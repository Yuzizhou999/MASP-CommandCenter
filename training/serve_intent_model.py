from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = 0
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None


def response_json_schema(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if response_format is None:
        return None
    response_type = response_format.get("type")
    if response_type == "json_object":
        return {"type": "object"}
    if response_type != "json_schema":
        raise ValueError(f"unsupported response_format type: {response_type}")
    descriptor = response_format.get("json_schema")
    if not isinstance(descriptor, dict):
        raise ValueError("json_schema response format requires json_schema")
    schema = descriptor.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("json_schema response format requires a schema object")
    return schema


def validate_generated_json(content: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("model produced incomplete JSON") from error
    try:
        validate_json_schema(payload, schema)
    except JsonSchemaValidationError as error:
        raise ValueError("model output does not match the requested schema") from error
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")
    return payload


def _dependencies() -> dict[str, Any]:
    try:
        import torch
        import uvicorn
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            LogitsProcessorList,
        )
    except ImportError as error:
        raise RuntimeError(
            "缺少本地推理依赖，请在独立环境安装 requirements-finetune.txt"
        ) from error
    try:
        import xgrammar
        from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor
    except ImportError:
        xgrammar = None
        XGrammarLogitsProcessor = None
    return {
        "torch": torch,
        "uvicorn": uvicorn,
        "xgrammar": xgrammar,
        "XGrammarLogitsProcessor": XGrammarLogitsProcessor,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "LogitsProcessorList": LogitsProcessorList,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 MASP 意图模型兼容 API")
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("models/masp-agent-lora-v2.3")
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="不加载 adapter，直接提供指定 Hugging Face 基座模型",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--require-xgrammar",
        action="store_true",
        help="要求 token 级 XGrammar 约束；缺少依赖时拒绝启动",
    )
    return parser.parse_args()


def resolve_model_spec(
    adapter_dir: Path, *, base_model: str | None = None, model_id: str | None = None
) -> dict[str, Any]:
    if base_model:
        return {
            "adapterDir": None,
            "tokenizerSource": base_model,
            "baseModel": base_model,
            "modelId": model_id or base_model,
            "mode": "base-model",
            "created": int(time()),
        }
    adapter_dir = adapter_dir.resolve()
    card_path = adapter_dir / "model-card.json"
    if not card_path.is_file():
        raise FileNotFoundError(f"未找到模型卡：{card_path}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    return {
        "adapterDir": adapter_dir,
        "tokenizerSource": adapter_dir,
        "baseModel": str(card["baseModel"]),
        "modelId": model_id or str(card["modelId"]),
        "mode": "lora-adapter",
        "created": int(card_path.stat().st_mtime),
    }


def create_app(
    adapter_dir: Path,
    *,
    base_model: str | None = None,
    model_id: str | None = None,
    max_new_tokens: int = 384,
    require_xgrammar: bool = False,
):
    deps = _dependencies()
    if require_xgrammar and deps["xgrammar"] is None:
        raise RuntimeError(
            "当前环境缺少 xgrammar，正式受约束评测不能启动"
        )
    torch = deps["torch"]
    spec = resolve_model_spec(
        adapter_dir, base_model=base_model, model_id=model_id
    )
    state: dict[str, Any] = {}
    generation_lock = Lock()

    @asynccontextmanager
    async def lifespan(_: Any):
        if not torch.cuda.is_available():
            raise RuntimeError("本地 4-bit 推理需要可用的 CUDA GPU")
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        tokenizer = deps["AutoTokenizer"].from_pretrained(
            spec["tokenizerSource"], trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base = deps["AutoModelForCausalLM"].from_pretrained(
            spec["baseModel"],
            torch_dtype=compute_dtype,
            quantization_config=quantization,
            device_map="auto",
            trust_remote_code=False,
        )
        model = (
            deps["PeftModel"].from_pretrained(base, spec["adapterDir"])
            if spec["adapterDir"] is not None
            else base
        )
        model.eval()
        grammar_compiler = None
        if deps["xgrammar"] is not None:
            tokenizer_info = deps["xgrammar"].TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=int(model.config.vocab_size)
            )
            grammar_compiler = deps["xgrammar"].GrammarCompiler(tokenizer_info)
        state.update(
            {
                "tokenizer": tokenizer,
                "model": model,
                "grammarCompiler": grammar_compiler,
            }
        )
        yield
        state.clear()

    app = FastAPI(
        title="MASP Intent Model API", version="0.1.0", lifespan=lifespan
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if state else "loading",
            "model": spec["modelId"],
            "baseModel": spec["baseModel"],
            "mode": spec["mode"],
            "structuredOutput": (
                "xgrammar"
                if state.get("grammarCompiler") is not None
                else "jsonschema-validation-only"
            ),
            "xgrammarRequired": require_xgrammar,
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": spec["modelId"],
                    "object": "model",
                    "created": spec["created"],
                    "owned_by": "masp-command-center",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.model != spec["modelId"]:
            raise HTTPException(status_code=404, detail="model not found")
        if not state:
            raise HTTPException(status_code=503, detail="model is loading")
        tokenizer = state["tokenizer"]
        model = state["model"]
        messages = [row.model_dump(exclude_none=True) for row in request.messages]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        limit = max(1, min(request.max_tokens or max_new_tokens, max_new_tokens))
        do_sample = request.temperature > 0
        generation_options = {
            "max_new_tokens": limit,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        try:
            schema = response_json_schema(request.response_format)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        compiler = state.get("grammarCompiler")
        if schema is not None and compiler is not None:
            if (request.response_format or {}).get("type") == "json_object":
                compiled_grammar = compiler.compile_builtin_json_grammar()
            else:
                compiled_grammar = compiler.compile_json_schema(
                    schema,
                    strict_mode=True,
                    any_whitespace=True,
                )
            xgrammar_processor = deps["XGrammarLogitsProcessor"](
                compiled_grammar
            )

            class CpuXGrammarLogitsProcessor:
                def __call__(self, input_ids, scores):
                    target_device = scores.device
                    constrained = xgrammar_processor(input_ids, scores.to("cpu"))
                    return constrained.to(target_device)

            generation_options["logits_processor"] = deps[
                "LogitsProcessorList"
            ]([CpuXGrammarLogitsProcessor()])
        if do_sample:
            generation_options["temperature"] = max(request.temperature, 1e-5)
        with generation_lock, torch.inference_mode():
            output = model.generate(
                **encoded,
                **generation_options,
            )
        completion_ids = output[0, encoded["input_ids"].shape[1] :]
        content = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        if schema is not None:
            try:
                validate_generated_json(content, schema)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        prompt_tokens = int(encoded["input_ids"].shape[1])
        completion_tokens = int(completion_ids.shape[0])
        return {
            "id": f"chatcmpl-{uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time()),
            "model": spec["modelId"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def main() -> None:
    args = _arguments()
    deps = _dependencies()
    app = create_app(
        args.adapter_dir,
        base_model=args.base_model,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        require_xgrammar=args.require_xgrammar,
    )
    deps["uvicorn"].run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
