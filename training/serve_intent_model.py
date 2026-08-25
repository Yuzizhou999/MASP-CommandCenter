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


def _dependencies() -> dict[str, Any]:
    try:
        import torch
        import uvicorn
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "缺少本地推理依赖，请在独立环境安装 requirements-finetune.txt"
        ) from error
    return {
        "torch": torch,
        "uvicorn": uvicorn,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 MASP 意图模型兼容 API")
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("models/masp-intent-lora")
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
):
    deps = _dependencies()
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
        state.update({"tokenizer": tokenizer, "model": model})
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
        if do_sample:
            generation_options["temperature"] = max(request.temperature, 1e-5)
        with generation_lock, torch.inference_mode():
            output = model.generate(
                **encoded,
                **generation_options,
            )
        completion_ids = output[0, encoded["input_ids"].shape[1] :]
        content = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
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
    )
    deps["uvicorn"].run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
