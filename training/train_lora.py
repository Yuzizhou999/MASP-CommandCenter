from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _dependencies():
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError(
            "缺少微调依赖，请在独立环境安装 requirements-finetune.txt"
        ) from error
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any
    torch: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(row["input_ids"]) for row in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        pad_id = int(self.tokenizer.pad_token_id)
        for row in features:
            padding = max_length - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * padding)
            attention_masks.append(row["attention_mask"] + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(
                attention_masks, dtype=self.torch.long
            ),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
        }


class TokenizedJsonlDataset:
    """Small map-style dataset that keeps training independent of Arrow/SSL setup."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 QLoRA 微调 MASP 意图模型")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/intent-lora.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/masp-intent-lora")
    )
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = _arguments()
    deps = _dependencies()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA 训练需要可用的 CUDA GPU")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_dir = args.dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = dataset_dir / manifest["files"]["train"]["path"]
    valid_path = dataset_dir / manifest["files"]["valid"]["path"]
    for split, path in (("train", train_path), ("valid", valid_path)):
        actual = _sha256(path)
        if actual != manifest["files"][split]["sha256"]:
            raise ValueError(f"{split} 数据文件摘要不匹配")

    tokenizer = deps["AutoTokenizer"].from_pretrained(
        config["baseModel"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    quantization_config = None
    if bool(config.get("loadIn4Bit", True)):
        quantization_config = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = deps["AutoModelForCausalLM"].from_pretrained(
        config["baseModel"],
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if quantization_config is not None:
        model = deps["prepare_model_for_kbit_training"](
            model, use_gradient_checkpointing=True
        )
    lora_config = deps["LoraConfig"](
        r=int(config["loraR"]),
        lora_alpha=int(config["loraAlpha"]),
        lora_dropout=float(config["loraDropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config["targetModules"]),
    )
    model = deps["get_peft_model"](model, lora_config)
    model.print_trainable_parameters()

    max_length = int(config["maxLength"])

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        encoded = tokenizer(
            full,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        labels = list(encoded["input_ids"])
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        if all(value == -100 for value in labels):
            raise ValueError("样本截断后没有保留 assistant token，请增大 maxLength")
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    train_rows = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid_rows = [
        json.loads(line)
        for line in valid_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenized_train = TokenizedJsonlDataset([tokenize(row) for row in train_rows])
    tokenized_valid = TokenizedJsonlDataset([tokenize(row) for row in valid_rows])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = deps["TrainingArguments"](
        output_dir=str(output_dir),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["perDeviceTrainBatchSize"]),
        per_device_eval_batch_size=int(config["perDeviceEvalBatchSize"]),
        gradient_accumulation_steps=int(config["gradientAccumulationSteps"]),
        learning_rate=float(config["learningRate"]),
        logging_steps=int(config["loggingSteps"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if quantization_config is not None else "adamw_torch",
        report_to=[],
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        remove_unused_columns=False,
    )
    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=CompletionOnlyCollator(tokenizer=tokenizer, torch=torch),
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    evaluation = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    adapter_file = output_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise RuntimeError("训练完成但未找到 adapter_model.safetensors")
    metrics = {
        key: round(float(value), 6)
        for key, value in {**train_result.metrics, **evaluation}.items()
        if isinstance(value, (int, float))
    }
    model_card = {
        "schemaVersion": 1,
        "modelId": "masp-intent-lora",
        "version": "0.1.0",
        "baseModel": config["baseModel"],
        "trainingMethod": "QLoRA-SFT-completion-only",
        "datasetVersion": manifest["datasetId"],
        "adapterFile": adapter_file.name,
        "adapterSha256": _sha256(adapter_file),
        "status": "candidate",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    (output_dir / "model-card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training-config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(model_card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
