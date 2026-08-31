from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LLMModelCard(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(default=1, alias="schemaVersion")
    model_id: str = Field(min_length=1, alias="modelId")
    version: str = Field(min_length=1)
    base_model: str = Field(min_length=1, alias="baseModel")
    training_method: str = Field(min_length=1, alias="trainingMethod")
    dataset_version: str = Field(min_length=1, alias="datasetVersion")
    adapter_file: str | None = Field(default=None, alias="adapterFile")
    adapter_sha256: str | None = Field(default=None, alias="adapterSha256")
    status: Literal["candidate", "active", "retired"] = "candidate"
    run_scope: Literal["sanity", "full"] | None = Field(
        default=None, alias="runScope"
    )
    sample_limits: dict[str, int | None] = Field(
        default_factory=dict, alias="sampleLimits"
    )
    checkpointing: dict[str, int | str | None] = Field(default_factory=dict)
    tokenization: dict[str, int | float | bool] = Field(default_factory=dict)
    training_environment: dict[str, str | None] = Field(
        default_factory=dict, alias="trainingEnvironment"
    )
    created_at: str = Field(alias="createdAt")
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_adapter_identity(self) -> "LLMModelCard":
        if bool(self.adapter_file) != bool(self.adapter_sha256):
            raise ValueError("adapterFile 和 adapterSha256 必须同时提供")
        if self.adapter_sha256 is not None and len(self.adapter_sha256) != 64:
            raise ValueError("adapterSha256 必须是 64 位 SHA-256")
        return self


def model_registration(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    result: dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "valid": False,
    }
    if not path.is_file():
        return result
    try:
        card = LLMModelCard.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result["error"] = str(error)
        return result

    result.update(card.model_dump(by_alias=True, mode="json"))
    result["valid"] = True
    if card.adapter_file and card.adapter_sha256:
        adapter_path = (path.parent / card.adapter_file).resolve()
        try:
            adapter_path.relative_to(path.parent.resolve())
        except ValueError:
            result["valid"] = False
            result["error"] = "adapterFile 不得指向模型目录之外"
            return result
        result["adapterPresent"] = adapter_path.is_file()
        if not adapter_path.is_file():
            result["valid"] = False
            return result
        actual = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
        result["adapterSha256Matches"] = actual == card.adapter_sha256
        if actual != card.adapter_sha256:
            result["valid"] = False
    return result
