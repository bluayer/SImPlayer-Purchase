from __future__ import annotations

import importlib
import json
import os
import sys
from types import ModuleType
from typing import Any


OPENAI_MODEL_PREFIXES = (
    "openai.",
    "global.openai.",
    "us.openai.",
)


def is_openai_bedrock_model(model_id: str) -> bool:
    return model_id.startswith(OPENAI_MODEL_PREFIXES)


def mantle_model_id(model_id: str) -> str:
    for prefix in ("global.", "us."):
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return model_id


def supports_temperature(model_id: str) -> bool:
    return "anthropic.claude-opus-5" not in model_id


def effective_max_tokens(model_id: str, requested: int) -> int:
    if "anthropic.claude-opus-5" in model_id:
        return max(requested, 2048)
    return requested


def _install_jiter_import_fallback() -> None:
    try:
        importlib.import_module("jiter")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "jiter":
            raise

    module = ModuleType("jiter")

    def from_json(
        json_data: str | bytes | bytearray,
        /,
        *,
        partial_mode: object = False,
        **_: object,
    ) -> Any:
        if partial_mode not in (False, None, Ellipsis):
            raise RuntimeError(
                "The pure-Python jiter fallback does not support partial JSON."
            )
        return json.loads(json_data)

    module.from_json = from_json  # type: ignore[attr-defined]
    module.__all__ = ["from_json"]
    sys.modules.setdefault("jiter", module)


def build_strands_model(
    *,
    model_id: str,
    region_name: str,
    max_tokens: int,
) -> Any:
    max_tokens = effective_max_tokens(model_id, max_tokens)
    if is_openai_bedrock_model(model_id):
        # OpenAI's package imports jiter's symbol while initializing chat
        # streaming, even though the Responses path used here never calls it.
        _install_jiter_import_fallback()
        from strands.models import OpenAIResponsesModel

        return OpenAIResponsesModel(
            client_args={
                "timeout": float(
                    os.getenv(
                        "PURCHASE_BEHAVIOR_MODEL_TIMEOUT_SECONDS",
                        "120",
                    )
                ),
                "max_retries": 0,
            },
            model_id=mantle_model_id(model_id),
            bedrock_mantle_config={"region": region_name},
            params={"max_output_tokens": max_tokens},
            stateful=False,
        )

    from strands.models import BedrockModel

    model_kwargs: dict[str, Any] = {
        "model_id": model_id,
        "region_name": region_name,
        "max_tokens": max_tokens,
    }
    if supports_temperature(model_id):
        model_kwargs["temperature"] = 0.0
    return BedrockModel(
        **model_kwargs,
    )
