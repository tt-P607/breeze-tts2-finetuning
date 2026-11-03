from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


PREPARE_DEFAULTS: dict[str, Any] = {
    "train-limit": 1024,
    "validation-limit": 128,
    "seed": 42,
    "speaker": "S0",
    "device": "cuda:0",
}
TRAIN_DEFAULTS: dict[str, Any] = {
    "device": "cuda:0",
    "rank": 16,
    "alpha": 32,
    "max-steps": 300,
    "stop-after-step": 250,
    "gradient-accumulation": 4,
    "learning-rate": 0.0001,
    "weight-decay": 0.01,
    "warmup-steps": 100,
    "save-every": 25,
    "validation-examples": 64,
    "seed": 42,
    "max-gradient-norm": 1.0,
}
EXPORT_DEFAULTS: dict[str, Any] = {"alpha": 32}

PATH_FIELDS = {
    "prepare": ("train-manifest", "validation-manifest", "output-root"),
    "train": ("cache-root", "output-root"),
    "export": ("checkpoint", "output"),
}
SECTION_DEFAULTS = {
    "prepare": PREPARE_DEFAULTS,
    "train": TRAIN_DEFAULTS,
    "export": EXPORT_DEFAULTS,
}
ALLOWED_FIELDS = {
    "prepare": frozenset((*PATH_FIELDS["prepare"], *PREPARE_DEFAULTS)),
    "train": frozenset((*PATH_FIELDS["train"], *TRAIN_DEFAULTS)),
    "export": frozenset((*PATH_FIELDS["export"], *EXPORT_DEFAULTS)),
}


def resolve_path(value: str | Path, workspace: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve(strict=False)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    return value


def _normalize_section(
    section_name: str,
    raw: Any,
    workspace: Path,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"'{section_name}' must be an object")
    unknown = sorted(set(raw) - ALLOWED_FIELDS[section_name])
    if unknown:
        raise ConfigError(f"unknown {section_name} option(s): {', '.join(unknown)}")

    result = dict(SECTION_DEFAULTS[section_name])
    result.update(raw)
    for field in PATH_FIELDS[section_name]:
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"'{section_name}.{field}' must be a non-empty path")
        result[field] = resolve_path(value, workspace)

    if section_name == "prepare":
        for key in ("train-limit", "validation-limit"):
            result[key] = _integer(result[key], f"prepare.{key}")
        result["seed"] = _integer(result["seed"], "prepare.seed")
        for key in ("device", "speaker"):
            if not isinstance(result[key], str) or not result[key]:
                raise ConfigError(f"prepare.{key} must be a non-empty string")
    elif section_name == "train":
        if "max-steps" in raw and "stop-after-step" not in raw:
            result.pop("stop-after-step", None)
        integer_fields = (
            "rank",
            "max-steps",
            "gradient-accumulation",
            "warmup-steps",
            "save-every",
            "validation-examples",
        )
        for key in integer_fields:
            result[key] = _integer(result[key], f"train.{key}")
        result["seed"] = _integer(result["seed"], "train.seed")
        for key in ("alpha", "learning-rate", "weight-decay", "max-gradient-norm"):
            result[key] = _number(result[key], f"train.{key}")
        if "stop-after-step" in result:
            result["stop-after-step"] = _integer(
                result["stop-after-step"], "train.stop-after-step"
            )
        if not isinstance(result["device"], str) or not result["device"]:
            raise ConfigError("train.device must be a non-empty string")
    else:
        result["alpha"] = _number(result["alpha"], "export.alpha")

    return result


def load_config(path: Path, workspace: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    required_sections = {"train"}
    optional_sections = {"prepare", "export"}
    allowed_root_fields = (
        required_sections
        | optional_sections
        | {"historical_only", "initialization", "notes"}
    )
    if not required_sections.issubset(raw) or not set(raw).issubset(
        allowed_root_fields
    ):
        missing = sorted(required_sections - set(raw))
        unknown = sorted(set(raw) - allowed_root_fields)
        detail = []
        if missing:
            detail.append(f"missing section(s): {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown section(s): {', '.join(unknown)}")
        raise ConfigError("; ".join(detail))
    historical_only = raw.get("historical_only", False)
    if not isinstance(historical_only, bool):
        raise ConfigError("'historical_only' must be true or false")
    result: dict[str, Any] = {
        section: _normalize_section(section, raw[section], workspace)
        for section in ("prepare", "train", "export")
        if section in raw
    }
    result["historical_only"] = historical_only
    for field in ("initialization", "notes"):
        if field in raw:
            result[field] = raw[field]
    return result
