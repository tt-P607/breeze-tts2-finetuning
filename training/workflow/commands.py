from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimePaths:
    workspace: Path
    python: Path
    training_repo: Path
    model_root: Path
    converter: Path
    gguf_deps: Path


@dataclass(frozen=True)
class PlannedCommand:
    stage: str
    argv: tuple[str, ...]
    cwd: Path
    uses_gpu: bool
    output_root: Path | None = None


@dataclass(frozen=True)
class Plan:
    workspace: Path
    commands: tuple[PlannedCommand, ...]
    log_path: Path
    resource_log_path: Path
    export_checkpoint: Path | None
    warnings: tuple[str, ...] = ()


class PlanError(ValueError):
    pass


def runtime_paths(workspace: Path) -> RuntimePaths:
    workspace = workspace.resolve(strict=False)
    prep = workspace / "training-preparation"
    return RuntimePaths(
        workspace=workspace,
        python=prep / ".venv" / "Scripts" / "python.exe",
        training_repo=workspace / "breeze-tts2-finetuning",
        model_root=prep / "base-model",
        converter=workspace
        / "Breeze-TTS-2.cpp"
        / "scripts"
        / "convert_lora_to_gguf.py",
        gguf_deps=prep / "gguf-export" / "deps",
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PlanError(f"{label} file does not exist: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise PlanError(f"{label} directory does not exist: {path}")


def _require_training_runtime(paths: RuntimePaths) -> None:
    _require_file(paths.python, "training Python")
    _require_directory(paths.training_repo, "training repository")
    _require_directory(paths.model_root, "base model")
    _require_directory(paths.gguf_deps, "GGUF Python dependencies")
    _require_file(
        paths.training_repo / "training" / "real_data.py", "real_data trainer"
    )
    _require_file(
        paths.training_repo / "training" / "real_lora.py", "real_lora trainer"
    )


def _require_export_runtime(paths: RuntimePaths) -> None:
    _require_file(paths.python, "training Python")
    _require_directory(paths.gguf_deps, "GGUF Python dependencies")
    _require_file(paths.converter, "LoRA GGUF converter")


def _command(
    stage: str,
    python: Path,
    cwd: Path,
    args: list[str],
    *,
    uses_gpu: bool,
    output_root: Path | None = None,
) -> PlannedCommand:
    return PlannedCommand(
        stage,
        tuple(str(value) for value in [python, *args]),
        cwd,
        uses_gpu,
        output_root,
    )


def _verify_cache(cache_root: Path) -> None:
    receipt_path = cache_root / "cache-receipt.json"
    _require_file(receipt_path, "prepared cache receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"invalid cache receipt: {receipt_path}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "deterministic_supervised_cache_complete"
    ):
        raise PlanError(f"cache is not complete: {receipt_path}")


def _validate_resume(
    checkpoint: Path,
    *,
    train: dict[str, Any],
    output_root: Path,
) -> int:
    checkpoint = checkpoint.resolve(strict=False)
    output_root = output_root.resolve(strict=False)
    if checkpoint.parent != output_root:
        raise PlanError(
            "resume checkpoint must be an immediate child of train.output-root"
        )
    match = re.fullmatch(r"checkpoint-step-(\d{6})", checkpoint.name)
    if not match:
        raise PlanError("resume path must name a checkpoint-step-NNNNNN directory")
    if not checkpoint.is_dir():
        raise PlanError(f"resume checkpoint directory does not exist: {checkpoint}")
    for name in (
        "adapter.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng.pt",
        "trainer-state.json",
        "checkpoint-receipt.json",
    ):
        _require_file(checkpoint / name, f"resume checkpoint {name}")
    _require_file(output_root / "run-config.json", "saved training configuration")
    if (output_root / "training-receipt.json").exists():
        raise PlanError(
            "training already has a completion receipt; start a new batch instead of resuming it"
        )
    try:
        state = json.loads(
            (checkpoint / "trainer-state.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise PlanError("resume checkpoint contains invalid trainer state") from exc
    if not isinstance(state, dict):
        raise PlanError("resume checkpoint trainer state must be a JSON object")
    try:
        step = int(state.get("global_step", -1))
    except (TypeError, ValueError) as exc:
        raise PlanError(
            "resume checkpoint trainer state has an invalid global_step"
        ) from exc
    if step != int(match.group(1)):
        raise PlanError(
            "resume checkpoint directory and trainer-state step do not match"
        )
    if not 0 <= step < train["max-steps"]:
        raise PlanError("resume checkpoint step must be lower than train.max-steps")
    return step


def build_plan(
    config: dict[str, Any],
    *,
    stage: str,
    resume: Path | None,
    paths: RuntimePaths,
    dry_run: bool = False,
) -> Plan:
    if stage not in {"prepare", "train", "export", "all"}:
        raise PlanError(f"unknown stage: {stage}")
    if resume is not None and stage not in {"train", "all"}:
        raise PlanError("--resume is supported only for --stage train or --stage all")
    if stage in {"prepare", "export"} and stage not in config:
        raise PlanError(
            f"--stage {stage} requires a '{stage}' section in the configuration"
        )

    prepare = config.get("prepare")
    train = config["train"]
    export = config.get("export")
    cache_root: Path = train["cache-root"]
    output_root: Path = train["output-root"]
    prepare_root: Path | None = prepare["output-root"] if prepare else None
    export_checkpoint: Path | None = export["checkpoint"] if export else None
    export_output: Path | None = export["output"] if export else None
    if stage == "all" and prepare is not None and prepare_root != cache_root:
        raise PlanError(
            "prepare.output-root and train.cache-root must match for --stage all"
        )
    if (
        stage == "all"
        and export is not None
        and not math.isclose(float(export["alpha"]), float(train["alpha"]))
    ):
        raise PlanError("export.alpha must match train.alpha for --stage all")

    selected = (
        [stage]
        if stage != "all"
        else (
            (["prepare"] if prepare is not None else [])
            + ["train"]
            + (["export"] if export is not None else [])
        )
    )
    if stage == "all" and resume is not None:
        selected = [value for value in selected if value != "prepare"]

    needs_training = any(value in selected for value in ("prepare", "train"))
    if needs_training:
        _require_training_runtime(paths)
    if "export" in selected:
        _require_export_runtime(paths)

    warnings: list[str] = []
    if "prepare" in selected:
        assert prepare is not None and prepare_root is not None
        _require_file(prepare["train-manifest"], "training manifest")
        _require_file(prepare["validation-manifest"], "validation manifest")
        partial_cache = prepare_root.with_name(prepare_root.name + ".partial")
        if prepare_root.exists() or partial_cache.exists():
            message = f"refusing to overwrite an existing cache or partial cache: {prepare_root}"
            if dry_run:
                warnings.append(message)
            else:
                raise PlanError(message)

    if "train" in selected:
        if "prepare" not in selected or resume is not None:
            _verify_cache(cache_root)
        if resume is None:
            if output_root.exists():
                message = (
                    f"refusing to train into an existing output root: {output_root}"
                )
                if dry_run:
                    warnings.append(message)
                else:
                    raise PlanError(message)
        else:
            resume_step = _validate_resume(
                resume,
                train=train,
                output_root=output_root,
            )
            if train.get("stop-after-step", train["max-steps"]) <= resume_step:
                raise PlanError(
                    "resume stop point must be higher than the checkpoint step"
                )

    target_checkpoint: Path | None = None
    if "export" in selected:
        assert (
            export is not None
            and export_checkpoint is not None
            and export_output is not None
        )
        if stage == "export":
            _require_directory(export_checkpoint, "export checkpoint")
            _require_file(
                export_checkpoint / "adapter.safetensors", "checkpoint adapter"
            )
        else:
            # A batch can train to 300 and deliberately export its saved 250-step model.
            target_checkpoint = export_checkpoint
        output_tmp = export_output.with_name(export_output.name + ".tmp")
        if export_output.exists() or output_tmp.exists():
            message = f"refusing to overwrite an existing model or partial export: {export_output}"
            if dry_run:
                warnings.append(message)
            else:
                raise PlanError(message)

    commands: list[PlannedCommand] = []
    if "prepare" in selected:
        assert prepare is not None and prepare_root is not None
        args = [
            "-u",
            "-m",
            "training.real_data",
            "--model-root",
            str(paths.model_root),
            "--train-manifest",
            str(prepare["train-manifest"]),
            "--validation-manifest",
            str(prepare["validation-manifest"]),
            "--output-root",
            str(prepare_root),
            "--train-limit",
            str(prepare["train-limit"]),
            "--validation-limit",
            str(prepare["validation-limit"]),
            "--seed",
            str(prepare["seed"]),
            "--speaker",
            prepare["speaker"],
            "--device",
            prepare["device"],
        ]
        commands.append(
            _command("prepare", paths.python, paths.training_repo, args, uses_gpu=True)
        )

    if "train" in selected:
        args = [
            "-u",
            "-m",
            "training.real_lora",
            "--model-root",
            str(paths.model_root),
            "--cache-root",
            str(cache_root),
            "--output-root",
            str(output_root),
            "--device",
            train["device"],
            "--rank",
            str(train["rank"]),
            "--alpha",
            str(train["alpha"]),
            "--max-steps",
            str(train["max-steps"]),
            "--gradient-accumulation",
            str(train["gradient-accumulation"]),
            "--learning-rate",
            str(train["learning-rate"]),
            "--weight-decay",
            str(train["weight-decay"]),
            "--warmup-steps",
            str(train["warmup-steps"]),
            "--save-every",
            str(train["save-every"]),
            "--validation-examples",
            str(train["validation-examples"]),
            "--seed",
            str(train["seed"]),
            "--max-gradient-norm",
            str(train["max-gradient-norm"]),
        ]
        if "stop-after-step" in train:
            args.extend(("--stop-after-step", str(train["stop-after-step"])))
        if resume is not None:
            args.extend(("--resume-checkpoint", str(resume.resolve(strict=False))))
        commands.append(
            _command(
                "train",
                paths.python,
                paths.training_repo,
                args,
                uses_gpu=True,
                output_root=output_root,
            )
        )

    if "export" in selected:
        assert export_checkpoint is not None and export_output is not None
        checkpoint = export_checkpoint if stage == "export" else target_checkpoint
        assert checkpoint is not None
        args = [
            str(paths.converter),
            str(checkpoint / "adapter.safetensors"),
            "--output",
            str(export_output),
            "--alpha",
            str(export["alpha"]),
        ]
        commands.append(
            _command("export", paths.python, paths.workspace, args, uses_gpu=False)
        )

    log_root = output_root.parent
    return Plan(
        workspace=paths.workspace,
        commands=tuple(commands),
        log_path=log_root / "统一训练.log",
        resource_log_path=log_root / "统一资源监测.jsonl",
        export_checkpoint=target_checkpoint or export_checkpoint,
        warnings=tuple(warnings),
    )
