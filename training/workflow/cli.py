from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from training.workflow.commands import build_plan, runtime_paths
from training.workflow.config import ConfigError, load_config, resolve_path
from training.workflow.runner import run_command

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, train, resume, and export one Breeze LoRA batch."
    )
    parser.add_argument(
        "--workspace-root",
        default=str(WORKSPACE_ROOT),
        help="Workspace containing datasets and shared resources",
    )
    parser.add_argument("--python", help="Training Python executable")
    parser.add_argument("--model-root", help="Original training model directory")
    parser.add_argument("--converter", help="C++ repository LoRA converter script")
    parser.add_argument("--gguf-deps", help="GGUF Python dependency directory")
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Runtime stop point without editing the batch recipe",
    )
    parser.add_argument(
        "--export-checkpoint", help="Checkpoint selected for this export"
    )
    parser.add_argument("--export-output", help="Destination GGUF for this export")
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config path; relative paths start at the workspace root",
    )
    parser.add_argument(
        "--stage", choices=("prepare", "train", "export", "all"), default="all"
    )
    parser.add_argument("--resume", help="checkpoint directory to resume from")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print commands without writing outputs",
    )
    return parser.parse_args(argv)


def show_plan(plan, *, dry_run: bool) -> None:
    title = "DRY RUN" if dry_run else "RUN"
    print(f"{title}: {len(plan.commands)} stage(s); workspace={plan.workspace}")
    for command in plan.commands:
        print(
            f"[{command.stage}] {subprocess.list2cmdline([str(x) for x in command.argv])}"
        )
    for warning in plan.warnings:
        print(f"NOTE: {warning}")
    print(f"Log: {plan.log_path}")
    print(f"Resource log: {plan.resource_log_path}")


def show_historical_config(config, *, stage: str, resume: Path | None) -> None:
    print(f"HISTORICAL ONLY: stage={stage}; no command will be run")
    print(
        "This recipe is archived for reference. Its initialization behavior is not implemented by the unified trainer."
    )
    for field in ("initialization", "notes"):
        if field in config:
            print(f"{field}: {json.dumps(config[field], ensure_ascii=False, indent=2)}")
    if resume is not None:
        print(f"resume: {resume}")
    for section in ("prepare", "train", "export"):
        if section not in config:
            continue
        print(f"[{section}]")
        values = {
            name: str(value) if isinstance(value, Path) else value
            for name, value in config[section].items()
        }
        print(json.dumps(values, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        workspace = Path(args.workspace_root).resolve()
        config_path = resolve_path(args.config, workspace)
        config = load_config(config_path, workspace)
        if args.export_checkpoint or args.export_output:
            export = config.setdefault("export", {"alpha": config["train"]["alpha"]})
            if args.export_checkpoint:
                export["checkpoint"] = resolve_path(args.export_checkpoint, workspace)
            if args.export_output:
                export["output"] = resolve_path(args.export_output, workspace)
            if not {"checkpoint", "output"}.issubset(export):
                raise ConfigError("export requires both a checkpoint and destination")
        if args.stop_after_step is not None:
            if not 0 < args.stop_after_step <= config["train"]["max-steps"]:
                raise ConfigError(
                    "stop point must be positive and within the original schedule"
                )
            if args.stop_after_step == config["train"]["max-steps"]:
                config["train"].pop("stop-after-step", None)
            else:
                config["train"]["stop-after-step"] = args.stop_after_step
        resume_path = resolve_path(args.resume, workspace) if args.resume else None
        if config["historical_only"]:
            if not args.dry_run:
                print(
                    "error: historical_only=true recipes are archive-only; use --dry-run to inspect them.",
                    file=sys.stderr,
                )
                return 2
            show_historical_config(config, stage=args.stage, resume=resume_path)
            return 0
        paths = runtime_paths(workspace)
        overrides = {
            key: resolve_path(getattr(args, key), workspace)
            for key in ("python", "model_root", "converter", "gguf_deps")
            if getattr(args, key)
        }
        paths = replace(paths, **overrides)
        plan = build_plan(
            config,
            stage=args.stage,
            resume=resume_path,
            paths=paths,
            dry_run=args.dry_run,
        )
    except (ConfigError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    show_plan(plan, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    try:
        for index, command in enumerate(plan.commands):
            output_root = config["train"]["output-root"]
            stop_file = output_root.with_name(output_root.name + ".stop-request")
            if stop_file.exists():
                stop_file.unlink()
                print("已在阶段边界暂停；现有缓存和检查点保留。", flush=True)
                return 0
            result = run_command(
                command,
                workspace=workspace,
                python=paths.python,
                training_repo=paths.training_repo,
                gguf_deps=paths.gguf_deps,
                log_path=plan.log_path,
                resource_log_path=plan.resource_log_path,
            )
            if result.paused:
                print(f"训练已保存可续训检查点：{result.checkpoint}")
                state = json.loads(
                    (result.checkpoint / "trainer-state.json").read_text(
                        encoding="utf-8"
                    )
                )
                if state.get("pause_reason") == "user_stop":
                    print("已按请求保存并暂停。", flush=True)
                    return 0
                later_export = next(
                    (
                        row
                        for row in plan.commands[index + 1 :]
                        if row.stage == "export"
                    ),
                    None,
                )
                expected_adapter = str(result.checkpoint / "adapter.safetensors")
                if (
                    later_export is not None
                    and later_export.argv[2] != expected_adapter
                ):
                    selected_adapter = Path(later_export.argv[2])
                    if not selected_adapter.is_file():
                        raise RuntimeError(
                            "训练在非目标步暂停；已保留检查点，请使用 --resume 继续到配置的导出步。"
                        )
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"See log: {plan.log_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
