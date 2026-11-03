from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config

_CHECKPOINT_RE = re.compile(r"checkpoint-step-(\d+)$")
_RUNTIME_FLAGS = {
    "python": "--python",
    "model_root": "--model-root",
    "converter": "--converter",
    "gguf_deps": "--gguf-deps",
    "stop_after_step": "--stop-after-step",
    "export_checkpoint": "--export-checkpoint",
    "export_output": "--export-output",
}
_CONFIG_ROOTS = ("角色模型", "风格模型", "数据集")
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "base-model",
    "gguf-export",
    "cache",
    "缓存",
    "checkpoints",
    "检查点",
}
_RESUME_FILES = (
    "adapter.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng.pt",
    "trainer-state.json",
    "checkpoint-receipt.json",
)


@dataclass
class _Job:
    job_id: str
    process: subprocess.Popen[bytes]
    config_path: Path
    stage: str
    console_log: Path
    started_at: float
    resume: Path | None
    dry_run: bool
    requested_stop: str | None = None
    ended_at: float | None = None
    returncode: int | None = None
    final_status: str | None = None


class JobManager:
    """供 WebUI 调用的单任务训练服务。"""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.source_repo = self.workspace / "breeze-tts2-finetuning"
        self._lock = threading.RLock()
        self._job: _Job | None = None
        self._resource_cache: tuple[float, dict[str, Any]] | None = None

    def _workspace_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"路径必须位于工作区内：{value}") from exc
        return resolved

    def _display_path(self, path: Path) -> str:
        try:
            return path.resolve(strict=False).relative_to(self.workspace).as_posix()
        except ValueError:
            return str(path)

    def list_configs(self) -> list[str]:
        """返回常用批次目录中的训练配置相对路径。"""
        found: set[Path] = set()
        for relative_root in _CONFIG_ROOTS:
            root = self.workspace / relative_root
            if not root.is_dir():
                continue
            for current, dirs, files in os.walk(root):
                dirs[:] = sorted(name for name in dirs if name not in _SKIP_DIRS)
                if "training-config.json" in files:
                    found.add((Path(current) / "training-config.json").resolve())
        return sorted(self._display_path(path) for path in found)

    def read_config(self, path: str | Path) -> dict[str, Any]:
        config_path = self._workspace_path(path)
        if config_path.name != "training-config.json":
            raise ValueError("配置文件名必须是 training-config.json")
        return self._json_safe(self._load_normalized(config_path))

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        return value

    def _load_normalized(self, path: Path) -> dict[str, Any]:
        try:
            return load_config(path, self.workspace)
        except (ConfigError, OSError, json.JSONDecodeError) as exc:
            raise ValueError(str(exc)) from exc

    @staticmethod
    def _has_training_output(config: dict[str, Any]) -> bool:
        train = config.get("train")
        if not isinstance(train, dict):
            return False
        output_root = train.get("output-root")
        if not isinstance(output_root, Path):
            return False
        if not output_root.is_dir():
            return False
        return any(output_root.iterdir())

    def save_config(self, path: str | Path, config: dict[str, Any]) -> None:
        config_path = self._workspace_path(path)
        if config_path.name != "training-config.json":
            raise ValueError("新批次配置文件名必须是 training-config.json")
        if not isinstance(config, dict):
            raise ValueError("配置根节点必须是 JSON 对象")

        with self._lock:
            if self._job and self._refresh_job_locked() in {"running", "stopping"}:
                if self._job.config_path == config_path:
                    raise ValueError("该批次任务正在运行，不能修改配置")

            if config_path.is_file():
                old = self._load_normalized(config_path)
                if self._has_training_output(old):
                    raise ValueError("批次已有训练输出，不能覆盖历史配置")
            candidate = config
            if candidate.get("historical_only") is True:
                raise ValueError("历史配方为只读记录，不能作为新批次配置保存")

            config_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    dir=config_path.parent,
                    suffix=".json.tmp",
                    delete=False,
                ) as handle:
                    temporary_path = Path(handle.name)
                    json.dump(candidate, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")

                new = self._load_normalized(temporary_path)
                if self._has_training_output(new):
                    raise ValueError("目标训练目录已有训练输出，不能修改批次配置")
                os.replace(temporary_path, config_path)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _stop_marker(output_root: Path) -> Path:
        return output_root.with_name(output_root.name + ".stop-request")

    def _build_command(
        self,
        config_path: Path,
        stage: str,
        resume: Path | None,
        dry_run: bool,
        runtime: dict[str, Any] | None,
    ) -> list[str]:
        runtime = runtime or {}
        launcher = str(runtime.get("python") or sys.executable)
        command = [
            launcher,
            "-u",
            "-m",
            "training.workflow.cli",
            "--workspace-root",
            str(self.workspace),
            "--config",
            str(config_path),
            "--stage",
            stage,
        ]
        if resume is not None:
            command.extend(("--resume", str(resume)))
        if dry_run:
            command.append("--dry-run")
        for key, flag in _RUNTIME_FLAGS.items():
            value = runtime.get(key)
            if value is not None:
                command.extend((flag, str(value)))
        return command

    def _refresh_job_locked(self) -> str | None:
        job = self._job
        if job is None:
            return None
        code = job.process.poll()
        if code is None:
            return "stopping" if job.requested_stop else "running"
        if job.returncode is None:
            job.returncode = code
            job.ended_at = time.monotonic()
            if code != 0:
                job.final_status = (
                    "stopped" if job.requested_stop == "force" else "failed"
                )
            else:
                state = self._training_state(job.config_path)
                config = self._load_normalized(job.config_path)
                export = config.get("export", {})
                exported = (
                    isinstance(export.get("output"), Path)
                    and export["output"].is_file()
                )
                if job.dry_run or job.stage in {"export", "prepare"}:
                    job.final_status = "completed"
                elif job.stage == "all" and exported and not job.requested_stop:
                    job.final_status = "completed"
                elif state and state.get("pause_reason") == "user_stop":
                    job.final_status = "paused"
                elif state and state.get("status") == "paused_for_fresh_process_resume":
                    job.final_status = "paused"
                elif job.requested_stop == "graceful" and code == 0:
                    job.final_status = "paused"
                else:
                    job.final_status = "completed"
        return job.final_status

    def _training_state(self, config_path: Path) -> dict[str, Any] | None:
        try:
            config = self._load_normalized(config_path)
            output_root = config["train"]["output-root"]
            if not isinstance(output_root, Path):
                return None
            latest_path = output_root / "latest.json"
            checkpoint: Path | None = None
            if latest_path.is_file():
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
                raw_path = Path(latest["checkpoint"])
                checkpoint = (
                    raw_path if raw_path.is_absolute() else self.workspace / raw_path
                )
                checkpoint = checkpoint.resolve(strict=False)
                if checkpoint.parent != output_root.resolve(strict=False):
                    checkpoint = None
            if checkpoint is None and output_root.is_dir():
                candidates = [
                    child
                    for child in output_root.iterdir()
                    if child.is_dir() and _CHECKPOINT_RE.fullmatch(child.name)
                ]
                if candidates:
                    checkpoint = max(
                        candidates,
                        key=lambda item: int(
                            _CHECKPOINT_RE.fullmatch(item.name).group(1)
                        ),
                    )
            if checkpoint is None:
                return None
            state_path = checkpoint / "trainer-state.json"
            if not state_path.is_file():
                return None
            value = json.loads(state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _tail_lines(path: Path, limit: int) -> list[str]:
        if limit <= 0 or not path.is_file():
            return []
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                position = handle.tell()
                chunk = b""
                while position > 0 and chunk.count(b"\n") <= limit:
                    size = min(8192, position)
                    position -= size
                    handle.seek(position)
                    chunk = handle.read(size) + chunk
            return chunk.decode("utf-8", errors="replace").splitlines()[-limit:]
        except OSError:
            return []

    def _log_paths(self, config_path: Path) -> tuple[Path, Path]:
        config = self._load_normalized(config_path)
        output_root = config["train"]["output-root"]
        return (
            config_path.parent / "统一WebUI控制台.log",
            output_root.parent / "统一训练.log",
        )

    def _combined_log(
        self, config_path: Path, *, active_console: Path | None = None
    ) -> str:
        try:
            console_path, native_path = self._log_paths(config_path)
        except (ValueError, KeyError, TypeError):
            console_path, native_path = config_path.parent / "统一WebUI控制台.log", None
        if active_console is not None:
            console_path = active_console

        console_lines = self._tail_lines(console_path, 35)
        native_lines = self._tail_lines(native_path, 85) if native_path else []
        if not console_lines:
            native_lines = self._tail_lines(native_path, 120) if native_path else []
        elif not native_lines:
            console_lines = self._tail_lines(console_path, 120)
        lines: list[str] = []
        if console_lines:
            lines.extend(("[控制台]", *console_lines))
        if native_lines:
            lines.extend(("[训练日志]", *native_lines))
        return "\n".join(lines[-120:])

    @staticmethod
    def _loss_value(value: Any, key: str = "total") -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if not isinstance(value, dict):
            return None
        aliases = {
            "total": ("total", "loss", "overall"),
            "backbone": ("backbone", "backbone_loss"),
            "depth": ("depth_decoder", "depth", "depth_loss"),
        }
        for name in aliases.get(key, (key,)):
            item = value.get(name)
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                return float(item)
        return None

    def _metrics(
        self,
        state: dict[str, Any] | None,
        native_log_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        if native_log_path is not None:
            for line in self._tail_lines(native_log_path, 1000):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("event") != "training_step":
                    continue
                try:
                    step = int(event["step"])
                except (KeyError, TypeError, ValueError):
                    continue
                training_loss = event.get("training_loss")
                by_step[step] = {
                    "step": step,
                    "training": self._loss_value(training_loss),
                    "backbone": self._loss_value(training_loss, "backbone"),
                    "depth": self._loss_value(training_loss, "depth"),
                    "validation": self._loss_value(event.get("validation_loss")),
                }

        history = state.get("history", []) if state else []
        if isinstance(history, list):
            for item in history:
                if not isinstance(item, dict):
                    continue
                try:
                    step = int(item["step"])
                except (KeyError, TypeError, ValueError):
                    continue
                training_loss = item.get("training_loss")
                validation_loss = item.get("validation_loss")
                row = by_step.setdefault(step, {"step": step})
                for key, value in (
                    ("training", self._loss_value(training_loss)),
                    ("backbone", self._loss_value(training_loss, "backbone")),
                    ("depth", self._loss_value(training_loss, "depth")),
                ):
                    if value is not None:
                        row[key] = value
                validation = self._loss_value(validation_loss)
                if validation is not None:
                    row["validation"] = validation

        return [by_step[step] for step in sorted(by_step)]

    def list_checkpoints(self, config_path: str | Path) -> list[dict[str, Any]]:
        path = self._workspace_path(config_path)
        config = self._load_normalized(path)
        train = config["train"]
        output_root: Path = train["output-root"]
        max_steps = int(train["max-steps"])
        completed = (output_root / "training-receipt.json").is_file()
        if not output_root.is_dir():
            return []

        rows: list[dict[str, Any]] = []
        for checkpoint in output_root.iterdir():
            match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
            if not checkpoint.is_dir() or not match:
                continue
            step = int(match.group(1))
            missing = [
                name for name in _RESUME_FILES if not (checkpoint / name).is_file()
            ]
            state: dict[str, Any] = {}
            state_path = checkpoint / "trainer-state.json"
            if state_path.is_file():
                try:
                    value = json.loads(state_path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        state = value
                except (OSError, json.JSONDecodeError):
                    missing.append("valid trainer-state.json")
            if int(state.get("global_step", -1)) != step:
                missing.append("matching trainer-state step")
            if not (output_root / "run-config.json").is_file():
                missing.append("run-config.json")
            if completed:
                missing.append("training is already complete")
            if step >= max_steps:
                missing.append("checkpoint step is not below train.max-steps")
            rows.append(
                {
                    "step": step,
                    "path": str(checkpoint.resolve()),
                    "resumable": not missing,
                    "reason": None if not missing else "; ".join(missing),
                }
            )
        return sorted(rows, key=lambda row: row["step"])

    def _output_files(self, config: dict[str, Any]) -> list[str]:
        paths: list[Path] = []
        train = config.get("train", {})
        output_root = train.get("output-root") if isinstance(train, dict) else None
        if isinstance(output_root, Path) and output_root.is_dir():
            for name in ("training-receipt.json", "latest.json", "run-config.json"):
                candidate = output_root / name
                if candidate.is_file():
                    paths.append(candidate)
            paths.extend(
                sorted(
                    child
                    for child in output_root.iterdir()
                    if child.is_dir() and _CHECKPOINT_RE.fullmatch(child.name)
                )
            )
        prepare = config.get("prepare", {})
        cache_root = prepare.get("output-root") if isinstance(prepare, dict) else None
        if (
            isinstance(cache_root, Path)
            and (cache_root / "cache-receipt.json").is_file()
        ):
            paths.append(cache_root / "cache-receipt.json")
        export = config.get("export", {})
        export_path = export.get("output") if isinstance(export, dict) else None
        if isinstance(export_path, Path) and export_path.is_file():
            paths.append(export_path)
        return [str(path.resolve()) for path in paths]

    def _sample_resources(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._resource_cache and now - self._resource_cache[0] < 1.0:
            return dict(self._resource_cache[1])
        try:
            disk_free_gib = round(shutil.disk_usage(self.workspace).free / (1024**3), 2)
        except OSError:
            disk_free_gib = None

        gpu: dict[str, Any] | None = None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=2,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            line = next(row for row in result.stdout.splitlines() if row.strip())
            name, used, total, utilization = (
                part.strip() for part in line.split(",", 3)
            )
            gpu = {
                "name": name,
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_percent": int(utilization),
            }
        except (OSError, subprocess.SubprocessError, ValueError, StopIteration):
            pass
        result = {"disk_free_gib": disk_free_gib, "gpu": gpu}
        self._resource_cache = (now, result)
        return dict(result)

    def snapshot(self, config_path: str | Path | None = None) -> dict[str, Any]:
        with self._lock:
            status = self._refresh_job_locked()
            job = self._job
            if job is not None and status in {"running", "stopping"}:
                selected_path = job.config_path
            elif config_path is not None:
                selected_path = self._workspace_path(config_path)
            else:
                selected_path = job.config_path if job else None

            config: dict[str, Any] = {}
            state: dict[str, Any] | None = None
            if selected_path is not None and selected_path.is_file():
                config = self._load_normalized(selected_path)
                state = self._training_state(selected_path)

            job_matches = job is not None and selected_path == job.config_path
            if job_matches and status in {
                "running",
                "stopping",
                "failed",
                "stopped",
                "paused",
                "completed",
            }:
                current_status = status or "idle"
                active_job = job
            elif state and state.get("pause_reason") == "user_stop":
                current_status, active_job = "paused", None
            elif state and state.get("status") == "paused_for_fresh_process_resume":
                current_status, active_job = "paused", None
            elif (
                isinstance(config.get("train"), dict)
                and isinstance(config["train"].get("output-root"), Path)
                and (config["train"]["output-root"] / "training-receipt.json").is_file()
            ):
                current_status, active_job = "completed", None
            elif state:
                current_status, active_job = "interrupted", None
            else:
                current_status, active_job = ("ready" if config else "idle"), None

            train = config.get("train", {}) if config else {}
            total_steps = (
                int(train.get("stop-after-step", train.get("max-steps", 0)))
                if isinstance(train, dict)
                else 0
            )
            if active_job is not None:
                args = active_job.process.args
                if isinstance(args, (list, tuple)) and "--stop-after-step" in args:
                    index = args.index("--stop-after-step")
                    if index + 1 < len(args):
                        try:
                            total_steps = int(args[index + 1])
                        except (TypeError, ValueError):
                            pass
            step = 0
            if state:
                try:
                    step = int(state.get("global_step", 0))
                except (TypeError, ValueError):
                    step = 0
            elapsed = 0.0
            pid: int | None = None
            returncode: int | None = None
            if active_job is not None:
                end = active_job.ended_at or time.monotonic()
                elapsed = max(0.0, end - active_job.started_at)
                pid = active_job.process.pid
                returncode = active_job.returncode
            elif state:
                runtime = state.get("runtime")
                if isinstance(runtime, dict):
                    try:
                        elapsed = float(
                            runtime.get("elapsed_seconds_this_process", 0.0)
                        )
                    except (TypeError, ValueError):
                        elapsed = 0.0

            console_path = active_job.console_log if active_job else None
            log = (
                self._combined_log(selected_path, active_console=console_path)
                if selected_path
                else ""
            )
            native_log_path = None
            if isinstance(config.get("train"), dict):
                output_root = config["train"].get("output-root")
                if isinstance(output_root, Path):
                    native_log_path = output_root.parent / "统一训练.log"
            metrics = self._metrics(state, native_log_path)
            if (
                metrics
                and active_job is not None
                and active_job.stage in {"train", "all"}
            ):
                step = max(step, max(row["step"] for row in metrics))
            return {
                "job_id": active_job.job_id if active_job else None,
                "status": current_status,
                "stage": active_job.stage if active_job else None,
                "dry_run": active_job.dry_run if active_job else False,
                "config": self._display_path(selected_path) if selected_path else None,
                "log": log,
                "step": step,
                "total_steps": total_steps,
                "metrics": metrics,
                "elapsed": round(elapsed, 1),
                "pid": pid,
                "returncode": returncode,
                "output_files": self._output_files(config) if config else [],
                "resources": self._sample_resources(),
            }

    def start(
        self,
        config_path: str | Path,
        stage: str = "all",
        resume: str | Path | None = None,
        dry_run: bool = False,
        runtime: dict[str, Any] | None = None,
    ) -> str:
        if stage not in {"prepare", "train", "export", "all"}:
            raise ValueError(f"未知训练阶段：{stage}")
        path = self._workspace_path(config_path)
        if path.name != "training-config.json" or not path.is_file():
            raise ValueError("请选择存在的 training-config.json")
        config = self._load_normalized(path)
        if config.get("historical_only") and not dry_run:
            raise ValueError("历史配方仅可使用 dry_run 查看，不能运行")
        resume_path = self._workspace_path(resume) if resume is not None else None

        with self._lock:
            current_status = self._refresh_job_locked()
            if current_status in {"running", "stopping"}:
                raise RuntimeError("已有训练任务正在运行")

            output_root: Path = config["train"]["output-root"]
            stop_marker = self._stop_marker(output_root)
            if stop_marker.is_file():
                stop_marker.unlink()

            console_log = path.parent / "统一WebUI控制台.log"
            command = self._build_command(path, stage, resume_path, dry_run, runtime)
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            source_path = str(self.source_repo)
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                [source_path, existing] if existing else [source_path]
            )

            console_log.parent.mkdir(parents=True, exist_ok=True)
            with console_log.open("ab") as output:
                output.write(
                    f"\n=== WebUI {time.strftime('%Y-%m-%d %H:%M:%S')} {stage} ===\n".encode(
                        "utf-8"
                    )
                )
                output.flush()
                process = subprocess.Popen(
                    command,
                    cwd=self.workspace,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )

            job_id = uuid.uuid4().hex
            self._job = _Job(
                job_id=job_id,
                process=process,
                config_path=path,
                stage=stage,
                console_log=console_log,
                started_at=time.monotonic(),
                resume=resume_path,
                dry_run=dry_run,
            )
            return job_id

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
        try:
            import psutil

            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                pass
            _, alive = psutil.wait_procs(children + [parent], timeout=5)
            for item in alive:
                try:
                    item.kill()
                except psutil.NoSuchProcess:
                    pass
        except ImportError:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            elif process.poll() is None:
                process.terminate()

    def stop(self, force: bool = False) -> str:
        with self._lock:
            status = self._refresh_job_locked()
            job = self._job
            if job is None or status not in {"running", "stopping"}:
                return "当前没有运行中的任务"
            if force:
                job.requested_stop = "force"
                self._terminate_tree(job.process)
                try:
                    job.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    job.process.kill()
                    job.process.wait(timeout=5)
                self._refresh_job_locked()
                return "已强制停止本任务的进程树"

            config = self._load_normalized(job.config_path)
            output_root: Path = config["train"]["output-root"]
            marker = self._stop_marker(output_root)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("user_stop\n", encoding="utf-8")
            job.requested_stop = "graceful"
            return "已请求安全暂停；训练将在当前步骤结束后保存可续训检查点"
