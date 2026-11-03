from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .commands import PlannedCommand
from .monitor import DISK_LIMIT_GIB, ResourceMonitor, append_jsonl


@dataclass(frozen=True)
class StageResult:
    returncode: int
    paused: bool = False
    checkpoint: Path | None = None


def child_environment(
    *,
    training_repo: Path,
    gguf_deps: Path,
    batch_root: Path,
) -> dict[str, str]:
    """Keep offline, UTF-8, CUDA, and process-scoped Windows Git settings."""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "6",
            "MKL_NUM_THREADS": "6",
            "NUMBA_CACHE_DIR": str(batch_root / "numba-cache"),
            "NUMBA_CACHE_LOCATOR_CLASSES": "numba.core.caching.UserProvidedCacheLocator",
            "TEMP": str(batch_root / "tmp"),
            "TMP": str(batch_root / "tmp"),
        }
    )
    existing = env.get("PYTHONPATH", "")
    paths = [str(training_repo), str(gguf_deps)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    try:
        git_config_count = max(0, int(env.get("GIT_CONFIG_COUNT", "0")))
    except ValueError:
        git_config_count = 0
    env["GIT_CONFIG_COUNT"] = str(git_config_count + 1)
    env[f"GIT_CONFIG_KEY_{git_config_count}"] = "safe.directory"
    env[f"GIT_CONFIG_VALUE_{git_config_count}"] = str(training_repo).replace("\\", "/")
    return env


def _disk_free_gib(root: Path) -> float:
    return shutil.disk_usage(root).free / (1024**3)


def _append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(message.rstrip() + "\n")
        log.flush()


def _terminate_tree(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    try:
        import psutil

        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=10)
        for child in alive:
            child.kill()
    except Exception:
        if process.poll() is None:
            process.terminate()


def _worker_pid(process: subprocess.Popen[str]) -> int:
    try:
        import psutil

        children = psutil.Process(process.pid).children(recursive=True)
        return children[-1].pid if children else process.pid
    except Exception:
        return process.pid


def _checkpoint_from_latest(output_root: Path | None) -> Path | None:
    if output_root is None:
        return None
    latest_path = output_root / "latest.json"
    if not latest_path.is_file():
        return None
    try:
        value = json.loads(latest_path.read_text(encoding="utf-8"))
        return Path(value["checkpoint"])
    except (OSError, KeyError, json.JSONDecodeError, TypeError):
        return None


def run_command(
    command: PlannedCommand,
    *,
    workspace: Path,
    python: Path,
    training_repo: Path,
    gguf_deps: Path,
    log_path: Path,
    resource_log_path: Path,
) -> StageResult:
    if command.uses_gpu:
        try:
            import psutil  # noqa: F401 - required for worker PID and process-tree monitoring
        except ImportError as exc:
            raise RuntimeError(
                "GPU resource monitoring requires psutil in the training venv"
            ) from exc
    batch_root = log_path.parent
    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "numba-cache").mkdir(exist_ok=True)
    (batch_root / "tmp").mkdir(exist_ok=True)
    free_gib = _disk_free_gib(workspace)
    if free_gib < DISK_LIMIT_GIB:
        raise RuntimeError(
            f"E: free space below {DISK_LIMIT_GIB:.0f} GiB before {command.stage}: {free_gib:.2f} GiB"
        )

    env = child_environment(
        training_repo=training_repo, gguf_deps=gguf_deps, batch_root=batch_root
    )
    _append_log(
        log_path,
        f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {command.stage} ===\n"
        + subprocess.list2cmdline(list(command.argv)),
    )
    log_stream = log_path.open("a", encoding="utf-8")
    monitor: ResourceMonitor | None = None
    stop_reason: str | None = None
    try:
        process = subprocess.Popen(
            command.argv,
            cwd=command.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if command.uses_gpu:
            time.sleep(3)
            monitor = ResourceMonitor(
                _worker_pid(process), command.stage, resource_log_path
            )
            monitor.start()
        last_disk_check = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - last_disk_check >= 5:
                free_gib = _disk_free_gib(workspace)
                if free_gib < DISK_LIMIT_GIB:
                    stop_reason = f"E: free space fell below {DISK_LIMIT_GIB:.0f} GiB: {free_gib:.2f} GiB"
                    append_jsonl(
                        resource_log_path,
                        {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "stage": command.stage,
                            "event": "disk_protective_stop",
                            "reason": stop_reason,
                        },
                    )
                    _terminate_tree(process)
                    break
                last_disk_check = time.monotonic()
            if monitor and monitor.stop_reason:
                stop_reason = monitor.stop_reason
                _terminate_tree(process)
                break
            time.sleep(1)
        try:
            returncode = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _terminate_tree(process)
            returncode = process.wait(timeout=15)
    finally:
        if monitor is not None:
            telemetry = monitor.stop_and_collect()
            append_jsonl(
                resource_log_path,
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "stage": command.stage,
                    "event": "stage_summary",
                    **telemetry,
                },
            )
        log_stream.close()
    if stop_reason:
        raise RuntimeError(
            f"{command.stage} stopped by resource protection: {stop_reason}"
        )
    if returncode == 75 and command.stage == "train":
        checkpoint = _checkpoint_from_latest(command.output_root)
        if checkpoint is None or not (checkpoint / "checkpoint-receipt.json").is_file():
            raise RuntimeError(
                "training returned resumable status 75 without a complete latest checkpoint"
            )
        try:
            state = json.loads(
                (checkpoint / "trainer-state.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot read paused checkpoint state: {checkpoint}"
            ) from exc
        if state.get("status") != "paused_for_fresh_process_resume":
            raise RuntimeError(
                f"training returned 75 with unexpected checkpoint state: {state.get('status')}"
            )
        return StageResult(returncode=returncode, paused=True, checkpoint=checkpoint)
    if returncode != 0:
        tail = ""
        try:
            tail = "".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines(
                    keepends=True
                )[-35:]
            )
        except OSError:
            pass
        raise RuntimeError(
            f"{command.stage} exited with code {returncode}. Last log lines:\n{tail[-3500:]}"
        )
    return StageResult(returncode=returncode)
