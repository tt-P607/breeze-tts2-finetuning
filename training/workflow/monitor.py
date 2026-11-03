from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

WARNING_MIB = (
    11469  # Informational threshold only; the historical policy does not stop here.
)
DISK_LIMIT_GIB = 8.0
_FILE_LOCK_GUARD = threading.Lock()
_FILE_LOCKS: dict[Path, threading.Lock] = {}


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    line = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    key = path.resolve(strict=False)
    with _FILE_LOCK_GUARD:
        lock = _FILE_LOCKS.setdefault(key, threading.Lock())
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


class ResourceMonitor:
    """Windows GPU monitor copied into the unified tools, independent of old batch folders."""

    def __init__(
        self, pid: int, label: str, resource_log: Path, *, protect: bool = True
    ) -> None:
        self.pid = pid
        self.label = label
        self.resource_log = resource_log
        self.protect = protect
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.gpu: list[dict[str, Any]] = []
        self.shared: list[dict[str, Any]] = []
        self.stop_reason: str | None = None
        self.warning_crossed = False
        self.shared_mib: float | None = None
        self.shared_time = 0.0
        self.pressure_since: float | None = None
        self.gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True)
        self.shared_thread = threading.Thread(target=self._shared_loop, daemon=True)
        self.shared_process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self._start_shared_counter()
        self.gpu_thread.start()
        self.shared_thread.start()

    def _start_shared_counter(self) -> None:
        ps = (
            "$ErrorActionPreference='SilentlyContinue';"
            "$ProgressPreference='SilentlyContinue';"
            f"$filter='pid_{self.pid}_*';"
            "while ($true) {"
            "try {"
            "$s=Get-Counter -Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction Stop;"
            "$v=@($s.CounterSamples | Where-Object {$_.InstanceName -like $filter} | Measure-Object -Property CookedValue -Sum).Sum;"
            "if ($null -eq $v) {[Console]::WriteLine('0')} else {[Console]::WriteLine(([long]$v).ToString([Globalization.CultureInfo]::InvariantCulture))}"
            "} catch {[Console]::WriteLine('ERR')}"
            "Start-Sleep -Seconds 10"
            "}"
        )
        try:
            self.shared_process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    ps,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self.shared_process = None

    def _shared_loop(self) -> None:
        process = self.shared_process
        if process is None or process.stdout is None:
            return
        while not self.stop.is_set():
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    return
                continue
            try:
                value: float | None = max(0, int(line.strip())) / (1024**2)
            except ValueError:
                value = None
            now = time.monotonic()
            with self.lock:
                self.shared_mib = value
                self.shared_time = now
                self.shared.append({"t": time.time(), "shared_mib": value})
            append_jsonl(
                self.resource_log,
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "process": self.label,
                    "pid": self.pid,
                    "shared_gpu_mib": value,
                },
            )

    def _gpu_loop(self) -> None:
        warned = False
        while not self.stop.wait(2.0):
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--id=0",
                        "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                used, total, util = [
                    int(value.strip())
                    for value in completed.stdout.splitlines()[0].split(",")
                ]
            except Exception:
                continue
            now = time.monotonic()
            with self.lock:
                shared = self.shared_mib if now - self.shared_time <= 20 else None
                self.gpu.append(
                    {
                        "t": time.time(),
                        "used_mib": used,
                        "total_mib": total,
                        "util_percent": util,
                        "process_shared_mib": shared,
                    }
                )
                if used > WARNING_MIB:
                    warned = True
                if (
                    self.protect
                    and used >= total - 128
                    and shared is not None
                    and shared >= 1024
                ):
                    if self.pressure_since is None:
                        self.pressure_since = now
                    elif now - self.pressure_since >= 20 and self.stop_reason is None:
                        self.stop_reason = (
                            f"GPU nearly full ({used}/{total} MiB) with sustained training-process "
                            f"shared GPU memory ({shared:.0f} MiB)"
                        )
                        append_jsonl(
                            self.resource_log,
                            {
                                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "process": self.label,
                                "event": "protective_stop",
                                "reason": self.stop_reason,
                            },
                        )
                        try:
                            os.kill(self.pid, signal.SIGTERM)
                        except OSError:
                            pass
                else:
                    self.pressure_since = None
                self.warning_crossed = self.warning_crossed or warned

    def stop_and_collect(self) -> dict[str, Any]:
        self.stop.set()
        if self.shared_process is not None:
            try:
                self.shared_process.terminate()
                self.shared_process.wait(timeout=5)
            except Exception:
                try:
                    self.shared_process.kill()
                except Exception:
                    pass
        self.gpu_thread.join(timeout=5)
        self.shared_thread.join(timeout=5)
        with self.lock:
            gpu = list(self.gpu)
            shared_values = [
                row["shared_mib"]
                for row in self.shared
                if row["shared_mib"] is not None
            ]
            reason = self.stop_reason
            crossed = self.warning_crossed
        return {
            "samples": len(gpu),
            "gpu_used_peak_mib": max((row["used_mib"] for row in gpu), default=None),
            "gpu_total_mib": gpu[-1]["total_mib"] if gpu else None,
            "gpu_util_mean_percent": round(
                sum(row["util_percent"] for row in gpu) / len(gpu), 2
            )
            if gpu
            else None,
            "gpu_util_peak_percent": max(
                (row["util_percent"] for row in gpu), default=None
            ),
            "training_process_shared_peak_mib": max(shared_values, default=None),
            "warning_crossed_11_2_gib": crossed,
            "protective_stop_reason": reason,
        }
