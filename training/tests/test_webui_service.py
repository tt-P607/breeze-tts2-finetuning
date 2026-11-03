from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.workflow.webui_service import JobManager


def batch_config(workspace: Path, batch: Path) -> dict[str, object]:
    return {
        "train": {
            "cache-root": str((batch / "cache").relative_to(workspace)),
            "output-root": str((batch / "output").relative_to(workspace)),
        }
    }


class WebUIServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.batch = self.workspace / "角色模型" / "测试角色" / "训练批次" / "测试批次"
        self.config_path = self.batch / "training-config.json"
        self.manager = JobManager(self.workspace)
        self.manager._sample_resources = lambda: {"disk_free_gib": 1.0, "gpu": None}  # type: ignore[method-assign]
        self.manager.save_config(
            self.config_path, batch_config(self.workspace, self.batch)
        )

    def _run_dummy(self, code: str) -> None:
        command = [sys.executable, "-u", "-c", code]
        with patch.object(self.manager, "_build_command", return_value=command):
            self.manager.start(self.config_path, stage="train")

    def _wait_for_status(self, wanted: str, timeout: float = 8.0) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.manager.snapshot()
            if snapshot["status"] == wanted:
                return snapshot
            time.sleep(0.05)
        self.fail(f"任务未在 {timeout} 秒内进入状态 {wanted!r}")

    def test_list_read_and_save_config_protects_trained_batch(self) -> None:
        relative = self.config_path.relative_to(self.workspace).as_posix()
        self.assertIn(relative, self.manager.list_configs())
        normalized = self.manager.read_config(relative)
        self.assertEqual(normalized["train"]["max-steps"], 300)
        self.assertIs(normalized["historical_only"], False)
        self.manager.save_config(relative, normalized)
        self.assertIs(self.manager.read_config(relative)["historical_only"], False)

        output_root = self.batch / "output"
        output_root.mkdir()
        (output_root / "training-receipt.json").write_text("{}", encoding="utf-8")
        changed = batch_config(self.workspace, self.batch)
        changed["train"]["learning-rate"] = 0.0002
        with self.assertRaisesRegex(ValueError, "已有训练输出"):
            self.manager.save_config(self.config_path, changed)

    def test_weight_only_checkpoint_is_not_resumable_and_metrics_are_parsed(
        self,
    ) -> None:
        output_root = self.batch / "output"
        checkpoint = output_root / "checkpoint-step-000025"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter.safetensors").write_bytes(b"weights only")
        state = {
            "global_step": 25,
            "status": "paused_for_fresh_process_resume",
            "history": [
                {
                    "step": 25,
                    "training_loss": {
                        "total": 1.25,
                        "backbone": 0.75,
                        "depth_decoder": 0.5,
                    },
                    "validation_loss": {
                        "total": 1.5,
                        "backbone": 0.9,
                        "depth_decoder": 0.6,
                    },
                }
            ],
        }
        (checkpoint / "trainer-state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        (output_root / "latest.json").write_text(
            json.dumps({"checkpoint": str(checkpoint), "global_step": 25}),
            encoding="utf-8",
        )
        log_path = output_root.parent / "统一训练.log"
        log_path.write_text(
            "\n".join(
                (
                    json.dumps(
                        {
                            "event": "training_step",
                            "step": 25,
                            "training_loss": {
                                "total": 1.1,
                                "backbone": 0.7,
                                "depth_decoder": 0.4,
                            },
                            "validation_loss": {"total": 99.0},
                        }
                    ),
                    json.dumps(
                        {
                            "event": "training_step",
                            "step": 26,
                            "training_loss": {
                                "total": 1.0,
                                "backbone": 0.6,
                                "depth_decoder": 0.4,
                            },
                        }
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        rows = self.manager.list_checkpoints(self.config_path)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["resumable"])
        self.assertIn("optimizer.pt", rows[0]["reason"])

        snapshot = self.manager.snapshot(self.config_path)
        self.assertEqual(snapshot["status"], "paused")
        self.assertEqual(snapshot["step"], 25)
        self.assertEqual(snapshot["total_steps"], 250)
        self.assertEqual(
            snapshot["metrics"],
            [
                {
                    "step": 25,
                    "training": 1.25,
                    "backbone": 0.75,
                    "depth": 0.5,
                    "validation": 1.5,
                },
                {
                    "step": 26,
                    "training": 1.0,
                    "backbone": 0.6,
                    "depth": 0.4,
                    "validation": None,
                },
            ],
        )

    def test_only_one_dummy_job_can_run_and_force_stop_kills_its_tree(self) -> None:
        pid_file = self.workspace / "child.pid"
        child_code = "import time; time.sleep(60)"
        parent_code = (
            "import subprocess,sys,time; "
            f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
            f"open({str(pid_file)!r},'w').write(str(child.pid)); "
            "time.sleep(60)"
        )
        command = [sys.executable, "-u", "-c", parent_code]
        with patch.object(self.manager, "_build_command", return_value=command):
            self.manager.start(self.config_path, stage="train")
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(pid_file.is_file())
            with self.assertRaisesRegex(RuntimeError, "已有训练任务"):
                self.manager.start(self.config_path, stage="train")

            child_pid = int(pid_file.read_text(encoding="utf-8"))
            self.manager.stop(force=True)

        try:
            import psutil

            self.assertFalse(psutil.pid_exists(child_pid))
        except ImportError:
            pass
        self.assertEqual(self.manager.snapshot()["status"], "stopped")

    def test_failed_dummy_command_exposes_cli_output(self) -> None:
        output_root = self.batch / "output"
        old_checkpoint = output_root / "checkpoint-step-000250"
        old_checkpoint.mkdir(parents=True)
        (old_checkpoint / "trainer-state.json").write_text(
            json.dumps(
                {
                    "global_step": 250,
                    "status": "paused_for_fresh_process_resume",
                }
            ),
            encoding="utf-8",
        )
        (output_root / "latest.json").write_text(
            json.dumps({"checkpoint": str(old_checkpoint), "global_step": 250}),
            encoding="utf-8",
        )
        self._run_dummy("print('dummy failure', flush=True); raise SystemExit(7)")
        snapshot = self._wait_for_status("failed")
        self.assertIn("dummy failure", snapshot["log"])
        self.assertEqual(snapshot["returncode"], 7)

    def test_safe_stop_writes_sibling_request_file(self) -> None:
        command = [sys.executable, "-u", "-c", "import time; time.sleep(60)"]
        marker = (self.batch / "output").with_name("output.stop-request")
        marker.write_text("stale\n", encoding="utf-8")
        with patch.object(self.manager, "_build_command", return_value=command):
            self.manager.start(self.config_path, stage="train")
            self.assertFalse(marker.exists())
            self.manager.stop(force=False)
            self.assertEqual(marker.read_text(encoding="utf-8"), "user_stop\n")
            self.manager.stop(force=True)

    def test_runtime_can_override_only_the_resume_pause_target(self) -> None:
        command = self.manager._build_command(
            self.config_path.resolve(),
            "train",
            (self.batch / "output" / "checkpoint-step-000250").resolve(),
            False,
            {
                "stop_after_step": 275,
                "export_checkpoint": "角色模型/测试角色/训练批次/测试批次/output/checkpoint-step-000250",
                "export_output": "角色模型/测试角色/训练批次/测试批次/model.gguf",
            },
        )
        self.assertIn("--resume", command)
        self.assertIn("--stop-after-step", command)
        self.assertEqual(command[command.index("--stop-after-step") + 1], "275")
        self.assertEqual(command[command.index("--stage") + 1], "train")
        self.assertIn("--export-checkpoint", command)
        self.assertEqual(
            command[command.index("--export-checkpoint") + 1],
            "角色模型/测试角色/训练批次/测试批次/output/checkpoint-step-000250",
        )
        self.assertIn("--export-output", command)

    def test_snapshot_uses_active_runtime_pause_target(self) -> None:
        command = [
            sys.executable,
            "-u",
            "-c",
            "import time; time.sleep(60)",
            "--stop-after-step",
            "275",
        ]
        with patch.object(self.manager, "_build_command", return_value=command):
            self.manager.start(
                self.config_path,
                stage="train",
                runtime={"stop_after_step": 275},
            )
            self.assertEqual(self.manager.snapshot()["total_steps"], 275)
            self.manager.stop(force=True)

    def test_successful_export_does_not_reuse_paused_training_status(self) -> None:
        checkpoint = self.batch / "output" / "checkpoint-step-000250"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer-state.json").write_text(
            json.dumps(
                {"global_step": 250, "status": "paused_for_fresh_process_resume"}
            ),
            encoding="utf-8",
        )
        with patch.object(
            self.manager, "_build_command", return_value=[sys.executable, "-c", "pass"]
        ):
            self.manager.start(self.config_path, stage="export")
        self.assertEqual(self._wait_for_status("completed")["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
