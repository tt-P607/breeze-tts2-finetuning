from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PREPARATION = Path(__file__).resolve().parents[2]
if str(PREPARATION) not in sys.path:
    sys.path.insert(0, str(PREPARATION))

from training.workflow import cli as train
from training.workflow.commands import PlanError, build_plan, runtime_paths
from training.workflow.config import load_config
from training.workflow.runner import child_environment


def make_workspace() -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
    temporary = tempfile.TemporaryDirectory()
    workspace = Path(temporary.name)
    prep = workspace / "training-preparation"
    python = prep / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test interpreter placeholder")
    model_root = prep / "base-model"
    model_root.mkdir(parents=True)
    gguf_deps = prep / "gguf-export" / "deps"
    gguf_deps.mkdir(parents=True)
    repo = workspace / "breeze-tts2-finetuning"
    (repo / "training").mkdir(parents=True)
    (repo / "training" / "real_data.py").write_text("", encoding="utf-8")
    (repo / "training" / "real_lora.py").write_text("", encoding="utf-8")
    converter = workspace / "Breeze-TTS-2.cpp" / "scripts" / "convert_lora_to_gguf.py"
    converter.parent.mkdir(parents=True)
    converter.write_text("", encoding="utf-8")
    batch = workspace / "datasets" / "new-batch"
    batch.mkdir(parents=True)
    (batch / "train.jsonl").write_text('{"audio":"train.wav","text":"a"}\n', encoding="utf-8")
    (batch / "validation.jsonl").write_text('{"audio":"valid.wav","text":"b"}\n', encoding="utf-8")
    return temporary, workspace, batch


def write_config(workspace: Path, batch: Path, *, historical: bool = False) -> Path:
    train: dict[str, object] = {
        "cache-root": str((batch / "cache").relative_to(workspace)),
        "output-root": str((batch / "checkpoints").relative_to(workspace)),
    }
    if historical:
        train["max-steps"] = 300
        train["stop-after-step"] = 250
    config = {
        "prepare": {
            "train-manifest": str((batch / "train.jsonl").relative_to(workspace)),
            "validation-manifest": str((batch / "validation.jsonl").relative_to(workspace)),
            "output-root": str((batch / "cache").relative_to(workspace)),
            "train-limit": 1,
            "validation-limit": 1,
        },
        "train": train,
        "export": {
            "checkpoint": str((batch / "checkpoints" / f"checkpoint-step-{250:06d}").relative_to(workspace)),
            "output": str((batch / "model.gguf").relative_to(workspace)),
            "alpha": 32,
        },
    }
    path = workspace / "training-preparation" / "example.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def write_train_only_config(workspace: Path, batch: Path, *, warmup_steps: int = 0) -> Path:
    config = {
        "train": {
            "cache-root": str((batch / "cache").relative_to(workspace)),
            "output-root": str((batch / "checkpoints").relative_to(workspace)),
            "warmup-steps": warmup_steps,
        }
    }
    path = workspace / "training-preparation" / "train-only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class UnifiedTrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary, self.workspace, self.batch = make_workspace()
        self.addCleanup(self.temporary.cleanup)
        self.runtime = runtime_paths(self.workspace)

    def _config(self, *, historical: bool = False):
        path = write_config(self.workspace, self.batch, historical=historical)
        return load_config(path, self.workspace)

    def test_relative_paths_defaults_and_command_plan(self) -> None:
        config = self._config()
        self.assertEqual(config["prepare"]["train-manifest"], self.batch / "train.jsonl")
        self.assertEqual(config["train"]["rank"], 16)
        self.assertEqual(config["train"]["alpha"], 32.0)
        self.assertEqual(config["train"]["max-steps"], 300)
        self.assertEqual(config["train"]["stop-after-step"], 250)
        self.assertEqual(config["train"]["save-every"], 25)
        self.assertEqual(config["train"]["learning-rate"], 0.0001)
        self.assertEqual(config["train"]["gradient-accumulation"], 4)
        self.assertEqual(config["train"]["warmup-steps"], 100)
        self.assertEqual(config["train"]["seed"], 42)

        plan = build_plan(config, stage="all", resume=None, paths=self.runtime)
        self.assertEqual([item.stage for item in plan.commands], ["prepare", "train", "export"])
        self.assertEqual(plan.commands[0].cwd, self.runtime.training_repo)
        self.assertIn("training.real_data", plan.commands[0].argv)
        self.assertIn("training.real_lora", plan.commands[1].argv)
        self.assertIn("16", plan.commands[1].argv)
        self.assertIn("0.0001", plan.commands[1].argv)
        self.assertEqual(plan.commands[1].argv[plan.commands[1].argv.index("--max-steps") + 1], "300")
        self.assertEqual(plan.commands[1].argv[plan.commands[1].argv.index("--stop-after-step") + 1], "250")
        self.assertIn(str(self.runtime.converter), plan.commands[2].argv)
        self.assertNotIn("--force", plan.commands[2].argv)
        self.assertEqual(plan.log_path, self.batch / "统一训练.log")
        self.assertEqual(plan.export_checkpoint, self.batch / "checkpoints" / "checkpoint-step-000250")
        self.assertFalse((self.batch / "cache").exists())
        self.assertFalse((self.batch / "checkpoints").exists())
        self.assertFalse(plan.log_path.exists())

    def test_historical_max_steps_and_pause_are_explicit(self) -> None:
        config = self._config(historical=True)
        plan = build_plan(config, stage="all", resume=None, paths=self.runtime)
        train_command = next(item for item in plan.commands if item.stage == "train")
        self.assertIn("300", train_command.argv)
        self.assertIn("250", train_command.argv)
        self.assertIn("--stop-after-step", train_command.argv)
        self.assertTrue(str(plan.export_checkpoint).endswith("checkpoint-step-000250"))

    def test_explicit_historical_max_steps_do_not_inherit_default_pause(self) -> None:
        config_path = write_config(self.workspace, self.batch, historical=True)
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        raw["train"]["max-steps"] = 1500
        raw["train"].pop("stop-after-step")
        config_path.write_text(json.dumps(raw), encoding="utf-8")
        config = load_config(config_path, self.workspace)
        self.assertNotIn("stop-after-step", config["train"])

        cache_root: Path = config["train"]["cache-root"]
        cache_root.mkdir()
        (cache_root / "cache-receipt.json").write_text(
            json.dumps({"status": "deterministic_supervised_cache_complete"}),
            encoding="utf-8",
        )
        plan = build_plan(config, stage="train", resume=None, paths=self.runtime)
        train_command = plan.commands[0]
        self.assertEqual(train_command.argv[train_command.argv.index("--max-steps") + 1], "1500")
        self.assertNotIn("--stop-after-step", train_command.argv)

    def test_child_environment_keeps_offline_and_scoped_git_settings(self) -> None:
        env = child_environment(
            training_repo=self.runtime.training_repo,
            gguf_deps=self.runtime.gguf_deps,
            batch_root=self.batch,
        )
        self.assertEqual(env["PYTHONUTF8"], "1")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["NUMBA_CACHE_LOCATOR_CLASSES"], "numba.core.caching.UserProvidedCacheLocator")
        config_index = int(env["GIT_CONFIG_COUNT"]) - 1
        self.assertEqual(env[f"GIT_CONFIG_KEY_{config_index}"], "safe.directory")
        self.assertEqual(env[f"GIT_CONFIG_VALUE_{config_index}"], str(self.runtime.training_repo).replace("\\", "/"))
        self.assertIn(str(self.runtime.gguf_deps), env["PYTHONPATH"].split(os.pathsep))

    def test_cli_dry_run_does_not_import_torch_or_create_outputs(self) -> None:
        config_path = write_config(self.workspace, self.batch)
        output = io.StringIO()
        with patch.object(train, "WORKSPACE_ROOT", self.workspace), contextlib.redirect_stdout(output):
            code = train.main(["--config", str(config_path.relative_to(self.workspace)), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", output.getvalue())
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import training.workflow.dataset; "
                "import training.workflow.webui_service; "
                "assert 'torch' not in sys.modules",
            ],
            cwd=PREPARATION,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertFalse((self.batch / "cache").exists())
        self.assertFalse((self.batch / "checkpoints").exists())
        self.assertFalse((self.batch / "统一训练.log").exists())
        self.assertFalse((self.batch / "统一资源监测.jsonl").exists())

    def test_train_only_config_all_runs_train_with_existing_cache(self) -> None:
        config_path = write_train_only_config(self.workspace, self.batch, warmup_steps=0)
        config = load_config(config_path, self.workspace)
        self.assertNotIn("prepare", config)
        self.assertNotIn("export", config)
        self.assertEqual(config["train"]["warmup-steps"], 0)
        cache_root: Path = config["train"]["cache-root"]
        cache_root.mkdir()
        (cache_root / "cache-receipt.json").write_text(
            json.dumps({"status": "deterministic_supervised_cache_complete"}),
            encoding="utf-8",
        )

        plan = build_plan(config, stage="all", resume=None, paths=self.runtime)
        self.assertEqual([item.stage for item in plan.commands], ["train"])
        self.assertIn("--warmup-steps", plan.commands[0].argv)
        self.assertIn("0", plan.commands[0].argv)

    def test_train_only_requires_cache_and_reports_explicit_missing_stages(self) -> None:
        config_path = write_train_only_config(self.workspace, self.batch)
        config = load_config(config_path, self.workspace)
        with self.assertRaisesRegex(PlanError, "prepared cache receipt"):
            build_plan(config, stage="all", resume=None, paths=self.runtime)
        with self.assertRaisesRegex(PlanError, "requires a 'prepare' section"):
            build_plan(config, stage="prepare", resume=None, paths=self.runtime)
        with self.assertRaisesRegex(PlanError, "requires a 'export' section"):
            build_plan(config, stage="export", resume=None, paths=self.runtime)

    def test_completed_outputs_are_visible_in_dry_run_and_block_execution(self) -> None:
        config = self._config()
        prepare_root: Path = config["prepare"]["output-root"]
        prepare_root.mkdir()
        (prepare_root / "cache-receipt.json").write_text(
            json.dumps({"status": "deterministic_supervised_cache_complete"}),
            encoding="utf-8",
        )
        output_root: Path = config["train"]["output-root"]
        checkpoint = output_root / "checkpoint-step-000250"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
        output: Path = config["export"]["output"]
        output.write_bytes(b"existing gguf")

        plan = build_plan(config, stage="all", resume=None, paths=self.runtime, dry_run=True)
        self.assertEqual([item.stage for item in plan.commands], ["prepare", "train", "export"])
        self.assertEqual(len(plan.warnings), 3)
        with self.assertRaisesRegex(PlanError, "refusing to overwrite an existing cache"):
            build_plan(config, stage="all", resume=None, paths=self.runtime)

    def test_historical_only_recipe_is_inspectable_but_not_executable(self) -> None:
        config_path = write_config(self.workspace, self.batch)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data["historical_only"] = True
        data["initialization"] = {
            "mode": "custom_init_checkpoint",
            "checkpoint": "已归档/缺失的checkpoint",
            "restored": ["adapter", "optimizer", "rng"],
            "scheduler": "reinitialized",
        }
        data["notes"] = ["历史初始化不等价于统一续训。"]
        data["prepare"]["train-manifest"] = "已归档/缺失/train.jsonl"
        data["prepare"]["validation-manifest"] = "已归档/缺失/validation.jsonl"
        config_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        output = io.StringIO()
        with patch.object(train, "WORKSPACE_ROOT", self.workspace), contextlib.redirect_stdout(output):
            code = train.main(["--config", str(config_path.relative_to(self.workspace)), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("HISTORICAL ONLY", output.getvalue())
        self.assertIn("custom_init_checkpoint", output.getvalue())
        self.assertIn("reinitialized", output.getvalue())

        errors = io.StringIO()
        with patch.object(train, "WORKSPACE_ROOT", self.workspace), contextlib.redirect_stderr(errors):
            code = train.main(["--config", str(config_path.relative_to(self.workspace))])
        self.assertEqual(code, 2)
        self.assertIn("archive-only", errors.getvalue())

    def test_cli_stop_target_accepts_schedule_limit_and_rejects_invalid_values(self) -> None:
        config_path = write_config(self.workspace, self.batch)
        original = config_path.read_bytes()
        cache_root = self.batch / "cache"
        cache_root.mkdir()
        (cache_root / "cache-receipt.json").write_text(
            json.dumps({"status": "deterministic_supervised_cache_complete"}),
            encoding="utf-8",
        )
        relative_config = str(config_path.relative_to(self.workspace))

        for target in (275, 300):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = train.main(
                    [
                        "--workspace-root",
                        str(self.workspace),
                        "--config",
                        relative_config,
                        "--stage",
                        "train",
                        "--stop-after-step",
                        str(target),
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            if target == 275:
                self.assertIn("--stop-after-step 275", output.getvalue())
            else:
                self.assertNotIn("--stop-after-step", output.getvalue())
            self.assertEqual(config_path.read_bytes(), original)

        for target in (0, 301):
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                code = train.main(
                    [
                        "--workspace-root",
                        str(self.workspace),
                        "--config",
                        relative_config,
                        "--stage",
                        "train",
                        "--stop-after-step",
                        str(target),
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("stop point must be positive", errors.getvalue())
            self.assertEqual(config_path.read_bytes(), original)

    def test_cli_export_overrides_do_not_rewrite_batch_config(self) -> None:
        config_path = write_config(self.workspace, self.batch)
        original = config_path.read_bytes()
        checkpoint = self.batch / "checkpoints" / "checkpoint-step-000100"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
        output_path = self.batch / "selected-export.gguf"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = train.main(
                [
                    "--workspace-root",
                    str(self.workspace),
                    "--config",
                    str(config_path.relative_to(self.workspace)),
                    "--stage",
                    "export",
                    "--export-checkpoint",
                    str(checkpoint.relative_to(self.workspace)),
                    "--export-output",
                    str(output_path.relative_to(self.workspace)),
                    "--dry-run",
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn(str(checkpoint / "adapter.safetensors"), output.getvalue())
        self.assertIn(str(output_path), output.getvalue())
        self.assertEqual(config_path.read_bytes(), original)
        self.assertFalse(output_path.exists())

    def test_stop_marker_after_prepare_skips_next_gpu_stage(self) -> None:
        config_path = write_config(self.workspace, self.batch)
        config = load_config(config_path, self.workspace)
        output_root: Path = config["train"]["output-root"]
        stop_marker = output_root.with_name(output_root.name + ".stop-request")
        stages: list[str] = []

        def fake_run(command, **_kwargs):
            stages.append(command.stage)
            if command.stage != "prepare":
                raise AssertionError(f"unexpected stage ran: {command.stage}")
            stop_marker.write_text("user_stop\n", encoding="utf-8")
            return SimpleNamespace(paused=False)

        output = io.StringIO()
        with (
            patch.object(train, "run_command", side_effect=fake_run),
            contextlib.redirect_stdout(output),
        ):
            code = train.main(
                [
                    "--workspace-root",
                    str(self.workspace),
                    "--config",
                    str(config_path.relative_to(self.workspace)),
                    "--stage",
                    "all",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(stages, ["prepare"])
        self.assertIn("已在阶段边界暂停", output.getvalue())
        self.assertFalse(stop_marker.exists())
        self.assertFalse(output_root.exists())

    def test_resume_adds_checkpoint_flag_and_requires_complete_checkpoint(self) -> None:
        config = self._config()
        train_config = config["train"]
        cache_root: Path = train_config["cache-root"]
        cache_root.mkdir()
        receipt = {"status": "deterministic_supervised_cache_complete"}
        receipt_path = cache_root / "cache-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        output_root: Path = train_config["output-root"]
        checkpoint = output_root / "checkpoint-step-000025"
        checkpoint.mkdir(parents=True)
        for name in (
            "adapter.safetensors", "optimizer.pt", "scheduler.pt", "rng.pt",
            "checkpoint-receipt.json",
        ):
            (checkpoint / name).write_bytes(b"stub")
        (checkpoint / "trainer-state.json").write_text(
            json.dumps({"global_step": 25, "status": "paused_for_fresh_process_resume"}),
            encoding="utf-8",
        )
        digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        saved = {
            "rank": 16,
            "alpha": 32.0,
            "max_steps": 300,
            "gradient_accumulation": 4,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "save_every": 25,
            "validation_examples": 64,
            "seed": 42,
            "max_gradient_norm": 1.0,
            "cache_receipt_sha256": digest,
        }
        (output_root / "run-config.json").write_text(json.dumps(saved), encoding="utf-8")

        plan = build_plan(config, stage="train", resume=checkpoint, paths=self.runtime)
        self.assertIn("--resume-checkpoint", plan.commands[0].argv)
        self.assertIn(str(checkpoint), plan.commands[0].argv)

        (checkpoint / "scheduler.pt").unlink()
        with self.assertRaisesRegex(PlanError, "scheduler.pt"):
            build_plan(config, stage="train", resume=checkpoint, paths=self.runtime)

    def test_fresh_train_refuses_existing_output_root(self) -> None:
        config = self._config()
        cache_root: Path = config["train"]["cache-root"]
        cache_root.mkdir()
        (cache_root / "cache-receipt.json").write_text(
            json.dumps({"status": "deterministic_supervised_cache_complete"}),
            encoding="utf-8",
        )
        output_root: Path = config["train"]["output-root"]
        output_root.mkdir()
        with self.assertRaisesRegex(PlanError, "refusing to train into an existing output root"):
            build_plan(config, stage="train", resume=None, paths=self.runtime)

    def test_historical_resume_can_export_the_selected_prior_checkpoint(self) -> None:
        config = self._config(historical=True)
        config["train"]["stop-after-step"] = 275
        cache_root: Path = config["train"]["cache-root"]
        cache_root.mkdir()
        receipt_path = cache_root / "cache-receipt.json"
        receipt_path.write_text(
            json.dumps({"status": "deterministic_supervised_cache_complete"}),
            encoding="utf-8",
        )
        output_root: Path = config["train"]["output-root"]
        checkpoint = output_root / "checkpoint-step-000250"
        checkpoint.mkdir(parents=True)
        for name in (
            "adapter.safetensors", "optimizer.pt", "scheduler.pt", "rng.pt",
            "checkpoint-receipt.json",
        ):
            (checkpoint / name).write_bytes(b"stub")
        (checkpoint / "trainer-state.json").write_text(
            json.dumps({"global_step": 250, "status": "paused_for_fresh_process_resume"}),
            encoding="utf-8",
        )
        digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        saved = {
            "rank": 16,
            "alpha": 32.0,
            "max_steps": 300,
            "gradient_accumulation": 4,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "save_every": 25,
            "validation_examples": 64,
            "seed": 42,
            "max_gradient_norm": 1.0,
            "cache_receipt_sha256": digest,
        }
        (output_root / "run-config.json").write_text(json.dumps(saved), encoding="utf-8")

        plan = build_plan(config, stage="all", resume=checkpoint, paths=self.runtime)
        self.assertEqual([item.stage for item in plan.commands], ["train", "export"])
        self.assertEqual(
            plan.commands[0].argv[plan.commands[0].argv.index("--stop-after-step") + 1],
            "275",
        )
        self.assertEqual(plan.export_checkpoint, checkpoint)
        self.assertIn(str(checkpoint / "adapter.safetensors"), plan.commands[-1].argv)

    def test_export_refuses_existing_model(self) -> None:
        config = self._config()
        checkpoint: Path = config["export"]["checkpoint"]
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
        output: Path = config["export"]["output"]
        output.write_bytes(b"existing model")
        with self.assertRaisesRegex(PlanError, "refusing to overwrite"):
            build_plan(config, stage="export", resume=None, paths=self.runtime)


if __name__ == "__main__":
    unittest.main()
