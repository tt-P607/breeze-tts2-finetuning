from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.webui import PARAMETERS, directory_name, new_batch, save_parameters
from training.workflow.dataset import read_dataset
from training.workflow.webui_service import JobManager


def test_new_role_batch_and_parameter_roundtrip(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for n in range(4):
        sf.write(source / f"{n}.wav", np.zeros(2400), 24000)
        (source / f"{n}.txt").write_text(f"台词{n}", encoding="utf-8")
    jobs = JobManager(tmp_path)
    items = read_dataset(source)
    path, result = new_batch(jobs, items, "角色模型", "新角色", "形态", "首批", 25, 42)
    assert result["train_count"] == 3
    assert result["validation_count"] == 1
    assert path == "角色模型/新角色/形态/训练批次/首批/training-config.json"
    config = jobs.read_config(path)
    assert config["train"]["stop-after-step"] == 250
    assert config["train"]["save-every"] == 25
    assert Path(config["export"]["checkpoint"]).name == "checkpoint-step-000250"
    values = [config["train"][key] for key in PARAMETERS]
    values[0] = 275
    saved = save_parameters(jobs, path, values)
    assert saved["train"]["stop-after-step"] == 275
    assert Path(saved["export"]["output"]).name == "新角色_275步.gguf"
    assert (source / "0.txt").read_text(encoding="utf-8") == "台词0"
    with pytest.raises(FileExistsError):
        new_batch(jobs, items, "角色模型", "新角色", "形态", "首批", 25, 42)


@pytest.mark.parametrize("name", ["../escape", "CON", "a:b", ""])
def test_batch_name_cannot_be_a_path(name):
    with pytest.raises(ValueError):
        directory_name(name)
