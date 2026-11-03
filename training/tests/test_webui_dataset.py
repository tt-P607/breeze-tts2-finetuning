from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.workflow.dataset import (
    DatasetImportError,
    DatasetItem,
    prepare_batch,
    read_dataset,
)

sf = pytest.importorskip("soundfile")


def write_audio(path: Path, frames: int = 800, sample_rate: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, [0.0] * frames, sample_rate)
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_jsonl_relative_paths_preserve_pipe_text_and_read_metadata(
    tmp_path: Path,
) -> None:
    audio = write_audio(tmp_path / "clips" / "a.wav")
    manifest = tmp_path / "items.jsonl"
    text = "  原样保留 | 竖线和空格  "
    manifest.write_text(
        json.dumps({"audio": "clips/a.wav", "text": text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    items = read_dataset(manifest)

    assert len(items) == 1
    assert items[0].audio == audio.resolve()
    assert items[0].text == text
    assert items[0].duration == pytest.approx(0.1)
    assert items[0].sample_rate == 8000
    assert items[0].channels == 1


def test_gsv_list_split_keeps_pipes_inside_text(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "gsv.wav")
    manifest = tmp_path / "train.list"
    manifest.write_text("gsv.wav|speaker|zh|第一句 | 第二句|第三句\n", encoding="utf-8")

    items = read_dataset(manifest)

    assert items[0].audio == audio.resolve()
    assert items[0].text == "第一句 | 第二句|第三句"


def test_directory_sidecars_support_txt_and_lab(tmp_path: Path) -> None:
    write_audio(tmp_path / "a.wav")
    write_audio(tmp_path / "nested" / "b.flac", sample_rate=16000)
    with (tmp_path / "a.txt").open("w", encoding="utf-8", newline="") as handle:
        handle.write("人工台词 | 不拆分\n")
    (tmp_path / "nested" / "b.lab").write_text("第二条台词", encoding="utf-8")

    items = read_dataset(tmp_path)

    assert [item.audio.name for item in items] == ["a.wav", "b.flac"]
    assert [item.text for item in items] == ["人工台词 | 不拆分\n", "第二条台词"]
    assert [item.sample_rate for item in items] == [8000, 16000]


def test_directory_requires_annotations_unless_filename_mode_is_explicit(
    tmp_path: Path,
) -> None:
    audio = write_audio(tmp_path / "named_take.wav")
    with pytest.raises(DatasetImportError, match="没有同名"):
        read_dataset(tmp_path)

    (tmp_path / "named_take.txt").write_text("人工标注", encoding="utf-8")
    items = read_dataset(tmp_path, transcript_mode="filename")
    assert items[0].audio == audio.resolve()
    assert items[0].text == "named_take"


def test_directory_rejects_both_txt_and_lab_for_one_audio(tmp_path: Path) -> None:
    write_audio(tmp_path / "take.wav")
    (tmp_path / "take.txt").write_text("第一份", encoding="utf-8")
    (tmp_path / "take.lab").write_text("第二份", encoding="utf-8")

    with pytest.raises(DatasetImportError, match="多个标注文件"):
        read_dataset(tmp_path)


def test_empty_text_missing_audio_and_duplicate_audio_are_errors(
    tmp_path: Path,
) -> None:
    audio = write_audio(tmp_path / "one.wav")
    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"audio":"one.wav","text":"  "}\n', encoding="utf-8")
    with pytest.raises(DatasetImportError, match="台词为空"):
        read_dataset(empty)

    missing = tmp_path / "missing.jsonl"
    missing.write_text('{"audio":"missing.wav","text":"text"}\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="不存在"):
        read_dataset(missing)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        "\n".join(
            json.dumps({"audio": name, "text": "text"})
            for name in ("one.wav", str(audio.resolve()))
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetImportError, match="重复引用"):
        read_dataset(duplicate)


def test_prepare_batch_copies_and_splits_reproducibly_without_touching_sources(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    for index in range(6):
        audio = write_audio(source_dir / f"take-{index}.wav", frames=800 + index)
        (audio.with_suffix(".txt")).write_text(f"台词 {index} | 原样", encoding="utf-8")
    items = read_dataset(source_dir)
    source_hashes = {str(item.audio): sha256(item.audio) for item in items}

    first = prepare_batch(items, tmp_path / "batch-one", validation_count=2, seed=9)
    second = prepare_batch(items, tmp_path / "batch-two", validation_count=2, seed=9)
    assert Path(second["train_manifest"]).is_file()
    assert Path(second["validation_manifest"]).is_file()

    assert first["train_count"] == 4
    assert first["validation_count"] == 2
    assert first["duration_seconds"] == pytest.approx(sum(i.duration for i in items))
    assert Path(first["audio_dir"]).is_absolute()

    first_rows: dict[str, str] = {}
    first_manifest_paths: set[str] = set()
    for manifest_key, expected_count in (
        ("train_manifest", 4),
        ("validation_manifest", 2),
    ):
        manifest = Path(first[manifest_key])
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == expected_count
        for row in rows:
            copied_audio = Path(row["audio"])
            assert copied_audio.is_absolute()
            assert copied_audio.is_file()
            assert len(copied_audio.stem) == 6
            assert copied_audio.stem.isdigit()
            assert copied_audio.suffix == ".wav"
            first_manifest_paths.add(str(copied_audio))
            source = next(item for item in items if item.text == row["text"])
            assert sha256(copied_audio) == source_hashes[str(source.audio)]
            first_rows[str(source.audio)] = manifest_key

    assert len(first_manifest_paths) == 6
    assert len(first_rows) == 6
    assert len(set(first_rows.values())) == 2

    second_record = json.loads(
        (tmp_path / "batch-two" / "dataset.json").read_text(encoding="utf-8")
    )
    second_rows = {
        item["source_audio"]: item["split"] for item in second_record["items"]
    }
    first_record = json.loads(
        (tmp_path / "batch-one" / "dataset.json").read_text(encoding="utf-8")
    )
    first_split = {
        item["source_audio"]: item["split"] for item in first_record["items"]
    }
    assert first_split == second_rows
    assert all(sha256(item.audio) == source_hashes[str(item.audio)] for item in items)


def test_prepare_batch_refuses_existing_directory_without_overwriting(
    tmp_path: Path,
) -> None:
    items = [
        DatasetItem(
            write_audio(tmp_path / f"{index}.wav"), f"line {index}", 0.1, 8000, 1
        )
        for index in range(2)
    ]
    batch = tmp_path / "existing"
    batch.mkdir()
    sentinel = batch / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        prepare_batch(items, batch, validation_count=1)

    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not (batch / "audio").exists()
