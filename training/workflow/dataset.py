"""CPU-only dataset import and immutable batch preparation helpers."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}
SIDECAR_SUFFIXES = {".txt", ".lab"}


@dataclass(frozen=True)
class DatasetItem:
    audio: Path
    text: str
    duration: float
    sample_rate: int
    channels: int


class DatasetImportError(ValueError):
    """Raised when a dataset row or its transcript cannot be imported."""


def _read_audio_info(audio: Path) -> tuple[float, int, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "读取音频元数据需要 soundfile，请安装项目音频依赖。"
        ) from exc

    try:
        info = sf.info(str(audio))
    except Exception as exc:
        raise DatasetImportError(f"无法读取音频元数据：{audio}（{exc}）") from exc

    duration = float(info.duration)
    sample_rate = int(info.samplerate)
    channels = int(info.channels)
    if (
        not math.isfinite(duration)
        or duration <= 0
        or sample_rate <= 0
        or channels <= 0
    ):
        raise DatasetImportError(f"音频元数据无效：{audio}")
    return duration, sample_rate, channels


def _resolve_audio(raw_audio: str, base_dir: Path, context: str) -> Path:
    if not isinstance(raw_audio, str) or not raw_audio.strip():
        raise DatasetImportError(f"音频路径为空：{context}")

    candidate = Path(raw_audio).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate

    try:
        audio = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"音频文件不存在：{context}：{candidate}") from exc
    if not audio.is_file():
        raise FileNotFoundError(f"音频路径不是文件：{context}：{audio}")
    if audio.suffix.lower() not in AUDIO_SUFFIXES:
        raise DatasetImportError(
            f"不支持的音频格式：{context}：{audio.suffix or '(无扩展名)'}"
        )
    return audio


def _read_text(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return handle.read()
    except UnicodeDecodeError as exc:
        raise DatasetImportError(f"标注文件不是 UTF-8 文本：{path}") from exc


def _jsonl_rows(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            context = f"{path}:{line_number}"
            if not line.strip():
                raise DatasetImportError(f"清单中有空行：{context}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetImportError(f"JSONL 格式错误：{context}：{exc}") from exc
            if not isinstance(value, dict):
                raise DatasetImportError(f"JSONL 每行必须是对象：{context}")
            audio = value.get("audio")
            text = value.get("text")
            if not isinstance(audio, str):
                raise DatasetImportError(f"JSONL 缺少字符串 audio 字段：{context}")
            if not isinstance(text, str):
                raise DatasetImportError(f"JSONL 缺少字符串 text 字段：{context}")
            rows.append((audio, text, context))
    return rows


def _gsv_rows(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
        context = f"{path}:{line_number}"
        if not line:
            raise DatasetImportError(f"GSV 清单中有空行：{context}")
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise DatasetImportError(
                f"GSV 清单每行必须为 audio|speaker|language|text：{context}"
            )
        audio, _speaker, _language, text = fields
        rows.append((audio, text, context))
    return rows


def _directory_rows(root: Path, transcript_mode: str) -> list[tuple[str, str, str]]:
    audio_files: list[Path] = []
    sidecars: dict[tuple[Path, str], list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in AUDIO_SUFFIXES:
            audio_files.append(path)
        elif suffix in SIDECAR_SUFFIXES:
            key = (path.parent.resolve(), path.stem.casefold())
            sidecars.setdefault(key, []).append(path)

    audio_files.sort(key=lambda item: str(item).casefold())
    if not audio_files:
        raise DatasetImportError(f"目录中没有支持的音频文件：{root}")

    rows: list[tuple[str, str, str]] = []
    missing: list[str] = []
    for audio in audio_files:
        context = f"目录音频 {audio}"
        if transcript_mode == "filename":
            rows.append((str(audio), audio.stem, context))
            continue

        key = (audio.parent.resolve(), audio.stem.casefold())
        matches = sorted(sidecars.get(key, []), key=lambda item: str(item).casefold())
        if not matches:
            missing.append(str(audio))
            continue
        if len(matches) > 1:
            listed = ", ".join(str(item) for item in matches)
            raise DatasetImportError(
                f"同一音频存在多个标注文件，请保留一个：{audio}：{listed}"
            )
        rows.append((str(audio), _read_text(matches[0]), context))

    if missing:
        listed = "\n".join(missing)
        raise DatasetImportError(
            "以下音频没有同名 .txt/.lab 人工标注；可补齐标注，或显式选择 "
            f"transcript_mode='filename'：\n{listed}"
        )
    return rows


def _build_items(rows: list[tuple[str, str, str]], base_dir: Path) -> list[DatasetItem]:
    if not rows:
        raise DatasetImportError("数据集没有音频条目。")

    items: list[DatasetItem] = []
    seen: dict[str, str] = {}
    missing: list[str] = []
    resolved_rows: list[tuple[Path, str, str]] = []
    for raw_audio, text, context in rows:
        if not isinstance(text, str) or not text.strip():
            raise DatasetImportError(f"台词为空：{context}")
        try:
            audio = _resolve_audio(raw_audio, base_dir, context)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue

        key = os.path.normcase(os.path.normpath(str(audio)))
        previous = seen.get(key)
        if previous is not None:
            raise DatasetImportError(
                f"重复引用同一音频：{audio}；首次出现在 {previous}，再次出现在 {context}"
            )
        seen[key] = context
        resolved_rows.append((audio, text, context))

    if missing:
        raise FileNotFoundError("清单引用了不存在的音频：\n" + "\n".join(missing))

    for audio, text, _context in resolved_rows:
        duration, sample_rate, channels = _read_audio_info(audio)
        items.append(
            DatasetItem(
                audio=audio,
                text=text,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
            )
        )
    return items


def read_dataset(
    source: str | Path, transcript_mode: str = "sidecar"
) -> list[DatasetItem]:
    """Read a JSONL/GSV manifest or scan a directory for annotated audio.

    Manifest rows always use their embedded text. ``transcript_mode`` applies
    to directory scans: ``sidecar`` requires a same-stem .txt/.lab file, while
    ``filename`` uses only the audio filename stem as its transcript.
    """
    if transcript_mode not in {"sidecar", "filename"}:
        raise ValueError("transcript_mode must be 'sidecar' or 'filename'")

    source_path = Path(source).expanduser()
    try:
        resolved_source = source_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"数据源不存在：{source_path}") from exc

    if resolved_source.is_dir():
        rows = _directory_rows(resolved_source, transcript_mode)
        base_dir = resolved_source
    elif resolved_source.is_file():
        suffix = resolved_source.suffix.lower()
        if suffix == ".jsonl":
            rows = _jsonl_rows(resolved_source)
        elif suffix == ".list":
            rows = _gsv_rows(resolved_source)
        else:
            raise DatasetImportError(
                f"仅支持 JSONL、GSV .list 清单或音频目录：{resolved_source}"
            )
        base_dir = resolved_source.parent
    else:
        raise DatasetImportError(f"数据源不是普通文件或目录：{resolved_source}")

    return _build_items(rows, base_dir)


def _validate_items(items: list[DatasetItem]) -> list[tuple[DatasetItem, Path]]:
    if len(items) < 2:
        raise ValueError("准备训练批次至少需要 2 条数据。")

    validated: list[tuple[DatasetItem, Path]] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(items, start=1):
        if not isinstance(item, DatasetItem):
            raise TypeError(f"第 {index} 条数据不是 DatasetItem。")
        if not isinstance(item.text, str) or not item.text.strip():
            raise DatasetImportError(f"第 {index} 条数据的台词为空。")
        try:
            audio = Path(item.audio).expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"第 {index} 条数据的音频不存在：{item.audio}"
            ) from exc
        if not audio.is_file():
            raise FileNotFoundError(f"第 {index} 条音频不是文件：{audio}")
        if audio.suffix.lower() not in AUDIO_SUFFIXES:
            raise DatasetImportError(f"不支持的音频格式：{audio}")
        key = os.path.normcase(os.path.normpath(str(audio)))
        if key in seen:
            raise DatasetImportError(
                f"重复引用同一音频：{audio}；首次为第 {seen[key]} 条，再次为第 {index} 条。"
            )
        seen[key] = index

        if (
            isinstance(item.duration, bool)
            or not isinstance(item.duration, (int, float))
            or isinstance(item.sample_rate, bool)
            or not isinstance(item.sample_rate, int)
            or isinstance(item.channels, bool)
            or not isinstance(item.channels, int)
        ):
            raise DatasetImportError(f"第 {index} 条音频元数据无效：{audio}")
        try:
            duration = float(item.duration)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DatasetImportError(f"第 {index} 条音频元数据无效：{audio}") from exc
        if (
            not math.isfinite(duration)
            or duration <= 0
            or item.sample_rate <= 0
            or item.channels <= 0
        ):
            raise DatasetImportError(f"第 {index} 条音频元数据无效：{audio}")
        normalized_item = DatasetItem(
            audio=audio,
            text=item.text,
            duration=duration,
            sample_rate=item.sample_rate,
            channels=item.channels,
        )
        validated.append((normalized_item, audio))
    return validated


def _write_jsonl(
    path: Path,
    indices: list[int],
    items: list[tuple[DatasetItem, Path]],
    final_audio_paths: list[Path],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in indices:
            row = {
                "audio": str(final_audio_paths[index]),
                "text": items[index][0].text,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_batch(
    items: list[DatasetItem],
    batch_dir: Path,
    validation_count: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Copy source audio into a new batch and write native-trainer manifests."""
    if not isinstance(validation_count, int) or isinstance(validation_count, bool):
        raise TypeError("validation_count 必须是整数。")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed 必须是整数。")

    validated = _validate_items(items)
    if not 1 <= validation_count < len(validated):
        raise ValueError("validation_count 必须满足 1 <= validation_count < 数据条数。")

    requested_dir = Path(batch_dir).expanduser()
    if requested_dir.exists() or requested_dir.is_symlink():
        raise FileExistsError(f"拒绝覆盖已有训练批次：{requested_dir}")
    final_dir = requested_dir.resolve(strict=False)
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"拒绝覆盖已有训练批次：{final_dir}")
    parent = final_dir.parent
    if not parent.exists():
        raise FileNotFoundError(f"训练批次父目录不存在：{parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"训练批次父路径不是目录：{parent}")
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"没有权限在该目录创建训练批次：{parent}")

    rng = random.Random(seed)
    validation_indices = set(rng.sample(range(len(validated)), validation_count))
    train_indices = [
        index for index in range(len(validated)) if index not in validation_indices
    ]
    ordered_validation = [
        index for index in range(len(validated)) if index in validation_indices
    ]

    suffixes = [audio.suffix.lower() for _item, audio in validated]
    final_audio_dir = final_dir / "audio"
    final_audio_paths = [
        (final_audio_dir / f"{index + 1:06d}{suffix}").resolve(strict=False)
        for index, suffix in enumerate(suffixes)
    ]

    total_duration = sum(item.duration for item, _audio in validated)
    dataset_items = []
    for index, (item, source_audio) in enumerate(validated):
        dataset_items.append(
            {
                "source_audio": str(source_audio),
                "batch_audio": str(final_audio_paths[index]),
                "text": item.text,
                "duration_seconds": item.duration,
                "sample_rate": item.sample_rate,
                "channels": item.channels,
                "split": "validation" if index in validation_indices else "train",
            }
        )

    dataset_record = {
        "schema_version": 1,
        "seed": seed,
        "source_count": len(validated),
        "train_count": len(train_indices),
        "validation_count": len(ordered_validation),
        "duration_seconds": total_duration,
        "audio_dir": str(final_audio_dir.resolve(strict=False)),
        "statistics": {
            "sample_rates": {
                str(rate): count
                for rate, count in sorted(
                    Counter(item.sample_rate for item, _audio in validated).items()
                )
            },
            "channels": {
                str(channels): count
                for channels, count in sorted(
                    Counter(item.channels for item, _audio in validated).items()
                )
            },
        },
        "items": dataset_items,
    }

    staging_dir: Path | None = None
    try:
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{final_dir.name}.prepare-", dir=parent)
        )
        staging_audio_dir = staging_dir / "audio"
        staging_audio_dir.mkdir()
        for index, (_item, source_audio) in enumerate(validated):
            destination = staging_audio_dir / final_audio_paths[index].name
            shutil.copyfile(source_audio, destination)

        _write_jsonl(
            staging_dir / "train.jsonl",
            train_indices,
            validated,
            final_audio_paths,
        )
        _write_jsonl(
            staging_dir / "validation.jsonl",
            ordered_validation,
            validated,
            final_audio_paths,
        )
        with (staging_dir / "dataset.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(dataset_record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        try:
            staging_dir.rename(final_dir)
        except FileExistsError as exc:
            raise FileExistsError(f"拒绝覆盖已有训练批次：{final_dir}") from exc
        staging_dir = None
    except PermissionError as exc:
        raise PermissionError(f"没有权限写入训练批次 {final_dir}：{exc}") from exc
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    return {
        "train_manifest": str(final_dir / "train.jsonl"),
        "validation_manifest": str(final_dir / "validation.jsonl"),
        "train_count": len(train_indices),
        "validation_count": len(ordered_validation),
        "duration_seconds": total_duration,
        "audio_dir": str(final_audio_dir),
    }
