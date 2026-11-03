"""Local Gradio UI for the same workflow used by the command line."""

from __future__ import annotations

import argparse
import math
import os
import re
from datetime import datetime
from pathlib import Path

import gradio as gr

from training.workflow.commands import runtime_paths
from training.workflow.config import TRAIN_DEFAULTS
from training.workflow.dataset import prepare_batch, read_dataset
from training.workflow.webui_service import JobManager

WORKSPACE = Path(__file__).resolve().parents[2]
PARAMETERS = (
    "stop-after-step",
    "save-every",
    "rank",
    "alpha",
    "learning-rate",
    "gradient-accumulation",
    "max-steps",
    "warmup-steps",
    "weight-decay",
    "validation-examples",
    "seed",
    "max-gradient-norm",
)
INTEGER_PARAMETERS = set(PARAMETERS) - {
    "alpha",
    "learning-rate",
    "weight-decay",
    "max-gradient-norm",
}
STATUS = {
    "idle": "尚未选择批次",
    "ready": "就绪",
    "running": "正在运行",
    "stopping": "正在保存并暂停",
    "paused": "已保存并暂停",
    "completed": "本次任务完成",
    "failed": "任务失败，请查看日志",
    "stopped": "已强制停止",
    "interrupted": "上次任务已中断",
}


def directory_name(value: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or re.search(r'[<>:"/\\|?*\x00-\x1f]', value):
        raise ValueError("名称不能为空，也不能包含路径分隔符或 Windows 文件名禁用字符")
    if value.endswith((".", " ")) or re.fullmatch(
        r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", value
    ):
        raise ValueError("请换一个可用的文件夹名称")
    return value


def new_batch(jobs, items, category, name, form, batch_name, validation_percent, seed):
    """Copy annotated audio, then create a recipe; never launch training."""
    if len(items) < 2:
        raise ValueError("请先导入至少两条有标注的音频")
    if category not in {"角色模型", "风格模型"}:
        raise ValueError("请选择角色模型或风格模型")
    if not 0 < validation_percent < 100:
        raise ValueError("验证集占比必须在 0 和 100 之间")
    name = directory_name(name)
    parent = jobs.workspace / category / name
    if form.strip():
        parent /= directory_name(form)
    batch = parent / "训练批次" / directory_name(batch_name)
    validation_count = max(
        1, min(len(items) - 1, math.ceil(len(items) * validation_percent / 100))
    )
    batch.parent.mkdir(parents=True, exist_ok=True)
    result = prepare_batch(
        items, batch, validation_count=validation_count, seed=int(seed)
    )
    config = {
        "prepare": {
            "train-manifest": result["train_manifest"],
            "validation-manifest": result["validation_manifest"],
            "output-root": str(batch / "cache"),
            "train-limit": result["train_count"],
            "validation-limit": result["validation_count"],
            "seed": int(seed),
        },
        "train": {
            **TRAIN_DEFAULTS,
            "cache-root": str(batch / "cache"),
            "output-root": str(batch / "检查点"),
            "validation-examples": min(64, result["validation_count"]),
            "seed": int(seed),
        },
        "export": {
            "checkpoint": str(batch / "检查点" / "checkpoint-step-000250"),
            "output": str(batch / "推理模型" / f"{name}_250步.gguf"),
            "alpha": 32,
        },
    }
    path = batch / "training-config.json"
    jobs.save_config(path, config)
    return path.relative_to(jobs.workspace).as_posix(), result


def save_parameters(jobs, path, values):
    config = jobs.read_config(path)
    train = config["train"]
    for key, value in zip(PARAMETERS, values, strict=True):
        if key in INTEGER_PARAMETERS:
            if int(value) != value:
                raise ValueError(f"{key} 必须是整数")
            value = int(value)
        train[key] = value
    target, schedule = train["stop-after-step"], train["max-steps"]
    if not 0 < target <= schedule or train["save-every"] <= 0:
        raise ValueError("训练停点必须大于 0 且不超过调度总步数；保存间隔必须大于 0")
    if schedule % train["save-every"]:
        raise ValueError("调度总步数必须能被保存间隔整除")
    if target == schedule:
        train.pop("stop-after-step")
    if "export" in config:
        config["export"]["alpha"] = train["alpha"]
        checkpoint = Path(train["output-root"]) / f"checkpoint-step-{target:06d}"
        config["export"]["checkpoint"] = str(checkpoint)
        output = Path(config["export"]["output"])
        config["export"]["output"] = str(
            output.with_name(f"{path_role(config)}_{target}步.gguf")
        )
    jobs.save_config(path, config)
    return jobs.read_config(path)


def path_role(config):
    output = Path(config["export"]["output"])
    return re.sub(r"_\d+步$", "", output.stem)


def build_app(workspace: Path = WORKSPACE):
    import pandas as pd

    jobs = JobManager(workspace)
    paths = runtime_paths(workspace)

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    def scan(source, mode):
        items = call(read_dataset, source, transcript_mode=mode)
        table = [
            [
                i + 1,
                p.audio.name,
                round(p.duration, 2),
                p.sample_rate,
                p.channels,
                p.text,
            ]
            for i, p in enumerate(items)
        ]
        stats = f"**{len(items)} 条 · {sum(p.duration for p in items) / 60:.2f} 分钟**　点击任意行试听并核对标注。"
        return items, table, stats

    def preview(items, evt: gr.SelectData):
        import soundfile as sf

        item = items[evt.index[0]]
        audio, rate = sf.read(item.audio, dtype="float32")
        return (rate, audio), item.text

    def create(items, *args):
        path, result = call(new_batch, jobs, items, *args)
        return (
            gr.Dropdown(choices=jobs.list_configs(), value=path),
            f"已复制 {result['train_count']} 条训练 + {result['validation_count']} 条验证。批次：{path}。还没有开始训练。",
        )

    def checkpoint_choices(path):
        return [
            (
                f"{r['step']} 步 · {'可续训' if r['resumable'] else '仅导出'}",
                r["path"],
            )
            for r in jobs.list_checkpoints(path)
        ]

    def load(path):
        if not path:
            return [gr.skip()] * (len(PARAMETERS) + 3)
        config = call(jobs.read_config, path)
        choices = call(checkpoint_choices, path)
        train = config["train"]
        values = [
            train.get(
                key,
                train["max-steps"] if key == "stop-after-step" else TRAIN_DEFAULTS[key],
            )
            for key in PARAMETERS
        ]
        return [
            *values,
            config,
            gr.Dropdown(choices=choices, value=choices[-1][1] if choices else None),
            config.get("export", {}).get("output", ""),
        ]

    def refresh_configs(selected):
        choices = jobs.list_configs()
        return gr.Dropdown(
            choices=choices, value=selected if selected in choices else None
        )

    def refresh_checkpoints(path):
        choices = call(checkpoint_choices, path)
        return gr.Dropdown(choices=choices, value=choices[-1][1] if choices else None)

    def save(path, *values):
        config = call(save_parameters, jobs, path, values)
        gr.Info("参数已保存；已有训练输出的历史批次不能改写。")
        return config

    def start(
        path,
        python,
        model,
        converter,
        deps,
        stage="all",
        dry_run=False,
        checkpoint=None,
        target=None,
        output=None,
    ):
        if not path:
            raise gr.Error("请先选择或准备一个训练批次")
        runtime = {
            "python": python,
            "model_root": model,
            "converter": converter,
            "gguf_deps": deps,
        }
        resume = None
        if target is not None:
            if not checkpoint:
                raise gr.Error("请先选择一个可续训检查点")
            runtime["stop_after_step"] = int(target)
            resume = checkpoint
        if stage == "export":
            if not checkpoint or not output.strip():
                raise gr.Error("请选择检查点并填写 GGUF 输出路径")
            runtime.update(export_checkpoint=checkpoint, export_output=output)
        call(
            jobs.start,
            path,
            stage=stage,
            resume=resume,
            dry_run=dry_run,
            runtime=runtime,
        )
        return (
            "配置检查已提交（不使用 GPU）。"
            if dry_run
            else "任务已提交，进度和日志将在下方更新。"
        )

    def poll(path):
        snapshot = call(jobs.snapshot, path or None)
        step, total = snapshot["step"], snapshot["total_steps"]
        percent = min(100, step * 100 / total) if total else 0
        status = STATUS.get(snapshot["status"], snapshot["status"])
        stage = {
            "prepare": "预处理",
            "train": "训练",
            "export": "导出",
            "all": "完整流程",
        }.get(snapshot["stage"], "历史记录")
        if snapshot["dry_run"]:
            stage = "配置检查（未训练）"
        status += f" · {stage}"
        resource = snapshot["resources"]
        gpu = resource.get("gpu")
        gpu_text = (
            f"显存 {gpu['memory_used_mib'] / 1024:.2f} / {gpu['memory_total_mib'] / 1024:.1f} GiB · GPU {gpu['utilization_percent']}%"
            if gpu
            else "GPU 信息暂不可用"
        )
        disk = resource.get("disk_free_gib")
        text = (
            f"### {status}　{step} / {total} 步 · {percent:.0f}%\n\n{gpu_text} · 磁盘剩余 {disk:.1f} GiB"
            if disk is not None
            else f"### {status}　{step} / {total} 步\n\n{gpu_text}"
        )
        if snapshot["config"]:
            text += f"\n\n当前批次：`{snapshot['config']}`　已运行 {snapshot['elapsed'] / 60:.1f} 分钟"
        rows = []
        for row in snapshot["metrics"]:
            for key, label in (("training", "训练"), ("validation", "验证")):
                if row.get(key) is not None:
                    rows.append({"步数": row["step"], "损失": row[key], "曲线": label})
        frame = pd.DataFrame(rows, columns=["步数", "损失", "曲线"])
        return text, snapshot["log"], frame, "\n".join(snapshot["output_files"])

    with gr.Blocks(title="Breeze · LoRA 训练", analytics_enabled=False) as app:
        gr.HTML(
            '<header class="breeze-header"><div class="eyebrow">BREEZE TTS 2 / FINETUNING</div><h1>LoRA 训练工作台</h1><p>整理一份数据，留下一组可继续训练的模型。</p></header>'
        )
        notice = gr.Markdown(
            "仅监听本机 · 默认 250 步，每 25 步保存 · 打开页面不会加载底模或开始训练"
        )
        items = gr.State([])
        with gr.Row():
            config_path = gr.Dropdown(
                choices=jobs.list_configs(),
                label="训练批次 / training-config.json",
                scale=8,
            )
            refresh = gr.Button("刷新批次", scale=1)
        with gr.Accordion("环境路径（通常不需要修改）", open=False):
            python = gr.Textbox(value=str(paths.python), label="训练 Python")
            model = gr.Textbox(
                value=str(paths.model_root), label="原始训练底模目录（不是推理 GGUF）"
            )
            converter = gr.Textbox(
                value=str(paths.converter), label="LoRA → GGUF 转换脚本"
            )
            deps = gr.Textbox(value=str(paths.gguf_deps), label="GGUF 依赖目录")
        with gr.Tabs():
            with gr.Tab("01 · 数据准备"):
                gr.Markdown(
                    "支持 JSONL、GPT-SoVITS `.list`，或带同名 `.txt` / `.lab` 的音频目录。只使用准确标注，不自动猜测台词。"
                )
                source = gr.Textbox(label="标注清单或音频目录的完整路径")
                mode = gr.Radio(
                    choices=[
                        ("同名标注文件", "sidecar"),
                        ("整个文件名就是台词", "filename"),
                    ],
                    value="sidecar",
                    label="目录的标注来源（清单文件不受影响）",
                )
                scan_button = gr.Button("读取并检查数据", variant="primary")
                stats = gr.Markdown("尚未读取数据")
                table = gr.Dataframe(
                    headers=["序号", "文件", "秒", "采样率", "声道", "标注"],
                    datatype=["number", "str", "number", "number", "number", "str"],
                    interactive=False,
                    wrap=True,
                    max_height=300,
                )
                with gr.Row():
                    audio = gr.Audio(label="选中音频试听", interactive=False)
                    transcript = gr.Textbox(
                        label="准确标注", lines=3, interactive=False
                    )
                with gr.Row():
                    category = gr.Dropdown(
                        choices=["角色模型", "风格模型"], value="角色模型", label="类型"
                    )
                    name = gr.Textbox(label="角色 / 风格名称")
                    form = gr.Textbox(label="形态（可留空）")
                with gr.Row():
                    batch_name = gr.Textbox(
                        value=datetime.now().strftime("%Y%m%d_新批次"),
                        label="新批次名称",
                    )
                    validation = gr.Number(
                        value=5, minimum=1, maximum=50, label="验证集占比 %"
                    )
                    data_seed = gr.Number(value=42, precision=0, label="划分随机种子")
                gr.Markdown(
                    "验证集只用于评估，不参与参数更新。准备时复制音频并保留来源表，原始数据不变；已有批次不会覆盖。"
                )
                create_button = gr.Button("复制数据，建立新训练批次", variant="primary")
            with gr.Tab("02 · 训练与损失"):
                gr.Markdown(
                    "先保存参数，再检查配置。首次训练可依次完成预处理、训练和 GGUF 导出；已有缓存时直接训练。"
                )
                fields = []
                labels = [
                    "本次训练到多少步",
                    "每多少步保存",
                    "LoRA Rank",
                    "LoRA Alpha",
                    "学习率",
                    "梯度累积",
                    "学习率调度总步数",
                    "预热步数",
                    "权重衰减",
                    "验证样本数",
                    "随机种子",
                    "梯度裁剪",
                ]
                for begin, end in ((0, 4), (4, 6)):
                    with gr.Row():
                        for key, label in zip(
                            PARAMETERS[begin:end], labels[begin:end], strict=True
                        ):
                            fields.append(
                                gr.Number(
                                    value=TRAIN_DEFAULTS[key],
                                    label=label,
                                    precision=0 if key in INTEGER_PARAMETERS else None,
                                )
                            )
                with gr.Accordion("高级训练参数", open=False):
                    gr.Markdown(
                        "默认调度预留到 300 步，本次在 250 步保存并暂停，之后可在原调度内续训。Rank 16 / Alpha 32 是本机已有用法。"
                    )
                    for begin, end in ((6, 9), (9, 12)):
                        with gr.Row():
                            for key, label in zip(
                                PARAMETERS[begin:end], labels[begin:end], strict=True
                            ):
                                fields.append(
                                    gr.Number(
                                        value=TRAIN_DEFAULTS[key],
                                        label=label,
                                        precision=0
                                        if key in INTEGER_PARAMETERS
                                        else None,
                                    )
                                )
                save_button = gr.Button("保存本批次参数")
                with gr.Row():
                    dry_button = gr.Button("检查配置（不训练）")
                    prepare_button = gr.Button("只预处理")
                    train_button = gr.Button("开始训练")
                    all_button = gr.Button("预处理 → 训练 → 导出", variant="primary")
                gr.Markdown(
                    "训练会使用 GPU。请先在管理页面停掉占用显存的语音 / 翻唱服务；此页面不会代停其他项目。参数保存不会改写已有训练输出的历史批次。"
                )
                with gr.Accordion("本批次完整配置", open=False):
                    config_json = gr.JSON(label="training-config.json")
            with gr.Tab("03 · 续训与导出"):
                gr.Markdown(
                    "选择已保存的检查点。只剩 LoRA 权重的旧检查点可以导出，缺少优化器等状态时不能精确续训。"
                )
                checkpoint = gr.Dropdown(label="检查点")
                checkpoint_refresh = gr.Button("刷新检查点")
                resume_target = gr.Number(
                    value=275,
                    precision=0,
                    label="续训至第几步（必须高于当前步数，且不超过原调度总步数）",
                )
                resume_button = gr.Button("从所选检查点继续训练")
                output = gr.Textbox(
                    label="导出 LoRA GGUF 的完整路径（已有文件不会覆盖）"
                )
                export_button = gr.Button("导出所选检查点", variant="primary")
        gr.Markdown("---")
        with gr.Row():
            status = gr.Markdown("### 就绪 · 等待选择批次", elem_id="job-status")
            with gr.Column(scale=0, min_width=180):
                pause_button = gr.Button("保存并暂停", variant="stop")
                force_button = gr.Button("强制停止本任务", size="sm")
                gr.Markdown(
                    "安全暂停会等待当前步结束并保存。强停只保留之前完整的检查点。"
                )
        plot = gr.LinePlot(
            x="步数",
            y="损失",
            color="曲线",
            color_map={"训练": "#0d9488", "验证": "#e8933c"},
            height=240,
            label="损失曲线",
            x_title="训练步数",
            y_title="损失值",
        )
        logs = gr.Textbox(
            label="实时日志", lines=13, max_lines=20, interactive=False, autoscroll=True
        )
        files = gr.Textbox(label="检查点与输出文件", lines=3, interactive=False)
        timer = gr.Timer(2)
        runtime_inputs = [config_path, python, model, converter, deps]
        refresh.click(
            refresh_configs, config_path, config_path, api_name="refresh_batches"
        )
        config_path.change(
            load,
            config_path,
            [*fields, config_json, checkpoint, output],
            api_name="load_batch",
        )
        scan_button.click(
            scan, [source, mode], [items, table, stats], api_name="scan_dataset"
        )
        table.select(preview, items, [audio, transcript], api_name=False)
        create_button.click(
            create,
            [items, category, name, form, batch_name, validation, data_seed],
            [config_path, notice],
            api_name="prepare_batch",
        )
        save_button.click(
            save, [config_path, *fields], config_json, api_name="save_parameters"
        )
        dry_button.click(
            lambda *a: start(*a, dry_run=True),
            runtime_inputs,
            notice,
            api_name="check_config",
        )
        prepare_button.click(
            lambda *a: start(*a, stage="prepare"),
            runtime_inputs,
            notice,
            api_name="preprocess",
        )
        train_button.click(
            lambda *a: start(*a, stage="train"),
            runtime_inputs,
            notice,
            api_name="train",
        )
        all_button.click(start, runtime_inputs, notice, api_name="run_all")
        checkpoint_refresh.click(
            refresh_checkpoints, config_path, checkpoint, api_name="list_checkpoints"
        )
        resume_button.click(
            lambda *a: start(*a[:5], stage="train", checkpoint=a[5], target=a[6]),
            [*runtime_inputs, checkpoint, resume_target],
            notice,
            api_name="resume",
        )
        export_button.click(
            lambda *a: start(*a[:5], stage="export", checkpoint=a[5], output=a[6]),
            [*runtime_inputs, checkpoint, output],
            notice,
            api_name="export",
        )
        pause_button.click(
            lambda: call(jobs.stop), outputs=notice, api_name="pause", queue=False
        )
        force_button.click(
            lambda: call(jobs.stop, force=True),
            outputs=notice,
            api_name="stop",
            queue=False,
        )
        timer.tick(
            poll,
            config_path,
            [status, logs, plot, files],
            api_name="status",
            queue=False,
            show_progress="hidden",
        )
    return app, jobs


CSS = """
.gradio-container {max-width: 1240px !important; margin: auto;}
.breeze-header {padding: 24px 0 12px; border-bottom: 2px solid #0d9488; margin-bottom: 16px;}
.breeze-header .eyebrow {font: 12px Consolas, monospace; letter-spacing: .18em; color: #0d9488;}
.breeze-header h1 {font-size: 32px; letter-spacing: -.03em; margin: 6px 0;}
.breeze-header p {opacity: .65; margin: 0 0 6px;}
#job-status {padding: 12px 16px; border-left: 3px solid #0d9488;}
footer {display: none !important;}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Breeze LoRA training WebUI")
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    workspace = args.workspace_root.resolve()
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault(
        "GRADIO_TEMP_DIR", str(workspace / "training-preparation" / "webui-cache")
    )
    import gradio as gr

    app, _ = build_app(workspace)
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        font=["Microsoft YaHei", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    )
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=args.open,
        theme=theme,
        css=CSS,
    )


if __name__ == "__main__":
    main()
