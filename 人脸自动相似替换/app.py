#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人脸自动相似替换桌面入口。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from face_guard import FaceEngine, cleanup_command, prepare_job
from identity_lock.job_contract import (
    generation_attachments,
    normalize_auxiliary_references,
)
from style_library import default_style_library, load_style_library


ROOT = Path(__file__).resolve().parent
PROMPT_TEMPLATE = ROOT / "提示词模板.txt"
SETTINGS_FILE = ROOT / "本地设置.json"
OUTPUT_ROOT = ROOT / "任务输出"
STYLE_LIBRARY_ROOT = default_style_library(ROOT)
IMAGE_TYPES = [
    ("图片", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
    ("全部文件", "*.*"),
]


class PortraitRerollApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI 人像锁脸生成")
        self.geometry("1000x1000")
        self.minsize(880, 820)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen | None = None
        settings = self._load_settings()

        self.source_var = tk.StringVar(value=settings.get("source", ""))
        raw_references = settings.get("references", [])
        if not isinstance(raw_references, list):
            raw_references = []
        self.reference_paths = normalize_auxiliary_references(
            self.source_var.get().strip() or None,
            raw_references,
            require_exists=False,
        )
        self.reference_summary_var = tk.StringVar()
        self.style_var = tk.StringVar()
        self.style_mode_var = tk.StringVar(value=settings.get("style_mode", "prompt_library"))
        if self.style_mode_var.get() not in {"prompt_library", "image_reference"}:
            self.style_mode_var.set("prompt_library")
        self.style_search_var = tk.StringVar()
        self.preferred_styles_only_var = tk.BooleanVar(
            value=bool(settings.get("preferred_styles_only", True))
        )
        self.style_preset_id_var = tk.StringVar(value=settings.get("style_preset_id", ""))
        self.style_preview_var = tk.StringVar()
        self.style_presets = []
        self.visible_presets = []
        self.style_library_error = ""
        try:
            self.style_presets = load_style_library(STYLE_LIBRARY_ROOT)
        except (OSError, UnicodeError, ValueError) as exc:
            self.style_library_error = str(exc)
            self.style_mode_var.set("image_reference")
        self.task_name_var = tk.StringVar(value="写真重roll")
        self.threshold_var = tk.DoubleVar(value=float(settings.get("threshold", 0.90)))
        self.max_rolls_var = tk.IntVar(value=int(settings.get("max_rolls", 6)))
        self.cleanup_var = tk.BooleanVar(value=bool(settings.get("cleanup_private_data", True)))
        self.agent_var = tk.StringVar(value=settings.get("agent", "codex"))
        self.custom_cmd_var = tk.StringVar(value=settings.get(
            "custom_cmd",
            "codex exec -C {ROOT} --skip-git-repo-check -s workspace-write "
            "-i {ATTACHMENTS} -o {OUTPUT} {INSTRUCTION}",
        ))
        self.status_var = tk.StringVar(value="先选择本人照片，再选择系统摄影风格或上传风格参考图。")
        self.last_session: Path | None = None

        self._build_ui(settings.get("prompt"))
        self.after(120, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_settings(self):
        data = {
            "source": self.source_var.get().strip(),
            "references": list(self.reference_paths),
            "style_mode": self.style_mode_var.get(),
            "style_preset_id": self.style_preset_id_var.get(),
            "preferred_styles_only": self.preferred_styles_only_var.get(),
            "threshold": round(self.threshold_var.get(), 3),
            "max_rolls": self.max_rolls_var.get(),
            "cleanup_private_data": self.cleanup_var.get(),
            "agent": self.agent_var.get(),
            "custom_cmd": self.custom_cmd_var.get(),
            "prompt": self.prompt.get("1.0", "end-1c").strip(),
        }
        SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_ui(self, saved_prompt: str | None):
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="AI 人像锁脸生成", font=("", 22, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="图1始终是唯一身份来源；可用系统摄影提示词锁定风格，也可上传图2精确参考动作与构图。",
            foreground="#444",
        ).pack(anchor="w", pady=(4, 16))

        form = ttk.Frame(outer)
        form.pack(fill="x")
        self._path_row(form, 0, "图片1（本人）", self.source_var, self._choose_source, "唯一身份来源；会自动记住")
        ttk.Label(form, text="辅助真人参考", width=14).grid(row=2, column=0, sticky="nw", pady=(10, 0))
        reference_controls = ttk.Frame(form)
        reference_controls.grid(row=2, column=1, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        ttk.Label(reference_controls, textvariable=self.reference_summary_var).pack(side="left")
        self.add_reference_button = ttk.Button(
            reference_controls, text="添加照片…", command=self._choose_references
        )
        self.add_reference_button.pack(side="right")
        self.clear_reference_button = ttk.Button(
            reference_controls, text="清空", command=self._clear_references
        )
        self.clear_reference_button.pack(side="right", padx=6)
        self.remove_reference_button = ttk.Button(
            reference_controls, text="移除所选", command=self._remove_selected_references
        )
        self.remove_reference_button.pack(side="right")
        self.reference_list = tk.Listbox(
            form, height=3, exportselection=False, selectmode=tk.EXTENDED,
            font=("PingFang SC", 10),
        )
        self.reference_list.grid(row=3, column=1, columnspan=2, sticky="ew", padx=8, pady=(4, 0))
        self.reference_mode_hint = ttk.Label(
            form,
            text="最多 7 张；第一张图片1仍是唯一基准，其余只补充本人不同角度。",
            foreground="#666",
        )
        self.reference_mode_hint.grid(row=4, column=1, columnspan=2, sticky="w", padx=8)
        ttk.Label(form, text="任务名称", width=14).grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.task_name_var).grid(row=5, column=1, sticky="ew", padx=8, pady=(8, 0))
        form.columnconfigure(1, weight=1)
        self._refresh_reference_list()

        style_source = ttk.LabelFrame(outer, text="风格来源（二选一）", padding=12)
        style_source.pack(fill="x", pady=(14, 0))
        self.prompt_style_radio = ttk.Radiobutton(
            style_source,
            text="系统摄影风格（更利于稳定锁脸）",
            value="prompt_library",
            variable=self.style_mode_var,
            command=self._on_style_mode_changed,
        )
        self.prompt_style_radio.pack(side="left")
        ttk.Radiobutton(
            style_source,
            text="上传风格参考图（更精确模仿动作与构图）",
            value="image_reference",
            variable=self.style_mode_var,
            command=self._on_style_mode_changed,
        ).pack(side="left", padx=(24, 0))
        if self.style_library_error:
            self.prompt_style_radio.configure(state="disabled")

        style_content = ttk.Frame(outer)
        style_content.pack(fill="x")
        self.prompt_style_frame = ttk.Frame(style_content)
        search_row = ttk.Frame(self.prompt_style_frame)
        search_row.pack(fill="x", pady=(10, 5))
        ttk.Label(search_row, text="搜索风格", width=14).pack(side="left")
        ttk.Entry(search_row, textvariable=self.style_search_var).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Checkbutton(
            search_row,
            text="仅显示锁脸优先",
            variable=self.preferred_styles_only_var,
            command=self._filter_styles,
        ).pack(side="right")
        list_row = ttk.Frame(self.prompt_style_frame)
        list_row.pack(fill="x")
        self.style_list = tk.Listbox(list_row, height=5, exportselection=False, font=("PingFang SC", 11))
        self.style_list.pack(side="left", fill="x", expand=True)
        style_scroll = ttk.Scrollbar(list_row, orient="vertical", command=self.style_list.yview)
        style_scroll.pack(side="right", fill="y")
        self.style_list.configure(yscrollcommand=style_scroll.set)
        self.style_list.bind("<<ListboxSelect>>", self._on_style_selected)
        ttk.Label(
            self.prompt_style_frame,
            textvariable=self.style_preview_var,
            foreground="#555",
            wraplength=940,
            justify="left",
        ).pack(fill="x", pady=(6, 0))

        self.image_style_frame = ttk.Frame(style_content)
        self._path_row(
            self.image_style_frame,
            0,
            "图片2（风格）",
            self.style_var,
            self._choose_style,
            "只提取动作、服饰、构图和光线；人物脸会被超粗马赛克覆盖",
        )
        self.image_style_frame.columnconfigure(1, weight=1)
        self.style_search_var.trace_add("write", lambda *_args: self._filter_styles())
        self._filter_styles()
        self._on_style_mode_changed()

        options = ttk.LabelFrame(outer, text="自动重 roll", padding=12)
        options.pack(fill="x", pady=14)
        ttk.Label(options, text="身份严格度").grid(row=0, column=0, sticky="w")
        threshold = ttk.Scale(options, from_=0.75, to=0.97, variable=self.threshold_var, orient="horizontal")
        threshold.grid(row=0, column=1, sticky="ew", padx=10)
        self.threshold_label = ttk.Label(options, width=8)
        self.threshold_label.grid(row=0, column=2, sticky="e")
        threshold.configure(command=lambda _value: self._update_threshold())
        self._update_threshold()
        ttk.Label(options, text="最多轮数").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(options, from_=1, to=20, textvariable=self.max_rolls_var, width=8).grid(row=1, column=1, sticky="w", padx=10, pady=(10, 0))
        ttk.Label(options, text="默认 6，防止无限消耗生图额度", foreground="#666").grid(row=1, column=2, sticky="e", pady=(10, 0))
        ttk.Label(options, text="风格隐私").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.privacy_value_label = ttk.Label(options)
        self.privacy_value_label.grid(row=2, column=1, sticky="w", padx=10, pady=(10, 0))
        self.privacy_hint_label = ttk.Label(options, foreground="#666")
        self.privacy_hint_label.grid(row=2, column=2, sticky="e", pady=(10, 0))
        ttk.Label(options, text="执行 AI").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(options, textvariable=self.agent_var, state="readonly",
                     values=("codex", "claude", "custom"), width=14
                     ).grid(row=3, column=1, sticky="w", padx=10, pady=(10, 0))
        ttk.Label(options, text="codex=OpenAI CLI｜claude=Claude Code CLI｜custom=自定义命令", foreground="#666"
                  ).grid(row=3, column=2, sticky="e", pady=(10, 0))
        ttk.Label(options, text="自定义命令（仅 custom；占位符 {ROOT} {SOURCE} {STYLE} {ATTACHMENTS} {OUTPUT} {INSTRUCTION}）",
                  foreground="#666").grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(options, textvariable=self.custom_cmd_var).grid(
            row=4, column=1, columnspan=2, sticky="ew", padx=10, pady=(4, 0))
        ttk.Checkbutton(
            options,
            text="完成后清理任务中的人脸复制件、身份向量和候选图（保留最终图与报告）",
            variable=self.cleanup_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 0))
        options.columnconfigure(1, weight=1)

        prompt_box = ttk.LabelFrame(outer, text="锁脸规则与补充要求（系统风格会自动压缩加入）", padding=10)
        prompt_box.pack(fill="both", expand=True)
        self.prompt = tk.Text(prompt_box, wrap="word", height=13, font=("PingFang SC", 12))
        self.prompt.pack(fill="both", expand=True)
        default_prompt = PROMPT_TEMPLATE.read_text(encoding="utf-8")
        self.prompt.insert("1.0", saved_prompt or default_prompt)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(14, 0))
        self.start_button = ttk.Button(actions, text="开始锁脸生成", command=self._start)
        self.start_button.pack(side="left")
        self.open_button = ttk.Button(actions, text="打开最近任务目录", command=self._open_recent)
        self.open_button.pack(side="left", padx=10)
        self.stop_button = ttk.Button(actions, text="停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="right")

        ttk.Label(outer, textvariable=self.status_var).pack(fill="x", anchor="w", pady=(12, 5))
        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill="x")
        self.log = tk.Text(outer, height=10, wrap="word", font=("Menlo", 11), state="disabled")
        self.log.pack(fill="both", expand=True, pady=(10, 0))
        self._on_style_mode_changed()

    def _path_row(self, parent, row, label, variable, command, hint):
        ttk.Label(parent, text=label, width=14).grid(row=row, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=(5, 0))
        ttk.Button(parent, text="选择…", command=command).grid(row=row, column=2, sticky="e", pady=(5, 0))
        ttk.Label(parent, text=hint, foreground="#666").grid(row=row + 1, column=1, sticky="w", padx=8)

    def _choose_source(self):
        path = filedialog.askopenfilename(title="选择图片1：本人照片", filetypes=IMAGE_TYPES)
        if path:
            self.source_var.set(path)
            self.reference_paths = normalize_auxiliary_references(
                path, self.reference_paths, require_exists=False
            )
            self._refresh_reference_list()

    def _choose_references(self):
        paths = filedialog.askopenfilenames(
            title="添加本人不同角度的真人参考（最多 7 张）", filetypes=IMAGE_TYPES
        )
        if not paths:
            return
        try:
            self.reference_paths = normalize_auxiliary_references(
                self.source_var.get().strip() or None,
                [*self.reference_paths, *paths],
                require_exists=True,
            )
        except ValueError as exc:
            messagebox.showerror("辅助参考无效", str(exc))
            return
        self._refresh_reference_list()

    def _remove_selected_references(self):
        selected = set(self.reference_list.curselection())
        self.reference_paths = [
            path for index, path in enumerate(self.reference_paths) if index not in selected
        ]
        self._refresh_reference_list()

    def _clear_references(self):
        self.reference_paths = []
        self._refresh_reference_list()

    def _refresh_reference_list(self):
        if not hasattr(self, "reference_list"):
            return
        self.reference_list.delete(0, "end")
        for index, path in enumerate(self.reference_paths, start=2):
            self.reference_list.insert("end", f"图片{index}：{Path(path).name}")
        self.reference_summary_var.set(
            f"已选 {len(self.reference_paths)} 张辅助图"
            if self.reference_paths
            else "未添加（可选）"
        )

    def _choose_style(self):
        path = filedialog.askopenfilename(title="选择图片2：动作/服饰/滤镜参考", filetypes=IMAGE_TYPES)
        if path:
            self.style_var.set(path)

    def _filter_styles(self):
        if not hasattr(self, "style_list"):
            return
        query = self.style_search_var.get().strip().casefold()
        preferred_only = self.preferred_styles_only_var.get()
        selected_id = self.style_preset_id_var.get()
        self.visible_presets = [
            preset for preset in self.style_presets
            if (not preferred_only or preset.compatibility == "preferred" or preset.preset_id == selected_id)
            and (not query or query in f"{preset.preset_id} {preset.title}".casefold())
        ]
        self.style_list.delete(0, "end")
        for preset in self.visible_presets:
            title = preset.title if len(preset.title) <= 76 else preset.title[:75] + "…"
            self.style_list.insert("end", f"[{preset.compatibility_label}] {title}")
        target_index = next(
            (index for index, preset in enumerate(self.visible_presets) if preset.preset_id == selected_id),
            0 if self.visible_presets else None,
        )
        if target_index is not None:
            self.style_list.selection_set(target_index)
            self.style_list.see(target_index)
            self._select_preset(self.visible_presets[target_index])
        else:
            self.style_preset_id_var.set("")
            self.style_preview_var.set("没有符合条件的风格；可清空搜索或取消“仅显示锁脸优先”。")

    def _on_style_selected(self, _event=None):
        selection = self.style_list.curselection()
        if selection and selection[0] < len(self.visible_presets):
            self._select_preset(self.visible_presets[selection[0]])

    def _select_preset(self, preset):
        self.style_preset_id_var.set(preset.preset_id)
        detail = f"{preset.compatibility_label}｜{preset.compact_prompt}"
        if preset.compatibility_reasons:
            detail += "｜注意：" + "；".join(preset.compatibility_reasons)
        self.style_preview_var.set(detail)

    def _selected_preset(self):
        preset_id = self.style_preset_id_var.get()
        return next((item for item in self.style_presets if item.preset_id == preset_id), None)

    def _on_style_mode_changed(self):
        if not hasattr(self, "prompt_style_frame"):
            return
        prompt_mode = self.style_mode_var.get() == "prompt_library"
        self.prompt_style_frame.pack_forget()
        self.image_style_frame.pack_forget()
        if prompt_mode:
            self.prompt_style_frame.pack(fill="x")
            privacy_value = "无需图2"
            privacy_hint = "发送已选真人参考；摄影风格在本地整理为精简文本"
            button_text = "使用系统风格开始锁脸生成"
            reference_state = "normal"
            reference_hint = "最多 7 张；第一张图片1仍是唯一基准，其余只补充本人不同角度。"
        else:
            self.image_style_frame.pack(fill="x", pady=(5, 0))
            privacy_value = "超粗马赛克（固定）"
            privacy_hint = "图2原图不传给外部服务"
            button_text = "遮蔽图2并开始锁脸生成"
            reference_state = "disabled"
            reference_hint = "图片2替换模式暂停辅助真人图，固定只发送图1与已遮脸图2。"
        for widget in (
            self.add_reference_button,
            self.remove_reference_button,
            self.clear_reference_button,
            self.reference_list,
        ):
            widget.configure(state=reference_state)
        self.reference_mode_hint.configure(text=reference_hint)
        if hasattr(self, "privacy_value_label"):
            self.privacy_value_label.configure(text=privacy_value)
            self.privacy_hint_label.configure(text=privacy_hint)
        if hasattr(self, "start_button"):
            self.start_button.configure(text=button_text)

    def _update_threshold(self):
        self.threshold_label.configure(text=f"{self.threshold_var.get():.3f}")

    def _append_log(self, message: str):
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_busy(self, busy: bool):
        self.start_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _start(self):
        source = Path(self.source_var.get().strip()).expanduser()
        style_mode = self.style_mode_var.get()
        style = Path(self.style_var.get().strip()).expanduser() if style_mode == "image_reference" else None
        preset = self._selected_preset() if style_mode == "prompt_library" else None
        prompt = self.prompt.get("1.0", "end-1c").strip()
        if not source.is_file():
            messagebox.showerror("缺少图片1", "请选择有效的本人照片。")
            return
        try:
            references = normalize_auxiliary_references(
                source,
                self.reference_paths,
                require_exists=True,
            )
        except ValueError as exc:
            messagebox.showerror("辅助参考无效", str(exc))
            return
        if style_mode != "prompt_library":
            references = []
        if style_mode == "image_reference" and (style is None or not style.is_file()):
            messagebox.showerror("缺少图片2", "请选择有效的风格参考图。")
            return
        if style_mode == "prompt_library" and preset is None:
            messagebox.showerror("缺少摄影风格", "请从系统摄影风格列表中选择一个风格。")
            return
        if not prompt:
            messagebox.showerror("缺少提示词", "提示词不能为空。")
            return
        if not 1 <= self.max_rolls_var.get() <= 20:
            messagebox.showerror("轮数错误", "最多轮数必须在 1 到 20 之间。")
            return
        if style is not None and source.resolve() == style.resolve():
            messagebox.showerror("图片重复", "图片1与图片2不能是同一文件。")
            return
        if preset and preset.compatibility != "preferred" and not messagebox.askokcancel(
            "锁脸可观测性提醒",
            f"这个风格标记为“{preset.compatibility_label}”："
            f"{'；'.join(preset.compatibility_reasons) or '脸部可能不够清晰'}。\n\n"
            "系统仍会执行身份评分，但可能需要先调整构图才能可靠验脸。是否继续？",
        ):
            return
        privacy_message = (
            f"图1本人照片与 {len(references)} 张辅助真人参考会发送到 config.json 配置的视觉/生图服务；"
            "选中的系统摄影风格只以本地精简文本发送，不会上传风格库原始文档。\n\n"
            if style_mode == "prompt_library"
            else "图1本人照片与已遮脸的图2会发送到 config.json 配置的视觉/生图服务；"
            "图2原图只在本机遮脸与本地评分时读取，不会传给外部服务。\n\n"
        )
        if not messagebox.askokcancel(
            "确认外部图片传输",
            privacy_message + "请确认你有权上传这些照片，并已接受该服务的隐私规则。",
        ):
            return

        self._save_settings()
        self._set_busy(True)
        if style_mode == "prompt_library":
            self.status_var.set("正在建立身份任务并整理系统摄影风格…")
            self._append_log(f"开始预处理。系统风格“{preset.title}”将在本地压缩为最多 8 个画面约束。")
        else:
            self.status_var.set("正在本地检测并彻底遮蔽图片2人脸…")
            self._append_log("开始预处理。图片2原图只在本地人脸检测器中读取，不会传给任何生图工具/AI。")
        values = {
            "source": str(source.resolve()),
            "style": str(style.resolve()) if style is not None else None,
            "style_preset": str((STYLE_LIBRARY_ROOT / preset.source_file).resolve()) if preset else None,
            "references": references,
            "prompt": prompt,
            "name": self.task_name_var.get().strip() or "写真重roll",
            "threshold": self.threshold_var.get(),
            "max_rolls": self.max_rolls_var.get(),
            "cleanup_private_data": self.cleanup_var.get(),
        }
        threading.Thread(target=self._worker, args=(values,), daemon=True).start()

    def _build_agent_command(self, job: dict, instruction: str) -> list[str]:
        """按用户选的 agent 类型构造执行命令。
        codex=OpenAI Codex CLI；claude=Claude Code CLI；custom=自定义命令模板。"""
        import shlex
        kind = self.agent_var.get()
        source = job["source_image"]
        style = job["masked_style_image"]
        attachments = generation_attachments(job)
        out = str(Path(job["session_dir"]) / "Agent最终消息.txt")
        if kind == "claude":
            return ["claude", "-p", instruction, *attachments]
        if kind == "custom":
            try:
                cmd = self.custom_cmd_var.get().format(
                    ROOT=str(ROOT), SOURCE=source, STYLE=style or "",
                    ATTACHMENTS=shlex.join(attachments),
                    OUTPUT=out, INSTRUCTION=instruction)
            except KeyError as exc:
                raise RuntimeError(
                    f"自定义命令里有不认识的占位符：{exc}。"
                    f"可用占位符：{{ROOT}} {{SOURCE}} {{STYLE}} {{ATTACHMENTS}} {{OUTPUT}} {{INSTRUCTION}}")
            parts = shlex.split(cmd)
            if not parts:
                raise RuntimeError("自定义命令为空。")
            return parts
        # 默认 codex
        return ["codex", "exec", "-C", str(ROOT), "--skip-git-repo-check",
                "-s", "workspace-write", "-i", *attachments,
                "-o", out, instruction]

    def _worker(self, values: dict):
        try:
            kind = self.agent_var.get()
            exe = {"codex": "codex", "claude": "claude"}.get(kind)
            if exe and not shutil_which(exe):
                raise RuntimeError(
                    f"没有找到 {exe} 命令，请先安装并登录对应 CLI。"
                    f"（codex=OpenAI Codex；claude=Claude Code）"
                )
            # 复用 argparse Namespace 结构，保证 GUI 与命令行走同一条安全链路。
            from argparse import Namespace

            job = prepare_job(
                Namespace(
                    source=values["source"],
                    style=values["style"],
                    style_preset=values["style_preset"],
                    reference=values["references"],
                    prompt=values["prompt"],
                    name=values["name"],
                    output=str(OUTPUT_ROOT),
                    threshold=values["threshold"],
                    leakage_threshold=0.70,
                    identity_margin=0.08,
                    max_rolls=values["max_rolls"],
                    privacy_mode="mosaic",
                    mosaic_blocks=4,
                    mask_side=0.18,
                    mask_top=0.22,
                    mask_bottom=0.18,
                )
            )
            session = Path(job["session_dir"])
            self.events.put(("session", session))
            if job["style_source"]["kind"] == "image_reference":
                self.events.put(("log", f"已遮蔽图片2中的 {job['style_faces_masked']} 张脸：{job['masked_style_image']}"))
                self.events.put(("status", "遮脸完成，AI 正在自动生成并评分…"))
                attachment_note = "你收到的两张附件依次是图片1本人和已彻底遮脸的图片2。不要寻找图片2原图；"
            else:
                self.events.put(("log", f"已载入系统摄影风格：{job['style_source']['preset_title']}"))
                self.events.put(("log", f"已载入 {job['generation_reference_count']} 张本人真人参考图。"))
                self.events.put(("status", "风格整理完成，AI 正在自动生成并评分…"))
                attachment_note = (
                    f"你收到 {job['generation_reference_count']} 张同一人的真人附件，"
                    "第一张是唯一身份基准，其余只补充不同角度；系统摄影风格已压缩写入提示词，不存在图片2；"
                )

            instruction = (
                "AUTO_PORTRAIT_REROLL\n"
                f"JOB_FILE={job['job_file']}\n"
                "严格读取本目录 AGENTS.md 并完成全部自动重roll流程。"
                f"{attachment_note}"
                "不要询问用户；达到门槛或最大轮数后完成最终报告。"
            )
            command = self._build_agent_command(job, instruction)
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )
            assert self.process.stdout is not None
            with (session / "运行日志.txt").open("w", encoding="utf-8") as log_file:
                for line in self.process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    self.events.put(("log", line.rstrip()))
            return_code = self.process.wait()
            self.process = None
            if return_code != 0:
                raise RuntimeError(f"Codex 自动任务退出，代码 {return_code}。请查看运行日志。")
            report = session / "最终报告.json"
            if not report.is_file():
                raise RuntimeError("AI 已结束，但没有生成最终报告。请查看 Agent最终消息.txt。")
            if values["cleanup_private_data"]:
                cleanup = cleanup_command(Namespace(job=job["job_file"]))
                self.events.put(("log", "已清理任务私密派生数据：" + "、".join(cleanup["removed"])))
            data = json.loads(report.read_text(encoding="utf-8"))
            data["_ui_style_source"] = job.get("style_source", {})
            self.events.put(("done", data))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "session":
                    self.last_session = Path(payload)
                elif kind == "done":
                    self._set_busy(False)
                    selected = payload["selected"]
                    identity_score = selected.get("source_similarity")
                    score_text = f"图1身份 {identity_score:.3f}" if identity_score is not None else "图1身份未测得"
                    style_source = payload.get("_ui_style_source", {})
                    if style_source.get("kind") == "image_reference":
                        style_score = selected.get("style_similarity")
                        style_text = f"图2身份泄漏检查 {style_score:.3f}" if style_score is not None else "图2身份泄漏检查不适用"
                    else:
                        style_text = f"系统风格：{style_source.get('preset_title') or '未指定'}"
                    self.status_var.set(f"完成：{payload['status']}；{score_text}；{style_text}")
                    self._append_log(f"最终图片：{payload['final_image']}")
                    messagebox.showinfo("任务完成", f"{payload['status']}\n\n结果：{payload['final_image']}")
                elif kind == "error":
                    self._set_busy(False)
                    self.status_var.set("任务失败，请查看日志。")
                    self._append_log("错误：" + str(payload))
                    messagebox.showerror("任务失败", str(payload))
        except queue.Empty:
            pass
        self.after(120, self._poll_events)

    def _stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self._append_log("已请求停止 AI 自动任务。已生成的候选不会删除。")
            self.status_var.set("正在停止…")

    def _open_recent(self):
        path = self.last_session or OUTPUT_ROOT
        path.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["open", str(path)])

    def _on_close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("任务仍在运行", "关闭会停止当前自动任务，确定关闭吗？"):
                return
            self.process.terminate()
        self._save_settings()
        self.destroy()


def shutil_which(command: str) -> str | None:
    # 延迟导入，保持界面启动路径清晰。
    import shutil
    return shutil.which(command)


if __name__ == "__main__":
    try:
        FaceEngine()
    except Exception as exc:
        messagebox.showerror("启动失败", str(exc))
        raise SystemExit(1)
    PortraitRerollApp().mainloop()
