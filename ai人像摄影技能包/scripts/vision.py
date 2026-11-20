#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision.py — 人脸特征建模 + 迭代修正（把参考图交给能看图的模型，翻译成锁脸文字特征）

用法:
  建模（默认）:
    python3 scripts/vision.py <参考图> [更多参考图...] [--out 特征.md] [--angle 正脸|45度侧脸|90度侧脸|斜上方|斜下方]
    已有 provisional 档案仅在显式加 --replace-provisional 时允许替换；validated 档案永不自动覆盖。
  迭代修正（不像时）:
    python3 scripts/vision.py --fix <参考图> --diff 不像的生成图.jpg --notes "不像点描述" [--base 原特征.md] [--out 修正结果.md]
  通用:
    --config <路径>  指定 config.json（默认自动找）

内置能力:
  - 10 项特征自检：模型漏项自动追问补全
  - 多角度锁脸段：按出图角度（正脸/侧脸/俯仰）调整锁脸段措辞
  - 模型降级重试：主模型（如 gemini-2.5-pro）失败时自动降级到 fallback（如 flash）
  - 大图自动压缩：>4MB 先压缩再传，避免超接口限制

配置: 读 config.json 的 vision 部分（host/key/model/fallback_model/agent_can_see）。
"""
import base64, json, os, re, subprocess, sys, tempfile, urllib.request, urllib.error

from profile_guard import guard_output, write_provisional_metadata

ITEM_NAMES = {
    1: "眼距（相对脸宽：偏宽/适中/偏窄 + 比例）",
    2: "脸型长宽比与整体轮廓",
    3: "眼-鼻-嘴整体空间位置关系",
    4: "眼睛（眼型/眼裂/眼角/眉眼距/卧蚕）",
    5: "鼻子（鼻长/鼻梁/鼻翼）",
    6: "眉毛（粗细/浓淡/走向/眉峰）",
    7: "嘴唇（厚薄/唇形，克制）",
    8: "发型与发色（保留本人轮廓）",
    9: "是否戴眼镜及样式",
    10: "皮肤/痣/穿着/情绪（一句话）",
}

ANGLE_HINTS = {
    "正脸": "",
    "45度侧脸": "注意：本次出图是45°侧脸，锁脸段要保留45°角度下的真实透视"
                "（近侧眼与远侧眼大小关系、鼻子前突、颌面轮廓），不要修成对称正脸。",
    "90度侧脸": "注意：本次出图是90°侧脸，锁脸段要强调侧面轮廓"
                "（额头-鼻-唇-下巴的侧面线条、下颌缘、颈颌角），不要对称化。",
    "斜上方": "注意：本次出图是斜上方45°俯拍机位，锁脸段要保留俯拍下的面部比例"
              "（额头相对显大、下颌相对收窄），不要改变本人比例。",
    "斜下方": "注意：本次出图是斜下方45°仰拍机位，锁脸段要保留仰拍下的面部比例"
              "（下颌相对显宽），不要改变本人比例。",
}

MODEL_PROMPT_BASE = """这是同一人的参考照片。供AI生成写真时锁脸。请按【人类认脸注意力优先级】分级输出特征：优先级越高写得越细越精确，优先级低的部位一句话带过。按条目编号用中文回答（编号在行首）：

【P0 · 核心空间关系 · 最重要，必须量化精确】
1. 眼距（相对脸宽：偏宽/适中/偏窄 + 大致比例）
2. 脸型长宽比与整体轮廓（长宽比、方圆/鹅蛋/瓜子、颧骨、下颌、下巴、三庭大致比例）
3. 眼-鼻-嘴的整体空间位置关系（眉眼距、鼻口距、口颏距大致比例）

【P1 · 眼睛优先 · 身份第一采样点，详细描述】
4. 眼睛（眼型单/双/内双、眼裂形状与长度、眼角方向、眉眼距、卧蚕、眼神）
5. 鼻子（鼻长、鼻梁高低、鼻头形状、鼻翼宽度）

【P2 · 结构外显 · 适中精确】
6. 眉毛（粗细、浓淡、走向、眉峰）
7. 嘴唇（厚薄、唇形——克制，嘴偏情绪不偏身份）
8. 发型与发色（长度、卷直、纹理、刘海，保留本人轮廓）
9. 是否戴眼镜及眼镜样式（框型）

【P3 · 低优先级 · 一句话或省略，不要过度描述】
10. 皮肤质感、痣/胡青、穿着、场景、情绪

每一条描述中性客观，不要评价美丑。

描述完以上10项后，最后额外输出一份「基线锁脸段成品」。它不是完整10项复述，而是正式首轮生图使用的身份基线：
基线锁脸段成品：以第一张参考照片（X1）里这位年轻亚洲[女性/男性]的脸为唯一基准：脸型、五官与发型轮廓不得改变。然后按 P0>P1>P2 的顺序内嵌身份特征。
要求：
- P0 与 P1 必须完整、精确；P2 只保留稳定的结构外显特征，尤其是眼镜和发型轮廓；
- P3 的皮肤、痣、胡青、穿着、场景、情绪完全不写入基线；
- 不受120字限制，但保持在一段内，不重复同一特征，不写"见上"或省略号；
- 性别按参考图自动填对并删除方括号占位符。"""

FIX_PROMPT = """你是人像锁脸修正专家。给你三样东西：
1. 参考人的照片（锁脸基准，身份唯一依据）
2. 一张 AI 生成的效果图（不像）
3. 使用者描述的不像点

任务：找出效果图相对参考人最影响辨识度的一个偏差。不要说"整体不像"，也不要同时修多个部位。输出严格使用以下格式：

偏差分析：一句话说明候选相对真人的主要偏差。
唯一修正重点：从 face_shape / eye_distance / eyes / nose / mouth / hair_glasses / global_identity 中选一个。
修正指令：一条正向、可直接追加到下一轮提示词的指令，不超过80个中文字符。

要求：
- 只输出一个修正重点，不输出完整锁脸段，不重写10项档案
- 使用正向描述，避免堆叠多个"不要"
- 如果下面给了原特征描述，只用它帮助判断，不得覆盖已验证基线
- 低优先级皮肤、情绪、穿着不得成为身份修正重点"""


def load_config(config_path=None):
    if not config_path:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (os.path.join(here, "..", "config.json"),
                     os.path.join(here, "config.json")):
            if os.path.exists(cand):
                config_path = cand
                break
    if not config_path:
        sys.exit("[vision.py] 找不到 config.json。请先按使用教程完成首次配置（或把 config.example.json 复制为 config.json 并填入你的 API 信息）。")
    with open(config_path) as f:
        return json.load(f), config_path


def http_json(host, payload, key, timeout=300):
    body = json.dumps(payload).encode("utf-8")
    url = host.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.read().decode("utf-8", "replace")


def b64(p):
    with open(p, "rb") as file:
        data = file.read()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    else:
        raise ValueError(f"不支持或无法识别的图片格式：{p}")
    return "data:%s;base64," % mime + base64.b64encode(data).decode()


def _maybe_compress(paths):
    """>4MB 的图先压缩到临时文件再传。压缩工具择优：PIL → opencv(cv2) → macOS sips → 兜底警告。
    Windows 用户用 .venv 的 python 跑就有 opencv，能压缩；纯标准库+无工具时走兜底。"""
    temps, result = [], []
    for p in paths:
        size = os.path.getsize(p)
        if size <= 4 * 1024 * 1024:
            result.append(p)
            continue
        tmp, ok = None, False
        try:
            from PIL import Image
            im = Image.open(p)
            im.thumbnail((1536, 1536))
            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            im.convert("RGB").save(tmp, "JPEG", quality=88)
            ok = True
        except Exception:
            ok = False
        if not ok:
            try:
                import cv2
                img = cv2.imread(p)
                if img is not None:
                    h, w = img.shape[:2]
                    scale = min(1.0, 1536.0 / max(h, w))
                    if scale < 1.0:
                        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                                         interpolation=cv2.INTER_AREA)
                    fd, tmp = tempfile.mkstemp(suffix=".jpg")
                    os.close(fd)
                    cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    ok = True
            except Exception:
                ok = False
        if not ok:
            try:
                fd, tmp = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                subprocess.run(["sips", "-Z", "1536", "-s", "format", "jpeg", p, "--out", tmp],
                               check=True, capture_output=True)
                ok = True
            except Exception:
                ok = False
        if not ok:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
            print(f"[vision.py] 警告：{os.path.basename(p)} 有 {size/1024/1024:.0f}MB 且无可用压缩工具(PIL/opencv/sips)，可能超接口限制")
            result.append(p)
            continue
        temps.append(tmp)
        result.append(tmp)
    return result, temps


def _parse_response(raw):
    parts = []
    for line in re.split(r"\r?\n", raw):
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line.startswith("{"):
            try:
                o = json.loads(line)
            except Exception:
                continue
            for c in (o.get("choices") or []):
                m = c.get("message") or {}
                d = c.get("delta") or {}
                if m.get("content"):
                    parts.append(m["content"])
                elif d.get("content"):
                    parts.append(d["content"])
    return "".join(parts)


def _fallback_model(v):
    if v.get("fallback_model"):
        return v["fallback_model"]
    m = v.get("model", "")
    if "pro" in m:
        return m.replace("pro", "flash")
    return None


def _call_vision(v, content, label):
    """调用 vision 接口；无有效返回且有 fallback 时降级重试。返回 (text, model_used, raw)。"""
    models = [v["model"]]
    fb = _fallback_model(v)
    if fb and fb != v["model"]:
        models.append(fb)
    for i, model in enumerate(models):
        payload = {"model": model, "max_tokens": 2048,
                   "messages": [{"role": "user", "content": content}]}
        print(f"[vision.py] {label}（模型 {model}）…")
        raw = http_json(v["host"], payload, v["key"])
        text = _parse_response(raw)
        if text.strip():
            return text, model, raw
        if i < len(models) - 1:
            print(f"[vision.py] 模型 {model} 无有效返回，降级到 {models[i + 1]} 重试…")
    return None, models[0], raw


def _check_items(text):
    found = set()
    for m in re.finditer(r"(?m)^\s*(\d{1,2})[\.、．]\s*\**", text):
        n = int(m.group(1))
        if 1 <= n <= 10:
            found.add(n)
    return [n for n in range(1, 11) if n not in found]


def _complete_items(v, paths, missing, angle):
    if not missing:
        return None
    prompt = (f"请只补充描述这张参考照片的人脸特征中缺失的 {len(missing)} 项，编号开头，不要写其他内容：\n"
              + "\n".join(f"{n}. {ITEM_NAMES[n]}" for n in missing))
    if angle and angle != "正脸":
        prompt += f"\n（注意本次出图目标角度：{angle}）"
    content = [{"type": "text", "text": prompt}]
    for p in paths:
        content.append({"type": "image_url", "image_url": {"url": b64(p)}})
    text, _model, _raw = _call_vision(v, content, "补全缺失特征")
    return text.strip() if text else None


def _split_lock(text):
    for marker in ("基线锁脸段成品：", "锁脸段成品："):
        if marker in text:
            feature, lock = text.split(marker, 1)
            return feature.strip(), ("基线锁脸段成品：" + lock.strip())
    return text.strip(), None


def _write_output(path, feature_part, lock_part, missing, model, references, title="# 图1 特征建模"):
    lines = [title, "", "档案状态：provisional（机器暂定，真人确认前不得当作已验证档案）", "",
             f"建模模型：{model}", f"真人参考图数量：{len(references)}", "", feature_part, ""]
    if lock_part:
        lines += ["---", "", "## 首轮身份基线（P0/P1/P2；正式首轮使用）", "", lock_part, ""]
    lines += ["---", "",
              "## 负面段（可直接粘贴到提示词结尾）", "",
              "不要塑料皮肤、不要过度磨皮、不要五官变形、不要头发糊成块、"
              "不要动漫感、不要文字乱码、不要背景杂乱", ""]
    if missing:
        lines += ["---", "", f"## 缺失待补项（自检未覆盖，需 AI 手工补）：{missing}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    write_provisional_metadata(path, references, model)
    print(f"[vision.py] 已写入: {os.path.abspath(path)}")


def cmd_model(v, paths, out_path, angle, replace_provisional=False):
    if out_path:
        guard_output(out_path, replace_provisional=replace_provisional)
    prepared, temps = _maybe_compress(paths)
    prompt = MODEL_PROMPT_BASE
    hint = ANGLE_HINTS.get(angle or "正脸", "")
    if hint:
        prompt += "\n\n" + hint
    content = [{"type": "text", "text": prompt}]
    for p in prepared:
        content.append({"type": "image_url", "image_url": {"url": b64(p)}})
    text, model, raw = _call_vision(v, content, f"分析 {len(prepared)} 张参考图")
    if not text:
        sys.exit("[vision.py] 模型未返回有效内容（负载高或额度不足），不要继续生图。原始响应：\n" + raw[:300])

    feature_part, lock_part = _split_lock(text)
    missing = _check_items(feature_part)
    if missing:
        print(f"[vision.py] 检测到缺失特征项：{missing}，自动补全…")
        add = _complete_items(v, prepared, missing, angle)
        if add:
            feature_part = feature_part.rstrip() + "\n\n" + add
            missing = _check_items(feature_part)
            if missing:
                print(f"[vision.py] 补全后仍缺：{missing}（写进输出文件，交给 AI 手工补）")
        else:
            print(f"[vision.py] 补全失败，缺项仍存在：{missing}")

    if out_path:
        _write_output(out_path, feature_part, lock_part, missing, model, paths)
    else:
        print(feature_part)
        if lock_part:
            print("\n" + lock_part)
    if not lock_part:
        print("[vision.py] 警告：模型输出里没找到「基线锁脸段成品：」，首轮身份基线需 AI 手工拼。")
    for t in temps:
        if t and os.path.exists(t):
            os.unlink(t)


def cmd_fix(v, paths, diff, notes, base_text, out_path):
    prepared, temps = _maybe_compress(paths + [diff])
    prompt = FIX_PROMPT
    if base_text:
        prompt += f"\n\n参考人原特征描述（以参考照片为准校正）：\n{base_text}"
    if notes:
        prompt += f"\n\n使用者说不像的点：{notes}"
    content = [{"type": "text", "text": prompt}]
    for p in prepared:
        content.append({"type": "image_url", "image_url": {"url": b64(p)}})
    text, model, raw = _call_vision(v, content, "分析单一修正重点")
    if not text:
        sys.exit("[vision.py] 修正失败，模型未返回有效内容。原始响应：\n" + raw[:300])
    for t in temps:
        if t and os.path.exists(t):
            os.unlink(t)

    if out_path:
        lines = ["# 单重点锁脸修正建议", "", "状态：advisory（交给 face_guard 状态机决定是否采用）", "", text.strip(), ""]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[vision.py] 已写入: {os.path.abspath(out_path)}")
    else:
        print(text)


VALUED_FLAGS = {"--config", "--out", "--angle", "--diff", "--notes", "--base"}
BOOL_FLAGS = {"--fix", "--replace-provisional"}


def extract_args(argv):
    flags, paths = {}, []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in VALUED_FLAGS:
            if i + 1 < len(argv):
                flags[a] = argv[i + 1]
                i += 2
            else:
                i += 1
            continue
        if a in BOOL_FLAGS:
            flags[a] = True
            i += 1
            continue
        if a.startswith("--"):
            i += 1
            continue
        paths.append(a)
        i += 1
    return flags, paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    flags, paths = extract_args(sys.argv)
    config, conf_path = load_config(flags.get("--config"))
    v = config.get("vision", {})
    if v.get("agent_can_see"):
        print("[vision.py] 提示：config 里 agent_can_see=true（你的 agent 自己能看图），"
              "这一步也可以直接让 agent 看参考图来完成。以下仍是外部 vision 的结果：")
    if not v.get("host") or not v.get("key") or not v.get("model"):
        sys.exit("[vision.py] config.json 的 vision 配置不完整（需要 host/key/model）。"
                  "请按使用教程检查 config.json。当前配置：\n" + json.dumps(v, ensure_ascii=False, indent=2))
    if not paths:
        sys.exit("[vision.py] 需要至少一张参考图。用法见文件头 docstring。")

    if flags.get("--fix"):
        diff = flags.get("--diff")
        if not diff:
            sys.exit("[vision.py] 修正模式需要 --diff <不像的生成图>。")
        base_text = None
        if flags.get("--base"):
            try:
                base_text = open(flags["--base"], encoding="utf-8").read()[:3000]
            except OSError as e:
                print(f"[vision.py] 警告：读 --base 失败（{e}），忽略原特征。")
        cmd_fix(v, paths, diff, flags.get("--notes"), base_text, flags.get("--out"))
    else:
        cmd_model(
            v,
            paths,
            flags.get("--out"),
            flags.get("--angle") or "正脸",
            replace_provisional=bool(flags.get("--replace-provisional")),
        )
