#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py — 参考图锁脸生图（双模式）

用法:
  python3 scripts/generate.py chat   "锁脸提示词" --ref 基准图.jpg [--ref 更多图.jpg] [--out 候选路径]
  python3 scripts/generate.py draw   "提示词" [--size auto] --ref 基准图.jpg [--out 候选路径]
  python3 scripts/generate.py export "提示词" --ref 基准图.jpg [--out 锁脸提示词.md]

三种模式:
  chat   走 /v1/chat/completions 带参考图（nano-banana 型模型，锁脸最强），
         从模型回复里提取图片 URL 并下载。参考图越多越稳。
  draw   走异步 draw 接口（gpt-image 型模型）：提交任务 → 轮询 → 下载。
  export 不联网，只把锁脸提示词 + 参考图路径写进一个 .md 文件，
         你自己粘到任意生图平台的参考图功能里用。任何平台都能兜底。

配置: 读本技能包根目录的 config.json（generate.host / key / model / draw_model / out_dir）。
     中途换模型只改 config.json，不用动脚本。
"""
import argparse, base64, json, os, re, subprocess, sys, tempfile, time, urllib.request, urllib.error
from pathlib import Path


def find_config():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "config.json"),
                 os.path.join(here, "config.json")):
        if os.path.exists(cand):
            return cand
    return None


def load_generate():
    path = find_config()
    if not path:
        sys.exit("[generate.py] 找不到 config.json。请先按使用教程完成首次配置（或把 config.example.json 复制为 config.json 并填入 API 信息）。")
    with open(path) as f:
        cfg = json.load(f)
    g = cfg.get("generate", {})
    if not g.get("host") or not g.get("key"):
        sys.exit("[generate.py] config.json 的 generate 配置缺少 host/key。请检查 config.json。")
    return g


def resolve_prompt(prompt=None, prompt_file=None):
    """允许闭环控制器每轮传入重新生成的干净提示词文件。"""
    if prompt_file:
        text = Path(prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    else:
        text = prompt or ""
    if not text.strip():
        raise ValueError("提示词不能为空；请提供文本或 --prompt-file")
    return text.strip()


def http_json(host, path, payload, key, timeout=300):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        host.rstrip("/") + path, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.read().decode("utf-8", "replace")


def parse_events(raw):
    objs = []
    for line in re.split(r"\r?\n", raw):
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line.startswith("{"):
            try:
                objs.append(json.loads(line))
            except Exception:
                pass
    return objs


def image_type(data):
    """按文件头识别图片，不能相信 URL、扩展名或响应头。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif", "image/gif"
    raise ValueError("下载结果不是可识别的 PNG/JPEG/WebP/GIF 图片")


def image_data_url(path):
    with open(path, "rb") as file:
        data = file.read()
    _extension, mime = image_type(data)
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def output_path(out_dir, prefix, extension):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(out_dir, f"{prefix}-{stamp}")
    candidate = base + "." + extension
    index = 2
    while os.path.exists(candidate):
        candidate = f"{base}-{index}.{extension}"
        index += 1
    return candidate


def download(url, out_dir, prefix, exact_path=None):
    requested = Path(exact_path).expanduser().resolve() if exact_path else None
    target_dir = requested.parent if requested else Path(out_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{prefix}-", suffix=".download", dir=str(target_dir))
    os.close(fd)
    ok = False
    for _ in range(3):
        r = subprocess.run(["curl", "-sL", "--max-time", "90", "-A", "Mozilla/5.0",
                            "--fail", "-o", temporary, url])
        if r.returncode == 0 and os.path.getsize(temporary) > 0:
            ok = True
            break
    if not ok:
        print(f"[generate.py] 下载失败: {url}")
        if os.path.exists(temporary):
            os.unlink(temporary)
        return None
    try:
        with open(temporary, "rb") as file:
            extension, _mime = image_type(file.read(32))
    except ValueError as error:
        os.unlink(temporary)
        print(f"[generate.py] 下载失败：{error}")
        return None
    if requested:
        if requested.suffix:
            normalized = "jpg" if requested.suffix.lower() in (".jpg", ".jpeg") else requested.suffix.lower()[1:]
            if normalized != extension:
                os.unlink(temporary)
                print(
                    f"[generate.py] 输出扩展名与真实图片格式不符：请求 {requested.suffix}，实际 {extension}"
                )
                return None
            out = str(requested)
        else:
            out = str(requested.with_suffix("." + extension))
        if os.path.exists(out):
            os.unlink(temporary)
            print(f"[generate.py] 拒绝覆盖已有候选图：{out}")
            return None
    else:
        out = output_path(str(target_dir), prefix, extension)
    os.replace(temporary, out)
    print(f"[generate.py] 完成: {out}")
    return out


def cmd_chat(g, prompt, refs, out_path=None):
    model = g.get("model") or "nano-banana-2"
    print(f"[generate.py] {model} 参考图生图中…")
    content = [{"type": "text", "text": prompt}]
    for p in refs or []:
        content.insert(0, {"type": "image_url",
                           "image_url": {"url": image_data_url(p)}})
    payload = {"model": model, "max_tokens": 8192,
               "messages": [{"role": "user", "content": content}]}
    raw = http_json(g["host"], "/v1/chat/completions", payload, g["key"], timeout=300)
    urls = []
    for o in parse_events(raw):
        if o.get("error"):
            print("[generate.py] 错误：" + json.dumps(o["error"], ensure_ascii=False))
            return 1
        for c in o.get("choices") or []:
            m = c.get("message") or {}
            if isinstance(m.get("content"), str):
                urls += re.findall(r"https?://[^\s)\"'`]+\.(?:png|jpg|jpeg|webp)[^\s)\"'`]*",
                                   m["content"])
    if not urls:
        print("[generate.py] 未提取到图片：" + raw[:400])
        return 1
    print(f"[generate.py] 原图：{urls[0]}")
    return 0 if download(urls[0], g.get("out_dir") or "出图", "pic", out_path) else 1


def cmd_draw(g, prompt, size, refs, out_path=None):
    model = g.get("draw_model") or "gpt-image-2"
    print(f"[generate.py] {model} 生图中：{prompt[:60]}")
    payload = {"model": model, "prompt": prompt, "n": 1}
    if size:
        payload["size"] = size
    if refs:
        imgs = []
        for p in refs:
            imgs.append(image_data_url(p))
        payload["image"] = imgs if len(imgs) > 1 else imgs[0]
        print(f"[generate.py] 参考图 {len(imgs)} 张：{', '.join(os.path.basename(p) for p in refs)}")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        g["host"].rstrip("/") + "/v1/draw/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + g["key"]})
    tid = None
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        print("[generate.py] 提交失败：" + e.read().decode("utf-8", "replace")[:500])
        return 1
    except Exception as e:
        print(f"[generate.py] 提交失败：{e}")
        return 1
    try:
        while True:
            line = resp.readline().decode("utf-8", "replace").strip()
            if not line:
                break
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line.startswith("{"):
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            tid = o.get("id") or (o.get("data") or {}).get("id")
            if tid:
                break
    finally:
        resp.close()
    if not tid:
        print("[generate.py] 提交失败：未能获取任务 ID")
        return 1
    print(f"[generate.py] 任务 {tid}，轮询中…")
    deadline = time.time() + 360
    while time.time() < deadline:
        time.sleep(3)
        rr = http_json(g["host"], "/v1/draw/result", {"id": tid}, g["key"])
        st = None
        for o in parse_events(rr):
            d = o.get("data")
            if isinstance(d, dict):
                st = d
                break
        if st is None:
            print("[generate.py] 查询失败：" + rr[:300])
            return 1
        status = st.get("status")
        if status == "succeeded":
            urls = [r.get("url") for r in (st.get("results") or []) if r.get("url")]
            if not urls:
                print("[generate.py] 成功但没拿到图片 URL：" + json.dumps(st, ensure_ascii=False))
                return 1
            print(f"[generate.py] 原图：{urls[0]}")
            return 0 if download(
                urls[0], g.get("out_dir") or "出图", "pic", out_path
            ) else 1
        if status in ("failed", "error"):
            print("[generate.py] 失败：" + json.dumps(st, ensure_ascii=False))
            return 1
        if st.get("progress") is not None:
            sys.stdout.write(f"\r[generate.py] 进度 {st['progress']}%")
            sys.stdout.flush()
    print("\n[generate.py] 超时")
    return 1


def cmd_export(g, prompt, refs, out_path):
    lines = []
    lines.append("# 锁脸生图提示词（由 ai人像摄影技能包 导出）")
    lines.append("")
    lines.append("## 参考图（第一张是锁脸基准，必须第一张传）")
    for i, p in enumerate(refs or []):
        tag = "（锁脸基准）" if i == 0 else f"（辅助角度{i}）"
        lines.append(f"- 第{i + 1}张{tag}: {os.path.abspath(p)}")
    lines.append("")
    lines.append("## 生图提示词（三段式，整体粘进你的生图平台的参考图功能）")
    lines.append("")
    for seg in prompt.split("\n"):
        lines.append(seg)
    lines.append("")
    lines.append("## 使用方法")
    lines.append("1. 在你生图平台的「参考图/参考人物」入口，第一张传锁脸基准图，其余传辅助图；")
    lines.append("2. 把上面「生图提示词」整体粘进提示词框；")
    lines.append("3. 中文文字建议用 no text / 留白，文字后期在排版层叠加（AI 生图写中文必乱码）。")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[generate.py] 已导出: {os.path.abspath(out_path)}")
    print("[generate.py] 把这个文件里的提示词 + 参考图，粘到你自己的生图平台即可。")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="参考图锁脸生图")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_chat = sub.add_parser("chat", help="chat-completions 带参考图（锁脸最强）")
    p_chat.add_argument("prompt", nargs="?")
    p_chat.add_argument("--prompt-file")
    p_chat.add_argument("--ref", action="append")
    p_chat.add_argument("--out", help="候选图目标路径；可不写扩展名，由真实格式自动补全")

    p_draw = sub.add_parser("draw", help="异步 draw 提交 + 轮询（gpt-image 型）")
    p_draw.add_argument("prompt", nargs="?")
    p_draw.add_argument("--prompt-file")
    p_draw.add_argument("--size", default=None)
    p_draw.add_argument("--ref", action="append")
    p_draw.add_argument("--out", help="候选图目标路径；可不写扩展名，由真实格式自动补全")

    p_exp = sub.add_parser("export", help="只导出提示词文件，手动粘平台")
    p_exp.add_argument("prompt", nargs="?")
    p_exp.add_argument("--prompt-file")
    p_exp.add_argument("--ref", action="append")
    p_exp.add_argument("--out", default="锁脸提示词.md")

    args = ap.parse_args()
    try:
        prompt = resolve_prompt(args.prompt, args.prompt_file)
    except ValueError as error:
        ap.error(str(error))
    if args.mode == "chat":
        g = load_generate()
        sys.exit(cmd_chat(g, prompt, args.ref, args.out))
    if args.mode == "draw":
        g = load_generate()
        sys.exit(cmd_draw(g, prompt, args.size, args.ref, args.out))
    if args.mode == "export":
        sys.exit(cmd_export({}, prompt, args.ref, args.out))
