"""OpenAI Chat Completions 兼容的视觉身份评分器。

评分器与生图模型解耦。默认只输出建议；只有配置明确声明已经完成真人样本校准，
结果才允许参与硬门槛，而且仍不能绕过本地身份模型与退化保护。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request


JUDGE_PROMPT = """你是人脸身份质检员，不是审美评委。图片顺序固定：
1. 身份参考图：唯一身份来源；
2. 上一轮最佳图（可能没有）；
3. 当前候选图。

本轮唯一修正重点：{focus}

只比较身份，不评价服装、背景、画质、漂亮程度和艺术风格。判断：
- 当前候选整体是否仍能被认作参考图中的同一个人；
- 当前唯一重点是否比上一轮更接近参考图；
- 是否破坏了脸型、眼鼻嘴空间关系或其他已经正确的身份特征。

只输出一个 JSON 对象，不要 Markdown：
{{
  "identity_score": 0到100的整数,
  "focus_score": 0到100的整数,
  "focus_improved": true或false,
  "focus_pass": true或false,
  "regressions": ["最多3条具体退化，没有则空数组"],
  "confidence": 0到1的小数,
  "evidence": ["最多3条简短可核对证据"]
}}
如果姿态、遮挡、表情或脸部尺寸使比较不可靠，降低 confidence，不要猜。"""


def _image_data_url(path: str | Path) -> str:
    data = Path(path).read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError(f"视觉评分不支持或无法识别图片格式：{path}")
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _chat_content(raw: str) -> str:
    parts: list[str] = []
    for line in re.split(r"\r?\n", raw):
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        for choice in payload.get("choices") or []:
            message = choice.get("message") or {}
            delta = choice.get("delta") or {}
            content = message.get("content") or delta.get("content")
            if isinstance(content, str):
                parts.append(content)
    if parts:
        return "".join(parts)
    # 某些兼容接口直接返回评分 JSON，不再包 choices。
    return raw.strip()


def _extract_json(text: str) -> Mapping[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("视觉评分必须返回 JSON 对象")
    return value


def _score(value: Any, name: str) -> int:
    number = int(round(float(value)))
    if not 0 <= number <= 100:
        raise ValueError(f"{name} 必须在 0 到 100 之间")
    return number


def _short_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} 必须是数组")
    return [str(item).strip() for item in value[:3] if str(item).strip()]


def parse_judge_response(raw: str, hard_gate_enabled: bool = False) -> dict[str, Any]:
    data = _extract_json(_chat_content(raw))
    confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
    return {
        "kind": "vision_identity_judgement",
        "identity_score": _score(data.get("identity_score"), "identity_score"),
        "focus_score": _score(data.get("focus_score"), "focus_score"),
        "focus_improved": bool(data.get("focus_improved", False)),
        "focus_pass": bool(data.get("focus_pass", False)),
        "regressions": _short_list(data.get("regressions"), "regressions"),
        "confidence": round(confidence, 6),
        "evidence": _short_list(data.get("evidence"), "evidence"),
        "hard_gate_enabled": bool(hard_gate_enabled),
        "gate_mode": "calibrated" if hard_gate_enabled else "advisory_shadow",
    }


def _http_transport(url: str, payload: Mapping[str, Any], key: str, timeout: int) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"视觉评分接口返回 HTTP {error.code}: {body[:300]}") from error


def load_vision_judge_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    judge = dict(config.get("vision_judge") or {})
    if not judge.get("enabled", False):
        raise ValueError("vision_judge 未启用")
    for key in ("host", "key", "model"):
        if not judge.get(key):
            raise ValueError(f"vision_judge 缺少 {key}")
    calibration = dict(judge.get("calibration") or {})
    calibration_file = judge.get("calibration_file")
    if calibration_file:
        calibration_path = Path(calibration_file).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = config_path.parent / calibration_path
        calibration = json.loads(calibration_path.resolve().read_text(encoding="utf-8"))
        ai_metric = (calibration.get("metrics") or {}).get("ai_identity") or {}
        if ai_metric.get("status") == "calibrated":
            judge["identity_threshold"] = float(ai_metric["threshold"])
        judge["calibration_file"] = str(calibration_path.resolve())
    judge["hard_gate_enabled"] = bool(
        judge.get("hard_gate_enabled", False)
        and calibration.get("status") == "calibrated"
        and int(calibration.get("sample_count", 0)) >= 20
        and calibration.get("hard_gate_eligible", True)
        and judge.get("identity_threshold") is not None
    )
    return judge


class OpenAICompatibleVisionJudge:
    def __init__(
        self,
        config: Mapping[str, Any],
        transport: Callable[[str, Mapping[str, Any], str, int], str] | None = None,
    ) -> None:
        self.config = dict(config)
        self.transport = transport or _http_transport

    def judge(
        self,
        reference: str | Path,
        candidate: str | Path,
        focus: str,
        previous: str | Path | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": JUDGE_PROMPT.format(focus=focus or "整体身份")},
            {"type": "text", "text": "图片1：身份参考图"},
            {"type": "image_url", "image_url": {"url": _image_data_url(reference)}},
        ]
        if previous:
            content.extend([
                {"type": "text", "text": "图片2：上一轮最佳图"},
                {"type": "image_url", "image_url": {"url": _image_data_url(previous)}},
            ])
        content.extend([
            {"type": "text", "text": "最后一张：当前候选图"},
            {"type": "image_url", "image_url": {"url": _image_data_url(candidate)}},
        ])
        payload = {
            "model": self.config["model"],
            "temperature": 0,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": content}],
        }
        raw = self.transport(
            self.config["host"].rstrip("/") + "/v1/chat/completions",
            payload,
            self.config["key"],
            int(self.config.get("timeout", 180)),
        )
        result = parse_judge_response(raw, bool(self.config.get("hard_gate_enabled", False)))
        result["model"] = self.config["model"]
        if self.config.get("identity_threshold") is not None:
            result["identity_threshold"] = float(self.config["identity_threshold"])
        return result
