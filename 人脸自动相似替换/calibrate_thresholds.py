#!/usr/bin/env python3
"""用真人“能认出/不能认出”标注校准各评分器的分界线。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_METRICS = ("sface", "arcface", "adaface", "ai_identity")
POSITIVE_LABELS = {"1", "true", "yes", "recognizable", "same", "能认出", "本人"}
NEGATIVE_LABELS = {"0", "false", "no", "unrecognizable", "different", "不能认出", "非本人"}


def parse_label(value: Any) -> int:
    normalized = str(value).strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"无法识别真人标注：{value}")


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("校准文件没有数据")
    return [dict(row) for row in rows]


def _metric_samples(
    rows: Iterable[Mapping[str, Any]], metric: str, label_column: str
) -> list[tuple[float, int]]:
    samples = []
    for row in rows:
        if row.get(metric) in (None, ""):
            continue
        value = float(row[metric])
        if not isfinite(value):
            continue
        samples.append((value, parse_label(row.get(label_column))))
    return samples


def calibrate_metric(samples: list[tuple[float, int]], min_per_class: int = 10) -> dict[str, Any]:
    positives = sum(label == 1 for _score, label in samples)
    negatives = sum(label == 0 for _score, label in samples)
    if positives < min_per_class or negatives < min_per_class:
        return {
            "status": "uncalibrated",
            "sample_count": len(samples),
            "positive_count": positives,
            "negative_count": negatives,
            "reason": f"至少需要 {min_per_class} 个能认出样本和 {min_per_class} 个不能认出样本",
        }

    values = sorted({score for score, _label in samples})
    candidates = [values[0] - 1e-9]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-9)
    best = None
    for threshold in candidates:
        true_positive = sum(score >= threshold and label == 1 for score, label in samples)
        false_negative = positives - true_positive
        true_negative = sum(score < threshold and label == 0 for score, label in samples)
        false_positive = negatives - true_negative
        recall = true_positive / positives
        specificity = true_negative / negatives
        balanced_accuracy = (recall + specificity) / 2.0
        false_accept_rate = false_positive / negatives
        false_reject_rate = false_negative / positives
        candidate = {
            "threshold": threshold,
            "balanced_accuracy": balanced_accuracy,
            "false_accept_rate": false_accept_rate,
            "false_reject_rate": false_reject_rate,
        }
        # 身份系统优先降低错放异人的概率；并列时选择更严格的阈值。
        rank = (balanced_accuracy, -false_accept_rate, threshold)
        if best is None or rank > best[0]:
            best = (rank, candidate)
    result = best[1]
    return {
        "status": "calibrated",
        "sample_count": len(samples),
        "positive_count": positives,
        "negative_count": negatives,
        **{key: round(value, 6) for key, value in result.items()},
    }


def calibrate_rows(
    rows: list[Mapping[str, Any]],
    metrics: Iterable[str] = DEFAULT_METRICS,
    label_column: str = "human_recognizable",
    min_per_class: int = 10,
    model_version: str = "unspecified",
) -> dict[str, Any]:
    reports = {
        metric: calibrate_metric(
            _metric_samples(rows, metric, label_column), min_per_class=min_per_class
        )
        for metric in metrics
    }
    calibrated = [name for name, report in reports.items() if report["status"] == "calibrated"]
    return {
        "schema_version": "1.0",
        "status": "calibrated" if calibrated else "uncalibrated",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": model_version,
        "sample_count": len(rows),
        "label_definition": "human_recognizable: 人能否认出候选图是参考图中的同一个人",
        "metrics": reports,
        "hard_gate_eligible": bool(len(rows) >= 20 and "sface" in calibrated),
        "warning": "阈值只适用于记录的模型版本和当前样本分布；不是通用的人类识别概率。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="校准人脸身份评分与真人识别判断的分界线")
    parser.add_argument("--input", required=True, help="CSV 或 JSONL 标注文件")
    parser.add_argument("--output", required=True, help="输出阈值 JSON")
    parser.add_argument("--label-column", default="human_recognizable")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--model-version", default="unspecified")
    args = parser.parse_args()
    rows = load_rows(args.input)
    report = calibrate_rows(
        rows,
        metrics=args.metrics,
        label_column=args.label_column,
        min_per_class=args.min_per_class,
        model_version=args.model_version,
    )
    output = Path(args.output).expanduser().resolve()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
