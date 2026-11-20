import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_judges import OpenAICompatibleVisionJudge, parse_judge_response
from vision_judges import load_vision_judge_config


class VisionJudgeTests(unittest.TestCase):
    def test_parses_wrapped_chat_response(self):
        judgement = {
            "identity_score": 82,
            "focus_score": 88,
            "focus_improved": True,
            "focus_pass": True,
            "regressions": [],
            "confidence": 0.91,
            "evidence": ["眼距接近参考图"],
        }
        raw = json.dumps({"choices": [{"message": {"content": json.dumps(judgement)}}]})
        result = parse_judge_response(raw)
        self.assertEqual(result["identity_score"], 82)
        self.assertEqual(result["gate_mode"], "advisory_shadow")

    def test_invalid_score_is_rejected(self):
        raw = json.dumps({
            "identity_score": 120,
            "focus_score": 50,
            "confidence": 0.5,
        })
        with self.assertRaises(ValueError):
            parse_judge_response(raw)

    def test_provider_builds_three_image_comparison(self):
        png = b"\x89PNG\r\n\x1a\n" + b"test"
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("reference.png", "previous.png", "candidate.png"):
                path = Path(directory) / name
                path.write_bytes(png)
                paths.append(path)
            captured = {}

            def transport(url, payload, key, timeout):
                captured["payload"] = payload
                return json.dumps({
                    "identity_score": 80,
                    "focus_score": 81,
                    "focus_improved": True,
                    "focus_pass": True,
                    "regressions": [],
                    "confidence": 0.8,
                    "evidence": [],
                })

            provider = OpenAICompatibleVisionJudge(
                {"host": "https://example.invalid", "key": "secret", "model": "vision-x"},
                transport=transport,
            )
            result = provider.judge(paths[0], paths[2], "眼距", paths[1])
            content = captured["payload"]["messages"][0]["content"]
            self.assertEqual(sum(item["type"] == "image_url" for item in content), 3)
            self.assertEqual(result["model"], "vision-x")

    def test_hard_gate_requires_calibration_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "thresholds.json"
            calibration.write_text(json.dumps({
                "status": "calibrated",
                "sample_count": 24,
                "hard_gate_eligible": True,
                "metrics": {
                    "ai_identity": {"status": "calibrated", "threshold": 81.5}
                },
            }), encoding="utf-8")
            config = root / "config.json"
            config.write_text(json.dumps({
                "vision_judge": {
                    "enabled": True,
                    "host": "https://example.invalid",
                    "key": "secret",
                    "model": "vision-x",
                    "hard_gate_enabled": True,
                    "calibration_file": "thresholds.json",
                }
            }), encoding="utf-8")
            loaded = load_vision_judge_config(config)
            self.assertTrue(loaded["hard_gate_enabled"])
            self.assertEqual(loaded["identity_threshold"], 81.5)


if __name__ == "__main__":
    unittest.main()
