import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ai人像摄影技能包"
    / "scripts"
    / "generate.py"
)
SPEC = importlib.util.spec_from_file_location("portrait_generate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GeneratePromptTests(unittest.TestCase):
    def test_prompt_file_is_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "next_prompt.txt"
            path.write_text("干净的新一轮提示词", encoding="utf-8")
            self.assertEqual(
                MODULE.resolve_prompt("旧提示词", path), "干净的新一轮提示词"
            )

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.resolve_prompt("", None)


if __name__ == "__main__":
    unittest.main()
