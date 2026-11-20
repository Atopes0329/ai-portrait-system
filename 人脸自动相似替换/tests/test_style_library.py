import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from style_library import (
    MAX_STYLE_CONSTRAINTS,
    default_style_library,
    load_style_library,
    load_style_preset,
    merge_style_prompt,
    resolve_library_preset,
)


class StyleLibraryTests(unittest.TestCase):
    def test_bundled_library_is_fully_audited(self):
        presets = load_style_library(default_style_library(ROOT))
        self.assertGreaterEqual(len(presets), 80)
        forbidden = (
            "亚洲男性", "亚洲女性", "短发", "长发", "卷发", "黑发",
            "眼镜", "墨镜", "脸型", "五官", "中文标题", "文字排版",
        )
        for preset in presets:
            self.assertTrue(preset.compact_prompt)
            self.assertLessEqual(len(preset.constraints), MAX_STYLE_CONSTRAINTS)
            self.assertNotIn("人物还原度", preset.title)
            for word in forbidden:
                self.assertNotIn(word, preset.compact_prompt)

    def test_style_is_compacted_and_identity_words_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "001_现代杂志，年轻亚洲男性，黑色卷发.md"
            path.write_text(
                "提示词可以写成：现代时尚杂志摄影，年轻亚洲男性半身肖像，"
                "黑色卷发，黑框眼镜，灰色西装，身体微侧，浅灰摄影棚背景，"
                "暖色侧光，低饱和灰蓝色调，真实胶片颗粒，顶部巨大中文标题。\n\n"
                "负面提示词：五官变形、文字乱码。",
                encoding="utf-8",
            )
            preset = load_style_preset(path)
            self.assertLessEqual(len(preset.constraints), MAX_STYLE_CONSTRAINTS)
            prompt = preset.compact_prompt.casefold()
            for forbidden in ("亚洲男性", "卷发", "眼镜", "中文标题", "五官"):
                self.assertNotIn(forbidden.casefold(), prompt)
            self.assertIn("摄影棚", prompt)
            self.assertIn("侧光", prompt)

    def test_empty_markdown_falls_back_to_safe_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "002_极简黑白摄影棚半身肖像.md"
            path.write_text("", encoding="utf-8")
            preset = load_style_preset(path)
            self.assertTrue(preset.constraints)
            self.assertIn("极简黑白", preset.compact_prompt)

    def test_identity_instruction_document_is_not_exposed_as_style(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "人物还原度提示词1.md").write_text("锁脸说明", encoding="utf-8")
            (root / "003_暖色棚拍.md").write_text("暖色棚拍，半身构图", encoding="utf-8")
            presets = load_style_library(root)
            self.assertEqual(len(presets), 1)
            self.assertNotIn("人物还原度", presets[0].title)

    def test_compatibility_marks_non_photographic_style(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "083_Q版二次元插画.md"
            path.write_text("Q版二次元插画，漫画网点质感，低饱和粉彩", encoding="utf-8")
            preset = load_style_preset(path)
            self.assertEqual(preset.compatibility, "incompatible")

    def test_merged_prompt_uses_safe_title_and_stays_before_quality_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "004_年轻亚洲男性黑色短发暖色杂志.md"
            path.write_text(
                "提示词可以写成：暖色时尚杂志摄影，年轻亚洲男性，黑色短发，"
                "米色摄影棚背景，半身构图，柔和侧光。",
                encoding="utf-8",
            )
            preset = load_style_preset(path)
            merged = merge_style_prompt(
                "身份硬约束。replacement 模式中的图片2只提供动作、服饰、构图、镜头、滤镜、布光和氛围；"
                "portrait 模式没有图片2。\n\n身份硬约束：不得猜测图片2遮挡区域。\n\n"
                "质量硬约束：自然皮肤。",
                preset,
            )
            self.assertLess(merged.index("【系统摄影风格】"), merged.index("质量硬约束"))
            self.assertNotIn("年轻亚洲男性", merged)
            self.assertNotIn("黑色短发", merged)
            self.assertNotIn("图片2", merged)
            self.assertIn("no text", merged)

    def test_preset_cannot_escape_library_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "styles"
            root.mkdir()
            outside = Path(directory) / "outside.md"
            outside.write_text("摄影风格", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_library_preset(outside, root)


if __name__ == "__main__":
    unittest.main()
