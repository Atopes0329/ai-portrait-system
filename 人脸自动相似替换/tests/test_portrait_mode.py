import argparse
from contextlib import redirect_stderr
from io import StringIO
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from identity_lock.job_contract import (
    add_identity_reference_arguments,
    generation_attachments,
    job_mode,
    normalize_auxiliary_references,
    style_source_kind,
)


class PortraitModeTests(unittest.TestCase):
    def _parser(self):
        parser = argparse.ArgumentParser()
        add_identity_reference_arguments(parser)
        parser.add_argument("--prompt")
        return parser

    def test_prepare_parser_accepts_portrait_without_style_image(self):
        args = self._parser().parse_args(
            [
                "--source",
                "本人.jpg",
                "--reference",
                "侧脸.jpg",
                "--prompt",
                "首轮提示词",
            ]
        )
        self.assertIsNone(args.style)
        self.assertEqual(args.reference, ["侧脸.jpg"])
        self.assertEqual(job_mode(args.style), "portrait")

    def test_replacement_mode_remains_available(self):
        args = self._parser().parse_args(
            [
                "--source",
                "本人.jpg",
                "--style",
                "风格人物.jpg",
                "--prompt",
                "换脸提示词",
            ]
        )
        self.assertEqual(args.style, "风格人物.jpg")
        self.assertEqual(job_mode(args.style), "replacement")

    def test_prompt_library_style_stays_on_portrait_identity_path(self):
        args = self._parser().parse_args(
            [
                "--source",
                "本人.jpg",
                "--style-preset",
                "现代杂志.md",
                "--prompt",
                "锁脸提示词",
            ]
        )
        self.assertIsNone(args.style)
        self.assertEqual(job_mode(args.style), "portrait")
        self.assertEqual(
            style_source_kind(args.style, args.style_preset), "prompt_library"
        )

    def test_style_image_and_prompt_preset_are_mutually_exclusive(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                self._parser().parse_args(
                    [
                        "--source",
                        "本人.jpg",
                        "--style",
                        "风格图.jpg",
                        "--style-preset",
                        "现代杂志.md",
                    ]
                )

    def test_none_style_source_is_explicit(self):
        self.assertEqual(style_source_kind(None, None), "none")

    def test_auxiliary_references_are_deduplicated_and_exclude_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            side = root / "side.jpg"
            source.write_bytes(b"source")
            side.write_bytes(b"side")
            result = normalize_auxiliary_references(
                source, [side, side, source], require_exists=True
            )
            self.assertEqual(result, [str(side.resolve())])

    def test_missing_auxiliary_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jpg"
            source.write_bytes(b"source")
            with self.assertRaisesRegex(ValueError, "不存在"):
                normalize_auxiliary_references(
                    source, [Path(directory) / "missing.jpg"], require_exists=True
                )

    def test_auxiliary_reference_limit_is_enforced(self):
        refs = [f"/tmp/ref-{index}.jpg" for index in range(8)]
        with self.assertRaisesRegex(ValueError, "最多允许 7 张"):
            normalize_auxiliary_references(None, refs, require_exists=False)

    def test_portrait_agent_receives_all_real_references(self):
        job = {
            "style_source": {"kind": "prompt_library"},
            "source_image": "/session/source.jpg",
            "masked_style_image": None,
            "generation_reference_images": [
                "/session/source.jpg",
                "/session/side.jpg",
                "/session/profile.jpg",
            ],
        }
        self.assertEqual(
            generation_attachments(job), job["generation_reference_images"]
        )

    def test_replacement_agent_keeps_two_image_privacy_boundary(self):
        job = {
            "style_source": {"kind": "image_reference"},
            "source_image": "/session/source.jpg",
            "masked_style_image": "/session/masked.png",
            "generation_reference_images": [
                "/session/source.jpg",
                "/session/unused-side.jpg",
            ],
        }
        self.assertEqual(
            generation_attachments(job),
            ["/session/source.jpg", "/session/masked.png"],
        )


if __name__ == "__main__":
    unittest.main()
