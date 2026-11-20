import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from landmark_providers import get_facemesh_provider


class DenseProviderFallbackTests(unittest.TestCase):
    def test_missing_optional_model_degrades_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            provider, status = get_facemesh_provider(Path(directory))
        self.assertIsNone(provider)
        self.assertEqual(status["state"], "missing")
        self.assertEqual(status["evidence_mode"], "advisory")


if __name__ == "__main__":
    unittest.main()
