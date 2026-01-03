import os
from pathlib import Path
import unittest

from backend.app.main import build_image_url, normalize_base_url


class TestMainHelpers(unittest.TestCase):
    def test_normalize_base_url_strips_and_appends_v1(self) -> None:
        cases = {
            "http://localhost:1234": "http://localhost:1234/v1",
            "http://localhost:1234/": "http://localhost:1234/v1",
            "http://localhost:1234/v1": "http://localhost:1234/v1",
            "http://localhost:1234/v1/": "http://localhost:1234/v1",
        }
        for incoming, expected in cases.items():
            with self.subTest(incoming=incoming):
                self.assertEqual(normalize_base_url(incoming), expected)

    def test_build_image_url_uses_file_scheme(self) -> None:
        tmp_file = Path("data/test-image.png")
        tmp_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp_file.write_bytes(b"fakepng")
            result = build_image_url(tmp_file)
            self.assertTrue(result.startswith("file://"))
            self.assertIn(tmp_file.resolve().name, result)
        finally:
            if tmp_file.exists():
                os.remove(tmp_file)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
