import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from train import read_split, safe_extract_archive


class TrainingDataTests(unittest.TestCase):
    def test_read_split_resolves_official_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            image_root = data_dir / "raw" / "color"
            image_path = image_root / "Corn___healthy" / "sample.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(image_path)

            split_file = data_dir / "splits" / "color_train.txt"
            split_file.parent.mkdir()
            split_file.write_text("raw/color/Corn___healthy/sample.jpg\n", encoding="utf-8")

            samples = read_split(image_root, split_file)

        self.assertEqual(samples, [(image_path.resolve(), "Corn___healthy")])

    def test_read_split_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            image_root = data_dir / "raw" / "color"
            image_root.mkdir(parents=True)
            split_file = data_dir / "splits" / "color_train.txt"
            split_file.parent.mkdir()
            split_file.write_text("raw/color/../../outside.jpg\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid dataset path"):
                read_split(image_root, split_file)

    def test_archive_extraction_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "malicious.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", "outside")

            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                safe_extract_archive(archive_path, root / "data")

            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()