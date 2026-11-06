import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppSetupTests(unittest.TestCase):
    def run_setup(self, status="idle"):
        original_is_file = Path.is_file
        with patch("pathlib.Path.is_file", autospec=True, side_effect=lambda path: (
            False if path.name == "plant_disease_resnet50.pth" else original_is_file(path)
        )), patch(
            "training_job.get_training_status", return_value=status
        ), patch("training_job.read_training_log", return_value=""):
            return AppTest.from_file(str(APP_PATH)).run(timeout=30)

    def test_missing_model_offers_recovery_without_predictions(self):
        app = self.run_setup()
        self.assertFalse(app.exception)
        self.assertEqual(app.warning[0].value, "No trained model is installed yet.")
        self.assertEqual(app.button[0].label, "Download data and train")
        self.assertFalse(app.button[0].disabled)
        self.assertTrue(app.button[1].disabled)
        self.assertFalse(app.metric)

    def test_running_training_disables_duplicate_start(self):
        app = self.run_setup("running")
        self.assertFalse(app.exception)
        self.assertTrue(app.button[0].disabled)
        self.assertIn("Training is running", app.info[0].value)

    def test_failed_training_offers_retry(self):
        app = self.run_setup("failed")
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertFalse(app.button[0].disabled)