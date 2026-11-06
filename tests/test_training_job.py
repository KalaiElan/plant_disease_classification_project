import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_job import get_training_status, read_training_log, run_training, training_lock


class TrainingJobTests(unittest.TestCase):
    def test_lock_prevents_duplicate_jobs_and_releases(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_dir = Path(temporary_directory)
            self.assertEqual(get_training_status(job_dir), "idle")
            with training_lock(job_dir):
                self.assertEqual(get_training_status(job_dir), "running")
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with training_lock(job_dir):
                        self.fail("Duplicate job acquired the lock")
            self.assertEqual(get_training_status(job_dir), "idle")

    def test_failed_training_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_dir = Path(temporary_directory)
            with patch("training_job.subprocess.run", return_value=subprocess.CompletedProcess([], 1)):
                run_training(job_dir)
            self.assertEqual(get_training_status(job_dir), "failed")

    def test_successful_training_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_dir = Path(temporary_directory)
            with patch("training_job.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
                run_training(job_dir)
            self.assertEqual(get_training_status(job_dir), "succeeded")

    def test_abandoned_job_is_reported_as_interrupted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_dir = Path(temporary_directory)
            (job_dir / "status.json").write_text('{"status": "running"}', encoding="utf-8")
            self.assertEqual(get_training_status(job_dir), "interrupted")

    def test_log_read_is_bounded(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_dir = Path(temporary_directory)
            (job_dir / "training.log").write_text("x" * 20000 + "last line", encoding="utf-8")
            log = read_training_log(job_dir)
            self.assertEqual(len(log), 12000)
            self.assertTrue(log.endswith("last line"))