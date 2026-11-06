import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


PROJECT_ROOT = Path(__file__).resolve().parent
JOB_DIR = PROJECT_ROOT / ".training"


@contextmanager
def training_lock(job_dir=JOB_DIR):
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / "training.lock").open("a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Training is already running.") from error
        try:
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def get_training_status(job_dir=JOB_DIR):
    try:
        with training_lock(job_dir):
            pass
    except RuntimeError:
        return "running"
    status_path = job_dir / "status.json"
    if not status_path.is_file():
        return "idle"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))["status"]
    except (OSError, ValueError, KeyError, TypeError):
        return "interrupted"
    return "interrupted" if status == "running" else status


def read_training_log(job_dir=JOB_DIR):
    try:
        with (job_dir / "training.log").open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, log_file.tell() - 12000))
            return log_file.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _write_status(job_dir, status):
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=job_dir, delete=False) as status_file:
            temporary_path = Path(status_file.name)
            json.dump({"status": status}, status_file)
        os.replace(temporary_path, job_dir / "status.json")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_training(job_dir=JOB_DIR):
    with training_lock(job_dir):
        _write_status(job_dir, "running")
        with (job_dir / "training.log").open("w", encoding="utf-8") as log_file:
            try:
                result = subprocess.run(
                    [sys.executable, "-u", str(PROJECT_ROOT / "train.py"), "--download-data"],
                    cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                status = "succeeded" if result.returncode == 0 else "failed"
            except OSError as error:
                log_file.write(str(error))
                status = "failed"
            _write_status(job_dir, status)


def start_training():
    if get_training_status() == "running":
        raise RuntimeError("Training is already running.")
    if shutil.disk_usage(PROJECT_ROOT).free < 5 * 1024**3:
        raise RuntimeError("At least 5 GB of free disk space is required for training.")
    process = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve())],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return process.pid


if __name__ == "__main__":
    try:
        run_training()
    except RuntimeError:
        sys.exit(1)