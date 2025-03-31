import shutil
import tempfile
import pytest
import os
import glob
from pathlib import Path

@pytest.fixture(scope="function", autouse=True)
def cleanup_pymp_directories():
    """Automatically removes 'pymp-*' directories after each test."""
    yield  # Run the test first
    tmpdir = Path(tempfile.gettempdir())
    for path in tmpdir.glob("pymp-*"):
        shutil.rmtree(path, ignore_errors=True)

@pytest.fixture(scope="function", autouse=True)
def cleanup_faces_directory():
    yield  # Run the test first
    faces_dir = Path("outputs/preparation/faces")
    for path in faces_dir.glob("**/*"):
        if path.is_file() and path.suffix != ".py":
            path.unlink()

@pytest.fixture(scope="function", autouse=True)
def cleanup_videos_directory():
    yield  # Run the test first
    faces_dir = Path("videos/")
    for path in faces_dir.glob("**/*"):
        if path.is_file() and path.suffix == ".txt":
            path.unlink()