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
