import os
import subprocess
import sys

import pytest

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)


@pytest.fixture(scope="session")
def lib():
    import foodcv
    subprocess.check_call([sys.executable, os.path.join(TOOLS, "build.py"), "lib"])
    name = "foodcv.dll" if os.name == "nt" else "libfoodcv.so"
    return foodcv.load_library(os.path.join(ROOT, name))
