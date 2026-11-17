"""Load the trainer's capture-time arithmetic from a checkout or installation."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trigs_source import trigs_file


_PATH = trigs_file("capture_timing.py")


def load():
    name = "trigs_capture_timing_shared"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
