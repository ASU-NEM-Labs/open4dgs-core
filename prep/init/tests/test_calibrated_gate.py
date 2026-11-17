import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_calibrated_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_calibrated_gate_opens_only_where_the_image_moves():
    flow = np.array([0.0, 0.9, 4.0, 40.0, 200.0])
    gate = MODULE.calibrated_gate_px(5.0, flow, motion_slack=1.0, motion_slack_max_px=12.0)
    np.testing.assert_allclose(gate, [5.0, 5.0, 9.0, 17.0, 17.0])


def test_calibrated_gate_is_fixed_without_flow_or_slack():
    assert MODULE.calibrated_gate_px(2.5, None, 1.0, 12.0) == 2.5
    assert MODULE.calibrated_gate_px(2.5, np.array([50.0]), 0.0, 12.0) == 2.5


def test_calibrated_motion_slack_defaults_on():
    parser_src = SCRIPT.read_text()
    assert "'--calibrated-motion-slack', action=argparse.BooleanOptionalAction, default=True" in parser_src
