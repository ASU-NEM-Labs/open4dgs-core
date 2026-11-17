# Camera calibration refinement

`run_calibration_v2.py` drives the refinement workflow implemented in
`calibration_v2.py`. Run `python run_calibration_v2.py --help` for inputs and
options, and use paths to your own capture and reference images.

Install the [preparation dependencies](../SETUP.md), validate the initial
COLMAP model, and inspect registration and reprojection quality before using
the refined cameras for point initialization. Keep calibration reports and
intermediate data outside the source checkout.
