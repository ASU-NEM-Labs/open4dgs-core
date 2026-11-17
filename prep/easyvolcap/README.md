# EasyVolCap / 4K4D interoperability

These scripts build COLMAP camera datasets and export depth or density grids
for use with initialization. They require a separate EasyVolCap environment;
follow the [upstream setup](https://github.com/zju3dv/EasyVolcap).

Use each script's `--help` for required input paths. Camera conversion tools
load COLMAP helpers from the companion Open4DGS-Trigs checkout; see
[combined setup](../../SETUP.md). Keep downloaded weights, capture data,
depth grids and experiment outputs outside the source checkout.
