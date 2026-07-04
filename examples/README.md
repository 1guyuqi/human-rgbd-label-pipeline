# Examples

This folder does **not** ship real RGB-D videos or labeled clips (too large, dataset-specific).

## Smoke test (synthetic, generated locally)

```bash
python scripts/create_minimal_fixture.py   # writes examples/minimal_rvideo/traj_000/ (gitignored)
python scripts/run_minimal_smoke_test.py   # runs 3D export, writes output/ (gitignored)
```

The fixture is a tiny **programmatic** RGB-D clip (colored block + depth map) used only to verify
that `pcd.npy`, `kpst_traj.npy`, and metadata JSON are produced. It is not meant as a visual demo.

For real data layout, see the main [README](../README.md) (RVideo trajectory folder structure).
