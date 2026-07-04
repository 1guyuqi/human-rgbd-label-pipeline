# Human RGB-D Label Pipeline

Extract **General Flow**-style labels from **human** RGB-D manipulation videos.  
This repo documents **how human data is annotated** (clips, masks, point trajectories).  
It does **not** include robot hardware capture code.

Two pipelines are provided:

| Pipeline | Input | Annotation source |
|----------|--------|-------------------|
| **HOI4D** (`hoi4d/`) | [HOI4D](https://github.com/leolyliu/HOI4D-Instructions) RGB-D + official human annotations | Action marks + object 6D pose in `HOI4D_annotations` |
| **RVideo** (`rvideo/`) | Your own human RGB-D trajectories (`rgb_*.png`, `depth_*.png`, `camera_in.npy`) | Semi-auto: GroundingDINO + SAM2 + CoTracker, with **manual mask correction** |

Based on the [General Flow](https://general-flow.github.io/) labeling methodology.

## What gets labeled

For each action clip the pipeline produces:

- **Clip boundaries** (start/end frames from human action marks or motion heuristics)
- **Hand / tool / object masks** (2D, per frame or keyframe)
- **KPST trajectories** — 3D point tracks on the manipulated object (`kpst_traj.npy`)
- **Colored point cloud** with part labels (`pcd.npy`: xyz + rgb + part id)
- **Metadata JSON** for training splits

## Install

```bash
conda create -n human-label python=3.10
conda activate human-label
pip install -r requirements.txt
```

Install external tools separately (not bundled):

| Tool | Purpose |
|------|---------|
| [CoTracker](https://github.com/facebookresearch/co-tracker) | Dense 2D/3D point tracking |
| [SAM2](https://github.com/facebookresearch/sam2) | Interactive / box-prompt segmentation |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Text/bbox HOI detection |
| [100DoH hand detector](https://github.com/ddshan/hand_object_detector) | Optional legacy hand-object detector |

Copy and edit checkpoint paths:

```bash
cp config/paths.example.yaml config/paths.yaml
# set cotracker_ckpt, sam2_checkpoint, groundingdino_ckpt, etc.
```

Or export environment variables: `TOOL_COTRACKER_CKPT`, `TOOL_SAM2_CHECKPOINT`, ...

## Pipeline A — HOI4D human annotations

Uses **existing human labels** from HOI4D (action intervals, object poses).

```bash
cd hoi4d

# Step 1–2: build clip list + object pose trajectories
python label_gen.py --data_root /path/to/HOI4D_release \
  --anno_root /path/to/HOI4D_annotations \
  --idx_file /path/to/release.txt \
  --output_root ../output/hoi4d
# Uncomment main_step1_clips_gen / main_step2_trajs_gen in label_gen.py as needed

# Step 3–4: kpst from human 6D pose (default entry point)
python label_gen.py --data_root ... --anno_root ... --idx_file ... --output_root ../output/hoi4d
# Optional: enable step3 confident masks (better object pcd, slower)
# python label_gen.py ... --enable_step3_masks

# Step 5–6: merge shards + train/val/test split
python label_gen_merge.py --output_root ../output/hoi4d
```

### HOI4D labeling steps (inside `label_gen.py`)

| Step | Function | Human annotation used |
|------|----------|------------------------|
| 1 | `main_step1_clips_gen` | Action mark JSON → clip `[st, ed]` |
| 2 | `main_step2_trajs_gen` | Object 6D pose annotations → per-frame transforms |
| 3 | `proc_step3_masks_gen` (optional, `--enable_step3_masks`) | HOI4D 2Dseg + camera extrinsics → temporally fused `confident_mask_*.npy` |
| 4 | `proc_step4_kpsts_gen` | Sample object points from semantic pcd; propagate with **annotated 6D pose** (not CoTracker) |
| 5–6 | `label_gen_merge.py` | Merge JSON caches, split by object instance |

**Note:** HOI4D kpst trajectories come from official object-pose annotations (`camera_coord_transformation`), not from CoTracker. Step 3 is off by default; step 4 still builds `pcd.npy` using HOI4D 2Dseg with a simpler erode fallback when confident masks are absent.

## Pipeline B — Custom human RGB-D videos

For egocentric / third-person RGB-D you recorded (no HOI4D annotation files).

Expected trajectory layout:

```
my_task/traj_000/
├── rgb_0.png, rgb_1.png, ...
├── depth_0.png, depth_1.png, ...
├── camera_in.npy
└── meta.json          # optional task description
```

```bash
cd rvideo

# Full 2D + 3D labeling
python label_gen.py \
  --raw_data_root /path/to/human_rgbd_root \
  --save_root ../output/rvideo/my_task

# Manual mask fix + re-track one clip (after automatic pass)
python redo_clip_manual_2d.py \
  --save_root ../output/rvideo/my_task \
  --clip_id 23 \
  --mode both

# Merge into HOI4D-format dataset
python merge_egosoft2hoi4d.py \
  --hoi4d_dir ../output/hoi4d \
  --egosoft_dir ../output/rvideo/my_task
```

### RVideo labeling steps (inside `label_gen.py`)

| Step | Stage | Method |
|------|-------|--------|
| 1 | Clip detection | Motion / task heuristics on RGB-D sequence |
| 2 | HOI detection | GroundingDINO text prompts (`tool`, `other`) |
| 3 | Segmentation | SAM2 box prompts; **manual points** via `--manual_seg_fallback` |
| 4 | Point sampling | Fixed N query points on tool/other masks |
| 5 | Tracking | CoTracker → `kpst_traj.npy` |
| 6 | Point cloud | Back-project depth, voxel downsample → `pcd.npy` |

**Human-in-the-loop:** use `redo_clip_manual_2d.py` to click SAM2 points on hand/object, then re-run CoTracker for ground-truth refresh.

## Output layout (per clip)

```
output/rvideo/my_task/
├── step1_kpst_2d_info.json      # clip index
├── step1_2d/{clip_id}/          # masks, tracks, vis
└── data/{clip_id}/
    ├── kpst_traj.npy            # (N, T, 3)
    ├── kpst_part_id.npy
    ├── pcd.npy                  # (M, 7) xyz + rgb + label
    └── metadata fields in JSON
```

## Repository layout

```
├── hoi4d/
│   ├── label_gen.py             # HOI4D annotation → kpst
│   ├── label_gen_merge.py       # merge + split
│   ├── hoi4d_tool.py
│   └── pcd_hoi4d/
├── rvideo/
│   ├── label_gen.py             # custom human RGB-D labeling
│   ├── redo_clip_manual_2d.py   # manual mask + re-track
│   ├── merge_egosoft2hoi4d.py
│   ├── dino_hoi_detector.py
│   └── utils.py
├── config/paths.example.yaml
├── scripts/
│   ├── batch_recompute_3d.sh    # batch 3D re-export (set GT_ROOT + RECORD_ROOT)
│   ├── create_minimal_fixture.py
│   └── run_minimal_smoke_test.py
└── repo_paths.py
```

## Smoke test (minimal example)

No GPU or external model weights required. **Sample images are not bundled in the repo**; the script generates a tiny synthetic clip locally:

```bash
python scripts/create_minimal_fixture.py
python scripts/run_minimal_smoke_test.py
```

This verifies the 3D export path and checks:

- `examples/minimal_rvideo/output/data/0/pcd.npy`
- `examples/minimal_rvideo/output/data/0/kpst_traj.npy`
- `examples/minimal_rvideo/output/metadata_egosoft_demo.json` (per-clip metadata; smoke test also writes `metadata.json`)

## Re-align existing RVideo 3D exports

If you generated `pcd.npy` before the kpst/pcd SE(3) fix, re-run 3D only (2D caches unchanged):

```bash
export GT_ROOT=/path/to/process_data          # parent of task folders with step1_kpst_2d_info.json
export RECORD_ROOT=/path/to/Record
export RECORDED_RGBD_ROOT=/path/to/recorded_rgbd   # optional alias / fallback

bash scripts/batch_recompute_3d.sh
# or single task:
cd rvideo && python label_gen.py \
  --save_root /path/to/process_data/TASK \
  --only_3d --recompute_3d \
  --record_root "$RECORD_ROOT" \
  --recorded_rgbd_root "$RECORDED_RGBD_ROOT"
```

## Related repos

- [General Flow (paper code)](https://github.com/MichaelYuancb/general_flow) — full training/eval stack
- [franka-rgbd-record](https://github.com/1guyuqi/franka-rgbd-record) — **robot-side** RGB-D capture (separate from this repo)

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [HOI4D](https://github.com/leolyliu/HOI4D-Instructions)
- [General Flow](https://general-flow.github.io/)
- CoTracker, SAM2, GroundingDINO, 100DoH
