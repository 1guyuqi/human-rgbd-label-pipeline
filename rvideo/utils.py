import numpy as np
import open3d as o3d


def apply_global_se3(xyz: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply the same global rotation and translation to Nx3 or NxTx3 points."""
    t = np.asarray(t, dtype=np.float32).reshape(3)
    if xyz.ndim == 2:
        return (xyz @ R.T) + t
    if xyz.ndim == 3:
        return (xyz @ R.T) + t.reshape(1, 1, 3)
    raise ValueError(f"apply_global_se3 expects 2D or 3D array, got shape {xyz.shape}")


def filter_kpst_long_trajectories(
    kpst_3d: np.ndarray,
    max_total_len: float = 0.60,
    max_step: float = 0.12,
    max_disp: float = 0.35,
    max_outlier_step_ratio: float = 0.35,
    outlier_step_thr: float = 0.08,
):
    """Drop trajectories that are too long or contain large frame-to-frame jumps."""
    if kpst_3d is None or kpst_3d.shape[0] == 0:
        return kpst_3d, np.zeros((0,), dtype=bool)

    steps = np.linalg.norm(kpst_3d[:, 1:] - kpst_3d[:, :-1], axis=-1)
    total_len = steps.sum(axis=1)
    max_step_each = steps.max(axis=1)
    disp = np.linalg.norm(kpst_3d[:, -1] - kpst_3d[:, 0], axis=-1)
    big_ratio = (steps > float(outlier_step_thr)).mean(axis=1)

    keep = (
        (total_len <= float(max_total_len))
        & (max_step_each <= float(max_step))
        & (disp <= float(max_disp))
        & (big_ratio <= float(max_outlier_step_ratio))
    )
    return kpst_3d[keep], keep


def filter_kpst_static_trajectories(
    kpst_3d: np.ndarray,
    min_disp: float = 0.008,
    min_total_len: float = 0.005,
):
    """Drop trajectories with too little 3D motion (meters)."""
    if kpst_3d is None or kpst_3d.shape[0] == 0:
        return kpst_3d, np.zeros((0,), dtype=bool)

    if kpst_3d.shape[1] < 2:
        return kpst_3d, np.zeros((kpst_3d.shape[0],), dtype=bool)

    steps = np.linalg.norm(kpst_3d[:, 1:] - kpst_3d[:, :-1], axis=-1)
    total_len = steps.sum(axis=1)
    disp = np.linalg.norm(kpst_3d[:, -1] - kpst_3d[:, 0], axis=-1)
    keep = (disp >= float(min_disp)) | (total_len >= float(min_total_len))
    return kpst_3d[keep], keep


def filter_bg_remove_islands(
    bg_xyz: np.ndarray,
    bg_rgb: np.ndarray,
    fg_xyz: np.ndarray,
    max_fg_dist: float = 0.6,
    dbscan_eps: float = 0.03,
    dbscan_min_points: int = 30,
    keep_largest_cluster: bool = True,
):
    """Remove background islands far from the foreground point cloud."""
    if bg_xyz is None or bg_xyz.shape[0] == 0:
        return bg_xyz, bg_rgb
    if fg_xyz is None or fg_xyz.shape[0] == 0:
        return bg_xyz, bg_rgb

    p_bg = o3d.geometry.PointCloud()
    p_bg.points = o3d.utility.Vector3dVector(bg_xyz.astype(np.float32))
    p_fg = o3d.geometry.PointCloud()
    p_fg.points = o3d.utility.Vector3dVector(fg_xyz.astype(np.float32))

    d = np.asarray(p_bg.compute_point_cloud_distance(p_fg), dtype=np.float32)
    keep = d <= float(max_fg_dist)

    bg_xyz2 = bg_xyz[keep]
    bg_rgb2 = bg_rgb[keep] if bg_rgb is not None and bg_rgb.shape[0] == bg_xyz.shape[0] else bg_rgb

    if bg_xyz2.shape[0] == 0:
        return bg_xyz2, bg_rgb2

    if keep_largest_cluster:
        p2 = o3d.geometry.PointCloud()
        p2.points = o3d.utility.Vector3dVector(bg_xyz2.astype(np.float32))
        labels = np.array(
            p2.cluster_dbscan(eps=float(dbscan_eps), min_points=int(dbscan_min_points), print_progress=False),
            dtype=np.int32,
        )
        valid = labels >= 0
        if valid.any():
            labels_v = labels[valid]
            counts = np.bincount(labels_v)
            best = int(np.argmax(counts))
            keep2 = labels == best
            bg_xyz2 = bg_xyz2[keep2]
            if bg_rgb2 is not None and bg_rgb2.shape[0] == keep2.shape[0]:
                bg_rgb2 = bg_rgb2[keep2]

    return bg_xyz2, bg_rgb2


def remove_fly_points_xyzrgb(
    xyz: np.ndarray,
    rgb: np.ndarray,
    z_min: float = 0.05,
    z_max: float = 2.0,
    use_statistical: bool = True,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
    use_radius: bool = True,
    radius: float = 0.03,
    min_neighbors: int = 6,
):
    """Remove depth fly points via hard Z limits and Open3D outlier filters."""
    if xyz is None or xyz.shape[0] == 0:
        return xyz, rgb

    xyz = xyz.astype(np.float32, copy=False)
    rgb = rgb.astype(np.float32, copy=False) if rgb is not None else None

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if rgb is not None:
        rgb = rgb[finite]

    z = xyz[:, 2]
    keep_z = (z > float(z_min)) & (z < float(z_max))
    xyz = xyz[keep_z]
    if rgb is not None:
        rgb = rgb[keep_z]

    if xyz.shape[0] == 0:
        return xyz, rgb

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None and rgb.shape[0] == xyz.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0))

    if use_statistical and len(pcd.points) >= max(nb_neighbors, 10):
        pcd, _ind = pcd.remove_statistical_outlier(
            nb_neighbors=int(nb_neighbors),
            std_ratio=float(std_ratio),
        )

    if use_radius and len(pcd.points) >= max(min_neighbors + 1, 10):
        pcd, _ind = pcd.remove_radius_outlier(
            nb_points=int(min_neighbors),
            radius=float(radius),
        )

    xyz2 = np.asarray(pcd.points).astype(np.float32)
    rgb2 = np.asarray(pcd.colors).astype(np.float32) if pcd.has_colors() else None

    if rgb is None:
        return xyz2, rgb
    return xyz2, rgb2


def filter_kpst_trajectory_flypoints(
    kpst_3d: np.ndarray,
    max_step: float = 0.15,
    max_total_jump: float = 1.0,
):
    """Drop trajectories with a single frame-to-frame jump caused by bad depth."""
    if kpst_3d is None or kpst_3d.shape[0] == 0:
        return kpst_3d, np.zeros((0,), dtype=bool)

    d = np.linalg.norm(kpst_3d[:, 1:] - kpst_3d[:, :-1], axis=-1)
    ok_step = (d <= float(max_step)).all(axis=1)
    ok_jump = np.max(d, axis=1) <= float(max_total_jump)
    keep = ok_step & ok_jump
    return kpst_3d[keep], keep
