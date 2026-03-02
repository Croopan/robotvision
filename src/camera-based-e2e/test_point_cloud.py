"""
Synthetic tests for multi-view point cloud fusion.
No GPU or Waymo dataset required — uses mock protos and fake images.
"""
import numpy as np
import open3d as o3d
import sys

# Mock proto classes so we can test without compiling protos
class MockTransform:
    def __init__(self, values):
        self.transform = values

class MockCalibration:
    def __init__(self, name, intrinsic, width, height, extrinsic_values):
        self.name = name
        self.intrinsic = intrinsic  # [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        self.width = width
        self.height = height
        self.extrinsic = MockTransform(extrinsic_values)

class MockContext:
    def __init__(self, calibrations):
        self.camera_calibrations = calibrations

class MockFrame:
    def __init__(self, calibrations):
        self.context = MockContext(calibrations)


def make_identity_extrinsic():
    """4x4 identity as a flat list of 16 doubles."""
    return list(np.eye(4).flatten())


def make_test_calibrations():
    """Two cameras: FRONT (1) and FRONT_LEFT (2) with known transforms."""
    # FRONT camera: identity extrinsic (camera = vehicle)
    front = MockCalibration(
        name=1,
        intrinsic=[100.0, 100.0, 16.0, 12.0, 0, 0, 0, 0, 0],
        width=32, height=24,
        extrinsic_values=make_identity_extrinsic()
    )
    # FRONT_LEFT camera: translated 1m to the left in vehicle frame
    ext = np.eye(4)
    ext[1, 3] = 1.0  # +Y = left in Waymo vehicle frame
    front_left = MockCalibration(
        name=2,
        intrinsic=[100.0, 100.0, 16.0, 12.0, 0, 0, 0, 0, 0],
        width=32, height=24,
        extrinsic_values=list(ext.flatten())
    )
    return [front, front_left]


def make_fake_image(h=24, w=32):
    """Random uint8 RGB image."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def make_fake_depth(h=24, w=32, min_d=1.0, max_d=5.0):
    """Random float32 depth map with plausible values."""
    return np.random.uniform(min_d, max_d, (h, w)).astype(np.float32)


# ─── Tests ──────────────────────────────────────────────────────────────────

from point_cloud import (
    create_point_cloud,
    transform_to_vehicle_frame,
    merge_point_clouds,
    create_multi_view_point_cloud,
    compute_metric_scale,
    T_CV_TO_WAYMO,
    T_WAYMO_TO_O3D,
)


def test_t_cv_to_waymo():
    """Verify the OpenCV→Waymo rotation matrix maps axes correctly."""
    # Correct mapping: Z_cv -> X_waymo, X_cv -> -Y_waymo, Y_cv -> -Z_waymo
    assert np.allclose(T_CV_TO_WAYMO @ [0, 0, 1, 1], [1, 0, 0, 1])   # Z_cv -> X_waymo (forward)
    assert np.allclose(T_CV_TO_WAYMO @ [1, 0, 0, 1], [0, -1, 0, 1])  # X_cv -> -Y_waymo (right)
    assert np.allclose(T_CV_TO_WAYMO @ [0, 1, 0, 1], [0, 0, -1, 1])  # Y_cv -> -Z_waymo (down)
    print("✓ test_t_cv_to_waymo passed")


def test_t_waymo_to_o3d():
    """Verify the Waymo->Open3D matrix maps axes for visualization."""
    # Waymo: X=Forward, Y=Left, Z=Up
    # Open3D: -Z=Forward, -X=Left, Y=Up
    assert np.allclose(T_WAYMO_TO_O3D @ [1, 0, 0, 1], [0, 0, -1, 1])   # X_waymo -> -Z_o3d
    assert np.allclose(T_WAYMO_TO_O3D @ [0, 1, 0, 1], [-1, 0, 0, 1])   # Y_waymo -> -X_o3d
    assert np.allclose(T_WAYMO_TO_O3D @ [0, 0, 1, 1], [0, 1, 0, 1])    # Z_waymo -> Y_o3d
    print("✓ test_t_waymo_to_o3d passed")


def test_create_point_cloud_basic():
    """Back-project a fake image — should get some points."""
    rgb = make_fake_image()
    depth = make_fake_depth()
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(32, 24, 100, 100, 16, 12)
    pcd = create_point_cloud(rgb, depth, intr)
    assert len(pcd.points) > 0
    print(f"✓ test_create_point_cloud_basic passed ({len(pcd.points)} points)")


def test_transform_to_vehicle_frame():
    """Points should move when a non-identity extrinsic is applied."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]))
    ext = np.eye(4)
    ext[0, 3] = 10.0  # translate +10 in X
    transform_to_vehicle_frame(pcd, ext)
    pts = np.asarray(pcd.points)
    # After T_CV_TO_WAYMO: (1,0,0) -> (0,-1,0)
    # After ext translate +10 in X: (0,-1,0) -> (10,-1,0)
    assert np.allclose(pts[0], [10.0, -1.0, 0.0], atol=1e-6)
    print("✓ test_transform_to_vehicle_frame passed")


def test_merge_point_clouds():
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.array([[1, 1, 1]]))
    merged = merge_point_clouds([pcd1, pcd2])
    assert len(merged.points) == 2
    print("✓ test_merge_point_clouds passed")


def test_merge_empty():
    merged = merge_point_clouds([])
    assert len(merged.points) == 0
    print("✓ test_merge_empty passed")


def test_compute_metric_scale():
    """Synthetic cloud: camera at Z=2m, ground at Z=-3 (wrong scale).
    Correct scale should move ground to Z≈0."""
    # Extrinsic: camera 2m above ground
    extr = np.eye(4)
    extr[2, 3] = 2.0  # cam_height = 2m

    # Build a fake point cloud where ground points sit at Z = -3
    # (instead of Z = 0 as they should in a correct metric world).
    n_ground = 200
    n_above = 50
    ground_pts = np.column_stack([
        np.random.uniform(-5, 5, n_ground),
        np.random.uniform(-5, 5, n_ground),
        np.random.uniform(-3.2, -2.8, n_ground),  # ground around Z=-3
    ])
    above_pts = np.column_stack([
        np.random.uniform(-2, 2, n_above),
        np.random.uniform(-2, 2, n_above),
        np.random.uniform(0, 2, n_above),  # some stuff above
    ])
    all_pts = np.vstack([ground_pts, above_pts])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)

    scale = compute_metric_scale(pcd, extr)

    # Apply the scale about the camera origin
    cam_origin = extr[:3, 3]
    scaled_pts = (all_pts - cam_origin) * scale + cam_origin

    # The ground points should now be near Z=0
    ground_z_after = scaled_pts[:n_ground, 2]
    assert abs(np.median(ground_z_after)) < 0.3, \
        f"Ground median Z should be ≈0, got {np.median(ground_z_after):.3f}"
    print(f"✓ test_compute_metric_scale passed (scale={scale:.3f}, "
          f"ground median Z={np.median(ground_z_after):.3f})")


def test_create_multi_view_point_cloud():
    """Full pipeline with two mock cameras."""
    rgb_images = [make_fake_image(), make_fake_image()]
    depth_maps = [make_fake_depth(), make_fake_depth()]
    
    # 2 cameras: intr = [fx, fy, cx, cy, width, height]
    intrinsics_list = [
        np.array([100.0, 100.0, 16.0, 12.0, 32.0, 24.0]),
        np.array([100.0, 100.0, 16.0, 12.0, 32.0, 24.0]),
    ]
    
    # 2 extrinsics: identity, and translated by 1m left
    ext1 = np.eye(4)
    ext2 = np.eye(4)
    ext2[1, 3] = 1.0  # +Y is left
    extrinsics_list = [ext1, ext2]

    fused = create_multi_view_point_cloud(rgb_images, depth_maps, intrinsics_list, extrinsics_list)
    n = len(fused.points)
    assert n > 0
    print(f"✓ test_create_multi_view_point_cloud passed ({n} points from 2 cameras)")


# ─── Run all ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_t_cv_to_waymo,
        test_t_waymo_to_o3d,
        test_create_point_cloud_basic,
        test_transform_to_vehicle_frame,
        test_merge_point_clouds,
        test_merge_empty,
        test_compute_metric_scale,
        test_create_multi_view_point_cloud,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        sys.exit(1)
