#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

import numpy as np
import cv2
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import base64
import io



from utils.paths import calculated_data_path
from calibration.pitch_keypoints_3d import (
	far_left_corner_3d, far_right_corner_3d,
	net_left_bottom_3d, net_right_bottom_3d, net_center_bottom_3d,
	close_white_line_center_left_far_3d, close_white_line_center_left_close_3d,
	close_white_line_center_right_far_3d, close_white_line_center_right_close_3d,
	net_left_top_3d, net_right_top_3d, net_center_top_3d
)
from calibration.pitch_corners_const import (
	far_left_corner, far_right_corner,
	net_left_bottom, net_right_bottom, net_center_bottom,
	close_white_line_center_left_far, close_white_line_center_left_close,
	close_white_line_center_right_far, close_white_line_center_right_close,
	net_left_top, net_right_top, net_center_top,
	image_width, image_height
)

# ─── Direct Linear Transform – 3 D→2 D projection ─────────────────────

#	⭾	TABS ONLY – copy / paste exactly

import numpy as np
import cv2
from scipy.optimize import least_squares

# ─── Helper – project 3-D → 2-D via 3×4 matrix ──────────────────────────
def _project_points(projection_matrix_3x4: np.ndarray, world_points_3d: np.ndarray) -> np.ndarray:
	homog = np.hstack([world_points_3d, np.ones((world_points_3d.shape[0], 1))])
	proj = (projection_matrix_3x4 @ homog.T).T
	return proj[:, :2] / proj[:, [2]]

# ─── New constrained solver ─────────────────────────────────────────────
def compute_projection_matrix_constrained(
	world_pts_3d: np.ndarray,
	image_pts_2d: np.ndarray,
	fixed_camera_x: float = 20.0,
	camera_y_bounds: tuple[float, float] = (3.0, 7.0),
	camera_z_bounds: tuple[float, float] = (0.5, 3.0)
) -> np.ndarray:
	# Use identity K (or DLT estimate), then fix K during optimisation
	P_dlt = compute_projection_matrix(world_pts_3d, image_pts_2d)
	K_init, _, _, _ = decompose_projection_matrix(P_dlt)

	# ── Initial camera pose guess ─────────────────────────────────────
	camera_center_init = np.array([fixed_camera_x, 5.0, 1.0])  # Initial C = (20, 5, 1)

	# Looking along -x → optical axis = [-1, 0, 0]
	# Build rotation matrix where columns = camera axes: [right, down, forward]
	R_init = np.array([
		[ 0,  0, -1],
		[ 0, -1,  0],
		[-1,  0,  0]
	], dtype=np.float64)
	rvec_init, _ = cv2.Rodrigues(R_init)

	param0 = np.hstack([rvec_init.ravel(), camera_center_init[1], camera_center_init[2]])

	def residuals(param_vec: np.ndarray) -> np.ndarray:
		rvec = param_vec[:3]
		cam_y = param_vec[3]
		cam_z = param_vec[4]
		camera_center = np.array([fixed_camera_x, cam_y, cam_z], dtype=np.float64)

		R_mat, _ = cv2.Rodrigues(rvec)
		t_vec = -R_mat @ camera_center

		P = K_init @ np.hstack([R_mat, t_vec.reshape(3, 1)])
		reprojected = _project_points(P, world_pts_3d)
		return (reprojected - image_pts_2d).ravel()

	# Bounds: rvec unbounded; y ∈ [3, 7], z ∈ [0.5, 3.0]
	lb = np.array([-np.inf, -np.inf, -np.inf, camera_y_bounds[0], camera_z_bounds[0]])
	ub = np.array([ np.inf,  np.inf,  np.inf, camera_y_bounds[1], camera_z_bounds[1]])

	result = least_squares(residuals, param0, bounds=(lb, ub), method="trf", verbose=0)

	rvec_opt = result.x[:3]
	cam_y_opt = result.x[3]
	cam_z_opt = result.x[4]
	camera_center_opt = np.array([fixed_camera_x, cam_y_opt, cam_z_opt], dtype=np.float64)

	R_opt, _ = cv2.Rodrigues(rvec_opt)
	t_opt = -R_opt @ camera_center_opt
	P_opt = K_init @ np.hstack([R_opt, t_opt.reshape(3, 1)])

	return P_opt / P_opt[-1, -1]

def compute_projection_matrix(world_pts, image_pts):
	num_points = world_pts.shape[0]
	design_matrix = np.zeros((2 * num_points, 12), dtype=np.float64)

	for point_index in range(num_points):
		x_world, y_world, z_world = world_pts[point_index]
		x_img, y_img = image_pts[point_index]

		design_matrix[2 * point_index] = [
			x_world, y_world, z_world, 1, 0, 0, 0, 0,
			-x_img * x_world, -x_img * y_world, -x_img * z_world, -x_img
		]
		design_matrix[2 * point_index + 1] = [
			0, 0, 0, 0, x_world, y_world, z_world, 1,
			-y_img * x_world, -y_img * y_world, -y_img * z_world, -y_img
		]

	_, _, vh_matrix = np.linalg.svd(design_matrix)
	proj_matrix = vh_matrix[-1].reshape(3, 4)
	return proj_matrix / proj_matrix[-1, -1]

# ─── Decompose P = K [R | t] ───────────────────────────────────────────
def decompose_projection_matrix(proj_matrix_3x4):
	camera_matrix_3x3 = proj_matrix_3x4[:, :3]

	def rq_decomposition(matrix_3x3):
		reversed_matrix = np.flipud(np.fliplr(matrix_3x3))
		q_matrix, r_matrix = np.linalg.qr(reversed_matrix.T)
		r_matrix = np.flipud(np.fliplr(r_matrix.T))
		q_matrix = np.flipud(np.fliplr(q_matrix.T))

		diagonal_signs = np.sign(np.diag(r_matrix))
		r_matrix *= diagonal_signs
		q_matrix *= diagonal_signs[:, np.newaxis]

		return r_matrix, q_matrix  # Intrinsics, Rotation

	intrinsic_matrix, rotation_matrix = rq_decomposition(camera_matrix_3x3)
	intrinsic_matrix /= intrinsic_matrix[-1, -1]
	translation_vector = np.linalg.inv(intrinsic_matrix) @ proj_matrix_3x4[:, 3]
	camera_position_world = -rotation_matrix.T @ translation_vector

	return intrinsic_matrix, rotation_matrix, translation_vector, camera_position_world

# ─── Project 3 D points via 3 × 4 matrix ──────────────────────────────
def reproject_points(proj_matrix_3x4, world_pts):
	homogeneous_world_pts = np.hstack([world_pts, np.ones((world_pts.shape[0], 1))])
	projected = (proj_matrix_3x4 @ homogeneous_world_pts.T).T
	return projected[:, :2] / projected[:, [2]]

# ─── 2-D diagnostic plot in an OpenCV window ──────────────────────────
def plot_2d_correspondences_opencv(
	image_rgb: np.ndarray,
	ground_truth_points_2d: np.ndarray,
	reprojected_points_2d: np.ndarray,
	test_points_2d: np.ndarray | None = None,
	test_points_3d: np.ndarray | None = None
):
	assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3
	assert ground_truth_points_2d.shape == reprojected_points_2d.shape
	assert ground_truth_points_2d.shape[1] == 2

	image_copy = image_rgb.copy()

	# ── Ground-truth vs reprojection ────────────────────────────────
	for point_index in range(len(ground_truth_points_2d)):
		x_gt, y_gt = map(int, ground_truth_points_2d[point_index])
		x_rp, y_rp = map(int, reprojected_points_2d[point_index])

		# Green – ground truth
		cv2.drawMarker(
			image_copy, (x_gt, y_gt),
			color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
			markerSize=10, thickness=2
		)

		# Blue – reprojected
		cv2.drawMarker(
			image_copy, (x_rp, y_rp),
			color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
			markerSize=10, thickness=2
		)

		# Red line between them
		cv2.line(image_copy, (x_gt, y_gt), (x_rp, y_rp), color=(0, 0, 255), thickness=1)

	# ── Extra test points (magenta) ────────────────────────────────
	if test_points_2d is not None and test_points_3d is not None:
		for point_index in range(len(test_points_2d)):
			x_tp, y_tp = map(int, test_points_2d[point_index])
			cv2.drawMarker(
				image_copy, (x_tp, y_tp),
				color=(255, 0, 255), markerType=cv2.MARKER_CROSS,
				markerSize=12, thickness=2
			)
			label_text = f"{test_points_3d[point_index][0]:.2f},{test_points_3d[point_index][1]:.2f},{test_points_3d[point_index][2]:.2f}"
			cv2.putText(
				image_copy, label_text,
				(x_tp + 5, y_tp - 5),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
				color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA
			)

	cv2.namedWindow("2D Correspondences", cv2.WINDOW_NORMAL)
	cv2.imshow("2D Correspondences", image_copy)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# ─── Optional Plotly 3-D scene ────────────────────────────────────────
def plot_3d_scene(world_pts, cam_position, cam_rotation):
	fig = go.Figure()

	fig.add_trace(go.Scatter3d(
		x=world_pts[:, 0], y=world_pts[:, 1], z=world_pts[:, 2],
		mode="markers",
		marker=dict(size=4, color="orange"),
		name="3D Points"
	))
	fig.add_trace(go.Scatter3d(
		x=[cam_position[0]],
		y=[cam_position[1]],
		z=[cam_position[2]],
		mode="markers",
		marker=dict(size=6, color="black", symbol="circle"),
		name="Camera"
	))

	cam_direction = cam_rotation.T @ np.array([0.0, 0.0, 1.0])
	axis_len = 5.0
	fig.add_trace(go.Scatter3d(
		x=[cam_position[0], cam_position[0] + axis_len * cam_direction[0]],
		y=[cam_position[1], cam_position[1] + axis_len * cam_direction[1]],
		z=[cam_position[2], cam_position[2] + axis_len * cam_direction[2]],
		mode="lines",
		line=dict(width=4, color="blue"),
		name="Optical Axis"
	))

	fig.update_layout(
		title="3D Keypoints and Camera Pose",
		scene=dict(
			xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
			aspectmode="data"
		)
	)
	fig.show()

# ─── Main entrypoint ────────────────────────────────────────────────
def main():
	# Known correspondences
	world_points = np.array([
		far_left_corner_3d, far_right_corner_3d,
		net_left_bottom_3d, net_right_bottom_3d, net_center_bottom_3d,
		close_white_line_center_left_far_3d, close_white_line_center_left_close_3d,
		close_white_line_center_right_far_3d, close_white_line_center_right_close_3d,
		net_left_top_3d, net_right_top_3d, net_center_top_3d
	], dtype=np.float64)

	image_points = np.array([
		far_left_corner, far_right_corner,
		net_left_bottom, net_right_bottom, net_center_bottom,
		close_white_line_center_left_far, close_white_line_center_left_close,
		close_white_line_center_right_far, close_white_line_center_right_close,
		net_left_top, net_right_top, net_center_top
	], dtype=np.float64)

	# Load reference frame
	average_frame_path = calculated_data_path / "game1_3.mp4" / "average_frame.bmp"
	img_bgr = cv2.imread(str(average_frame_path))
	assert img_bgr is not None, f"Image not found: {average_frame_path}"
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

	# Compute projection
	# projection_mat = compute_projection_matrix(world_points, image_points)
	projection_mat = compute_projection_matrix_constrained(world_points, image_points)
	intrinsics, rotation, translation, cam_position = decompose_projection_matrix(projection_mat)
	reprojected_image_points = reproject_points(projection_mat, world_points)

	# ── Define EXTRA test 3-D points ──────────────────────────────
	test_points_3d = np.array([
		(net_left_top_3d + net_center_top_3d) / 2.0,			# Midpoint left-top ↔ center-top
		(net_center_top_3d + net_right_top_3d) / 2.0,			# Midpoint center-top ↔ right-top
		np.array([5.0, 10.0, 1.0]),							# Over the net center (1 m high)
		np.array([5.0, 12.0, 1.0])							# Two metres behind the net, 1 m high
	], dtype=np.float64)
	test_points_2d = reproject_points(projection_mat, test_points_3d)

	# ── 2-D diagnostics including test points ────────────────────
	plot_2d_correspondences_opencv(
		img_rgb,
		image_points,
		reprojected_image_points,
		test_points_2d=test_points_2d,
		test_points_3d=test_points_3d
	)
	plot_3d_scene(world_points, cam_position, rotation)		# Optional 3-D view

	# ── Console output ───────────────────────────────────────────
	np.set_printoptions(precision=4, suppress=True)
	print("\nCamera Intrinsics (K):")
	print(intrinsics)
	print("\nCamera Rotation Matrix (R):")
	print(rotation)
	print("\nCamera Translation Vector (t):")
	print(translation)
	print("\nCamera World Position:")
	print(cam_position)

if __name__ == "__main__":
	main()
