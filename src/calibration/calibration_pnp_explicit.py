#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

import numpy as np
import cv2
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import base64
import io
import plotly.io as pio


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

def reproject_points(proj_matrix_3x4, world_pts):
	homogeneous_world_pts = np.hstack([world_pts, np.ones((world_pts.shape[0], 1))])
	projected = (proj_matrix_3x4 @ homogeneous_world_pts.T).T
	return projected[:, :2] / projected[:, [2]]

def plot_2d_correspondences(img_rgb, gt_2d_pts, reprojected_pts):
	# Convert numpy array to base64 encoded image
	pil_img = Image.fromarray(img_rgb)
	buffer = io.BytesIO()
	pil_img.save(buffer, format="PNG")
	img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
	img_src = f"data:image/png;base64,{img_str}"

	fig = go.Figure()

	fig.add_layout_image(
		dict(
			source=img_src,
			xref="x", yref="y",
			x=0, y=0,
			sizex=image_width, sizey=image_height,
			sizing="stretch", layer="below"
		)
	)
	fig.update_xaxes(visible=False, range=[0, image_width])
	fig.update_yaxes(visible=False, range=[image_height, 0])

	fig.add_trace(go.Scatter(
		x=gt_2d_pts[:, 0], y=gt_2d_pts[:, 1],
		mode="markers", name="Ground Truth",
		marker=dict(symbol="x", size=14, color="green")
	))
	fig.add_trace(go.Scatter(
		x=reprojected_pts[:, 0], y=reprojected_pts[:, 1],
		mode="markers", name="Reprojected",
		marker=dict(symbol="x", size=14, color="blue")
	))

	for i, (orig, proj) in enumerate(zip(gt_2d_pts, reprojected_pts)):
		fig.add_trace(go.Scatter(
			x=[orig[0], proj[0]],
			y=[orig[1], proj[1]],
			mode="lines", line=dict(color="red"),
			showlegend=False
		))

	fig.update_layout(
		width=image_width,
		height=image_height,
		title="2D Correspondences: Green=Ground Truth, Blue=Reprojected"
	)

	fig.show()

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

def main():
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

	average_frame_path = calculated_data_path / "game1_3.mp4" / "average_frame.bmp"
	img_bgr = cv2.imread(str(average_frame_path))
	assert img_bgr is not None, f"Image not found: {average_frame_path}"
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

	projection_mat = compute_projection_matrix(world_points, image_points)
	intrinsics, rotation, translation, cam_position = decompose_projection_matrix(projection_mat)
	reprojected_image_points = reproject_points(projection_mat, world_points)

	plot_2d_correspondences(img_rgb, image_points, reprojected_image_points)
	plot_3d_scene(world_points, cam_position, rotation)

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