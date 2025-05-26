#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

import numpy as np
import cv2
from utils.paths import calculated_data_path
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ─── 2-D image points ─────────────────────────────────────────────────
from calibration.pitch_corners_const import (
	net_left_bottom, net_right_bottom, net_center_bottom,
	far_left_corner, far_right_corner,
	image_width, image_height, net_left_top, net_right_top, net_center_top,
	close_white_line_center_right_far, close_white_line_center_right_close,
	close_white_line_center_left_close, close_white_line_center_left_far,
)

from calibration.pitch_keypoints_3d import (
	far_left_corner_3d, far_right_corner_3d,
	net_left_bottom_3d, net_right_bottom_3d, net_center_bottom_3d,
	net_left_top_3d, net_right_top_3d, net_center_top_3d,
	close_white_line_center_left_close_3d, close_white_line_center_left_far_3d,
	close_white_line_center_right_close_3d, close_white_line_center_right_far_3d
)

class Camera:
	def __init__(self, image_width, image_height, object_points_3d, image_points_2d):
		self._image_width = image_width
		self._image_height = image_height
		self._object_points_3d = np.asarray(object_points_3d, dtype=np.float32)
		self._image_points_2d = np.asarray(image_points_2d, dtype=np.float32)

		self._camera_matrix = cv2.initCameraMatrix2D(
			[self._object_points_3d],
			[self._image_points_2d],
			(self._image_width, self._image_height),
			aspectRatio=1.0
		)
		self._dist_coeffs = np.zeros((4, 1))	# Assume no distortion

		self._rotation_vector = None
		self._translation_vector = None
		self._rotation_matrix = None
		self._projection_matrix = None

		self._calibrate()

	def _calibrate(self):
		success, rvec, tvec = cv2.solvePnP(
			objectPoints=self._object_points_3d,
			imagePoints=self._image_points_2d,
			cameraMatrix=self._camera_matrix,
			distCoeffs=self._dist_coeffs,
			flags=cv2.SOLVEPNP_ITERATIVE
		)
		if not success:
			raise RuntimeError("cv2.solvePnP() failed during calibration.")

		self._rotation_vector = rvec
		self._translation_vector = tvec
		self._rotation_matrix, _ = cv2.Rodrigues(self._rotation_vector)

		extrinsic_matrix = np.hstack((self._rotation_matrix, self._translation_vector))
		self._projection_matrix = self._camera_matrix @ extrinsic_matrix

	@property
	def camera_matrix(self):
		return self._camera_matrix.copy()

	@property
	def rotation_vector(self):
		return self._rotation_vector.copy()

	@property
	def translation_vector(self):
		return self._translation_vector.copy()

	@property
	def rotation_matrix(self):
		return self._rotation_matrix.copy()

	@property
	def projection_matrix(self):
		return self._projection_matrix.copy()

	def project_points(self, object_points_3d):
		object_points_3d = np.asarray(object_points_3d, dtype=np.float32)
		projected, _ = cv2.projectPoints(
			object_points_3d,
			self._rotation_vector,
			self._translation_vector,
			self._camera_matrix,
			self._dist_coeffs
		)

		error = np.linalg.norm(self._image_points_2d - projected.reshape(-1, 2), axis=1)
		print(f"\n[INFO] Reprojection error per point:\n{error}")
		print(f"[INFO] Mean reprojection error: {np.mean(error)} pixels")
		return projected.reshape(-1, 2)

# ─── Build Input Points ────────────────────────────────────────────────
image_points_2d = [
	far_left_corner,
	far_right_corner,
	# net_left_bottom,
	# net_right_bottom,
	net_center_bottom,
	close_white_line_center_left_far,
	# close_white_line_center_right_far,
	# close_white_line_center_left_close,
	# close_white_line_center_right_close,
	net_left_top,
	net_right_top,
	net_center_top
]

object_points_3d = [
	far_left_corner_3d,
	far_right_corner_3d,
	# net_left_bottom_3d,
	# net_right_bottom_3d,
	net_center_bottom_3d,
	close_white_line_center_left_far_3d,
	# close_white_line_center_right_far_3d,
	# close_white_line_center_left_close_3d,
	# close_white_line_center_right_close_3d,
	net_left_top_3d,
	net_right_top_3d,
	net_center_top_3d
]

# ─── Instantiate and Run ───────────────────────────────────────────────
camera = Camera(
	image_width=image_width,
	image_height=image_height,
	object_points_3d=object_points_3d,
	image_points_2d=image_points_2d
)

print("[INFO] Camera Matrix:")
print(camera.camera_matrix)

print("\n[INFO] Rotation Vector:")
print(camera.rotation_vector)

print("\n[INFO] Translation Vector:")
print(camera.translation_vector)

print("\n[INFO] Rotation Matrix:")
print(camera.rotation_matrix)

print("\n[INFO] Projection Matrix (3x4):")
print(camera.projection_matrix)

# ─── Optional: Reproject and Compare ───────────────────────────────────
projected = camera.project_points(object_points_3d)

print("\n[INFO] Reprojected 2D points:")
for i, pt in enumerate(projected):
	print(f"  Reprojected: {pt}  ← Original: {image_points_2d[i]}")

# ─── Visualize 2D Keypoints vs Reprojections ──────────────────────────
assert calculated_data_path.is_dir()
average_frame_path = calculated_data_path / "game1_3.mp4" / "average_frame.bmp"
assert average_frame_path.is_file()

pitch_image = cv2.imread(str(average_frame_path))
assert pitch_image is not None

# Resize factor
scale_factor = 0.25
resized_image = cv2.resize(pitch_image, (0, 0), fx=scale_factor, fy=scale_factor)
points_on_image = resized_image.copy()

# Draw all
for i, ((x_img, y_img), (x_proj, y_proj)) in enumerate(zip(image_points_2d, projected)):
	x_img_scaled	= int(x_img * scale_factor)
	y_img_scaled	= int(y_img * scale_factor)
	x_proj_scaled	= int(x_proj * scale_factor)
	y_proj_scaled	= int(y_proj * scale_factor)

	# Original 2D point – green cross
	cv2.drawMarker(
		points_on_image,
		position=(x_img_scaled, y_img_scaled),
		color=(0, 255, 0),	# Green
		markerType=cv2.MARKER_CROSS,
		markerSize=15,
		thickness=2
	)

	# Projected point – blue cross
	cv2.drawMarker(
		points_on_image,
		position=(x_proj_scaled, y_proj_scaled),
		color=(255, 0, 0),	# Blue
		markerType=cv2.MARKER_CROSS,
		markerSize=15,
		thickness=2
	)

	# Line between them – red
	cv2.line(
		points_on_image,
		pt1=(x_img_scaled, y_img_scaled),
		pt2=(x_proj_scaled, y_proj_scaled),
		color=(0, 0, 255),	# Red
		thickness=1,
		lineType=cv2.LINE_AA
	)

# Show window
plt.figure(figsize=(10, 8))
cv2.imshow("2D Points vs Projections (scaled)", cv2.cvtColor(points_on_image, cv2.COLOR_BGR2RGB))
plt.title("2D Points vs Projections (scaled)")
plt.axis('off')
plt.show()

# Save to file
output_path = calculated_data_path / "reprojection_overlay.scaled.png"
cv2.imwrite(str(output_path), points_on_image)
print(f"[INFO] Saved reprojection visualization → {output_path}")

# visualize 3d with plotly

def visualize_3d_scene_plotly(object_points_3d, camera):
	object_points_3d = np.asarray(object_points_3d)
	xs, ys, zs = object_points_3d[:, 0], object_points_3d[:, 1], object_points_3d[:, 2]

	# Camera pose
	R = camera.rotation_matrix
	T = camera.translation_vector.reshape(3)
	camera_center = -R.T @ T

	# Camera axes (length = 1.0)
	axis_length = 1.0
	x_axis = R.T @ np.array([axis_length, 0, 0])
	y_axis = R.T @ np.array([0, axis_length, 0])
	z_axis = R.T @ np.array([0, 0, axis_length])

	x_axis_end = camera_center + x_axis
	y_axis_end = camera_center + y_axis
	z_axis_end = camera_center + z_axis

	fig = go.Figure()

	# 3D keypoints
	fig.add_trace(go.Scatter3d(
		x=xs,
		y=ys,
		z=zs,
		mode='markers+text',
		text=[f"P{i}" for i in range(len(xs))],
		textposition='top center',
		marker=dict(size=4, color='blue'),
		name='3D keypoints'
	))

	# Camera center
	fig.add_trace(go.Scatter3d(
		x=[camera_center[0]],
		y=[camera_center[1]],
		z=[camera_center[2]],
		mode='markers+text',
		text=["Camera"],
		textposition='bottom center',
		marker=dict(size=6, color='red', symbol='circle'),
		name='Camera Center'
	))

	# Axis lines
	def axis_line(start, end, color, name):
		return go.Scatter3d(
			x=[start[0], end[0]],
			y=[start[1], end[1]],
			z=[start[2], end[2]],
			mode='lines',
			line=dict(color=color, width=5),
			name=name
		)

	fig.add_trace(axis_line(camera_center, x_axis_end, 'red', 'X-axis'))
	fig.add_trace(axis_line(camera_center, y_axis_end, 'green', 'Y-axis'))
	fig.add_trace(axis_line(camera_center, z_axis_end, 'blue', 'Z-axis'))

	# Axis cones
	def axis_cone(end, vec, color, name):
		return go.Cone(
			x=[end[0]], y=[end[1]], z=[end[2]],
			u=[vec[0]], v=[vec[1]], w=[vec[2]],
			sizemode="absolute",
			sizeref=0.2,  # Replaced the invalid 'sizemax' with 'sizeref'
			anchor="tail",
			colorscale=[[0, color], [1, color]],
			showscale=False,
			name=name
		)

	fig.add_trace(axis_cone(x_axis_end, x_axis * 0.001, 'red', 'X-dir'))
	fig.add_trace(axis_cone(y_axis_end, y_axis * 0.001, 'green', 'Y-dir'))
	fig.add_trace(axis_cone(z_axis_end, z_axis * 0.001, 'blue', 'Z-dir'))

	fig.update_layout(
		title="3D Calibration Visualization with Camera Direction",
		scene=dict(
			xaxis_title='X (m)',
			yaxis_title='Y (m)',
			zaxis_title='Z (m)',
			aspectmode='data'
		),
		width=900,
		height=700,
		showlegend=True
	)

	fig.show()

visualize_3d_scene_plotly(object_points_3d, camera)