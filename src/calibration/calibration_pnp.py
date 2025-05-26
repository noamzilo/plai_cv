#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

import numpy as np
import cv2
from utils.paths import calculated_data_path

# ─── 2-D image points ─────────────────────────────────────────────────
from calibration.pitch_corners_const import (
	net_left_bottom, net_right_bottom, net_center_bottom,
	far_left_corner, far_right_corner, close_white_line_center,
	image_width, image_height
)

from calibration.pitch_keypoints import (
	far_left_corner_3d, far_right_corner_3d,
	net_left_bottom_3d, net_right_bottom_3d, net_center_bottom_3d,
	close_white_line_center_3d
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
	net_left_bottom,
	net_right_bottom,
	net_center_bottom,
	close_white_line_center
]

object_points_3d = [
	far_left_corner_3d,
	far_right_corner_3d,
	net_left_bottom_3d,
	net_right_bottom_3d,
	net_center_bottom_3d,
	close_white_line_center_3d
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

# Draw original 2D points as green +
for (x, y) in image_points_2d:
	x_scaled = int(x * scale_factor)
	y_scaled = int(y * scale_factor)
	cv2.drawMarker(
		points_on_image,
		position=(x_scaled, y_scaled),
		color=(0, 255, 0),	# Green
		markerType=cv2.MARKER_CROSS,
		markerSize=15,
		thickness=2
	)

# Draw projected 2D points as blue +
for (x, y) in projected:
	x_scaled = int(x * scale_factor)
	y_scaled = int(y * scale_factor)
	cv2.drawMarker(
		points_on_image,
		position=(x_scaled, y_scaled),
		color=(255, 0, 0),	# Blue
		markerType=cv2.MARKER_CROSS,
		markerSize=15,
		thickness=2
	)

# Show window
cv2.imshow("2D Points vs Projections (scaled)", points_on_image)
cv2.waitKey(0)

# Save to file
output_path = calculated_data_path / "reprojection_overlay.scaled.png"
cv2.imwrite(str(output_path), points_on_image)
print(f"[INFO] Saved reprojection visualization → {output_path}")
