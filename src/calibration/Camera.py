#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

import numpy as np
import cv2
import plotly.graph_objects as go
from pathlib import Path
from typing import Tuple
from scipy.optimize import least_squares

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
	net_left_top, net_right_top, net_center_top
)


class Camera:
	def __init__(
		self,
		world_points_3d: np.ndarray,
		image_points_2d: np.ndarray,
		fixed_camera_x_position: float = 20.0,
		camera_y_bounds: Tuple[float, float] = (3.0, 7.0),
		camera_z_bounds: Tuple[float, float] = (0.5, 3.0)
	):
		self._world_points_3d = world_points_3d
		self._image_points_2d = image_points_2d
		self._fixed_camera_x = fixed_camera_x_position
		self._camera_y_bounds = camera_y_bounds
		self._camera_z_bounds = camera_z_bounds

		self._compute_calibration()

	def _project(self, projection_matrix: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
		homogeneous_points = np.hstack([world_pts, np.ones((world_pts.shape[0], 1))])
		projected = (projection_matrix @ homogeneous_points.T).T
		return projected[:, :2] / projected[:, [2]]

	def _compute_calibration(self):
		projection_matrix_initial = self._compute_dlt_projection_matrix()
		camera_matrix_initial, _, _, _ = self._decompose_projection_matrix(projection_matrix_initial)
		self._intrinsic_matrix_initial = camera_matrix_initial

		initial_camera_center = np.array([self._fixed_camera_x, 5.0, 1.0], dtype=np.float64)

		rotation_matrix_initial = np.array([
			[ 0,  0, -1],
			[ 0, -1,  0],
			[-1,  0,  0]
		], dtype=np.float64)

		rotation_vector_initial, _ = cv2.Rodrigues(rotation_matrix_initial)

		initial_parameters = np.hstack([
			rotation_vector_initial.ravel(),
			initial_camera_center[1],
			initial_camera_center[2]
		])

		def residual_function(parameter_vector: np.ndarray) -> np.ndarray:
			rotation_vector = parameter_vector[:3]
			camera_y = parameter_vector[3]
			camera_z = parameter_vector[4]
			camera_center = np.array([self._fixed_camera_x, camera_y, camera_z])

			rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
			translation_vector = -rotation_matrix @ camera_center

			projection_matrix = camera_matrix_initial @ np.hstack([rotation_matrix, translation_vector.reshape(3, 1)])
			reprojected = self._project(projection_matrix, self._world_points_3d)

			return (reprojected - self._image_points_2d).ravel()

		lower_bounds = np.array([
			-np.inf, -np.inf, -np.inf,
			self._camera_y_bounds[0], self._camera_z_bounds[0]
		])
		upper_bounds = np.array([
			np.inf, np.inf, np.inf,
			self._camera_y_bounds[1], self._camera_z_bounds[1]
		])

		result = least_squares(
			residual_function,
			initial_parameters,
			bounds=(lower_bounds, upper_bounds),
			method="trf"
		)

		rotation_vector_optimized = result.x[:3]
		camera_y_optimized = result.x[3]
		camera_z_optimized = result.x[4]

		self._camera_center = np.array([self._fixed_camera_x, camera_y_optimized, camera_z_optimized])
		self._rotation_matrix, _ = cv2.Rodrigues(rotation_vector_optimized)
		self._translation_vector = -self._rotation_matrix @ self._camera_center

		self._projection_matrix = camera_matrix_initial @ np.hstack([
			self._rotation_matrix, self._translation_vector.reshape(3, 1)
		])
		self._projection_matrix /= self._projection_matrix[-1, -1]

		self._camera_matrix, _, _, _ = self._decompose_projection_matrix(self._projection_matrix)

	def _compute_dlt_projection_matrix(self) -> np.ndarray:
		num_points = self._world_points_3d.shape[0]
		design_matrix = np.zeros((2 * num_points, 12))

		for i in range(num_points):
			x, y, z = self._world_points_3d[i]
			u, v = self._image_points_2d[i]

			design_matrix[2 * i] = [
				x, y, z, 1, 0, 0, 0, 0,
				-u * x, -u * y, -u * z, -u
			]
			design_matrix[2 * i + 1] = [
				0, 0, 0, 0, x, y, z, 1,
				-v * x, -v * y, -v * z, -v
			]

		_, _, vh = np.linalg.svd(design_matrix)
		projection_matrix = vh[-1].reshape(3, 4)
		return projection_matrix / projection_matrix[-1, -1]

	def _decompose_projection_matrix(self, projection_matrix: np.ndarray):
		camera_matrix = projection_matrix[:, :3]

		def rq(matrix_3x3):
			reversed_matrix = np.flipud(np.fliplr(matrix_3x3))
			q, r = np.linalg.qr(reversed_matrix.T)
			r = np.flipud(np.fliplr(r.T))
			q = np.flipud(np.fliplr(q.T))
			signs = np.sign(np.diag(r))
			r *= signs
			q *= signs[:, np.newaxis]
			return r, q

		intrinsic, rotation = rq(camera_matrix)
		intrinsic /= intrinsic[-1, -1]
		translation = np.linalg.inv(intrinsic) @ projection_matrix[:, 3]
		camera_center = -rotation.T @ translation

		return intrinsic, rotation, translation, camera_center

	def project_world_to_image(self, world_points: np.ndarray) -> np.ndarray:
		return self._project(self._projection_matrix, world_points)

	def backproject_point_with_z(self, image_point_2d: Tuple[float, float], z_world_known: float) -> np.ndarray:
		"""
        Backproject a 2D image point to 3D world coordinates assuming known Z (height).

        Args:
            image_point_2d: (u, v) pixel coordinates
            z_world_known: known Z (height) in world coordinates

        Returns:
            Numpy array [x, y, z] in world coordinates
        """
		u, v = image_point_2d
		pixel_homogeneous = np.array([u, v, 1.0], dtype=np.float64)

		# Ray in camera coordinates
		ray_direction_camera = np.linalg.inv(self._camera_matrix) @ pixel_homogeneous

		# Ray in world coordinates
		ray_direction_world = self._rotation_matrix.T @ ray_direction_camera
		camera_origin_world = self._camera_center

		# Solve for intersection with plane Z = z_world_known
		scale_numerator = z_world_known - camera_origin_world[2]
		scale_denominator = ray_direction_world[2]

		if abs(scale_denominator) < 1e-6:
			raise ValueError("Ray is parallel to the Z = constant plane")

		scale = scale_numerator / scale_denominator
		point_3d_world = camera_origin_world + scale * ray_direction_world

		return point_3d_world

	@property
	def camera_matrix(self) -> np.ndarray:
		return self._camera_matrix

	@property
	def rotation_matrix(self) -> np.ndarray:
		return self._rotation_matrix

	@property
	def translation_vector(self) -> np.ndarray:
		return self._translation_vector

	@property
	def camera_center_world(self) -> np.ndarray:
		return self._camera_center

	@property
	def projection_matrix(self) -> np.ndarray:
		return self._projection_matrix

	@property
	def focal_length(self) -> Tuple[float, float]:
		return self._camera_matrix[0, 0], self._camera_matrix[1, 1]


def plot_2d_correspondences_opencv(image_rgb, ground_truth_points_2d, reprojected_points_2d, test_points_2d=None, test_points_3d=None):
	assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3
	image_copy = image_rgb.copy()

	for i in range(len(ground_truth_points_2d)):
		x_gt, y_gt = map(int, ground_truth_points_2d[i])
		x_rp, y_rp = map(int, reprojected_points_2d[i])
		cv2.drawMarker(image_copy, (x_gt, y_gt), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
		cv2.drawMarker(image_copy, (x_rp, y_rp), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
		cv2.line(image_copy, (x_gt, y_gt), (x_rp, y_rp), (0, 0, 255), 1)

	if test_points_2d is not None and test_points_3d is not None:
		for i in range(len(test_points_2d)):
			x_tp, y_tp = map(int, test_points_2d[i])
			cv2.drawMarker(image_copy, (x_tp, y_tp), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
			label = f"{test_points_3d[i][0]:.2f},{test_points_3d[i][1]:.2f},{test_points_3d[i][2]:.2f}"
			cv2.putText(image_copy, label, (x_tp + 5, y_tp - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

	cv2.namedWindow("2D Correspondences", cv2.WINDOW_NORMAL)
	cv2.imshow("2D Correspondences", image_copy)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def plot_3d_scene(world_pts, cam_position, cam_rotation):
	fig = go.Figure()
	fig.add_trace(go.Scatter3d(
		x=world_pts[:, 0], y=world_pts[:, 1], z=world_pts[:, 2],
		mode="markers", marker=dict(size=4, color="orange"), name="3D Points"
	))
	fig.add_trace(go.Scatter3d(
		x=[cam_position[0]], y=[cam_position[1]], z=[cam_position[2]],
		mode="markers", marker=dict(size=6, color="black", symbol="circle"), name="Camera"
	))
	cam_direction = cam_rotation.T @ np.array([0.0, 0.0, 1.0])
	fig.add_trace(go.Scatter3d(
		x=[cam_position[0], cam_position[0] + 5.0 * cam_direction[0]],
		y=[cam_position[1], cam_position[1] + 5.0 * cam_direction[1]],
		z=[cam_position[2], cam_position[2] + 5.0 * cam_direction[2]],
		mode="lines", line=dict(width=4, color="blue"), name="Optical Axis"
	))
	fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"))
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

	camera_model = Camera(world_points, image_points)


	image_path = calculated_data_path / "game1_3.mp4" / "average_frame.bmp"
	img_bgr = cv2.imread(str(image_path))
	assert img_bgr is not None
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


	reprojected_points = camera_model.project_world_to_image(world_points)

	test_points_3d = np.array([
		(net_left_top_3d + net_center_top_3d) / 2.0,
		(net_center_top_3d + net_right_top_3d) / 2.0,
		np.array([12., 3., 1.0]),
		np.array([15, 8., 1.0])
	], dtype=np.float64)

	test_points_2d = camera_model.project_world_to_image(test_points_3d)

	plot_2d_correspondences_opencv(
		img_rgb,
		image_points,
		reprojected_points,
		test_points_2d=test_points_2d,
		test_points_3d=test_points_3d
	)

	np.set_printoptions(precision=4, suppress=True)
	print("\nCamera Intrinsics (K):")
	print(camera_model.camera_matrix)
	print("\nCamera Rotation Matrix (R):")
	print(camera_model.rotation_matrix)
	print("\nCamera Translation Vector (t):")
	print(camera_model.translation_vector)
	print("\nCamera World Position:")
	print("Final Camera Center:", camera_model.camera_center_world)

if __name__ == "__main__":
	main()
