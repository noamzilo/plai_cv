#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

from collections import OrderedDict
import \
	numpy as np
import \
	pandas as pd
import \
	cv2
from pathlib import \
	Path

# ─── Project-specific paths ───────────────────────────────────────────
from utils.paths import \
	(
	raw_data_path,
	tracked_detections_csv_path,
)
from acquisition.VideoReader import \
	VideoReader

from calibration.pitch_keypoints_3d import \
	(
	far_left_corner_3d,
	far_right_corner_3d,
	net_left_bottom_3d,
	net_right_bottom_3d,
)
from calibration.pitch_corners_const import \
	(
	far_left_corner,
	far_right_corner,
	net_left_bottom,
	net_right_bottom,
	net_center_bottom,
)

# ─── Display constants ────────────────────────────────────────────────
image_scale_factor			= 0.25			# show camera view at 25 %
pitch_inner_size_px			= (600, 300)	# (width, height) of pitch area
buffer_px					= 100			# buffer on all four sides

# ─── Player colours ───────────────────────────────────────────────────
player_colors_bgr = [
	(255,   0,   0),	# Player 0 – Red
	(  0, 255,   0),	# Player 1 – Green
	(  0,   0, 255),	# Player 2 – Blue
	(255, 255,   0),	# Player 3 – Cyan
]
player_color_array = np.array(player_colors_bgr, dtype=np.uint8)

# ─── 2-D top-view pitch representation ───────────────────────────────
class Pitch2d:
	def __init__(
			self,
			length_m: float							= 20.0,
			width_m: float							= 10.0,
			canvas_inner_px: tuple[int, int]		= pitch_inner_size_px,
			pitch_2d_corners_4x2: np.ndarray | None	= None,
			img_2d_corners_4x2: np.ndarray | None	= None,
			orientation: str							= "vertical",	# default vertical
	):
		self.len_m, self.wid_m	= length_m, width_m
		self.orientation		= orientation.lower()
		if self.orientation not in ("horizontal", "vertical"):
			raise ValueError("orientation must be 'horizontal' or 'vertical'.")

		inner_w_px, inner_h_px	= (
			canvas_inner_px if self.orientation == "horizontal" else canvas_inner_px[::-1]
		)
		self.inner_w_px, self.inner_h_px	= inner_w_px, inner_h_px
		self.canvas_w_px						= inner_w_px + buffer_px * 2
		self.canvas_h_px						= inner_h_px + buffer_px * 2

		if pitch_2d_corners_4x2 is not None and img_2d_corners_4x2 is not None:
			self.homography_3x3, _ = cv2.findHomography(
				pitch_2d_corners_4x2,
				img_2d_corners_4x2,
				method = cv2.RANSAC,
				ransacReprojThreshold = 3.0,
			)
			self.inv_homography_3x3 = np.linalg.inv(self.homography_3x3)
		else:
			self.homography_3x3		= None
			self.inv_homography_3x3	= None

	def blank_canvas(self) -> np.ndarray:
		img = np.zeros((self.canvas_h_px, self.canvas_w_px, 3), np.uint8)
		cv2.rectangle(
			img,
			(buffer_px, buffer_px),
			(buffer_px + self.inner_w_px - 1, buffer_px + self.inner_h_px - 1),
			(255, 255, 255),
			2,
		)
		return img

	def pitch_to_canvas(self, pitch_xy_m: tuple[float, float]) -> tuple[int, int]:
		x_m, y_m = pitch_xy_m
		if self.orientation == "horizontal":
			cx_inner = int((x_m / self.len_m) * self.inner_w_px)
			cy_inner = int((1.0 - y_m / self.wid_m) * self.inner_h_px)
		else:	# vertical
			cx_inner = int((y_m / self.wid_m) * self.inner_w_px)
			cy_inner = int((x_m / self.len_m) * self.inner_h_px)
		return cx_inner + buffer_px, cy_inner + buffer_px

	def draw_points(
			self,
			canvas: np.ndarray,
			points: list[tuple[tuple[float, float], tuple[int, int, int], str]],
			font_scale: float = 0.5,
	) -> np.ndarray:
		out = canvas.copy()
		for (px, py), col, lbl in points:
			if np.isnan(px):
				continue
			cx, cy = self.pitch_to_canvas((px, py))
			cv2.circle(out, (cx, cy), 6, col, -1)
			if lbl:
				cv2.putText(out, lbl, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 1, cv2.LINE_AA)
		return out

	def draw_polyline(
			self,
			canvas: np.ndarray,
			pitch_xy_seq: list[tuple[float, float]],
			color_bgr: tuple[int, int, int],
			thickness: int = 2,
	) -> np.ndarray:
		if len(pitch_xy_seq) < 2:
			return canvas
		out = canvas.copy()
		p_canvas = [self.pitch_to_canvas(pt) for pt in pitch_xy_seq]
		for p0, p1 in zip(p_canvas[:-1], p_canvas[1:]):
			cv2.line(out, p0, p1, color_bgr, thickness)
		return out

	def image_to_pitch_batch(self, img_xy_batch: np.ndarray) -> np.ndarray:
		return self._apply_homography(img_xy_batch, self.inv_homography_3x3)

	def _apply_homography(self, xy_batch: np.ndarray, homography: np.ndarray) -> np.ndarray:
		if homography is None:
			raise ValueError("Homography has not been initialized.")
		if len(xy_batch.shape) == 1:
			xy_batch = xy_batch[np.newaxis, :]
		homo = np.hstack([xy_batch, np.ones((xy_batch.shape[0], 1), dtype=np.float32)])
		transformed = (homography @ homo.T).T
		return transformed[:, :2] / transformed[:, 2:3]

# ─── Static pitch background ─────────────────────────────────────────
def create_static_pitch_image(pitch2d: Pitch2d) -> np.ndarray:
	# static image points → pitch points
	named_img_points = OrderedDict([
		("deadzone0", np.array([0, 1055])),
		("deadzone1", np.array([0, 2159])),
		("deadzone2", np.array([3839, 2159])),
		("deadzone3", np.array([3839, 1174])),
		("far_right_corner", far_right_corner),
		("far_left_corner", far_left_corner),
		("net_right_bottom", net_right_bottom),
		("net_left_bottom", net_left_bottom),
	])
	# convert & build poly-line order
	img_pts		= np.array(list(named_img_points.values()), dtype=np.float32)
	pitch_pts	= pitch2d.image_to_pitch_batch(img_pts)

	# build canvas
	canvas = pitch2d.blank_canvas()

	# polyline around court (white)
	poly_order_names = [
		"far_left_corner", "far_right_corner", "net_right_bottom",
		"deadzone3", "deadzone2", "deadzone1", "deadzone0",
		"net_left_bottom", "far_left_corner",
	]
	poly_pts = [pitch_pts[list(named_img_points.keys()).index(n)] for n in poly_order_names]
	canvas = pitch2d.draw_polyline(canvas, poly_pts, (255, 255, 255), 2)

	# net line – centre of court
	net_pts = [(pitch2d.len_m / 2.0, 0.0), (pitch2d.len_m / 2.0, pitch2d.wid_m)]
	canvas  = pitch2d.draw_polyline(canvas, net_pts, (255, 255, 255), 2)

	return canvas

# ─── Video + pitch overlay ───────────────────────────────────────────
class PitchTrackerVisualizer:
	def __init__(
			self,
			video_path: Path,
			detections_dataframe: pd.DataFrame,
			pitch_2d_corners_4x2: np.ndarray,
			img_2d_corners_4x2: np.ndarray,
	):
		self.detections_df	= detections_dataframe
		self.pitch			= Pitch2d(
			pitch_2d_corners_4x2 = pitch_2d_corners_4x2,
			img_2d_corners_4x2	 = img_2d_corners_4x2,
			orientation			 = "vertical",
		)
		self.static_canvas	= create_static_pitch_image(self.pitch)
		self.video_reader	= VideoReader(video_path)

	def show_video_with_pitch_overlay(self) -> None:
		for frame_idx, full_frame in self.video_reader.video_frames_generator():
			ind_df = self.detections_df[self.detections_df.frame_ind == frame_idx]
			if ind_df.empty:
				continue

			# detection → pitch XY (metres)
			dx = ((ind_df.x1 + ind_df.x2) / 2).to_numpy(np.float32)
			dy = ind_df.y2.to_numpy(np.float32)
			img_xy = np.stack([dx, dy], axis=1)
			pitch_xy = self.pitch.image_to_pitch_batch(img_xy)

			player_ids	= ind_df.player_id.to_numpy(np.int32)
			valid_mask	= ind_df.is_valid.to_numpy(bool)

			# dynamic player point-specs
			player_points = []
			for pid, (px, py), valid in zip(player_ids, pitch_xy, valid_mask):
				if not valid:
					continue
				col = player_color_array[pid % len(player_color_array)].tolist()
				player_points.append(((float(px), float(py)), col, f"P{pid}"))

			# build pitch for this frame
			pitch_img = self.static_canvas.copy()
			pitch_img = self.pitch.draw_points(pitch_img, player_points, font_scale = 0.5)

			# draw detection boxes on camera frame
			scaled_h = int(full_frame.shape[0] * image_scale_factor)
			scaled_w = int(full_frame.shape[1] * image_scale_factor)
			camera_small = cv2.resize(full_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

			for pid, (x1, y1, x2, y2), valid in zip(
					player_ids,
					ind_df[["x1", "y1", "x2", "y2"]].values,
					valid_mask,
			):
				if not valid:
					continue
				col	= player_color_array[pid % len(player_color_array)].tolist()
				sx1, sy1, sx2, sy2 = [int(c * image_scale_factor) for c in (x1, y1, x2, y2)]
				cv2.rectangle(camera_small, (sx1, sy1), (sx2, sy2), col, 2)
				cv2.putText(camera_small, f"P{pid}", (sx1, sy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

			# resize pitch to camera height
			target_h = scaled_h
			target_w = int(pitch_img.shape[1] * (target_h / pitch_img.shape[0]))
			pitch_small = cv2.resize(pitch_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

			combined = np.hstack([camera_small, pitch_small])

			cv2.imshow("Pitch Tracking (25 % camera + vertical top-view)", combined)
			if cv2.waitKey(1) == ord("q"):
				break
		cv2.destroyAllWindows()

# ─── Optional static debug view ───────────────────────────────────────
def visualize_basic_pitch(pitch2d: Pitch2d) -> None:
	cv2.imshow("Homography Debug View", create_static_pitch_image(pitch2d))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# ─── Entry-point ──────────────────────────────────────────────────────
def main() -> None:
	pitch_2d_corners_4x2 = np.stack([
		far_left_corner_3d[:2],
		far_right_corner_3d[:2],
		net_right_bottom_3d[:2],
		net_left_bottom_3d[:2],
	])
	img_2d_corners_4x2 = np.stack([
		far_left_corner,
		far_right_corner,
		net_right_bottom,
		net_left_bottom,
	])

	visualize_pitch_basic	= True
	visualize_2d_tracking	= False

	if visualize_pitch_basic:
		pitch2d = Pitch2d(
			pitch_2d_corners_4x2 = pitch_2d_corners_4x2,
			img_2d_corners_4x2	 = img_2d_corners_4x2,
			orientation			 = "vertical",
		)
		visualize_basic_pitch(pitch2d)

	if visualize_2d_tracking:
		video_path = raw_data_path / "game1_3.mp4"
		tracked_df = pd.read_csv(tracked_detections_csv_path)	# frame_ind,x1,y1,x2,y2,player_id,is_valid
		PitchTrackerVisualizer(
			video_path,
			tracked_df,
			pitch_2d_corners_4x2 = pitch_2d_corners_4x2,
			img_2d_corners_4x2	 = img_2d_corners_4x2,
		).show_video_with_pitch_overlay()

if __name__ == "__main__":
	main()
