#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

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
)

# ─── Display constants ────────────────────────────────────────────────
image_scale_factor = 0.25  # show camera view at 25 %
pitch_canvas_base_size = (
	600,
	300,
)  # (width, height) before scaling to match image

# ─── Player colours ───────────────────────────────────────────────────
player_colors_bgr = [
	(
		255,
		0,
		0,
	),
	# Player 0 – Red
	(
		0,
		255,
		0,
	),
	# Player 1 – Green
	(
		0,
		0,
		255,
	),
	# Player 2 – Blue
	(
		255,
		255,
		0,
	),
	# Player 3 – Cyan
]
player_color_array = np.array(
	player_colors_bgr,
	dtype=np.uint8,
)


# ─── Affine-only homography (pitch↔image) ─────────────────────────────
class PitchHomography:
	def __init__(
			self,
			pitch_2d_corners_4x2,
			img_2d_corners_4x2,
	):
		self.affine_2x3, _ = cv2.estimateAffine2D(
			pitch_2d_corners_4x2,
			img_2d_corners_4x2,
			method=cv2.RANSAC,
			ransacReprojThreshold=3.0,
		)

		inv_affine_3x3 = np.eye(
			N=3,
			dtype=np.float32,
		)
		inv_affine_3x3[
		:2,
		:3] = self.affine_2x3
		self.inv_affine_3x3 = np.linalg.inv(
			inv_affine_3x3,
		)

	def image_to_pitch_batch(
			self,
			img_xy_batch: np.ndarray,
	) -> np.ndarray:
		n = \
			img_xy_batch.shape[
				0]
		homo = np.hstack(
			[
				img_xy_batch,
				np.ones(
					(
						n,
						1,
					),
					np.float32,
				),
			],
		)
		pitch = (
				self.inv_affine_3x3 @ homo.T).T
		return pitch[
		:,
		:2] / pitch[
		:,
		2:3]  # (n, 2) in metres

	def pitch_to_image_batch(
			self,
			pitch_xy_batch: np.ndarray,
	) -> np.ndarray:
		n = \
			pitch_xy_batch.shape[
				0]
		homo = np.hstack(
			[
				pitch_xy_batch,
				np.ones(
					(
						n,
						1,
					),
					np.float32,
				),
			],
		)
		img = (
				self.affine_2x3 @ homo.T).T
		return img.astype(
			np.float32,
		)  # (n, 2) pixels


# ─── 2-D top-view pitch canvas ────────────────────────────────────────
class Pitch2d:
	def __init__(self, length_m=20.0, width_m=10.0, canvas_size_px=pitch_canvas_base_size):
		self.len_m, self.wid_m = length_m, width_m
		self.base_w_px, self.base_h_px = canvas_size_px

	def blank_canvas(self) -> np.ndarray:
		img = np.zeros((self.base_h_px, self.base_w_px, 3), np.uint8)
		cv2.rectangle(img, (0, 0), (self.base_w_px - 1, self.base_h_px - 1), (255, 255, 255), 2)
		return img

	def pitch_to_canvas(self, pitch_xy_m: tuple[float, float]) -> tuple[int, int]:
		x, y = pitch_xy_m
		cx = int((x / self.len_m) * self.base_w_px)
		cy = int((1.0 - y / self.wid_m) * self.base_h_px)
		return cx, cy


class PitchTrackerVisualizer:
	def __init__(self, video_path: Path, detections_dataframe: pd.DataFrame, homography: PitchHomography):
		self.video_path = video_path
		self.detections_df = detections_dataframe
		self.homography = homography
		self.pitch_converter = Pitch2d()
		self.base_canvas = self.pitch_converter.blank_canvas()
		self.video_reader = VideoReader(video_path)

	def draw_pitch_players(self, pitch_canvas: np.ndarray, players_dict: dict[int, tuple[float, float]]) -> np.ndarray:
		out = pitch_canvas.copy()
		for pid, (px, py) in players_dict.items():
			if np.isnan(px):
				continue
			cx, cy = self.pitch_converter.pitch_to_canvas((px, py))
			color = player_color_array[pid % len(player_color_array)].tolist()
			cv2.circle(out, (cx, cy), 6, color, -1)
			cv2.putText(
				out, f"P{pid} ({px:.1f},{py:.1f})", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
				cv2.LINE_AA,
			)
		return out

	def show_video_with_pitch_overlay(self) -> None:
		for frame_idx, full_frame in self.video_reader.video_frames_generator():
			ind_detections = self.detections_df[self.detections_df.frame_ind == frame_idx]
			if ind_detections.empty:
				continue

			detections_x = ((ind_detections.x1 + ind_detections.x2) / 2).to_numpy(np.float32)
			detections_y = ind_detections.y2.to_numpy(np.float32)
			detections_xy = np.stack([detections_x, detections_y], axis=1)
			pitch_xy = self.homography.image_to_pitch_batch(detections_xy)
			player_ids = ind_detections.player_id.to_numpy(np.int32)
			valid_mask = ind_detections.is_valid.to_numpy(bool)

			player_pitch_dict = {
				int(pid): (float(px), float(py))
				for pid, (px, py), v in zip(player_ids, pitch_xy, valid_mask) if v
			}

			scaled_h, scaled_w = int(full_frame.shape[0] * image_scale_factor), int(
				full_frame.shape[1] * image_scale_factor,
			)
			scaled_frame = cv2.resize(full_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

			for pid, (x1, y1, x2, y2), v in zip(player_ids, ind_detections[["x1", "y1", "x2", "y2"]].values, valid_mask):
				if not v:
					continue
				col = player_color_array[pid % len(player_color_array)].tolist()
				sx1, sy1, sx2, sy2 = [int(c * image_scale_factor) for c in (x1, y1, x2, y2)]
				cv2.rectangle(scaled_frame, (sx1, sy1), (sx2, sy2), col, 2)
				cv2.putText(scaled_frame, f"P{pid}", (sx1, sy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

			pitch_image = self.draw_pitch_players(self.base_canvas, player_pitch_dict)
			target_pitch_h = scaled_h
			target_pitch_w = int(pitch_image.shape[1] * (target_pitch_h / pitch_image.shape[0]))
			pitch_resized = cv2.resize(pitch_image, (target_pitch_w, target_pitch_h), interpolation=cv2.INTER_LINEAR)
			combined = np.hstack([scaled_frame, pitch_resized])

			cv2.imshow("Pitch Tracking (25 % camera + top view)", combined)
			if cv2.waitKey(1) == ord("q"):
				break

		cv2.destroyAllWindows()


# ─── Entry point ──────────────────────────────────────────────────────
def main() -> None:
	tacked_detections_df = pd.read_csv(
		tracked_detections_csv_path,
	)  # frame_ind,x1,y1,x2,y2,player_id,is_valid
	pitch_homography = PitchHomography(
		pitch_2d_corners_4x2=np.stack(
			[
				far_left_corner_3d[
				:2],
				far_right_corner_3d[
				:2],
				net_right_bottom_3d[
				:2],
				net_left_bottom_3d[
				:2],
			],
		),
		img_2d_corners_4x2=np.stack(
			[
				far_left_corner,
				far_right_corner,
				net_right_bottom,
				net_left_bottom,
			],
		),
	)
	video_path = raw_data_path / "game1_3.mp4"
	visualizer = PitchTrackerVisualizer(
		video_path,
		tacked_detections_df,
		pitch_homography,
	)
	visualizer.show_video_with_pitch_overlay()


if __name__ == "__main__":
	main()
