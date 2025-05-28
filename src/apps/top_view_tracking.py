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

# ─── Project-specific paths ──────────────────────────────────────────
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

# ─── Display constants ───────────────────────────────────────────────
image_scale_factor			= 0.25
pitch_inner_size_px			= (600, 300)		# (w, h) of playable area (before buffer swap)
buffer_px					= 100

# ─── Colours ─────────────────────────────────────────────────────────
player_colors_bgr = [
	(255,   0,   0),
	(  0, 255,   0),
	(  0,   0, 255),
	(255, 255,   0),
]
player_color_array = np.array(player_colors_bgr, dtype=np.uint8)

# ─── 2-D pitch helper ────────────────────────────────────────────────
class Pitch2d:
	def __init__(
			self,
			length_m: float							= 20.0,
			width_m: float							= 10.0,
			canvas_inner_px: tuple[int, int]		= pitch_inner_size_px,
			pitch_2d_corners_4x2: np.ndarray | None	= None,
			img_2d_corners_4x2: np.ndarray | None	= None,
			orientation: str							= "vertical",
	):
		self.len_m, self.wid_m	= length_m, width_m
		self.orientation		= orientation.lower()
		if self.orientation not in ("horizontal", "vertical"):
			raise ValueError("orientation must be 'horizontal' or 'vertical'.")
		inner_w_px, inner_h_px	= (
			canvas_inner_px if self.orientation == "horizontal" else canvas_inner_px[::-1]
		)
		self.inner_w_px, self.inner_h_px	= inner_w_px, inner_h_px
		self.canvas_w_px						= inner_w_px + 2 * buffer_px
		self.canvas_h_px						= inner_h_px + 2 * buffer_px
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
		else:
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

	def draw_polyline(self, canvas: np.ndarray, pitch_xy_seq, col, t=2) -> np.ndarray:
		if len(pitch_xy_seq) < 2:
			return canvas
		out = canvas.copy()
		canvas_pts = [self.pitch_to_canvas(pt) for pt in pitch_xy_seq]
		for p0, p1 in zip(canvas_pts[:-1], canvas_pts[1:]):
			cv2.line(out, p0, p1, col, t)
		return out

	def image_to_pitch_batch(self, img_xy_batch: np.ndarray) -> np.ndarray:
		return self._apply_homography(img_xy_batch, self.inv_homography_3x3)

	def _apply_homography(self, xy_batch: np.ndarray, H: np.ndarray) -> np.ndarray:
		if H is None:
			raise ValueError("Pitch homography not initialised.")
		if len(xy_batch.shape) == 1:
			xy_batch = xy_batch[np.newaxis, :]
		homo = np.hstack([xy_batch, np.ones((xy_batch.shape[0], 1), np.float32)])
		xp = (H @ homo.T).T
		return xp[:, :2] / xp[:, 2:3]

# ─── Pre-compute static pitch canvas ──────────────────────────────────
def build_static_pitch_canvas(pitch2d: Pitch2d) -> np.ndarray:
	named_img_pts = OrderedDict([
		("deadzone0",			np.array([0,   1055])),
		("deadzone1",			np.array([0,   2159])),
		("deadzone2",			np.array([3839,2159])),
		("deadzone3",			np.array([3839,1174])),
		("far_left_corner",		far_left_corner),
		("far_right_corner",	far_right_corner),
		("net_right_bottom",	net_right_bottom),
		("net_left_bottom",		net_left_bottom),
	])
	img_pts		= np.array(list(named_img_pts.values()), np.float32)
	pitch_pts	= pitch2d.image_to_pitch_batch(img_pts)

	canvas = pitch2d.blank_canvas()

	# outline polyline
	poly_seq = [
		"far_left_corner", "far_right_corner", "net_right_bottom",
		"deadzone3", "deadzone2", "deadzone1", "deadzone0",
		"net_left_bottom", "far_left_corner",
	]
	poly_pts = [pitch_pts[list(named_img_pts.keys()).index(n)] for n in poly_seq]
	canvas  = pitch2d.draw_polyline(canvas, poly_pts, (255, 255, 255), 2)

	# net line
	net_pts = [(pitch2d.len_m/2, 0), (pitch2d.len_m/2, pitch2d.wid_m)]
	canvas  = pitch2d.draw_polyline(canvas, net_pts, (255, 255, 255), 2)

	return canvas

# ─── STEP-1 convert and dump to pitch-coords CSV ──────────────────────
def convert_tracked_to_pitch(src_csv: Path, dst_csv: Path, pitch: Pitch2d) -> None:

	df = pd.read_csv(src_csv)
	centre_img = np.stack([
		((df.x1 + df.x2) / 2).to_numpy(np.float32),
		df.y2.to_numpy(np.float32),
	], axis=1)
	centre_pitch = pitch.image_to_pitch_batch(centre_img)
	df["p_x"] = centre_pitch[:, 0]
	df["p_y"] = centre_pitch[:, 1]

	# corners
	c1_img	= df[["x1", "y1"]].to_numpy(np.float32)
	c2_img	= df[["x2", "y2"]].to_numpy(np.float32)
	c1_pt	= pitch.image_to_pitch_batch(c1_img)
	c2_pt	= pitch.image_to_pitch_batch(c2_img)
	df[["p_x1", "p_y1"]] = c1_pt
	df[["p_x2", "p_y2"]] = c2_pt

	df.to_csv(dst_csv, index=False)
	print(f"Pitch-converted detections written → {dst_csv}")

# ─── STEP-2 visualiser (reads pitch CSV) ──────────────────────────────
class PitchTrackerVisualizer:
	def __init__(self, video_path: Path, detections_df: pd.DataFrame, pitch2d: Pitch2d, static_canvas: np.ndarray):
		self.df				= detections_df
		self.pitch			= pitch2d
		self.static_canvas	= static_canvas
		self.reader			= VideoReader(video_path)

	def show(self) -> None:
		for frame_idx, frame_bgr in self.reader.video_frames_generator():
			fd = self.df[self.df.frame_ind == frame_idx]
			if fd.empty:
				continue
			player_ids	= fd.player_id.to_numpy(np.int32)
			valid		= fd.is_valid.to_numpy(bool)

			pitch_xy   = fd[["p_x", "p_y"]].to_numpy(np.float32)

			# build player point specs
			pp = []
			for pid, (px, py), v in zip(player_ids, pitch_xy, valid):
				if not v:
					continue
				col = player_color_array[pid % len(player_color_array)].tolist()
				pp.append(((float(px), float(py)), col, f"P{pid}"))

			pitch_img = self.pitch.draw_points(self.static_canvas.copy(), pp, 0.5)

			# detection boxes on camera frame
			cam_small = cv2.resize(
				frame_bgr,
				(int(frame_bgr.shape[1]*image_scale_factor), int(frame_bgr.shape[0]*image_scale_factor)),
				interpolation = cv2.INTER_LINEAR,
			)
			for pid, (x1,y1,x2,y2), v in zip(player_ids, fd[["x1","y1","x2","y2"]].values, valid):
				if not v:
					continue
				col = player_color_array[pid % len(player_color_array)].tolist()
				sx1, sy1, sx2, sy2 = [int(c*image_scale_factor) for c in (x1,y1,x2,y2)]
				cv2.rectangle(cam_small,(sx1,sy1),(sx2,sy2),col,2)
				cv2.putText(cam_small,f"P{pid}",(sx1,sy1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2,cv2.LINE_AA)

			# resize pitch to cam height
			tgt_h = cam_small.shape[0]
			tgt_w = int(pitch_img.shape[1]*(tgt_h/pitch_img.shape[0]))
			pitch_small = cv2.resize(pitch_img,(tgt_w,tgt_h),interpolation=cv2.INTER_LINEAR)

			cv2.imshow("Tracking (25 % cam + vertical top-view)", np.hstack([cam_small, pitch_small]))
			if cv2.waitKey(1) == ord("q"):
				break
		cv2.destroyAllWindows()

# ─── STEP-3 per-player heatmaps ───────────────────────────────────────
def generate_heatmaps(pitch_df: pd.DataFrame, pitch2d: Pitch2d, static_canvas: np.ndarray) -> None:
	h, w = pitch2d.canvas_h_px, pitch2d.canvas_w_px
	heatmaps = {pid: np.zeros((h, w), np.float32) for pid in range(4)}

	for pid, sub in pitch_df[pitch_df.is_valid.astype(bool)].groupby("player_id"):
		sub_sorted = sub.sort_values("frame_ind")
		if len(sub_sorted) < 2:
			continue
		px = sub_sorted.p_x.to_numpy()
		py = sub_sorted.p_y.to_numpy()
		for (x0,y0),(x1,y1) in zip(zip(px[:-1],py[:-1]), zip(px[1:],py[1:])):
			d = np.hypot(x1-x0, y1-y0)
			if d == 0:		# avoid ∞
				continue
			intensity = 1.0 / d
			c0 = pitch2d.pitch_to_canvas((x0,y0))
			c1 = pitch2d.pitch_to_canvas((x1,y1))
			cv2.line(heatmaps[pid], c0, c1, intensity, 1)

	# blur-normalise-colorise each
	coloured_maps = []
	for pid in range(4):
		hm = cv2.GaussianBlur(heatmaps[pid], (0,0), 15)
		cv2.normalize(hm, hm, 0, 255, cv2.NORM_MINMAX)
		hm_uint8 = hm.astype(np.uint8)
		color_hm = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
		# overlay on static canvas for context
		overlay = cv2.addWeighted(static_canvas, 0.3, color_hm, 0.7, 0)
		coloured_maps.append(overlay)

	combined = np.hstack(coloured_maps)
	cv2.imshow("Per-player Heatmaps (0-3 L→R)", combined)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# ─── MAIN ─────────────────────────────────────────────────────────────
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

	# Initialise pitch helper
	pitch2d = Pitch2d(
		pitch_2d_corners_4x2 = pitch_2d_corners_4x2,
		img_2d_corners_4x2	 = img_2d_corners_4x2,
		orientation			 = "vertical",
	)
	static_canvas = build_static_pitch_canvas(pitch2d)

	# Derived CSV path
	pitch_csv_path = tracked_detections_csv_path.with_stem(
		tracked_detections_csv_path.stem + "_pitch"
	)

	run_convert   = True
	run_visualise = True
	run_heatmap   = True

	if run_convert:
		convert_tracked_to_pitch(tracked_detections_csv_path, pitch_csv_path, pitch2d)

	if run_visualise:
		pitch_df = pd.read_csv(pitch_csv_path)
		vis = PitchTrackerVisualizer(
			raw_data_path / "game1_3.mp4",
			pitch_df,
			pitch2d,
			static_canvas,
		)
		vis.show()

	if run_heatmap:
		pitch_df = pd.read_csv(pitch_csv_path)
		generate_heatmaps(pitch_df, pitch2d, static_canvas)

if __name__ == "__main__":
	main()
