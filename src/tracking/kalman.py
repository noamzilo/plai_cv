#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

import pandas as pd
import numpy as np
from pathlib import Path
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque
import matplotlib.pyplot as plt

# ─── Paths ─────────────────────────────────────────────────────────────
from utils.paths import detections_csv_path, tracked_detections_csv_path

# ─── Kalman Filter Wrapper ─────────────────────────────────────────────
class Tracker:
	def __init__(self, id_):
		self.id = id_
		self.kalman_filter = KalmanFilter(dim_x=8, dim_z=4)
		self.kalman_filter.F = np.eye(8)
		for i in range(4):
			self.kalman_filter.F[i, i + 4] = 1
		self.kalman_filter.H = np.eye(4, 8)
		self.kalman_filter.R *= 10
		self.kalman_filter.P *= 100
		self.kalman_filter.Q *= 0.01
		self.age = 0
		self.time_since_update = 0
		self.history = deque(maxlen=10)
		self.hit_streak = 0
		self.hit = False

	def update(self, bbox):
		x1, y1, x2, y2 = bbox
		self.kalman_filter.update([x1, y1, x2, y2])
		self.history.append(self.kalman_filter.x[:4])
		self.hit = True
		self.hit_streak += 1
		self.time_since_update = 0

	def predict(self):
		self.kalman_filter.predict()
		self.age += 1
		self.time_since_update += 1
		self.history.append(self.kalman_filter.x[:4])
		self.hit = False
		return self.kalman_filter.x[:4]

	def get_state(self):
		# always return a flat (x1, y1, x2, y2) array
		return self.kalman_filter.x[:4].ravel()

# ─── Hungarian Matcher ─────────────────────────────────────────────────
def iou(bb_test, bb_gt):
	xx1 = max(bb_test[0], bb_gt[0])
	yy1 = max(bb_test[1], bb_gt[1])
	xx2 = min(bb_test[2], bb_gt[2])
	yy2 = min(bb_test[3], bb_gt[3])
	w = max(0.0, xx2 - xx1)
	h = max(0.0, yy2 - yy1)
	intersection = w * h
	area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
	area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
	union = area1 + area2 - intersection
	return intersection / union if union > 0 else 0.0

def assign_detections_to_trackers(trackers, detections, iou_threshold=0.1):
	if len(trackers) == 0:
		return np.empty((0, 2), dtype=int), np.arange(len(detections)), []

	iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
	for i_det, det in enumerate(detections):
		for i_trk, trk in enumerate(trackers):
			iou_matrix[i_det, i_trk] = iou(det, trk.get_state().tolist())

	matched_indices = np.array(linear_sum_assignment(-iou_matrix)).T

	unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
	unmatched_trks = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

	matches = []
	for det_idx, trk_idx in matched_indices:
		if iou_matrix[det_idx, trk_idx] < iou_threshold:
			unmatched_dets.append(det_idx)
			unmatched_trks.append(trk_idx)
		else:
			matches.append([det_idx, trk_idx])

	return np.array(matches), np.array(unmatched_dets), np.array(unmatched_trks)

# ─── Plotting ──────────────────────────────────────────────────────────
def plot_player_locations(tracking_df):
	fig, ax = plt.subplots()
	for player_id in range(4):
		player_df = tracking_df[tracking_df["player_id"] == str(player_id)]
		ax.plot(player_df["frame_ind"], player_df["x1"], label=f"Player {player_id} - x1")
		ax.plot(player_df["frame_ind"], player_df["y1"], label=f"Player {player_id} - y1", linestyle="--")

	ax.set_xlabel("Frame Index")
	ax.set_ylabel("Position")
	ax.set_title("x1/y1 Position Over Time per Player")
	ax.legend()
	ax.grid(True)
	plt.tight_layout()
	out_png = tracked_detections_csv_path.with_suffix(".xy1.png")
	plt.savefig(out_png)
	plt.close()
	print(f"[INFO] x1/y1 plot saved → {out_png}")


def create_tracking_over_detections_df(detections_df):
	n_frames = int(detections_df["frame_ind"].max()) + 1
	print(f"[INFO] {len(detections_df)} detections across {n_frames} frames")

	trackers = []
	next_id = 0
	output_rows = []

	for frame_idx in range(n_frames):
		frame_dets = detections_df[detections_df["frame_ind"] == frame_idx][["x1", "y1", "x2", "y2"]].values.tolist()

		# ── Predict step ──
		for trk in trackers:
			trk.predict()

		# ── Associate ──
		matches, unmatched_dets, unmatched_trks = assign_detections_to_trackers(trackers, frame_dets)
		if frame_idx % 100 == 0:
			print(f"[FRAME {frame_idx:>5}/{n_frames}] {len(frame_dets)} detections")
			print(f"Matched: {len(matches)} | New dets: {len(unmatched_dets)} | Lost: {len(unmatched_trks)}")

		# ── Update matched trackers ──
		for det_idx, trk_idx in matches:
			trackers[trk_idx].update(frame_dets[det_idx])

		# ── Create new trackers for unmatched detections ──
		for det_idx in unmatched_dets:
			trk = Tracker(next_id)
			trk.update(frame_dets[det_idx])
			trackers.append(trk)
			print(f" [NEW] Tracker {next_id}")
			next_id += 1

		# ── Prune dead trackers ──
		trackers = [t for t in trackers if t.time_since_update <= 10]

		# ── Keep exactly 4 trackers (pads with dummies) ──
		sorted_trks = sorted(trackers, key=lambda t: t.id)[:4]
		while len(sorted_trks) < 4:
			dummy = Tracker(-1)
			dummy.kalman_filter.x[:4] = np.zeros((4, 1))
			sorted_trks.append(dummy)

		for slot_idx, trk in enumerate(sorted_trks[:4]):
			x1, y1, x2, y2 = trk.get_state()
			output_rows.append([frame_idx, int(x1), int(y1), int(x2), int(y2), slot_idx])

	tracking_df = pd.DataFrame(output_rows, columns=["frame_ind", "x1", "y1", "x2", "y2", "player_id"])
	return tracking_df

# ─── Main ──────────────────────────────────────────────────────────────
def main():
	print("[INFO] Loading detections CSV …")
	do_tracking = False
	save_tracking = not do_tracking
	if do_tracking:
		detections_df = pd.read_csv(detections_csv_path)
		tracking_df = create_tracking_over_detections_df(detections_df)
	else:
		tracking_df = pd.read_csv(tracked_detections_csv_path)

	if save_tracking:
		print("[INFO] Saving final tracked data …")
		tracking_df.to_csv(tracked_detections_csv_path, index=False)
		print(f"[INFO] CSV saved → {tracked_detections_csv_path}")

	plot_player_locations(tracking_df)

if __name__ == "__main__":
	main()
