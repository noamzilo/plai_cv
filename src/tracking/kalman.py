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
			self.kalman_filter.F[i, i+4] = 1
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
		return self.kalman_filter.x[:4]

# ─── Hungarian Matcher ─────────────────────────────────────────────────
def iou(bb_test, bb_gt):
	xx1 = max(bb_test[0], bb_gt[0])
	yy1 = max(bb_test[1], bb_gt[1])
	xx2 = min(bb_test[2], bb_gt[2])
	yy2 = min(bb_test[3], bb_gt[3])
	w = max(0., xx2 - xx1)
	h = max(0., yy2 - yy1)
	intersection = w * h
	area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
	area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
	union = area1 + area2 - intersection
	return intersection / union if union > 0 else 0

def assign_detections_to_trackers(trackers, detections, iou_threshold=0.1):
	if len(trackers) == 0:
		return np.empty((0,2), dtype=int), np.arange(len(detections)), []
	iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
	for i_det, det in enumerate(detections):
		for i_tracker, tracker in enumerate(trackers):
			iou_matrix[i_det, i_tracker] = iou(det, tracker.get_state())

	matched_indices = linear_sum_assignment(-iou_matrix)
	matched_indices = np.array(matched_indices).T

	unmatched_detections = []
	for i_det in range(len(detections)):
		if i_det not in matched_indices[:, 0]:
			unmatched_detections.append(i_det)

	unmatched_trackers = []
	for i_tracker in range(len(trackers)):
		if i_tracker not in matched_indices[:, 1]:
			unmatched_trackers.append(i_tracker)

	matches = []
	for m in matched_indices:
		if iou_matrix[m[0], m[1]] < iou_threshold:
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else:
			matches.append(m.reshape(1, 2))
	if matches:
		matches = np.concatenate(matches, axis=0)
	else:
		matches = np.empty((0, 2), dtype=int)
	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def plot_player_locations():
	fig, ax = plt.subplots()
	for player_id in range(4):
		player_df = df_out[df_out["player_id"] == str(player_id)]
		ax.plot(player_df["x1"], player_df["y1"], label=f"Player {player_id}")

	ax.set_xlabel("x1 (left)")
	ax.set_ylabel("y1 (top)")
	ax.set_title("Player Locations Over Time")
	ax.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(tracked_detections_csv_path.with_suffix(".png"))
	plt.close()
	print(f"Saved player trajectory plot to {tracked_detections_csv_path.with_suffix('.png')}")

def main():
		# ─── Main Tracking Loop ────────────────────────────────────────────────
	df = pd.read_csv(detections_csv_path)
	all_frames = df["frame_ind"].max() + 1

	trackers = []
	next_id = 0
	output_rows = []

	for frame_ind in range(all_frames):
		frame_detections = df[df["frame_ind"] == frame_ind][["x1", "y1", "x2", "y2"]].values.tolist()

		for tracker in trackers:
			tracker.predict()

		matches, unmatched_detection_inds, unmatched_trks = assign_detections_to_trackers(trackers, frame_detections)

		# update matched trackers
		for match in matches:
			tracker = trackers[match[1]]
			tracker.update(frame_detections[match[0]])

		# create new trackers
		for unmatched_detection_ind in unmatched_detection_inds:
			tracker = Tracker(next_id)
			tracker.update(frame_detections[unmatched_detection_ind])
			trackers.append(tracker)
			next_id += 1

		# Keep only recent trackers
		trackers = [t for t in trackers if t.time_since_update <= 10]

		# ─── Store 4 tracks per frame ─────
		sorted_trackers = sorted(trackers, key=lambda t: t.id)[:4]
		while len(sorted_trackers) < 4:
			# Add dummy trackers if needed
			dummy = Tracker(-1)
			dummy.kalman_filter.x[:4] = np.array([0, 0, 0, 0])[:, np.newaxis]
			sorted_trackers.append(dummy)

		for sorted_tracker_ind, sorted_tracker in enumerate(sorted_trackers[:4]):
			x1, y1, x2, y2 = sorted_tracker.get_state()
			player_id = f"{sorted_tracker_ind}"
			output_rows.append([frame_ind, int(x1), int(y1), int(x2), int(y2), player_id])

	# ─── Dump Final Output ─────────────────────────────────────────────────
	df_out = pd.DataFrame(output_rows, columns=["frame_ind", "x1", "y1", "x2", "y2", "player_id"])
	df_out.to_csv(tracked_detections_csv_path, index=False)
	print(f"Saved tracked player data to {tracked_detections_csv_path}")


if __name__ == "__main__":
	main()