import pandas as pd
import numpy as np
from pathlib import Path
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque

# ─── Paths ─────────────────────────────────────────────────────────────
from utils.paths import calculated_data_path
bboxes_csv = calculated_data_path / "player_bboxes.csv"
output_csv = calculated_data_path / "tracked_players.csv"

# ─── Kalman Filter Wrapper ─────────────────────────────────────────────
class Tracker:
	def __init__(self, id):
		self.id = id
		self.kf = KalmanFilter(dim_x=8, dim_z=4)
		self.kf.F = np.eye(8)
		for i in range(4):
			self.kf.F[i, i+4] = 1
		self.kf.H = np.eye(4, 8)
		self.kf.R *= 10
		self.kf.P *= 100
		self.kf.Q *= 0.01
		self.age = 0
		self.time_since_update = 0
		self.history = deque(maxlen=10)
		self.hit_streak = 0
		self.hit = False

	def update(self, bbox):
		x1, y1, x2, y2 = bbox
		self.kf.update([x1, y1, x2, y2])
		self.history.append(self.kf.x[:4])
		self.hit = True
		self.hit_streak += 1
		self.time_since_update = 0

	def predict(self):
		self.kf.predict()
		self.age += 1
		self.time_since_update += 1
		self.history.append(self.kf.x[:4])
		self.hit = False
		return self.kf.x[:4]

	def get_state(self):
		return self.kf.x[:4]

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
	for d, det in enumerate(detections):
		for t, trk in enumerate(trackers):
			iou_matrix[d, t] = iou(det, trk.get_state())

	matched_indices = linear_sum_assignment(-iou_matrix)
	matched_indices = np.array(matched_indices).T

	unmatched_detections = []
	for d in range(len(detections)):
		if d not in matched_indices[:, 0]:
			unmatched_detections.append(d)

	unmatched_trackers = []
	for t in range(len(trackers)):
		if t not in matched_indices[:, 1]:
			unmatched_trackers.append(t)

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

# ─── Main Tracking Loop ────────────────────────────────────────────────
df = pd.read_csv(bboxes_csv)
all_frames = df["frame_ind"].max() + 1

trackers = []
next_id = 0
output_rows = []

for frame_ind in range(all_frames):
	frame_dets = df[df["frame_ind"] == frame_ind][["x1", "y1", "x2", "y2"]].values.tolist()

	for t in trackers:
		t.predict()

	matches, unmatched_dets, unmatched_trks = assign_detections_to_trackers(trackers, frame_dets)

	# update matched trackers
	for m in matches:
		trk = trackers[m[1]]
		trk.update(frame_dets[m[0]])

	# create new trackers
	for i in unmatched_dets:
		t = Tracker(next_id)
		t.update(frame_dets[i])
		trackers.append(t)
		next_id += 1

	# Keep only recent trackers
	trackers = [t for t in trackers if t.time_since_update <= 10]

	# ─── Store 4 tracks per frame ─────
	sorted_trackers = sorted(trackers, key=lambda t: t.id)[:4]
	while len(sorted_trackers) < 4:
		# Add dummy trackers if needed
		dummy = Tracker(-1)
		dummy.kf.x[:4] = [0, 0, 0, 0]
		sorted_trackers.append(dummy)

	for i, trk in enumerate(sorted_trackers[:4]):
		x1, y1, x2, y2 = trk.get_state()
		player_id = f"player_{i}"
		output_rows.append([frame_ind, int(x1), int(y1), int(x2), int(y2), player_id])

# ─── Dump Final Output ─────────────────────────────────────────────────
df_out = pd.DataFrame(output_rows, columns=["frame_ind", "x1", "y1", "x2", "y2", "player_id"])
df_out.to_csv(output_csv, index=False)
print(f"Saved tracked player data to {output_csv}")
