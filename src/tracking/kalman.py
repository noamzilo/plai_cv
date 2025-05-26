#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

import pandas as pd, numpy as np
from pathlib import Path
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque
import matplotlib.pyplot as plt
import cv2

# ─── Paths ─────────────────────────────────────────────────────────────
from utils.paths import detections_csv_path, tracked_detections_csv_path

# ─── Net reference points (constants) ──────────────────────────────────
net_left_bottom		= np.array([840, 705])
net_right_bottom	= np.array([2955, 751])

def side_sign(pt_xy: np.ndarray) -> float:
	"""Return sign (+ / -) indicating which half-court the point is on."""
	x1, y1 = net_left_bottom
	x2, y2 = net_right_bottom
	x,  y  = pt_xy
	return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

# ─── Kalman Filter Wrapper ─────────────────────────────────────────────
class Tracker:
	def __init__(self, id_):
		self.id = id_
		self.kf = KalmanFilter(dim_x=8, dim_z=4)
		self.kf.F = np.eye(8)
		for i in range(4):
			self.kf.F[i, i + 4] = 1
		self.kf.H = np.eye(4, 8)
		self.kf.R *= 10
		self.kf.P *= 100
		self.kf.Q *= 0.01
		self.age = 0
		self.time_since_update = 0
		self.history = deque(maxlen=10)

	def update(self, bbox):
		self.kf.update(bbox)
		self.time_since_update = 0

	def predict(self):
		self.kf.predict()
		self.age += 1
		self.time_since_update += 1
		return self.kf.x[:4]

	def get_state(self):
		return self.kf.x[:4].ravel()

# ─── IOU + side-aware Hungarian matcher ────────────────────────────────
def iou(bb1, bb2):
	xx1, yy1 = np.maximum(bb1[:2], bb2[:2])
	xx2, yy2 = np.minimum(bb1[2:], bb2[2:])
	w, h = np.maximum(0, [xx2-xx1, yy2-yy1])
	inter = w*h
	union = ((bb1[2]-bb1[0])*(bb1[3]-bb1[1]) +
			 (bb2[2]-bb2[0])*(bb2[3]-bb2[1]) - inter)
	return inter/union if union>0 else 0. # no iou from opposite sides of net

def assign_detections_to_trackers(trackers, detections, iou_thr=0.1):
	if not trackers:
		return np.empty((0,2),int), np.arange(len(detections)), []

	M = np.zeros((len(detections), len(trackers)), np.float32)
	for d_i, det in enumerate(detections):
		d_side = side_sign([(det[0]+det[2])/2, (det[1]+det[3])/2])
		for t_i, trk in enumerate(trackers):
			tc = trk.get_state()
			t_side = side_sign([(tc[0]+tc[2])/2, (tc[1]+tc[3])/2])
			if d_side * t_side > 0:				# same half-court
				M[d_i, t_i] = iou(det, tc)
			else:
				M[d_i, t_i] = 0.

	matched = np.array(linear_sum_assignment(-M)).T
	unmatched_dets = [i for i in range(len(detections)) if i not in matched[:,0]]
	unmatched_trks = [i for i in range(len(trackers))   if i not in matched[:,1]]
	good_matches = []
	for d, t in matched:
		if M[d, t] >= iou_thr:	good_matches.append([d,t])
		else:					unmatched_dets.append(d); unmatched_trks.append(t)
	return np.array(good_matches), np.array(unmatched_dets), np.array(unmatched_trks)

# ─── Plotting (8 PNGs) ─────────────────────────────────────────────────
def plot_player_locations(df):
	for pid in range(4):
		pdf = df[(df.player_id==pid) & (df.is_valid)]
		if pdf.empty: continue
		centers = pd.DataFrame({
			"frame": pdf.frame_ind,
			"x": (pdf.x1+pdf.x2)/2,
			"y": (pdf.y1+pdf.y2)/2
		})
		for coord in ("x","y"):
			plt.figure(figsize=(12,8))
			plt.plot(centers.frame, centers[coord])
			plt.xlabel("Frame"); plt.ylabel(f"{coord}_center")
			plt.title(f"Player {pid} – {coord}_center")
			plt.grid(True); plt.tight_layout()
			out = tracked_detections_csv_path.with_suffix(f".player{pid}_{coord}.png")
			plt.savefig(out); plt.close()
			print(f"[INFO] saved {out}")

# ─── Tracking orchestration ────────────────────────────────────────────
def create_tracking_df(detections_df):
	n_frames = int(detections_df.frame_ind.max()) + 1
	trackers, next_id, rows = [], 0, []
	for i_frame in range(n_frames):
		if i_frame % 100 == 0:
			print(f"processing frame #{i_frame}")
		detections = detections_df[detections_df.frame_ind == i_frame][["x1", "y1", "x2", "y2"]].values.tolist()
		for t in trackers: t.predict()

		matches, um_d, um_t = assign_detections_to_trackers(trackers, detections)
		for d,t in matches: trackers[t].update(detections[d])
		for d in um_d:
			trk = Tracker(next_id); trk.update(detections[d]); trackers.append(trk); next_id+=1
		trackers = [t for t in trackers if t.time_since_update<=10]

		slots = sorted(trackers, key=lambda x:x.id)[:4]
		while len(slots)<4:
			dummy = Tracker(-1); dummy.kf.x[:4]=np.nan; slots.append(dummy)

		for slot,trk in enumerate(slots):
			x1,y1,x2,y2 = trk.get_state()
			valid = not np.isnan(x1)
			rows.append([i_frame, x1, y1, x2, y2, slot, valid])
	return pd.DataFrame(rows, columns=["frame_ind","x1","y1","x2","y2","player_id","is_valid"])

def draw_tracking_overlay(frame, tracks_frame):
	colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
	for _, row in tracks_frame.iterrows():
		if not row["is_valid"]:
			continue
		x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
		pid = int(row.player_id)
		cv2.rectangle(frame, (x1,y1), (x2,y2), colors[pid%len(colors)], 2)
		cv2.putText(frame, f"Player {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[pid%len(colors)], 2)
	return frame

def view_tracking_on_video(tracks_df: pd.DataFrame, video_path: Path):
	from acquisition.VideoReader import VideoReader
	video_reader = VideoReader(video_path)

	for frame_ind, frame in video_reader.video_frames_generator(start_frame=0):
		frame_tracks = tracks_df[tracks_df.frame_ind == frame_ind]
		overlay = draw_tracking_overlay(frame.copy(), frame_tracks)
		cv2.imshow("Tracked Detections", overlay)
		key = cv2.waitKey(0)
		if key == ord('q'):
			break
	cv2.destroyAllWindows()

# ─── Main ──────────────────────────────────────────────────────────────
def main():
	is_tracking = False
	is_viewing = True
	if is_tracking:
		detections = pd.read_csv(detections_csv_path)
		tracks = create_tracking_df(detections)
		tracks.to_csv(tracked_detections_csv_path,index=False)
		print(f"[INFO] CSV saved → {tracked_detections_csv_path}")
	else:
		tracks = pd.read_csv(tracked_detections_csv_path)
	plot_player_locations(tracks)

	if is_viewing:
		from utils.paths import raw_data_path
		video_path = raw_data_path / "game1_3.mp4"
		view_tracking_on_video(tracks, video_path)

if __name__ == "__main__":
	main()
