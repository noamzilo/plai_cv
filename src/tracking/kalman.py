#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly
#	⭾	FULL MEANINGFUL VARIABLE NAMES – no shortcuts, no single-letter names

import pandas as pandas_lib, numpy as numpy_lib
from pathlib import Path
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque
import matplotlib.pyplot as pyplot_lib
import cv2

# ─── Paths (project specific) ──────────────────────────────────────────
from utils.paths import detections_csv_path, tracked_detections_csv_path, raw_data_path

# ─── Constants ─────────────────────────────────────────────────────────
MAX_CENTER_DISTANCE_THRESHOLD	= 250		# distance gate for matching
IOU_SCORE_THRESHOLD				= 0.30		# minimum IOU score for a valid match
MAX_TRACKER_AGE_FRAMES			= 20		# how many missed frames before a tracker is discarded

# Net (paddle-ball) reference points, already measured in the image
net_left_bottom_point	= numpy_lib.array([840, 705])
net_right_bottom_point	= numpy_lib.array([2955, 751])

# ─── Geometry helpers ──────────────────────────────────────────────────
def court_side_sign(point_xy: tuple[float, float]) -> int:
	"""
	Return +1 for the right half-court, -1 for the left half-court, 0 if exactly on the net.
	"""
	x1, y1 = net_left_bottom_point
	x2, y2 = net_right_bottom_point
	x_coord, y_coord = point_xy
	determinant_value = (x2 - x1) * (y_coord - y1) - (y2 - y1) * (x_coord - x1)
	return 1 if determinant_value > 0 else -1 if determinant_value < 0 else 0

def intersection_over_union(bbox_a: numpy_lib.ndarray, bbox_b: numpy_lib.ndarray) -> float:
	"""
	Compute intersection-over-union of two bounding boxes.
	Each bbox is [x1, y1, x2, y2] in absolute pixel coordinates.
	"""
	xx1, yy1 = numpy_lib.maximum(bbox_a[:2], bbox_b[:2])
	xx2, yy2 = numpy_lib.minimum(bbox_a[2:], bbox_b[2:])
	intersection_width, intersection_height = numpy_lib.maximum(0, [xx2 - xx1, yy2 - yy1])
	intersection_area = intersection_width * intersection_height
	union_area = (
		(bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]) +
		(bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]) -
		intersection_area
	)
	return intersection_area / union_area if union_area > 0 else 0.0

# ─── Kalman-based tracker ──────────────────────────────────────────────
class PlayerTracker:
	def __init__(self, slot_identifier: int, initial_detection: numpy_lib.ndarray):
		self.slot_identifier			= slot_identifier				# 0,1 top; 2,3 bottom
		self.court_side				= -1 if slot_identifier < 2 else 1
		self.missed_frame_counter	= 0
		self.kalman_filter			= KalmanFilter(dim_x=8, dim_z=4)
		self.kalman_filter.F		= numpy_lib.eye(8)
		for index in range(4):
			self.kalman_filter.F[index, index + 4] = 1
		self.kalman_filter.H		= numpy_lib.eye(4, 8)
		self.kalman_filter.R		*= 10
		self.kalman_filter.P		*= 100
		self.kalman_filter.Q		*= 0.01
		self.kalman_filter.x[:4]	= initial_detection.reshape(-1, 1)
	def update(self, detection_bbox: numpy_lib.ndarray) -> None:
		self.kalman_filter.update(detection_bbox)
		self.missed_frame_counter = 0
	def predict(self) -> None:
		self.kalman_filter.predict()
		self.missed_frame_counter += 1
	def is_expired(self) -> bool:
		return self.missed_frame_counter > MAX_TRACKER_AGE_FRAMES
	def current_state_bbox(self) -> numpy_lib.ndarray:
		return self.kalman_filter.x[:4].ravel()

# ─── Matching logic (Hungarian with side + distance gating) ────────────
def match_detections_to_trackers(
	existing_trackers: list[PlayerTracker],
	current_detections: list[numpy_lib.ndarray]
):
	if not existing_trackers:
		return (
			numpy_lib.empty((0, 2), dtype=int),
			numpy_lib.arange(len(current_detections)),
			numpy_lib.empty(0, dtype=int)
		)

	cost_matrix = numpy_lib.zeros((len(current_detections), len(existing_trackers)), numpy_lib.float32)

	for detection_index, detection_bbox in enumerate(current_detections):
		detection_center_x	= (detection_bbox[0] + detection_bbox[2]) / 2
		detection_center_y	= (detection_bbox[1] + detection_bbox[3]) / 2
		for tracker_index, tracker in enumerate(existing_trackers):
			if tracker.court_side != court_side_sign((detection_center_x, detection_center_y)):
				# Detection is on the wrong side for this tracker
				continue
			tracker_bbox = tracker.current_state_bbox()
			tracker_center_x	= (tracker_bbox[0] + tracker_bbox[2]) / 2
			tracker_center_y	= (tracker_bbox[1] + tracker_bbox[3]) / 2
			center_distance = numpy_lib.hypot(
				detection_center_x - tracker_center_x,
				detection_center_y - tracker_center_y
			)
			if center_distance > MAX_CENTER_DISTANCE_THRESHOLD:
				# Distance gate ➜ skip unlikely matches
				continue
			cost_matrix[detection_index, tracker_index] = intersection_over_union(
				detection_bbox, tracker_bbox
			)

	assignment_pairs = numpy_lib.array(linear_sum_assignment(-cost_matrix)).T

	valid_matches, unmatched_detection_indices, unmatched_tracker_indices = [], [], []
	for detection_index, tracker_index in assignment_pairs:
		if cost_matrix[detection_index, tracker_index] >= IOU_SCORE_THRESHOLD:
			valid_matches.append([detection_index, tracker_index])
		else:
			unmatched_detection_indices.append(detection_index)
			unmatched_tracker_indices.append(tracker_index)

	for detection_index in range(len(current_detections)):
		if detection_index not in assignment_pairs[:, 0]:
			unmatched_detection_indices.append(detection_index)
	for tracker_index in range(len(existing_trackers)):
		if tracker_index not in assignment_pairs[:, 1]:
			unmatched_tracker_indices.append(tracker_index)

	return (
		numpy_lib.array(valid_matches),
		numpy_lib.array(unmatched_detection_indices),
		numpy_lib.array(unmatched_tracker_indices)
	)

# ─── Slot management helpers ───────────────────────────────────────────
def free_slot_identifiers(active_trackers: list[PlayerTracker], desired_side: int) -> list[int]:
	available_slots = [0, 1] if desired_side == -1 else [2, 3]
	occupied_slots = [tracker.slot_identifier for tracker in active_trackers if tracker.court_side == desired_side]
	return [identifier for identifier in available_slots if identifier not in occupied_slots]

# ─── Drawing helpers (re-usable) ───────────────────────────────────────
def draw_bounding_boxes(
	image_frame: numpy_lib.ndarray,
	rows_dataframe: pandas_lib.DataFrame,
	color_bgr: tuple[int, int, int] = (128, 128, 128),
	label_prefix: str = "Det"
) -> numpy_lib.ndarray:
	for row_index, row_data in rows_dataframe.iterrows():
		x1, y1, x2, y2 = map(int, row_data[["x1", "y1", "x2", "y2"]])
		cv2.rectangle(image_frame, (x1, y1), (x2, y2), color_bgr, 2)
		cv2.putText(
			image_frame, f"{label_prefix} {row_index}", (x1, y1 - 8),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1
		)
	return image_frame

def draw_tracker_overlay(
	image_frame: numpy_lib.ndarray,
	tracker_rows_dataframe: pandas_lib.DataFrame,
	resize_scale: float = 1.0
) -> numpy_lib.ndarray:
	slot_colors_bgr = [
		(255, 0,   0),		# Player 0 – Red
		(0,   255, 0),		# Player 1 – Green
		(0,   0,   255),	# Player 2 – Blue
		(255, 255, 0)		# Player 3 – Cyan
	]

	if resize_scale != 1.0:
		image_frame = cv2.resize(image_frame, (0, 0), fx=resize_scale, fy=resize_scale)

	for _, row_data in tracker_rows_dataframe.iterrows():
		if not bool(row_data["is_valid"]):
			continue
		x1 = int(row_data.x1 * resize_scale)
		y1 = int(row_data.y1 * resize_scale)
		x2 = int(row_data.x2 * resize_scale)
		y2 = int(row_data.y2 * resize_scale)
		player_identifier = int(row_data.player_id)
		color_bgr = slot_colors_bgr[player_identifier % len(slot_colors_bgr)]
		cv2.rectangle(image_frame, (x1, y1), (x2, y2), color_bgr, 2)
		cv2.putText(
			image_frame, f"Player {player_identifier}", (x1, y1 - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2
		)
	return image_frame

# ─── Visualization utilities ───────────────────────────────────────────
def preview_detections_only(video_file_path: Path, detections_dataframe: pandas_lib.DataFrame) -> None:
	from acquisition.VideoReader import VideoReader
	video_reader = VideoReader(video_file_path)
	for frame_index, image_frame in video_reader.video_frames_generator():
		detections_for_frame = detections_dataframe[detections_dataframe.frame_ind == frame_index]
		image_with_boxes = draw_bounding_boxes(image_frame.copy(), detections_for_frame)
		cv2.imshow("Raw detections preview", image_with_boxes)
		if cv2.waitKey(1) == ord("q"):
			break
	cv2.destroyAllWindows()

def preview_tracking_overlay(video_file_path: Path, tracking_dataframe: pandas_lib.DataFrame) -> None:
	from acquisition.VideoReader import VideoReader
	video_reader = VideoReader(video_file_path)
	for frame_index, image_frame in video_reader.video_frames_generator():
		tracking_rows = tracking_dataframe[tracking_dataframe.frame_ind == frame_index]
		image_with_overlay = draw_tracker_overlay(image_frame.copy(), tracking_rows, resize_scale=0.25)
		cv2.imshow("Kalman tracking", image_with_overlay)
		if cv2.waitKey(1) == ord("q"):
			break
	cv2.destroyAllWindows()

# ─── Plotting (x and y center for each player) ────────────────────────
def plot_player_center_positions(tracking_dataframe: pandas_lib.DataFrame) -> None:
	for player_identifier in range(4):
		player_rows = tracking_dataframe[
			(tracking_dataframe.player_id == player_identifier) &
			(tracking_dataframe.is_valid)
		]
		if player_rows.empty:
			continue
		center_dataframe = pandas_lib.DataFrame({
			"frame_index": player_rows.frame_ind,
			"x_center": (player_rows.x1 + player_rows.x2) / 2,
			"y_center": (player_rows.y1 + player_rows.y2) / 2
		})
		for coordinate_name in ("x_center", "y_center"):
			pyplot_lib.figure(figsize=(12, 8))
			pyplot_lib.plot(center_dataframe.frame_index, center_dataframe[coordinate_name], alpha=0.8)
			pyplot_lib.xlabel("Frame index")
			pyplot_lib.ylabel(coordinate_name)
			pyplot_lib.title(f"Player {player_identifier} – {coordinate_name}")
			pyplot_lib.grid(True)
			pyplot_lib.tight_layout()
			output_png_path = tracked_detections_csv_path.with_suffix(
				f".player{player_identifier}_{coordinate_name}.png"
			)
			pyplot_lib.savefig(output_png_path)
			pyplot_lib.close()
			print(f"[INFO] saved → {output_png_path}")

def update_trackers_for_side(
	trackers_list: list[PlayerTracker],
	detections_list: list[numpy_lib.ndarray],
	slot_offset: int
) -> list[PlayerTracker]:
	for tracker in trackers_list:
		tracker.predict()

	match_pairs, unmatched_detections, _ = match_detections_to_trackers(trackers_list, detections_list)

	for detection_index, tracker_index in match_pairs:
		trackers_list[tracker_index].update(detections_list[detection_index])

	side = -1 if slot_offset == 0 else 1
	for detection_index in unmatched_detections:
		available_slots = free_slot_identifiers(trackers_list, side)
		if available_slots:
			new_tracker = PlayerTracker(available_slots[0], detections_list[detection_index])
			trackers_list.append(new_tracker)

	return [t for t in trackers_list if not t.is_expired()]

def snapshot_trackers_for_side(
	trackers_list: list[PlayerTracker],
	current_frame_index: int,
	expected_slots: list[int]
) -> list[list]:
	tracker_by_slot = {t.slot_identifier: t for t in trackers_list}
	rows = []

	for slot_id in expected_slots:
		tracker = tracker_by_slot.get(slot_id)
		if tracker is None:
			dummy_tracker = PlayerTracker(slot_id, numpy_lib.array([numpy_lib.nan] * 4))
			dummy_tracker.kalman_filter.x[:4] = numpy_lib.nan
			tracker = dummy_tracker
		x1, y1, x2, y2 = tracker.current_state_bbox()
		is_valid = not numpy_lib.isnan(x1)
		rows.append([current_frame_index, x1, y1, x2, y2, slot_id, is_valid])

	return rows

def create_tracking_dataframe(
	detections_dataframe: pandas_lib.DataFrame,
	start_frame_index: int = 0,
	end_frame_index_exclusive: int | None = None,
	frame_interval: int = 1
) -> pandas_lib.DataFrame:
	absolute_last_frame_index = int(detections_dataframe.frame_ind.max()) + 1
	if end_frame_index_exclusive is None:
		end_frame_index_exclusive = absolute_last_frame_index

	total_frames_to_process = (
		(end_frame_index_exclusive - start_frame_index + frame_interval - 1)
		// frame_interval
	)

	top_trackers_list: list[PlayerTracker] = []
	bottom_trackers_list: list[PlayerTracker] = []
	output_rows = []

	for processed_frame_counter, current_frame_index in enumerate(
		range(start_frame_index, end_frame_index_exclusive, frame_interval)
	):
		if processed_frame_counter % 100 == 0:
			print(f"[INFO] processing frame #{current_frame_index} ({processed_frame_counter + 1}/{total_frames_to_process})")

		detections_for_frame = detections_dataframe[
			detections_dataframe.frame_ind == current_frame_index
		][["x1", "y1", "x2", "y2"]].values

		top_detections, bottom_detections = [], []
		for bbox in detections_for_frame:
			center_x = (bbox[0] + bbox[2]) / 2
			bottom_y = bbox[3]
			side = court_side_sign((center_x, bottom_y))
			(top_detections if side <= 0 else bottom_detections).append(bbox)

		top_trackers_list = update_trackers_for_side(top_trackers_list, top_detections, slot_offset=0)
		bottom_trackers_list = update_trackers_for_side(bottom_trackers_list, bottom_detections, slot_offset=2)

		output_rows.extend(snapshot_trackers_for_side(top_trackers_list, current_frame_index, [0, 1]))
		output_rows.extend(snapshot_trackers_for_side(bottom_trackers_list, current_frame_index, [2, 3]))

	return pandas_lib.DataFrame(output_rows, columns=[
		"frame_ind", "x1", "y1", "x2", "y2", "player_id", "is_valid"
	])


# ─── Main entry point ──────────────────────────────────────────────────
def main() -> None:
	VISUALIZE_DETECTIONS_ONLY	= False
	GENERATE_NEW_TRACKING_CSV	= True
	is_plot_player_center_positions = False
	VISUALIZE_TRACKING_OVERLAY	= True

	if VISUALIZE_DETECTIONS_ONLY:
		detections_dataframe = pandas_lib.read_csv(detections_csv_path)
		preview_detections_only(raw_data_path / "game1_3.mp4", detections_dataframe)
		return

	if GENERATE_NEW_TRACKING_CSV:
		detections_dataframe = pandas_lib.read_csv(detections_csv_path)
		tracking_dataframe = create_tracking_dataframe(detections_dataframe)
		tracking_dataframe.to_csv(tracked_detections_csv_path, index=False)
		print(f"[INFO] tracking CSV saved → {tracked_detections_csv_path}")
	else:
		tracking_dataframe = pandas_lib.read_csv(tracked_detections_csv_path)

	if is_plot_player_center_positions:
		plot_player_center_positions(tracking_dataframe)

	if VISUALIZE_TRACKING_OVERLAY:
		preview_tracking_overlay(raw_data_path / "game1_3.mp4", tracking_dataframe)

if __name__ == "__main__":
	main()
