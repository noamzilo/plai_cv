import pandas as pd
from pathlib import Path
from collections import defaultdict

from utils.paths import raw_data_path, detections_csv_path
from acquisition.VideoReader import VideoReader
from detection.detection_utils import detect_players, load_yolo_detector

def find_detections_on_video():
	video_path = raw_data_path / "game1_3.mp4"
	print(f"started detecting on video {video_path}")
	video_reader = VideoReader(video_path=video_path)
	frames_generator = video_reader.video_frames_generator(start_frame=0, interval=1, end_frame=-1)
	yolo_detector = load_yolo_detector()

	detections = []

	for i_frame, frame in enumerate(frames_generator):
		print(f"processing frame #{i_frame}")
		players_bboxes = detect_players(frame, yolo_detector)
		for (x1, y1), (x2, y2) in players_bboxes:
			detections.append([i_frame, x1, y1, x2, y2])

	return detections

def save_detections(detections):
	df = pd.DataFrame(detections, columns=["frame_ind", "x1", "y1", "x2", "y2"])
	df.to_csv(detections_csv_path, index=False)
	print(f"Saved to {detections_csv_path}")


if __name__ == "__main__":
	detections = find_detections_on_video
	save_detections(detections)
