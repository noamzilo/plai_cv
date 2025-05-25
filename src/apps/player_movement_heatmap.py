import pandas as pd
from pathlib import Path
from collections import defaultdict

from utils.paths import raw_data_path, calculated_data_path
from acquisition.VideoReader import VideoReader
from detection.detection_utils import detect_players, load_yolo_detector

def main():
	video_path = raw_data_path / "game1_3.mp4"
	video_reader = VideoReader(video_path=video_path)
	frames_generator = video_reader.video_frames_generator(start_frame=0, interval=10, end_frame=1000)
	yolo_detector = load_yolo_detector()

	records = []

	for i_frame, frame in enumerate(frames_generator):
		players_bboxes = detect_players(frame, yolo_detector)
		for (x1, y1), (x2, y2) in players_bboxes:
			records.append([i_frame, x1, y1, x2, y2])

	df = pd.DataFrame(records, columns=["frame_ind", "x1", "y1", "x2", "y2"])
	output_csv_path = calculated_data_path / "player_bboxes.csv"
	df.to_csv(output_csv_path, index=False)
	print(f"Saved to {output_csv_path}")

if __name__ == "__main__":
	main()
