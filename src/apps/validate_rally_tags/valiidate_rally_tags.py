import os
import cv2
import pandas as pd

import validate_rally_data_config as config

VIDEO_EXTENSIONS = [".mp4", ".MP4", ".avi", ".mkv"]


def load_csv_data():
	print(f"Loading: {config.RALLIES_CSV}, {config.HITS_CSV}, {config.HIT_ASSIGNMENTS_XLSX}")

	rallies_df = pd.read_csv(config.RALLIES_CSV)
	hits_df = pd.read_csv(config.HITS_CSV)
	hit_assignments_df = pd.read_excel(config.HIT_ASSIGNMENTS_XLSX)

	return rallies_df, hits_df, hit_assignments_df


def draw_hit_marker(frame, x, y, color, label):
	cv2.circle(frame, (x, y), 20, color, 3)
	cv2.putText(frame, label, (x + 25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def process_video(video_path, rally_info, hits_df, hit_assignments_df, output_path):
	print(f"Processing {video_path}")

	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

	start_frame = rally_info['start_frame']
	end_frame = rally_info['end_frame']

	frame_idx = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		if frame_idx < start_frame:
			frame_idx += 1
			continue
		if frame_idx > end_frame:
			break

		cv2.putText(frame, f"Frame {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

		hits_in_frame = hits_df[hits_df['frame'] == frame_idx]
		for _, hit_row in hits_in_frame.iterrows():
			draw_hit_marker(frame, 100, 100, (0, 0, 255), "HIT")

		assignments_in_frame = hit_assignments_df[hit_assignments_df['frame'] == frame_idx]
		for _, assign_row in assignments_in_frame.iterrows():
			draw_hit_marker(frame, 200, 200, (0, 255, 0), f"Player {assign_row['player']}")

		writer.write(frame)
		frame_idx += 1

	cap.release()
	writer.release()
	print(f"Saved {output_path}")


def find_video_file(rally_video_name):
	for ext in VIDEO_EXTENSIONS:
		candidate = os.path.join(config.VIDEO_DIR, rally_video_name + ext)
		if os.path.isfile(candidate):
			return candidate
	return None


def main():
	rallies_df, hits_df, hit_assignments_df = load_csv_data()

	for idx, rally_row in rallies_df.iterrows():
		video_name = rally_row['video_name'] if 'video_name' in rally_row else rally_row['filename']
		video_path = find_video_file(video_name)
		if video_path is None:
			print(f"[WARNING] Could not find video for {video_name}")
			continue

		output_path = os.path.join(config.OUTPUT_DIR, f"{video_name}_validated.mp4")

		process_video(
			video_path,
			rally_row,
			hits_df[hits_df['rally_id'] == rally_row['rally_id']] if 'rally_id' in hits_df else hits_df,
			hit_assignments_df[hit_assignments_df['rally_id'] == rally_row['rally_id']] if 'rally_id' in hit_assignments_df else hit_assignments_df,
			output_path
		)


if __name__ == "__main__":
	main()
