import os
import cv2
import pandas as pd

import validate_rally_data_config as config

VIDEO_EXTENSIONS = [".mp4", ".MP4", ".avi", ".mkv"]


def load_annotations():
	print(f"Loading: {config.RALLIES_CSV}, {config.HITS_CSV}, {config.HIT_ASSIGNMENTS_XLSX}")

	rallies_df = pd.read_csv(config.RALLIES_CSV)
	hits_df = pd.read_csv(config.HITS_CSV)
	hit_assignments_df = pd.read_excel(config.HIT_ASSIGNMENTS_XLSX)

	return rallies_df, hits_df, hit_assignments_df

def load_video_metadata(video_dir):
	video_records = []

	for filename in os.listdir(video_dir):
		if not any(filename.endswith(ext) for ext in VIDEO_EXTENSIONS):
			continue

		video_path = os.path.join(video_dir, filename)

		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		cap.release()

		video_name = os.path.splitext(filename)[0]
		video_extension = os.path.splitext(filename)[1]

		video_records.append({
			"video_name": video_name,
			"video_extension": video_extension,
			"video_name_with_ext": f"{video_name}{video_extension}",
			"video_path": video_path,
			"fps": fps,
			"total_frames": total_frames,
			"width": width,
			"height": height
		})

	video_df = pd.DataFrame(video_records)
	print(f"Found {len(video_df)} videos in {video_dir}")
	return video_df


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

	start_frame = rally_info['start']
	end_frame = rally_info['end']

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

def extract_uncut_from_cut_filename(video_name, ext):
	return "_".join(video_name.split("_")[:-1]) + ext


def main():
	rallies_df, hits_df, hit_assignments_df = load_annotations()
	videos_df = load_video_metadata(config.VIDEO_DIR)

	for idx, video_row in videos_df.iterrows():
		video_path = video_row['video_path']
		video_name = video_row['video_name']
		video_name_with_ext = video_row['video_name_with_ext']
		ext = video_row['video_extension']
		# Extract rally_id from filename (assuming it is embedded)
		# Example: "rally_01234_part1.mp4"
		rally_id = extract_uncut_from_cut_filename(video_name_with_ext, ext)

		rally_row = rallies_df[rallies_df['filename'] == rally_id]
		if rally_row.empty:
			print(f"[WARNING] No rally row found for {video_name_with_ext}")
			continue

		hits_in_video = hits_df[hits_df['filename'] == video_name_with_ext]
		assignments_in_video = hit_assignments_df[hit_assignments_df['video'] == video_name]

		output_path = os.path.join(config.OUTPUT_DIR, f"{video_name_with_ext}_validated.mp4")

		process_video(
			video_path,
			rally_row.iloc[0],
			hits_in_video,
			assignments_in_video,
			output_path
		)


if __name__ == "__main__":
	main()
