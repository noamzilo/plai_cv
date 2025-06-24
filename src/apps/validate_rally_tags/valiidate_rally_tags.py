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

import os
import cv2
import pandas as pd


import os
import cv2
import pandas as pd


def process_video(video_path: str,
				  hits_df: pd.DataFrame,
				  hit_assignments_df: pd.DataFrame,
				  output_path: str) -> None:
	"""
	Render `video_path` with visual markers for every hit and (if available)
	the player assignment.
	✓	uses *only* tabs for indentation
	✓	no lambdas / inner-defs
	✓	handles per-video filtering and “unknown” assignments gracefully
	"""

	# ────────────────── 1.  Video I/O setup ─────────────────────────────────
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open {video_path}")

	fps = cap.get(cv2.CAP_PROP_FPS)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	writer = cv2.VideoWriter(
		output_path,
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(width, height)
	)

	# ────────────────── 2.  Filter rows for this video ──────────────────────
	video_name = os.path.basename(video_path)

	hits_df = hits_df[hits_df["filename"] == video_name].copy()
	hit_assignments_df = hit_assignments_df[hit_assignments_df["video"] == video_name].copy()

	# if there are no hits at all, just copy the video and return
	if hits_df.empty:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			writer.write(frame)
		cap.release()
		writer.release()
		print(f"Saved {output_path} (no hits)")
		return

	# ────────────────── 3.  Pre-compute helper columns ──────────────────────
	hits_df["start_frame"] = (hits_df["start"] * fps).round().astype(int)
	hits_df["end_frame"] = (hits_df["end"] * fps).round().astype(int)
	hits_df["assigned_player"] = "unknown"

	hit_assignments_df["timestamp_sec"] = (
		pd.to_timedelta(hit_assignments_df["timestamp"]).dt.total_seconds()
	)
	hit_assignments_df["frame_num"] = (hit_assignments_df["timestamp_sec"] * fps).round().astype(int)
	frame_to_player = hit_assignments_df.set_index("frame_num")["player"]

	# ── assign a player (if any) to every hit row
	for idx, row in hits_df.iterrows():
		player_value = "unknown"
		for f in range(row["start_frame"], row["end_frame"] + 1):
			if f in frame_to_player:
				player_value = frame_to_player[f]
				break
		hits_df.at[idx, "assigned_player"] = player_value

	# ────────────────── 4.  Frame-by-frame rendering ────────────────────────
	frame_idx = 0
	timestamp_sec = 0.0

	current_hit_idx = 0
	current_hit = hits_df.iloc[current_hit_idx]

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		# advance pointer once we pass the current hit’s window
		while frame_idx > current_hit["end_frame"]:
			current_hit_idx += 1
			if current_hit_idx >= len(hits_df):
				current_hit = None
				break
			current_hit = hits_df.iloc[current_hit_idx]

		# ─────────── on-screen diagnostics (frame & time) ────────────────
		cv2.putText(
			frame,
			f"Frame {frame_idx}",
			(40, 40),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.0,
			(255, 255, 255),
			2
		)
		cv2.putText(
			frame,
			f"t {timestamp_sec:.3f}s",
			(40, 80),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(255, 255, 255),
			2
		)

		# ─────────── hit-specific overlays ────────────────────────────────
		has_current_hit = current_hit is not None
		if has_current_hit:
			start_frame = current_hit["start_frame"]
			end_frame = current_hit["end_frame"]
			player_lbl = current_hit["assigned_player"]
		else:
			start_frame = end_frame = player_lbl = None

		if has_current_hit and start_frame <= frame_idx <= end_frame:
			draw_hit_marker(frame, 100, 100, (0, 0, 255), "HIT")
			if player_lbl != "unknown":
				draw_hit_marker(frame, 200, 200, (0, 255, 0), f"Player {player_lbl}")

		writer.write(frame)
		frame_idx += 1
		timestamp_sec += 1.0 / fps

	# ────────────────── 5.  Cleanup ────────────────────────────────────────
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
	rallies_df = rallies_df.reset_index(drop=True)
	hits_df = hits_df.reset_index(drop=True)
	hit_assignments_df = hit_assignments_df.reset_index(drop=True)
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

		if len(assignments_in_video) == 0:
			continue
		output_path = os.path.join(config.OUTPUT_DIR, f"{video_name_with_ext}_validated.mp4")

		process_video(
			video_path,
			video_row,
			hits_in_video,
			assignments_in_video,
			output_path
		)


if __name__ == "__main__":
	main()
