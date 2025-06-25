import os
import cv2
import pandas as pd

import validate_rally_data_config as config

VIDEO_EXTENSIONS = [".mp4", ".MP4", ".avi", ".mkv"]
def _validate_order(group, starts, ends, video_name, label):
	starts_values = group[starts].values
	ends_values = group[ends].values

	bad_idx_start = (starts_values[1:] < starts_values[:-1]).nonzero()[0]
	bad_idx_end = (ends_values[1:] < ends_values[:-1]).nonzero()[0]

	if len(bad_idx_start) > 0:
		print(f"{label} START ORDER ERROR in video: {video_name} ({len(bad_idx_start)} issues)")
		for i in bad_idx_start:
			print(f"    Row {i}:   start={starts_values[i]:.2f}  -> next start={starts_values[i+1]:.2f}")

	if len(bad_idx_end) > 0:
		print(f"{label} END ORDER ERROR in video: {video_name} ({len(bad_idx_end)} issues)")
		for i in bad_idx_end:
			print(f"    Row {i}:   end={ends_values[i]:.2f}  -> next end={ends_values[i+1]:.2f}")

	if len(bad_idx_start) > 0 or len(bad_idx_end) > 0:
		assert False, f"{label} ORDER ERROR in video: {video_name} (start: {len(bad_idx_start)}, end: {len(bad_idx_end)})"

def load_annotations():
	print(f"Loading: {config.RALLIES_CSV}, {config.HITS_CSV}, {config.HIT_ASSIGNMENTS_XLSX}")

	rallies_df = pd.read_csv(config.RALLIES_CSV)
	hits_df = pd.read_csv(config.HITS_CSV)
	hit_assignments_df = pd.read_excel(config.HIT_ASSIGNMENTS_XLSX)

	# Sort hits_df by filename and start
	hits_df = hits_df.sort_values(
		by=["filename", "start"],
		ascending=[True, True]
	).reset_index(drop=True)

	# Validate hits_df
	for video_name, group in hits_df.groupby("filename"):
		_validate_order(group, "start", "end", video_name, "HIT")

	# Sort and validate hit_assignments_df
	hit_assignments_df = hit_assignments_df.sort_values(
		by=["video", "timestamp"],
		ascending=[True, True]
	).reset_index(drop=True)

	for video_name, group in hit_assignments_df.groupby("video"):
		_validate_order(group, "timestamp", "timestamp", video_name, "ASSIGNMENT")

	print(f"Loaded {len(rallies_df)} rallies, {len(hits_df)} hits, {len(hit_assignments_df)} assignments")

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


def process_video(videos_df: pd.DataFrame,
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
	video_metadata = videos_df.iloc[0]
	video_path = video_metadata["video_path"]
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
	video_name_with_ext = video_metadata["video_name_with_ext"]
	video_name_no_ext = video_metadata["video_name"]

	hits_df = hits_df[hits_df["filename"] == video_name_with_ext].copy()
	hit_assignments_df = hit_assignments_df[hit_assignments_df["video"] == video_name_no_ext].copy()

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

	hit_assignments_df["timestamp_fixed"] = "00:" + hit_assignments_df["timestamp"]
	hit_assignments_df["timestamp_sec"] = pd.to_timedelta(hit_assignments_df["timestamp_fixed"]).dt.total_seconds()

	hit_assignments_df["frame_num"] = (hit_assignments_df["timestamp_sec"] * fps).round().astype(int)
	frame_tolerance = int(round(fps * 0.5))

	frame_to_player = hit_assignments_df.set_index("frame_num")["player"]

	# assign player to each hit — based on closest assignment within tolerance
	hits_df["assigned_player"] = "unknown"

	for hit_index, (idx, row) in enumerate(hits_df.iterrows()):
		hit_center_frame = (row["start_frame"] + row["end_frame"]) // 2

		# compute distances to all hit assignments
		distances = {
			f: abs(hit_center_frame - f)
			for f in frame_to_player.index
			# if abs(hit_center_frame - f) <= frame_tolerance
		}

		if len(distances) == 0:
			raise ValueError(f"Unknown not allowed for this test, {video_metadata['video_name']}")
			player_value = "unknown"
		elif len(distances) == 1:
			closest_f = min(distances, key=distances.get)
			player_value = frame_to_player[closest_f]
		else:
			# multiple hit assignments within tolerance window
			closest_dist = min(distances.values())
			frames_closest = [
				f for f, dist in distances.items() if dist == closest_dist
			]

			players_found = {frame_to_player[f] for f in frames_closest}

			if len(players_found) == 1:
				player_value = players_found.pop()
			else:
				raise RuntimeError(
					f"Hit [{idx}] center={hit_center_frame} overlaps multiple players: {players_found}"
				)

		hits_df.at[idx, "assigned_player"] = player_value

	# ────────────────── 4.  Frame-by-frame rendering ────────────────────────
	n_hits = len(hits_df)
	frame_idx = 0
	timestamp_sec = 0.0

	current_hit_idx = 0
	current_hit = hits_df.iloc[current_hit_idx]

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		# advance pointer once we pass the current hit’s window
		while True:
			is_no_more_hits = current_hit is None
			is_current_frame_in_hit = not is_no_more_hits and frame_idx <= current_hit["end_frame"]
			is_last_hit = current_hit_idx + 1 >= len(hits_df)

			if is_no_more_hits or is_current_frame_in_hit:
				break

			if is_last_hit:
				current_hit = None
				break

			# safe to advance
			next_hit_idx = current_hit_idx + 1
			next_hit = hits_df.iloc[next_hit_idx]

			current_hit_idx = next_hit_idx
			current_hit = next_hit

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
			draw_hit_marker(frame, 1000, 500, (0, 0, 255), "HIT")
			draw_hit_marker(frame, 1200, 400, (255, 255, 0), f"HitInd {current_hit_idx}")
			draw_hit_marker(frame, 1200, 300, (0, 255, 255), f"{start_frame}: {end_frame}")
			if player_lbl != "unknown":
				draw_hit_marker(frame, 1200, 500, (0, 255, 0), f"Player {player_lbl}")
			else:
				pass

		writer.write(frame)
		frame_idx += 1
		timestamp_sec += 1.0 / fps

	# ────────────────── 5.  Cleanup ────────────────────────────────────────
	cap.release()
	writer.release()
	print(f"Saved {output_path}, #marked hits: {'x'}/{n_hits}")




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
			videos_df.iloc[[idx]],
			hits_in_video,
			assignments_in_video,
			output_path
		)


if __name__ == "__main__":
	main()
