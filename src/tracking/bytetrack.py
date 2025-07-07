# type: ignore
"""
Player tracking pipeline using YOLOv8 for detection and ByteTrack (via deep_sort_realtime) for tracking.
Assigns players to 'far' and 'close' teams (2 per side) per frame, with robust ID assignment.
Caches intermediates to disk.
"""

from pathlib import Path
import pandas as pd
from src.detection.PlayerDetection import PlayerDetection
from src.tracking.PlayerTracker import PlayerTracker
from src.tracking.TrackingVisualizer import TrackingVisualizer
from src.utils.paths import project_root
# --- CONFIG ---
VIDEO_PATH = Path(project_root / "data/proprietary/001.mp4")
CACHE_DIR = Path(project_root / "cache/bytetrack_cache")
CACHE_DIR.mkdir(exist_ok=True)
DETECTIONS_CSV = CACHE_DIR / "detections.csv"
TRACKING_CSV = CACHE_DIR / "player_positions.csv"
VISUALIZATION_VIDEO = CACHE_DIR / "tracking_overlay.mp4"


def main():
	# Step 1: Detect players (cache to CSV)
	detector = PlayerDetection()
	if DETECTIONS_CSV.is_file():
		detections_df = pd.read_csv(DETECTIONS_CSV)
	else:
		detections_df = detector.detect_and_save(VIDEO_PATH, DETECTIONS_CSV)

	# Step 2: Track players and assign teams/IDs
	net_left: Tuple[float, float] = (73, 482) # hard coded for current video
	net_right: Tuple[float, float] = (1154, 490) # hard coded for current video
	tracker = PlayerTracker(net_left=net_left, net_right=net_right)
	tracking_df = tracker.run_tracking(VIDEO_PATH, detections_df)
	tracking_df.to_csv(TRACKING_CSV, index=False)
	print(f"Saved player positions to {TRACKING_CSV}")

	# Step 3: Visualize tracking results
	visualizer = TrackingVisualizer(VIDEO_PATH, tracking_df)
	visualizer.visualize_and_save(VISUALIZATION_VIDEO)

if __name__ == "__main__":
	main()