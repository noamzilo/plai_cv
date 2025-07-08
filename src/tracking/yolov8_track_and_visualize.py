import pandas as pd
from pathlib import Path
from src.detection.PlayerDetection import PlayerDetection
from src.tracking.TrackingVisualizer import TrackingVisualizer
from src.utils.paths import project_root, test_video_name, output_path
from src.tracking.PlayerTracker import PlayerTracker
from src.acquisition.VideoReader import VideoReader
from typing import Tuple
# --- CONFIG ---
output_path.mkdir(exist_ok=True)
VIDEO_PATH = Path(project_root / f"data/proprietary/{test_video_name}")
CACHE_DIR = Path(project_root / "cache/yolov8_tracking_cache")
CACHE_DIR.mkdir(exist_ok=True)
tracking_dir = output_path / "tracking"
tracking_dir.mkdir(exist_ok=True)
TRACKING_CSV = tracking_dir / f"player_tracks_{test_video_name}.csv"
VISUALIZATION_VIDEO = tracking_dir / f"tracking_overlay_{test_video_name}"
TEAM_TRACKING_CSV = tracking_dir / f"player_team_tracks_{test_video_name}.csv"
DETECTIONS_CSV = tracking_dir / f"player_detections_{test_video_name}.csv"

net_left: Tuple[float, float] = (73, 482)
net_right: Tuple[float, float] = (1154, 490)

def main():
    detector = PlayerDetection()

    # Step 1: Detect players and save detections (no tracking)
    if DETECTIONS_CSV.is_file():
        detections_df = pd.read_csv(DETECTIONS_CSV)
    else:
        detections_df = detector.detect_and_save(VIDEO_PATH, DETECTIONS_CSV)
        print(f"Saved player detections to {DETECTIONS_CSV}")

    # Step 2: Run custom PlayerTracker on detections to assign consistent IDs and teams
    tracker = PlayerTracker(net_left=net_left, net_right=net_right)
    video_reader = VideoReader(VIDEO_PATH)
    tracks_df = tracker.run_tracking(detections_df, video_reader)

    # Save tracker outputs
    tracks_df.to_csv(TRACKING_CSV, index=False)
    print(f"Saved player tracks to {TRACKING_CSV}")

    # Optionally, convert to wide format and save
    team_tracking_df = PlayerTracker.tracks_to_team_df(tracks_df)
    team_tracking_df.to_csv(TEAM_TRACKING_CSV, index=False)
    print(f"Saved team/ID assigned tracks to {TEAM_TRACKING_CSV}")

    # Step 3: Visualize tracking results using the assigned track IDs
    if VISUALIZATION_VIDEO.is_file():
        print(f"Visualization video already exists at {VISUALIZATION_VIDEO}")
    else:
        visualizer = TrackingVisualizer(VIDEO_PATH, tracks_df, net_left=tracker.net_left, net_right=tracker.net_right)
        visualizer.visualize_and_save(VISUALIZATION_VIDEO)
        print(f"Saved visualization video to {VISUALIZATION_VIDEO}")

if __name__ == "__main__":
    main() 