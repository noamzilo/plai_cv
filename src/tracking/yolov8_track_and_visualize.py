import pandas as pd
from pathlib import Path
from src.detection.PlayerDetection import PlayerDetection
from src.tracking.TrackingVisualizer import TrackingVisualizer
from src.utils.paths import project_root, test_video_name, output_path
from src.tracking.PlayerTracker import PlayerTracker

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

def main():
    # Step 1: Track players using YOLOv8 and save to CSV
    detector = PlayerDetection()
    if TRACKING_CSV.is_file():
        tracking_df = pd.read_csv(TRACKING_CSV)
    else:
        tracking_df = detector.track_and_save(VIDEO_PATH, TRACKING_CSV)

    print(f"Saved player tracks to {TRACKING_CSV}")

    # Step 2: Assign teams and consistent IDs using PlayerTracker
    tracker = PlayerTracker()
    team_tracking_df = tracker.run_tracking(tracking_df)
    team_tracking_df.to_csv(TEAM_TRACKING_CSV, index=False)
    print(f"Saved team/ID assigned tracks to {TEAM_TRACKING_CSV}")

    # Step 3: Visualize tracking results (using original tracking_df for bounding boxes)
    visualizer = TrackingVisualizer(VIDEO_PATH, tracking_df)
    visualizer.visualize_and_save(VISUALIZATION_VIDEO)
    print(f"Saved visualization video to {VISUALIZATION_VIDEO}")

if __name__ == "__main__":
    main() 