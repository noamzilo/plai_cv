import pandas as pd
from pathlib import Path
from src.detection.PlayerDetection import PlayerDetection
from src.tracking.TrackingVisualizer import TrackingVisualizer
from src.utils.paths import project_root

# --- CONFIG ---
VIDEO_PATH = Path(project_root / "data/proprietary/001.mp4")
CACHE_DIR = Path(project_root / "cache/yolov8_tracking_cache")
CACHE_DIR.mkdir(exist_ok=True)
TRACKING_CSV = CACHE_DIR / "player_tracks.csv"
VISUALIZATION_VIDEO = CACHE_DIR / "tracking_overlay.mp4"

def main():
    # Step 1: Track players using YOLOv8 and save to CSV
    detector = PlayerDetection()
    if TRACKING_CSV.is_file():
        tracking_df = pd.read_csv(TRACKING_CSV)
    else:
        tracking_df = detector.track_and_save(VIDEO_PATH, TRACKING_CSV)

    print(f"Saved player tracks to {TRACKING_CSV}")

    # Step 2: Visualize tracking results
    visualizer = TrackingVisualizer(VIDEO_PATH, tracking_df)
    visualizer.visualize_and_save(VISUALIZATION_VIDEO)
    print(f"Saved visualization video to {VISUALIZATION_VIDEO}")

if __name__ == "__main__":
    main() 