import cv2  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path
from src.acquisition.VideoReader import VideoReader
from tqdm import tqdm

class TrackingVisualizer:
    """
    Visualizes tracking results by overlaying bounding boxes and player names on the video.
    """
    def __init__(self, video_path: Path, tracking_df: pd.DataFrame):
        self.video_path = video_path
        self.tracking_df = tracking_df

    def visualize_and_save(self, output_path: Path, box_color=(0, 255, 0), thickness=2, font_scale=0.7):
        print("Starting visualization...")
        video_reader = VideoReader(self.video_path)
        # Get video properties
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        # Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        # Group tracking data by frame
        grouped = self.tracking_df.groupby('frame')
        frame_count = int(self.tracking_df['frame'].max()) + 1 if not self.tracking_df.empty else 0
        for frame_idx, frame in tqdm(video_reader.video_frames_generator(), total=frame_count, desc='Visualizing frames'):
            frame_draw = frame.copy()
            if frame_idx in grouped.groups:
                rows = grouped.get_group(frame_idx)
                for _, row in rows.iterrows():
                    # Skip if any coordinate is NaN
                    if any(pd.isna(row[c]) for c in ['x1', 'y1', 'x2', 'y2']):
                        continue
                    x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                    track_id = int(row['track_id']) if 'track_id' in row else -1
                    # Draw bounding box
                    cv2.rectangle(frame_draw, (x1, y1), (x2, y2), box_color, thickness)
                    # Draw track_id label
                    label = f'ID {track_id}'
                    cv2.putText(frame_draw, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)
            out.write(frame_draw)
        out.release()
        print(f"Saved visualization video to {output_path}") 