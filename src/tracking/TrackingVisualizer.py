import cv2  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path
from src.acquisition.VideoReader import VideoReader

class TrackingVisualizer:
    """
    Visualizes tracking results by overlaying bounding boxes and player names on the video.
    """
    def __init__(self, video_path: Path, tracking_df: pd.DataFrame):
        self.video_path = video_path
        self.tracking_df = tracking_df

    def visualize_and_save(self, output_path: Path, box_color=(0, 255, 0), thickness=2, font_scale=0.7):
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
        for frame_idx, frame in video_reader.video_frames_generator():
            frame_draw = frame.copy()
            if frame_idx in grouped.groups:
                row = grouped.get_group(frame_idx)
                for pid in range(4):
                    x_col = f'player{pid}_x'
                    y_col = f'player{pid}_y'
                    if x_col in row and y_col in row:
                        x = int(row[x_col].values[0])
                        y = int(row[y_col].values[0])
                        # Draw a circle at the player's position
                        cv2.circle(frame_draw, (x, y), 20, box_color, thickness)
                        # Draw player name
                        cv2.putText(frame_draw, f'Player {pid}', (x + 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)
            out.write(frame_draw)
        out.release()
        print(f"Saved visualization video to {output_path}") 