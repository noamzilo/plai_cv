import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from ultralytics import YOLO  # type: ignore
from pathlib import Path
from src.acquisition.VideoReader import VideoReader
from typing import Optional
from tqdm import tqdm

class PlayerDetection:
    """
    Player detector using YOLOv8. Outputs a DataFrame with one row per detection and columns:
    frame, x1, y1, x2, y2, conf
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "yolov8n.pt"
        self.model = YOLO(self.model_path)

    def detect_video(self, video_path: Path) -> pd.DataFrame:
        video_reader = VideoReader(video_path)
        detections = []
        for frame_idx, frame in video_reader.video_frames_generator():
            results = self.model(frame)[0]
            for r in results.boxes:
                if int(r.cls[0]) == 0:  # person class
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    conf = float(r.conf[0])
                    detections.append({
                        "frame": frame_idx,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": conf
                    })
        df = pd.DataFrame(detections)
        return df

    def detect_and_save(self, video_path: Path, output_csv: Path) -> pd.DataFrame:
        df = self.detect_video(video_path)
        df.to_csv(output_csv, index=False)
        return df

    def track_video(self, video_path: Path, tracker: str = 'bytetrack.yaml') -> pd.DataFrame:
        """
        Run YOLOv8 tracking on a video. Returns DataFrame with columns:
        frame, track_id, x1, y1, x2, y2, conf
        Only includes person class.
        """
        print("Starting YOLOv8 tracking...")
        results = self.model.track(source=str(video_path), tracker=tracker, stream=True, verbose=True)
        detections = []
        for frame_idx, result in tqdm(enumerate(results), desc='Tracking frames'):
            boxes = result.boxes
            ids = getattr(boxes, 'id', None)
            for i, r in enumerate(boxes):
                if int(r.cls[0]) == 0:  # person class
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    conf = float(r.conf[0])
                    track_id = int(ids[i]) if ids is not None else -1
                    detections.append({
                        "frame": frame_idx,
                        "track_id": track_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": conf
                    })
        print("Tracking complete.")
        df = pd.DataFrame(detections)
        return df

    def track_and_save(self, video_path: Path, output_csv: Path, tracker: str = 'bytetrack.yaml') -> pd.DataFrame:
        df = self.track_video(video_path, tracker=tracker)
        df.to_csv(output_csv, index=False)
        return df 