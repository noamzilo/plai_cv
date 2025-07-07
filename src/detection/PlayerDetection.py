import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from ultralytics import YOLO  # type: ignore
from pathlib import Path
from src.acquisition.VideoReader import VideoReader
from typing import Optional
from tqdm import tqdm
from matplotlib.path import Path as MplPath

class PlayerDetection:
    """
    Player detector using YOLOv8. Outputs a DataFrame with one row per detection and columns:
    frame, x1, y1, x2, y2, conf
    """
    def __init__(self, model_path: Optional[str] = None, pitch_polygon: Optional[np.ndarray] = None):
        self.model_path = model_path or "yolov8n.pt"
        self.model = YOLO(self.model_path)
        # Default polygon if none provided
        if pitch_polygon is None:
            self._pitch_polygon = np.array([
                [70, 319],
                [82, 1079],
                [1919, 1079],
                [1919, 738],
                [740, 370],
                [70, 319]
            ])
        else:
            self._pitch_polygon = pitch_polygon
        self._pitch_path = MplPath(self._pitch_polygon)

    def _is_inside_pitch(self, px: float, py: float) -> bool:
        """
        Returns True if the point (px, py) is inside the pitch polygon.
        """
        return self._pitch_path.contains_point((px, py))

    def filter_detections_by_pitch(self, detections: list[dict]) -> list[dict]:
        """
        Filters detections to only those whose bottom center is inside the pitch polygon.
        Each detection is a dict with x1, y1, x2, y2.
        """
        filtered = []
        for det in detections:
            bx = (det["x1"] + det["x2"]) / 2
            by = det["y2"]  # bottom center
            if self._is_inside_pitch(bx, by):
                filtered.append(det)
        return filtered

    def detect_video(self, video_path: Path) -> pd.DataFrame:
        video_reader = VideoReader(video_path)
        detections = []
        for frame_idx, frame in video_reader.video_frames_generator():
            if frame_idx % 50 == 0:
                print(f"Detecting frame {frame_idx}")
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
        detections = self.filter_detections_by_pitch(detections)
        df = pd.DataFrame(detections)
        return df

    def detect_and_save(self, video_path: Path, output_csv: Path) -> pd.DataFrame:
        df = self.detect_video(video_path)
        df.to_csv(output_csv, index=False)
        return df

    def detect_and_save_no_tracking(self, video_path: Path, output_csv: Path) -> pd.DataFrame:
        """
        Run YOLOv8 detection (no tracking) on a video and save results to CSV. Returns DataFrame with columns:
        frame, x1, y1, x2, y2, conf
        """
        df = self.detect_video(video_path)
        df.to_csv(output_csv, index=False)
        return df

    def track_and_save(self, video_path: Path, output_csv: Path, tracker: str = 'bytetrack.yaml') -> pd.DataFrame:
        """
        Run YOLOv8 tracking on a video and save results to CSV. Returns DataFrame with columns:
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
        df = pd.DataFrame(detections)
        df.to_csv(output_csv, index=False)
        print(f"Tracking complete. Saved to {output_csv}")
        return df 