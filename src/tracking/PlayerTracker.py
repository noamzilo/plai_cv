import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
from typing import Optional, List, Dict, Tuple
import cv2  # type: ignore
from pathlib import Path

class PlayerTracker:
    """
    Player tracker using ByteTrack or DeepSORT (via deep_sort_realtime).
    Assigns consistent IDs and teams.
    Output: DataFrame with one row per frame, columns for each player's (x, y) position.
    The net is defined by two points (net_left, net_right) as (x, y) tuples.
    tracker_type: 'bytetrack' (default) or 'deepsort'
    """
    def __init__(self, net_left: Tuple[float, float] = (840, 705), net_right: Tuple[float, float] = (2955, 751), num_players: int = 4, players_per_side: int = 2, tracker_type: str = 'bytetrack'):
        self.net_left = np.array(net_left, dtype=np.float32)
        self.net_right = np.array(net_right, dtype=np.float32)
        self.num_players = num_players
        self.players_per_side = players_per_side
        assert tracker_type in ('bytetrack', 'deepsort'), "tracker_type must be 'bytetrack' or 'deepsort'"
        self.tracker_type = tracker_type
        self.tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, embedder="mobilenet", half=True, bgr=True, backend=tracker_type)

    def _is_below_net(self, cx: float, cy: float) -> bool:
        # Returns True if the point (cx, cy) is below the net line (using cross product)
        x1, y1 = self.net_left
        x2, y2 = self.net_right
        # Net vector
        dx, dy = x2 - x1, y2 - y1
        # Vector from net_left to point
        px, py = cx - x1, cy - y1
        # Cross product (z-component)
        cross = dx * py - dy * px
        return cross > 0  # convention: below net if cross > 0

    def run_tracking(self, video_path: Path, detections_df: pd.DataFrame) -> pd.DataFrame:
        import acquisition.VideoReader
        VideoReader = acquisition.VideoReader.VideoReader
        video_reader = VideoReader(video_path)
        frames = []
        for _, frame in video_reader.video_frames_generator():
            frames.append(frame)
        # Group detections by frame
        grouped = detections_df.groupby("frame")
        all_tracks = []
        for frame_idx, frame in enumerate(frames):
            dets = grouped.get_group(frame_idx) if frame_idx in grouped.groups else pd.DataFrame()
            dets_arr = dets[["x1", "y1", "x2", "y2", "conf"]].values if not dets.empty else np.zeros((0, 5))
            tracks = self.tracker.update_tracks(dets_arr, frame=frame)
            frame_tracks = []
            for t in tracks:
                if not t.is_confirmed():
                    continue
                track_id = t.track_id
                ltrb = t.to_ltrb()
                frame_tracks.append({
                    "track_id": track_id,
                    "bbox": ltrb,
                })
            all_tracks.append(frame_tracks)
        return self._assign_teams_and_ids(all_tracks)

    def _assign_teams_and_ids(self, all_tracks: List[List[Dict]]) -> pd.DataFrame:
        side_slots = {"close": [0, 1], "far": [2, 3]}
        records = []
        last_positions = {pid: (np.nan, np.nan) for pid in range(self.num_players)}
        for frame_idx, tracks in enumerate(all_tracks):
            close, far = [], []
            for t in tracks:
                x1, y1, x2, y2 = t["bbox"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if self._is_below_net(cx, cy):
                    close.append((t["track_id"], x1, y1, x2, y2))
                else:
                    far.append((t["track_id"], x1, y1, x2, y2))
            close = sorted(close, key=lambda x: x[1])[:self.players_per_side]
            far = sorted(far, key=lambda x: x[1])[:self.players_per_side]
            row = {"frame": frame_idx}
            for slot, player in enumerate(close):
                pid = side_slots["close"][slot]
                x1, y1, x2, y2 = player[1:]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                row[f"player{pid}_x"] = cx
                row[f"player{pid}_y"] = cy
                last_positions[pid] = (cx, cy)
            for slot, player in enumerate(far):
                pid = side_slots["far"][slot]
                x1, y1, x2, y2 = player[1:]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                row[f"player{pid}_x"] = cx
                row[f"player{pid}_y"] = cy
                last_positions[pid] = (cx, cy)
            for pid in range(self.num_players):
                if f"player{pid}_x" not in row:
                    row[f"player{pid}_x"], row[f"player{pid}_y"] = last_positions[pid]
            records.append(row)
        df = pd.DataFrame(records)
        return df 