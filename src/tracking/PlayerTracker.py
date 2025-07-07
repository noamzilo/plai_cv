import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore
from typing import List, Dict, Tuple

class PlayerTracker:
    """
    Player tracker for assigning consistent IDs and teams.
    Input: DataFrame with columns: frame, track_id, x1, y1, x2, y2, conf (from YOLOv8 tracking)
    Output: DataFrame with one row per frame, columns for each player's (x, y) position.
    The net is defined by two points (net_left, net_right) as (x, y) tuples.
    """
    def __init__(self, net_left: Tuple[float, float] = (840, 705), net_right: Tuple[float, float] = (2955, 751), num_players: int = 4, players_per_side: int = 2):
        self.net_left = np.array(net_left, dtype=np.float32)
        self.net_right = np.array(net_right, dtype=np.float32)
        self.num_players = num_players
        self.players_per_side = players_per_side

    def _is_below_net(self, cx: float, cy: float) -> bool:
        x1, y1 = self.net_left
        x2, y2 = self.net_right
        dx, dy = x2 - x1, y2 - y1
        px, py = cx - x1, cy - y1
        cross = dx * py - dy * px
        return cross > 0

    def _bbox_middle_bottom(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = y2
        return cx, cy

    def run_tracking(self, detections_df: pd.DataFrame) -> pd.DataFrame:
        # Group detections by frame
        grouped = detections_df.groupby("frame")
        all_tracks = []
        frame_indices = sorted(grouped.groups.keys())
        for frame_idx in tqdm(frame_indices, desc="Assigning teams/IDs"):
            dets = grouped.get_group(frame_idx)
            frame_tracks = []
            for _, row in dets.iterrows():
                bbox = (row["x1"], row["y1"], row["x2"], row["y2"])
                cx, cy = self._bbox_middle_bottom(bbox)
                frame_tracks.append({
                    "track_id": row["track_id"],
                    "bbox": bbox,
                    "cx": cx,
                    "cy": cy,
                })
            all_tracks.append(frame_tracks)
        tracking_df = self._assign_teams_and_ids(all_tracks)
        return tracking_df

    def _assign_teams_and_ids(self, all_tracks: List[List[Dict]]) -> pd.DataFrame:
        """
        Assigns players to teams and consistent IDs for each frame.

        Output DataFrame:
            - Each row corresponds to a single video frame (indexed by 'frame').
            - For each player slot (e.g., player0, player1, ...), the columns:
                - 'player{pid}_x': x coordinate (middle of bottom of bounding box) for player pid in this frame
                - 'player{pid}_y': y coordinate (middle of bottom of bounding box) for player pid in this frame
            - If a player is not detected in a frame, their coordinates are filled with their last known position (or NaN if never seen).
            - Players are assigned to 'close' or 'far' teams based on their position relative to the net, and sorted left-to-right within each team.
        """
        side_slots = {"close": [0, 1], "far": [2, 3]}
        records = []
        last_positions = {pid: (np.nan, np.nan) for pid in range(self.num_players)}
        for frame_idx, tracks in enumerate(all_tracks):
            close, far = [], []
            for t in tracks:
                cx, cy = t["cx"], t["cy"]
                if self._is_below_net(cx, cy):
                    close.append((t["track_id"], t["bbox"], cx, cy))
                else:
                    far.append((t["track_id"], t["bbox"], cx, cy))
            close = sorted(close, key=lambda x: x[1][0])[:self.players_per_side]
            far = sorted(far, key=lambda x: x[1][0])[:self.players_per_side]
            row = {"frame": frame_idx}
            for slot, player in enumerate(close):
                pid = side_slots["close"][slot]
                _, _, cx, cy = player
                row[f"player{pid}_x"] = cx
                row[f"player{pid}_y"] = cy
                last_positions[pid] = (cx, cy)
            for slot, player in enumerate(far):
                pid = side_slots["far"][slot]
                _, _, cx, cy = player
                row[f"player{pid}_x"] = cx
                row[f"player{pid}_y"] = cy
                last_positions[pid] = (cx, cy)
            for pid in range(self.num_players):
                if f"player{pid}_x" not in row:
                    row[f"player{pid}_x"], row[f"player{pid}_y"] = last_positions[pid]
            records.append(row)
        df = pd.DataFrame(records)
        return df 