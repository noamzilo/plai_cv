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
        self.history_length = 2  # for extrapolation

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
        frame_indices = sorted(grouped.groups.keys())
        side_slots = {"close": [0, 1], "far": [2, 3]}
        slot_team = {0: "close", 1: "close", 2: "far", 3: "far"}
        last_positions = {pid: (np.nan, np.nan) for pid in range(self.num_players)}
        position_history = {pid: [] for pid in range(self.num_players)}
        last_active = {pid: -1 for pid in range(self.num_players)}
        track_records: List[Dict] = []
        for frame_idx in frame_indices:
            dets = grouped.get_group(frame_idx)
            det_list = []
            for _, row in dets.iterrows():
                bbox = (row["x1"], row["y1"], row["x2"], row["y2"])
                cx, cy = self._bbox_middle_bottom(bbox)
                team = "close" if self._is_below_net(cx, cy) else "far"
                det_list.append({"bbox": bbox, "cx": cx, "cy": cy, "team": team})
            assigned = set()
            slot_assignment = {pid: None for pid in range(self.num_players)}
            # 1. Try to match detections to previous slots by proximity and team
            for pid in range(self.num_players):
                prev_cx, prev_cy = last_positions[pid]
                team = slot_team[pid]
                min_dist = float("inf")
                min_det = None
                min_idx = -1
                for idx, det in enumerate(det_list):
                    if det["team"] != team or idx in assigned:
                        continue
                    dist = np.hypot(det["cx"] - prev_cx, det["cy"] - prev_cy) if not np.isnan(prev_cx) else 0
                    if dist < min_dist:
                        min_dist = dist
                        min_det = det
                        min_idx = idx
                if min_det is not None:
                    slot_assignment[pid] = min_det
                    assigned.add(min_idx)
                    last_positions[pid] = (min_det["cx"], min_det["cy"])
                    position_history[pid].append((min_det["cx"], min_det["cy"]))
                    if len(position_history[pid]) > self.history_length:
                        position_history[pid] = position_history[pid][-self.history_length:]
                    last_active[pid] = frame_idx
            # 2. For any remaining detections, assign to empty slots on correct team by interpolation/extrapolation and proximity
            for team in ["close", "far"]:
                empty_slots = [pid for pid in side_slots[team] if slot_assignment[pid] is None]
                unassigned_dets = [idx for idx, det in enumerate(det_list) if det["team"] == team and idx not in assigned]
                if len(empty_slots) > 1 and len(unassigned_dets) > 1:
                    remaining_dets = set(unassigned_dets)
                    for pid in empty_slots:
                        hist = position_history[pid]
                        if len(hist) >= 2:
                            (x1, y1), (x2, y2) = hist[-2], hist[-1]
                            pred = (x2 + (x2 - x1), y2 + (y2 - y1))
                        elif len(hist) == 1:
                            pred = hist[-1]
                        else:
                            pred = (np.nan, np.nan)
                        best_idx = None
                        best_dist = float("inf")
                        for idx in list(remaining_dets):
                            dx, dy = det_list[idx]["cx"], det_list[idx]["cy"]
                            dist = np.hypot(dx - pred[0], dy - pred[1]) if not np.isnan(pred[0]) else 0
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = idx
                        if best_idx is not None:
                            idx = best_idx
                        det = det_list[idx]
                        slot_assignment[pid] = det
                        assigned.add(idx)
                        remaining_dets.remove(idx)
                        last_positions[pid] = (det["cx"], det["cy"])
                        position_history[pid].append((det["cx"], det["cy"]))
                        if len(position_history[pid]) > self.history_length:
                            position_history[pid] = position_history[pid][-self.history_length:]
                        last_active[pid] = frame_idx
            # 3. For every player, append a record (even if missing)
            for pid in range(self.num_players):
                det = slot_assignment[pid]
                if det is not None and isinstance(det, dict):
                    bbox = det["bbox"]
                    track_records.append({
                        "frame": frame_idx,
                        "track_id": pid,
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                        "conf": 1.0,
                    })
                else:
                    # Use last known position if available, else NaN
                    last_pos = last_positions.get(pid, (np.nan, np.nan))
                    track_records.append({
                        "frame": frame_idx,
                        "track_id": pid,
                        "x1": np.nan,
                        "y1": np.nan,
                        "x2": np.nan,
                        "y2": np.nan,
                        "conf": np.nan,
                    })
        tracks_df = pd.DataFrame(track_records)
        return tracks_df

    @staticmethod
    def tracks_to_team_df(tracks_df: pd.DataFrame, num_players: int = 4) -> pd.DataFrame:
        """
        Convert long format tracks_df to wide format team_df (one row per frame, columns for each player's x/y).
        """
        # Pivot to wide format: one row per frame, columns player{pid}_x, player{pid}_y
        frames = sorted(tracks_df['frame'].unique())
        records = []
        for frame in frames:
            row = {'frame': frame}
            frame_tracks = tracks_df[tracks_df['frame'] == frame]
            for pid in range(num_players):
                player_row = frame_tracks[frame_tracks['track_id'] == pid]
                if not player_row.empty:
                    row[f'player{pid}_x'] = player_row.iloc[0]['x1'] + (player_row.iloc[0]['x2'] - player_row.iloc[0]['x1']) / 2
                    row[f'player{pid}_y'] = player_row.iloc[0]['y2']
                else:
                    row[f'player{pid}_x'] = np.nan
                    row[f'player{pid}_y'] = np.nan
            records.append(row)
        return pd.DataFrame(records) 