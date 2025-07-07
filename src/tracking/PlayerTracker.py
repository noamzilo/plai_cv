import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment

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
        # Initialize slot memory
        side_slots = {"close": [0, 1], "far": [2, 3]}
        slot_team = {0: "close", 1: "close", 2: "far", 3: "far"}
        last_positions = {pid: (np.nan, np.nan) for pid in range(self.num_players)}
        # Keep a history of positions for each slot
        position_history = {pid: [] for pid in range(self.num_players)}
        last_active = {pid: -1 for pid in range(self.num_players)}
        records = []
        for frame_idx in frame_indices:
            dets = grouped.get_group(frame_idx)
            # Compute (cx, cy) for each detection
            det_list = []
            for _, row in dets.iterrows():
                bbox = (row["x1"], row["y1"], row["x2"], row["y2"])
                cx, cy = self._bbox_middle_bottom(bbox)
                team = "close" if self._is_below_net(cx, cy) else "far"
                det_list.append({"bbox": bbox, "cx": cx, "cy": cy, "team": team})
            # Assign detections to slots
            assigned = set()
            slot_assignment = {pid: None for pid in range(self.num_players)}
            # 1. Try to match detections to previous slots by proximity and team
            for pid in range(self.num_players):
                prev_cx, prev_cy = last_positions[pid]
                team = slot_team[pid]
                # Find closest detection on same team not yet assigned
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
                    # Update history
                    position_history[pid].append((min_det["cx"], min_det["cy"]))
                    if len(position_history[pid]) > self.history_length:
                        position_history[pid] = position_history[pid][-self.history_length:]
                    last_active[pid] = frame_idx
            # 2. For any remaining detections, assign to empty slots on correct team by interpolation/extrapolation and proximity
            for team in ["close", "far"]:
                empty_slots = [pid for pid in side_slots[team] if slot_assignment[pid] is None]
                unassigned_dets = [idx for idx, det in enumerate(det_list) if det["team"] == team and idx not in assigned]
                if len(empty_slots) > 1 and len(unassigned_dets) > 1:
                    # Predict positions for each empty slot
                    pred_positions = []
                    for pid in empty_slots:
                        hist = position_history[pid]
                        if len(hist) >= 2:
                            # Linear extrapolation
                            (x1, y1), (x2, y2) = hist[-2], hist[-1]
                            pred_x = x2 + (x2 - x1)
                            pred_y = y2 + (y2 - y1)
                        elif len(hist) == 1:
                            pred_x, pred_y = hist[-1]
                        else:
                            pred_x, pred_y = np.nan, np.nan
                        pred_positions.append((pred_x, pred_y))
                    # Build cost matrix
                    det_positions = [(det_list[idx]["cx"], det_list[idx]["cy"]) for idx in unassigned_dets]
                    cost_matrix = np.zeros((len(empty_slots), len(unassigned_dets)))
                    for i, (px, py) in enumerate(pred_positions):
                        for j, (dx, dy) in enumerate(det_positions):
                            if np.isnan(px) or np.isnan(dx):
                                cost_matrix[i, j] = 1e6
                            else:
                                cost_matrix[i, j] = np.hypot(dx - px, dy - py)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for i, j in zip(row_ind, col_ind):
                        pid = empty_slots[i]
                        idx = unassigned_dets[j]
                        det = det_list[idx]
                        slot_assignment[pid] = det
                        assigned.add(idx)
                        last_positions[pid] = (det["cx"], det["cy"])
                        position_history[pid].append((det["cx"], det["cy"]))
                        if len(position_history[pid]) > self.history_length:
                            position_history[pid] = position_history[pid][-self.history_length:]
                        last_active[pid] = frame_idx
                else:
                    # Assign by left-right order as fallback
                    for idx in unassigned_dets:
                        det = det_list[idx]
                        empty_slots = [pid for pid in side_slots[team] if slot_assignment[pid] is None]
                        if empty_slots:
                            # Assign by left-right order
                            empty_slots = sorted(empty_slots, key=lambda pid: last_positions[pid][0] if not np.isnan(last_positions[pid][0]) else det["cx"])
                            pid = empty_slots[0]
                            slot_assignment[pid] = det
                            assigned.add(idx)
                            last_positions[pid] = (det["cx"], det["cy"])
                            position_history[pid].append((det["cx"], det["cy"]))
                            if len(position_history[pid]) > self.history_length:
                                position_history[pid] = position_history[pid][-self.history_length:]
                            last_active[pid] = frame_idx
            # 3. Build row for this frame
            row = {"frame": frame_idx}
            for pid in range(self.num_players):
                det = slot_assignment[pid]
                if det is not None and isinstance(det, dict):
                    row[f"player{pid}_x"] = det["cx"]
                    row[f"player{pid}_y"] = det["cy"]
                    last_positions[pid] = (det["cx"], det["cy"])
                    position_history[pid].append((det["cx"], det["cy"]))
                    if len(position_history[pid]) > self.history_length:
                        position_history[pid] = position_history[pid][-self.history_length:]
                else:
                    # Use last known position if available, else NaN
                    last_pos = last_positions.get(pid, (np.nan, np.nan))
                    if isinstance(last_pos, tuple) and len(last_pos) == 2:
                        row[f"player{pid}_x"], row[f"player{pid}_y"] = last_pos
                        last_positions[pid] = last_pos
                    else:
                        print(f"[DEBUG] Unexpected last_pos for pid={pid}: {last_pos}, using (np.nan, np.nan)")
                        row[f"player{pid}_x"] = np.nan
                        row[f"player{pid}_y"] = np.nan
                        last_positions[pid] = (np.nan, np.nan)
            records.append(row)
        df = pd.DataFrame(records)
        return df 