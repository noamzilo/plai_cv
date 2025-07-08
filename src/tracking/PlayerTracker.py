import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore
from typing import List, Dict, Tuple, Callable
import cv2  # type: ignore

# -----------------------------------------------------------------------------
# Configuration constants for ID jitter removal
# -----------------------------------------------------------------------------
# Minimum IOU required to allow an ID change between consecutive (or look-back)
# frames. A higher value makes ID switches less likely.
IOU_THRESHOLD: float = 0.5

# How many frames back we search when validating whether an existing ID should
# remain the same. If no suitable match is found within this window, we consider
# that the object might genuinely be new or disappeared.
MAX_LOOKBACK_FRAMES_EXISTING_ID: int = 5

# How many frames back we search when deciding whether a newly detected box is
# truly new or corresponds to a previously missed detection.
MAX_LOOKBACK_FRAMES_NEW_BOX: int = 5

# Maximum allowed distance (in pixels) between the centers of two boxes to still
# consider them the *same* physical player, even if IOU is low because of slight
# bbox jitter. This helps avoid ID switches when the bounding box barely moves.
CENTER_DISTANCE_THRESHOLD: float = 40.0

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

    def _extract_histogram(self, image: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            # Invalid bbox, return zeros
            return np.zeros(512, dtype=np.float32)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _hist_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        # Use correlation (1.0 = perfect match)
        return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

    # ---------------------------------------------------------------------
    # Helper methods for ID-jitter removal
    # ---------------------------------------------------------------------
    def _compute_iou(self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
        """Compute Intersection over Union (IOU) between two bounding boxes."""
        # Handle NaNs early
        if any(np.isnan(v) for v in bbox1 + bbox2):
            return 0.0
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])
        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        box1_area = max(0.0, (bbox1[2] - bbox1[0])) * max(0.0, (bbox1[3] - bbox1[1]))
        box2_area = max(0.0, (bbox2[2] - bbox2[0])) * max(0.0, (bbox2[3] - bbox2[1]))
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _center_distance(self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
        """Compute Euclidean distance between the centers of two bounding boxes."""
        if any(np.isnan(v) for v in bbox1 + bbox2):
            return float("inf")
        c1x = (bbox1[0] + bbox1[2]) / 2.0
        c1y = (bbox1[1] + bbox1[3]) / 2.0
        c2x = (bbox2[0] + bbox2[2]) / 2.0
        c2y = (bbox2[1] + bbox2[3]) / 2.0
        return float(np.hypot(c1x - c2x, c1y - c2y))

    def _get_side_label(self, bbox: Tuple[float, float, float, float]) -> str:
        """Return 'close' if bbox is below the net, else 'far'."""
        cx, cy = self._bbox_middle_bottom(bbox)
        return "close" if self._is_below_net(cx, cy) else "far"

    def _find_best_matching_id(
        self,
        detection_bbox: Tuple[float, float, float, float],
        candidate_ids: List[int],
        last_bbox_per_id: Dict[int, Tuple[int, Tuple[float, float, float, float]]],
        current_frame: int,
        max_lookback: int,
    ) -> Tuple[int, float, float]:
        """Return (best_id, best_iou, best_center_distance)."""
        best_id: int = -1
        best_iou: float = 0.0
        best_dist: float = float("inf")
        for pid in candidate_ids:
            if pid not in last_bbox_per_id:
                continue
            last_frame, prev_bbox = last_bbox_per_id[pid]
            if current_frame - last_frame > max_lookback:
                continue
            iou_val = self._compute_iou(detection_bbox, prev_bbox)
            dist_val = self._center_distance(detection_bbox, prev_bbox)
            # Prefer higher IOU; if IOU ties, prefer smaller distance
            if (iou_val > best_iou) or (np.isclose(iou_val, best_iou) and dist_val < best_dist):
                best_iou = iou_val
                best_dist = dist_val
                best_id = pid
        return best_id, best_iou, best_dist

    def remove_id_jittering(self, tracks_df: pd.DataFrame) -> pd.DataFrame:
        """Post-process `tracks_df` to mitigate ID jittering using IOU consistency.

        Rules implemented:
        1. IOU between boxes from opposite court sides is considered 0.
        2. A detection keeps its previous ID unless a different ID has an IOU
           larger than IOU_THRESHOLD within `MAX_LOOKBACK_FRAMES_EXISTING_ID`.
        3. Newly appearing detections are matched against historical boxes up
           to `MAX_LOOKBACK_FRAMES_NEW_BOX` to decide if they are truly new.
        4. Side-consistency: Each side ('close' or 'far') can have at most
           `players_per_side` unique IDs. If only one ID exists on a side in
           consecutive frames, it is enforced to remain the same.
        """
        # Prepare look-up helpers
        side_slots = {"close": [0, 1], "far": [2, 3]}
        id_to_side = {pid: ("close" if pid in side_slots["close"] else "far") for pid in range(self.num_players)}

        # History: id -> (last_frame, bbox)
        last_bbox_per_id: Dict[int, Tuple[int, Tuple[float, float, float, float]]] = {}

        # We'll build a list of updated records and concatenate at the end
        updated_records: List[Dict] = []

        frames_sorted = sorted(tracks_df["frame"].unique())
        for frame_idx in frames_sorted:
            frame_rows = tracks_df[tracks_df["frame"] == frame_idx].copy()

            # Collect all row_index -> id assignments for the current frame
            assignments_for_frame: Dict[int, int] = {}

            # Group detections by side based on bbox geometry (more robust than using current ID)
            detections_by_side: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]] = {"close": [], "far": []}
            for row_index, row in frame_rows.iterrows():
                bbox = (row["x1"], row["y1"], row["x2"], row["y2"])
                if any(pd.isna(x) for x in bbox):
                    continue  # skip NaN boxes for matching purposes
                side = self._get_side_label(bbox)
                detections_by_side[side].append((row_index, bbox))

            # Process each side independently so IOU across sides is effectively 0.
            for side in ("close", "far"):
                candidate_ids = side_slots[side]
                side_detections = detections_by_side[side]

                # ------------------------------------------------------------------
                # Rule 6 – If exactly one detection on this side and previously there
                # was also exactly one active ID, keep that ID regardless of IOU.
                # ------------------------------------------------------------------
                if len(side_detections) == 1:
                    # Historical IDs active recently on this side
                    recent_ids = [
                        pid for pid in candidate_ids
                        if pid in last_bbox_per_id and frame_idx - last_bbox_per_id[pid][0] <= MAX_LOOKBACK_FRAMES_EXISTING_ID
                    ]
                    if len(recent_ids) == 1:
                        row_index, bbox = side_detections[0]
                        chosen_id = recent_ids[0]
                        assignments_for_frame[row_index] = chosen_id
                        last_bbox_per_id[chosen_id] = (frame_idx, bbox)
                        # Skip further processing for this side
                        continue

                # ------------------------------------------------------------------
                # Step 1: Attempt to match detections to existing IDs by IOU
                # ------------------------------------------------------------------
                unassigned_detections: List[Tuple[int, Tuple[float, float, float, float]]] = []
                assigned_ids = set()
                side_assignments: Dict[int, int] = {}  # row_index -> new_id

                # Build all (id, detection) IOU pairs
                pairs: List[Tuple[float, float, int, int, Tuple[float, float, float, float]]] = []  # (iou, dist, id, row_idx, bbox)
                for row_index, bbox in side_detections:
                    best_id, best_iou, best_dist = self._find_best_matching_id(
                        bbox,
                        candidate_ids,
                        last_bbox_per_id,
                        frame_idx,
                        MAX_LOOKBACK_FRAMES_EXISTING_ID,
                    )
                    if best_id != -1 and (best_iou >= IOU_THRESHOLD or best_dist <= CENTER_DISTANCE_THRESHOLD):
                        pairs.append((best_iou, best_dist, best_id, row_index, bbox))
                    else:
                        unassigned_detections.append((row_index, bbox))

                # Sort by IOU desc then distance asc
                pairs.sort(key=lambda p: (-p[0], p[1]))
                for iou_val, _, pid, row_index, bbox in pairs:
                    if pid in assigned_ids:
                        unassigned_detections.append((row_index, bbox))
                        continue
                    side_assignments[row_index] = pid
                    assigned_ids.add(pid)
                    last_bbox_per_id[pid] = (frame_idx, bbox)

                # ------------------------------------------------------------------
                # Step 2: Handle detections not matched above (potential new IDs)
                # ------------------------------------------------------------------
                available_ids = [pid for pid in candidate_ids if pid not in assigned_ids]
                for row_index, bbox in unassigned_detections:
                    # Look back to see if this bbox existed recently but was missed
                    best_id, best_iou, best_dist = self._find_best_matching_id(
                        bbox,
                        available_ids,
                        last_bbox_per_id,
                        frame_idx,
                        MAX_LOOKBACK_FRAMES_NEW_BOX,
                    )
                    if best_id != -1 and (best_iou >= IOU_THRESHOLD or best_dist <= CENTER_DISTANCE_THRESHOLD):
                        chosen_id = best_id
                    elif available_ids:
                        chosen_id = available_ids.pop(0)
                    else:
                        # Fallback in rare cases – keep original ID
                        chosen_id = int(frame_rows.loc[row_index, "track_id"])
                    side_assignments[row_index] = chosen_id
                    last_bbox_per_id[chosen_id] = (frame_idx, bbox)

            # Merge side-specific assignments into frame-level mapping
            assignments_for_frame.update(side_assignments)

            # ----------------------------------------------------------------------
            # Apply the resolved IDs for this frame
            # ----------------------------------------------------------------------
            for row_index, resolved_id in assignments_for_frame.items():
                frame_rows.at[row_index, "track_id"] = resolved_id

            updated_records.append(frame_rows)

        cleaned_df = pd.concat(updated_records, ignore_index=True)
        return cleaned_df

    def run_tracking(self, detections_df: pd.DataFrame, video_reader) -> pd.DataFrame:
        """
        Run player tracking in a streaming, memory-efficient way using VideoReader.
        Args:
            detections_df: DataFrame with columns: frame, track_id, x1, y1, x2, y2, conf
            video_reader: VideoReader instance with video_frames_generator()
        Returns:
            DataFrame with tracking results.
        """
        grouped = detections_df.groupby("frame")
        frame_indices = sorted(grouped.groups.keys())
        frame_indices_set = set(frame_indices)
        side_slots = {"close": [0, 1], "far": [2, 3]}
        slot_team = {0: "close", 1: "close", 2: "far", 3: "far"}
        last_positions = {pid: (np.nan, np.nan) for pid in range(self.num_players)}
        position_history = {pid: [] for pid in range(self.num_players)}
        last_active = {pid: -1 for pid in range(self.num_players)}
        # --- HISTOGRAM HISTORY ---
        hist_history = {pid: [] for pid in range(self.num_players)}
        track_records: List[Dict] = []
        frame_iter = video_reader.video_frames_generator()
        for frame_idx, frame_img in frame_iter:
            if frame_idx not in frame_indices_set:
                continue  # Only process frames with detections
            i = frame_indices.index(frame_idx)
            if i % 50 == 0:
                print(f"Tracking frame {i+1}/{len(frame_indices)} (frame_idx={frame_idx})")
            dets = grouped.get_group(frame_idx)
            det_list = []
            det_histograms = []
            for _, row in dets.iterrows():
                bbox = (row["x1"], row["y1"], row["x2"], row["y2"])
                cx, cy = self._bbox_middle_bottom(bbox)
                team = "close" if self._is_below_net(cx, cy) else "far"
                det_list.append({"bbox": bbox, "cx": cx, "cy": cy, "team": team})
                hist = self._extract_histogram(frame_img, bbox)
                det_histograms.append(hist)
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
                    # --- HISTOGRAM ---
                    hist = det_histograms[min_idx]
                    hist_history[pid].append(hist)
                    if len(hist_history[pid]) > 100:
                        hist_history[pid] = hist_history[pid][-100:]
            # 2. For any remaining detections, assign to empty slots on correct team by interpolation/extrapolation, proximity, and histogram
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
                        best_sim = -float("inf")
                        best_score = float("inf")
                        for idx in list(remaining_dets):
                            dx, dy = det_list[idx]["cx"], det_list[idx]["cy"]
                            dist = np.hypot(dx - pred[0], dy - pred[1]) if not np.isnan(pred[0]) else 0
                            # --- HISTOGRAM ---
                            det_hist = det_histograms[idx]
                            # Compare to average of last N histograms
                            sim_scores = []
                            for N in [10, 20, 50, 75, 100]:
                                if len(hist_history[pid]) >= N:
                                    avg_hist = np.mean(hist_history[pid][-N:], axis=0)
                                    sim = self._hist_similarity(det_hist, avg_hist)
                                    sim_scores.append(sim)
                            if sim_scores:
                                sim = float(np.mean(sim_scores))
                            else:
                                sim = 0.0
                            # Combine distance and similarity (tunable: here, prioritize similarity if available)
                            score = dist - sim * 100  # Higher sim reduces score
                            if score < best_score:
                                best_score = score
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
                        # --- HISTOGRAM ---
                        hist = det_histograms[idx]
                        hist_history[pid].append(hist)
                        if len(hist_history[pid]) > 100:
                            hist_history[pid] = hist_history[pid][-100:]
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
        # ------------------------------------------------------------------
        # Post-process to mitigate ID jittering
        # ------------------------------------------------------------------
        tracks_df = self.remove_id_jittering(tracks_df)
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