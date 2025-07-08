import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore
from typing import List, Dict, Tuple, Callable, Optional
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

# -----------------------------------------------------------------------------
# New configuration constants for prioritised ID assignment
# -----------------------------------------------------------------------------
# Frames to look back (incrementally) when searching for IOU matches
LOOKBACK_STEPS: list[int] = [1, 2, 3, 4, 5, 10]

# IOU threshold above which a previous bbox is considered the **same** player
IOU_MATCH_THRESHOLD: float = 0.5  # Step-1 threshold

# Number of historical frames to use when computing extrapolated positions
EXTRAPOLATION_HISTORY: int = 5  # Step-4 history length

# Swap out the older CENTER_DISTANCE_THRESHOLD with a slightly lower value for histogram fallback.
HISTOGRAM_DISTANCE_BIAS: float = 100.0  # Larger => histogram similarity more important

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
        """Refactored tracker that follows strict priority rules and guarantees
        exactly four unique IDs (0-3) per frame.
        """
        side_slots = {"close": [0, 1], "far": [2, 3]}

        # ----------------------------------------
        # State that persists across frames
        # ----------------------------------------
        last_positions: Dict[int, Tuple[float, float]] = {pid: (np.nan, np.nan) for pid in range(self.num_players)}
        position_history: Dict[int, List[Tuple[float, float]]] = {pid: [] for pid in range(self.num_players)}
        hist_history: Dict[int, List[np.ndarray]] = {pid: [] for pid in range(self.num_players)}
        last_bbox_per_id: Dict[int, Tuple[int, Tuple[float, float, float, float]]] = {}

        # Cache detections by frame for quick access
        detections_by_frame = {f: grp for f, grp in detections_df.groupby("frame")}

        track_records: List[Dict] = []

        for frame_idx, frame_img in video_reader.video_frames_generator():
            if frame_idx not in detections_by_frame:
                # Still need to output 4 rows (all NaN) to keep continuity
                for pid in range(self.num_players):
                    track_records.append({
                        "frame": frame_idx,
                        "track_id": pid,
                        "x1": np.nan,
                        "y1": np.nan,
                        "x2": np.nan,
                        "y2": np.nan,
                        "conf": np.nan,
                    })
                continue

            detections_frame = detections_by_frame[frame_idx]

            # Build detection objects with histograms & metadata
            det_objects: List[Dict] = []
            for row_idx, row in detections_frame.iterrows():
                bbox = (row["x1"], row["y1"], row["x2"], row["y2"])
                cx, cy = self._bbox_middle_bottom(bbox)
                team = "close" if self._is_below_net(cx, cy) else "far"
                hist = self._extract_histogram(frame_img, bbox)
                det_objects.append({
                    "bbox": bbox,
                    "cx": cx,
                    "cy": cy,
                    "team": team,
                    "hist": hist,
                    "row_idx": row_idx,
                })

            # Split detections by side and assign IDs using new helper
            assignments: Dict[int, int] = {}  # row_idx -> pid

            for team_label in ("close", "far"):
                side_dets = [d for d in det_objects if d["team"] == team_label]
                candidate_ids = side_slots[team_label]
                side_assign, last_bbox_per_id = self._assign_ids_for_side(
                    team=team_label,
                    detections=side_dets,
                    candidate_ids=candidate_ids,
                    current_frame=frame_idx,
                    last_bbox_per_id=last_bbox_per_id,
                    last_positions=last_positions,
                    position_history=position_history,
                    hist_history=hist_history,
                )
                assignments.update(side_assign)

            # Build per-pid slot assignment dict
            slot_assignment: Dict[int, Optional[Dict]] = {pid: None for pid in range(self.num_players)}
            for det in det_objects:
                ridx = det["row_idx"]
                if ridx in assignments:
                    pid = assignments[ridx]
                    slot_assignment[pid] = det

            # Update histories and produce track rows
            for pid in range(self.num_players):
                det = slot_assignment[pid]
                if det is not None:
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

                    # history updates
                    last_positions[pid] = (det["cx"], det["cy"])
                    position_history[pid].append((det["cx"], det["cy"]))
                    if len(position_history[pid]) > EXTRAPOLATION_HISTORY:
                        position_history[pid] = position_history[pid][-EXTRAPOLATION_HISTORY:]
                    hist_history[pid].append(det["hist"])
                    if len(hist_history[pid]) > 100:
                        hist_history[pid] = hist_history[pid][-100:]
                else:
                    # Missing detection → NaN row
                    track_records.append({
                        "frame": frame_idx,
                        "track_id": pid,
                        "x1": np.nan,
                        "y1": np.nan,
                        "x2": np.nan,
                        "y2": np.nan,
                        "conf": np.nan,
                    })

            # Sanity: ensure uniqueness (should always hold)
            pids_in_frame = {rec["track_id"] for rec in track_records[-self.num_players:]}
            assert len(pids_in_frame) == self.num_players, "Duplicate or missing IDs in frame assignment"

        # Build final dataframe – no post-jittering pass needed because logic already enforces consistency
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

    # ---------------------------------------------------------------------
    # New helper utilities (side-agnostic)                                               
    # ---------------------------------------------------------------------
    def _iou(self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
        """Wrapper around `_compute_iou` to keep public helpers clustered together."""
        return self._compute_iou(bbox1, bbox2)

    def _best_iou_match(
        self,
        detection_bbox: Tuple[float, float, float, float],
        candidate_ids: List[int],
        last_bbox_per_id: Dict[int, Tuple[int, Tuple[float, float, float, float]]],
        current_frame: int,
    ) -> Tuple[int, float]:
        """Return `(best_id, frame_delta)` given incremental `LOOKBACK_STEPS`.

        The function walks through `LOOKBACK_STEPS` (1,2,3,...) and returns the
        first ID that has an IOU above `IOU_MATCH_THRESHOLD` within that exact
        look-back window. If no ID satisfies the condition, returns `(-1, inf)`.
        """
        for lookback in LOOKBACK_STEPS:
            best_id = -1
            for pid in candidate_ids:
                if pid not in last_bbox_per_id:
                    continue
                last_frame, prev_bbox = last_bbox_per_id[pid]
                if current_frame - last_frame != lookback:
                    continue
                iou_val = self._iou(detection_bbox, prev_bbox)
                if iou_val >= IOU_MATCH_THRESHOLD:
                    best_id = pid
                    break  # Found a valid match at this look-back step
            if best_id != -1:
                return best_id, lookback
        return -1, float("inf")

    def _histogram_similarity(self, det_hist: np.ndarray, hist_history: List[np.ndarray]) -> float:
        """Compute average histogram similarity against historical entries."""
        if not hist_history:
            return 0.0
        sims = [self._hist_similarity(det_hist, h) for h in hist_history[-100:]]
        return float(np.mean(sims)) if sims else 0.0

    def _predict_position(self, history: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Extrapolate next (cx,cy) position using the last two points."""
        if len(history) >= 2:
            (x1, y1), (x2, y2) = history[-2], history[-1]
            return x2 + (x2 - x1), y2 + (y2 - y1)
        elif len(history) == 1:
            return history[-1]
        else:
            return (np.nan, np.nan)

    def _assign_ids_for_side(
        self,
        team: str,
        detections: List[Dict],  # dict keys: bbox, cx, cy, hist, row_idx
        candidate_ids: List[int],
        current_frame: int,
        last_bbox_per_id: Dict[int, Tuple[int, Tuple[float, float, float, float]]],
        last_positions: Dict[int, Tuple[float, float]],
        position_history: Dict[int, List[Tuple[float, float]]],
        hist_history: Dict[int, List[np.ndarray]],
    ) -> Tuple[Dict[int, int], Dict[int, Tuple[int, Tuple[float, float, float, float]]]]:
        """Return mapping `row_idx -> pid` for this side, enforcing priority order.

        Steps implemented:
        1. IOU matching with incremental lookbacks.
        2. Single-assignment shortcut (exactly one IOU assigned → other det gets remaining ID).
        3. Histogram-based re-identification for any remaining detections.
        4. Extrapolation fallback using recent trajectories when histogram fails.
        """
        assignments: Dict[int, int] = {}
        assigned_ids: set[int] = set()
        unassigned_dets: List[Dict] = []

        # STEP-1: Incremental IOU matching
        for det in detections:
            best_id, _ = self._best_iou_match(det["bbox"], candidate_ids, last_bbox_per_id, current_frame)
            if best_id != -1 and best_id not in assigned_ids:
                assignments[det["row_idx"]] = best_id
                assigned_ids.add(best_id)
                last_bbox_per_id[best_id] = (current_frame, det["bbox"])
            else:
                unassigned_dets.append(det)

        # STEP-2: Single IOU shortcut
        if len(assignments) == 1 and len(unassigned_dets) == 1:
            remaining_id = [pid for pid in candidate_ids if pid not in assigned_ids][0]
            det = unassigned_dets.pop()
            assignments[det["row_idx"]] = remaining_id
            assigned_ids.add(remaining_id)
            last_bbox_per_id[remaining_id] = (current_frame, det["bbox"])

        # STEP-3: Histogram-based matching for any leftover detections
        still_unassigned: List[Dict] = []
        for det in unassigned_dets:
            best_score = -float("inf")
            chosen_id = -1
            for pid in candidate_ids:
                if pid in assigned_ids:
                    continue
                sim = self._histogram_similarity(det["hist"], hist_history[pid])
                if sim > best_score:
                    best_score = sim
                    chosen_id = pid
            if chosen_id != -1:
                assignments[det["row_idx"]] = chosen_id
                assigned_ids.add(chosen_id)
                last_bbox_per_id[chosen_id] = (current_frame, det["bbox"])
            else:
                still_unassigned.append(det)

        # STEP-4: Extrapolation using last positions if two detections remain for same side
        for det in still_unassigned:
            best_dist = float("inf")
            chosen_id = -1
            for pid in candidate_ids:
                if pid in assigned_ids:
                    continue
                pred_x, pred_y = self._predict_position(position_history[pid])
                dist = (
                    np.hypot(det["cx"] - pred_x, det["cy"] - pred_y)
                    if not np.isnan(pred_x)
                    else float("inf")
                )
                if dist < best_dist:
                    best_dist = dist
                    chosen_id = pid
            if chosen_id == -1:
                # If everything fails, assign first available id deterministically
                remaining = [pid for pid in candidate_ids if pid not in assigned_ids]
                if not remaining:
                    continue  # Should not occur, but guard anyway
                chosen_id = remaining[0]
            assignments[det["row_idx"]] = chosen_id
            assigned_ids.add(chosen_id)
            last_bbox_per_id[chosen_id] = (current_frame, det["bbox"])

        return assignments, last_bbox_per_id 