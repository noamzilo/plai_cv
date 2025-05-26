import cv2
import numpy as np
from pathlib import Path
from utils.paths import yolo_model_path, calculated_data_path

# ─────────────────────────── Configuration ───────────────────────────
YOLO_MODEL_PATH = yolo_model_path  
YOLO_CONFIDENCE = 0.4
PERSON_CLASS_ID = 0  # COCO class index for person

# ─────────────────────── YOLOv8 Detector Setup ───────────────────────
def load_yolo_detector():
	import ultralytics
	from ultralytics import YOLO
	return YOLO(YOLO_MODEL_PATH)

# ───────────────────────────── Detection ─────────────────────────────
def detect_players(img, model):
	"""
	Detect players using YOLOv8 and return bounding boxes in image coordinates.
	"""
	results = model(img)[0]
	boxes = []
	for r in results.boxes:
		if int(r.cls[0]) == PERSON_CLASS_ID and r.conf[0] > YOLO_CONFIDENCE:
			x1, y1, x2, y2 = map(int, r.xyxy[0])
			boxes.append(((x1, y1), (x2, y2)))
	return boxes

# ─────────────────────────── Visualization ───────────────────────────
def draw_detections(img, boxes, color=(0, 255, 0)):
	"""
	Draw bounding boxes for visual inspection.
	"""
	for (x1, y1), (x2, y2) in boxes:
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
	return img

# ───────────────────────────── Main Test ─────────────────────────────
def run_detection_on_image(image_path, is_draw=False):
	model = load_yolo_detector()
	img = cv2.imread(str(image_path))
	boxes = detect_players(img, model)
	if is_draw:
		img_with_boxes = draw_detections(img, boxes)
		cv2.imshow("Detections", img_with_boxes)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def detect_pitch_corners(image: np.ndarray):
	# known points, hard coded for now. Corresponds to game 1_3
	net_left_bottom = np.array([840, 705])
	net_right_bottom = np.array([2955, 751])
	far_left_corner = np.array([1372, 485])
	far_right_corner = np.array([2451, 510])

	# inferred by extrapolation: close = net + (net - far)
	# THIS IS WRONG! because of perspective.
	# the correct way: this exercise is true IN 3D. so infer camera intrinsics from the known pitch edge sizes
	close_left_corner = net_left_bottom + (net_left_bottom - far_left_corner)
	close_right_corner = net_right_bottom + (net_right_bottom - far_right_corner)



	return {
		"far_left_corner": far_left_corner,
		"far_right_corner": far_right_corner,
		"close_left_corner": close_left_corner,
		"close_right_corner": close_right_corner,
		"left_bottom_net": left_bottom_net,
		"left_bottom_right": left_bottom_right,
	}

if __name__ == "__main__":
	image_path = calculated_data_path / "frame_068.png"
	run_detection_on_image(image_path)
