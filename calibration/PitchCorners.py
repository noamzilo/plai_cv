import cv2

class PitchCorners:
	def __init__(self, image_path: str):
		self.image_path = image_path
		self.image = cv2.imread(image_path)
		self.corners = self.detect_corners()

