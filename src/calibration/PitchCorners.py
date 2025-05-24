import cv2
import numpy as np
from pathlib import Path
from utils.visualizations import cv2_imshow

# ---------- main class ----------
class PitchCorners:
	def __init__(self, image_path: str | Path):
		self.image_path = Path(image_path)
		self.image = cv2.imread(str(self.image_path))
		if self.image is None:
			raise FileNotFoundError(f"Can't read image at {self.image_path}")
		self.edges = None
		self.hough_lines = None
		self.corners = self.calculate_corners()			# ← populate on construction

	@staticmethod
	def _hough_lines(edge_img, p=false):
		if p:
			lines = cv2.HoughLines(edge_img, 1, np.pi / 180, 150, None, 0, 0)
			return [] if lines is None else lines[:, 0, :]
		else:
			lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 150, minLineLength=100, maxLineGap=20)
			return [] if lines is None else lines[:, 0, :]

	@staticmethod
	def _is_near_angle(theta, target, tol_deg=10):
		return abs(np.rad2deg(theta) - target) < tol_deg

	@staticmethod
	def _line_coeffs(p1, p2):
		# retorna (a,b,c) tal que ax+by+c=0 usando cross product homogéneo
		return np.cross(np.array([*p1, 1.0]), np.array([*p2, 1.0]))

	@staticmethod
	def _intersection_homog(l1, l2):
		pt = np.cross(l1, l2)		# homogeneous (x,y,w)
		if abs(pt[2]) < 1e-6:		# parallel
			return None
		return (pt[0] / pt[2], pt[1] / pt[2])

	def calculate_corners(self):
		# 1. edges
		gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5, 5), 0)
		self.edges = cv2.Canny(blur, 50, 150)

		# 2. Hough
		lines = self._hough_lines(self.edges)
		self.hough_lines = lines
		if len(lines) == 0:
			return []

		# 3. separar horizontales (near 0/180 deg) y laterales (resto)
		horiz, side = [], []
		for x1, y1, x2, y2 in lines:
			theta = np.arctan2((y2 - y1), (x2 - x1))
			if self._is_near_angle(theta, 0) or self._is_near_angle(theta, 180):
				horiz.append(((x1, y1), (x2, y2)))
			else:
				side.append(((x1, y1), (x2, y2)))

		if len(horiz) == 0 or len(side) < 2:
			return []

		# 4. elegir la línea horizontal más alta (menor y medio)
		horiz.sort(key=lambda l: (l[0][1] + l[1][1]) / 2)
		base_h = horiz[0]

		# 5. agrupar lados por pendiente (negativa vs positiva) para izquierda/derecha
		left_candidates, right_candidates = [], []
		for p1, p2 in side:
			slope = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-6)
			(left_candidates if slope > 0 else right_candidates).append((p1, p2))

		if len(left_candidates) == 0 or len(right_candidates) == 0:
			return []

		
		img_w = self.image.shape[1]
		left_candidates.sort(key=lambda l: min(l[0][0], l[1][0]))			
		right_candidates.sort(key=lambda l: img_w - max(l[0][0], l[1][0]))	
		left_l, right_l = left_candidates[0], right_candidates[0]

		# 6. intersecciones con la horizontal
		h_line = self._line_coeffs(*base_h)
		l_line = self._line_coeffs(*left_l)
		r_line = self._line_coeffs(*right_l)

		left_pt = self._intersection_homog(h_line, l_line)
		right_pt = self._intersection_homog(h_line, r_line)
		if left_pt is None or right_pt is None:
			return []

		return [(int(round(left_pt[0])), int(round(left_pt[1]))),
				(int(round(right_pt[0])), int(round(right_pt[1])))]

	# ---------- debug helpers ----------
	def _dump_and_show(self, img, name: str, out_dir: Path):
		out_path = out_dir / f"{name}.png"
		cv2.imwrite(str(out_path), img)
		cv2_imshow(img)
		print(f"Saved {out_path}")

	def debug_pipeline(self, out_dir: Path):
		out_dir.mkdir(parents=True, exist_ok=True)

		# edges
		self._dump_and_show(cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR), "edges", out_dir)

		# lines
		lines_img = self.image.copy()
		for (x1, y1, x2, y2) in self.hough_lines:
			cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
		self._dump_and_show(lines_img, "hough_lines", out_dir)

		# intersections
		inters_img = lines_img.copy()
		for (x, y) in self.corners:
			cv2.circle(inters_img, (x, y), 6, (0, 255, 255), -1)
		self._dump_and_show(inters_img, "intersections", out_dir)

	def draw_corners(self, out_path: Path | str, radius: int = 6):
		if len(self.corners) == 0:
			raise RuntimeError("No corners detected")

		min_x = min(p[0] for p in self.corners + [(0, 0)])
		min_y = min(p[1] for p in self.corners + [(0, 0)])
		max_x = max(p[0] for p in self.corners + [(self.image.shape[1] - 1, 0)])
		max_y = max(p[1] for p in self.corners + [(0, self.image.shape[0] - 1)])

		offset_x = -min(0, min_x)
		offset_y = -min(0, min_y)
		new_w = int(max_x + offset_x + 1)
		new_h = int(max_y + offset_y + 1)

		canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
		canvas[offset_y:offset_y + self.image.shape[0], offset_x:offset_x + self.image.shape[1]] = self.image

		for x, y in self.corners:
			cv2.circle(canvas, (x + offset_x, y + offset_y), radius, (0, 255, 0), -1)

		cv2.imwrite(str(out_path), canvas)
		return canvas

# ---------- quick CLI ----------
if __name__ == "__main__":
	import sys, json
	from utils.paths import calculated_data_path				# tu módulo

	img_path = calculated_data_path / "game1_3.mp4" / "average_frame.bmp"
	out_dir	 = calculated_data_path / "game1_3.mp4" / "debug"
	out_img	 = calculated_data_path / "game1_3.mp4" / "average_frame_with_corners.png"

	pc = PitchCorners(img_path)
	print(json.dumps(pc.corners))								# imprime coordenadas
	pc.debug_pipeline(out_dir)									# guarda/enseña etapas
	pc.draw_corners(out_img)
	print(f"Corners drawn → {out_img}")
