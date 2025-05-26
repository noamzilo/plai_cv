import cv2
from pathlib import Path

class VideoReader():
	def __init__(self, video_path: Path):
		self.video_path: Path = video_path
		assert video_path.is_file()
		self.video: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
		assert self.video.isOpened()
		self._average_frame = None

	@property
	def average_frame(self):
		if self._average_frame is None:
			raise ValueError("_average_frame not yet defined")
		return self._average_frame

	def video_frames_generator(self, start_frame: int = 0, interval: int = 1, end_frame: int = -1):
		self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
		curr_frame = start_frame
		while self.video.isOpened():
			ret, frame = self.video.read()
			if not ret:
				break
			if (curr_frame - start_frame) % interval == 0:
				yield curr_frame, frame
			curr_frame += 1
			if 0 < end_frame <= curr_frame:
				break
		self.video.release()