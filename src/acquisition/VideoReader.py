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

	def video_frames_generator(self, start_frame: int = 0, interval: int = 1, end_frame=-1):
		# Seek to the start frame directly
		self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
		frame_count = 0
		
		while self.video.isOpened():
			ret, frame = self.video.read()
			if not ret:
				break
			if frame_count % interval == 0:
				yield frame
			frame_count += 1
			if frame_count % 20 == 0:
				print(f"video_frames_generator processing frame #{frame_count}")
			if end_frame < start_frame + interval * frame_count:
				break

		self.video.release()