import cv2
import numpy as np
from pathlib import Path

class VideoAverager:
	def __init__(self, video_path: Path):
		self.video_path: Path = video_path
		self.video: cv2.VideoCapture = cv2.VideoCapture(video_path)
		
	@property
	def average_frame(self) -> np.ndarray:
		return self._average_frame
	
	@average_frame.setter
	def average_frame(self, value: np.ndarray):
		self._average_frame = value

	def video_frames_generator(self, start_frame: int = 0, interval: int = 1):
		# Seek to the start frame directly
		self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
		frame_count = 0
		
		while self.video.isOpened():
			ret, frame = self.video.read()
			if not ret:
				break
			if frame_count % interval == 0:
				yield frame, frame_count
			frame_count += 1

		self.video.release()

	@average_frame.setter
	def average_frames(self, interval: int = 10):
		frame_gen = self.video_frames_generator(interval)
		try:
			# Initialize with first frame
			self.average_frame, _ = next(frame_gen).astype(np.float64)
			count = 1
			
			# Running average for remaining frames
			for current_frame, _ in frame_gen:
				self.average_frame = (self.average_frame * count + current_frame) / (count + 1)
				count += 1
				
			self.average_frame = self._average_frame.astype(np.uint8)
		except StopIteration:
			raise ValueError("No frames were processed")

	def save_average_frame(self, output_path: str):
		cv2.imwrite(output_path, self._average_frame)

