import cv2
import numpy as np
from pathlib import Path

class VideoAverager:
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
				yield frame, frame_count
			frame_count += 1
			if frame_count % 20 == 0:
				print(f"video_frames_generator processing frame #{frame_count}")
			if end_frame < start_frame + interval * frame_count:
				break

		self.video.release()

	def average_frames(self, start_frame=0, interval: int = 1, end_frame=-1):
		frame_gen = self.video_frames_generator(start_frame, interval, end_frame)
		try:
			# Initialize with first frame
			next_frame, _ = next(frame_gen)
			self._average_frame = next_frame.astype(np.float64)
			count = 1
			
			# Running average for remaining frames
			for current_frame, _ in frame_gen:
				self._average_frame = (self._average_frame * count + current_frame) / (count + 1)
				count += 1
				
			self._average_frame = self._average_frame.astype(np.uint8)
		except StopIteration:
			raise ValueError("No frames were processed")

	def save_average_frame(self, output_path: str):
		cv2.imwrite(output_path, self._average_frame)

