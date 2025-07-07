from src.utils.paths import raw_data_path, calculated_data_path
from pathlib import Path
import os

from src.calibration.VideoAverager import VideoAverager
from src.calibration.PitchCorners import PitchCorners
from src.utils.paths import test_video_name

def main():
	video_name = test_video_name
	raw_data_path.is_dir()
	video_path = raw_data_path / video_name 
	assert video_path.is_file()
	video_averager = VideoAverager(video_path)
	video_averager.average_frames(start_frame=0, interval=10, end_frame=20000)
	
	video_calculated_path = calculated_data_path /f"{video_name}"
	os.makedirs(video_calculated_path, exist_ok=True)
	average_frame_name = f"average_frame.bmp"
	average_frame_path = video_calculated_path / average_frame_name 
	video_averager.save_average_frame(str(average_frame_path))

	pitch_corners = PitchCorners(image_path=average_frame_path)
	pitch_corners.calculate_corners()


if __name__ == "__main__":
	main()