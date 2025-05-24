from utils.paths import raw_data_path, calculated_data_path
from pathlib import Path
import os

from calibration.VideoAverager import VideoAverager


def main():
	video_name = "game1_3.mp4"
	raw_data_path.is_dir()
	video_path = raw_data_path / video_name 
	assert video_path.is_file()
	video_averager = VideoAverager(video_path)
	video_averager.average_frames(start_frame=0, interval=10, end_frame=2000)
	
	video_calculated_path = calculated_data_path /f"{video_name}"
	os.makedirs(video_calculated_path, exist_ok=True)
	average_frame_path = video_calculated_path / f"average_frame.bmp"
	video_averager.save_average_frame(calculated_data_path / average_frame_path)


if __name__ == "__main__":
	main()