import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(project_root, "data", "raw", "video.mp4")
raw_data_path = os.path.join(project_root, "data", "raw")
calculated_data_path = os.path.join(project_root, "data", "calculated") 
