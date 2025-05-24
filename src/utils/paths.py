from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "raw" / "video.mp4"
raw_data_path = project_root / "data" / "raw"
calculated_data_path = project_root / "data" / "calculated"
