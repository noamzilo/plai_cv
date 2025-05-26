from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
print(f"project_root: {project_root}")
data_path = project_root / "data"
print(f"data_path: {data_path}")
raw_data_path = project_root / "data" / "raw"
calculated_data_path = project_root / "data" / "calculated"
models_path = project_root / "models"
yolo_model_path = models_path / "yolo_model.pt"
test_video_name = "game1_3.mp4"
detections_csv_path = calculated_data_path / test_video_name / "player_bboxes.csv"
tracked_detections_csv_path = calculated_data_path / test_video_name / "tracked_player_bboxes.csv"