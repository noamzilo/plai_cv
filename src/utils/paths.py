from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
print(f"project_root: {project_root}")
data_path = project_root / "data"
print(f"data_path: {data_path}")
raw_data_path = project_root / "data" / "raw"
calculated_data_path = project_root / "data" / "calculated"
