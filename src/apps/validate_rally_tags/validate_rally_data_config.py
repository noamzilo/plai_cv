# Configuration for rally data validation
import os

VIDEO_DIR = "/home/noams/src/plai_cv/data/decorte/rallies"
OUTPUT_DIR = "/home/noams/src/plai_cv/output/validation"

RALLIES_CSV = "/home/noams/src/plai_cv/data/decorte/metadata/rallies.csv"
HITS_CSV = "/home/noams/src/plai_cv/data/decorte/metadata/hits.csv"
HIT_ASSIGNMENTS_XLSX = "/home/noams/src/plai_cv/data/decorte/metadata/hit_assignments.xlsx"

assert os.path.isdir(VIDEO_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
assert os.path.isfile(HIT_ASSIGNMENTS_XLSX)
assert os.path.isfile(RALLIES_CSV)
assert os.path.isfile(HITS_CSV)
