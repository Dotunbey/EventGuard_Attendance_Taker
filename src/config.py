from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets" / "guest_photos"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "encodings.pkl"
LOG_PATH = DATA_DIR / "access_log.csv"

# CV Constants
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 2
FRAME_RESIZE_SCALE = 0.25
TOLERANCE = 0.5
