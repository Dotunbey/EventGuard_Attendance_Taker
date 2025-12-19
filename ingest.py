import face_recognition
import cv2
import pickle
import numpy as np
from src.config import ASSETS_DIR, DATA_DIR, DB_PATH
from src.utils import save_db

def run_ingestion():
    if not ASSETS_DIR.exists():
        ASSETS_DIR.mkdir(parents=True)
        return

    DATA_DIR.mkdir(exist_ok=True)
    known_encodings = []
    known_names = []
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in ASSETS_DIR.iterdir() if p.suffix.lower() in valid_extensions]
    
    print(f"🚀 Processing {len(images)} images (Grayscale Mode)...")

    for img_path in images:
        name = img_path.stem.replace("_", " ").title()
        
        # 1. Load using OpenCV (Standard BGR)
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"❌ Could not read {name}")
            continue

        try:
            # 2. Convert to GRAYSCALE (1 Channel)
            # This bypasses the RGB memory layout issues causing the crash
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 3. Detect Faces on the Grayscale image
            # HOG detector works perfectly on grayscale
            boxes = face_recognition.face_locations(gray, model="hog")
            
            # 4. For encoding, we still need RGB, so we convert back just for this step
            # We assume if detection worked, the memory is now "warmed up" safe
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb, boxes)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"✅ Indexed: {name}")
            else:
                print(f"⚠️  Skipped: {name} (No face detected)")

        except Exception as e:
            print(f"❌ Error on {name}: {e}")
            # Debugging info
            if 'gray' in locals():
                print(f"   Image Shape: {gray.shape}, Type: {gray.dtype}")

    if known_names:
        save_db({"encodings": known_encodings, "names": known_names}, DB_PATH)
        print(f"\n🎉 Database compiled. Total: {len(known_names)}")
    else:
        print("\n⚠️ Database empty.")

if __name__ == "__main__":
    run_ingestion()
