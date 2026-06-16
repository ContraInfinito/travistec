from services.emotion_deepface import analyze_image_file
import sys
from pathlib import Path
import json

# import from backend root
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def main(paths):
    for p in paths:
        res = analyze_image_file(p)
        print("==>", p)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/test_emotions_deepface.py <image1> [<image2> ...]")
        sys.exit(1)
    main(sys.argv[1:])
