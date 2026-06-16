from pathlib import Path
from services.emotion_local_simple import analyze_image_file

IMG = Path(__file__).parent.parent / 'datasets' / 'image1.webp'


def main():
    if not IMG.exists():
        print('Test image not found:', IMG)
        return

    res = analyze_image_file(IMG)
    print('Emotion analysis:', res)


if __name__ == '__main__':
    main()
