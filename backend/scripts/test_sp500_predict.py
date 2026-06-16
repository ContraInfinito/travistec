from services.model_runner import ModelRunner
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


def main():
    mr = ModelRunner()
    print('Available models:', mr.get_available_models())
    try:
        res = mr.predict('sp500_model', params={'days': 30})
        print('sp500 predict =>', res)
    except Exception as e:
        print('Error:', e)


if __name__ == '__main__':
    main()
