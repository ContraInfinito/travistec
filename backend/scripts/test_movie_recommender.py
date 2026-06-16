from services.model_runner import ModelRunner
import sys
from pathlib import Path
# ensure backend/ is on sys.path so we can import services
BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


def main():
    mr = ModelRunner()
    print("Available models:", mr.get_available_models())
    try:
        res = mr.predict('movie_recommender', params={
                         'top_k': 5, 'genre': 'Drama', 'year': 1994})
        print('Result:', res)
    except Exception as e:
        print('Error calling movie_recommender:', e)


if __name__ == '__main__':
    main()
