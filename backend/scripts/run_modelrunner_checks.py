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
        car = mr.predict('car_model', params={'year': 2015, 'km': 50000})
        print('Car prediction:', car)
    except Exception as e:
        print('Car error:', e)

    try:
        btc = mr.predict('bitcoin_model', params={'years': 1})
        print('Bitcoin prediction:', btc)
    except Exception as e:
        print('Bitcoin error:', e)

    try:
        movie = mr.predict('movie_recommender', params={
                           'top_k': 3, 'genre': 'Drama', 'year': 1994})
        print('Movie prediction:', movie)
    except Exception as e:
        print('Movie error:', e)


if __name__ == '__main__':
    main()
