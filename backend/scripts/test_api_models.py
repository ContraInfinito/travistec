import app
from fastapi.testclient import TestClient
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


client = TestClient(app.app)


def call(endpoint, payload):
    print(f"POST {endpoint} -> {payload}")
    r = client.post(endpoint, json=payload)
    print(r.status_code, r.json())


def main():
    call('/api/v1/predict/car', {'year': 2015, 'km': 50000})
    call('/api/v1/predict/bitcoin', {'years': 1})
    call('/api/v1/predict/movie', {'top_k': 3, 'genre': 'Drama', 'year': 1994})
    call('/api/v1/models/movie_recommender',
         {'params': {'top_k': 2, 'genre': 'Comedy'}})


if __name__ == '__main__':
    main()
