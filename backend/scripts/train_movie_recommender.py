"""Train a lightweight content/popularity-based movie recommender.

Reads:
  - backend/datasets/movies/movies.csv
  - backend/datasets/movies/ratings.csv

Produces:
  - backend/models/movie_recommender.joblib  (dict with DataFrame 'movies')

The recommender is simple: movies are scored by mean_rating * log(count+1).
We store a DataFrame with title, movieId, year, genres (string), genres_list, popularity.
"""
from sklearn.neighbors import NearestNeighbors
import re
import math
from pathlib import Path
import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'datasets' / 'movies'
OUT_DIR = ROOT / 'models'


def extract_year(title: str):
    if not isinstance(title, str):
        return None
    m = re.search(r"\((\d{4})\)", title)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None


def main():
    movies_fp = DATA_DIR / 'movies.csv'
    ratings_fp = DATA_DIR / 'ratings.csv'

    if not movies_fp.exists():
        print(f"movies.csv not found at: {movies_fp}")
        return
    if not ratings_fp.exists():
        print(f"ratings.csv not found at: {ratings_fp}")
        return

    print("Loading movies...")
    movies = pd.read_csv(movies_fp)
    print(f"Movies: {len(movies)} rows")

    print("Loading ratings (this may take a few seconds)...")
    ratings = pd.read_csv(ratings_fp)
    print(f"Ratings: {len(ratings)} rows")

    # Aggregate ratings: count and mean per movieId
    agg = ratings.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index(
    ).rename(columns={'count': 'rating_count', 'mean': 'rating_mean'})

    df = movies.merge(agg, how='left', left_on='movieId', right_on='movieId')
    df['rating_count'] = df['rating_count'].fillna(0).astype(int)
    df['rating_mean'] = df['rating_mean'].fillna(0.0)

    # extract year and clean title
    df['year'] = df['title'].apply(extract_year)
    df['clean_title'] = df['title'].str.replace(
        r"\s*\(\d{4}\)", '', regex=True).str.strip()

    # genres list and string
    df['genres_list'] = df['genres'].fillna('').apply(
        lambda s: [g.strip() for g in s.split('|') if g.strip()])
    df['genres_str'] = df['genres'].fillna('')

    # popularity score: mean_rating * log10(count+1)
    df['popularity'] = df.apply(lambda r: float(
        r['rating_mean']) * math.log10(max(1, int(r['rating_count'])) + 1), axis=1)

    # fallback popularity for movies with no ratings: small random-ish value based on movieId
    df['popularity'] = df['popularity'].fillna(0.0)

    # sort by popularity desc
    df = df.sort_values('popularity', ascending=False).reset_index(drop=True)

    # Keep only relevant columns
    keep_cols = ['movieId', 'title', 'clean_title', 'year', 'genres_str',
                 'genres_list', 'rating_count', 'rating_mean', 'popularity']
    df_small = df[keep_cols].copy()

    # Ensure output dir exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'movie_recommender.joblib'

    print(f"Saving recommender to: {out_path}")
    joblib.dump({'movies': df_small}, out_path)
    print("Saved. Example top titles:")
    print(df_small.head()[['title', 'year', 'genres_str',
          'rating_count', 'rating_mean', 'popularity']])


if __name__ == '__main__':
    main()
"""Train a simple movie recommendation model using collaborative filtering.

Uses the movies.csv and ratings.csv datasets to build a basic recommendation system.
"""

MOVIES_DATA = Path(__file__).parent.parent / 'datasets' / 'movies.csv'
RATINGS_DATA = Path(__file__).parent.parent / 'datasets' / 'ratings.csv'
MODEL_OUT = Path(__file__).parent.parent / \
    'models' / 'movie_recommender.joblib'


def main():
    if not MOVIES_DATA.exists() or not RATINGS_DATA.exists():
        print(f"Datasets not found. Skipping training.")
        return

    # Load datasets
    movies = pd.read_csv(MOVIES_DATA)
    ratings = pd.read_csv(RATINGS_DATA)

    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings")

    # Create user-item matrix
    user_item_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    # Train KNN model for collaborative filtering
    model = NearestNeighbors(
        metric='cosine', algorithm='brute', n_neighbors=20)
    model.fit(user_item_matrix.T)

    # Save model and metadata
    model_data = {
        'model': model,
        'user_item_matrix': user_item_matrix,
        'movies': movies
    }

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, MODEL_OUT)
    print(f'Saved movie recommender model to {MODEL_OUT}')
    print(f'Model can recommend from {len(movies)} movies')


if __name__ == '__main__':
    main()
