import pandas as pd
from pathlib import Path
import joblib

DATA = Path(__file__).parent.parent / 'datasets' / 'ratings.csv'
OUT = Path(__file__).parent.parent / 'models' / 'movie_recommender.joblib'

def main():
    if not DATA.exists():
        print('ratings dataset missing:', DATA)
        return
    df = pd.read_csv(DATA)
    # simple popularity model: top N movies by average rating
    if not {'movieId','rating'}.issubset(df.columns):
        print('ratings.csv missing expected columns movieId,rating')
        return
    pop = df.groupby('movieId')['rating'].mean().sort_values(ascending=False)
    model = {'top_movies': pop.head(50).index.tolist()}
    joblib.dump(model, OUT)
    print('Saved simple movie_recommender to', OUT)

if __name__ == '__main__':
    main()
