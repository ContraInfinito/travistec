import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

DATA = Path(__file__).parent.parent / 'datasets' / 'DelayedFlights.csv'
OUT = Path(__file__).parent.parent / 'models' / 'airline_delay.joblib'

def main():
    if not DATA.exists():
        print('airline dataset missing:', DATA)
        return
    df = pd.read_csv(DATA)
    # create a simple binary target: ARR_DELAY > 15
    if 'ARR_DELAY' in df.columns:
        df = df.dropna(subset=['ARR_DELAY'])
        df['delayed'] = (df['ARR_DELAY'] > 15).astype(int)
    else:
        # try alternate column names
        possible = [c for c in df.columns if 'ARR' in c.upper() and 'DELAY' in c.upper()]
        if possible:
            df = df.dropna(subset=[possible[0]])
            df['delayed'] = (df[possible[0]] > 15).astype(int)
        else:
            print('ARR_DELAY column missing')
            return

    # use numeric columns only
    X = df.select_dtypes(include=['number']).drop(columns=['ARR_DELAY','delayed'], errors='ignore')
    y = df['delayed']

    pipe = Pipeline([('clf', RandomForestClassifier(n_estimators=50, random_state=42))])
    pipe.fit(X.fillna(0), y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT)
    print('Saved airline_delay model to', OUT)

if __name__ == '__main__':
    main()
