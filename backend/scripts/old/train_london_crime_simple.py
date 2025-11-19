import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib

DATA = Path(__file__).parent.parent / 'datasets' / 'london_crime_by_lsoa.csv'
OUT = Path(__file__).parent.parent / 'models' / 'london_crime.joblib'

def main():
    if not DATA.exists():
        print('london crime dataset missing:', DATA)
        return
    df = pd.read_csv(DATA)
    # if there is a 'count' or similar, use it; otherwise use last numeric column
    numeric = df.select_dtypes(include=['number']).columns
    if len(numeric) < 1:
        print('no numeric columns found in london crime dataset')
        return
    target = numeric[-1]
    X = df[numeric[:-1]] if len(numeric) > 1 else df[[numeric[0]]]
    y = df[target]
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X.fillna(0), y)
    joblib.dump(model, OUT)
    print('Saved london_crime model to', OUT)

if __name__ == '__main__':
    main()
