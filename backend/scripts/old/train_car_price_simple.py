import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

DATA = Path(__file__).parent.parent / 'datasets' / 'car data.txt'
OUT = Path(__file__).parent.parent / 'models' / 'car_price.joblib'

def main():
    if not DATA.exists():
        print('car dataset missing:', DATA)
        return
    df = pd.read_csv(DATA)
    # target: Selling_Price
    X = df.drop(columns=['Selling_Price'])
    y = df['Selling_Price']
    # simple preprocessing: encode categorical
    cat = X.select_dtypes(include=['object']).columns.tolist()
    num = X.select_dtypes(include=['int64','float64']).columns.tolist()
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
    ], remainder='passthrough')

    pipe = Pipeline([
        ('pre', pre),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    pipe.fit(X, y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT)
    print('Saved car_price model to', OUT)

if __name__ == '__main__':
    main()
