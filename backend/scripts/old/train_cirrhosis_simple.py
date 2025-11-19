import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

DATA = Path(__file__).parent.parent / 'datasets' / 'cirrhosis.csv'
OUT = Path(__file__).parent.parent / 'models' / 'cirrhosis_classifier.joblib'

def main():
    if not DATA.exists():
        print('cirrhosis dataset missing:', DATA)
        return
    df = pd.read_csv(DATA)
    if 'Result' in df.columns:
        X = df.drop(columns=['Result'])
        y = df['Result']
    else:
        # assume last column is label
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # select numeric columns only for a simple baseline
    X_num = X.select_dtypes(include=['number'])
    if X_num.shape[1] == 0:
        print('No numeric features available for cirrhosis training; aborting')
        return

    # drop NaN targets
    notnull = y.notnull()
    X_num = X_num.loc[notnull]
    y = y.loc[notnull]

    pipe = Pipeline([('clf', RandomForestClassifier(n_estimators=50, random_state=42))])
    pipe.fit(X_num.fillna(0), y)
    joblib.dump(pipe, OUT)
    print('Saved cirrhosis model to', OUT)

if __name__ == '__main__':
    main()
