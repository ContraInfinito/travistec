"""Train a simple BMI/BodyFat model if dataset exists.

This script is safe to run even if the dataset isn't present; it will exit
with a friendly message.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

CHARTS_DIR = Path(__file__).parent.parent.parent / 'docs' / 'charts'

DATA = Path(__file__).parent.parent / 'datasets' / 'bodymass' / 'bodyfat.csv'
MODEL_OUT = Path(__file__).parent.parent / 'models' / 'bmi_model.joblib'

def main():
    if not DATA.exists():
        print(f"Dataset not found: {DATA}. Skipping training.")
        return

    df = pd.read_csv(DATA)
    # Expecting columns: Age, Weight, Height, BodyFat (real column names)
    if not {'Age','Weight','Height','BodyFat'}.issubset(df.columns):
        print("Dataset doesn't contain required columns. Expected Age, Weight, Height, BodyFat")
        return

    feature_cols = ['Height', 'Weight', 'Age']
    X = df[feature_cols]
    y = df['BodyFat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print('Test MAE:', mean_absolute_error(y_test, preds))
    print('R2:', r2_score(y_test, preds))

    print('--- 5-fold Cross-Validation ---')
    cv = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    base = cross_val_score(DummyRegressor(strategy='mean'), X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'Model CV MAE : {-cv.mean():.3f} +/- {cv.std():.3f}')
    print(f'Baseline MAE : {-base.mean():.3f} +/- {base.std():.3f}')

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(feature_cols)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in idx], rotation=0, ha='center', fontsize=10)
    ax.set_title('Feature Importances — BMI / Body Fat')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance_bmi.png', dpi=150)
    plt.close()
    print('Saved docs/charts/feature_importance_bmi.png')

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print('Saved model to', MODEL_OUT)

if __name__ == '__main__':
    main()
