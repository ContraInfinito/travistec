"""Train a Bitcoin price prediction model using time series data.

Uses historical Bitcoin price data to train a regression model.
"""
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')

CHARTS_DIR = Path(__file__).parent.parent.parent / 'docs' / 'charts'

# The repo provides CSV files under datasets/bitcoin
DATA = Path(__file__).parent.parent / 'datasets' / \
    'bitcoin' / 'bitcoin_price_Training - Training.csv'
MODEL_OUT = Path(__file__).parent.parent / 'models' / 'bitcoin_model.joblib'


def main():
    if not DATA.exists():
        print(f"Dataset not found: {DATA}. Skipping training.")
        return

    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} Bitcoin price records")
    print(f"Columns: {df.columns.tolist()}")

    # Check if we have price data
    price_cols = [col for col in df.columns if 'price' in col.lower()
                  or 'close' in col.lower()]
    if not price_cols:
        print("No price column found in dataset")
        return

    # Use first price column found
    price_col = price_cols[0]
    print(f"Using price column: {price_col}")

    # Create simple features: use rolling averages and lag features
    df = df.copy()
    df['price'] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=['price'])

    # Create lag features and rolling means
    for lag in [1, 2, 3, 7]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)

    df['rolling_mean_7'] = df['price'].rolling(window=7).mean()
    df['rolling_mean_30'] = df['price'].rolling(window=30).mean()

    # We'll create supervised examples for several horizons (days): 1,7,30,90,180,365
    horizons = [1, 7, 30, 90, 180, 365]
    rows = []
    for i in range(len(df)):
        if i < 30:
            continue
        for h in horizons:
            tgt_idx = i + h
            if tgt_idx >= len(df):
                continue
            tgt_price = df.iloc[tgt_idx]['price']
            feat = {
                'price_lag_1': df.iloc[i]['price_lag_1'],
                'price_lag_2': df.iloc[i]['price_lag_2'],
                'price_lag_3': df.iloc[i]['price_lag_3'],
                'price_lag_7': df.iloc[i]['price_lag_7'],
                'rolling_mean_7': df.iloc[i]['rolling_mean_7'],
                'rolling_mean_30': df.iloc[i]['rolling_mean_30'],
                'horizon_days': h,
                'target_price': tgt_price
            }
            rows.append(feat)

    df_sup = pd.DataFrame(rows)
    df_sup = df_sup.dropna()
    if len(df_sup) < 50:
        print('Not enough supervised examples to train')
        return

    feature_cols = ['price_lag_1', 'price_lag_2', 'price_lag_3',
                    'price_lag_7', 'rolling_mean_7', 'rolling_mean_30', 'horizon_days']
    X = df_sup[feature_cols]
    y = df_sup['target_price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f'Test MAE: ${mae:.2f}')
    print(f'R2 Score: {r2:.4f}')
    print(f'Mean price: ${y_test.mean():.2f}')

    print('--- 5-fold Cross-Validation ---')
    cv = cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    base = cross_val_score(DummyRegressor(strategy='mean'),
                           X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'Model CV MAE : ${-cv.mean():.2f} +/- {cv.std():.2f}')
    print(f'Baseline MAE : ${-base.mean():.2f} +/- {base.std():.2f}')

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(feature_cols)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i]
                       for i in idx], rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances — Bitcoin Price')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance_bitcoin.png', dpi=150)
    plt.close()
    print('Saved docs/charts/feature_importance_bitcoin.png')

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    # Save model with metadata: include last date from original df
    last_date = pd.to_datetime(
        df['Date'].iloc[-1]) if 'Date' in df.columns else None
    package = {'model': model, 'feature_cols': feature_cols,
               'last_date': last_date}
    joblib.dump(package, MODEL_OUT)
    print(f'Saved model+metadata to {MODEL_OUT} (last_date={last_date})')


if __name__ == '__main__':
    main()
