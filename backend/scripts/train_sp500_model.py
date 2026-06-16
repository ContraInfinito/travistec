"""Train a lightweight S&P-like model using the provided all_stocks_5yr.csv.

This script aggregates by date (mean close across tickers) to create a proxy S&P index,
creates lag features and rolling means, then generates supervised examples for horizons
and trains a RandomForestRegressor. Saves package to backend/models/sp500_model.joblib
containing {'model','feature_cols','last_date'}.
"""
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')

CHARTS_DIR = Path(__file__).parent.parent.parent / 'docs' / 'charts'


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets' / 'sp500' / 'all_stocks_5yr.csv'
OUT = ROOT / 'models' / 'sp500_model.joblib'


def main():
    if not DATA.exists():
        print(f"Dataset not found: {DATA}. Skipping training.")
        return

    print('Loading S&P dataset (this may be large)...')
    df = pd.read_csv(DATA, parse_dates=['date'])
    # aggregate by date to produce a proxy index (mean close across tickers)
    df_index = df.groupby('date')['close'].mean().reset_index().rename(
        columns={'close': 'index_close'})
    df_index = df_index.sort_values('date').reset_index(drop=True)

    # create lag features and rolling means
    for lag in [1, 2, 3, 5, 10]:
        df_index[f'close_lag_{lag}'] = df_index['index_close'].shift(lag)

    df_index['rolling_mean_5'] = df_index['index_close'].rolling(5).mean()
    df_index['rolling_mean_20'] = df_index['index_close'].rolling(20).mean()

    # price change and high_low_diff approximations are not available in aggregated data,
    # so compute simple daily change of the proxy index
    df_index['price_change'] = df_index['index_close'].pct_change() * 100.0

    # build supervised examples for horizons
    horizons = [1, 7, 30, 90, 180, 365]
    rows = []
    for i in range(len(df_index)):
        if i < 30:
            continue
        for h in horizons:
            tgt_idx = i + h
            if tgt_idx >= len(df_index):
                continue
            feat = {
                'open': df_index['index_close'].iloc[i] * 0.998,
                'high': df_index['index_close'].iloc[i] * 1.002,
                'low': df_index['index_close'].iloc[i] * 0.995,
                'volume': 1000000 + int(h * 10000),
                'price_change': df_index['price_change'].iloc[i] if not pd.isna(df_index['price_change'].iloc[i]) else 0.0,
                'high_low_diff': (df_index['index_close'].iloc[i]*1.002) - (df_index['index_close'].iloc[i]*0.995),
                'close_lag_1': df_index['close_lag_1'].iloc[i],
                'close_lag_5': df_index['close_lag_5'].iloc[i] if 'close_lag_5' in df_index.columns else df_index['close_lag_1'].iloc[i],
                'close_lag_10': df_index['close_lag_10'].iloc[i] if 'close_lag_10' in df_index.columns else df_index['close_lag_1'].iloc[i],
                'rolling_mean_5': df_index['rolling_mean_5'].iloc[i],
                'rolling_mean_20': df_index['rolling_mean_20'].iloc[i],
                'horizon_days': h,
                'target': df_index['index_close'].iloc[tgt_idx]
            }
            rows.append(feat)

    df_sup = pd.DataFrame(rows).dropna()
    if len(df_sup) < 50:
        print('Not enough supervised examples to train')
        return

    feature_cols = ['open', 'high', 'low', 'volume', 'price_change', 'high_low_diff', 'close_lag_1',
                    'close_lag_5', 'close_lag_10', 'rolling_mean_5', 'rolling_mean_20', 'horizon_days']
    X = df_sup[feature_cols]
    y = df_sup['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=12)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'Test MAE: {mae:.2f}')
    print(f'R2 score: {r2:.4f}')

    print('--- 5-fold Cross-Validation ---')
    cv = cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    base = cross_val_score(DummyRegressor(strategy='mean'),
                           X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'Model CV MAE : {-cv.mean():.2f} +/- {cv.std():.2f}')
    print(f'Baseline MAE : {-base.mean():.2f} +/- {base.std():.2f}')

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(feature_cols)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i]
                       for i in idx], rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances — SP500')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance_sp500.png', dpi=150)
    plt.close()
    print('Saved docs/charts/feature_importance_sp500.png')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    last_date = df_index['date'].iloc[-1]
    pkg = {'model': model, 'feature_cols': feature_cols, 'last_date': last_date}
    joblib.dump(pkg, OUT)
    print(f'Saved sp500 model to {OUT} (last_date={last_date})')


if __name__ == '__main__':
    main()
