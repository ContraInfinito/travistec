"""Train a Bitcoin price prediction model on live yfinance data.

The deployed inference path (services/model_runner.py) builds lag/rolling
features from *live* BTC-USD prices via yfinance. This trainer therefore sources
the same live data so the model's training range matches what it sees at
inference time — a RandomForest cannot extrapolate, so training on stale
(2013-era) prices would make live predictions meaningless.

Outputs:
  - 5-fold CV vs. a mean baseline
  - a time-based 90-day backtest (1-day-ahead MAE / MAPE on a held-out window)
  - feature-importance chart in docs/charts/
  - the saved model package backend/models/bitcoin_model.joblib
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
# CSV fallback (stale 2013 data) used only if yfinance is unavailable.
CSV_FALLBACK = Path(__file__).parent.parent / 'datasets' / \
    'bitcoin' / 'bitcoin_price_Training - Training.csv'
MODEL_OUT = Path(__file__).parent.parent / 'models' / 'bitcoin_model.joblib'

TICKER = 'BTC-USD'
HORIZONS = [1, 7, 30, 90, 180, 365]
FEATURE_COLS = ['price_lag_1', 'price_lag_2', 'price_lag_3',
                'price_lag_7', 'rolling_mean_7', 'rolling_mean_30', 'horizon_days']


def load_prices():
    """Return a DataFrame with columns ['Date', 'price'] sorted ascending.

    Prefers live yfinance data; falls back to the bundled CSV.
    """
    try:
        import yfinance as yf
        data = yf.download(TICKER, period='8y', interval='1d',
                           auto_adjust=True, progress=False)
        if data is not None and len(data) > 100:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            out = data.reset_index()[['Date', 'Close']].rename(
                columns={'Close': 'price'})
            out['price'] = pd.to_numeric(out['price'], errors='coerce')
            out = out.dropna(subset=['price']).reset_index(drop=True)
            print(f"Loaded {len(out)} BTC-USD daily records from yfinance "
                  f"({out['Date'].iloc[0].date()} -> {out['Date'].iloc[-1].date()})")
            return out
        print("[WARN] yfinance returned too little data; falling back to CSV.")
    except Exception as e:
        print(f"[WARN] yfinance unavailable ({e}); falling back to CSV.")

    if not CSV_FALLBACK.exists():
        return None
    df = pd.read_csv(CSV_FALLBACK)
    price_col = next((c for c in df.columns if 'price' in c.lower()
                      or 'close' in c.lower()), None)
    if price_col is None:
        return None
    df['price'] = pd.to_numeric(df[price_col], errors='coerce')
    df['Date'] = pd.to_datetime(df.get('Date'), errors='coerce')
    df = df.dropna(subset=['price']).sort_values('Date').reset_index(drop=True)
    print(f"Loaded {len(df)} records from CSV fallback (stale).")
    return df[['Date', 'price']]


def add_features(df):
    df = df.sort_values('Date').reset_index(drop=True).copy()
    for lag in [1, 2, 3, 7]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    df['rolling_mean_7'] = df['price'].rolling(7).mean()
    df['rolling_mean_30'] = df['price'].rolling(30).mean()
    return df


def build_supervised(df, horizons):
    rows = []
    for i in range(len(df)):
        if i < 30:
            continue
        for h in horizons:
            tgt_idx = i + h
            if tgt_idx >= len(df):
                continue
            rows.append({
                'date': df.iloc[i]['Date'],
                'price_lag_1': df.iloc[i]['price_lag_1'],
                'price_lag_2': df.iloc[i]['price_lag_2'],
                'price_lag_3': df.iloc[i]['price_lag_3'],
                'price_lag_7': df.iloc[i]['price_lag_7'],
                'rolling_mean_7': df.iloc[i]['rolling_mean_7'],
                'rolling_mean_30': df.iloc[i]['rolling_mean_30'],
                'horizon_days': h,
                'target_price': df.iloc[tgt_idx]['price'],
            })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def backtest_90d(df):
    """Time-based 1-day-ahead backtest over the last 90 days."""
    sup = build_supervised(df, horizons=[1])
    if len(sup) < 120:
        print("[backtest] not enough history for a 90-day backtest")
        return None
    cutoff = sup['date'].max() - pd.Timedelta(days=90)
    train = sup[sup['date'] <= cutoff]
    test = sup[sup['date'] > cutoff]
    if len(test) < 10 or len(train) < 50:
        print("[backtest] insufficient train/test split")
        return None
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12)
    model.fit(train[FEATURE_COLS], train['target_price'])
    preds = model.predict(test[FEATURE_COLS])
    actual = test['target_price'].values
    mae = mean_absolute_error(actual, preds)
    mape = float(np.mean(np.abs((actual - preds) / actual)) * 100)
    print(f"\n=== 90-day backtest (1-day-ahead, {len(test)} days) ===")
    print(f"Backtest MAE  : ${mae:,.2f}")
    print(f"Backtest MAPE : {mape:.2f}%")
    print(f"Mean BTC price in window: ${actual.mean():,.2f}")
    return {'mae': mae, 'mape': mape, 'n': len(test)}


def main():
    df = load_prices()
    if df is None or len(df) < 120:
        print("No usable price data. Skipping training.")
        return

    df = add_features(df)

    # Time-based backtest first (honest out-of-sample estimate)
    backtest_90d(df)

    # Full supervised set across all horizons for the deployed model
    df_sup = build_supervised(df, HORIZONS)
    if len(df_sup) < 50:
        print('Not enough supervised examples to train')
        return

    X = df_sup[FEATURE_COLS]
    y = df_sup['target_price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f'\nTest MAE: ${mean_absolute_error(y_test, preds):,.2f}')
    print(f'R2 Score: {r2_score(y_test, preds):.4f}')

    print('--- 5-fold Cross-Validation ---')
    cv = cross_val_score(model, X, y, cv=5,
                         scoring='neg_mean_absolute_error', n_jobs=-1)
    base = cross_val_score(DummyRegressor(strategy='mean'),
                           X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'Model CV MAE : ${-cv.mean():,.2f} +/- {cv.std():,.2f}')
    print(f'Baseline MAE : ${-base.mean():,.2f} +/- {base.std():,.2f}')

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(FEATURE_COLS)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in idx],
                       rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances — Bitcoin Price')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance_bitcoin.png', dpi=150)
    plt.close()
    print('Saved docs/charts/feature_importance_bitcoin.png')

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    package = {'model': model, 'feature_cols': FEATURE_COLS,
               'last_date': last_date}
    joblib.dump(package, MODEL_OUT)
    print(f'Saved model+metadata to {MODEL_OUT} (last_date={last_date.date()})')


if __name__ == '__main__':
    main()
