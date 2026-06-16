"""Train an S&P 500 (^GSPC) price model on live yfinance data.

Mirrors services/model_runner.py's live inference: it pulls ^GSPC OHLCV from
yfinance and builds the exact same features, so the model's training range
matches what it sees at inference time. (The previous version trained on a
mean-of-stocks proxy from a 2018 CSV — a different scale entirely — which made
live predictions meaningless.)

Outputs: 5-fold CV vs. baseline, a 90-day backtest, a feature-importance chart,
and backend/models/sp500_model.joblib.
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
OUT = Path(__file__).parent.parent / 'models' / 'sp500_model.joblib'

TICKER = '^GSPC'
HORIZONS = [1, 7, 30, 90, 180, 365]
FEATURE_COLS = ['open', 'high', 'low', 'volume', 'price_change', 'high_low_diff',
                'close_lag_1', 'close_lag_5', 'close_lag_10',
                'rolling_mean_5', 'rolling_mean_20', 'horizon_days']


def load_ohlcv():
    import yfinance as yf
    data = yf.download(TICKER, period='10y', interval='1d',
                       auto_adjust=True, progress=False)
    if data is None or len(data) < 200:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['Date', 'open', 'high', 'low', 'close', 'volume']
    df = df.dropna().sort_values('Date').reset_index(drop=True)
    print(f"Loaded {len(df)} {TICKER} daily records from yfinance "
          f"({df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()})")
    return df


def add_features(df):
    df = df.sort_values('Date').reset_index(drop=True).copy()
    df['price_change'] = df['close'] - df['open']
    df['high_low_diff'] = df['high'] - df['low']
    for lag in [1, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    df['rolling_mean_5'] = df['close'].rolling(5).mean()
    df['rolling_mean_20'] = df['close'].rolling(20).mean()
    return df


def build_supervised(df, horizons):
    rows = []
    for i in range(len(df)):
        if i < 20:
            continue
        for h in horizons:
            tgt_idx = i + h
            if tgt_idx >= len(df):
                continue
            r = df.iloc[i]
            rows.append({
                'date': r['Date'],
                'open': r['open'], 'high': r['high'], 'low': r['low'],
                'volume': r['volume'], 'price_change': r['price_change'],
                'high_low_diff': r['high_low_diff'],
                'close_lag_1': r['close_lag_1'], 'close_lag_5': r['close_lag_5'],
                'close_lag_10': r['close_lag_10'],
                'rolling_mean_5': r['rolling_mean_5'],
                'rolling_mean_20': r['rolling_mean_20'],
                'horizon_days': h,
                'target': df.iloc[tgt_idx]['close'],
            })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def backtest_90d(df):
    sup = build_supervised(df, horizons=[1])
    if len(sup) < 120:
        print("[backtest] not enough history")
        return None
    cutoff = sup['date'].max() - pd.Timedelta(days=90)
    train, test = sup[sup['date'] <= cutoff], sup[sup['date'] > cutoff]
    if len(test) < 10 or len(train) < 50:
        print("[backtest] insufficient split")
        return None
    m = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12)
    m.fit(train[FEATURE_COLS], train['target'])
    preds = m.predict(test[FEATURE_COLS])
    actual = test['target'].values
    mae = mean_absolute_error(actual, preds)
    mape = float(np.mean(np.abs((actual - preds) / actual)) * 100)
    print(f"\n=== 90-day backtest (1-day-ahead, {len(test)} days) ===")
    print(f"Backtest MAE  : {mae:,.2f}")
    print(f"Backtest MAPE : {mape:.2f}%")
    print(f"Mean ^GSPC in window: {actual.mean():,.2f}")
    return {'mae': mae, 'mape': mape, 'n': len(test)}


def main():
    df = load_ohlcv()
    if df is None or len(df) < 200:
        print("No usable ^GSPC data. Skipping training.")
        return

    df = add_features(df)
    backtest_90d(df)

    df_sup = build_supervised(df, HORIZONS)
    if len(df_sup) < 50:
        print('Not enough supervised examples to train')
        return

    X, y = df_sup[FEATURE_COLS], df_sup['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f'\nTest MAE: {mean_absolute_error(y_test, preds):,.2f}')
    print(f'R2 score: {r2_score(y_test, preds):.4f}')

    print('--- 5-fold Cross-Validation ---')
    cv = cross_val_score(model, X, y, cv=5,
                         scoring='neg_mean_absolute_error', n_jobs=-1)
    base = cross_val_score(DummyRegressor(strategy='mean'),
                           X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'Model CV MAE : {-cv.mean():,.2f} +/- {cv.std():,.2f}')
    print(f'Baseline MAE : {-base.mean():,.2f} +/- {base.std():,.2f}')

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(FEATURE_COLS)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in idx],
                       rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances — S&P 500')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance_sp500.png', dpi=150)
    plt.close()
    print('Saved docs/charts/feature_importance_sp500.png')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    joblib.dump({'model': model, 'feature_cols': FEATURE_COLS,
                 'last_date': last_date}, OUT)
    print(f'Saved sp500 model to {OUT} (last_date={last_date.date()})')


if __name__ == '__main__':
    main()
