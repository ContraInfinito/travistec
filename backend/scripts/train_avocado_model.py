"""Train an avocado price model predicting AveragePrice for a given horizon in months.

Creates monthly supervised examples from the weekly dataset by resampling to monthly mean per region.
Saves model package to backend/models/avocado_model.joblib
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

CHARTS_DIR = Path(__file__).parent.parent.parent / 'docs' / 'charts'

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets' / 'avocado' / 'avocado.csv'
OUT = ROOT / 'models' / 'avocado_model.joblib'


def main():
    if not DATA.exists():
        print(f"Dataset not found: {DATA}. Skipping training.")
        return

    print('Loading avocado dataset...')
    df = pd.read_csv(DATA, parse_dates=['Date'])
    # keep conventional type only for simplicity
    df = df[df['type'] == 'conventional']

    # resample to monthly frequency per region (mean price and sums for volumes)
    df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    monthly = df.groupby(['month','region']).agg({
        'AveragePrice':'mean',
        'Total Volume':'sum',
        '4046':'sum','4225':'sum','4770':'sum',
        'Total Bags':'sum','Small Bags':'sum','Large Bags':'sum','XLarge Bags':'sum'
    }).reset_index().rename(columns={'month':'date'})

    monthly = monthly.sort_values(['region','date']).reset_index(drop=True)

    # Create lag/rolling features per region
    rows = []
    for region, g in monthly.groupby('region'):
        g = g.sort_values('date').reset_index(drop=True)
        g['lag_1'] = g['AveragePrice'].shift(1)
        g['lag_3'] = g['AveragePrice'].shift(3)
        g['rolling_3'] = g['AveragePrice'].rolling(3).mean()
        for i in range(len(g)):
            if i < 6:
                continue
            for months_h in [1,3,6,12]:
                tgt_idx = i + months_h
                if tgt_idx >= len(g):
                    continue
                row = {
                    'date': g.loc[i, 'date'],
                    'region': region,
                    'avg_price': g.loc[i, 'AveragePrice'],
                    'total_volume': g.loc[i, 'Total Volume'],
                    'v_4046': g.loc[i, '4046'],
                    'v_4225': g.loc[i, '4225'],
                    'v_4770': g.loc[i, '4770'],
                    'total_bags': g.loc[i, 'Total Bags'],
                    'small_bags': g.loc[i, 'Small Bags'],
                    'large_bags': g.loc[i, 'Large Bags'],
                    'xlarge_bags': g.loc[i, 'XLarge Bags'],
                    'lag_1': g.loc[i, 'lag_1'],
                    'lag_3': g.loc[i, 'lag_3'],
                    'rolling_3': g.loc[i, 'rolling_3'],
                    'horizon_months': months_h,
                    'target_price': g.loc[tgt_idx, 'AveragePrice']
                }
                rows.append(row)

    df_sup = pd.DataFrame(rows).dropna()
    if len(df_sup) < 50:
        print('Not enough supervised examples to train')
        return

    feature_cols = ['avg_price','total_volume','v_4046','v_4225','v_4770','total_bags','small_bags','large_bags','xlarge_bags','lag_1','lag_3','rolling_3','horizon_months']
    X = df_sup[feature_cols]
    y = df_sup['target_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'Test MAE: {mae:.2f}, R2: {r2:.4f}')

    print('--- 5-fold Cross-Validation ---')
    cv = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    base = cross_val_score(DummyRegressor(strategy='mean'), X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'Model CV MAE : {-cv.mean():.3f} +/- {cv.std():.3f}')
    print(f'Baseline MAE : {-base.mean():.3f} +/- {base.std():.3f}')

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(feature_cols)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in idx], rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances — Avocado Price')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance_avocado.png', dpi=150)
    plt.close()
    print('Saved docs/charts/feature_importance_avocado.png')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    last_date = monthly['date'].max()
    pkg = {'model': model, 'feature_cols': feature_cols, 'last_date': last_date}
    joblib.dump(pkg, OUT)
    print(f'Saved avocado model to {OUT} (last_date={last_date})')


if __name__ == '__main__':
    main()
