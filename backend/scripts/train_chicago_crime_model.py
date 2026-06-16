"""
Train a Chicago crime model by querying the public BigQuery dataset using bq_helper.

This script expects `bq_helper` to be installed in the environment and that the user
has configured Google Cloud authentication (e.g. set GOOGLE_APPLICATION_CREDENTIALS
or have `gcloud` configured locally).

It queries `bigquery-public-data.chicago_crime` to fetch recent incidents, aggregates
counts per day (or per community area) and trains a RandomForestRegressor to predict
expected incidents for a given day_of_week/month/community_area.

If you don't have BigQuery access, adapt the query or export the dataset to CSV and
point the script to a local file.

Example run (PowerShell):
  cd backend
  .\.venv\Scripts\Activate.ps1
  python .\scripts\train_chicago_crime_model.py

"""
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models' / 'chicago_crime.joblib'

try:
    import bq_helper
    from bq_helper import BigQueryHelper
except Exception as e:
    print('bq_helper not installed or import failed:', e)
    print('Install with: pip install bq_helper')
    sys.exit(1)

matplotlib.use('Agg')

CHARTS_DIR = Path(__file__).parent.parent.parent / 'docs' / 'charts'


def query_chicago_sample(bqh: BigQueryHelper, limit_years: int = 3):
    # Query a few recent years to keep the dataset manageable
    sql = f"""
    SELECT
      CAST(`date` AS DATE) as occ_date,
      EXTRACT(YEAR FROM `date`) as year,
      EXTRACT(MONTH FROM `date`) as month,
      EXTRACT(DAYOFWEEK FROM `date`) as day_of_week, -- 1=Sunday
      community_area,
      COUNT(1) as incident_count
    FROM `bigquery-public-data.chicago_crime.crime`
    WHERE EXTRACT(YEAR FROM `date`) >= (EXTRACT(YEAR FROM CURRENT_DATE()) - {limit_years})
    GROUP BY occ_date, year, month, day_of_week, community_area
    ORDER BY occ_date DESC
    LIMIT 200000
    """
    print('Running BigQuery. This may take a while...')
    df = bqh.query_to_pandas_safe(sql)
    return df


def prepare_features(df: pd.DataFrame):
    # community_area can be null; fill with -1
    df['community_area'] = df['community_area'].fillna(-1).astype(int)
    # day_of_week: convert to 0..6 (BigQuery uses 1=Sunday)
    df['dow0'] = ((df['day_of_week'] - 1) % 7).astype(int)
    # Features: dow0, month, community_area
    X = df[['dow0', 'month', 'community_area']].astype(float)
    y = df['incident_count'].astype(float)
    return X, y


def train():
    bqh = BigQueryHelper(active_project="bigquery-public-data",
                         dataset_name="chicago_crime")

    # Sanity: list tables
    try:
        print('Available tables:', bqh.list_tables())
    except Exception as e:
        print('Error listing tables (check credentials):', e)

    df = query_chicago_sample(bqh, limit_years=3)
    if df is None or df.shape[0] == 0:
        print('No data returned from BigQuery. Exiting.')
        return

    print('Rows returned:', df.shape[0])
    X, y = prepare_features(df)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print('Training RandomForestRegressor...')
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'Test MAE: {mae:.3f}, R2: {r2:.3f}')

    feature_cols = ['dow0', 'month', 'community_area']
    print('--- 5-fold Cross-Validation ---')
    cv = cross_val_score(
        rf, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    base = cross_val_score(DummyRegressor(strategy='mean'),
                           X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'Model CV MAE : {-cv.mean():.3f} +/- {cv.std():.3f}')
    print(f'Baseline MAE : {-base.mean():.3f} +/- {base.std():.3f}')

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(feature_cols)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i]
                       for i in idx], rotation=0, ha='center', fontsize=10)
    ax.set_title('Feature Importances — Chicago Crime')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance_chicago_crime.png', dpi=150)
    plt.close()
    print('Saved docs/charts/feature_importance_chicago_crime.png')

    pkg = {
        'model': rf,
        'feature_cols': feature_cols
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pkg, OUT)
    print('Saved chicago crime model to', OUT)


if __name__ == '__main__':
    train()
