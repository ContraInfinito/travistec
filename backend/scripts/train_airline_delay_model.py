"""Train a binary classifier to predict whether a flight will arrive >15 minutes late.

Reads datasets/airline/DelayedFlights.csv, samples rows, builds features and trains
a RandomForestClassifier, then saves a package to backend/models/airline_delay_model.joblib
including encoders for categorical fields.
"""
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')

CHARTS_DIR = Path(__file__).parent.parent.parent / 'docs' / 'charts'

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'datasets' / 'airline' / 'DelayedFlights.csv'
OUT = ROOT / 'models' / 'airline_delay_model.joblib'


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def prepare_dataframe(nrows=None, sample_frac=0.05):
    # Read CSV in chunks if large, sample to keep memory small
    usecols = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime',
               'Distance', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'UniqueCarrier', 'Cancelled']
    if nrows:
        df = pd.read_csv(CSV, usecols=usecols, nrows=nrows)
    else:
        df = pd.read_csv(CSV, usecols=usecols)

    # Create label: delayed if ArrDelay > 15, fallback to DepDelay > 15
    df['ArrDelay'] = pd.to_numeric(df['ArrDelay'], errors='coerce')
    df['DepDelay'] = pd.to_numeric(df['DepDelay'], errors='coerce')
    df['delayed'] = ((df['ArrDelay'] > 15) | (df['DepDelay'] > 15)).astype(int)

    # Basic cleaning
    df['CRSDepTime'] = pd.to_numeric(df['CRSDepTime'], errors='coerce')
    df['CRSArrTime'] = pd.to_numeric(df['CRSArrTime'], errors='coerce')
    df['CRSElapsedTime'] = pd.to_numeric(df['CRSElapsedTime'], errors='coerce')
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')

    # Drop very bad rows
    df = df.dropna(subset=['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime',
                   'CRSArrTime', 'CRSElapsedTime', 'Distance', 'Origin', 'Dest', 'UniqueCarrier'])

    # Sample to speed training if big
    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    return df


def train():
    if not CSV.exists():
        print(f"CSV not found: {CSV}")
        return
    print('Preparing dataframe (may take a while)')
    df = prepare_dataframe(sample_frac=0.05)
    print(f'Rows after sample/clean: {len(df)}')

    # Limit number of unique airports: keep top 50 origins/dests, others->OTHER
    top_orig = df['Origin'].value_counts().nlargest(50).index.tolist()
    top_dest = df['Dest'].value_counts().nlargest(50).index.tolist()
    df['Origin_enc'] = df['Origin'].where(df['Origin'].isin(top_orig), 'OTHER')
    df['Dest_enc'] = df['Dest'].where(df['Dest'].isin(top_dest), 'OTHER')

    # Label-encode categorical fields
    le_origin = LabelEncoder()
    le_dest = LabelEncoder()
    le_carrier = LabelEncoder()
    df['Origin_le'] = le_origin.fit_transform(df['Origin_enc'])
    df['Dest_le'] = le_dest.fit_transform(df['Dest_enc'])
    df['Carrier_le'] = le_carrier.fit_transform(df['UniqueCarrier'])

    feature_cols = ['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime',
                    'CRSElapsedTime', 'Distance', 'Origin_le', 'Dest_le', 'Carrier_le']
    X = df[feature_cols].astype(float)
    y = df['delayed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print('Training RandomForestClassifier...')
    clf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42, max_depth=20)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = float('nan')
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    print(
        f'Accuracy: {acc:.4f}, AUC: {auc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

    print('--- 5-fold Cross-Validation ---')
    cv = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    base = cross_val_score(DummyClassifier(
        strategy='most_frequent'), X, y, cv=5, scoring='accuracy')
    print(f'Model CV Accuracy : {cv.mean():.4f} +/- {cv.std():.4f}')
    print(f'Baseline Accuracy : {base.mean():.4f} +/- {base.std():.4f}')

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(feature_cols)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i]
                       for i in idx], rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances — Airline Delay')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance_airline.png', dpi=150)
    plt.close()
    print('Saved docs/charts/feature_importance_airline.png')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    last_date = None
    # Attempt to infer last_date from CSV by reading Year/Month max
    try:
        df_dates = pd.read_csv(CSV, usecols=['Year', 'Month'], nrows=100000)
        max_year = int(df_dates['Year'].max())
        max_month = int(df_dates['Month'].max())
        last_date = pd.to_datetime(f"{max_year}-{max_month}-01")
    except Exception:
        last_date = None

    pkg = {
        'model': clf,
        'feature_cols': feature_cols,
        'encoders': {'origin': le_origin, 'dest': le_dest, 'carrier': le_carrier},
        'last_date': last_date
    }
    joblib.dump(pkg, OUT)
    print(f'Saved model to {OUT} (last_date={last_date})')


if __name__ == '__main__':
    train()
