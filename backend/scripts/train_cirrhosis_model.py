"""Train a classifier to predict cirrhosis Stage (1-4) from clinical features.

Saves model and label encoder to backend/models/cirrhosis_model.joblib
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

CHARTS_DIR = Path(__file__).resolve().parents[2] / 'docs' / 'charts'

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'datasets' / 'cirrosis' / 'cirrhosis.csv'
OUT = ROOT / 'models' / 'cirrhosis_model.joblib'


def to_numeric(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def bool_map(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper()
    if s in ('Y','YES','1','TRUE'):
        return 1
    if s in ('N','NO','0','FALSE'):
        return 0
    return np.nan


def train():
    if not CSV.exists():
        print(f"CSV not found: {CSV}")
        return
    print('Loading cirrhosis dataset...')
    df = pd.read_csv(CSV)

    # Target: Stage (values like 1.0..4.0) - drop missing
    df['Stage'] = pd.to_numeric(df['Stage'], errors='coerce')
    df = df.dropna(subset=['Stage'])
    df['Stage'] = df['Stage'].astype(int).astype(str)  # as categorical strings for LabelEncoder

    # Features to use
    cols_bool = ['Ascites','Hepatomegaly','Spiders','Edema']
    cols_num = ['N_Days','Age','Bilirubin','Cholesterol','Albumin','Copper','Alk_Phos','SGOT','Tryglicerides','Platelets','Prothrombin']

    # Clean and convert
    for c in cols_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    for c in cols_bool:
        if c in df.columns:
            df[c+'_bin'] = df[c].apply(bool_map)
        else:
            df[c+'_bin'] = np.nan

    # Sex and Drug as categorical
    df['Sex'] = df['Sex'].astype(str).fillna('U')
    df['Drug'] = df['Drug'].astype(str).fillna('Unknown')

    # Fill numeric NaNs with median
    for c in cols_num:
        med = df[c].median()
        df[c] = df[c].fillna(med)

    for c in cols_bool:
        df[c+'_bin'] = df[c+'_bin'].fillna(0)

    # Label encode Sex and Drug
    le_sex = LabelEncoder(); df['Sex_le'] = le_sex.fit_transform(df['Sex'])
    le_drug = LabelEncoder(); df['Drug_le'] = le_drug.fit_transform(df['Drug'])

    feature_cols = cols_num + [c+'_bin' for c in cols_bool] + ['Sex_le','Drug_le']
    X = df[feature_cols].astype(float)
    y = df['Stage']

    le_target = LabelEncoder(); y_enc = le_target.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    print('Training RandomForestClassifier...')
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    print(f'Accuracy: {acc:.4f}, F1(macro): {f1:.4f}')
    print('Classification report:')
    print(classification_report(y_test, preds, target_names=le_target.classes_))

    print('--- 5-fold Cross-Validation ---')
    cv = cross_val_score(clf, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    base = cross_val_score(DummyClassifier(strategy='most_frequent'), X, y_enc, cv=5, scoring='accuracy')
    print(f'Model CV Accuracy : {cv.mean():.4f} +/- {cv.std():.4f}')
    print(f'Baseline Accuracy : {base.mean():.4f} +/- {base.std():.4f}')

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(feature_cols)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in idx], rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances — Cirrhosis Stage')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance_cirrhosis.png', dpi=150)
    plt.close()
    print('Saved docs/charts/feature_importance_cirrhosis.png')

    pkg = {
        'model': clf,
        'feature_cols': feature_cols,
        'le_target': le_target,
        'encoders': {'sex': le_sex, 'drug': le_drug}
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pkg, OUT)
    print(f'Saved cirrhosis model to {OUT}')


if __name__ == '__main__':
    train()
