import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

CHARTS_DIR = Path(__file__).parent.parent.parent / 'docs' / 'charts'


def encode_fuel(x: str) -> int:
    if not isinstance(x, str):
        return 0
    x = x.strip().lower()
    if 'diesel' in x:
        return 1
    if 'cng' in x:
        return 2
    return 0


def encode_seller(x: str) -> int:
    if isinstance(x, str) and x.strip().lower() == 'individual':
        return 1
    return 0


def encode_transmission(x: str) -> int:
    if isinstance(x, str) and 'auto' in x.lower():
        return 1
    return 0


def main():
    repo_root = Path(__file__).parent.parent
    data_path = repo_root / 'datasets' / 'cars' / 'car data.txt'
    models_dir = repo_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Basic cleaning
    df = df.dropna(subset=['Year', 'Present_Price', 'Kms_Driven', 'Selling_Price'])

    # Ensure numeric types
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Present_Price'] = pd.to_numeric(df['Present_Price'], errors='coerce')
    df['Kms_Driven'] = pd.to_numeric(df['Kms_Driven'], errors='coerce')
    df['Selling_Price'] = pd.to_numeric(df['Selling_Price'], errors='coerce')
    df = df.dropna(subset=['Year', 'Present_Price', 'Kms_Driven', 'Selling_Price'])

    # Encode categorical columns into integers matching ModelRunner expectations
    df['fuel_enc'] = df['Fuel_Type'].apply(encode_fuel)
    df['seller_enc'] = df['Seller_Type'].apply(encode_seller)
    df['trans_enc'] = df['Transmission'].apply(encode_transmission)

    # Owner column exists; ensure numeric
    if 'Owner' in df.columns:
        df['Owner'] = pd.to_numeric(df['Owner'], errors='coerce').fillna(0).astype(int)
    else:
        df['Owner'] = 0

    # Feature order required by ModelRunner: [year, present_price, km, fuel_type, seller_type, transmission, owner]
    feature_cols = ['Year', 'Present_Price', 'Kms_Driven', 'fuel_enc', 'seller_enc', 'trans_enc', 'Owner']

    X = df[feature_cols].astype(float).to_numpy()
    y = df['Selling_Price'].astype(float).to_numpy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training RandomForestRegressor on {X_train.shape[0]} samples, {X_train.shape[1]} features")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # compute RMSE without using the 'squared' kw for compatibility
    rmse_train = mean_squared_error(y_train, y_pred_train) ** 0.5
    rmse_test = mean_squared_error(y_test, y_pred_test) ** 0.5

    mae_test = mean_absolute_error(y_test, y_pred_test)
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Test  RMSE: {rmse_test:.4f}")
    print(f"Test  MAE : {mae_test:.4f}")

    print('--- 5-fold Cross-Validation ---')
    cv = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    base = cross_val_score(DummyRegressor(strategy='mean'), X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'Model CV MAE : {-cv.mean():.4f} +/- {cv.std():.4f}')
    print(f'Baseline MAE : {-base.mean():.4f} +/- {base.std():.4f}')

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(feature_cols)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in idx], rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances — Car Price')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance_car.png', dpi=150)
    plt.close()
    print('Saved docs/charts/feature_importance_car.png')

    # Save model to backend/models/car_price.joblib
    out_path = models_dir / 'car_price.joblib'
    joblib.dump(model, out_path)
    print(f"Saved trained model to: {out_path}")


if __name__ == '__main__':
    main()
