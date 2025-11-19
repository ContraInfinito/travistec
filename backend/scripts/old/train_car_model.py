"""Train a Car price prediction model.

Uses car features to predict selling price.
"""
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

DATA = Path(__file__).parent.parent / 'datasets' / 'car data.txt'
MODEL_OUT = Path(__file__).parent.parent / 'models' / 'car_model.joblib'

def main():
    if not DATA.exists():
        print(f"Dataset not found: {DATA}. Skipping training.")
        return

    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} car records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare features
    df = df.copy()
    
    # Encode categorical variables
    le_fuel = LabelEncoder()
    le_seller = LabelEncoder()
    le_transmission = LabelEncoder()
    
    df['Fuel_Type_encoded'] = le_fuel.fit_transform(df['Fuel_Type'])
    df['Seller_Type_encoded'] = le_seller.fit_transform(df['Seller_Type'])
    df['Transmission_encoded'] = le_transmission.fit_transform(df['Transmission'])
    
    # Select features
    feature_cols = ['Year', 'Present_Price', 'Kms_Driven', 
                    'Fuel_Type_encoded', 'Seller_Type_encoded', 
                    'Transmission_encoded', 'Owner']
    
    X = df[feature_cols]
    y = df['Selling_Price']
    
    # Drop any NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f'Test MAE: ${mae:.2f}')
    print(f'R2 Score: {r2:.4f}')
    print(f'Mean selling price: ${y_test.mean():.2f}')
    print(f'Trained on {len(X_train)} samples, tested on {len(X_test)} samples')
    
    # Save model with encoders
    model_data = {
        'model': model,
        'fuel_encoder': le_fuel,
        'seller_encoder': le_seller,
        'transmission_encoder': le_transmission,
        'feature_cols': feature_cols
    }
    
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, MODEL_OUT)
    print(f'Saved model to {MODEL_OUT}')

if __name__ == '__main__':
    main()
