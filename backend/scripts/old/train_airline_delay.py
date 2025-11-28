"""Train an Airline Delay prediction model.

Predicts whether a flight will be delayed based on various features.
Uses a sample of the data for faster training.
"""
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

DATA = Path(__file__).parent.parent / 'datasets' / 'airline' / 'DelayedFlights.csv'
MODEL_OUT = Path(__file__).parent.parent / 'models' / 'airline_delay_model.joblib'

def main():
    if not DATA.exists():
        print(f"Dataset not found: {DATA}. Skipping training.")
        return

    print(f"Loading airline delay data...")
    # Read only first 100k rows for faster training
    df = pd.read_csv(DATA, nrows=100000)
    print(f"Loaded {len(df)} flight records (sample)")
    
    # Create binary target: delayed or not
    if 'ArrDelay' in df.columns:
        df['Delayed'] = (df['ArrDelay'] > 15).astype(int)  # Delayed if > 15 minutes
    elif 'DepDelay' in df.columns:
        df['Delayed'] = (df['DepDelay'] > 15).astype(int)
    else:
        print("No delay column found")
        return
    
    # Select useful features
    potential_features = ['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 
                         'CRSDepTime', 'CRSArrTime', 'Distance']
    
    feature_cols = [col for col in potential_features if col in df.columns]
    
    # Encode categorical if needed
    encoders = {}
    for col in ['UniqueCarrier', 'Origin', 'Dest']:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            feature_cols.append(f'{col}_encoded')
            encoders[col] = le
    
    X = df[feature_cols].fillna(0)
    y = df['Delayed']
    
    # Remove rows with NaN target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"Training with {len(X)} samples")
    print(f"Delayed flights: {y.sum()} ({y.mean()*100:.1f}%)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    print(f'\nTest Accuracy: {accuracy:.4f}')
    print(f'\nClassification Report:')
    print(classification_report(y_test, preds, target_names=['On-time', 'Delayed']))
    
    # Save model with encoders
    model_data = {
        'model': model,
        'encoders': encoders,
        'feature_cols': feature_cols
    }
    
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, MODEL_OUT)
    print(f'\nSaved model to {MODEL_OUT}')

if __name__ == '__main__':
    main()
