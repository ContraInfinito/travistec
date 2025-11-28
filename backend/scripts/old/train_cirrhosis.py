"""Train a Cirrhosis patient outcome prediction model.

Predicts patient status (C=censored, CL=censored due to liver tx, D=death) 
based on clinical features.
"""
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

DATA = Path(__file__).parent.parent / 'datasets' / 'cirrhosis.csv'
MODEL_OUT = Path(__file__).parent.parent / 'models' / 'cirrhosis_model.joblib'

def main():
    if not DATA.exists():
        print(f"Dataset not found: {DATA}. Skipping training.")
        return

    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} patient records")
    
    # Prepare features
    df = df.copy()
    
    # Encode categorical variables
    encoders = {}
    for col in ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']:
        if col in df.columns:
            le = LabelEncoder()
            # Convert to string first to handle mixed types
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    # Handle Stage as numeric if available
    if 'Stage' in df.columns:
        df['Stage'] = pd.to_numeric(df['Stage'], errors='coerce').fillna(0)
    
    # Select numeric and encoded features
    feature_cols = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 
                    'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 
                    'Prothrombin', 'Stage'] + \
                   [f'{col}_encoded' for col in ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 
                                                   'Spiders', 'Edema']]
    
    # Remove cols that don't exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    # Encode target
    le_status = LabelEncoder()
    y = le_status.fit_transform(df['Status'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'\nClassification Report:')
    print(classification_report(y_test, preds, target_names=le_status.classes_))
    print(f'Trained on {len(X_train)} samples, tested on {len(X_test)} samples')
    
    # Save model with encoders
    model_data = {
        'model': model,
        'status_encoder': le_status,
        'feature_encoders': encoders,
        'feature_cols': feature_cols
    }
    
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, MODEL_OUT)
    print(f'\nSaved model to {MODEL_OUT}')

if __name__ == '__main__':
    main()
