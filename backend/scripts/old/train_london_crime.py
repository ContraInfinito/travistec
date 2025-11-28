"""Train a London Crime prediction/classification model.

Since shapefile processing requires geopandas which may not be installed,
this creates a simple placeholder model that demonstrates the concept.
For production, install geopandas: pip install geopandas
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

MODEL_OUT = Path(__file__).parent.parent / 'models' / 'london_crime_model.joblib'

def main():
    print("Creating London Crime placeholder model...")
    print("(For production use, install geopandas to process shapefile data)")
    
    # Create synthetic data to demonstrate the model structure
    # In production, this would read from st99_d00.shp using geopandas
    np.random.seed(42)
    n_samples = 5000
    
    # Synthetic features: day of week, hour, district (encoded), month
    X = pd.DataFrame({
        'day_of_week': np.random.randint(0, 7, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'district': np.random.randint(0, 33, n_samples),  # 33 London boroughs
        'month': np.random.randint(1, 13, n_samples),
        'population_density': np.random.rand(n_samples) * 100,
    })
    
    # Synthetic target: crime type (0=low, 1=medium, 2=high severity)
    y = np.random.randint(0, 3, n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    print(f'\nTest Accuracy: {accuracy:.4f}')
    print(f'Trained on {len(X_train)} samples, tested on {len(X_test)} samples')
    print('\nNote: This is a placeholder model with synthetic data.')
    print('For real implementation, process st99_d00.shp with geopandas.')
    
    # Save model with metadata
    model_data = {
        'model': model,
        'feature_cols': X.columns.tolist(),
        'crime_classes': ['Low', 'Medium', 'High'],
        'is_placeholder': True
    }
    
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, MODEL_OUT)
    print(f'\nSaved model to {MODEL_OUT}')

if __name__ == '__main__':
    main()
