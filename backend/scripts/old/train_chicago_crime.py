"""
Script de entrenamiento para modelo de predicción de crímenes en Chicago
Predice la cantidad de crímenes por día de la semana

Dataset: Sintético basado en patrones históricos de crímenes en Chicago
Modelo: Random Forest Classifier
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def generate_chicago_crime_data(n_samples=10000):
    """
    Genera datos sintéticos de crímenes en Chicago
    Basado en patrones históricos conocidos
    """
    np.random.seed(42)
    
    # Días de la semana (0=Lunes, 6=Domingo)
    days = np.random.randint(0, 7, n_samples)
    
    # Patrones de crímenes por día (basado en estadísticas reales)
    # Chicago tiene más crímenes los fines de semana
    crime_patterns = {
        0: (420, 80),  # Lunes - media 420, std 80
        1: (450, 85),  # Martes
        2: (430, 75),  # Miércoles
        3: (460, 90),  # Jueves
        4: (520, 100), # Viernes - aumenta
        5: (580, 120), # Sábado - pico
        6: (550, 110)  # Domingo - alto
    }
    
    crimes = []
    for day in days:
        mean, std = crime_patterns[day]
        # Agregar variación temporal y estacional
        temporal_factor = np.random.uniform(0.85, 1.15)
        crime_count = int(np.random.normal(mean, std) * temporal_factor)
        crime_count = max(crime_count, 0)  # No valores negativos
        crimes.append(crime_count)
    
    df = pd.DataFrame({
        'day_of_week': days,
        'crime_count': crimes
    })
    
    return df

def train_chicago_crime_model():
    """
    Entrena modelo de Random Forest para predecir crímenes en Chicago
    """
    print("=" * 70)
    print("🚔 ENTRENAMIENTO: Modelo de Predicción de Crímenes en Chicago")
    print("=" * 70)
    
    # Generar datos
    print("\n[1/5] Generando datos sintéticos...")
    df = generate_chicago_crime_data(n_samples=10000)
    
    print(f"   ✓ Dataset generado: {len(df)} registros")
    print(f"\n📊 Estadísticas por día:")
    print(df.groupby('day_of_week')['crime_count'].describe()[['mean', 'std', 'min', 'max']])
    
    # Preparar features
    print("\n[2/5] Preparando features...")
    X = df[['day_of_week']].values
    y = df['crime_count'].values
    
    print(f"   ✓ Features: {X.shape}")
    print(f"   ✓ Target: {y.shape}")
    
    # Split
    print("\n[3/5] Dividiendo train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   ✓ Train: {len(X_train)} muestras")
    print(f"   ✓ Test: {len(X_test)} muestras")
    
    # Entrenar
    print("\n[4/5] Entrenando Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("   ✓ Modelo entrenado")
    
    # Evaluar
    print("\n[5/5] Evaluando modelo...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)
    
    print(f"\n📈 MÉTRICAS DE EVALUACIÓN:")
    print(f"   MAE (Train): {mae_train:.2f} crímenes")
    print(f"   MAE (Test):  {mae_test:.2f} crímenes")
    print(f"   RMSE (Test): {rmse_test:.2f} crímenes")
    print(f"   R² (Test):   {r2_test:.4f}")
    
    # Predicciones por día
    print(f"\n🔮 PREDICCIONES POR DÍA DE LA SEMANA:")
    days_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    for day_idx in range(7):
        pred = model.predict([[day_idx]])[0]
        actual_mean = df[df['day_of_week'] == day_idx]['crime_count'].mean()
        print(f"   {days_names[day_idx]:10s}: {pred:.0f} crímenes (real: {actual_mean:.0f})")
    
    # Guardar modelo
    print("\n💾 Guardando modelo...")
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'chicago_crime.joblib'
    
    joblib.dump(model, model_path)
    print(f"   ✓ Modelo guardado: {model_path}")
    print(f"   ✓ Tamaño: {model_path.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO: chicago_crime.joblib")
    print("=" * 70)
    
    return model, {
        'mae_train': mae_train,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'n_samples': len(df)
    }

if __name__ == '__main__':
    model, metrics = train_chicago_crime_model()
    
    print("\n🎯 RESUMEN FINAL:")
    print(f"   - Modelo: Random Forest Regressor")
    print(f"   - Dataset: {metrics['n_samples']} registros sintéticos")
    print(f"   - MAE: {metrics['mae_test']:.2f} crímenes")
    print(f"   - R²: {metrics['r2_test']:.4f}")
    print(f"   - Uso: Predecir crímenes en Chicago por día de la semana")
    print(f"   - Input: día_semana (0=Lunes, 6=Domingo)")
    print(f"   - Output: Cantidad estimada de crímenes")
    print()
