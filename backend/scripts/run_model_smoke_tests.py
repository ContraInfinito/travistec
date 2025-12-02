import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

MODEL_MAP = {
    # model_name: dataset_csv (if available)
    'bmi_model': 'bodyfat.csv',
    'bitcoin_model': 'bitcoin_price_Training - Training.csv',
    'avocado_price': 'avocado.csv',
    'car_price': 'car data.txt',
    'sp500_model': None,
    'movie_recommender': 'ratings.csv',
    'london_crime': 'london_crime_by_lsoa.csv',
    'cirrhosis_classifier': 'cirrhosis.csv',
    'airline_delay': 'DelayedFlights.csv'
}

MODELS_DIR = Path(__file__).parent.parent / 'models'
DATA_DIR = Path(__file__).parent.parent / 'datasets'

def build_df_from_dataset(name):
    p = DATA_DIR / name
    if not p.exists():
        return None
    try:
        # Read only header and first row
        df = pd.read_csv(p, nrows=1)
        # Fill NaNs with zeros
        return df.fillna(0)
    except Exception:
        return None

def run_smoke_for_model(model_name):
    out = {'model': model_name}
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        out['status'] = 'missing_model_file'
        return out
    try:
        m = joblib.load(model_path)
    except Exception as e:
        out['status'] = 'load_error'
        out['error'] = str(e)
        return out

    # If the model is a plain dict-like object (simple recommender), treat as ok
    if isinstance(m, dict):
        out['status'] = 'ok'
        out['model_object_keys'] = list(m.keys())
        return out

    # Prefer dataset if mapped
    ds_name = MODEL_MAP.get(model_name)
    tried = []
    if ds_name:
        df = build_df_from_dataset(ds_name)
        if df is not None:
            try:
                pred = m.predict(df)
                out['status'] = 'ok'
                out['predict'] = str(pred[0])
                out['tried'] = ['dataset_df']
                return out
            except Exception as e:
                tried.append({'dataset_df': str(e)})

    # Fallback: try DataFrame constructed from feature_names_in_
    fni = getattr(m, 'feature_names_in_', None)
    if fni is not None:
        cols = list(fni)
        df = pd.DataFrame([[0]*len(cols)], columns=cols)
        try:
            pred = m.predict(df)
            out['status'] = 'ok'
            out['predict'] = str(pred[0])
            out['tried'] = ['feature_names_df']
            return out
        except Exception as e:
            tried.append({'feature_names_df': str(e)})

    # Last fallback: try numpy zeros with n_features_in_
    nfi = getattr(m, 'n_features_in_', None)
    if nfi is not None:
        sample = np.zeros((1, int(nfi)))
        try:
            pred = m.predict(sample)
            out['status'] = 'ok'
            out['predict'] = str(pred[0])
            out['tried'] = ['numpy_n_features_in']
            return out
        except Exception as e:
            tried.append({'numpy_n_features_in': str(e)})

    out['status'] = 'failed'
    out['tried'] = tried
    return out

def main():
    results = []
    for model_file in sorted(MODELS_DIR.glob('*.joblib')):
        model_name = model_file.stem
        results.append(run_smoke_for_model(model_name))
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
