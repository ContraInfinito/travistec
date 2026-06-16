import json
import joblib
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / 'models'
DATA_DIR = Path(__file__).parent.parent / 'datasets'


def try_predict(model_path: Path):
    name = model_path.stem
    out = {'model': name}
    try:
        m = joblib.load(model_path)
    except Exception as e:
        out['status'] = 'load_error'
        out['error'] = str(e)
        return out

    # Prefer DataFrame if model indicates feature names
    tried = []

    # Helper to run predict and capture exceptions
    def run_pred(input_obj):
        try:
            pred = m.predict(input_obj)
            return True, pred
        except Exception as e:
            return False, str(e)

    # 1) If model exposes n_features_in_
    nfi = getattr(m, 'n_features_in_', None)
    if nfi is not None:
        sample = np.zeros((1, int(nfi)))
        ok, res = run_pred(sample)
        tried.append({'type': 'numpy_n_features_in',
                     'ok': ok, 'result': str(res)})
        if ok:
            out['status'] = 'ok'
            out['predict'] = str(res[0]) if hasattr(
                res, '__len__') else str(res)
            out['tried'] = tried
            return out

    # 2) If model exposes feature_names_in_
    fni = getattr(m, 'feature_names_in_', None)
    if fni is not None:
        cols = list(fni)
        df = pd.DataFrame([[0]*len(cols)], columns=cols)
        ok, res = run_pred(df)
        tried.append({'type': 'df_feature_names_in',
                     'ok': ok, 'result': str(res)})
        if ok:
            out['status'] = 'ok'
            out['predict'] = str(res[0]) if hasattr(
                res, '__len__') else str(res)
            out['tried'] = tried
            return out

    # 3) Try a generic small numpy array (3 features)
    sample = np.zeros((1, 3))
    ok, res = run_pred(sample)
    tried.append({'type': 'numpy_3', 'ok': ok, 'result': str(res)})
    if ok:
        out['status'] = 'ok'
        out['predict'] = str(res[0]) if hasattr(res, '__len__') else str(res)
        out['tried'] = tried
        return out

    # 4) Try to find a dataset with similar name and use its columns
    candidate = None
    for ext in ['.csv', '.zip']:
        p = DATA_DIR / (name + ext)
        if p.exists():
            candidate = p
            break

    if candidate and candidate.suffix == '.csv':
        try:
            df0 = pd.read_csv(candidate, nrows=1)
            df = pd.DataFrame([df0.iloc[0].fillna(0).to_dict()])
            ok, res = run_pred(df)
            tried.append({'type': 'df_from_dataset', 'dataset': str(
                candidate.name), 'ok': ok, 'result': str(res)})
            if ok:
                out['status'] = 'ok'
                out['predict'] = str(res[0]) if hasattr(
                    res, '__len__') else str(res)
                out['tried'] = tried
                return out
        except Exception as e:
            tried.append({'type': 'df_from_dataset', 'error': str(e)})

    out['status'] = 'failed'
    out['tried'] = tried
    return out


def main():
    results = []
    for p in sorted(MODELS_DIR.glob('*.joblib')):
        results.append(try_predict(p))

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
