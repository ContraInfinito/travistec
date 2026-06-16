"""
Run smoke tests against the FastAPI app using TestClient.
Skips file upload / external service endpoints.
"""
import json
import asyncio
import inspect
from pathlib import Path

import app as api

tests = [
    {'name': 'root', 'fn': api.root, 'args': ()},
    {'name': 'health', 'fn': api.health_check, 'args': ()},
    {'name': 'list_models', 'fn': api.list_models, 'args': ()},
    {'name': 'predict_bitcoin', 'fn': api.predict_bitcoin,
        'args': ({'years': 0.01},)},
    {'name': 'predict_avocado', 'fn': api.predict_avocado,
        'args': ({'months': 1},)},
    {'name': 'predict_car', 'fn': api.predict_car,
        'args': ({'year': 2015, 'km': 50000},)},
    {'name': 'predict_bmi_endpoint', 'fn': api.predict_bmi_endpoint,
        'args': ({'height': 1.75, 'weight': 70, 'age': 30},)},
    {'name': 'predict_sp500', 'fn': api.predict_sp500, 'args': ({'days': 7},)},
    {'name': 'predict_london', 'fn': api.predict_london,
        'args': ({'day': 'viernes'},)},
    {'name': 'predict_chicago', 'fn': api.predict_chicago,
        'args': ({'day_of_week': 4},)},
    {'name': 'predict_cirrhosis', 'fn': api.predict_cirrhosis,
        'args': ({'age': 50, 'bilirubin': 2.0, 'N_Days': 1000},)},
    {'name': 'predict_airline', 'fn': api.predict_airline,
        'args': ({'month': 6, 'day': 15, 'distance': 500},)},
    {'name': 'predict_movie', 'fn': api.predict_movie,
        'args': ({'top_k': 1},)},
    {'name': 'airline_metadata', 'fn': api.airline_metadata, 'args': ()},
    {'name': 'predict_bmi', 'fn': api.predict_bmi, 'args': (
        {'height': 1.75, 'weight': 70, 'age': 30},)},
]

results = []


async def _maybe_call(fn, *args):
    try:
        if inspect.iscoroutinefunction(fn):
            return await fn(*args)
        else:
            return fn(*args)
    except Exception as e:
        return {'_error': str(e)}


async def run_all():
    for t in tests:
        name = t['name']
        fn = t['fn']
        args = t.get('args', ())
        print(f"Running test: {name} args={args}")
        res = await _maybe_call(fn, *args)
        print(f" -> Result: {type(res)} | {res}\n")
        results.append({'name': name, 'result': res})

if __name__ == '__main__':
    asyncio.run(run_all())
    out = Path(__file__).parent / 'endpoint_test_results.json'
    out.write_text(json.dumps(results, default=str, indent=2))
    print('Saved results to', out)
