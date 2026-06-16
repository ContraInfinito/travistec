# TravisTEC — Backend

FastAPI service exposing the multimodal features (speech-to-text, emotion, gesture,
image classification) and 10 ML prediction models. Entry point: [`app.py`](app.py).

Base URL (local): `http://localhost:8000` · interactive docs at `/docs`.

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

All configuration is optional — see [`.env.example`](.env.example). With no `.env`,
the service uses the local Whisper STT model and the DeepFace emotion detector.

---

## Services (`services/`)

| Module | Responsibility |
|---|---|
| `model_runner.py` | Loads `*.joblib` and Keras models from `models/`, exposes `predict()` / `run_model()`. Builds live features (yfinance) for Bitcoin/SP500. |
| `stt_service.py` | Speech-to-text. `STT_SERVICE` selects `local_whisper` (default), `local_ctc`, or `azure`. |
| `emotion_deepface.py` | Emotion detection via DeepFace, with a Haar-cascade smile-heuristic fallback. |
| `emotion_local_onnx.py` | Alternative FER+ ONNX emotion detector. |
| `gesture_classifier.py` | Hand-gesture classification (MobileNetV3Large, 5 classes). |
| `gesture_mediapipe.py` | Alternative MediaPipe landmark detector. |

---

## Multimodal endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET`  | `/api/health` | — | `{status, services: {stt_service, model_runner, local_face_detector}}` |
| `POST` | `/api/v1/transcribe` (alias `/api/transcribe`) | `audio` (multipart) | `{transcription}` |
| `POST` | `/api/v1/face/sentiment` | `image` (multipart) | `{dominant_emotion, face_count, details}` |
| `POST` | `/api/v1/classify/gesture` | `image` (multipart) | `{gesture, confidence, ...}` |
| `POST` | `/api/v1/classify/image?model=<name>` | `image` (multipart) | classifier output (default: dog breeds) |
| `POST` | `/api/v1/command/execute` | `{text, task, params}` | `{response}` — runs the model for `task` |
| `POST` | `/api/process` | `{text}` | `{response}` — quick text→model mapping |

---

## Model endpoints

- `GET /api/v1/models` → `{"models": ["bitcoin_model", "car_price", ...]}`
- `POST /api/v1/models/{model_name}` → generic predict; body `{"features":[...]}` **or** `{"params":{...}}`
- Convenience wrappers (one per model):

| Path | Example body | Notes |
|---|---|---|
| `POST /api/v1/predict/bitcoin` | `{ "years": 1 }` | Live BTC-USD features via yfinance; target date computed from today. |
| `POST /api/v1/predict/sp500` | `{ "days": 30 }` | Live ^GSPC features. |
| `POST /api/v1/predict/avocado` | `{ "months": 3 }` | Also accepts `days`/`years` (converted to months). Returns `target_date` if available. |
| `POST /api/v1/predict/bmi` · `/api/v1/bmi` | `{ "height": 1.78, "weight": 78, "age": 30 }` | Falls back to the BMI formula if no model. |
| `POST /api/v1/predict/car` | `{ "year": 2015, "km": 50000 }` | Returns dataset units + optional rupee/USD conversion. |
| `POST /api/v1/predict/cirrhosis` | `{ "age": 50, "bilirubin": 1.5 }` | Decodes the predicted stage label if the encoder is saved. |
| `POST /api/v1/predict/airline` | `{ "month": 6, "day": 15, "distance": 500, "origin": "IAD", "dest": "TPA", "carrier": "WN" }` | `{prediction: {delayed, probability}}`. |
| `POST /api/v1/predict/london` | `{ "day": "viernes" }` | Expected crimes/day for a borough/day. |
| `POST /api/v1/predict/chicago` | `{ "day": "viernes", "month": 11 }` | Expected incidents/day. |
| `POST /api/v1/predict/movie` | `{ "top_k": 5, "genre": "Drama", "year": 1994 }` | List of recommended titles. |

### Metadata helpers

- `GET /api/v1/meta/movies` → available genres & years
- `GET /api/v1/meta/airports` → origins / destinations / carriers (from the dataset)
- `GET /api/v1/airline/metadata` → airport & carrier codes with best-effort full names

### Example (fetch)

```javascript
fetch('/api/v1/predict/car', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ year: 2015, km: 50000 })
}).then(r => r.json()).then(console.log);
```

---

## Training & evaluation

Every supervised trainer in [`scripts/`](scripts) reports **5-fold cross-validation
against a `Dummy` baseline** and writes a feature-importance chart to `../docs/charts/`.

```powershell
python scripts\train_bmi_model.py         # CV vs baseline + chart
python scripts\run_model_smoke_tests.py   # load & exercise every saved model
```

Saved models live in `models/` (gitignored). The server loads whatever is present at
startup and returns a clear error for any model that's missing — it never crashes on
absence. Model packages may be plain estimators or dicts like
`{'model', 'feature_cols', 'last_date', 'encoders'}`; `ModelRunner` handles both.
