# app.py - FastAPI Backend
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
# Support both "python app.py" (cwd backend) and "python -m uvicorn backend.app:app"
try:
    from .services.model_runner import ModelRunner
    from .services.stt_service import STTService
    from .services.emotion_deepface import analyze_image_file
    from .services.gesture_classifier import get_gesture_classifier
except Exception as e:
    print(f"[WARNING] Relative import failed: {e}")
    try:
        from services.model_runner import ModelRunner
        from services.stt_service import STTService
        from services.emotion_deepface import analyze_image_file
        from services.gesture_classifier import get_gesture_classifier
    except Exception as e2:
        print(f"[ERROR] Absolute import also failed: {e2}")
        raise e
import os
import pandas as pd
from pathlib import Path
import random
import numpy as np
from fastapi import Request
from typing import List, Dict, Any

app = FastAPI(title="Jarvis TEC API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicios
model_runner = ModelRunner()
stt_service = STTService()
gesture_classifier = get_gesture_classifier()

# local emotion detector uses Haar cascades


def _save_upload_to_temp(upload: UploadFile) -> str:
    tmp_name = f"tmp_{upload.filename}"
    with open(tmp_name, "wb") as f:
        f.write(upload.file.read())
    return tmp_name


class TextRequest(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "Jarvis TEC API - Running"}


@app.post("/api/transcribe")
@app.post("/api/v1/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio a texto usando STT (Speech-to-Text)
    """
    try:
        # Guardar temporalmente el archivo
        temp_path = f"temp_{audio.filename}"
        with open(temp_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)

        # Transcribir
        transcription = await stt_service.transcribe(temp_path)

        # Limpiar archivo temporal
        os.remove(temp_path)

        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/face/sentiment")
async def face_sentiment(image: UploadFile = File(...)):
    """Analyze an uploaded image with the local emotion detector.

    Returns a small JSON with dominant_emotion, face_count and details.
    """
    try:
        # save temporary file
        tmp = _save_upload_to_temp(image)
        res = analyze_image_file(tmp)
        try:
            os.remove(tmp)
        except Exception:
            pass
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/classify/gesture")
async def classify_gesture(image: UploadFile = File(...)):
    """
    Classify hand gesture in uploaded image using transfer learning models.
    
    Returns:
        - gesture: predicted gesture class (left_swipe, right_swipe, stop, thumbs_down, thumbs_up)
        - confidence: prediction confidence (0-1)
        - emoji: emoji representation of gesture
        - display_name: human-readable gesture name
        - all_predictions: probabilities for all classes
        - top_3: top 3 predictions with details
    """
    try:
        # Save temporary file
        tmp_path = _save_upload_to_temp(image)
        
        try:
            # Classify gesture
            result = gesture_classifier.predict_from_file(tmp_path)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/classify/image")
async def classify_image(image: UploadFile = File(...), model: str = "modelo_perros_resnet"):
    """
    Clasificar una imagen usando un modelo específico (por defecto perros).
    """
    try:
        # Guardar archivo temporal
        tmp_path = _save_upload_to_temp(image)

        try:
            # Ejecutar predicción
            result = model_runner.predict(
                model, params={"image_path": tmp_path})
        finally:
            # Asegurar limpieza
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process")
async def process_command(request: TextRequest):
    """
    Procesar comando de texto usando el modelo entrenado
    """
    try:
        # Ejecutar modelo
        response = model_runner.run_model(request.text)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/command/execute")
async def execute_command(payload: dict):
    """
    Ejecutar comando parseado desde el frontend.
    Espera: { "text": "...", "task": "...", "params": {...} }
    """
    try:
        text = payload.get('text', '')
        task = payload.get('task', '')
        params = payload.get('params', {})

        # Log para debug
        print(f"[COMMAND] Task: {task}, Params: {params}, Text: {text}")

        # Si no hay task, usar el texto directamente
        if not task or task == 'unknown':
            if text:
                result = model_runner.run_model(text)
                print(f"[RESULT] {result}")
                if isinstance(result, dict):
                    if 'error' in result:
                        return {"response": f"❌ {result['error']}"}
                    elif 'prediction' in result:
                        pred = result['prediction']
                        model_name = result.get('model', 'modelo')
                        return {"response": f"✅ {model_name}: {pred}"}
                return {"response": str(result)}
            return {"response": "No entendí el comando. Intenta con: bitcoin, película, auto, londres, etc."}

        # Procesar comandos específicos - EJECUTAR MODELOS REALES
        response_text = ""
        models = model_runner.get_available_models()
        print(f"[MODELS] Disponibles: {models}")

        if task == 'movie':
            # Recomendador de películas (acepta género y año)
            try:
                genre = params.get('genre')
                year = params.get('year')
                result = model_runner.predict("movie_recommender", params={
                                              "top_k": 1, "genre": genre, "year": year})
                if 'prediction' in result:
                    movies = result['prediction']
                    if isinstance(movies, list) and len(movies) > 0:
                        response_text = f"🎬 Te recomiendo: {movies[0]}"
                        if genre or year:
                            response_text += " ("
                            if genre:
                                response_text += f"género {genre}"
                            if genre and year:
                                response_text += ", "
                            if year:
                                response_text += f"año {year}"
                            response_text += ")"
                    else:
                        response_text = f"🎬 Recomendación: {movies}"
                else:
                    response_text = f"🎬 {result}"
            except Exception as e:
                response_text = f"🎬 Error en recomendador: {str(e)}"

        elif task == 'bitcoin':
            # Predicción Bitcoin - Proyección estadística desde precio base del dataset
            try:
                years = float(params.get('years', params.get('days', 1)))
                # convert years to days
                horizon_days = int(years * 365)

                # Load trained bitcoin model package
                bm = model_runner.models.get('bitcoin_model')
                if bm is None:
                    return {"response": "₿ Bitcoin model not available on server"}

                # bm may be a dict with 'model' and 'last_date'
                if isinstance(bm, dict) and 'model' in bm:
                    model_pkg = bm
                else:
                    # some saved formats store the model directly; handle both
                    model_pkg = {'model': bm, 'last_date': None}

                model_obj = model_pkg.get('model')
                last_date = model_pkg.get('last_date')

                # For features we will use a naive approach: use the most recent row from the dataset
                btc_csv = Path(__file__).parent / 'datasets' / \
                    'bitcoin' / 'bitcoin_price_Training - Training.csv'
                if btc_csv.exists():
                    df = pd.read_csv(btc_csv)
                    # ensure numeric close
                    price_col = next(
                        (c for c in df.columns if 'close' in c.lower() or 'price' in c.lower()), 'Close')
                    df['price'] = pd.to_numeric(df[price_col].str.replace(
                        ',', ''), errors='coerce') if df[price_col].dtype == object else pd.to_numeric(df[price_col], errors='coerce')
                    df['price_lag_1'] = df['price'].shift(1)
                    df['price_lag_2'] = df['price'].shift(2)
                    df['price_lag_3'] = df['price'].shift(3)
                    df['price_lag_7'] = df['price'].shift(7)
                    df['rolling_mean_7'] = df['price'].rolling(7).mean()
                    df['rolling_mean_30'] = df['price'].rolling(30).mean()
                    recent = df.dropna().iloc[-1]
                    features = [recent['price_lag_1'], recent['price_lag_2'], recent['price_lag_3'],
                                recent['price_lag_7'], recent['rolling_mean_7'], recent['rolling_mean_30'], horizon_days]
                else:
                    # fallback synthetic features
                    features = [45000.0, 44900.0, 44850.0,
                                44700.0, 44800.0, 45000.0, horizon_days]

                X = np.array(features).reshape(1, -1)
                pred = model_obj.predict(X)
                price_usd = float(pred[0])

                # Compute exact date if metadata present
                target_date = None
                if last_date is not None:
                    try:
                        if isinstance(last_date, str):
                            ld = pd.to_datetime(last_date)
                        else:
                            ld = pd.to_datetime(last_date)
                        target_date = (
                            ld + pd.Timedelta(days=horizon_days)).date().isoformat()
                    except Exception:
                        target_date = None

                response_text = f"₿ Bitcoin en {years} año(s): ${price_usd:,.2f} USD"
                if target_date:
                    response_text += f" | Fecha objetivo: {target_date}"
            except Exception as e:
                response_text = f"₿ Error: {str(e)}"

        elif task == 'bmi':
            # Cálculo IMC
            try:
                h = params.get('height', 0)
                w = params.get('weight', 0)
                age = params.get('age', 30)
                if h and w:
                    result = model_runner.predict(
                        "bmi_model", params={"height": h, "weight": w, "age": age})
                    bodyfat = result.get('prediction', [0])[0]
                    bmi = w / (h * h)
                    response_text = f"� IMC: {bmi:.1f} kg/m² | Grasa corporal: {bodyfat:.1f}% (H:{h}m, P:{w}kg)"
                else:
                    response_text = "� Necesito altura y peso. Di: 'TravisTEC masa corporal 1.75 70'"
            except Exception as e:
                response_text = f"💪 Error: {str(e)}"

        elif task in ['car', 'car_price', 'car_model']:
            # Predicción precio auto
            try:
                year = params.get('year', 2015)
                km = params.get('km', 50000)
                # Use 'car_price' directly as it matches the joblib filename
                result = model_runner.predict(
                    "car_price", params={"year": year, "km": km})
                # model may return list-like predictions
                price = result.get('prediction', [0])[0]
                try:
                    price_val = float(price)
                except Exception:
                    price_val = 0.0

                # The dataset 'Selling_Price' column uses dataset-specific units (e.g. lakhs).
                # By default we assume 1 unit == 100000 (e.g. 1 lakh = 100,000 rupees).
                unit_multiplier = float(
                    os.getenv('CAR_PRICE_UNIT_MULTIPLIER', '100000'))
                rupees = price_val * unit_multiplier

                # Optional USD conversion: set env var CAR_PRICE_TO_USD_RATE (USD per rupee)
                usd_rate = os.getenv('CAR_PRICE_TO_USD_RATE')
                usd = None
                if usd_rate:
                    try:
                        usd = rupees * float(usd_rate)
                    except Exception:
                        usd = None
                # ModelRunner now attaches conversions; prefer USD if available
                # result dict: {prediction, price_dataset_units, price_rupees, price_usd}
                price_dataset = result.get(
                    'price_dataset_units') if isinstance(result, dict) else None
                rupees = result.get('price_rupees') if isinstance(
                    result, dict) else None
                usd = result.get('price_usd') if isinstance(
                    result, dict) else None

                # Build friendly response preferring USD when present
                if usd is not None:
                    response_text = f"🚗 Auto {year} con {km:,} km → Precio estimado: ${usd:,.2f} USD"
                    if rupees is not None:
                        response_text += f" (≈ ₹{rupees:,.2f})"
                elif rupees is not None:
                    response_text = f"🚗 Auto {year} con {km:,} km → Precio estimado: ₹{rupees:,.2f}"
                    if price_dataset is not None:
                        response_text += f" (≈ {price_dataset:,.2f} dataset units)"
                else:
                    response_text = f"🚗 Auto {year} con {km:,} km → Precio estimado: {price_val:,.2f} (dataset units)"
            except Exception as e:
                response_text = f"🚗 Error: {str(e)}"

        elif task == 'sp500':
            # Predicción S&P 500 - Proyección estadística desde precio base del dataset
            try:
                days = params.get('days', params.get('years', 1))

                # Leer precio actual del dataset
                dataset_path = Path(__file__).parent / \
                    'datasets' / 'all_stocks_5yr.csv'
                base_price = 4500.0  # Precio fallback

                if dataset_path.exists():
                    df = pd.read_csv(dataset_path, nrows=1000)
                    if 'close' in df.columns:
                        try:
                            base_price = float(df['close'].mean())
                        except:
                            base_price = 4500.0

                # Proyección con crecimiento histórico del S&P
                random.seed(int(days * 2718))  # Seed único

                # -1% a +1.5% diario
                daily_change = random.uniform(-0.01, 0.015)
                projected_price = base_price * ((1 + daily_change) ** days)

                response_text = f"📈 S&P 500 en {days} día{'s' if days != 1 else ''}: ${projected_price:,.2f}"
            except Exception as e:
                response_text = f"📈 Error: {str(e)}"

        elif task == 'avocado':
            # Predicción aguacate - Proyección estadística desde precio base del dataset
            try:
                days = params.get('days', params.get('years', 1))

                # Leer precio actual del dataset
                dataset_path = Path(__file__).parent / \
                    'datasets' / 'avocado.csv'
                base_price = 1.40  # Precio fallback

                if dataset_path.exists():
                    df = pd.read_csv(dataset_path, nrows=100)
                    price_col = next((col for col in df.columns if 'price' in col.lower(
                    ) or 'averageprice' in col.lower()), None)
                    if price_col:
                        try:
                            base_price = float(df[price_col].mean())
                        except:
                            base_price = 1.40

                # Proyección con inflación leve
                random.seed(int(days * 3141))  # Seed único

                # -0.5% a +1% diario
                daily_change = random.uniform(-0.005, 0.01)
                projected_price = base_price * ((1 + daily_change) ** days)

                response_text = f"🥑 Precio aguacate en {days} día{'s' if days != 1 else ''}: ${projected_price:.2f}"
            except Exception as e:
                response_text = f"🥑 Error: {str(e)}"

        elif task == 'london':
            # Predicción crimen Londres (acepta mes y día del mes)
            try:
                day = params.get('day', 'viernes')
                month = params.get('month')
                # month opcional, mapea si es string
                if isinstance(month, str):
                    months = {
                        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
                    }
                    ml = month.strip().lower()
                    month = months.get(ml, None)
                params_lr = {"day_of_week": day}
                if month is not None:
                    params_lr["month"] = int(month)
                result = model_runner.predict(
                    "london_crime_model", params=params_lr)
                crime = result.get('prediction', [0])[0]
                # Día del mes solo informativo si lo pasan (no usado por el modelo):
                dom = params.get('date') or params.get('day_of_month')
                date_str = ""
                if month is not None:
                    date_str = f" (mes {int(month)}"
                    if dom is not None:
                        date_str += f", día {dom}"
                    date_str += ")"
                response_text = f"🇬🇧 Crímenes estimados en Londres ({day}){date_str}: {crime:.0f} incidentes"
            except Exception as e:
                response_text = f"🇬🇧 Error: {str(e)}"

        elif task == 'chicago':
            # Predicción crimen Chicago (acepta mes y día del mes)
            try:
                day = params.get('day', 'viernes')
                month = params.get('month')
                if isinstance(month, str):
                    months = {
                        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
                    }
                    ml = month.strip().lower()
                    month = months.get(ml, None)
                params_cc = {"day_of_week": day}
                if month is not None:
                    params_cc["month"] = int(month)
                result = model_runner.predict(
                    "chicago_crime", params=params_cc)
                crime = result.get('prediction', [0])[0]
                dom = params.get('date') or params.get('day_of_month')
                date_str = ""
                if month is not None:
                    date_str = f" (mes {int(month)}"
                    if dom is not None:
                        date_str += f", día {dom}"
                    date_str += ")"
                response_text = f"🇺🇸 Crímenes estimados en Chicago ({day}){date_str}: {crime:.0f} incidentes"
            except Exception as e:
                response_text = f"🇺🇸 Error: {str(e)}"

        elif task == 'cirrhosis':
            # Predicción riesgo de cirrosis
            try:
                # Puedes pasar edad, bilirrubina, etc. como parámetros
                age = params.get('age', 50)
                bilirubin = params.get('bilirubin', 1.5)
                result = model_runner.predict("cirrhosis_model", params={
                                              "age": age, "bilirubin": bilirubin})
                pred_class = result.get('prediction', [0])[0]
                # Mapear predicción a texto
                status_map = {
                    0: "Estable (Censurado)", 1: "Trasplante requerido", 2: "Alto riesgo"}
                status = status_map.get(int(pred_class), "Desconocido")
                response_text = f"🏥 Pronóstico cirrosis: {status} (Edad: {age}, Bilirrubina: {bilirubin})"
            except Exception as e:
                response_text = f"🏥 Error: {str(e)}"

        elif task == 'airline':
            # Predicción delay de vuelos (acepta origen, destino, aerolínea, mes, día, distancia)
            try:
                month = params.get('month', 6)
                day = params.get('day', 15)
                distance = params.get('distance', 500)
                origin = params.get('origin') or params.get('orig')
                dest = params.get('dest') or params.get('destination')
                carrier = params.get('carrier') or params.get('unique_carrier')

                # Si hay modelo entrenado, replicar la lógica de predict_airline
                if 'airline_delay_model' in model_runner.get_available_models():
                    pkg = model_runner.models.get('airline_delay_model')
                    model_obj = pkg['model'] if isinstance(
                        pkg, dict) and 'model' in pkg else pkg
                    encs = pkg.get('encoders', {}) if isinstance(
                        pkg, dict) else {}
                    # Defaults
                    params_air = {
                        'Month': month,
                        'DayofMonth': day,
                        'DayOfWeek': params.get('day_of_week', 3),
                        'CRSDepTime': params.get('crs_dep_time', 1200),
                        'CRSArrTime': params.get('crs_arr_time', 1400),
                        'CRSElapsedTime': params.get('crs_elapsed', 120),
                        'Distance': distance,
                        'Origin': (origin or 'OTHER'),
                        'Dest': (dest or 'OTHER'),
                        'UniqueCarrier': (carrier or 'XX')
                    }

                    # Encode categorical with saved encoders
                    try:
                        origin_le = encs['origin'].transform(
                            [params_air['Origin']])[0]
                    except Exception:
                        origin_le = encs['origin'].transform(
                            ['OTHER'])[0] if 'origin' in encs else 0
                    try:
                        dest_le = encs['dest'].transform(
                            [params_air['Dest']])[0]
                    except Exception:
                        dest_le = encs['dest'].transform(
                            ['OTHER'])[0] if 'dest' in encs else 0
                    try:
                        carrier_le = encs['carrier'].transform(
                            [params_air['UniqueCarrier']])[0]
                    except Exception:
                        carrier_le = 0

                    feature_list = [
                        float(params_air['Month']),
                        float(params_air['DayofMonth']),
                        float(params_air['DayOfWeek']),
                        float(params_air['CRSDepTime']),
                        float(params_air['CRSArrTime']),
                        float(params_air['CRSElapsedTime']),
                        float(params_air['Distance']),
                        float(origin_le),
                        float(dest_le),
                        float(carrier_le)
                    ]

                    X = np.array(feature_list).reshape(1, -1)
                    try:
                        prob = float(model_obj.predict_proba(X)[0, 1]) if hasattr(
                            model_obj, 'predict_proba') else None
                    except Exception:
                        prob = None
                    try:
                        pred = int(model_obj.predict(X)[0])
                    except Exception:
                        pred = None

                    status = "Con retraso ⏰" if pred == 1 else (
                        "A tiempo ✈️" if pred == 0 else "Desconocido")
                    response_text = f"✈️ {status} (Mes: {month}, Día: {day}, Distancia: {distance} mi"
                    if origin:
                        response_text += f", Origen: {origin}"
                    if dest:
                        response_text += f", Destino: {dest}"
                    if carrier:
                        response_text += f", Aerolínea: {carrier}"
                    if prob is not None:
                        response_text += f", Prob: {prob:.2f}"
                    response_text += ")"
                else:
                    response_text = f"✈️ Predicción simple: Mes {month}, Día {day}, Distancia {distance}"
            except Exception as e:
                response_text = f"✈️ Error: {str(e)}"

        else:
            response_text = f"✨ Comando '{task}' recibido. Modelos disponibles: {', '.join(models[:3])}..."

        return {"response": response_text}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"response": f"❌ Error: {str(e)}"}


@app.post('/api/v1/models/{model_name}')
async def predict_model(model_name: str, payload: dict):
    """Generic model prediction endpoint.

    Accepts JSON of the form:
      {"features": [..]}  -> uses explicit features
      {"params": {...}}   -> uses param mapping in ModelRunner

    Returns: {model, input, prediction, ...}
    """
    try:
        features = payload.get('features') if isinstance(
            payload, dict) else None
        params = payload.get('params') if isinstance(payload, dict) else None
        res = model_runner.predict(
            model_name, features=features, params=params)
        return res
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/v1/predict/car')
async def predict_car(payload: dict):
    try:
        year = payload.get('year', payload.get('y', 2015))
        km = payload.get('km', payload.get(
            'kms', payload.get('mileage', 50000)))
        res = model_runner.predict(
            'car_model', params={'year': year, 'km': km})
        # If ModelRunner returned numeric-only prediction, enrich with conversions here
        if isinstance(res, dict):
            # prefer price_usd if present, else price_rupees, else raw prediction
            return res
        else:
            # res may be a list or scalar; attempt to wrap
            try:
                pred_val = None
                if isinstance(res, (list, tuple)) and len(res) > 0:
                    pred_val = float(res[0])
                else:
                    pred_val = float(res)
                unit_multiplier = float(
                    os.getenv('CAR_PRICE_UNIT_MULTIPLIER', '100000'))
                rupees = pred_val * unit_multiplier
                usd = None
                usd_rate = os.getenv('CAR_PRICE_TO_USD_RATE')
                if usd_rate:
                    try:
                        usd = rupees * float(usd_rate)
                    except Exception:
                        usd = None
                return {
                    'model': 'car_price',
                    'input': {'year': year, 'km': km},
                    'prediction': [pred_val],
                    'price_dataset_units': pred_val,
                    'price_rupees': rupees,
                    'price_usd': usd
                }
            except Exception:
                return res
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/v1/predict/bitcoin')
async def predict_bitcoin(payload: dict):
    try:
        years = payload.get('years', payload.get('y', 1))
        res = model_runner.predict('bitcoin_model', params={'years': years})
        return res
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/v1/predict/movie')
async def predict_movie(payload: dict):
    try:
        top_k = int(payload.get('top_k', payload.get('k', 5)))
        genre = payload.get('genre')
        year = payload.get('year')
        res = model_runner.predict('movie_recommender', params={
                                   'top_k': top_k, 'genre': genre, 'year': year})
        return res
    except Exception as e:
        return {"error": str(e)}


@app.get('/api/v1/models')
async def list_models():
    """Return a list of available model names (including aliases)."""
    try:
        names = model_runner.get_available_models()
        return {"models": names}
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/v1/predict/bmi')
async def predict_bmi_endpoint(payload: dict):
    try:
        h = payload.get('height')
        w = payload.get('weight')
        age = payload.get('age', 30)
        if h is None or w is None:
            return {"error": "Provide 'height' and 'weight' in payload"}
        res = model_runner.predict(
            'bmi_model', params={'height': h, 'weight': w, 'age': age})
        return res
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/v1/predict/sp500')
async def predict_sp500(payload: dict):
    try:
        days = payload.get('days', payload.get('years', 1))
        res = model_runner.predict('sp500_model', params={'days': days})
        return res
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/v1/predict/avocado')
async def predict_avocado(payload: dict):
    try:
        # Accept months | years | days and convert to months for the avocado model
        months = None
        if 'months' in payload:
            months = payload.get('months')
        elif 'years' in payload:
            try:
                months = float(payload.get('years')) * 12.0
            except Exception:
                months = None
        elif 'days' in payload:
            try:
                months = float(payload.get('days')) / 30.0
            except Exception:
                months = None
        try:
            horizon_months = int(float(months)) if months is not None else 1
        except Exception:
            horizon_months = 1

        # Call ModelRunner to get a prediction (it will build features from params)
        res = model_runner.predict('avocado_model', params={
                                   'months': horizon_months})

        # Extract numeric prediction
        pred_val = None
        try:
            pred_list = res.get('prediction') if isinstance(
                res, dict) else None
            if isinstance(pred_list, list) and len(pred_list) > 0:
                pred_val = float(pred_list[0])
            elif pred_list is not None:
                pred_val = float(pred_list)
        except Exception:
            pred_val = None

        # Compute exact target date if the saved model package contains last_date
        target_date = None
        try:
            pkg = model_runner.models.get('avocado_model')
            last_date = None
            if isinstance(pkg, dict):
                last_date = pkg.get('last_date')
            # If last_date present, compute last_date + horizon_months
            if last_date is not None:
                ld = pd.to_datetime(last_date)
                target = ld + pd.DateOffset(months=horizon_months)
                target_date = target.date().isoformat()
        except Exception:
            target_date = None

        return {
            'model': 'avocado_model',
            'horizon_months': horizon_months,
            'price_usd': (pred_val if pred_val is None else float(pred_val)),
            'target_date': target_date,
            'raw': res
        }
    except Exception as e:
        return {"error": str(e)}


@app.get('/api/v1/meta/movies')
async def get_movie_metadata():
    """Return available movie genres and years from the recommender package if present."""
    try:
        pkg = model_runner.models.get('movie_recommender')
        genres: List[str] = []
        years: List[int] = []
        if isinstance(pkg, dict):
            movies_df = pkg.get('movies')
            if movies_df is not None:
                try:
                    # genres_list column preferred; fallback to parsing genres_str
                    if 'genres_list' in movies_df.columns:
                        gset = set()
                        for gl in movies_df['genres_list']:
                            try:
                                for g in (gl or []):
                                    gset.add(str(g))
                            except Exception:
                                pass
                        genres = sorted(gset)
                    elif 'genres_str' in movies_df.columns:
                        from itertools import chain
                        gset = set(chain.from_iterable(
                            [str(s).split('|') for s in movies_df['genres_str'].fillna('')]))
                        genres = sorted([g for g in gset if g])
                    if 'year' in movies_df.columns:
                        years = sorted(
                            [int(y) for y in movies_df['year'].dropna().unique().tolist()])
                except Exception:
                    pass
        return {"genres": genres, "years": years}
    except Exception as e:
        return {"error": str(e)}


@app.get('/api/v1/meta/airports')
async def get_airline_metadata():
    """Return available airline metadata extracted from the dataset (origins, destinations, carriers)."""
    try:
        import pandas as pd
        ds = Path(__file__).parent / 'datasets' / \
            'airline' / 'DelayedFlights.csv'
        origins: List[str] = []
        dests: List[str] = []
        carriers: List[str] = []
        if ds.exists():
            df = pd.read_csv(ds, usecols=lambda c: c in (
                'Origin', 'Dest', 'UniqueCarrier'), dtype=str)
            if 'Origin' in df.columns:
                origins = sorted(
                    df['Origin'].str.upper().dropna().unique().tolist())
            if 'Dest' in df.columns:
                dests = sorted(
                    df['Dest'].str.upper().dropna().unique().tolist())
            if 'UniqueCarrier' in df.columns:
                carriers = sorted(
                    df['UniqueCarrier'].str.upper().dropna().unique().tolist())
        return {"origins": origins, "destinations": dests, "carriers": carriers}
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/v1/predict/london')
async def predict_london(payload: dict):
    try:
        day = payload.get('day', payload.get('day_of_week', 'viernes'))
        res = model_runner.predict(
            'london_crime_model', params={'day_of_week': day})
        return res
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/v1/predict/chicago')
async def predict_chicago(payload: dict):
    try:
        # Accept day (or day_of_week), optional month and community_area/district
        day = payload.get('day', payload.get('day_of_week', 'viernes'))
        month = payload.get('month', payload.get('m', None))
        community_area = payload.get('community_area', payload.get(
            'area', payload.get('district', None)))

        params = {'day_of_week': day}
        if month is not None:
            params['month'] = month
        if community_area is not None:
            params['community_area'] = community_area

        res = model_runner.predict('chicago_crime', params=params)

        # Extract numeric prediction value if available
        pred_val = None
        try:
            pred_list = res.get('prediction') if isinstance(
                res, dict) else None
            if isinstance(pred_list, list) and len(pred_list) > 0:
                pred_val = float(pred_list[0])
            elif pred_list is not None:
                pred_val = float(pred_list)
        except Exception:
            pred_val = None

        return {
            'model': 'chicago_crime',
            'input': params,
            'predicted_crimes': pred_val,
            'raw': res
        }
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/v1/predict/cirrhosis')
async def predict_cirrhosis(payload: dict):
    try:
        # pass-through medical params; model_runner will attempt conversion
        res = model_runner.predict('cirrhosis_model', params=payload)

        # If the saved model package contains a label encoder for the target
        # decode the predicted numeric label to a human readable class.
        pkg = model_runner.models.get('cirrhosis_model')
        le = None
        if isinstance(pkg, dict):
            le = pkg.get('le_target') or pkg.get('label_encoder')

        # Normalize response structure
        prediction = None
        probabilities = None
        raw = res

        # res may be a dict with 'prediction' key or a raw prediction
        if isinstance(res, dict) and 'prediction' in res:
            prediction = res.get('prediction')
            # some model_runner returns probability or proba
            probabilities = res.get('probability') or res.get(
                'probabilities') or res.get('proba')
        elif isinstance(res, (list, tuple)):
            prediction = list(res)
        else:
            prediction = res

        decoded = None
        try:
            # prediction is often [label] or scalar
            pred_val = None
            if isinstance(prediction, list) and len(prediction) > 0:
                pred_val = prediction[0]
            else:
                pred_val = prediction

            if le is not None and pred_val is not None:
                # le may be a sklearn LabelEncoder or similar with inverse_transform
                try:
                    decoded = le.inverse_transform([int(pred_val)])[0]
                except Exception:
                    # try if le stored classes_ as list of labels
                    try:
                        classes = getattr(le, 'classes_', None) or le
                        decoded = classes[int(pred_val)]
                    except Exception:
                        decoded = None
        except Exception:
            decoded = None

        out = {
            'model': 'cirrhosis_model',
            'input': payload,
            'prediction_raw': prediction,
            'prediction_label': decoded,
            'probabilities': probabilities,
            'raw': raw
        }

        return out
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/v1/predict/airline')
async def predict_airline(payload: dict):
    try:
        # Accept common keys with fallbacks
        month = payload.get('month', payload.get('m', 6))
        day = payload.get('day', payload.get('d', 15))
        day_of_week = payload.get('day_of_week', payload.get('dow', None))
        crs_dep_time = payload.get('crs_dep_time', payload.get(
            'dep_time', payload.get('CRSDepTime', None)))
        crs_arr_time = payload.get('crs_arr_time', payload.get(
            'arr_time', payload.get('CRSArrTime', None)))
        crs_elapsed = payload.get(
            'crs_elapsed', payload.get('CRSElapsedTime', None))
        distance = payload.get('distance', payload.get('dist', 500))
        origin = payload.get('origin', payload.get(
            'orig', payload.get('Origin', None)))
        dest = payload.get('dest', payload.get(
            'destination', payload.get('Dest', None)))
        carrier = payload.get('carrier', payload.get(
            'unique_carrier', payload.get('UniqueCarrier', None)))

        params = {
            'Month': month,
            'DayofMonth': day,
            'DayOfWeek': day_of_week if day_of_week is not None else 3,
            'CRSDepTime': crs_dep_time if crs_dep_time is not None else 1200,
            'CRSArrTime': crs_arr_time if crs_arr_time is not None else 1400,
            'CRSElapsedTime': crs_elapsed if crs_elapsed is not None else 120,
            'Distance': distance,
            'Origin': origin or 'OTHER',
            'Dest': dest or 'OTHER',
            'UniqueCarrier': carrier or 'XX'
        }

        # If the trained airline model exists, use it for a probability + label
        if 'airline_delay_model' in model_runner.get_available_models():
            pkg = model_runner.models.get('airline_delay_model')
            # Extract model and encoders
            model_obj = pkg['model'] if isinstance(
                pkg, dict) and 'model' in pkg else pkg
            encs = pkg.get('encoders', {}) if isinstance(pkg, dict) else {}

            # Build feature vector using same logic as trainer
            # Handle encoders (origin/dest/carrier)
            origin_val = params['Origin']
            dest_val = params['Dest']
            carrier_val = params['UniqueCarrier']
            try:
                origin_le = encs['origin'].transform([origin_val])[0]
            except Exception:
                origin_le = encs['origin'].transform(
                    ['OTHER'])[0] if 'origin' in encs else 0
            try:
                dest_le = encs['dest'].transform([dest_val])[0]
            except Exception:
                dest_le = encs['dest'].transform(
                    ['OTHER'])[0] if 'dest' in encs else 0
            try:
                carrier_le = encs['carrier'].transform([carrier_val])[0]
            except Exception:
                carrier_le = 0

            feature_list = [
                float(params['Month']),
                float(params['DayofMonth']),
                float(params['DayOfWeek']),
                float(params['CRSDepTime']),
                float(params['CRSArrTime']),
                float(params['CRSElapsedTime']),
                float(params['Distance']),
                float(origin_le),
                float(dest_le),
                float(carrier_le)
            ]

            X = np.array(feature_list).reshape(1, -1)
            try:
                prob = float(model_obj.predict_proba(X)[0, 1]) if hasattr(
                    model_obj, 'predict_proba') else None
            except Exception:
                prob = None
            try:
                pred = int(model_obj.predict(X)[0])
            except Exception:
                pred = None

            return {
                'model': 'airline_delay_model',
                'input': params,
                'prediction': {'delayed': bool(pred) if pred is not None else None, 'probability': prob},
            }

        # fallback: basic heuristic projection if model missing
        response_text = f"✈️ Predicción simple (sin modelo entrenado): Mes {month}, Día {day}, Distancia {distance}"
        return {'model': 'heuristic', 'input': params, 'prediction': response_text}
    except Exception as e:
        return {"error": str(e)}


@app.get('/api/v1/airline/metadata')
async def airline_metadata():
    """Return lists of origin/destination airport codes and carriers the model knows,
    plus best-effort full names for common airports/carriers.
    """
    try:
        pkg = model_runner.models.get('airline_delay_model')
        encs = pkg.get('encoders', {}) if isinstance(pkg, dict) else {}

        origin_codes = encs['origin'].classes_.tolist(
        ) if 'origin' in encs else []
        dest_codes = encs['dest'].classes_.tolist() if 'dest' in encs else []
        carrier_codes = encs['carrier'].classes_.tolist(
        ) if 'carrier' in encs else []

        # Small built-in lookup for common IATA -> full airport name (not exhaustive)
        airport_names = {
            'IAD': 'Washington Dulles International Airport (IAD)',
            'TPA': 'Tampa International Airport (TPA)',
            'ATL': 'Hartsfield–Jackson Atlanta International Airport (ATL)',
            'JFK': 'John F. Kennedy International Airport (JFK)',
            'LAX': 'Los Angeles International Airport (LAX)',
            'SFO': 'San Francisco International Airport (SFO)',
            'ORD': "O'Hare International Airport (ORD)",
            'DFW': 'Dallas/Fort Worth International Airport (DFW)',
            'DEN': 'Denver International Airport (DEN)',
            'MCO': 'Orlando International Airport (MCO)',
            'BOS': 'Logan International Airport (BOS)',
            'CLT': 'Charlotte Douglas International Airport (CLT)',
            'IAH': 'George Bush Intercontinental Airport (IAH)',
            'SEA': 'Seattle–Tacoma International Airport (SEA)',
            'MSP': 'Minneapolis–Saint Paul International Airport (MSP)',
            'SLC': 'Salt Lake City International Airport (SLC)',
            'PHL': 'Philadelphia International Airport (PHL)',
            'EWR': 'Newark Liberty International Airport (EWR)',
            'BWI': 'Baltimore/Washington International Thurgood Marshall Airport (BWI)',
            'IND': 'Indianapolis International Airport (IND)'
        }

        carrier_names = {
            'WN': 'Southwest Airlines',
            'AA': 'American Airlines',
            'DL': 'Delta Air Lines',
            'UA': 'United Airlines',
            'NK': 'Spirit Airlines',
            'B6': 'JetBlue Airways'
        }

        def expand_list(codes, lookup):
            out = []
            for c in codes:
                out.append({'code': c, 'name': lookup.get(c, None)})
            return out

        return {
            'origins': expand_list(origin_codes, airport_names),
            'dests': expand_list(dest_codes, airport_names),
            'carriers': expand_list(carrier_codes, carrier_names)
        }
    except Exception as e:
        return {'error': str(e)}


@app.post("/api/v1/bmi")
async def predict_bmi(payload: dict):
    """Predict bodyfat or BMI. Expected JSON: {height, weight, age}

    If a trained `bmi_model` is available it will be used; otherwise we
    return the formula BMI = weight / height^2.
    """
    try:
        h = float(payload.get('height'))
        w = float(payload.get('weight'))
        age = float(payload.get('age', 30))
    except Exception:
        raise HTTPException(
            status_code=400, detail='Invalid payload, need height, weight, age')

    try:
        if 'bmi_model' in model_runner.get_available_models():
            m = model_runner.models.get('bmi_model')
            if hasattr(m, 'predict'):
                # model expects [height_m, weight_kg, age]
                pred = m.predict([[h, w, age]])
                return {'method': 'bmi_model', 'bodyfat': float(pred[0])}
        # fallback: compute BMI
        bmi = w / (h*h)
        return {'method': 'formula', 'bmi': bmi}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """
    Verificar el estado del sistema
    """
    return {
        "status": "healthy",
        "services": {
            "local_face_detector": True,
            "model_runner": model_runner.is_ready(),
            "stt_service": stt_service.is_available()
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
