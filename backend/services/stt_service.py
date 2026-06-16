# stt_service.py - Servicio de Speech-to-Text
import os
from dotenv import load_dotenv
import numpy as np
import tensorflow as tf
import keras
import librosa
import imageio_ffmpeg
import shutil
import difflib
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Add ffmpeg to PATH so librosa/audioread can find it
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")

# audioread requires 'ffmpeg.exe', but imageio-ffmpeg provides a versioned name
if not os.path.exists(target_ffmpeg):
    try:
        shutil.copy(ffmpeg_exe, target_ffmpeg)
        print(
            f"[INFO] Copied ffmpeg to {target_ffmpeg} for audioread compatibility")
    except Exception as e:
        print(f"[WARN] Failed to copy ffmpeg: {e}")

os.environ["PATH"] += os.pathsep + ffmpeg_dir

load_dotenv()


class STTService:
    def __init__(self):
        # Default to local_whisper if not specified
        self.service_type = os.getenv('STT_SERVICE', 'local_whisper')
        self.configure_service()

    def configure_service(self):
        if self.service_type == 'azure':
            self.setup_azure()
        elif self.service_type == 'local_ctc':
            self.setup_local_ctc()
        elif self.service_type == 'local_whisper':
            self.setup_local_whisper()
        else:
            print(f'[WARN] Servicio STT no configurado: {self.service_type}')

    def setup_local_whisper(self):
        try:
            # Path to the unzipped model folder
            model_path = os.path.join(os.path.dirname(
                __file__), '../models/whisper-small-es-finetune-final/whisper-small-es-finetune-final')

            if not os.path.exists(model_path):
                print(f'[WARN] Modelo Whisper no encontrado en {model_path}')
                self.configured = False
                return

            print(f"[INFO] Loading Whisper model from {model_path}...")
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_path)
            # Disable forced language/task to let it detect or use defaults
            self.model.config.forced_decoder_ids = None

            self.configured = True
            print('[OK] Modelo Local Whisper cargado correctamente')
        except Exception as e:
            print(f'[ERROR] Fallo al cargar modelo Whisper: {e}')
            self.configured = False

    def setup_local_ctc(self):
        try:
            model_path = os.path.join(os.path.dirname(
                __file__), '../models/audio_ctc_model.keras')
            vocab_path = os.path.join(os.path.dirname(
                __file__), '../models/audio_ctc_model_vocab.txt')

            if not os.path.exists(model_path) or not os.path.exists(vocab_path):
                print(f'[WARN] Modelo CTC no encontrado en {model_path}')
                self.configured = False
                return

            self.ctc_model = keras.models.load_model(model_path, compile=False)

            with open(vocab_path, 'r') as f:
                vocab_str = f.read()
            self.vocab = list(vocab_str)

            self.char_to_num = keras.layers.StringLookup(
                vocabulary=self.vocab, mask_token='')
            self.num_to_char = keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), mask_token='', invert=True)

            self.configured = True
            print('[OK] Modelo Local CTC cargado correctamente')
        except Exception as e:
            print(f'[ERROR] Fallo al cargar modelo CTC: {e}')
            self.configured = False

    def setup_azure(self):
        try:
            import azure.cognitiveservices.speech as speechsdk
            speech_key = os.getenv('AZURE_SPEECH_KEY', '')
            service_region = os.getenv('AZURE_SERVICE_REGION', '')
            if speech_key and service_region:
                self.speech_config = speechsdk.SpeechConfig(
                    subscription=speech_key, region=service_region)
                self.speech_config.speech_recognition_language = 'es-ES'
                self.configured = True
                print('[OK] Azure Speech configurado')
            else:
                self.configured = False
                print('[WARN] Azure Speech no configurado')
        except ImportError:
            self.configured = False
            print('[WARN] Instale azure-cognitiveservices-speech')

    def correct_text(self, text):
        """Applies spell checking against a known vocabulary."""
        if not text:
            return text

        # List of expected keywords/commands based on the system's domain
        keywords = [
            "travistec", "airline", "avocado", "aguacate", "bitcoin", "bmi", "bodymass", "masa",
            "cars", "coche", "carro", "chicago", "cirrhosis", "cirrosis", "london", "londres",
            "movie", "movies", "pelicula", "cine", "sp500", "acciones", "action", "thriller",
            "drama", "comedy", "romance", "horror", "adventure", "crime", "crimen", "fantasy",
            "mystery", "scifi", "war", "western", "avion",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december",
            "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
            "septiembre", "octubre", "noviembre", "diciembre",
            "toyota", "honda", "ford", "bmw", "nissan", "audi", "mercedes", "tesla", "chevrolet", "hyundai",
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"
        ]

        # Map Spanish/Synonyms to Canonical Commands
        # This ensures the frontend receives the exact keywords it expects
        canonical_map = {
            "londres": "london crime",
            "london": "london crime",
            "chicago": "chicago crime",
            "crimen": "crime",
            "pelicula": "movie",
            "cine": "movie",
            "acciones": "sp500",
            "aguacate": "avocado",
            "masa": "bmi",
            "coche": "car_price",
            "carro": "car_price",
            "autos": "car_price",
            "coches": "car_price",
            "avion": "airline",
            "cirrosis": "cirrhosis"
        }

        # Find closest match
        # cutoff=0.75 allows more flexibility for phonetic variants
        matches = difflib.get_close_matches(
            text.lower(), keywords, n=1, cutoff=0.75)

        if matches:
            matched_word = matches[0]
            print(f"[INFO] Corrected '{text}' -> '{matched_word}'")

            # Return canonical form if exists, else the matched word
            return canonical_map.get(matched_word, matched_word)

        # Check if it's a number (e.g. 2015, 50000)
        if text.isdigit():
            return text

        # Strict mode: If no match and not a number, return empty string to suppress gibberish
        # This addresses the user's complaint about "agael", "pomcm" etc.
        return ""

    def correct_full_transcription(self, full_text):
        """
        Applies phonetic correction to the full transcription.
        Handles multi-word phrases like "agua cate" -> "aguacate" -> "avocado"
        and "si rrosis" -> "cirrosis" -> "cirrhosis"
        """
        import re

        # First, normalize the text (remove punctuation, lowercase)
        clean_text = re.sub(r'[^\w\s]', '', full_text).lower().strip()

        # Phonetic phrase mappings - handle split words and common misrecognitions
        phrase_patterns = [
            # Aguacate variations (avocado)
            (r'\bagua\s*cat[ea]?\b', 'avocado'),
            (r'\bagua\s*c[aá]tida\b', 'avocado'),
            (r'\baguacat[ea]?\b', 'avocado'),
            (r'\babocado\b', 'avocado'),
            (r'\baguacate\b', 'avocado'),

            # Cirrosis variations (cirrhosis)
            (r'\bsi\s*r+os+is\b', 'cirrhosis'),
            (r'\bsi\s*ros+is\b', 'cirrhosis'),
            (r'\bsi\s*bruc?ys\b', 'cirrhosis'),
            (r'\bsi\s*roc?e[s]?\b', 'cirrhosis'),
            (r'\bse\s*roc?e[s]?\b', 'cirrhosis'),
            (r'\bcirrosis\b', 'cirrhosis'),
            (r'\bsirrosis\b', 'cirrhosis'),

            # Bitcoin variations
            (r'\bbit\s*co[iy]n?\b', 'bitcoin'),
            (r'\bbitcine\b', 'bitcoin'),
            (r'\bbitcoy\b', 'bitcoin'),

            # Avión -> airline
            (r'\bavi[oó]n\b', 'airline'),
            (r'\baerline\b', 'airline'),
            (r'\bairline\b', 'airline'),

            # Londres/London
            (r'\blondres\b', 'london crime'),
            (r'\blondon\b', 'london crime'),

            # Chicago
            (r'\bchicago\b', 'chicago crime'),

            # Película/Movie
            (r'\bpel[ií]cula\b', 'movie'),
            (r'\bmovie\b', 'movie'),
            (r'\bcine\b', 'movie'),

            # Carro/Coche -> car_price
            (r'\bcarro\b', 'car_price'),
            (r'\bcoche\b', 'car_price'),
            (r'\bauto\b', 'car_price'),

            # Masa/BMI
            (r'\bmasa\s*corporal\b', 'bmi'),
            (r'\bmasa\b', 'bmi'),
            (r'\bbmi\b', 'bmi'),

            # SP500
            (r'\bsp\s*500\b', 'sp500'),
            (r'\bacciones\b', 'sp500'),

            # TravisTEC
            (r'\btravis\s*tec\b', 'travistec'),
            (r'\btrabis\s*tec\b', 'travistec'),
        ]

        # Try each pattern
        for pattern, replacement in phrase_patterns:
            if re.search(pattern, clean_text, re.IGNORECASE):
                print(
                    f"[INFO] Phrase match: '{clean_text}' matched '{pattern}' -> '{replacement}'")
                return replacement

        # If no phrase match, fall back to word-by-word correction
        return None

    async def transcribe(self, audio_path):
        if not self.configured:
            return 'Error: Servicio STT no configurado'

        if self.service_type == 'local_whisper':
            return self.transcribe_local_whisper(audio_path)
        elif self.service_type == 'local_ctc':
            return self.transcribe_local_ctc(audio_path)
        elif self.service_type == 'azure':
            return self.transcribe_azure(audio_path)
        return ''

    def transcribe_local_whisper(self, audio_path):
        temp_wav = None
        try:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print(f"[ERROR] Audio file is missing or empty: {audio_path}")
                return ""

            # Convert to WAV using ffmpeg (16kHz mono for Whisper)
            import subprocess
            import uuid

            temp_wav = f"temp_whisper_{uuid.uuid4()}.wav"

            cmd = [
                ffmpeg_exe, "-y",
                "-i", audio_path,
                "-ar", "16000",
                "-ac", "1",
                temp_wav
            ]

            print(
                f"[INFO] Converting {audio_path} to {temp_wav} for Whisper...")
            result = subprocess.run(
                cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                print(
                    f"[ERROR] FFmpeg failed: {result.stderr.decode('utf-8', errors='ignore')}")
                return ''

            # Load audio with librosa
            audio, rate = librosa.load(temp_wav, sr=16000)

            # Process audio
            input_features = self.processor(
                audio, sampling_rate=16000, return_tensors="pt").input_features

            # Generate token ids
            # We force language='es' to ensure it focuses on Spanish
            predicted_ids = self.model.generate(input_features, language="es")

            # Decode token ids to text
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True)[0]

            print(f"[INFO] Whisper Raw Transcription: '{transcription}'")

            # Clean up text (remove punctuation, lowercase)
            import re
            clean_text = re.sub(r'[^\w\s]', '', transcription).lower().strip()

            # FIRST: Try full phrase matching for multi-word patterns like "agua cate"
            phrase_result = self.correct_full_transcription(clean_text)
            if phrase_result:
                return phrase_result

            # SECOND: Fall back to word-by-word correction
            words = clean_text.split()
            final_words = []
            for word in words:
                corrected = self.correct_text(word)
                if corrected:
                    final_words.append(corrected)

            if final_words:
                return " ".join(final_words)

            # If no keyword found, return empty string (Strict Mode)
            # This prevents "large phrases I never said" from being returned
            return ""

        except Exception as e:
            print(f'[ERROR] Error en inferencia Whisper: {e}')
            import traceback
            traceback.print_exc()
            return ''
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass

    def transcribe_local_ctc(self, audio_path):
        temp_wav = None
        try:
            SAMPLE_RATE = 22050
            FFT_LENGTH = 384
            FRAME_LENGTH = 256
            FRAME_STEP = 160

            # Convert to WAV using ffmpeg if needed (especially for webm)
            # We always convert to ensure consistent sample rate and format
            import subprocess
            import uuid

            temp_wav = f"temp_convert_{uuid.uuid4()}.wav"

            # Use the ffmpeg we found/configured earlier
            # -y: overwrite output
            # -ar: set sample rate
            # -ac: set channels (1 for mono)
            # Use the full path to the ffmpeg executable
            cmd = [
                ffmpeg_exe, "-y",
                "-i", audio_path,
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                temp_wav
            ]

            # Run conversion
            print(f"[INFO] Converting {audio_path} to {temp_wav}...")
            # Capture stderr to debug ffmpeg issues
            result = subprocess.run(
                cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                print(f"[ERROR] FFmpeg failed with code {result.returncode}")
                print(
                    f"[ERROR] FFmpeg stderr: {result.stderr.decode('utf-8', errors='ignore')}")
                return ''

            # Load the converted WAV file
            audio, _ = librosa.load(temp_wav, sr=SAMPLE_RATE)

            stft = librosa.stft(audio, n_fft=FFT_LENGTH,
                                hop_length=FRAME_STEP, win_length=FRAME_LENGTH)
            spectrogram = np.abs(stft)
            spectrogram = np.log1p(spectrogram)
            spectrogram = np.transpose(spectrogram)

            means = np.mean(spectrogram, axis=0)
            stds = np.std(spectrogram, axis=0)
            spectrogram = (spectrogram - means) / (stds + 1e-10)

            spectrogram = np.expand_dims(spectrogram, axis=0)

            preds = self.ctc_model.predict(spectrogram, verbose=0)

            input_len = np.ones(preds.shape[0]) * preds.shape[1]
            input_len = tf.cast(input_len, dtype=tf.int32)

            # Transpose for tf.nn.ctc_greedy_decoder (time, batch, classes)
            preds_t = tf.transpose(preds, perm=[1, 0, 2])

            decoded, _ = tf.nn.ctc_greedy_decoder(
                preds_t, sequence_length=input_len, merge_repeated=True)
            results = tf.sparse.to_dense(decoded[0], default_value=-1)

            output_text = []
            for res in results:
                res = res[res != -1]
                res = tf.strings.reduce_join(
                    self.num_to_char(res)).numpy().decode('utf-8')
                output_text.append(res)

            # Apply spell correction
            final_text = self.correct_text(output_text[0])
            return final_text
        except Exception as e:
            print(f'[ERROR] Error en inferencia CTC: {e}')
            import traceback
            traceback.print_exc()
            return ''
        finally:
            # Cleanup temp wav
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass

    def transcribe_azure(self, audio_path):
        import azure.cognitiveservices.speech as speechsdk
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, audio_config=audio_config)
        result = speech_recognizer.recognize_once_async().get()
        return result.text
