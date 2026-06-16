# TravisTEC — Frontend

React 19 + Vite 7 single-page app for the TravisTEC multimodal assistant. It captures
webcam frames and audio, sends them to the backend (or a mock server), and visualizes
emotion, gesture, transcription, and ML-prediction results.

## Scripts

```bash
npm install        # install dependencies
npm run dev        # dev server with HMR  → http://localhost:3000
npm run build      # production build → dist/
npm run preview    # preview the production build
npm run lint       # ESLint
```

## Configuration (`.env`)

Copy `.env.example` to `.env`:

```env
VITE_API_URL=http://localhost:8000     # real FastAPI backend
VITE_USE_MOCK=false                    # true → use the Express mock server
VITE_MOCK_API_URL=http://localhost:3001
```

When `VITE_USE_MOCK=true`, `src/services/api-client.js` points every request at the
mock server in [`../mock-server`](../mock-server), so you can develop the UI without
running the ML backend or any cloud keys.

## Structure

```
src/
├── pages/
│   ├── Home.jsx       Landing page (/)
│   ├── Capture.jsx    Live capture: camera + audio (/capture)
│   └── Results.jsx    Session summary & logs (/results)
├── components/
│   ├── CameraCapture.jsx   Webcam stream, auto + manual snapshots
│   ├── AudioRecorder.jsx   Web Speech API + MediaRecorder, command parsing
│   ├── EmotionDisplay.jsx  Emotion distribution + dominant emoji
│   ├── GestureDisplay.jsx  Predicted gesture + confidence
│   └── DogClassifier.jsx   Image upload → dog-breed classifier
├── services/
│   └── api-client.js  Axios client (mock/real switch via .env)
└── main.jsx           App entry + React Router
```

## API client

`api-client.js` wraps the backend endpoints used by the UI:

| Method | Backend endpoint |
|---|---|
| `transcribeAudio(blob)` | `POST /api/v1/transcribe` |
| `recognizeFace(blob)` | `POST /api/v1/face/sentiment` |
| `classifyGesture(blob)` | `POST /api/v1/classify/gesture` |
| `classifyImage(blob, model)` | `POST /api/v1/classify/image` |
| `processCommand(command)` | `POST /api/v1/command/execute` |
| `calculateBMI(params)` | `POST /api/v1/bmi` |
| `healthCheck()` | `GET /api/health` |
| `getMovieMeta()` / `getAirlineMeta()` | `GET /api/v1/meta/*` |

See the [root README](../README.md) for the full API reference and architecture.
