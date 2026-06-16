# Audit & Refinement Log

This document records the issues found when reopening TravisTEC after its initial
university-project phase, and how each was fixed. It is the honest "before → after"
trail behind the polished version in [`README.md`](README.md).

The guiding principle: a university project and a portfolio project differ less in
talent than in the willingness to audit your own work and fix what's wrong — with
the numbers to prove it.

---

## Issues found & resolved

| # | Issue | Severity | Resolution |
|---|---|---|---|
| 1 | **Fake financial predictions.** `model_runner.py` generated features with `random.seed(horizon * constant)` for Bitcoin, S&P 500 and avocado — noise disguised as inference. | Blocking | Real lag/rolling features from live `yfinance` data (BTC-USD, ^GSPC) and the avocado dataset, with a 1-hour cache and graceful fallback. |
| 2 | **Duplicated fake logic in the API.** `app.py`'s voice-command path (`/api/v1/command/execute`) re-implemented the same fake `random.seed` projections, bypassing the model entirely. | Blocking | Routed Bitcoin/SP500/avocado voice commands through the single fixed `ModelRunner` path — one source of truth. |
| 3 | **CTC "speech recognition" overclaim.** 108 samples across ~30 classes is a keyword spotter, not ASR; WER was meaningless at that scale. | High | Reframed honestly as a keyword spotter; expanded the dataset and added a real held-out WER metric (see #4, #5). |
| 4 | **No validation split on the CTC model.** `EarlyStopping` monitored *training* loss, so it never detected overfitting and "best" = most overfit. | High | Added an 80/20 stratified split; all callbacks now monitor `val_loss`; added a `jiwer` WER metric computed on the held-out set. |
| 5 | **Tiny voice dataset.** Only 108 clips, ~3 per command. | High | Expanded to **1,596 clips** (49–58 per command) via neural TTS (`edge-tts`, 8 voices) plus librosa augmentation (pitch, time-stretch, noise). |
| 6 | **Corrupt training label.** `metadata.csv` had `bitcoin+` (and a `recomned` typo), poisoning the vocabulary. | High | Corrected to `bitcoin` / `recommend`. |
| 7 | **Duplicate avocado trainer.** Two model definitions lived in one file; the weaker V2 silently shadowed V1 and used an incompatible feature schema. | High | Deleted V2; kept and documented the V1 lag-feature approach. |
| 8 | **No model validation rigor.** The 10 ML models used a single `train_test_split`, no baselines, no feature importance. | Medium | Added 5-fold cross-validation, `Dummy` baselines, and feature-importance charts (`docs/charts/`) to every supervised trainer. |
| 9 | **Two frontends in the repo.** A legacy vanilla-JS `frontend/` sat next to `frontend-react/`, confusing the structure. | Medium | Deleted the legacy `frontend/`. |
| 10 | **No CI.** No automated linting or testing. | Medium | Added GitHub Actions: flake8, model smoke tests, ESLint. |
| 11 | **Stale / inaccurate docs.** The README referenced deleted files (`azure_face.py`, `emotion_local.py`, the old `frontend/`), the old "Jarvis TEC" name, Azure Face API (replaced by DeepFace), and contained a corrupted section. | Medium | Rewrote all documentation to match the actual codebase. |
| 12 | **DeepFace framing.** `emotion_deepface.py` is an integrated wrapper, not a trained model. | Low | Documented honestly as "integrated DeepFace", not "built a face model". |
| 13 | **No architecture diagram.** | Low | Added a Mermaid system diagram to the README. |

---

## Honest framing retained

- **Voice** is a *command keyword spotter* over a fixed vocabulary, not general ASR.
- **Emotion** is an *integrated DeepFace wrapper*, not a model trained in this project.
- **Gesture** is the strongest deep-learning story: MobileNetV3Large transfer learning
  on LeapGestRecog, a defensible architecture choice (fewer parameters than ResNet50
  for comparable accuracy on a CPU-served app).

## Stale-model finding (discovered during metric generation)

| # | Issue | Severity | Resolution |
|---|---|---|---|
| 14 | **Models trained on stale/mismatched data.** The Bitcoin model was trained on a 2013-era CSV (BTC ~$100) and the S&P 500 model on a 2018 mean-of-stocks proxy (~$80–150) — yet the fixed inference path feeds *live* prices ($60k+ BTC, ~5,300 index). A RandomForest can't extrapolate, so live predictions were still meaningless even after the inference fix. | High | Retrained both on live yfinance data (`BTC-USD`, `^GSPC`) so the training range matches inference, with a time-based 90-day backtest. |

## Measured results

Generated locally (`python backend/scripts/train_*.py`). Models whose datasets are
present locally; airline/London/Chicago datasets are gitignored or need BigQuery.

| Model | Type | Headline metric | Model | Baseline |
|---|---|---|---|---|
| Bitcoin | Regression | 90-day backtest, 1-day-ahead | **MAPE 3.20%** (MAE $2,332) | — |
| Bitcoin | Regression | 5-fold CV MAE (all horizons) | $16,200 | $32,266 |
| S&P 500 | Regression | 90-day backtest, 1-day-ahead | **MAPE 4.43%** (MAE 325) | — |
| S&P 500 | Regression | 5-fold CV MAE | 476 | 1,372 |
| Avocado | Regression | 5-fold CV MAE | 0.143 | 0.219 |
| Car price | Regression | 5-fold CV MAE | 0.840 | 3.686 |
| BMI / body-fat | Regression | 5-fold CV MAE | 5.064 | 6.884 |
| Cirrhosis | Classification | 5-fold CV accuracy | 0.522 | 0.376 |

Every model beats its naive baseline. The Bitcoin/S&P backtests are honest
out-of-sample estimates on a held-out 90-day window of live prices.

**Pending (needs GPU/Colab or absent data):**
- **Voice CTC WER** — dataset (1,596 clips) and the 80/20-split + `jiwer` WER
  pipeline are ready, but training needs TensorFlow on a GPU. Run
  `notebooks/train_audio_ctc_colab.ipynb`. (TensorFlow has no Python 3.13 build,
  so it can't run in this local env.)
- **Gesture confusion matrix** — needs the LeapGestRecog images (`backend/data/`,
  gitignored).
- **Airline / London / Chicago CV** — datasets absent locally / need BigQuery.

## Known remaining debt

- **Frontend lint.** The original React components carry pre-existing lint debt
  (unused variables, an empty catch block, a `while (true)` loop). These are
  demoted to ESLint *warnings* so CI stays green, and should be cleaned up
  incrementally. The backend passes `flake8` with no warnings.
- **Charts on retrain.** Feature-importance PNGs in `docs/charts/` are generated
  by the trainers; only the models trained locally are committed so far.

## Verifying the fixes

```powershell
cd backend
python scripts\run_model_smoke_tests.py   # every saved model loads & predicts
python scripts\train_bmi_model.py          # prints CV vs Dummy baseline + writes chart
```
