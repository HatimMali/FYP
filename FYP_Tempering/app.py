import os
import logging
import tempfile
from contextlib import asynccontextmanager

# Suppress NumPy 1.x vs 2.x warnings from pyarrow at import time
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import joblib
import librosa
import numpy as np
import shap
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (MATCH TRAINING)
# ---------------------------------------------------------------------------
MODEL_PATH = "rf_model1.joblib"
TARGET_SR = 22050
SEGMENT_DURATION = 1.0
MFCC_COUNT = 27

SUPPORTED_EXT = {".wav", ".flac"}

# FIXED FEATURE SIZE
_N_RAW = MFCC_COUNT + 4        # 27 + 4 = 31
N_FEATURES = _N_RAW * 3        # 31 x 3 = 93

# ---------------------------------------------------------------------------
# Feature names (for SHAP)
# ---------------------------------------------------------------------------
BASE_FEATURES = [f"mfcc_{i+1}" for i in range(MFCC_COUNT)] + [
    "centroid", "bandwidth", "zcr", "rms"
]

FEATURE_NAMES = (
    [f"{f}_mean" for f in BASE_FEATURES] +
    [f"{f}_var"  for f in BASE_FEATURES] +
    [f"{f}_maxdiff" for f in BASE_FEATURES]
)

# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------
class ModelState:
    scaler    = None
    model     = None
    explainer = None
    ready     = False

model_state = ModelState()

# ---------------------------------------------------------------------------
# Load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model...")

    model_state.scaler, model_state.model = joblib.load(MODEL_PATH)

    # SHAP Explainer — tree_path_dependent avoids background data requirement
    model_state.explainer = shap.TreeExplainer(
        model_state.model,
        feature_perturbation="tree_path_dependent"
    )

    model_state.ready = True
    log.info("Model loaded successfully")

    yield

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Audio Tampering Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    prediction:  str
    confidence:  float
    filename:    str
    explanation: list

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def segment_audio(y, sr):
    seg_len = int(SEGMENT_DURATION * sr)
    return [y[i:i + seg_len] for i in range(0, len(y) - seg_len + 1, seg_len)]


def extract_segment_features(segment, sr):
    mfcc      = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=MFCC_COUNT)
    mfcc_mean = np.mean(mfcc, axis=1)

    centroid  = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))
    zcr       = np.mean(librosa.feature.zero_crossing_rate(y=segment))
    rms       = np.mean(librosa.feature.rms(y=segment))

    return np.concatenate([mfcc_mean, [centroid, bandwidth, zcr, rms]])


def aggregate(features):
    mean = np.mean(features, axis=0)
    var  = np.var(features,  axis=0)

    if len(features) > 1:
        maxdiff = np.max(np.abs(np.diff(features, axis=0)), axis=0)
    else:
        maxdiff = np.zeros(features.shape[1])

    return np.concatenate([mean, var, maxdiff])


def build_feature_vector(audio_bytes, filename):
    suffix = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, _ = librosa.load(tmp_path, sr=TARGET_SR)
    finally:
        os.remove(tmp_path)

    segments = segment_audio(y, TARGET_SR)

    if len(segments) == 0:
        raise HTTPException(status_code=400, detail="Audio too short")

    seg_feats      = np.array([extract_segment_features(s, TARGET_SR) for s in segments])
    feature_vector = aggregate(seg_feats)

    if feature_vector.shape[0] != N_FEATURES:
        raise HTTPException(status_code=500, detail="Feature size mismatch")

    return feature_vector

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(audio: UploadFile = File(...)):

    if not model_state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    filename = audio.filename
    ext      = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .wav or .flac")

    audio_bytes = await audio.read()

    feature_vector  = build_feature_vector(audio_bytes, filename)
    features_scaled = model_state.scaler.transform([feature_vector])

    # -------- Prediction --------
    pred = model_state.model.predict(features_scaled)[0]
    prob = model_state.model.predict_proba(features_scaled)[0]

    prediction = "Tampered" if pred == 1 else "Authentic"
    confidence = float(prob[pred])

    # -------- SHAP Explanation --------
    shap_values = model_state.explainer.shap_values(features_scaled)

    # Compatible with both old SHAP (returns list per class)
    # and new SHAP (returns a single ndarray)
    if isinstance(shap_values, list):
        shap_vals = shap_values[int(pred)][0]   # use predicted class index
    else:
        if shap_values.ndim == 3:
            shap_vals = shap_values[0, :, int(pred)]  # shape: (n_samples, n_features, n_classes)
        else:
            shap_vals = shap_values[0]                 # shape: (n_samples, n_features)

    # .tolist() converts NumPy integers to plain Python ints for list indexing
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5].tolist()

    explanation = [
        {
            "feature": FEATURE_NAMES[i],
            "impact":  float(shap_vals[i])
        }
        for i in top_idx
    ]

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        filename=filename,
        explanation=explanation,
    )

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)