from fastapi import FastAPI, UploadFile, File
import torch
import tempfile
import numpy as np

from model import load_model
from preprocess import extract_features

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model("best_cnn_model.pth", device)


@app.get("/")
def home():
    return {"message": "LFCC Deepfake Detection API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    # Feature extraction
    features = extract_features(temp_path)

    # Convert to tensor → (1,1,20,400)
    features = torch.tensor(features).unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(features)
        prob = torch.sigmoid(logits).item()

    label = "Spoof" if prob > 0.5 else "Bonafide"

    return {
        "prediction": label,
        "confidence": float(prob)
    }