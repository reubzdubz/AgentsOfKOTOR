from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import joblib

app = FastAPI()

print("Loading SVM edge classifier...")

# Load the trained model and label encoder
model_path = "output/svm_edge_classifier.pkl"
encoder_path = "output/label_encoder.pkl"

clf = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

print("SVM classifier loaded successfully!")

def classify_features(edge_density_corners: float, edge_density_global: float):
    # Prepare feature vector
    X = np.array([[edge_density_corners, edge_density_global]])
    # Predict
    pred = clf.predict(X)
    label = label_encoder.inverse_transform(pred)[0]
    # Confidence scores (probabilities)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        scores = dict(zip(label_encoder.classes_, probs.tolist()))
    else:
        scores = {label: 1.0}
    return label, scores

@app.post("/classify")
async def classify(
    edge_density_corners: float = Form(...),
    edge_density_global: float = Form(...),
):
    label, scores = classify_features(edge_density_corners, edge_density_global)
    return JSONResponse({
        "predicted_label": label,
        "scores": scores
    })

@app.get("/health")
async def health():
    return {"status": "ok", "model": "Linear SVM Edge Classifier"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
