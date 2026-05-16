from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import io
import numpy as np
from PIL import Image
import joblib
import cv2

app = FastAPI()

print("Loading SVM edge classifier...")

# Load the trained model and label encoder
model_path = "output/svm_edge_classifier.pkl"
encoder_path = "output/label_encoder.pkl"

clf = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

print("SVM classifier loaded successfully!")

def edge_metrics(image_bytes: bytes, low_thresh: int, high_thresh: int, blur_ksize: int):
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image bytes")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edges = cv2.Canny(gray, low_thresh, high_thresh)

    h, w = edges.shape
    density = edges.sum().astype(np.float32) / 255.0 / (h * w)

    bottom_right_corner = edges[3 * h // 4 :, w // 2 :]
    bottom_right_density = bottom_right_corner.sum() / 255.0 / (h * w / 8)
    top_right_corner = edges[: h // 4, w // 2 :]
    top_right_density = top_right_corner.sum() / 255.0 / (h * w / 8)
    bottom_left_corner = edges[3 * h // 4 :, : w // 2]
    bottom_left_density = bottom_left_corner.sum() / 255.0 / (h * w / 8)
    top_left_corner = edges[: h // 4, : w // 2]
    top_left_density = top_left_corner.sum() / 255.0 / (h * w / 8)
    centre_quadrant = edges[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    centre_density = centre_quadrant.sum() / 255.0 / (h * w / 4)

    return {
        "edge_density_global": density,
        "edge_density_corners": (bottom_right_density + top_right_density + bottom_left_density + top_left_density) / 4,
        "edge_density_centre": centre_density,
        "width": int(w),
        "height": int(h),
    }

def classify_features(image_bytes: bytes):
    # Prepare feature vector
    edge_metrics_dict = edge_metrics(image_bytes, low_thresh=100, high_thresh=200, blur_ksize=3)
    edge_density_corners = edge_metrics_dict["edge_density_corners"]
    edge_density_global = edge_metrics_dict["edge_density_global"]
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
async def classify(file: UploadFile = File(...)):                   
    image_bytes = await file.read()
    label, scores = classify_features(image_bytes)
    return JSONResponse({
        "predicted_label": label,
        "scores": scores
    })

@app.get("/health")
async def health():
    return {"status": "ok", "model": "Linear SVM Edge Classifier"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
