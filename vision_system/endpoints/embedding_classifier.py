from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import uvicorn
import io
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

app = FastAPI()

print("Loading Qwen3-VL embedding model...")

model_id = "Qwen/Qwen3-VL-Embedding-2B"
model = SentenceTransformer(
    model_id,
    trust_remote_code=True,
)

LABEL_TEXTS = {
    "combat": "A game UI screenshot showing combat gameplay, battle actions, hostile targets, health bars, loot pickup overlays, item collected popups, or post-combat reward screens.",
    "narrative": "A game UI screenshot showing narrative dialogue, story conversation, subtitles, character interaction, or cutscene text.",
    "leveling": "A game UI screenshot showing character leveling, progression, skills, attributes, inventory stats, or upgrade menus.",
}

labels = list(LABEL_TEXTS.keys())
label_embs = np.asarray(
    model.encode([LABEL_TEXTS[label] for label in labels], normalize_embeddings=True),
    dtype=np.float32,
)

print("Embedding model loaded successfully!")


def classify_image(image_bytes: bytes, prompt: str) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_emb = np.asarray(
        model.encode([image], normalize_embeddings=True),
        dtype=np.float32,
    )[0]

    scores = label_embs @ image_emb
    best_idx = int(np.argmax(scores))
    predicted_label = labels[best_idx]
    print(f"Predicted label: {predicted_label} (scores: {dict(zip(labels, scores))})")

    return predicted_label


@app.post("/analyze")
async def analyze_screenshot(
    file: UploadFile = File(...),
    prompt: str = Form(
        "Classify the following KOTOR user interface screenshot into exactly one of these categories: combat, narrative, or leveling. Only answer with one word: combat, narrative, or leveling."
    ),
):
    image_bytes = await file.read()
    return StreamingResponse(
        iter([classify_image(image_bytes, prompt)]),
        media_type="application/x-ndjson",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)