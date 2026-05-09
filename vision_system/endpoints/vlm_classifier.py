from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import uvicorn
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import io
import json
import re
from peft import PeftModel

app = FastAPI()

# Load Molmo2-4B on startup
print("Loading Molmo2-4B model...")

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=[
        # Module names can also be relative like "ff_norm" which would apply to all such layers
        "model.vision_backbone", "model.transformer.ff_out", "model.transformer.ln_f"
    ]
)

model_id="allenai/Molmo2-4B"

# load the processor
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto",
    token=True
)

# load the model
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto",
    quantization_config=nf4_config,
    token=True
)

model = PeftModel.from_pretrained(model, "checkpoint-3000")

print("Model loaded successfully!")

def classify_image(image_bytes: bytes, prompt: str) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image}
        ]
    }
    ]
        
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
        
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # Generate with streaming (token-by-token)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    
    # Only get generated tokens
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text.strip()

@app.post("/analyze")
async def analyze_screenshot(file: UploadFile = File(...), prompt: str = Form("Classify the following KOTOR user interface screenshot into exactly one of these categories: combat, narrative, or leveling. Only answer with one word: combat, narrative, or leveling.")):
    """Analyze screenshot and return streaming Molmo response."""
    image_bytes = await file.read()
    return StreamingResponse(
        classify_image(image_bytes, prompt),
        media_type="application/x-ndjson"
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": model_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
