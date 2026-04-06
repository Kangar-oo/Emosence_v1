from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import base64
import tensorflow as tf
import ollama
import os
import logging

logging.basicConfig(level=logging.INFO)

# === LOAD MODEL ===
# Matches the path where train_model.py saves the file
MODEL_PATH = '../ml-engine/emosense_cnn.h5'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

print("⏳ Loading Vision Model...")
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Vision Model Online.")
else:
    print(f"⚠️ Warning: Model not found at {MODEL_PATH}. Run train_model.py first.")
    model = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatPayload(BaseModel):
    text: str
    image: str | None = None

@app.post("/api/analyze")
async def analyze(payload: ChatPayload):
    detected_emotion = "Neutral"
    confidence = 0.0

    # --- VISION ---
    if payload.image and model:
        try:
            img_str = payload.image.split(",")[1] if "," in payload.image else payload.image
            img_bytes = base64.b64decode(img_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (48, 48))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)

            predictions = model.predict(img, verbose=0)
            max_index = np.argmax(predictions[0])
            detected_emotion = EMOTION_LABELS[max_index]
            confidence = float(np.max(predictions[0]))
            print(f"👁️ Emotion: {detected_emotion} ({confidence:.2f})")
        except Exception as e:
            print(f"❌ Vision Error: {e}")

    # --- CHAT (Ollama) ---
    try:
        system_prompt = f"""
        You are EmoSense, a therapy AI.
        User's emotion: {detected_emotion}.
        User said: "{payload.text}"
        Reply with empathy in 2 sentences.
        """
        
        # Ensure 'ollama serve' is running in another terminal
        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'user', 'content': system_prompt},
        ])
        ai_text = response['message']['content']
    except Exception as e:
        logging.exception("❌ Chat Error: Ollama chat failed")
        ai_text = "I hear you. (Local Brain Offline: temporary technical issue.)"

    return {
        "response": ai_text,
        "mood": detected_emotion,
        "analysis": f"Confidence: {confidence*100:.1f}%"
    }