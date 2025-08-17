from fastapi import FastAPI, WebSocket
import whisper
import requests
from gtts import gTTS
from io import BytesIO
import base64
import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-r1-0528:free"

app = FastAPI()
model = whisper.load_model("tiny")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()  # receive base64 audio chunk
        audio_bytes = base64.b64decode(data)
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes)

        # Transcribe
        result = model.transcribe("temp.wav", language="hi")
        user_text = result["text"]

        # Call OpenRouter
        messages = [{"role": "user", "content": user_text}]
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": MODEL, "messages": messages},
        )
        assistant_text = resp.json()["choices"][0]["message"]["content"]

        # Convert to speech
        tts = gTTS(assistant_text, lang="hi")
        audio_out = BytesIO()
        tts.write_to_fp(audio_out)
        audio_out.seek(0)
        audio_b64 = base64.b64encode(audio_out.read()).decode("utf-8")

        # Send assistant response
        await ws.send_json({"text": assistant_text, "audio": audio_b64})
