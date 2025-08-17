from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import os
from dotenv import load_dotenv
from utils import transcribe_audio, call_openrouter, generate_tts

load_dotenv("secrets.env")  # Load API keys

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = FastAPI(title="AnnaData Voice Assistant API")

@app.post("/voice-assistant/")
async def voice_assistant(audio_file: UploadFile = File(...)):
    # 1️⃣ Save uploaded audio
    file_path = f"temp_{audio_file.filename}"
    with open(file_path, "wb") as f:
        f.write(await audio_file.read())

    # 2️⃣ Transcribe
    user_text = transcribe_audio(file_path)

    # 3️⃣ Call OpenRouter
    assistant_text = call_openrouter(user_text, OPENROUTER_API_KEY)

    # 4️⃣ Generate TTS
    audio_bytes = generate_tts(assistant_text)

    # 5️⃣ Return JSON with transcription + assistant + audio
    return StreamingResponse(audio_bytes, media_type="audio/mp3", headers={
        "X-User-Text": user_text,
        "X-Assistant-Text": assistant_text
    })
