import whisper
from gtts import gTTS
from io import BytesIO
import requests
import os

# Load Whisper model once
model = whisper.load_model("tiny")

def transcribe_audio(file_path: str, language="hi") -> str:
    result = model.transcribe(file_path, language=language)
    return result["text"]

def call_openrouter(prompt: str, api_key: str, model_name="deepseek/deepseek-r1-0528:free") -> str:
    messages = [{"role": "user", "content": prompt}]
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model_name, "messages": messages},
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error: Could not get response from OpenRouter."

def generate_tts(text: str, lang="hi") -> BytesIO:
    tts = gTTS(text, lang=lang)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes
