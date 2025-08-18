from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, json
import os

app = Flask(__name__)
CORS(app)  # allow frontend to call this backend

# ✅ Load API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL = "deepseek/deepseek-r1-0528:free"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        # Send request to OpenRouter API
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": user_message}
                ],
            },
        )

        if response.status_code != 200:
            return jsonify({"reply": "⚠️ Error from OpenRouter API"}), 500

        data_json = response.json()
        bot_reply = data_json["choices"][0]["message"]["content"]

        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"reply": f"⚠️ Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
