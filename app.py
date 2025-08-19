from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os

app = Flask(__name__)
CORS(app)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-r1-0528:free"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # ✅ Check if API key is set
        if not OPENROUTER_API_KEY:
            return jsonify({"reply": "⚠️ Server misconfigured: API key missing"}), 500

        data = request.get_json()
        user_message = data.get("message", "")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI assistant for Indian agriculture, designed to support farmers. "
                            "Your main goal is to provide clear, practical, and easy-to-understand advice about "
                            "farming, crops, soil, irrigation, fertilizers, pest control, weather updates, and government schemes. "
                            "Avoid unnecessary technical jargon—explain things simply, as if speaking to rural farmers. "
                            "Always reply in the SAME language as the user's message. "
                            "If the user writes in Hindi, reply in Hindi. "
                            "If the user writes in English, reply in English. "
                            "If the user mixes languages (Hinglish), reply in the same style. "
                            "Keep answers concise, friendly, and supportive, so farmers can take direct action from your guidance."
                        ),
                    },
                    {"role": "user", "content": user_message},
                ],
            },
        )

        if response.status_code != 200:
            return jsonify({"reply": "⚠️ Error from OpenRouter API"}), 500

        data_json = response.json()
        bot_reply = data_json["choices"][0]["message"].get("content", "").strip()

        # ✅ Fallback if model doesn't reply
        if not bot_reply:
            bot_reply = (
                "माफ़ कीजिए, मैं इस सवाल का सही जवाब नहीं दे पा रहा हूँ। "
                "कृपया दूसरा सवाल पूछें।"
            )

        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"reply": f"⚠️ Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
