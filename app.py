from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os, re  # added re for cleaning markdown

# RAG imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

app = Flask(__name__)
CORS(app)

# ======================
# 🔑 API Keys & Models
# ======================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
MODEL = "deepseek/deepseek-r1-0528:free"

# ✅ Load FAISS database for schemes
try:
    db = FAISS.load_local("vector_db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
except Exception as e:
    print("⚠️ Could not load FAISS DB:", e)
    db = None

# ======================
# 🤖 Chatbot Route
# ======================
@app.route("/chat", methods=["POST"])
def chat():
    try:
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
                            "Keep answers concise, friendly, and supportive."
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

        # ✅ Clean unwanted markdown like **bold**
        bot_reply = re.sub(r"\*\*(.*?)\*\*", r"\1", bot_reply)

        if not bot_reply:
            bot_reply = (
                "माफ़ कीजिए, मैं इस सवाल का सही जवाब नहीं दे पा रहा हूँ। "
                "कृपया दूसरा सवाल पूछें।"
            )

        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"reply": f"⚠️ Server error: {str(e)}"}), 500


# ======================
# 🌦️ Weather Route
# ======================
@app.route("/weather", methods=["POST"])
def weather():
    try:
        data = request.get_json()
        location = data.get("location", "")

        if not location:
            return jsonify({"weather": "⚠️ Please provide city or pincode"}), 400

        if not WEATHER_API_KEY:
            return jsonify({"weather": "⚠️ Weather API key missing"}), 500

        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url)

        if res.status_code != 200:
            return jsonify({"weather": "⚠️ Error fetching weather"}), 500

        weather_data = res.json()
        temp = weather_data["main"]["temp"]
        desc = weather_data["weather"][0]["description"]

        # ✅ Return structured JSON for frontend widgets
        return jsonify({
            "type": "weather",
            "location": location,
            "temp": temp,
            "desc": desc
        })

    except Exception as e:
        return jsonify({"weather": f"⚠️ Server error: {str(e)}"}), 500


# ======================
# 📜 Government Schemes (RAG)
# ======================
@app.route("/schemes", methods=["POST"])
def schemes():
    try:
        if db is None:
            return jsonify({"schemes": "⚠️ Scheme database not loaded"}), 500

        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"schemes": "⚠️ Please provide a query"}), 400

        docs = db.similarity_search(query, k=2)
        context = "\n".join([d.page_content for d in docs])

        return jsonify({"schemes": context if context else "⚠️ No relevant schemes found"})

    except Exception as e:
        return jsonify({"schemes": f"⚠️ Server error: {str(e)}"}), 500


# ======================
# 🚀 Run Server
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
