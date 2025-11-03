from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
from flask_cors import CORS
import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
import json
from datetime import datetime
from PIL import Image, UnidentifiedImageError
import io
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()
app = Flask(__name__, static_folder=".")
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")
client = Groq(api_key=GROQ_API_KEY)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")
genai.configure(api_key=GEMINI_API_KEY)

def log_event(filename, message, file="system.log"):
    """Generic logger for all events"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {filename}: {message}\n")

def log_chat_message(user_message, status):
    log_event("Chat", f"{user_message} | Status: {status}", "chat_logs.txt")

def log_pest_upload(filename, status):
    log_event("Pest", f"{filename} | Status: {status}", "pest_uploads.txt")

UPLOADS_DIR = "uploaded_images"
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"reply": "Please enter a valid message."}), 400

        system_prompt = (
            "You are KrishiMitra, an AI assistant for farmers. "
            "Respond only to queries related to agriculture, crops, soil, fertilizer, weather, or pest management. "
            "If user speaks Hindi, always reply in Hindi (not Urdu). "
            "Politely redirect users to farming topics if the query is off-topic. "
            "Always respond in the language the user uses."
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=400
        )

        ai_reply = response.choices[0].message.content
        log_chat_message(user_message, "Response OK")
        return jsonify({"reply": ai_reply})

    except Exception as e:
        print("Chat error:", e)
        log_chat_message(str(e), "Chat generation failed")
        return jsonify({"reply": "Error processing your message."}), 500

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        with open(temp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language="hi"
            )

        os.remove(temp_path)
        return jsonify({"text": result.text.strip()})

    except Exception as e:
        print("Transcription error:", e)
        return jsonify({"error": "Transcription failed"}), 500

#  Pest Detection
@app.route("/detect_pest", methods=["POST"])
def detect_pest():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        image_bytes = file.read()

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()
        except UnidentifiedImageError:
            return jsonify({"error": "Invalid or corrupted image file"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(UPLOADS_DIR, f"{timestamp}_{file.filename.replace(' ', '_')}")
        with open(save_path, "wb") as out:
            out.write(image_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            img_path = tmp.name

        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.4,
                "max_output_tokens": 4096
            },
        )

        prompt = """
        You are KrishiMitra, an agricultural expert.
        Analyze this crop leaf image and identify if any pest, insect, or disease is visible.

        Return strictly JSON:

        {
          "pest_name": "name or 'No Pest Detected'",
          "confidence": "High/Medium/Low",
          "description": "short summary",
          "severity": "Mild/Moderate/Severe/Unknown",
          "organic_treatments": ["..."],
          "chemical_treatments": ["..."],
          "prevention_tips": ["..."]
        }
        """

        with Image.open(img_path) as image:
            result = model.generate_content([prompt, image])

        os.remove(img_path)

        text = ""
        if result and getattr(result, "candidates", None):
            for c in result.candidates:
                for part in getattr(c.content, "parts", []):
                    text += getattr(part, "text", "")

        if not text.strip():
            raise ValueError("Empty response from model")

        try:
            pest_data = json.loads(text)
            log_pest_upload(file.filename, "Detection OK")
        except json.JSONDecodeError:
            pest_data = {
                "pest_name": "Unknown Pest",
                "confidence": "N/A",
                "description": text.strip() or "Unable to parse model output.",
                "severity": "Unknown",
                "organic_treatments": [],
                "chemical_treatments": [],
                "prevention_tips": []
            }
            log_pest_upload(file.filename, "JSON parse fallback")

        return jsonify(pest_data)

    except Exception as e:
        print("Pest detection error:", e)
        log_pest_upload(file.filename if "file" in request.files else "Unknown", f"Error: {e}")
        return jsonify({"error": "Pest detection failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)