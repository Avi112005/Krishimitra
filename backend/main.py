from flask import Flask, request, jsonify, send_from_directory
from google.cloud import texttospeech
import psycopg2
import requests
from urllib.parse import quote
from flask import send_file
import io
from groq import Groq
from flask_cors import CORS
import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
import json
from datetime import datetime
from PIL import Image
import traceback

# Load environment variables
load_dotenv()

# PostgreSQL connection
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    port=os.getenv("DB_PORT")
)

cursor = conn.cursor()

app = Flask(__name__, static_folder=".")
CORS(app)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")
client = Groq(api_key=GROQ_API_KEY)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")
genai.configure(api_key=GEMINI_API_KEY)
tts_client = texttospeech.TextToSpeechClient()

# Directory setup
UPLOADS_DIR = "uploaded_images"
LOG_FILES = ["chat_logs.txt", "pest_uploads.txt", "system.log"]

os.makedirs(UPLOADS_DIR, exist_ok=True)
for file_name in LOG_FILES:
    path = os.path.join(UPLOADS_DIR, file_name)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("")

# Logging helpers
def log_event(filename, message, file="system.log"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(UPLOADS_DIR, file)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {filename}: {message}\n")

def log_chat_message(user_message, status):
    log_event("Chat", f"{user_message} | Status: {status}", "chat_logs.txt")

def log_pest_upload(filename, status):
    log_event("Pest", f"{filename} | Status: {status}", "pest_uploads.txt")

# Routes
@app.route("/", methods=["GET"])
def home():
    return send_from_directory("../frontend", "index.html")
    
@app.route("/health", methods=["GET"])
def health():
    """Simple uptime route for Render/UptimeRobot."""
    return jsonify({"status": "ok"}), 200


# Chat (Groq)
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)

        user_message = (data.get("message") or "").strip()
        session_id = data.get("session_id")

        if not user_message:
            return jsonify({"reply": "Please enter a valid message."}), 400

        if not session_id:
            session_id = "default"

        system_prompt = (
            "You are KrishiMitra, an AI assistant for farmers. "
            "Respond only to queries related to agriculture, crops, soil, fertilizer, weather, or pest management. "
            "If the user speaks Hindi, always reply in Hindi (not Urdu). "
            "Politely redirect users to farming topics if the query is off-topic. "
            "Always respond in the language the user uses."
            "Always reply in the SAME language as the user's LAST message. "
            "Do not switch languages unless the user switches language."
        )

        # save user message
        cursor.execute(
            "INSERT INTO chat_messages (session_id, role, message) VALUES (%s,%s,%s)",
            (session_id, "user", user_message)
        )
        conn.commit()

        # load previous messages
        cursor.execute(
            """
            SELECT role, message
            FROM chat_messages
            WHERE session_id=%s
            ORDER BY id DESC
            LIMIT 8
            """,
            (session_id,)
        )

        rows = cursor.fetchall()

        history = []
        for r in reversed(rows):
            history.append({
                "role": r[0],
                "content": r[1]
            })

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}] + history,
            temperature=0.5,
            max_tokens=900,
        )

        ai_reply = response.choices[0].message.content

        # save AI reply
        cursor.execute(
            "INSERT INTO chat_messages (session_id, role, message) VALUES (%s,%s,%s)",
            (session_id, "assistant", ai_reply)
        )
        conn.commit()

        return jsonify({"reply": ai_reply})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": "Error processing your message."}), 500

# Audio Transcription (Groq Whisper)
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
            # no language param = auto-detect
        )

        os.remove(temp_path)
        return jsonify({"text": result.text.strip()}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Transcription failed"}), 500

# ─────────────────────────────────────────────
# Text-to-Speech (Fast Google Translate TTS)
# ─────────────────────────────────────────────

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "en")

    voice_map = {
        "en": ("en-IN", "en-IN-Wavenet-B"),
        "hi": ("hi-IN", "hi-IN-Wavenet-A"),
        "gu": ("gu-IN", "gu-IN-Wavenet-A"),
        "ta": ("ta-IN", "ta-IN-Wavenet-A"),
        "mr": ("mr-IN", "mr-IN-Standard-A"),
        "ml": ("ml-IN", "ml-IN-Standard-A"),
    }

    language_code, voice_name = voice_map.get(lang, ("en-IN", "en-IN-Wavenet-B"))

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.05
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return send_file(io.BytesIO(response.audio_content), mimetype="audio/mpeg")

# Chat History Retrieval
@app.route("/chat_history", methods=["GET"])
def chat_history():
    try:
        session_id = request.args.get("session_id")

        cursor.execute(
            """
            SELECT role, message
            FROM chat_messages
            WHERE session_id=%s
            ORDER BY id ASC
            LIMIT 50
            """,
            (session_id,)
        )

        rows = cursor.fetchall()

        history = []
        for r in rows:
            history.append({
                "role": r[0],
                "message": r[1]
            })

        return jsonify(history)

    except Exception as e:
        traceback.print_exc()
        return jsonify([])

# Pest Detection (Gemini Vision)
@app.route("/detect_pest", methods=["POST"])
def detect_pest():
    try:

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded image
        filename = os.path.basename(file.filename)
        saved_path = os.path.join(UPLOADS_DIR, filename)
        file.save(saved_path)

        # session id from frontend
        session_id = request.form.get("session_id", "default")

        # selected language
        lang = request.form.get("lang", "en")

        language_instruction = {
            "en": "Respond in English.",
            "hi": "Respond ONLY in Hindi using Devanagari script.",
            "ml": "Respond in Malayalam.",
            "gu": "Respond ONLY in Gujarati script.",
            "ta": "Respond ONLY in Tamil script.",
            "mr": "Respond ONLY in Marathi using Devanagari script."
        }.get(lang, "Respond in English.")

        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.4,
                "max_output_tokens": 4096,
            },
        )

        prompt = f"""
You are KrishiMitra, an agricultural expert.

Analyze this crop leaf image and identify if any pest, insect, or disease is visible.

{language_instruction}
DO NOT return English if another language is requested.

Return strictly JSON:

{{
  "pest_name": "name or 'No Pest Detected'",
  "confidence": "High/Medium/Low",
  "description": "short summary",
  "severity": "Mild/Moderate/Severe/Unknown",
  "organic_treatments": ["..."],
  "chemical_treatments": ["..."],
  "prevention_tips": ["..."]
}}
"""

        with Image.open(saved_path) as image:
            result = model.generate_content([prompt, image])

        text = result.text.strip()

        pest_data = json.loads(text)

        # Save detection to PostgreSQL
        cursor.execute(
            """
            INSERT INTO pest_detections
            (session_id, pest_name, confidence, description, severity, image_path)
            VALUES (%s,%s,%s,%s,%s,%s)
            """,
            (
                session_id,
                pest_data.get("pest_name"),
                pest_data.get("confidence"),
                pest_data.get("description"),
                pest_data.get("severity"),
                saved_path
            )
        )

        conn.commit()

        # remove local image
        os.remove(saved_path)

        return jsonify(pest_data), 200

    except Exception as e:
        print("Pest detection error:", e)
        return jsonify({"error": "Pest detection failed"}), 500

# Pest Detection Statistics
@app.route("/pest_stats", methods=["GET"])
def pest_stats():
    try:

        cursor.execute(
            """
            SELECT pest_name, COUNT(*) 
            FROM pest_detections
            GROUP BY pest_name
            ORDER BY COUNT(*) DESC
            """
        )

        rows = cursor.fetchall()

        stats = []

        for r in rows:
            stats.append({
                "pest": r[0],
                "count": r[1]
            })

        return jsonify(stats)

    except Exception as e:
        print("Stats error:", e)
        return jsonify({"error": "Failed to load stats"}), 500

# Run Server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
    host="0.0.0.0",
    port=port,
    debug=False,
    threaded=True
)