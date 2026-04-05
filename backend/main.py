from flask import Flask, request, jsonify, send_from_directory
from google.cloud import texttospeech
import psycopg2
import requests
import re
import time
from time import sleep
from datetime import datetime, timedelta
from urllib.parse import quote
from flask import send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from collections import defaultdict
import io
from groq import Groq
from flask_cors import CORS
import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
import json
import schedule
import threading
from PIL import Image
import traceback

ALLOWED_CROPS = {"wheat", "rice", "maize", "onion", "garlic"}

# Database Helper
def get_cursor():
    return conn.cursor()

def get_user_key():
    data = request.get_json(silent=True) or {}
    return data.get("session_id") or get_remote_address()

# ---------------- SECURITY HELPERS ---------------- #

def sanitize_text(text, max_length=500):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text[:max_length]


def sanitize_session_id(session_id):
    if not re.match(r'^[a-zA-Z0-9\-]{1,100}$', session_id or ""):
        return "invalid_session"
    return session_id


def sanitize_city(city):
    if not re.match(r'^[a-zA-Z\s]{1,50}$', city or ""):
        return None
    return city.strip()


def sanitize_lang(lang):
    allowed = {"en", "hi", "gu", "ta", "mr", "ml"}
    return lang if lang in allowed else "en"


def validate_image(file):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.mimetype not in allowed_types:
        return False, "Invalid file type"

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    if size > 5 * 1024 * 1024:
        return False, "File too large"

    return True, None


def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {
                "pest_name": "Unknown",
                "confidence": "Low",
                "description": "Failed to parse AI response",
                "severity": "Unknown",
                "organic_treatments": [],
                "chemical_treatments": [],
                "prevention_tips": []
            }

# Normalize commodity names by removing extra details in parentheses and trimming whitespace
def normalize_commodity(name):
    if not name:
        return ""
    return name.split("(")[0].strip().lower()

MARKET_CACHE = []
MARKET_CACHE_TIME = 0
MARKET_CACHE_TTL = 1800  # 30 minutes
last_api_call_time = 0

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

app = Flask(__name__, static_folder="static")
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="redis://localhost:6379",
    default_limits=["200 per day", "50 per hour"]
)
CORS(app)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")
client = Groq(api_key=GROQ_API_KEY)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    raise ValueError("Missing OPENWEATHER_API_KEY in .env")

AGMARKNET_API_KEY = os.getenv("AGMARKNET_API_KEY")
if not AGMARKNET_API_KEY:
    raise ValueError("Missing AGMARKNET_API_KEY in .env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")
genai.configure(api_key=GEMINI_API_KEY)
tts_client = texttospeech.TextToSpeechClient()

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
@limiter.limit("8/minute")
def chat():
    try:
        data = request.get_json(force=True)

        user_message = sanitize_text(data.get("message"), 500)
        session_id = sanitize_session_id(data.get("session_id"))

        if not user_message:
            return jsonify({"reply": "Please enter a valid message."}), 400

        if not session_id:
            session_id = "default"

        system_prompt = (
            "You are KrishiMitra, an AI assistant for farmers. "
            "Respond only to queries related to agriculture, crops, soil, fertilizer, weather, or pest management. "
            "If the user speaks Hindi, always reply in Hindi (not Urdu). "
            "Politely redirect users to farming topics if the query is off-topic. "
            "Always strictly respond in the language the user used."
            "If query in English, respond only in English."
            "Never mix languages if query is in English, Hindi, Gujarati, Tamil, Marathi, or Malayalam or any other language - respond ONLY in that language's script. "
        )

        # save user message
        cur = get_cursor()
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, message) VALUES (%s,%s,%s)",
            (session_id, "user", user_message)
        )
        conn.commit()
        cur.close()

        # load previous messages
        cur = get_cursor()
        cur.execute(
            """
            SELECT role, message
            FROM chat_messages
            WHERE session_id=%s
            ORDER BY id DESC
            LIMIT 8
            """,
            (session_id,)
        )
        rows = cur.fetchall()
        cur.close()

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
        cur = get_cursor()
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, message) VALUES (%s,%s,%s)",
            (session_id, "assistant", ai_reply)
        )
        conn.commit()
        cur.close()

        return jsonify({"reply": ai_reply})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": "Error processing your message."}), 500

# Audio Transcription (Groq Whisper)
@app.route("/transcribe", methods=["POST"])
@limiter.limit("5/minute")
def transcribe_audio():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # File size validation (10MB max)
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > 10 * 1024 * 1024:
            return jsonify({"error": "File too large"}), 400

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        # Transcribe
        with open(temp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
            )

        # Cleanup
        os.remove(temp_path)

        return jsonify({"text": result.text.strip()}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Transcription failed"}), 500

# Text-to-Speech (Fast Google Translate TTS)
@app.route("/speak", methods=["POST"])
@limiter.limit("5/minute")
def speak():
    data = request.get_json()
    text = sanitize_text(data.get("text", ""), 1000)
    lang = sanitize_lang(data.get("lang", "en"))

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

# Weather (OpenWeatherMap)
@app.route("/weather")
@limiter.limit("5/minute")
def weather():
    try:

        city = sanitize_city(request.args.get("city"))

        if not city:
            return jsonify({"error": "Invalid city name"}), 400

        url = "https://api.openweathermap.org/data/2.5/forecast"

        params = {
            "q": f"{city},IN",
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        res = requests.get(url, params=params, timeout=15)
        data = res.json()

        if "list" not in data:
            return jsonify({"error": "City not found"}), 404

        current = data["list"][0]

        # current weather
        temperature = current["main"]["temp"]
        feels_like = current["main"]["feels_like"]
        humidity = current["main"]["humidity"]
        wind = current["wind"]["speed"]
        condition = current["weather"][0]["description"]

        rain_probability = int(current.get("pop", 0) * 100)

        # 3 day forecast
        forecast = []
        for i in range(0, 24, 8):
            day = data["list"][i]

            forecast.append({
                "temp_max": day["main"]["temp_max"],
                "temp_min": day["main"]["temp_min"],
                "condition": day["weather"][0]["description"],
                "rain_probability": int(day.get("pop", 0) * 100)
            })

        return jsonify({
            "temperature": temperature,
            "feels_like": feels_like,
            "humidity": humidity,
            "wind": wind,
            "condition": condition,
            "rain_probability": rain_probability,
            "forecast": forecast,
            "sunrise": data["city"]["sunrise"],
            "sunset": data["city"]["sunset"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Weather fetch failed"}), 500

# Weather by Coordinates (OpenWeatherMap)
@app.route("/weather_by_coords")
@limiter.limit("3/minute")
def weather_by_coords():

    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except:
        return jsonify({"error": "Invalid coordinates"}), 400

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"

    res = requests.get(url, timeout=10).json()

    if res.get("sys", {}).get("country") != "IN":
        return jsonify({"error": "Service available only in India"}), 400

    return jsonify({
        "city": res["name"],
        "temperature": res["main"]["temp"],
        "feels_like": res["main"]["feels_like"],
        "humidity": res["main"]["humidity"],
        "wind": res["wind"]["speed"],
        "condition": res["weather"][0]["description"],
        "rain_probability": 0,
        "sunrise": res["sys"]["sunrise"],
        "sunset": res["sys"]["sunset"]
    })
        
# Weather Forecast (OpenWeatherMap)
forecast_cache = {}
CACHE_TTL = 1800  # 30 minutes


def fetch_with_retry(url, retries=2):
    for i in range(retries):
        try:
            return requests.get(url, timeout=10)
        except requests.exceptions.Timeout:
            if i == retries - 1:
                raise
            sleep(1)

@app.route("/forecast")
@limiter.limit("8/minute")
def forecast():

    try:
        city = sanitize_city(request.args.get("city"))

        if not city:
            return jsonify({"error": "Invalid city name"}), 400

        now = datetime.utcnow().timestamp()

        # CACHE FIRST
        if city in forecast_cache:
            data, ts = forecast_cache[city]
            if now - ts < CACHE_TTL:
                return jsonify(data)

        url = f"https://api.openweathermap.org/data/2.5/forecast?q={quote(city)},IN&appid={OPENWEATHER_API_KEY}&units=metric"

        res = fetch_with_retry(url)
        data = res.json()

        if "list" not in data:
            print("Forecast API error:", data)
            return jsonify({"error": "Forecast unavailable"}), 500

        days = defaultdict(list)

        for item in data["list"]:
            if "dt_txt" not in item:
                continue
            date = item["dt_txt"].split(" ")[0]
            days[date].append(item)

        forecast_data = []
        today = datetime.utcnow().date()

        for date in sorted(days.keys()):

            date_obj = datetime.strptime(date, "%Y-%m-%d").date()

            if date_obj == today:
                continue

            items = days[date]
            if not items:
                continue

            temps = [i["main"]["temp"] for i in items if "main" in i]

            if not temps:
                continue

            forecast_data.append({
                "date": date,
                "temp_max": round(max(temps)),
                "temp_min": round(min(temps)),
                "condition": items[0]["weather"][0]["description"],
                "icon": items[0]["weather"][0]["icon"],
                "rain_probability": int(items[0].get("pop", 0) * 100)
            })

            if len(forecast_data) == 3:
                break

        # SAVE CACHE
        forecast_cache[city] = (forecast_data, now)

        return jsonify(forecast_data)

    except Exception as e:
        print("Forecast error:", e)

        return jsonify({
            "error": "Forecast API timeout"
        }), 500

# City Search (OpenWeatherMap Geocoding)
@app.route("/city_search")
@limiter.limit("5/minute")
def city_search():

    query = sanitize_text(request.args.get("q", ""), 50)

    if len(query) < 2:
        return jsonify([])

    url = "https://api.openweathermap.org/geo/1.0/direct"

    params = {
        "q": query,
        "limit": 10,
        "appid": OPENWEATHER_API_KEY
    }

    res = requests.get(url, params=params, timeout=15)
    data = res.json()

    indian_cities = []

    for city in data:
        if city.get("country") == "IN":
            indian_cities.append({
                "name": city["name"],
                "state": city.get("state", ""),
                "country": "IN"
            })

    return jsonify(indian_cities)

# Chat History Retrieval
@app.route("/chat_history", methods=["GET"])
def chat_history():
    try:
        session_id = sanitize_session_id(request.args.get("session_id"))

        cur = get_cursor()
        cur.execute(
            """
            SELECT role, message
            FROM chat_messages
            WHERE session_id=%s
            ORDER BY id ASC
            LIMIT 50
            """,
            (session_id,)
        )
        rows = cur.fetchall()
        cur.close()

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

# Save User Location
@app.route("/save_location", methods=["POST"])
@limiter.limit("5/minute")
def save_location():
    try:
        data = request.get_json()

        session_id = sanitize_session_id(data.get("session_id"))
        city = sanitize_city(data.get("city"))
        lat = data.get("lat")
        lon = data.get("lon")

        cur = get_cursor()
        cur.execute(
            """
            INSERT INTO user_locations (session_id, city, lat, lon)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_id)
            DO UPDATE SET 
                city = EXCLUDED.city,
                lat = EXCLUDED.lat,
                lon = EXCLUDED.lon,
                created_at = NOW()
            """,
            (session_id, city, lat, lon)
        )
        conn.commit()
        cur.close()

        return jsonify({"status": "saved"}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed"}), 500

# Fetch Market Data from Agmarknet API
def fetch_market_data():
    global last_api_call_time

    last_api_call_time = time.time()

    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    params = {
        "api-key": AGMARKNET_API_KEY,
        "format": "json",
        "limit": 500
    }

    for attempt in range(3):
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()

            records = data.get("records", [])

            print(f"Fetched {len(records)} records")

            cleaned = []
            for r in records:
                if not r.get("modal_price"):
                    continue

                cleaned.append({
                    "commodity": r.get("commodity", "").lower(),
                    "market": r.get("market"),
                    "state": r.get("state"),
                    "price": int(r.get("modal_price")),
                    "date": r.get("arrival_date")
                })

            if cleaned:
                return cleaned

        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)
            sleep(1)

    print("API FAILED → returning cache")
    return MARKET_CACHE

def fetch_from_db():
    try:
        cur = get_cursor()
        cur.execute("""
            SELECT commodity, market, state, price, date
            FROM market_history
            ORDER BY date DESC
            LIMIT 300
        """)
        rows = cur.fetchall()
        cur.close()

        return [
            {
                "commodity": r[0].lower(),
                "market": r[1],
                "state": r[2],
                "price": r[3],
                "date": str(r[4])
            }
            for r in rows
        ]
    except Exception as e:
        print("DB fallback failed:", e)
        return []
        
# Daily Job to fetch and store market data in PostgreSQL
def run_daily_market_job():
    global MARKET_CACHE, MARKET_CACHE_TIME

    print("Running daily market job...")

    records = fetch_market_data()

    if not records:
        print("No data fetched")
        return

    today = datetime.utcnow().date()
    cur = get_cursor()

    try:
        for item in records:
            if not item.get("price"):
                continue

            cur.execute("""
                INSERT INTO market_history (commodity, market, state, price, date)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (commodity, date)
                DO UPDATE SET
                    price = EXCLUDED.price,
                    market = EXCLUDED.market,
                    state = EXCLUDED.state
            """, (
                item["commodity"].title(),
                item["market"],
                item["state"],
                item["price"],
                today
            ))

        conn.commit()
        print("Daily market data stored")

    except Exception as e:
        conn.rollback()
        print("DAILY JOB ERROR:", e)

    finally:
        cur.close()

    # update cache also
    MARKET_CACHE = records
    MARKET_CACHE_TIME = time.time()


# Market Prices (Agmarknet)
@app.route("/market_prices")
@limiter.limit("10/minute")
def market_prices():

    global MARKET_CACHE, MARKET_CACHE_TIME

    commodity = sanitize_text(request.args.get("commodity"), 50)
    query = normalize_commodity(commodity) if commodity else None

    now = time.time()

    if MARKET_CACHE and (now - MARKET_CACHE_TIME < MARKET_CACHE_TTL):
        print("Using cached data")
        records = MARKET_CACHE

    else:
        print("Fetching from API...")
        records = fetch_market_data()

        today = datetime.utcnow().date()

        try:
            cur = get_cursor()
            cur.execute(
                "SELECT 1 FROM market_history WHERE date=%s LIMIT 1",
                (today,)
            )
            exists = cur.fetchone()
            cur.close()

            if not exists:
                print("No data for today → saving to DB")
                run_daily_market_job()
            else:
                print("Today's data already exists")

        except Exception as e:
            print("DB check failed:", e)

        if not records:
            print("API empty → using DB fallback")
            records = fetch_from_db()

        if not records:
            print("DB empty → using cache")
            records = MARKET_CACHE

        # update cache
        if records:
            MARKET_CACHE = records
            MARKET_CACHE_TIME = now

    filtered = []

    for item in records:
        name = normalize_commodity(item.get("commodity"))

        if query and query not in name:
            continue

        if item.get("price"):
            filtered.append({
                "commodity": item["commodity"].title(),
                "market": item["market"],
                "state": item["state"],
                "price": item["price"],
                "date": item["date"]
            })

    # optional: sort by price
    filtered = sorted(filtered, key=lambda x: x["price"], reverse=True)

    return jsonify({
        "data": filtered,
        "status": "ok"
    })

# Endpoint to trigger daily job manually (for testing)
@app.route("/run-daily-job")
def run_daily_job():
    run_daily_market_job()
    return jsonify({"status": "job executed"})

# Price Trend for last 30 days
@app.route("/price_trend")
def price_trend():

    commodity = sanitize_text(request.args.get("commodity"), 50)

    if not commodity:
        return jsonify([])

    try:
        cur = get_cursor()

        cur.execute("""
            SELECT date, AVG(price)
            FROM market_history
            WHERE LOWER(commodity) = %s
            AND date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY date
            ORDER BY date ASC
            LIMIT 10
        """, (commodity.lower(),))

        rows = cur.fetchall()
        cur.close()

        return jsonify([
            {
                "date": str(r[0]),
                "price": int(r[1])
            }
            for r in rows
        ])

    except Exception as e:
        print("Trend error:", e)
        return jsonify([])

# Pest Detection (Gemini Vision)
@app.route("/detect_pest", methods=["POST"])
@limiter.limit("5/minute")
def detect_pest():
    try:

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        is_valid, error = validate_image(file)
        if not is_valid:
            return jsonify({"error": error}), 400

        # session id from frontend
        session_id = sanitize_session_id(request.form.get("session_id", "default"))

        # selected language
        lang = sanitize_lang(request.form.get("lang", "en"))

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

        image = Image.open(file.stream)
        result = model.generate_content([prompt, image])
        text = result.text.strip()

        pest_data = safe_json_parse(text)

        # Save detection to PostgreSQL
        cur = get_cursor()
        cur.execute(
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
                None
            )
        )
        conn.commit()
        cur.close()
        return jsonify(pest_data), 200

    except Exception as e:
        print("Pest detection error:", e)
        return jsonify({"error": "Pest detection failed"}), 500

# Pest Detection Statistics
@app.route("/pest_stats", methods=["GET"])
def pest_stats():
    try:

        cur = get_cursor()
        cur.execute(
            """
            SELECT pest_name, COUNT(*) 
            FROM pest_detections
            GROUP BY pest_name
            ORDER BY COUNT(*) DESC
            """
        )

        rows = cur.fetchall()
        cur.close()

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

# Rate Limit Handler
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Too many requests. Please slow down."
    }), 429        

def start_scheduler():

    # run once at startup
    run_daily_market_job()

    # run every day
    schedule.every().day.at("06:00").do(run_daily_market_job)

    while True:
        schedule.run_pending()
        time.sleep(60)

# start scheduler in background thread
threading.Thread(target=start_scheduler, daemon=True).start()

# Run Server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
    host="0.0.0.0",
    port=port,
    debug=False,
    threaded=True
)