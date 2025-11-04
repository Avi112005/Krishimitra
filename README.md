# Krishimitra

---
## 🎯 Objective
The main objective of KrishiMitra is to develop an AI-powered, multilingual agricultural assistant that empowers farmers with real-time, personalized, and scientifically accurate guidance for crop management, pest detection, and sustainable farming practices.

It aims to bridge the gap between rural farmers and modern agri-tech, by providing easy access to expert advice through voice, text, and image-based interaction — all in the farmer’s native language.

---

## 🧪 How to Run the Project
- Python 3.11. or above 
### Local Setup:
```bash
# Clone the repo
git clone https://github.com/Avi112005/Krishimitra.git

# Navigate to backend
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

### Environment Variables (.env Setup)
# Inside your backend/ directory, create a file named .env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Start development server
python main.py

# Then open your browser and go to:
http://127.0.0.1:5000/
```