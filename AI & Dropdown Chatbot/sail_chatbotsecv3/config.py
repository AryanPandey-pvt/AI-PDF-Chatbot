import os
from werkzeug.security import generate_password_hash

# Application secret key (for Flask sessions)
SECRET_KEY = os.getenv("SECRET_KEY", "sail-chatbot-secure-key-change-in-production")

# Fetch the OpenAI API key from the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","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")

# Check if the API key was successfully loaded
if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found. Please set the environment variable.")

# Admin credentials
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_PASSWORD_HASH = generate_password_hash(ADMIN_PASSWORD)

# Database configuration
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sail_chatbot.db')

# External API keys
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Chatbot configuration
ENABLE_GENERAL_KNOWLEDGE = os.getenv("ENABLE_GENERAL_KNOWLEDGE", "True").lower() == "true"
GENERAL_KNOWLEDGE_DEFAULT = os.getenv("GENERAL_KNOWLEDGE_DEFAULT", "False").lower() == "true"

# Model configuration
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 150
TEMPERATURE_STANDARD = 0.3  # Lower temperature for more factual responses
TEMPERATURE_GENERAL = 0.7   # Higher temperature for general knowledge responses

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 10
MAX_REQUESTS_PER_DAY = 200

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "logs/chatbot.log"

# Voice configuration
ENABLE_VOICE = True
TEXT_TO_SPEECH_ENGINE = "pyttsx3"  # Options: "pyttsx3", "gtts"
SPEECH_RECOGNITION_SERVICE = "google"  # Currently only Google supported

# Feedback configuration
COLLECT_FEEDBACK = True
FEEDBACK_THRESHOLD = 3  # Minimum rating (out of 5) to consider feedback positive
