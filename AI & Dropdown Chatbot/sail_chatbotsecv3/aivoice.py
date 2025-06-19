# aivoice.py (fixed handle_voice_query to use local Recognizer/Microphone)
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import pandas as pd
import logging
import os
import threading
import queue
import tempfile
import time
from datetime import datetime
from rapidfuzz import fuzz
from chatbot_model import enhanced_chatbot_response
import argparse

# —— Logging setup ——
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# File handler
file_handler = logging.FileHandler('logs/voice.log')
file_handler.setLevel(logging.DEBUG)
h_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(h_formatter)
logger.addHandler(file_handler)
# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(h_formatter)
logger.addHandler(console_handler)

# —— Parse CLI args ——
parser = argparse.ArgumentParser(description="Voice assistant config")
parser.add_argument("--general", action="store_true", help="Enable general knowledge fallback")
parser.add_argument("--sensitivity", type=int, default=150, help="Energy threshold (lower is more sensitive)")
parser.add_argument("--tts-engine", choices=['pyttsx3','gtts'], default='pyttsx3', help="TTS engine")
parser.add_argument("--language", default='en-IN', help="Speech recognition language")
args = parser.parse_args()

# —— Global config from args ——
ALLOW_GENERAL = args.general
SENSITIVITY   = args.sensitivity
TTS_ENGINE    = args.tts_engine
LANGUAGE      = args.language

# —— Load FAQ data ——
try:
    data = pd.read_csv('data/sail_faq.csv')
except Exception as e:
    logging.error(f"Error loading CSV data: {e}")
    data = pd.DataFrame({
        'Level1': ['Error'], 'Level2': ['Loading'], 'Level3': ['-'],
        'Response': ['Sorry, there was an error loading the knowledge base.']
    })

# —— Shared queue for recognized phrases ——
question_queue = queue.Queue()

# —— Global recognizer for background ——
recognizer_bg = sr.Recognizer()
recognizer_bg.energy_threshold = SENSITIVITY
recognizer_bg.dynamic_energy_threshold = True
mic_bg = sr.Microphone()

# —— Fuzzy + fallback search ——
def search_response(question, allow_general_knowledge=False):
    if not question:
        return {"response": "Sorry, I couldn't understand that.", "query": ""}
    q = question.lower().strip()
    best_score, best_resp = 0, None
    for _, row in data.iterrows():
        combined = ' '.join(str(x).lower() for x in [row['Level1'], row['Level2'], row['Level3']] if x and str(x)!='nan' and x!='-')
        score = fuzz.token_set_ratio(q, combined)
        logger.debug(f"Fuzzy: '{q}' vs '{combined}' -> {score}")
        if score > best_score:
            best_score, best_resp = score, row['Response']
    if best_score >= 60:
        return {"response": best_resp, "query": question, "feedback": True}
    try:
        response_data = enhanced_chatbot_response(q, "voice_user", allow_general_knowledge)
        response_data["query"] = question
        return response_data
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return {"response": "Sorry, I couldn't find an answer.", "query": question}

# —— TTS engines ——
def speak_pyttsx3(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def speak_gtts(text):
    try:
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.write_to_fp(fp)
            temp_path = fp.name
        os.system(f"mpg123 {temp_path} 2>/dev/null")
        os.unlink(temp_path)
    except Exception as e:
        logging.error(f"gTTS error: {e}")

# —— Background callback ——
def callback(recog, audio):
    try:
        text = recog.recognize_google(audio, language=LANGUAGE)
        logger.debug(f"Background recognized: {text}")
        question_queue.put(text)
    except sr.UnknownValueError:
        logger.debug("Background: Unintelligible speech")
    except sr.RequestError as e:
        logger.error(f"Background speech service error: {e}")
    except Exception as e:
        logger.error(f"Background listener exception: {e}")

# —— Worker thread ——
def worker():
    while True:
        q_text = question_queue.get()
        if q_text is None:
            break
        try:
            allow = ALLOW_GENERAL
            if "use general knowledge" in q_text.lower():
                allow = True
                q_text = q_text.lower().replace("use general knowledge", "").strip()
            resp = search_response(q_text, allow)
            tts = speak_pyttsx3 if TTS_ENGINE=='pyttsx3' else speak_gtts
            tts(resp['response'])
        except Exception as e:
            logger.error(f"Worker processing exception: {e}")
        finally:
            question_queue.task_done()

def worker():
    while True:
        q_text = question_queue.get()
        if q_text is None:
            break
        allow = ALLOW_GENERAL
        if "use general knowledge" in q_text.lower():
            allow = True
            q_text = q_text.lower().replace("use general knowledge", "").strip()
        resp = search_response(q_text, allow)
        tts = speak_pyttsx3 if TTS_ENGINE=='pyttsx3' else speak_gtts
        tts(resp['response'])
        question_queue.task_done()

# —— Synchronous handler for Flask ——
def handle_voice_query(allow_general_knowledge=False):
    # use fresh recognizer & mic to avoid nested context
    recog = sr.Recognizer()
    recog.energy_threshold = SENSITIVITY
    with sr.Microphone() as source:
        recog.adjust_for_ambient_noise(source, duration=0.2)
        audio = recog.listen(source, timeout=5, phrase_time_limit=6)
    try:
        text = recog.recognize_google(audio, language=LANGUAGE)
    except sr.UnknownValueError:
        return {"response":"Sorry, I couldn't understand that.","query":""}
    except sr.RequestError:
        return {"response":"Speech service error.","query":""}
    return search_response(text, allow_general_knowledge)

if __name__=='__main__':
    # start background listener
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    stop = recognizer_bg.listen_in_background(mic_bg, callback)
    try:
        while True: time.sleep(0.1)
    except KeyboardInterrupt:
        stop(wait_for_stop=False)
        question_queue.put(None)
        t.join()
