import openai
import pandas as pd
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import schedule
import time
import threading
import numpy as np
import html
from thefuzz import fuzz, process
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='logs/chatbot_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Use environment variable for API key
openai.api_key = os.getenv("key")

# Initialize conversation context with thread safety
conversation_contexts = {}
context_lock = threading.Lock()
model_lock = threading.Lock()

# Try to import sentence transformers, but don't fail if not available
try:
    from sentence_transformers import SentenceTransformer

    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    logging.warning("Sentence-transformers not available. Vector search disabled.")


def fuzzy_preprocess(text):
    """Preprocess text specifically for fuzzy matching"""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove common stop words
    stop_words = ['the', 'and', 'is', 'of', 'to', 'in', 'a', 'for', 'with', 'on', 'at']
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


def preprocess(text):
    """Preprocess text by lowercasing, removing special characters, and lemmatizing"""
    if not isinstance(text, str):
        return ""
    # Sanitize input to prevent XSS
    text = html.escape(text.lower())
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def load_data():
    """Load FAQ data from CSV file"""
    try:
        data = pd.read_csv('data/sail_faq.csv')
        # Sanitize data to prevent XSS
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].apply(lambda x: html.escape(str(x)) if isinstance(x, str) else x)
        logging.info(f"Data loaded successfully with {len(data)} rows")
    except FileNotFoundError:
        logging.warning("FAQ data file not found, creating default data")
        data = pd.DataFrame({
            'Level1': ['About SAIL'],
            'Level2': ['Vision'],
            'Level3': ['-'],
            'Response': ['SAIL aims to be a leading steel producer.']
        })
    return data


def train_and_save_model(data):
    """Train and save the model to a pickle file"""
    with model_lock:
        try:
            logging.info("Training model...")
            # Create model directory if it doesn't exist
            os.makedirs('model', exist_ok=True)
            with open('model/sail_tree.pkl', 'wb') as f:
                pickle.dump(data, f)
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            # Log error to file
            with open('logs/model_errors.log', 'a') as f:
                f.write(f"{datetime.now()}: {str(e)}\n")


def load_model():
    """Load the model from a pickle file"""
    with model_lock:
        try:
            with open('model/sail_tree.pkl', 'rb') as f:
                model_data = pickle.load(f)
                logging.info("Model loaded successfully")
                return model_data
        except FileNotFoundError:
            logging.warning("Model file not found, creating new model")
            # If model doesn't exist, create it
            data = load_data()
            train_and_save_model(data)
            with open('model/sail_tree.pkl', 'rb') as f:
                return pickle.load(f)


def get_generative_response(prompt, data=None, allow_general_knowledge=False):
    """Get a generative response using OpenAI API with context from knowledge base"""
    try:
        # Find relevant context from knowledge base
        relevant_context = ""
        if data is not None:
            # Simple keyword matching to find relevant entries
            keywords = prompt.lower().split()
            relevant_entries = []

            for _, row in data.iterrows():
                entry_text = f"{row['Level1']} {row['Level2']} {row['Level3']} {row['Response']}".lower()
                if any(keyword in entry_text for keyword in keywords):
                    relevant_entries.append(row['Response'])

            # Use top 3 most relevant entries as context
            if relevant_entries:
                relevant_context = "Information from SAIL knowledge base:\n" + "\n".join(relevant_entries[:3]) + "\n\n"

        # Construct a better prompt with context and instructions
        if allow_general_knowledge:
            enhanced_prompt = f"""
{relevant_context}
You are SAIL's AI assistant. Answer the following question using the information provided above when relevant.
If the information isn't in the provided context, you can use your general knowledge to provide a helpful response.
When using general knowledge, start your response with "Based on general knowledge: "

Question: {prompt}

Answer:
"""
        else:
            enhanced_prompt = f"""
{relevant_context}
You are SAIL's AI assistant. Answer the following question using the information provided above when relevant.
If the information isn't in the provided context, say you don't have that specific information.

Question: {prompt}

Answer:
"""

        # Use the current OpenAI API format
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("key"))

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are SAIL's AI assistant. Answer based on provided information or general knowledge if instructed."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.3 if not allow_general_knowledge else 0.7,  # Higher temperature for general knowledge
                max_tokens=150
            )

            return response.choices[0].message.content.strip()
        except (ImportError, AttributeError):
            # Fall back to legacy API if needed
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",  # Updated model
                prompt=enhanced_prompt,
                temperature=0.3 if not allow_general_knowledge else 0.7,
                max_tokens=150
            )
            return response.choices[0].text.strip()

    except Exception as e:
        # Log error
        logging.error(f"Error generating response: {e}")
        with open('logs/api_errors.log', 'a') as f:
            f.write(f"{datetime.now()}: {str(e)}\n")
        return "I'm sorry, I couldn't generate a response at this time."


def get_next_level_options(user_input, data, threshold=70):
    """Get next level options based on user input using multiple fuzzy matching strategies"""
    logging.debug(f"Finding options for input: {user_input}")

    # Sanitize and preprocess input
    user_input = fuzzy_preprocess(html.escape(user_input.strip()))

    level1_options = data['Level1'].dropna().unique()
    level2_options = data['Level2'].dropna().unique()
    level3_options = data['Level3'].dropna().unique()

    # Remove '-' from level3 options
    level3_options = [opt for opt in level3_options if opt != '-']

    # Preprocess all options for better matching
    level1_processed = [fuzzy_preprocess(opt) for opt in level1_options]
    level2_processed = [fuzzy_preprocess(opt) for opt in level2_options]
    level3_processed = [fuzzy_preprocess(opt) for opt in level3_options]

    # Try different matching strategies
    matchers = [
        (fuzz.ratio, threshold),
        (fuzz.partial_ratio, threshold),
        (fuzz.token_sort_ratio, threshold),
        (fuzz.token_set_ratio, threshold - 10)  # Lower threshold for token_set_ratio
    ]

    # Try matching with level1 options
    for matcher, thresh in matchers:
        if level1_processed:
            match = process.extractOne(user_input, level1_processed, scorer=matcher)
            if match and match[1] >= thresh:
                idx = level1_processed.index(match[0])
                matched_option = level1_options[idx]
                logging.debug(f"Level1 match found: {matched_option} with score {match[1]}")
                filtered = data[data['Level1'] == matched_option]
                return list(filtered['Level2'].dropna().unique())

    # Try matching with level2 options
    for matcher, thresh in matchers:
        if level2_processed:
            match = process.extractOne(user_input, level2_processed, scorer=matcher)
            if match and match[1] >= thresh:
                idx = level2_processed.index(match[0])
                matched_option = level2_options[idx]
                logging.debug(f"Level2 match found: {matched_option} with score {match[1]}")
                filtered = data[data['Level2'] == matched_option]
                level3_opts = list(filtered['Level3'].dropna().unique())
                level3_opts = [opt for opt in level3_opts if opt != '-']

                if not level3_opts:
                    if not filtered.empty:
                        return filtered.iloc[0]['Response']
                return level3_opts

    # Try matching with level3 options
    for matcher, thresh in matchers:
        if level3_processed:
            match = process.extractOne(user_input, level3_processed, scorer=matcher)
            if match and match[1] >= thresh:
                idx = level3_processed.index(match[0])
                matched_option = level3_options[idx]
                logging.debug(f"Level3 match found: {matched_option} with score {match[1]}")
                filtered = data[data['Level3'] == matched_option]
                if not filtered.empty:
                    return filtered.iloc[0]['Response']

    logging.debug("No match found in hierarchical options")
    return None


def suggest_correction(user_input, all_options, threshold=65):
    """Suggest a correction for user input based on available options"""
    processed_input = fuzzy_preprocess(user_input)
    processed_options = [fuzzy_preprocess(opt) for opt in all_options]

    match = process.extractOne(processed_input, processed_options, scorer=fuzz.token_set_ratio)
    if match and match[1] >= threshold:
        idx = processed_options.index(match[0])
        suggestion = all_options[idx]
        logging.debug(f"Suggested correction: {suggestion} with score {match[1]}")
        return suggestion
    return None


def initialize_vector_search():
    """Initialize sentence transformer model for vector search"""
    if not VECTOR_SEARCH_AVAILABLE:
        logging.warning("Vector search initialization skipped - sentence-transformers not available")
        return False

    global sentence_model, vector_db, vector_responses

    logging.info("Initializing vector search...")

    try:
        # Load sentence transformer model
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load data
        data = load_data()

        # Create vectors for all responses
        questions = []
        vector_responses = []

        for _, row in data.iterrows():
            # Create question from hierarchy
            question = f"{row['Level1']} {row['Level2']}"
            if row['Level3'] != '-':
                question += f" {row['Level3']}"

            questions.append(question)
            vector_responses.append(row['Response'])

        # Encode questions to vectors
        vector_db = sentence_model.encode(questions)

        logging.info(f"Vector search initialized with {len(vector_db)} entries")
        return True
    except Exception as e:
        logging.error(f"Error initializing vector search: {e}")
        return False


def vector_search(query, threshold=0.6):
    """Search for similar questions using vector similarity"""
    if not VECTOR_SEARCH_AVAILABLE or 'sentence_model' not in globals():
        logging.warning("Vector search called but not available")
        return None

    try:
        # Encode query
        query_vector = sentence_model.encode([query])

        # Calculate similarity
        similarities = np.dot(vector_db, query_vector.T).flatten()

        # Get best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        logging.debug(f"Vector search best match score: {best_score}")

        if best_score >= threshold:
            return vector_responses[best_idx]
        return None
    except Exception as e:
        logging.error(f"Error in vector search: {e}")
        return None


def classify_intent(user_input):
    """Classify user intent to route to appropriate handler"""
    # Preprocess input
    processed_input = fuzzy_preprocess(user_input.lower())
    logging.debug(f"Classifying intent for: {processed_input}")

    # Define intent patterns
    intent_patterns = {
        'weather': ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'climate'],
        'news': ['news', 'latest', 'update', 'headlines', 'recent', 'announcement'],
        'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
        'farewell': ['bye', 'goodbye', 'see you', 'talk to you later', 'farewell'],
        'thanks': ['thank', 'thanks', 'appreciate', 'grateful', 'gratitude'],
        'help': ['help', 'assist', 'support', 'guide', 'how to', 'explain'],
        'contact': ['contact', 'phone', 'email', 'reach', 'call', 'address'],
        'faq': []  # Default intent for FAQ lookup
    }

    # Check for intent matches
    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            if pattern in processed_input:
                logging.debug(f"Intent classified as: {intent}")
                return intent

    # Default to FAQ lookup
    logging.debug("Intent classified as: faq (default)")
    return 'faq'


def extract_entities(user_input, intent):
    """Extract relevant entities based on intent"""
    logging.debug(f"Extracting entities for intent: {intent}")

    if intent == 'weather':
        # Extract location for weather intent
        location_indicators = ['in', 'at', 'for', 'of']
        words = user_input.split()

        for i, word in enumerate(words):
            if word.lower() in location_indicators and i < len(words) - 1:
                location = words[i + 1]
                logging.debug(f"Extracted location: {location}")
                return location

        logging.debug("No location found, using default")
        return 'current location'  # Default

    return None


def manage_context(user_id, user_input=None, context_update=None):
    """Manage conversation context for a specific user"""
    with context_lock:
        logging.debug(f"Managing context for user {user_id}")

        # Initialize context if not exists
        if user_id not in conversation_contexts:
            conversation_contexts[user_id] = {
                'history': [],
                'current_topic': None,
                'last_question': None,
                'follow_up_mode': False,
                'allow_general_knowledge': False  # Added field for general knowledge
            }

        # Update context if provided
        if context_update:
            conversation_contexts[user_id].update(context_update)
            logging.debug(f"Updated context with: {context_update}")

        # Add user input to history if provided
        if user_input:
            conversation_contexts[user_id]['history'].append({
                'user': user_input,
                'timestamp': datetime.now()
            })

            # Limit history to last 10 exchanges
            if len(conversation_contexts[user_id]['history']) > 10:
                conversation_contexts[user_id]['history'] = conversation_contexts[user_id]['history'][-10:]

        return conversation_contexts[user_id]


def handle_follow_up_questions(user_id, user_input):
    """Handle follow-up questions based on conversation context"""
    context = manage_context(user_id)
    logging.debug(f"Checking for follow-up question. Follow-up mode: {context['follow_up_mode']}")

    # Check if in follow-up mode
    if context['follow_up_mode'] and context['current_topic']:
        topic = context['current_topic']
        logging.debug(f"Current topic: {topic}")

        # Load data
        data = load_model()

        # Find responses related to current topic
        if topic in data['Level1'].values:
            filtered = data[data['Level1'] == topic]
        elif topic in data['Level2'].values:
            filtered = data[data['Level2'] == topic]
        else:
            filtered = None

        if filtered is not None and not filtered.empty:
            # Try to find a match within the current topic
            for _, row in filtered.iterrows():
                question = f"{row['Level2']} {row['Level3']}".lower()
                score = fuzz.token_set_ratio(user_input.lower(), question)
                logging.debug(f"Follow-up match score: {score} for '{question}'")
                if score > 70:
                    logging.info(f"Follow-up response found for topic: {topic}")
                    return row['Response']

    logging.debug("No follow-up response found")
    return None


def get_topic_from_query(query, data):
    """Extract the most likely topic from a query"""
    # Preprocess query
    processed_query = fuzzy_preprocess(query)
    logging.debug(f"Finding topic for query: {processed_query}")

    # Check each level for the best match
    best_match = None
    best_score = 0

    for level in ['Level1', 'Level2', 'Level3']:
        options = data[level].dropna().unique()
        options = [opt for opt in options if opt != '-']

        if options:
            processed_options = [fuzzy_preprocess(opt) for opt in options]
            match = process.extractOne(processed_query, processed_options, scorer=fuzz.token_set_ratio)
            if match and match[1] > best_score:
                best_score = match[1]
                idx = processed_options.index(match[0])
                best_match = options[idx]
                logging.debug(f"Topic match found in {level}: {best_match} with score {match[1]}")

    return best_match


def get_weather_info(location):
    """Get weather information from external API"""
    try:
        import requests

        # Replace with actual API key and endpoint
        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            logging.warning("Weather API key not found")
            return f"Sorry, I'm not configured to provide weather information yet."

        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"

        logging.debug(f"Fetching weather for location: {location}")
        response = requests.get(url)
        data = response.json()

        if 'current' in data:
            temp = data['current']['temp_c']
            condition = data['current']['condition']['text']
            return f"The current weather in {location} is {condition} with a temperature of {temp}°C."
        else:
            logging.warning(f"No weather data found for location: {location}")
            return f"Sorry, I couldn't find weather information for {location}."
    except Exception as e:
        logging.error(f"Error getting weather: {e}")
        return "Sorry, I couldn't retrieve weather information at this time."


def get_news_updates(topic="steel industry"):
    """Get latest news updates from external API"""
    try:
        import requests

        # Replace with actual API key and endpoint
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            logging.warning("News API key not found")
            return f"Sorry, I'm not configured to provide news updates yet."

        url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}&pageSize=3"

        logging.debug(f"Fetching news for topic: {topic}")
        response = requests.get(url)
        data = response.json()

        if data['status'] == 'ok' and len(data['articles']) > 0:
            news = "Here are the latest news updates:\n\n"
            for article in data['articles']:
                news += f"- {article['title']}\n"
            return news
        else:
            logging.warning(f"No news found for topic: {topic}")
            return f"Sorry, I couldn't find any news about {topic}."
    except Exception as e:
        logging.error(f"Error getting news: {e}")
        return "Sorry, I couldn't retrieve news updates at this time."


def record_feedback(user_id, feedback, query, response, used_general_knowledge=False):
    """Record user feedback for continuous improvement"""
    try:
        # Create feedback directory if it doesn't exist
        os.makedirs('feedback', exist_ok=True)

        # Record feedback
        with open('feedback/user_feedback.csv', 'a') as f:
            f.write(f"{datetime.now()},{user_id},{feedback},{query},{response},{used_general_knowledge}\n")

        logging.info(f"Recorded {feedback} feedback from user {user_id}")
        return True
    except Exception as e:
        logging.error(f"Error recording feedback: {e}")
        return False


def analyze_feedback():
    """Analyze feedback to identify improvement areas"""
    try:
        feedback_data = pd.read_csv('feedback/user_feedback.csv',
                                    names=['timestamp', 'user_id', 'feedback', 'query', 'response',
                                           'used_general_knowledge'])

        # Calculate satisfaction rate
        positive_feedback = feedback_data[feedback_data['feedback'] == 'positive'].shape[0]
        total_feedback = feedback_data.shape[0]
        satisfaction_rate = positive_feedback / total_feedback if total_feedback > 0 else 0

        # Identify common negative feedback queries
        negative_feedback = feedback_data[feedback_data['feedback'] == 'negative']
        common_issues = negative_feedback['query'].value_counts().head(5)

        # Calculate satisfaction rate for general knowledge responses
        general_knowledge_feedback = feedback_data[feedback_data['used_general_knowledge'] == True]
        if len(general_knowledge_feedback) > 0:
            general_positive = general_knowledge_feedback[general_knowledge_feedback['feedback'] == 'positive'].shape[0]
            general_satisfaction_rate = general_positive / len(general_knowledge_feedback)
        else:
            general_satisfaction_rate = 0

        logging.info(f"Feedback analysis: {satisfaction_rate:.2f} satisfaction rate")
        return {
            'satisfaction_rate': satisfaction_rate,
            'general_knowledge_satisfaction': general_satisfaction_rate,
            'common_issues': common_issues.to_dict()
        }
    except Exception as e:
        logging.error(f"Error analyzing feedback: {e}")
        return None


def expand_knowledge_base():
    """Function to expand the knowledge base with more topics and questions"""
    try:
        data = load_data()

        # Add new categories and responses
        new_data = pd.DataFrame({
            'Level1': ['HR Policies', 'Technical Information', 'Customer Support', 'Products'],
            'Level2': ['Leave Policy', 'Steel Manufacturing', 'Order Tracking', 'Steel Grades'],
            'Level3': ['-', 'Blast Furnace', '-', 'Structural Steel'],
            'Response': [
                'SAIL offers various leave policies including casual leave, earned leave, and sick leave...',
                'Blast furnace is used for smelting to produce industrial metals using hot air...',
                'You can track your order by visiting our website and entering your order number...',
                'SAIL produces various structural steel grades including IS 2062, IS 808, and IS 1161...'
            ]
        })

        # Append new data to existing data
        expanded_data = pd.concat([data, new_data], ignore_index=True)

        # Save expanded data
        expanded_data.to_csv('data/sail_faq.csv', index=False)

        # Retrain model
        train_and_save_model(expanded_data)

        # Reinitialize vector search if available
        if VECTOR_SEARCH_AVAILABLE and 'sentence_model' in globals():
            initialize_vector_search()

        logging.info("Knowledge base expanded successfully")
        return True
    except Exception as e:
        logging.error(f"Error expanding knowledge base: {e}")
        return False


def scrape_website_content(url, category):
    """Scrape content from website and add to knowledge base"""
    try:
        import requests
        from bs4 import BeautifulSoup

        logging.info(f"Scraping content from {url} for category {category}")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract relevant content (customize based on website structure)
        content_div = soup.find('div', class_='content-area')

        if content_div:
            content = content_div.get_text()
        else:
            # Fallback to main content
            content = soup.get_text()

        # Process and add to knowledge base
        new_entry = pd.DataFrame({
            'Level1': [category],
            'Level2': ['Web Content'],
            'Level3': [url],
            'Response': [content[:500]]  # Limit length of response
        })

        # Add to existing data
        data = load_data()
        expanded_data = pd.concat([data, new_entry], ignore_index=True)
        expanded_data.to_csv('data/sail_faq.csv', index=False)

        # Retrain model
        train_and_save_model(expanded_data)

        # Reinitialize vector search if available
        if VECTOR_SEARCH_AVAILABLE and 'sentence_model' in globals():
            initialize_vector_search()

        logging.info(f"Added scraped content from {url} to knowledge base")
        return True
    except Exception as e:
        logging.error(f"Error scraping website: {e}")
        return False


def schedule_knowledge_update():
    """Schedule periodic updates to the knowledge base"""

    def update_job():
        logging.info("Performing scheduled knowledge base update...")
        try:
            # Scrape latest information from company website
            scrape_website_content("https://www.sail.co.in/news", "Latest News")
            # Update product information
            scrape_website_content("https://www.sail.co.in/products", "Products")
            logging.info("Knowledge base update completed")
        except Exception as e:
            logging.error(f"Error in scheduled update: {e}")

    # Schedule daily update at 2 AM
    schedule.every().day.at("02:00").do(update_job)

    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)

    # Run scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

    logging.info("Knowledge update scheduler started")
    return True


def initialize_chatbot():
    """Initialize all chatbot components"""
    try:
        logging.info("Starting chatbot initialization...")

        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('model', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('feedback', exist_ok=True)

        # Initialize data if not exists
        if not os.path.exists('data/sail_faq.csv'):
            logging.info("Creating initial FAQ dataset...")
            data = pd.DataFrame({
                'Level1': ['About SAIL', 'Products', 'HR Policies', 'Customer Support'],
                'Level2': ['Vision', 'Steel Grades', 'Leave Policy', 'Contact'],
                'Level3': ['-', 'Structural Steel', '-', '-'],
                'Response': [
                    'SAIL aims to be a leading steel producer in the world.',
                    'SAIL produces various structural steel grades including IS 2062, IS 808, and IS 1161.',
                    'SAIL offers various leave policies including casual leave, earned leave, and sick leave.',
                    'You can contact SAIL through Email: contact@sail.co.in, Phone: +91-11-2436-7481'
                ]
            })
            data.to_csv('data/sail_faq.csv', index=False)

        # Load and train model
        data = load_data()
        train_and_save_model(data)

        # Initialize vector search if available
        if VECTOR_SEARCH_AVAILABLE:
            initialize_vector_search()

        # Schedule knowledge updates
        schedule_knowledge_update()

        logging.info("Chatbot initialization complete!")
        return True
    except Exception as e:
        logging.error(f"Error initializing chatbot: {e}")
        return False


def enhanced_chatbot_response(user_input, user_id="anonymous", allow_general_knowledge=False):
    """Enhanced chatbot response function that uses all new features"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    logging.info(f"Processing input from user {user_id}: {user_input}")

    try:
        # Update user context
        context = manage_context(user_id, user_input, {'allow_general_knowledge': allow_general_knowledge})

        # Log user input
        with open('logs/chat_history.log', 'a') as f:
            f.write(f"{datetime.now()} - User {user_id}: {user_input}\n")

        # Sanitize input
        user_input = html.escape(user_input)

        # Load data early so it's available throughout the function
        try:
            data = load_model()
        except (FileNotFoundError, pickle.UnpicklingError):
            data = load_data()
            train_and_save_model(data)
            data = load_model()

        # Check for follow-up questions
        follow_up_response = handle_follow_up_questions(user_id, user_input)
        if follow_up_response:
            response_data = {
                "response": follow_up_response,
                "dropdown": None,
                "feedback": True
            }
            # Log response
            with open('logs/chat_history.log', 'a') as f:
                f.write(f"{datetime.now()} - Bot to {user_id}: {response_data['response']}\n")
            return response_data

        # Classify intent
        intent = classify_intent(user_input)

        # Handle different intents
        if intent == 'greeting':
            response_data = {
                "response": "Hello! How can I assist you with SAIL today?",
                "dropdown": ["About SAIL", "Products", "HR Policies", "Technical Information", "Customer Support"]
            }

        elif intent == 'farewell':
            response_data = {
                "response": "Thank you for chatting with SAIL assistant. Have a great day!",
                "dropdown": None
            }

        elif intent == 'thanks':
            response_data = {
                "response": "You're welcome! Is there anything else I can help you with?",
                "dropdown": ["Yes", "No"]
            }

        elif intent == 'help':
            response_data = {
                "response": "I can help you with information about SAIL, our products, HR policies, and more. What would you like to know?",
                "dropdown": ["About SAIL", "Products", "HR Policies", "Technical Information", "Customer Support"]
            }

        elif intent == 'weather':
            location = extract_entities(user_input, 'weather')
            weather_info = get_weather_info(location)
            response_data = {
                "response": weather_info,
                "dropdown": None
            }

        elif intent == 'news':
            news_info = get_news_updates()
            response_data = {
                "response": news_info,
                "dropdown": None
            }

        elif intent == 'contact':
            response_data = {
                "response": "You can contact SAIL through:\nEmail: contact@sail.co.in\nPhone: +91-11-2436-7481\nWebsite: www.sail.co.in",
                "dropdown": None
            }

        else:  # Default FAQ lookup
            # Try tree-based lookup first
            result = get_next_level_options(user_input, data)

            if result is None:
                # Try vector search if available
                vector_result = None
                if VECTOR_SEARCH_AVAILABLE and 'sentence_model' in globals():
                    vector_result = vector_search(user_input)

                if vector_result:
                    response_data = {
                        "response": vector_result,
                        "dropdown": None
                    }

                    # Update context for follow-up questions
                    manage_context(user_id, context_update={
                        'follow_up_mode': True,
                        'last_question': user_input,
                        'current_topic': get_topic_from_query(user_input, data)
                    })
                else:
                    # Try to suggest a correction
                    all_options = list(data['Level1'].dropna().unique()) + \
                                  list(data['Level2'].dropna().unique()) + \
                                  list(data['Level3'].dropna().unique())
                    all_options = [opt for opt in all_options if opt != '-']

                    suggestion = suggest_correction(user_input, all_options)
                    if suggestion:
                        response_data = {
                            "response": f"Did you mean '{suggestion}'? Please select from the options below:",
                            "dropdown": [suggestion]
                        }
                    else:
                        # Use generative response as fallback
                        generated_response = get_generative_response(user_input, data, allow_general_knowledge)
                        response_data = {
                            "response": generated_response,
                            "dropdown": None,
                            "feedback": True,
                            "used_general_knowledge": allow_general_knowledge
                        }
            elif isinstance(result, list):
                if result:
                    # Update context with current topic
                    current_topic = None
                    for level in ['Level1', 'Level2', 'Level3']:
                        matches = data[data[level].str.lower() == user_input.lower()]
                        if not matches.empty:
                            current_topic = matches.iloc[0][level]
                            break

                    if current_topic:
                        manage_context(user_id, context_update={
                            'current_topic': current_topic,
                            'follow_up_mode': True
                        })

                    response_data = {
                        "response": "Please choose one of the following options:",
                        "dropdown": result
                    }
                else:
                    response_data = {
                        "response": "No further options available.",
                        "dropdown": None
                    }
            elif isinstance(result, str):
                response_data = {
                    "response": result,
                    "dropdown": None
                }

                # Update context for potential follow-up
                manage_context(user_id, context_update={
                    'follow_up_mode': True,
                    'last_question': user_input
                })

        # Add feedback option to response
        if 'dropdown' not in response_data or response_data['dropdown'] is None:
            response_data['feedback'] = True

        # Log response
        with open('logs/chat_history.log', 'a') as f:
            f.write(f"{datetime.now()} - Bot to {user_id}: {response_data['response']}\n")

        return response_data

    except Exception as e:
        logging.error(f"Error in chatbot response: {e}")
        # Log error
        with open('logs/error.log', 'a') as f:
            f.write(f"{datetime.now()}: {str(e)}\n")
        return {
            "response": "I'm sorry, something went wrong. Please try again later.",
            "dropdown": None
        }


def test_components():
    """Test individual chatbot components"""
    logging.info("Testing chatbot components...")

    print("Testing intent classification...")
    test_inputs = ["hello", "bye", "thanks", "help", "weather in Delhi",
                   "latest news", "contact information", "about SAIL"]
    for input_text in test_inputs:
        intent = classify_intent(input_text)
        print(f"Input: {input_text} → Intent: {intent}")

    print("\nTesting fuzzy matching...")
    data = load_data()
    test_inputs = ["about company", "steel products", "HR policy", "contact details"]
    for input_text in test_inputs:
        result = get_next_level_options(input_text, data)
        print(f"Input: {input_text} → Result: {result}")

    if VECTOR_SEARCH_AVAILABLE and 'sentence_model' in globals():
        print("\nTesting vector search...")
        test_inputs = ["what is SAIL's vision?", "tell me about steel grades", "leave policy"]
        for input_text in test_inputs:
            result = vector_search(input_text)
            print(f"Input: {input_text} → Result: {result[:50]}..." if result else "No match")

    print("\nTesting generative responses...")
    test_inputs = ["What is the price of steel today?", "Tell me about SAIL's international operations"]
    for input_text in test_inputs:
        print(f"\nInput: {input_text}")
        print(f"Standard response: {get_generative_response(input_text, data, False)[:100]}...")
        print(f"General knowledge response: {get_generative_response(input_text, data, True)[:100]}...")

    print("\nTesting complete!")


# Initialize the chatbot when the module is imported
if __name__ == "__main__":
    initialize_chatbot()
    print("Chatbot initialized and ready to use!")
    print("Use enhanced_chatbot_response(user_input, user_id, allow_general_knowledge) to get responses.")
