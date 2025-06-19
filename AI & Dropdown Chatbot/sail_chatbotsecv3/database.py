import sqlite3
import os
import logging
from datetime import datetime
import bcrypt

# Get the database path from config
from config import DATABASE_PATH

# Set up logging
logging.basicConfig(
    filename='logs/database.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_db_connection():
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def close_connection(conn):
    """Close the database connection"""
    if conn:
        conn.close()


def init_db():
    """Initialize the database with required tables"""
    try:
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Create users table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS users
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           email
                           TEXT
                           UNIQUE
                           NOT
                           NULL,
                           password
                           TEXT
                           NOT
                           NULL,
                           approved
                           INTEGER
                           DEFAULT
                           0,
                           created_at
                           TEXT
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Create password_requests table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS password_requests
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           email
                           TEXT
                           NOT
                           NULL,
                           token
                           TEXT,
                           expiry
                           DATETIME,
                           status
                           TEXT,
                           request_time
                           DATETIME
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Create account_requests table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS account_requests
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           email
                           TEXT
                           NOT
                           NULL,
                           password
                           TEXT
                           NOT
                           NULL,
                           created_at
                           TEXT
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Create chat_logs table for tracking conversations
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS chat_logs
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           INTEGER
                           NOT
                           NULL,
                           user_input
                           TEXT
                           NOT
                           NULL,
                           bot_response
                           TEXT
                           NOT
                           NULL,
                           used_general_knowledge
                           INTEGER
                           DEFAULT
                           0,
                           timestamp
                           TEXT
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           id
                       )
                           )
                       ''')

        # Create feedback table for user feedback
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS feedback
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           INTEGER
                           NOT
                           NULL,
                           chat_log_id
                           INTEGER,
                           rating
                           INTEGER
                           NOT
                           NULL,
                           comments
                           TEXT,
                           timestamp
                           TEXT
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           id
                       ),
                           FOREIGN KEY
                       (
                           chat_log_id
                       ) REFERENCES chat_logs
                       (
                           id
                       )
                           )
                       ''')

        # Create voice_logs table for voice interactions
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS voice_logs
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           INTEGER
                           NOT
                           NULL,
                           query
                           TEXT
                           NOT
                           NULL,
                           response
                           TEXT
                           NOT
                           NULL,
                           used_general_knowledge
                           INTEGER
                           DEFAULT
                           0,
                           timestamp
                           TEXT
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           id
                       )
                           )
                       ''')

        conn.commit()
        conn.close()

        logging.info("Database initialized successfully")
        print("✅ Database and tables created successfully!")
        return True
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        print(f"❌ Error initializing database: {e}")
        return False


def add_user(email, password, approved=0):
    """Add a new user to the database"""
    try:
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO users (email, password, approved) VALUES (?, ?, ?)",
            (email, hashed_password, approved)
        )

        user_id = cursor.lastrowid

        conn.commit()
        conn.close()

        logging.info(f"Added new user: {email}")
        return user_id
    except sqlite3.IntegrityError:
        logging.warning(f"Attempted to add duplicate user: {email}")
        return None
    except Exception as e:
        logging.error(f"Error adding user: {e}")
        return None


def add_account_request(email, password):
    """Add a new account request"""
    try:
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO account_requests (email, password) VALUES (?, ?)",
            (email, hashed_password)
        )

        request_id = cursor.lastrowid

        conn.commit()
        conn.close()

        logging.info(f"Added new account request: {email}")
        return request_id
    except Exception as e:
        logging.error(f"Error adding account request: {e}")
        return None


def add_password_request(email, token, expiry):
    """Add a new password reset request"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO password_requests (email, token, expiry, status) VALUES (?, ?, ?, ?)",
            (email, token, expiry, "pending")
        )

        request_id = cursor.lastrowid

        conn.commit()
        conn.close()

        logging.info(f"Added new password reset request for: {email}")
        return request_id
    except Exception as e:
        logging.error(f"Error adding password request: {e}")
        return None


def log_chat_interaction(user_id, user_input, bot_response, used_general_knowledge=False):
    """Log chat interactions to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO chat_logs (user_id, user_input, bot_response, used_general_knowledge, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, user_input, bot_response, 1 if used_general_knowledge else 0,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

        chat_log_id = cursor.lastrowid

        conn.commit()
        conn.close()

        logging.info(f"Logged chat interaction for user {user_id}")
        return chat_log_id
    except Exception as e:
        logging.error(f"Error logging chat interaction: {e}")
        return None


def log_voice_interaction(user_id, query, response, used_general_knowledge=False):
    """Log voice interactions to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO voice_logs (user_id, query, response, used_general_knowledge, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, query, response, 1 if used_general_knowledge else 0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

        voice_log_id = cursor.lastrowid

        conn.commit()
        conn.close()

        logging.info(f"Logged voice interaction for user {user_id}")
        return voice_log_id
    except Exception as e:
        logging.error(f"Error logging voice interaction: {e}")
        return None


def save_feedback(user_id, chat_log_id, rating, comments=None):
    """Save user feedback about a chat interaction"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO feedback (user_id, chat_log_id, rating, comments, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, chat_log_id, rating, comments, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

        feedback_id = cursor.lastrowid

        conn.commit()
        conn.close()

        logging.info(f"Saved feedback for user {user_id}, chat {chat_log_id}, rating: {rating}")
        return feedback_id
    except Exception as e:
        logging.error(f"Error saving feedback: {e}")
        return None


def get_user_chat_history(user_id, limit=20):
    """Get chat history for a specific user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        history = cursor.execute(
            "SELECT * FROM chat_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()

        conn.close()
        return history
    except Exception as e:
        logging.error(f"Error retrieving chat history: {e}")
        return []


def get_feedback_statistics():
    """Get statistics about user feedback"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get overall feedback statistics
        cursor.execute("""
                       SELECT COUNT(*)                                     as total_feedback,
                              SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive_feedback,
                              AVG(rating)                                  as average_rating
                       FROM feedback
                       """)
        overall_stats = cursor.fetchone()

        # Get feedback statistics for general knowledge responses
        cursor.execute("""
                       SELECT COUNT(*)                                       as total_feedback,
                              SUM(CASE WHEN f.rating >= 4 THEN 1 ELSE 0 END) as positive_feedback,
                              AVG(f.rating)                                  as average_rating
                       FROM feedback f
                                JOIN chat_logs c ON f.chat_log_id = c.id
                       WHERE c.used_general_knowledge = 1
                       """)
        general_knowledge_stats = cursor.fetchone()

        conn.close()

        return {
            'overall': {
                'total': overall_stats['total_feedback'],
                'positive': overall_stats['positive_feedback'],
                'average_rating': overall_stats['average_rating']
            },
            'general_knowledge': {
                'total': general_knowledge_stats['total_feedback'],
                'positive': general_knowledge_stats['positive_feedback'],
                'average_rating': general_knowledge_stats['average_rating']
            }
        }
    except Exception as e:
        logging.error(f"Error retrieving feedback statistics: {e}")
        return {
            'overall': {'total': 0, 'positive': 0, 'average_rating': 0},
            'general_knowledge': {'total': 0, 'positive': 0, 'average_rating': 0}
        }


# Initialize the database if this script is run directly
if __name__ == "__main__":
    init_db()
