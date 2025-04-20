import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threading
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import requests
from functools import wraps
from openai import OpenAI
from supabase import create_client
import google.generativeai as genai
import logging
from flask_cors import CORS
from utils.ebook_handler import EBookHandler
from utils.summary_handler import SummaryHandler

# Configure logging to use StreamHandler (stdout) instead of FileHandler
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

logger.info("Application started successfully!")

# Load environment variables
load_dotenv()


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:4000", "https://resofront.vercel.app"]}})

# Initialize clients
supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_ANON_KEY')
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

if not supabase_url or not supabase_key:
    raise ValueError("Missing Supabase credentials")

supabase_client = create_client(supabase_url, supabase_key)

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        logger.info("Authenticating request")
        data = request.get_json()
        is_admin = data.get('is_administrator', False)

        if is_admin:
            logger.info("Creating admin client with service role")
            authenticated_supabase = create_client(
                supabase_url,
                os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            )
            request.supabase = authenticated_supabase
            logger.info("Request authenticated successfully as admin")
            return f(*args, **kwargs)

        # Regular user authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("Unauthorized request")
            return jsonify({'message': 'Unauthorized'}), 401
        
        token = auth_header.split(' ')[1]
        try:
            authenticated_supabase = create_client(
                supabase_url,
                supabase_key,
                {'headers': {'Authorization': f'Bearer {token}'}}
            )
            request.supabase = authenticated_supabase
            logger.info("Request authenticated successfully as regular user")
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return jsonify({'message': 'Authentication failed', 'error': str(e)}), 401
    
    return decorated


def process_ebook_async(ebook_url, book_id, page_count, file_type, callback_url, openai_client, supabase_client, genai, use_gemini):
    """Background PDF processing with webhook notification"""
    try:
        # Process PDF
        with app.app_context():
            ebook_handler = EBookHandler(openai_client, supabase_client, genai)
            ebook_handler.process_ebook(ebook_url, book_id, page_count, file_type, use_gemini)
            logging.info(f"✅ Ebook Processing Completed: book_id={book_id}")

            # Generate section summaries if TOC is available
            process_section_summary(openai_client, supabase_client, book_id)

        # Send webhook notification if callback_url is provided
        if callback_url:
            payload = {
                "book_id": book_id,
                "status": "completed",
                "message": "PDF processing and summary generation completed successfully."
            }
            try:
                response = requests.post(callback_url, json=payload, timeout=5)
                logging.info(f"✅ Webhook Sent! URL: {callback_url} | Status: {response.status_code} | Response: {response.text}")
            except requests.exceptions.RequestException as e:
                logging.error(f"❌ Webhook Failed! URL: {callback_url} | Error: {str(e)}")
    except Exception as e:
        logging.error(f"❌ Error in background processing: {str(e)}")
        if callback_url:
            requests.post(callback_url, json={"book_id": book_id, "status": "error", "message": str(e)})

def process_section_summary(openai_client, supabase_client, book_id):
    with app.app_context():
        book_response = supabase_client.table('library')\
            .select('id, title, author, toc')\
            .eq('id', book_id)\
            .execute()
    
        book_data = book_response.data[0] if book_response.data else None

        if not book_data:
            return jsonify({'error': 'Book not found'}), 404

        try:
            summary_handler = SummaryHandler(openai_client, supabase_client)

            result = summary_handler.process_all_sections(
                book_response.data[0]['id'],
                book_response.data[0]['title'],
                book_response.data[0]['author'],
                book_response.data[0]['toc']
            )
            return jsonify(result)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error generating section summaries: {str(e)}")
            return jsonify({'error': f'Error generating section summaries: {str(e)}'}), 500

@app.route('/', methods=['POST'])
def home():
    data = request.json  # Get JSON data from the request body
    return jsonify({"message": "Hello, World!", "received_data": data})


@app.route('/parse-ebook', methods=['POST'])
@require_auth
def parse_ebook():
    """Starts PDF processing in the background with webhook callback"""
    data = request.get_json()
    book_id = data.get('book_id')
    ebook_url = data.get('ebook_url')
    page_count = data.get('page_count', 1)
    use_gemini = data.get('use_gemini', False)
    file_type = data.get('file_type', 'pdf')
    callback_url = data.get('callback_url')  # Webhook URL (optional)

    print(f"use_gemini: {use_gemini}")

    if not book_id or not ebook_url:
        return jsonify({'error': 'Missing required parameters'}), 400

    logging.info(f"📢 Starting PDF Processing: book_id={book_id}, callback_url={callback_url}")

    # ✅ Fix: Pass required arguments explicitly to avoid request context issues
    thread = threading.Thread(
        target=process_ebook_async,
        args=(ebook_url, book_id, page_count, file_type, callback_url, openai_client, supabase_client, genai, use_gemini)
    )
    thread.start()

    return jsonify({'message': 'Processing started in background', 'book_id': book_id}), 202

@app.route('/generate-section-summary', methods=['POST'])
@require_auth
def generate_section_summary():
    logger.info("Received section summary generation request")
    data = request.get_json()
    
    required_fields = ['book_id']
    if not data or not all(field in data for field in required_fields):
        logger.warning("Missing required parameters")
        return jsonify({'error': 'Missing required parameters'}), 400
    
    output = process_section_summary(openai_client, supabase_client, data['book_id'])
    return output
  

@app.route('/parse-epub', methods=['POST'])
@require_auth
def parse_epub():
    """Starts EPUB processing in the background with webhook callback"""
    data = request.get_json()
    book_id = data.get('book_id')
    epub_url = data.get('epub_url')
    chapter_limit = data.get('chapter_limit', 100)
    use_gemini = data.get('use_gemini', False)
    callback_url = data.get('callback_url')  # Webhook URL (optional)

    if not book_id or not epub_url:
        return jsonify({'error': 'Missing required parameters'}), 400

    logging.info(f"📢 Starting EPUB Processing: book_id={book_id}, callback_url={callback_url}")

    # Start processing in background thread
    thread = threading.Thread(
        target=process_epub_async,
        args=(epub_url, book_id, chapter_limit, callback_url, openai_client, supabase_client, genai, use_gemini)
    )
    thread.start()

    return jsonify({'message': 'Processing started in background', 'book_id': book_id}), 202

def process_epub_async(epub_url, book_id, chapter_limit, callback_url, openai_client, supabase_client, genai, use_gemini):
    """Background EPUB processing with webhook notification"""
    try:
        # Process EPUB
        with app.app_context():
            ebook_handler = EBookHandler(openai_client, supabase_client, genai)
            ebook_handler.process_ebook(epub_url, book_id, chapter_limit, 'epub', use_gemini)
            logging.info(f"✅ EPUB Processing Completed: book_id={book_id}")

            # Generate section summaries if TOC is available
            process_section_summary(openai_client, supabase_client, book_id)

        # Send webhook notification if callback_url is provided
        if callback_url:
            payload = {
                "book_id": book_id,
                "status": "completed",
                "message": "EPUB processing and summary generation completed successfully."
            }
            try:
                response = requests.post(callback_url, json=payload, timeout=5)
                logging.info(f"✅ Webhook Sent! URL: {callback_url} | Status: {response.status_code} | Response: {response.text}")
            except requests.exceptions.RequestException as e:
                logging.error(f"❌ Webhook Failed! URL: {callback_url} | Error: {str(e)}")
    except Exception as e:
        logging.error(f"❌ Error in background processing: {str(e)}")
        if callback_url:
            requests.post(callback_url, json={"book_id": book_id, "status": "error", "message": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080)) #Use env var or default to 5000

    print(f"Running on port: {port}")

    app.run(debug=False, host='0.0.0.0', port=port)