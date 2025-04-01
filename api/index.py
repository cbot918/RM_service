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
import logging
from flask_cors import CORS
from utils.pdf_handler import PDFHandler
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

supabase = create_client(supabase_url, supabase_key)


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
        PDFHandler
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

@app.route('/', methods=['POST'])
def home():
    data = request.json  # Get JSON data from the request body
    return jsonify({"message": "Hello, World!", "received_data": data})


def process_pdf_async(pdf_url, book_id, page_count, callback_url, openai_client, supabase_client, toc, title, author):
    """Background PDF processing with webhook notification"""
    try:
        # Process PDF
        pdf_handler = PDFHandler(openai_client, supabase_client)
        pdf_handler.process_pdf(pdf_url, book_id, page_count)
        logging.info(f"✅ PDF Processing Completed: book_id={book_id}")

        # Generate section summaries if TOC is available
        try:
            summary_handler = SummaryHandler(openai_client, supabase_client)
            summary_handler.process_all_sections(
                book_id,
                title,
                author,
                toc
            )
            logging.info(f"✅ Section Summaries Generated: book_id={book_id}")
        except Exception as summary_error:
            logging.error(f"❌ Error generating section summaries: {str(summary_error)}")

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


@app.route('/parse-pdf', methods=['POST'])
def parse_pdf():
    """Starts PDF processing in the background with webhook callback"""
    data = request.get_json()
    book_id = data.get('book_id')
    pdf_url = data.get('pdf_url')
    page_count = data.get('page_count', 1)
    toc = data.get('toc')
    title = data.get('title')
    author = data.get('author')
    callback_url = data.get('callback_url')  # Webhook URL (optional)


    if not book_id or not pdf_url:
        return jsonify({'error': 'Missing required parameters'}), 400

    logging.info(f"📢 Starting PDF Processing: book_id={book_id}, callback_url={callback_url}")

    # ✅ Fix: Pass required arguments explicitly to avoid request context issues
    thread = threading.Thread(
        target=process_pdf_async,
        args=(pdf_url, book_id, page_count, callback_url, openai_client, supabase, toc, title, author)
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
    
    book_response = supabase.table('reading_records')\
        .select('id, title, author, toc')\
        .eq('id', data['book_id'])\
        .execute()
    
    book_data = book_response.data[0] if book_response.data else None

    if not book_data:
        return jsonify({'error': 'Book not found'}), 404
    

    try:
        summary_handler = SummaryHandler(openai_client, request.supabase)

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
  
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080)) #Use env var or default to 5000

    print(f"Running on port: {port}")

    app.run(debug=False, host='0.0.0.0', port=port)