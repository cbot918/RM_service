import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import requests
from functools import wraps
from openai import OpenAI
from supabase import create_client
import logging

from utils.pdf_handler import PDFHandler
from utils.summary_handler import SummaryHandler

# Configure logging to use StreamHandler (stdout) instead of FileHandler
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

logger.info("Application started successfully!")

# Load environment variables
load_dotenv()


app = Flask(__name__)

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

@app.route('/parse-pdf', methods=['POST'])
@require_auth
def parse_pdf():
    logger.info("Received PDF parsing request")
    data = request.get_json()
    
    if not data or 'pdf_url' not in data or 'book_id' not in data:
        logger.warning("Missing required parameters")
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        pdf_handler = PDFHandler(openai_client, request.supabase)
        result = pdf_handler.process_pdf(
            data['pdf_url'],
            data['book_id'],
            data.get('page_count')
        )
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to download PDF: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500


@app.route('/generate-section-summary', methods=['POST'])
@require_auth
def generate_section_summary():
    logger.info("Received section summary generation request")
    data = request.get_json()
    
    required_fields = ['book_id', 'toc']
    if not data or not all(field in data for field in required_fields):
        logger.warning("Missing required parameters")
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        summary_handler = SummaryHandler(openai_client, request.supabase)

        result = summary_handler.process_all_sections(
            data['book_id'],
            data['book_title'],
            data['book_author'],
            data['toc']
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