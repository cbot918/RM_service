from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from openai import OpenAI
from supabase import create_client
import logging

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


app = Flask(__name__)

@app.route('/', methods=['POST'])
def home():
    data = request.json  # Get JSON data from the request body
    return jsonify({"message": "Hello, World!", "received_data": data})

@app.route('/parse-pdf', methods=['POST'])
def parse_pdf():
    logger.info("Received PDF parsing request")
    data = request.json
    return jsonify({"message": "PDF Parsing", "received_data": data})


@app.route('/generate-section-summary', methods=['POST'])
def generate_section_summary():
    logger.info("Received section summary generation request")
    data = request.json
    return jsonify({"message": "Section Summary Generation", "received_data": data})
  
