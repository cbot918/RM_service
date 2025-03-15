import logging
import tempfile
import requests
import pdfplumber
from openai import OpenAI
import os
from pdf2image import convert_from_path
import pytesseract

logger = logging.getLogger(__name__)

class PDFHandler:
    def __init__(self, openai_client, supabase_client):
        self.openai_client = openai_client
        self.supabase = supabase_client

    def generate_embedding(self, text):
        """Generates embeddings for the given text using OpenAI's API."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None  # ❌ Instead of crashing, return None to continue processing

    def write_to_supabase(self, book_id, page_data):
        """Writes a single page to Supabase immediately after processing it."""
        try:
            logger.info(f"Writing page {page_data['page_number']} to Supabase for book_id: {book_id}")
            self.supabase.table('book_pages').insert(page_data).execute()
        except Exception as e:
            logger.error(f"Failed to write to Supabase: {str(e)}")

    def download_pdf(self, pdf_url):
        """Downloads a PDF from the given URL using a streaming approach to avoid memory overload."""
        logger.info(f"Downloading PDF from URL: {pdf_url}")

        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/pdf'
        }

        with requests.get(pdf_url, stream=True, headers=headers) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_filename = temp_file.name
                for chunk in response.iter_content(chunk_size=8192):  # ✅ Streams instead of loading all
                    temp_file.write(chunk)

        logger.info(f"Successfully downloaded PDF to {temp_filename}")
        return temp_filename

    def extract_text_from_pdf(self, pdf_path, page_count, book_id):
        """
        Extracts text from a PDF file while keeping memory usage low by processing one page at a time.
        """
        logger.info(f"Starting text extraction from PDF: {pdf_path}")

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")

            for page_num in range(min(page_count, total_pages)):
                logger.info(f"Processing page {page_num + 1}/{page_count}")

                # ✅ Open and process one page at a time to free memory
                with pdfplumber.open(pdf_path) as temp_pdf:
                    page = temp_pdf.pages[page_num]
                    text = page.extract_text()

                if text:
                    embedding = self.generate_embedding(text)
                    page_data = {
                        "book_id": book_id,
                        "page_number": page_num + 1,
                        "text": text,
                        "embedding": embedding,
                    }

                    self.write_to_supabase(book_id, page_data)  # ✅ Insert immediately to reduce memory use

        logger.info("Completed text extraction from PDF")

    def is_text_based_pdf(self, pdf_path):
        """Checks if the PDF is text-based by attempting to extract text from the first page."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) > 0:
                    text = pdf.pages[0].extract_text()
                    # If we get substantial text, consider it text-based
                    return bool(text and len(text.strip()) > 50)
            return False
        except Exception as e:
            logger.error(f"Error checking PDF type: {str(e)}")
            return False
    
    def extract_text_using_ocr(self, pdf_path, page_count, book_id):
        """Extracts text from a PDF file using OCR."""
        logger.info(f"Starting OCR text extraction from PDF: {pdf_path}")
        
        try:
            images = convert_from_path(pdf_path, dpi=100)
            total_pages = len(images)
            
            for page_num in range(min(page_count, total_pages)):
                logger.info(f"Processing page {page_num + 1}/{page_count} with OCR")
                
                text = pytesseract.image_to_string(images[page_num],config="--psm 6")
                
                if text:
                    embedding = self.generate_embedding(text)
                    page_data = {
                        "book_id": book_id,
                        "page_number": page_num + 1,
                        "text": text,
                        "embedding": embedding,
                    }
                    
                    self.write_to_supabase(book_id, page_data)
                
                # Free up memory
                images[page_num] = None
                
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            raise

        logger.info("Completed OCR text extraction from PDF")

    def process_pdf(self, pdf_url, book_id, page_count):
        """Main method to process a PDF file, handling both text-based and OCR-based PDFs."""
        try:
            temp_filename = self.download_pdf(pdf_url)
            logger.info("Detecting PDF type and extracting text")

            if self.is_text_based_pdf(temp_filename):
                logger.info("Processing text-based PDF")
                self.extract_text_from_pdf(temp_filename, page_count, book_id)
            else:
                logger.info("Processing image-based PDF using OCR")
                self.extract_text_using_ocr(temp_filename, page_count, book_id)

            os.unlink(temp_filename)
            logger.info(f"Successfully processed PDF with {page_count} pages")

            return {
                'success': True,
                'message': 'PDF processed and stored successfully',
                'pageCount': page_count
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download PDF: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
