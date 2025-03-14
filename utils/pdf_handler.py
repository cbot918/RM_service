import logging
import tempfile
import requests
import pdfplumber
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

# {
#   "pdf_url": "https://khuipsgjvqjwydhzodjt.supabase.co/storage/v1/object/public/pdfs/f90d8869-9a47-4165-ad23-b1d463895b36/89bcb8de-7296-4e0c-b9e5-93d853e6dcd4.1984",
#   "book_id": "e33ce238-7468-4b8b-a586-8876c757f5b6", 
#   "page_count": 394,
#   "is_administrator": true
# }

class PDFHandler:
    def __init__(self, openai_client, supabase_client):
        self.openai_client = openai_client
        self.supabase = supabase_client

    def generate_embedding(self, text):
        """
        Generates embeddings for the given text using OpenAI's API
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise Exception(f"Failed to generate embedding: {str(e)}")

    def write_to_supabase(self, book_id, pages_data, page_count):
        """
        Writes PDF page data and embeddings to Supabase book_pages table
        """
        try:
            logger.info(f"Writing {len(pages_data)} pages to Supabase for book_id: {book_id}")
            records = []
            for page in pages_data:
                if page_count and page['page_number'] <= page_count:
                    # embedding = self.generate_embedding(page['text'])
                    record = {
                        'book_id': book_id,
                        'page_number': page['page_number'],
                        'text': page['text'],
                        # 'embedding': embedding,
                    }
                    records.append(record)
            
            self.supabase.table('book_pages').insert(records).execute()
            logger.info(f"Successfully wrote {len(records)} pages to Supabase")
            return True
        except Exception as e:
            logger.error(f"Failed to write to Supabase: {str(e)}")
            raise Exception(f"Failed to write to Supabase: {str(e)}")

    def download_pdf(self, pdf_url):
        """
        Downloads a PDF from the given URL and saves it to a temporary file.
        """
        logger.info(f"Downloading PDF from URL: {pdf_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(pdf_url, stream=True, headers=headers)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_filename = temp_file.name
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
        
        logger.info(f"Successfully downloaded PDF to {temp_filename}")
        return temp_filename

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file
        """
        logger.info(f"Starting text extraction from PDF: {pdf_path}")
        structured_data = {
            "pages": []
        }

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                logger.debug(f"Processing page {page_num}/{total_pages}")
                text = page.extract_text()
                structured_data["pages"].append({
                    "page_number": page_num,
                    "text": text
                })

        logger.info("Completed text extraction from PDF")
        return structured_data

    def process_pdf(self, pdf_url, book_id, page_count):
        """
        Main method to process a PDF file
        """
        try:
            temp_filename = self.download_pdf(pdf_url)
            logger.info("Extracting text from PDF")
            text_content = self.extract_text_from_pdf(temp_filename)
            self.write_to_supabase(book_id, text_content['pages'], page_count)
            os.unlink(temp_filename)
            logger.info(f"Successfully processed PDF with {len(text_content['pages'])} pages")
            return {
                'success': True,
                'message': 'PDF processed and stored successfully',
                'pageCount': len(text_content['pages'])
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download PDF: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise 