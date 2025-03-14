import logging
import tempfile
import requests
import pdfplumber
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

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

    def write_to_supabase(self, book_id, pages_generator, batch_size=10):
        """
        Writes PDF page data to Supabase in batches to reduce memory usage.
        """
        try:
            logger.info(f"Writing pages to Supabase for book_id: {book_id}")
            batch = []

            for page in pages_generator:
                embedding = self.generate_embedding(page['text'])
                batch.append({
                    'book_id': book_id,
                    'page_number': page['page_number'],
                    'text': page['text'],
                    'embedding': embedding,
                })

                if len(batch) >= batch_size:  # ✅ Inserts every `batch_size` pages
                    self.supabase.table('book_pages').insert(batch).execute()
                    logger.info(f"Inserted {len(batch)} pages into Supabase")
                    batch.clear()

            # Insert remaining pages
            if batch:
                self.supabase.table('book_pages').insert(batch).execute()
                logger.info(f"Inserted last {len(batch)} pages into Supabase")

            return True
        except Exception as e:
            logger.error(f"Failed to write to Supabase: {str(e)}")
            raise Exception(f"Failed to write to Supabase: {str(e)}")

    def download_pdf(self, pdf_url):
        """
        Downloads a PDF from the given URL in chunks, avoiding full memory usage.
        """
        logger.info(f"Downloading PDF from URL: {pdf_url}")

        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/pdf'
        }

        with requests.get(pdf_url, stream=True, headers=headers) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_filename = temp_file.name
                for chunk in response.iter_content(chunk_size=8192):  # ✅ Streams in chunks
                    temp_file.write(chunk)

        logger.info(f"Successfully downloaded PDF to {temp_filename}")
        return temp_filename

    def extract_text_from_pdf(self, pdf_path, page_count):
        """
        Extracts text from a PDF file, processing pages one by one to reduce memory usage.
        """
        logger.info(f"Starting text extraction from PDF: {pdf_path}")
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")

            for page_num, page in enumerate(pdf.pages[:page_count], start=1):  # ✅ Limits to `page_count`
                logger.info(f"Processing page {page_num}/{page_count}")

                text = page.extract_text()
                if text:
                    yield {  # ✅ Yields one page at a time instead of storing all pages
                        "page_number": page_num,
                        "text": text
                    }
        
        logger.info("Completed text extraction from PDF")

    def process_pdf(self, pdf_url, book_id, page_count):
        """
        Main method to process a PDF file, optimized for memory usage.
        """
        try:
            temp_filename = self.download_pdf(pdf_url)
            logger.info("Extracting text from PDF")

            pages_generator = self.extract_text_from_pdf(temp_filename, page_count)
            self.write_to_supabase(book_id, pages_generator, batch_size=10)

            os.unlink(temp_filename)  # ✅ Delete temp file after processing
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
