import logging
import os
import tiktoken  # Install via `pip install tiktoken`
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class SummaryHandler:
    def __init__(self, openai_client, supabase_client):
        self.openai_client = openai_client
        self.supabase = supabase_client

    def split_text_128k(self, text, max_tokens=128000):
        """
        Splits large text into 128K token chunks to fit within GPT-4o Mini's limits.
        """
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        tokens = encoding.encode(text)

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk = encoding.decode(tokens[i:i+max_tokens])
            chunks.append(chunk)

        return chunks

    def generate_section_summary(self, book_id, book_title, book_author, section_index, section_data):
        """
        Generates a summary for a book section, handling large inputs by processing in 128K chunks.
        """
        try:
            # Extract section information
            section_title = section_data['title']
            start_page = section_data['start_page']
            end_page = section_data['end_page']

            # Fetch section text from Supabase
            pages_response = self.supabase.table('book_pages')\
                .select('text')\
                .eq('book_id', book_id)\
                .gte('page_number', start_page)\
                .lte('page_number', end_page)\
                .execute()

            if not pages_response.data:
                raise ValueError(f"No pages found for section between pages {start_page} and {end_page}")

            # Combine all page text
            section_text = ' '.join(page['text'] for page in pages_response.data)

            # Split text if too long
            chunks = self.split_text_128k(section_text)
            summaries = []

            system_prompt = f"""
            Objective:
            Generate a detailed yet structured summary with the section title of "{section_title}" from "{book_title}" by "{book_author}" using the page text provided. 
            
            The summary should:

            Capture the main ideas and structure of the section.
            Include key details from the book.
            Retain relevant raw text to allow for use in podcasts, interactive discussions, or in-depth analysis.
            Provide clear references to specific topics so users can easily locate them in the book.

            Guidelines for Generating the Summary:
            1. Maintain a Balance Between Gist and Detail
            - Start with a high-level overview of the section.
            - Expand into detailed explanations of key concepts.
            - Use bullet points for clarity when listing examples or experiments.

            2. Retain Key Raw Text & Book Structure
            - Where appropriate, quote short excerpts verbatim from the book.
            - Describe key details in a way that preserves their impact.

            3. Ensure Clarity & Easy Navigation
            - Use section headings that align with the book's flow.
            - Provide clear references to major ideas, making it easy for users to connect their questions to specific concepts.
            - Use phrases from the book to keep the summary true to the original tone.
            """

            for idx, chunk in enumerate(chunks):
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Here is the page text (Part {idx+1}): {chunk}"}
                        ],
                        temperature=0.7
                    )

                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty response from OpenAI API")

                    summaries.append(response.choices[0].message.content)

                except Exception as e:
                    logger.error(f"OpenAI API error: {str(e)} - Using raw text as fallback.")
                    summaries.append(chunk)  # Use raw text as a fallback

            # Merge all partial summaries into a final summary
            final_summary = "\n\n".join(summaries)

            # Generate embedding
            embedding = self.generate_embedding(final_summary)

            # Store the summary in Supabase
            summary_data = {
                'book_id': book_id,
                "index": section_index,
                'section_title': section_title,
                'start_page': start_page,
                'end_page': end_page,
                'summary': final_summary,
                'embedding': embedding
            }
            
            return summary_data

        except Exception as e:
            logger.error(f"Error generating section summary: {str(e)}")
            raise

    def process_all_sections(self, book_id, book_title, book_author, toc):
        """
        Process all sections in the table of contents in parallel.
        """
        
        results = []
        # Use ThreadPoolExecutor to process sections in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Create a list of future tasks
            future_to_section = {
                executor.submit(self.generate_section_summary, book_id, book_title, book_author, section_index, section_data): 
                (section_index, section_data) for section_index, section_data in enumerate(toc)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_section):
                section_index, section_data = future_to_section[future]
                try:
                    summary_data = future.result()
                    results.append(summary_data)
                except Exception as e:
                    logger.error(f"Error processing section {section_data['title']}: {str(e)}")

        # Insert all results at once
        if results:
            self.supabase.table('book_sections').insert(results).execute()

        return {
            'success': True,
            'message': f'Processed {len(results)} sections',
        }

    def generate_embedding(self, text):
        """
        Generates embeddings for the given text using OpenAI's API.
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
            raise