# src/main_1b.py
import json
import os
from datetime import datetime
from . import document_parser
from . import semantic_analyzer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_challenge_1b(input_path: str, output_path: str):
    # 1. Load Inputs
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    pdf_filenames = [doc['filename'] for doc in input_data['documents']]
    persona = input_data['persona']['role']
    job_to_be_done = input_data['job_to_be_done']
    pdf_folder = os.path.join(os.path.dirname(input_path), "PDFs")

    # 2. Initialize Analyzer
    # Path to the model you downloaded in Step 1
    model_path = 'models/all-MiniLM-L6-v2' 
    analyzer = semantic_analyzer.SemanticAnalyzer(model_path)

    # 3. Process all PDFs to get chunks
    all_chunks = []
    for filename in pdf_filenames:
        full_pdf_path = os.path.join(pdf_folder, filename)
        chunks = document_parser.parse_pdf_to_chunks(full_pdf_path)
        all_chunks.extend(chunks)

    # 4. Generate Query Embedding
    query_text = f"{persona}: {job_to_be_done}"
    query_embedding = analyzer.get_embedding(query_text)

    # 5. Score all Chunks
    for chunk in all_chunks:
        chunk_embedding = analyzer.get_embedding(chunk['content'])
        chunk['relevance_score'] = analyzer.get_similarity(query_embedding, chunk_embedding)

    # 6. Rank Chunks and Sections
    # Sort all chunks by relevance for the 'subsection_analysis'
    all_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Aggregate scores to rank the parent headings for 'extracted_sections'
    heading_scores = {}
    for chunk in all_chunks:
        heading = chunk['parent_heading']
        score = chunk['relevance_score']
        if heading not in heading_scores or score > heading_scores[heading]['relevance_score']:
             heading_scores[heading] = {
                 'relevance_score': score,
                 'source_pdf': chunk['source_pdf'],
                 'page_number': chunk['page_number']
             }

    ranked_headings = sorted(heading_scores.items(), key=lambda item: item[1]['relevance_score'], reverse=True)

    # 7. Assemble and Save Final Output
    output_json = {
        "metadata": {
            "input_documents": pdf_filenames,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": [
            {
                "document": details['source_pdf'],
                "section_title": heading,
                "importance_rank": i + 1,
                "page_number": details['page_number']
            } for i, (heading, details) in enumerate(ranked_headings[:10]) # Get top 10 sections
        ],
        "subsection_analysis": [
            {
                "document": chunk['source_pdf'],
                "refined_text": chunk['content'],
                "page_number": chunk['page_number']
            } for chunk in all_chunks[:10] # Get top 10 subsections
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_json, f, indent=4)
    
    print(f"Successfully processed {input_path} and saved output to {output_path}")

# Example of how to run it
# NEW CODE - USE THIS
if __name__ == '__main__':
    # Define the base directory for the challenge collections
    collections_base_dir = os.path.join(PROJECT_ROOT, 'Challenge_1b')
    output_base_dir = os.path.join(PROJECT_ROOT, 'output_1b')

    # Ensure the main output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Automatically find all collection folders (e.g., "Collection 1", "Collection 2")
    collection_folders = [d for d in os.listdir(collections_base_dir) if os.path.isdir(os.path.join(collections_base_dir, d))]

    print(f"Found collections: {collection_folders}")

    # Loop through each collection and process it
    for folder_name in collection_folders:
        print(f"\n--- Processing {folder_name} ---")
        
        # Construct the input and output paths dynamically
        input_filename = os.path.join('Challenge_1b', folder_name, 'challenge1b_input.json')
        
        # Sanitize folder name for the output file
        safe_output_name = folder_name.replace(" ", "_").lower()
        output_filename = os.path.join('output_1b', f'{safe_output_name}_output.json')

        try:
            run_challenge_1b(input_filename, output_filename)
        except FileNotFoundError as e:
            print(f"Could not process {folder_name}. Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {folder_name}: {e}")