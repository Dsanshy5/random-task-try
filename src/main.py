# src/main.py

import os
import json
import pandas as pd
import fitz # PyMuPDF
import joblib

from . import config
from . import feature_extractor

def process_pdfs():
    """
    Processes all PDFs in the input directory and generates structured JSON output.
    """
    # Load the trained model
    model = joblib.load(config.MODEL_PATH)
    feature_names = model.feature_names_in_

    input_files = [f for f in os.listdir(config.INPUT_DIR) if f.endswith('.pdf')]

    for pdf_filename in input_files:
        pdf_path = os.path.join(config.INPUT_DIR, pdf_filename)
        output_path = os.path.join(config.OUTPUT_DIR, pdf_filename.replace('.pdf', '.json'))
        
        doc = fitz.open(pdf_path)
        
        # --- Title Extraction Logic ---
        # A simple rule: find the largest text on the first page.
        # This can be made more sophisticated.
        title = ""
        max_font_size_page1 = 0
        if len(doc) > 0:
            page1_blocks = doc[0].get_text("dict")["blocks"]
            for block in page1_blocks:
                 if 'lines' not in block: continue
                 for line in block['lines']:
                     for span in line['spans']:
                         if span['size'] > max_font_size_page1:
                             max_font_size_page1 = span['size']
                             title = span['text'].strip()

        # --- Outline Extraction Logic ---
        outline = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_rect = page.rect
            
            font_sizes = [span['size'] for block in blocks if 'lines' in block for line in block['lines'] for span in line['spans']]
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

            for block in blocks:
                if 'lines' not in block: continue
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text'].strip()
                        if not text: continue
                        
                        features = feature_extractor.extract_features(
                            span, page_rect.width, page_rect.height, avg_font_size
                        )
                        
                        # Ensure features are in the same order as during training
                        features_df = pd.DataFrame([features])[feature_names]
                        
                        # Predict the label
                        prediction = model.predict(features_df)[0]
                        
                        if prediction in ["H1", "H2", "H3"]:
                            outline.append({
                                "level": prediction,
                                "text": text,
                                "page": page_num + 1 # Page numbers are 1-indexed in output
                            })
        
        # Final JSON structure
        result = {
            "title": title,
            "outline": outline
        }
        
        # Save the output JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
        
        doc.close()
    
    print(f"Processing complete, files are in '{config.OUTPUT_DIR}'.")


if __name__ == '__main__':
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    process_pdfs()