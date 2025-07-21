# src/create_training_data.py

import os
import json
import pandas as pd
import fitz  # PyMuPDF

from . import config
from . import feature_extractor

def create_dataset():
    """
    Creates a labeled dataset from the sample PDFs and JSONs.
    """
    all_rows = []
    sample_files = [f for f in os.listdir(config.SAMPLES_DIR) if f.endswith('.pdf')]

    for pdf_filename in sample_files:
        json_filename = pdf_filename.replace('.pdf', '.json')
        pdf_path = os.path.join(config.SAMPLES_DIR, pdf_filename)
        json_path = os.path.join(config.SAMPLES_DIR, json_filename)

        if not os.path.exists(json_path):
            continue

        # Load the ground truth headings from the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        # Create a quick lookup for headings: {(page_num, text): "H1"}
        heading_lookup = {}
        if ground_truth.get("outline"):
            for heading in ground_truth["outline"]:
                # Normalize text for better matching
                clean_text = heading['text'].strip()
                heading_lookup[(heading['page'], clean_text)] = heading['level']

        # Process the PDF
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_rect = page.rect
            
            # Calculate average font size for normalization
            font_sizes = [span['size'] for block in blocks if 'lines' in block for line in block['lines'] for span in line['spans']]
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

            for block in blocks:
                if 'lines' not in block:
                    continue
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text'].strip()
                        if not text:
                            continue

                        # Extract features
                        features = feature_extractor.extract_features(
                            span, page_rect.width, page_rect.height, avg_font_size
                        )

                        # Label the data
                        label = heading_lookup.get((page_num, text), "Body")
                        features['label'] = label
                        all_rows.append(features)
        doc.close()

    # Create and save the DataFrame
    df = pd.DataFrame(all_rows)
    df.to_csv(config.TRAINING_DATA_PATH, index=False)
    print(f"Training data created and saved to {config.TRAINING_DATA_PATH}")

if __name__ == '__main__':
    create_dataset()