# src/document_parser.py
import fitz  # PyMuPDF
import os
import joblib  # To load your 1A model
import pandas as pd # To format data for the model

# IMPORTANT: Assuming your 1A feature extractor is in a file
# called 'feature_extractor.py' inside the 'src' folder.
from . import feature_extractor 

# Load the trained model from Round 1A once when the module is imported
MODEL_PATH = 'models/heading_classifier.joblib'
if os.path.exists(MODEL_PATH):
    HEADING_MODEL = joblib.load(MODEL_PATH)
    FEATURE_NAMES = HEADING_MODEL.feature_names_in_
else:
    HEADING_MODEL = None
    print(f"Warning: Heading model not found at {MODEL_PATH}")


def parse_pdf_to_chunks(pdf_path: str) -> list:
    """
    Parses a PDF using the trained 1A model to identify headings and
    then associates text paragraphs with those headings.
    """
    if not os.path.exists(pdf_path) or not HEADING_MODEL:
        return []

    doc = fitz.open(pdf_path)
    chunks = []
    current_heading = "Introduction"  # Default heading

    for page_num, page in enumerate(doc):
        # Calculate average font size on the page for feature normalization
        font_sizes = [span['size'] for block in page.get_text("dict")["blocks"] if 'lines' in block for line in block['lines'] for span in line['spans']]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

        # We process text block by block
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if 'lines' not in block:
                continue
            
            # Combine all spans in a block to treat it as one unit
            block_text = " ".join([span['text'] for line in block['lines'] for span in line['spans']]).strip()
            if not block_text:
                continue

            # --- REPLACEMENT LOGIC STARTS HERE ---
            
            # 1. Extract features for the block using your 1A feature extractor
            # We'll use the features of the first span to represent the block
            first_span = block['lines'][0]['spans'][0]
            features = feature_extractor.extract_features(first_span, page.rect.width, page.rect.height, avg_font_size)
            
            # 2. Predict using your trained 1A model
            features_df = pd.DataFrame([features])[FEATURE_NAMES] # Ensure column order is correct
            prediction = HEADING_MODEL.predict(features_df)[0]
            
            # 3. Use the prediction to guide the logic
            if prediction in ["H1", "H2", "H3"]:
                # The model says this block is a heading
                current_heading = block_text.replace('\n', ' ')
            else:
                # The model says this block is 'Body' text
                chunk = {
                    "content": block_text.replace('\n', ' '),
                    "source_pdf": os.path.basename(pdf_path),
                    "page_number": page_num + 1,
                    "parent_heading": current_heading
                }
                chunks.append(chunk)

            # --- REPLACEMENT LOGIC ENDS HERE ---
                
    doc.close()
    return chunks