# src/feature_extractor.py

import re

def extract_features(span, page_width, page_height, avg_font_size):
    """
    Extracts features from a single text span (a block of text from PyMuPDF).

    Args:
        span (dict): The text span dictionary from PyMuPDF.
        page_width (float): The width of the page.
        page_height (float): The height of the page.
        avg_font_size (float): The average font size on the page.

    Returns:
        dict: A dictionary of features for the model.
    """
    features = {}
    
    # Text-based features
    text = span['text'].strip()
    features['text_length'] = len(text)
    features['starts_with_number'] = 1 if re.match(r'^\d+(\.\d+)*', text) else 0
    features['is_all_caps'] = 1 if text.isupper() and len(text) > 1 else 0
    
    # Font and style features
    font_size = span['size']
    features['font_size'] = font_size
    features['size_ratio'] = font_size / avg_font_size if avg_font_size > 0 else 1
    features['is_bold'] = 1 if "bold" in span['font'].lower() else 0
    
    # Positional features
    bbox = span['bbox'] # (x0, y0, x1, y1)
    features['x_position_ratio'] = bbox[0] / page_width if page_width > 0 else 0
    features['y_position_ratio'] = bbox[1] / page_height if page_height > 0 else 0
    
    # Centered text feature
    span_center = (bbox[0] + bbox[2]) / 2
    page_center = page_width / 2
    # Calculate how far from the center it is, as a percentage of page width
    features['center_distance_ratio'] = abs(span_center - page_center) / page_width if page_width > 0 else 0

    return features