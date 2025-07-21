# src/config.py

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and Output folders
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Folder with sample PDFs and their ground-truth JSONs
SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')

# Location to save the trained model
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'heading_classifier.joblib')

# Location for the generated training data
TRAINING_DATA_PATH = os.path.join(BASE_DIR, 'training_data.csv')

# Define the labels you're trying to predict
# 'Body' will be our label for any text that is not a heading
LABELS = ["H1", "H2", "H3", "Body"]