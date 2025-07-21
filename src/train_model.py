# src/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

from . import config

def train():
    """
    Trains the heading classification model and saves it.
    """
    # Load data
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    
    # Define features (X) and target (y)
    X = df.drop('label', axis=1)
    y = df['label']

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train the model
    # RandomForest is a good choice for this kind of tabular data
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Evaluation Report:")
    print(classification_report(y_test, y_pred, labels=config.LABELS[:-1])) # Report on H1-H3

    # Save the trained model
    if not os.path.exists(config.MODELS_DIR):
        os.makedirs(config.MODELS_DIR)
    joblib.dump(model, config.MODEL_PATH)
    print(f"Definitive generic model trained and saved to {config.MODEL_PATH}")


if __name__ == '__main__':
    train()