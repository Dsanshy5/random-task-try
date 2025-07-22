# src/semantic_analyzer.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticAnalyzer:
    def __init__(self, model_path: str):
        """
        Initializes the analyzer by loading the offline sentence-transformer model.
        """
        self.model = SentenceTransformer(model_path)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates a numerical embedding for a given piece of text.
        """
        return self.model.encode(text, convert_to_tensor=False)

    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two embeddings.
        """
        # Reshape for sklearn's cosine_similarity function
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]