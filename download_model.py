# download_model.py
from sentence_transformers import SentenceTransformer

# Define the model you want to use
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Define the local path to save it to
local_model_path = 'models/all-MiniLM-L6-v2'

# Save the model to the local path
model.save(local_model_path)

print(f"Model '{model_name}' downloaded and saved to '{local_model_path}'")