import numpy as np
import json
import sentence_transformers

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

# Load JSON data from file
with open('/Users/krishna/Desktop/Study/NLP Assignments/Course_Project/PDF/output/chunks.json', 'r') as file:
    json_data = json.load(file)

# Extract content and ids
contents = [item["content"] for item in json_data]
ids = [item["id"] for item in json_data]

# Generate embeddings for the contents
embeddings = model.encode(contents)

# Save the embeddings and ids to .npy files
np.save('embeddings.npy', embeddings)
np.save('ids.npy', ids)

user_query = "What is self attention?"
query_embed = model.encode([user_query])

from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings and ids from .npy files
embeddings = np.load('embeddings.npy')
ids = np.load('ids.npy', allow_pickle=True)

# Calculate cosine similarity between query and all embeddings
similarities = cosine_similarity(query_embed, embeddings)

# Find the top 3 indices
top_indices = np.argsort(similarities[0])[::-1][:4]

# Retrieve and print the corresponding strings
for index in top_indices:
    content_id = ids[index]
    # Assuming json_data is the list of dictionaries loaded from 'chunks.json'
    corresponding_string = next(item["content"] for item in chunks if item["id"] == content_id)
    print(corresponding_string)