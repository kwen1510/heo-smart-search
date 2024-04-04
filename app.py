
import streamlit as st
import json
from annoy import AnnoyIndex
import cohere
import os

# Initialize Cohere client
co = cohere.Client(os.getenv('COHERE_API_KEY'))

# Load JSON Data
def load_json_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Function to build or load an Annoy index - this depends on your setup
def build_or_load_index():
    N = 4096  # Assuming a fixed embedding size, adjust as necessary
    index_file = 'text_embeddings.ann'
    u = AnnoyIndex(N, 'angular')
    u.load(index_file)  # Load your pre-built index
    return u

# Search function
def search(query, index, metadata, num_results=1):
    # Generate query embedding with Cohere
    response = co.embed(texts=[query], model='large')
    query_embedding = response.embeddings[0]
    
    nearest_neighbors = index.get_nns_by_vector(query_embedding, num_results)
    return [metadata[i] for i in nearest_neighbors]

# Load data and index on app start
loaded_data = load_json_file("split_text.json")
ann_index = build_or_load_index()

# Streamlit interface
st.title("Your App Title Here")

query = st.text_input("Enter your query here", "why study science")
num_results = st.slider("Number of results", 1, 10, 1)

if st.button("Search"):
    # Preparing metadata array; adjust according to your structure if needed
    metadata_array = []
    for key, values in loaded_data.items():
        for value in values:
            metadata_array.append({"page_number": key, "text": value})
            
    search_results = search(query, ann_index, metadata_array, num_results)

    for result in search_results:
        st.text(f"Page: {result['page_number']}
Text: {result['text'].replace('\n', ' ')}\n------\n")

# Include 'cohere' in your requirements.txt for deployment
