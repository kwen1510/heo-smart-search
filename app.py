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
def search(query, index, metadata, links, num_results=1):
    # Generate query embedding with Cohere
    response = co.embed(texts=[query], model='large')
    query_embedding = response.embeddings[0]
    
    nearest_neighbors = index.get_nns_by_vector(query_embedding, num_results)
    results = []
    for i in nearest_neighbors:
        result = metadata[i]
        page_number = result['page_number']
        text = result['text']
        # Extract page number from text
        page_number = page_number.split()[-1]
        # Compare with keys in links.json
        for key, value in links.items():
            if page_number.strip().lower() in key.strip().lower():
                results.append((text, value))
                break
    return results

# Load data and index on app start
loaded_data = load_json_file("split_text.json")
links_data = load_json_file("links.json")
ann_index = build_or_load_index()

# Streamlit interface
st.title("HEO Search Tool")

query = st.text_input("Enter your query here")
num_results = st.slider("Number of results", 1, 5, 3, 1)

if st.button("Search"):
    # Preparing metadata array; adjust according to your structure if needed
    metadata_array = []
    for key, values in loaded_data.items():
        for value in values:
            metadata_array.append({"page_number": key, "text": value})
            
    search_results = search(query, ann_index, metadata_array, links_data, num_results)

    for text, link in search_results:
        words = text.replace('\n', ' ').strip().split()
        truncated_text = ' '.join(words[:30]) + "..."
        st.text("Page: " + "\nContext: " + truncated_text + "\n------\n")
        st.markdown(f"[Link](link)")
