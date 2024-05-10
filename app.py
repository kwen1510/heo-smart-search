import streamlit as st
import json
import os
from annoy import AnnoyIndex
import cohere
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# password = st.secrets(["MONGO_DB"])

uri = f"mongodb+srv://kwen1510:applepear123@cluster0.bwtbeur.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

print(client)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Initialize Cohere client
co = cohere.Client(os.getenv('COHERE_API_KEY'))
client = MongoClient(st.secrets["MONGO_DB"])
db = client.HEO
collection = db.HEO_queries

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
            
    search_results = search(query, ann_index, metadata_array, num_results)

    for result in search_results:
        words = result['text'].replace('\n', ' ').strip().split()
        truncated_text = ' '.join(words[:30]) + "..."
        
        page_key = result['page_number'].split(" page")[0].strip()
        
        link = links_data.get(page_key, "No link available")
        
        st.text("Page: " + result['page_number'] + "\nContext: " + truncated_text + "\n------\n")
        st.markdown(f"[Click here to access the document]({link})")

# Include 'cohere' in your requirements.txt for deployment
