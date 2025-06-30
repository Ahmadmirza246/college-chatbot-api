import os
import weaviate
from dotenv import load_dotenv
from faqs_data import college_faqs
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Property, DataType, Configure # Import necessary classes for schema v4

# 1. Load environment variables
load_dotenv(dotenv_path='./bot.env')

# 2. Weaviate Configuration
WEAVIATE_URL = "http://localhost:8080" # This is your local Weaviate instance running in Docker

# 3. Initialize Weaviate Client (v4 syntax)
try:
    client = weaviate.WeaviateClient(
        connection_params=weaviate.ConnectionParams.from_url(WEAVIATE_URL, grpc_port=50051)
    )
    client.connect() # ADDED THIS LINE TO EXPLICITLY CONNECT
    
    # After connecting, we can check if it's live
    if client.is_live():
        print("Successfully connected to Weaviate!")
    else:
        print("Failed to connect to Weaviate: Weaviate instance is not live after connecting.")
        exit()
except Exception as e:
    print(f"Failed to connect to Weaviate: {e}")
    exit()

# 4. Initialize Sentence Transformer Model
print("Loading Sentence Transformer model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# 5. Define Weaviate Schema (Collection) for FAQs
collection_name = "CollegeFAQ" # Name for your data collection in Weaviate (v4 uses Collections)

# Define the Collection (like a table in traditional DBs, v4 terminology)
try:
    # Check if the collection already exists and delete if it does (for clean runs during development)
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting it for a fresh start...")
        client.collections.delete(collection_name)
        print(f"Collection '{collection_name}' deleted.")

    print(f"Creating collection '{collection_name}' in Weaviate...")
    college_faq_collection = client.collections.create(
        name=collection_name,
        properties=[
            Property(name="question", data_type=DataType.TEXT, description="The question part of the FAQ."),
            Property(name="answer", data_type=DataType.TEXT, description="The answer part of the FAQ."),
        ],
        # Configure the vectorizer to use pre-computed vectors
        vectorizer_config=Configure.Vectorizer.none(), # Important: Use none() because we provide vectors
        # Configure the Generative module if you want to use it later (optional for now)
        # generative_config=Configure.Generative.palm() # Example if using Google PaLM for generation
    )
    print(f"Collection '{collection_name}' created.")

except weaviate.exceptions.WeaviateBaseError as e:
    print(f"Error creating/deleting collection: {e}")
    exit()


# 6. Generate Embeddings and Import Data
print(f"Importing {len(college_faqs)} FAQs into Weaviate...")

# Use the data import methods directly on the collection object
# The batching is handled automatically and efficiently by the v4 client
with college_faq_collection.batch.dynamic() as batch:
    for i, faq in enumerate(college_faqs):
        Youtube_text = f"{faq['question']} {faq['answer']}"
        
        # Generate embedding for the combined question and answer
        embedding = embedding_model.encode(Youtube_text).tolist()

        data_object = {
            "question": faq["question"],
            "answer": faq["answer"],
        }
        
        # Add the data object to the batch with the pre-computed embedding
        batch.add_object(
            properties=data_object,
            vector=embedding
        )
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} of {len(college_faqs)} FAQs.")

# Batch is automatically flushed when exiting the 'with' block
print("All FAQs imported successfully!")