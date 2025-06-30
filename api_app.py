import os
import weaviate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import Rerank, QueryReference
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Load environment variables
load_dotenv(dotenv_path='./bot.env')

# 2. Weaviate Configuration
WEAVIATE_URL = "http://localhost:8080"
WEAVIATE_COLLECTION_NAME = "CollegeFAQ"

# 3. DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

if not DEEPSEEK_API_KEY:
    logging.error("Error: DEEPSEEK_API_KEY not found in bot.env file. Please add it.")
    raise ValueError("DEEPSEEK_API_KEY not set.")

# 4. Initialize Weaviate Client
# We'll initialize this once globally for the API
try:
    client = weaviate.WeaviateClient(
        connection_params=weaviate.ConnectionParams.from_url(WEAVIATE_URL, grpc_port=50051)
    )
    client.connect()
    if client.is_live():
        logging.info("Successfully connected to Weaviate!")
    else:
        logging.error("Failed to connect to Weaviate: Weaviate instance is not live after connecting.")
        raise ConnectionError("Weaviate not live.")
except Exception as e:
    logging.error(f"Failed to connect to Weaviate: {e}")
    raise ConnectionError(f"Weaviate connection error: {e}")

# Get the collection object for easier interaction
college_faqs_collection = client.collections.get(WEAVIATE_COLLECTION_NAME)

# 5. Initialize Sentence Transformer Model
logging.info("Loading Sentence Transformer model for queries...")
query_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("Model loaded.")

# --- API Setup ---
app = FastAPI(
    title="College Chatbot API",
    description="API for the College Chatbot, powered by Weaviate and DeepSeek AI.",
    version="1.0.0",
)

# Pydantic model for incoming chat requests
class ChatRequest(BaseModel):
    query: str

# --- Chatbot Functions (adapted for API) ---

def get_relevant_faq(query_text: str, top_k: int = 1) -> list:
    """
    Finds the most relevant FAQs from Weaviate based on the query.
    """
    logging.info(f"Searching Weaviate for: '{query_text}'")
    query_vector = query_embedding_model.encode(query_text).tolist()

    try:
        response = college_faqs_collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=["question", "answer"],
        )

        relevant_results = []
        for o in response.objects:
            if o.properties:
                relevant_results.append({
                    "question": o.properties.get("question"),
                    "answer": o.properties.get("answer")
                })
        logging.info(f"Found {len(relevant_results)} relevant FAQs.")
        return relevant_results
    except Exception as e:
        logging.error(f"Error querying Weaviate: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving FAQs from database.")

def generate_llm_response(user_query: str, relevant_faq_text: str) -> str:
    """
    Generates a conversational response using the DeepSeek API.
    """
    system_prompt = (
        "You are a helpful assistant for Punjab Group of Colleges Jaranwala. "
        "Use the provided college FAQ to answer the user's question. "
        "If the FAQ does not contain the answer, politely state that you don't have information on that topic. "
        "Keep your answer concise and directly related to the provided FAQ. "
        "Always prioritize information from the provided FAQ."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User's original question: {user_query}\n\nRelevant College FAQ:\n{relevant_faq_text}"}
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": "deepseek-chat", # You can try other DeepSeek models if available and suitable
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        
        if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content'].strip()
        else:
            logging.warning("No response content from DeepSeek API.")
            return "I couldn't generate a response for that. Please try rephrasing."
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error from DeepSeek API: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 402: # Insufficient Balance
            raise HTTPException(status_code=503, detail="DeepSeek API: Insufficient Balance. Please top up your account.")
        raise HTTPException(status_code=500, detail=f"DeepSeek API Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logging.error(f"Error generating LLM response: {e}")
        raise HTTPException(status_code=500, detail="Error generating AI response.")

# --- API Endpoint ---

@app.post("/chat/")
async def chat_with_bot(request: ChatRequest):
    """
    Endpoint for chatting with the College Chatbot.
    """
    user_query = request.query
    
    # 1. Retrieve relevant FAQs from Weaviate
    relevant_faqs = get_relevant_faq(user_query, top_k=1) # Get top 1 most relevant FAQ

    if relevant_faqs:
        top_faq = relevant_faqs[0]
        context_text = f"Question: {top_faq['question']}\nAnswer: {top_faq['answer']}"
        
        # 2. Generate LLM response using DeepSeek
        chatbot_response = generate_llm_response(user_query, context_text)
        return {"response": chatbot_response, "source_faq": top_faq}
    else:
        logging.info(f"No relevant FAQ found for query: '{user_query}'")
        return {"response": "I couldn't find a relevant FAQ for your question. Please try rephrasing or ask about general college topics.", "source_faq": None}


# Add a simple root endpoint for health check or info
@app.get("/")
async def root():
    return {"message": "College Chatbot API is running. Go to /docs for API documentation."}

# Add startup and shutdown events for client connection management
@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Closing Weaviate client connection...")
    if client.is_connected():
        client.close()
    logging.info("Weaviate client closed.")