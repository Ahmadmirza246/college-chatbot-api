import os
import weaviate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import Rerank, QueryReference # Import for v4 query syntax
import requests # For DeepSeek API
import json # For handling JSON responses

# 1. Load environment variables
load_dotenv(dotenv_path='./bot.env')

# 2. Weaviate Configuration
WEAVIATE_URL = "http://localhost:8080" # Your local Weaviate instance
WEAVIATE_COLLECTION_NAME = "CollegeFAQ" # Name of the collection we created

# 3. DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions" # DeepSeek Chat API endpoint

if not DEEPSEEK_API_KEY:
    print("Error: DEEPSEEK_API_KEY not found in bot.env file. Please add it.")
    exit()

# 4. Initialize Weaviate Client
try:
    client = weaviate.WeaviateClient(
        connection_params=weaviate.ConnectionParams.from_url(WEAVIATE_URL, grpc_port=50051)
    )
    client.connect()
    if client.is_live():
        print("Successfully connected to Weaviate!")
    else:
        print("Failed to connect to Weaviate: Weaviate instance is not live after connecting.")
        exit()
except Exception as e:
    print(f"Failed to connect to Weaviate: {e}")
    exit()

# Get the collection object for easier interaction
college_faqs_collection = client.collections.get(WEAVIATE_COLLECTION_NAME)

# 5. Initialize Sentence Transformer Model (must be the same as used for embedding data)
print("Loading Sentence Transformer model for queries...")
query_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# --- Chatbot Functions ---

def get_relevant_faq(query: str, top_k: int = 3) -> list:
    """
    Finds the most relevant FAQs from Weaviate based on the query.
    """
    print(f"\nSearching Weaviate for: '{query}'")
    query_vector = query_embedding_model.encode(query).tolist()

    try:
        # Perform a vector search
        response = college_faqs_collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=["question", "answer"], # Specify which properties to return
            # We can also add a reranker here if we wanted to refine results, e.g.,
            # query_reranker=Rerank(prop="question", query=query)
        )

        relevant_results = []
        for o in response.objects:
            if o.properties:
                relevant_results.append({
                    "question": o.properties.get("question"),
                    "answer": o.properties.get("answer")
                })
        print(f"Found {len(relevant_results)} relevant FAQs.")
        return relevant_results
    except Exception as e:
        print(f"Error querying Weaviate: {e}")
        return []

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
        "max_tokens": 300, # Limit response length
        "temperature": 0.7, # Creativity level
        "stream": False # We want a single response
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        
        if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content'].strip()
        else:
            print("Warning: No response content from DeepSeek API.")
            return "I couldn't generate a response for that. Please try rephrasing."
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error from DeepSeek API: {e.response.status_code} - {e.response.text}")
        return "I'm having trouble connecting to the AI. Please try again later."
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return "I'm having trouble generating a response. Please try again later."

# --- Main Chatbot Loop ---

def run_chatbot():
    print("\n--- College Chatbot (Powered by Weaviate & DeepSeek) ---")
    print("Type your questions about Punjab Group of Colleges Jaranwala. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        # 1. Retrieve relevant FAQs from Weaviate
        relevant_faqs = get_relevant_faq(user_input, top_k=1) # Get top 1 most relevant FAQ

        if relevant_faqs:
            # For simplicity, we'll use the top 1 result
            top_faq = relevant_faqs[0]
            context_text = f"Question: {top_faq['question']}\nAnswer: {top_faq['answer']}"
            
            # 2. Generate LLM response using DeepSeek
            chatbot_response = generate_llm_response(user_input, context_text)
            print(f"Chatbot: {chatbot_response}")
        else:
            print("Chatbot: I couldn't find a relevant FAQ for your question. Please try rephrasing or ask about general college topics.")

# Close the Weaviate client when the script finishes or exits
# This ensures proper resource cleanup
def cleanup_client():
    if client.is_connected():
        client.close()
        print("\nWeaviate client closed.")

if __name__ == "__main__":
    try:
        run_chatbot()
    except KeyboardInterrupt:
        print("\nExiting chatbot.")
    finally:
        cleanup_client()