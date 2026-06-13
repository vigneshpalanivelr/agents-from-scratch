from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

def get_gemini_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")