import os
import sys
# Add current directory to path so we can import app modules
sys.path.append(os.getcwd())

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from app.core.config import settings
from app.core.logging import logger

# Ensure environment variables are loaded
load_dotenv()

def inspect_chroma():
    print("--- ChromaDB Internal Inspection ---\n")
    
    # 1. Initialize the same embedding function
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=settings.GOOGLE_API_KEY,
        model=settings.GEMINI_EMBEDDING_MODEL
    )

    # 2. Connect to the existing database
    db = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="rag_documents"
    )

    # 3. Use .get() to retrieve stored items
    # We include 'embeddings' to see the actual math/numbers
    data = db.get(include=['documents', 'metadatas', 'embeddings'])
    
    ids = data.get('ids', [])
    documents = data.get('documents', [])
    metadatas = data.get('metadatas', [])
    embeddings = data.get('embeddings', [])

    print(f"📊 Total Chunks in DB: {len(ids)}")
    print(f"📂 Persistent Storage: {settings.CHROMA_PERSIST_DIRECTORY}\n")

    if not ids:
        print("Empty database. Please upload a document first.")
        return

    # 4. Show a sample of 1 chunk with its VECTOR
    sample_index = 0
    print(f"--- Deep Inspection: Vector Representation ---\n")

    print(f"🔹 CHUNK ID: {ids[sample_index]}")
    print(f"📄 CONTENT: \"{documents[sample_index][:100]}...\"")
    
    vector = embeddings[sample_index]
    print(f"🧬 VECTOR SIZE: {len(vector)} dimensions")
    print(f"🔢 VECTOR SAMPLE (First 10 values):")
    formatted_vector = [round(val, 6) for val in vector[:10]]
    print(f"{formatted_vector} ...")
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    inspect_chroma()
