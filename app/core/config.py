import os
from dotenv import load_dotenv

if not os.getenv("PINECONE_API_KEY"):
    load_dotenv()

ALLOWED_EXTENSIONS = {"docx", "odt", "pdf", "txt"}

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_SERVERLESS_CLOUD = os.getenv("PINECONE_SERVERLESS_CLOUD")
PINECONE_SERVERLESS_REGION = os.getenv("PINECONE_SERVERLESS_REGION")

CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS"))
OVERLAP_PCT = float(os.getenv("OVERLAP_PCT"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
ENCODING_NAME = os.getenv("ENCODING_NAME")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS"))
TOP_K = int(os.getenv("TOP_K"))
