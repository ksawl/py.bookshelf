import os
from dotenv import load_dotenv

if not os.getenv("PINECONE_API_KEY"):
    load_dotenv()

PINECONE_FREE_TIER = True
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")  # e.g., "us-west1-gcp"
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "bookshelf-index")
PINECONE_SERVERLESS_CLOUD = os.getenv(
    "PINECONE_SERVERLESS_CLOUD", "aws"
)  # e.g. ['gcp', 'aws', 'azure']
PINECONE_SERVERLESS_REGION = os.getenv("PINECONE_SERVERLESS_REGION", "us-east-1")

CHUNK_SIZE = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
OVERLAP_PCT = float(os.getenv("OVERLAP_PCT", "0.2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
ENCODING_NAME = "cl100k_base"  # for tiktoken

# choose model: "all-MiniLM-L6-v2" (fast, 384 dim) or "all-mpnet-base-v2" (768 dim)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# https://platform.openai.com/api-keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required for embeddings in this example
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "text-embedding-3-small"
)  # change if needed

ALLOWED_EXTENSIONS = {"docx", "odt", "pdf", "txt"}
