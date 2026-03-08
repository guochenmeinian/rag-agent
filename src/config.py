"""Centralized configuration. Import this module early to ensure .env is loaded."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Models ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-instruct")

# --- API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# --- Memory ---
SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "6"))
MAX_RECENT_MESSAGES = int(os.getenv("MAX_RECENT_MESSAGES", "5"))

# --- Agent ---
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

# --- RAG / Storage ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rag-agent/
MILVUS_URI = os.getenv("MILVUS_URI", os.path.join(_PROJECT_ROOT, "milvus.db"))
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(_PROJECT_ROOT, "data"))
NIO_CAR_MODELS = [m.strip() for m in os.getenv("NIO_CAR_MODELS", "EC6,EC7,ES6,ES8,ET5,ET5T,ET7,ET9").split(",")]
