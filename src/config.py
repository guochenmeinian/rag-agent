"""Centralized configuration. Import this module early to ensure .env is loaded."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rag-agent/

# --- API Keys (fallbacks) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# --- Executor (Agent / 推理引擎) ---
# 可切换 OpenAI / Kimi / DeepSeek 等，通过 EXECUTOR_* 指定
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o")
EXECUTOR_API_KEY = os.getenv("EXECUTOR_API_KEY") or OPENAI_API_KEY
EXECUTOR_BASE_URL = os.getenv("EXECUTOR_BASE_URL", "")  # 空则用 OpenAI 默认

# --- Qwen 系 (Rewriter / Reflector / Memory 改写、反思、记忆) ---  
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-flash")
QWEN_API_KEY = os.getenv("QWEN_API_KEY") or DASHSCOPE_API_KEY
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Backward compat
OPENAI_MODEL = EXECUTOR_MODEL

# --- Memory ---
MAX_RECENT_MESSAGES = int(os.getenv("MAX_RECENT_MESSAGES", "5"))
MEMORY_DIR = os.getenv("MEMORY_DIR", os.path.join(_PROJECT_ROOT, ".sessions"))

# --- Agent ---
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

# --- RAG / Storage ---
MILVUS_URI = os.getenv("MILVUS_URI", os.path.join(_PROJECT_ROOT, "milvus.db"))
GREP_INDEX_PATH = os.getenv("GREP_INDEX_PATH", os.path.join(_PROJECT_ROOT, "grep_index.db"))
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(_PROJECT_ROOT, "data"))
NIO_CAR_MODELS = [m.strip() for m in os.getenv("NIO_CAR_MODELS", "EC6,EC7,ES6,ES8,ET5,ET5T,ET7,ET9").split(",")]


def get_executor_cfg() -> dict:
    """Executor 模型配置，用于 AgentWorkflow。可从 .env 覆盖。"""
    cfg = {"model": EXECUTOR_MODEL}
    if EXECUTOR_API_KEY:
        cfg["api_key"] = EXECUTOR_API_KEY
    if EXECUTOR_BASE_URL:
        cfg["base_url"] = EXECUTOR_BASE_URL
    return cfg


def get_qwen_cfg() -> dict:
    """Qwen 系模型配置（Rewriter / Reflector / Memory）。"""
    cfg = {"model": QWEN_MODEL}
    if QWEN_API_KEY:
        cfg["api_key"] = QWEN_API_KEY
    if QWEN_BASE_URL:
        cfg["base_url"] = QWEN_BASE_URL
    return cfg
