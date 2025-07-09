# RAG-Agent: Retrieval-Augmented Generation with Agent Workflow

This project demonstrates the integration of Retrieval-Augmented Generation (RAG) and Agent-based reasoning for intelligent document and web-augmented question answering. It is designed for learning and experimenting with multi-step reasoning, tool selection, and dynamic answer generation.

<img width="926" alt="Screenshot 2025-07-09 at 9 34 08 AM" src="https://github.com/user-attachments/assets/4ae291bd-0036-4cc7-9084-22ef285ed798" />

## Key Features
- **Local Document RAG**: Parse, chunk, embed, and semantically retrieve information from local PDFs and other documents.
- **Web-Augmented QA**: Optionally enhance answers with real-time web search (via Serper API and lightweight crawling).
- **Agent Workflow**: The agent plans, decomposes user queries, selects tools (local/web), and executes multi-step reasoning.
- **Session Memory**: Each reasoning session accumulates retrieved information and intermediate results as memory/context, which is used for final answer synthesis.
- **Reflection & Reasoning**: The agent uses reflection prompts to check if enough information has been gathered, and finally leverages a powerful reasoning model (e.g., deepseek-r1) to generate a persuasive and styled answer.

## Directory Structure
```
rag-agent/
├── rag-agent.ipynb         # Main notebook: Agent/RAG workflow and demos
├── websearch/              # (Optional) Web search and augmentation modules
│   └── src/
│       └── ...             # Web crawling, retrieval, config, etc.
├── data/                   # Local PDF/document samples
├── BAAI/bge-m3/            # Local embedding model (e.g., bge-m3)
├── milvus.db               # Local vector database
└── ...
```

## Installation
Python 3.10+ is recommended. Install core dependencies:

```bash
pip install langchain openai chromadb sentence-transformers pymilvus transformers tqdm requests pyyaml bs4
```

For local embedding models (e.g., bge-m3), see [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding).

## Configuration
1. Copy and edit `websearch/src/config/config.yaml` with your OpenAI API Key, Serper API Key, and model settings.
2. For Milvus or local models, follow the notebook examples to set up paths and database connections.

## Quick Start
- Run `rag-agent.ipynb` to experience local document parsing, RAG retrieval, web-augmented QA, and multi-step agent reasoning.
- Refer to scripts in `websearch/src/` for modular usage if needed.

## Core Workflow
1. **Planning**: The agent analyzes the user query, decomposes it, and decides which tools to use (local/document search or web search).
2. **Session Memory**: Each retrieval or search result is stored in a session memory list, which is passed to the final answer stage.
3. **Reflection**: The agent uses a reflection prompt to check if the gathered information is sufficient, and may trigger further retrieval if needed.
4. **Reasoning & Answer Generation**: All collected memory is fed to a reasoning model (e.g., deepseek-r1) to generate a comprehensive, persuasive answer, often in a sales-oriented style.

> Note: The current memory mechanism is session-based (short-term). For persistent or long-term memory, consider extending with a database or file storage.

## Main Dependencies
- LangChain
- OpenAI
- ChromaDB
- Sentence-Transformers
- FlagEmbedding/BGE-M3
- Milvus
- Serper API
- BeautifulSoup4, Requests, PyYAML, tqdm

## Reference
- Sample product documents (e.g., for ES9 EV) are included for RAG demos.
- For detailed logic and workflow, see `rag-agent.ipynb` and `Agent+RAG实现检索.pdf`.

---
For questions or contributions, feel free to open an issue or contact the author.
