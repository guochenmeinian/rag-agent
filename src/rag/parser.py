from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import nest_asyncio
from llama_cloud_services import LlamaParse
import os
from dotenv import load_dotenv

try:
    nest_asyncio.apply()
except ValueError:
    pass  # uvloop doesn't need patching
load_dotenv()

def build_llama_parser():

    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

    return LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown", # "markdown" and "text" are available
        num_workers=3, # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="ch_sim", # Optionally you can define a language, default=en
    )


def parse_documents(data_dir, parser):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist.")
    
    if parser is None:
        raise ValueError("Parser is not initialized.")
    
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        data_dir, file_extractor=file_extractor
    ).load_data()
    return documents


def parse_single_file(filepath: str, parser) -> str:
    if not os.path.exists(filepath):
        raise ValueError(f"File {filepath} does not exist.")
    
    if parser is None:
        raise ValueError("Parser is not initialized.")

    documents = parser.load_data(filepath)
    return merge_documents(documents)


def merge_documents(documents):
    return "".join(doc.text for doc in documents)