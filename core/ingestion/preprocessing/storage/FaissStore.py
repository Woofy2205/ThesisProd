import os
import sys
from typing import List

import faiss
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore

from core.ingestion.preprocessing.formatter.Formatter import TextCleaner

# Load environment variables from a .env file
load_dotenv()

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

class FaissStore:
    """
    FaissStore class for the Faiss vector store.
    """
    documents: list = None
    vector_store: FaissVectorStore = None
    def __init__(self, 
                 documents_path: str = None, 
                 embed_dim: int = 512,
                 **kwargs):
        """
        Initializes the FaissStore.
        
        :param documents_path: str. The path to the documents.
        """
        super()
        if documents_path is None:
            print("No documents provided.")
        else:
            node_parser = SimpleDirectoryReader(input_dir=documents_path, required_exts=[".pdf"])
            _documents = node_parser.load_data()
            self.documents = _documents
            faiss_index = faiss.IndexFlatL2(embed_dim)
            self.vector_store = FaissVectorStore(faiss_index=faiss_index)
            print(f"Loaded {len(self.documents)} documents.")
            
    
    # def get_vectorstore(self, embed_dim: int = 512, **kwargs):
    #     """
    #     Get the retriever for the specified nodes.
        
    #     :param embed_dim: int. The embedding dimension.
    #     """
    #     faiss_index = faiss.IndexFlatL2(embed_dim)
    #     vector_store = FaissVectorStore(faiss_index=faiss_index)
    #     return vector_store
    
    def get_nodes(self, _chunk_size: int = 200, _chunk_overlap: int = 50, **kwargs):
        """
        Get the context for the specified query.
        """
        if self.vector_store is None:
            print("No vector store provided.")
        
        if self.documents is None:
            print("No documents provided.")
        
        text_splitter = SentenceSplitter(chunk_size = _chunk_size, chunk_overlap = _chunk_overlap)
        pipeline = IngestionPipeline(
            transformations=[
                TextCleaner(),
                text_splitter,
            ],
            vector_store=self.vector_store,
        )
        nodes = pipeline.run(documents = self.documents)
        return nodes