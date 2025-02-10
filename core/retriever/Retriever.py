import os
import random
import sys

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import \
    VectorIndexRetriever

from core.llm.AssistantLLM import AssistantBot

llm = AssistantBot()

class Retriever(VectorStoreIndex):
    """
    Retriever class for the vector store index.
    """    
    vector_store_index: VectorStoreIndex = None,
    
    def __init__(self, 
                 nodes: list = None, 
                 **kwargs):
        """
        Initializes the Retriever.

        :param vector_store_index: VectorStoreIndex. The vector store index.
        :param nodes: list. The nodes to use.
        :param kwargs: Any. Additional arguments.
        """
        if nodes is None:
            print("No nodes provided.") 
        else:
            self.vector_store_index = VectorStoreIndex(nodes)
            print(f"Successfully initialized for {len(nodes)} nodes.")

    def get_retriever(self, top_k: int = 5):
        """
        Get the retriever for the specified nodes.
        
        :param top_k: int. The number of retrievals to return.
        :return: list. The retrievals.
        """
        retriever = self.vector_store_index.as_retriever(similarity_top_k=top_k)
        return retriever
    
    def get_big_context(self, retriever: VectorIndexRetriever, query: str):
        """
        Get the context for the specified query.
        
        :param query: str. The query to get the context for.
        :return: str. The context.
        """
        _context = retriever.retrieve(query)
        paragraph = ""
        for cot in _context:
            paragraph += cot.text
        paragraph = paragraph.replace("\n", " ")
        prompt = f"Clean the following context: {paragraph} into one united paragraph. If there is any mathematical symbols, return them in the correct format that can be displayed in a web page."
        context = llm.complete(prompt)
        context = context.text
        return context