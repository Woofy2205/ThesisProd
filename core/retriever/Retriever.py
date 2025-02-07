import os
import sys

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


class Retriever(VectorStoreIndex):
    def __init__(self, vector_store_path: str):
        """
        Initializes the Retriever.

        :param vector_store_path: str. The path to the vector store.
        """
        super().__init__(vector_store_path=vector_store_path)

    def search(self, query: str, top_k: int = 10):
        """
        Searches the vector store for the most similar vectors to the query.

        :param query: str. The query to search for.
        :param top_k: int, default 10. The number of results to return.
        :return: list. A list of the most similar vectors to the query.
        """
        return self.get_nearest_neighbours(query, top_k)