from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

EMBED_DIMENSION = 512

class OAIEmbedding(OAIEmbedding):
    def __init__(self,
                 model: str = "text-embedding-3-small", 
                 dimension: int = EMBED_DIMENSION):
        """
        Initializes the OpenAIEmbedding.
        
        :param model: str, default "text-embedding-3-small". The model to use for embedding.
        :param dimension: int, default 512. The dimension of the embedding.
        """
        
        super().__init__(**kwargs)
        self.model = OpenAIEmbedding(model = model, dimension = dimension)
        print("Initialized OpenAIEmbedding")
    
    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name.
        """
        return "OAIEmbedding"
    
    def get_embeddings():
        """
        Embeds the input text.
        
        :param text: str. The text to be embedded.
        :return: np.array. The embedding of the text.
        """
        return self.get_embedding(text)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Get the query embedding.
        """
        return self.get_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Get the text embedding.
        """
        return self.get_embedding(text)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self.get_embeddings(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self.get_embeddings(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embeddings(text) for text in texts]