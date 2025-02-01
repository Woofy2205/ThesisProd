from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

EMBED_DIMENSION = 512

class OpenAIEmbedding(OpenAIEmbedding):
    def __init__(self,
                 model: str = "text-embedding-3-small", 
                 dimension: int = EMBED_DIMENSION):
        """
        Initializes the OpenAIEmbedding.
        
        :param model: str, default "text-embedding-3-small". The model to use for embedding.
        :param dimension: int, default 512. The dimension of the embedding.
        """
        
        super().__init__(model=model, dimension=dimension)
    
    def get_embeddings():
        """
        Embeds the input text.
        
        :param text: str. The text to be embedded.
        :return: np.array. The embedding of the text.
        """
        return self.get_embedding(text)