import os
import warnings
from typing import Any, List, Union

from dotenv import load_dotenv
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.llms import LLM

#import Assistant LLM from the project
from core.llm.AssistantLLM import AssistantBot

warnings.filterwarnings("ignore")
load_dotenv("../../.env")

llm = AssistantBot()

class HyDETransformer(HyDEQueryTransform):
    def __init__(self, 
                 llm: LLM = llm,
                 hyde_prompt: str = None,
                 include_original: bool = True):
        """
        Initializes the Hypothetical Document Embeddings 

        :param llm: str, default None. The LLM model to use.
        :param hyde_prompt: str, default None. The prompt to use for the HyDE model.
        :param include_original: bool, default True. Whether to include the original text in the output.
        """
        if isinstance(llm, str):
            self.llm = AssistantBot(model=llm)
        else:
            self.llm = llm
        
        super().__init__(
            llm=self.llm,
            hyde_prompt=hyde_prompt,
            include_original=include_original
        )
        

    def transform(
        self,
        text: Union[str, List[str], List[List[str]]] = None
    ):
        """
        Transforms the input text into hypothetical document embeddings.

        :param text: str. The text to transform.
        :return: str. The transformed text.
        """
        if text is None:
            return "Please provide a text to transform."
        
        if isinstance(text, str):
            _text = [text]

        if isinstance(text[0], list) and len(text) == 1:
            _text = text[0]

        try:
            response = [self.run(t) for t in _text]
        except Exception as e:
            raise Exception(f"Error transforming the input text: {str(e)}")
        response = [r for r in response if r.custom_embedding_strs]
        response = [r.custom_embedding_strs[:-1] for r in response]
        return response
    
if __name__ == "__main__":
    transformer = HyDETransformer()
    response = transformer.transform(text=[["Hello, how are you?", "What is the purpose of life?"]])
    print(response)