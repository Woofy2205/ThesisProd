import os
import sys
from typing import List
from warnings import filterwarnings

from llama_index.core.schema import BaseNode, TransformComponent

filterwarnings('ignore')

class TextCleaner(TransformComponent):
    """
    Transformation to be used within the ingestion pipeline.
    Cleans clutters from texts.
    """
    def __call__(self, nodes, **kwargs) -> List[BaseNode]:
        
        for node in nodes:
            node.text = node.text.replace('\t', ' ') # Replace tabs with spaces
            node.text = node.text.replace(' \n', ' ') # Replace paragraph seperator with spacaes
            
        return nodes