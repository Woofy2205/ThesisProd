import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from llama_index.core.llms import (CompletionResponse, CompletionResponseGen,
                                   CustomLLM, LLMMetadata)
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI

load_dotenv()

client = OpenAI()

def get_response(query, 
                 history: Optional[List[Dict[str, Any]]] = None,
                 model="gpt-3.5-turbo",
                 **kwargs) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    if history:
        messages.extend(history)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 0.7),
            stream=kwargs.get("stream", False)
        )
    except Exception as e:
        raise Exception(f"Error: {str(e)}")
    
    return response.choices[0].message.content

class AssistantBot(CustomLLM):
    """
    Custom LLM class using OpenAI API.
    """
    model: str = "gpt-3.5-turbo"

    def __init__(self, model: str = None, **kwargs) -> None:
        """
        Initializes the Custom LLM with the specified model.
        """
        super().__init__(**kwargs)
        self.model = model or self.model
        print(f"Custom LLM initialized with model: {self.model}")

    @property
    def metadata(self) -> LLMMetadata:
        """
        Returns the metadata for the LLM.
        """
        return LLMMetadata(
            num_output=1,
            model_name=self.model
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str,
                 history: Optional[List[Dict[str, Any]]] = None,
                 **kwargs) -> CompletionResponse:
        """
        Completion endpoint for the LLM.
        """
        try:
            response = get_response(prompt, history, self.model, **kwargs)
        except Exception as e:
            response = str(f"Error: {e}")

        additional_kwargs = {"model": self.model, **kwargs}

        return CompletionResponse(
            text=response,
            additional_kwargs=additional_kwargs
        )

    @llm_completion_callback()
    def stream_complete(self, prompt, formatted=False, **kwargs):
        """
        Stream completion endpoint for the LLM.
        """
        try:
            response = get_response(prompt, model=self.model, stream=True, **kwargs)
        except Exception as e:
            yield CompletionResponse(text="", delta=f"Error: {str(e)}")

        accumulated_text = ""
        for char in response:
            accumulated_text += char
            if char == "\n":
                yield CompletionResponse(text=accumulated_text)
                accumulated_text = ""

        if accumulated_text:
            yield CompletionResponse(text=accumulated_text)

if __name__ == "__main__":
    # llm = AssistantBot(model="gpt-3.5-turbo")
    # response = llm.complete("Hello, how are you?")
    # print(response)
    pass