import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

load_dotenv()

client = OpenAI()

def complete(user_text: str,
             model: str = "gpt-4-turbo",
             history: Optional[List[dict]] = None) -> str:
    """
    Perform a completion using OpenAI's API.

    :param user_text: The input prompt for the model.
    :param model: The model to use for completion.
    :param history: Optional history of previous interactions.
    :return: The generated completion text.
    """
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

class AssistantOpenAIBot(CustomLLM):
    """
    OpenAIBot is a custom LLM model that uses OpenAI's API to generate text completions.

    :param model: str, default "gpt-4-turbo". The model name to use.
    """
    model: str = "gpt-3.5-turbo"

    def __init__(
        self, 
        model: str = "gpt-3.5-turbo",
        **kwargs: Any
    ):
        """
        Initialize the OpenAIBot instance.
        """
        super().__init__(**kwargs)
        self.model = model
        print(f"AssistantOpenAIBot initialized with model: {self.model}")

    @property
    def metadata(self) -> LLMMetadata:
        """
        Provide metadata about the LLM.
        """
        return LLMMetadata(
            num_output=1,
            model_name=self.model
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """
        Generate a completion for the given prompt.
        """
        try:
            response_text = complete(
                user_text=prompt,
                model=self.model,
                history=history
            )
        except Exception as e:
            response_text = f"Error: {e}"

        return CompletionResponse(text=response_text)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> CompletionResponseGen:
        """
        Generate a streamed completion for the given prompt.
        """
        try:
            full_response = complete(
                user_text=prompt,
                model=self.model,
                history=history
            )
        except Exception as e:
            yield CompletionResponse(text="", delta=f"Error: {e}")
            return

        accumulated_text = ""
        for char in full_response:
            accumulated_text += char
            yield CompletionResponse(text=accumulated_text, delta=char)

if __name__ == "__main__":
    llm = AssistantOpenAIBot(model="gpt-4-turbo")
    response = llm.complete("Hello, how are you?")
    print(response)
