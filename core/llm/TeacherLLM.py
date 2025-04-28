import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from llama_index.core.llms import (CompletionResponse, CompletionResponseGen,
                                   CustomLLM, LLMMetadata)
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI

from core.llm.Prompt import (analyzing_questions_prompt,
                             applying_questions_prompt,
                             original_greeting_prompt,
                             original_instruction_questions,
                             remembering_questions_prompt,
                             understanding_questions_prompt)

load_dotenv()

client = OpenAI()

question_prompt = {
    'Remember': remembering_questions_prompt,
    'Understand': understanding_questions_prompt,
    'Apply': applying_questions_prompt,
    'Analyze': analyzing_questions_prompt,
}

def get_response(user_text: str,
                model: str = "gpt-3.5-turbo",
                question_type: str = None,
                history: Optional[List[dict]] = None) -> str:
    """
    Perform a completion using OpenAI's API.
    
    :param user_text: The input prompt for the model.
    :param model: The model to use for completion.
    :param history: Optional history of previous interactions.
    :return: The generated completion text.
    """
    messages = [{"role": "system", "content": original_greeting_prompt},
                {"role": "user", "content": question_prompt.get(question_type)},
                {"role": "user", "content": original_instruction_questions}]
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

def get_response_no_context(user_text: str = None, 
                    history: Optional[List[Dict[str, Any]]] = None,
                    model="gpt-3.5-turbo",
                    **kwargs) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text}
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

class TeacherBot(CustomLLM):
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
    
    def create_question(self, context: str, 
                        ques_type: str = None,
                        history: Optional[List[Dict[str, Any]]] = None,
                        num_questions: int = 5,
                        total_options: int = 4,
                        num_correct_options: int = 1,
                        **kwargs) -> list:
        """
        Generate a question using the LLM.
        """
        if context is None:
            response = get_response_no_context(self.model, prompt, history)
        else:
            prompt = f"""
                context: {context}
                num_questions: {num_questions}
                total_options: {total_options}
                correct_options: {num_correct_options}
            """
            response = get_response(prompt, self.model, ques_type, history)
            # print(response)
            try:
                final_response = json.loads(response)
            except:
                final_response = None
                print("Error in generating questions: ", response)
        return final_response
    
    @llm_completion_callback()
    def complete(self, prompt: str,
                    history: Optional[List[Dict[str, Any]]] = None,
                    **kwargs) -> CompletionResponse:
        """
        Completion endpoint for the LLM.
        """
        try:
            response = get_response_no_context(prompt, history, self.model, **kwargs)
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
            response = get_response_no_context(prompt, model=self.model, stream=True, **kwargs)
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