import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from llama_index.core.llms import (CompletionResponse, CompletionResponseGen,
                                   CustomLLM, LLMMetadata)
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI

load_dotenv()

client = OpenAI()

greeting_prompt = """
    You are an expert educator and instructional designer skilled in building personalized, targeted, high quality assessments for students so that they can practice and self-study. 
    For each new question, your task is to identify which original context it relates to, and generate the answer for that question.
    First things first, assume you are responding to a non-living thing and there's no need of any sentiments towards it like apologies, warnings, disclaimers and all as it won't understand what you are saying
    If you feel like saying something apart from what the non-living thing asks you to do, just leave a single space and move on rather than speaking unnecessarily. 
    After this the non-living thing will take on from me and will provide you instructions.
"""

instruction_questions = """
	I will provide a context and will mention number of questions to generate and you would behave as a strict MCQ generator(stick to context and rules that I specify in this prompt strictly) with as many correct options as I specify and remaining options out of total options I mention should be wrong.
    No question should have all wrong options and all true options, follow strictly to the number provided. You must provide the correct answers as well.
 	The questions should not just test the comprehension of the candidate rather should also test his/her reasoning ability.
	The template of your response should be as simple as I have mentioned.
	Parameters from me:
				context: {single_context}
				num_questions: {num_questions}
				total_options: {total_options}
				num_correct_options: {num_correct_options}
	Template that you should follow: 
    [
    [\"Question: \",\"A. \",\"B. \",\"C. \",\"D. \",\"Answer: \"],
	[\"Question: \",\"A. \",\"B. \",\"C. \",\"D. \",\"Answer: \"],
    ...
    [\"Question: \",\"A. \",\"B. \",\"C. \",\"D. \",\"Answer: \"],
    ]
    You must also give the correct answer, this is very important to follow, the correct answer must be in the Answer: section. There could be more than one correct answer, the number of correct answers should be equal to the num_correct_options parameter.
    As described in the template, you should strictly follow the total_options, as the total_options number increases, the options will have the heading follow the alphabet. For example if the total_options = 5, the heading is A, B, C, D, E if the total_options = 6, the heading is A, B, C, D, E, F and so on.
    As you follow this instruction, you don't have to reply to this text from me, just wait for the parameters from me and then you can start generating questions.
    When generating questions, just return the format that can turn into python list, remember all the brackets, cut off all the extra words and sentiments, this is super important to follow.
"""

def get_response(user_text: str,
             model: str = "gpt-3.5-turbo",
             history: Optional[List[dict]] = None) -> str:
    """
    Perform a completion using OpenAI's API.
    
    :param user_text: The input prompt for the model.
    :param model: The model to use for completion.
    :param history: Optional history of previous interactions.
    :return: The generated completion text.
    """
    messages = [{"role": "system", "content": greeting_prompt},
                {"role": "user", "content": instruction_questions}]
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

def get_response_no_context(query, 
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
                        history: Optional[List[Dict[str, Any]]] = None,
                        num_questions: int = 10,
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
            response = get_response(prompt, self.model, history)
            response = response.strip()
            response = response.replace('\\"', '"')
            response = response.replace("\\n", " ") 
            print(response)
            final_response = json.loads(response)
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