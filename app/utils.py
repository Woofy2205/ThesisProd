import math
import os
from typing import List, Union

os.chdir("D:/Project/ThesisProd/")
# os.chdir("C:/Users/Learning/Project/ThesisProd")

import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

from core.ingestion.preprocessing.storage.FaissStore import FaissStore
from core.llm.AssistantLLM import AssistantBot
from core.llm.TeacherLLM import TeacherBot
from core.retriever.Retriever import Retriever


def process_pdf(doc_path: str) -> list:
    faiss_store = FaissStore(documents_path = doc_path)
    nodes = faiss_store.get_nodes()
    return nodes

def create_store(nodes: list):
    _store = Retriever(nodes = nodes)
    return _store

def refactor(response: list[str] = None) -> list:
    # for i in range(len(response)):
    #         for j in range(len(response[i])):
    #             # if j == 0:
    #             #     _response[i][0] = "###" + _response[i][0]
    #             if j == 5:
    #                 response[i][5] = ":green[" + response[i][5] + "]"
    #             if j == 6:
    #                 response[i][6] = ":green[" + response[i][6] + "]"
    return response

def show_question_raw(response: Union[str, List[str]] = None):
    if type(response) == str:
        st.markdown(response)
    else:
        for i in range(len(response)):
            for j in range(len(response[i])):
                if j == 0:
                    st.subheader(response[i][j])
                elif j == 5 or j == 6:
                    continue
                else:
                    st.markdown(response[i][j])

def show_answer_raw(response: Union[str, List[str]] = None):
    if type(response) == str:
        st.markdown(response)
    else:
        for i in range(len(response)):
            for j in range(len(response[i])):
                if j == 0:
                    st.subheader(response[i][j])
                else:
                    st.markdown(response[i][j])
                    
def answer_check(user: str, answer: str) -> bool:
    if user[0] == answer[0]:
        return True
    else:
        return False

def show_question(response: Union[str, List[str]] = None, id: int = 0):
    if type(response) == str:
        st.markdown(response)
    else:
        with st.form(f'quiz {id}'):
            st.header('Let\'s practice!')
            number_of_questions = len(response)
            if number_of_questions > 1:
                pass_score = st.select_slider('Pass score',
                                                range(1, number_of_questions + 1),
                                                value=math.ceil(0.8 * number_of_questions))
            else:
                pass_score = 1
            answers = []
            containers = []
            for i in range(len(response)):
                container = st.container()
                answer = container.radio(f'{i + 1}. {response[i][0]}',
                                            response[i][1:5],
                                            )
                answers.append(response[i][1:5].index(answer)+1)
                containers.append(container)
            submit_quiz = st.form_submit_button('Submit my answers')
        if submit_quiz:
            score = 0
            for i in range(len(response)):
                if answer_check(response[i][answers[i]], response[i][5].replace("Answer: ", "")):
                    containers[i].success("Correct🎉🎊")
                    containers[i].write(f"Context in documents: {response[i][6].replace('Context: ', '')}")
                    score += 1
                else:
                    containers[i].error("Incorrect😢")
                    containers[i].write(f"The answer is: {response[i][5].replace('Answer: ', '')}.")
                    containers[i].write(f"Context in documents: {response[i][6].replace('Context: ', '')}")
            
            message = f'Your final score is: {score}/{number_of_questions}'
            if score >= pass_score:
                st.success(message)
                st.success(':partying_face: Well done! Keep it up!')
            else:
                st.error(message)
                st.error('Not this time :grimacing: Please Try again!')

os.chdir("D:/Project/ThesisProd/app/")
# os.chdir("C:/Users/Learning/Project/ThesisProd/app")