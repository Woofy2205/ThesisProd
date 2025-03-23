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

def show_question(response: Union[str, List[str]] = None):
    if type(response) == str:
        st.markdown(response)
    else:
        for i in range(len(response)):
            form = st.form(key=f"quiz_{i}")
            form.markdown(f"{i+1}. {response[i][0]}")
            user_choice = form.radio("Choose an answer:", response[i][1:5])
            submitted = form.form_submit_button("Submit")
            if submitted:
                print(response[i][5].replace('Answer: ', ''))
                if answer_check(user_choice, response[i][5].replace('Answer: ', '')):
                    st.success("Correct! 🎉")
                    st.markdown(f"Context: {response[i][6].replace('Context: ', '')}")
                else:
                    st.error("Incorrect! 😢")
                    st.markdown(f"Context: {response[i][6].replace('Context: ', '')}")

os.chdir("D:/Project/ThesisProd/app/")
# os.chdir("C:/Users/Learning/Project/ThesisProd/app")