import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

os.chdir("D:/Project/ThesisProd/")
# os.chdir("C:/Users/Learning/Project/ThesisProd")

from app.utils import *
from core.ingestion.preprocessing.storage.FaissStore import FaissStore
from core.llm.AssistantLLM import AssistantBot
from core.llm.TeacherLLM import TeacherBot
from core.retriever.Retriever import Retriever

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# def process_pdf(doc_path: str) -> list:
#     faiss_store = FaissStore(documents_path = doc_path)
#     nodes = faiss_store.get_nodes()
#     return nodes

# def create_store(nodes: list):
#     _store = Retriever(nodes = nodes)
#     return _store

st.header("DocumentsQA Practice Mode :books:")

if 'pdf' not in ss:
    ss['pdf'] = None

if 'store' not in ss:
    ss['store'] = None

if 'nodes' not in ss:
    ss['nodes'] = None

col1, col2 = st.columns([2,2])

with col2:
    pdf_file = ss.pdf
    save_folder = "app/static/pdfdir/"
    
    if ss.pdf:
        binary_data = ss.pdf.getvalue()
        pdf_viewer(input=binary_data, width=1000, height=1000)
    
    if pdf_file is not None:
        save_path = os.path.join(save_folder, pdf_file.name)
        for file in os.listdir(save_folder):
            os.remove(os.path.join(save_folder, file))
        with open(save_path, mode='wb') as w:
            w.write(pdf_file.getvalue())
    else:
        st.info("Please upload a PDF file.")
            
    if ss.store:
        retriever = ss.store.get_retriever(top_k = 5)        
        st.success("Test Bank is ready! 🎉")
    # os.chdir("C:/Users/Learning/Project/ThesisProd/app")
    os.chdir("D:/Project/ThesisProd/app/")

with col1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    mes_container = st.container(height=600)
    num = 0
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with mes_container.chat_message(message["role"]):
            num += 1
            show_question(message["content"], num)
    
    smallcol2, smallcol3 = st.columns([1,1])
    
    types = smallcol2.selectbox("Select a question difficulty",
        ["Remember", "Understand", "Apply", "Analyze"]
    )
    num_ques = smallcol3.number_input("Number of questions", min_value=1, max_value=10, value=5, step = 1)
    
    if prompt := st.chat_input("Type some topic you want to practice!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with mes_container.chat_message("user"):
            st.markdown(prompt)
        
        context = ss.store.get_big_context(retriever=retriever, query=prompt)
        
        with st.spinner("Teacher choosing the best question..."):
            teacher = TeacherBot()
            _response = teacher.create_question(context = context, ques_type = types, num_questions = num_ques)
        
        if _response is not None:
            response = refactor(_response)

            # Display assistant response in chat message container
            with mes_container.chat_message("assistant"):
                show_question(response, id = num)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Question generated failed! Please try again.")
    # os.chdir("C:/Users/Learning/Project/ThesisProd/app")
    os.chdir("D:/Project/ThesisProd/app/")