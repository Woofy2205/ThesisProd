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
    subcol1, subcol2 = st.columns([7,3])
    pdf_file = subcol1.file_uploader("Upload PDF file", type=('pdf'), label_visibility = "collapsed")
    
    ss.pdf = pdf_file
    save_folder = "app/static/pdfdir/"
    
    if subcol2.button("Process PDF 🚩", use_container_width=True):
        with st.spinner("Teacher digesting the PDF"):
            ss.nodes = process_pdf(save_folder)

    if subcol2.button("Creating Test Bank👍", use_container_width=True):
        with st.spinner("Teacher preparing questions..."):
            ss.store = create_store(ss.nodes)
    
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
    
    mes_container = st.container(height=1150)
    num = 0
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with mes_container.chat_message(message["role"]):
            num += 1
            show_question(message["content"], num)
            
    # if st.button("Answer", use_container_width=True):
    #     with st.spinner("Generating Answer..."):
    #         for message in st.session_state.messages:
    #             if message["role"] == "assistant":
    #                 with mes_container.chat_message(message["role"]):
    #                     show_answer_raw(message["content"])
    
    smallcol1 , smallcol2, smallcol3 = st.columns([1,1,1])
    options  = smallcol1.selectbox( "Select a question type to create",
        ["Multiple Choice Question", "Fill in the Blank", "True or False", "Short Answer Question"]                        
    )
    
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
            _response = teacher.create_question(context = context, question_type = types, num_questions = num_ques)
        
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