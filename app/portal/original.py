import os

import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

os.chdir("D:/Project/ThesisProd/")

from core.ingestion.preprocessing.storage.FaissStore import FaissStore
from core.llm.AssistantLLM import AssistantBot
from core.llm.TeacherLLM import TeacherBot
from core.retriever.Retriever import Retriever

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def process_pdf(doc_path: str) -> list:
    faiss_store = FaissStore(documents_path = doc_path)
    nodes = faiss_store.get_nodes()
    return nodes

def create_store(nodes: list):
    _store = Retriever(nodes = nodes)
    return _store

#  st.set_page_config(page_title="DocumentsQA", page_icon="💬", layout="wide")
st.header("DocumentsQA :books:")

if 'pdf' not in ss:
    ss['pdf'] = None

if 'store' not in ss:
    ss['store'] = None

if 'nodes' not in ss:
    ss['nodes'] = None

with st.sidebar:
    # Access the uploaded ref via a key.
    pdf_file = st.file_uploader("Upload PDF file", type=('pdf'))
    ss.pdf = pdf_file
    save_folder = "app/static/pdfdir/"
    if pdf_file is not None:
        save_path = os.path.join(save_folder, pdf_file.name)
        for file in os.listdir(save_folder):
            os.remove(os.path.join(save_folder, file))
        with open(save_path, mode='wb') as w:
            w.write(pdf_file.getvalue())
    else:
        st.info("Please upload a PDF file.")
    if ss.pdf:
        binary_data = ss.pdf.getvalue()
        pdf_viewer(input=binary_data, width=700)

if st.button("Process PDF"):
    with st.spinner("Teacher digesting the PDF"):
        ss.nodes = process_pdf(save_folder)

if st.button("Preparing questions"):
    with st.spinner("Teacher preparing questions..."):
        ss.store = create_store(ss.nodes)

if ss.store:
    retriever = ss.store.get_retriever(top_k = 5)        
    st.success("PDF processed successfully!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if prompt := st.chat_input("Type some topic you want to practice!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    context = ss.store.get_big_context(retriever=retriever, query=prompt)
    
    with st.spinner("Teacher is creating questions..."):
        teacher = TeacherBot()
        _response = teacher.create_question(context = context)
        
    for i in range(len(_response)):
        for j in range(len(_response[i])):
            # if j == 0:
            #     _response[i][0] = "###" + _response[i][0]
            if j == 5:
                _response[i][5] = ":green[" + _response[i][5] + "]"
            if j == 6:
                _response[i][6] = ":green[" + _response[i][6] + "]"

    # for i in range(len(respond)):
    #     for j in range(len(respond[i])):
    #         print(respond[i][j] + "\n")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        for i in range(len(_response)):
            for j in range(len(_response[i])):
                if j == 0:
                    st.subheader(_response[i][j])
                else:
                    st.markdown(_response[i][j])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": _response})

os.chdir("D:/Project/ThesisProd/app/")