import os
from pathlib import Path

os.chdir(Path(os.getcwd()).parent)

import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

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


st.set_page_config(page_title="DocQA", page_icon="💬", layout="wide")

home_page = st.Page("app/portal/home.py", title="Home", icon="🏠")
settings_page = st.Page("app/portal/settings.py", title="Settings", icon="⚙️")
practice_page = st.Page("app/portal/practice.py", title="Practice", icon="👨‍🏫")
video_page = st.Page("app/portal/video.py", title="Video", icon="📹")


navigation = st.navigation(
    {"Home": [home_page, settings_page], "Subpage": [practice_page, video_page]},
    expanded=True,
)

                
if __name__ == "__main__":
    os.chdir(Path(os.getcwd()).joinpath('app'))
    navigation.run()