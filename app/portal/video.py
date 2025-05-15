import os

import streamlit as st
from streamlit import session_state as ss

os.chdir("D:/Project/ThesisProd/")
# os.chdir("C:/Users/Learning/Project/ThesisProd")

from app.utils import *
from core.ingestion.preprocessing.advanced.hyde.HyDe import HyDETransformer
from core.ingestion.preprocessing.audio.AudioProcessing import AudioProcessor
from core.llm.TeacherLLM import TeacherBot

st.header("RAGQA Practice Mode :books:")

if 'video_url' not in ss:
    ss['video_url'] = None

if 'audio_processor' not in ss:
    ss['audio_processor'] = None

if 'transformer' not in ss:
    ss['transformer'] = HyDETransformer()

if 'context' not in ss:
    ss['context'] = None

save_folder = '/'.join(os.getcwd().split('/')[:3]) + '/app/static/speechdir'
audio_path = save_folder + '/audio_cont'
json_path = save_folder + '/json_cont'

col1, col2 = st.columns([2,2])
with col2:
    if ss.video_url:
        if ss.context:
            st.video(ss.video_url)
            st.write(ss.context)
    # os.chdir("C:/Users/Learning/Project/ThesisProd/app")
    os.chdir("D:/Project/ThesisProd/app")

with col1:
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
        
        # context = ss.store.get_big_context(retriever=retriever, query=prompt)
        num = 0
        with st.spinner("Teacher is creating questions..."):
            teacher = TeacherBot()
            _response = teacher.create_question(context = ss.context, ques_type = types, num_questions = num_ques)
            
        if _response is not None:
            response = refactor(_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.button("Let's start practicing!"):
                pass
        else:
            st.error("Question generated failed! Please try again.")
    # os.chdir("C:/Users/Learning/Project/ThesisProd/app")
    os.chdir("D:/Project/ThesisProd/app")