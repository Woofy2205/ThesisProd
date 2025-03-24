import os

import streamlit as st
from streamlit import session_state as ss

# os.chdir("D:/Project/ThesisProd/")
os.chdir("C:/Users/Learning/Project/ThesisProd")

from app.utils import *
from core.ingestion.preprocessing.advanced.hyde.HyDe import HyDETransformer
from core.ingestion.preprocessing.audio.AudioProcessing import AudioProcessor
from core.llm.TeacherLLM import TeacherBot

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
with col1:
    ss.video_url = st.text_input("Enter video URL here")
    if ss.video_url:
        st.video(ss.video_url)
        processor = AudioProcessor(audio_path = ss.video_url)
        ss.audio_processor = processor
        if ss.context is None:
            with st.spinner("Processing Video..."):
                vid_title = processor.process_download(video_url=ss.video_url, audio_path = audio_path)
                processor.transcript(audio_path = audio_path, json_path = json_path, video_title=vid_title)
                transcription_file = json_path + '/' + vid_title + '.json'
                respond = processor.context_transcript(transcription_file)
                context = ss.transformer.transform(respond)
                ss.context = context[0][0]
            st.write(ss.context)
        else:
            st.write(ss.context)
    os.chdir("C:/Users/Learning/Project/ThesisProd/app")
    # os.chdir("D:/Project/ThesisProd/app")

with col2:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    mes_container = st.container(height=1000)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with mes_container.chat_message(message["role"]):
            show_question(message["content"])
            
    smallcol1 , smallcol2, smallcol3 = st.columns([1,1,1])
    options  = smallcol1.selectbox( "Select a question type to create",
        ["Multiple Choice Question", "Fill in the Blank", "True or False", "Short Answer Question"]                        
    )
    
    types = smallcol2.selectbox("Select a question difficulty",
        ["Remembering", "Understanding", "Applying", "Analyzing", "Evaluating", "Creating"]
    )
    if smallcol3.button("Answer", use_container_width=True):
        with st.spinner("Generating Answer..."):
            for message in st.session_state.messages:
                if message["role"] == "assistant":
                    with mes_container.chat_message(message["role"]):
                        show_answer(message["content"])
    
    if prompt := st.chat_input("Type some topic you want to practice!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with mes_container.chat_message("user"):
            st.markdown(prompt)
        
        # context = ss.store.get_big_context(retriever=retriever, query=prompt)
        
        with st.spinner("Teacher is creating questions..."):
            teacher = TeacherBot()
            _response = teacher.create_question(context = ss.context)
            
        if _response is not None:
            response = refactor(_response)

            # Display assistant response in chat message container
            with mes_container.chat_message("assistant"):
                show_question(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Question generated failed! Please try again.")
    os.chdir("C:/Users/Learning/Project/ThesisProd/app")
    # os.chdir("D:/Project/ThesisProd/app")