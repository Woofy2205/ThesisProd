import os

import streamlit as st
from streamlit import session_state as ss

os.chdir("D:/Project/ThesisProd/")

from core.ingestion.preprocessing.audio.AudioProcessing import AudioProcessor

if 'video_url' not in ss:
    ss['video_url'] = None
    
if 
    
col1, col2 = st.columns([2,2])
with col1:
    ss.video_url = st.text_input("Enter video URL here")
    save_folder = "app/static/speechdir/"
    
    if ss.video_url:
        st.video(ss.video_url)
        st.write("Transcript Here (not yet, bug-ing 😭😭)")
    
    os.chdir("D:/Project/ThesisProd/app")

with col2:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    mes_container = st.container(height=700)

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
        
        context = ss.store.get_big_context(retriever=retriever, query=prompt)
        
        with st.spinner("Teacher is creating questions..."):
            teacher = TeacherBot()
            _response = teacher.create_question(context = context)
            
        response = refactor(_response)

        # Display assistant response in chat message container
        with mes_container.chat_message("assistant"):
            show_question(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})