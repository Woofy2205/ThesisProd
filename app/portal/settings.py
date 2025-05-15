import os
from pathlib import Path

import streamlit as st
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

os.chdir("D:/Project/ThesisProd/")
# os.chdir("C:/Users/Learning/Project/ThesisProd")

from app.utils import *
from core.ingestion.preprocessing.advanced.hyde.HyDe import HyDETransformer
from core.ingestion.preprocessing.audio.AudioProcessing import AudioProcessor

st.title("Let's customize your learning 😁")
tab1, tab2 = st.tabs(["PDF", "Video"])

with tab1:
    os.chdir("D:/Project/ThesisProd/")
    if 'pdf' not in ss:
        ss['pdf'] = None

    if 'store' not in ss:
        ss['store'] = None

    if 'nodes' not in ss:
        ss['nodes'] = None
    
    pdf_file = st.file_uploader("Upload PDF file", type=('pdf'), label_visibility = "collapsed")
    
    ss.pdf = pdf_file
    save_folder = "app/static/pdfdir/"
    
    with st.expander("PDF Preview", expanded=True):
        if ss.pdf:
            binary_data = ss.pdf.getvalue()
            pdf_viewer(input=binary_data, width=1500, height=1500)
        
        if pdf_file is not None:
            save_path = os.path.join(save_folder, pdf_file.name)
            for file in os.listdir(save_folder):
                os.remove(os.path.join(save_folder, file))
            with open(save_path, mode='wb') as w:
                w.write(pdf_file.getvalue())
        else:
            st.info("Please upload a PDF file.")
    
    subcol1, subcol2 = st.columns([5,5])
    if subcol1.button("Process PDF 🚩", use_container_width=True):
        with st.spinner("Teacher digesting the PDF"):
            ss.nodes = process_pdf(save_folder)
        if ss.nodes:
            subcol1.success("PDF processed successfully!")
        else:
            subcol1.error("Failed to process PDF.")

    if subcol2.button("Creating Test Bank👍", use_container_width=True):
        with st.spinner("Teacher preparing questions..."):
            ss.store = create_store(ss.nodes)
        if ss.store:
            subcol2.success("Test Bank is ready! 🎉")
        else:
            subcol2.error("Failed to create test bank.")
    # os.chdir("C:/Users/Learning/Project/ThesisProd/app")
    os.chdir("D:/Project/ThesisProd/app/")

with tab2:
    os.chdir("D:/Project/ThesisProd/")
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
    
    ss.video_url = st.text_input("Enter video URL here")
    with st.expander("Video Preview", expanded=True):
        if ss.video_url:
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
                st.video(ss.video_url)
                st.write(ss.context)
            else:
                st.video(ss.video_url)
                st.write(ss.context)
        else:
            st.info("Please enter a video URL.")
    # os.chdir("C:/Users/Learning/Project/ThesisProd/app")
    os.chdir("D:/Project/ThesisProd/app")
os.chdir("D:/Project/ThesisProd/app")   