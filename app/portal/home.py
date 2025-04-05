import os

import streamlit as st

os.chdir("D:/Project/ThesisProd/")
# os.chdir("C:/Users/Learning/Project/ThesisProd")

st.title("Chat with your Lecture 😁")
st.write(
   """This is a chatbot that can create the questions from your lecture and answer them. """
)

st.subheader("Instructions for Student")
st.markdown(
   """
1. Upload a PDF file.
   - Upload a PDF file to create a test bank, any PDF files are accepted.
2. Process the PDF file.
   - Click the 'Process PDF' button to extract the text from the PDF file.
   - When you click the button, the teacher will start digesting the PDF and then return the nodes of the PDF for embedding and retrieving.
3. Create a test bank.
   - Click the 'Creating Test Bank' button to create a test bank from the nodes of the PDF.
   - When you click the button, the teacher will start preparing the questions.
4. Chat with the lecture.
   - You can chat with the lecture and type in keywords.
   - The lecture will generate the questions and answer them.
   """
)

os.chdir("D:/Project/ThesisProd/app/")
# os.chdir("C:/Users/Learning/Project/ThesisProd/app")