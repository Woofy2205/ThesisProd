import streamlit as st

st.title("Chat with your Lecture 😁")
st.write(
   """This is a chatbot that can create the questions from your lecture"""
)

col1, col2 = st.columns(2)

with col1:
   st.subheader("Instructions for Teacher")
   st.markdown(
      """
      Write Something here
      """
   )

with col2:
   st.subheader("Instructions for Student")
   st.markdown(
      """
      Write Something here
      """
   )