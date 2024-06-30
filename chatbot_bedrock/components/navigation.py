import streamlit as st

def page_navigation():
    if st.button("Home"):
        st.switch_page("homepage.py")
    if st.button("Chatbot"):
        st.switch_page("chatbot.py")
    if st.button("Pdfchat"):
        st.switch_page("pdfchat.py")