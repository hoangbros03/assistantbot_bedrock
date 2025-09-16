import streamlit as st


def page_navigation():
    if st.button("Home"):
        st.switch_page("./pages/homepage.py")
    if st.button("Chatbot"):
        st.switch_page("./pages/chatbot.py")
    if st.button("Pdfchat"):
        st.switch_page("./pages/pdfchat.py")
