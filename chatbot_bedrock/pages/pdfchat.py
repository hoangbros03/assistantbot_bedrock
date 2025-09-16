import streamlit as st

import backend as demo
from components.sidebar import sidebar
from utils import get_model_ids

select_event, max_token_limit, total_usage_price = sidebar()

if "text_uploader_key" not in st.session_state:
    st.session_state.text_uploader_key = 0

if "db" not in st.session_state:
    st.session_state.db = None

if "embed_model" not in st.session_state:
    st.session_state.embed_model = demo.get_embedding()


def new_text_file():
    with open(f"temp_file.{uploaded_text_file.name[-3:]}", "wb") as file:
        file.write(uploaded_text_file.getvalue())
    document_list = demo.get_document_list(f"temp_file.{uploaded_text_file.name[-3:]}")
    st.session_state.db = demo.get_faiss_db(document_list, st.session_state.embed_model)
    # Get embed usage
    demo.get_current_token_usage(
        st.session_state.token_status_obj, None, st.session_state.embed_model
    )


uploaded_text_file = st.file_uploader(
    "Choose a file",
    type=["txt", "pdf"],
    key=f"uploader_{str(st.session_state.text_uploader_key)}",
    on_change=new_text_file,
)

input_text = st.chat_input(
    "Start chat with pdf...", disabled=bool(uploaded_text_file is None)
)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)

    ref_list = demo.get_similarity_docs(input_text, db=st.session_state.db)
    ref_list = "\n".join(
        [f"Page {doc.metadata['page']}: {doc.page_content}" for doc in ref_list]
    )
    ref_part = f"References found when retrieve data:\n{ref_list}"
    with st.chat_message("assistant"):
        st.markdown(f"{ref_part[:50]}...")
    print("Ref part: ", ref_part[:50])
    st.session_state.memory.save_context(
        {"input": "Found relevant information to answer the last question"},
        {"output": ref_part},
    )

    chat_response = demo.demo_conversation(
        input_text=input_text, memory=st.session_state.memory, model=select_event
    )

    # Get usage
    current_token_usage = demo.count_text_token_usage(
        input_text,
        chat_response,
        st.session_state.memory,
        get_model_ids()[select_event],
    )

    # Update usage and cost
    demo.get_current_token_usage(st.session_state.token_status_obj, current_token_usage)

    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append({"role": "user", "text": input_text})
    st.session_state.chat_history.append({"role": "assistant", "text": ref_part})
    st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
    print(st.session_state.token_status_obj)
    st.rerun()
