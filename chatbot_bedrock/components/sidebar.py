import copy

import backend as demo
import pandas as pd
import streamlit as st
from constant import LIMIT_PRICE
from backend.utils import format_float_dict


def change_max_token_limit():
    mtl = st.session_state.max_token_limit
    print("Update new memory.")
    st.session_state.memory = demo.demo_memory(mtl)


def edit_get_table_usage(df):
    if not df.empty:
        df.index = ["Input tokens", "Output tokens", "Price"]
        return copy.deepcopy(df).drop(columns="total")
    else:
        return df


def sidebar():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize session state if it does not exist
    if "max_token_limit" not in st.session_state:
        st.session_state.max_token_limit = 1024

    if "token_status_obj" not in st.session_state:
        st.session_state.token_status_obj = {}

    max_token_limit = st.sidebar.select_slider(
        "Select max token length of memory",
        options=[128, 256, 512, 1024, 2048],
        value=st.session_state.max_token_limit,
        on_change=change_max_token_limit,
    )

    if "memory" not in st.session_state:
        st.session_state.memory = demo.demo_memory(max_token_limit)

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    st.sidebar.download_button(
        "Export the chat",
        data="\n".join(
            [
                f"role: {i['role']}, content: {i['text']}"
                for i in st.session_state.chat_history
            ]
        ),
        file_name="chat_history.txt",
    )

    select_event = st.sidebar.selectbox(
        "Select model",
        [
            "Amazon Titan Text Premier",
            "Claude 3.5 Sonnet",
            "Claude 3 Haiku",
            "Amazon Titan Text Express",
        ],
    )

    st.sidebar.write(f"Limit price: {LIMIT_PRICE} USD")

    st.sidebar.table(
        edit_get_table_usage(
            pd.DataFrame(
                format_float_dict(copy.deepcopy(st.session_state.token_status_obj))
            )
        ).T
    )
    if "total" in list(st.session_state.token_status_obj.keys()):
        st.sidebar.write(
            f"Total price: {format_float_dict(copy.deepcopy(st.session_state.token_status_obj))['total']} USD"
        )
        total_usage_price = st.session_state.token_status_obj["total"]
    else:
        st.sidebar.write("Total: 0.0 USD")
        total_usage_price = 0.0

    return select_event, max_token_limit, total_usage_price
