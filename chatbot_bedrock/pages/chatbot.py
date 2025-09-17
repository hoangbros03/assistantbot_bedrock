import copy
from io import StringIO

import backend as demo
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from components.sidebar import sidebar
from constant import LIMIT_PRICE
from backend.utils import get_model_ids
from backend.utils import get_random_string
from backend.enums import AllowImageFileExtensions
from backend.prompts import DEFAULT_PROMPT

st.title("Your AI assistant is here!")
select_event, max_token_limit, total_usage_price = sidebar()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

if "Claude" in select_event:
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image",
        type=AllowImageFileExtensions.values(),
        key=f"uploader_{str(st.session_state.uploader_key)}",
    )
else:
    st.session_state.uploader_key += 1
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image",
        type=AllowImageFileExtensions.values(),
        disabled=True,
        key=f"uploader_{str(st.session_state.uploader_key)}",
    )
    st.sidebar.write(
        ":red[Only support upload image with Claude-family models]"
    )

image = None
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    st.sidebar.image(image, caption="Uploaded image", channels="BGR")

input_text = st.chat_input(
    "Hello it's testing...", disabled=bool(total_usage_price > LIMIT_PRICE)
)


if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)

    st.session_state.chat_history.append({"role": "user", "text": input_text})
    if image is not None:
        # Edit input prompt
        prompt = DEFAULT_PROMPT.format(
            st.session_state.memory.load_memory_variables({})["history"]
        )

        # Make API request
        message = {
            "role": "user",
            "content": [
                {"text": prompt},
                {"image": {"format": "png", "source": {"bytes": bytes_data}}},
            ],
        }

        messages = [message]

        # Send the message.
        chat_response = demo.get_client().converse(
            modelId=demo.get_model_id(select_event), messages=messages
        )

        # Get usage
        current_token_usage = {
            "model_id": demo.get_model_id(select_event),
            "input_tokens": chat_response["usage"]["inputTokens"],
            "output_tokens": chat_response["usage"]["outputTokens"],
        }
        # Update usage and cost
        demo.get_current_token_usage(
            st.session_state.token_status_obj, current_token_usage
        )

        # Verbose to console
        print(chat_response)
        chat_response = chat_response["output"]["message"]["content"][0][
            "text"
        ]

        # Save to memory
        st.session_state.memory.save_context(
            {"input": input_text}, {"output": chat_response}
        )

        # Resetting flag
        image = None
        uploaded_file = None
        st.session_state.image = None
        st.session_state.uploaded_file = []
        st.session_state.uploader_key += 1
    else:
        chat_response = demo.demo_conversation(
            input_text=input_text,
            memory=st.session_state.memory,
            model=select_event,
        )

        # Get usage
        current_token_usage = demo.count_text_token_usage(
            input_text,
            chat_response,
            st.session_state.memory,
            get_model_ids()[select_event],
        )

        # Update usage and cost
        demo.get_current_token_usage(
            st.session_state.token_status_obj, current_token_usage
        )

    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append(
        {"role": "assistant", "text": chat_response}
    )
    print(st.session_state.token_status_obj)

    # Remove history
    st.session_state.memory = demo.demo_memory(max_token_limit)
    st.rerun()
