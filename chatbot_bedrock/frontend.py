import copy

from io import StringIO
import streamlit as st
import cv2
import numpy as np
import pandas as pd

import backend as demo
from utils import get_random_string, get_model_ids, format_float_dict
from constant import LIMIT_PRICE

st.title("Your AI assistant is here!")

select_event = st.sidebar.selectbox('Select model',
                                    ["Claude 3.5 Sonnet",'Claude 3 Haiku', "Amazon Titan Text Premier",'Amazon Titan Text Express'])
resetting = False

st.sidebar.write(f"Limit price: {LIMIT_PRICE} USD")

def change_max_token_limit():
    mtl = st.session_state.max_token_limit
    print("Update new memory.")
    st.session_state.memory = demo.demo_memory(mtl)

# Initialize session state if it does not exist
if 'max_token_limit' not in st.session_state:
    st.session_state.max_token_limit = 1024

max_token_limit = st.sidebar.select_slider(
    "Select max token length of memory",
    options=[128, 256, 512, 1024, 2048],
    value=st.session_state.max_token_limit,
    on_change=change_max_token_limit
)

if 'token_status_obj' not in st.session_state:
    st.session_state.token_status_obj = {}

if 'memory' not in st.session_state:
    st.session_state.memory = demo.demo_memory(max_token_limit)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['text'])

if 'Claude' in select_event:
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'], key=f"uploader_{str(st.session_state.uploader_key)}")
else:
    st.session_state.uploader_key +=1
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'], disabled=True, key=f"uploader_{str(st.session_state.uploader_key)}")
    st.sidebar.write(":red[Only support upload image with Claude-family models]")

image = None
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    st.sidebar.image(image, caption="Uploaded image", channels="BGR")

def edit_get_table_usage(df):
    if not df.empty:
        df.index = ['Input tokens', 'Output tokens', 'Price']
        return copy.deepcopy(df).drop(columns='total')
    else:
        return df

st.sidebar.table(
    edit_get_table_usage(pd.DataFrame(format_float_dict(copy.deepcopy(st.session_state.token_status_obj)))).T
    )
if 'total' in list(st.session_state.token_status_obj.keys()):
    st.sidebar.write(f"Total price: {format_float_dict(copy.deepcopy(st.session_state.token_status_obj))['total']} USD")
    total_usage_price = st.session_state.token_status_obj['total']
else:
    st.sidebar.write("Total: 0.0 USD")
    total_usage_price = 0.0

input_text = st.chat_input("Hello it's testing...", disabled=bool(total_usage_price>LIMIT_PRICE))


if input_text:
    with st.chat_message('user'):
        st.markdown(input_text)

    st.session_state.chat_history.append({"role": "user", "text": input_text})
    if image is not None:
        # Edit input prompt
        prompt = f"""The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n\n{st.session_state.memory.load_memory_variables({})['history']}"""

        # Make API request
        message = {
            "role": "user",
            "content": [
                {
                    "text": prompt
                },
                {
                        "image": {
                            "format": 'png',
                            "source": {
                                "bytes": bytes_data
                            }
                        }
                }
            ]
        }

        messages = [message]

        # Send the message.
        chat_response = demo.get_client().converse(
            modelId=demo.get_model_id(select_event),
            messages=messages
        )

        # Get usage
        current_token_usage = {
            "model_id": demo.get_model_id(select_event),
            "input_tokens": chat_response['usage']['inputTokens'],
            "output_tokens": chat_response['usage']['outputTokens'],
        }
        # Update usage and cost
        demo.get_current_token_usage(st.session_state.token_status_obj, current_token_usage)

        # Verbose to console 
        print(chat_response)
        chat_response = chat_response['output']['message']['content'][0]['text']
        
        # Save to memory
        st.session_state.memory.save_context({"input": input_text}, {"output": chat_response})

        # Resetting flag
        image = None
        uploaded_file = None
        st.session_state.image = None
        st.session_state.uploaded_file = []
        st.session_state.uploader_key +=1
    else:
        chat_response = demo.demo_conversation(input_text=input_text, memory=st.session_state.memory, model=select_event)

        # Get usage
        current_token_usage = demo.count_text_token_usage(input_text, chat_response, st.session_state.memory, get_model_ids()[select_event])

        # Update usage and cost
        demo.get_current_token_usage(st.session_state.token_status_obj, current_token_usage)

    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
    print(st.session_state.token_status_obj)
    st.rerun()