import os 
from os import environ

import boto3
from langchain_aws import ChatBedrock, BedrockLLM 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain 
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Setup AWS
load_dotenv()
os.environ['AWS_DEFAULT_REGION'] = environ.get('AWS_DEFAULT_REGION')
os.environ['AWS_SECRET_ACCESS_KEY'] = environ.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_ACCESS_KEY_ID'] = environ.get('AWS_ACCESS_KEY_ID')
client = boto3.client("bedrock-runtime")

def demo_chatbot():
    demo_llm = BedrockLLM(
        model_id="amazon.titan-text-express-v1",
        model_kwargs ={
            "temperature": 0.5,
            "topP": 0.9
        },
    )
    return demo_llm

# response = demo_chatbot("hi, what is your name?")
# print(response)

def demo_memory():
    memory = ConversationBufferMemory(max_token_limit=2048) 
    return memory

def demo_conversation(input_text, memory):
    # llm_chain_data = demo_chatbot()
    llm_conversation = ConversationChain(llm=demo_chatbot(), memory=memory, verbose=True)

    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply

