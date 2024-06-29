import os 
from os import environ

from dotenv import load_dotenv
import boto3
from langchain_aws import ChatBedrock, BedrockLLM 
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain 
from langchain_core.messages import HumanMessage, AIMessage

# Setup AWS
load_dotenv()
os.environ['AWS_DEFAULT_REGION'] = environ.get('AWS_DEFAULT_REGION')
os.environ['AWS_SECRET_ACCESS_KEY'] = environ.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_ACCESS_KEY_ID'] = environ.get('AWS_ACCESS_KEY_ID')
client = boto3.client("bedrock-runtime")

def get_client():
    return client

# Available models
# anthropic.claude-3-haiku-20240307-v1:0
# amazon.titan-text-express-v1
def demo_chatbot(model_id="amazon.titan-text-express-v1"):
    if 'claude' in model_id:
        return BedrockChat(
            model_id=model_id,
            model_kwargs ={
            "temperature": 0.5,
            "top_k": 250
            },
        )
    demo_llm = BedrockLLM(
        model_id=model_id,
        model_kwargs ={
            "temperature": 0.5,
            "topP": 0.9
        },
    )
    return demo_llm
    
def demo_memory(max_token_limit=1024):
    memory = ConversationBufferMemory(max_token_limit=max_token_limit) 
    return memory

def get_model_id(model):
    if model=="Claude 3 Haiku":
        return "anthropic.claude-3-haiku-20240307-v1:0"
    elif model=="Claude 3.5 Sonnet":
        return "anthropic.claude-3-5-sonnet-20240620-v1:0"
    elif model=="Amazon Titan Text Premier":
        return "amazon.titan-text-premier-v1:0"
    return "amazon.titan-text-express-v1"

def demo_conversation(input_text, memory, model):
    llm_conversation = ConversationChain(llm=demo_chatbot(get_model_id(model)), memory=memory, verbose=True)
    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply

