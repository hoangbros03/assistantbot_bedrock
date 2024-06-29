import os 
from os import environ
import copy

from dotenv import load_dotenv
import boto3
from langchain_aws import ChatBedrock, BedrockLLM 
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain 
from langchain_core.messages import HumanMessage, AIMessage

from utils import get_model_ids, get_model_prices

# Setup AWS
load_dotenv()
os.environ['AWS_DEFAULT_REGION'] = environ.get('AWS_DEFAULT_REGION')
os.environ['AWS_SECRET_ACCESS_KEY'] = environ.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_ACCESS_KEY_ID'] = environ.get('AWS_ACCESS_KEY_ID')
client = boto3.client("bedrock-runtime")

def get_client():
    return client

# Available models
def demo_chatbot(model_id="amazon.titan-text-express-v1"):
    if 'claude' in model_id:
        return BedrockChat(
            model_id=model_id,
            model_kwargs ={
            "temperature": 1.0,
            "top_k": 250
            },
        )
    demo_llm = BedrockLLM(
        model_id=model_id,
        model_kwargs ={
            "temperature": 1.0,
            "topP": 0.9
        },
    )
    return demo_llm
    
def demo_memory(max_token_limit=1024):
    memory = ConversationBufferMemory(max_token_limit=max_token_limit) 
    return memory

def get_model_id(model_display_name):
    return get_model_ids()[model_display_name]

def demo_conversation(input_text, memory, model):
    llm_conversation = ConversationChain(llm=demo_chatbot(get_model_id(model)), memory=memory, verbose=True)
    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply

def count_text_token_usage(input_text, response, memory, model_id):
    input_prompt=f"""The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n\n{memory.load_memory_variables({})['history']}"""

    # Count token
    model = demo_chatbot(model_id)
    input_tokens = model.get_num_tokens(input_prompt)
    output_tokens = model.get_num_tokens(response)
    
    # Return
    return {
        "model_id": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

def get_current_token_usage(token_status_obj, count_result):
    # init if empty status
    if len(list(token_status_obj.keys())) ==0:
        for name in list(get_model_ids().values()):
            token_status_obj[name] = [0,0,0.0] # Input token, output token, price
        token_status_obj['total'] = 0.0
    token_status_obj[count_result['model_id']][0] += count_result['input_tokens']
    token_status_obj[count_result['model_id']][1] += count_result['output_tokens']
    token_status_obj[count_result['model_id']][2] = \
        token_status_obj[count_result['model_id']][0]*\
        get_model_prices()[count_result['model_id']][0]/1000.0 + \
        token_status_obj[count_result['model_id']][1]*\
        get_model_prices()[count_result['model_id']][1]/1000.0
    token_status_obj['total'] = sum([token_status_obj[list(get_model_ids().values())[i]][2] for i in range(len(list(get_model_ids().values())))])
    # return token_status_obj
