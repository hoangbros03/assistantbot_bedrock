import copy
import os
from os import environ

import boto3
from constant import EMBED_MODEL_ID
from dotenv import load_dotenv
from backend.embedding import BedrockEmbeddings  # Overwrite
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_aws import BedrockLLM
from langchain_aws import ChatBedrock
from langchain_community.chat_models import BedrockChat
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_text_splitters import CharacterTextSplitter
from settings import get_settings
from backend.utils import get_model_ids
from backend.utils import get_model_prices
from backend.prompts import DEFAULT_PROMPT

# CONSTANTS
DEMO_CHATBOT_TEMPERATURE = 1.0
DEMO_CHATBOT_TOP_P = 0.9
DEMO_CHATBOT_TOP_K = 250

# Setup AWS
settings = get_settings()


class BedrockClientSingleton:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = boto3.client("bedrock-runtime")
        return cls._client


def get_client():
    return BedrockClientSingleton.get_client()


# Available models
def demo_chatbot(model_id="amazon.titan-text-express-v1"):
    if "claude" in model_id:
        return BedrockChat(
            model_id=model_id,
            model_kwargs={
                "temperature": DEMO_CHATBOT_TEMPERATURE,
                "top_k": DEMO_CHATBOT_TOP_K,
            },
        )
    demo_llm = BedrockLLM(
        model_id=model_id,
        model_kwargs={
            "temperature": DEMO_CHATBOT_TEMPERATURE,
            "topP": DEMO_CHATBOT_TOP_P,
        },
    )
    return demo_llm


def demo_memory(max_token_limit=1024):
    memory = ConversationBufferMemory(max_token_limit=max_token_limit)
    return memory


def get_model_id(model_display_name):
    return get_model_ids()[model_display_name]


def demo_conversation(input_text, memory, model):
    llm_conversation = ConversationChain(
        llm=demo_chatbot(get_model_id(model)), memory=memory, verbose=True
    )
    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply


def count_text_token_usage(input_text, response, memory, model_id):
    input_prompt = DEFAULT_PROMPT.format(
        current_conversation=memory.load_memory_variables({})["history"]
    )

    # Count token
    model = demo_chatbot(model_id)
    input_tokens = model.get_num_tokens(input_prompt)
    output_tokens = model.get_num_tokens(response)

    # Return
    return {
        "model_id": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def get_current_token_usage(token_status_obj, count_result, embed_model=None):
    if len(list(token_status_obj.keys())) == 0:
        for name in list(get_model_ids().values()):
            token_status_obj[name] = [
                0,
                0,
                0.0,
            ]  # Input token, output token, price
        token_status_obj["total"] = 0.0

    # init if empty status
    if embed_model is not None:
        token_status_obj[EMBED_MODEL_ID][0] = embed_model.response_token_count
        token_status_obj[EMBED_MODEL_ID][2] = (
            token_status_obj[EMBED_MODEL_ID][0]
            * get_model_prices()[EMBED_MODEL_ID][0]
            / 1000.0
        )
    else:
        token_status_obj[count_result["model_id"]][0] += count_result[
            "input_tokens"
        ]
        token_status_obj[count_result["model_id"]][1] += count_result[
            "output_tokens"
        ]
        token_status_obj[count_result["model_id"]][2] = (
            token_status_obj[count_result["model_id"]][0]
            * get_model_prices()[count_result["model_id"]][0]
            / 1000.0
            + token_status_obj[count_result["model_id"]][1]
            * get_model_prices()[count_result["model_id"]][1]
            / 1000.0
        )
    token_status_obj["total"] = sum(
        [
            token_status_obj[list(get_model_ids().values())[i]][2]
            for i in range(len(list(get_model_ids().values())))
        ]
    )
    # return token_status_obj


def get_document_list(file_path: str):
    if file_path[-3:] == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load_and_split()
    elif file_path[-3:] == "txt":
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(documents)
    else:
        raise ValueError("File is not txt or pdf")


def get_embedding():
    br_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0", client=get_client()
    )
    return br_embeddings


def get_faiss_db(docs, embed_model):
    db = FAISS.from_documents(docs, embed_model)
    print(db.index.ntotal)
    return db


def get_similarity_docs(query, db, k=5):
    return db.similarity_search(query, k=k)
