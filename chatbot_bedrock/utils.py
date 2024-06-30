import random
import string

def get_random_string(length):
    letters = string.ascii_lowercase  # Use string.ascii_uppercase for uppercase letters
    return ''.join(random.choice(letters) for i in range(length))

def get_model_ids():
    return {
        "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "Amazon Titan Text Premier": "amazon.titan-text-premier-v1:0",
        'Amazon Titan Text Express': "amazon.titan-text-express-v1",
        'Amazon Text Embedding V2': "amazon.titan-embed-text-v2:0"
    }

def get_model_prices():
    return {
        "anthropic.claude-3-haiku-20240307-v1:0": [0.00025,0.00125],
        "anthropic.claude-3-5-sonnet-20240620-v1:0": [0.003,0.015],
        "amazon.titan-text-premier-v1:0": [0.0005, 0.0015],
        "amazon.titan-text-express-v1": [0.0002, 0.0006],
        "amazon.titan-embed-text-v2:0": [0.00002,0.0]
    }

def format_float_dict(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, float):
            output_dict[key] = f"{value:.4f}"
        else:
            output_dict[key] = value
    return output_dict