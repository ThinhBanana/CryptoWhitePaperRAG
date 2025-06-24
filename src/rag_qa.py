import torch
import data_embedding
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_answer(query: str, context: str) -> str:
    result = ''
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForCausalLM.from_pretrained("google/flan-t5-base", device_map="auto")


    return result
