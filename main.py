import os
os.environ["HF_HOME"] = "C:/Games/hf/huggingface"

import vector_store
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig
import time

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")


bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto"
)

question = 'Как мне провезти оружие в аэропорте?'

start = time.time()

vs = vector_store.VectorStore()
retrieved_chunks = vs.find_similar(question, 5)

prompt = f"""
Context information is below.
---------------------
{retrieved_chunks}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **input_ids,
    max_new_tokens=400,
    eos_token_id=tokenizer.eos_token_id,          # меньше — быстрее
    do_sample=False,            # без сэмплинга — быстрее
    num_beams=1,                # без beam search
    use_cache=True              # кэширование внимания
)

print(tokenizer.decode(outputs[0]))
print(f"Время генерации: {time.time() - start:.2f} сек")