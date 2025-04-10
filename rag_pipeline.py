import os
os.environ["HF_HOME"] = "C:/Games/hf/huggingface"

import vector_store
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig
import time
from typing import Optional, Tuple

class RagPipeline():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("yandex/YandexGPT-5-Lite-8B-instruct")#mistralai/Mistral-7B-Instruct-v0.3
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            "yandex/YandexGPT-5-Lite-8B-instruct",
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.vs = vector_store.VectorStore()

    def generate(self, question:str) -> Tuple[str, float]:
        start = time.time()
        retrieved_chunks = self.vs.find_similar(question, 5)

        prompt = f"""
        Ты — чат-бот контактного центра, обученный отвечать на вопросы на основе предоставленных документов.
        Ниже представлен контекст, извлечённый из базы данных:
        <context>
        {retrieved_chunks}
        </context>
        Вопрос пользователя:
        {question}
        
        Ответь кратко, ясно и по существу, строго основываясь на предоставленном контексте. Если информации недостаточно — честно скажи, что ты не можешь ответить на этот вопрос.
        Ответ:
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=400,
            eos_token_id=self.tokenizer.eos_token_id,
            #do_sample=False,            # без сэмплинга — быстрее
            #num_beams=1,                # без beam search
            use_cache=True              # кэширование внимания
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip(), time.time() - start