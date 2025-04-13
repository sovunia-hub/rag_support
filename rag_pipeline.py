import os

os.environ["HF_HOME"] = "C:/Games/hf/huggingface"

import vector_store
from generative_model import LLM
import time
from typing import Optional, Tuple

class RagPipeline:
    def __init__(self):
        self.llm = LLM()
        self.vs = vector_store.VectorStore()
        self.messages = []

    def clear_history(self):
        self.messages = []

    def rephrase(self) -> str:
        prompt = f"""
        Ты — помощник, который помогает переформулировать пользовательский вопрос так, чтобы он был понятен без дополнительного контекста и включал важную информацию из диалога. Используй только релевантные детали, избегай лишней воды.
        
        Контекст диалога:
        {self.messages[:-1]}
        
        Оригинальный вопрос:
        {self.messages[-1]}
        
        Переформулированный вопрос:
        """
        new_question = self.llm.generate(prompt=prompt)
        return new_question

    def generate(self, question:str, save_message:bool=True, return_context:bool = False):
        start = time.time()
        if save_message:
            self.messages.append({'Вопрос': question})
            new_question = self.rephrase()
        else:
            new_question = question
        retrieved_chunks = self.vs.find_similar(new_question, 5)

        prompt = f"""
        Ты — чат-бот контактного центра, обученный отвечать на вопросы на основе предоставленных документов.

        Ниже представлен контекст, извлечённый из базы данных:
        <context>
        {retrieved_chunks}
        </context>

        Вопрос пользователя:
        {question}

        Перефразированный вопрос пользователя:
        {new_question}

        Вопросы делятся на категории: Авиабилеты, ЖД билеты, Отели, Страховки, Личный кабинет, ЭДО, Баланс.

        Если для точного ответа информации недостаточно, **не сообщай об отсутствии данных**, вместо этого:
        — Задай вежливый уточняющий вопрос, чтобы получить недостающую информацию,
        — Или признайся, что у тебя нет информации.

        Ответь кратко, ясно и по существу, строго основываясь на предоставленном контексте.

        Результат:
        """
        answer = self.llm.generate(prompt=prompt)
        print(f'Время: \n\t{time.time() - start}\nВопрос: \n\t{question}\nПерефразированный вопрос: \n\t{new_question}')
        if save_message:
            self.messages.append({'Ответ': answer})
        if return_context:
            return '    ' + answer, retrieved_chunks
        return '     ' + answer, time.time() - start