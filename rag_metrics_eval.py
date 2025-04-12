import os

os.environ["HF_HOME"] = "C:/Games/hf/huggingface"

from rag_pipeline import RagPipeline
from generative_model import LLM

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
import pickle
import re

class JudgeModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "yandex/YandexGPT-5-Lite-8B-instruct"):
        self.model_name = model_name
        self.model = LLM()

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        return self.model.generate(prompt=prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


def generate_qa():
    rag = RagPipeline()
    # documents = random.sample(rag.vs.get_documents(), k=k)
    test_cases = []
    # llm = LLM()

    # for doc in documents:
    #     prompt_q = f"Сгенерируй вопрос на основе контекста:\n\n{doc}"
    #     question = llm.generate(prompt_q)
    #
    #     prompt_a = f"Ответь на вопрос на основе контекста:\n\n{question}\n\nКонтекст: {doc}"
    #     expected_output = llm.generate(prompt_a)
    #
    #     actual_output, retrieval_context = rag.generate(question, save_message=False, return_context=True)
    #
    #     # test_cases.append(LLMTestCase(input=question, actual_output=actual_output,
    #     #             expected_output=expected_output,
    #     #             retrieval_context=retrieval_context))
    with open('data/G_QA.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    # Шаблон для поиска вопросов и ответов
    qa_blocks = re.split(r'\n\n+', content.strip())

    # Обрабатываем каждую пару
    qa_pairs = []
    for block in qa_blocks[:9]:
        if block:
            # Разделяем на вопрос и ответ
            parts = block.split('\n', 1)
            if len(parts) == 2:
                question = parts[0].strip()
                answer = parts[1].strip().replace('\n', ' ')
                actual_output, retrieval_context = rag.generate(question, save_message=False, return_context=True)

                test_cases.append(LLMTestCase(input=question, actual_output=actual_output,
                            expected_output=answer,
                            retrieval_context=[doc.page_content for doc in retrieval_context]))
    with open('data/tests.pkl', 'wb') as f:
        pickle.dump(test_cases, f)


if __name__ == '__main__':
    # generate_qa()
    judge = JudgeModel()
    with open('data/tests.pkl', 'rb') as f:
        tests = pickle.load(f)
    print(tests)
    answer_relevancy_metric = AnswerRelevancyMetric(model=judge, threshold=0.7)
    context_precision_metric = ContextualPrecisionMetric(model=judge, threshold=0.7)
    context_recall_metric = ContextualRecallMetric(model=judge, threshold=0.7)

    print(evaluate(test_cases=tests[:9], metrics=[answer_relevancy_metric, context_precision_metric, context_recall_metric]))