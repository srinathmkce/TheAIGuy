import mlflow
from mlflow.pyfunc import PythonModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from mlflow.models import set_model


class LangchainModel(PythonModel):
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=10,  # Controls the maximum burst size.
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        rate_limiter=rate_limiter,
    )

    def __init__(self):
        super().__init__()

    @classmethod
    def predict(cls, model_input: dict) -> str:
        prompt_template = mlflow.genai.load_prompt(
            "news_classifier", version=1
        ).template
        prompt = prompt_template.format(article=model_input)
        response = cls.llm.invoke(prompt)
        print(response)
        return response.content


set_model(LangchainModel())
