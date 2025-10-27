import mlflow
from mlflow.pyfunc import PythonModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from mlflow.models import set_model


class LangchainModel(PythonModel):
    def __init__(self):
        super().__init__()

    def load_context(self, context):
        print("Loading context")
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
            check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
            max_bucket_size=10,  # Controls the maximum burst size.
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            rate_limiter=rate_limiter,
        )

    def predict(self, model_input: list[dict[str, str]]) -> list[str]:
        responses = []
        for data in model_input:
            print("Received Input: ", data)
            article = data.get("article")
            prompt_uri = data.get("prompt_uri")
            if not prompt_uri:
                raise ValueError("prompt_uri is required")
            if not article:
                raise ValueError("article is required")
            prompt_template = mlflow.genai.load_prompt(prompt_uri).template
            prompt = prompt_template.format(article=article)
            response = self.llm.invoke(prompt)
            responses.append(response.content)
        return responses


set_model(LangchainModel())
