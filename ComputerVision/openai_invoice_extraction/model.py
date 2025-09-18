import mlflow
import openai
import os
from dotenv import load_dotenv

load_dotenv()


def log_invoice_extraction_model(model_name, reasoning, prompt_name, prompt_version):
    system_prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}/{prompt_version}").template
    with mlflow.start_run(run_name=f"{model_name}-{reasoning}") as run:
        model_info = mlflow.openai.log_model(
            model=model_name,
            reasoning={
                "effort": reasoning,
            },
            task=openai.chat.completions,
            name=f"{model_name}-{reasoning}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ],
        )

    return model_info


