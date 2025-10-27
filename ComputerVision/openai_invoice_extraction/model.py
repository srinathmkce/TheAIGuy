import mlflow
import os
import openai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

load_dotenv()


def log_openai_invoice_extraction_model(model_name, prompt_name, prompt_version, reasoning="low"):
    """
    Log OpenAI model using MLflow OpenAI flavor for invoice extraction.
    Supports both GPT-5 (with reasoning) and GPT-4o-mini models.
    """
    # Load the prompt template
    system_prompt_template = mlflow.genai.load_prompt(
        name_or_uri=prompt_name, version=prompt_version
    )
    
    # Create the messages structure for the model
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": system_prompt_template.template},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ]
    
    # Define model signature for input/output
    signature = ModelSignature(
        inputs=Schema([
            ColSpec(type="string", name="image_base64"),
            ColSpec(type="string", name="schema")
        ]),
        outputs=Schema([ColSpec(type="string", name=None)]),
        params=ParamSchema([
            ParamSpec(name="temperature", default=0, dtype="float"),
            ParamSpec(name="max_tokens", default=4000, dtype="int"),
        ])
    )
    
    with mlflow.start_run(run_name=f"{model_name}-invoice-extraction") as run:
        if model_name.startswith("gpt-5"):
            # For GPT-5 models, we need to handle reasoning differently
            # Since MLflow OpenAI flavor doesn't directly support responses.create,
            # we'll use chat.completions with custom parameters
            model_info = mlflow.openai.log_model(
                model=model_name,
                task=openai.chat.completions,
                name="invoice_extraction_model",
                messages=messages,
                signature=signature,
                # Store reasoning as a custom parameter
                extra_params={"reasoning_effort": reasoning}
            )
        else:
            # For standard models like GPT-4o-mini
            model_info = mlflow.openai.log_model(
                model=model_name,
                task=openai.chat.completions,
                name="invoice_extraction_model", 
                messages=messages,
                signature=signature
            )
        
        # Log model parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("prompt_name", prompt_name)
        mlflow.log_param("prompt_version", prompt_version)
        if model_name.startswith("gpt-5"):
            mlflow.log_param("reasoning_effort", reasoning)
    
    return model_info


def log_gemini_invoice_extraction_model(model_name, prompt_name, prompt_version):
    """
    Log Gemini model using MLflow LangChain flavor for invoice extraction.
    """
    system_prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}/{prompt_version}").template
    
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    with mlflow.start_run(run_name=f"{model_name}-invoice-extraction") as run:
        # Log the model using LangChain format
        model_info = mlflow.langchain.log_model(
            lc_model=llm,
            artifact_path="gemini_model",
            registered_model_name=f"{model_name}",
        )
        
        # Log model parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("prompt_name", prompt_name)
        mlflow.log_param("prompt_version", prompt_version)

    return model_info


