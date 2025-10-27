import mlflow

def register_prompt(prompt_name):
    system_prompt = """You are a Vision Language Model designed to extract structured data from invoice receipts.
    Task:
    Convert the invoice receipt into a well-formed JSON object strictly following the schema provided.

    Requirements:
    1. Identify and extract only these sections (if present): `menu`, `sub_menu`, `sub_total`, `total`.  
    2. Preserve exact formatting for all the extracted values.  
    3. Do not output fields that lack data—omit empty keys.  
    4. Do not add any information not present in the invoice.
    5. In case of prices and currencies, ensure to maintain the original format without any modifications.

    Schema:
    {{schema}}

    Output:
    Return valid, minimal JSON matching this schema - no extraneous keys or null values.
    """

    few_shot_prompt = """---

    Few-shot Examples:

    Example 1:
    {{example1}}

    ---

    Example 2:
    {{example2}}

    ---
    Example 3:
    {{example3}}"""

    if "few-shot" in prompt_name:
        system_prompt = system_prompt + few_shot_prompt

    print(system_prompt)

    mlflow.genai.register_prompt(
        name = prompt_name,
        template = system_prompt
    )

    # if not mlflow.genai.search_prompts(filter_string=f"name='{prompt_name}'"):
    #     mlflow.genai.register_prompt(
    #         name = prompt_name,
    #         template = system_prompt
    #     )
    # else:
    #     print(f"Prompt - {prompt_name} already exists.")
