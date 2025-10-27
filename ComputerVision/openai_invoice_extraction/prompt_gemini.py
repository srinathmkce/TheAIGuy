import mlflow

def register_prompt(prompt_name):
    system_prompt = """You are an AI Vision-Language Model specialized in document understanding and structured data extraction.

    Task:
    Extract data from the provided invoice receipt and convert it into a well-formed JSON object.

    Context:

    Only extract these sections if present: menu, sub_menu, sub_total, total.

    Preserve the exact formatting of values (including numbers, text, prices, and currencies).

    Do not generate fields with missing data—omit empty keys.

    Do not infer or add information that is not explicitly present in the invoice.

    Format:
    Follow this schema exactly:
    {{schema}}

    Return only valid JSON, minimal and strict—no extra keys, no null values, no explanations.

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

    if "few_shot" in prompt_name:
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
