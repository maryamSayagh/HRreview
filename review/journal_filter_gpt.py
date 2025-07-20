import re
import ast
from typing import List, Dict
from more_itertools import chunked
import pandas as pd
import openai


def gpt_filter_by_list_response(
    items: List[str],
    client: openai.OpenAI,
    column_name: str,
    context_text: str
) -> List[str]:
    """
    Uses GPT to return a list of irrelevant items directly.
    The model is prompted to output a Python list of items that are not useful.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a classification agent, specializing in identifying irrelevant items from lists based on project context."
        },
        {
            "role": "system",
            "content": f"You understand the following project context:\n{context_text}"
        },
        {
            "role": "user",
            "content": f"Extract the irrelevant entries from the column '{column_name}':\n{items}"
        },
        {
            "role": "user",
            "content": f"Return ONLY a Python list of irrelevant items, e.g. ['item1', 'item2']"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        temperature=0
    )
    raw = response.choices[0].message.content
    raw = re.sub(r"```(?:python)?", "", raw).strip()

    try:
        result: List[str] = ast.literal_eval(raw)
        return result if isinstance(result, list) else []
    except Exception as e:
        print("Failed to parse GPT output (list response):", e)
        print("Raw response:\n", raw[:1000])
        return []


def gpt_filter_by_dict_classification(
    items: List[str],
    client: openai.OpenAI,
    column_name: str,
    context_text: str
) -> List[str]:
    """
    Uses GPT to classify items with a 0/1 label and returns items classified as 0 (irrelevant).
    """
    messages = [
        {
            "role": "system",
            "content": "You are a classification agent that returns whether each item is relevant (1) or irrelevant (0)."
        },
        {
            "role": "system",
            "content": f"The following is the project context:\n\n{context_text}"
        },
        {
            "role": "user",
            "content": (
                f"Given the list of values from column '{column_name}', classify each item "
                f"as relevant (1) or irrelevant (0).\n\n"
                f"Return ONLY a valid Python dictionary like this:\n"
                f"{{'item1': 1, 'item2': 0}}\n\n"
                f"Here is the list:\n{items}"
            )
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        temperature=0
    )

    raw = response.choices[0].message.content
    raw = re.sub(r"```(?:python)?", "", raw).strip()

    try:
        result: Dict[str, int] = ast.literal_eval(raw)
        return [item for item, val in result.items() if val == 0]
    except Exception as e:
        print("Failed to parse GPT output (dict response):", e)
        print("Raw response:\n", raw[:1000])
        return []


def filter_journal_dataframe(
    df: pd.DataFrame,
    client: openai.OpenAI,
    column_names: List[str],
    context_text: str,
    strategy: str = "dict",  # can be "dict" or "list"
    chunk_size: int = 100
) -> pd.DataFrame:
    """
    Filters irrelevant rows from a DataFrame based on GPT classification of selected columns.

    Parameters:
        strategy: Either 'dict' (for classification) or 'list' (for list-returning prompt)
    """
    irrelevant_all = set()

    for col in column_names:
        unique_values = df[col].dropna().unique().tolist()
        for chunk in chunked(unique_values, chunk_size):
            if strategy == "dict":
                irrelevant = gpt_filter_by_dict_classification(chunk, client, col, context_text)
            elif strategy == "list":
                irrelevant = gpt_filter_by_list_response(chunk, client, col, context_text)
            else:
                raise ValueError("Strategy must be either 'dict' or 'list'")
            irrelevant_all.update(irrelevant)

        df = df[~df[col].isin(irrelevant_all)]

    return df
