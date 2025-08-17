import re
import ast
from typing import List, Dict
from more_itertools import chunked
import pandas as pd
from mistralai import Mistral

try:
    from tqdm.notebook import tqdm
    use_tqdm = True
except ImportError:
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        tqdm = None
        use_tqdm = False

import re
import ast


import re, ast, json
import json, ast, re

def extract_dict_from_response(text: str) -> dict:
    # Use regex to find a dictionary in the text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return {}




def gpt_filter_by_list_response(
    items: List[str],
    client: Mistral,
    column_name: str,
    context_text: str,
    model:str = 'mistral-small'
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

    response = response = client.chat.complete(
        model= model,
        messages=messages,
        temperature=0
    )
    raw = response.choices[0].message.content
    raw = re.sub(r"```[\w]*", "", raw).strip()
    result = extract_dict_from_response(raw)
    if not result:
        print("Failed to parse model response for chunk:", chunk[:50])
        return [False] * len(chunk)  # or handle as needed
    return [val == 0 for val in result.values()]



import re
import ast
from typing import List, Dict

def gpt_filter_by_dict_classification(
    items: List[str],
    client: Mistral,
    column_name: str,
    context_text: str,
    model: str = 'mistral-small'
) -> Dict[str, int]:

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
                f"Here is the list:\n{items}")
        }
    ]

    try:
        response = client.chat.complete(
            model=model,
            messages=messages,
            temperature=0.5
        )
        raw = response.choices[0].message.content
        raw = re.sub(r"```[\w]*", "", raw).strip()
        print("Raw response:", raw)  # Debug print
        result = extract_dict_from_response(raw)
        if not result:
            print("Failed to parse model response for chunk:", items[:50])
            return {}
        return result
    except Exception as e:
        print(f"Error in classification: {e}")
        return {}





def filter_journal_dataframe(
            df: pd.DataFrame,
            client: Mistral,
            column_names: List[str],
            context_text: str,
            strategy: str = "dict",
            chunk_size: int = 100,
            model: str = 'mistral-small',
            use_tqdm: bool = True
    ) -> pd.DataFrame:
        filtered_df = df.copy()
        try:
            for col in column_names:
                irrelevant_col = set()
                unique_values = filtered_df[col].dropna().unique().tolist()
                chunks = list(chunked(unique_values, chunk_size))
                total_chunks = len(chunks)
                print(f"[Modeld Filter] Starting filtering for column '{col}' with {total_chunks} chunk(s)...")
                chunk_iter = tqdm(enumerate(chunks), total=total_chunks, desc=f"Filtering {col}", leave=True,
                                  dynamic_ncols=True) if use_tqdm and tqdm else enumerate(chunks)
                for i, chunk in chunk_iter:
                    print(f"Chunk size: {len(chunk)}")
                    if strategy == "dict":
                        irrelevant = gpt_filter_by_dict_classification(chunk, client, col, context_text, model)
                    elif strategy == "list":
                        irrelevant = gpt_filter_by_list_response(chunk, client, col, context_text, model)
                    else:
                        raise ValueError("Strategy must be either 'dict' or 'list'")
                    irrelevant_col.update(irrelevant)
                    if not use_tqdm:
                        percent = ((i + 1) / total_chunks) * 100
                        print(f"[Model Filter] {col}: Chunk {i + 1}/{total_chunks} ({percent:.1f}%) done.")
                filtered_df = filtered_df[~filtered_df[col].isin(irrelevant_col)]
                print(f"[Model Filter] Completed filtering for column '{col}'.")
            return filtered_df
        except Exception as e:
            print(f"[Model Filter] Exception occurred: {e}. ")
            print("[Model Filter] Returning partially filtered DataFrame.")
            return filtered_df
