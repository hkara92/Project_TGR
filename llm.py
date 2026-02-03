"""
llm.py

Unified interface for LLM calls and embeddings.
Supports: OpenAI GPT-5, Qwen2.5-7B-Instruct
"""

import os
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()


# ============ TOKENIZER ============

def get_tokenizer(model: str = "gpt"):
    """Get tokenizer for the specified model."""
    if model == "gpt":
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    elif model == "qwen":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    else:
        raise ValueError(f"Unknown model: {model}")


# ============ LLM CALLS ============

def call_llm(prompt: str, model: str = "gpt", max_tokens: int = 1024) -> str:
    """
    Call LLM and return response text.
    
    Args:
        prompt: Input prompt
        model: "gpt" (GPT-5) or "qwen" (Qwen2.5-7B-Instruct)
        max_tokens: Maximum tokens to generate
    """
    if model == "gpt":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            print("[WARNING] LLM output truncated due to Max Tokens limit!")
        print(f"[DEBUG] Finish reason: {finish_reason}")
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content.strip()
    
    elif model == "qwen":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        llm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        outputs = llm.generate(**inputs, max_new_tokens=max_tokens)
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    else:
        raise ValueError(f"Unknown model: {model}")


# ============ EMBEDDINGS ============

def get_embeddings(texts: Union[str, List[str]]) -> List[List[float]]:
    """Get embeddings using OpenAI text-embedding-3-large."""
    from openai import OpenAI
    
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_embeddings = []
    
    for i in range(0, len(texts), 100):  # Batch size 100
        batch = texts[i:i + 100]
        response = client.embeddings.create(model="text-embedding-3-large", input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    
    return all_embeddings


def get_embedding_dim() -> int:
    """Return embedding dimension (3072 for text-embedding-3-large)."""
    return 3072