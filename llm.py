"""
llm.py

Unified interface for LLM calls and embeddings.
Supports: OpenAI GPT-5, Qwen2.5-7B-Instruct
"""

import os
from typing import List, Union
from dotenv import load_dotenv
import json

load_dotenv()

# Global token counters
INPUT_TOKENS = 0
OUTPUT_TOKENS = 0

def reset_token_usage():
    global INPUT_TOKENS, OUTPUT_TOKENS
    INPUT_TOKENS = 0
    OUTPUT_TOKENS = 0

def get_token_usage():
    return INPUT_TOKENS, OUTPUT_TOKENS


# ============ TOKENIZER ============

def get_tokenizer(model: str = "gpt"):
    """Get tokenizer for the specified model."""
    if model == "gpt":
        import tiktoken
        return tiktoken.encoding_for_model("gpt-5-mini")
    elif model == "qwen":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    else:
        raise ValueError(f"Unknown model: {model}")


# ============ LLM CALLS ============

def call_llm(prompt: str, model: str = "gpt", max_tokens: int = 4096) -> str:
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
        # Using the new Responses API capable of handling GPT-5 reasoning models
        resp = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            max_output_tokens=max_tokens,
            reasoning={"effort": "low"},
        )
        # print(json.dumps(resp.model_dump(), indent=2)[:6000])
        
        # Track usage
        if hasattr(resp, 'usage'):
            # Note: Responses API usage structure might vary
            if resp.usage:
                # Try standard OpenAI usage attributes (completion/prompt or output/input)
                in_tok = getattr(resp.usage, 'input_tokens', 0) or getattr(resp.usage, 'prompt_tokens', 0)
                out_tok = getattr(resp.usage, 'output_tokens', 0) or getattr(resp.usage, 'completion_tokens', 0)
                
                global INPUT_TOKENS, OUTPUT_TOKENS
                INPUT_TOKENS += in_tok
                OUTPUT_TOKENS += out_tok
                
                print(f"[LLM] Call Usage: In={in_tok}, Out={out_tok}")
        
        text = resp.output_text or ""
        return text.strip()
    
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

def get_embeddings(texts: Union[str, List[str]], model: str = "text-embedding-3-large") -> List[List[float]]:
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
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    
    return all_embeddings


def get_embedding_dim() -> int:
    """Return embedding dimension (3072 for text-embedding-3-large)."""
    return 3072