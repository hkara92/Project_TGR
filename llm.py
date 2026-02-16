"""
llm.py

Handles LLM calls (GPT-5 or Qwen 2.5 14B) and embedding generation
(OpenAI or BAAI BGE). Models are cached after first load and freed
with unload_model().
"""

import os
import gc
import json
import torch
from dotenv import load_dotenv

load_dotenv()

# where local models get downloaded
QWEN_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# token counters (only used for OpenAI billing tracking)
INPUT_TOKENS = 0
OUTPUT_TOKENS = 0

def reset_token_usage():
    global INPUT_TOKENS, OUTPUT_TOKENS
    INPUT_TOKENS = 0
    OUTPUT_TOKENS = 0

def get_token_usage():
    return INPUT_TOKENS, OUTPUT_TOKENS


# ---- tokenizer ----

def get_tokenizer(model="gpt"):
    """Return a tokenizer for text chunking."""
    if model == "gpt":
        import tiktoken
        return tiktoken.encoding_for_model("gpt-5-mini")

    elif model == "qwen":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            QWEN_MODEL_NAME, trust_remote_code=True, cache_dir=MODELS_DIR
        )

    else:
        raise ValueError(f"Unknown model: {model}")


# ---- local model cache ----

_loaded_model = None
_loaded_tokenizer = None
_loaded_embedder = None

def _load_qwen():
    """Load the Qwen model and tokenizer into GPU. Skips if already loaded."""
    global _loaded_model, _loaded_tokenizer

    if _loaded_model is not None:
        return _loaded_model, _loaded_tokenizer

    print(f"[LLM] Loading {QWEN_MODEL_NAME} ...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _loaded_tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_NAME, trust_remote_code=True, cache_dir=MODELS_DIR
    )
    _loaded_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=MODELS_DIR,
    )
    print("[LLM] Model loaded.")
    return _loaded_model, _loaded_tokenizer


def _load_bge():
    """Load the BGE embedding model. Skips if already loaded."""
    global _loaded_embedder

    if _loaded_embedder is not None:
        return _loaded_embedder

    print(f"[EMB] Loading {BGE_MODEL_NAME} ...")
    from sentence_transformers import SentenceTransformer

    _loaded_embedder = SentenceTransformer(BGE_MODEL_NAME, cache_folder=MODELS_DIR)
    print("[EMB] BGE loaded.")
    return _loaded_embedder


def unload_model():
    """Remove the LLM from memory and free GPU."""
    global _loaded_model, _loaded_tokenizer

    if _loaded_model is None:
        return

    print("[LLM] Unloading model ...")
    _loaded_model = None
    _loaded_tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    print("[LLM] GPU memory cleared.")


# ---- llm calls ----

def call_llm(prompt, model="gpt", max_tokens=4096):
    """
    Send a prompt and get a text response.
    model="gpt"  uses OpenAI API, model="qwen" runs locally.
    """
    if model == "gpt":
        return _call_gpt(prompt, max_tokens)
    elif model == "qwen":
        return _call_qwen(prompt, max_tokens)
    else:
        raise ValueError(f"Unknown model: {model}")


def _call_gpt(prompt, max_tokens):
    """OpenAI GPT-5-mini API call."""
    global INPUT_TOKENS, OUTPUT_TOKENS

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    resp = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        max_output_tokens=max_tokens,
        reasoning={"effort": "low"},
    )

    if resp.usage:
        in_tok = getattr(resp.usage, "input_tokens", 0) or getattr(resp.usage, "prompt_tokens", 0)
        out_tok = getattr(resp.usage, "output_tokens", 0) or getattr(resp.usage, "completion_tokens", 0)
        INPUT_TOKENS += in_tok
        OUTPUT_TOKENS += out_tok
        print(f"[LLM] Tokens: in={in_tok}, out={out_tok}")

    return (resp.output_text or "").strip()


def _call_qwen(prompt, max_tokens):
    """Local Qwen 2.5 14B inference."""
    model_obj, tokenizer = _load_qwen()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([formatted], return_tensors="pt").to(model_obj.device)
    output_ids = model_obj.generate(**inputs, max_new_tokens=max_tokens)

    # only decode the newly generated tokens
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ---- embeddings ----

def get_embeddings(texts, model="text-embedding-3-large"):
    """
    Return a list of embedding vectors.
    model="text-embedding-3-large" calls OpenAI API.
    model="bge" runs BAAI BGE locally (1024 dim).
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []

    if model == "bge":
        embedder = _load_bge()
        return embedder.encode(texts, show_progress_bar=False).tolist()

    # openai path
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_embeddings = []

    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


def get_embedding_dim(model="text-embedding-3-large"):
    """Return the vector size for the chosen embedding model."""
    if model == "bge":
        return 1024
    return 3072