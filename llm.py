"""
A central hub for talking to our AI models.
It handles all text generation (via local Qwen or LM Studio) and vector embeddings (via BGE), 
keeping them in memory only when necessary to save GPU VRAM.
"""

import os
import gc
import torch
from dotenv import load_dotenv

load_dotenv()

QWEN_MODEL_NAME = "./models/Qwen2.5-7B-Instruct"
BGE_MODEL_NAME = "BAAI/bge-m3"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Track token usage so we can monitor API costs (if using OpenAI)
INPUT_TOKENS = 0
OUTPUT_TOKENS = 0

# Keeping models in memory so we don't have to load them from disk every time
_qwen_model = None
_qwen_tokenizer = None
_bge_embedder = None


def reset_token_usage():
    global INPUT_TOKENS, OUTPUT_TOKENS
    INPUT_TOKENS = 0
    OUTPUT_TOKENS = 0


def get_token_usage():
    return INPUT_TOKENS, OUTPUT_TOKENS


def get_tokenizer(model="gpt"):
    """Grabs the correct tokenizer so we can accurately count how big our text chunks are."""
    if model == "gpt":
        import tiktoken
        return tiktoken.encoding_for_model("gpt-5-mini")
    
    if model == "qwen":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            QWEN_MODEL_NAME, trust_remote_code=True, cache_dir=MODELS_DIR
        )
    
    raise ValueError(f"Unknown model: {model}")


def unload_model():
    """Wipes the heavy local LLMs out of VRAM so the next script has a fresh GPU to work with."""
    global _qwen_model, _qwen_tokenizer
    
    if _qwen_model is None:
        return
    
    print("[LLM] Unloading model...")
    _qwen_model = None
    _qwen_tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    print("[LLM] GPU memory cleared.")


def call_llm(prompt, model="gpt", max_tokens=4096):
    """Sends a prompt to either our local instance or an external API and returns the generated text."""
    if model == "gpt":
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
    
    if model == "qwen":
        global _qwen_model, _qwen_tokenizer
        
        if _qwen_model is None:
            print(f"[LLM] Loading {QWEN_MODEL_NAME}...")
            os.makedirs(MODELS_DIR, exist_ok=True)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            _qwen_tokenizer = AutoTokenizer.from_pretrained(
                QWEN_MODEL_NAME, trust_remote_code=True, cache_dir=MODELS_DIR
            )
            
            _qwen_model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=MODELS_DIR,
            )
            print("[LLM] Qwen3-14B loaded (bfloat16).")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        formatted = _qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        
        inputs = _qwen_tokenizer([formatted], return_tensors="pt").to(_qwen_model.device)
        output_ids = _qwen_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.8,
            top_k=20,
        )
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        response = _qwen_tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    if model == "lmstudio":
        # Forward the request to our local LM Studio server running in the background
        from openai import OpenAI
        
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        
        try:
            resp = client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, 
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[LLM] LM Studio Error: {e}")
            return "Error generating response"
    
    raise ValueError(f"Unknown model: {model}")


def get_embeddings(texts, model="text-embedding-3-large"):
    """Converts raw text into mathematical vectors using our chosen embedding model."""
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    
    if model == "bge":
        global _bge_embedder
        
        if _bge_embedder is None:
            print(f"[EMB] Loading {BGE_MODEL_NAME}...")
            from sentence_transformers import SentenceTransformer
            _bge_embedder = SentenceTransformer(BGE_MODEL_NAME, cache_folder=MODELS_DIR)
            print("[EMB] BGE loaded.")
        
        return _bge_embedder.encode(texts, show_progress_bar=False).tolist()
    
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_embeddings = []
    
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    
    return all_embeddings


def get_embedding_dim(model="text-embedding-3-large"):
    """Returns the exact dimension size of our vectors so we can initialize the FAISS databases correctly."""
    return 1024 if model == "bge" else 3072


def preload_models(llm_model="qwen"):
    """Forces the models to load into memory right away instead of waiting for the first actual query."""
    get_embeddings(["warmup"], model="bge")
    if llm_model == "qwen":
        call_llm("warmup", model="qwen", max_tokens=1)