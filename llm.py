import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_tokenizer(llm_choice: str = "gpt"):
    """Get tokenizer matching the LLM."""
    if llm_choice == "gpt":
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    elif llm_choice == "qwen":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    else:
        raise ValueError(f"Unknown llm_choice: {llm_choice}")


def call_llm(prompt: str, llm_choice: str = "gpt", max_tokens: int = 1024) -> str:
    """Call LLM and return response text."""
    if llm_choice == "gpt":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    
    elif llm_choice == "qwen":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    else:
        raise ValueError(f"Unknown llm_choice: {llm_choice}")


def get_embedding(text, model="text-embedding-3-large"):
    """Get embedding using OpenAI API. Handles single string or list of strings."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Ensure input is a list for the API, even if single string
    if isinstance(text, str):
        text = [text]
        
    # Replace newlines (recommended by OpenAI)
    text = [t.replace("\n", " ") for t in text]
    
    response = client.embeddings.create(input=text, model=model)
    
    # Return list of embeddings (list of lists) or single list depending on input
    embeddings = [data.embedding for data in response.data]
    
    # If using numpy elsewhere, you might want to return np.array(embeddings)
    # But for compatibility, let's return a list of lists (or single list if appropriate)
    import numpy as np
    return np.array(embeddings)