import os
from dotenv import load_dotenv

# Load environment variables from .env file
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