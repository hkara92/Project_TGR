"""
prompts.py

Centralized store for LLM prompts used in evaluation scripts.
Separates Multiple Choice (InfiniteChoice) from Open-Ended QA (InfiniteQA) logic.
"""

PROMPT_CHOICE = """You are a helpful assistant. You are given a multiple-choice question and evidence context.
Your task is to select the correct answer based ONLY on the evidence provided.

Instructions:
1. Read the question and options carefully.
2. Analyze the evidence to find the correct answer.
3. You must select exactly one option: "A", "B", "C", or "D".
4. Output ONLY the single letter of the correct answer. Do not include any reasoning or extra text.

Question: {question}
Evidence: {evidence}

Answer: """

PROMPT_OPEN = """You are a helpful assistant. You are given a question and evidence.
Please answer the question based on the given evidences.
The answer should be a short sentence supported by the given evidences and matches the requirements of the question.
You should not assume any information beyond the evidence.
You should only output the answer.

Question: {question}
Evidence: {evidence}

Answer: """
