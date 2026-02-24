"""
prompts.py

Centralized store for LLM prompts used in evaluation scripts.
Separates Multiple Choice (InfiniteChoice) from Open-Ended QA (InfiniteQA) logic.
"""

PROMPT_CHOICE = """You are a helpful assistant. Use ONLY the evidence below to answer the question.
Evidence:
{evidence}
Question: {question}
Select the correct option. Output ONLY the letter (A, B, C, or D). No explanation.
Answer: """

PROMPT_OPEN = """You are a helpful assistant. You are given a question and evidence.
Please answer the question based on the given evidences.
The answer should be a short sentence supported by the given evidences and matches the requirements of the question.
You should not assume any information beyond the evidence.
You should only output the answer.

Question: {question}
Evidence: {evidence}

Answer: """
