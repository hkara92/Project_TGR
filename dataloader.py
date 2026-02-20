import os
import json
from typing import Dict, Any


def load_novelqa(data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load NovelQA dataset.       
    Returns:
        List of book dictionaries with unified format
    """
    books_path = os.path.join(data_path, "Books")
    qa_path = os.path.join(data_path, "Data")
    
    # Load all book texts
    
    book_texts = {}
    
    for category in os.listdir(books_path):
        category_path = os.path.join(books_path, category)
        if not os.path.isdir(category_path):
            continue
            
        for filename in os.listdir(category_path):
            if not filename.endswith(".txt"):
                continue
            
            # Extract book_id: "B00.txt" -> "00", "B45.txt" -> "45"
            book_id = filename[1:-4]  # Remove 'B' prefix and '.txt' suffix
            
            filepath = os.path.join(category_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                book_texts[book_id] = f.read()
    
    # Load all QA pairs
    book_qa = {}
    
    for category in os.listdir(qa_path):
        category_path = os.path.join(qa_path, category)
        if not os.path.isdir(category_path):
            continue
            
        for filename in os.listdir(category_path):
            if not filename.endswith(".json"):
                continue
            
            # Extract book_id: "B00.json" -> "00", "B45.json" -> "45"
            book_id = filename[1:-5]  # Remove 'B' prefix and '.json' suffix
            
            filepath = os.path.join(category_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
            
            book_qa[book_id] = qa_data
    
    # Combine into unified format (dict keyed by book_id)
    results = {}
    
    for book_id in sorted(book_texts.keys(), key=lambda x: int(x)):
        if book_id not in book_qa:
            print(f"Warning: No QA found for book {book_id}, skipping")
            continue
        
        #  Format QA pairs
        qa_pairs = []
        for qa_id, qa_item in book_qa[book_id].items():
            # Extract options as list [A_text, B_text, C_text, D_text]
            # Use .get() to handle cases where some options might be missing
            opt = qa_item.get("Options", {})
            options = [
                opt.get("A", ""),
                opt.get("B", ""),
                opt.get("C", ""),
                opt.get("D", "")
            ]
            
            qa_pairs.append({
                "question": qa_item["Question"],
                "options": options,
                "answer": qa_item["Gold"]  # "A", "B", "C", or "D"
            })
        
        results[book_id] = {
            "book_text": book_texts[book_id],
            "qa_pairs": qa_pairs
        }
    
    print(f"Loaded {len(results)} books from NovelQA")
    return results


def load_infinite_choice(data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load InfiniteChoice (InfiniteBench longbook_choice_eng) dataset.   
    Returns:
        List of book dictionaries with unified format
    """
    # Group questions by context (book text)
    # Multiple questions can share the same context
    context_to_data = {}
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
            
            context = item["context"]
            question = item["input"]
            options = item["options"]  # List of 4 options
            answer_text = item["answer"]  # Answer is the text, not the letter
            
            # Convert answer text to letter (A, B, C, D)
            answer_letter = None
            for i, opt in enumerate(options):
                if opt == answer_text:
                    answer_letter = ["A", "B", "C", "D"][i]
                    break
            
            if answer_letter is None:
                # Sometimes answer might be in a list
                if isinstance(answer_text, list) and len(answer_text) > 0:
                    for i, opt in enumerate(options):
                        if opt == answer_text[0]:
                            answer_letter = ["A", "B", "C", "D"][i]
                            break
            
            if answer_letter is None:
                print(f"Warning: Could not match answer '{answer_text}' to options, skipping")
                continue
            
            # Group by context
            if context not in context_to_data:
                context_to_data[context] = []
            
            # Augment question with options for retrieval context
            formatted_options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            augmented_question = f"{question}\n{formatted_options}"

            context_to_data[context].append({
                "question": augmented_question,  # Now includes options
                "options": options,
                "answer": answer_letter
            })
    
    # Convert to dict format keyed by book_id
    results = {}
    for i, (context, qa_pairs) in enumerate(context_to_data.items()):
        book_id = str(i)
        results[book_id] = {
            "book_text": context,
            "qa_pairs": qa_pairs
        }
    
    print(f"Loaded {len(results)} books from InfiniteChoice")
    return results


def load_infinite_qa(data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load InfiniteQA (InfiniteBench longbook_qa_eng) dataset.
    This is an open-ended QA task (no multiple-choice options).
    Returns:
        Dict keyed by book_id with unified format
    """
    context_to_data = {}

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue

            context = item["context"]
            question = item["input"]
            # answer is a list of strings; join them
            raw_answer = item["answer"]
            if isinstance(raw_answer, list):
                answer = "; ".join(raw_answer)
            else:
                answer = str(raw_answer)

            if context not in context_to_data:
                context_to_data[context] = []

            context_to_data[context].append({
                "question": question,
                "options": [],       # open-ended, no options
                "answer": answer
            })

    results = {}
    for i, (context, qa_pairs) in enumerate(context_to_data.items()):
        book_id = str(i)
        results[book_id] = {
            "book_text": context,
            "qa_pairs": qa_pairs
        }

    print(f"Loaded {len(results)} books from InfiniteQA")
    return results


def load_dataset(dataset_name: str, data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Unified dataset loader.
    List of book dictionaries with unified format
    """
    if dataset_name == "NovelQA":
        return load_novelqa(data_path)
    elif dataset_name == "InfiniteChoice":
        return load_infinite_choice(data_path)
    elif dataset_name == "InfiniteQA":
        return load_infinite_qa(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'NovelQA', 'InfiniteChoice', or 'InfiniteQA'")


