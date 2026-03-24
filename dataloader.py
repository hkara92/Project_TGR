import os
import json
from typing import Dict, Any


def load_novelqa(data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads the NovelQA dataset from the specified path.
    Returns a dictionary of books, where each book has its text and QA pairs.
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
            
            # Get the book ID from the filename (e.g., changing "B00.txt" to "00")
            book_id = filename[1:-4]  # Drop the "B" at the start and ".txt" at the end
            
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
            
            # Get the book ID from the QA file (e.g., changing "B00.json" to "00")
            book_id = filename[1:-5]  # Drop the "B" at the start and ".json" at the end
            
            filepath = os.path.join(category_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
            
            book_qa[book_id] = qa_data
    
    # Put everything together into a single dictionary organized by book ID
    results = {}
    
    for book_id in sorted(book_texts.keys(), key=lambda x: int(x)):
        if book_id not in book_qa:
            print(f"Warning: No QA found for book {book_id}, skipping")
            continue
        
        # Organize the questions and answers
        qa_pairs = []
        for qa_id, qa_item in book_qa[book_id].items():
            # Put the four multiple choice options into a simple list
            # We use .get() here just in case a question is missing an option
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
                "answer": qa_item["Gold"]  # The correct answer letter
            })
        
        results[book_id] = {
            "book_text": book_texts[book_id],
            "qa_pairs": qa_pairs
        }
    
    print(f"Loaded {len(results)} books from NovelQA")
    return results


def load_infinite_choice(data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads the longbook_choice_eng task from the InfiniteBench dataset.
    Returns a dictionary of books, each containing the book text and its QA pairs.
    """
    # Because one book can have multiple questions, we group all questions that share the same book text together.
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
            options = item["options"]  # The list of choices
            answer_text = item["answer"]  # The actual answer text (not just "A" or "B")
            
            # We need to figure out which letter (A, B, C, or D) corresponds to the answer text
            answer_letter = None
            for i, opt in enumerate(options):
                if opt == answer_text:
                    answer_letter = ["A", "B", "C", "D"][i]
                    break
            
            if answer_letter is None:
                # In some edge cases, the dataset puts the answer inside a list, so we handle that here
                if isinstance(answer_text, list) and len(answer_text) > 0:
                    for i, opt in enumerate(options):
                        if opt == answer_text[0]:
                            answer_letter = ["A", "B", "C", "D"][i]
                            break
            
            if answer_letter is None:
                print(f"Warning: Could not match answer '{answer_text}' to options, skipping")
                continue
            
            # Add the question to the right book
            if context not in context_to_data:
                context_to_data[context] = []
            
            # Attach the multiple choice options directly to the question text to help with retrieval
            formatted_options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            augmented_question = f"{question}\n{formatted_options}"

            context_to_data[context].append({
                "question": augmented_question,  # We use the combined question and options here
                "options": options,
                "answer": answer_letter
            })
    
    # Package everything up into a dictionary using the book index as the ID
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
    Loads the longbook_qa_eng task from the InfiniteBench dataset.
    Unlike the choice tasks, this is open-ended QA so there are no multiple choice options.
    Returns a dictionary of books with their text and QA pairs.
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
            
            # The answers come as a list of strings, so we just join them together with a semicolon
            raw_answer = item["answer"]
            if isinstance(raw_answer, list):
                answer = "; ".join(raw_answer)
            else:
                answer = str(raw_answer)

            if context not in context_to_data:
                context_to_data[context] = []

            context_to_data[context].append({
                "question": question,
                "options": [],       # Keep this empty since it's an open-ended question
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
    A simple wrapper that calls the correct loading function based on the dataset name string.
    """
    if dataset_name == "NovelQA":
        return load_novelqa(data_path)
    elif dataset_name == "InfiniteChoice":
        return load_infinite_choice(data_path)
    elif dataset_name == "InfiniteQA":
        return load_infinite_qa(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'NovelQA', 'InfiniteChoice', or 'InfiniteQA'")
