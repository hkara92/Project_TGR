# Project TGR: Tree-Graph RAG

This project implements a Knowledge Graph-enhanced Retrieval Augmented Generation (KG-RAG) system that combines hierarchical summary trees (RAPTOR-like), knowledge graph construction, and advanced retrieval strategies  to improve information retrieval and question answering over long documents (InfiniteBench).

## Project Overview

The system integrates knowledge graph construction with hierarchical document processing, creating a parent child relationship between them, to create a robust RAG pipeline. The main components are:

*   **Hierarchical Indexing**: Builds a tree of summaries to capture high-level context.
*   **Knowledge Graph Construction**: Extracts entities by NLP and relations by LLM to push to Neo4j.
*   **Retrieval**: Implements adaptive retrieval strategies (Graph Traversal, Vector Search, Entity Filtering).

## Dataset Setup

Please refer to [DATA_SETUP.md](./DATA_SETUP.md) for detailed instructions on how to download and set up the **InfiniteBench** and **NovelQA** datasets.

## Key Components

### `run_indexing.py`
This is the main orchestration file that:
*   Loads and preprocesses the dataset (InfiniteBench)
*   Builds the Summary Tree (RAPTOR-like structure)
*   Extracts Entities (SpaCy) and Relations (LLM)
*   Builds FAISS Vector Indexes
*   Saves all artifacts to a local cache

To run:
```bash
python run_indexing.py
```

### `build_graph.py`
This script loads the processed data into Neo4j:
*   Creates Entity and Chunk nodes
*   Creates RELATION edges
*   Ensures isolation between different Books using `book_id`

To run:
```bash
python build_graph.py
```

### `C1_run_eval.py` (Step 1: Baseline)
Executes the baseline retrieval evaluation (Standard E2Retrieval with enhanced graph and tree).
```bash
python C1_run_eval.py
```
- **Output**: `cache/InfiniteChoice/<book_id>/predictions.json`

### `C2_run_eval.py` (Step 2: Region-Restricted)
Executes the advanced retrieval evaluation using hierarchical region constraints.
```bash
python C2_run_eval.py
```
- **Output**: `cache/InfiniteChoice/<book_id>/predictions_C2.json`

### `calculate_metrics.py`
This script calculates the final evaluation metrics based on the predictions generated:
*   Aggregates results from all processed books
*   Reports **Accuracy (EM)** for Multiple Choice tasks
*   Reports **ROUGE-L** for Free Generation tasks

To run:
```bash
python calculate_metrics.py
```

## Directory Structure

*   `run_indexing.py`: Main pipeline orchestration
*   `run_eval.py`: Retrieval and Evaluation pipeline
*   `calculate_metrics.py`: Metrics calculation (EM / ROUGE)
*   `preprocessing.py`: Text cleaning and chunking modules
*   `summary_tree.py`: Recursive tree building logic (RAPTOR-like)
*   `entity_extraction.py`: SpaCy-based entity extraction
*   `relation_extraction_llm.py`: LLM-based relation extraction
*   `build_graph.py`: Neo4j loader script
*   `C1_retrieval.py`: Core retrieval logic (Graph + Vector)
*   `llm.py`: Unified LLM interface (OpenAI)
*   `data/`: Contains dataset files
*   `cache/`: Stores generated artifacts (Entities, Trees, Indexes)

## Neo4j Integration

To use Neo4j for this project, run the Docker container:

```bash
docker-compose up -d
```

Or manually:
```bash
docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/testpassword \
    neo4j:latest
```

## Requirements

Main dependencies:
*   `neo4j`
*   `openai`
*   `spacy` (`en_core_web_lg`)
*   `faiss-cpu`
*   `numpy`
*   `python-dotenv`
*   `rouge` (for metrics)

Install via:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```
