# Project TGR: Tree-Graph RAG

This project implements a Knowledge Graph-enhanced Retrieval Augmented Generation (KG-RAG) system that combines hierarchical summary trees (RAPTOR-style), knowledge graph construction, and advanced retrieval strategies to improve information retrieval and question answering over long documents (InfiniteBench).

## Project Overview

The system integrates knowledge graph construction with hierarchical document processing, creating a parent-child relationship between them, to create a robust RAG pipeline. The main components are:

*   **Hierarchical Indexing**: Builds a RAPTOR-style tree of summaries using UMAP/GMM clustering.
*   **Knowledge Graph Construction**: Extracts entities via SpaCy NER and relations via LLM, then loads everything into Neo4j.
*   **Retrieval**: Implements two adaptive retrieval strategies (C1 Baseline and C2 Region-Restricted).
*   **Evaluation**: Measures Accuracy (EM), ROUGE-L, Macro F1, and RAGAS metrics.

## Dataset Setup

Please refer to [DATASET_SETUP.md](./DATASET_SETUP.md) for detailed instructions on how to download and set up the **InfiniteBench** and **NovelQA** datasets.

## How to Run (Step-by-Step)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Step 2: Run the Offline Indexing Pipeline

This processes all books in the dataset: chunking the text, building the summary tree, extracting entities and relations, and creating FAISS indexes. All outputs are cached to `./cache/`.

```bash
python run_indexing.py
```

### Step 3: Start Neo4j (Docker)

The knowledge graph needs a Neo4j instance. Start one using Docker Compose:

```bash
docker-compose up -d
```

This creates a Neo4j container accessible at:
- **Browser**: http://localhost:7474
- **Bolt**: bolt://localhost:7687
- **Credentials**: `neo4j` / `testpassword`

### Step 4: Build the Knowledge Graph

This reads the extracted relations from the cache and uploads them into Neo4j as Entity nodes and RELATION edges.

```bash
python build_graph.py
```

### Step 5: Run Evaluation

#### C1 Baseline Evaluation
Uses the standard E2GraphRAG adaptive strategy.

```bash
python C1_run_eval.py
```
- **Output**: `cache/<dataset>/<book_id>/predictions.json`

#### C2 Region-Restricted Evaluation
Uses the advanced region-restricted retrieval with tree pruning, MMR diversity, cross-encoder reranking, and Lost-in-the-Middle reordering.

```bash
python C2_run_eval.py
```
- **Output**: `cache/<dataset>/<book_id>/predictions_C2.json`

**Switching C2 Retrieval Strategies**: `C2_run_eval.py` supports two retrieval strategies controlled by the `RETRIEVAL_METHOD` variable at the top of the file:
- `"shortest_path"` — Uses `C2_retrieval.py` (batched shortestPath queries in Neo4j). This is the primary method.
- `"hop"` — Uses `C2_retrieval_hop.py` (fixed 1-hop or 2-hop graph traversal within the region).

### Step 6: Calculate Metrics

Aggregates predictions from all books and prints accuracy, F1, confusion matrices, and timing data.

```bash
python calculate_metrics.py
```

- Reports **Accuracy (EM)** and **Macro F1** for InfiniteChoice (multiple-choice)
- Reports **ROUGE-L** for InfiniteQA (open-ended)

### Step 7: RAGAS Evaluation

Runs RAGAS framework metrics (Faithfulness, Answer Relevancy, Context Recall, Context Precision, Answer Correctness) using a local LM Studio server as the judge LLM.

```bash
python run_ragas_eval.py
```

**Requires**: LM Studio running locally at `http://localhost:1234` with a loaded model and embedding model.

## Key Scripts

| Script | Purpose |
|---|---|
| `run_indexing.py` | Main indexing pipeline (chunking, summary tree, entities, relations, FAISS) |
| `build_graph.py` | Loads extracted relations into Neo4j |
| `C1_retrieval.py` | Baseline retrieval engine (graph + FAISS adaptive strategy) |
| `C1_run_eval.py` | Runs C1 retrieval evaluation on the dataset |
| `C2_retrieval.py` | Region-restricted retrieval (shortest-path method) |
| `C2_retrieval_hop.py` | Region-restricted retrieval (1-hop / 2-hop traversal variant) |
| `C2_run_eval.py` | Runs C2 retrieval evaluation (supports both strategies) |
| `calculate_metrics.py` | Calculates Accuracy, ROUGE-L, F1, confusion matrix, and timing |
| `run_ragas_eval.py` | RAGAS framework evaluation via LM Studio |
| `dataloader.py` | Dataset loading for NovelQA, InfiniteChoice, InfiniteQA |
| `preprocessing.py` | Text normalization and chunking (recursive character splitting) |
| `summary_tree.py` | RAPTOR-style hierarchical summary tree (UMAP + GMM clustering) |
| `entity_extraction.py` | SpaCy-based NER and entity canonicalization |
| `relation_extraction_llm.py` | LLM-based relation extraction and edge merging |
| `build_indexes.py` | FAISS index and inverted index construction |
| `llm.py` | Unified interface for GPT, Qwen (local), and LM Studio |
| `prompts.py` | LLM prompt templates for multiple-choice and open-ended QA |

## Directory Structure

```
Project_TGR/
├── run_indexing.py          # Step 2: Offline indexing pipeline
├── build_graph.py           # Step 4: Neo4j graph loader
├── C1_retrieval.py          # Baseline retrieval logic
├── C1_run_eval.py           # Step 5a: Baseline evaluation
├── C2_retrieval.py          # Region-restricted retrieval (shortest-path)
├── C2_retrieval_hop.py      # Region-restricted retrieval (hop variant)
├── C2_run_eval.py           # Step 5b: Region-restricted evaluation
├── calculate_metrics.py     # Step 6: Metrics aggregation
├── run_ragas_eval.py        # Step 7: RAGAS evaluation
├── dataloader.py            # Dataset loading utilities
├── preprocessing.py         # Text cleaning and chunking
├── summary_tree.py          # RAPTOR-style tree construction
├── entity_extraction.py     # SpaCy NER
├── relation_extraction_llm.py  # LLM relation extraction
├── build_indexes.py         # FAISS index builder
├── llm.py                   # LLM interface (GPT / Qwen / LM Studio)
├── prompts.py               # Prompt templates
├── docker-compose.yml       # Neo4j Docker setup
├── requirements.txt         # Python dependencies
├── data/                    # Dataset files
├── cache/                   # Generated artifacts (trees, indexes, predictions)
├── models/                  # Local model weights (Qwen, BGE)
└── neo4j_data/              # Neo4j persistent storage
```

## Models Used

| Component | Model | Notes |
|---|---|---|
| **Text Generation** | Qwen2.5-7B-Instruct (local) | Loaded via HuggingFace Transformers |
| **Text Generation** | GPT-5-mini (API) | Via OpenAI API |
| **Text Generation** | LM Studio (local) | Any model loaded in LM Studio |
| **Embeddings** | BAAI/bge-m3 | 1024-dim vectors, loaded via SentenceTransformers |
| **NER** | SpaCy en_core_web_lg | Entity extraction |
| **Cross-Encoder** | ms-marco-MiniLM-L-6-v2 | Reranking in C2 pipeline |

## Requirements

Install all Python dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```
