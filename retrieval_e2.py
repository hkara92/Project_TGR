"""
E²GraphRAG-style Retriever
==========================
Implements E²GraphRAG retrieval strategy with Neo4j graph, FAISS index, and custom tree structure.

Retrieval Modes:
1. Global Search: Dense retrieval when no entities found
2. Local Search: Graph-based entity pair retrieval  
3. Occurrence Rerank: Dense + entity filtering when local returns 0
4. EntityAware Filter: When iterative tightening hits 0
"""

import json
import re
import copy
import logging
import numpy as np
import faiss
import spacy
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from itertools import combinations

logger = logging.getLogger(__name__)

# Constants for entity extraction
NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT"}
PRONOUN_LIKE = {"he", "she", "it", "they", "we", "i", "you", "this", "that"}


class Retriever:
    """E²GraphRAG-style retriever with Neo4j graph and FAISS dense retrieval."""
    
    def __init__(self, tree: Dict, neo4j_driver, book_id: str,
                 I_e2c: Dict, I_c2e: Dict, I_s2e: Dict, I_e2s: Dict,
                 faiss_index: faiss.Index, node_id_list: List[str],
                 embedder_func, nlp, **kwargs):
        # Core data
        self.tree = tree
        self.driver = neo4j_driver
        self.book_id = book_id
        
        # Indexes
        self.I_e2c = I_e2c  # entity -> [chunk_ids]
        self.I_c2e = I_c2e  # chunk_id -> [entities]
        self.I_s2e = I_s2e  # summary_node -> [entities]
        self.I_e2s = I_e2s  # entity -> [summary_nodes]
        
        # Dense retrieval
        self.faiss_index = faiss_index
        self.node_id_list = node_id_list
        self.embedder_func = embedder_func
        self.nlp = nlp
        
        # Config
        self.max_chunk_setting = kwargs.get("max_chunk_setting", 25)
        self.shortest_path_k = kwargs.get("shortest_path_k", 4)
        
        logger.info(f"Retriever initialized: {len(tree)} nodes, {len(I_e2c)} entities")

    # -------------------------------------------------------------------------
    # ENTITY EXTRACTION
    # -------------------------------------------------------------------------
    
    def extract_query_entities(self, query: str) -> List[str]:
        """Extract and canonicalize entities from query using SpaCy NER."""
        doc = self.nlp(query)
        entities = set()
        
        for ent in doc.ents:
            if ent.label_ not in NER_LABELS:
                continue
            # Canonicalize: lowercase, normalize whitespace, strip quotes
            e = ent.text.lower().strip()
            e = re.sub(r"\s+", " ", e)
            e = re.sub(r"^[\"'""'']+|[\"'""'']+$", "", e)
            
            if len(e) >= 3 and e not in PRONOUN_LIKE:
                entities.add(e)
        
        return list(entities)

    # -------------------------------------------------------------------------
    # GRAPH OPERATIONS (Neo4j)
    # -------------------------------------------------------------------------
    
    def graph_filter(self, entities: List[str], k: int) -> List[Tuple[str, str]]:
        """Find entity pairs with shortest path <= k hops in Neo4j."""
        pairs = []
        for head, tail in combinations(entities, 2):
            length = self._get_shortest_path_length(head, tail)
            if length is not None and length <= k:
                pairs.append((head, tail))
        return pairs
    
    def _get_shortest_path_length(self, e1: str, e2: str) -> Optional[int]:
        """Query Neo4j for shortest path length between entities."""
        query = """
        MATCH (a:Entity {name: $name1, book_id: $book_id}),
              (b:Entity {name: $name2, book_id: $book_id}),
              path = shortestPath((a)-[:RELATION*..10]-(b))
        RETURN length(path) as path_length
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, name1=e1, name2=e2, book_id=self.book_id)
                record = result.single()
                return record["path_length"] if record else None
        except Exception as e:
            logger.debug(f"No path {e1}-{e2}: {e}")
            return None

    # -------------------------------------------------------------------------
    # INDEX MAPPING & KEY MERGING
    # -------------------------------------------------------------------------
    
    def index_mapping(self, entities: list) -> Dict[str, List[str]]:
        """Map entities/pairs to chunk IDs. Pairs use intersection."""
        chunk_ids = {}
        
        for entity in entities:
            if isinstance(entity, str):
                if entity in self.I_e2c:
                    chunk_ids[entity] = self.I_e2c[entity].copy()
            elif isinstance(entity, tuple):
                # Intersection for pairs
                key = "_".join(sorted(entity))
                chunks_set = None
                for e in entity:
                    if e in self.I_e2c:
                        e_chunks = set(self.I_e2c[e])
                        chunks_set = e_chunks if chunks_set is None else chunks_set & e_chunks
                if chunks_set:
                    chunk_ids[key] = sorted(list(chunks_set))
        
        return chunk_ids
    
    def merge_keys(self, res: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge chunks appearing under multiple entity keys."""
        chunks_to_keys = defaultdict(set)
        for key, chunks in res.items():
            for chunk in chunks:
                chunks_to_keys[chunk].add(key)
        
        merged = {}
        for chunk, keys in chunks_to_keys.items():
            if len(keys) > 1:
                all_entities = set()
                for k in keys:
                    all_entities.update(k.split("_"))
                new_key = "_".join(sorted(all_entities))
            else:
                new_key = keys.pop()
            merged.setdefault(new_key, []).append(chunk)
        
        return merged

    # -------------------------------------------------------------------------
    # RETRIEVAL METHODS
    # -------------------------------------------------------------------------
    
    def local_retrieval(self, entities: List[str], k: int) -> Dict[str, List[str]]:
        """Graph-based retrieval: find pairs -> map to chunks -> merge keys."""
        pairs = self.graph_filter(entities, k)
        init_chunks = self.index_mapping(pairs) if pairs else self.index_mapping(entities)
        return self.merge_keys(init_chunks)
    
    def dense_retrieval(self, query: str, k: int) -> Dict[str, List[str]]:
        """FAISS dense retrieval over all tree nodes."""
        query_embed = np.array(self.embedder_func(query)).reshape(1, -1).astype('float32')
        _, indices = self.faiss_index.search(query_embed, k=k)
        
        candidates = [self.node_id_list[i] for i in indices[0] 
                      if 0 <= i < len(self.node_id_list)]
        return {"": candidates}

    # -------------------------------------------------------------------------
    # RANKING METHODS
    # -------------------------------------------------------------------------
    
    def occurrence_ranking(self, candidates: List[str], entities: List[str], 
                          top_k: int) -> Dict[str, List[str]]:
        """Rank candidates by entity occurrence count."""
        scores = [self._count_entity_matches(c, entities) for c in candidates]
        sorted_idx = np.argsort(scores)[::-1]
        
        filtered = [candidates[i] for i in sorted_idx if scores[i] > 0][:top_k]
        if not filtered:
            return {"": candidates[:top_k]}
        
        res = self._assign_entity_keys(filtered, entities)
        return self.merge_keys(res) if res else {"": filtered}
    
    def entityaware_filter(self, candidates: Dict[str, List[str]], 
                          entities: List[str], top_k: int) -> Dict[str, List[str]]:
        """Filter by entity count in key and chunk coverage."""
        info = []
        for key, nodes in candidates.items():
            for node in nodes:
                info.append({
                    "node": node, "key": key,
                    "key_count": len(key.split("_")) if key else 0,
                    "entity_count": self._count_entity_matches(node, entities)
                })
        
        info.sort(key=lambda x: (x["key_count"], x["entity_count"]), reverse=True)
        
        result = {}
        for item in info[:top_k]:
            key = item["key"] or "unknown"
            result.setdefault(key, []).append(item["node"])
        return self.merge_keys(result)

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    
    def _node_id_to_chunk_id(self, node_id: str) -> str:
        """Normalize node ID to chunk_X format for I_c2e/I_e2c lookup."""
        if node_id.startswith("L0_"):
            return node_id[3:]  # "L0_chunk_5" → "chunk_5"
        return node_id  # Already "chunk_5" or summary node
    
    def _count_entity_matches(self, node_id: str, entities: List[str]) -> int:
        """Count query entities present in a node. Handles both chunk_X and L0_chunk_X formats."""
        # Normalize: convert L0_chunk_X → chunk_X for I_c2e lookup
        if node_id.startswith("L0_"):
            chunk_id = node_id[3:]  # "L0_chunk_5" → "chunk_5"
        else:
            chunk_id = node_id  # Already "chunk_5" format
        
        # Try chunk lookup (I_c2e uses chunk_X format)
        if chunk_id in self.I_c2e:
            node_entities = set(self.I_c2e[chunk_id])
        # Try summary lookup (I_s2e uses L0_chunk_X, L1_CX format)
        elif node_id in self.I_s2e:
            node_entities = set(self.I_s2e[node_id])
        # Try summary with L0_ prefix
        elif f"L0_{chunk_id}" in self.I_s2e:
            node_entities = set(self.I_s2e[f"L0_{chunk_id}"])
        else:
            return 0
        
        return len(set(entities) & node_entities)
    
    def _assign_entity_keys(self, nodes: List[str], entities: List[str]) -> Dict[str, List[str]]:
        """Assign entity keys to nodes based on contained entities. Handles both ID formats."""
        result = {}
        for node_id in nodes:
            # Normalize: convert L0_chunk_X → chunk_X for I_c2e lookup
            if node_id.startswith("L0_"):
                chunk_id = node_id[3:]
            else:
                chunk_id = node_id
            
            # Try chunk lookup
            if chunk_id in self.I_c2e:
                node_ents = set(self.I_c2e[chunk_id])
            # Try summary lookup
            elif node_id in self.I_s2e:
                node_ents = set(self.I_s2e[node_id])
            elif f"L0_{chunk_id}" in self.I_s2e:
                node_ents = set(self.I_s2e[f"L0_{chunk_id}"])
            else:
                result.setdefault("", []).append(node_id)
                continue
            
            matching = [e for e in entities if e in node_ents]
            key = "_".join(sorted(matching)) if matching else ""
            result.setdefault(key, []).append(node_id)
        
        return result
    
    def _count_chunks(self, res: Dict[str, List[str]]) -> int:
        return sum(len(v) for v in res.values())
    
    def format_res(self, res: Dict[str, List[str]]) -> str:
        """Format results for LLM prompt. Handles both chunk_X and L0_chunk_X formats."""
        parts = []
        for key, nodes in res.items():
            for node_id in nodes:
                # Try direct lookup (for L0_chunk_X, L1_CX from dense retrieval)
                if node_id in self.tree:
                    text = self.tree[node_id]["text"]
                # Try with L0_ prefix (for chunk_X from local retrieval / I_e2c)
                elif f"L0_{node_id}" in self.tree:
                    text = self.tree[f"L0_{node_id}"]["text"]
                else:
                    logger.warning(f"Node {node_id} not found in tree")
                    continue
                
                parts.append(f"{key}: {text}" if key else text)
        return "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # MAIN QUERY METHOD
    # -------------------------------------------------------------------------
    
    def query(self, question: str, full_query: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Main retrieval implementing E²GraphRAG adaptive strategy.
        
        Args:
            question: The question text (used for Entity Extraction).
            full_query: content for dense retrieval (e.g. Question + Options). 
                        If None, defaults to `question`.
        """
        max_chunks = kwargs.get("max_chunk_setting", self.max_chunk_setting)
        k = kwargs.get("shortest_path_k", self.shortest_path_k)
        
        # Use provided full_query (Question + Options) or fallback to question
        dense_input = full_query if full_query else question
        
        # Step 1: Extract entities from Question Only
        # (We no longer need strict newline splitting if the caller separates them)
        entities = self.extract_query_entities(question)
        logger.info(f"Entities: {entities}")
        
        # Step 2: No entities -> Global dense retrieval (Using Dense Input)
        if not entities:
            logger.info("No entities -> Global Search")
            res = self.dense_retrieval(dense_input, max_chunks)
            return self._build_result(res, entities, "Global Search", [])
        
        # Step 3: Local retrieval (Graph-based, using Entities)
        local_res = self.local_retrieval(entities, k)
        count = self._count_chunks(local_res)
        history = [(k, count)]
        logger.info(f"Local: k={k}, count={count}")
        
        # Step 4: Zero results -> Occurrence rerank
        if count == 0:
            logger.info("Local=0 -> Occurrence Rerank")
            # Dense retrieval gets 2x candidates -> filtered by Entities
            dense = self.dense_retrieval(dense_input, max_chunks * 2)
            res = self.occurrence_ranking(dense.get("", []), entities, max_chunks)
            return self._build_result(res, entities, "Occurrence Rerank", history)
        
        # Step 5: Too many -> Iterative tightening
        prev_res = None
        while count > max_chunks and k > 1:
            prev_res = copy.deepcopy(local_res)
            k -= 1
            local_res = self.local_retrieval(entities, k)
            count = self._count_chunks(local_res)
            history.append((k, count))
            logger.info(f"Tighten: k={k}, count={count}")
        
        # Step 6: Return result
        if count > 0:
            rtype = f"Local, Loop for {len(history)-1} times"
            return self._build_result(local_res, entities, rtype, history)
        
        # Tightening hit 0 -> EntityAware filter
        logger.info("Tightening hit 0 -> EntityAware Filter")
        if prev_res:
            res = self.entityaware_filter(prev_res, entities, max_chunks)
            rtype = f"EntityAware Filter, Loop for {len(history)-1} times"
        else:
            # Fallback to Global Search if everything failed
            res = self.dense_retrieval(dense_input, max_chunks)
            rtype = "Global Search (fallback)"
        
        return self._build_result(res, entities, rtype, history)
    
    def _build_result(self, res: Dict, entities: List, rtype: str, history: List) -> Dict:
        return {
            "chunks": self.format_res(res),
            "chunk_ids": res,
            "entities": entities,
            "retrieval_type": rtype,
            "len_chunks": self._count_chunks(res),
            "chunk_counts_history": history
        }


# =============================================================================
# LOADER FUNCTION
# =============================================================================

def load_retriever_from_cache(cache_dir: str, book_id: str, neo4j_uri: str,
                              neo4j_user: str, neo4j_password: str,
                              embedder_func, spacy_model: str = "en_core_web_lg",
                              **kwargs) -> Retriever:
    """Load retriever with all artifacts from cache directory."""
    import os
    from neo4j import GraphDatabase
    
    # Load tree
    with open(os.path.join(cache_dir, "summary_tree", "nodes.json"), "r") as f:
        tree = json.load(f)
    
    # Load indexes
    with open(os.path.join(cache_dir, "summary_tree", "I_s2e.json"), "r") as f:
        I_s2e = json.load(f)
    with open(os.path.join(cache_dir, "summary_tree", "I_e2s.json"), "r") as f:
        I_e2s = json.load(f)
    with open(os.path.join(cache_dir, "entities", "I_c2e.json"), "r") as f:
        I_c2e = json.load(f)
    with open(os.path.join(cache_dir, "entities", "I_e2c.json"), "r") as f:
        I_e2c = json.load(f)
    
    # Load FAISS
    faiss_index = faiss.read_index(os.path.join(cache_dir, "summary_tree", "tree.index"))
    with open(os.path.join(cache_dir, "summary_tree", "faiss_id_map.json"), "r") as f:
        node_id_list = json.load(f)
    
    # Connect Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as s:
        s.run("RETURN 1")
    
    nlp = spacy.load(spacy_model)
    
    logger.info(f"Loaded: {len(tree)} nodes, {len(I_e2c)} entities, {faiss_index.ntotal} vectors")
    
    return Retriever(
        tree=tree, neo4j_driver=driver, book_id=book_id,
        I_e2c=I_e2c, I_c2e=I_c2e, I_s2e=I_s2e, I_e2s=I_e2s,
        faiss_index=faiss_index, node_id_list=node_id_list,
        embedder_func=embedder_func, nlp=nlp, **kwargs
    )