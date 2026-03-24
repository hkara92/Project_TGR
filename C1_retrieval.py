"""
This is our Baseline C1 Retrieval engine. It tries to use the Neo4j graph to find exact shortest-path connections 
between entities mentioned in the user's question. If it can't find anything in the graph, it falls back to a 
standard FAISS dense vector search over the summary tree.
"""

import json
import re
import copy

import numpy as np
import faiss
import spacy
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from itertools import combinations

# Constants for entity extraction
NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT"}
PRONOUN_LIKE = {"he", "she", "it", "they", "we", "i", "you", "this", "that"}


class Retriever:
    """The main retriever class that handles talking to Neo4j and our FAISS indexes."""
    
    def __init__(self, tree: Dict, neo4j_driver, book_id: str,
                 I_e2c: Dict, I_c2e: Dict, I_s2e: Dict, I_e2s: Dict,
                 faiss_index: faiss.Index, node_id_list: List[str],
                 embedder_func, nlp, **kwargs):
        # Core data
        self.tree = tree
        self.driver = neo4j_driver
        self.book_id = book_id
        
        # Dictionaries to quickly look up which entities are in which chunks/summaries
        self.I_e2c = I_e2c  
        self.I_c2e = I_c2e  
        self.I_s2e = I_s2e  
        self.I_e2s = I_e2s  
        
        # Dense retrieval
        self.faiss_index = faiss_index
        self.node_id_list = node_id_list
        self.embedder_func = embedder_func
        self.nlp = nlp
        
        # Config
        self.max_chunk_setting = kwargs.get("max_chunk_setting", 25)
        self.shortest_path_k = kwargs.get("shortest_path_k", 4)

    def extract_query_entities(self, query: str):
        """Runs the user's question through SpaCy to figure out exactly who or what they are asking about."""
        # If query contains options (starts with 'A. '), extract only the question part
        if "\nA. " in query:
            query = query.split("\nA. ")[0]

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

    def graph_filter(self, entities: list, k: int):
        """Looks at all pairs of entities from the question and asks Neo4j if they are connected within 'k' hops."""
        pairs = []
        for head, tail in combinations(entities, 2):
            length = self._get_shortest_path_length(head, tail)
            if length is not None and length <= k:
                pairs.append((head, tail))
        return pairs
    
    def _get_shortest_path_length(self, e1: str, e2: str):
        """A tiny helper to run the exact shortestPath Cypher query against Neo4j."""
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
            return None

    def index_mapping(self, entities: list):
        """Takes our found entities and figures out exactly which raw context chunks mention them."""
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
    
    def merge_keys(self, res: dict):
        """If a chunk mentions multiple entities we're looking for, we merge its keys together so we don't process it twice."""
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

    def local_retrieval(self, entities: list, k: int, allow_fallback: bool = True):
        """This tries to find graph pairs first. If it can't find pairs but allow_fallback is True, it returns chunks that just mention the individual entities."""
        if len(entities) < 2:
            return {} # If there's only one entity, we can't find a path between pairs, so force a fallback

        pairs = self.graph_filter(entities, k)
        
        if pairs:
             init_chunks = self.index_mapping(pairs)
        elif allow_fallback:
             init_chunks = self.index_mapping(entities)
        else:
             return {}
             
        return self.merge_keys(init_chunks)
    
    def dense_retrieval(self, query: str, k: int):
        """The classic vector search fallback. Embeds the question and grabs the most mathematically similar chunks from FAISS."""
        # print(f"dense_retrieval: embedding query '{query[:50]}...'")
        query_embed = np.array(self.embedder_func(query)).reshape(1, -1).astype('float32')
        # print("dense_retrieval: searching FAISS index...")
        _, indices = self.faiss_index.search(query_embed, k=k)
        
        candidates = [self.node_id_list[i] for i in indices[0] 
                      if 0 <= i < len(self.node_id_list)]
        return {"": candidates}

    def occurrence_ranking(self, candidates: list, entities: list, top_k: int):
        """Takes a list of candidate chunks and ranks them based on how often our query entities actually show up in them."""
        scores = [self._count_entity_matches(c, entities) for c in candidates]
        sorted_idx = np.argsort(scores)[::-1]
        
        filtered = [candidates[i] for i in sorted_idx if scores[i] > 0][:top_k]
        if not filtered:
            return {"": candidates[:top_k]}
        
        res = self._assign_entity_keys(filtered, entities)
        return self.merge_keys(res) if res else {"": filtered}
    
    def entityaware_filter(self, candidates: dict, entities: list, top_k: int):
        """A stricter filter that prioritizes chunks which contain multiple different target entities at once."""
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

    def _count_entity_matches(self, node_id: str, entities: list):
        """Simply checks how many of our target entities exist inside a specific node."""
        chunk_id = node_id  # IDs are now aligned (L0_chunk_X)
        
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
    
    def _assign_entity_keys(self, nodes: list, entities: list):
        """Tags nodes with the specific entities they contain so we can track them easier."""
        result = {}
        for node_id in nodes:
            chunk_id = node_id  # IDs are now aligned
            
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
    
    def _count_chunks(self, res: dict):
        return sum(len(v) for v in res.values())
    
    def format_res(self, res: dict):
        """Combines the extracted chunks into one big string that we'll eventually feed to the Generator LLM."""
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
                    print(f"Warning: Node {node_id} not found in tree")
                    continue
                
                parts.append(f"{key}: {text}" if key else text)
        return "\n\n".join(parts)

    def query(self, question: str, full_query: str = None, **kwargs):
        """
        This is the main brain of the C1 baseline. It adaptively decides whether to use the graph, 
        fallback to dense vector search, or do a mix of both depending on what it finds.
        """
        max_chunks = kwargs.get("max_chunk_setting", self.max_chunk_setting)
        k = kwargs.get("shortest_path_k", self.shortest_path_k)
        
        # Use provided full_query (Question + Options) or fallback to question
        dense_input = full_query if full_query else question
        
        # Step 1: Extract entities from Question Only
        entities = self.extract_query_entities(question)

        
        # If the question had no named entities, we can't use the graph. Go straight to vector search.
        if not entities:
            print("No entities -> Global Search")
            res = self.dense_retrieval(dense_input, max_chunks)
            return self._build_result(res, entities, "Global Search", [])
        
        # Try to find structural graph paths between the entities.
        print("Starting local retrieval...")
        local_res = self.local_retrieval(entities, k)
        count = self._count_chunks(local_res)
        history = [(k, count)]
        print(f"Local: k={k}, count={count}")
        
        # If there are no paths in the graph, we do a wide vector search and rerank the results based on entity mentions.
        if count == 0:
            print("Local=0 to Occurrence Rerank")
            # Dense retrieval gets 2x candidates to filtered by Entities
            dense = self.dense_retrieval(dense_input, max_chunks * 2)
            res = self.occurrence_ranking(dense.get("", []), entities, max_chunks)
            return self._build_result(res, entities, "Occurrence Rerank", history)
        
        # If the graph returned WAY too many chunks, we shrink the allowed hop distance to tighten the net.
        prev_res = None
        while count > max_chunks:
            prev_res = copy.deepcopy(local_res)
            k -= 1
            new_res = self.local_retrieval(entities, k, allow_fallback=False)
            
            new_count = self._count_chunks(new_res)
                
            local_res = new_res
            count = new_count
            history.append((k, count))
            print(f"Tighten: k={k}, count={count}")
        
        # We got a good amount of chunks, bundle them up and return them!
        if count > 0:
            rtype = f"Local, Loop for {len(history)-1} times"
            return self._build_result(local_res, entities, rtype, history)
        
        # If tightening the hops made us lose all our chunks, we fall back to the previous hop radius and heavily filter it.
        print("Tightening hit 0 -> EntityAware Filter")
        if prev_res:
            res = self.entityaware_filter(prev_res, entities, max_chunks)
            rtype = f"EntityAware Filter, Loop for {len(history)-1} times"
        else:
            # Fallback to Global Search if everything failed
            res = self.dense_retrieval(dense_input, max_chunks)
            rtype = "Global Search (fallback)"
        
        return self._build_result(res, entities, rtype, history)
    
    def _build_result(self, res: dict, entities: list, rtype: str, history: list):
        return {
            "chunks": self.format_res(res),
            "chunk_ids": res,
            "entities": entities,
            "retrieval_type": rtype,
            "len_chunks": self._count_chunks(res),
            "chunk_counts_history": history
        }


def load_retriever_from_cache(cache_dir: str, book_id: str, neo4j_uri: str,
                              neo4j_user: str, neo4j_password: str,
                              embedder_func, spacy_model: str = "en_core_web_lg",
                              **kwargs):
    """A convenient setup function that reads everything from the hard drive and initializes our retriever perfectly."""
    import os
    from neo4j import GraphDatabase
    
    # Load tree
    with open(os.path.join(cache_dir, "summary_tree", "nodes.json"), "r", encoding="utf-8") as f:
        tree = json.load(f)
    
    # Load indexes
    with open(os.path.join(cache_dir, "summary_tree", "I_s2e.json"), "r", encoding="utf-8") as f:
        I_s2e = json.load(f)
    with open(os.path.join(cache_dir, "summary_tree", "I_e2s.json"), "r", encoding="utf-8") as f:
        I_e2s = json.load(f)
    with open(os.path.join(cache_dir, "entities", "I_c2e.json"), "r", encoding="utf-8") as f:
        I_c2e = json.load(f)
    with open(os.path.join(cache_dir, "entities", "I_e2c.json"), "r", encoding="utf-8") as f:
        I_e2c = json.load(f)
    
    # Load FAISS
    faiss_index = faiss.read_index(os.path.join(cache_dir, "summary_tree", "tree.index"))
    with open(os.path.join(cache_dir, "summary_tree", "faiss_id_map.json"), "r", encoding="utf-8") as f:
        node_id_list = json.load(f)
    
    # Connect Neo4j
    driver = GraphDatabase.driver(
        neo4j_uri, 
        auth=(neo4j_user, neo4j_password),
        notifications_min_severity="WARNING", # Suppress INFO/PERFORMANCE
        # Or you can disable categories specifically if your driver version supports it
    )
    with driver.session() as s:
        s.run("RETURN 1")
    
    nlp = spacy.load(spacy_model)
    

    
    return Retriever(
        tree=tree, neo4j_driver=driver, book_id=book_id,
        I_e2c=I_e2c, I_c2e=I_c2e, I_s2e=I_s2e, I_e2s=I_e2s,
        faiss_index=faiss_index, node_id_list=node_id_list,
        embedder_func=embedder_func, nlp=nlp, **kwargs
    )