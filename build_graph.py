from neo4j import GraphDatabase
import os, json, time
from tqdm import tqdm

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

# ---- Main Configuration ----
# Change this variable if you want to switch to a different dataset.
DATASET_NAME = "InfiniteChoice"  # "InfiniteChoice", "InfiniteQA", or "NovelQA"

BASE_CACHE_DIR = f"./cache/{DATASET_NAME}"

CLEAR_DB = True
CREATE_CHUNK_NODES = True

# Configuration
RUN_MODE = "all"  # Options: "single", "range", "all"

# If you just want to process one specific book, put its index here.
BOOK_IDX = 0

# If you want to process a specific chunk of books, set the start and end here (the end is not included).
RANGE_START = 0
RANGE_END = 20

# For "all" mode
TOTAL_BOOKS = 58

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def idx_to_book_id(i: int):
    """We prefix the book ID with the dataset name so that books from different datasets don't crash into each other in Neo4j."""
    return f"{DATASET_NAME}_{i}"


def load_triples(cache_dir):
    """Loads the merged graph edges we extracted earlier."""
    path = os.path.join(cache_dir, "edges.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_schema(tx):
    """Sets up some strict rules in Neo4j to make sure we don't accidentally create duplicate nodes or bad data."""
    tx.run("""
    CREATE CONSTRAINT entity_unique IF NOT EXISTS
    FOR (e:Entity)
    REQUIRE (e.book_id, e.name) IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT chunk_unique IF NOT EXISTS
    FOR (c:Chunk)
    REQUIRE (c.book_id, c.chunk_id) IS UNIQUE
    """)

    tx.run("""
    CREATE INDEX rel_relation IF NOT EXISTS
    FOR ()-[r:RELATION]-()
    ON (r.relation)
    """)

    tx.run("""
    CREATE INDEX entity_book IF NOT EXISTS
    FOR (e:Entity)
    ON (e.book_id)
    """)

    tx.run("""
    CREATE INDEX chunk_book IF NOT EXISTS
    FOR (c:Chunk)
    ON (c.book_id)
    """)


def clear_db(tx):
    """A dangerous but handy function that completely wipes everything in the database."""
    tx.run("MATCH (n) DETACH DELETE n")


def normalize(triple, book_id):
    """Cleans up the raw data from our JSON files so it fits nicely into our graph schema."""
    chunk_ids = triple.get("chunk_ids", []) or []
    if not isinstance(chunk_ids, list):
        chunk_ids = [chunk_ids]

    return {
        "book_id": book_id,
        "source": (triple.get("source") or "").strip(),
        "target": (triple.get("target") or "").strip(),
        "relation": (triple.get("relation") or "").strip(),
        "weight": float(triple.get("weight", 1.0) or 1.0),
        "chunk_ids": chunk_ids,
        "support_count": len(set(chunk_ids)),
    }


def insert_triple(tx, row):
    """
    The actual Cypher query that pushes our data into Neo4j.
    MERGE means it will only create the entities if they don't already exist.
    """
    q = """
    MERGE (s:Entity {book_id: $book_id, name: $source})
    MERGE (t:Entity {book_id: $book_id, name: $target})

    MERGE (s)-[r:RELATION {book_id: $book_id, relation: $relation}]->(t)
    SET r.weight = $weight,
        r.support_count = $support_count,
        r.chunk_ids = $chunk_ids
    """
    tx.run(q, **row)

    if CREATE_CHUNK_NODES and row["chunk_ids"]:
        q2 = """
        UNWIND $chunk_ids AS cid
          MERGE (c:Chunk {book_id: $book_id, chunk_id: cid})
          MERGE (s:Entity {book_id: $book_id, name: $source})
          MERGE (t:Entity {book_id: $book_id, name: $target})
          MERGE (s)-[:MENTIONED_IN {book_id: $book_id}]->(c)
          MERGE (t)-[:MENTIONED_IN {book_id: $book_id}]->(c)
        """
        tx.run(
            q2,
            book_id=row["book_id"],
            source=row["source"],
            target=row["target"],
            chunk_ids=row["chunk_ids"],
        )


def main():
    if RUN_MODE == "all":
        book_indices = range(TOTAL_BOOKS)
    elif RUN_MODE == "range":
        book_indices = range(RANGE_START, RANGE_END)
    else:
        book_indices = [BOOK_IDX]

    print(f"Dataset: {DATASET_NAME}")
    print(f"Cache:   {BASE_CACHE_DIR}")
    print(f"Books:   {list(book_indices)}")

    with driver.session() as session:
        if CLEAR_DB:
            session.execute_write(clear_db)

        session.execute_write(setup_schema)

        build_times = []

        for i in book_indices:
            cache_dir = os.path.join(BASE_CACHE_DIR, str(i))
            edges_path = os.path.join(cache_dir, "edges.json")

            if not os.path.exists(edges_path):
                print(f"Skip book {i}: edges.json not found at {edges_path}")
                continue

            book_id = idx_to_book_id(i)
            print(f"\nIngesting book {i} -> book_id={book_id}")

            book_start = time.time()

            triples = load_triples(cache_dir)
            rows = [normalize(t, book_id) for t in triples]
            rows = [r for r in rows if r["source"] and r["target"] and r["relation"]]

            for row in tqdm(rows, desc=f"Book {i}", unit="triple"):
                session.execute_write(insert_triple, row)

            book_time = time.time() - book_start
            build_times.append(book_time)
            with open(os.path.join(cache_dir, "graph_build_time.txt"), "w") as f:
                f.write(f"{book_time:.4f}")
            print(f"  Book {i} graph done in {book_time:.2f}s")

    driver.close()

    if build_times:
        print(f"\nTotal books processed: {len(build_times)}")
        print(f"Average graph build time per book: {sum(build_times)/len(build_times):.2f}s")
        print(f"Total graph build time: {sum(build_times):.2f}s")

    print("\nDone. Open http://localhost:7474")


if __name__ == "__main__":
    main()