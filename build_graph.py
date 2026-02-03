from neo4j import GraphDatabase
import os, json
from tqdm import tqdm

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

BASE_CACHE_DIR = "./cache/InfiniteChoice"

CLEAR_DB = False
CREATE_CHUNK_NODES = True

# Configuration
RUN_MODE = "range"  # Options: "single", "range", "all"

# For "single" mode
BOOK_IDX = 0  

# For "range" mode (python range style: start inclusive, end exclusive)
RANGE_START = 1
RANGE_END = 10 # Processes 1 to 9

# For "all" mode
TOTAL_BOOKS = 58

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def idx_to_book_id(i: int) -> str:
    return f"InfiniteChoice_{i}"


def load_triples(cache_dir):
    path = os.path.join(cache_dir, "edges.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_schema(tx):
    # Unique per book: John in book_0 != John in book_5
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

    # Optional: makes filtering by relation faster
    tx.run("""
    CREATE INDEX rel_relation IF NOT EXISTS
    FOR ()-[r:RELATION]-()
    ON (r.relation)
    """)

    # Optional: makes filtering by book_id faster
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
    tx.run("MATCH (n) DETACH DELETE n")


def normalize(triple, book_id):
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

    # Single DB session (Neo4j Community)
    with driver.session() as session:
        if CLEAR_DB:
            session.execute_write(clear_db)

        session.execute_write(setup_schema)

        for i in book_indices:
            cache_dir = os.path.join(BASE_CACHE_DIR, str(i))
            edges_path = os.path.join(cache_dir, "edges.json")

            if not os.path.exists(edges_path):
                print(f"Skip book {i}: edges.json not found at {edges_path}")
                continue

            book_id = idx_to_book_id(i)
            print(f"\nIngesting book {i} -> book_id={book_id}")

            triples = load_triples(cache_dir)
            rows = [normalize(t, book_id) for t in triples]
            rows = [r for r in rows if r["source"] and r["target"] and r["relation"]]

            for row in tqdm(rows, desc=f"Book {i}", unit="triple"):
                session.execute_write(insert_triple, row)

    driver.close()
    print("\nDone. Open http://localhost:7474")


if __name__ == "__main__":
    main()
