"""Neo4j AuraDB knowledge graph: documents -> chunks -> entities -> relations."""
from neo4j import GraphDatabase
from backend.config import settings


class Neo4jKG:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        )

    def close(self):
        self.driver.close()

    def ensure_constraints(self):
        with self.driver.session(database=settings.NEO4J_DATABASE) as s:
            s.run("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
            s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
            s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")

    def add_document(self, doc_id: str, doc_name: str, num_pages: int):
        with self.driver.session(database=settings.NEO4J_DATABASE) as s:
            s.run(
                "MERGE (d:Document {doc_id: $doc_id}) "
                "SET d.name = $doc_name, d.num_pages = $num_pages",
                doc_id=doc_id, doc_name=doc_name, num_pages=num_pages,
            )

    def add_chunk_with_entities(self, doc_id: str, chunk: dict, entities: list[str], triples: list[list[str]]):
        with self.driver.session(database=settings.NEO4J_DATABASE) as s:
            s.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.text = $text, c.page = $page, c.doc_name = $doc_name
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                doc_id=doc_id, chunk_id=chunk["chunk_id"], text=chunk["text"],
                page=chunk["page"], doc_name=chunk["doc_name"],
            )
            for ent in entities:
                s.run(
                    """
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (e:Entity {name: $name})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    chunk_id=chunk["chunk_id"], name=ent,
                )
            for triple in triples:
                if len(triple) != 3:
                    continue
                head, rel, tail = triple
                rel_type = "".join(ch if ch.isalnum() else "_" for ch in str(rel).upper())[:40] or "RELATED_TO"
                s.run(
                    f"""
                    MERGE (h:Entity {{name: $head}})
                    MERGE (t:Entity {{name: $tail}})
                    MERGE (h)-[:{rel_type}]->(t)
                    """,
                    head=head, tail=tail,
                )

    def get_chunks_for_entities(self, entity_names: list[str], top_k: int) -> list[dict]:
        """1-2 hop expansion: entity -> chunks mentioning it or connected entities' chunks."""
        if not entity_names:
            return []
        with self.driver.session(database=settings.NEO4J_DATABASE) as s:
            result = s.run(
                """
                UNWIND $names AS name
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower(name) OR toLower(name) CONTAINS toLower(e.name)
                OPTIONAL MATCH (e)<-[:MENTIONS]-(c1:Chunk)
                OPTIONAL MATCH (e)--(related:Entity)<-[:MENTIONS]-(c2:Chunk)
                WITH collect(DISTINCT c1) + collect(DISTINCT c2) AS chunks
                UNWIND chunks AS c
                WITH DISTINCT c
                WHERE c IS NOT NULL
                RETURN c.chunk_id AS chunk_id, c.text AS text, c.page AS page, c.doc_name AS doc_name
                LIMIT $limit
                """,
                names=entity_names, limit=top_k,
            )
            return [dict(r) for r in result]

    def count_entities(self) -> int:
        with self.driver.session(database=settings.NEO4J_DATABASE) as s:
            r = s.run("MATCH (e:Entity) RETURN count(e) AS c").single()
            return r["c"] if r else 0


kg = Neo4jKG()
