import pytest
from rag.vector_rag import NaiveVectorRAG


def test_vector_rag_indexing_and_query():
    rag = NaiveVectorRAG()
    docs = [
        "Python is a popular language for artificial intelligence and machine learning.",
        "SpaceX builds reusable rockets for space exploration missions.",
        "Quantum computing uses qubits and superposition to solve complex problems."
    ]
    rag.index(docs)
    
    # Query for machine learning
    results = rag.query("machine learning with Python", top_k=1)
    assert len(results) == 1
    assert "Python" in results[0]["chunk"]
    assert results[0]["score"] > 0.0


def test_vector_rag_empty_handling():
    rag = NaiveVectorRAG()
    rag.index([])
    assert rag.query("test") == []
