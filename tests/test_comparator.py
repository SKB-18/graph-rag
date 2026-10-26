import pytest
from rag.graph import KnowledgeGraph
from rag.vector_rag import NaiveVectorRAG
from rag.comparator import MultiHopComparator


def test_multihop_comparator_graph_advantage():
    # Setup Knowledge Graph with 2-hop connection
    kg = KnowledgeGraph()
    kg.add_edge("Wernher von Braun", "Saturn V", "designed")
    kg.add_edge("Saturn V", "Apollo 11", "launched")

    # Setup Vector RAG with isolated text chunks
    docs = [
        "Wernher von Braun designed Saturn V for heavy lift propulsion.",
        "Saturn V launched Apollo 11 to the moon in 1969.",
        "The lunar rover was used in later missions."
    ]
    vector_rag = NaiveVectorRAG()
    vector_rag.index(docs)

    comparator = MultiHopComparator(knowledge_graph=kg, vector_rag=vector_rag)
    
    query = "Find the connection between Wernher von Braun and Apollo 11."
    eval_result = comparator.evaluate_query(
        query=query,
        start_entity="Wernher von Braun",
        target_entity="Apollo 11",
        top_k=1  # Only top 1 retrieved by vector search
    )
    
    # GraphRAG succeeds in finding the full path across hops
    assert eval_result["graph_rag"]["multi_hop_connected"] is True
    assert eval_result["graph_rag"]["hops_count"] == 2
    assert "(Wernher von Braun) -[designed]-> (Saturn V)" in eval_result["graph_rag"]["verbalized_path"]
    assert eval_result["graph_advantage"] is True
