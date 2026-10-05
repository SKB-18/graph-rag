import pytest
from rag.graph import KnowledgeGraph

def test_knowledge_graph_basic():
    kg = KnowledgeGraph()
    
    # Add nodes
    kg.add_node("Alice", "Person")
    kg.add_node("Bob", "Person")
    kg.add_node("Paris", "Location")
    
    # Add edges
    kg.add_edge("Alice", "Bob", "KNOWS")
    kg.add_edge("Alice", "Paris", "VISITED")
    
    # Multi-hop query: Who knows someone who visited Paris?
    # Alice visited Paris. Bob knows Alice. (Assuming bidirectional KNOWS for simplicity, or we just do a direct traversal)
    # Let's say we want to find paths from "Bob" to "Paris"
    
    kg.add_edge("Bob", "Alice", "KNOWS")
    
    path = kg.find_path("Bob", "Paris", max_depth=2)
    assert path is not None
    assert path == ["Bob", "Alice", "Paris"]

def test_knowledge_graph_no_path():
    kg = KnowledgeGraph()
    kg.add_node("A", "Node")
    kg.add_node("B", "Node")
    kg.add_node("C", "Node")
    
    kg.add_edge("A", "B", "CONNECTS")
    
    path = kg.find_path("A", "C")
    assert path is None
