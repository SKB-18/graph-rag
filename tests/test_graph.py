import pytest
from rag.graph import KnowledgeGraph


def test_knowledge_graph_basic():
    kg = KnowledgeGraph()
    kg.add_node("Alice", "Person")
    kg.add_node("Bob", "Person")
    kg.add_node("Paris", "Location")
    
    kg.add_edge("Alice", "Bob", "KNOWS")
    kg.add_edge("Alice", "Paris", "VISITED")
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


def test_knowledge_graph_verbalize_path():
    kg = KnowledgeGraph()
    kg.add_edge("Google", "DeepMind", "acquired")
    kg.add_edge("DeepMind", "AlphaFold", "developed")
    
    path = kg.find_path("Google", "AlphaFold")
    assert path == ["Google", "DeepMind", "AlphaFold"]
    
    verbalized = kg.verbalize_path(path)
    assert "(Google) -[acquired]-> (DeepMind)" in verbalized
    assert "(DeepMind) -[developed]-> (AlphaFold)" in verbalized


def test_knowledge_graph_subgraph_and_verbalize():
    kg = KnowledgeGraph()
    kg.add_edge("Stanford", "Silicon Valley", "located in")
    kg.add_edge("Larry Page", "Stanford", "studied at")
    kg.add_edge("Larry Page", "Google", "founded")

    subgraph = kg.get_subgraph("Larry Page", depth=2)
    assert len(subgraph["nodes"]) >= 3
    assert len(subgraph["edges"]) >= 2
    
    text = kg.verbalize_subgraph("Larry Page", depth=1)
    assert "Larry Page" in text
    assert "founded Google" in text or "studied at Stanford" in text


def test_knowledge_graph_search_and_serialization():
    kg = KnowledgeGraph()
    kg.add_node("Elon Musk", "Person")
    kg.add_edge("Elon Musk", "SpaceX", "founded")
    
    matches = kg.search_entities("musk")
    assert "Elon Musk" in matches
    
    # Test JSON serialization
    data = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(data)
    assert "Elon Musk" in kg2.nodes
    assert kg2.find_path("Elon Musk", "SpaceX") == ["Elon Musk", "SpaceX"]
