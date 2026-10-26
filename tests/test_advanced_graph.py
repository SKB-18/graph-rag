import pytest
from rag.graph import KnowledgeGraph


def test_cyclic_graph_bfs_termination():
    kg = KnowledgeGraph()
    # A -> B -> C -> A (cycle)
    kg.add_edge("A", "B", "points_to")
    kg.add_edge("B", "C", "points_to")
    kg.add_edge("C", "A", "points_to")
    kg.add_edge("C", "Target", "points_to")
    
    path = kg.find_path("A", "Target", max_depth=5)
    assert path == ["A", "B", "C", "Target"]


def test_multiple_paths_finds_shortest():
    kg = KnowledgeGraph()
    # Path 1: Start -> Intermediate1 -> Intermediate2 -> Goal (3 hops)
    kg.add_edge("Start", "Intermediate1", "rel")
    kg.add_edge("Intermediate1", "Intermediate2", "rel")
    kg.add_edge("Intermediate2", "Goal", "rel")
    
    # Path 2: Start -> DirectShortcut -> Goal (2 hops)
    kg.add_edge("Start", "DirectShortcut", "rel")
    kg.add_edge("DirectShortcut", "Goal", "rel")
    
    path = kg.find_path("Start", "Goal", max_depth=5)
    # BFS guarantees shortest path (length 3 nodes = 2 hops)
    assert len(path) == 3
    assert path == ["Start", "DirectShortcut", "Goal"]


def test_empty_graph():
    kg = KnowledgeGraph()
    assert kg.find_path("X", "Y") is None
    assert kg.get_subgraph("X", depth=2)["nodes"] == []
    assert kg.search_entities("anything") == []


def test_self_loop_edge():
    kg = KnowledgeGraph()
    kg.add_edge("Agent", "Agent", "self_reflects")
    assert "Agent" in kg.nodes
    subgraph = kg.get_subgraph("Agent", depth=1)
    assert "Agent" in subgraph["nodes"]


def test_subgraph_depth_zero():
    kg = KnowledgeGraph()
    kg.add_edge("Center", "Neighbor1", "rel")
    kg.add_edge("Center", "Neighbor2", "rel")
    
    subgraph = kg.get_subgraph("Center", depth=0)
    assert subgraph["nodes"] == ["Center"]
    assert len(subgraph["edges"]) == 0


def test_large_chain_traversal():
    kg = KnowledgeGraph()
    chain_length = 6
    for i in range(chain_length - 1):
        kg.add_edge(f"Node_{i}", f"Node_{i+1}", "connects_to")
        
    path = kg.find_path("Node_0", "Node_5", max_depth=10)
    assert path == [f"Node_{i}" for i in range(chain_length)]
    
    # Exceeding max_depth returns None
    short_path = kg.find_path("Node_0", "Node_5", max_depth=3)
    assert short_path is None


def test_verbalize_edge_cases():
    kg = KnowledgeGraph()
    assert kg.verbalize_path([]) == ""
    assert kg.verbalize_path(["SingleNode"]) == ""


def test_case_insensitive_entity_search():
    kg = KnowledgeGraph()
    kg.add_node("Artificial Intelligence", "Field")
    kg.add_node("Deep Learning", "Subfield")
    
    matches = kg.search_entities("inTeLLiGenCe")
    assert "Artificial Intelligence" in matches
    assert "Deep Learning" not in matches
