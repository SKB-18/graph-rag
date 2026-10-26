from typing import Dict, Any, List, Optional
from rag.graph import KnowledgeGraph
from rag.vector_rag import NaiveVectorRAG


class MultiHopComparator:
    """
    Empirical Benchmark comparing GraphRAG vs Naive Vector RAG on multi-hop reasoning.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph, vector_rag: NaiveVectorRAG):
        self.kg = knowledge_graph
        self.vector_rag = vector_rag

    def evaluate_query(
        self, 
        query: str, 
        start_entity: str, 
        target_entity: str, 
        top_k: int = 2
    ) -> Dict[str, Any]:
        """
        Runs both retrievers on a multi-hop reasoning question and compares evidence completeness.
        """
        # 1. Vector RAG Retrieval
        vector_results = self.vector_rag.query(query, top_k=top_k)
        vector_text = " ".join([r["chunk"] for r in vector_results])
        vector_found_target = target_entity.lower() in vector_text.lower()
        vector_found_start = start_entity.lower() in vector_text.lower()

        # 2. GraphRAG Retrieval
        path = self.kg.find_path(start_entity, target_entity, max_depth=4)
        subgraph = self.kg.get_subgraph(start_entity, depth=2)
        verbalized_path = self.kg.verbalize_path(path) if path else ""
        verbalized_subgraph = self.kg.verbalize_subgraph(start_entity, depth=2)

        graph_found_target = bool(path and target_entity.lower() in [n.lower() for n in path])

        return {
            "query": query,
            "start_entity": start_entity,
            "target_entity": target_entity,
            "vector_rag": {
                "retrieved_chunks": [r["chunk"] for r in vector_results],
                "found_start_entity": vector_found_start,
                "found_target_entity": vector_found_target,
                "multi_hop_connected": vector_found_start and vector_found_target
            },
            "graph_rag": {
                "path": path,
                "verbalized_path": verbalized_path,
                "hops_count": len(path) - 1 if path else None,
                "multi_hop_connected": graph_found_target
            },
            "graph_advantage": (graph_found_target and not (vector_found_start and vector_found_target))
        }
