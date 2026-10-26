from typing import List, Dict, Optional, Set, Any, Tuple
import json


class KnowledgeGraph:
    """
    Production In-Memory Knowledge Graph for Multi-Hop GraphRAG.
    Maintains directed typed edges between entity nodes, supporting
    BFS pathfinding, k-hop subgraph extraction, and prompt verbalization.
    """
    def __init__(self):
        # adj: source -> {target: relation}
        self.adj: Dict[str, Dict[str, str]] = {}
        # nodes: node_id -> {"type": type, "properties": dict}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        
    def add_node(self, node_id: str, node_type: str = "entity", properties: Optional[Dict[str, Any]] = None):
        node_id_clean = node_id.strip()
        if node_id_clean not in self.adj:
            self.adj[node_id_clean] = {}
        self.nodes[node_id_clean] = {
            "type": node_type,
            "properties": properties or {}
        }
        
    def add_edge(self, source: str, target: str, relation: str):
        src = source.strip()
        tgt = target.strip()
        rel = relation.strip()
        
        if src not in self.adj:
            self.add_node(src)
        if tgt not in self.adj:
            self.add_node(tgt)
            
        self.adj[src][tgt] = rel

    def find_path(self, start: str, end: str, max_depth: int = 4) -> Optional[List[str]]:
        """
        Multi-hop pathfinding using Breadth-First Search (BFS).
        Returns the shortest sequence of node IDs connecting start to end.
        """
        start = start.strip()
        end = end.strip()
        
        if start not in self.adj or end not in self.adj:
            return None
            
        queue: List[Tuple[str, List[str]]] = [(start, [start])]
        visited: Set[str] = set()
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) - 1 > max_depth:
                continue
                
            if current.lower() == end.lower():
                return path
                
            if current not in visited:
                visited.add(current)
                for neighbor in self.adj.get(current, {}):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
                        
        return None

    def verbalize_path(self, path: List[str]) -> str:
        """Converts a node path into explicit relational reasoning propositions."""
        if not path or len(path) < 2:
            return ""
            
        statements = []
        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            rel = self.adj.get(src, {}).get(tgt, "is connected to")
            statements.append(f"({src}) -[{rel}]-> ({tgt})")
            
        return " -> ".join(statements)

    def get_subgraph(self, start: str, depth: int = 2) -> Dict[str, Any]:
        """Extracts all nodes and edges within a k-hop neighborhood."""
        start = start.strip()
        if start not in self.adj:
            return {"nodes": [], "edges": []}

        visited_nodes: Set[str] = {start}
        edges: List[Dict[str, str]] = []
        current_level: Set[str] = {start}

        for _ in range(depth):
            next_level: Set[str] = set()
            for node in current_level:
                for neighbor, relation in self.adj.get(node, {}).items():
                    edges.append({"source": node, "target": neighbor, "relation": relation})
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level
            if not current_level:
                break

        return {
            "nodes": list(visited_nodes),
            "edges": edges
        }

    def verbalize_subgraph(self, start: str, depth: int = 2) -> str:
        """Verbalizes a k-hop subgraph into natural language facts for LLM context."""
        subgraph = self.get_subgraph(start, depth=depth)
        if not subgraph["edges"]:
            return f"No relational graph data found for {start}."

        lines = [f"Relational Facts for '{start}':"]
        for edge in subgraph["edges"]:
            lines.append(f"- {edge['source']} {edge['relation']} {edge['target']}.")
        return "\n".join(lines)

    def search_entities(self, query: str) -> List[str]:
        """Returns entity names matching the query as substring or case-insensitive match."""
        q = query.lower()
        matches = [node for node in self.nodes if q in node.lower() or node.lower() in q]
        return matches

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "adj": self.adj
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        kg = cls()
        kg.nodes = data.get("nodes", {})
        kg.adj = data.get("adj", {})
        return kg
