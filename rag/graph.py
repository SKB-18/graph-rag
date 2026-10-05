from typing import List, Dict, Optional, Set

class KnowledgeGraph:
    def __init__(self):
        # adjacency list: node -> dict of {neighbor: relation_type}
        self.adj: Dict[str, Dict[str, str]] = {}
        # metadata: node -> type
        self.nodes: Dict[str, str] = {}
        
    def add_node(self, node_id: str, node_type: str):
        if node_id not in self.adj:
            self.adj[node_id] = {}
        self.nodes[node_id] = node_type
        
    def add_edge(self, source: str, target: str, relation: str):
        if source not in self.adj:
            self.adj[source] = {}
        if target not in self.adj:
            self.adj[target] = {}
            
        self.adj[source][target] = relation
        
    def find_path(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """
        Multi-hop retrieval: Finds a path between start and end nodes using BFS.
        """
        if start not in self.adj or end not in self.adj:
            return None
            
        queue = [(start, [start])]
        visited: Set[str] = set()
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) - 1 > max_depth:
                continue
                
            if current == end:
                return path
                
            if current not in visited:
                visited.add(current)
                for neighbor in self.adj.get(current, {}):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
                        
        return None
