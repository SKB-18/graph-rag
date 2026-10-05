# GraphRAG

This is Repo 4 of the AI Project Portfolio Roadmap.

It implements a Knowledge Graph tailored for multi-hop Retrieval-Augmented Generation (GraphRAG). While traditional vector databases struggle with multi-hop reasoning (e.g. "Who is the CEO of the company that acquired X?"), GraphRAG excels by traversing explicit edges in a graph structure.

## Architecture
- **`rag/graph.py`**: A lightweight, in-memory Knowledge Graph that allows arbitrary nodes and edges, using Breadth-First Search (BFS) to retrieve exact relation paths between entities to supply as context to the LLM.

## Testing
```bash
python -m pytest tests/
```
