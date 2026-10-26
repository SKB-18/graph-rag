import re
from typing import List, Tuple, Dict, Any, Optional
from rag.graph import KnowledgeGraph


class EntityRelationExtractor:
    """
    Extracts Subject-Relation-Object (SPO) triples from text documents
    and builds an explicit KnowledgeGraph.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
        # Subject: Capitalized proper noun phrase (supports alphanumeric and hyphens)
        subj = r"\b([A-Z][a-zA-Z0-9\-_]*(?:\s+[A-Z][a-zA-Z0-9\-_]*)*)"
        # Object: Capitalized proper noun or brand identifier (supports alphanumeric and hyphens, e.g. iPhone, GPT-4)
        obj = r"([A-Za-z0-9\-_]+(?:\s+[A-Za-z0-9\-_]+)*)"

        self.patterns = [
            (rf"{subj}\s+(?:was founded by|is founded by)\s+{obj}", "founded by", False),
            (rf"{subj}\s+(?:founded)\s+{obj}", "founded", False),
            (rf"{subj}\s+(?:acquired|purchased|bought)\s+{obj}", "acquired", False),
            (rf"{subj}\s+(?:is located in|located in|based in|headquartered in)\s+{obj}", "located in", False),
            (rf"{subj}\s+(?:developed|created|built|designed)\s+{obj}", "developed", False),
            (rf"{subj}\s+(?:graduated from|studied at|attended)\s+{obj}", "studied at", False),
            (rf"{subj}\s+(?:is the CEO of|is CEO of|became CEO of|leads)\s+{obj}", "is CEO of", False),
            (rf"{subj}\s+(?:is a subsidiary of|owned by)\s+{obj}", "subsidiary of", False),
            (rf"{subj}\s+(?:partnered with|collaborated with)\s+{obj}", "partnered with", False),
        ]

    def _clean_entity(self, name: str) -> str:
        # Strip trailing prepositions and prepositional phrases (e.g. "in Cupertino", "for mobile...")
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", name.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?:in|for|at|on|by|to|with|of)\b.*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def extract_triples_rules(self, text: str) -> List[Tuple[str, str, str]]:
        """Extracts entity triples using regex relational heuristics."""
        triples = []
        sentences = re.split(r"[.\n;]", text)
        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue
            for pattern, relation, reverse in self.patterns:
                matches = re.findall(pattern, sent_clean)
                for src, tgt in matches:
                    src_clean = self._clean_entity(src)
                    tgt_clean = self._clean_entity(tgt)
                    if src_clean and tgt_clean and src_clean.lower() != tgt_clean.lower():
                        if reverse:
                            triples.append((tgt_clean, relation, src_clean))
                        else:
                            triples.append((src_clean, relation, tgt_clean))
        return triples

    def build_graph_from_documents(self, documents: List[str]) -> KnowledgeGraph:
        """Constructs a KnowledgeGraph by extracting relational triples across documents."""
        kg = KnowledgeGraph()
        for doc in documents:
            triples = self.extract_triples_rules(doc)
            for src, rel, tgt in triples:
                kg.add_edge(src, tgt, rel)
        return kg
