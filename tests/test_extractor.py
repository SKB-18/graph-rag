import pytest
from rag.extractor import EntityRelationExtractor


def test_rule_based_triple_extraction():
    extractor = EntityRelationExtractor()
    text = "Google acquired DeepMind in 2014. DeepMind developed AlphaFold."
    
    triples = extractor.extract_triples_rules(text)
    assert len(triples) >= 2
    
    # Check first triple
    assert triples[0] == ("Google", "acquired", "DeepMind")
    assert triples[1] == ("DeepMind", "developed", "AlphaFold")


def test_build_graph_from_documents():
    extractor = EntityRelationExtractor()
    docs = [
        "Steve Jobs founded Apple in Cupertino.",
        "Apple developed iPhone for mobile communication."
    ]
    
    kg = extractor.build_graph_from_documents(docs)
    path = kg.find_path("Steve Jobs", "iPhone")
    assert path == ["Steve Jobs", "Apple", "iPhone"]
