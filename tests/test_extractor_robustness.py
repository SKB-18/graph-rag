import pytest
from rag.extractor import EntityRelationExtractor


def test_extractor_multiple_relations():
    extractor = EntityRelationExtractor()
    text = (
        "Satya Nadella is CEO of Microsoft. "
        "Microsoft acquired GitHub. "
        "GitHub developed Copilot."
    )
    triples = extractor.extract_triples_rules(text)
    
    assert len(triples) == 3
    assert ("Satya Nadella", "is CEO of", "Microsoft") in triples
    assert ("Microsoft", "acquired", "GitHub") in triples
    assert ("GitHub", "developed", "Copilot") in triples


def test_extractor_passive_and_active():
    extractor = EntityRelationExtractor()
    text1 = "Google was founded by Larry Page."
    text2 = "Larry Page founded Google."
    
    triples1 = extractor.extract_triples_rules(text1)
    triples2 = extractor.extract_triples_rules(text2)
    
    # Both extract the core relation
    assert len(triples1) >= 1
    assert len(triples2) >= 1


def test_extractor_empty_and_noise():
    extractor = EntityRelationExtractor()
    assert extractor.extract_triples_rules("") == []
    assert extractor.extract_triples_rules("!@#$%^&*() 123456") == []
    assert extractor.extract_triples_rules("   \n\n\t   ") == []


def test_extractor_build_graph_from_documents():
    extractor = EntityRelationExtractor()
    docs = [
        "OpenAI developed GPT-4.",
        "Sam Altman is CEO of OpenAI.",
        "Microsoft partnered with OpenAI."
    ]
    kg = extractor.build_graph_from_documents(docs)
    
    assert "OpenAI" in kg.nodes
    assert "GPT-4" in kg.nodes
    assert "Sam Altman" in kg.nodes
    
    # Path traversal works over extracted graph
    path = kg.find_path("Sam Altman", "GPT-4")
    assert path == ["Sam Altman", "OpenAI", "GPT-4"]
