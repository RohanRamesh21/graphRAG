import pytest
from pydantic import ValidationError

from graphrag.schemas import Entity, Extraction, Passage, Triple, normalize_name


def test_normalize_name_casefolds_strips_collapses_whitespace():
    assert normalize_name("  Xawery   Żuławski ") == normalize_name("xawery żuławski")
    assert normalize_name("Marie Curie") == "marie curie"


def test_entity_valid():
    e = Entity(name="Marie Curie", type="PERSON")
    assert e.name == "Marie Curie"


def test_entity_rejects_empty_name():
    with pytest.raises(ValidationError):
        Entity(name="   ", type="PERSON")


def test_triple_valid():
    t = Triple(head="Marie Curie", relation="birth_place", tail="Warsaw")
    assert t.relation == "birth_place"


def test_triple_rejects_empty_field():
    with pytest.raises(ValidationError):
        Triple(head="", relation="birth_place", tail="Warsaw")


def test_extraction_accepts_well_formed_payload():
    extraction = Extraction(
        passage_id="abc123",
        entities=[{"name": "Marie Curie", "type": "PERSON"}],
        triples=[{"head": "Marie Curie", "relation": "birth_place", "tail": "Warsaw"}],
    )
    assert len(extraction.entities) == 1
    assert extraction.error is None


def test_extraction_rejects_malformed_entity_in_payload():
    with pytest.raises(ValidationError):
        Extraction(
            passage_id="abc123",
            entities=[{"name": "Marie Curie"}],  # missing required "type"
            triples=[],
        )


def test_extraction_error_record_has_empty_lists_by_default():
    extraction = Extraction(passage_id="abc123", error="malformed JSON after 3 attempts")
    assert extraction.entities == []
    assert extraction.triples == []


def test_passage_round_trips_through_json():
    p = Passage(id="p1", title="Title", text="Some text.", source_question_ids=["q1"])
    p2 = Passage.model_validate_json(p.model_dump_json())
    assert p2 == p
