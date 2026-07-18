from graphrag.graph.neo4j_client import sanitize_relation_type


def test_sanitize_relation_type_basic():
    assert sanitize_relation_type("birth_place") == "BIRTH_PLACE"


def test_sanitize_relation_type_handles_spaces_and_hyphens():
    assert sanitize_relation_type("publication date") == "PUBLICATION_DATE"
    assert sanitize_relation_type("co-founder") == "CO_FOUNDER"


def test_sanitize_relation_type_strips_illegal_characters():
    assert sanitize_relation_type("director (film)!") == "DIRECTOR_FILM"


def test_sanitize_relation_type_handles_leading_digit():
    assert sanitize_relation_type("1st_spouse").startswith("REL_")


def test_sanitize_relation_type_falls_back_when_empty_after_sanitizing():
    assert sanitize_relation_type("???") == "RELATED_TO"


def test_sanitize_relation_type_is_bounded_length():
    assert len(sanitize_relation_type("x" * 500)) <= 100
