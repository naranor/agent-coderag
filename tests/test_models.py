import json
from code_rag.core.models import KnowledgeUnit, UnitKind, Relation, RelationType


def test_knowledge_unit_serialization():
    """Test serialization and deserialization of KnowledgeUnit."""
    unit = KnowledgeUnit(
        id="test_id",
        kind=UnitKind.FUNCTION,
        name="test_func",
        path="test.py",
        signature="def test_func(): ...",
        summary="A test function",
        code_hash="hash123",
        tags=["test", "unit"],
        metadata={"start_line": 1, "end_line": 5},
        relations=[
            Relation(from_id="test_id", to_id="other_id", type=RelationType.CALLS)
        ],
    )

    # Serialize to dict
    data = unit.model_dump()
    assert data["id"] == "test_id"
    assert data["kind"] == "function"
    assert len(data["relations"]) == 1
    assert data["relations"][0]["type"] == "calls"

    # Serialize to JSON
    json_data = unit.model_dump_json()
    parsed = json.loads(json_data)
    assert parsed["id"] == "test_id"

    # Deserialize from dict
    unit2 = KnowledgeUnit.model_validate(data)
    assert unit2 == unit
    assert unit2.kind == UnitKind.FUNCTION


def test_relation_serialization():
    """Test serialization of Relation."""
    rel = Relation(from_id="a", to_id="b", type=RelationType.IMPORTS)
    data = rel.model_dump()
    assert data["type"] == "imports"

    rel2 = Relation.model_validate(data)
    assert rel2 == rel
