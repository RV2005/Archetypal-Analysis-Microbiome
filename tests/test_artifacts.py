from archetypal_analysis.pipeline.artifacts import stable_hash


def test_stable_hash_ignores_mapping_order():
    assert stable_hash({"a": 1, "b": {"x": 2, "y": 3}}) == stable_hash({"b": {"y": 3, "x": 2}, "a": 1})