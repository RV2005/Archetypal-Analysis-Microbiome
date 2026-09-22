from archetypal_analysis.pipeline.artifacts import stable_hash


def test_stable_hash_ignores_mapping_order():
    assert stable_hash({"a": 1, "b": {"x": 2, "y": 3}}) == stable_hash({"b": {"y": 3, "x": 2}, "a": 1})

def test_reuse_checks_every_recorded_output(tmp_path):
    from archetypal_analysis.pipeline.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "results")
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("ok")
    store.mark_success("load", "abc", [artifact])
    assert store.is_reusable("load", "abc", [])
    artifact.unlink()
    assert not store.is_reusable("load", "abc", [])
