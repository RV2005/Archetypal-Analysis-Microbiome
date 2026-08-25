import pandas as pd

from archetypal_analysis.data.conversion import merge_compendium_metadata


def test_compendium_metadata_merge():
    idx = pd.Index(["p1_r1", "p2_r2"], name="sample")
    sample = pd.DataFrame({"project": ["p1", "p2"], "srr": ["r1", "r2"], "region": ["sample1", "sample2"], "age": [10, 20]})
    projects = pd.DataFrame({"region": ["project1", "project2"], "sample_type": ["stool", "skin"]}, index=["p1", "p2"])
    tags = pd.DataFrame({"project": ["p1", "p2"], "srr": ["r1", "r2"], "tag": ["diet", "diet"], "value": ["A", "B"]})
    merged = merge_compendium_metadata(idx, sample, projects, tags)
    assert merged.loc["p1_r1", "region"] == "project1"
    assert merged.loc["p2_r2", "sample_type"] == "skin"
    assert merged.loc["p1_r1", "diet"] == "A"
    assert merged.loc["p2_r2", "age"] == 20