import pandas as pd
import pytest

from m5research.hierarchy import hierarchy_counts, validate_official_hierarchy


def test_hierarchy_counts_for_small_fixture():
    frame = pd.DataFrame({
        "item_id": ["i1", "i1", "i2", "i2"],
        "dept_id": ["d1", "d1", "d2", "d2"],
        "cat_id": ["c1", "c1", "c2", "c2"],
        "store_id": ["s1", "s2", "s1", "s2"],
        "state_id": ["st1", "st1", "st1", "st1"],
    })
    counts = hierarchy_counts(frame)
    assert counts["Level1"] == 1
    assert counts["Level2"] == 1
    assert counts["Level3"] == 2
    assert counts["Level10"] == 2
    assert counts["Level11"] == 2
    assert counts["Level12"] == 4


def test_official_hierarchy_rejects_wrong_counts():
    with pytest.raises(ValueError, match="do not match"):
        validate_official_hierarchy({"Level1": 1})
