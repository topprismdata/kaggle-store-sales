from __future__ import annotations

EXPECTED_LEVEL_COUNTS = {
    "Level1": 1,
    "Level2": 3,
    "Level3": 10,
    "Level4": 3,
    "Level5": 7,
    "Level6": 9,
    "Level7": 21,
    "Level8": 30,
    "Level9": 70,
    "Level10": 3049,
    "Level11": 9147,
    "Level12": 30490,
}
EXPECTED_TOTAL_SERIES = 42840


def hierarchy_counts(frame) -> dict[str, int]:
    """Count the 12 official M5 hierarchy levels from a bottom-level metadata frame."""
    required = {"item_id", "dept_id", "cat_id", "store_id", "state_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing hierarchy columns: {sorted(missing)}")

    counts = {
        "Level1": 1,
        "Level2": int(frame["state_id"].nunique()),
        "Level3": int(frame["store_id"].nunique()),
        "Level4": int(frame["cat_id"].nunique()),
        "Level5": int(frame["dept_id"].nunique()),
        "Level6": int(frame[["state_id", "cat_id"]].drop_duplicates().shape[0]),
        "Level7": int(frame[["state_id", "dept_id"]].drop_duplicates().shape[0]),
        "Level8": int(frame[["store_id", "cat_id"]].drop_duplicates().shape[0]),
        "Level9": int(frame[["store_id", "dept_id"]].drop_duplicates().shape[0]),
        "Level10": int(frame["item_id"].nunique()),
        "Level11": int(frame[["item_id", "state_id"]].drop_duplicates().shape[0]),
        "Level12": int(frame[["item_id", "store_id"]].drop_duplicates().shape[0]),
    }
    return counts


def validate_official_hierarchy(counts: dict[str, int]) -> None:
    if counts != EXPECTED_LEVEL_COUNTS:
        mismatches = {
            level: {"expected": EXPECTED_LEVEL_COUNTS[level], "actual": counts.get(level)}
            for level in EXPECTED_LEVEL_COUNTS
            if counts.get(level) != EXPECTED_LEVEL_COUNTS[level]
        }
        raise ValueError(f"M5 hierarchy counts do not match official contract: {mismatches}")
    if sum(counts.values()) != EXPECTED_TOTAL_SERIES:
        raise ValueError("hierarchy total does not equal 42,840")
