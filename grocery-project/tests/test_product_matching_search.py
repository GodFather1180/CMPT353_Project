import pandas as pd

from product_matching import find_best_matches


def test_finds_correct_candidate():
    query = pd.Series(["Old El Paso Salsa Hot 650ml"])
    candidates = pd.Series(
        ["Unrelated Product", "Old El Paso Salsa Hot 650 ml", "Another Unrelated Product"]
    )
    result = find_best_matches(query, candidates)
    assert result["match_index"].iloc[0] == 1
    assert result["score"].iloc[0] > 0.9


def test_returns_nan_below_threshold():
    query = pd.Series(["Completely Different Item"])
    candidates = pd.Series(["Old El Paso Salsa Hot 650ml", "Kraft Peanut Butter Crunchy 500g"])
    result = find_best_matches(query, candidates)
    assert pd.isna(result["match_index"].iloc[0])


def test_respects_qualifier_veto():
    query = pd.Series(["Kraft Peanut Butter Crunchy 500g"])
    candidates = pd.Series(["Kraft Peanut Butter Smooth 500g"])
    result = find_best_matches(query, candidates)
    assert pd.isna(result["match_index"].iloc[0])


def test_keeps_candidate_original_index():
    query = pd.Series(["Old El Paso Salsa Hot 650ml"])
    candidates = pd.Series(["Old El Paso Salsa Hot 650 ml"], index=[99])
    result = find_best_matches(query, candidates)
    assert result["match_index"].iloc[0] == 99


def test_preserves_query_index():
    query = pd.Series(["Old El Paso Salsa Hot 650ml"], index=[7])
    candidates = pd.Series(["Old El Paso Salsa Hot 650 ml"])
    result = find_best_matches(query, candidates)
    assert list(result.index) == [7]


def test_exclude_self_index_does_not_match_itself():
    # same product listed twice under different ids, plus one real other
    # product. searching a catalog against itself should never return a
    # row's own index as its own best match.
    names = pd.Series(
        ["Old El Paso Salsa Hot 650ml", "Old El Paso Salsa Hot 650 ml", "Kraft Peanut Butter 500g"],
        index=[10, 11, 12],
    )
    result = find_best_matches(names, names, exclude_self_index=True)

    assert result.loc[10, "match_index"] == 11
    assert result.loc[11, "match_index"] == 10
    assert pd.isna(result.loc[12, "match_index"])
