"""Decide whether two product name strings are the same product.

Combines TF-IDF cosine similarity on the name (size stripped out) with a
check against a list of qualifier words like crunchy/smooth. Similarity
alone isn't enough, "Crunchy 500g" and "Smooth 500g" score high since
they share almost every word, but they're different products.
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from package_parser import UNIT_GROUP

_SIZE_RE = re.compile(rf"\d+(?:\.\d+)?\s*{UNIT_GROUP}\b", re.IGNORECASE)

# flips a match to a non-match if only one name has the word
_QUALIFIER_WORDS = [
    "smooth", "crunchy",
    "original", "spicy", "mild", "hot",
    "light", "regular", "diet",
    "salted", "unsalted",
    "sweetened", "unsweetened",
    "decaf",
]


def _strip_size(names: pd.Series) -> pd.Series:
    return names.astype("string").str.replace(_SIZE_RE, "", regex=True)


def _has_qualifier_conflict(name_a: pd.Series, name_b: pd.Series) -> pd.Series:
    lower_a = name_a.astype("string").str.lower()
    lower_b = name_b.astype("string").str.lower()
    conflict = pd.Series(False, index=name_a.index)
    # loop is over 14 words, not the data
    for word in _QUALIFIER_WORDS:
        has_a = lower_a.str.contains(word, regex=False)
        has_b = lower_b.str.contains(word, regex=False)
        conflict = conflict | (has_a != has_b)
    return conflict


def is_match(name_a: pd.Series, name_b: pd.Series, threshold: float = 0.9) -> pd.Series:
    cleaned_a = _strip_size(name_a)
    cleaned_b = _strip_size(name_b)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(pd.concat([cleaned_a, cleaned_b]))
    matrix_a = vectorizer.transform(cleaned_a)
    matrix_b = vectorizer.transform(cleaned_b)

    # rows are already normalized, so the dot product is the cosine similarity
    similarity = np.asarray(matrix_a.multiply(matrix_b).sum(axis=1)).flatten()
    similar_enough = pd.Series(similarity >= threshold, index=name_a.index)

    return similar_enough & ~_has_qualifier_conflict(name_a, name_b)


def find_best_matches(
    query_names: pd.Series,
    candidate_names: pd.Series,
    threshold: float = 0.9,
    exclude_self_index: bool = False,
) -> pd.DataFrame:
    cleaned_query = _strip_size(query_names)
    cleaned_candidates = _strip_size(candidate_names)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(pd.concat([cleaned_query, cleaned_candidates]))
    query_matrix = vectorizer.transform(cleaned_query)
    candidate_matrix = vectorizer.transform(cleaned_candidates)

    # nearest neighbor search instead of a full similarity matrix, that
    # matrix would be too big to hold for two large vendor catalogs.
    # asking for a second neighbor when excluding self, since a query's
    # own row would otherwise be its own closest match.
    n_neighbors = 2 if exclude_self_index else 1
    neighbors = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
    neighbors.fit(candidate_matrix)
    distance, position = neighbors.kneighbors(query_matrix)

    if exclude_self_index:
        query_position = np.arange(len(query_names))
        is_self = candidate_names.index.to_numpy()[position] == query_names.index.to_numpy()[:, None]
        # first neighbor that isn't the query's own row
        pick = np.where(is_self[:, 0], 1, 0)
        similarity = 1 - distance[query_position, pick]
        position = position[query_position, pick]
    else:
        similarity = 1 - distance[:, 0]
        position = position[:, 0]

    matched_names = candidate_names.iloc[position].reset_index(drop=True)
    conflict = _has_qualifier_conflict(query_names.reset_index(drop=True), matched_names)

    is_close_enough = (similarity >= threshold) & ~conflict.to_numpy()
    match_index = np.where(is_close_enough, candidate_names.index.to_numpy()[position], np.nan)

    return pd.DataFrame({"match_index": match_index, "score": similarity}, index=query_names.index)
