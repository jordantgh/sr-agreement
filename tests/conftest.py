# conftest.py
import pytest
import numpy as np


# Fixture for simple, valid rank data (numeric IDs 1-based)
@pytest.fixture
def simple_ranks_arr():
    return np.array(
        [
            [1, 2, 3, 4],
            [1, 3, 2, 5],  # Item 5 appears
            [2, 1, 4, 3],
        ],
        dtype=float,
    )


@pytest.fixture
def simple_nitems():
    return 5  # Max ID in simple_ranks_arr is 5


# Fixture for rank data with NaNs and different lengths implicitly
@pytest.fixture
def nan_ranks_arr():
    return np.array(
        [
            [1, 2, 3, np.nan, np.nan],
            [1, 3, np.nan, np.nan, np.nan],
            [2, 1, 4, 3, 5],  # Longer list with item 5
        ],
        dtype=float,
    )


@pytest.fixture
def nan_nitems():
    return 5  # Max ID is 5


# Fixture for item lists (strings)
@pytest.fixture
def item_lists_str():
    return [
        ["A", "B", "C", "D"],
        ["A", "C", "B", "E"],  # Item E appears
        ["B", "A", "D", "C"],
        ["A", "B", None, "D"],  # Mixed length / None
    ]


# Fixture for large/sparse ID ranks
@pytest.fixture
def sparse_ranks_arr():
    return np.array(
        [[10, 5000, 20, np.nan], [20, 10, 60000, 5000], [10, 20, 5000, 60000]],
        dtype=float,
    )


@pytest.fixture
def sparse_nitems():
    return 60000


# Fixture for scores
@pytest.fixture
def scores_lists():
    return [
        [0.1, 0.5, 0.05, 0.8],  # Ranks should be 2, 3, 1, 4 (ascending)
        [10, 5, 15, 2],  # Ranks should be 3, 2, 4, 1 (ascending)
        [0.9, 0.8, np.nan, 0.7],  # Ranks should be 3, 2, nan, 1 (ascending)
    ]
