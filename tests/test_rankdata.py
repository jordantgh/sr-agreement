# test_rankdata.py
import pytest
import numpy as np
from sr_agreement.superranker import RankData


# --- Test from_items ---
def test_from_items_list_basic(item_lists_str):
    ranks_arr, mapping = RankData.from_items(item_lists_str)

    assert isinstance(ranks_arr, np.ndarray)
    assert ranks_arr.shape == (4, 4)  # 4 lists, max length 4
    assert np.issubdtype(
        ranks_arr.dtype, np.floating
    )  # Should be float for NaN

    # Check mapping
    assert "item_to_id" in mapping
    assert "id_to_item" in mapping
    assert len(mapping["item_to_id"]) == 5  # A, B, C, D, E
    assert len(mapping["id_to_item"]) == 5
    assert (
        mapping["item_to_id"]["A"] == 1
    )  # Assuming sorted order A=1, B=2, ...
    assert mapping["item_to_id"]["E"] == 5
    assert mapping["id_to_item"][1] == "A"
    assert mapping["id_to_item"][5] == "E"

    # Check ranks_arr content (example row)
    # Row 0: ["A", "B", "C", "D"] -> [1, 2, 3, 4]
    np.testing.assert_array_equal(ranks_arr[0, :], np.array([1, 2, 3, 4]))
    # Row 1: ["A", "C", "B", "E"] -> [1, 3, 2, 5]
    np.testing.assert_array_equal(ranks_arr[1, :], np.array([1, 3, 2, 5]))
    # Row 3: ["A", "B", None, "D"] -> [1, 2, nan, 4]
    np.testing.assert_array_equal(ranks_arr[3, :], np.array([1, 2, np.nan, 4]))


def test_from_items_np_array_str():
    items_np = np.array(
        [
            ["A", "B", "C"],
            ["B", "A", "na"],  # 'na' should become NaN
            ["C", " ", "A"],  # ' ' should become NaN
        ],
        dtype=object,
    )  # Use object to allow mixed types / Nones easily
    ranks_arr, mapping = RankData.from_items(items_np)

    assert ranks_arr.shape == (3, 3)
    assert len(mapping["item_to_id"]) == 3  # A, B, C
    # Row 1: [B, A, na] -> [2, 1, nan]
    np.testing.assert_array_equal(ranks_arr[1, :], np.array([2, 1, np.nan]))
    # Row 2: [C, ' ', A] -> [3, nan, 1]
    np.testing.assert_array_equal(ranks_arr[2, :], np.array([3, np.nan, 1]))


def test_from_items_np_array_str_custom_na():
    items_np = np.array(
        [
            ["A", "B", "MISSING"],
            ["B", "A", "na"],
        ],
        dtype=object,
    )
    ranks_arr, mapping = RankData.from_items(
        items_np, na_strings=["na", "missing", "N/A"]
    )  # 'na' is default

    assert ranks_arr.shape == (2, 3)
    # Row 0: [A, B, MISSING] -> [1, 2, nan]
    np.testing.assert_array_equal(ranks_arr[0, :], np.array([1, 2, np.nan]))
    # Row 1: [B, A, na] -> [2, 1, nan] ('na' is default NA)
    np.testing.assert_array_equal(ranks_arr[1, :], np.array([2, 1, np.nan]))


def test_from_items_np_array_numeric():
    items_np = np.array(
        [
            [10, 20, 30.0],
            [20, np.nan, 10],  # Explicit NaN
        ]
    )
    ranks_arr, mapping = RankData.from_items(
        items_np
    )  # Should treat numbers as items

    assert ranks_arr.shape == (2, 3)
    assert len(mapping["item_to_id"]) == 3  # Items are 10, 20, 30
    assert mapping["item_to_id"][10] == 1  # Assuming 10<20<30
    assert mapping["item_to_id"][20] == 2
    assert mapping["item_to_id"][30.0] == 3
    # Row 0: [10, 20, 30] -> [1, 2, 3]
    np.testing.assert_array_equal(ranks_arr[0, :], np.array([1, 2, 3]))
    # Row 1: [20, nan, 10] -> [2, nan, 1]
    np.testing.assert_array_equal(ranks_arr[1, :], np.array([2, np.nan, 1]))


def test_from_items_unsortable_items(item_lists_str, caplog):
    # Add an unsortable item (like a dictionary)
    item_lists_unsortable = item_lists_str + [[{"a": 1}, "A"]]
    with pytest.warns(UserWarning, match="Items are not naturally sortable"):
        ranks_arr, mapping = RankData.from_items(item_lists_unsortable)

    # Check if mapping uses string representations
    assert str({"a": 1}) in mapping["item_to_id"]
    dict_id = mapping["item_to_id"][str({"a": 1})]
    assert mapping["id_to_item"][dict_id] == str(
        {"a": 1}
    )  # It stores the string rep

    # Check the rank array uses the ID derived from the string rep
    # List 4: [{"a":1}, "A"] -> [dict_id, 1] (assuming A is ID 1)
    np.testing.assert_array_equal(ranks_arr[4, :2], np.array([dict_id, 1]))


def test_from_items_empty_input():
    ranks_arr, mapping = RankData.from_items([])
    assert ranks_arr.shape == (0, 0)
    assert not mapping["item_to_id"]
    assert not mapping["id_to_item"]

    ranks_arr, mapping = RankData.from_items([[], []])
    assert ranks_arr.shape == (2, 0)
    assert not mapping["item_to_id"]


def test_from_items_all_na():
    with pytest.warns(UserWarning, match="No valid items found"):
        ranks_arr, mapping = RankData.from_items([[None, np.nan], ["", "na"]])
    assert ranks_arr.shape == (2, 2)
    assert np.all(np.isnan(ranks_arr))
    assert not mapping["item_to_id"]


# --- Test from_scores ---
def test_from_scores_ascending(scores_lists):
    ranks_arr = RankData.from_scores(scores_lists, ascending=True)
    assert ranks_arr.shape == (3, 4)  # 3 lists, max length 4
    # Row 0: [0.1, 0.5, 0.05, 0.8] -> ranks [2, 3, 1, 4]
    np.testing.assert_array_equal(ranks_arr[0, :], np.array([2, 3, 1, 4]))
    # Row 1: [10, 5, 15, 2] -> ranks [3, 2, 4, 1]
    np.testing.assert_array_equal(ranks_arr[1, :], np.array([3, 2, 4, 1]))
    # Row 2: [0.9, 0.8, nan, 0.7] -> ranks [3, 2, nan, 1] (padded with nan first)
    np.testing.assert_array_equal(ranks_arr[2, :], np.array([3, 2, np.nan, 1]))


def test_from_scores_descending(scores_lists):
    ranks_arr = RankData.from_scores(scores_lists, ascending=False)
    assert ranks_arr.shape == (3, 4)
    # Row 0: [0.1, 0.5, 0.05, 0.8] -> ranks [3, 2, 4, 1] (higher score = lower rank)
    np.testing.assert_array_equal(ranks_arr[0, :], np.array([3, 2, 4, 1]))
    # Row 1: [10, 5, 15, 2] -> ranks [2, 3, 1, 4]
    np.testing.assert_array_equal(ranks_arr[1, :], np.array([2, 3, 1, 4]))
    # Row 2: [0.9, 0.8, nan, 0.7] -> ranks [1, 2, nan, 3]
    np.testing.assert_array_equal(ranks_arr[2, :], np.array([1, 2, np.nan, 3]))


def test_from_scores_ties():
    scores = [[0.1, 0.5, 0.1, 0.8]]  # Ties at 0.1
    # rankdata method='ordinal' breaks ties by appearance order
    ranks_arr = RankData.from_scores(scores, ascending=True)
    # Expected: 0.1(1st) -> rank 1, 0.5 -> rank 3, 0.1(2nd) -> rank 2, 0.8 -> rank 4
    np.testing.assert_array_equal(ranks_arr[0, :], np.array([1, 3, 2, 4]))


# --- Test map_ids_to_items ---
def test_map_ids_to_items_basic(item_lists_str):
    ranks_arr, mapping = RankData.from_items(item_lists_str)
    # Map the first row back: [1, 2, 3, 4] -> ["A", "B", "C", "D"]
    items_row0 = RankData.map_ids_to_items(ranks_arr[0, :], mapping)
    np.testing.assert_array_equal(
        items_row0, np.array(["A", "B", "C", "D"], dtype=object)
    )

    # Map the row with NaN: [1, 2, nan, 4] -> ["A", "B", None, "D"]
    items_row3 = RankData.map_ids_to_items(ranks_arr[3, :], mapping)
    np.testing.assert_array_equal(
        items_row3, np.array(["A", "B", None, "D"], dtype=object)
    )


def test_map_ids_to_items_unmapped():
    mapping = {"id_to_item": {1: "A", 2: "B"}}
    ids_to_map = np.array([1, 3, np.nan, 2])  # ID 3 is not in map
    items_arr = RankData.map_ids_to_items(ids_to_map, mapping)
    # Expected: ["A", None, None, "B"]
    np.testing.assert_array_equal(
        items_arr, np.array(["A", None, None, "B"], dtype=object)
    )


def test_map_ids_to_items_invalid_mapping():
    mapping = {"items": {1: "A"}}  # Missing 'id_to_item' key
    with pytest.raises(ValueError):
        RankData.map_ids_to_items(np.array([1]), mapping)
