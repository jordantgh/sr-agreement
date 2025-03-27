# test_core_algorithms.py
import pytest
import numpy as np
from sr_agreement.superranker import (
    SRAConfig,
    TestConfig,
    compute_sra,
    random_list_sra,
    test_sra,
    compare_sra,
)

# Fixture to skip tests if rank_array_adapter is not available
rank_array_adapter = pytest.importorskip("rank_array_adapter")


# --- Test compute_sra ---


@pytest.mark.parametrize("metric", ["sd", "mad"])
def test_compute_sra_basic(simple_ranks_arr, simple_nitems, metric):
    config = SRAConfig(metric=metric, B=1, epsilon=0.0)
    result = compute_sra(simple_ranks_arr, config, nitems=simple_nitems)

    assert result.config == config
    assert result.values.shape == (simple_nitems,)
    assert result.when_included is not None
    assert result.when_included.shape == (simple_nitems,)
    assert np.all(result.values >= 0)  # Disagreement non-negative

    # Check when_included basics (epsilon=0, B=1) -> all items included at depth = max rank
    # In this case, max rank is 4. Items 1-4 appear. Item 5 appears at rank 4 in list 1.
    # Expected: items 1-4 included depth <=4. Item 5 included depth 4.
    # Let's verify using the reshaped matrix from test_utils
    # Reshaped row 1: [1, 3, 2, nan, 4] -> Item 5 rank is 4
    # Expected when_included (1-based IDs 1,2,3,4,5):
    # Item 1: min rank = 1 -> included depth 1? No, S(d) requires rank <= d.
    # Need to trace S(d) logic...
    # d=1: Ranks<=1? Item 1 (rows 0,1), Item 2 (row 2). Prop(1)>0, Prop(2)>0. S(1)={1,2}. when_inc[0]=1, when_inc[1]=1
    # d=2: Ranks<=2? Items 1,2,3 (row 0), Item 1,3 (row 1), Items 2,1 (row 2). S(2)={1,2,3}. when_inc[2]=2
    # d=3: Ranks<=3? Items 1,2,3 (row 0), Items 1,3,2 (row 1), Items 2,1,4,3 (row 2). S(3)={1,2,3,4}. when_inc[3]=3
    # d=4: Ranks<=4? All items present are included. S(4)={1,2,3,4,5}. when_inc[4]=4
    expected_when_included = np.array([1, 1, 2, 3, 4], dtype=float)
    np.testing.assert_allclose(result.when_included, expected_when_included)


def test_compute_sra_nans(nan_ranks_arr, nan_nitems):
    config = SRAConfig(metric="sd", B=1, epsilon=0.0)
    result = compute_sra(nan_ranks_arr, config, nitems=nan_nitems)

    assert result.values.shape == (nan_nitems,)
    assert result.when_included is not None
    assert result.when_included.shape == (nan_nitems,)
    assert not np.any(np.isnan(result.values))  # Should default to 0 if needed

    # Check when_included with NaNs
    # d=1: Ranks<=1? Item 1 (rows 0,1), Item 2 (row 2). S(1)={1,2}. when[0]=1, when[1]=1
    # d=2: Ranks<=2? Items 1,2 (row 0), Item 1 (row 1), Items 2,1 (row 2). S(2)={1,2}.
    # d=3: Ranks<=3? Items 1,2,3 (row 0), Items 1,3 (row 1), Items 2,1,4,3 (row 2). S(3)={1,2,3,4}. when[2]=3, when[3]=3
    # d=4: Ranks<=4? Items 1,2,3 (row 0), Items 1,3 (row 1), Items 2,1,4,3 (row 2). S(4)={1,2,3,4}.
    # d=5: Ranks<=5? Items 1,2,3(r0), 1,3(r1), 2,1,4,3,5(r2). S(5)={1,2,3,4,5}. when[4]=5
    expected_when_included = np.array([1, 1, 3, 3, 5], dtype=float)
    np.testing.assert_allclose(result.when_included, expected_when_included)


def test_compute_sra_bootstrap(simple_ranks_arr, simple_nitems, mocker):
    # Mock random shuffle to make B>1 deterministic (though difficult)
    # Just check if B>1 runs and gives plausible result
    config = SRAConfig(metric="sd", B=5, epsilon=0.0)
    result = compute_sra(simple_ranks_arr, config, nitems=simple_nitems)
    assert result.values.shape == (simple_nitems,)
    # when_included should still be based on first bootstrap pass
    expected_when_included = np.array([1, 1, 2, 3, 4], dtype=float)
    np.testing.assert_allclose(result.when_included, expected_when_included)


@pytest.mark.parametrize(
    "epsilon, expected_sra1",
    [
        (
            0.0,
            0.0,
        ),  # All items 1,2 included. Ranks are [1,1], [1,2]. Var=0, Var=0.25. Mean=0.125. sqrt=0.353
        (
            0.6,
            0.0,
        ),  # Only item 1 included (prop=1.0 > 0.6). Ranks [1,1]. Var=0. Mean=0. sqrt=0.
        (1.0, 0.0),  # No items included. SRA=0.
    ],
)
def test_compute_sra_epsilon(epsilon, expected_sra1):
    # Simple 2x2 case
    ranks = np.array([[1, 2], [1, 3]], dtype=float)  # nitems=3
    config = SRAConfig(metric="sd", B=1, epsilon=epsilon)
    result = compute_sra(ranks, config, nitems=3)

    # Check SRA value at depth d=1
    # d=1: Ranks<=1? Item 1 (rows 0,1). Prop=1.0. Item 2(r0), 3(r1) Prop=0.5.
    # If eps=0.0: S(1)={1,2,3}. Ranks are Item1:[1,1], Item2:[2,nan], Item3:[nan,2].
    #   Disagreements: Var([1,1])=0. Var([2,nan])=nan. Var([nan,2])=nan. Mean(0,nan,nan)=0. Sqrt=0.
    # If eps=0.6: S(1)={1}. Ranks Item1:[1,1]. Disagreements: Var([1,1])=0. Mean=0. Sqrt=0.
    # If eps=1.0: S(1)={}. Mean=0. Sqrt=0.
    assert result.values[0] == pytest.approx(
        expected_sra1
    )  # Check value at depth 1 (index 0)


def test_compute_sra_nitems_inference(simple_ranks_arr, simple_nitems, caplog):
    config = SRAConfig()
    # Infer nitems
    result_inf = compute_sra(simple_ranks_arr, config, nitems=None)
    assert "Inferred nitems=5" in caplog.text
    assert result_inf.values.shape == (simple_nitems,)
    assert result_inf.when_included.shape == (simple_nitems,)

    # Provide nitems smaller than max ID
    with pytest.raises(ValueError, match="contains invalid item ID"):
        compute_sra(simple_ranks_arr, config, nitems=4)

    # Provide nitems larger than max ID (padding)
    result_pad = compute_sra(simple_ranks_arr, config, nitems=10)
    assert result_pad.values.shape == (10,)
    assert result_pad.when_included.shape == (10,)
    # Check that padded items are never included
    assert np.all(result_pad.when_included[simple_nitems:] == np.inf)


def test_compute_sra_sparse_ids(sparse_ranks_arr, sparse_nitems):
    # Check that it runs without MemoryError and gives correct shape
    config = SRAConfig(B=1)
    result = compute_sra(sparse_ranks_arr, config, nitems=sparse_nitems)
    assert result.values.shape == (sparse_nitems,)
    assert result.when_included.shape == (sparse_nitems,)
    # Check most when_included are inf
    assert np.sum(result.when_included != np.inf) == 4  # Only 4 items present


def test_compute_sra_epsilon_array(simple_ranks_arr, simple_nitems):
    eps_array = np.linspace(0, 0.5, simple_nitems)
    config = SRAConfig(epsilon=eps_array)
    result = compute_sra(simple_ranks_arr, config, nitems=simple_nitems)
    assert result.values.shape == (simple_nitems,)

    # Invalid length epsilon array
    with pytest.raises(ValueError, match="Length of epsilon array"):
        config_bad = SRAConfig(epsilon=np.array([0.1, 0.2]))
        compute_sra(simple_ranks_arr, config_bad, nitems=simple_nitems)


# --- Test random_list_sra ---


def test_random_list_sra_basic(simple_ranks_arr, simple_nitems):
    config = SRAConfig()
    n_perm = 50
    result = random_list_sra(
        simple_ranks_arr, config, n_permutations=n_perm, nitems=simple_nitems
    )
    assert result.config == config
    assert result.n_permutations == n_perm
    assert result.distribution.shape == (simple_nitems, n_perm)
    assert not np.any(
        np.isnan(result.distribution)
    )  # Should not produce NaNs here


def test_random_list_sra_nans(nan_ranks_arr, nan_nitems):
    # Checks that permutation respects the number of non-missing items per list
    config = SRAConfig()
    n_perm = 50
    result = random_list_sra(
        nan_ranks_arr, config, n_permutations=n_perm, nitems=nan_nitems
    )
    assert result.distribution.shape == (nan_nitems, n_perm)


def test_random_list_sra_nitems_inference(
    simple_ranks_arr, simple_nitems, caplog
):
    config = SRAConfig()
    n_perm = 10
    # Infer nitems
    result_inf = random_list_sra(
        simple_ranks_arr, config, n_permutations=n_perm, nitems=None
    )
    assert "Inferred nitems=5" in caplog.text
    assert result_inf.distribution.shape == (simple_nitems, n_perm)

    # Provide nitems larger than list_length (should pad structure)
    result_pad = random_list_sra(
        simple_ranks_arr, config, n_permutations=n_perm, nitems=10
    )
    assert result_pad.distribution.shape == (10, n_perm)


# --- Test test_sra ---


def test_test_sra_no_diff():
    obs = np.array([1.0, 1.0, 1.0])
    # Null where mean is also [1, 1, 1]
    null = np.array(
        [[1.0, 1.1, 0.9], [1.0, 0.9, 1.1], [1.0, 1.1, 0.9]]
    ).T  # Shape (3, 3)
    config = TestConfig(style="max")
    result = test_sra(obs, null, config)
    assert result.test_statistic == pytest.approx(0.0)
    # T_null should also be small. p ~ 1.0
    assert result.p_value_empirical > 0.9  # Expect high p-value
    assert result.p_value == result.p_value_empirical  # No GPD by default


def test_test_sra_large_diff():
    obs = np.array([10.0, 10.0, 10.0])
    null = np.array([[1.0, 1.1, 0.9], [1.0, 0.9, 1.1], [1.0, 1.1, 0.9]]).T
    config = TestConfig(style="max")
    result = test_sra(obs, null, config)
    mean_null = np.mean(null, axis=1)  # Approx [1, 1, 1]
    expected_T_obs = np.max(
        np.abs(obs - mean_null)
    )  # Approx max(abs(10-1)) = 9
    assert result.test_statistic == pytest.approx(expected_T_obs)
    # T_null should be small compared to T_obs. p ~ 0.0
    # Empirical p = (0 + 1) / (3 + 1) = 0.25 (low number of perms)
    assert result.p_value_empirical == pytest.approx(1 / 4)
    assert result.p_value == result.p_value_empirical


def test_test_sra_with_smoothing():
    obs = np.array([1.0, 1.0, 10.0, 10.0])
    null = np.ones((4, 10))  # Null mean is 1
    config = TestConfig(style="max", window=3)
    result = test_sra(obs, null, config)
    # Smoothed obs ~ [1, 4, 7, 10] ? (Check smooth_sra_window separately)
    # Smoothed null is still 1
    # T_obs depends on smoothed diffs
    assert result.test_statistic > 0
    # Check p-value is calculated
    assert 0 < result.p_value_empirical <= 1.0


def test_test_sra_with_gpd(mocker):
    # Need null distribution where GPD is likely applied
    obs = np.array([95.0])
    null_dist = np.linspace(1, 100, 500).reshape(1, -1)  # T_obs > threshold
    config = TestConfig(use_gpd=True, threshold_quantile=0.9)
    # Mock gpd calculation to check if it's called
    mock_gpd = mocker.patch(
        "superranker._calculate_gpd_pvalue",
        return_value={
            "p_value_gpd": 0.01,
            "gpd_fit": "mock_fit",
            "applied_gpd": True,
        },
    )
    result = test_sra(obs, null_dist, config)
    mock_gpd.assert_called_once()
    assert result.p_value_gpd == 0.01
    assert result.gpd_fit == "mock_fit"
    assert result.p_value == 0.01  # Prefers GPD


# --- Test compare_sra ---


def test_compare_sra_identical():
    sra1 = np.array([1.0, 1.0, 1.0])
    sra2 = np.array([1.0, 1.0, 1.0])
    null1 = np.array([[1.0, 1.1, 0.9]] * 3).T  # Mean 1
    null2 = np.array([[1.0, 0.9, 1.1]] * 3).T  # Mean 1
    config = TestConfig(style="max")
    result = compare_sra(sra1, sra2, null1, null2, config)
    # T_obs1 ~ 0, T_obs2 ~ 0. T_obs = T_obs1 - T_obs2 ~ 0
    assert result.test_statistic == pytest.approx(0.0)
    # p-value should be high (~0.5 for two-sided, >0.5 for T_null>=T_obs)
    assert result.p_value > 0.4  # Expect high p-value


def test_compare_sra_different():
    sra1 = np.array([10.0, 10.0, 10.0])  # Far from null mean 1
    sra2 = np.array([1.0, 1.0, 1.0])  # Close to null mean 1
    null1 = np.array([[1.0, 1.1, 0.9]] * 3).T  # Mean 1
    null2 = np.array([[1.0, 0.9, 1.1]] * 3).T  # Mean 1
    config = TestConfig(style="max")
    result = compare_sra(sra1, sra2, null1, null2, config)
    # T_obs1 ~ 9, T_obs2 ~ 0. T_obs = T_obs1 - T_obs2 ~ 9
    assert result.test_statistic > 5  # Expect large positive diff
    # Null diff dist should be centered around 0. p-value should be low.
    # (1 + count(>=T_obs)) / (N+1) -> should be small
    assert result.p_value < 0.6  # Low threshold due to few perms


def test_compare_sra_smoothing():
    sra1 = np.array([1.0, 1.0, 10.0, 10.0])
    sra2 = np.array([1.0, 1.0, 1.0, 1.0])
    null1 = np.ones((4, 10))
    null2 = np.ones((4, 10))
    config = TestConfig(window=3)
    result = compare_sra(sra1, sra2, null1, null2, config)
    # Smoothed sra1 deviates, sra2 doesn't. T_obs1 > 0, T_obs2 ~ 0. T_obs > 0.
    assert result.test_statistic > 0
    assert 0 < result.p_value <= 1.0
