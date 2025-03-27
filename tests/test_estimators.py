# test_estimators.py
import pytest
import numpy as np
from sr_agreement.superranker import (
    SRA,
    RandomListSRA,
    SRATest,
    SRACompare,
    SRAResult,
    RandomListSRAResult,
)

# Fixture to skip tests if rank_array_adapter is not available
rank_array_adapter = pytest.importorskip("rank_array_adapter")


# --- Test SRA Estimator ---
def test_sra_estimator_init():
    estimator = SRA(epsilon=0.1, metric="mad", B=5)
    assert estimator.config.epsilon == 0.1
    assert estimator.config.metric == "mad"
    assert estimator.config.B == 5
    assert not estimator.fitted_


def test_sra_estimator_generate(simple_ranks_arr, simple_nitems):
    estimator = SRA()
    # Check raises before fit
    with pytest.raises(ValueError):
        estimator.get_result()
    with pytest.raises(ValueError):
        estimator.values()
    with pytest.raises(ValueError):
        estimator.when_included()
    with pytest.raises(ValueError):
        estimator.smooth()

    # Fit data
    result_self = estimator.generate(simple_ranks_arr, nitems=simple_nitems)
    assert result_self is estimator  # Check returns self
    assert estimator.fitted_

    # Check results after fit
    sra_result = estimator.get_result()
    assert isinstance(sra_result, SRAResult)
    assert sra_result.values.shape == (simple_nitems,)
    vals = estimator.values()
    assert isinstance(vals, np.ndarray)
    assert vals.shape == (simple_nitems,)
    when = estimator.when_included()
    assert isinstance(when, np.ndarray)
    assert when.shape == (simple_nitems,)
    smoothed = estimator.smooth()
    assert isinstance(smoothed, np.ndarray)
    assert smoothed.shape == (simple_nitems,)


def test_sra_estimator_input_validation():
    estimator = SRA()
    # Non-2D
    with pytest.raises(ValueError, match="Input must be a 2D array"):
        estimator.generate(np.array([1, 2, 3]))
    # Non-numeric
    with pytest.raises(ValueError, match="Input array must be numeric"):
        estimator.generate(np.array([["A", "B"], ["C", "D"]]))


# --- Test RandomListSRA Estimator ---
def test_randomlist_sra_estimator_init():
    estimator = RandomListSRA(epsilon=0.1, metric="mad", B=5, n_permutations=50)
    assert estimator.config.epsilon == 0.1
    assert estimator.config.metric == "mad"
    assert estimator.config.B == 5
    assert estimator.n_permutations == 50
    assert not estimator.fitted_


def test_randomlist_sra_estimator_generate(simple_ranks_arr, simple_nitems):
    n_perm = 10
    estimator = RandomListSRA(n_permutations=n_perm)
    # Check raises before fit
    with pytest.raises(ValueError):
        estimator.get_result()
    with pytest.raises(ValueError):
        estimator.distribution()
    with pytest.raises(ValueError):
        estimator.confidence_band()
    with pytest.raises(ValueError):
        estimator.quantiles(0.5)

    # Fit data
    result_self = estimator.generate(simple_ranks_arr, nitems=simple_nitems)
    assert result_self is estimator
    assert estimator.fitted_

    # Check results after fit
    null_result = estimator.get_result()
    assert isinstance(null_result, RandomListSRAResult)
    assert null_result.distribution.shape == (simple_nitems, n_perm)
    dist = estimator.distribution()
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (simple_nitems, n_perm)
    band = estimator.confidence_band()
    assert isinstance(band, dict)
    assert "lower" in band and "upper" in band
    assert band["lower"].shape == (simple_nitems,)
    quants = estimator.quantiles([0.1, 0.9])
    assert isinstance(quants, dict)
    assert 0.1 in quants and 0.9 in quants
    assert quants[0.1].shape == (simple_nitems,)


def test_randomlist_sra_input_validation():
    estimator = RandomListSRA()
    # Non-2D
    with pytest.raises(ValueError, match="Input must be a 2D array"):
        estimator.generate(np.array([1, 2, 3]))
    # Non-numeric
    with pytest.raises(ValueError, match="Input array must be numeric"):
        estimator.generate(np.array([["A", "B"], ["C", "D"]]))


# --- Test SRATest Estimator ---
@pytest.fixture
def sra_test_inputs(simple_nitems):
    obs_res = SRAResult(
        values=np.linspace(1, 0, simple_nitems), config=SRAConfig()
    )
    null_res = RandomListSRAResult(
        distribution=np.random.rand(simple_nitems, 20),
        config=SRAConfig(),
        n_permutations=20,
    )
    return obs_res, null_res


def test_sratest_estimator_init():
    estimator = SRATest(
        style="l2", window=3, use_gpd=True, threshold_quantile=0.95
    )
    assert estimator.config.style == "l2"
    assert estimator.config.window == 3
    assert estimator.config.use_gpd
    assert estimator.config.threshold_quantile == 0.95
    assert not estimator.fitted_


def test_sratest_estimator_generate(sra_test_inputs):
    obs_res, null_res = sra_test_inputs
    estimator = SRATest()
    # Check raises before fit
    with pytest.raises(ValueError):
        estimator.get_result()
    with pytest.raises(ValueError):
        estimator.p_value()

    # Fit data
    result_self = estimator.generate(obs_res, null_res)
    assert result_self is estimator
    assert estimator.fitted_

    # Check results after fit
    test_result = estimator.get_result()
    assert test_result.p_value_empirical is not None
    assert test_result.test_statistic is not None
    assert test_result.null_statistics is not None
    p_val = estimator.p_value()
    assert isinstance(p_val, float) or np.isnan(p_val)


def test_sratest_estimator_generate_arrays(sra_test_inputs):
    obs_res, null_res = sra_test_inputs
    estimator = SRATest()
    estimator.generate(obs_res.values, null_res.distribution)  # Use arrays
    assert estimator.fitted_
    assert estimator.p_value() is not None


def test_sratest_estimator_input_validation(sra_test_inputs):
    obs_res, null_res = sra_test_inputs
    estimator = SRATest()
    # Mismatched shapes
    with pytest.raises(ValueError, match="must have same number of depths"):
        estimator.generate(obs_res.values[:-1], null_res.distribution)
    # Wrong dimensions
    with pytest.raises(ValueError, match="observed_sra must be 1D"):
        estimator.generate(obs_res.values.reshape(-1, 1), null_res.distribution)
    with pytest.raises(ValueError, match="null_dist must be 2D"):
        estimator.generate(obs_res.values, null_res.distribution[:, 0])


# --- Test SRACompare Estimator ---
@pytest.fixture
def sra_compare_inputs(simple_nitems):
    sra1 = SRAResult(
        values=np.linspace(1, 0, simple_nitems), config=SRAConfig()
    )
    sra2 = SRAResult(
        values=np.linspace(0.5, 0, simple_nitems), config=SRAConfig()
    )
    null1 = RandomListSRAResult(
        distribution=np.random.rand(simple_nitems, 20),
        config=SRAConfig(),
        n_permutations=20,
    )
    null2 = RandomListSRAResult(
        distribution=np.random.rand(simple_nitems, 20) + 0.1,
        config=SRAConfig(),
        n_permutations=20,
    )
    return sra1, sra2, null1, null2


def test_sracompare_estimator_init():
    estimator = SRACompare(style="l2", window=3)
    assert estimator.config.style == "l2"
    assert estimator.config.window == 3
    assert not estimator.fitted_


def test_sracompare_estimator_generate(sra_compare_inputs):
    sra1, sra2, null1, null2 = sra_compare_inputs
    estimator = SRACompare()
    # Check raises before fit
    with pytest.raises(ValueError):
        estimator.get_result()
    with pytest.raises(ValueError):
        estimator.p_value()

    # Fit data
    result_self = estimator.generate(sra1, sra2, null1, null2)
    assert result_self is estimator
    assert estimator.fitted_

    # Check results after fit
    comp_result = estimator.get_result()
    assert comp_result.p_value is not None
    assert comp_result.test_statistic is not None
    assert comp_result.null_statistics is not None
    p_val = estimator.p_value()
    assert isinstance(p_val, float) or np.isnan(p_val)


def test_sracompare_estimator_generate_arrays(sra_compare_inputs):
    sra1, sra2, null1, null2 = sra_compare_inputs
    estimator = SRACompare()
    estimator.generate(
        sra1.values, sra2.values, null1.distribution, null2.distribution
    )  # Use arrays
    assert estimator.fitted_
    assert estimator.p_value() is not None


def test_sracompare_estimator_input_validation(sra_compare_inputs):
    sra1, sra2, null1, null2 = sra_compare_inputs
    estimator = SRACompare()
    # Mismatched shapes
    with pytest.raises(
        ValueError, match="All inputs must have the same number of depths"
    ):
        estimator.generate(
            sra1.values[:-1],
            sra2.values,
            null1.distribution,
            null2.distribution,
        )
    with pytest.raises(
        ValueError, match="All inputs must have the same number of depths"
    ):
        estimator.generate(
            sra1.values,
            sra2.values,
            null1.distribution[:-1, :],
            null2.distribution,
        )
