# test_datamodels.py
import pytest
import numpy as np
from sr_agreement.superranker import (
    SRAConfig,
    TestConfig,
    SRAResult,
    RandomListSRAResult,
    GPDFit,
    TestResult,
    ComparisonResult,
)


# --- SRAConfig Tests ---
def test_sra_config_defaults():
    config = SRAConfig()
    assert config.epsilon == 0.0
    assert config.metric == "sd"
    assert config.B == 1


@pytest.mark.parametrize(
    "metric, B, epsilon, expected_error",
    [
        ("foo", 1, 0.0, ValueError),  # Invalid metric
        ("sd", 0, 0.0, ValueError),  # Invalid B
        ("sd", -1, 0.0, ValueError),
        ("sd", 1.5, 0.0, ValueError),
        ("sd", 1, -0.1, ValueError),  # Invalid epsilon (scalar)
        ("sd", 1, 1.1, ValueError),
        ("sd", 1, np.array([-0.1, 0.5]), ValueError),  # Invalid epsilon (array)
        ("sd", 1, np.array([0.1, 1.1]), ValueError),
        ("sd", 1, [0.1, 0.2], TypeError),  # Invalid epsilon type
    ],
)
def test_sra_config_invalid_init(metric, B, epsilon, expected_error):
    with pytest.raises(expected_error):
        SRAConfig(metric=metric, B=B, epsilon=epsilon)


def test_sra_config_valid_init():
    config = SRAConfig(metric="mad", B=10, epsilon=0.1)
    assert config.metric == "mad"
    assert config.B == 10
    assert config.epsilon == 0.1

    eps_arr = np.array([0.1, 0.2])
    config_arr = SRAConfig(epsilon=eps_arr)
    np.testing.assert_array_equal(config_arr.epsilon, eps_arr)


# --- TestConfig Tests ---
def test_test_config_defaults():
    config = TestConfig()
    assert (
        config.style == "max"
    )  # Default changed based on discussion? Check defaults.
    assert config.window == 1
    assert not config.use_gpd
    assert config.threshold_quantile == 0.90


@pytest.mark.parametrize(
    "style, window, use_gpd, threshold_quantile, expected_error",
    [
        ("foo", 1, False, 0.9, ValueError),  # Invalid style
        ("l2", 0, False, 0.9, ValueError),  # Invalid window
        ("l2", -1, False, 0.9, ValueError),
        ("l2", 1.5, False, 0.9, ValueError),
        ("l2", 1, False, -0.1, ValueError),  # Invalid threshold
        ("l2", 1, False, 1.0, ValueError),
        ("l2", 1, False, 1.1, ValueError),
    ],
)
def test_test_config_invalid_init(
    style, window, use_gpd, threshold_quantile, expected_error
):
    with pytest.raises(expected_error):
        TestConfig(
            style=style,
            window=window,
            use_gpd=use_gpd,
            threshold_quantile=threshold_quantile,
        )


def test_test_config_valid_init():
    config = TestConfig(
        style="l2", window=5, use_gpd=True, threshold_quantile=0.95
    )
    assert config.style == "l2"
    assert config.window == 5
    assert config.use_gpd
    assert config.threshold_quantile == 0.95


# --- SRAResult Tests ---
def test_sra_result_init():
    vals = np.array([1, 2, 3])
    when = np.array([1, 1, 2])
    conf = SRAConfig()
    res = SRAResult(values=vals, config=conf, when_included=when)
    assert isinstance(res.values, np.ndarray)
    np.testing.assert_array_equal(res.values, vals)
    assert isinstance(res.when_included, np.ndarray)
    np.testing.assert_array_equal(res.when_included, when)
    assert res.config == conf


def test_sra_result_init_list_input():
    vals = [1, 2, 3]
    when = [1, 1, 2]
    conf = SRAConfig()
    res = SRAResult(values=vals, config=conf, when_included=when)
    assert isinstance(res.values, np.ndarray)
    assert isinstance(res.when_included, np.ndarray)


def test_sra_result_smooth(mocker):
    vals = np.array([1, 2, 3, 4, 5], dtype=float)
    conf = SRAConfig()
    res = SRAResult(values=vals, config=conf)
    # Mock the underlying smoothing function to check if it's called
    mock_smooth = mocker.patch(
        "sr_agreement.superranker.smooth_sra_window",
        return_value=np.array([2.0] * 5),
    )
    smoothed = res.smooth(window_size=3)
    mock_smooth.assert_called_once_with(vals, 3)
    assert isinstance(smoothed, np.ndarray)


# --- RandomListSRAResult Tests ---
def test_random_list_sra_result_init():
    dist = np.random.rand(10, 100)
    conf = SRAConfig()
    n_perm = 100
    res = RandomListSRAResult(
        distribution=dist, config=conf, n_permutations=n_perm
    )
    assert isinstance(res.distribution, np.ndarray)
    np.testing.assert_array_equal(res.distribution, dist)
    assert res.config == conf
    assert res.n_permutations == n_perm


def test_random_list_sra_result_confidence_band():
    dist = np.array(
        [[1, 2, 3, 4, 10], [5, 6, 7, 8, 15]], dtype=float
    ).T  # Shape (5, 2)
    conf = SRAConfig()
    res = RandomListSRAResult(distribution=dist, config=conf, n_permutations=5)
    band = res.confidence_band(confidence=0.80)  # alpha = 0.1
    # Quantiles at 0.1 and 0.9
    # For depth 0: sorted [1, 2, 3, 4, 10] -> 10% is 1.4, 90% is 7.6 (approx interpolation)
    # Use np.quantile for exact values:
    expected_lower = np.quantile(dist, 0.1, axis=1)
    expected_upper = np.quantile(dist, 0.9, axis=1)
    np.testing.assert_allclose(band["lower"], expected_lower)
    np.testing.assert_allclose(band["upper"], expected_upper)


# --- TestResult Tests ---
def test_test_result_init():
    null_stats = np.random.rand(100)
    conf = TestConfig()
    res = TestResult(
        p_value_empirical=0.05,
        test_statistic=5.0,
        null_statistics=null_stats,
        config=conf,
    )
    assert res.p_value_empirical == 0.05
    assert res.test_statistic == 5.0
    assert isinstance(res.null_statistics, np.ndarray)
    assert res.config == conf
    assert res.p_value_gpd is None
    assert res.gpd_fit is None
    assert res.p_value == 0.05  # Should return empirical if GPD is None


def test_test_result_p_value_preference():
    null_stats = np.random.rand(100)
    conf = TestConfig()
    gpd_fit = GPDFit(xi=0.1, beta=1.0, threshold=4.0)
    res = TestResult(
        p_value_empirical=0.05,
        test_statistic=5.0,
        null_statistics=null_stats,
        config=conf,
        p_value_gpd=0.01,
        gpd_fit=gpd_fit,
    )
    assert res.p_value_gpd == 0.01
    assert res.p_value == 0.01  # Should prefer GPD


# --- ComparisonResult Tests ---
def test_comparison_result_init():
    null_stats = np.random.rand(100)
    conf = TestConfig()
    res = ComparisonResult(
        p_value=0.1,
        test_statistic=-2.0,
        null_statistics=null_stats,
        config=conf,
    )
    assert res.p_value == 0.1
    assert res.test_statistic == -2.0
    assert isinstance(res.null_statistics, np.ndarray)
    assert res.config == conf
