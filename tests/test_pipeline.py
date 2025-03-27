# test_pipeline.py
import pytest
import numpy as np
from sr_agreement.superranker import RankPipeline, SRA, RandomListSRA, SRATest

# Fixture to skip tests if rank_array_adapter is not available
rank_array_adapter = pytest.importorskip("rank_array_adapter")


# --- Test RankPipeline ---


def test_pipeline_init():
    pipe = RankPipeline()
    assert pipe._input_data is None
    assert pipe.ranked_data is None
    assert pipe.sra is None
    assert pipe.null_dist is None
    assert pipe.test is None
    assert not pipe._generated


# --- Input Methods ---
def test_pipeline_with_ranked_data(simple_ranks_arr):
    pipe = RankPipeline().with_ranked_data(simple_ranks_arr)
    assert pipe._input_type == "ranks"
    assert pipe._input_data is simple_ranks_arr
    assert not pipe._generated  # Preparation deferred

    # Check data preparation happens
    pipe._prepare_data()
    assert pipe._generated
    assert pipe.ranked_data is not None
    assert pipe.nitems == 5  # Inferred from max ID
    assert pipe.item_mapping is None


def test_pipeline_with_items_lists(item_lists_str):
    pipe = RankPipeline().with_items_lists(item_lists_str)
    assert pipe._input_type == "items"
    assert pipe._input_data["items_lists"] is item_lists_str
    assert not pipe._generated

    # Check data preparation happens
    pipe._prepare_data()
    assert pipe._generated
    assert pipe.ranked_data is not None
    assert pipe.ranked_data.shape[1] == 4  # Max length
    assert pipe.item_mapping is not None
    assert pipe.nitems == 5  # Unique items A,B,C,D,E


def test_pipeline_with_scores_raises_error(scores_lists):
    pipe = RankPipeline()
    with pytest.raises(
        NotImplementedError,
        match="Pipeline 'with_scores' requires internal conversion",
    ):
        pipe.with_scores(scores_lists)


def test_pipeline_double_input_raises():
    pipe = RankPipeline().with_ranked_data(np.array([[1]]))
    with pytest.raises(ValueError, match="Input data already set"):
        pipe.with_items_lists([["A"]])


# --- Compute Methods ---
def test_pipeline_compute_sra(simple_ranks_arr):
    pipe = RankPipeline().with_ranked_data(simple_ranks_arr)
    pipe.compute_sra(epsilon=0.1, metric="mad", B=2)

    assert pipe._generated  # Should trigger data prep
    assert pipe.sra is not None
    assert isinstance(pipe.sra, SRA)
    assert pipe.sra.fitted_
    assert pipe.sra_config is not None
    assert pipe.sra_config.epsilon == 0.1
    assert pipe.sra_config.metric == "mad"
    assert pipe.sra_config.B == 2


def test_pipeline_compute_sra_no_data():
    pipe = RankPipeline()
    with pytest.raises(ValueError, match="No input data provided"):
        pipe.compute_sra()


def test_pipeline_random_list_sra(simple_ranks_arr):
    pipe = RankPipeline().with_ranked_data(simple_ranks_arr)
    pipe.random_list_sra(n_permutations=50)

    assert pipe._generated
    assert pipe.null_dist is not None
    assert isinstance(pipe.null_dist, RandomListSRA)
    assert pipe.null_dist.fitted_
    assert pipe.null_config == {"n_permutations": 50}
    # Check if default SRA config was used if compute_sra wasn't called
    assert pipe.null_dist.config.epsilon == 0.0  # Default SRAConfig epsilon


def test_pipeline_random_list_sra_uses_sra_config(simple_ranks_arr):
    pipe = RankPipeline().with_ranked_data(simple_ranks_arr)
    pipe.compute_sra(epsilon=0.2)  # Set non-default config
    pipe.random_list_sra(n_permutations=50)
    assert (
        pipe.null_dist.config.epsilon == 0.2
    )  # Should use the config from compute_sra


def test_pipeline_test_significance(simple_ranks_arr):
    pipe = RankPipeline().with_ranked_data(simple_ranks_arr)
    # Check errors if steps missing
    with pytest.raises(ValueError, match="SRA must be computed"):
        pipe.test_significance()
    pipe.compute_sra()
    with pytest.raises(ValueError, match="Null distribution must be generated"):
        pipe.test_significance()
    pipe.random_list_sra(n_permutations=10)  # Use few perms for speed
    pipe.test_significance(style="l2", window=3)

    assert pipe.test is not None
    assert isinstance(pipe.test, SRATest)
    assert pipe.test.fitted_
    assert pipe.test_config is not None
    assert pipe.test_config.style == "l2"
    assert pipe.test_config.window == 3


# --- Build Method ---
def test_pipeline_build_empty():
    pipe = RankPipeline()
    with pytest.warns(
        UserWarning, match="Pipeline built with no analysis steps"
    ):
        results = pipe.build()
    assert isinstance(results, dict)
    assert not results  # Empty dict


def test_pipeline_build_data_only(item_lists_str):
    pipe = RankPipeline().with_items_lists(item_lists_str)
    results = pipe.build()
    assert "item_mapping" in results
    assert results["item_mapping"] is not None
    # Should not contain sra, null, test results
    assert "sra_result" not in results
    assert "null_result" not in results
    assert "test_result" not in results


def test_pipeline_build_full(simple_ranks_arr):
    pipe = (
        RankPipeline()
        .with_ranked_data(simple_ranks_arr)
        .compute_sra()
        .random_list_sra(n_permutations=10)
        .test_significance()
    )
    results = pipe.build()

    assert "sra_estimator" in results
    assert "sra_result" in results
    assert "sra_values" in results
    assert "when_included" in results
    assert "null_estimator" in results
    assert "null_result" in results
    assert "null_distribution" in results
    assert "confidence_band" in results
    assert "test_estimator" in results
    assert "test_result" in results
    assert "p_value" in results
    assert "significant_05" in results
    # Check types
    assert isinstance(results["sra_values"], np.ndarray)
    assert isinstance(results["null_distribution"], np.ndarray)
    assert isinstance(results["p_value"], float) or np.isnan(results["p_value"])


def test_pipeline_build_handles_nan_pvalue(mocker):
    # Mock test_sra to return NaN p-value
    mock_test_result = mocker.MagicMock()
    mock_test_result.p_value = np.nan
    mocker.patch("superranker.test_sra", return_value=mock_test_result)

    pipe = (
        RankPipeline()
        .with_ranked_data(np.array([[1, 2], [1, 2]]))  # Need valid input
        .compute_sra()
        .random_list_sra(n_permutations=10)
        .test_significance()
    )
    results = pipe.build()

    assert "p_value" in results
    assert np.isnan(results["p_value"])
    assert "significant_05" in results
    assert results["significant_05"] is None  # Should be None if p=NaN


def test_pipeline_build_handles_data_prep_error(mocker):
    mocker.patch(
        "superranker.RankData.from_items", side_effect=ValueError("Bad items")
    )
    pipe = RankPipeline().with_items_lists([["A"]])
    results = pipe.build()
    assert "error" in results
    assert "Data preparation failed: Bad items" in results["error"]
