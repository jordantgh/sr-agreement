"""
SuperRanker: A package for computing and analyzing Sequential Rank Agreement.

This package provides tools for comparing ranked lists, generating null distributions,
and testing the significance of rank agreement.
"""

import functools
import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Any, Dict, Tuple
import numpy as np
from scipy.stats import genpareto, rankdata
from enum import Enum
import pandas as pd
import collections.abc
from joblib import Parallel, delayed

# Import necessary components from the adapter module
try:
    from ma_utils.rank_array_adapter import (
        convert_rank_data,
        RankShape,
        RankDataAdapter,
        _validate_rankcol_no_midrow_nulls,
        _validate_rankcol_no_duplicates,
    )
except ImportError:
    # Provide dummy definitions if the module is not found,
    # though the relevant functions will fail.
    warnings.warn(
        "rank_array_adapter module not found. Functionality relying on it will fail."
    )

    class RankShape(str, Enum):
        LISTROW_RANKCOL = "listrow_rankcol"
        LISTCOL_RANKROW = "listcol_rankrow"
        LISTROW_ITEMCOL = "listrow_itemcol"
        LISTCOL_ITEMROW = "listcol_itemrow"

    @dataclass
    class RankDataAdapter:
        data: np.ndarray
        id_to_index_mapping: Dict[float, int]

    def convert_rank_data(*args, **kwargs) -> RankDataAdapter:
        raise NotImplementedError("rank_array_adapter module is required.")


def _worker_rls_joblib(
    num_lists: int,
    final_nitems: int,
    real_lengths: np.ndarray,
    all_possible_ids: np.ndarray,
    sra_config: "SRAConfig",
    compute_sra_func: callable,
    rng_entropy: int,
) -> np.ndarray:
    """
    Build ONE permuted data set and return its SRA curve.

    A fresh independent Generator is created from the supplied entropy so that
    every worker is reproducible and independent of the global RNG.
    """
    rng = np.random.default_rng(rng_entropy)

    # build a (num_lists × n_items) matrix filled with NaN
    current_obj_for_rep = np.full(
        (num_lists, final_nitems), np.nan, dtype=float
    )

    for row_idx, nn in enumerate(real_lengths):
        nn = min(nn, final_nitems)
        if nn:
            sample = rng.choice(all_possible_ids, size=nn, replace=False)
            current_obj_for_rep[row_idx, :nn] = sample

    sra_curve = compute_sra_func(
        current_obj_for_rep,
        sra_config,
        nitems=final_nitems,
        rng=rng,
    ).values
    return sra_curve


def _worker_gnd_joblib(
    current_col_from_null_matrix: np.ndarray,
    n_depths: int,
    colsums: np.ndarray,
    valid_counts: np.ndarray,
    aggregator_style: str,
    aggregator_func: callable,
) -> float:
    """
    Worker function for a single column in _generate_null_distribution using joblib.
    """
    if np.all(np.isnan(current_col_from_null_matrix)):
        return np.nan

    loo_sum = colsums - np.nan_to_num(current_col_from_null_matrix)
    loo_count = valid_counts - (~np.isnan(current_col_from_null_matrix)).astype(
        int
    )

    loo_mean = np.full(n_depths, np.nan)
    valid_mask = loo_count > 0
    if np.any(valid_mask):
        loo_mean[valid_mask] = loo_sum[valid_mask] / loo_count[valid_mask]

    diffs = np.abs(current_col_from_null_matrix - loo_mean)
    return aggregator_func(diffs, aggregator_style)


def _worker_smooth_col_joblib(
    col_data: np.ndarray, window_size: int, smooth_sra_window_func: callable
) -> np.ndarray:
    """Worker function to smooth a single column using joblib."""
    return smooth_sra_window_func(col_data, window_size)


def require_generated(method):
    """Decorator that checks if an estimator was generated before calling a method."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, "fitted_") or not self.fitted_:
            raise ValueError(
                f"This {self.__class__.__name__} instance was not generated yet. Call 'generate' first."
            )
        return method(self, *args, **kwargs)

    return wrapped


###################
# Core Data Models
###################


@dataclass(frozen=True)
class SRAConfig:
    """Immutable configuration for SRA computation.

    Parameters
    ----------
    epsilon : float or array-like, default=0.0
        Threshold for an item to be included in S(d).
        Must be between 0 and 1.
    metric : str, default="sd"
        Method to measure dispersion of ranks.
        Must be one of ["sd", "mad"].
    B : int, default=1
        Number of bootstrap samples for handling missing values.
        Must be positive.
    """

    epsilon: float | np.ndarray = 0.0
    metric: Literal["sd", "mad"] = "sd"
    B: int = 1

    def __post_init__(self):
        # Validate metric
        if self.metric not in ["sd", "mad"]:
            raise ValueError(f"Metric must be 'sd' or 'mad', got {self.metric}")

        # Validate B
        if not isinstance(self.B, int) or self.B < 1:
            raise ValueError(f"B must be a positive integer, got {self.B}")

        # Validate epsilon
        if isinstance(self.epsilon, (int, float)):
            if not 0 <= self.epsilon <= 1:
                raise ValueError(
                    f"Epsilon must be between 0 and 1, got {self.epsilon}"
                )
        elif isinstance(self.epsilon, np.ndarray):
            if np.any((self.epsilon < 0) | (self.epsilon > 1)):
                raise ValueError(
                    "All values in epsilon array must be between 0 and 1"
                )
            # Epsilon array length validation happens inside compute_sra based on nitems
        else:
            raise TypeError(
                f"Epsilon must be float or numpy.ndarray, got {type(self.epsilon)}"
            )


@dataclass(frozen=True)
class TestConfig:
    """Configuration for statistical testing of SRA values.

    Parameters
    ----------
    style : str, default="max"
        Method to aggregate differences.
        Must be one of ["l2", "max"].
    window : int, default=1
        Size of smoothing window. Use 1 for no smoothing.
        Must be positive.
    use_gpd : bool, default=False
        Whether to use generalized Pareto distribution for extreme p-values.
    threshold_quantile : float, default=0.90
        Quantile to use as threshold for GPD fitting.
        Must be between 0 and 1.
    tails : str, default="one-tailed"
        For comparison tests, specifies if the test is one-tailed
        (e.g., is sra1's deviation significantly GREATER than sra2's deviation)
        or two-tailed (is sra1's deviation significantly DIFFERENT from sra2's deviation).
        Applies only to compare_sra/SRACompare.
        Must be one of ["one-tailed", "two-tailed"].
    """

    style: Literal["l2", "max"] = "max"
    window: int = 1
    use_gpd: bool = False
    threshold_quantile: float = 0.90
    sra_comparison_tails: Literal["one-tailed", "two-tailed"] = "one-tailed"

    def __post_init__(self):
        # Validate style
        if self.style not in ["l2", "max"]:
            raise ValueError(f"Style must be 'l2' or 'max', got {self.style}")

        # Validate window
        if not isinstance(self.window, int) or self.window < 1:
            raise ValueError(
                f"Window size must be a positive integer, got {self.window}"
            )

        # Validate threshold_quantile
        if not 0 < self.threshold_quantile < 1:
            raise ValueError(
                f"Threshold quantile must be between 0 and 1, got {self.threshold_quantile}"
            )

        # Validate tails
        if self.sra_comparison_tails not in ["one-tailed", "two-tailed"]:
            raise ValueError(
                f"Tails must be 'one-tailed' or 'two-tailed', got {self.sra_comparison_tails}"
            )


###################
# Result Containers
###################


@dataclass(frozen=True)
class SRAResult:
    """Immutable container for SRA computation results.

    Attributes
    ----------
    values : numpy.ndarray
        SRA values for each depth (1 to nitems). Shape (nitems,).
    config : SRAConfig
        Configuration used for computation.
    when_included : numpy.ndarray, optional
        Depth at which each item (1 to nitems) was first included.
        Shape (nitems,). Index corresponds to item_id - 1. np.inf if never included.
    """

    values: np.ndarray
    config: SRAConfig
    when_included: Optional[np.ndarray] = None

    def __post_init__(self):
        if not isinstance(self.values, np.ndarray):
            object.__setattr__(self, "values", np.asarray(self.values))

        if self.when_included is not None and not isinstance(
            self.when_included, np.ndarray
        ):
            object.__setattr__(
                self, "when_included", np.asarray(self.when_included)
            )

    def smooth(self, window_size: int = 10) -> np.ndarray:
        """Return smoothed SRA values.

        Parameters
        ----------
        window_size : int, default=10
            Size of the rolling window.

        Returns
        -------
        smoothed_values : numpy.ndarray
            Smoothed SRA curve.
        """
        return smooth_sra_window(self.values, window_size)


@dataclass(frozen=True)
class RandomListSRAResult:
    """Immutable container for null distribution generation results.

    Attributes
    ----------
    distribution : numpy.ndarray
        Matrix of shape (n_depths, n_permutations) containing SRA values
        for each permutation. n_depths is typically nitems.
    config : SRAConfig
        Configuration used for computation.
    n_permutations : int
        Number of permutations used.
    """

    distribution: np.ndarray
    config: SRAConfig
    n_permutations: int

    def __post_init__(self):
        if not isinstance(self.distribution, np.ndarray):
            object.__setattr__(
                self, "distribution", np.asarray(self.distribution)
            )

    def confidence_band(
        self, confidence: float = 0.95
    ) -> dict[str, np.ndarray]:
        """Compute confidence band for the null distribution.

        Parameters
        ----------
        confidence : float, default=0.95
            Confidence level.

        Returns
        -------
        band : dict
            Dictionary with 'lower' and 'upper' arrays.
        """
        alpha = (1 - confidence) / 2
        lower = np.quantile(self.distribution, alpha, axis=1)
        upper = np.quantile(self.distribution, 1 - alpha, axis=1)
        return {"lower": lower, "upper": upper}


@dataclass(frozen=True)
class GPDFit:
    """Generalized Pareto Distribution fit results.

    Attributes
    ----------
    xi : float
        Shape parameter.
    beta : float
        Scale parameter.
    threshold : float
        Threshold value.
    """

    xi: float
    beta: float
    threshold: float


@dataclass(frozen=True)
class TestResult:
    """Immutable container for test results.

    Attributes
    ----------
    p_value_empirical : float
        Empirical p-value.
    test_statistic : float
        Test statistic value.
    null_statistics : numpy.ndarray
        Null distribution of test statistics.
    config : TestConfig
        Test configuration.
    p_value_gpd : float, optional
        GPD-adjusted p-value.
    gpd_fit : GPDFit, optional
        GPD fit parameters.
    """

    p_value_empirical: float
    test_statistic: float
    null_statistics: np.ndarray
    config: TestConfig
    p_value_gpd: Optional[float] = None
    gpd_fit: Optional[GPDFit] = None

    def __post_init__(self):
        if not isinstance(self.null_statistics, np.ndarray):
            object.__setattr__(
                self, "null_statistics", np.asarray(self.null_statistics)
            )

    @property
    def p_value(self) -> float:
        """Get the best available p-value (GPD if available, otherwise empirical)."""
        if self.p_value_gpd is not None:
            return self.p_value_gpd
        return self.p_value_empirical


@dataclass(frozen=True)
class ComparisonResult:
    """Immutable container for SRA comparison results.

    Attributes
    ----------
    p_value : float
        P-value for the comparison.
    test_statistic : float
        Test statistic value.
    null_statistics : numpy.ndarray
        Null distribution of test statistics.
    config : TestConfig
        Test configuration.
    """

    p_value: float
    test_statistic: float
    null_statistics: np.ndarray
    config: TestConfig

    def __post_init__(self):
        if not isinstance(self.null_statistics, np.ndarray):
            object.__setattr__(
                self, "null_statistics", np.asarray(self.null_statistics)
            )


###################
# Base Estimator
###################


class BaseEstimator:
    """Base class for all estimators."""

    def __init__(self):
        self.fitted_ = False

    def generate(self, X: np.ndarray, **kwargs) -> "BaseEstimator":
        """Fit the estimator to the data."""
        raise NotImplementedError("Subclasses must implement fit method")

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate input data."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                "Input must be a 2D array with shape (n_lists, list_length)"
            )
        # Ensure numeric type for internal processing
        if not np.issubdtype(X.dtype, np.number):
            # Attempt conversion if possible (e.g., object array of numbers)
            try:
                X = X.astype(float)
            except ValueError as e:
                raise ValueError(
                    "Input array must be numeric or convertible to numeric."
                ) from e
        return X


###################
# Utility Functions
###################


# Use the robust converter instead of the old _reshape_rank_matrix
def _reshape_to_item_cols(
    ranked_lists_array: np.ndarray,
) -> Tuple[np.ndarray, Dict[float, int]]:
    """
    Converts a LISTROW_RANKCOL array (item IDs in cells) to a
    LISTROW_ITEMCOL array (ranks in cells, columns indexed by item)
    using the robust convert_rank_data function.

    Parameters
    ----------
    ranked_lists_array : numpy.ndarray
        Array in LISTROW_RANKCOL format (shape num_lists, list_length).
        Values are numeric item IDs. Assumed validated & numeric.

    Returns
    -------
    rank_matrix : np.ndarray
        Array in LISTROW_ITEMCOL format (shape num_lists, num_unique_items).
        Values are ranks (1-based). Columns correspond to unique items found,
        ordered numerically by their original ID. NaNs indicate an item was
        not ranked in a list.
    id_to_index_map : Dict[float, int]
        Mapping from original numeric item ID to the column index in rank_matrix.

    Raises
    ------
    ValueError
        Propagated from convert_rank_data if input constraints are violated
        or expansion fails.
    ImportError
        If rank_array_adapter is not installed/found.
    """
    try:
        adapter_result = convert_rank_data(
            data=ranked_lists_array,
            from_shape=RankShape.LISTROW_RANKCOL,
            to_shape=RankShape.LISTROW_ITEMCOL,
            na_value=np.nan,  # Use NaN for missing ranks
        )
        return adapter_result.data, adapter_result.id_to_index_mapping
    except NameError:  # If convert_rank_data was not imported
        raise ImportError(
            "rank_array_adapter module is required for robust reshaping."
        )
    except ValueError as e:
        # Add context if helpful
        raise ValueError(
            f"Error reshaping rank matrix using convert_rank_data: {e}"
        )


def smooth_sra_window(
    sra_values: np.ndarray, window_size: int = 10
) -> np.ndarray:
    """
    Smooth the SRA curve using a rolling window average.

    Parameters
    ----------
    sra_values : numpy.ndarray
        1D array of SRA values.
    window_size : int, default=10
        Window size for rolling mean.

    Returns
    -------
    smoothed_values : numpy.ndarray
        1D array of smoothed SRA values.
    """
    n = len(sra_values)
    half_w = (window_size - 1) // 2

    if window_size <= 1 or n <= 1:
        return sra_values.copy()

    if window_size >= n:
        # If window is larger than array, return array average
        mean_val = np.nanmean(sra_values)  # Use nanmean
        return np.full(n, mean_val if not np.isnan(mean_val) else 0.0)

    # Use convolution for efficient rolling mean, handling NaNs
    weights = np.ones(window_size) / window_size
    # Pad with edge values to handle boundaries better than zero padding
    padded_sra = np.pad(sra_values.astype(float), half_w, mode="edge")

    # Handle NaNs: Calculate weighted sum and count of non-NaNs separately
    valid_mask = ~np.isnan(padded_sra)
    padded_sra_zeros = np.where(valid_mask, padded_sra, 0)

    weighted_sum = np.convolve(padded_sra_zeros, weights, mode="valid")
    valid_count = (
        np.convolve(valid_mask.astype(float), weights, mode="valid")
        * window_size
    )

    # Avoid division by zero where count is zero
    smoothed = np.full_like(weighted_sum, np.nan)
    non_zero_mask = valid_count > 1e-9  # Use tolerance for float comparison
    smoothed[non_zero_mask] = weighted_sum[non_zero_mask] / (
        valid_count[non_zero_mask] / window_size
    )

    # If any result is NaN (e.g., all NaNs in window), fill with 0 or another strategy?
    smoothed = np.nan_to_num(smoothed, nan=0.0)

    # Ensure output has the same length as input
    if len(smoothed) > n:
        # This might happen depending on convolution implementation details, trim if needed
        smoothed = smoothed[:n]
    elif len(smoothed) < n:
        # Should not happen with 'valid' mode and correct padding, but handle defensively
        smoothed = np.pad(smoothed, (0, n - len(smoothed)), mode="edge")

    return smoothed


def _aggregator(diffs: np.ndarray, style: str = "l2") -> float:
    """
    Aggregate a vector of differences. Handles NaNs by ignoring them.

    Parameters
    ----------
    diffs : numpy.ndarray
        1D array of differences.
    style : str, default="l2"
        Aggregation style; "max" returns the maximum, "l2" returns the sum of squares.

    Returns
    -------
    aggregated_value : float
        Aggregated value. Returns 0.0 if diffs contains only NaNs.
    """
    valid_diffs = diffs[~np.isnan(diffs)]
    if valid_diffs.size == 0:
        return 0.0

    if style == "max":
        return np.max(valid_diffs)
    elif style == "l2":  # Default to l2 if style is not 'max'
        return np.sum(valid_diffs**2)
    else:  # Should not happen due to TestConfig validation, but safer
        return np.sum(valid_diffs**2)


def _calculate_gpd_pvalue(
    T_obs: float, T_null: np.ndarray, threshold_quantile: float = 0.90
) -> dict[str, Any]:
    """
    Calculate p-value using Generalized Pareto Distribution (GPD) for extreme values.

    Parameters
    ----------
    T_obs : float
        The observed test statistic
    T_null : numpy.ndarray
        Array of null test statistics
    threshold_quantile : float, default=0.90
        Quantile of T_null to use as threshold for GPD fitting

    Returns
    -------
    result : dict
        Dictionary containing:
        - p_value_gpd: GPD-adjusted p-value
        - gpd_fit: Dictionary with GPD parameters (xi, beta, threshold)
        - applied_gpd: Boolean indicating if GPD was applied
    """
    # Filter NaNs from null distribution before calculating quantile/threshold
    T_null_valid = T_null[~np.isnan(T_null)]
    if T_null_valid.size == 0:
        warnings.warn("Null distribution contains only NaNs. Cannot apply GPD.")
        # Return empirical based on original T_null which might also be all NaN
        # Avoid division by zero if len(T_null) is 0
        emp_p = np.nan if len(T_null) == 0 else np.mean(T_null >= T_obs)
        return {
            "p_value_gpd": emp_p,
            "gpd_fit": None,
            "applied_gpd": False,
        }

    threshold = np.quantile(T_null_valid, threshold_quantile)

    # If observed statistic isn't extreme, don't apply GPD
    # Also handle case where T_obs might be NaN
    if np.isnan(T_obs) or T_obs <= threshold:
        # Calculate empirical p-value using original T_null to handle potential NaNs correctly
        emp_p = np.nan if len(T_null) == 0 else np.mean(T_null >= T_obs)
        return {
            "p_value_gpd": emp_p,
            "gpd_fit": None,
            "applied_gpd": False,
        }

    tail_data = T_null_valid[T_null_valid > threshold]

    # Check if we have enough tail points for stable fitting
    min_tail_points = 30  # Minimum number of points for reliable GPD fit
    if tail_data.size < min_tail_points:
        emp_p = np.nan if len(T_null) == 0 else np.mean(T_null >= T_obs)
        return {
            "p_value_gpd": emp_p,
            "gpd_fit": None,
            "applied_gpd": False,
        }

    # Fit GPD to excesses
    excesses = tail_data - threshold
    try:
        # floc=0 fixes the location parameter to 0 for excesses
        xi, _, beta = genpareto.fit(excesses, floc=0)

        # Check for potentially unstable fit parameters
        if beta <= 0:
            raise ValueError("GPD fit resulted in non-positive scale (beta).")
        # Consider adding checks for extreme xi values if needed

        excess_obs = T_obs - threshold
        # Use survival function (sf = 1 - cdf) for P(X > x)
        tail_prob = genpareto.sf(excess_obs, c=xi, loc=0, scale=beta)

        # Proportion of data below threshold (using original T_null)
        prop_below_threshold = np.mean(T_null < threshold)
        p_value_gpd = (1 - prop_below_threshold) * tail_prob

        # Ensure p-value is valid
        p_value_gpd = np.clip(p_value_gpd, 0.0, 1.0)

        return {
            "p_value_gpd": p_value_gpd,
            "gpd_fit": GPDFit(xi=xi, beta=beta, threshold=threshold),
            "applied_gpd": True,
        }
    except (RuntimeError, ValueError, FloatingPointError) as e:
        warnings.warn(
            f"GPD fitting failed: {str(e)}. Using empirical p-value instead."
        )
        emp_p = np.nan if len(T_null) == 0 else np.mean(T_null >= T_obs)
        return {
            "p_value_gpd": emp_p,
            "gpd_fit": None,
            "applied_gpd": False,
        }


def _generate_null_distribution(
    null_matrix: np.ndarray,
    aggregator_style: str,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Generate null distribution of test statistics using leave-one-out means.

    Parameters
    ----------
    null_matrix : numpy.ndarray
        Matrix of shape (n_depths, n_permutations) containing null SRA values.
    aggregator_style : str
        Aggregation style; "max" or "l2".

    Returns
    -------
    T_null : numpy.ndarray
        Array of test statistics from null distribution. Shape (n_permutations,).
        May contain NaNs if input columns are all NaN.
    """
    n_depths, B = null_matrix.shape
    if B <= 1:
        warnings.warn(
            "Cannot generate null distribution with <= 1 permutation."
        )
        return np.full(B, np.nan)

    current_n_jobs = n_jobs if n_jobs is not None and n_jobs != 0 else 1

    colsums = np.nansum(null_matrix, axis=1)
    valid_counts = np.sum(~np.isnan(null_matrix), axis=1)

    T_null_list = []

    if current_n_jobs == 1:
        for i in range(B):
            T_null_list.append(
                _worker_gnd_joblib(
                    current_col_from_null_matrix=null_matrix[:, i],
                    n_depths=n_depths,
                    colsums=colsums,
                    valid_counts=valid_counts,
                    aggregator_style=aggregator_style,
                    aggregator_func=_aggregator,
                )
            )
    else:
        T_null_list = Parallel(n_jobs=current_n_jobs, verbose=0)(
            delayed(_worker_gnd_joblib)(
                current_col_from_null_matrix=null_matrix[:, i],
                n_depths=n_depths,
                colsums=colsums,
                valid_counts=valid_counts,
                aggregator_style=aggregator_style,
                aggregator_func=_aggregator,
            )
            for i in range(B)
        )

    return np.array(T_null_list)


###################
# Core Algorithms
###################


def _nanmad(x: np.ndarray) -> float:
    """Calculate Median Absolute Deviation, ignoring NaNs."""
    x_valid = x[~np.isnan(x)]
    if len(x_valid) == 0:
        return np.nan
    med = np.median(x_valid)
    # Scale factor 1.4826 assumes normality for consistency
    return np.median(np.abs(x_valid - med)) * 1.4826


def compute_sra(
    ranked_lists: np.ndarray,
    config: SRAConfig,
    nitems: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> SRAResult:
    """
    Compute the Sequential Rank Agreement (SRA) for a set of ranked lists.

    This is the pure functional core of the SRA algorithm. It handles the case
    of incomplete lists by imputing missing values through random resampling.
    It uses a robust method to reshape ranks for calculation.

    Parameters
    ----------
    ranked_lists : numpy.ndarray
        Array of shape (n_lists, list_length) where each row is a ranked list
        containing numeric item IDs (assumed 1-based).
    config : SRAConfig
        Configuration for SRA computation.
    nitems : int, optional
        Total number of items in the universe (determines the max item ID, assumed 1 to nitems).
        If None, inferred as max(ranked_lists) if possible, otherwise from list_length.
        Crucial for correct imputation and sizing of `when_included`.

    Returns
    -------
    result : SRAResult
        Container for SRA computation results.
    """
    # Validate and prepare input data
    if not isinstance(ranked_lists, np.ndarray):
        ranked_lists = np.array(ranked_lists)
    if not np.issubdtype(ranked_lists.dtype, np.number):
        raise ValueError("Input ranked_lists must be numeric.")

    if rng is None:
        rng = np.random.default_rng()

    ranked_lists = ranked_lists.astype(float)  # Work with floats internally
    num_lists, list_length = ranked_lists.shape

    # Determine nitems (total universe size)
    original_nitems = nitems
    if nitems is None:
        # Infer nitems cautiously
        present_ids = ranked_lists[~np.isnan(ranked_lists)]
        if present_ids.size > 0:
            max_id = int(np.nanmax(present_ids))
            # Use max ID found, or list_length if max ID is smaller (less likely)
            nitems = max(max_id, list_length)
            if nitems > list_length and original_nitems is None:
                warnings.warn(
                    f"Inferred nitems={nitems} from max item ID found, "
                    f"which is greater than list_length={list_length}. Ensure this is intended."
                )
        else:
            nitems = list_length  # Fallback if all NaNs or empty
        if nitems == 0:
            raise ValueError("Cannot compute SRA with nitems=0.")
    elif not isinstance(nitems, int) or nitems <= 0:
        raise ValueError("nitems must be a positive integer.")

    if list_length < nitems:
        # Pad with missing values (NaN) if list_length is smaller than universe size
        pad_width = nitems - list_length
        pad = np.full((num_lists, pad_width), np.nan)
        ranked_lists = np.concatenate([ranked_lists, pad], axis=1)
        list_length = nitems  # Update list_length to reflect padding
    elif list_length > nitems:
        # Truncate lists if they are longer than the specified universe size
        warnings.warn(
            f"Input list_length ({list_length}) > specified nitems ({nitems}). Truncating lists."
        )
        ranked_lists = ranked_lists[:, :nitems]
        list_length = nitems

    # Prepare epsilon array
    if isinstance(config.epsilon, (int, float)):
        epsilons = np.full(nitems, float(config.epsilon))
    else:
        # Ensure numpy array and correct type
        epsilons = np.asarray(config.epsilon, dtype=float)
        if epsilons.ndim == 0:  # Handle scalar array
            epsilons = np.full(nitems, float(epsilons))
        elif epsilons.shape != (nitems,):
            raise ValueError(
                f"Length of epsilon array ({len(epsilons)}) must match nitems ({nitems})"
            )

    # Store SRA curves from each resample
    sra_curves = np.full((config.B, nitems), np.nan)  # Initialize with NaN
    # Initialize when_included for the first bootstrap run (b=0)
    # Use nitems + 1 as initial value, convert to inf later
    when_included = np.full(nitems, nitems + 1, dtype=float)
    id_to_index_map_bs0 = None  # Store map from first bootstrap

    # For each resample, impute missing values and compute the SRA curve
    all_items_set = set(range(1, nitems + 1))  # Universe of item IDs

    for b in range(config.B):
        imputed = np.empty_like(ranked_lists)

        for i in range(num_lists):
            list_i = ranked_lists[i, :]
            observed_mask = ~np.isnan(list_i)
            observed = list_i[observed_mask]

            # Ensure observed IDs are integers for set operations
            if observed.size > 0:
                # Check for non-integer values that might cause issues
                if not np.all(np.isclose(observed, np.round(observed))):
                    warnings.warn(
                        f"List {i} contains non-integer item IDs. Casting to int."
                    )
                observed_int = observed.astype(int)
                # Validate IDs are within expected range [1, nitems]
                if np.any(observed_int < 1) or np.any(observed_int > nitems):
                    invalid_ids = observed_int[
                        (observed_int < 1) | (observed_int > nitems)
                    ]
                    raise ValueError(
                        f"List {i} contains invalid item ID(s) (e.g., {invalid_ids[0]}) "
                        f"outside the expected range [1, {nitems}]."
                    )
                observed_set = set(observed_int)
            else:
                observed_set = set()

            missing_idx = np.where(~observed_mask)[0]
            missing_items = np.array(
                list(all_items_set - observed_set), dtype=int
            )

            if missing_items.size > 0:
                rng.shuffle(missing_items)

            new_list = list_i.copy()
            # Impute missing values
            num_to_impute = min(len(missing_idx), len(missing_items))
            if num_to_impute < len(missing_idx):
                warnings.warn(
                    f"List {i}, Resample {b}: More missing slots ({len(missing_idx)}) "
                    f"than available unique items to impute ({len(missing_items)}). "
                    f"Leaving {len(missing_idx) - num_to_impute} slots as NaN."
                )

            imputed_indices = missing_idx[:num_to_impute]
            new_list[imputed_indices] = missing_items[:num_to_impute]
            imputed[i, :] = new_list

        # Use the robust reshaping function
        try:
            rank_mat, id_to_index_map = _reshape_to_item_cols(imputed)
            if b == 0:
                id_to_index_map_bs0 = id_to_index_map  # Save for when_included
        except ValueError as e:
            raise ValueError(
                f"Error during rank matrix reshaping in bootstrap {b}: {e}"
            )

        # rank_mat shape: (num_lists, num_unique_items_found)
        # Values are ranks (1-based), NaNs indicate absence.

        # Calculate disagreements per item (across lists)
        if config.metric.lower() == "sd":
            # np.nanvar ignores NaNs, ddof=1 for sample variance
            disagreements = np.nanvar(rank_mat, axis=0, ddof=1)
        elif config.metric.lower() == "mad":
            # Apply nanmad across lists (axis 0) for each item (column)
            disagreements = np.apply_along_axis(_nanmad, 0, rank_mat)
            # Replace NaNs (e.g., item never appeared) with 0 disagreement? Or keep NaN?
            # Keeping NaN seems safer, let downstream handle it.
            # disagreements = np.nan_to_num(disagreements, nan=0.0) # Old: filled with 0
        else:
            # This case should be caught by SRAConfig validation
            raise ValueError(
                f"Unsupported metric: {config.metric}"
            )  # Should not happen

        # disagreements shape: (num_unique_items_found,)

        sra_curve_b = np.full(
            nitems, np.nan
        )  # Initialize curve for this bootstrap run

        if b == 0 and id_to_index_map_bs0 is None:
            # Handle case where reshaping failed or returned no map on first run
            warnings.warn(
                "Could not obtain item ID map on first bootstrap; 'when_included' will not be calculated."
            )

        # Build inverse map for when_included (only need to do once)
        index_to_id_map_bs0 = (
            {v: k for k, v in id_to_index_map_bs0.items()}
            if id_to_index_map_bs0
            else {}
        )

        for d in range(1, nitems + 1):  # Iterate through depths 1 to nitems
            # Proportion of lists where item rank is <= d
            # rank_mat contains ranks (1-based) or NaN
            prop = np.nanmean(
                (rank_mat <= d).astype(float), axis=0
            )  # nanmean ignores NaNs

            current_epsilon = epsilons[d - 1]

            # Indices of items (columns in rank_mat) satisfying the condition
            depth_set_indices = np.where(prop > current_epsilon)[0]

            # Update when_included based on the first bootstrap run (b=0)
            if b == 0 and len(index_to_id_map_bs0) > 0:
                for item_col_idx in depth_set_indices:
                    # Map column index back to original item ID
                    original_item_id = index_to_id_map_bs0.get(item_col_idx)
                    if original_item_id is not None:
                        # Convert 1-based item ID to 0-based index for when_included
                        when_included_idx = int(original_item_id) - 1
                        # Check bounds just in case
                        if 0 <= when_included_idx < nitems:
                            if d < when_included[when_included_idx]:
                                when_included[when_included_idx] = d

            # Calculate SRA value for depth d
            if depth_set_indices.size > 0:
                # Get disagreements for items in the current depth set
                # disagreements array corresponds to rank_mat columns
                disagreements_at_depth = disagreements[depth_set_indices]
                # Calculate mean, ignoring potential NaNs in disagreements
                mean_disagreement_at_depth = np.nanmean(disagreements_at_depth)
                # If all disagreements were NaN, nanmean returns NaN. Handle this?
                sra_curve_b[d - 1] = np.nan_to_num(
                    mean_disagreement_at_depth, nan=0.0
                )  # Fill NaN result with 0
            else:
                sra_curve_b[d - 1] = 0.0  # No items in set, 0 disagreement

        sra_curves[b, :] = sra_curve_b

    # Average SRA curves across bootstrap runs, ignoring NaNs
    avg_sra = np.nanmean(sra_curves, axis=0)

    # If metric is sd, take sqrt. Handle NaNs safely.
    if config.metric.lower() == "sd":
        # Avoid RuntimeWarning for sqrt(NaN) and sqrt(negative)
        with np.errstate(invalid="ignore"):  # Suppress warning locally
            avg_sra_sqrt = np.sqrt(
                np.maximum(0, avg_sra)
            )  # Use maximum(0,...) for tiny negatives
        avg_sra = np.where(np.isnan(avg_sra), np.nan, avg_sra_sqrt)

    # Finalize when_included: replace placeholder with inf
    when_included[when_included > nitems] = np.inf

    # Fill any remaining NaNs in avg_sra with 0? Consistent with empty set case.
    avg_sra = np.nan_to_num(avg_sra, nan=0.0)

    return SRAResult(values=avg_sra, config=config, when_included=when_included)


def random_list_sra(
    ranked_lists: np.ndarray | list[list[float]],
    config: SRAConfig,
    n_permutations: int = 100,
    nitems: Optional[int] = None,
    n_jobs: int = 1,
) -> RandomListSRAResult:
    """
    Generate null distribution for SRA by permuting lists, respecting original list lengths.

    Parameters
    ----------
    ranked_lists : numpy.ndarray or list of lists
        Array or list of shape (n_lists, list_length) where each row/list
        contains numeric item IDs (assumed 1-based).
    config : SRAConfig
        Configuration for SRA computation (used by compute_sra).
    n_permutations : int, default=100
        Number of permutations to generate.
    nitems : int, optional
        Total number of items in the universe (max item ID). If None, inferred.
        Crucial for defining the universe from which permutations are drawn.

    Returns
    -------
    result : RandomListSRAResult
        Container for null distribution results.
    """

    if not isinstance(ranked_lists, np.ndarray):
        try:
            if isinstance(ranked_lists, list) and ranked_lists:
                max_len = (
                    max(len(sublist) for sublist in ranked_lists)
                    if ranked_lists
                    else 0
                )
                padded_lists = []
                for sublist in ranked_lists:
                    padded_sublist = list(sublist) + [np.nan] * (
                        max_len - len(sublist)
                    )
                    padded_lists.append(padded_sublist)
                ranked_lists_arr = np.array(padded_lists, dtype=float)
            else:
                ranked_lists_arr = np.array(ranked_lists, dtype=float)
        except (ValueError, TypeError):
            raise TypeError(
                "Input ranked_lists must be array-like and numeric or list of numeric lists."
            )
    else:
        ranked_lists_arr = ranked_lists.astype(float)  # Ensure float

    if ranked_lists_arr.ndim != 2:
        if (
            ranked_lists_arr.size == 0
            and isinstance(ranked_lists, list)
            and not ranked_lists
        ):
            num_lists, original_list_length = 0, 0
        else:
            raise ValueError("Input ranked_lists must be 2D.")
    else:
        num_lists, original_list_length = ranked_lists_arr.shape

    real_lengths = np.sum(~np.isnan(ranked_lists_arr), axis=1).astype(int)

    final_nitems = nitems
    if final_nitems is None:
        present_ids = ranked_lists_arr[~np.isnan(ranked_lists_arr)]
        if present_ids.size > 0:
            max_id = int(np.nanmax(present_ids))
            final_nitems = max(max_id, original_list_length, 1)
        else:
            final_nitems = max(original_list_length, 1)
        if nitems is None and final_nitems > original_list_length:
            warnings.warn(
                f"Inferred nitems={final_nitems} from max item ID found, "
                f"which is greater than original list_length={original_list_length}. Using this for permutations."
            )
    elif (
        not isinstance(final_nitems, int) or final_nitems <= 0
    ):  # nitems was passed
        raise ValueError("nitems must be a positive integer.")
    # final_nitems is now set

    # Note: processed_ranked_lists from original code is not directly used by the worker
    # as worker reconstructs permutations from scratch using all_possible_ids.

    all_possible_ids = np.arange(1, final_nitems + 1)
    sra_results = []

    # Actual number of jobs for joblib (joblib handles n_jobs=-1 itself)
    # If n_jobs is None or 0, treat as 1 (serial)
    current_n_jobs = n_jobs if n_jobs is not None and n_jobs != 0 else 1

    if current_n_jobs == 1:
        # Ensure reproducibility for serial path if a global seed was set before calling this
        # or match the seeding strategy of the parallel path for consistency.
        master_seed_seq = np.random.SeedSequence()
        worker_seeds_entropy = [
            s.generate_state(1)[0]  # <- unique 32-bit integer
            for s in master_seed_seq.spawn(n_permutations)
        ]

        for i in range(n_permutations):
            sra_results.append(
                _worker_rls_joblib(
                    num_lists=num_lists,
                    final_nitems=final_nitems,
                    real_lengths=real_lengths,
                    all_possible_ids=all_possible_ids,
                    sra_config=config,
                    compute_sra_func=compute_sra,
                    rng_entropy=worker_seeds_entropy[i],
                )
            )
    else:
        master_seed_seq = np.random.SeedSequence()
        worker_seeds_entropy = [
            s.generate_state(1)[0]  # unique 32-bit integer
            for s in master_seed_seq.spawn(n_permutations)
        ]

        sra_results = Parallel(n_jobs=current_n_jobs, verbose=0)(
            delayed(_worker_rls_joblib)(
                num_lists=num_lists,
                final_nitems=final_nitems,
                real_lengths=real_lengths,
                all_possible_ids=all_possible_ids,
                sra_config=config,
                compute_sra_func=compute_sra,
                rng_entropy=worker_seeds_entropy[i],
            )
            for i in range(n_permutations)
        )

    if not sra_results:
        null_distribution = np.empty((final_nitems, 0), dtype=float)
    else:
        null_distribution = np.column_stack(sra_results)

    return RandomListSRAResult(
        distribution=null_distribution,
        config=config,
        n_permutations=n_permutations,
    )


def test_sra(
    observed_sra: np.ndarray,
    null_distribution: np.ndarray,
    config: TestConfig,
    n_jobs: int = 1,
) -> TestResult:
    """
    Test observed SRA values against null distribution.

    Parameters
    ----------
    observed_sra : numpy.ndarray
        Observed SRA values (1D array).
    null_distribution : numpy.ndarray
        Null distribution of SRA values (2D array, shape n_depths x n_permutations).
    config : TestConfig
        Configuration for the test.

    Returns
    -------
    result : TestResult
        Container for test results.
    """
    observed_sra = np.asarray(observed_sra)
    null_distribution = np.asarray(null_distribution)

    if observed_sra.ndim != 1:
        raise ValueError("observed_sra must be 1D.")
    if null_distribution.ndim != 2:
        raise ValueError("null_distribution must be 2D.")
    if observed_sra.shape[0] != null_distribution.shape[0]:
        raise ValueError(
            "observed_sra and null_distribution must have the same number of depths."
        )

    current_n_jobs = n_jobs if n_jobs is not None and n_jobs != 0 else 1
    null_distribution_smooth = null_distribution

    # Apply smoothing if requested
    if config.window > 1:
        observed_sra_smooth = smooth_sra_window(observed_sra, config.window)

        if current_n_jobs == 1:
            null_distribution_smooth = np.apply_along_axis(
                lambda n: smooth_sra_window(n, config.window),
                0,
                null_distribution,
            )
        else:
            smoothed_cols = Parallel(n_jobs=current_n_jobs, verbose=0)(
                delayed(_worker_smooth_col_joblib)(
                    col_data=null_distribution[:, i],
                    window_size=config.window,
                    smooth_sra_window_func=smooth_sra_window,
                )
                for i in range(null_distribution.shape[1])
            )
            if smoothed_cols:
                null_distribution_smooth = np.column_stack(smoothed_cols)
            else:
                null_distribution_smooth = np.empty(
                    (null_distribution.shape[0], 0)
                )

    else:
        observed_sra_smooth = observed_sra
        null_distribution_smooth = null_distribution

    # Compute test statistic for observed data
    # Calculate mean of the null distribution *per depth*, ignoring NaNs
    mean_null_sra = np.nanmean(null_distribution_smooth, axis=1)
    diffs_obs = np.abs(observed_sra_smooth - mean_null_sra)
    T_obs = _aggregator(diffs_obs, style=config.style)

    # Generate null distribution of test statistics
    T_null = _generate_null_distribution(
        null_distribution_smooth, config.style, n_jobs=current_n_jobs
    )

    # Compute empirical p-value (handle potential NaNs in T_null)
    # Compare T_obs with valid (non-NaN) values in T_null
    valid_T_null = T_null[~np.isnan(T_null)]
    if valid_T_null.size == 0:
        # Handle case where null generation failed completely
        p_value_empirical = np.nan
        warnings.warn(
            "Could not compute empirical p-value; null distribution generation yielded only NaNs."
        )
    elif np.isnan(T_obs):
        p_value_empirical = np.nan
        warnings.warn(
            "Could not compute empirical p-value; observed test statistic is NaN."
        )
    else:
        # Count how many null stats are >= observed stat
        # Add 1 to numerator and denominator for adjusted p-value
        p_value_empirical = (np.sum(valid_T_null >= T_obs) + 1) / (
            len(valid_T_null) + 1
        )

    # Use GPD if requested
    gpd_results = None
    p_value_gpd = None
    gpd_fit = None

    # Only apply GPD if empirical p-value and T_obs are valid
    if (
        config.use_gpd
        and not np.isnan(p_value_empirical)
        and not np.isnan(T_obs)
        and valid_T_null.size > 0
    ):
        # Pass only valid T_null values to GPD calculation
        gpd_results = _calculate_gpd_pvalue(
            T_obs, valid_T_null, config.threshold_quantile
        )
        p_value_gpd = gpd_results["p_value_gpd"]
        gpd_fit = gpd_results["gpd_fit"]
        # If GPD failed and returned empirical, set p_value_gpd to None?
        # Let's keep the potentially empirical value returned by _calculate_gpd_pvalue
        # It handles internal failures gracefully.

    return TestResult(
        p_value_empirical=p_value_empirical,
        test_statistic=T_obs,
        null_statistics=T_null,
        config=config,
        p_value_gpd=p_value_gpd,
        gpd_fit=gpd_fit,
    )


def compare_sra(
    sra1: np.ndarray,
    sra2: np.ndarray,
    null1: np.ndarray,
    null2: np.ndarray,
    config: TestConfig,
    n_jobs: int = 1,
) -> ComparisonResult:
    """
    Compare two SRA curves based on their deviation from their respective nulls,
    preserving permutation pairing.

    Parameters
    ----------
    sra1 : numpy.ndarray
        First SRA curve (1D array).
    sra2 : numpy.ndarray
        Second SRA curve (1D array).
    null1 : numpy.ndarray
        Null distribution for first curve (2D array, depths x permutations).
    null2 : numpy.ndarray
        Null distribution for second curve (2D array, depths x permutations).
        Must have the same number of permutations as null1.
    config : TestConfig
        Configuration for the test (style, window, tails).

    Returns
    -------
    result : ComparisonResult
        Container for comparison results.

    Raises
    ------
    ValueError
        If inputs have incompatible shapes or null distributions have different
        numbers of permutations.
    """
    sra1, sra2 = np.asarray(sra1), np.asarray(sra2)
    null1, null2 = np.asarray(null1), np.asarray(null2)

    if sra1.ndim != 1 or sra2.ndim != 1:
        raise ValueError("SRA curves must be 1D arrays.")
    if null1.ndim != 2 or null2.ndim != 2:
        raise ValueError("Null distributions must be 2D arrays.")
    if (
        sra1.shape[0] != sra2.shape[0]
        or sra1.shape[0] != null1.shape[0]
        or sra1.shape[0] != null2.shape[0]
    ):
        raise ValueError(
            "All inputs must have the same number of depths (first dimension)."
        )
    if null1.shape[1] != null2.shape[1]:
        raise ValueError(
            "Null distributions must have the same number of permutations (columns) for comparison."
        )
    if null1.shape[1] == 0:
        raise ValueError("Null distributions cannot have zero permutations.")

    current_n_jobs = n_jobs if n_jobs is not None and n_jobs != 0 else 1

    sra1_smooth, sra2_smooth = sra1, sra2
    null1_smooth, null2_smooth = null1, null2

    if config.window > 1:
        sra1_smooth = smooth_sra_window(sra1, config.window)
        sra2_smooth = smooth_sra_window(sra2, config.window)

        if current_n_jobs == 1:
            null1_smooth = np.apply_along_axis(
                lambda n: smooth_sra_window(n, config.window), 0, null1
            )
            null2_smooth = np.apply_along_axis(
                lambda n: smooth_sra_window(n, config.window), 0, null2
            )
        else:
            smoothed_cols1 = Parallel(n_jobs=current_n_jobs, verbose=0)(
                delayed(_worker_smooth_col_joblib)(
                    col_data=null1[:, i],
                    window_size=config.window,
                    smooth_sra_window_func=smooth_sra_window,
                )
                for i in range(null1.shape[1])
            )
            if smoothed_cols1:
                null1_smooth = np.column_stack(smoothed_cols1)
            else:
                null1_smooth = np.empty((null1.shape[0], 0))

            smoothed_cols2 = Parallel(n_jobs=current_n_jobs, verbose=0)(
                delayed(_worker_smooth_col_joblib)(
                    col_data=null2[:, i],
                    window_size=config.window,
                    smooth_sra_window_func=smooth_sra_window,
                )
                for i in range(null2.shape[1])
            )
            if smoothed_cols2:
                null2_smooth = np.column_stack(smoothed_cols2)
            else:
                null2_smooth = np.empty((null2.shape[0], 0))
    else:
        sra1_smooth = sra1
        sra2_smooth = sra2
        null1_smooth = null1
        null2_smooth = null2

    mean_null1 = np.nanmean(null1_smooth, axis=1)
    mean_null2 = np.nanmean(null2_smooth, axis=1)
    diffs_obs1 = np.abs(sra1_smooth - mean_null1)
    diffs_obs2 = np.abs(sra2_smooth - mean_null2)
    T_obs1 = _aggregator(diffs_obs1, config.style)
    T_obs2 = _aggregator(diffs_obs2, config.style)
    T_obs = T_obs1 - T_obs2

    T_null1 = _generate_null_distribution(
        null1_smooth, config.style, n_jobs=current_n_jobs
    )
    T_null2 = _generate_null_distribution(
        null2_smooth, config.style, n_jobs=current_n_jobs
    )

    T_null_diff = T_null1 - T_null2

    valid_T_null_diff = T_null_diff[~np.isnan(T_null_diff)]

    if valid_T_null_diff.size == 0:
        warnings.warn(
            "Could not compute comparison p-value; null difference distribution lacks valid values."
        )
        p_value = np.nan
    elif np.isnan(T_obs):
        warnings.warn(
            "Could not compute comparison p-value; observed test statistic difference is NaN."
        )
        p_value = np.nan
    else:
        if config.sra_comparison_tails == "one-tailed":
            p_value = (np.sum(valid_T_null_diff >= T_obs) + 1) / (
                len(valid_T_null_diff) + 1
            )
        elif config.sra_comparison_tails == "two-tailed":
            p_value = (
                np.sum(np.abs(valid_T_null_diff) >= np.abs(T_obs)) + 1
            ) / (len(valid_T_null_diff) + 1)
        else:  # Should be caught by TestConfig validation
            raise ValueError(f"Invalid tails: {config.sra_comparison_tails}")

    return ComparisonResult(
        p_value=p_value,
        test_statistic=T_obs,
        null_statistics=T_null_diff,
        config=config,
    )


###################
# Estimator Classes
###################


class RankData:
    """Utility for handling and transforming ranked data."""

    @staticmethod
    def from_scores(
        scores: list[list[float]], ascending: bool = True
    ) -> np.ndarray:
        """Convert lists of scores to ranks (e.g., p-values, correlations).

        Produces a LISTROW_ITEMCOL like array where columns implicitly represent items
        based on original position, and values are ranks.

        Parameters
        ----------
        scores : list of lists
            Each inner list contains numeric scores.
        ascending : bool, default=True
            If True, lower scores get lower ranks (e.g., for p-values).
            If False, higher scores get lower ranks (e.g., for correlations).

        Returns
        -------
        rank_matrix : numpy.ndarray
            Array where each row contains the corresponding ranks (1-based).
            Shape (n_lists, n_items). NaNs are preserved.
        """
        result = []
        max_len = 0

        # First pass to find max length
        for score_list in scores:
            max_len = max(max_len, len(score_list))

        if max_len == 0:
            return np.empty((len(scores), 0), dtype=float)

        # Second pass to compute ranks and pad
        for score_list in scores:
            scores_array = np.array(score_list, dtype=float)
            # Pad if necessary before ranking
            if len(scores_array) < max_len:
                padded_scores = np.pad(
                    scores_array,
                    (0, max_len - len(scores_array)),
                    mode="constant",
                    constant_values=np.nan,
                )
            else:
                padded_scores = scores_array

            mask = ~np.isnan(padded_scores)
            valid_scores = padded_scores[mask]

            # Get ranks (argsort of argsort gives ranks, 0-based)
            if valid_scores.size > 0:
                if ascending:
                    # Use scipy.stats.rankdata for better tie handling ('average', 'min', etc.)
                    # Using 'ordinal' breaks ties by order of appearance, giving unique ranks 1..N
                    valid_ranks = rankdata(
                        valid_scores, method="ordinal"
                    )  # Gives 1-based ranks
                else:
                    valid_ranks = rankdata(
                        -valid_scores, method="ordinal"
                    )  # Gives 1-based ranks
            else:
                valid_ranks = np.array([])

            ranks = np.full(max_len, np.nan)  # Initialize with NaN
            ranks[mask] = valid_ranks
            result.append(ranks)

        return np.array(result)

    @staticmethod
    def from_items(
        items_lists: list[list[Any]] | np.ndarray,
        na_strings: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, dict]:
        """Convert lists of items or numpy array of items to numeric ranks (LISTROW_RANKCOL).

        Assigns dense, 1-based numeric IDs to unique items found. Items that are
        not sortable or hashable (like dicts) will be converted to strings for mapping.

        Parameters
        ----------
        items_lists : list of lists or numpy.ndarray
            Either a list of lists where each inner list contains items in their ranked order,
            or a numpy array of items (often strings) with possible NA values.
        na_strings : Optional[list[str]], default=["", "nan", "na", "null", "none"]
            Case-insensitive strings to interpret as NA/missing values. Applied to both
            list and array inputs.

        Returns
        -------
        rank_matrix : numpy.ndarray
            Array in LISTROW_RANKCOL format. Shape (n_lists, max_list_length).
            Values are the assigned 1-based numeric IDs. NaNs represent missing items.
        item_mapping : dict
            Dictionary with 'item_to_id' (original item or its string rep -> 1-based int ID) and
            'id_to_item' (1-based int ID -> original item or its string rep) mappings.
        """
        processed_lists = []
        is_np_input = isinstance(items_lists, np.ndarray)

        # Consistent NA handling setup
        default_na = ["", "nan", "na", "null", "none"]
        current_na_strings = (
            na_strings if na_strings is not None else default_na
        )
        na_set_lower = set(s.lower() for s in current_na_strings)

        max_len = 0

        if is_np_input:
            items_array = items_lists
            current_max_len = (
                items_array.shape[1] if items_array.ndim == 2 else 0
            )

            if (
                items_array.ndim == 1 and items_array.size > 0
            ):  # Handle 1D array as single list
                items_array = items_array.reshape(1, -1)
                current_max_len = items_array.shape[1]
                is_np_input = False  # Treat as list of lists now
                items_lists = (
                    items_array.tolist()
                )  # Convert for list processing loop

            elif items_array.ndim != 2:
                if items_array.size == 0:
                    items_lists = []
                    is_np_input = False
                else:
                    raise ValueError(
                        "Input numpy array must be 2D or convertible to 1D->2D."
                    )

            if is_np_input:  # If it was originally 2D
                max_len = current_max_len  # Use shape[1]
                na_mask = np.zeros_like(items_array, dtype=bool)
                na_mask |= pd.isna(items_array)

                if (
                    np.issubdtype(items_array.dtype, np.character)
                    or items_array.dtype == object
                ):
                    try:
                        items_str = items_array.astype(str)
                        items_lower = np.char.lower(items_str)
                        for indicator in na_set_lower:
                            na_mask |= items_lower == indicator
                        na_mask |= np.char.strip(items_str) == ""
                    except (TypeError, ValueError):
                        pass

                for i in range(items_array.shape[0]):
                    row_list = [
                        items_array[i, j] if not na_mask[i, j] else None
                        for j in range(items_array.shape[1])
                    ]
                    processed_lists.append(row_list)

        if not is_np_input:  # Process list of lists (or converted 1D array)
            if not isinstance(items_lists, list):
                raise TypeError("Input must be list of lists or numpy array.")

            temp_processed_lists = []
            current_max_len = 0
            for i, sublist_orig in enumerate(items_lists):
                # Ensure sublist is actually a list or tuple before processing
                if not isinstance(sublist_orig, (list, tuple)):
                    try:
                        sublist = list(sublist_orig)  # Attempt conversion
                    except TypeError:
                        raise TypeError(
                            f"Inner element at index {i} is not list-like."
                        )
                else:
                    sublist = sublist_orig

                processed_row = []
                current_max_len = max(current_max_len, len(sublist))
                for item in sublist:
                    if item is None or pd.isna(item):
                        processed_row.append(None)
                    elif isinstance(item, str):
                        item_lower = item.lower()
                        if item_lower in na_set_lower or item.strip() == "":
                            processed_row.append(None)
                        else:
                            processed_row.append(item)  # Keep original case
                    else:
                        processed_row.append(item)
                temp_processed_lists.append(processed_row)

            processed_lists = temp_processed_lists
            max_len = current_max_len

        # Collect all non-None items from processed lists
        all_items_collected = []
        for items in processed_lists:
            all_items_collected.extend(
                [item for item in items if item is not None]
            )

        if not all_items_collected:
            warnings.warn("No valid items found in input.")
            return np.full((len(processed_lists), max_len), np.nan), {
                "item_to_id": {},
                "id_to_item": {},
            }

        unique_hashable = []
        seen_hashable = set()
        unique_unhashable = []

        for item in all_items_collected:
            # Use isinstance check which is generally safer than try-except hash()
            if isinstance(item, collections.abc.Hashable):
                if item not in seen_hashable:
                    seen_hashable.add(item)
                    unique_hashable.append(item)
            else:  # Unhashable item
                is_new = True
                for existing_unhashable in unique_unhashable:
                    try:
                        # Use equality check (==)
                        if item == existing_unhashable:
                            is_new = False
                            break
                    except Exception:
                        # If equality check fails, consider them different
                        pass
                if is_new:
                    unique_unhashable.append(item)

        # Combine the unique items found
        unique_items_raw = unique_hashable + unique_unhashable

        # Attempt to sort unique items for deterministic mapping
        items_for_mapping = []
        lookup_is_string = False
        try:
            # Sorting will likely fail if unique_unhashable is not empty
            # or if mixed types exist in unique_hashable.
            items_for_mapping = sorted(unique_items_raw)
        except TypeError:
            warnings.warn(
                "Items are not naturally sortable (e.g., mixed types, unhashable items); "
                "converting to string for deterministic ID assignment. "
                "Different items with the same string representation will get the same ID."
            )
            # Fallback: Convert all to strings, ensure uniqueness *after* conversion, then sort strings
            unique_items_str_dict = {}
            for item in unique_items_raw:
                str_repr = str(item)
                if str_repr not in unique_items_str_dict:
                    unique_items_str_dict[str_repr] = (
                        str_repr  # Store string key -> string value
                    )

            items_for_mapping = sorted(
                unique_items_str_dict.keys()
            )  # Sort the unique strings
            lookup_is_string = True

        # Assign dense 1-based IDs
        item_to_id = {
            item: id_val for id_val, item in enumerate(items_for_mapping, 1)
        }
        # Note: Keys in item_to_id are either original items or strings.
        # Values in id_to_item are also either original items or strings, matching keys.
        id_to_item = {id_val: item for item, id_val in item_to_id.items()}
        item_mapping = {"item_to_id": item_to_id, "id_to_item": id_to_item}

        # Convert each list to numeric IDs, preserving NA pattern (None -> np.nan)
        result_numeric = np.full(
            (len(processed_lists), max_len), np.nan, dtype=float
        )
        for i, items in enumerate(processed_lists):
            # Pad the list with None if it's shorter than max_len
            padded_items = items + [None] * (max_len - len(items))
            for j, item in enumerate(padded_items):
                if item is not None:
                    # Determine the key for lookup based on whether strings were used
                    lookup_key = str(item) if lookup_is_string else item
                    item_id = item_to_id.get(lookup_key)
                    if item_id is not None:
                        result_numeric[i, j] = item_id
                    # else: Item maps to None or key wasn't found (shouldn't happen if logic is correct)

        # Validation (optional, requires adapter module)
        try:
            _validate_rankcol_no_midrow_nulls(result_numeric)
            _validate_rankcol_no_duplicates(result_numeric)
        except NameError:  # Functions not found
            pass  # Ignore validation if not available
        except ImportError:  # Module not found
            pass
        except ValueError as e:
            warnings.warn(
                f"Internal validation failed after creating rank matrix: {e}"
            )

        return result_numeric, item_mapping

    @staticmethod
    def map_ids_to_items(
        id_array: np.ndarray, item_mapping: dict
    ) -> np.ndarray:
        """Convert array of numeric IDs back to original items (or their string reps).

        Parameters
        ----------
        id_array : numpy.ndarray
            Array containing numeric IDs (typically 1-based).
        item_mapping : dict
            Mapping dict returned by from_items method, containing 'id_to_item'.

        Returns
        -------
        item_array : numpy.ndarray
            Array with the same shape as id_array but with original items
            (or string reps if string conversion occurred) instead of IDs.
            NaNs and unmapped IDs result in None.
        """
        id_to_item = item_mapping.get("id_to_item")
        if id_to_item is None:
            raise ValueError(
                "Invalid item_mapping provided; missing 'id_to_item'."
            )

        # Create array of object dtype to hold items of any type
        result = np.full(id_array.shape, None, dtype=object)

        # Iterate and map IDs back to items
        for idx, val in np.ndenumerate(id_array):
            if not pd.isna(val):
                try:
                    item_id_int = int(val)
                    # Get original item or its string representation from the map
                    original_item_or_rep = id_to_item.get(item_id_int)
                    result[idx] = (
                        original_item_or_rep  # This is None if ID not found
                    )
                except (ValueError, TypeError):
                    result[idx] = None  # Treat non-int IDs as unmapped
            # else: Leave as None for NaNs

        return result


class SRA(BaseEstimator):
    """Sequential Rank Agreement estimator.

    Computes agreement between ranked lists as a function of list depth.

    Parameters
    ----------
    epsilon : float or array-like, default=0.0
        Threshold for an item to be included in S(d).
    metric : {'sd', 'mad'}, default='sd'
        Method to measure dispersion of ranks.
    B : int, default=1
        Number of bootstrap samples for handling missing values.
    """

    def __init__(
        self,
        epsilon: float | np.ndarray = 0.0,
        metric: Literal["sd", "mad"] = "sd",
        B: int = 1,
    ):
        super().__init__()
        self.config = SRAConfig(epsilon=epsilon, metric=metric, B=B)
        self.result_ = None

    def generate(self, X: np.ndarray, nitems: Optional[int] = None) -> "SRA":
        """
        Compute SRA values for ranked lists X.

        Parameters
        ----------
        X : array-like of shape (n_lists, list_length)
            Ranked lists data containing numeric item IDs (assumed 1-based).
            Use RankData.from_items() or RankData.from_scores() first
            if starting from item names or scores.
        nitems : int, optional
            Total number of items in the universe (max item ID). If None,
            inferred inside compute_sra. Providing it is recommended for clarity
            and consistency, especially if lists are truncated or padded.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_input(X)  # Ensures X is 2D numeric array
        self.result_ = compute_sra(X, self.config, nitems)
        self.fitted_ = True
        return self

    @require_generated
    def get_result(self) -> SRAResult:
        """Get the SRA computation result."""
        if self.result_ is None:
            # This should be caught by @require_generated, but defensive check
            raise ValueError("SRA result not available. Call 'generate' first.")
        return self.result_

    @require_generated
    def values(self) -> np.ndarray:
        """Get the SRA values."""
        return self.get_result().values

    @require_generated
    def when_included(self) -> Optional[np.ndarray]:
        """Get the depth at which each item was first included (or np.inf)."""
        # Check if when_included was successfully computed
        res = self.get_result()
        if res.when_included is None:
            warnings.warn(
                "'when_included' data is not available for this result."
            )
        return res.when_included

    @require_generated
    def smooth(self, window_size: int = 10) -> np.ndarray:
        """
        Smooth the SRA curve using a rolling window average.

        Parameters
        ----------
        window_size : int, default=10
            Size of the rolling window.

        Returns
        -------
        smoothed_values : ndarray
            Smoothed SRA curve.
        """
        return self.get_result().smooth(window_size)


class RandomListSRA(BaseEstimator):
    """Generate null distribution for SRA through permutation.

    Parameters
    ----------
    epsilon : float or array-like, default=0.0
        Threshold for an item to be included in S(d). Passed to compute_sra.
    metric : {'sd', 'mad'}, default='sd'
        Method to measure dispersion of ranks. Passed to compute_sra.
    B : int, default=1
        Number of bootstrap samples for handling missing values. Passed to compute_sra.
    n_permutations : int, default=100
        Number of permutations to generate.
    """

    def __init__(
        self,
        epsilon: float | np.ndarray = 0.0,
        metric: Literal["sd", "mad"] = "sd",
        B: int = 1,
        n_permutations: int = 100,
        n_jobs: int = 1,
    ):
        super().__init__()
        if n_permutations < 1:
            raise ValueError("n_permutations must be at least 1.")
        self.config = SRAConfig(epsilon=epsilon, metric=metric, B=B)
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs
        self.result_ = None

    def generate(
        self, X: np.ndarray, nitems: Optional[int] = None
    ) -> "RandomListSRA":
        """
        Generate null distribution of SRA values by permuting lists.

        Parameters
        ----------
        X : array-like of shape (n_lists, list_length)
            Ranked lists data containing numeric item IDs (assumed 1-based).
            Represents the structure (list lengths, number of lists) for permutation.
        nitems : int, optional
            Total number of items in the universe (max item ID). If None,
            inferred inside random_list_sra. Providing it is recommended.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_input(X)
        self.result_ = random_list_sra(
            X, self.config, self.n_permutations, nitems, n_jobs=self.n_jobs
        )
        self.fitted_ = True
        return self

    @require_generated
    def get_result(self) -> RandomListSRAResult:
        """Get the null distribution generation result."""
        if self.result_ is None:
            raise ValueError(
                "Null distribution not generated. Call 'generate' first."
            )
        return self.result_

    @require_generated
    def distribution(self) -> np.ndarray:
        """Get the null distribution matrix."""
        return self.get_result().distribution

    @require_generated
    def confidence_band(
        self, confidence: float = 0.95
    ) -> dict[str, np.ndarray]:
        """
        Compute confidence band for the null distribution.

        Parameters
        ----------
        confidence : float, default=0.95
            Confidence level (must be between 0 and 1).

        Returns
        -------
        band : dict
            Dictionary with 'lower' and 'upper' arrays.
        """
        if not 0 < confidence < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        return self.get_result().confidence_band(confidence)

    @require_generated
    def quantiles(self, probs: float | list[float]) -> dict[float, np.ndarray]:
        """
        Compute quantiles of the null distribution for each depth.

        Parameters
        ----------
        probs : float or list of float
            Probability point(s) at which to compute quantiles (e.g., 0.025, 0.5, 0.975).
            Each probability must be between 0 and 1.

        Returns
        -------
        quantiles : dict
            Dictionary mapping probabilities to quantile arrays.
        """
        if isinstance(probs, (int, float)):
            probs = [float(probs)]
        elif not isinstance(probs, list) or not all(
            isinstance(p, (int, float)) for p in probs
        ):
            raise TypeError("probs must be a float or a list of floats.")

        if not all(0 <= p <= 1 for p in probs):
            raise ValueError(
                "All quantile probabilities must be between 0 and 1."
            )

        result = {}
        dist = self.distribution()  # Get the null distribution matrix
        for prob in probs:
            # Calculate quantiles along the permutation axis (axis=1)
            result[prob] = np.nanquantile(dist, prob, axis=1)  # Use nanquantile
        return result


class SRATest(BaseEstimator):
    """Test for significance of SRA values against a null distribution.

    Parameters
    ----------
    style : {'l2', 'max'}, default='max'
        Method to aggregate differences.
    window : int, default=1
        Size of smoothing window. Use 1 for no smoothing.
    use_gpd : bool, default=False
        Whether to use generalized Pareto distribution for extreme p-values.
    threshold_quantile : float, default=0.90
        Quantile to use as threshold for GPD fitting.
    """

    def __init__(
        self,
        style: Literal["l2", "max"] = "max",
        window: int = 1,
        use_gpd: bool = False,
        threshold_quantile: float = 0.90,
        n_jobs: int = 1,
    ):
        super().__init__()
        self.config = TestConfig(
            style=style,
            window=window,
            use_gpd=use_gpd,
            threshold_quantile=threshold_quantile,
        )
        self.n_jobs = n_jobs
        self.result_ = None

    def generate(
        self,
        observed_sra: SRAResult | np.ndarray,
        null_dist: RandomListSRAResult | np.ndarray,
    ) -> "SRATest":
        """
        Test observed SRA values against null distribution.

        Parameters
        ----------
        observed_sra : SRAResult or array-like
            Observed SRA values (1D array) or result object.
        null_dist : RandomListSRAResult or array-like
            Null distribution of SRA values (2D array, depths x permutations)
            or result object.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Extract values if result objects are provided
        if isinstance(observed_sra, SRAResult):
            observed_values = observed_sra.values
        elif isinstance(observed_sra, (np.ndarray, list, tuple)):
            observed_values = np.asarray(observed_sra)
        else:
            raise TypeError("observed_sra must be SRAResult or array-like.")

        if isinstance(null_dist, RandomListSRAResult):
            null_values = null_dist.distribution
        elif isinstance(null_dist, (np.ndarray, list, tuple)):
            null_values = np.asarray(null_dist)
        else:
            raise TypeError(
                "null_dist must be RandomListSRAResult or array-like."
            )

        # Validate input dimensions after extraction
        if observed_values.ndim != 1:
            raise ValueError(
                "observed_sra must be 1D array or yield a 1D array."
            )
        if null_values.ndim != 2:
            raise ValueError("null_dist must be 2D array or yield a 2D array.")
        if observed_values.shape[0] != null_values.shape[0]:
            raise ValueError(
                "observed_sra and null_dist must have same number of depths (first dimension)."
            )

        self.result_ = test_sra(
            observed_values, null_values, self.config, self.n_jobs
        )
        self.fitted_ = True
        return self

    @require_generated
    def get_result(self) -> TestResult:
        """Get the test result."""
        if self.result_ is None:
            raise ValueError("Test not performed. Call 'generate' first.")
        return self.result_

    @require_generated
    def p_value(self) -> float:
        """Get the p-value (GPD-based if available, otherwise empirical)."""
        # Returns NaN if p-value calculation failed
        return self.get_result().p_value


class SRACompare(BaseEstimator):
    """Compare two SRA curves based on their deviation from respective nulls.

    Parameters
    ----------
    style : {'l2', 'max'}, default='max'
        Method to aggregate differences for comparison statistic.
    window : int, default=1
        Size of smoothing window. Use 1 for no smoothing.
    tails : {'one-tailed', 'two-tailed'}, default='one-tailed'
        Specifies the tails of the comparison test.
        "one-tailed": tests if sra1's deviation > sra2's deviation.
        "two-tailed": tests if sra1's deviation != sra2's deviation.
    n_jobs : int, default=1
        Number of parallel jobs to use.
    """

    def __init__(
        self,
        style: Literal["l2", "max"] = "max",
        window: int = 1,
        tails: Literal["one-tailed", "two-tailed"] = "one-tailed",
        n_jobs: int = 1,
    ):
        super().__init__()
        # GPD not typically used in comparison context
        self.config = TestConfig(
            style=style, window=window, use_gpd=False, sra_comparison_tails=tails
        )
        self.n_jobs = n_jobs
        self.result_ = None

    def generate(
        self,
        sra1: SRAResult | np.ndarray,
        sra2: SRAResult | np.ndarray,
        null1: RandomListSRAResult | np.ndarray,
        null2: RandomListSRAResult | np.ndarray,
    ) -> "SRACompare":
        """
        Compare two SRA curves.

        Parameters
        ----------
        sra1 : SRAResult or array-like
            First SRA curve (1D) or result object.
        sra2 : SRAResult or array-like
            Second SRA curve (1D) or result object.
        null1 : RandomListSRAResult or array-like
            Null distribution for first curve (2D) or result object.
        null2 : RandomListSRAResult or array-like
            Null distribution for second curve (2D) or result object.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Extract values if result objects are provided
        if isinstance(sra1, SRAResult):
            sra1_values = sra1.values
        elif isinstance(sra1, (np.ndarray, list, tuple)):
            sra1_values = np.asarray(sra1)
        else:
            raise TypeError("sra1 must be SRAResult or array-like.")

        if isinstance(sra2, SRAResult):
            sra2_values = sra2.values
        elif isinstance(sra2, (np.ndarray, list, tuple)):
            sra2_values = np.asarray(sra2)
        else:
            raise TypeError("sra2 must be SRAResult or array-like.")

        if isinstance(null1, RandomListSRAResult):
            null1_values = null1.distribution
        elif isinstance(null1, (np.ndarray, list, tuple)):
            null1_values = np.asarray(null1)
        else:
            raise TypeError("null1 must be RandomListSRAResult or array-like.")

        if isinstance(null2, RandomListSRAResult):
            null2_values = null2.distribution
        elif isinstance(null2, (np.ndarray, list, tuple)):
            null2_values = np.asarray(null2)
        else:
            raise TypeError("null2 must be RandomListSRAResult or array-like.")

        self.result_ = compare_sra(
            sra1_values,
            sra2_values,
            null1_values,
            null2_values,
            self.config,
            self.n_jobs,
        )
        self.fitted_ = True
        return self

    @require_generated
    def get_result(self) -> ComparisonResult:
        if self.result_ is None:
            raise ValueError("Comparison not performed. Call 'generate' first.")
        return self.result_

    @require_generated
    def p_value(self) -> float:
        """Get the p-value for the comparison."""
        # Returns NaN if p-value calculation failed
        return self.get_result().p_value


class RankPipeline:
    """Builder for common rank analysis pipelines.

    This utility class provides a fluent API for building common analysis
    pipelines for ranked data, making it easy to perform typical analyses
    with minimal code. Handles data preparation using RankData internally.

    Attributes
    ----------
    input_data : Any
        Stores the initial data provided (list of lists, scores, array).
    input_type : str
        Indicates the type of input ('items', 'scores', 'ranks').
    ranked_data : numpy.ndarray | None
        Numeric ranked data ready for SRA (LISTROW_RANKCOL format with 1-based IDs).
    item_mapping : dict | None
        Mapping generated by RankData.from_items, if used.
    nitems : int | None
        Total number of unique items, determined during data preparation.
    sra_config : SRAConfig | None
        Configuration used for SRA computation.
    sra : SRA | None
        SRA estimator instance.
    null_config : dict | None
        Configuration used for null distribution generation.
    null_dist : RandomListSRA | None
        Null distribution estimator instance.
    test_config : TestConfig | None
        Configuration used for significance testing.
    test : SRATest | None
        Test estimator instance.
    """

    def __init__(self):
        self._input_data = None
        self._input_type = None
        self.ranked_data = None
        self.item_mapping = None
        self.nitems = None
        self.sra_config = None
        self.sra = None
        self.null_config = None
        self.null_dist = None
        self.test_config = None
        self.test = None
        self._generated = (
            False  # Track if pipeline steps requiring data are done
        )

    def _prepare_data(self, nitems: Optional[int] = None) -> None:
        """Internal method to process input data into numeric ranks."""
        if self.ranked_data is not None:  # Already prepared
            return

        if self._input_data is None:
            raise ValueError(
                "No input data provided. Use with_... methods first."
            )

        if self._input_type == "ranks":
            # Input is assumed to be pre-formatted numeric ranks (LISTROW_RANKCOL)
            try:
                self.ranked_data = np.asarray(self._input_data, dtype=float)
            except ValueError:
                raise ValueError(
                    "Input data for 'with_ranked_data' must be numeric or convertible."
                )
            if self.ranked_data.ndim != 2:
                raise ValueError(
                    "Input data for 'with_ranked_data' must be 2D."
                )

            if nitems is not None:
                self.nitems = nitems
            else:
                # Infer nitems from max ID if possible
                present_ids = self.ranked_data[~np.isnan(self.ranked_data)]
                if present_ids.size > 0:
                    self.nitems = int(np.nanmax(present_ids))
                else:  # Fallback to width if all NaN or empty
                    self.nitems = self.ranked_data.shape[1]
            self.item_mapping = None  # No mapping for direct ranks

        elif self._input_type == "items":
            # Input is list of lists or array of items
            data, mapping = RankData.from_items(
                self._input_data["items_lists"],
                self._input_data.get("na_strings"),
            )
            self.ranked_data = data
            self.item_mapping = mapping
            # Use provided nitems or infer from mapping
            if nitems is not None:
                self.nitems = nitems
            else:
                self.nitems = len(
                    mapping["id_to_item"]
                )  # Number of unique items found

        elif self._input_type == "scores":
            # Input is list of lists of scores
            data = RankData.from_scores(
                self._input_data["scores"], self._input_data["ascending"]
            )
            self.ranked_data = data  # This is LISTROW_ITEMCOL format
            self.item_mapping = None
            # Use provided nitems if available
            if nitems is not None:
                self.nitems = nitems
            # Otherwise, cannot easily determine nitems
            # Needs conversion from ITEMCOL to RANKCOL first.
            raise NotImplementedError(
                "Pipeline support for 'with_scores' requires conversion step "
                "to LISTROW_RANKCOL format expected by SRA. Use 'with_items_lists' "
                "or 'with_ranked_data' (with numeric item IDs) instead."
            )

        else:
            raise ValueError("Invalid input type state.")

        if self.ranked_data is None or self.nitems is None:
            raise ValueError("Data preparation failed.")
        self._generated = True

    def with_ranked_data(
        self, data: np.ndarray | list[list[float]]
    ) -> "RankPipeline":
        """
        Set the ranked data directly. Assumes LISTROW_RANKCOL format with
        numeric (1-based recommended) item IDs.

        Parameters
        ----------
        data : array-like of shape (n_lists, list_length)
            Ranked lists data with numeric item IDs.

        Returns
        -------
        self : RankPipeline
            For method chaining.
        """
        if self._input_data is not None:
            raise ValueError("Input data already set.")
        self._input_data = data
        self._input_type = "ranks"
        # Defer actual processing until needed
        self.ranked_data = None
        self.item_mapping = None
        self.nitems = None
        self._generated = False
        return self

    def with_items_lists(
        self,
        items_lists: list[list[Any]] | np.ndarray,
        na_strings: Optional[list[str]] = None,
    ) -> "RankPipeline":
        """
        Set ranked data from lists/arrays of named items. Items will be mapped
        to dense 1-based numeric IDs internally.

        Parameters
        ----------
        items_lists : list of lists or numpy.ndarray
            Each inner list/row contains items in their ranked order.
        na_strings : Optional[list[str]], default=["", "nan", "na", "null", "none"]
            Case-insensitive strings to interpret as NA in array input.

        Returns
        -------
        self : RankPipeline
            For method chaining.
        """
        if self._input_data is not None:
            raise ValueError("Input data already set.")
        self._input_data = {
            "items_lists": items_lists,
            "na_strings": na_strings,
        }
        self._input_type = "items"
        self.ranked_data = None
        self.item_mapping = None
        self.nitems = None
        self._generated = False
        return self

    def with_scores(
        self, scores: list[list[float]], ascending: bool = True
    ) -> "RankPipeline":
        """
        Set ranked data from lists of scores. (Currently Not Fully Supported in Pipeline).

        Parameters
        ----------
        scores : list of lists
            Each inner list contains numeric scores.
        ascending : bool, default=True
            If True, lower scores get lower ranks (e.g., for p-values).

        Returns
        -------
        self : RankPipeline
            For method chaining.

        Raises
        ------
        NotImplementedError
            This method is not fully supported as the pipeline expects LISTROW_RANKCOL input.
        """
        if self._input_data is not None:
            raise ValueError("Input data already set.")
        # Store input config, but mark as not directly usable yet
        self._input_data = {"scores": scores, "ascending": ascending}
        self._input_type = "scores"
        self.ranked_data = None
        self.item_mapping = None
        self.nitems = None
        self._generated = False
        # Raise error immediately or during _prepare_data? Let's raise here for clarity.
        raise NotImplementedError(
            "Pipeline 'with_scores' requires internal conversion to LISTROW_RANKCOL "
            "which is not yet implemented. Please use 'with_items_lists' or "
            "'with_ranked_data' with numeric IDs."
        )
        # return self # If we were to implement it later

    def compute_sra(
        self,
        epsilon: float | np.ndarray = 0.0,
        metric: Literal["sd", "mad"] = "sd",
        B: int = 1,
        nitems: Optional[int] = None,
    ) -> "RankPipeline":
        """
        Compute SRA for the prepared ranked data.

        Parameters
        ----------
        epsilon : float or array-like, default=0.0
            Threshold for an item to be included in S(d).
        metric : {'sd', 'mad'}, default='sd'
            Method to measure dispersion of ranks.
        B : int, default=1
            Number of bootstrap samples for handling missing values.

        Returns
        -------
        self : RankPipeline
            For method chaining.
        """
        if not self._generated:
            self._prepare_data(nitems)  # Ensure ranked_data and nitems are set

        if self.ranked_data is None or self.nitems is None:
            # Should have been caught by _prepare_data, but double-check
            raise ValueError(
                "Ranked data must be successfully prepared before computing SRA."
            )

        self.sra_config = SRAConfig(epsilon=epsilon, metric=metric, B=B)
        self.sra = SRA(
            epsilon=self.sra_config.epsilon,
            metric=self.sra_config.metric,
            B=self.sra_config.B,
        ).generate(self.ranked_data, nitems=self.nitems)
        return self

    def random_list_sra(
        self,
        n_permutations: int = 1000,
        nitems: Optional[int] = None,
        n_jobs: int = 1,
    ) -> "RankPipeline":
        """
        Generate null distribution for the prepared ranked data. Uses SRA parameters
        if compute_sra was called, otherwise uses defaults.

        Parameters
        ----------
        n_permutations : int, default=1000
            Number of permutations to generate.

        Returns
        -------
        self : RankPipeline
            For method chaining.
        """
        if not self._generated:
            self._prepare_data(nitems)

        if self.ranked_data is None or self.nitems is None:
            raise ValueError(
                "Ranked data must be successfully prepared before generating null distribution."
            )

        # Use SRA config if available, otherwise default
        sra_conf = (
            self.sra_config if self.sra_config is not None else SRAConfig()
        )

        self.null_config = {"n_permutations": n_permutations}

        self.null_dist = RandomListSRA(
            epsilon=sra_conf.epsilon,
            metric=sra_conf.metric,
            B=sra_conf.B,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        ).generate(self.ranked_data, nitems=self.nitems)

        return self

    def test_significance(
        self,
        style: Literal["l2", "max"] = "max",
        window: int = 1,
        use_gpd: bool = False,
        threshold_quantile: float = 0.90,
        n_jobs: int = 1,
    ) -> "RankPipeline":
        """
        Test significance of observed SRA against null distribution.

        Requires compute_sra() and random_list_sra() to be called first.

        Parameters
        ----------
        style : {'l2', 'max'}, default='max'
            Method to aggregate differences.
        window : int, default=1
            Size of smoothing window. Use 1 for no smoothing.
        use_gpd : bool, default=False
            Whether to use generalized Pareto distribution for extreme p-values.
        threshold_quantile : float, default=0.90
            Quantile to use as threshold for GPD fitting.

        Returns
        -------
        self : RankPipeline
            For method chaining.
        """
        if self.sra is None or not self.sra.fitted_:
            raise ValueError(
                "SRA must be computed before testing significance (call compute_sra)."
            )
        if self.null_dist is None or not self.null_dist.fitted_:
            raise ValueError(
                "Null distribution must be generated before testing significance (call random_list_sra)."
            )

        self.test_config = TestConfig(
            style=style,
            window=window,
            use_gpd=use_gpd,
            threshold_quantile=threshold_quantile,
        )

        self.test = SRATest(
            style=style,
            window=window,
            use_gpd=use_gpd,
            threshold_quantile=threshold_quantile,
            n_jobs=n_jobs,
        ).generate(self.sra.get_result(), self.null_dist.get_result())

        return self

    def build(self) -> dict[str, Any]:
        """
        Build and return the analysis results collected in the pipeline.

        Ensures data preparation is done if not already performed.

        Returns
        -------
        results : dict
            Dictionary containing analysis results ('sra', 'null_distribution', 'test',
            'item_mapping', 'p_value', 'significant', etc., where applicable).
        """
        if not self._generated:
            try:
                self._prepare_data()
            except Exception as e:
                return {"error": f"Data preparation failed: {e}"}

        result = {}

        if self.item_mapping is not None:
            result["item_mapping"] = self.item_mapping

        if self.sra is not None and self.sra.fitted_:
            result["sra_estimator"] = self.sra
            result["sra_result"] = self.sra.get_result()
            result["sra_values"] = result["sra_result"].values
            result["when_included"] = result["sra_result"].when_included

        if self.null_dist is not None and self.null_dist.fitted_:
            result["null_estimator"] = self.null_dist
            result["null_result"] = self.null_dist.get_result()
            result["null_distribution"] = result["null_result"].distribution

            # Add confidence band only if null distribution is available
            try:
                result["confidence_band"] = self.null_dist.confidence_band()
            except Exception as e:
                warnings.warn(f"Could not compute confidence band: {e}")

        if self.test is not None and self.test.fitted_:
            result["test_estimator"] = self.test
            result["test_result"] = self.test.get_result()
            result["p_value"] = self.test.p_value()
            # Handle potential NaN p-value
            if result["p_value"] is not None and not np.isnan(
                result["p_value"]
            ):
                result["significant_05"] = result["p_value"] < 0.05
            else:
                result["significant_05"] = (
                    None  # Indicate significance couldn't be determined
                )

        if not result:
            warnings.warn(
                "Pipeline built with no analysis steps performed or data prepared."
            )

        return result


###################
# Example Usage
###################


# (Keep example_usage function as is, it doesn't need modification for the internal fixes)
def example_usage():
    """
    Demonstrate example usage of the SuperRanker package, utilizing 12 workers.
    """
    N_JOBS_TO_USE = 12  # Define the number of workers
    N_PERMUTATIONS_EXAMPLE = 10000
    N_PERMUTATIONS_COMPARISON = (
        50000  # Fewer for the comparison example for speed
    )

    print(f"--- Running examples with n_jobs = {N_JOBS_TO_USE} ---")

    # Sample ranked lists with numeric IDs (1-based)
    ranks_numeric = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 5, 6, 7, 4, 8],
            [1, 5, 3, 4, 2, 8, 7, 6],
        ]
    )

    # Using the Pipeline API with pre-defined numeric ranks
    print("\nUsing Pipeline API with numeric ranks:")
    n_items_in_example = (
        8  # This can often be inferred, but good to be explicit
    )
    results_numeric = (
        RankPipeline()
        .with_ranked_data(ranks_numeric)
        .compute_sra(epsilon=0.0, metric="sd", B=1)
        .random_list_sra(
            n_permutations=N_PERMUTATIONS_EXAMPLE, n_jobs=N_JOBS_TO_USE
        )  # Use n_jobs
        .test_significance(
            style="max", window=1, use_gpd=False, n_jobs=N_JOBS_TO_USE
        )  # Use n_jobs
        .build()
    )

    if "error" in results_numeric:
        print(f"Pipeline failed: {results_numeric['error']}")
    else:
        print(
            f"SRA first 5 values: {results_numeric.get('sra_values', [])[:5]}"
        )
        print(f"P-value: {results_numeric.get('p_value', 'N/A')}")
        print(
            f"Significant (p<0.05): {results_numeric.get('significant_05', 'N/A')}"
        )

    # Using the direct API (more verbose, shows underlying steps)
    print("\nUsing Direct API:")
    try:
        sra_config = SRAConfig(epsilon=0.0, metric="sd", B=1)
        sra_estimator = SRA().generate(ranks_numeric, nitems=n_items_in_example)
        sra_result = sra_estimator.get_result()

        null_estimator = RandomListSRA(
            epsilon=sra_config.epsilon,
            metric=sra_config.metric,
            B=sra_config.B,
            n_permutations=N_PERMUTATIONS_EXAMPLE,
            n_jobs=N_JOBS_TO_USE,  # Use n_jobs
        ).generate(ranks_numeric, nitems=n_items_in_example)
        null_result = null_estimator.get_result()

        test_config = TestConfig(style="max", use_gpd=False)
        test_estimator = SRATest(
            style=test_config.style,
            use_gpd=test_config.use_gpd,
            n_jobs=N_JOBS_TO_USE,
        ).generate(sra_result, null_result)
        test_result = test_estimator.get_result()

        print(f"Direct SRA first 5 values: {sra_result.values[:5]}")
        print(f"Direct P-value: {test_result.p_value}")

    except Exception as e:
        print(f"Direct API execution failed: {e}")
        import traceback

        traceback.print_exc()

    # Using Item Lists (Strings)
    print("\nUsing Item Lists (Strings):")
    gene_lists = [
        [
            "GeneA",
            "GeneB",
            "GeneC",
            "GeneD",
            "GeneE",
            "GeneF",
            "GeneG",
            "GeneH",
        ],
        [
            "GeneA",
            "GeneB",
            "GeneC",
            "GeneE",
            "GeneF",
            "GeneG",
            "GeneD",
            "GeneH",
        ],
        [
            "GeneA",
            "GeneE",
            "GeneC",
            "GeneD",
            "GeneB",
            "GeneH",
            "GeneG",
            "GeneF",
        ],
        ["GeneA", "GeneC", "GeneB", None, "GeneG"],
        [
            "GeneI",
            "GeneA",
            "GeneH",
            "GeneB",
            "GeneC",
            "GeneD",
            "GeneE",
            "GeneF",
        ],
    ]

    results_items = (
        RankPipeline()
        .with_items_lists(gene_lists)
        .compute_sra(epsilon=0.0, metric="sd", B=1)
        .random_list_sra(
            n_permutations=N_PERMUTATIONS_EXAMPLE, n_jobs=N_JOBS_TO_USE
        )  # Use n_jobs
        .test_significance(
            style="max", window=1, use_gpd=False, n_jobs=N_JOBS_TO_USE
        )  # Use n_jobs
        .build()
    )

    if "error" in results_items:
        print(f"Pipeline failed: {results_items['error']}")
    else:
        print(
            f"Item List SRA first 5 values: {results_items.get('sra_values', [])[:5]}"
        )
        print(f"Item List P-value: {results_items.get('p_value', 'N/A')}")
        print(
            f"Significant (p<0.05): {results_items.get('significant_05', 'N/A')}"
        )

        if "item_mapping" in results_items:
            print("\nItem mapping created:")
            items_to_print = list(
                results_items["item_mapping"]["item_to_id"].items()
            )[:5]
            for item, id_val in items_to_print:
                print(f"  {item} -> ID {id_val}")
            if len(results_items["item_mapping"]["item_to_id"]) > 5:
                print(
                    f"  ... ({len(results_items['item_mapping']['item_to_id'])} total items)"
                )

        when_inc = results_items.get("when_included")
        if when_inc is not None:
            print(f"\nWhen Included (first 5 items by ID 1-5): {when_inc[:5]}")
        else:
            print("\nWhen Included data was not calculated.")

    # Example with sparse/large IDs
    print("\nUsing large/sparse numeric IDs (data prep check):")
    ranks_large_ids = np.array(
        [[10, 5000, 20, np.nan], [20, 10, 60000, 5000], [10, 20, 5000, 60000]]
    )
    # n_items_large = 60000 # nitems will be inferred by pipeline if not given

    results_large_prep = (
        RankPipeline()
        .with_ranked_data(ranks_large_ids)
        # Only building to check data prep, not full computation here for speed
        .build()
    )

    if "error" in results_large_prep:
        print(f"Pipeline failed: {results_large_prep['error']}")
    elif results_large_prep.get("ranked_data") is not None:
        print(
            f"Data prepared successfully. Inferred nitems: {results_large_prep.get('nitems')}"
        )
        print(
            "Note: Full SRA computation with very large nitems can be slow/memory intensive "
            "even with parallelization for permutations. This example only checks data prep."
        )
        # If you want to run the full SRA with parallelization:
        # print("Attempting full SRA for large sparse IDs with parallelization...")
        # results_large_full = (
        #     RankPipeline()
        #     .with_ranked_data(ranks_large_ids)
        #     .compute_sra(B=1)
        #     .random_list_sra(n_permutations=100, n_jobs=N_JOBS_TO_USE) # Reduced perms for example
        #     .test_significance(n_jobs=N_JOBS_TO_USE)
        #     .build()
        # )
        # if "error" not in results_large_full:
        #    print(f"SRA Values (first 5): {results_large_full.get('sra_values', [])[:5]}")
        #    print(f"P-value: {results_large_full.get('p_value', 'N/A')}")
        # else:
        #    print(f"Full SRA for large IDs failed: {results_large_full['error']}")
    else:
        print("Pipeline build step failed for large IDs.")

    # Compare two methods example
    print("\nComparing Two Methods:")
    method1_ranks = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 6, 7, 4, 8]]
    )
    n1 = 8
    method2_ranks = np.array(
        [[1, 5, 3, 4, 2, 8, 7, 6], [2, 1, 5, 3, 4, 7, 8, 6]]
    )
    n2 = 8

    try:
        sra1 = SRA().generate(method1_ranks, nitems=n1).get_result()
        sra2 = SRA().generate(method2_ranks, nitems=n2).get_result()

        # Use n_jobs in RandomListSRA constructor
        null1 = (
            RandomListSRA(
                n_permutations=N_PERMUTATIONS_COMPARISON, n_jobs=N_JOBS_TO_USE
            )
            .generate(method1_ranks, nitems=n1)
            .get_result()
        )
        null2 = (
            RandomListSRA(
                n_permutations=N_PERMUTATIONS_COMPARISON, n_jobs=N_JOBS_TO_USE
            )
            .generate(method2_ranks, nitems=n2)
            .get_result()
        )

        # Use n_jobs in SRACompare constructor
        compare_estimator = SRACompare(
            style="max", tails="two-tailed", n_jobs=N_JOBS_TO_USE
        ).generate(sra1, sra2, null1, null2)
        compare_result = compare_estimator.get_result()

        print(f"Method comparison p-value: {compare_result.p_value}")

    except Exception as e:
        print(f"Comparison execution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    example_usage()
