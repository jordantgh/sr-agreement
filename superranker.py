"""
SuperRanker: A package for computing and analyzing Sequential Rank Agreement.

This package provides tools for comparing ranked lists, generating null distributions,
and testing the significance of rank agreement.
"""

import functools
import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Any
import numpy as np
from scipy.stats import genpareto


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
        else:
            raise TypeError(
                f"Epsilon must be float or numpy.ndarray, got {type(self.epsilon)}"
            )


@dataclass(frozen=True)
class TestConfig:
    """Configuration for statistical testing of SRA values.

    Parameters
    ----------
    style : str, default="l2"
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
    """

    style: Literal["l2", "max"] = "max"
    window: int = 1
    use_gpd: bool = False
    threshold_quantile: float = 0.90

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


###################
# Result Containers
###################


@dataclass(frozen=True)
class SRAResult:
    """Immutable container for SRA computation results.

    Attributes
    ----------
    values : numpy.ndarray
        SRA values for each depth.
    config : SRAConfig
        Configuration used for computation.
    when_included : numpy.ndarray, optional
        Depth at which each item was first included.
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
        for each permutation.
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
        return X


###################
# Utility Functions
###################


def _reshape_rank_matrix(ranked_lists_array: np.ndarray) -> np.ndarray:
    """
    Create a reshaped rank matrix from the input ranked lists.

    Parameters
    ----------
    ranked_lists_array : numpy.ndarray
        Array of shape (num_lists, nitems) where each row is a ranked list.

    Returns
    -------
    rank_matrix : numpy.ndarray
        Array of shape (num_lists, nitems) where rank_matrix[i, j]
        is the rank (1-indexed) of item (j+1) in list i.
    """
    num_lists, nitems = ranked_lists_array.shape
    rank_matrix = np.zeros((num_lists, nitems), dtype=int)
    rows = np.arange(num_lists)[:, np.newaxis]
    cols = ranked_lists_array.astype(int) - 1
    rank_matrix[rows, cols] = np.tile(np.arange(1, nitems + 1), (num_lists, 1))
    return rank_matrix


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

    if window_size > n:
        return np.full(n, np.mean(sra_values))

    cs = np.concatenate(([0], np.cumsum(sra_values)))
    smoothed = np.empty(n)

    for i in range(n):
        start_idx = max(0, i - half_w)
        end_idx = min(n - 1, i + half_w)
        count = end_idx - start_idx + 1
        smoothed[i] = (cs[end_idx + 1] - cs[start_idx]) / count

    return smoothed


def _aggregator(diffs: np.ndarray, style: str = "l2") -> float:
    """
    Aggregate a vector of differences.

    Parameters
    ----------
    diffs : numpy.ndarray
        1D array of differences.
    style : str, default="l2"
        Aggregation style; "max" returns the maximum, "l2" returns the sum of squares.

    Returns
    -------
    aggregated_value : float
        Aggregated value.
    """
    if style == "max":
        return np.max(diffs)
    else:
        return np.sum(diffs**2)


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
    threshold = np.quantile(T_null, threshold_quantile)

    # If observed statistic isn't extreme, don't apply GPD
    if T_obs <= threshold:
        return {
            "p_value_gpd": np.mean(T_null >= T_obs),
            "gpd_fit": None,
            "applied_gpd": False,
        }

    tail_data = T_null[T_null > threshold]

    # Check if we have enough tail points for stable fitting
    if tail_data.size < 30:
        return {
            "p_value_gpd": np.mean(T_null >= T_obs),
            "gpd_fit": None,
            "applied_gpd": False,
        }

    # Fit GPD to excesses
    excesses = tail_data - threshold
    try:
        xi, _, beta = genpareto.fit(excesses, floc=0)
        excess_obs = T_obs - threshold
        tail_prob = genpareto.sf(excess_obs, c=xi, loc=0, scale=beta)

        F_threshold = np.mean(T_null < threshold)
        p_value_gpd = (1 - F_threshold) * tail_prob

        return {
            "p_value_gpd": p_value_gpd,
            "gpd_fit": GPDFit(xi=xi, beta=beta, threshold=threshold),
            "applied_gpd": True,
        }
    except (RuntimeError, ValueError) as e:
        warnings.warn(
            f"GPD fitting failed: {str(e)}. Using empirical p-value instead."
        )
        return {
            "p_value_gpd": np.mean(T_null >= T_obs),
            "gpd_fit": None,
            "applied_gpd": False,
        }


def _generate_null_distribution(
    null_matrix: np.ndarray,
    aggregator_style: str,
) -> np.ndarray:
    """
    Generate null distribution of test statistics.

    Parameters
    ----------
    null_matrix : numpy.ndarray
        Matrix of shape (n_depths, n_permutations) containing null SRA values.
    aggregator_style : str
        Aggregation style; "max" or "l2".

    Returns
    -------
    T_null : numpy.ndarray
        Array of test statistics from null distribution.
    """
    B = null_matrix.shape[1]
    colsums = np.sum(null_matrix, axis=1)
    T_null = np.empty(B)

    for i in range(B):
        loo_mean = (colsums - null_matrix[:, i]) / (B - 1)
        diffs = np.abs(null_matrix[:, i] - loo_mean)
        T_null[i] = _aggregator(diffs, aggregator_style)

    return T_null


###################
# Core Algorithms
###################


def compute_sra(
    ranked_lists: np.ndarray, config: SRAConfig, nitems: Optional[int] = None
) -> SRAResult:
    """
    Compute the Sequential Rank Agreement (SRA) for a set of ranked lists.

    This is the pure functional core of the SRA algorithm. It handles the case
    of incomplete lists by imputing missing values through random resampling.

    Parameters
    ----------
    ranked_lists : numpy.ndarray
        Array of shape (n_lists, list_length) where each row is a ranked list.
    config : SRAConfig
        Configuration for SRA computation.
    nitems : int, optional
        Total number of items. If None, inferred from ranked_lists.

    Returns
    -------
    result : SRAResult
        Container for SRA computation results.
    """
    ranked_lists = np.array(ranked_lists, dtype=float)
    num_lists, list_length = ranked_lists.shape

    if nitems is None:
        nitems = list_length
    elif list_length < nitems:
        # Pad with missing values
        pad_width = nitems - list_length
        pad = np.full((num_lists, pad_width), np.nan)
        ranked_lists = np.concatenate([ranked_lists, pad], axis=1)

    if np.isscalar(config.epsilon):
        epsilons = np.full(nitems, float(config.epsilon))
    else:
        epsilons = np.asarray(config.epsilon, dtype=float)
        if len(epsilons) != nitems:
            raise ValueError(
                f"Length of epsilon ({len(epsilons)}) must match nitems ({nitems})"
            )

    # Store SRA curves from each resample
    sra_curves = np.zeros((config.B, nitems))
    when_included = np.full(nitems, nitems + 1, dtype=float)

    # For each resample, impute missing values and compute the SRA curve
    for b in range(config.B):
        imputed = np.empty_like(ranked_lists)
        for i in range(num_lists):
            list_i = ranked_lists[i, :]
            observed_mask = ~np.isnan(list_i)
            observed = list_i[observed_mask].astype(int)
            missing_idx = np.where(~observed_mask)[0]

            all_items = set(range(1, nitems + 1))
            observed_set = set(observed)
            missing_items = np.array(list(all_items - observed_set), dtype=int)

            if missing_items.size > 0:
                np.random.shuffle(missing_items)

            new_list = list_i.copy()
            for j, idx in enumerate(missing_idx):
                if j < len(missing_items):
                    new_list[idx] = missing_items[j]

            imputed[i, :] = new_list

        rank_mat = _reshape_rank_matrix(imputed)

        if config.metric.lower() == "sd":
            disagreements = np.var(rank_mat, axis=0, ddof=1)
        elif config.metric.lower() == "mad":

            def mad(x):
                med = np.median(x)
                return np.median(np.abs(x - med)) * 1.4826

            disagreements = np.array(
                [mad(rank_mat[:, j]) for j in range(nitems)]
            )
        else:
            raise ValueError(f"Unsupported metric: {config.metric}")

        sra_curve = np.zeros(nitems)
        for d in range(1, nitems + 1):
            prop = np.mean(rank_mat <= d, axis=0)
            depth_set = prop > epsilons[d - 1]
            if b == 0:
                for j in range(nitems):
                    if depth_set[j] and d < when_included[j]:
                        when_included[j] = d

            if np.any(depth_set):
                sra_curve[d - 1] = np.mean(disagreements[depth_set])
            else:
                sra_curve[d - 1] = 0.0

        sra_curves[b, :] = sra_curve

    avg_sra = np.mean(sra_curves, axis=0)

    if config.metric.lower() == "sd":
        avg_sra = np.sqrt(avg_sra)

    when_included[when_included > nitems] = np.inf

    return SRAResult(values=avg_sra, config=config, when_included=when_included)


def random_list_sra(
    ranked_lists: np.ndarray | list[list[float]],
    config: SRAConfig,
    n_permutations: int = 100,
    nitems: Optional[int] = None,
) -> RandomListSRAResult:
    ranked_lists = np.asarray(ranked_lists, dtype=float)
    num_lists, list_length = ranked_lists.shape

    # If needed, pad out to nitems
    if nitems is None:
        nitems = list_length
    elif list_length < nitems:
        pad_width = nitems - list_length
        pad = np.full((num_lists, pad_width), np.nan)
        ranked_lists = np.concatenate([ranked_lists, pad], axis=1)
        list_length = ranked_lists.shape[1]

    # Count how many items are actually non-missing in each row
    notmiss = np.sum(~np.isnan(ranked_lists), axis=1).astype(int)

    # Pre-generate random partial-permutations for each row
    sample_list = []
    for nn in notmiss:
        # nn is how many items are actually observed
        # generate n_permutations permutations of [1..nitems], each truncated to length nn
        row_samples = [
            np.random.permutation(nitems)[:nn] + 1
            for _ in range(n_permutations)
        ]
        sample_list.append(row_samples)

    # Now build each random replicate and compute SRA
    sra_results = []
    for i in range(n_permutations):
        # For each replicate i, assemble a new matrix from row i's partial permutations
        current_obj = np.full((num_lists, list_length), np.nan, dtype=float)
        for row_idx in range(num_lists):
            nn = notmiss[row_idx]
            if nn > 0:
                current_obj[row_idx, :nn] = sample_list[row_idx][i]
        # compute SRA on that entire matrix
        sra_curve = compute_sra(current_obj, config, nitems).values
        sra_results.append(sra_curve)

    # Combine into a distribution of shape (nitems, n_permutations)
    null_distribution = np.column_stack(sra_results)
    return RandomListSRAResult(
        distribution=null_distribution,
        config=config,
        n_permutations=n_permutations,
    )


def test_sra(
    observed_sra: np.ndarray, null_distribution: np.ndarray, config: TestConfig
) -> TestResult:
    """
    Test observed SRA values against null distribution.

    Parameters
    ----------
    observed_sra : numpy.ndarray
        Observed SRA values.
    null_distribution : numpy.ndarray
        Null distribution of SRA values.
    config : TestConfig
        Configuration for the test.

    Returns
    -------
    result : TestResult
        Container for test results.
    """
    observed_sra = np.asarray(observed_sra)
    null_distribution = np.asarray(null_distribution)

    # Apply smoothing if requested
    if config.window > 1:
        observed_sra = smooth_sra_window(observed_sra, config.window)
        null_distribution = np.apply_along_axis(
            lambda n: smooth_sra_window(n, config.window), 0, null_distribution
        )

    # Compute test statistic
    diffs_obs = np.abs(observed_sra - np.mean(null_distribution, axis=1))
    T_obs = _aggregator(diffs_obs, style=config.style)

    # Generate null distribution of test statistics
    T_null = _generate_null_distribution(null_distribution, config.style)

    # Compute empirical p-value
    p_value_empirical = (np.sum(T_null >= T_obs) + 1) / (len(T_null) + 1)

    # Use GPD if requested
    gpd_results = None
    p_value_gpd = None
    gpd_fit = None

    if config.use_gpd:
        gpd_results = _calculate_gpd_pvalue(
            T_obs, T_null, config.threshold_quantile
        )
        p_value_gpd = gpd_results["p_value_gpd"]
        gpd_fit = gpd_results["gpd_fit"]

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
) -> ComparisonResult:
    """
    Compare two SRA curves.

    Parameters
    ----------
    sra1 : numpy.ndarray
        First SRA curve.
    sra2 : numpy.ndarray
        Second SRA curve.
    null1 : numpy.ndarray
        Null distribution for first curve.
    null2 : numpy.ndarray
        Null distribution for second curve.
    config : TestConfig
        Configuration for the test.

    Returns
    -------
    result : ComparisonResult
        Container for comparison results.
    """
    sra1 = np.asarray(sra1)
    sra2 = np.asarray(sra2)
    null1 = np.asarray(null1)
    null2 = np.asarray(null2)

    # Apply smoothing if requested
    if config.window > 1:
        sra1 = smooth_sra_window(sra1, config.window)
        sra2 = smooth_sra_window(sra2, config.window)
        null1 = np.apply_along_axis(
            lambda n: smooth_sra_window(n, config.window), 0, null1
        )
        null2 = np.apply_along_axis(
            lambda n: smooth_sra_window(n, config.window), 0, null2
        )

    # Compute test statistics for each curve
    T_obs1 = _aggregator(np.abs(sra1 - np.mean(null1, axis=1)), config.style)
    T_obs2 = _aggregator(np.abs(sra2 - np.mean(null2, axis=1)), config.style)

    # Compute observed difference
    T_obs = T_obs1 - T_obs2

    # Generate null distribution by comparing random permutations
    combined = np.hstack([null1, null2])
    B = combined.shape[1]
    T_null = np.empty(B)

    for i in range(B):
        cols = np.random.choice(B, size=2, replace=False)
        T1 = _aggregator(
            np.abs(combined[:, cols[0]] - np.mean(combined, axis=1)),
            config.style,
        )
        T2 = _aggregator(
            np.abs(combined[:, cols[1]] - np.mean(combined, axis=1)),
            config.style,
        )
        T_null[i] = T1 - T2

    # Compute p-value
    p_value = (np.sum(T_null >= T_obs) + 1) / (B + 1)

    return ComparisonResult(
        p_value=p_value,
        test_statistic=T_obs,
        null_statistics=T_null,
        config=config,
    )


###################
# Estimator Classes
###################


class RankData:
    """Utility for handling and transforming ranked data."""

    @staticmethod
    def from_items(items_lists: list[list[Any]]) -> np.ndarray:
        """Convert lists of items to numeric ranks.

        Parameters
        ----------
        items_lists : list of lists
            Each inner list contains items in their ranked order.

        Returns
        -------
        rank_matrix : numpy.ndarray
            Array of shape (len(items_lists), max_items) where each row
            is a ranked list converted to numeric ranks.
        """
        result = []

        # Find unique items across all lists
        all_items = set()
        for items in items_lists:
            all_items.update(items)

        # Create a mapping from items to integer IDs
        item_to_id = {
            item: idx + 1 for idx, item in enumerate(sorted(all_items))
        }

        # Convert each list to numeric IDs
        for items in items_lists:
            result.append([item_to_id[item] for item in items])

        # Find maximum length and pad with NaN
        max_len = max(len(lst) for lst in result)
        padded_result = []

        for lst in result:
            padded = lst + [np.nan] * (max_len - len(lst))
            padded_result.append(padded)

        return np.array(padded_result)

    @staticmethod
    def from_scores(
        scores: list[list[float]], ascending: bool = True
    ) -> np.ndarray:
        """Convert lists of scores to ranks (e.g., p-values, correlations).

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
            Array where each row contains the corresponding ranks.
        """
        result = []

        for score_list in scores:
            # Convert to numpy array
            scores_array = np.array(score_list)

            # Handle NaN values
            mask = ~np.isnan(scores_array)
            valid_scores = scores_array[mask]

            # Get ranks (argsort of argsort gives ranks)
            if ascending:
                valid_ranks = np.argsort(np.argsort(valid_scores)) + 1
            else:
                valid_ranks = np.argsort(np.argsort(-valid_scores)) + 1

            # Create full array with NaN for missing values
            ranks = np.full_like(scores_array, np.nan)
            ranks[mask] = valid_ranks

            result.append(ranks)

        return np.array(result)


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
            Ranked lists data. Each row represents a list.
        nitems : int, optional
            Total number of items. If None, inferred from X.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_input(X)
        self.result_ = compute_sra(X, self.config, nitems)
        self.fitted_ = True
        return self

    @require_generated
    def get_result(self) -> SRAResult:
        """Get the SRA computation result."""
        return self.result_

    @require_generated
    def values(self) -> np.ndarray:
        """Get the SRA values."""
        return self.result_.values

    @require_generated
    def when_included(self) -> np.ndarray:
        """Get the depth at which each item was first included."""
        return self.result_.when_included

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
        return self.result_.smooth(window_size)


class RandomListSRA(BaseEstimator):
    """Generate null distribution for SRA through permutation.

    Parameters
    ----------
    epsilon : float or array-like, default=0.0
        Threshold for an item to be included in S(d).
    metric : {'sd', 'mad'}, default='sd'
        Method to measure dispersion of ranks.
    B : int, default=1
        Number of bootstrap samples for handling missing values.
    n_permutations : int, default=100
        Number of permutations to generate.
    n_jobs : int, default=1
        Number of jobs for parallel processing. Use -1 to use all available cores.
    """

    def __init__(
        self,
        epsilon: float | np.ndarray = 0.0,
        metric: Literal["sd", "mad"] = "sd",
        B: int = 1,
        n_permutations: int = 100,
    ):
        super().__init__()
        self.config = SRAConfig(epsilon=epsilon, metric=metric, B=B)
        self.n_permutations = n_permutations
        self.result_ = None

    def generate(
        self, X: np.ndarray, nitems: Optional[int] = None
    ) -> "RandomListSRA":
        """
        Generate null distribution of SRA values by permuting lists.

        Parameters
        ----------
        X : array-like of shape (n_lists, list_length)
            Ranked lists data. Each row represents a list.
        nitems : int, optional
            Total number of items. If None, inferred from X.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_input(X)
        self.result_ = random_list_sra(
            X, self.config, self.n_permutations, nitems
        )
        self.fitted_ = True
        return self

    @require_generated
    def get_result(self) -> RandomListSRAResult:
        """Get the null distribution generation result."""
        return self.result_

    @require_generated
    def distribution(self) -> np.ndarray:
        """Get the null distribution matrix."""
        return self.result_.distribution

    @require_generated
    def confidence_band(
        self, confidence: float = 0.95
    ) -> dict[str, np.ndarray]:
        """
        Compute confidence band for the null distribution.

        Parameters
        ----------
        confidence : float, default=0.95
            Confidence level.

        Returns
        -------
        band : dict
            Dictionary with 'lower' and 'upper' arrays.
        """
        return self.result_.confidence_band(confidence)

    @require_generated
    def quantiles(self, probs: list[float]) -> dict[float, np.ndarray]:
        """
        Compute quantiles of the null distribution for each depth.

        Parameters
        ----------
        probs : list of float
            Probability points at which to compute quantiles.

        Returns
        -------
        quantiles : dict
            Dictionary mapping probabilities to quantile arrays.
        """
        result = {}
        for prob in probs:
            result[prob] = np.quantile(self.result_.distribution, prob, axis=1)
        return result


class SRATest(BaseEstimator):
    """Test for significance of SRA values against a null distribution.

    Parameters
    ----------
    style : {'l2', 'max'}, default='l2'
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
        style: Literal["l2", "max"] = "l2",
        window: int = 1,
        use_gpd: bool = False,
        threshold_quantile: float = 0.90,
    ):
        super().__init__()
        self.config = TestConfig(
            style=style,
            window=window,
            use_gpd=use_gpd,
            threshold_quantile=threshold_quantile,
        )
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
            Observed SRA values or result object.
        null_dist : RandomListSRAResult or array-like
            Null distribution of SRA values or result object.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Extract values if result objects are provided
        if isinstance(observed_sra, SRAResult):
            observed_values = observed_sra.values
        else:
            observed_values = np.asarray(observed_sra)

        if isinstance(null_dist, RandomListSRAResult):
            null_values = null_dist.distribution
        else:
            null_values = np.asarray(null_dist)

        # Validate input
        if observed_values.ndim != 1:
            raise ValueError("observed_sra must be a 1D array or SRAResult")
        if null_values.ndim != 2:
            raise ValueError(
                "null_dist must be a 2D array or RandomListSRAResult"
            )
        if observed_values.shape[0] != null_values.shape[0]:
            raise ValueError(
                "observed_sra and null_dist must have same number of depths"
            )

        self.result_ = test_sra(observed_values, null_values, self.config)
        self.fitted_ = True
        return self

    @require_generated
    def get_result(self) -> TestResult:
        """Get the test result."""
        return self.result_

    @require_generated
    def p_value(self) -> float:
        """Get the p-value (GPD-based if available, otherwise empirical)."""
        return self.result_.p_value


class SRACompare(BaseEstimator):
    """Compare two SRA curves.

    Parameters
    ----------
    style : {'l2', 'max'}, default='l2'
        Method to aggregate differences.
    window : int, default=1
        Size of smoothing window. Use 1 for no smoothing.
    """

    def __init__(self, style: Literal["l2", "max"] = "l2", window: int = 1):
        super().__init__()
        self.config = TestConfig(style=style, window=window)
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
            First SRA curve or result object.
        sra2 : SRAResult or array-like
            Second SRA curve or result object.
        null1 : RandomListSRAResult or array-like
            Null distribution for first curve or result object.
        null2 : RandomListSRAResult or array-like
            Null distribution for second curve or result object.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Extract values if result objects are provided
        if isinstance(sra1, SRAResult):
            sra1_values = sra1.values
        else:
            sra1_values = np.asarray(sra1)

        if isinstance(sra2, SRAResult):
            sra2_values = sra2.values
        else:
            sra2_values = np.asarray(sra2)

        if isinstance(null1, RandomListSRAResult):
            null1_values = null1.distribution
        else:
            null1_values = np.asarray(null1)

        if isinstance(null2, RandomListSRAResult):
            null2_values = null2.distribution
        else:
            null2_values = np.asarray(null2)

        # Validate input
        if sra1_values.ndim != 1 or sra2_values.ndim != 1:
            raise ValueError(
                "SRA curves must be 1D arrays or SRAResult objects"
            )
        if sra1_values.shape[0] != sra2_values.shape[0]:
            raise ValueError("SRA curves must have same length")
        if null1_values.ndim != 2 or null2_values.ndim != 2:
            raise ValueError(
                "Null distributions must be 2D arrays or RandomListSRAResult objects"
            )

        self.result_ = compare_sra(
            sra1_values, sra2_values, null1_values, null2_values, self.config
        )
        self.fitted_ = True
        return self

    @require_generated
    def get_result(self) -> ComparisonResult:
        """Get the comparison result."""
        return self.result_

    @require_generated
    def p_value(self) -> float:
        """Get the p-value for the comparison."""
        return self.result_.p_value


class RankPipeline:
    """Builder for common rank analysis pipelines.

    This utility class provides a fluent API for building common analysis
    pipelines for ranked data, making it easy to perform typical analyses
    with minimal code.

    Attributes
    ----------
    ranked_data : numpy.ndarray
        Ranked data for analysis.
    sra_config : SRAConfig
        Configuration for SRA computation.
    sra : SRA
        SRA estimator.
    null_config : dict
        Configuration for null distribution generation.
    null_dist : RandomListSRA
        Null distribution estimator.
    test_config : TestConfig
        Configuration for significance testing.
    test : SRATest
        Test estimator.
    """

    def __init__(self):
        self.ranked_data = None
        self.sra_config = None
        self.sra = None
        self.null_config = None
        self.null_dist = None
        self.test_config = None
        self.test = None

    def with_ranked_data(self, data: np.ndarray) -> "RankPipeline":
        """
        Set the ranked data for analysis.

        Parameters
        ----------
        data : array-like
            Ranked lists data. Each row represents a list.

        Returns
        -------
        self : RankPipeline
            For method chaining.
        """
        self.ranked_data = np.asarray(data)
        return self

    def with_items_lists(self, items_lists: list[list[Any]]) -> "RankPipeline":
        """
        Set ranked data from lists of items.

        Parameters
        ----------
        items_lists : list of lists
            Each inner list contains items in their ranked order.

        Returns
        -------
        self : RankPipeline
            For method chaining.
        """
        self.ranked_data = RankData.from_items(items_lists)
        return self

    def with_scores(
        self, scores: list[list[float]], ascending: bool = True
    ) -> "RankPipeline":
        """
        Set ranked data from lists of scores.

        Parameters
        ----------
        scores : list of lists
            Each inner list contains numeric scores.
        ascending : bool, default=True
            If True, lower scores get lower ranks (e.g., for p-values).
            If False, higher scores get lower ranks (e.g., for correlations).

        Returns
        -------
        self : RankPipeline
            For method chaining.
        """
        self.ranked_data = RankData.from_scores(scores, ascending=ascending)
        return self

    def compute_sra(
        self,
        epsilon: float | np.ndarray = 0.0,
        metric: Literal["sd", "mad"] = "sd",
        B: int = 1,
        nitems: Optional[int] = None,
    ) -> "RankPipeline":
        """
        Compute SRA for the ranked data.

        Parameters
        ----------
        epsilon : float or array-like, default=0.0
            Threshold for an item to be included in S(d).
        metric : {'sd', 'mad'}, default='sd'
            Method to measure dispersion of ranks.
        B : int, default=1
            Number of bootstrap samples for handling missing values.
        nitems : int, optional
            Total number of items. If None, inferred from ranked_data.

        Returns
        -------
        self : RankPipeline
            For method chaining.
        """
        if self.ranked_data is None:
            raise ValueError("Ranked data must be set before computing SRA")

        self.sra_config = SRAConfig(epsilon=epsilon, metric=metric, B=B)
        self.sra = SRA(epsilon=epsilon, metric=metric, B=B).generate(
            self.ranked_data, nitems=nitems
        )

        return self

    def random_list_sra(
        self,
        n_permutations: int = 100,
        nitems: Optional[int] = None,
    ) -> "RankPipeline":
        """
        Generate null distribution for the ranked data.

        Parameters
        ----------
        n_permutations : int, default=100
            Number of permutations to generate.
        nitems : int, optional
            Total number of items. If None, inferred from ranked_data.
        n_jobs : int, default=1
            Number of jobs for parallel processing. Use -1 to use all available cores.

        Returns
        -------
        self : RankPipeline
            For method chaining.
        """
        if self.ranked_data is None:
            raise ValueError(
                "Ranked data must be set before generating null distribution"
            )

        if self.sra_config is None:
            warnings.warn("Using default SRA parameters for null distribution")
            self.sra_config = SRAConfig()

        self.null_config = {"n_permutations": n_permutations}

        self.null_dist = RandomListSRA(
            epsilon=self.sra_config.epsilon,
            metric=self.sra_config.metric,
            B=self.sra_config.B,
            n_permutations=n_permutations,
        ).generate(self.ranked_data, nitems=nitems)

        return self

    def test_significance(
        self,
        style: Literal["l2", "max"] = "l2",
        window: int = 1,
        use_gpd: bool = False,
        threshold_quantile: float = 0.90,
    ) -> "RankPipeline":
        """
        Test significance of observed SRA against null distribution.

        Parameters
        ----------
        style : {'l2', 'max'}, default='l2'
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
        if self.sra is None:
            raise ValueError("SRA must be computed before testing significance")
        if self.null_dist is None:
            raise ValueError(
                "Null distribution must be generated before testing significance"
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
        ).generate(self.sra.get_result(), self.null_dist.get_result())

        return self

    def build(self) -> dict[str, Any]:
        """
        Build and return the analysis results.

        Returns
        -------
        results : dict
            Dictionary containing analysis results.
        """
        result = {}

        if self.sra is not None:
            result["sra"] = self.sra
            result["sra_result"] = self.sra.get_result()

        if self.null_dist is not None:
            result["null_distribution"] = self.null_dist
            result["null_result"] = self.null_dist.get_result()

            if self.sra is not None:
                result["confidence_band"] = self.null_dist.confidence_band()

        if self.test is not None:
            result["test"] = self.test
            result["test_result"] = self.test.get_result()
            result["p_value"] = self.test.p_value()
            result["significant"] = self.test.p_value() < 0.05

        return result


###################
# Example Usage
###################


def example_usage():
    """
    Demonstrate example usage of the SuperRanker package.
    """
    # Sample ranked lists
    ranks = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 5, 6, 7, 4, 8],
            [1, 5, 3, 4, 2, 8, 7, 6],
        ]
    )

    # Using the Pipeline API
    print("Using Pipeline API:")
    results = (
        RankPipeline()
        .with_ranked_data(ranks)
        .compute_sra(epsilon=0.0, metric="sd", B=1)
        .random_list_sra(n_permutations=10000)
        .test_significance(style="max", window=1, use_gpd=False)
        .build()
    )

    print(f"SRA first 5 values: {results['sra_result'].values[:5]}")
    print(f"P-value: {results['p_value']}")
    print(f"Significant: {results['significant']}")

    # Using the direct API
    print("\nUsing Direct API:")
    sra_config = SRAConfig(epsilon=0.0, metric="sd", B=1)
    sra_estimator = SRA(
        epsilon=sra_config.epsilon, metric=sra_config.metric, B=sra_config.B
    ).generate(ranks)
    sra_result = sra_estimator.get_result()

    null_estimator = RandomListSRA(
        epsilon=sra_config.epsilon,
        metric=sra_config.metric,
        B=sra_config.B,
        n_permutations=10000,
    ).generate(ranks)
    null_result = null_estimator.get_result()

    test_config = TestConfig(style="max", use_gpd=False)
    test = SRATest(
        style=test_config.style, use_gpd=test_config.use_gpd
    ).generate(sra_result, null_result)
    test_result = test.get_result()

    print(f"Direct SRA first 5 values: {sra_result.values[:5]}")
    print(f"Direct P-value: {test_result.p_value}")

    # Working with items directly
    print("\nWorking with Item Lists:")
    gene_lists = [
        [
            "Gene1",
            "Gene2",
            "Gene3",
            "Gene4",
            "Gene5",
            "Gene6",
            "Gene7",
            "Gene8",
        ],
        [
            "Gene1",
            "Gene2",
            "Gene3",
            "Gene5",
            "Gene6",
            "Gene7",
            "Gene4",
            "Gene8",
        ],
        [
            "Gene1",
            "Gene5",
            "Gene3",
            "Gene4",
            "Gene2",
            "Gene8",
            "Gene7",
            "Gene6",
        ],
    ]

    # With the same parameters as the original example
    results_genes = (
        RankPipeline()
        .with_items_lists(gene_lists)
        .compute_sra(epsilon=0.0, metric="sd", B=1)
        .random_list_sra(n_permutations=10000)
        .test_significance(style="max", window=1, use_gpd=False)
        .build()
    )

    print(f"Gene SRA values: {results_genes['sra_result'].values[:5]}")
    print(f"Gene test p-value: {results_genes['p_value']}")
    print(f"Significant: {results_genes['significant']}")
    # Compare two methods
    print("\nComparing Two Methods:")
    method1_ranks = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 6, 7, 4, 8]]
    )

    method2_ranks = np.array(
        [[1, 5, 3, 4, 2, 8, 7, 6], [2, 1, 5, 3, 4, 7, 8, 6]]
    )

    # Calculate SRAs for both methods
    sra1 = SRA().generate(method1_ranks).get_result()
    sra2 = SRA().generate(method2_ranks).get_result()

    # Generate null distributions
    null1 = (
        RandomListSRA(n_permutations=50).generate(method1_ranks).get_result()
    )
    null2 = (
        RandomListSRA(n_permutations=50).generate(method2_ranks).get_result()
    )

    # Compare methods
    compare = SRACompare().generate(sra1, sra2, null1, null2)
    compare_result = compare.get_result()

    print(f"Method comparison p-value: {compare_result.p_value}")


if __name__ == "__main__":
    example_usage()
