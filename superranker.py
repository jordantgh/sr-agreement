import numpy as np
from scipy.stats import norm, t, genpareto

def _reshape_rank_matrix(ranked_lists_array: np.ndarray) -> np.ndarray:
    """
    Create a rank matrix from the input ranked lists.

    Each row corresponds to a list and each column to an item (ID from 1 to nitems).
    The output matrix has element [i, j] equal to the rank of item (j+1) in list i.

    Args:
        ranked_lists_array: ndarray of shape (num_lists, nitems)
            Each row must be a permutation of the integers 1 to nitems.

    Returns:
        rank_matrix: ndarray of shape (num_lists, nitems) where rank_matrix[i, j]
            is the rank (1-indexed) of item (j+1) in list i.
    """
    num_lists, nitems = ranked_lists_array.shape
    rank_matrix = np.zeros((num_lists, nitems), dtype=int)
    # For each list, the element in position j is the item ID.
    # We then set rank_matrix[i, itemID - 1] = (j+1), i.e. the rank.
    rows = np.arange(num_lists)[:, np.newaxis]
    cols = ranked_lists_array.astype(int) - 1  # convert to 0-indexed
    # Each row gets assigned the ranks 1,2,...,nitems
    rank_matrix[rows, cols] = np.tile(np.arange(1, nitems + 1), (num_lists, 1))
    return rank_matrix


def sra(
    ranked_lists_array: np.ndarray,
    epsilon: float | np.ndarray = 0.0,
    B: int = 1,
    nitems: int | None = None,
    metric: str = "sd",
) -> np.ndarray:
    """
    Compute the Sequential Rank Agreement (SRA) curve.

    This function handles incomplete/censored lists by imputing missing items
    via random resampling (B resamples). It supports two agreement metrics:
    'sd' (variance-based, with a square root transformation applied for interpretability)
    and 'mad' (median absolute deviation scaled by 1.4826).

    Args:
        ranked_lists_array: ndarray of shape (num_lists, k)
            Each row is a ranked list (item IDs from 1 to nitems).
            Lists can be incomplete (k < nitems) or contain missing values (np.nan).
        epsilon: Scalar or array-like of length nitems.
            Threshold for an item to be considered “included” in S(d).
        B: int, default 1.
            Number of resamples to perform when imputing missing items.
        nitems: int or None.
            The total number of items. If not provided, it is assumed to equal k (the number of columns).
        metric: str, either "sd" or "mad".
            Specifies the dispersion measure. "sd" uses variance (with square root later),
            "mad" uses median absolute deviation (scaled by 1.4826).

    Returns:
        sra_values: ndarray of shape (nitems,)
            The SRA curve, where the dth entry is the sequential rank agreement for depth d.
    """
    ranked_lists_array = np.array(ranked_lists_array, dtype=float)
    num_lists, k = ranked_lists_array.shape

    # If nitems is provided and k < nitems, pad each list with missing values (np.nan)
    if nitems is None:
        nitems = k
    elif k < nitems:
        pad_width = nitems - k
        pad = np.full((num_lists, pad_width), np.nan)
        ranked_lists_array = np.concatenate([ranked_lists_array, pad], axis=1)

    # Convert epsilon to an array of length nitems
    if np.isscalar(epsilon):
        epsilons = np.full(nitems, float(epsilon))
    else:
        epsilons = np.asarray(epsilon, dtype=float)
        if epsilons.shape[0] != nitems:
            raise ValueError("Length of epsilon must match nitems.")

    # Prepare to store SRA curves from each resample
    sra_curves = np.zeros((B, nitems))

    # For each resample, impute missing values and compute the SRA curve
    for b in range(B):
        # Impute missing values for each list separately.
        # Each list is treated as a (possibly partial) permutation.
        imputed = np.empty_like(ranked_lists_array)
        for i in range(num_lists):
            list_i = ranked_lists_array[i, :]
            # Identify observed (non-missing) entries
            observed_mask = ~np.isnan(list_i)
            observed = list_i[observed_mask].astype(int)
            # Determine positions that are missing
            missing_idx = np.where(~observed_mask)[0]
            # The full set of items is assumed to be 1...nitems
            all_items = set(range(1, nitems + 1))
            observed_set = set(observed)
            missing_items = np.array(list(all_items - observed_set), dtype=int)
            # Randomly shuffle the missing items
            if missing_items.size > 0:
                np.random.shuffle(missing_items)
            # Create the imputed list: keep observed values in place, fill missing positions with shuffled missing items
            new_list = list_i.copy()
            for j, idx in enumerate(missing_idx):
                new_list[idx] = missing_items[j]
            # In case the list is complete, new_list remains unchanged
            imputed[i, :] = new_list

        # At this point each row of imputed is a complete permutation of 1...nitems.
        rank_mat = _reshape_rank_matrix(imputed)  # shape: (num_lists, nitems)

        # Compute dispersion for each item (i.e. for each column) using the chosen metric.
        if metric.lower() == "sd":
            # Use variance with ddof=1.
            disagreements = np.var(rank_mat, axis=0, ddof=1)
        elif metric.lower() == "mad":
            # Compute MAD: median(|x - median(x)|) * 1.4826 for each column.
            def mad(x):
                med = np.median(x)
                return np.median(np.abs(x - med)) * 1.4826

            disagreements = np.array(
                [mad(rank_mat[:, j]) for j in range(nitems)]
            )
        else:
            raise ValueError("Unsupported metric. Choose 'sd' or 'mad'.")

        # For each depth d, define S(d) as the set of items where the proportion of lists
        # ranking the item in the top d exceeds the threshold epsilon.
        sra_curve = np.zeros(nitems)
        for d in range(1, nitems + 1):
            # Compute, for each item, the proportion of lists with rank <= d.
            prop = np.mean(rank_mat <= d, axis=0)
            # Determine the "included" items based on epsilon for depth d.
            depth_set = prop > epsilons[d - 1]
            if np.any(depth_set):
                sra_curve[d - 1] = np.mean(disagreements[depth_set])
            else:
                sra_curve[d - 1] = 0.0
        sra_curves[b, :] = sra_curve

    # Average the SRA curves across resamples.
    avg_sra = np.mean(sra_curves, axis=0)

    # For the variance-based metric ("sd"), return the square root transformation.
    if metric.lower() == "sd":
        avg_sra = np.sqrt(avg_sra)

    return avg_sra


def _smooth_sra_window(
    sra_values: np.ndarray, window_size: int = 10
) -> np.ndarray:
    """
    Smooth the SRA curve using a rolling window average.

    If window_size > len(sra_values), returns an array filled with the overall mean.

    Args:
        sra_values: 1D array of SRA values.
        window_size: Window size for rolling mean.

    Returns:
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

    Args:
        diffs: 1D array of differences.
        style: Aggregation style; "max" returns the maximum, otherwise the sum of squares.

    Returns:
        Aggregated value.
    """
    if style == "max":
        return np.max(diffs)
    else:
        return np.sum(diffs**2)


def _loo_var(
    nullobj: np.ndarray, i: int, sumall: np.ndarray, sumsqs: np.ndarray, B: int
) -> np.ndarray:
    """
    Compute the leave-one-out variance for permutation i.

    Args:
        nullobj: 2D array (D, B) of null SRA curves (each column is a permutation).
        i: Column index for leave-one-out.
        sumall: 1D array of row sums of nullobj.
        sumsqs: 1D array of row sums of squares of nullobj.
        B: Total number of permutations.

    Returns:
        1D array of leave-one-out variances.
    """
    return (
        (sumsqs - nullobj[:, i] ** 2)
        - ((sumall - nullobj[:, i]) ** 2) / (B - 1)
    ) / (B - 2)


def test_sra(
    observed_sra: np.ndarray,
    null_sra: np.ndarray,
    window_size: int = 1,
    style: str = "l2",
    standardise: bool = False,
    recompute_sd_for_loo: bool = True,
) -> dict:
    """
    Extended SRA test with optional smoothing and standardisation.

    Args:
        observed_sra: 1D array of observed SRA values.
        null_sra: 2D array (D, B) of null SRA curves (each column is one permutation run).
        window_size: Window size for smoothing (default 1 means no smoothing).
        style: Aggregation style ("l2" or "max").
        standardise: Whether to standardise differences by their standard deviation.
        recompute_sd_for_loo: If standardising, whether to recompute sd in the leave-one-out step.

    Returns:
        Dictionary with keys:
          - p_value: Permutation-based p-value.
          - T_obs: Aggregated statistic for the observed SRA.
          - T_null: Array of aggregated statistics from null permutations.
          - style, window_size, standardise.
    """
    if window_size > 1:
        obs_smoothed = _smooth_sra_window(observed_sra, window_size)
        D, B = null_sra.shape
        null_smoothed = np.empty_like(null_sra)
        for b in range(B):
            null_smoothed[:, b] = _smooth_sra_window(
                null_sra[:, b], window_size
            )
    else:
        obs_smoothed = observed_sra
        null_smoothed = null_sra

    D, B = null_smoothed.shape

    if standardise:
        global_sd = np.std(null_smoothed, axis=1, ddof=1)
        eps = np.finfo(float).eps
        mask = global_sd < eps
        if np.any(mask):
            non_zero_mean = np.mean(global_sd[~mask]) if np.any(~mask) else 1e-6
            if non_zero_mean < eps:
                non_zero_mean = 1e-6
            global_sd[mask] = non_zero_mean

    ref_all = np.mean(null_smoothed, axis=1)
    raw_diffs_obs = np.abs(obs_smoothed - ref_all)
    diffs_obs = raw_diffs_obs / global_sd if standardise else raw_diffs_obs
    T_obs = _aggregator(diffs_obs, style=style)

    T_null = np.empty(B)
    sumall = np.sum(null_smoothed, axis=1)
    if standardise and recompute_sd_for_loo:
        sumsqs = np.sum(null_smoothed**2, axis=1)
    for i in range(B):
        ref_loo = (sumall - null_smoothed[:, i]) / (B - 1)
        raw_diffs_loo = np.abs(null_smoothed[:, i] - ref_loo)
        if standardise:
            if recompute_sd_for_loo:
                sd_loo = np.sqrt(_loo_var(null_smoothed, i, sumall, sumsqs, B))
                mask = sd_loo < np.finfo(float).eps
                if np.any(mask):
                    non_zero_mean = (
                        np.mean(sd_loo[~mask]) if np.any(~mask) else 1e-6
                    )
                    if non_zero_mean < np.finfo(float).eps:
                        non_zero_mean = 1e-6
                    sd_loo[mask] = non_zero_mean
                diffs_loo = raw_diffs_loo / sd_loo
            else:
                diffs_loo = raw_diffs_loo / global_sd
        else:
            diffs_loo = raw_diffs_loo
        T_null[i] = _aggregator(diffs_loo, style=style)
    p_value = (np.sum(T_null >= T_obs) + 1) / (B + 1)
    return {
        "p_value": p_value,
        "T_obs": T_obs,
        "T_null": T_null,
        "style": style,
        "window_size": window_size,
        "standardise": standardise,
    }


def random_list_sra(
    object_input: np.ndarray,
    B: int = 1,
    n: int = 1,
    na_strings=None,
    nitems: int = None,
    metric: str = "sd",
    epsilon: float = 0.0,
) -> np.ndarray:
    """
    Generate simulated SRA curves using random permutations with enhanced features.

    Accepts either a list of ranked lists or a numpy array. The ranked lists may be
    incomplete or contain missing values. Note that our Python convention is that each
    row represents one ranked list.

    Args:
        object_input: List of lists/arrays or a 2D numpy array of ranked lists.
        B: Number of randomisation runs for SRA imputation.
        n: Number of permutation runs.
        na_strings: Additional values (besides np.nan) to treat as missing.
        nitems: Total number of items; if not provided, determined from the data.
        metric: 'sd' (default) or 'mad'.
        epsilon: Threshold for inclusion in S(d).

    Returns:
        A 2D numpy array of shape (list_length, n) where each column is a simulated SRA curve.
    """
    if na_strings is not None:
        for na_val in na_strings:
            object_input[object_input == na_val] = np.nan

    num_lists, list_length = object_input.shape
    notmiss = np.sum(~np.isnan(object_input), axis=1)
    if nitems is None:
        unique_non_na = (
            len(np.unique(object_input[~np.isnan(object_input)]))
            if np.any(~np.isnan(object_input))
            else 1
        )
        nitems = max(list_length, unique_non_na, 1)
    else:
        nitems = max(
            nitems,
            list_length,
            (
                len(np.unique(object_input[~np.isnan(object_input)]))
                if np.any(~np.isnan(object_input))
                else 1
            ),
        )

    # For each list, generate n random permutations (without replacement) of length equal to the number of observed items.
    sample_list = []
    for nn in notmiss:
        nn = int(nn)
        samples_for_list = [
            np.random.permutation(np.arange(1, nitems + 1))[:nn]
            for _ in range(n)
        ]
        sample_list.append(samples_for_list)

    # For each permutation run, construct a new ranked list matrix.
    sra_results = []
    for i in range(n):
        current_obj = np.full((num_lists, list_length), np.nan, dtype=float)
        for j in range(num_lists):
            nn = int(notmiss[j])
            if nn > 0:
                current_obj[j, :nn] = sample_list[j][i][:nn]
        # Compute SRA curve using the previously defined sra() function.
        sra_curve = sra(
            current_obj, epsilon=epsilon, B=B, nitems=nitems, metric=metric
        )
        sra_results.append(sra_curve)
    return np.column_stack(sra_results)


def test_delta_sra(
    object1: np.ndarray,
    object2: np.ndarray,
    nullobject1: np.ndarray,
    nullobject2: np.ndarray,
    window_size: int = 1,
    style: str = "l2",
    standardise: bool = False,
    recompute_sd_for_loo: bool = True,
    use_normal_approx: bool = True,
    tail_correction: bool = True,
) -> dict:
    """
    Compare two observed SRA curves via the difference in their aggregated statistics.

    The function optionally smooths the curves using a rolling window,
    standardises differences using row-wise standard deviations computed from the null curves,
    and then computes both an empirical and a parametric (normal or t-based) p-value.

    Parameters
    ----------
    object1 : np.ndarray
        Observed SRA curve for method 1 (1D array of length D).
    object2 : np.ndarray
        Observed SRA curve for method 2 (1D array of length D).
    nullobject1 : np.ndarray
        Null SRA curves for method 1 (2D array of shape (D, B)).
    nullobject2 : np.ndarray
        Null SRA curves for method 2 (2D array of shape (D, B)).
    window_size : int, optional
        Smoothing window size; if greater than 1 a rolling window average is applied.
    style : str, optional
        Aggregation style. Options include "l2" (sum-of-squares) or "max".
    standardise : bool, optional
        If True, differences are standardised by their estimated SD.
    recompute_sd_for_loo : bool, optional
        When standardising, recompute the leave‑one‑out SD rather than use the overall estimate.
    use_normal_approx : bool, optional
        Flag to use a parametric p‑value based on a (t‑distributed) normal approximation.
    tail_correction : bool, optional
        If True, apply a slight tail correction using a t-distribution with df=25.

    Returns
    -------
    dict
        A dictionary containing:
          - p_value: The parametric p‑value.
          - p_value_empirical: The empirical (permutation) p‑value.
          - T_obs: The difference in aggregated statistics for the observed curves.
          - T_null: A vector of differences from the null resamples.
          - null_fit: A dict with keys "mean" and "sd" (if using normal approximation), else None.
          - style, window_size, standardise, use_normal_approx, tail_correction.
    """
    eps = np.finfo(float).eps

    # Smooth observed curves if required
    if window_size > 1:
        obj1 = _smooth_sra_window(object1, window_size=window_size)
        obj2 = _smooth_sra_window(object2, window_size=window_size)
    else:
        obj1, obj2 = object1, object2

    # Smooth null curves along each column if required.
    if window_size > 1:
        nullobj1 = np.column_stack(
            [
                _smooth_sra_window(nullobject1[:, i], window_size=window_size)
                for i in range(nullobject1.shape[1])
            ]
        )
        nullobj2 = np.column_stack(
            [
                _smooth_sra_window(nullobject2[:, i], window_size=window_size)
                for i in range(nullobject2.shape[1])
            ]
        )
    # Else, keep as is.
    B = nullobj1.shape[1]
    if nullobj2.shape[1] != B:
        raise ValueError(
            "Both null matrices must have the same number of columns."
        )

    if standardise:
        # Compute row-wise SD (ddof=1) for each null object.
        global_sd1 = np.std(nullobj1, axis=1, ddof=1)
        global_sd2 = np.std(nullobj2, axis=1, ddof=1)
        # Replace any near-zero values by the mean of nonzero entries (or 1e-6).
        for global_sd in (global_sd1, global_sd2):
            mask = global_sd < eps
            if np.any(mask):
                non_zero_mean = (
                    np.mean(global_sd[~mask]) if np.any(~mask) else 1e-6
                )
                if non_zero_mean < eps:
                    non_zero_mean = 1e-6
                global_sd[mask] = non_zero_mean
    else:
        global_sd1 = global_sd2 = None

    # Compute row means over null columns.
    ref_all1 = np.mean(nullobj1, axis=1)
    ref_all2 = np.mean(nullobj2, axis=1)

    raw_diffs_obs1 = np.abs(obj1 - ref_all1)
    raw_diffs_obs2 = np.abs(obj2 - ref_all2)
    if standardise:
        diffs_obs1 = raw_diffs_obs1 / global_sd1
        diffs_obs2 = raw_diffs_obs2 / global_sd2
    else:
        diffs_obs1, diffs_obs2 = raw_diffs_obs1, raw_diffs_obs2

    T_obs1 = _aggregator(diffs_obs1, style=style)
    T_obs2 = _aggregator(diffs_obs2, style=style)
    T_obs = T_obs1 - T_obs2

    # Prepare for leave-one-out (LOO) calculations.
    T_null = np.zeros(B)
    sumall1 = np.sum(nullobj1, axis=1)
    sumall2 = np.sum(nullobj2, axis=1)
    if standardise and recompute_sd_for_loo:
        sumsqs1 = np.sum(nullobj1**2, axis=1)
        sumsqs2 = np.sum(nullobj2**2, axis=1)

    # Loop over each permutation (null column)
    for i in range(B):
        # For nullobject1:
        ref_loo1 = (sumall1 - nullobj1[:, i]) / (B - 1)
        raw_diffs_loo1 = np.abs(nullobj1[:, i] - ref_loo1)
        if standardise:
            if recompute_sd_for_loo:
                sd_loo1 = np.sqrt(_loo_var(nullobj1, i, sumall1, sumsqs1, B))
                mask = sd_loo1 < eps
                if np.any(mask):
                    non_zero_mean = (
                        np.mean(sd_loo1[~mask]) if np.any(~mask) else 1e-6
                    )
                    if non_zero_mean < eps:
                        non_zero_mean = 1e-6
                    sd_loo1[mask] = non_zero_mean
                diffs_loo1 = raw_diffs_loo1 / sd_loo1
            else:
                diffs_loo1 = raw_diffs_loo1 / global_sd1
        else:
            diffs_loo1 = raw_diffs_loo1
        T_null1 = _aggregator(diffs_loo1, style=style)

        # For nullobject2:
        ref_loo2 = (sumall2 - nullobj2[:, i]) / (B - 1)
        raw_diffs_loo2 = np.abs(nullobj2[:, i] - ref_loo2)
        if standardise:
            if recompute_sd_for_loo:
                sd_loo2 = np.sqrt(_loo_var(nullobj2, i, sumall2, sumsqs2, B))
                mask = sd_loo2 < eps
                if np.any(mask):
                    non_zero_mean = (
                        np.mean(sd_loo2[~mask]) if np.any(~mask) else 1e-6
                    )
                    if non_zero_mean < eps:
                        non_zero_mean = 1e-6
                    sd_loo2[mask] = non_zero_mean
                diffs_loo2 = raw_diffs_loo2 / sd_loo2
            else:
                diffs_loo2 = raw_diffs_loo2 / global_sd2
        else:
            diffs_loo2 = raw_diffs_loo2
        T_null2 = _aggregator(diffs_loo2, style=style)

        T_null[i] = T_null1 - T_null2

    p_value_empirical = (np.sum(T_null >= T_obs) + 1) / (B + 1)

    if use_normal_approx:
        null_mean = np.mean(T_null)
        null_sd = np.std(T_null, ddof=1)
        z_obs = (T_obs - null_mean) / null_sd
        if tail_correction:
            # Use t-distribution with df=25 for a slight tail correction.
            p_value_parametric = t.sf(z_obs, df=25)
        else:
            p_value_parametric = norm.sf(z_obs)
        if p_value_parametric < eps:
            p_value_parametric = np.exp(norm.logsf(z_obs))
    else:
        p_value_parametric = p_value_empirical
        null_mean, null_sd = None, None

    return {
        "p_value": p_value_parametric,
        "p_value_empirical": p_value_empirical,
        "T_obs": T_obs,
        "T_null": T_null,
        "null_fit": {"mean": null_mean, "sd": null_sd}
        if use_normal_approx
        else None,
        "style": style,
        "window_size": window_size,
        "standardise": standardise,
        "use_normal_approx": use_normal_approx,
        "tail_correction": tail_correction,
    }


def survival_gpd(x: float, xi: float, beta: float) -> float:
    """
    Compute the survival function for a Generalised Pareto Distribution (GPD).

    For xi near zero the GPD approximates an exponential tail.

    Parameters
    ----------
    x : float
        The value (or excess) at which to evaluate the survival.
    xi : float
        The shape parameter.
    beta : float
        The scale parameter.

    Returns
    -------
    float
        The survival probability.
    """
    if xi == 0:
        return np.exp(-x / beta)
    else:
        inner = 1 + (xi * x / beta)
        if inner <= 0:
            return 0.0
        else:
            return inner ** (-1 / xi)


def test_sra_gpd(
    object_: np.ndarray,
    nullobject: np.ndarray,
    window_size: int = 1,
    style: str = "l2",
    standardise: bool = False,
    recompute_sd_for_loo: bool = True,
    threshold_quantile: float = 0.90,
) -> dict:
    """
    Extend the SRA test by fitting a Generalised Pareto Distribution (GPD) to the tail
    of the null distribution if the observed aggregated statistic is extreme.

    First the standard test_sra procedure is run (assumed to be available) to obtain T_obs and T_null.
    If T_obs is below the threshold (the quantile of T_null) or if there are fewer than 30 null values
    in the tail, no tail fit is performed and the base result is returned.
    Otherwise, a GPD is fit to the excesses over the threshold and used to calculate a tail-adjusted p-value.

    Parameters
    ----------
    object_ : np.ndarray
        Observed SRA curve (1D array).
    nullobject : np.ndarray
        Null SRA curves (2D array with shape (D, B)).
    window_size : int, optional
        Smoothing window size.
    style : str, optional
        Aggregation style ("l2" or "max").
    standardise : bool, optional
        Whether to standardise differences.
    recompute_sd_for_loo : bool, optional
        Whether to recompute leave-one-out standard deviations.
    threshold_quantile : float, optional
        Quantile (between 0 and 1) to use as the threshold for tail fitting.

    Returns
    -------
    dict
        A dictionary with the base test_sra results and (if applicable) additional keys:
          - p_value_gpd: The tail-adjusted p-value.
          - gpd_fit: A dict containing the fitted GPD parameters (xi, beta) and threshold.
    """
    # Run the base SRA test (assumed to be defined in your library)
    base_result = test_sra(
        observed_sra=object_,
        null_sra=nullobject,
        window_size=window_size,
        style=style,
        standardise=standardise,
        recompute_sd_for_loo=recompute_sd_for_loo,
    )

    T_null = base_result["T_null"]
    T_obs = base_result["T_obs"]

    threshold = np.quantile(T_null, threshold_quantile)

    if T_obs < threshold:
        print("No tail fit needed")
        return base_result

    tail_data = T_null[T_null > threshold]
    if tail_data.size < 30:
        print(
            "Fewer than 30 null T-values exceed threshold; GPD tail fit may be unstable. Returning the empirical result."
        )
        return base_result

    # Fit GPD to the excesses over the threshold.
    excesses = tail_data - threshold
    # Fix location to 0 (i.e. excesses) during fitting.
    xi, loc, beta = genpareto.fit(excesses, floc=0)

    # Compute tail-based p-value for T_obs.
    if T_obs <= threshold:
        p_value_gpd = np.mean(T_null >= T_obs)
    else:
        excess_obs = T_obs - threshold
        tail_prob = genpareto.sf(excess_obs, c=xi, loc=0, scale=beta)
        # Combine the empirical CDF at threshold with the tail fit.
        F_threshold = np.mean(T_null < threshold)
        p_value_gpd = F_threshold + (1 - F_threshold) * tail_prob

    # Add the GPD tail-adjusted p-value and fit parameters to the result.
    base_result.update(
        {
            "p_value_gpd": p_value_gpd,
            "gpd_fit": {"xi": xi, "beta": beta, "threshold": threshold},
        }
    )
    return base_result
