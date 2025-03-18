import numpy as np


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


def random_list_sra(
    num_lists: int, nitems: int, n_permutations: int
) -> np.ndarray:
    """Generate simulated SRAs using random permutations

    Args:
        num_lists: Number of lists (L in paper)
        nitems: Number of items (P in paper)
        n_permutations: Number of permutation runs (n in paper)

    Returns:
        Matrix where each column is a simulated SRA curve
    """
    null_sra_matrix = np.zeros((nitems, n_permutations))

    for i in range(n_permutations):
        # Generate random permutations by sorting random values
        random_values = np.random.rand(num_lists, nitems)
        permutations = np.argsort(random_values, axis=1) + 1  # 1-based ranks
        null_sra_matrix[:, i] = sra(permutations)

    return null_sra_matrix


def test_sra(
    observed_sra: np.ndarray,
    null_sra_matrix: np.ndarray,
    weights: np.ndarray | float = 1.0,
) -> float:
    """Compute permutation-based p-value for sequential rank agreement.

    Args:
        observed_sra: Array of shape (nitems,) containing observed SRA values
        null_sra_matrix: Matrix of shape (nitems, n_permutations) containing
                         null SRA curves from random_list_sra
        weights: Weight vector for depths. Scalar or array of shape (nitems,)

    Returns:
        p-value for observed SRA being different from null distribution
    """

    if isinstance(weights, (int, float)):
        weights = np.full_like(observed_sra, weights)
    elif weights.shape != observed_sra.shape:
        raise ValueError("Weights must match shape of observed_sra")

    null_mean = np.mean(null_sra_matrix, axis=1)
    t_obs = np.max(weights * np.abs(observed_sra - null_mean))

    # Distribution of test statistic under null
    B = null_sra_matrix.shape[1]
    t_null = np.zeros(B)
    for i in range(B):
        # Jackknife leave-one-out mean
        loo_mean = np.mean(np.delete(null_sra_matrix, i, axis=1), axis=1)
        t_null[i] = np.max(weights * np.abs(null_sra_matrix[:, i] - loo_mean))

    p_value = (np.sum(t_null >= t_obs) + 1) / (B + 1)

    return p_value


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


def test_sra_extended(
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


def random_list_sra_enhanced(
    object_input,
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
    # Convert input to an array where each row is a ranked list.
    if isinstance(object_input, list):
        num_lists = len(object_input)
        max_len = max(len(lst) for lst in object_input)
        obj = np.full((num_lists, max_len), np.nan, dtype=float)
        for i, lst in enumerate(object_input):
            lst_array = np.array(lst, dtype=float)
            if na_strings is not None:
                for na_val in na_strings:
                    lst_array[lst_array == na_val] = np.nan
            obj[i, : len(lst_array)] = lst_array
    else:
        obj = np.array(object_input, dtype=float)
        if na_strings is not None:
            for na_val in na_strings:
                obj[obj == na_val] = np.nan

    num_lists, list_length = obj.shape
    notmiss = np.sum(~np.isnan(obj), axis=1)
    if nitems is None:
        unique_non_na = (
            len(np.unique(obj[~np.isnan(obj)])) if np.any(~np.isnan(obj)) else 1
        )
        nitems = max(list_length, unique_non_na, 1)
    else:
        nitems = max(
            nitems,
            list_length,
            (
                len(np.unique(obj[~np.isnan(obj)]))
                if np.any(~np.isnan(obj))
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
