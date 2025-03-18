import numpy as np


def compute_rank_matrix(ranked_lists_array: np.ndarray) -> np.ndarray:
    """Create matrix where entry [i,j] = rank of item j in list i (R_l(X_p))

    Args:
        ranked_lists_array: ndarray (num_lists, nitems)
            - Rows (axis=0): ranked lists, from 1 to num_lists
            - Columns (axis=1): ranks, from 1 to nitems
            - Values: 1-based indices of items, specifying item ID

    Returns:
        rank_matrix: ndarray (num_lists, nitems)
            - Rows (axis=0): item rank lists, from 1 to num_lists
            - Columns (axis=1): items, from 1 to nitems
            - Values: rank of item in list, from 1 to nitems
    """
    num_lists, nitems = ranked_lists_array.shape
    rank_matrix = np.zeros((num_lists, nitems), dtype=int)
    rows = np.arange(num_lists)[:, np.newaxis]  # Shape (num_lists, 1)
    cols = ranked_lists_array - 1  # Convert to 0-based indices
    rank_matrix[rows, cols] = np.arange(
        1, nitems + 1
    )  # Set ranks for each list

    return rank_matrix


def sra(
    ranked_lists_array: np.ndarray, epsilon: float | np.ndarray = 0.0
) -> np.ndarray:
    """Compute sequential rank agreement

    Args:
        ranked_lists_array: ndarray (num_lists, nitems)
            - Rows (axis=0): ranked lists, from 1 to num_lists
            - Columns (axis=1): ranks, from 1 to nitems
            - Values: 1-based indices of items, specifying item ID

        epsilon: Threshold for agreement, either a scalar or array of length nitems

    Returns:
        sra_values: Array where sra_values[d-1] = sra(d) for depth d=1..nitems
    """

    # Compute rank matrix R_l(X_p) for all lists and items
    ranks = compute_rank_matrix(ranked_lists_array)
    nitems = ranks.shape[1]

    # Convert epsilon to depth-indexed array
    if isinstance(epsilon, (int, float)):
        epsilons = np.full(nitems, float(epsilon))
    else:
        if len(epsilon) != nitems:
            raise ValueError("epsilon length must match number of items")
        epsilons = np.asarray(epsilon)

    # Calculate empirical agreements (vars) A_L(X_p) for each item (Eq 2.2)
    item_agreements = np.var(ranks, axis=0, ddof=1)

    sra_values = np.zeros(nitems)
    for depth in range(1, nitems + 1):
        e = epsilons[depth - 1]
        # Compute S_L(d) - items in top d of any list (Eq 2.4 with ε=0)
        depth_set = np.mean(ranks <= depth, axis=0) > e

        if np.any(depth_set):
            # Compute sra(d) = mean{A_L(X_p) | X_p ∈ S_L(d)} (Eq 2.6)
            sra_values[depth - 1] = np.mean(item_agreements[depth_set])
        else:
            sra_values[depth - 1] = 0.0  # Case when |S_L(d)| = 0

    return sra_values


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
