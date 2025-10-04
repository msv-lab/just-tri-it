from typing import List

from typing import List

from typing import List

from typing import List


def fiber_max_sum_with_constraints_wrt_num_black_balls(
        max_value_sum_of_selected_balls: int,
        num_black_balls: int,
        black_ball_values: List[int],
        white_ball_values: List[int]
) -> List[int]:
    """
    Finds all possible numbers of white balls that result in a specific maximum sum.

    Given a target sum, a number of black balls with their values, and a pool of
    available white balls, this function returns a list of all integers M
    (number of white balls chosen) such that the maximum possible sum under the
    constraint (number of black balls >= number of white balls) equals the
    target sum.

    Args:
        max_value_sum_of_selected_balls: The target maximum sum to achieve.
        num_black_balls: The total number of black balls available (N).
        black_ball_values: A list of the values of the black balls.
        white_ball_values: A list of the values for the available pool of white balls.

    Returns:
        A sorted list of all possible values for M (the number of white balls chosen)
        that can produce the target maximum sum.
    """
    N = num_black_balls
    B = sorted(black_ball_values, reverse=True)

    # --- 1. Precomputation on Black Balls ---

    # prefix_B[k] = sum of the k largest black ball values
    prefix_B = [0] * (N + 1)
    for i in range(N):
        prefix_B[i + 1] = prefix_B[i] + B[i]

    # pos_B_suffix[k] = sum of positive black balls from index k to the end
    pos_B_suffix = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        pos_B_suffix[i] = pos_B_suffix[i + 1]
        if B[i] > 0:
            pos_B_suffix[i] += B[i]

    # f[k] = max sum from black balls if we commit to taking the top k
    f = [0] * (N + 1)
    for k in range(N + 1):
        f[k] = prefix_B[k] + pos_B_suffix[k]

    # --- 2. Precomputation on White Balls ---

    max_M = len(white_ball_values)
    W = sorted(white_ball_values, reverse=True)

    # prefix_W[k] = sum of the k largest white ball values
    prefix_W = [0] * (max_M + 1)
    for i in range(max_M):
        prefix_W[i + 1] = prefix_W[i] + W[i]

    # --- 3. Precompute Maximum Possible Sums ---

    # The number of pairs `k` can go up to min(N, max_M)
    k_limit = min(N, max_M)

    # max_sum_upto_k_pairs[k] will store the maximum possible sum if we can
    # form up to k pairs of (black, white) balls.
    max_sum_upto_k_pairs = [-float('inf')] * (k_limit + 1)

    # The baseline is choosing 0 balls (sum=0) or only positive black balls.
    current_max = max(0, f[0])
    if k_limit >= 0:
        max_sum_upto_k_pairs[0] = current_max

    # Calculate the running maximum for k = 1 to k_limit
    for k in range(1, k_limit + 1):
        # Sum for exactly k pairs is sum of top k black + top k white + remaining positive black
        sum_for_k_pairs = f[k] + prefix_W[k]
        current_max = max(current_max, sum_for_k_pairs)
        max_sum_upto_k_pairs[k] = current_max

    # --- 4. Find All Valid M ---

    solutions = []
    target_sum = max_value_sum_of_selected_balls

    # Iterate through every possible number of white balls to choose
    for M in range(max_M + 1):
        # For a given M, the number of pairs we can form is at most min(N, M)
        k_idx = min(N, M)

        # The maximum possible sum for this M is the precomputed max over all
        # possible pair counts (from 0 to k_idx)
        achieved_max_sum = max_sum_upto_k_pairs[k_idx]

        if achieved_max_sum == target_sum:
            solutions.append(M)

    return solutions


in_put = [
              8,
              3,
              [
                2,
                3
              ],
              [
                1,
                2,
                3
              ]
            ]

print(fiber_max_sum_with_constraints_wrt_num_black_balls(*in_put))