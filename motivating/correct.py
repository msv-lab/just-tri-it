# signature: def max_sum_with_constraints(num_black_balls: int, num_white_balls: int, black_ball_values: List[int], white_ball_values: List[int]) -> int
from typing import List

def max_sum_with_constraints(num_black_balls: int, num_white_balls: int, black_ball_values: List[int], white_ball_values: List[int]) -> int:
    if not (1 <= num_black_balls <= 200000) or not (1 <= num_white_balls <= 200000):
        raise ValueError("Invalid input")
    if len(black_ball_values) != num_black_balls or len(white_ball_values) != num_white_balls:
        raise ValueError("Invalid input")
    if any(not (-10**9 <= value <= 10**9) for value in black_ball_values + white_ball_values):
        raise ValueError("Invalid input")

    # Sort black and white ball values in descending order
    black_ball_values.sort(reverse=True)
    white_ball_values.sort(reverse=True)

    max_sum = 0
    current_black_sum = 0
    current_white_sum = 0

    # Calculate prefix sums for black balls
    black_prefix_sum = [0] * (num_black_balls + 1)
    for i in range(1, num_black_balls + 1):
        black_prefix_sum[i] = black_prefix_sum[i - 1] + black_ball_values[i - 1]

    # Calculate prefix sums for white balls
    white_prefix_sum = [0] * (num_white_balls + 1)
    for i in range(1, num_white_balls + 1):
        white_prefix_sum[i] = white_prefix_sum[i - 1] + white_ball_values[i - 1]

    # For each possible number of white balls chosen, find the maximum possible sum
    for w in range(0, num_white_balls + 1):
        if w > num_black_balls:
            break
        current_black_sum = black_prefix_sum[w]
        current_white_sum = white_prefix_sum[w]
        max_sum = max(max_sum, current_black_sum + current_white_sum)

        # Try including more black balls than white balls chosen
        for b in range(w + 1, num_black_balls + 1):
            current_black_sum = black_prefix_sum[b]
            max_sum = max(max_sum, current_black_sum + current_white_sum)

    return max_sum

in_put = [
              3,
              2,
              [
                -3,
                5,
                1
              ],
              [
                2,
                -4
              ]
            ]
print(max_sum_with_constraints(*in_put))