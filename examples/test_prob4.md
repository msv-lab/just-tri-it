You are a treasure hunter exploring the vast lands of Algoria. You have come across a mystical cave filled with rare art
ifacts. Each artifact has a specific value and weight, and you are equipped with a knapsack that can carry a limited wei
ght. Your goal is to maximize the total value of the artifacts you can carry in your knapsack.

**Input:**

The first line contains two integers \( n \) and \( W \) — the number of artifacts and the maximum weight capacity of your knapsack.

The next \( n \) lines each contain two integers \( v_i \) and \( w_i \) — the value and weight of the \( i^{th} \) artifact.

**Output:**

Output a single integer — the maximum total value of artifacts you can carry without exceeding the weight capacity of the knapsack.

**Constraints:**

- \( 1 \leq n \leq 1000 \)
- \( 1 \leq W \leq 1000 \)
- \( 1 \leq v_i, w_i \leq 1000 \)

**Example:**

Input:
```
4 10
5 3
7 4
9 5
3 2
```

Output:
```
16
```

**Explanation:**

To achieve the maximum value of 16, you can choose the 2nd artifact (value 7, weight 4) and the 3rd artifact (value 9, weight 5), which together fit within the weight capacity of 10.