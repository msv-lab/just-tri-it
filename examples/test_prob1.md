The Fibonacci sequence is a famous mathematical sequence defined as follows:

* **F(0) = 0**
* **F(1) = 1**
* For **n ≥ 2**, **F(n) = F(n - 1) + F(n - 2)**

Given a non-negative integer **n**, compute the **n-th Fibonacci number**.

**Input:**
A single integer **n** (**0 ≤ n ≤ 10^6**)

**Output:**
Output a single integer — the **n-th Fibonacci number modulo 10^9 + 7**.

**Example Input 1:**
```
5
```

**Example Output 1:**
```
5
```

**Example Input 2:**
```
10
```

**Example Output 2:**
```
55
```

**Note**

* Since **n** can be large, make sure your solution is efficient enough to handle large inputs within reasonable time and memory limits.
* Use fast algorithms like matrix exponentiation or iterative dynamic programming to compute the answer efficiently.