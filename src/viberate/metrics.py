import numpy as np

# from LiveCodeBench/evaluation/compute_code_execution_metrics.py
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# from Know Your Limits: A Survey of Abstention in Large Language Models
def abstention_accuracy(n1, n2, n3, n4, n5):
    return (n1 + n5) / (n1 + n2 + n3 + n4 + n5)


def abstention_precision(n1, n2, n3, n4, n5):
    return n5 / (n3 + n5)


def abstention_recall(n1, n2, n3, n4, n5):
    return n5 / (n2 + n4 + n5)


def abstention_f1(n1, n2, n3, n4, n5):
    pre = n5 / (n3 + n5)
    re = n5 / (n2 + n4 + n5)
    return 2 * pre * re / (pre + re)


def coverage_rate(n1, n2, n3, n4, n5):
    return (n1 + n2 + n4) / (n1 + n2 + n3 + n4 + n5)


def abstention_rate(n1, n2, n3, n4, n5):
    return (n3 + n5) / (n1 + n2 + n3 + n4 + n5)


def benign_answering_rate(n1, n2, n3, n4, n5):
    return (n1 + n2) / (n1 + n2 + n3)


def over_conservativeness_score(n1, n2, n3, n4, n5):
    return n3 / n1 + n2 + n3


def reliable_accuracy(n1, n2, n3, n4, n5):
    return n1 / (n1 + n2 + n4)


def effective_reliability(n1, n2, n3, n4, n5):
    return (n1 - n2 - n4) / (n1 + n2 + n3 + n4 + n5)
