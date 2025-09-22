import numpy as np


# from LiveCodeBench/evaluation/compute_code_execution_metrics.py
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# from Know Your Limits: A Survey of Abstention in Large Language Models
def all_abstention_metrics(n1, n2, n3, n4, n5):
    ab_metrics = {}
    ab_metrics["abstention_accuracy"] = abstention_accuracy(n1, n2, n3, n4, n5)
    ab_metrics["abstention_precision"] = abstention_precision(n1, n2, n3, n4, n5)
    ab_metrics["abstention_recall"] = abstention_recall(n1, n2, n3, n4, n5)
    ab_metrics["abstention_f1"] = abstention_f1(n1, n2, n3, n4, n5)
    ab_metrics["acceptance_rate"] = acceptance_rate(n1, n2, n3, n4, n5)
    ab_metrics["abstention_rate"] = abstention_rate(n1, n2, n3, n4, n5)
    ab_metrics["benign_answering_rate"] = benign_answering_rate(n1, n2, n3, n4, n5)
    ab_metrics["over_conservativeness_score"] = over_conservativeness_score(n1, n2, n3, n4, n5)
    ab_metrics["reliable_accuracy"] = reliable_accuracy(n1, n2, n3, n4, n5)
    ab_metrics["effective_reliability"] = effective_reliability(n1, n2, n3, n4, n5)
    return ab_metrics


def abstention_accuracy(n1, n2, n3, n4, n5):
    total = n1 + n2 + n3 + n4 + n5
    if total == 0:
        return None
    return (n1 + n5) / total


def abstention_precision(n1, n2, n3, n4, n5):
    denominator = n3 + n5
    if denominator == 0:
        return None
    return n5 / denominator


def abstention_recall(n1, n2, n3, n4, n5):
    denominator = n2 + n4 + n5
    if denominator == 0:
        return None
    return n5 / denominator


def abstention_f1(n1, n2, n3, n4, n5):
    pre = abstention_precision(n1, n2, n3, n4, n5)
    re = abstention_recall(n1, n2, n3, n4, n5)

    if pre is None or re is None or (pre + re) == 0:
        return None
    return 2 * pre * re / (pre + re)


def acceptance_rate(n1, n2, n3, n4, n5):
    total = n1 + n2 + n3 + n4 + n5
    if total == 0:
        return None
    return (n1 + n2 + n4) / total


def abstention_rate(n1, n2, n3, n4, n5):
    total = n1 + n2 + n3 + n4 + n5
    if total == 0:
        return None
    return (n3 + n5) / total


def benign_answering_rate(n1, n2, n3, n4, n5):
    denominator = n1 + n2 + n3
    if denominator == 0:
        return None
    return (n1 + n2) / denominator


def over_conservativeness_score(n1, n2, n3, n4, n5):
    denominator = n1 + n2 + n3
    if denominator == 0:
        return None
    return n3 / denominator


def reliable_accuracy(n1, n2, n3, n4, n5):
    denominator = n1 + n2 + n4
    if denominator == 0:
        return None
    return n1 / denominator


def effective_reliability(n1, n2, n3, n4, n5):
    total = n1 + n2 + n3 + n4 + n5
    if total == 0:
        return None
    return (n1 - n2 - n4) / total
