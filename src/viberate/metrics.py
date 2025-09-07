import numpy as np

from viberate.utils import print_annotated_hr


# from LiveCodeBench/evaluation/compute_code_execution_metrics.py
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# from Know Your Limits: A Survey of Abstention in Large Language Models
def all_metrics_abt(n1, n2, n3, n4, n5):
    print_annotated_hr("abstention_accuracy")
    print(abstention_accuracy(n1, n2, n3, n4, n5))

    print_annotated_hr("abstention_precision")
    print(abstention_precision(n1, n2, n3, n4, n5))

    print_annotated_hr("abstention_recall")
    print(abstention_recall(n1, n2, n3, n4, n5))

    print_annotated_hr("abstention_f1")
    print(abstention_f1(n1, n2, n3, n4, n5))

    print_annotated_hr("coverage_rate")
    print(coverage_rate(n1, n2, n3, n4, n5))

    print_annotated_hr("abstention_rate")
    print(abstention_rate(n1, n2, n3, n4, n5))

    print_annotated_hr("benign_answering_rate")
    print(benign_answering_rate(n1, n2, n3, n4, n5))

    print_annotated_hr("over_conservativeness_score")
    print(over_conservativeness_score(n1, n2, n3, n4, n5))

    print_annotated_hr("reliable_accuracy")
    print(reliable_accuracy(n1, n2, n3, n4, n5))

    print_annotated_hr("reliability_precision")
    print(effective_reliability(n1, n2, n3, n4, n5))


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


def coverage_rate(n1, n2, n3, n4, n5):
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
    denominator = n1 + n2 + n3  # 修复了原代码中的语法错误
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
