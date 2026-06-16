# Pure math functions (nDCG, MAP, etc.)
import math

def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))

def ndcg_at_k(relevances: list[int], k: int) -> float:
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances) / ideal_dcg

def average_precision(relevances: list[int]) -> float:
    hits = 0
    precision_sum = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel:
            hits += 1
            precision_sum += hits / idx
    if hits == 0:
        return 0.0
    return precision_sum / hits