import numpy as np

def calculate_recall_at_k(recommended_items, true_items, k=10):
    if not true_items:
        return 0.0
    hits = len(set(recommended_items[:k]) & set(true_items))
    return hits / len(true_items)

def calculate_ndcg_at_k(recommended_items, true_items, k=10):
    if not true_items:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended_items[:k]):
        if item in true_items:
            dcg += 1.0 / np.log2(rank + 2)
            
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(true_items), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_recsys(recommend_func, actual_interactions, k=10):
    recalls, ndcgs = [], []
    for user_id, true_items in actual_interactions.items():
        recs = recommend_func(user_id, k=k)
        recalls.append(calculate_recall_at_k(recs, true_items, k))
        ndcgs.append(calculate_ndcg_at_k(recs, true_items, k))
    return np.mean(recalls), np.mean(ndcgs)
