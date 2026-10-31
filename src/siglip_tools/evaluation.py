"""Retrieval metrics computed from complete rankings and fixed relevance labels."""

from math import fsum


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, list[str]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Return per-query metrics and their unweighted arithmetic means.

    Every ranking must contain each ID in the same nonempty gallery exactly
    once. Query sets must match, and each query must have nonempty, unique
    relevant IDs drawn from that gallery. Cutoffs must be nonempty, unique
    positive integers; booleans are not accepted as integers.

    Hit@K indicates any positive in the first K results. Recall@K is the
    fraction of all positives in those results. MRR uses the first positive's
    rank; AP averages precision at every positive's rank in the full ranking.
    Aggregate ``average_precision`` is therefore mAP. Query IDs and cutoffs
    are sorted in the output, and the input dictionaries and lists are unchanged.

    Raise ValueError when any of these input requirements is violated.
    """
    if not rankings or rankings.keys() != qrels.keys():
        raise ValueError("rankings and qrels must have matching nonempty query sets")
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers and cannot be empty")
    if len(set(ks)) != len(ks):
        raise ValueError("ks must contain unique cutoffs")

    per_query = {}
    gallery = None
    for query_id in sorted(rankings):
        ranking = rankings[query_id]
        if not ranking:
            raise ValueError(f"ranking for {query_id!r} must be nonempty")
        ranked_ids = set(ranking)
        if len(ranked_ids) != len(ranking):
            raise ValueError(f"ranking for {query_id!r} contains duplicate image IDs")
        if gallery is None:
            gallery = ranked_ids
        elif ranked_ids != gallery:
            raise ValueError("all rankings must contain the same complete gallery")

        relevant = set(qrels[query_id])
        if not relevant:
            raise ValueError(f"qrels for {query_id!r} must contain at least one relevant image ID")
        if len(relevant) != len(qrels[query_id]):
            raise ValueError(f"qrels for {query_id!r} contains duplicate relevant image IDs")
        if not relevant <= ranked_ids:
            raise ValueError(
                f"qrels for {query_id!r} contains image IDs outside the ranking gallery"
            )

        metrics = {}
        for k in sorted(ks):
            retrieved = sum(image_id in relevant for image_id in ranking[:k])
            metrics[f"hit_at_{k}"] = float(retrieved > 0)
            metrics[f"recall_at_{k}"] = retrieved / len(relevant)

        relevant_ranks = [rank for rank, image_id in enumerate(ranking, 1) if image_id in relevant]
        metrics["mrr"] = 1 / relevant_ranks[0]
        metrics["average_precision"] = fsum(
            count / rank for count, rank in enumerate(relevant_ranks, 1)
        ) / len(relevant)
        per_query[query_id] = metrics

    aggregate = {
        name: fsum(metrics[name] for metrics in per_query.values()) / len(per_query)
        for name in next(iter(per_query.values()))
    }
    return {"per_query": per_query, "aggregate": aggregate}
