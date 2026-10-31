from copy import deepcopy
from json import dumps

import pytest

from siglip_tools.evaluation import evaluate_rankings


def test_metrics_use_all_relevant_items_and_full_ranking():
    result = evaluate_rankings({"query": ["b", "a", "c"]}, {"query": ["a", "c"]})

    expected = {
        "hit_at_1": 0.0,
        "recall_at_1": 0.0,
        "hit_at_5": 1.0,
        "recall_at_5": 1.0,
        "hit_at_10": 1.0,
        "recall_at_10": 1.0,
        "mrr": 0.5,
        "average_precision": (1 / 2 + 2 / 3) / 2,
    }
    assert result["per_query"]["query"] == pytest.approx(expected)
    assert result["aggregate"] == pytest.approx(expected)


def test_aggregate_means_weight_queries_equally_with_custom_cutoffs():
    result = evaluate_rankings(
        {"many": ["b", "a", "c"], "one": ["c", "b", "a"]},
        {"many": ["a", "c"], "one": ["c"]},
        ks=(1, 2, 10),
    )

    assert result["per_query"]["many"] == pytest.approx(
        {
            "hit_at_1": 0.0,
            "recall_at_1": 0.0,
            "hit_at_2": 1.0,
            "recall_at_2": 0.5,
            "hit_at_10": 1.0,
            "recall_at_10": 1.0,
            "mrr": 0.5,
            "average_precision": 7 / 12,
        }
    )
    assert result["per_query"]["one"] == pytest.approx(
        {
            "hit_at_1": 1.0,
            "recall_at_1": 1.0,
            "hit_at_2": 1.0,
            "recall_at_2": 1.0,
            "hit_at_10": 1.0,
            "recall_at_10": 1.0,
            "mrr": 1.0,
            "average_precision": 1.0,
        }
    )
    assert result["aggregate"] == pytest.approx(
        {
            "hit_at_1": 0.5,
            "recall_at_1": 0.5,
            "hit_at_2": 1.0,
            "recall_at_2": 0.75,
            "hit_at_10": 1.0,
            "recall_at_10": 1.0,
            "mrr": 0.75,
            "average_precision": 19 / 24,
        }
    )


def test_average_precision_uses_relevant_items_beyond_largest_cutoff():
    result = evaluate_rankings(
        {"query": ["a", "b", "c", "d"]}, {"query": ["a", "d"]}, ks=(1,)
    )

    assert result["aggregate"] == pytest.approx(
        {"hit_at_1": 1.0, "recall_at_1": 0.5, "mrr": 1.0, "average_precision": 0.75}
    )


def test_output_is_deterministic_and_inputs_are_unchanged():
    rankings = {"z": ["c", "b", "a"], "a": ["b", "a", "c"]}
    qrels = {"a": ["a", "c"], "z": ["c"]}
    original = deepcopy((rankings, qrels))

    result = evaluate_rankings(rankings, qrels, ks=(10, 2, 1))
    reordered = evaluate_rankings(
        {"a": ["b", "a", "c"], "z": ["c", "b", "a"]},
        {"z": ["c"], "a": ["c", "a"]},
        ks=(1, 2, 10),
    )

    assert dumps(result) == dumps(reordered)
    assert list(result["per_query"]) == ["a", "z"]
    assert (rankings, qrels) == original


@pytest.mark.parametrize(
    ("rankings", "qrels"),
    [
        ({}, {}),
        ({"query": ["a"]}, {}),
        ({}, {"query": ["a"]}),
        ({"query": ["a"]}, {"different_query": ["a"]}),
        ({"query": ["a"], "extra_query": ["a"]}, {"query": ["a"]}),
    ],
    ids=["both-empty", "qrels-empty", "rankings-empty", "different-ids", "missing-qrels"],
)
def test_rejects_nonmatching_or_empty_query_sets(rankings, qrels):
    with pytest.raises(ValueError):
        evaluate_rankings(rankings, qrels)


@pytest.mark.parametrize(
    "ks",
    [(), (0,), (-1,), (True,), (False,), (1.0,), ("1",), (1, 1)],
    ids=["empty", "zero", "negative", "true", "false", "float", "string", "duplicate"],
)
def test_rejects_invalid_cutoffs(ks):
    with pytest.raises(ValueError):
        evaluate_rankings({"query": ["a"]}, {"query": ["a"]}, ks=ks)


@pytest.mark.parametrize(
    ("ranking", "relevant"),
    [([], ["a"]), (["a", "a"], ["a"]), (["a"], []), (["a"], ["a", "a"]), (["a"], ["b"])],
    ids=[
        "empty-ranking",
        "duplicate-ranking",
        "empty-positives",
        "duplicate-positives",
        "unknown-id",
    ],
)
def test_rejects_invalid_rankings_or_relevance_labels(ranking, relevant):
    with pytest.raises(ValueError):
        evaluate_rankings({"query": ranking}, {"query": relevant})


@pytest.mark.parametrize("second_ranking", [["a"], ["a", "c"]], ids=["missing-id", "different-id"])
def test_rejects_rankings_that_do_not_cover_the_same_gallery(second_ranking):
    with pytest.raises(ValueError):
        evaluate_rankings(
            {"first": ["a", "b"], "second": second_ranking},
            {"first": ["a"], "second": ["a"]},
        )
