from graphrag.eval.metrics import aggregate, exact_match, f1_score, normalize_answer, supporting_fact_f1


def test_normalize_answer_strips_articles_punctuation_case():
    assert normalize_answer("The Mask of Fu Manchu.") == normalize_answer("mask of fu manchu")


def test_exact_match_identical_after_normalization():
    assert exact_match("The Mask Of Fu Manchu", "mask of fu manchu.") == 1.0


def test_exact_match_different_answers():
    assert exact_match("Blind Shaft", "The Mask Of Fu Manchu") == 0.0


def test_f1_score_partial_overlap():
    # pred shares 1 of 2 gold tokens ("fu", "manchu") -> precision=1/1? check exact calc
    score = f1_score("Manchu", "Fu Manchu")
    assert 0.0 < score < 1.0


def test_f1_score_perfect_match():
    assert f1_score("Małgorzata Braunek", "Małgorzata Braunek") == 1.0


def test_f1_score_no_overlap():
    assert f1_score("completely different", "answer") == 0.0


def test_f1_score_empty_prediction_vs_nonempty_gold():
    assert f1_score("", "some answer") == 0.0


def test_supporting_fact_f1_perfect():
    p, r, f1 = supporting_fact_f1(["a", "b"], ["a", "b"])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_supporting_fact_f1_partial():
    p, r, f1 = supporting_fact_f1(["a", "c"], ["a", "b"])
    assert p == 0.5
    assert r == 0.5
    assert f1 == 0.5


def test_supporting_fact_f1_no_predictions():
    p, r, f1 = supporting_fact_f1([], ["a", "b"])
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_supporting_fact_f1_no_gold_and_no_prediction_is_perfect():
    assert supporting_fact_f1([], []) == (1.0, 1.0, 1.0)


def test_aggregate_empty_predictions():
    stats = aggregate([])
    assert stats == {"n": 0, "em": 0.0, "f1": 0.0, "support_f1": 0.0}


def test_aggregate_mean_across_two_questions():
    predictions = [
        {
            "predicted_answer": "Paris",
            "gold_answer": "Paris",
            "supporting_passage_ids": ["p1"],
            "gold_supporting_passage_ids": ["p1"],
        },
        {
            "predicted_answer": "wrong",
            "gold_answer": "Paris",
            "supporting_passage_ids": ["p2"],
            "gold_supporting_passage_ids": ["p1"],
        },
    ]
    stats = aggregate(predictions)
    assert stats["n"] == 2
    assert stats["em"] == 0.5
    # q1 supporting-fact F1 = 1.0 (exact match), q2 = 0.0 (no overlap) -> mean 0.5
    assert stats["support_f1"] == 0.5
