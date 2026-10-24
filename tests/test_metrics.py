from ragbench import metrics


def test_normalize_strips_articles_and_punctuation():
    assert metrics.normalize("The  Sunday Times!") == "sunday times"


def test_exact_match_and_f1():
    assert metrics.exact_match("The Sunday Times", "Sunday Times") == 1.0
    assert metrics.f1("Sunday Times newspaper", "Sunday Times") == 0.8
    assert metrics.f1("nothing", "Sunday Times") == 0.0


def test_contain_and_context_recall():
    assert metrics.contain("It was the Sunday Times.", "Sunday Times") == 1.0
    assert metrics.contain("", "x") == 0.0
    assert metrics.context_recall(["he worked at the sunday times ."], "Sunday Times") == 1.0
    assert metrics.context_recall(["unrelated"], "Sunday Times") == 0.0


def test_bootstrap_ci_brackets_mean():
    lo, hi = metrics.bootstrap_ci([0, 1] * 50)
    assert lo < 0.5 < hi
