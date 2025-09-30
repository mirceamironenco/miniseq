from miniseq.datasets import (
    ChainedVerifier,
    EqualityVerifier,
    MapVerifier,
    Verifier,
)


def test_equality_verifier():
    verifier = EqualityVerifier()
    assert verifier(guess="a", gold="a") == {"equality": 1.0}
    assert verifier(guess="a", gold="b") == {"equality": 0.0}


def test_map_verifier():
    def my_verifier_map(guess, gold, document):
        return {"my_score": 1.0 if guess.startswith(gold) else 0.0}

    verifier = MapVerifier(my_verifier_map)
    assert verifier(guess="apple", gold="app", document=None) == {"my_score": 1.0}
    assert verifier(guess="banana", gold="app", document=None) == {"my_score": 0.0}


def test_chained_verifier():
    verifier1 = EqualityVerifier()
    verifier2 = Verifier.from_outcome_map(
        lambda guess, gold, **kwargs: len(guess) == len(gold), name="same_len"
    )

    chained = ChainedVerifier(verifier1, verifier2)
    scores = chained(guess="a", gold="a")
    assert scores == {"equality": 1.0, "same_len": 1.0}

    scores = chained(guess="a", gold="b")
    assert scores == {"equality": 0.0, "same_len": 1.0}

    scores = chained(guess="ab", gold="c")
    assert scores == {"equality": 0.0, "same_len": 0.0}


def test_chained_verifier_with_name_clash():
    verifier1 = EqualityVerifier()
    verifier2 = EqualityVerifier()  # Same name

    chained = ChainedVerifier(verifier1, verifier2)
    scores = chained(guess="a", gold="a")
    assert scores == {"equality": 1.0, "equality_1": 1.0}
