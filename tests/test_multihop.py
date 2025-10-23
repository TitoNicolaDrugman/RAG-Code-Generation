import pytest
from models.retrieval_providers import Doc
from S6.multihop import _dedup_and_rank

def test_dedup_and_rank_basic():
    docs = [
        Doc(repo="r1", doc_id=1, text="a", score=1.0),
        Doc(repo="r1", doc_id=1, text="a", score=0.9),
        Doc(repo="r1", doc_id=2, text="b", score=0.8),
        Doc(repo="r2", doc_id=3, text="c", score=0.7),
    ]
    out = _dedup_and_rank(docs, k_final=3)
    assert [(d.repo, d.doc_id) for d in out] == [("r1", 1), ("r1", 2), ("r2", 3)]
