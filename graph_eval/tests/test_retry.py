import pytest

from graph_eval.extraction.retry import RetryPolicy, classify_error


class Status(Exception):
    def __init__(self, code):
        super().__init__(str(code))
        self.status_code = code


class APITimeoutError(Exception):
    pass


class WeirdError(Exception):
    pass


def test_classify_error():
    assert classify_error(Status(429)) == "retry"
    assert classify_error(Status(503)) == "retry"
    assert classify_error(Status(400)) == "fail"
    assert classify_error(Status(401)) == "fail"
    assert classify_error(Status(404)) == "fail"
    assert classify_error(APITimeoutError()) == "retry"
    assert classify_error(WeirdError()) == "fail"


def test_retry_recovers_after_transient():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Status(429)
        return "ok"

    assert RetryPolicy(max_retries=5, sleep=lambda s: None).run(fn) == "ok"
    assert calls["n"] == 3


def test_retry_exhausts_then_raises():
    def always_500():
        raise Status(500)

    with pytest.raises(Status):
        RetryPolicy(max_retries=2, sleep=lambda s: None).run(always_500)


def test_fail_fast_does_not_retry():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise Status(400)

    with pytest.raises(Status):
        RetryPolicy(max_retries=5, sleep=lambda s: None).run(fn)
    assert calls["n"] == 1
