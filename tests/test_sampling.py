from collections import Counter
from datetime import UTC, datetime

from thermal import sampling


def test_stratified_sample_counts_and_cells():
    start, end = datetime(2019, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)
    ts = sampling.stratified_sample(start, end, 576, 15, seed=1)
    assert len(ts) == 576
    assert all(start <= t < end for t in ts)
    assert all(t.minute % 15 == 0 for t in ts)
    cells = Counter((t.month, t.hour) for t in ts)
    assert len(cells) == 288 and set(cells.values()) == {2}


def test_stratified_sample_exclusion_and_determinism():
    start, end = datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    excl = lambda t: t.month == 6  # noqa: E731
    a = sampling.stratified_sample(start, end, 100, 15, exclude=excl, seed=3)
    b = sampling.stratified_sample(start, end, 100, 15, exclude=excl, seed=3)
    assert a == b
    assert not any(t.month == 6 for t in a)


def test_list_roundtrip(tmp_path):
    ts = [datetime(2024, 3, 16, 22, 0, tzinfo=UTC), datetime(2024, 3, 17, 1, 15, tzinfo=UTC)]
    p = tmp_path / "l.csv"
    sampling.write_list(p, ts, "msg-0deg")
    assert sampling.read_list(p) == [("msg-0deg", t) for t in ts]


def test_regular_range():
    r = sampling.regular_range(datetime(2024, 1, 1), datetime(2024, 1, 1, 3), 60)
    assert len(r) == 3 and r[0].tzinfo is UTC
