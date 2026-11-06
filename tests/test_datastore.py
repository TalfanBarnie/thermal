from datetime import UTC, datetime

import pytest

from thermal import datastore


def test_nominal_time_rounds_to_cycle():
    end = datetime(2024, 3, 16, 21, 57, 43, tzinfo=UTC)
    assert datastore.nominal_time(end, 15) == datetime(2024, 3, 16, 22, 0, tzinfo=UTC)
    end = datetime(2024, 3, 16, 23, 56, 0, tzinfo=UTC)
    assert datastore.nominal_time(end, 15) == datetime(2024, 3, 17, 0, 0, tzinfo=UTC)


class FakeCustomisation:
    def __init__(self, statuses):
        self._statuses = list(statuses)

    @property
    def status(self):
        s = self._statuses[0]
        if len(self._statuses) > 1:
            self._statuses.pop(0)
        return s


def test_wait_for_customisation_done():
    c = FakeCustomisation(["QUEUED", "RUNNING", "RUNNING", "DONE"])
    sleeps = []
    status = datastore.wait_for_customisation(c, 1, 100, sleep=sleeps.append)
    assert status == "DONE"
    assert len(sleeps) == 3


def test_wait_for_customisation_failed_returns_status():
    c = FakeCustomisation(["RUNNING", "FAILED"])
    assert datastore.wait_for_customisation(c, 1, 100, sleep=lambda s: None) == "FAILED"


def test_wait_for_customisation_timeout():
    c = FakeCustomisation(["RUNNING"])
    with pytest.raises(TimeoutError):
        datastore.wait_for_customisation(c, 1, -1, sleep=lambda s: None)
