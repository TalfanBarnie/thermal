from datetime import UTC, date, datetime

from thermal.calendar import Calendar, Eruption, load


def test_window_and_exclusion(tmp_path):
    e = Eruption("Sundhnukur", date(2024, 3, 16), date(2024, 5, 9))
    cal = Calendar([e], buffer_days=30)
    assert cal.excluded(datetime(2024, 3, 16, 0, 0, tzinfo=UTC))
    assert not cal.excluded(datetime(2024, 3, 15, 23, 45, tzinfo=UTC))
    assert cal.excluded(datetime(2024, 6, 8, 12, 0, tzinfo=UTC))  # within 30 day tail
    assert not cal.excluded(datetime(2024, 6, 9, 0, 0, tzinfo=UTC))
    assert cal.eruption_at(datetime(2024, 4, 1, tzinfo=UTC)) is e
    assert cal.eruption_at(datetime(2024, 6, 1, tzinfo=UTC)) is None


def test_ongoing_eruption_excludes_until_today():
    e = Eruption("X", date(2020, 1, 1), None)
    cal = Calendar([e])
    assert cal.excluded(datetime.now(UTC))


def test_load_csv(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text(
        "volcano,start_date,end_date,source,notes\n"
        "B,2014-08-29,2015-02-27,url,\n"
        "A,2010-03-20,2010-04-12,url,fissure\n"
        ",,,,\n"
    )
    es = load(p)
    assert [e.volcano for e in es] == ["A", "B"]
    assert es[1].end == date(2015, 2, 27)
