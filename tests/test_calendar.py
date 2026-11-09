"""The timestamp -> session mapping, checked on hand-picked edge cases."""
import pandas as pd
import pytest

from marketmood.calendar import localise_publication_times

NY = "America/New_York"


def one(cal, s):
    return cal.assign_session(pd.Series([pd.Timestamp(s, tz=NY)])).iloc[0]


def test_before_close_same_day(cal):
    assert one(cal, "2019-03-13 15:59:59") == pd.Timestamp("2019-03-13")


def test_exactly_at_close_goes_to_next_session(cal):
    assert one(cal, "2019-03-13 16:00:00") == pd.Timestamp("2019-03-14")


def test_after_close_goes_to_next_session(cal):
    assert one(cal, "2019-03-13 18:30") == pd.Timestamp("2019-03-14")


def test_premarket_counts_for_same_day(cal):
    assert one(cal, "2019-03-13 07:00") == pd.Timestamp("2019-03-13")


def test_friday_evening_and_weekend_go_to_monday(cal):
    assert one(cal, "2019-03-15 17:00") == pd.Timestamp("2019-03-18")
    assert one(cal, "2019-03-16 12:00") == pd.Timestamp("2019-03-18")


def test_holiday_goes_to_next_session(cal):
    # 4 July 2019 was a Thursday and a market holiday
    assert one(cal, "2019-07-04 10:00") == pd.Timestamp("2019-07-05")


def test_early_close_day(cal):
    # 29 Nov 2019 (day after Thanksgiving) closed at 13:00
    assert one(cal, "2019-11-29 12:59") == pd.Timestamp("2019-11-29")
    assert one(cal, "2019-11-29 14:00") == pd.Timestamp("2019-12-02")


def test_utc_input_and_dst(cal):
    # 20:30 UTC is 16:30 EDT in summer (after close) but 15:30 EST in winter (before close)
    summer = cal.assign_session(pd.Series([pd.Timestamp("2019-07-10 20:30", tz="UTC")])).iloc[0]
    winter = cal.assign_session(pd.Series([pd.Timestamp("2019-01-10 20:30", tz="UTC")])).iloc[0]
    assert summer == pd.Timestamp("2019-07-11")
    assert winter == pd.Timestamp("2019-01-10")


def test_naive_timestamps_are_rejected(cal):
    with pytest.raises(ValueError):
        cal.assign_session(pd.Series([pd.Timestamp("2019-03-13 10:00")]))


def test_every_item_lands_strictly_before_its_decision_point(cal, news_items):
    s = cal.assign_session(news_items["published_utc"])
    close = cal.decision_time(s)
    assert (news_items["published_utc"].values < close.values).all()


def test_date_only_items_are_pushed_past_the_close(cal):
    raw = pd.Series(["2019-03-13 00:00:00 UTC", "2019-03-13 14:05:00 UTC"])
    mask = pd.Series([True, False])
    ts = localise_publication_times(raw, "UTC", mask)
    s = cal.assign_session(ts)
    # date-only item cannot be assumed to be pre-close -> next session
    assert s.iloc[0] == pd.Timestamp("2019-03-14")
    # 14:05 UTC = 10:05 New York -> same session
    assert s.iloc[1] == pd.Timestamp("2019-03-13")


def test_fnspid_items_inform_the_next_session_only(cal):
    """FNSPID dates are used at day precision: an item dated D can only reach
    the first session that closes after D has ended."""
    from marketmood.data.news import end_of_stated_day

    stated = pd.Series([
        "2019-03-13 00:00:00 UTC",   # date only, Wednesday
        "2019-03-13 21:30:00 UTC",   # clock time is ignored
        "2019-03-15 10:00:00 UTC",   # Friday -> Monday
        "2019-07-03 00:00:00 UTC",   # day before the 4 July holiday -> 5 July
    ])
    s = cal.assign_session(end_of_stated_day(stated))
    assert list(s) == [pd.Timestamp("2019-03-14"), pd.Timestamp("2019-03-14"),
                       pd.Timestamp("2019-03-18"), pd.Timestamp("2019-07-05")]
    # never the stated day itself, so after-close news cannot reach that day's forecast
    days = pd.to_datetime(stated.str.slice(0, 10))
    assert (s.values > days.values).all()
