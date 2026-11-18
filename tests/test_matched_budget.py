"""Tests for the matched false-alarm budget reading of regime predictions."""

import numpy as np
import pandas as pd

from tfm.data import hupa
from tfm.eval.alarm import recall_at_budget
from tfm.eval.metrics import Predictions
from tfm.training import external


def _events_fn(events: dict):
    return lambda patient, preds: external.scorable(events[patient], preds)


def _series(pid: str, seed: int, drop_60_head: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    target = pd.date_range("2018-06-13", periods=800, freq="15min")
    y = 110 + 60 * np.sin(np.arange(800) / 25)
    out = {}
    for h in (30, 60):
        keep = slice(drop_60_head if h == 60 else 0, None)
        mu = y + rng.normal(0, 15, 800)
        pred = Predictions(y[keep], mu[keep], (target - pd.Timedelta(minutes=h))[keep].to_numpy(),
                           pid, h, sigma=np.full(800, 15.0)[keep])
        pred.meta["episode_gap_min"] = hupa.ALARM_EPISODE_GAP_MIN
        out[h] = pred
    return out


def _events(pid: str) -> pd.DataFrame:
    pred = _series(pid, 0)[30]
    low = pred.y_true < 70
    starts = np.flatnonzero(low & ~np.r_[False, low[:-1]])
    ends = np.flatnonzero(low & ~np.r_[low[1:], False])
    t = pd.DatetimeIndex(pred.target_ts)
    return pd.DataFrame({"start": t[starts], "end": t[ends]})


def _run(drop_60_head: int = 0) -> dict:
    return {(s, 1): {p: _series(p, s, drop_60_head) for p in ("a", "b")} for s in (0, 1)}


def test_matched_recall_stays_within_the_budget():
    events = {p: _events(p) for p in "ab"}
    table = recall_at_budget(_run(), _events_fn(events), budget=1.5)
    assert set(table["horizon_min"]) == {30, 60}
    assert table["within_budget"].all()
    assert (table["false_alarms_per_day"] <= 1.5).all()
    loose = recall_at_budget(_run(), _events_fn(events), budget=50.0)
    assert (loose["recall"].to_numpy() > table["recall"].to_numpy()).all()


def test_an_unreachable_budget_is_flagged_not_hidden():
    table = recall_at_budget(_run(), _events_fn({p: _events(p) for p in "ab"}), budget=1.0)
    assert not table["within_budget"].any()
    assert (table["false_alarms_per_day"] > 1.0).all()


def test_matched_recall_can_be_restricted_to_some_patients():
    events = {p: _events(p) for p in "ab"}
    both = recall_at_budget(_run(), _events_fn(events))
    only_a = recall_at_budget(_run(), _events_fn(events), patients=["a"])
    assert (only_a["n_patients"] == 1).all()
    assert (only_a["n_events"] == len(events["a"])).all()
    assert (both["n_events"] == len(events["a"]) + len(events["b"])).all()


def test_an_onset_one_horizon_misses_is_not_scored():
    events = {p: _events(p) for p in "ab"}
    first = pd.Timestamp(events["a"]["start"].min())
    head = int((first - pd.Timestamp("2018-06-13")) / pd.Timedelta(minutes=15)) + 1
    full = recall_at_budget(_run(), _events_fn(events))
    cut = recall_at_budget(_run(drop_60_head=head), _events_fn(events))
    assert (cut["n_events"] == full["n_events"] - 2).all()
