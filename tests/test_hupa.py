"""Tests for the HUPA-UCM reader: raw Libre formats, unit repair, events, windows."""

import numpy as np
import pandas as pd
import pytest

from tfm.data import hupa
from tfm.data import features as f
from tfm.data import quality as q
from tfm.paths import HUPA_RAW_DIR

LIBRE_HEADER_ES = (
    "ID;Hora;Tipo de registro;Histórico glucosa (mg/dL);Glucosa leída (mg/dL);"
    "Insulina de acción rápida (unidades)"
)


def _write(path, text, newline="\n", bom=False):
    path.write_bytes(((("﻿" if bom else "") + text.replace("\n", newline)).encode("utf-8")))
    return path


# --- raw Libre exports -------------------------------------------------------

def test_read_libre_keeps_only_historic_readings(tmp_path):
    text = "\n".join([
        "HUPA0001P;;;;;",
        LIBRE_HEADER_ES,
        "1;2018/06/13 17:15;0;208;;",
        "2;2018/06/13 17:22;1;;215;",     # scan: dropped
        "3;2018/06/13 17:30;0;209;;",
        "4;2018/06/13 17:40;5;;;2",       # insulin record: dropped
        "5;2018/06/13 17:45;0;216;;",
    ])
    df = hupa.read_libre(_write(tmp_path / "a.csv", text))
    assert df["value"].tolist() == [208, 209, 216]
    assert df["ts"].iloc[0] == pd.Timestamp("2018-06-13 17:15")


def test_read_libre_handles_tabs_short_dates_and_old_mac_line_endings(tmp_path):
    text = "\n".join([
        LIBRE_HEADER_ES.replace(";", "\t"),
        "1\t27/3/19 16:22\t0\t60\t\t",
        "2\t27/3/19 16:37\t0\t56\t\t",
        "3\t27/3/19 16:52\t0\t50\t\t",
    ])
    df = hupa.read_libre(_write(tmp_path / "b.csv", text, newline="\r", bom=True))
    assert df["ts"].tolist() == list(pd.date_range("2019-03-27 16:22", periods=3, freq="15min"))


def test_read_libre_handles_the_librelink_report_layout(tmp_path):
    text = "\n".join([
        "Informe del paciente HUPA0027P;Generado el;05-11-2020 00:45 UTC",
        "HUPA0027P,",
        "Dispositivo,Número de serial,Sello de tiempo del dispositivo,Tipo de registro,"
        "Historial de glucosa mg/dL,Escaneo de glucosa mg/dL",
        "FreeStyle LibreLink,,17-11-2021 11:52,0,124,",
        "FreeStyle LibreLink,,17-11-2021 12:07,0,125,",
        "FreeStyle LibreLink,,17-11-2021 12:22,0,111,",
    ])
    df = hupa.read_libre(_write(tmp_path / "c.csv", text, newline="\r\n"))
    assert df["value"].tolist() == [124, 125, 111]
    assert df["ts"].iloc[0] == pd.Timestamp("2021-11-17 11:52")


def test_ambiguous_day_month_order_is_resolved_by_the_libre_cadence():
    raw = pd.Series(["01-02-2021 23:30", "01-02-2021 23:45", "02-02-2021 00:00",
                     "02-02-2021 00:15"])
    ts = hupa._parse_timestamps(raw)
    assert ts.iloc[2] == pd.Timestamp("2021-02-02 00:00")
    assert ts.diff().dropna().eq(pd.Timedelta(minutes=15)).all()


def test_an_export_without_historic_rows_is_empty(tmp_path):
    text = "\n".join([LIBRE_HEADER_ES, "1;2018/09/19 15:10;4;;;0"])
    assert hupa.read_libre(_write(tmp_path / "d.csv", text)).empty


# --- treatment channels ------------------------------------------------------

def _pre(n=60, basal=0.075, zero=(), bolus=None, carbs=None):
    t = pd.date_range("2018-06-13 00:00", periods=n, freq="5min")
    rate = np.full(n, basal)
    rate[list(zero)] = 0.0
    return pd.DataFrame({
        "time": t,
        "basal_rate": rate,
        "bolus_volume_delivered": np.zeros(n) if bolus is None else bolus,
        "carb_input": np.zeros(n) if carbs is None else carbs,
    })


def test_basal_is_converted_from_units_per_cell_to_units_per_hour():
    basal, _ = hupa.basal_events("HUPA0001P", _pre(basal=0.075))
    assert basal["value"].iloc[0] == pytest.approx(0.9)


def test_a_short_zero_basal_on_a_pump_is_a_suspension():
    pre = _pre(zero=range(10, 16))  # 30 min
    _, temp = hupa.basal_events("HUPA0001P", pre)
    assert len(temp) == 1
    assert temp["ts_begin"].iloc[0] == pre["time"].iloc[10]
    assert temp["ts_end"].iloc[0] == pre["time"].iloc[15] + pd.Timedelta(minutes=5)
    assert temp["value"].iloc[0] == 0.0


def test_a_long_zero_basal_is_missing_data_not_a_suspension():
    pre = _pre(n=80, zero=range(10, 10 + hupa.MAX_SUSPEND_MIN // 5 + 1))
    _, temp = hupa.basal_events("HUPA0001P", pre)
    assert temp.empty


def test_pens_never_suspend():
    _, temp = hupa.basal_events("HUPA0017P", _pre(zero=range(10, 16)))
    assert temp.empty


def test_carbohydrates_are_converted_to_grams_per_patient():
    carbs = np.zeros(60)
    carbs[5] = 4.0
    servings = hupa.treatment_channels("HUPA0001P", _pre(carbs=carbs))
    grams = hupa.treatment_channels("HUPA0002P", _pre(carbs=carbs))
    assert servings["meal"]["carbs"].tolist() == [40.0]
    assert grams["meal"]["carbs"].tolist() == [4.0]


def test_unquantified_doses_are_dropped_not_fed_as_one_unit():
    ones = np.zeros(60)
    ones[[5, 20]] = 1.0
    ch = hupa.treatment_channels("HUPA0019P", _pre(bolus=ones, carbs=ones))
    assert "bolus" not in ch and "meal" not in ch
    assert "basal" in ch


def test_missing_channels_reach_the_features_as_zeros():
    grid = pd.date_range("2018-06-13", periods=24, freq="5min")
    cgm = pd.DataFrame({"ts": grid, "value": np.linspace(120, 140, 24)})
    frame = f.build_feature_frame("x", "all", q.add_segments(cgm), wearable=False,
                                  channels={"glucose_level": cgm})
    assert frame[["iob", "cob", "basal_rate", "pump_suspended"]].abs().sum().sum() == 0.0


# --- synthetic Libre patient -------------------------------------------------

def _libre_patient(values, start="2018-06-13 08:02", gap_after=None, gap_min=60):
    """Readings every 15 minutes, optionally with a dropout after an index."""
    ts = list(pd.date_range(start, periods=len(values), freq="15min"))
    if gap_after is not None:
        ts = ts[: gap_after + 1] + [t + pd.Timedelta(minutes=gap_min) for t in ts[gap_after + 1:]]
    cgm = pd.DataFrame({"ts": ts, "value": np.asarray(values, dtype=float)})
    return {"glucose_level": cgm}


def test_feature_frame_marks_readings_and_cadence_cells():
    frame = hupa.feature_frame("HUPA0001P", _libre_patient([100, 103, 106, 109, 112]))
    assert frame["is_reading"].sum() == 5
    # 4 intervals x 2 cells between readings, all due to the cadence.
    assert frame["libre_cadence"].sum() == 8
    assert frame["is_imputed"].sum() == 8


def test_a_dropout_beyond_the_cadence_keeps_its_imputed_meaning():
    # 15 + 15 = 30 min between two readings: 5 empty cells, filled but a dropout.
    frame = hupa.feature_frame("HUPA0001P", _libre_patient([100] * 8, gap_after=3, gap_min=15))
    dropout = frame["is_imputed"].astype(bool) & ~frame["libre_cadence"]
    assert dropout.sum() == 5


def test_one_low_reading_is_not_an_event_but_two_are():
    one = hupa.feature_frame("HUPA0001P", _libre_patient([90, 80, 65, 80, 90]))
    two = hupa.feature_frame("HUPA0001P", _libre_patient([90, 80, 65, 62, 80, 90]))
    assert hupa.libre_events(one).empty
    ev = hupa.libre_events(two)
    assert len(ev) == 1 and ev["nadir"].iloc[0] == 62 and ev["minutes"].iloc[0] == 15


def test_an_event_does_not_run_across_a_segment_break():
    frame = hupa.feature_frame(
        "HUPA0001P", _libre_patient([90, 65, 60, 60, 65, 90], gap_after=2, gap_min=60)
    )
    assert frame["segment"].nunique() == 2
    ev = hupa.libre_events(frame)
    assert ev["n_readings"].tolist() == [2, 2]


def test_windows_are_anchored_on_readings_with_reading_targets():
    values = 120 + 30 * np.sin(np.arange(40) / 4)
    frame = hupa.feature_frame("HUPA0001P", _libre_patient(values))
    ws = hupa.patient_windows(frame)
    readings = set(frame.loc[frame["is_reading"], "ts"])
    assert len(ws) > 0
    for t in pd.to_datetime(ws.ts):
        assert t in readings
        for h in ws.horizons:
            assert t + pd.Timedelta(minutes=5 * h) in readings
    # Every third cell is a reading, so about a third of the windows survive.
    assert len(ws) == pytest.approx(ws.meta["n_all_windows"] / 3, abs=1)


def test_targets_match_the_readings():
    values = np.arange(100, 140, dtype=float)
    frame = hupa.feature_frame("HUPA0001P", _libre_patient(values))
    ws = hupa.patient_windows(frame)
    by_ts = frame.set_index("ts")["value"]
    t0 = pd.Timestamp(ws.ts[0])
    assert ws.y[0, 0] == by_ts[t0 + pd.Timedelta(minutes=30)]


def test_imputed_flag_modes():
    frame = hupa.feature_frame("HUPA0001P", _libre_patient(np.full(40, 120.0)))
    col = hupa.FEATURE_COLUMNS.index("is_imputed")
    dropouts = hupa.patient_windows(frame, imputed_flag="dropouts")
    everything = hupa.patient_windows(frame, imputed_flag="all")
    assert dropouts.X[:, :, col].sum() == 0
    # Two of every three cells in the history are interpolated.
    assert everything.X[:, :, col].mean() == pytest.approx(2 / 3, abs=0.1)
    with pytest.raises(ValueError):
        hupa.patient_windows(frame, imputed_flag="none")


# --- the per-patient lists, checked against the real files -------------------

requires_hupa = pytest.mark.skipif(
    not (HUPA_RAW_DIR / "Preprocessed").exists(), reason="HUPA-UCM not on this machine"
)


@requires_hupa
def test_excluded_patients_have_no_libre_history():
    for patient in hupa.EXCLUDED:
        pre = hupa.read_preprocessed(patient)
        window = (pre["time"].min(), pre["time"].max())
        assert hupa.libre_glucose(patient, window).empty


@requires_hupa
@pytest.mark.parametrize("patient", [p for p in hupa.PATIENTS if p not in hupa.EXCLUDED])
def test_unit_and_tier_lists_match_the_data(patient):
    pre = hupa.read_preprocessed(patient)
    carbs = pre["carb_input"][pre["carb_input"] > 0]
    bolus = pre["bolus_volume_delivered"][pre["bolus_volume_delivered"] > 0]
    if patient in hupa.CARBS_IN_GRAMS:
        assert carbs.median() >= 15
    elif len(carbs):
        assert carbs.median() <= 10
    if patient in hupa.UNQUANTIFIED_DOSES:
        assert set(carbs) <= {1.0} and set(bolus) <= {1.0}
    if hupa.RECORD_TIER[patient] == "sin_registro":
        assert carbs.empty and bolus.empty and (pre["basal_rate"] == 0).all()
    if hupa.RECORD_TIER[patient] == "completo":
        assert len(carbs) and len(bolus) and (pre["basal_rate"] > 0).any()


# --- split and event budget --------------------------------------------------

def test_split_is_temporal_and_holds_out_the_last_readings():
    frame = hupa.feature_frame("HUPA0001P", _libre_patient(np.full(100, 120.0)))
    parts = hupa.split_frame(frame, test_fraction=0.3)
    assert parts["adapt"]["ts"].max() < parts["test"]["ts"].min()
    assert len(parts["adapt"]) + len(parts["test"]) == len(frame)
    assert parts["test"]["is_reading"].sum() == 30
    # The test part starts on a reading, so its first window can anchor there.
    assert parts["test"]["is_reading"].iloc[0]


def test_no_window_straddles_the_split():
    values = 120 + 30 * np.sin(np.arange(100) / 4)
    frame = hupa.feature_frame("HUPA0001P", _libre_patient(values))
    parts = hupa.split_frame(frame)
    cut = parts["test"]["ts"].min()
    adapt = hupa.patient_windows(parts["adapt"])
    last_target = pd.to_datetime(adapt.ts) + pd.Timedelta(minutes=5 * max(adapt.horizons))
    assert (last_target < cut).all()


def test_event_budget_assigns_events_by_onset():
    values = [100] * 10 + [60, 60] + [100] * 70 + [60, 60, 60] + [100] * 17
    frame = hupa.feature_frame("HUPA0001P", _libre_patient(values))
    budget = hupa.event_budget({"HUPA0001P": frame})
    assert budget[["events_all", "events_test"]].iloc[0].tolist() == [2, 1]
