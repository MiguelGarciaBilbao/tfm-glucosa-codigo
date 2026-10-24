"""Tests for the raw XML parser.
"""

import pandas as pd
import pytest

from tfm.data import parser
from tfm.paths import RAW_DIR

requires_raw = pytest.mark.skipif(
    not RAW_DIR.exists(), reason="OhioT1DM raw XML not available (DUA)"
)


@pytest.fixture(scope="module")
def train_559():
    return parser.load_patient_channels("559", "train")


def test_patient_ids_are_twelve_and_disjoint():
    assert len(parser.PATIENTS) == 12
    assert not set(parser.PATIENTS_2018) & set(parser.PATIENTS_2020)


def test_cohort_lookup():
    assert parser.cohort_of("559") == "2018"
    assert parser.cohort_of("540") == "2020"
    with pytest.raises(ValueError):
        parser.cohort_of("000")


@requires_raw
def test_glucose_is_parsed_with_numeric_values(train_559):
    _, channels = train_559
    cgm = channels["glucose_level"]
    assert len(cgm) == 10796
    assert pd.api.types.is_datetime64_any_dtype(cgm["ts"])
    assert pd.api.types.is_numeric_dtype(cgm["value"])
    assert cgm["ts"].is_monotonic_increasing
    # Physiological range of the Medtronic Enlite sensor used in the study.
    assert cgm["value"].between(38, 402).all()


@requires_raw
def test_cgm_median_gap_is_five_minutes(train_559):
    _, channels = train_559
    gaps = channels["glucose_level"]["ts"].diff().dropna()
    assert gaps.median() == pd.Timedelta(minutes=parser.SAMPLING_MINUTES)


@requires_raw
def test_interval_channels_have_begin_and_end(train_559):
    _, channels = train_559
    bolus = channels["bolus"]
    assert {"ts_begin", "ts_end", "dose", "type"} <= set(bolus.columns)
    assert (bolus["ts_end"] >= bolus["ts_begin"]).all()


@requires_raw
def test_weight_attribute_is_filler_and_must_not_be_a_feature():
    weights = {parser.load_patient_channels(p, "train")[0]["weight"] for p in parser.PATIENTS}
    assert weights == {"99"}


@requires_raw
@pytest.mark.parametrize("patient", parser.PATIENTS_2020)
def test_2020_cohort_has_no_heart_rate(patient):
    _, channels = parser.load_patient_channels(patient, "train")
    assert channels["basis_heart_rate"].empty
    assert not channels["acceleration"].empty


@requires_raw
@pytest.mark.parametrize("patient", parser.PATIENTS_2018)
def test_2018_cohort_has_heart_rate_and_no_acceleration(patient):
    _, channels = parser.load_patient_channels(patient, "train")
    assert not channels["basis_heart_rate"].empty
    assert "acceleration" not in channels


@requires_raw
def test_all_files_parse_and_every_tag_is_known():
    for patient in parser.PATIENTS:
        for split in ("train", "test"):
            attrs, channels = parser.load_patient_channels(patient, split)
            assert attrs["id"] == patient
            assert not channels["glucose_level"].empty


# Expected scored test points, Table 2 of Marling & Bunescu (2020).
PAPER_TEST_EXAMPLES = {
    "540": 2884, "544": 2704, "552": 2352, "567": 2377, "584": 2653, "596": 2731,
    "559": 2514, "563": 2570, "570": 2745, "575": 2590, "588": 2791, "591": 2760,
}


@requires_raw
@pytest.mark.parametrize("patient,expected", sorted(PAPER_TEST_EXAMPLES.items()))
def test_official_test_size_matches_the_dataset_paper(patient, expected):
    """The 2020 cohort drops its first hour; the 2018 cohort scores every point."""
    assert len(parser.official_test_frame(patient)) == expected
