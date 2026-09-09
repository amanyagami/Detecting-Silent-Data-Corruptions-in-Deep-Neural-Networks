"""Unit tests for the tau-score building blocks in DRDNA/b10c32/config.py.
Pure logic, synthetic inputs -- no model, dataset, or GPU needed."""
from DRDNA.b10c32.config import (
    abnormality_score2,
    abnormality_score3,
    abnormility_score1,
    listtohistogram,
    normalize_histogram,
    reset_histogram,
    update_histogram,
)


def test_listtohistogram_bins_are_contiguous_and_cover_the_data():
    data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    hist = listtohistogram(data, bins=10)
    assert len(hist) == 10
    edges = sorted(hist.keys())
    assert edges[0][0] == min(data)
    assert edges[-1][1] == max(data)
    for (lo1, hi1), (lo2, _hi2) in zip(edges, edges[1:]):
        assert hi1 == lo2


def test_update_histogram_increments_the_right_bin():
    hist = {(0.0, 5.0): 0, (5.0, 10.0): 0}
    updated = update_histogram(3.0, dict(hist))
    assert updated[(0.0, 5.0)] == 1
    assert updated[(5.0, 10.0)] == 0


def test_reset_histogram_zeroes_counts_but_keeps_bins():
    hist = {(0.0, 5.0): 7, (5.0, 10.0): 3}
    reset = reset_histogram(hist)
    assert set(reset.keys()) == set(hist.keys())
    assert all(v == 0 for v in reset.values())


def test_normalize_histogram_sums_to_one():
    hist = {(0.0, 5.0): 2, (5.0, 10.0): 6}
    normalized = normalize_histogram(hist)
    assert abs(sum(normalized.values()) - 1.0) < 1e-9


def test_normalize_histogram_handles_all_zero_without_dividing_by_zero():
    hist = {(0.0, 5.0): 0, (5.0, 10.0): 0}
    assert normalize_histogram(hist) == hist


def test_abnormility_score1_is_lower_for_a_common_value():
    hist = {(0.0, 5.0): 90, (5.0, 10.0): 10}
    common = abnormility_score1(1.0, hist)  # falls in the 90-count bin
    rare = abnormility_score1(6.0, hist)  # falls in the 10-count bin
    assert common < rare


def test_abnormity_score1_defaults_to_1_outside_all_bins():
    hist = {(0.0, 5.0): 10}
    assert abnormility_score1(99.0, hist) == 1


def test_abnormality_score2_is_zero_for_identical_histograms():
    hist = {(0.0, 5.0): 4, (5.0, 10.0): 6}
    assert abnormality_score2(hist, hist) == 0


def test_abnormality_score2_is_positive_for_a_shifted_histogram():
    baseline = {(0.0, 5.0): 8, (5.0, 10.0): 2}
    shifted = {(0.0, 5.0): 2, (5.0, 10.0): 8}
    assert abnormality_score2(shifted, baseline) > 0


def test_abnormality_score3_matches_the_implemented_scoring_rule():
    # README.md describes this as "0 if neurons are same, 0.5 if one neuron
    # is same, 1 if both neurons are different" -- but the actual
    # implementation compares a['min_location'] against b['max_location']
    # (not b['min_location']) in both branches. That's almost certainly a
    # typo in the original script, but it's the formula that produced this
    # repo's validated results, so this test documents the code as written
    # rather than the description, matching this repo's general policy of
    # not silently "fixing" scoring behavior it hasn't separately re-validated.
    a = {"max_location": (0, 1), "min_location": (0, 2)}
    b_neither = {"max_location": (0, 9), "min_location": (0, 8)}
    b_max_matches_only = {"max_location": (0, 1), "min_location": (0, 8)}
    assert abnormality_score3(a, b_neither) == 0
    assert abnormality_score3(a, b_max_matches_only) == 0.5

    # Both conditions fire only when a's max and min positions coincide
    # (e.g. a constant activation) and that shared position equals b's max.
    degenerate_a = {"max_location": (0, 1), "min_location": (0, 1)}
    b_full_match = {"max_location": (0, 1), "min_location": (0, 9)}
    assert abnormality_score3(degenerate_a, b_full_match) == 1
