import pytest

from oscarlic.text_metrics import (
    character_counts,
    edit_counts,
    evaluate_pairs,
    normalize_text,
)


def test_unicode_nfkc_and_whitespace():
    assert normalize_text("ＡＢＣ  12", "unicode_strict_v1") == "ABC 12"


def test_casefold_sharp_s():
    assert normalize_text("Straße", "unicode_casefold_v1") == "strasse"
    assert normalize_text("STRASSE", "unicode_casefold_v1") == "strasse"


def test_legacy_alnum_profile_hides_punctuation_by_design():
    assert normalize_text("Price: $8", "latin_alnum_ci_v1") == "price 8"
    assert normalize_text("price 8", "latin_alnum_ci_v1") == "price 8"
    assert normalize_text("Price: $8", "unicode_strict_v1") != normalize_text("price 8", "unicode_strict_v1")


def test_substitution_counts():
    counts = character_counts("8", "3", "unicode_strict_v1")
    assert counts.substitutions == 1
    assert counts.deletions == 0
    assert counts.insertions == 0
    assert counts.distance == 1
    assert counts.error_rate == 1.0


def test_known_levenshtein_distance():
    counts = edit_counts(list("kitten"), list("sitting"))
    assert counts.distance == 3
    assert counts.reference_length == 6


def test_corpus_micro_metrics():
    metrics = evaluate_pairs([("abc", "adc"), ("hello world", "hello world")])
    assert metrics.samples == 2
    assert metrics.char_substitutions == 1
    assert metrics.exact_matches == 1
    assert metrics.cer_micro == pytest.approx(1 / (3 + 11))
    assert metrics.exact_rate == pytest.approx(0.5)


def test_empty_reference_policy():
    assert character_counts("", "", "unicode_strict_v1").error_rate == 0.0
    assert character_counts("", "x", "unicode_strict_v1").error_rate is None


def test_invalid_profile():
    with pytest.raises(ValueError):
        normalize_text("abc", "invented")
