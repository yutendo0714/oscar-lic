"""Unicode-aware OCR sequence metrics with explicit normalization profiles."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re
import unicodedata
from typing import Iterable, Sequence


SUPPORTED_PROFILES = {
    "raw_exact_v1",
    "unicode_strict_v1",
    "unicode_casefold_v1",
    "latin_alnum_ci_v1",
}


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _collapse_unicode_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_text(text: str, profile: str = "unicode_strict_v1") -> str:
    """Normalize OCR text under a named, versioned profile.

    The implementation mirrors ``evaluation/OCR_NORMALIZATION.md``. The
    ``digits_exact_v1`` dataset-field profile is intentionally excluded because
    it requires annotation metadata rather than a generic string transform.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text)!r}")
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported normalization profile: {profile}")

    text = _normalize_line_endings(text)
    if profile == "raw_exact_v1":
        return text

    text = unicodedata.normalize("NFKC", text)
    if profile == "unicode_casefold_v1":
        return _collapse_unicode_whitespace(text).strip().casefold()
    if profile == "unicode_strict_v1":
        return _collapse_unicode_whitespace(text).strip()

    # Legacy/compatibility profile. Keep Unicode letters and numbers, while
    # converting any whitespace run to a single separator.
    text = text.casefold()
    kept: list[str] = []
    pending_space = False
    for char in text:
        if char.isspace():
            pending_space = bool(kept)
            continue
        category = unicodedata.category(char)
        if category.startswith("L") or category.startswith("N"):
            if pending_space and kept and kept[-1] != " ":
                kept.append(" ")
            kept.append(char)
            pending_space = False
        # punctuation, marks and symbols are deliberately removed in this
        # compatibility profile. This is why it cannot be the sole main metric.
    return "".join(kept).strip()


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int
    hypothesis_length: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float | None:
        if self.reference_length == 0:
            return 0.0 if self.distance == 0 else None
        return self.distance / self.reference_length

    @property
    def normalized_edit_similarity(self) -> float:
        denom = max(self.reference_length, self.hypothesis_length, 1)
        return 1.0 - self.distance / denom


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Compute Levenshtein distance and one deterministic S/D/I decomposition.

    Tie-breaking prefers a diagonal operation, then deletion, then insertion.
    Aggregate distance is invariant; the detailed decomposition can differ from
    other valid alignments when multiple optimal paths exist.
    """
    n, m = len(reference), len(hypothesis)
    # Each cell stores (distance, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        current = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                current.append(previous[j - 1])
                continue
            diag = previous[j - 1]
            delete = previous[j]
            insert = current[j - 1]
            candidates = [
                (diag[0] + 1, diag[1] + 1, diag[2], diag[3], 0),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3], 1),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1, 2),
            ]
            best = min(candidates, key=lambda x: (x[0], x[4]))
            current.append(best[:4])
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    assert distance == substitutions + deletions + insertions
    return EditCounts(substitutions, deletions, insertions, n, m)


def character_counts(reference: str, hypothesis: str, profile: str) -> EditCounts:
    ref = normalize_text(reference, profile)
    hyp = normalize_text(hypothesis, profile)
    return edit_counts(list(ref), list(hyp))


def word_counts(reference: str, hypothesis: str, profile: str) -> EditCounts:
    ref = normalize_text(reference, profile).split()
    hyp = normalize_text(hypothesis, profile).split()
    return edit_counts(ref, hyp)


@dataclass
class CorpusMetrics:
    normalization_profile: str
    samples: int
    char_substitutions: int
    char_deletions: int
    char_insertions: int
    reference_characters: int
    word_substitutions: int
    word_deletions: int
    word_insertions: int
    reference_words: int
    exact_matches: int
    ned_sum: float

    @property
    def cer_micro(self) -> float | None:
        errors = self.char_substitutions + self.char_deletions + self.char_insertions
        if self.reference_characters == 0:
            return 0.0 if errors == 0 else None
        return errors / self.reference_characters

    @property
    def wer_micro(self) -> float | None:
        errors = self.word_substitutions + self.word_deletions + self.word_insertions
        if self.reference_words == 0:
            return 0.0 if errors == 0 else None
        return errors / self.reference_words

    @property
    def exact_rate(self) -> float | None:
        return self.exact_matches / self.samples if self.samples else None

    @property
    def ned_mean(self) -> float | None:
        return self.ned_sum / self.samples if self.samples else None

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            cer_micro=self.cer_micro,
            wer_micro=self.wer_micro,
            exact_rate=self.exact_rate,
            ned_mean=self.ned_mean,
        )
        return result


def evaluate_pairs(
    pairs: Iterable[tuple[str, str]], profile: str = "unicode_strict_v1"
) -> CorpusMetrics:
    totals = CorpusMetrics(
        normalization_profile=profile,
        samples=0,
        char_substitutions=0,
        char_deletions=0,
        char_insertions=0,
        reference_characters=0,
        word_substitutions=0,
        word_deletions=0,
        word_insertions=0,
        reference_words=0,
        exact_matches=0,
        ned_sum=0.0,
    )
    for reference, hypothesis in pairs:
        ref_n = normalize_text(reference, profile)
        hyp_n = normalize_text(hypothesis, profile)
        chars = edit_counts(list(ref_n), list(hyp_n))
        words = edit_counts(ref_n.split(), hyp_n.split())
        totals.samples += 1
        totals.char_substitutions += chars.substitutions
        totals.char_deletions += chars.deletions
        totals.char_insertions += chars.insertions
        totals.reference_characters += chars.reference_length
        totals.word_substitutions += words.substitutions
        totals.word_deletions += words.deletions
        totals.word_insertions += words.insertions
        totals.reference_words += words.reference_length
        totals.exact_matches += int(ref_n == hyp_n)
        totals.ned_sum += chars.normalized_edit_similarity
    return totals
