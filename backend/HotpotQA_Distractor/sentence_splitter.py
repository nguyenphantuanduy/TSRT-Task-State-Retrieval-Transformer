"""
Hybrid sentence splitter with confidence-based fallback.

Primary:
    spaCy ``en_core_web_sm`` — fast, rule-based, no native confidence.

Fallback (lazy-loaded):
    ``wtpsplit.SaT`` neural model — slower, but exposes sentence-boundary
    probabilities.

A heuristic structural score is computed over the spaCy output.
When that score is below ``CONFIDENCE_THRESHOLD``, the splitter
re-segments the text with ``wtpsplit`` and reports the model's
average segment probability instead.

The public surface is intentionally narrow: callers (e.g. the
collator) only see :func:`split_sentences_with_offsets`, which
returns the same ``(sentence, start_char, end_char)`` shape as
the previous in-collator implementation.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import spacy
from spacy.cli import download

SentenceBoundary = Tuple[str, int, int]


CONFIDENCE_THRESHOLD = 0.85

SPACY_MODEL_NAME = "en_core_web_sm"

WTP_MODEL_NAME = "sat-3l-sm"


# Common English abbreviations that frequently fool rule-based
# sentence splitters (e.g. "Dr. Smith met Mr. Brown.").
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "st", "vs", "etc", "e.g", "i.e", "u.s", "u.k",
    "inc", "ltd", "co", "no", "fig",
}


def _load_spacy():
    try:
        return spacy.load(SPACY_MODEL_NAME)
    except OSError:
        print(f"{SPACY_MODEL_NAME} not found. Downloading...")
        download(SPACY_MODEL_NAME)
        return spacy.load(SPACY_MODEL_NAME)


class HybridSentenceSplitter:
    """
    Sentence splitter with a confidence threshold that falls back
    to a slower, neural segmenter when the primary output looks
    unreliable.

    Args:
        confidence_threshold:
            Heuristic score in ``[0, 1]`` below which the fallback
            is triggered.

        wtp_model_name:
            Any wtpsplit-compatible Hugging Face checkpoint.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        wtp_model_name: str = WTP_MODEL_NAME,
    ) -> None:
        self._spacy_nlp = _load_spacy()

        self._wtp = None
        self._wtp_model_name = wtp_model_name

        self.confidence_threshold = confidence_threshold

    # ==========================================================
    # Public API
    # ==========================================================

    def split(self, text: str) -> Tuple[List[SentenceBoundary], str, float]:
        """
        Split ``text`` into sentences with character offsets.

        Returns:
            boundaries:
                List of ``(sentence, start_char, end_char)`` tuples.

            source:
                ``"spacy"`` if the primary path was used,
                ``"wtpsplit"`` if the fallback was used.

            confidence:
                The score that drove the routing decision. For the
                spaCy path, this is the heuristic score. For
                wtpsplit, this is the model's average per-segment
                probability.
        """

        if not text or not text.strip():
            return [], "spacy", 1.0

        boundaries = self._spacy_split(text)
        confidence = self._score_boundaries(text, boundaries)

        if confidence >= self.confidence_threshold:
            return boundaries, "spacy", confidence

        wtp_boundaries, wtp_confidence = self._wtp_split(text)
        return wtp_boundaries, "wtpsplit", wtp_confidence

    # ==========================================================
    # Internals
    # ==========================================================

    def _spacy_split(self, text: str) -> List[SentenceBoundary]:
        doc = self._spacy_nlp(text)
        return [
            (sent.text, sent.start_char, sent.end_char)
            for sent in doc.sents
        ]

    def _wtp_split(self, text: str) -> Tuple[List[SentenceBoundary], float]:
        """Run SaT and return ``(boundaries, avg_boundary_probability)``."""

        sat = self._lazy_wtp()
        segments = sat.split(text)
        probabilities = sat.predict_proba(text)

        boundaries: List[SentenceBoundary] = []
        cursor = 0

        for segment in segments:
            start = cursor
            end = start + len(segment)
            boundaries.append((segment, start, end))
            cursor = end

        # SaT may split input newlines without retaining them in individual
        # segments. Rebuild offsets by locating each segment sequentially when
        # direct concatenation does not cover the complete input.
        if cursor != len(text):
            boundaries = []
            cursor = 0
            for segment in segments:
                start = text.find(segment, cursor)
                if start < 0:
                    raise ValueError("SaT returned a segment not found in the input text")
                end = start + len(segment)
                boundaries.append((segment, start, end))
                cursor = end

        boundary_probs = []
        for _, _, end in boundaries[:-1]:
            probability_index = end - 1
            if 0 <= probability_index < len(probabilities):
                boundary_probs.append(float(probabilities[probability_index]))

        avg_prob = (
            sum(boundary_probs) / len(boundary_probs)
            if boundary_probs
            else 1.0
        )

        return boundaries, avg_prob

    def _lazy_wtp(self):
        if self._wtp is None:
            from wtpsplit import SaT
            self._wtp = SaT(self._wtp_model_name)
        return self._wtp

    @staticmethod
    def _score_boundaries(
        text: str,
        boundaries: List[SentenceBoundary],
    ) -> float:
        """
        Heuristic structural score in ``[0, 1]`` for the spaCy
        boundaries.

        Heuristics:
            + bonus when the boundary is followed by an uppercase
              letter (likely a real sentence start).
            - penalty when the pre-boundary token is a known
              abbreviation (likely a false sentence break).
            - penalty when the boundary is inside a quoted string.
            - penalty when the next character is non-whitespace
              (suggests the boundary was spurious).
        """

        if not boundaries:
            return 0.0

        score_sum = 0.0

        for sent_text, start, end in boundaries:
            s = 1.0  # baseline

            # ----- pre-boundary token -----
            if start > 0:
                pre = text[:start].rstrip().lower().split(" ")[-1]
                pre_clean = pre.strip(".,;:!?\"").lower()

                if pre_clean in _ABBREVIATIONS:
                    s -= 0.6

                if text[start - 1] == '"':
                    s -= 0.2

            # ----- post-boundary token -----
            if end < len(text):
                post = text[end:end + 1]

                if post and post.isupper():
                    s += 0.1
                elif post and not post.isspace():
                    s -= 0.2

            score_sum += max(0.0, min(1.0, s))

        return score_sum / len(boundaries)


# ==========================================================
# Module-level singleton
# ==========================================================

_splitter: Optional[HybridSentenceSplitter] = None


def get_splitter() -> HybridSentenceSplitter:
    """
    Lazily instantiate the module-level splitter.

    A singleton is used so the spaCy pipeline (and any future
    wtpsplit model) is loaded exactly once per process.
    """

    global _splitter
    if _splitter is None:
        _splitter = HybridSentenceSplitter()
    return _splitter


def split_sentences_with_offsets(
    text: str,
    splitter: Optional[HybridSentenceSplitter] = None,
):
    """
    Backwards-compatible entry point.

    Returns:
        List of ``(sentence, start_char, end_char)`` tuples.

    The selected source and confidence are exposed via
    :func:`split_with_metadata` for callers that want to log the
    fallback rate.
    """

    splitter = splitter or get_splitter()
    boundaries, _, _ = splitter.split(text)
    return boundaries


def split_with_metadata(
    text: str,
    splitter: Optional[HybridSentenceSplitter] = None,
):
    """
    Same as :func:`split_sentences_with_offsets` but also returns
    the segmenter source and confidence score.
    """

    splitter = splitter or get_splitter()
    return splitter.split(text)