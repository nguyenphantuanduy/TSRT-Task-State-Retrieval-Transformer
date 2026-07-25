"""
Smoke test for the new hybrid sentence_splitter.

Verifies:
    1. spaCy path returns sane boundaries + offsets on a tricky text
       (abbreviations, quoted speech, lowercase after period).
    2. ``split_sentences_with_offsets`` (backwards-compatible entry
       point used by collator.py) agrees with ``split_with_metadata``.
    3. Empty / whitespace input returns [] without errors.

Usage (from ``backend/HotpotQA_Distractor/``)::

    python -m test.test_sentence_splitter
"""

from __future__ import annotations

from sentence_splitter import (
    get_splitter,
    split_sentences_with_offsets,
    split_with_metadata,
)


def main() -> None:

    # 1. Lazy-load -- downloads en_core_web_sm on first run.
    splitter = get_splitter()
    print("splitter loaded:", type(splitter).__name__)
    print()

    text = (
        'Dr. Smith met Mr. Brown in Washington, D.C. '
        'They discussed U.S. policy. '
        'The meeting lasted two hours. '
        'Afterwards, they went to the park. '
        '"It was great," he said.'
    )

    # 2. Primary path with metadata.
    boundaries, source, confidence = split_with_metadata(text)

    print(f"source     : {source}")
    print(f"confidence : {confidence:.4f}")
    print(f"segments   : {len(boundaries)}")
    print("-" * 70)
    for sent, start, end in boundaries:
        print(f"  [{start:>3}..{end:<3}]  {sent!r}")
    print("-" * 70)

    # 3. Offsets must reconstruct the original text exactly.
    reconstructed = "".join(s for s, _, _ in boundaries)
    assert reconstructed == text, (
        f"offset reconstruction mismatch:\n"
        f"  got: {reconstructed!r}\n"
        f"  exp: {text!r}"
    )
    print("offset reconstruction: OK")

    # 4. Backwards-compatible entry point must agree on (text, offsets).
    compat = split_sentences_with_offsets(text)
    assert [t[:2] for t in compat] == [t[:2] for t in boundaries], (
        "split_sentences_with_offsets diverged from split_with_metadata"
    )
    print("backwards-compatible API: OK")

    # 5. Empty / whitespace inputs must not blow up.
    for empty in ("", "   ", "\n\t"):
        out = split_sentences_with_offsets(empty)
        assert out == [], f"expected [] for {empty!r}, got {out!r}"
    print("empty / whitespace inputs: OK")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
