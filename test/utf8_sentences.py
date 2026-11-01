from random import Random, choice, randint, seed
import pytest
import stringzilla as sz
from stringzilla import Str
from test.sz_helpers import (
    SEED_VALUES,
    scale_iterations,
    assert_backends_agree,
    baseline_sentence_boundaries,
    malformed_utf8_corpus,
    representatives_by_class,
    run_across_backends,
    vector_width_bracketing_strings,
)
from test.utf8_helpers import (
    SEGMENTATION_PALETTE,
    adversarial_utf8_inputs,
    assert_segments_tile,
    byte_boundaries,
    class_adjacency_strings,
    corpus_of_byte_length,
    icu_segmenter,
    window_seam_lengths,
)
_byte_boundaries = byte_boundaries
def test_utf8_sentences_basic():
    result = [str(s) for s in sz.utf8_sentences("Hello. World!")]
    assert len(result) == 2
    assert "".join(result) == "Hello. World!"
    assert [str(s) for s in sz.utf8_sentences("")] == []
    assert [str(s) for s in sz.utf8_sentences("no terminator here")] == ["no terminator here"]
def test_utf8_sentences_skip_empty():
    result = [str(s) for s in sz.utf8_sentences("A. B. C.")]
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)
def test_utf8_sentences_unicode():
    result = [str(s) for s in sz.utf8_sentences("Привет. Мир.")]
    assert "".join(result) == "Привет. Мир."
    assert len(result) >= 2
    result = [str(s) for s in sz.utf8_sentences("你好。世界。")]
    assert "".join(result) == "你好。世界。"
def test_utf8_sentences_str_method():
    s = Str("Hello. World!")
    method_result = [str(x) for x in s.utf8_sentences(s)]
    module_result = [str(x) for x in sz.utf8_sentences(s)]
    assert method_result == module_result
    assert method_result == [str(x) for x in sz.utf8_sentences("Hello. World!")]
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_safety(seed_value: int):
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        assert_segments_tile(sz.utf8_sentences(raw), raw)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_seam(seed_value: int):
    rng = Random(seed_value)
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_sentences(raw), raw)
    body = corpus_of_byte_length(96, rng)
    for phase in range(64):
        raw = b"a" * phase + body
        assert_segments_tile(sz.utf8_sentences(raw), raw)
def test_utf8_sentence_boundary_official_conformance(sentence_break_cases):
    test_cases = sentence_break_cases
    failures = []
    for test_str, expected_byte_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(test_str))
        if sz_boundaries != expected_byte_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_byte_boundaries}, got {sz_boundaries}")
    assert not failures, "UAX-29 SentenceBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )
def test_utf8_sentence_class_adjacency(sentence_break_props):
    icu_sentences = icu_segmenter("sentence")
    representatives = representatives_by_class(sentence_break_props, count=1)
    cases = class_adjacency_strings(representatives, arity=2)
    failures = []
    for text in cases:
        if byte_boundaries(sz.utf8_sentences(text)) != byte_boundaries(icu_sentences(text)):
            failures.append(" ".join(f"{ord(character):04X}" for character in text))
    agreement = 1.0 - len(failures) / max(1, len(cases))
    assert agreement >= 0.80, "StringZilla vs ICU sentence class-adjacency agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_differential_icu(seed_value: int):
    seed(seed_value)
    def icu_boundaries():
        icu = pytest.importorskip("icu", reason="PyICU not installed")
        bi = icu.BreakIterator.createSentenceInstance(icu.Locale.getRoot())
        def segments(text):
            uni = icu.UnicodeString(text)
            bi.setText(uni)
            parts = []
            start = bi.first()
            end = bi.nextBoundary()
            while end != icu.BreakIterator.DONE:
                parts.append(str(uni[start:end]))
                start = end
                end = bi.nextBoundary()
            return parts
        return segments
    try:
        ref_segments = icu_boundaries()
    except Exception:
        pysbd = pytest.importorskip("pysbd", reason="neither PyICU nor pysbd installed")
        segmenter = pysbd.Segmenter(language="en", clean=False)
        ref_segments = lambda text: segmenter.segment(text)
    palette = list("abZ59 \t\n.,;:!?'\"()éαаカ中" "\u2028\u2029")
    iterations = scale_iterations(500)
    failures = []
    for _ in range(iterations):
        text = "".join(choice(palette) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(text))
        ref_boundaries = _byte_boundaries(ref_segments(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    ref={ref_boundaries}")
    agreement = 1.0 - len(failures) / iterations
    assert agreement >= 0.80, "StringZilla vs reference sentence agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_boundary_fuzz(seed_value: int, sentence_break_props):
    seed(seed_value)
    icu_sentences = None
    try:
        icu_sentences = icu_segmenter("sentence")
    except Exception:
        icu_sentences = None
    total = 0
    disagreements = []
    for _ in range(scale_iterations(200)):
        text = "".join(choice(SEGMENTATION_PALETTE) for _ in range(randint(1, 100)))
        try:
            expected = baseline_sentence_boundaries(text, sentence_break_props)
        except Exception:
            continue
        total += 1
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(text))
        if sz_boundaries == expected:
            continue
        if icu_sentences is not None and sz_boundaries == _byte_boundaries(icu_sentences(text)):
            continue
        disagreements.append((" ".join(f"{ord(c):04X}" for c in text), expected, sz_boundaries))
    agreement = 1.0 - len(disagreements) / max(total, 1)
    assert agreement >= 0.98, "Sentence boundary agreement {:.2%} too low (seed {}):\n{}".format(
        agreement, seed_value, "\n".join(f"  {c}\n    baseline={e}\n    got={g}" for c, e, g in disagreements[:10])
    )
def test_utf8_sentences_prose():
    from test.utf8_helpers import (
        PROSE_HOTEL_REVIEW,
        PROSE_CONCERT_POST,
        PROSE_NEWS_LEDE,
        PROSE_MICRO_HARDBREAKS,
        assert_segments_tile,
        icu_segmenter,
    )
    cases = [
        (PROSE_HOTEL_REVIEW, 5),
        (PROSE_CONCERT_POST, 4),
        (PROSE_NEWS_LEDE, 6),
        (PROSE_MICRO_HARDBREAKS, 3),
    ]
    for text, expected in cases:
        segments = list(sz.utf8_sentences(text))
        assert_segments_tile(segments, text)
        assert len(segments) == expected
    oracle = icu_segmenter("sentence")
    for text, _ in cases:
        assert len(list(sz.utf8_sentences(text))) == len(oracle(text))
SENTENCE_REALISTIC_TEXTS = [
    "Hello. World!",
    "no terminator here",
    "Привет. Мир.",
    "你好。世界。",
    "Dr. Smith went home. He slept.",
]
@pytest.mark.parametrize(
    "text", SENTENCE_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_sentences_backend_differential(text):
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_sentences(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")
