from random import Random, choice, randint, seed
import pytest
import stringzilla as sz
from stringzilla import Str
from test.sz_helpers import (
    SEED_VALUES,
    scale_iterations,
    assert_backends_agree,
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
    window_seam_lengths,
)
_byte_boundaries = byte_boundaries
def test_utf8_linebreaks_basic():
    result = [str(seg) for seg in sz.utf8_linebreaks("first\nsecond")]
    assert "".join(result) == "first\nsecond"
    assert [str(seg) for seg in sz.utf8_linebreaks("")] == []
    result = [str(seg) for seg in sz.utf8_linebreaks("just one line")]
    assert "".join(result) == "just one line"
def test_utf8_linebreaks_tiling():
    text = "alpha\nbeta\ngamma"
    forward = [str(seg) for seg in sz.utf8_linebreaks(text)]
    assert "".join(forward) == text
def test_utf8_linebreaks_skip_empty():
    result = [str(seg) for seg in sz.utf8_linebreaks("a\n\nb")]
    assert len(result) > 0
    assert all(len(seg) > 0 for seg in result)
def test_utf8_linebreaks_unicode():
    result = [str(seg) for seg in sz.utf8_linebreaks("Größe\u2028привет")]
    assert "".join(result) == "Größe\u2028привет"
    result = [str(seg) for seg in sz.utf8_linebreaks("a\r\nb")]
    assert "".join(result) == "a\r\nb"
def test_utf8_linebreaks_str_method():
    s = Str("first\nsecond")
    method_result = [str(seg) for seg in s.utf8_linebreaks(s)]
    module_result = [str(seg) for seg in sz.utf8_linebreaks(s)]
    assert method_result == module_result
    assert method_result == [str(seg) for seg in sz.utf8_linebreaks("first\nsecond")]
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_linewrap_safety(seed_value: int):
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        assert_segments_tile(sz.utf8_linebreaks(raw), raw)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_linewrap_seam(seed_value: int):
    rng = Random(seed_value)
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_linebreaks(raw), raw)
    body = corpus_of_byte_length(96, rng)
    for phase in range(64):
        raw = b"a" * phase + body
        assert_segments_tile(sz.utf8_linebreaks(raw), raw)
def test_utf8_linewrap_boundary_official_conformance(line_break_cases):
    test_cases = line_break_cases
    failures = []
    for test_str, expected_byte_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_linebreaks(test_str))[1:]
        if sz_boundaries != expected_byte_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_byte_boundaries}, got {sz_boundaries}")
    assert not failures, "UAX-14 LineBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )
def test_utf8_linewrap_class_adjacency(line_break_props):
    uniseg_linebreak = pytest.importorskip("uniseg.linebreak", reason="uniseg not installed")
    representatives = representatives_by_class(line_break_props, count=1)
    cases = class_adjacency_strings(representatives, arity=2)
    failures = []
    for text in cases:
        sz_boundaries = byte_boundaries(sz.utf8_linebreaks(text))
        reference_boundaries = byte_boundaries(list(uniseg_linebreak.line_break_units(text)))
        if sz_boundaries != reference_boundaries:
            codepoints = " ".join(f"{ord(character):04X}" for character in text)
            failures.append(f"  {codepoints}\n    sz={sz_boundaries}\n    uniseg={reference_boundaries}")
    agreement = 1.0 - len(failures) / max(1, len(cases))
    assert agreement >= 0.80, "StringZilla vs uniseg line class-adjacency agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_linewrap_differential_uniseg(seed_value: int):
    uniseg_linebreak = pytest.importorskip("uniseg.linebreak", reason="uniseg not installed")
    seed(seed_value)
    def ref_segments(text):
        return list(uniseg_linebreak.line_break_units(text))
    iterations = scale_iterations(2000)
    failures = []
    for _ in range(iterations):
        text = "".join(choice(SEGMENTATION_PALETTE) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_linebreaks(text))
        ref_boundaries = _byte_boundaries(ref_segments(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    uniseg={ref_boundaries}")
    agreement = 1.0 - len(failures) / iterations
    assert agreement >= 0.80, "StringZilla vs uniseg line-boundary agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )
def test_utf8_linebreaks_prose():
    from test.utf8_helpers import (
        PROSE_HOTEL_REVIEW,
        PROSE_SCIENCE_ABSTRACT,
        PROSE_NEWS_LEDE,
        assert_segments_tile,
        icu_segmenter,
    )
    cases = [
        (PROSE_HOTEL_REVIEW, 45),
        (PROSE_SCIENCE_ABSTRACT, 43),
        (PROSE_NEWS_LEDE, 32),
    ]
    for text, expected in cases:
        segments = list(sz.utf8_linebreaks(text))
        assert_segments_tile(segments, text)
        assert len(segments) == expected
    oracle = icu_segmenter("line")
    for text, _ in cases:
        assert len(list(sz.utf8_linebreaks(text))) == len(oracle(text))
LINEBREAK_REALISTIC_TEXTS = [
    "first\nsecond",
    "just one line",
    "Größe\u2028привет",
    "a\r\nb",
    "alpha\nbeta\ngamma",
]
@pytest.mark.parametrize(
    "text", LINEBREAK_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_linebreaks_backend_differential(text):
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_linebreaks(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")
