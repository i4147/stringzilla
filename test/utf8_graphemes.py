from random import Random, choice, randint
import pytest
import stringzilla as sz
from stringzilla import Str
from test.sz_helpers import (
    vector_width_bracketing_strings,
    SEED_VALUES,
    scale_iterations,
    seed_random_generators,
    run_across_backends,
    assert_backends_agree,
    get_random_string,
    malformed_utf8_corpus,
    representatives_by_class,
    baseline_grapheme_boundaries,
)
from test.utf8_helpers import (
    SEGMENTATION_PALETTE,
    adversarial_utf8_inputs,
    assert_segments_tile,
    byte_boundaries,
    class_adjacency_strings,
    corpus_of_byte_length,
    icu_segmenter,
    icu_unicode_at_least,
    window_seam_lengths,
)
_byte_boundaries = byte_boundaries
_SEGMENTATION_PALETTE = SEGMENTATION_PALETTE
def test_utf8_graphemes_basic():
    result = [str(g) for g in sz.utf8_graphemes("hello")]
    assert result == ["h", "e", "l", "l", "o"]
    assert [str(g) for g in sz.utf8_graphemes("")] == []
    result = [str(g) for g in sz.utf8_graphemes("é")]
    assert result == ["é"]
    family = "\U0001f468‍\U0001f469‍\U0001f467"
    result = [str(g) for g in sz.utf8_graphemes(family)]
    assert result == [family]
def test_utf8_graphemes_skip_empty():
    result = [str(g) for g in sz.utf8_graphemes("héllo")]
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)
def test_utf8_graphemes_unicode():
    assert [str(g) for g in sz.utf8_graphemes("ä")] == ["ä"]
    flag = "\U0001f1e6\U0001f1fa"
    assert [str(g) for g in sz.utf8_graphemes(flag)] == [flag]
    waver = "\U0001f44b\U0001f3fb"
    assert [str(g) for g in sz.utf8_graphemes(waver)] == [waver]
def test_utf8_graphemes_str_method():
    s = Str("héllo")
    method_result = [str(g) for g in s.utf8_graphemes(s)]
    module_result = [str(g) for g in sz.utf8_graphemes(s)]
    assert method_result == module_result
    assert method_result == [str(g) for g in sz.utf8_graphemes("héllo")]
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_safety(seed_value: int):
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        assert_segments_tile(sz.utf8_graphemes(raw), raw)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_seam(seed_value: int):
    rng = Random(seed_value)
    icu_graphemes = icu_segmenter("grapheme")
    icu_current = icu_unicode_at_least("17")
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_graphemes(raw), raw)
        text = raw.decode("utf-8")
        if icu_current:
            assert byte_boundaries(sz.utf8_graphemes(text)) == byte_boundaries(icu_graphemes(text))
    body = corpus_of_byte_length(96, rng).decode("utf-8")
    for phase in range(64):
        text = "a" * phase + body
        assert_segments_tile(sz.utf8_graphemes(text), text)
        if icu_current:
            assert byte_boundaries(sz.utf8_graphemes(text)) == byte_boundaries(icu_graphemes(text))
def test_utf8_grapheme_boundary_official_conformance(grapheme_break_cases):
    test_cases = grapheme_break_cases
    failures = []
    for test_str, expected_byte_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(test_str))
        if sz_boundaries != expected_byte_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_byte_boundaries}, got {sz_boundaries}")
    assert not failures, "UAX-29 GraphemeBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )
def test_baseline_grapheme_matches_official(
    grapheme_break_cases, grapheme_break_props, indic_conjunct_breaks, extended_pictographic
):
    test_cases = grapheme_break_cases
    properties = grapheme_break_props
    failures = []
    for text, expected_byte_boundaries in test_cases:
        if not text:
            continue
        got = baseline_grapheme_boundaries(text, properties, indic_conjunct_breaks, extended_pictographic)
        if got != expected_byte_boundaries:
            codepoints = " ".join(f"{ord(character):04X}" for character in text)
            failures.append(f"  {codepoints}: expected {expected_byte_boundaries}, got {got}")
    assert not failures, "baseline_grapheme_boundaries vs official GraphemeBreakTest ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )
def test_utf8_grapheme_class_adjacency(grapheme_break_props):
    icu = pytest.importorskip("icu", reason="PyICU not installed")
    icu_graphemes = icu_segmenter("grapheme")
    representatives = representatives_by_class(grapheme_break_props, count=1)
    cases = class_adjacency_strings(representatives, arity=2)
    cases += class_adjacency_strings(representatives, arity=3, max_cases=8000)
    failures = []
    for text in cases:
        if not all(icu.Char.isdefined(ord(character)) for character in text):
            continue
        sz_boundaries = byte_boundaries(sz.utf8_graphemes(text))
        icu_boundaries = byte_boundaries(icu_graphemes(text))
        if sz_boundaries != icu_boundaries:
            codepoints = " ".join(f"{ord(character):04X}" for character in text)
            failures.append(f"  {codepoints}\n    sz={sz_boundaries}\n    icu={icu_boundaries}")
    assert not failures, "StringZilla vs ICU grapheme class-adjacency divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_boundary_fuzz(seed_value: int, grapheme_break_props):
    seed_random_generators(seed_value)
    for _ in range(scale_iterations(200)):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 100)))
        try:
            expected = baseline_grapheme_boundaries(text, grapheme_break_props)
        except Exception:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(text))
        assert (
            sz_boundaries == expected
        ), "Grapheme boundary mismatch (seed {}):\n  text cps: {}\n  expected: {}\n  got:      {}".format(
            seed_value, " ".join(f"{ord(c):04X}" for c in text), expected, sz_boundaries
        )
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_differential_grapheme(seed_value: int):
    seed_random_generators(seed_value)
    try:
        graphemes_fn = pytest.importorskip("grapheme", reason="grapheme not installed").graphemes
    except Exception:
        uniseg_gc = pytest.importorskip("uniseg.graphemecluster", reason="uniseg not installed")
        graphemes_fn = uniseg_gc.grapheme_clusters
    failures = []
    for _ in range(scale_iterations(2000)):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(text))
        ref_boundaries = _byte_boundaries(graphemes_fn(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    ref={ref_boundaries}")
    assert not failures, "StringZilla vs reference grapheme-boundary divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_differential_icu(seed_value: int):
    icu_graphemes = icu_segmenter("grapheme")
    icu_current = icu_unicode_at_least("17")
    rng = Random(seed_value)
    iterations = scale_iterations(2000)
    failures = []
    for _ in range(iterations):
        text = "".join(rng.choice(SEGMENTATION_PALETTE) for _ in range(rng.randint(1, 32)))
        sz_boundaries = byte_boundaries(sz.utf8_graphemes(text))
        icu_boundaries = byte_boundaries(icu_graphemes(text))
        if sz_boundaries != icu_boundaries:
            codepoints = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {codepoints}\n    sz={sz_boundaries}\n    icu={icu_boundaries}")
    if icu_current:
        assert not failures, "StringZilla vs ICU grapheme-boundary divergences ({}):\n{}".format(
            len(failures), "\n".join(failures[:20])
        )
    else:
        agreement = 1.0 - len(failures) / iterations
        assert agreement >= 0.99, "StringZilla vs stale-ICU grapheme agreement {:.2%} too low:\n{}".format(
            agreement, "\n".join(failures[:20])
        )
def test_utf8_graphemes_prose():
    from test.utf8_helpers import (
        PROSE_PRIDE_CAPTION,
        PROSE_DEVANAGARI_TIP,
        PROSE_CONCERT_POST,
        PROSE_RTL_SCRIPTS,
        PROSE_MICRO_PREPEND,
        assert_segments_tile,
        icu_segmenter,
    )
    cases = [
        (PROSE_PRIDE_CAPTION, 206),
        (PROSE_DEVANAGARI_TIP, 252),
        (PROSE_CONCERT_POST, 134),
        (PROSE_RTL_SCRIPTS, 256),
        (PROSE_MICRO_PREPEND, 3),
    ]
    for text, expected in cases:
        clusters = list(sz.utf8_graphemes(text))
        assert_segments_tile(clusters, text)
        assert len(clusters) == expected
    oracle = icu_segmenter("grapheme")
    for text, _ in cases:
        assert len(list(sz.utf8_graphemes(text))) == len(oracle(text))
    assert sz.utf8_count(PROSE_PRIDE_CAPTION) == 222
    assert sz.utf8_count(PROSE_PRIDE_CAPTION) > sum(1 for _ in sz.utf8_graphemes(PROSE_PRIDE_CAPTION))
def _grapheme_segment_bytes(text_or_bytes) -> list:
    return [bytes(segment) for segment in sz.utf8_graphemes(text_or_bytes)]
GRAPHEME_BACKEND_DIFFERENTIAL_TEXTS = [
    "hello",
    "héllo",
    "\U0001f468‍\U0001f469‍\U0001f467",
    "\U0001f1e6\U0001f1fa",
    "\U0001f44b\U0001f3fb",
    "é",
    "\r\n",
    "a\r\nb\r\n",
    "café\r\ncafé",
] + vector_width_bracketing_strings()
@pytest.mark.parametrize("text", GRAPHEME_BACKEND_DIFFERENTIAL_TEXTS)
def test_utf8_graphemes_backend_differential(text):
    results = run_across_backends(lambda: _grapheme_segment_bytes(text))
    assert_backends_agree(results, format_inputs=lambda: repr(text))
@pytest.mark.parametrize("raw", malformed_utf8_corpus())
def test_utf8_graphemes_backend_differential_malformed(raw):
    def segment_and_check():
        segments = _grapheme_segment_bytes(raw)
        assert_segments_tile(segments, raw)
        return segments
    results = run_across_backends(segment_and_check)
    assert_backends_agree(results, format_inputs=lambda: raw.hex())
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_graphemes_backend_differential_random(seed_value):
    seed_random_generators(seed_value)
    text = get_random_string()
    results = run_across_backends(lambda: _grapheme_segment_bytes(text))
    assert_backends_agree(results, format_inputs=lambda: repr(text))
