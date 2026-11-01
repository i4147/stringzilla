from random import Random
import pytest
import stringzilla as sz
from test.sz_helpers import (
    vector_width_bracketing_strings,
    SEED_VALUES,
    scale_iterations,
    seed_random_generators,
    run_across_backends,
    assert_backends_agree,
    get_random_string,
    malformed_utf8_corpus,
)
from test.utf8_helpers import adversarial_utf8_inputs, random_segmentation_corpus
def test_unit_utf8_count():
    assert sz.utf8_count("hello") == 5
    assert sz.utf8_count("") == 0
    assert sz.utf8_count("a") == 1
    assert sz.utf8_count("café") == 4
    assert sz.utf8_count("日本語") == 3
    assert sz.utf8_count("🎉") == 1
    assert sz.utf8_count("👨‍👩‍👧") == 5
    assert sz.utf8_count("hello世界") == 7
    assert sz.utf8_count("naïve") == 5
    assert sz.Str("café").utf8_count() == 4
    assert sz.utf8_count(b"caf\xc3\xa9") == 4
    assert sz.utf8_count("αβγδ") == 4
    assert sz.utf8_count("АБВГ") == 4
    assert sz.utf8_count("𐍈") == 1
def test_unit_utf8_codepoints():
    assert list(sz.utf8_codepoints("")) == []
    assert list(sz.utf8_codepoints("AB")) == [65, 66]
    for text in ["café", "日本語", "🎉", "αβγδ", "АБВГ", "𐍈", "hello世界"]:
        assert list(sz.utf8_codepoints(text)) == [ord(c) for c in text]
    assert list(sz.utf8_codepoints(sz.Str("café"))) == [ord(c) for c in "café"]
    assert list(sz.utf8_codepoints("café".encode())) == [ord(c) for c in "café"]
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_count_roundtrip(seed_value: int):
    rng = Random(seed_value)
    for _ in range(scale_iterations(200)):
        raw = random_segmentation_corpus(rng.randint(1, 400), "valid", None, rng)
        text = raw.decode("utf-8")
        assert sz.utf8_count(text) == len(text)
        assert sz.utf8_count(raw) == len(text)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_codepoints_roundtrip(seed_value: int):
    rng = Random(seed_value)
    for _ in range(scale_iterations(200)):
        raw = random_segmentation_corpus(rng.randint(1, 400), "valid", None, rng)
        text = raw.decode("utf-8")
        expected = [ord(c) for c in text]
        assert list(sz.utf8_codepoints(text)) == expected
        assert list(sz.utf8_codepoints(raw)) == expected
def test_utf8_count_malformed():
    rng = Random(0)
    for raw in adversarial_utf8_inputs(rng, random_input_count=scale_iterations(500)):
        count = sz.utf8_count(raw)
        assert 0 <= count <= len(raw)
def test_utf8_codepoints_malformed():
    assert list(sz.utf8_codepoints(b"a\xffb")) == [0x61, 0xFFFD, 0x62]
    rng = Random(0)
    for raw in adversarial_utf8_inputs(rng, random_input_count=scale_iterations(500)):
        for codepoint in sz.utf8_codepoints(raw):
            assert 0 <= codepoint <= 0x10FFFF and not (0xD800 <= codepoint <= 0xDFFF)
UTF8_CODEPOINTS_BACKEND_DIFFERENTIAL_TEXTS = [
    "hello",
    "café",
    "日本語",
    "🎉",
    "αβγδ",
    "АБВГ",
    "𐍈",
    "hello世界",
    "👨‍👩‍👧",
] + vector_width_bracketing_strings()
@pytest.mark.parametrize("text", UTF8_CODEPOINTS_BACKEND_DIFFERENTIAL_TEXTS)
def test_utf8_count_backend_differential(text):
    results = run_across_backends(lambda: sz.utf8_count(text))
    assert_backends_agree(results, oracle=len(text), format_inputs=lambda: repr(text))
@pytest.mark.parametrize("text", UTF8_CODEPOINTS_BACKEND_DIFFERENTIAL_TEXTS)
def test_utf8_codepoints_backend_differential(text):
    results = run_across_backends(lambda: list(sz.utf8_codepoints(text)))
    assert_backends_agree(results, oracle=[ord(c) for c in text], format_inputs=lambda: repr(text))
@pytest.mark.parametrize("raw", malformed_utf8_corpus())
def test_utf8_count_backend_differential_malformed(raw):
    results = run_across_backends(lambda: sz.utf8_count(raw))
    assert_backends_agree(results, format_inputs=lambda: raw.hex())
    assert 0 <= next(iter(results.values())) <= len(raw)
@pytest.mark.parametrize("raw", malformed_utf8_corpus())
def test_utf8_codepoints_backend_differential_malformed(raw):
    results = run_across_backends(lambda: list(sz.utf8_codepoints(raw)))
    assert_backends_agree(results, format_inputs=lambda: raw.hex())
    for codepoint in next(iter(results.values())):
        assert 0 <= codepoint <= 0x10FFFF and not (0xD800 <= codepoint <= 0xDFFF)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_count_backend_differential_random(seed_value):
    seed_random_generators(seed_value)
    text = get_random_string()
    results = run_across_backends(lambda: sz.utf8_count(text))
    assert_backends_agree(results, oracle=len(text), format_inputs=lambda: repr(text))
