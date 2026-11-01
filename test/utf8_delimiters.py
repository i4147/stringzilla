from random import Random
import pytest
import stringzilla as sz
from stringzilla import Str
from test.sz_helpers import (
    SEED_VALUES,
    assert_backends_agree,
    malformed_utf8_corpus,
    run_across_backends,
    vector_width_bracketing_strings,
)
from test.utf8_helpers import adversarial_utf8_inputs
def test_utf8_delimiters():
    assert list(sz.utf8_split_delimiters("Hi, world")) == [Str("Hi"), Str(""), Str("world")]
    assert list(sz.utf8_split_delimiters("Hi, world", skip_empty=True)) == [Str("Hi"), Str("world")]
    assert list(sz.utf8_split_delimiters("a.b")) == [Str("a"), Str("b")]
    assert list(sz.utf8_split_delimiters("")) == [Str("")]
    assert list(sz.utf8_split_delimiters("hello")) == [Str("hello")]
    assert list(sz.utf8_split_delimiters("a—b", skip_empty=True)) == [Str("a"), Str("b")]
    assert list(sz.utf8_split_delimiters("日本。語", skip_empty=True)) == [Str("日本"), Str("語")]
    assert list(sz.utf8_split_delimiters("a\U0001f600b", skip_empty=True)) == [Str("a"), Str("b")]
    assert list(sz.Str("Hi, world").utf8_split_delimiters(skip_empty=True)) == [Str("Hi"), Str("world")]
    assert list(sz.utf8_split_delimiters("abc123")) == [Str("abc123")]
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_delimiters_safety(seed_value: int):
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        segments = list(sz.utf8_split_delimiters(raw))
        assert sum(len(bytes(segment)) for segment in segments) <= len(raw)
DELIMITER_REALISTIC_TEXTS = [
    "Hi, world",
    "a.b,c",
    "日本。語",
    "a—b",
    "abc123",
]
@pytest.mark.parametrize(
    "text", DELIMITER_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_delimiters_backend_differential(text):
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_split_delimiters(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")
