import pytest
import stringzilla as sz
from stringzilla import Str
from test.sz_helpers import (
    assert_backends_agree,
    malformed_utf8_corpus,
    run_across_backends,
    vector_width_bracketing_strings,
)
def test_between_vs_separators():
    text = "a b  c"
    assert list(sz.utf8_split_whitespaces(text)) == [Str("a"), Str("b"), Str(""), Str("c")]
    assert list(sz.utf8_whitespaces(text)) == [Str(" "), Str(" "), Str(" ")]
    assert list(sz.utf8_split_newlines("a\nb")) == [Str("a"), Str("b")]
    assert list(sz.utf8_newlines("a\nb")) == [Str("\n")]
    assert list(sz.utf8_delimiters("a,b")) == [Str(",")]
def test_with_separators_round_trip():
    for text in ["a b  c", "  x  ", "", "abc", "a\r\nb", ",,a,,"]:
        joined = "".join(str(s) for s in sz.utf8_split_delimiters(text, with_separators=True))
        assert joined == text
        joined_nl = "".join(str(s) for s in sz.utf8_split_newlines(text, with_separators=True))
        assert joined_nl == text
def test_member_form_matches_free_function():
    text = "one two  three"
    assert list(sz.Str(text).utf8_split_whitespaces(skip_empty=True)) == [Str("one"), Str("two"), Str("three")]
    assert list(sz.Str(text).utf8_whitespaces()) == list(sz.utf8_whitespaces(text))
def test_wordbreaks_tiles():
    segments = list(sz.utf8_wordbreaks("Hello, world!"))
    assert "".join(str(s) for s in segments) == "Hello, world!"
    assert len(segments) == 5
def test_empty_and_skip_empty():
    assert list(sz.utf8_split_whitespaces("")) == [Str("")]
    assert list(sz.utf8_split_whitespaces("a  b")) == [Str("a"), Str(""), Str("b")]
    assert list(sz.utf8_split_whitespaces("a  b", skip_empty=True)) == [Str("a"), Str("b")]
def test_zero_width_format_chars_are_not_whitespace():
    assert list(sz.utf8_split_whitespaces("a b")) == [Str("a"), Str("b")]
    for cp in ("​", "‌", "‍"):
        assert list(sz.utf8_split_whitespaces(f"a{cp}b")) == [Str(f"a{cp}b")]
    assert list(sz.utf8_split_whitespaces("\U0001f468‍\U0001f469")) == [Str("\U0001f468‍\U0001f469")]
SPLIT_MODE_REALISTIC_TEXTS = [
    "a b  c",
    "  x  ",
    "one two  three",
    "日本 語",
    "привет мир",
]
@pytest.mark.parametrize(
    "text", SPLIT_MODE_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_split_modes_backend_differential(text):
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_split_whitespaces(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")
