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
def test_utf8_lines():
    assert list(sz.utf8_split_newlines("a\nb\nc")) == [Str("a"), Str("b"), Str("c")]
    assert list(sz.utf8_split_newlines("a\rb\rc")) == [Str("a"), Str("b"), Str("c")]
    assert list(sz.utf8_split_newlines("a\r\nb\r\nc")) == [Str("a"), Str("b"), Str("c")]
    result = list(sz.utf8_split_newlines("line1\r\nline2"))
    assert len(result) == 2
    assert result[0] == Str("line1")
    assert result[1] == Str("line2")
    assert list(sz.utf8_split_newlines("")) == [Str("")]
    assert list(sz.utf8_split_newlines("hello")) == [Str("hello")]
    result = list(sz.utf8_split_newlines("a\n\nb"))
    assert len(result) == 3
    assert result[1] == Str("")
    result = list(sz.utf8_split_newlines("a\nb\n"))
    assert len(result) == 3
    assert result[2] == Str("")
    result = list(sz.utf8_split_newlines("a\nb\nc", with_separators=True))
    assert "".join(str(s) for s in result) == "a\nb\nc"
    assert result == [Str("a"), Str("\n"), Str("b"), Str("\n"), Str("c")]
    result = list(sz.utf8_split_newlines("a\r\nb", with_separators=True))
    assert result == [Str("a"), Str("\r\n"), Str("b")]
    assert len(list(sz.utf8_split_newlines("a\vb\fc"))) == 3
    nel = "\u0085"
    line_sep = "\u2028"
    para_sep = "\u2029"
    assert len(list(sz.utf8_split_newlines(f"a{nel}b"))) == 2
    assert len(list(sz.utf8_split_newlines(f"a{line_sep}b"))) == 2
    assert len(list(sz.utf8_split_newlines(f"a{para_sep}b"))) == 2
    result = list(sz.Str("a\nb").utf8_split_newlines())
    assert result == [Str("a"), Str("b")]
    assert list(sz.utf8_split_newlines("", skip_empty=True)) == []
    assert list(sz.utf8_split_newlines("a\nb\n", skip_empty=True)) == [Str("a"), Str("b")]
    assert list(sz.utf8_split_newlines("a\n\nb", skip_empty=True)) == [Str("a"), Str("b")]
    assert list(sz.utf8_split_newlines("\na\n", skip_empty=True)) == [Str("a")]
def test_utf8_tokens():
    result = list(sz.utf8_split_whitespaces("hello world"))
    assert result == [Str("hello"), Str("world")]
    result = list(sz.utf8_split_whitespaces("a\tb\nc"))
    assert result == [Str("a"), Str("b"), Str("c")]
    assert list(sz.utf8_split_whitespaces("")) == [Str("")]
    result = list(sz.utf8_split_whitespaces("   "))
    assert all(str(s) == "" for s in result)
    result = list(sz.utf8_split_whitespaces("a   b"))
    assert len(result) == 4
    assert str(result[0]) == "a"
    assert str(result[-1]) == "b"
    result = list(sz.utf8_split_whitespaces("  hello  "))
    assert "hello" in [str(s) for s in result]
    assert list(sz.utf8_split_whitespaces("hello")) == [Str("hello")]
    nbsp = "\u00a0"
    result = list(sz.utf8_split_whitespaces(f"a{nbsp}b"))
    assert len(result) == 2
    assert str(result[0]) == "a"
    assert str(result[1]) == "b"
    ideo_space = "\u3000"
    result = list(sz.utf8_split_whitespaces(f"日本{ideo_space}語"))
    assert len(result) == 2
    assert str(result[0]) == "日本"
    assert str(result[1]) == "語"
    en_space = "\u2002"
    em_space = "\u2003"
    result = list(sz.utf8_split_whitespaces(f"a{en_space}b{em_space}c"))
    assert len(result) == 3
    assert str(result[0]) == "a"
    assert str(result[1]) == "b"
    assert str(result[2]) == "c"
    result = list(sz.Str("hello world").utf8_split_whitespaces())
    assert len(result) == 2
    assert str(result[0]) == "hello"
    assert str(result[1]) == "world"
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_tokens_safety(seed_value: int):
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        for splitter in (sz.utf8_split_whitespaces, sz.utf8_split_newlines):
            segments = list(splitter(raw))
            assert sum(len(bytes(segment)) for segment in segments) <= len(raw)
@pytest.mark.parametrize(
    "text",
    ["hello\nworld", "a\r\nb\nc", "no newline", "trailing\n"],
)
def test_utf8_lines_matches_python(text):
    py_result = text.splitlines()
    sz_result = [str(s) for s in sz.utf8_split_newlines(text, skip_empty=True)]
    assert sz_result == py_result
@pytest.mark.parametrize(
    "text",
    ["hello world", "  spaced  ", "a\tb\nc", "no-spaces", "   ", ""],
)
def test_utf8_tokens_matches_python(text):
    py_result = text.split()
    sz_result = [str(s) for s in sz.utf8_split_whitespaces(text, skip_empty=True)]
    assert sz_result == py_result
TOKEN_REALISTIC_TEXTS = [
    "hello world",
    "a\tb\nc",
    "日本 語",
    "  hello  ",
    "привет мир",
]
@pytest.mark.parametrize(
    "text", TOKEN_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_tokens_backend_differential(text):
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_split_whitespaces(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")
