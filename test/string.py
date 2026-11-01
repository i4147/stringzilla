import os
import sys
import math
import tempfile
from string import ascii_lowercase
from typing import Sequence
import pytest
import stringzilla as sz
from stringzilla import Str, Strs
from test.sz_helpers import (
    SEED_VALUES,
    DEGENERATE_HAYSTACKS,
    scale_iterations,
    seed_random_generators,
    get_random_string,
    numpy_available,
    pyarrow_available,
    run_across_backends,
    assert_backends_agree,
    VECTOR_WIDTH_LENGTHS,
    differential_bodies,
    unaligned_views,
)
try:
    import numpy as np
except:
    pass
try:
    import pyarrow as pa
except:
    pass
SPLIT_PARITY_MAXSPLITS = [-1, 0, 1, 2, 99]
SPLITLINES_CASES = ["", "a", "a\n", "\n", "\n\n", "a\nb", "a\n\nb", "a\nb\n"]
def test_library_properties():
    assert len(sz.__version__.split(".")) == 3, "Semantic versioning must be preserved"
    assert "serial" in sz.__capabilities__, "Serial backend must be present"
    assert isinstance(sz.__capabilities_str__, str) and len(sz.__capabilities_str__) > 0
    sz.reset_capabilities(sz.__capabilities__)
@pytest.mark.parametrize("native_type", [str, bytes, bytearray])
def test_unit_construct(native_type):
    native = "aaaaa"
    if native_type is bytes or native_type is bytearray:
        native = native_type(native, "utf-8")
    big = Str(native)
    assert len(big) == len(native)
def test_str_repr():
    native = "abcdef"
    big = Str(native)
    assert repr(big) == f"sz.Str({repr(native)})"
    assert str(big) == native
def test_unit_indexing():
    native = "abcdef"
    big = Str(native)
    for i in range(len(native)):
        assert big[i] == native[i]
def test_unit_str_rich_comparisons():
    assert Str("aa") == "aa"
    assert Str("a") != "b"
    assert Str("abc") == Str("abc")
    assert Str("abc") != Str("abd")
    assert Str("aa") < "b"
    assert Str("ab") <= "ab"
    assert Str("a") < Str("b")
    assert Str("abc") <= Str("abcd")
    assert Str("b") > "aa"
    assert Str("ab") >= "ab"
    assert Str("b") > Str("a")
    assert Str("abcd") >= Str("abc")
    s2 = Str("abb")
    assert s2[1:] == "bb"
    assert s2[:-1] == "ab"
    assert s2[-1:] == "b"
    assert s2[1:] != "abb"
    assert s2[:-2] == "a"
    assert s2[-2:] == "bb"
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_unit_buffer_protocol():
    sz_str = Str("hello")
    np_array = np.array(sz_str)
    assert np_array.dtype == np.dtype("c")
    assert np_array.shape == (len("hello"),)
    assert "".join([c.decode("utf-8") for c in np_array.tolist()]) == "hello"
    np_array = np.arange(ord("a"), ord("z"), dtype=np.uint8)
    sz_str = sz.Str(memoryview(np_array))
    assert len(np_array) == len(sz_str)
    np_array = np.arange(ord("a"), ord("a") + 8, dtype=np.uint8).reshape((2, 2, 2))
    sz_str = sz.Str(memoryview(np_array))
    assert np_array.size == len(sz_str)
    np_array = np.arange(ord("a"), ord("z"), dtype=np.uint8)[::2]
    with pytest.raises(ValueError):
        sz.Str(memoryview(np_array))
    with pytest.raises(TypeError):
        sz.Str(np.array())
    with pytest.raises(TypeError):
        sz.Str(dict())
def test_str_write_to():
    native = "line1\nline2\nline3"
    big = Str(native)
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        temp_filename = tmpfile.name
    try:
        big.write_to(temp_filename)
        with open(temp_filename, "r") as file:
            content = file.read()
            assert content == native, "The content of the file does not match the expected output"
    finally:
        os.remove(temp_filename)
def test_unit_split():
    native = "line1\nline2\nline3"
    big = Str(native)
    lines = sz.split(big, "\n")
    assert lines == ["line1", "line2", "line3"]
    lines = sz.rsplit(big, "\n")
    assert lines == ["line1", "line2", "line3"]
    lines = sz.split(big, "\n", keepseparator=True)
    assert lines == ["line1\n", "line2\n", "line3"]
    letters = sz.split("a b c d")
    assert letters == ["a", "b", "c", "d"]
    letters = sz.split_byteset("a b_c d", " _")
    assert letters == ["a", "b", "c", "d"]
    letters = sz.rsplit_byteset("a b_c d", " _")
    assert letters == ["a", "b", "c", "d"]
    assert native.splitlines() == list(big.splitlines())
    assert native.splitlines(True) == list(big.splitlines(keeplinebreaks=True))
    assert native.split("l") == list(big.split("l"))
    assert native.split("li") == list(big.split("li"))
    assert native.split("lin") == list(big.split("lin"))
    assert native.split("line") == list(big.split("line"))
    assert native.split("line1") == list(big.split("line1"))
    assert native.split("line3") == list(big.split("line3"))
    assert native.split("\n", maxsplit=0) == list(big.split("\n", maxsplit=0))
    assert native.split("\n", maxsplit=1) == list(big.split("\n", maxsplit=1))
    assert native.split("\n", maxsplit=2) == list(big.split("\n", maxsplit=2))
    assert native.split("\n", maxsplit=3) == list(big.split("\n", maxsplit=3))
    assert native.split("\n", maxsplit=4) == list(big.split("\n", maxsplit=4))
    assert native.rsplit("line1") == list(big.rsplit("line1"))
    assert native.rsplit("line3") == list(big.rsplit("line3"))
    assert native.rsplit("\n", maxsplit=0) == list(big.rsplit("\n", maxsplit=0))
    assert native.rsplit("\n", maxsplit=1) == list(big.rsplit("\n", maxsplit=1))
    assert native.rsplit("\n", maxsplit=2) == list(big.rsplit("\n", maxsplit=2))
    assert native.rsplit("\n", maxsplit=3) == list(big.rsplit("\n", maxsplit=3))
    assert native.rsplit("\n", maxsplit=4) == list(big.rsplit("\n", maxsplit=4))
    with pytest.raises(ValueError):
        sz.split(big, "")
    with pytest.raises(ValueError):
        sz.rsplit(big, "")
    with pytest.raises(ValueError):
        sz.split_byteset(big, "")
    with pytest.raises(ValueError):
        sz.rsplit_byteset(big, "")
def test_unit_split_skip_empty():
    def strs(x):
        return [str(s) for s in x]
    assert strs(Str("a,,b,").split(",", skip_empty=True)) == ["a", "b"]
    assert strs(Str(",a,,b,").split(",", skip_empty=True)) == ["a", "b"]
    assert strs(Str("a,,b,").split(",")) == ["a", "", "b", ""]
    assert strs(Str("a,,b,").rsplit(",", skip_empty=True)) == ["a", "b"]
    assert strs(Str("a,,b,").rsplit(",")) == ["a", "", "b", ""]
    assert strs(Str("a,;b;").split_byteset(",;", skip_empty=True)) == ["a", "b"]
    assert strs(Str("a,;b;").rsplit_byteset(",;", skip_empty=True)) == ["a", "b"]
    assert strs(Str("a,;b;").split_byteset(",;")) == ["a", "", "b", ""]
    assert strs(sz.split_iter("a,,b,", ",", skip_empty=True)) == ["a", "b"]
    assert strs(sz.rsplit_iter("a,,b,", ",", skip_empty=True)) == ["b", "a"]
    assert strs(sz.split_byteset_iter("a,;b;", ",;", skip_empty=True)) == ["a", "b"]
    assert strs(sz.rsplit_byteset_iter("a,;b;", ",;", skip_empty=True)) == ["b", "a"]
    assert strs(Str("a,,b,c").split(",", maxsplit=1, skip_empty=True)) == ["a", ",b,c"]
    assert strs(Str(",,,").split(",", skip_empty=True)) == []
    assert strs(sz.split_iter(",,,", ",", skip_empty=True)) == []
def test_unit_split_iterators():
    native = "line1\nline2\nline3"
    big = Str(native)
    lines = list(sz.split_iter(big, "\n"))
    assert lines == ["line1", "line2", "line3"]
    lines = list(sz.rsplit_iter(big, "\n"))
    assert lines == ["line3", "line2", "line1"]
    lines = list(sz.split_iter(big, "\n", keepseparator=True))
    assert lines == ["line1\n", "line2\n", "line3"]
    lines = list(sz.rsplit_iter(big, "\n", keepseparator=True))
    assert lines == ["\nline3", "\nline2", "line1"]
    letters = list(sz.split_iter("a b c d"))
    assert letters == ["a", "b", "c", "d"]
    letters = list(sz.split_byteset_iter("a-b_c-d", "-_"))
    assert letters == ["a", "b", "c", "d"]
    letters = list(sz.rsplit_byteset_iter("a-b_c-d", "-_"))
    assert letters == ["d", "c", "b", "a"]
    assert native.split("line1") == list(big.split_iter("line1"))
    assert native.split("line3") == list(big.split_iter("line3"))
    assert native.split("\n", maxsplit=0) == list(big.split_iter("\n", maxsplit=0))
    assert native.split("\n", maxsplit=1) == list(big.split_iter("\n", maxsplit=1))
    assert native.split("\n", maxsplit=2) == list(big.split_iter("\n", maxsplit=2))
    assert native.split("\n", maxsplit=3) == list(big.split_iter("\n", maxsplit=3))
    assert native.split("\n", maxsplit=4) == list(big.split_iter("\n", maxsplit=4))
    def rlist(seq):
        seq = list(seq)
        seq.reverse()
        return seq
    assert native.rsplit("line1") == rlist(big.rsplit_iter("line1"))
    assert native.rsplit("line3") == rlist(big.rsplit_iter("line3"))
    assert native.rsplit("\n", maxsplit=0) == rlist(big.rsplit_iter("\n", maxsplit=0))
    assert native.rsplit("\n", maxsplit=1) == rlist(big.rsplit_iter("\n", maxsplit=1))
    assert native.rsplit("\n", maxsplit=2) == rlist(big.rsplit_iter("\n", maxsplit=2))
    assert native.rsplit("\n", maxsplit=3) == rlist(big.rsplit_iter("\n", maxsplit=3))
    assert native.rsplit("\n", maxsplit=4) == rlist(big.rsplit_iter("\n", maxsplit=4))
    with pytest.raises(ValueError):
        sz.split_iter(big, "")
    with pytest.raises(ValueError):
        sz.rsplit_iter(big, "")
    with pytest.raises(ValueError):
        sz.split_byteset_iter(big, "")
    with pytest.raises(ValueError):
        sz.rsplit_byteset_iter(big, "")
def test_unit_strip():
    native_whitespace = "  \t\n hello world \r\f\v  "
    big_whitespace = Str(native_whitespace)
    assert native_whitespace.lstrip() == str(sz.lstrip(big_whitespace))
    assert native_whitespace.lstrip() == str(big_whitespace.lstrip())
    assert native_whitespace.rstrip() == str(sz.rstrip(big_whitespace))
    assert native_whitespace.rstrip() == str(big_whitespace.rstrip())
    assert native_whitespace.strip() == str(sz.strip(big_whitespace))
    assert native_whitespace.strip() == str(big_whitespace.strip())
    native_custom = "aaabbbhello worldcccaaa"
    big_custom = Str(native_custom)
    chars = "abc"
    assert native_custom.lstrip(chars) == str(sz.lstrip(big_custom, chars))
    assert native_custom.lstrip(chars) == str(big_custom.lstrip(chars))
    assert native_custom.rstrip(chars) == str(sz.rstrip(big_custom, chars))
    assert native_custom.rstrip(chars) == str(big_custom.rstrip(chars))
    assert native_custom.strip(chars) == str(sz.strip(big_custom, chars))
    assert native_custom.strip(chars) == str(big_custom.strip(chars))
    empty = ""
    big_empty = Str(empty)
    assert empty.strip() == str(sz.strip(big_empty))
    assert empty.strip() == str(big_empty.strip())
    only_whitespace = " \t\n\r\f\v "
    big_only_whitespace = Str(only_whitespace)
    assert only_whitespace.strip() == str(sz.strip(big_only_whitespace))
    assert only_whitespace.strip() == str(big_only_whitespace.strip())
    only_custom = "aaabbbccc"
    big_only_custom = Str(only_custom)
    assert only_custom.strip("abc") == str(sz.strip(big_only_custom, "abc"))
    assert only_custom.strip("abc") == str(big_only_custom.strip("abc"))
    no_strip = "hello world"
    big_no_strip = Str(no_strip)
    assert no_strip.strip() == str(sz.strip(big_no_strip))
    assert no_strip.strip() == str(big_no_strip.strip())
    native_bytes = b"  hello world  "
    big_bytes = Str(native_bytes)
    assert native_bytes.strip() == bytes(sz.strip(big_bytes))
    assert native_bytes.strip() == bytes(big_bytes.strip())
    native_asymmetric = "aaahello worldbbb"
    big_asymmetric = Str(native_asymmetric)
    assert native_asymmetric.lstrip("a") == str(sz.lstrip(big_asymmetric, "a"))
    assert native_asymmetric.lstrip("a") == str(big_asymmetric.lstrip("a"))
    assert native_asymmetric.rstrip("b") == str(sz.rstrip(big_asymmetric, "b"))
    assert native_asymmetric.rstrip("b") == str(big_asymmetric.rstrip("b"))
    native_special = "!!!###hello world***!!!"
    big_special = Str(native_special)
    special_chars = "!#*"
    assert native_special.strip(special_chars) == str(sz.strip(big_special, special_chars))
    assert native_special.strip(special_chars) == str(big_special.strip(special_chars))
    native_single = "aaa"
    big_single = Str(native_single)
    assert native_single.strip("a") == str(sz.strip(big_single, "a"))
    assert native_single.strip("a") == str(big_single.strip("a"))
    native_no_match = "hello world"
    big_no_match = Str(native_no_match)
    assert native_no_match.strip("xyz") == str(sz.strip(big_no_match, "xyz"))
    assert native_no_match.strip("xyz") == str(big_no_match.strip("xyz"))
def test_unit_slicing():
    native = "abcdef"
    big = Str(native)
    assert big[1:3] == "bc"
    assert big[1:] == "bcdef"
    assert big[:3] == "abc"
    assert big[-1:] == "f"
    assert big[:-1] == "abcde"
    assert big[-3:] == "def"
    assert big[:-3] == "abc"
@pytest.mark.parametrize("native", SPLITLINES_CASES)
def test_unit_splitlines_parity(native):
    got = [str(s) for s in Str(native).splitlines()]
    assert got == native.splitlines()
@pytest.mark.parametrize("maxsplit", SPLIT_PARITY_MAXSPLITS)
def test_unit_split_iter_parity(maxsplit):
    haystack = Str("a,b,c,d,e")
    lazy = list(map(str, haystack.split_iter(",", maxsplit)))
    assert lazy == haystack.split(",", maxsplit)
@pytest.mark.parametrize("maxsplit", SPLIT_PARITY_MAXSPLITS)
def test_unit_rsplit_iter_parity(maxsplit):
    haystack = Str("a,b,c,d,e")
    lazy_reversed = list(reversed(list(map(str, haystack.rsplit_iter(",", maxsplit)))))
    assert lazy_reversed == haystack.rsplit(",", maxsplit)
@pytest.mark.parametrize("maxsplit", SPLIT_PARITY_MAXSPLITS)
def test_unit_split_byteset_iter_parity(maxsplit):
    haystack = Str("a,b;c,d;e")
    lazy = list(map(str, haystack.split_byteset_iter(",;", maxsplit)))
    assert lazy == haystack.split_byteset(",;", maxsplit)
@pytest.mark.parametrize("maxsplit", SPLIT_PARITY_MAXSPLITS)
def test_unit_rsplit_byteset_iter_parity(maxsplit):
    haystack = Str("a,b;c,d;e")
    lazy_reversed = list(reversed(list(map(str, haystack.rsplit_byteset_iter(",;", maxsplit)))))
    assert lazy_reversed == haystack.rsplit_byteset(",;", maxsplit)
def test_unit_str_memoryview_is_readonly():
    assert memoryview(Str("abc")).readonly is True
def test_unit_strs_sequence_slicing():
    native = "1, 2, 3, 4, 5, 6"
    big = Str(native)
    big_sequence = big.split(", ")
    def to_str(seq):
        return "".join([str(x) for x in seq])
    assert big_sequence[1:3] == ["2", "3"], to_str(big_sequence[1:3])
    assert big_sequence[1:] == ["2", "3", "4", "5", "6"], to_str(big_sequence[1:])
    assert big_sequence[:3] == ["1", "2", "3"], to_str(big_sequence[:3])
    assert big_sequence[-1:] == ["6"], to_str(big_sequence[-1:])
    assert big_sequence[:-1] == ["1", "2", "3", "4", "5"], to_str(big_sequence[:-1])
    assert big_sequence[-3:] == ["4", "5", "6"], to_str(big_sequence[-3:])
    assert big_sequence[:-3] == ["1", "2", "3"], to_str(big_sequence[:-3])
    assert big_sequence[::2] == ["1", "3", "5"], to_str(big_sequence[::2])
    assert big_sequence[::-1] == ["6", "5", "4", "3", "2", "1"], to_str(big_sequence[::-1])
    assert big_sequence[1:][::-1] == ["6", "5", "4", "3", "2"]
    assert big_sequence[1:][::-2] == ["6", "4", "2"]
    assert big_sequence[1:][::-3] == ["6", "3"]
    assert big_sequence[0:1] == ["1"], "Single element slice [0:1] failed"
    assert big_sequence[1:2] == ["2"], "Single element slice [1:2] failed"
    assert big_sequence[-1:] == ["6"], "Single element slice [-1:] failed"
    assert big_sequence[-2:-1] == ["5"], "Single element slice [-2:-1] failed"
    assert big_sequence[0:0] == [], "Empty slice [0:0] failed"
    assert big_sequence[1:1] == [], "Empty slice [1:1] failed"
    assert big_sequence[5:5] == [], "Empty slice [5:5] failed"
    assert big_sequence[10:10] == [], "Empty slice [10:10] out of bounds failed"
    assert big_sequence[1:0] == [], "Empty slice [1:0] (start > end) failed"
    assert big_sequence[3:1] == [], "Empty slice [3:1] (start > end) failed"
    single_slice = big_sequence[0:1]
    empty_slice = big_sequence[0:0]
    assert len(single_slice) == 1, "Single slice length incorrect"
    assert len(empty_slice) == 0, "Empty slice length incorrect"
    assert str(single_slice[0]) == "1", "Single slice element access failed"
    single_items = [str(item) for item in single_slice]
    assert single_items == ["1"], "Single slice iteration failed"
    empty_items = [str(item) for item in empty_slice]
    assert empty_items == [], "Empty slice iteration failed"
    assert isinstance(repr(single_slice), str), "Single slice repr failed"
    assert isinstance(repr(empty_slice), str), "Empty slice repr failed"
    assert isinstance(str(single_slice), str), "Single slice str failed"
    assert isinstance(str(empty_slice), str), "Empty slice str failed"
def test_unit_strs_rich_comparisons():
    arr: Strs = Str("a b c d e f g h").split()
    identical_arr: Strs = Str("a b c d e f g h").split()
    different_arr: Strs = Str("a b c d e f g i").split()
    shorter_arr: Strs = Str("a b c d e").split()
    longer_arr: Strs = Str("a b c d e f g h i j").split()
    assert arr == identical_arr
    assert arr != different_arr
    assert arr != shorter_arr
    assert arr != longer_arr
    assert shorter_arr < arr
    assert longer_arr > arr
    list_equal = ["a", "b", "c", "d", "e", "f", "g", "h"]
    list_different = ["a", "b", "c", "d", "x", "f", "g", "h"]
    tuple_equal = ("a", "b", "c", "d", "e", "f", "g", "h")
    tuple_different = ("a", "b", "c", "d", "e", "f", "g", "i")
    assert arr == list_equal
    assert arr != list_different
    assert arr == tuple_equal
    assert arr != tuple_different
    generator_equal = (x for x in "a b c d e f g h".split())
    generator_different = (x for x in "a b c d e f g i".split())
    generator_shorter = (x for x in "a b c d e".split())
    generator_longer = (x for x in "a b c d e f g h i j".split())
    assert arr == generator_equal
    assert arr != generator_different
    assert arr != generator_shorter
    assert arr != generator_longer
def test_string_lengths():
    assert 4 == len(sz.Str("abcd"))
    assert 8 == len(sz.Str("αβγδ"))
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("length", range(1, 300))
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_translations(length: int, seed_value: int):
    seed_random_generators(seed_value)
    map_identity = np.arange(256, dtype=np.uint8)
    map_invert = np.arange(255, -1, -1, dtype=np.uint8)
    map_threshold = np.where(np.arange(256) > 127, 255, 0).astype(np.uint8)
    view_identity = memoryview(map_identity)
    view_invert = memoryview(map_invert)
    view_threshold = memoryview(map_threshold)
    body = get_random_string(length=length)
    body_bytes = body.encode("utf-8")
    assert sz.translate(body, view_identity) == body
    assert sz.translate(body_bytes, view_identity) == body_bytes
    assert sz.translate(body_bytes, view_identity) == body_bytes.translate(view_identity)
    assert sz.translate(body_bytes, view_invert) == body_bytes.translate(view_invert)
    assert sz.translate(body_bytes, view_threshold) == body_bytes.translate(view_threshold)
    out = np.full((1, len(body_bytes) + 1), 0xA5, dtype=np.uint8)
    for table in (view_identity, view_invert, view_threshold):
        assert sz.translate(body_bytes, table, out=out) is out
        assert out[0, :-1].tobytes() == body_bytes.translate(table)
        assert out[0, -1] == 0xA5
    after_identity = bytearray(body_bytes)
    assert sz.translate(after_identity, view_identity, inplace=True) is None
    assert sz.equal(after_identity, body_bytes.translate(view_identity))
    after_invert = bytearray(body_bytes)
    assert sz.translate(after_invert, view_invert, inplace=True) is None
    assert sz.equal(after_invert, body_bytes.translate(view_invert))
    after_threshold = bytearray(body_bytes)
    assert sz.translate(after_threshold, view_threshold, inplace=True) is None
    assert sz.equal(after_threshold, body_bytes.translate(view_threshold))
def test_translation_out_buffers():
    source = bytearray(b"_abc_")
    out = memoryview(source)[1:-1]
    assert Str(source).translate({"a": "A"}, start=1, end=-1, out=out) is out
    assert source == b"_Abc_"
    for start in (0, 2):
        with pytest.raises(ValueError, match="overlap"):
            sz.translate(out, {}, out=memoryview(source)[start : start + 3])
    assert Str().translate({}, out=bytearray(b"!")) == b"!"
    assert sz.translate("abc", {}, out=None) == "abc"
def test_translation_out_errors():
    source = bytearray(b"abc")
    for out, error in (
        (b"...", BufferError),
        (memoryview(bytearray(6))[::2], BufferError),
        (memoryview(bytearray(12)).cast("I"), TypeError),
        (bytearray(2), ValueError),
    ):
        with pytest.raises(error):
            sz.translate(source, {}, out=out)
    with pytest.raises(ValueError):
        sz.translate(source, {}, inplace=True, out=bytearray(3))
    with pytest.raises(TypeError):
        sz.translate(source, {}, inplace=True, unexpected=True)
    with pytest.raises(TypeError):
        Str(source).translate({}, True, inplace=True)
    assert source == b"abc"
@pytest.mark.skipif(sys.version_info < (3, 12), reason="Python buffer exporters require 3.12")
@pytest.mark.parametrize("kind", ["view", "str", "bytearray"])
def test_translation_out_callback(kind):
    backing = bytearray(b"abc")
    view = memoryview(backing)
    source = Str(backing) if kind == "bytearray" else view
    if kind == "str":
        source = Str(source)
    out = bytearray(b"...")
    class ResizeOnExport:
        def __buffer__(self, flags):
            view.release()
            backing.clear()
            return memoryview(out)
    with pytest.raises(ValueError):
        sz.translate(source, {}, out=ResizeOnExport())
    assert out == b"..."
    out.extend(b"!")
@pytest.mark.parametrize("length", list(range(0, 300)) + [1024, 4096, 100000])
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_translations_random(length: int, seed_value: int):
    seed_random_generators(seed_value)
    body = get_random_string(length=length)
    lut = np.random.randint(0, 256, size=256, dtype=np.uint8)
    assert sz.translate(body, memoryview(lut)) == baseline_translate(body, lut)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_fill_random_slice(seed_value: int):
    original = bytearray(64)
    updated_in_slices = bytearray(original)
    start, end = 10, 30
    sz.fill_random(updated_in_slices, nonce=seed_value, start=start, end=end)
    assert bytes(updated_in_slices[:start]) == bytes(original[:start])
    assert bytes(updated_in_slices[end:]) == bytes(original[end:])
    assert bytes(updated_in_slices[start:end]) != bytes(original[start:end])
def test_fill_random_different_nonces():
    first_buffer = bytearray(64)
    second_buffer = bytearray(64)
    sz.fill_random(first_buffer, nonce=1)
    sz.fill_random(second_buffer, nonce=2)
    assert bytes(first_buffer) != bytes(second_buffer)
@pytest.mark.parametrize("length", [0, 1, 7, 64])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_fill_random_alphabet(length: int, seed_value: int):
    random_string = sz.random(length, nonce=seed_value)
    same_nonce_random_string = sz.random(length, nonce=seed_value)
    assert isinstance(random_string, (bytes, bytearray))
    assert len(random_string) == length
    assert random_string == same_nonce_random_string
    alphabet = b"0123456789"
    random_digits = sz.random(128, nonce=seed_value, alphabet=alphabet)
    assert set(random_digits).issubset(set(alphabet))
@pytest.mark.parametrize(
    "byte_string, encoding, expected",
    [
        (b"hello world", "utf-8", "hello world"),
        (b"\xf0\x9f\x98\x81", "utf-8", "😁"),
        (b"hello world", "ascii", "hello world"),
        (b"\xf0hello world", "latin-1", "ðhello world"),
        (b"", "utf-8", ""),
    ],
)
def test_decoding_valid_strings(byte_string, encoding, expected):
    assert byte_string.decode(encoding) == expected
    assert sz.Str(byte_string).decode(encoding) == expected
DEGENERATE_WINDOWS = [(3, 1), (100, 200), (-1, -3), (2, 2)]
OUT_OF_SSIZE_T_INDICES = [2**63, 2**64, -(2**63) - 1]
OUT_OF_RANGE_SEEDS = [-1, 2**70]
STARTSWITH_ENDSWITH_BOUNDS = [-10, -2, 0, 2, 5, 6, 10**7, -(10**7)]
TRANSLATE_DEGENERATE_STARTS = [0, 3, 100, -3, 10**6]
@pytest.mark.parametrize("start, stop", DEGENERATE_WINDOWS)
def test_unit_slicing_degenerate(start, stop):
    native = "abcdef"
    big = Str(native)
    assert big[start:stop] == native[start:stop]
@pytest.mark.parametrize("container", [Str("hello"), Str("hello").split("l")], ids=["str", "strs"])
@pytest.mark.parametrize("bad_index", OUT_OF_SSIZE_T_INDICES)
def test_unit_index_out_of_ssize_t(container, bad_index):
    with pytest.raises((IndexError, OverflowError)):
        container[bad_index]
@pytest.mark.parametrize("haystack", DEGENERATE_HAYSTACKS)
def test_unit_contains_empty_needle(haystack):
    assert ("" in Str(haystack)) == ("" in haystack) == True
@pytest.mark.parametrize("start", STARTSWITH_ENDSWITH_BOUNDS)
def test_unit_startswith_endswith_degenerate(start):
    haystack = "hello"
    assert sz.startswith(haystack, "lo", start) == haystack.startswith("lo", start)
    assert sz.endswith(haystack, "lo", start) == haystack.endswith("lo", start)
@pytest.mark.parametrize(
    "sz_func, native_method, haystack, needle",
    [
        (sz.startswith, "startswith", b"\x00x", b"\x00y"),
        (sz.endswith, "endswith", b"ab\x00cd", b"\x00ZZ"),
    ],
    ids=["startswith-embedded_nul", "endswith-embedded_nul"],
)
def test_unit_startswith_endswith_binary_safe(sz_func, native_method, haystack, needle):
    assert sz_func(haystack, needle) == getattr(haystack, native_method)(needle) == False
@pytest.mark.parametrize("start", TRANSLATE_DEGENERATE_STARTS)
def test_unit_translate_degenerate_window(start):
    table = {"a": "A"}
    mutable = bytearray(b"abcabc")
    sz.translate(mutable, table, True, start)
    assert len(mutable) == 6
    sz.translate("abcdef", table, False, start)
@pytest.mark.parametrize(
    "operation, expected",
    [
        (lambda strs: list(strs.shuffled()), []),
        (lambda strs: list(strs.sorted()), []),
        (lambda strs: tuple(strs.argsort()), ()),
        (lambda strs: list(strs.sample(5)), []),
    ],
    ids=["shuffled", "sorted", "argsort", "sample"],
)
def test_unit_strs_reorder_empty(operation, expected):
    empty = Str("").split(",", skip_empty=True)
    assert len(empty) == 0
    assert operation(empty) == expected
def test_unit_strs_sample_zero():
    assert list(Str("a,b,c").split(",").sample(0)) == []
@pytest.mark.parametrize(
    "needle, haystack",
    [
        (Str("zzz"), Str("hello world")),
        (Str(""), Str("hello world")),
        (Str("hello world"), Str("zzz")),
    ],
    ids=["disjoint", "empty_needle_separate_allocation", "haystack_shorter_than_needle"],
)
def test_unit_offset_within_not_contained(needle, haystack):
    assert needle.offset_within(haystack) == -1
@pytest.mark.parametrize(
    "path, expected_exception",
    [
        (os.path.join("no", "such", "dir", "f.bin"), OSError),
        ("with\x00null", ValueError),
    ],
    ids=["missing_directory", "embedded_nul_in_path"],
)
def test_unit_write_to_errors_are_catchable(path, expected_exception):
    with pytest.raises(expected_exception):
        Str("payload").write_to(path)
@pytest.mark.parametrize("seed_value", OUT_OF_RANGE_SEEDS)
def test_unit_strs_sample_bad_seed_is_catchable(seed_value):
    strs = Str("a,b,c").split(",")
    with pytest.raises((OverflowError, ValueError, TypeError)):
        strs.sample(2, seed=seed_value)
@pytest.mark.parametrize("seed_value", OUT_OF_RANGE_SEEDS)
def test_unit_strs_shuffled_bad_seed_is_catchable(seed_value):
    strs = Str("a,b,c").split(",")
    with pytest.raises((OverflowError, ValueError, TypeError)):
        strs.shuffled(seed=seed_value)
def test_unit_str_reinit_rebinds():
    s = Str("abc")
    s.__init__("xyz")
    assert str(s) == "xyz"
def test_unit_strs_reinit_rebinds():
    ss = Str("a,b").split(",")
    ss.__init__(("p", "q"))
    assert [str(x) for x in ss] == ["p", "q"]
def _file_mapping_available() -> bool:
    import gc
    with tempfile.NamedTemporaryFile(delete=False) as probe_file:
        probe_file.write(b"probe")
        probe_filename = probe_file.name
    mapped = None
    try:
        mapped = sz.File(probe_filename)
        return True
    except Exception:
        return False
    finally:
        mapped = None
        gc.collect()
        os.remove(probe_filename)
file_mapping_available = _file_mapping_available()
@pytest.mark.skipif(not file_mapping_available, reason="Platform maps no files")
def test_unit_file_reinit_rebinds():
    import gc
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(b"hello reinit")
        temp_filename = tmpfile.name
    file = None
    try:
        file = sz.File(temp_filename)
        file.__init__(temp_filename)
        file.__init__(temp_filename)
        assert bytes(sz.Str(file)) == b"hello reinit"
    finally:
        file = None
        gc.collect()
        os.remove(temp_filename)
@pytest.mark.parametrize(
    "byte_string, encoding",
    [
        (bytes.fromhex("ff"), "utf-8"),
        (bytes.fromhex("80") + b"hello", "ascii"),
    ],
)
def test_decoding_exceptions(byte_string, encoding):
    with pytest.raises(UnicodeDecodeError):
        byte_string.decode(encoding)
    with pytest.raises(UnicodeDecodeError):
        sz.Str(byte_string).decode(encoding)
def test_invalid_utf8_handling():
    test_arrays = [
        [b"hello", bytes.fromhex("80") + b"world", b"valid"],
        [bytes.fromhex("fffe"), bytes.fromhex("80"), bytes.fromhex("f4908080")],
        [b"normal", b"string with " + bytes.fromhex("80") + b" bytes"],
    ]
    for test_array in test_arrays:
        strs_obj = Strs(test_array)
        repr_result = repr(strs_obj)
        str_result = str(strs_obj)
        assert isinstance(repr_result, str)
        assert isinstance(str_result, str)
        assert len(repr_result) > 0
        assert len(str_result) > 0
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_str_to_pyarrow_conversion():
    native = "hello"
    big = Str(native)
    assert isinstance(big.address, int) and big.address != 0
    assert isinstance(big.nbytes, int) and big.nbytes == len(native)
    arrow_buffer = pa.foreign_buffer(big.address, big.nbytes, big)
    assert arrow_buffer.to_pybytes() == native.encode("utf-8")
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_to_pyarrow_conversion():
    native_list = ["hello", "world", "test", "", "python"]
    strs = Strs(native_list)
    assert isinstance(strs.tape_address, int) and strs.tape_address != 0, "tape_address should be non-zero"
    assert isinstance(strs.offsets_address, int) and strs.offsets_address != 0, "offsets_address should be non-zero"
    assert isinstance(strs.tape_nbytes, int) and strs.tape_nbytes >= 0, "tape_nbytes should be non-negative"
    assert isinstance(strs.offsets_nbytes, int) and strs.offsets_nbytes > 0, "offsets_nbytes should be positive"
    assert isinstance(strs.offsets_are_large, bool), "offsets_are_large should be a boolean"
    expected_tape_nbytes = sum(len(s) for s in native_list)
    assert (
        strs.tape_nbytes == expected_tape_nbytes
    ), f"Expected tape_nbytes={expected_tape_nbytes}, got {strs.tape_nbytes}"
    expected_offsets_count = len(native_list) + 1
    if strs.offsets_are_large:
        expected_offsets_nbytes = expected_offsets_count * 8
    else:
        expected_offsets_nbytes = expected_offsets_count * 4
    assert (
        strs.offsets_nbytes == expected_offsets_nbytes
    ), f"Expected offsets_nbytes={expected_offsets_nbytes}, got {strs.offsets_nbytes}"
    tape_buffer = pa.foreign_buffer(strs.tape_address, strs.tape_nbytes, strs)
    offsets_buffer = pa.foreign_buffer(strs.offsets_address, strs.offsets_nbytes, strs)
    concatenated = "".join(native_list)
    assert tape_buffer.to_pybytes() == concatenated.encode("utf-8"), "Tape should contain concatenated strings"
    if strs.offsets_are_large:
        arrow_array = pa.Array.from_buffers(pa.large_string(), len(native_list), [None, offsets_buffer, tape_buffer])
    else:
        arrow_array = pa.Array.from_buffers(pa.string(), len(native_list), [None, offsets_buffer, tape_buffer])
    assert arrow_array.to_pylist() == native_list, "Arrow array should match original list"
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_pyarrow_empty():
    strs = Strs([])
    tape_addr = strs.tape_address
    assert tape_addr is not None and tape_addr >= 0
    offsets_addr = strs.offsets_address
    assert offsets_addr is not None and offsets_addr >= 0
    tape_bytes = strs.tape_nbytes
    assert tape_bytes == 0
    offsets_bytes = strs.offsets_nbytes
    assert offsets_bytes == 4
    are_large = strs.offsets_are_large
    assert are_large == False
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_pyarrow_large_strings():
    large_string = "x" * 10000
    native_list = [large_string, "small", large_string, ""]
    strs = Strs(native_list)
    assert isinstance(strs.offsets_are_large, bool), "offsets_are_large should be boolean"
    assert strs.tape_nbytes == sum(len(s) for s in native_list), "tape_nbytes should match total length"
    expected_offsets_count = len(native_list) + 1
    offset_size = 8 if strs.offsets_are_large else 4
    assert strs.offsets_nbytes == expected_offsets_count * offset_size, "offsets_nbytes should be correct"
    tape_buffer = pa.foreign_buffer(strs.tape_address, strs.tape_nbytes, strs)
    offsets_buffer = pa.foreign_buffer(strs.offsets_address, strs.offsets_nbytes, strs)
    if strs.offsets_are_large:
        arrow_array = pa.Array.from_buffers(pa.large_string(), len(native_list), [None, offsets_buffer, tape_buffer])
    else:
        arrow_array = pa.Array.from_buffers(pa.string(), len(native_list), [None, offsets_buffer, tape_buffer])
    assert arrow_array.to_pylist() == native_list, "Arrow array should match original list"
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_pyarrow_fragmented_conversion():
    text = Str("apple banana cherry date")
    tape_strs = text.split(" ")
    fragmented_strs = tape_strs.shuffled(seed=42)
    tape_addr = fragmented_strs.tape_address
    assert tape_addr > 0, "tape_address should be valid"
    offsets_addr = fragmented_strs.offsets_address
    assert offsets_addr > 0, "offsets_address should be valid"
    tape_bytes = fragmented_strs.tape_nbytes
    expected_bytes = sum(len(str(fragmented_strs[i])) for i in range(len(fragmented_strs)))
    assert tape_bytes == expected_bytes, f"tape_nbytes should be {expected_bytes}, got {tape_bytes}"
    offsets_bytes = fragmented_strs.offsets_nbytes
    expected_offsets = (len(fragmented_strs) + 1) * 4
    assert offsets_bytes == expected_offsets, f"offsets_nbytes should be {expected_offsets}, got {offsets_bytes}"
    are_large = fragmented_strs.offsets_are_large
    assert are_large == False, "Small strings should use u32 offsets"
    tape_buffer = pa.foreign_buffer(fragmented_strs.tape_address, fragmented_strs.tape_nbytes, fragmented_strs)
    offsets_buffer = pa.foreign_buffer(fragmented_strs.offsets_address, fragmented_strs.offsets_nbytes, fragmented_strs)
    arrow_array = pa.Array.from_buffers(pa.string(), len(fragmented_strs), [None, offsets_buffer, tape_buffer])
    arrow_list = arrow_array.to_pylist()
    original_list = [str(fragmented_strs[i]) for i in range(len(fragmented_strs))]
    assert arrow_list == original_list, "Arrow array should match shuffled strings"
@pytest.mark.parametrize("container_class", [tuple, list, iter])
@pytest.mark.parametrize("view", [False, True])
def test_strs_from_python_basic(container_class: type, view: bool):
    base_items = ["hello", "world", "test", " ", "from", " ", "container", ""]
    container = container_class(base_items)
    if container_class == iter and view:
        with pytest.raises(ValueError, match="View mode.*not supported for iterators"):
            Strs(container, view=view)
        return
    strs = Strs(container, view=view)
    assert len(strs) == len(base_items)
    assert strs[0] == "hello"
    assert strs[1] == "world"
    assert strs[2] == "test"
    assert strs[3] == " "
    assert strs[4] == "from"
    assert strs[5] == " "
    assert strs[6] == "container"
    assert strs[7] == ""
@pytest.mark.skipif(sys.maxsize <= 2**32, reason="64-bit system required for 4GB+ test")
def test_strs_from_4gb_list():
    try:
        import gc
        import ctypes
    except ImportError:
        pytest.skip("ctypes & gc not available (e.g., PyPy)")
    part_size = 64 * 1024 * 1024
    parts_count = math.ceil(UINT32_MAX / part_size) + 1
    try:
        parts_pythonic = [
            long_repeated_string(ctypes, ascii_lowercase[part_index % len(ascii_lowercase)], part_size)
            for part_index in range(parts_count)
        ]
        parts_stringzilla = Strs(parts_pythonic)
        last_used_char = ascii_lowercase[(parts_count - 1) % len(ascii_lowercase)]
        assert len(parts_stringzilla) == parts_count
        assert parts_stringzilla[0] == long_repeated_string(ctypes, "a", part_size)
        assert parts_stringzilla[-1] == long_repeated_string(ctypes, last_used_char, part_size)
        del parts_pythonic
        del parts_stringzilla
    except (MemoryError, OSError):
        pytest.skip("Memory allocation failed")
    finally:
        gc.collect()
@pytest.mark.skipif(sys.maxsize <= 2**32, reason="64-bit system required for 4GB+ test")
def test_strs_from_4gb_generator():
    try:
        import gc
        import ctypes
    except ImportError:
        pytest.skip("ctypes & gc not available (e.g., PyPy)")
    part_size = 64 * 1024 * 1024
    parts_count = math.ceil(UINT32_MAX / part_size) + 1
    try:
        parts_stringzilla = Strs(
            long_repeated_string(ctypes, ascii_lowercase[part_index % len(ascii_lowercase)], part_size)
            for part_index in range(parts_count)
        )
        last_used_char = ascii_lowercase[(parts_count - 1) % len(ascii_lowercase)]
        assert len(parts_stringzilla) == parts_count
        assert parts_stringzilla[0] == long_repeated_string(ctypes, "a", part_size)
        assert parts_stringzilla[-1] == long_repeated_string(ctypes, last_used_char, part_size)
        del parts_stringzilla
    except (MemoryError, OSError):
        pytest.skip("Memory allocation failed")
    finally:
        gc.collect()
@pytest.mark.parametrize("container_class", [tuple, list, iter])
@pytest.mark.parametrize("view", [False, True])
def test_strs_reference_counting(container_class: type, view: bool):
    if not hasattr(sys, "getrefcount"):
        pytest.skip("Reference counting semantics are not available")
    import gc
    base_items = [word.encode().decode() for word in ("ref", "count", "test")]
    container = container_class(base_items)
    if container_class == iter and view:
        with pytest.raises(ValueError, match="View mode.*not supported for iterators"):
            Strs(container, view=view)
        return
    initial_refcount = sys.getrefcount(container)
    strs = Strs(container, view=view)
    during_refcount = sys.getrefcount(container)
    if container_class != iter:
        if not view:
            assert during_refcount == initial_refcount, "Copy mode should not change refcount"
        elif container_class == tuple:
            assert during_refcount == initial_refcount + 1, "A tuple is pinned directly"
        else:
            assert during_refcount == initial_refcount, "A list is pinned through its snapshot, not directly"
    assert len(strs) == 3
    assert strs[0] == "ref"
    assert strs[1] == "count"
    assert strs[2] == "test"
    del strs
    gc.collect()
    if container_class != iter:
        final_refcount = sys.getrefcount(container)
        assert final_refcount == initial_refcount, "Refcount should return to initial value"
@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("mutation", [list.clear, list.reverse, lambda source: source.__delitem__(slice(1, None))])
def test_strs_survives_source_mutation(view: bool, mutation):
    expected = ["prefix-" + "x" * 200 + f"-{index}" for index in range(8)]
    source = ["prefix-" + "x" * 200 + f"-{index}" for index in range(8)]
    strs = Strs(source, view=view)
    mutation(source)
    churn = [("z" * 208).encode() for _ in range(scale_iterations(5000))]
    assert [str(part) for part in strs] == expected, f"spans went stale across {len(churn)} reuses"
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
@pytest.mark.parametrize("view", [False, True])
def test_strs_from_arrow_basic(view: bool):
    arrow_array = pa.array(["hello", "world", "test", "arrow"])
    strs = Strs(arrow_array, view=view)
    assert len(strs) == 4
    assert strs[0] == "hello"
    assert strs[1] == "world"
    assert strs[2] == "test"
    assert strs[3] == "arrow"
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_from_arrow_binary_array():
    binary_data = [b"hello", b"world", b"binary", b"data"]
    arrow_array = pa.array(binary_data, type=pa.binary())
    strs = Strs(arrow_array)
    assert len(strs) == 4
    for i, expected in enumerate(binary_data):
        str_bytes = strs[i].encode("latin-1") if isinstance(strs[i], str) else bytes(strs[i])
        assert str_bytes == expected
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_from_arrow_large_strings():
    arrow_array = pa.array(["hello", "world", "large", "strings"], type=pa.large_string())
    strs = Strs(arrow_array)
    assert len(strs) == 4
    assert strs[0] == "hello"
    assert strs[1] == "world"
    assert strs[2] == "large"
    assert strs[3] == "strings"
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_from_arrow_error_cases():
    with pytest.raises(TypeError):
        Strs(123)
    with pytest.raises(TypeError):
        Strs(None)
    int_array = pa.array([1, 2, 3, 4])
    with pytest.raises((TypeError, ValueError)):
        Strs(int_array)
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_from_arrow_c_interface():
    arrow_array = pa.array(["test", "c", "interface"])
    assert hasattr(arrow_array, "__arrow_c_array__")
    schema_capsule, array_capsule = arrow_array.__arrow_c_array__()
    if sys.version_info >= (3, 1):
        assert str(type(schema_capsule)) == "<class 'PyCapsule'>"
        assert str(type(array_capsule)) == "<class 'PyCapsule'>"
    strs = Strs(arrow_array)
    assert len(strs) == 3
    assert strs[0] == "test"
    assert strs[1] == "c"
    assert strs[2] == "interface"
@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_from_arrow_with_nulls():
    arrow_array = pa.array(["hello", None, "world", None, "test"])
    strs = Strs(arrow_array)
    assert len(strs) == 5
    assert strs[0] == "hello"
    assert strs[1] == ""
    assert strs[2] == "world"
    assert strs[3] == ""
    assert strs[4] == "test"
    all_nulls = pa.array([None, None, None], type=pa.string())
    strs_nulls = Strs(all_nulls)
    assert len(strs_nulls) == 3
    assert all(s == "" for s in strs_nulls)
    mixed = pa.array(["", None, "hello", "", None, "world"])
    strs_mixed = Strs(mixed)
    assert len(strs_mixed) == 6
    assert strs_mixed[0] == ""
    assert strs_mixed[1] == ""
    assert strs_mixed[2] == "hello"
    assert strs_mixed[3] == ""
    assert strs_mixed[4] == ""
    assert strs_mixed[5] == "world"
def baseline_translate(body: str, lut: Sequence) -> str:
    return "".join([chr(lut[ord(c)]) for c in body])
def long_repeated_string(ctypes, fill_char: str, string_size: int) -> str:
    buffer = ctypes.create_string_buffer(string_size)
    ctypes.memset(buffer, ord(fill_char), string_size)
    return buffer.value.decode("ascii")
def safe_windows(length: int) -> Sequence[Sequence[int]]:
    windows = {
        (0, length),
        (0, 0),
        (length, length),
        (0, length // 2),
        (length // 2, length),
        (length // 3, (2 * length) // 3),
        (1, length - 1) if length >= 2 else (0, length),
    }
    return sorted(windows)
def lookup_table_oracle(data: bytes, table: bytes) -> bytes:
    return bytes(table[byte] for byte in data)
UINT32_MAX = 2**32 - 1
_TRANSLATE_IDENTITY_TABLE = bytes(range(256))
_TRANSLATE_INVERT_TABLE = bytes(255 - value for value in range(256))
_TRANSLATE_SCRAMBLE_TABLE = bytes((value * 167 + 41) % 256 for value in range(256))
TRANSLATE_TABLES = (_TRANSLATE_IDENTITY_TABLE, _TRANSLATE_INVERT_TABLE, _TRANSLATE_SCRAMBLE_TABLE)
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_backend_differential_equal(length):
    for body in differential_bodies(length, length):
        matching = body
        mismatching = body[:-1] + ("z" if body[-1:] != "z" else "y") if length > 0 else "nonempty"
        assert_backends_agree(run_across_backends(lambda body=body: Str(body) == matching), oracle=True)
        assert_backends_agree(run_across_backends(lambda body=body: Str(body) == mismatching), oracle=False)
        assert_backends_agree(run_across_backends(lambda body=body: sz.equal(body, matching)), oracle=True)
        assert_backends_agree(run_across_backends(lambda body=body: sz.equal(body, mismatching)), oracle=False)
def test_unit_equal_compares_both_operands():
    assert sz.equal("abc", "abc") is True
    assert sz.equal("abc", "xyz") is False
    assert sz.equal("abc", "ab") is False
    assert sz.equal(b"\x00\x01", b"\x00\x01") is True
    assert sz.equal(b"\x00\x01", b"\x02\x03") is False
    assert sz.equal("", "") is True
    assert sz.equal("", "a") is False
@pytest.mark.parametrize("offset", [0, 1, 3, 7, 15])
def test_unit_backend_differential_equal_unaligned(offset):
    text = "ab" * 200
    for _offset, view in unaligned_views(text, offsets=(offset,)):
        matching = text[offset:]
        mismatching = "z" + text[offset + 1 :] if len(matching) > 0 else "nonempty"
        assert_backends_agree(run_across_backends(lambda view=view, matching=matching: view == matching), oracle=True)
        assert_backends_agree(
            run_across_backends(lambda view=view, mismatching=mismatching: view == mismatching), oracle=False
        )
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_backend_differential_contains(length):
    for needle in differential_bodies(length, length):
        haystack = f"head_{needle}_tail"
        absent_needle = needle + "\x01"
        assert_backends_agree(
            run_across_backends(lambda needle=needle: needle in Str(haystack)), oracle=needle in haystack
        )
        assert_backends_agree(
            run_across_backends(lambda absent_needle=absent_needle: absent_needle in Str(haystack)),
            oracle=absent_needle in haystack,
        )
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_backend_differential_copy_slice(length):
    for body in differential_bodies(length, length):
        body_bytes = body.encode()
        assert_backends_agree(run_across_backends(lambda body=body: bytes(Str(body))), oracle=body_bytes)
        for start, stop in safe_windows(length):
            oracle = body_bytes[start:stop]
            assert_backends_agree(
                run_across_backends(lambda body=body, start=start, stop=stop: bytes(Str(body)[start:stop])),
                oracle=oracle,
            )
@pytest.mark.parametrize("offset", [0, 1, 3, 7, 15])
def test_unit_backend_differential_copy_unaligned(offset):
    text = "xy" * 200
    for _offset, view in unaligned_views(text, offsets=(offset,)):
        oracle = text[offset:].encode()
        assert_backends_agree(run_across_backends(lambda view=view: bytes(view)), oracle=oracle)
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_backend_differential_concat(length):
    seed_random_generators(length + 1)
    right = get_random_string(length=37)
    for left in differential_bodies(length, length):
        oracle = (left + right).encode()
        assert_backends_agree(run_across_backends(lambda left=left: bytes(Str(left) + Str(right))), oracle=oracle)
        assert_backends_agree(run_across_backends(lambda left=left: bytes(Str(left) + right)), oracle=oracle)
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_backend_differential_translate(length):
    for body in differential_bodies(length, length):
        body_bytes = body.encode()
        for table in TRANSLATE_TABLES:
            full_oracle = lookup_table_oracle(body_bytes, table)
            assert_backends_agree(
                run_across_backends(lambda body_bytes=body_bytes, table=table: sz.translate(body_bytes, table)),
                oracle=full_oracle,
            )
            for start, stop in safe_windows(length):
                windowed_slice_oracle = lookup_table_oracle(body_bytes[start:stop], table)
                windowed_full_oracle = bytearray(body_bytes)
                windowed_full_oracle[start:stop] = windowed_slice_oracle
                def windowed_copy(body_bytes=body_bytes, table=table, start=start, stop=stop):
                    return sz.translate(body_bytes, table, False, start, stop)
                assert_backends_agree(run_across_backends(windowed_copy), oracle=windowed_slice_oracle)
                def windowed_inplace(body_bytes=body_bytes, table=table, start=start, stop=stop):
                    mutable = bytearray(body_bytes)
                    sz.translate(mutable, table, True, start, stop)
                    return bytes(mutable)
                assert_backends_agree(run_across_backends(windowed_inplace), oracle=bytes(windowed_full_oracle))
@pytest.mark.parametrize("offset", [0, 1, 3, 7, 15])
def test_unit_backend_differential_translate_unaligned(offset):
    text = "ab" * 200
    for _offset, view in unaligned_views(text, offsets=(offset,)):
        oracle = lookup_table_oracle(str(view).encode(), _TRANSLATE_INVERT_TABLE)
        assert_backends_agree(
            run_across_backends(lambda view=view: sz.translate(view, _TRANSLATE_INVERT_TABLE)), oracle=oracle
        )
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
@pytest.mark.parametrize("nonce", [0, 1, 42, 314159])
def test_unit_backend_differential_fill_random(length, nonce):
    def fill():
        buffer = bytearray(length)
        sz.fill_random(buffer, nonce=nonce)
        return bytes(buffer)
    assert_backends_agree(run_across_backends(fill))
@pytest.mark.parametrize("start, end", [(1, 9), (3, 7), (0, 1), (15, 16), (7, 64)])
def test_unit_backend_differential_fill_random_windowed(start, end):
    buffer_length = 96
    nonce = 2026
    def fill():
        buffer = bytearray(buffer_length)
        sz.fill_random(buffer, nonce=nonce, start=start, end=end)
        return bytes(buffer)
    assert_backends_agree(run_across_backends(fill))
