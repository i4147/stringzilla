from random import randint
from typing import Optional
import pytest
import stringzilla as sz
from stringzilla import Str, Strs
from test.sz_helpers import (
    SEED_VALUES,
    DEGENERATE_HAYSTACKS,
    DEGENERATE_BOUNDS,
    UNCASED_DEGENERATE_BOUNDS,
    seed_random_generators,
    get_random_string,
    is_equal_strings,
    run_across_backends,
    assert_backends_agree,
    VECTOR_WIDTH_LENGTHS,
    boundary_strings,
    unaligned_views,
)
FIND_BYTESET_EMPTY_SET_CASES = [
    pytest.param(sz.find_first_of, ("abc", ""), -1, id="first_of-empty_set"),
    pytest.param(sz.find_last_of, ("abc", ""), -1, id="last_of-empty_set"),
    pytest.param(sz.find_first_not_of, ("abc", ""), 0, id="first_not_of-empty_set"),
    pytest.param(sz.find_last_not_of, ("abc", ""), 2, id="last_not_of-empty_set"),
    pytest.param(sz.find_first_not_of, ("abc", "", 0, 0), -1, id="first_not_of-empty_window"),
    pytest.param(sz.find_last_of, ("abc", "", 1, 2), -1, id="last_of-empty_window"),
]
def test_unit_count():
    native = "aaaaa"
    big = Str(native)
    assert big.count("a") == 5
    assert big.count("aa") == 2
    assert big.count("aa", allowoverlap=True) == 4
def test_unit_count_byteset():
    native = "abcdef"
    big = Str(native)
    assert big.count_byteset("abc") == 3
    assert big.count_byteset("xyz") == 0
    assert big.count_byteset("aeiou") == 2
    assert sz.count_byteset("", "abc") == 0
    assert sz.count_byteset("abc", "") == 0
    assert sz.count_byteset("", "") == 0
    assert sz.count_byteset("hello", "l") == 2
    assert sz.count_byteset("hello", "x") == 0
    assert sz.count_byteset("mississippi", "si") == 8
    assert sz.count_byteset("aaaaaa", "a") == 6
    native = "abcdefghij"
    big = Str(native)
    assert big.count_byteset("abc", 0, 3) == 3
    assert big.count_byteset("abc", 1, 3) == 2
    assert big.count_byteset("abc", 3) == 0
    assert big.count_byteset("hij", 7) == 3
    assert big.count_byteset("hij", -3) == 3
    assert big.count_byteset("abc", 0, -7) == 3
    assert sz.count_byteset("a", "a", 0, 0) == 0
    assert sz.count_byteset("abc", "abc", 10, 20) == 0
    assert sz.count_byteset("abc", "abc", -10, -5) == 0
def test_unit_contains():
    big = Str("abcdef")
    assert "a" in big
    assert "ab" in big
    assert "xxx" not in big
@pytest.mark.parametrize("func, args, expected", FIND_BYTESET_EMPTY_SET_CASES)
def test_unit_find_byteset_empty_set(func, args, expected):
    assert func(*args) == expected
@pytest.mark.parametrize("func", [sz.find_first_not_of, sz.find_last_not_of], ids=["first_not_of", "last_not_of"])
def test_unit_find_byteset_empty_set_index_is_subscriptable(func):
    haystack = "abc"
    idx = func(haystack, "")
    assert idx == -1 or haystack[idx]
@pytest.mark.parametrize(
    "sz_func, native_method, args",
    [
        (sz.find, "find", (0, None)),
        (sz.find, "find", (None,)),
        (sz.find, "find", (2**63,)),
        (sz.find, "find", (0, 2**63)),
        (sz.rfind, "rfind", (None, None)),
        (sz.count, "count", (0, None)),
    ],
    ids=["find-start_none", "find-none", "find-overflow_start", "find-overflow_end", "rfind-none_none", "count-none"],
)
def test_unit_index_args_none_and_overflow(sz_func, native_method, args):
    haystack = "banana"
    assert sz_func(haystack, "a", *args) == getattr(haystack, native_method)("a", *args)
def test_unit_globals():
    assert sz.find("abcdef", "bcdef") == 1
    assert sz.find("abcdef", "x") == -1
    assert sz.rfind("abcdef", "bcdef") == 1
    assert sz.rfind("abcdef", "x") == -1
    assert sz.find("abcdef", "") == "abcdef".find("")
    assert sz.rfind("abcdef", "") == "abcdef".rfind("")
    assert sz.find("abcdef", "", 1) == "abcdef".find("", 1)
    assert sz.rfind("abcdef", "", 1) == "abcdef".rfind("", 1)
    assert sz.find("abcdef", "", 1, 3) == "abcdef".find("", 1, 3)
    assert sz.rfind("abcdef", "", 1, 3) == "abcdef".rfind("", 1, 3)
    assert sz.find("", "abcdef") == "".find("abcdef")
    assert sz.rfind("", "abcdef") == "".rfind("abcdef")
    assert sz.find("Hello, world!", "world", 0, 11) == "Hello, world!".find("world", 0, 11)
    assert sz.rfind("Hello, world!", "world", 0, 11) == "Hello, world!".rfind("world", 0, 11)
    assert sz.find_first_of("abcdef", "cde") == 2
    assert sz.find_first_of("abcdef", "xyz") == -1
    assert sz.find_first_of("hello world", "aeiou") == 1
    assert sz.find_last_of("abcdef", "abc") == 2
    assert sz.find_last_of("abcdef", "xyz") == -1
    assert sz.find_last_of("hello world", "aeiou") == 7
    assert sz.find_first_not_of("aaabbbccc", "ab") == 6
    assert sz.find_first_not_of("abcdef", "abcdef") == -1
    assert sz.find_first_not_of("   hello", " ") == 3
    assert sz.find_last_not_of("aaabbbccc", "bc") == 2
    assert sz.find_last_not_of("abcdef", "abcdef") == -1
    assert sz.find_last_not_of("hello   ", " ") == 4
    assert sz.count_byteset("abcdef", "abc") == 3
    assert sz.count_byteset("abcdef", "xyz") == 0
    assert sz.count_byteset("hello world", "aeiou") == 3
    assert sz.count_byteset("mississippi", "si") == 8
    assert sz.count_byteset("", "abc") == 0
    assert sz.count_byteset("abc", "") == 0
    assert sz.count_byteset("abcdef", "abc", 1, 4) == 2
    assert sz.count_byteset("hello world", "aeiou", 2) == 2
    assert sz.count_byteset("hello world", "aeiou", 0, 5) == 2
    assert sz.partition("abcdef", "c") == ("ab", "c", "def")
    assert sz.rpartition("abcdef", "c") == ("ab", "c", "def")
    assert sz.partition("abcdef", "x") == ("abcdef", "", "")
    assert sz.rpartition("abcdef", "x") == ("", "", "abcdef")
    assert Str("abcdef").rpartition("x") == ("", "", "abcdef")
    parts = sz.partition(bytearray(b"abc:def"), b":")
    assert parts == (b"abc", b":", b"def")
    with pytest.raises(ValueError):
        sz.partition("abcdef", "")
    with pytest.raises(ValueError):
        sz.rpartition("abcdef", "")
    assert sz.count("abcdef", "x") == 0
    assert sz.count("aaaaa", "a") == 5
    assert sz.count("aaaaa", "aa") == 2
    assert sz.count("aaaaa", "aa", allowoverlap=True) == 4
    assert sz.bytesum("hello") > 0
    assert sz.bytesum("hello") > sz.bytesum("hell")
    assert sz.translate("ABC", {"A": "X", "B": "Y", "C": "Z"}) == "XYZ"
    assert sz.translate("ABC", {"A": "X", "B": "Y"}) == "XYC"
    assert sz.translate("ABC", {"A": "X", "B": "Y"}, start=1, end=-1) == "Y"
    assert sz.translate("ABC", bytes(range(256))) == "ABC"
    with pytest.raises(TypeError):
        sz.translate("ABC", {"A": "X", "B": "Y"}, start=1, end=-1, inplace=True)
    mutable_buffer = bytearray(b"ABC")
    assert sz.fill_random(mutable_buffer) is None
    assert sz.fill_random(mutable_buffer, 42) is None
    assert sz.split("hello world test", " ") == ["hello", "world", "test"]
    assert sz.rsplit("hello world test", " ", 1) == ["hello world", "test"]
def test_slice_of_split():
    def impl(native_str: str):
        native_split = native_str.split()
        text = sz.Str(native_str)
        sz_split = text.split()
        for slice_idx in range(len(native_split)):
            assert str(sz_split[slice_idx]) == native_split[slice_idx]
    native_str = "Weebles wobble before they fall down, don't they?"
    impl(native_str)
    copies = int(len(native_str) / 5e9)
    impl(native_str * copies)
@pytest.mark.parametrize("haystack", DEGENERATE_HAYSTACKS)
@pytest.mark.parametrize("start", DEGENERATE_BOUNDS)
@pytest.mark.parametrize("end", DEGENERATE_BOUNDS)
def test_unit_empty_needle_offset_parity(haystack, start, end):
    assert sz.find(haystack, "", start) == haystack.find("", start)
    assert sz.rfind(haystack, "", start) == haystack.rfind("", start)
    assert sz.find(haystack, "", start, end) == haystack.find("", start, end)
    assert sz.rfind(haystack, "", start, end) == haystack.rfind("", start, end)
@pytest.mark.parametrize("haystack", DEGENERATE_HAYSTACKS)
@pytest.mark.parametrize("window", [None, (1, 2), (10, 20), (-100, -1)], ids=["full", "1:2", "10:20", "-100:-1"])
def test_unit_count_empty_needle(haystack, window):
    if window is None:
        assert sz.count(haystack, "") == haystack.count("")
    else:
        start, end = window
        assert sz.count(haystack, "", start, end) == haystack.count("", start, end)
@pytest.mark.parametrize("haystack", ["", "abc", "Straße"])
@pytest.mark.parametrize("start", UNCASED_DEGENERATE_BOUNDS)
@pytest.mark.parametrize("end", UNCASED_DEGENERATE_BOUNDS)
def test_unit_empty_needle_uncased_search_degenerate(haystack, start, end):
    assert sz.utf8_uncased_search(haystack, "", start) == haystack.find("", start)
    assert sz.utf8_uncased_search(haystack, "", start, end) == haystack.find("", start, end)
@pytest.mark.parametrize(
    "haystack, needle, start, end, expected",
    [
        (b"hello", b"", 2, None, 2),
        (b"hello", b"", 5, None, 5),
        (b"hello", b"", 6, None, -1),
        (b"hello", b"", 2, 1, -1),
    ],
    ids=["start_mid", "start_at_len", "start_past_len", "inverted_window"],
)
def test_unit_empty_needle_uncased_search_degenerate_bytes(haystack, needle, start, end, expected):
    if end is None:
        assert sz.utf8_uncased_search(haystack, needle, start) == expected
    else:
        assert sz.utf8_uncased_search(haystack, needle, start, end) == expected
def check_identical(
    native: str,
    big: Str,
    needle: Optional[str] = None,
    check_iterators: bool = False,
):
    if needle is None:
        part_offset = randint(0, len(native) - 1)
        part_length = randint(1, len(native) - part_offset)
        needle = native[part_offset:part_length]
    present_in_native: bool = needle in native
    present_in_big = needle in big
    assert present_in_native == present_in_big
    assert native.find(needle) == big.find(needle)
    assert native.rfind(needle) == big.rfind(needle)
    assert native.count(needle) == big.count(needle)
    len_half = len(native) // 2
    len_quarter = len(native) // 4
    assert native.find(needle, len_half) == big.find(needle, len_half)
    assert native.find(needle, len_quarter, 3 * len_quarter) == big.find(needle, len_quarter, 3 * len_quarter)
    native_strings = native.split(needle)
    big_strings: Strs = big.split(needle)
    assert len(native_strings) == len(big_strings)
    if check_iterators:
        for i in range(len(native_strings)):
            assert len(native_strings[i]) == len(big_strings[i])
            assert (
                native_strings[i] == big_strings[i]
            ), f"Mismatch between `{native_strings[i]}` and `{str(big_strings[i])}`"
            assert [c for c in native_strings[i]] == [c for c in big_strings[i]]
    is_equal_strings(native_strings, big_strings)
@pytest.mark.parametrize("repetitions", range(1, 10))
def test_fuzzy_repetitions(repetitions: int):
    native = "abcd" * repetitions
    big = Str(native)
    check_identical(native, big, "a", True)
    check_identical(native, big, "ab", True)
    check_identical(native, big, "abc", True)
    check_identical(native, big, "abcd", True)
    check_identical(native, big, "abcde", True)
@pytest.mark.parametrize("pattern_length", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("haystack_length", [n for n in VECTOR_WIDTH_LENGTHS if n <= 65])
@pytest.mark.parametrize("variability", [1, 2, 4, 26])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_fuzzy_substrings(pattern_length: int, haystack_length: int, variability: int, seed_value: int):
    seed_random_generators(seed_value)
    native = get_random_string(variability=variability, length=haystack_length)
    big = Str(native)
    pattern = get_random_string(variability=variability, length=pattern_length)
    assert (pattern in native) == big.contains(
        pattern
    ), f"Failed to check if {pattern} at offset {native.find(pattern)} is present in {native}"
    assert native.find(pattern) == big.find(
        pattern
    ), f"Failed to locate {pattern} at offset {native.find(pattern)} in {native}"
def oracle_find_first_of(haystack: str, chars: str) -> int:
    charset = set(chars)
    for index, char in enumerate(haystack):
        if char in charset:
            return index
    return -1
def oracle_find_last_of(haystack: str, chars: str) -> int:
    charset = set(chars)
    for index in range(len(haystack) - 1, -1, -1):
        if haystack[index] in charset:
            return index
    return -1
def oracle_find_first_not_of(haystack: str, chars: str) -> int:
    charset = set(chars)
    for index, char in enumerate(haystack):
        if char not in charset:
            return index
    return -1
def oracle_find_last_not_of(haystack: str, chars: str) -> int:
    charset = set(chars)
    for index in range(len(haystack) - 1, -1, -1):
        if haystack[index] not in charset:
            return index
    return -1
def oracle_count_byteset(haystack: str, chars: str) -> int:
    charset = set(chars)
    return sum(1 for char in haystack if char in charset)
def assert_find_family_matches_oracles(haystack: str, needle: str):
    assert_backends_agree(run_across_backends(lambda: sz.find(haystack, needle)), oracle=haystack.find(needle))
    assert_backends_agree(run_across_backends(lambda: sz.rfind(haystack, needle)), oracle=haystack.rfind(needle))
    assert_backends_agree(run_across_backends(lambda: sz.count(haystack, needle)), oracle=haystack.count(needle))
    assert_backends_agree(run_across_backends(lambda: needle in Str(haystack)), oracle=needle in haystack)
    assert_backends_agree(
        run_across_backends(lambda: sz.find_first_of(haystack, needle)), oracle=oracle_find_first_of(haystack, needle)
    )
    assert_backends_agree(
        run_across_backends(lambda: sz.find_last_of(haystack, needle)), oracle=oracle_find_last_of(haystack, needle)
    )
    assert_backends_agree(
        run_across_backends(lambda: sz.find_first_not_of(haystack, needle)),
        oracle=oracle_find_first_not_of(haystack, needle),
    )
    assert_backends_agree(
        run_across_backends(lambda: sz.find_last_not_of(haystack, needle)),
        oracle=oracle_find_last_not_of(haystack, needle),
    )
    assert_backends_agree(
        run_across_backends(lambda: sz.count_byteset(haystack, needle)), oracle=oracle_count_byteset(haystack, needle)
    )
def embed_at_offset(filler: str, needle: str, offset: int, total_length: int) -> str:
    return filler * offset + needle + filler * (total_length - offset - len(needle))
def needle_for_strategy(haystack: str, alphabet: str, strategy: str) -> str:
    if strategy == "absent":
        return "Q"
    if strategy == "single_char":
        return haystack[:1]
    if strategy == "periodic_unit":
        return alphabet
    if strategy == "whole_haystack":
        return haystack
    if strategy == "longer_than_haystack":
        return haystack + "Q"
    if strategy == "empty":
        return ""
    raise ValueError(strategy)
LANE_STRADDLE_OFFSETS = list(range(14, 19)) + list(range(30, 35)) + list(range(62, 67))
NEEDLE_LENGTHS = [1, 2, 3, 4, 5]
LANE_HAYSTACK_LENGTH = 80
HAYSTACK_ALPHABETS = ["ab", "a"]
BOUNDARY_HAYSTACKS_BY_ALPHABET = {
    alphabet: dict(zip(VECTOR_WIDTH_LENGTHS, boundary_strings(alphabet))) for alphabet in HAYSTACK_ALPHABETS
}
NEEDLE_STRATEGIES = ["absent", "single_char", "periodic_unit", "whole_haystack", "longer_than_haystack", "empty"]
UNALIGNED_VIEW_NEEDLES = ["", "a", "ab", "ba", "abab", "Q"]
@pytest.mark.parametrize("offset", LANE_STRADDLE_OFFSETS)
@pytest.mark.parametrize("needle_length", NEEDLE_LENGTHS)
def test_unit_backend_differential_find_lane_boundary(needle_length, offset):
    needle = "N" * needle_length
    haystack = embed_at_offset("x", needle, offset, LANE_HAYSTACK_LENGTH)
    assert_find_family_matches_oracles(haystack, needle)
@pytest.mark.parametrize("strategy", NEEDLE_STRATEGIES)
@pytest.mark.parametrize("alphabet", HAYSTACK_ALPHABETS)
@pytest.mark.parametrize("haystack_length", VECTOR_WIDTH_LENGTHS)
def test_unit_backend_differential_boundary_lengths(haystack_length, alphabet, strategy):
    haystack = BOUNDARY_HAYSTACKS_BY_ALPHABET[alphabet][haystack_length]
    needle = needle_for_strategy(haystack, alphabet, strategy)
    assert_find_family_matches_oracles(haystack, needle)
@pytest.mark.parametrize("needle", UNALIGNED_VIEW_NEEDLES)
def test_unit_backend_differential_unaligned_views(needle):
    parent_text = ("ab" * 40)[:77]
    for offset, view in unaligned_views(parent_text):
        native_slice = parent_text[offset:]
        assert_backends_agree(
            run_across_backends(lambda view=view: sz.find(view, needle)), oracle=native_slice.find(needle)
        )
        assert_backends_agree(
            run_across_backends(lambda view=view: sz.rfind(view, needle)), oracle=native_slice.rfind(needle)
        )
        assert_backends_agree(
            run_across_backends(lambda view=view: sz.count(view, needle)), oracle=native_slice.count(needle)
        )
        assert_backends_agree(
            run_across_backends(lambda view=view: needle in view),
            oracle=needle in native_slice,
        )
@pytest.mark.parametrize("needle_length", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("haystack_length", [0, 1, 5, 17, 33, 65])
@pytest.mark.parametrize("variability", [1, 2, 4])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_random_fuzz(seed_value, variability, haystack_length, needle_length):
    seed_random_generators(seed_value)
    haystack = get_random_string(variability=variability, length=haystack_length)
    needle = get_random_string(variability=variability, length=needle_length)
    assert_find_family_matches_oracles(haystack, needle)
