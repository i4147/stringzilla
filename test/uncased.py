from random import randint
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
UNCASED_DEGENERATE_HAYSTACKS = ["abcabc", "Straße"]
UNCASED_DEGENERATE_BOUNDS = [-11, -2, 0, 2, 6, 7]
def test_unit_utf8_uncased_fold():
    assert sz.utf8_uncased_fold("HELLO") == b"hello"
    assert sz.utf8_uncased_fold("Hello World") == b"hello world"
    assert sz.utf8_uncased_fold("") == b""
    assert sz.utf8_uncased_fold("already lowercase") == b"already lowercase"
    assert sz.utf8_uncased_fold("Straße") == b"strasse"
    assert sz.utf8_uncased_fold("GROSSE") == b"grosse"
    assert sz.Str("HELLO").utf8_uncased_fold() == b"hello"
    assert sz.utf8_uncased_fold(b"HELLO") == b"hello"
@pytest.mark.parametrize(
    "haystack, needle, expected",
    [
        ("Hello World", "WORLD", 6),
        ("Hello World", "hello", 0),
        ("Hello World", "xyz", -1),
        ("HELLO", "hello", 0),
        ("hello", "HELLO", 0),
        ("HeLLo WoRLd", "world", 6),
        ("abcdef", "CD", 2),
        ("Über allen Gipfeln", "ÜBER", 0),
        ("ÜBER", "über", 0),
        ("Das schöne Mädchen", "SCHÖNE", 4),
        ("Café au lait", "CAFÉ", 0),
        ("naïve approach", "NAÏVE", 0),
        ("El niño juega", "NIÑO", 3),
        ("Straße", "STRASSE", 0),
        ("STRASSE", "straße", 0),
        ("die Straße", "STRASSE", 4),
        ("groß", "GROSS", 0),
        ("GROSS", "groß", 0),
        ("Fußball", "FUSSBALL", 0),
        ("ПРИВЕТ", "привет", 0),
        ("привет", "ПРИВЕТ", 0),
        ("Москва столица", "МОСКВА", 0),
        ("добрый день", "ДОБРЫЙ", 0),
        ("ΑΒΓΔ", "αβγδ", 0),
        ("αβγδ", "ΑΒΓΔ", 0),
        ("Ελλάδα", "ΕΛΛΆΔΑ", 0),
        ("Hello Мир World", "МИР", 6),
        ("Café МОСКВА", "москва", 5),
        ("hello", "", 0),
        ("", "x", -1),
        ("", "", 0),
        ("a", "A", 0),
        ("A", "a", 0),
    ],
)
def test_utf8_uncased_search(haystack, needle, expected):
    assert sz.utf8_uncased_search(haystack, needle) == expected
def test_utf8_uncased_search_method():
    s = sz.Str("Hello World")
    assert s.utf8_uncased_search("WORLD") == 6
    assert s.utf8_uncased_search("hello") == 0
    assert s.utf8_uncased_search("xyz") == -1
def test_utf8_uncased_search_offsets():
    assert sz.utf8_uncased_search("Hëllo Wörld", "WÖRLD") == 6
    assert sz.utf8_uncased_search("Café", "FÉ") == 2
    assert sz.utf8_uncased_search("Hëllo Wörld".encode(), "WÖRLD".encode()) == 7
    assert sz.utf8_uncased_search("Café".encode(), "FÉ".encode()) == 2
def test_utf8_uncased_search_start_end():
    text = "Hëllo Hëllo"
    assert sz.utf8_uncased_search(text, "HËLLO", 0) == 0
    assert sz.utf8_uncased_search(text, "HËLLO", 1) == 6
    assert sz.utf8_uncased_search(text, "HËLLO", 0, 5) == 0
    assert sz.utf8_uncased_search(text, "HËLLO", 0, 4) == -1
    text_bytes = text.encode()
    assert sz.utf8_uncased_search(text_bytes, "HËLLO".encode(), 0) == 0
    assert sz.utf8_uncased_search(text_bytes, "HËLLO".encode(), 1) == 7
    assert sz.utf8_uncased_search(text_bytes, "HËLLO".encode(), 0, 7) == 0
    assert sz.utf8_uncased_search(text_bytes, "HËLLO".encode(), 0, 5) == -1
def test_utf8_uncased_search_bytes():
    assert sz.utf8_uncased_search(b"Hello World", b"WORLD") == 6
    assert sz.utf8_uncased_search(b"Strasse", b"STRASSE") == 0
    assert sz.utf8_uncased_search("Straße".encode(), b"STRASSE") == 0
    assert sz.utf8_uncased_search("XXStraße".encode(), b"STRASSE") == 2
def test_utf8_uncased_search_slicing():
    text = "Café au lait"
    idx = sz.utf8_uncased_search(text, "AU")
    assert text[idx : idx + 2].lower() == "au"
    text_bytes = text.encode()
    idx = sz.utf8_uncased_search(text_bytes, b"AU")
    assert text_bytes[idx : idx + 2].lower() == b"au"
@pytest.mark.parametrize(
    "haystack, needle, expected_matches",
    [
        ("Hello HELLO hello HeLLo", "hello", ["Hello", "HELLO", "hello", "HeLLo"]),
        ("Hello World", "world", ["World"]),
        ("Hello World", "xyz", []),
        ("Straße STRASSE strasse", "strasse", ["Straße", "STRASSE", "strasse"]),
        ("groß GROSS", "gross", ["groß", "GROSS"]),
        ("ПРИВЕТ привет Привет", "привет", ["ПРИВЕТ", "привет", "Привет"]),
        ("ΑΒΓΔ αβγδ", "αβγδ", ["ΑΒΓΔ", "αβγδ"]),
        ("", "hello", []),
        ("hello", "xyz", []),
    ],
)
def test_utf8_uncased_matches(haystack, needle, expected_matches):
    matches = [str(m) for m in sz.utf8_uncased_matches(haystack, needle)]
    assert matches == expected_matches
    matches_method = [str(m) for m in sz.Str(haystack).utf8_uncased_matches(needle)]
    assert matches_method == expected_matches
def test_utf8_uncased_matches_overlapping():
    assert len(list(sz.utf8_uncased_matches("aaaa", "aa"))) == 2
    assert len(list(sz.utf8_uncased_matches("aaaa", "aa", include_overlapping=True))) == 3
    assert len(list(sz.utf8_uncased_matches(b"Hello HELLO", b"hello"))) == 2
def test_utf8_uncased_order():
    assert sz.utf8_uncased_order("hello", "HELLO") == 0
    assert sz.utf8_uncased_order("HELLO", "hello") == 0
    assert sz.utf8_uncased_order("HeLLo", "hElLO") == 0
    assert sz.utf8_uncased_order("Straße", "STRASSE") == 0
    assert sz.utf8_uncased_order("strasse", "STRAẞE") == 0
    assert sz.utf8_uncased_order("apple", "BANANA") < 0
    assert sz.utf8_uncased_order("APPLE", "banana") < 0
    assert sz.utf8_uncased_order("a", "B") < 0
    assert sz.utf8_uncased_order("ZEBRA", "apple") > 0
    assert sz.utf8_uncased_order("zebra", "APPLE") > 0
    assert sz.utf8_uncased_order("z", "A") > 0
    assert sz.utf8_uncased_order("", "") == 0
    assert sz.utf8_uncased_order("", "a") < 0
    assert sz.utf8_uncased_order("a", "") > 0
    assert sz.utf8_uncased_order("hello", "HELLO WORLD") < 0
    assert sz.utf8_uncased_order("HELLO WORLD", "hello") > 0
    assert sz.utf8_uncased_order("ΑΒΓΔ", "αβγδ") == 0
    assert sz.utf8_uncased_order("αβγ", "ΑΒΓ") == 0
    assert sz.utf8_uncased_order("ПРИВЕТ", "привет") == 0
    s = sz.Str("hello")
    assert s.utf8_uncased_order("HELLO") == 0
    assert sz.utf8_uncased_order(b"hello", b"HELLO") == 0
    assert sz.utf8_uncased_order(b"HELLO", b"hello") == 0
    assert sz.utf8_uncased_order(b"apple", b"BANANA") < 0
    assert sz.utf8_uncased_order(b"ZEBRA", b"apple") > 0
    assert sz.utf8_uncased_order("Straße".encode(), b"STRASSE") == 0
    assert sz.utf8_uncased_order(b"strasse", "Straße".encode()) == 0
    assert sz.utf8_uncased_order("ΑΒΓΔ".encode(), "αβγδ".encode()) == 0
    assert sz.utf8_uncased_order("ПРИВЕТ".encode(), "привет".encode()) == 0
@pytest.mark.parametrize("haystack", UNCASED_DEGENERATE_HAYSTACKS)
@pytest.mark.parametrize("start", UNCASED_DEGENERATE_BOUNDS)
def test_utf8_uncased_search_degenerate_offsets(haystack, start):
    assert sz.utf8_uncased_search(haystack, "", start) >= -1
    expected = haystack.find("bc", start)
    assert sz.utf8_uncased_search(haystack, "BC", start) == expected
@pytest.mark.parametrize(
    "haystack, needle, start, end, expected",
    [
        ("hello", "", 6, None, -1),
        ("hello", "", 2, 1, -1),
        (b"hello", b"", 6, None, -1),
    ],
    ids=["str_start_past_end", "str_inverted_window", "bytes_start_past_end"],
)
def test_utf8_uncased_search_degenerate_offsets_report_negative_one(haystack, needle, start, end, expected):
    if end is None:
        assert sz.utf8_uncased_search(haystack, needle, start) == expected
    else:
        assert sz.utf8_uncased_search(haystack, needle, start, end) == expected
@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("A", b"a"),
        ("Z", b"z"),
        ("ß", b"ss"),
        ("ẞ", b"ss"),
        ("ﬁ", b"fi"),
        ("ﬀ", b"ff"),
        ("ﬂ", b"fl"),
        ("ﬃ", b"ffi"),
        ("ﬄ", b"ffl"),
        ("Σ", "σ".encode("utf-8")),
        ("Ω", "ω".encode("utf-8")),
        ("Ä", "ä".encode("utf-8")),
        ("É", "é".encode("utf-8")),
        ("Ñ", "ñ".encode("utf-8")),
    ],
)
def test_utf8_uncased_fold_expansions(input_str, expected):
    assert sz.utf8_uncased_fold(input_str) == expected
def test_utf8_uncased_fold_all_codepoints(unicode_folds):
    mismatches = []
    missing_folds = []
    extra_folds = []
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        try:
            char = chr(codepoint)
            char_bytes = char.encode("utf-8")
            sz_folded = sz.utf8_uncased_fold(char)
            expected = unicode_folds.get(codepoint, char_bytes)
            if sz_folded != expected:
                entry = (f"U+{codepoint:04X}", repr(char), expected.hex(), sz_folded.hex())
                if codepoint in unicode_folds and sz_folded == char_bytes:
                    missing_folds.append(entry)
                elif codepoint not in unicode_folds and sz_folded != char_bytes:
                    extra_folds.append(entry)
                else:
                    mismatches.append(entry)
        except (ValueError, UnicodeEncodeError):
            continue
    total_errors = len(mismatches) + len(missing_folds) + len(extra_folds)
    assert total_errors == 0, (
        f"Case folding errors vs Unicode 17.0 ({len(unicode_folds)} rules): "
        f"{len(mismatches)} wrong targets, {len(missing_folds)} missing, {len(extra_folds)} extra. "
        f"First 10: {(mismatches + missing_folds + extra_folds)[:10]}"
    )
def _reference_uncased_find(haystack_bytes: bytes, needle_folded: bytes, unicode_folds: dict) -> int:
    if not needle_folded:
        return 0
    text = haystack_bytes.decode("utf-8")
    codepoint_byte_offsets = []
    codepoint_folds = []
    byte_offset = 0
    for char in text:
        char_bytes = char.encode("utf-8")
        codepoint_byte_offsets.append(byte_offset)
        codepoint_folds.append(unicode_folds.get(ord(char), char_bytes))
        byte_offset += len(char_bytes)
    for start in range(len(text)):
        accumulated = b""
        for index in range(start, len(text)):
            accumulated += codepoint_folds[index]
            if len(accumulated) >= len(needle_folded):
                break
        if accumulated == needle_folded:
            return codepoint_byte_offsets[start]
    return -1
def test_utf8_uncased_search_preimages(unicode_folds):
    mismatches = []
    for codepoint in sorted(unicode_folds):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        preimage_bytes = chr(codepoint).encode("utf-8")
        folded_bytes = unicode_folds[codepoint]
        if preimage_bytes == folded_bytes:
            continue
        for offset in (0, 30, 62, 63, 64):
            haystack = b"y" * offset + preimage_bytes + b"tail"
            expected = _reference_uncased_find(haystack, folded_bytes, unicode_folds)
            actual = sz.utf8_uncased_search(haystack, folded_bytes)
            if actual != expected:
                mismatches.append((f"U+{codepoint:04X}", offset, expected, actual))
    assert not mismatches, (
        f"{len(mismatches)} uncased find mismatches vs Unicode 17.0 reference. "
        f"First 10 (codepoint, offset, expected, actual): {mismatches[:10]}"
    )
def _reference_uncased_find_subset(haystack_bytes: bytes, needle_folded: bytes, unicode_folds: dict) -> int:
    if not needle_folded:
        return 0
    text = haystack_bytes.decode("utf-8")
    folded_blob = b""
    source_byte_offset_for_folded_byte = []
    byte_offset = 0
    for char in text:
        char_bytes = char.encode("utf-8")
        folded_run = unicode_folds.get(ord(char), char_bytes)
        folded_blob += folded_run
        source_byte_offset_for_folded_byte.extend([byte_offset] * len(folded_run))
        byte_offset += len(char_bytes)
    match_position = folded_blob.find(needle_folded)
    if match_position == -1:
        return -1
    return source_byte_offset_for_folded_byte[match_position]
def test_utf8_uncased_search_crossing_expansions(unicode_folds):
    def fold_needle(needle_str: str) -> bytes:
        folded = b""
        for char in needle_str:
            folded += unicode_folds.get(ord(char), char.encode("utf-8"))
        return folded
    crossing_cases = [
        ("ßß", "sss"),
        ("ßß", "ſß"),
        ("ẞß", "ssss"),
        ("ẞß", "sss"),
        ("ﬃ", "fi"),
        ("ﬃ", "ffi"),
        ("ﬀﬁ", "ffi"),
    ]
    filler_char = "z"
    prefix_paddings = [0, 30, 62, 63, 64, 65]
    mismatches = []
    for haystack_str, needle_str in crossing_cases:
        needle_folded = fold_needle(needle_str)
        for padding in prefix_paddings:
            haystack_str_padded = filler_char * padding + haystack_str
            haystack_bytes = haystack_str_padded.encode("utf-8")
            needle_bytes = needle_str.encode("utf-8")
            expected = _reference_uncased_find_subset(haystack_bytes, needle_folded, unicode_folds)
            actual = sz.utf8_uncased_search(haystack_bytes, needle_bytes)
            if actual != expected:
                mismatches.append((haystack_str, needle_str, padding, expected, actual))
    assert not mismatches, (
        f"{len(mismatches)} crossing-expansion find mismatches vs Unicode 17.0 reference. "
        f"(haystack, needle, padding, expected, actual): {mismatches[:10]}"
    )
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_uncased_fold_random_strings(seed_value: int):
    seed_random_generators(seed_value)
    for _ in range(scale_iterations(50)):
        length = randint(1, 100)
        test_str = "".join(chr(randint(0x41, 0x5A)) for _ in range(length))
        python_folded = test_str.casefold().encode("utf-8")
        sz_folded = sz.utf8_uncased_fold(test_str)
        assert python_folded == sz_folded, f"Mismatch for: {test_str!r}"
    for _ in range(scale_iterations(50)):
        length = randint(1, 50)
        codepoints = [randint(0x41, 0x5A) for _ in range(length)]
        codepoints += [randint(0xC0, 0xFF) for _ in range(length // 2)]
        test_str = "".join(chr(cp) for cp in codepoints)
        python_folded = test_str.casefold().encode("utf-8")
        sz_folded = sz.utf8_uncased_fold(test_str)
        assert python_folded == sz_folded, f"Mismatch for: {test_str!r}"
def test_utf8_uncased_prose():
    from test.utf8_helpers import PROSE_HOTEL_REVIEW, PROSE_SCIENCE_ABSTRACT, PROSE_LANGUAGE_LESSON
    cases = [
        (PROSE_HOTEL_REVIEW, "strasse"),
        (PROSE_SCIENCE_ABSTRACT, "film"),
        (PROSE_LANGUAGE_LESSON, "strasse"),
    ]
    for haystack, needle in cases:
        assert haystack.find(needle) == -1
        result = sz.utf8_uncased_search(haystack, needle)
        assert result is not None and result >= 0
UNCASED_FOLD_BACKEND_DIFFERENTIAL_TEXTS = [
    "hello world",
    "HELLO WORLD",
    "Straße",
    "STRASSE",
    "Café",
    "CAFÉ",
    "naïve",
    "ΑΒΓΔ",
    "αβγδ",
    "ПРИВЕТ",
    "привет",
    "ß",
    "ẞ",
    "ﬁ",
    "ﬀ",
    "ﬂ",
    "ﬃ",
    "ﬄ",
] + vector_width_bracketing_strings()
UNCASED_SEARCH_BACKEND_DIFFERENTIAL_CASES = [
    ("Hello World", "WORLD"),
    ("Straße", "STRASSE"),
    ("STRASSE", "straße"),
    ("ßß", "sss"),
    ("ẞß", "ssss"),
    ("ﬃ", "ffi"),
    ("ΑΒΓΔ", "αβγδ"),
    ("ПРИВЕТ", "привет"),
    ("Café au lait", "AU"),
]
UNCASED_ORDER_BACKEND_DIFFERENTIAL_PAIRS = [
    ("hello", "HELLO"),
    ("Straße", "STRASSE"),
    ("strasse", "STRAẞE"),
    ("apple", "BANANA"),
    ("ZEBRA", "apple"),
    ("ΑΒΓΔ", "αβγδ"),
    ("ПРИВЕТ", "привет"),
]
@pytest.mark.parametrize("text", UNCASED_FOLD_BACKEND_DIFFERENTIAL_TEXTS)
def test_utf8_uncased_fold_backend_differential(text):
    assert_backends_agree(run_across_backends(lambda: sz.utf8_uncased_fold(text)), format_inputs=lambda: repr(text))
@pytest.mark.parametrize("raw", malformed_utf8_corpus())
def test_utf8_uncased_fold_backend_differential_malformed(raw):
    assert_backends_agree(run_across_backends(lambda: sz.utf8_uncased_fold(raw)), format_inputs=lambda: raw.hex())
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_uncased_fold_backend_differential_random(seed_value):
    seed_random_generators(seed_value)
    text = get_random_string()
    assert_backends_agree(run_across_backends(lambda: sz.utf8_uncased_fold(text)), format_inputs=lambda: repr(text))
@pytest.mark.parametrize("haystack, needle", UNCASED_SEARCH_BACKEND_DIFFERENTIAL_CASES)
def test_utf8_uncased_search_backend_differential(haystack, needle):
    assert_backends_agree(
        run_across_backends(lambda: sz.utf8_uncased_search(haystack, needle)),
        format_inputs=lambda: f"{haystack!r}/{needle!r}",
    )
@pytest.mark.parametrize("text", vector_width_bracketing_strings())
def test_utf8_uncased_search_backend_differential_vector_width(text):
    assert_backends_agree(
        run_across_backends(lambda: sz.utf8_uncased_search(text, "TAIL")), format_inputs=lambda: repr(text)
    )
@pytest.mark.parametrize("raw", malformed_utf8_corpus())
def test_utf8_uncased_search_backend_differential_malformed(raw):
    needle = raw[:2] if len(raw) >= 2 else raw
    assert_backends_agree(
        run_across_backends(lambda: sz.utf8_uncased_search(raw, needle)), format_inputs=lambda: raw.hex()
    )
@pytest.mark.parametrize("left, right", UNCASED_ORDER_BACKEND_DIFFERENTIAL_PAIRS)
def test_utf8_uncased_order_backend_differential(left, right):
    assert_backends_agree(
        run_across_backends(lambda: sz.utf8_uncased_order(left, right)), format_inputs=lambda: f"{left!r}/{right!r}"
    )
@pytest.mark.parametrize("raw", malformed_utf8_corpus())
def test_utf8_uncased_order_backend_differential_malformed(raw):
    assert_backends_agree(
        run_across_backends(lambda: sz.utf8_uncased_order(raw, raw)), format_inputs=lambda: raw.hex()
    )
