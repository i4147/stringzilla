import itertools
from random import Random, choice, randint
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
    seed_random_generators,
    vector_width_bracketing_strings,
)
from test.utf8_helpers import (
    adversarial_utf8_inputs,
    assert_segments_tile,
    byte_boundaries,
    class_adjacency_strings,
    corpus_of_byte_length,
    window_seam_lengths,
)
def test_utf8_wordbreaks_basic():
    result = [str(w) for w in sz.utf8_wordbreaks("hello world")]
    assert result == ["hello", " ", "world"]
    result = [str(w) for w in sz.utf8_wordbreaks("")]
    assert result == []
    result = [str(w) for w in sz.utf8_wordbreaks("a")]
    assert result == ["a"]
def test_utf8_wordbreaks_skip_empty():
    result = [str(w) for w in sz.utf8_wordbreaks("hello  world")]
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)
def test_utf8_wordbreaks_contractions():
    result = [str(w) for w in sz.utf8_wordbreaks("don't")]
    assert "don't" in result or result == ["don't"]
def test_utf8_wordbreaks_unicode():
    result = [str(w) for w in sz.utf8_wordbreaks("Größe")]
    assert "Größe" in result
    result = [str(w) for w in sz.utf8_wordbreaks("привет мир")]
    assert len(result) >= 2
    result = [str(w) for w in sz.utf8_wordbreaks("你好")]
    assert len(result) >= 1
def test_utf8_wordbreaks_emoji():
    result = [str(w) for w in sz.utf8_wordbreaks("hello 👋 world")]
    assert "hello" in result
    assert "world" in result
def test_utf8_wordbreaks_numbers():
    result = [str(w) for w in sz.utf8_wordbreaks("test123")]
    assert "test123" in result or result == ["test123"]
    result = [str(w) for w in sz.utf8_wordbreaks("3.14")]
    assert "3.14" in result or len(result) <= 3
@pytest.mark.parametrize("run_length", [3, 8, 15, 31, 32, 33, 100])
def test_utf8_wordbreaks_deferred_mid(run_length: int):
    marks = "̀" * run_length
    he = "ה"
    emoji = "\U0001f600"
    zwj = "‍"
    math_five = "\U0001d7d7"
    goldens = [
        (f"5,{marks}6", [f"5,{marks}6"]),
        (f"5,{marks}{math_five}", [f"5,{marks}{math_five}"]),
        (f"a:{marks}b", [f"a:{marks}b"]),
        (f'{he}"{marks}{he}', [f'{he}"{marks}{he}']),
        (f"5,{marks}a", ["5", f",{marks}", "a"]),
        (f"a:{marks}5", ["a", f":{marks}", "5"]),
        (f"5,{marks}", ["5", f",{marks}"]),
        (f"5,{marks},5", ["5", f",{marks}", ",", "5"]),
        (f"{he}'{marks}x", [f"{he}'{marks}x"]),
        (f"{he}'{marks}5", [f"{he}'{marks}", "5"]),
        (f'{he}"{marks}x', [he, f'"{marks}', "x"]),
        (f"a:{marks}{zwj}{emoji}", ["a", f":{marks}{zwj}{emoji}"]),
        (f"5,{emoji}", ["5", ",", emoji]),
    ]
    for text, expected in goldens:
        assert [str(w) for w in sz.utf8_wordbreaks(text)] == expected
def test_utf8_wordbreaks_str_method():
    s = Str("hello world")
    result = [str(w) for w in s.utf8_wordbreaks(s)]
    assert result == ["hello", " ", "world"]
    result = [str(w) for w in sz.utf8_wordbreaks(s)]
    assert result == ["hello", " ", "world"]
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_word_safety(seed_value: int):
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        assert_segments_tile(sz.utf8_wordbreaks(raw), raw)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_word_seam(seed_value: int):
    rng = Random(seed_value)
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_wordbreaks(raw), raw)
    body = corpus_of_byte_length(96, rng)
    for phase in range(64):
        raw = b"a" * phase + body
        assert_segments_tile(sz.utf8_wordbreaks(raw), raw)
def test_utf8_word_boundary_official_conformance(word_break_cases):
    test_cases = word_break_cases
    failures = []
    for test_str, expected_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = [0]
        for word in sz.utf8_wordbreaks(test_str):
            sz_boundaries.append(sz_boundaries[-1] + len(str(word).encode("utf-8")))
        if sz_boundaries != expected_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_boundaries}, got {sz_boundaries}")
    assert not failures, "UAX-29 WordBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )
def test_utf8_word_class_adjacency(word_break_props):
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    representatives = representatives_by_class(word_break_props, count=1)
    cases = class_adjacency_strings(representatives, arity=2)
    cases += class_adjacency_strings(representatives, arity=3, max_cases=8000)
    failures = []
    for text in cases:
        sz_boundaries = byte_boundaries(sz.utf8_wordbreaks(text))
        reference_boundaries = byte_boundaries(list(uniseg_wordbreak.words(text)))
        if sz_boundaries != reference_boundaries:
            codepoints = " ".join(f"{ord(character):04X}" for character in text)
            failures.append(f"  {codepoints}\n    sz={sz_boundaries}\n    uniseg={reference_boundaries}")
    assert not failures, "StringZilla vs uniseg word class-adjacency divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_word_boundary_differential_uniseg(seed_value: int):
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    seed_random_generators(seed_value)
    palette = list(
        "abZ59 \t\n\r.,;:'\"_-"
        "éαаאבカタ中국"
        "̀́̈"
        "­⁠‍"
        "　  "
        "\U0001f600\U0001f6d1Ⓜ©❤"
        "\U0001f1e6\U0001f1fa\U0001f3fb"
    )
    def boundaries(segments):
        out = [0]
        for seg in segments:
            out.append(out[-1] + len(seg.encode("utf-8")))
        return out
    seam_regressions = [
        bytes.fromhex(
            "E382AB2D0AF09F87A662C2ADF09F87A6CC800A5FF09F87A6F09F87A6C2AD0ACC88F09F87BAF09F8FBBCC88C2ADE281A0CC88F0"
            "9F87A6F09F87A6E4B8ADE281A05F"
        ).decode("utf-8"),
        bytes.fromhex(
            "5F27F09F87BAE4B8ADCC81F09F87A6F09F87A65FCC885FD7902DE29382CC8062F09F87A6E281A0F09F87BACC80CC88F09F8FBB"
            "F09F87BACC81F09F87BA0AC2ADF09F87BACC88E382AB5F272ECC880AE4B8AD"
        ).decode("utf-8"),
        bytes.fromhex(
            "5FE281A05F352CF09F988027F09F98805FCC88CC81E4B8ADE281A05F62CC8061E4B8ADE382AB35E281A0CC88CC88F09F8FBBCC"
            "88E281A0CC80CC812735CC80E4B8ADCC81E2808DC3A9E380803527"
        ).decode("utf-8"),
    ]
    failures = []
    random_samples = ("".join(choice(palette) for _ in range(randint(1, 100))) for _ in range(scale_iterations(2000)))
    for text in itertools.chain(seam_regressions, random_samples):
        sz_boundaries = boundaries(str(w) for w in sz.utf8_wordbreaks(text))
        ref_boundaries = boundaries(uniseg_wordbreak.words(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    uniseg={ref_boundaries}")
    assert not failures, "StringZilla vs uniseg word-boundary divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_word_boundary_fuzz(seed_value: int):
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    seed_random_generators(seed_value)
    test_chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " \t\n\r"
        ".,;:!?'\"_-"
        "äöüßαβγδабвг"
        "אב"
        "カタ"
        "中文"
        "̀́̈"
        "­⁠‍"
        "　  "
        "\U0001f600\U0001f6d1Ⓜ©❤"
        "\U0001f1e6\U0001f1fa"
    )
    for _ in range(scale_iterations(200)):
        length = randint(1, 100)
        text = "".join(choice(test_chars) for _ in range(length))
        expected_boundaries = [0]
        for seg in uniseg_wordbreak.words(text):
            expected_boundaries.append(expected_boundaries[-1] + len(seg.encode("utf-8")))
        sz_boundaries = [0]
        for word in sz.utf8_wordbreaks(text):
            sz_boundaries.append(sz_boundaries[-1] + len(str(word).encode("utf-8")))
        assert (
            sz_boundaries == expected_boundaries
        ), "Word boundary mismatch (seed {}):\n  text cps: {}\n  expected: {}\n  got:      {}".format(
            seed_value, " ".join(f"{ord(c):04X}" for c in text), expected_boundaries, sz_boundaries
        )
def test_utf8_wordbreaks_prose():
    from test.utf8_helpers import (
        PROSE_HOTEL_REVIEW,
        PROSE_NEWS_LEDE,
        PROSE_CONCERT_POST,
        PROSE_RTL_SCRIPTS,
        PROSE_MICRO_APOSTROPHE,
        assert_segments_tile,
    )
    cases = [
        (PROSE_HOTEL_REVIEW, 100),
        (PROSE_NEWS_LEDE, 83),
        (PROSE_CONCERT_POST, 69),
        (PROSE_RTL_SCRIPTS, 98),
        (PROSE_MICRO_APOSTROPHE, 5),
    ]
    for text, expected in cases:
        segments = list(sz.utf8_wordbreaks(text))
        assert_segments_tile(segments, text)
        assert len(segments) == expected
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak")
    for text, _ in cases:
        assert len(list(sz.utf8_wordbreaks(text))) == len(list(uniseg_wordbreak.words(text)))
WORDBREAK_REALISTIC_TEXTS = [
    "hello world",
    "don't stop",
    "Größe café",
    "привет мир",
    "你好 世界",
    "test123 3.14",
    "hello 👋 world",
]
@pytest.mark.parametrize(
    "text", WORDBREAK_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_wordbreaks_backend_differential(text):
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_wordbreaks(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")
