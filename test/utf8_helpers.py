import itertools
from typing import Callable, Iterable, List, Optional, Sequence, Union
import pytest
from test.sz_helpers import scale_iterations
_PALETTE_CODEPOINTS = [
    0x00E9,
    0x03B1,
    0x0430,
    0x05D0,
    0x05D1,
    0x30AB,
    0x30BF,
    0x4E2D,
    0xAD6D,
    0x0300,
    0x0301,
    0x0308,
    0x00AD,
    0x2060,
    0x200D,
    0x3000,
    0x0020,
    0x0020,
    0x1F600,
    0x1F6D1,
    0x24C2,
    0x00A9,
    0x2764,
    0x1F1E6,
    0x1F1FA,
    0x1F3FB,
    0x2028,
    0x2029,
]
SEGMENTATION_PALETTE: List[str] = list("abZ59 \t\n\r.,;:!?'\"_-()") + [chr(cp) for cp in _PALETTE_CODEPOINTS]
ASTRAL_FIXTURES: List[bytes] = [
    b"\xf0\x9f\x87\xba\xf0\x9f\x87\xb8",
    b"\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f\x87\xab\xf0\x9f\x87\xb7",
    b"\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa7",
    b"\xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd",
    b"\xf0\x9f\x98\x80\xf0\x9f\x98\x81",
    b"a\xf0\x9f\x98\x80b",
    b"\xf0\x90\x80\x80\xf0\x90\x80\x81",
    b"\xf0\x9f\x87\xbaa\xf0\x9f\x87\xb8",
]
MALFORMED_CLASSES: List[bytes] = [
    b"\xc0\x80",
    b"\xc1\xbf",
    b"\xe0\x80\x80",
    b"\xe0\x9f\xbf",
    b"\xf0\x80\x80\x80",
    b"\xf0\x8f\xbf\xbf",
    b"\xed\xa0\x80",
    b"\xed\xbf\xbf",
    b"\x80",
    b"\xbf",
    b"\xfe",
    b"\xff",
    b"\xc2",
    b"\xe2\x80",
    b"\xf0\x9f\x98",
    b"\xf5\x80\x80\x80",
    b"\xef\xb7\x90",
    b"\xef\xb7\xaf",
    b"\xef\xbf\xbe",
    b"\xef\xbf\xbf",
]
_NAMED_ADVERSARIAL: List[bytes] = [
    b"\x80",
    b"\xc0\x80",
    b"\xed\xa0\x80",
    b"hello\xf0\x9f\x98",
]
_LEAD_BYTES = [0x00, 0x41, 0x7F, 0x80, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xED, 0xEF, 0xF0, 0xF4, 0xF5, 0xFF]
_BOUNDARY_CODEPOINTS = [0x007F, 0x0080, 0x07FF, 0x0800, 0xFFFF, 0x10000, 0x10FFFF, 0xFEFF, 0x200D, 0xFE0F]
_SNIPPETS: List[bytes] = [
    b"a",
    b"Hello, world! ",
    b"don't ",
    b"3,14 ",
    b"\xe4\xbd\xa0\xe5\xa5\xbd",
    b"\r\n",
    b"one. two. ",
    b"A! B? ",
    b"e\xcc\x81",
    b"\xea\xb0\x80",
    b"\xe0\xa4\x95\xe0\xa5\x8d\xe0\xa4\xb7",
    b"line\nbreak ",
    b"soft wrap here ",
    b"\xe2\x80\xa8",
    b"\xe2\x80\xa9",
]
_UNIT_CAPACITY = 70
def byte_boundaries(segments: Iterable) -> List[int]:
    out = [0]
    for segment in segments:
        out.append(out[-1] + len(str(segment).encode("utf-8")))
    return out
def char_boundaries_to_bytes(text: str, char_boundaries: Iterable[int]) -> List[int]:
    char_boundaries = set(char_boundaries)
    out = []
    byte_position = 0
    for index, char in enumerate(text):
        if index in char_boundaries:
            out.append(byte_position)
        byte_position += len(char.encode("utf-8"))
    if len(text) in char_boundaries:
        out.append(byte_position)
    return out
def segment_bytes(segments: Iterable) -> bytes:
    return b"".join(bytes(segment) for segment in segments)
def assert_segments_tile(segments: Iterable, original: Union[str, bytes]) -> None:
    original_bytes = original.encode("utf-8") if isinstance(original, str) else bytes(original)
    rebuilt = segment_bytes(segments)
    assert rebuilt == original_bytes, "segments do not tile the input: in={} out={}".format(
        original_bytes.hex(), rebuilt.hex()
    )
def append_malformed_class(rng) -> bytes:
    return rng.choice(MALFORMED_CLASSES)
def apply_mutation_passes(data: bytes, rng) -> bytes:
    buffer = bytearray(data)
    if not buffer:
        return bytes(buffer)
    to_corrupt = len(buffer) // 10
    for _ in range(to_corrupt):
        buffer[rng.randrange(len(buffer))] = 0
    for _ in range(to_corrupt):
        first, second = rng.randrange(len(buffer)), rng.randrange(len(buffer))
        buffer[first], buffer[second] = buffer[second], buffer[first]
    truncate_by = min(rng.randint(1, 3), len(buffer))
    del buffer[len(buffer) - truncate_by :]
    if buffer:
        buffer.insert(rng.randint(0, len(buffer)), rng.randint(0x80, 0xBF))
    return bytes(buffer)
def random_segmentation_corpus(min_length: int, flavor: str, motifs: Optional[Sequence[bytes]], rng) -> bytes:
    motifs = list(motifs or [])
    out = bytearray()
    while len(out) < min_length:
        roll = rng.randint(0, 99)
        if flavor == "malformed" and roll < 10:
            out += append_malformed_class(rng)
        elif roll < 30:
            out += chr(rng.choice(_BOUNDARY_CODEPOINTS)).encode("utf-8")
        elif roll < 45:
            out += rng.choice(ASTRAL_FIXTURES)
        elif roll < 65 and motifs:
            out += rng.choice(motifs)
        else:
            out += rng.choice(_SNIPPETS)
    return bytes(out)
def corpus_of_byte_length(target_length: int, rng, palette: Optional[Sequence[str]] = None) -> bytes:
    palette = palette or SEGMENTATION_PALETTE
    out = bytearray()
    while len(out) < target_length:
        encoded = rng.choice(palette).encode("utf-8")
        if len(out) + len(encoded) > target_length:
            break
        out += encoded
    out += b"x" * (target_length - len(out))
    return bytes(out)
def window_seam_lengths() -> List[int]:
    return [1, 3, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256]
def adversarial_utf8_inputs(rng, random_input_count: Optional[int] = None) -> Iterable[bytes]:
    yield from _NAMED_ADVERSARIAL
    yield from ASTRAL_FIXTURES
    for byte in range(256):
        yield bytes([byte])
    for lead in _LEAD_BYTES:
        for trailing in range(256):
            yield bytes([lead, trailing])
    count = scale_iterations(4000) if random_input_count is None else random_input_count
    for _ in range(count):
        length = rng.randint(1, _UNIT_CAPACITY)
        yield bytes(rng.randrange(256) for _ in range(length))
def icu_segmenter(kind: str) -> Callable[[str], List[str]]:
    icu = pytest.importorskip("icu", reason="PyICU not installed")
    factories = {
        "word": icu.BreakIterator.createWordInstance,
        "grapheme": icu.BreakIterator.createCharacterInstance,
        "sentence": icu.BreakIterator.createSentenceInstance,
        "line": icu.BreakIterator.createLineInstance,
    }
    iterator = factories[kind](icu.Locale.getRoot())
    def segments(text: str) -> List[str]:
        unicode_text = icu.UnicodeString(text)
        iterator.setText(unicode_text)
        parts = []
        start = iterator.first()
        end = iterator.nextBoundary()
        while end != icu.BreakIterator.DONE:
            parts.append(str(unicode_text[start:end]))
            start = end
            end = iterator.nextBoundary()
        return parts
    return segments
def icu_unicode_at_least(version: str) -> bool:
    icu = pytest.importorskip("icu", reason="PyICU not installed")
    def major_minor(value: str) -> tuple:
        return tuple(int(piece) for piece in str(value).split(".")[:2] if piece.isdigit())
    return major_minor(icu.UNICODE_VERSION) >= major_minor(version)
def icu_normalizer(form: str) -> Callable[[str], str]:
    icu = pytest.importorskip("icu", reason="PyICU not installed")
    getters = {
        "NFC": icu.Normalizer2.getNFCInstance,
        "NFD": icu.Normalizer2.getNFDInstance,
        "NFKC": icu.Normalizer2.getNFKCInstance,
        "NFKD": icu.Normalizer2.getNFKDInstance,
    }
    normalizer = getters[form]()
    return lambda text: normalizer.normalize(text)
def class_adjacency_strings(
    representatives: dict, arity: int = 2, representatives_per_class: int = 1, max_cases: Optional[int] = None
) -> List[str]:
    class_names = sorted(representatives)
    chosen = {name: representatives[name][:representatives_per_class] for name in class_names}
    results: List[str] = []
    for class_combination in itertools.product(class_names, repeat=arity):
        for codepoint_combination in itertools.product(*[chosen[name] for name in class_combination]):
            results.append("".join(chr(codepoint) for codepoint in codepoint_combination))
            if max_cases is not None and len(results) >= max_cases:
                return results
    return results
def combining_scrambles(combining_classes: dict, rng, count: int, max_marks: int = 4) -> List[str]:
    marks = [codepoint for codepoint, combining_class in combining_classes.items() if codepoint <= 0xFFFF]
    base_letters = "aeoun"
    results: List[str] = []
    for _ in range(count):
        base_letter = rng.choice(base_letters)
        mark_count = rng.randint(2, max_marks)
        scrambled_marks = [rng.choice(marks) for _ in range(mark_count)]
        results.append(base_letter + "".join(chr(codepoint) for codepoint in scrambled_marks))
    return results
def decomposition_pairs(decomposition_mappings: dict, canonical_only: bool = True, limit: Optional[int] = None) -> list:
    pairs = []
    for codepoint in sorted(decomposition_mappings):
        kind, targets = decomposition_mappings[codepoint]
        if canonical_only and kind != "canonical":
            continue
        pairs.append((chr(codepoint), "".join(chr(target) for target in targets)))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs
PROSE_HOTEL_REVIEW = (
    "Last spring we strolled down M\u00fcnchner Stra\u00dfe; the cafe\u0301 cortado cost 3,50"
    "\u00a0\u20ac and was unreal. Dr. Vogel, our guide, swore it's the city's finest. Worth the d"
    "etour?! Absolutely \u2014 and \u6771\u4eac\u30bf\u30ef\u30fc the next week, all 333\u00a0m o"
    "f it, was breathtaking at dusk\u2026"
)
PROSE_PRIDE_CAPTION = (
    "Best Pride yet \U0001f3f3\ufe0f\u200d\U0001f308 \u2014 the whole crew showed up. Even my par"
    "ents \U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466 and grandma \U0001f44d"
    "\U0001f3fd came through! We met at booth 5\ufe0f\u20e3, then waved every flag we packed "
    "\U0001f1fa\U0001f1f8\U0001f1ef\U0001f1f5\U0001f1eb. Texting \u260e\ufe0e over calling \u2708"
    "\ufe0f all day; 10/10, would march again."
)
PROSE_CONCERT_POST = (
    "\uc624\ub298 \ucf58\uc11c\ud2b8, \uc9c4\uc9dc \ubbf8\ucce4\ub2e4!! \u1112\u1161\u11ab\uad6d "
    "\ud32c\ub4e4\uc774 \ub2e4 \ubaa8\uc600\uace0, the staff bowed and said \uc548\ub155\ud788 "
    "\uac00\uc138\uc694. Setlist was pure \u30cf\u30fc\u30c9\u30b3\u30a2; \u4eca\u65e5\u306f"
    "\u6700\u9ad8\u3060\u3063\u305f\u3002 We screamed \uc0ac\ub791\ud574 till 11 p.m. sharp."
)
PROSE_DEVANAGARI_TIP = (
    "Quick Devanagari tip: \u0915\u094d\u0937 is one cluster (\u0915 + \u094d + \u0937), not thre"
    "e. Force the half-form with ZWJ \u2014 \u0915\u094d\u200d\u0937 \u2014 or split it with ZWNJ"
    " \u2014 \u0915\u094d\u200c\u0937. The same logic hits \u0915\u094d\u0937\u0924\u094d\u0930"
    "\u093f\u092f and spacing vowel signs like \u0915\u0940. Renderers disagree, so test (\u00bd "
    "the bugs are font bugs) before you ship!"
)
PROSE_SCIENCE_ABSTRACT = (
    "The \ufb01lm grew at 300\u00a0\u212a on a 5\u00a0\u212b buffer (\u2248 2\u00b2 monolayers). "
    "Section \u216b covers the \uff21-phase; see Fig. 2 for the \u03a3-band dispersion. Resistivi"
    "ty scaled as T\u00b2, vanishing at the 4.2\u00a0\u212a transition. Full dataset: doi:10.1000"
    "\u2060/\u200bxyz (mirror in Box \u2461)."
)
PROSE_NEWS_LEDE = (
    "The U.S.A. wasn't ready, analysts said. \u201cWe lost 1,000 jobs,\u201d the mayor warned. "
    "\u201cRecovery starts now.\u201d Filings spiked 2024/06\u20132024/09, topping $1,000 per cla"
    "im. Will it hold?! No one knows for sure."
)
PROSE_LANGUAGE_LESSON = (
    "Greek lesson: \u039f\u0394\u039f\u03a3 becomes \u03bf\u03b4\u03cc\u03c2 when lowercased, end"
    "ing in a final \u03c2. Russian's easy too \u2014 \u041c\u041e\u0421\u041a\u0412\u0410 \u2194"
    " \u043c\u043e\u0441\u043a\u0432\u0430, no drama. Croatian has the digraph \u01c4: titlecase "
    "\u01c5, lowercase \u01c6. Quiz \u2014 does \u201cstra\u00dfe\u201d match STRASSE? Yes, once "
    "you fold."
)
PROSE_RTL_SCRIPTS = (
    "Hebrew acronyms take gershayim: \u05e6\u05d4\u05f4\u05dc and \u05d0\u05e8\u05d4\u05f4\u05d1 "
    "aren't typos. Arabic flows right-to-left too \u2014 \u0645\u0631\u062d\u0628\u0627 \u0628"
    "\u0627\u0644\u0639\u0627\u0644\u0645 \u2014 and finance text can carry the number sign "
    "\u0600\u0664. Niqqud stacks marks: \u05e9\u05c1\u05b8\u05dc\u05d5\u05b9\u05dd must reorder u"
    "nder NFC. Malayalam even has a true prepend, the dot-reph \u0d4e\u0d15."
)
PROSE_MICRO_APOSTROPHE = "it\u2019s worth it"
PROSE_MICRO_PREPEND = "\u0600\u0664 \u0d4e\u0d15"
PROSE_MICRO_HARDBREAKS = "A.\u000d\u000aB.\u2028C."
