from random import randint
import pytest
from stringzilla import Str, Strs
from test.sz_helpers import (
    SEED_VALUES,
    seed_random_generators,
    get_random_string,
    numpy_available,
    run_across_backends,
    assert_backends_agree,
)
try:
    import numpy as np
except:
    pass
def test_unit_strs_sequence():
    native = "p3\np2\np1"
    big = Str(native)
    lines = big.splitlines()
    assert [2, 1, 0] == list(lines.argsort())
    assert "p3" in lines
    assert "p4" not in lines
    assert repr(lines) == "sz.Strs(['p3', 'p2', 'p1'])"
    assert repr(Str("a" * 1_000_000).split()).endswith("... ])")
    assert str(lines) == "['p3', 'p2', 'p1']"
    assert str(Str("a" * 1_000_000).split()).startswith("['aaa")
    assert str(Str("a" * 1_000_000).split()).endswith("aaa']")
    lines_sorted = lines.sorted()
    assert [0, 1, 2] == list(lines_sorted.argsort())
    assert ["p1", "p2", "p3"] == list(lines_sorted)
    assert [2, 1, 0] == list(lines_sorted.argsort(reverse=True))
    lines_sorted_reverse = lines.sorted(reverse=True)
    assert ["p3", "p2", "p1"] == list(lines_sorted_reverse)
    assert ["p1", "p2"] == list(lines.sorted(top=2))
    assert [0, 1] == list(lines.argsort(top=2, reverse=True))
    assert list(lines.sorted()) == list(lines.sorted(top=999))
    mixed = Str("Banana\napple\nBANANA\nApple").splitlines()
    assert ["apple", "Apple", "Banana", "BANANA"] == list(mixed.sorted(uncased=True))
    assert [1, 3, 0, 2] == list(mixed.argsort(uncased=True))
    assert ["Banana", "BANANA", "apple", "Apple"] == list(mixed.sorted(uncased=True, reverse=True))
    try:
        lines.sorted(True)
        assert False, "sorted() should reject positional arguments"
    except TypeError:
        pass
    sampled = lines.sample(100, seed=42)
    assert "p3" in sampled
    assert "p4" not in sampled
def test_unit_strs_intersect():
    first = Str("banana\napple\ncherry").splitlines()
    second = Str("cherry\norange\npineapple\nbanana").splitlines()
    ours, theirs = first.intersect(second)
    assert len(ours) == len(theirs) == 2
    assert {str(first[i]) for i in ours} == {"banana", "cherry"}
    for i, j in zip(ours, theirs):
        assert str(first[i]) == str(second[j])
    seeded_ours, seeded_theirs = first.intersect(second, seed=42)
    assert sorted(seeded_ours) == sorted(ours)
    assert sorted(seeded_theirs) == sorted(theirs)
    assert first.intersect(Str("kiwi\nmango").splitlines()) == ((), ())
    assert first.intersect(Str("").splitlines()) == ((), ())
    dup_ours, dup_theirs = Str("apple\napple\nbanana").splitlines().intersect(Str("apple\napple").splitlines())
    assert len(dup_ours) == len(dup_theirs) == 1
    try:
        first.intersect(["banana"])
        assert False, "intersect() should reject non-Strs arguments"
    except TypeError:
        pass
def test_unit_strs_argsort_out():
    import array
    import struct
    strs = Str("banana\napple\ncherry\nApple\nBANANA").splitlines()
    n = len(strs)
    index_code = next(code for code in "LQ" if array.array(code).itemsize == struct.calcsize("P"))
    buf = array.array(index_code, [0] * n)
    assert strs.argsort(uncased=True, out=buf) is buf
    assert tuple(buf) == strs.argsort(uncased=True)
    wide = array.array(index_code, [12345] * n)
    strs.argsort(top=2, out=wide)
    assert tuple(wide[:2]) == strs.argsort(top=2)
    assert all(x == 12345 for x in wide[2:]), "argsort(out=...) clobbered past `top`"
    with pytest.raises(TypeError):
        strs.argsort(out=array.array("H", [0] * n))
    with pytest.raises(ValueError):
        strs.argsort(out=array.array(index_code, [0] * (n - 1)))
    with pytest.raises((TypeError, BufferError)):
        strs.argsort(out=bytes(8 * n))
    with pytest.raises(TypeError):
        strs.sorted(out=buf)
def randomly_cased(text: str) -> str:
    return "".join(char.upper() if randint(0, 1) else char for char in text)
def assert_sort_family_matches_oracles(native_list: list, *, top=None, reverse: bool = False, uncased: bool = False):
    strs = Strs(native_list)
    sorted_results = run_across_backends(lambda: list(map(str, strs.sorted(top=top, reverse=reverse, uncased=uncased))))
    argsort_results = run_across_backends(lambda: strs.argsort(top=top, reverse=reverse, uncased=uncased))
    assert_backends_agree(sorted_results)
    assert_backends_agree(argsort_results)
    sorted_order = next(iter(sorted_results.values()))
    permutation = next(iter(argsort_results.values()))
    assert [native_list[index] for index in permutation] == sorted_order
    key = (lambda item: item.casefold()) if uncased else None
    expected_full = sorted(native_list, key=key, reverse=reverse)
    expected_count = len(native_list) if top is None else min(top, len(native_list))
    assert sorted_order == expected_full[:expected_count]
@pytest.mark.parametrize("list_length", [10, 20, 30, 40, 50])
@pytest.mark.parametrize("part_length", [5, 10])
@pytest.mark.parametrize("variability", [2, 3])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_fuzzy_sorting(list_length: int, part_length: int, variability: int, seed_value: int):
    seed_random_generators(seed_value)
    native_list = [get_random_string(variability=variability, length=part_length) for _ in range(list_length)]
    native_joined = ".".join(native_list)
    big_joined = Str(native_joined)
    big_list = big_joined.split(".")
    def py_cmp(a: str, b: str) -> int:
        return -1 if a < b else (1 if a > b else 0)
    def sz_cmp(a: str, b: str) -> int:
        sa, sb = Str(a), Str(b)
        if sa < sb:
            return -1
        if sa > sb:
            return 1
        return 0
    for i in range(len(native_list) - 1):
        a, b = native_list[i], native_list[i + 1]
        assert py_cmp(a, b) == sz_cmp(a, b), f"Comparator mismatch at {i}: '{a}' vs '{b}'"
    native_ordered = sorted(native_list)
    native_order = big_list.argsort()
    for i in range(list_length):
        assert native_ordered[i] == native_list[native_order[i]], "Order is wrong"
        assert native_ordered[i] == str(big_list[int(native_order[i])]), "Split is wrong?!"
    if numpy_available:
        out = np.zeros(list_length, dtype=np.uintp)
        returned = big_list.argsort(out=out)
        assert returned is out
        assert out.tolist() == list(native_order)
    native_list.sort()
    big_list = big_list.sorted()
    assert len(native_list) == len(big_list)
    for native_str, big_str in zip(native_list, big_list):
        assert native_str == str(big_str), "Order is wrong"
BATCH_SIZES = [1, 2, 33, 64, 1000]
def random_batch(size: int) -> list:
    return [randomly_cased(get_random_string(variability=6, length=randint(0, 16))) for _ in range(size)]
def all_equal_batch(size: int) -> list:
    return ["repeated"] * size
def single_char_batch(size: int) -> list:
    return [randomly_cased(get_random_string(variability=20, length=1)) for _ in range(size)]
def empty_mixed_batch(size: int) -> list:
    return [
        "" if index % 3 == 0 else randomly_cased(get_random_string(variability=6, length=randint(1, 10)))
        for index in range(size)
    ]
BATCH_BUILDERS = {
    "random": random_batch,
    "all_equal": all_equal_batch,
    "single_char": single_char_batch,
    "empty_mixed": empty_mixed_batch,
}
@pytest.mark.parametrize("uncased", [False, True])
@pytest.mark.parametrize("batch_name", sorted(BATCH_BUILDERS))
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_sorted_argsort(seed_value, batch_size, batch_name, uncased):
    seed_random_generators(seed_value)
    native_list = BATCH_BUILDERS[batch_name](batch_size)
    assert_sort_family_matches_oracles(native_list, uncased=uncased)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("top", [None, 1, 5, 50, 10_000])
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_sorted_argsort_top_reverse(seed_value, batch_size, top, reverse):
    seed_random_generators(seed_value)
    native_list = random_batch(batch_size)
    assert_sort_family_matches_oracles(native_list, top=top, reverse=reverse)
def test_unit_backend_differential_sorted_argsort_empty():
    for uncased in (False, True):
        for reverse in (False, True):
            assert_sort_family_matches_oracles([], uncased=uncased, reverse=reverse)
