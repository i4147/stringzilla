from random import randint
from string import ascii_lowercase
from typing import Literal
import pytest
import numpy as np
import affine_gaps as ag
import stringzillas as szs
from stringzilla import Strs
from test.sz_helpers import SEED_VALUES, seed_random_generators, get_random_string
from test.szs_helpers import (
    DeviceName,
    DEVICE_NAMES,
    InputSizeConfig,
    INPUT_SIZE_CONFIGS,
    device_scope_and_capabilities,
    generate_string_batches,
    run_across_engines,
    assert_engines_agree,
)
def baseline_levenshtein_distance(s1, s2) -> int:
    matrix = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    for i in range(len(s1) + 1):
        matrix[i, 0] = i
    for j in range(len(s2) + 1):
        matrix[0, j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i, j] = min(
                matrix[i - 1, j] + 1,
                matrix[i, j - 1] + 1,
                matrix[i - 1, j - 1] + cost,
            )
    return matrix[len(s1), len(s2)]
def affine_gap_edit_distance(first_sequence, second_sequence, mismatch=1, open=1, extend=1):
    first_length, second_length = len(first_sequence), len(second_sequence)
    infinity = float("inf")
    substitution_scores = [[infinity] * (second_length + 1) for _ in range(first_length + 1)]
    deletion_scores = [[infinity] * (second_length + 1) for _ in range(first_length + 1)]
    insertion_scores = [[infinity] * (second_length + 1) for _ in range(first_length + 1)]
    substitution_scores[0][0] = 0
    for first_index in range(1, first_length + 1):
        deletion_scores[first_index][0] = open + extend * (first_index - 1)
        substitution_scores[first_index][0] = deletion_scores[first_index][0]
    for second_index in range(1, second_length + 1):
        insertion_scores[0][second_index] = open + extend * (second_index - 1)
        substitution_scores[0][second_index] = insertion_scores[0][second_index]
    for first_index in range(1, first_length + 1):
        for second_index in range(1, second_length + 1):
            substitution_cost = 0 if first_sequence[first_index - 1] == second_sequence[second_index - 1] else mismatch
            best_predecessor = min(
                substitution_scores[first_index - 1][second_index - 1],
                deletion_scores[first_index - 1][second_index - 1],
                insertion_scores[first_index - 1][second_index - 1],
            )
            substitution_scores[first_index][second_index] = best_predecessor + substitution_cost
            deletion_scores[first_index][second_index] = min(
                substitution_scores[first_index - 1][second_index] + open,
                deletion_scores[first_index - 1][second_index] + extend,
            )
            insertion_scores[first_index][second_index] = min(
                substitution_scores[first_index][second_index - 1] + open,
                insertion_scores[first_index][second_index - 1] + extend,
            )
    return int(
        min(
            substitution_scores[first_length][second_length],
            deletion_scores[first_length][second_length],
            insertion_scores[first_length][second_length],
        )
    )
LevenshteinCostMode = Literal["unit", "linear", "affine"]
LEVENSHTEIN_COST_MODES = ["unit", "linear", "affine"]
def levenshtein_costs_for_mode(cost_mode: LevenshteinCostMode):
    if cost_mode == "unit":
        return dict(match=0, mismatch=1, open=1, extend=1)
    elif cost_mode == "linear":
        return dict(match=0, mismatch=3, open=2, extend=2)
    elif cost_mode == "affine":
        return dict(match=0, mismatch=4, open=5, extend=2)
    else:
        raise ValueError(f"Unknown Levenshtein cost mode: {cost_mode}")
AlignmentCostMode = Literal["linear", "affine"]
ALIGNMENT_COST_MODES = ["linear", "affine"]
def alignment_costs_for_mode(cost_mode: AlignmentCostMode):
    if cost_mode == "linear":
        return -5, -5
    elif cost_mode == "affine":
        return ag.default_gap_opening, ag.default_gap_extension
    else:
        raise ValueError(f"Unknown alignment cost mode: {cost_mode}")
def protein_substitution_tables():
    alphabet = ag.default_proteins_alphabet
    residue_count = len(alphabet)
    byte_to_class = np.zeros(256, dtype=np.uint8)
    byte_to_class[np.frombuffer(alphabet.encode(), dtype=np.uint8)] = np.arange(1, residue_count + 1, dtype=np.uint8)
    class_costs = np.zeros((32, 32), dtype=np.int8)
    class_costs[1 : residue_count + 1, 1 : residue_count + 1] = ag.default_proteins_matrix[
        :residue_count, :residue_count
    ]
    return alphabet, byte_to_class, class_costs
@pytest.mark.parametrize("max_edit_distance", [150])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_levenshtein_distance_insertions(max_edit_distance: int, seed_value: int):
    def insert_char_at(s, char_to_insert, index):
        return s[:index] + char_to_insert + s[index:]
    seed_random_generators(seed_value)
    binary_engine = szs.LevenshteinDistances()
    a = get_random_string(length=20)
    b = a
    for i in range(max_edit_distance):
        source_offset = randint(0, len(ascii_lowercase) - 1)
        target_offset = randint(0, len(b) - 1)
        b = insert_char_at(b, ascii_lowercase[source_offset], target_offset)
        a_strs = Strs([a])
        b_strs = Strs([b])
        results = binary_engine(a_strs, b_strs)
        assert results.shape == (1, 1), "Binary engine should return a 1x1 matrix for single query and candidate"
        assert results[0, 0] == i + 1, f"Edit distance mismatch after {i + 1} insertions: {a} -> {b}"
@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_levenshtein_distances_with_simple_cases(capabilities_mode: str, device_name: DeviceName):
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    binary_engine = szs.LevenshteinDistances(capabilities=base_caps if capabilities_mode == "base" else device_scope)
    def binary_distance(a: str, b: str) -> int:
        a_strs = Strs([a])
        b_strs = Strs([b])
        results = binary_engine(a_strs, b_strs, device=device_scope)
        assert results.shape == (1, 1), "Binary engine should return a 1x1 matrix"
        return int(results[0, 0])
    assert binary_distance("hello", "hello") == 0
    assert binary_distance("hello", "hell") == 1
    assert binary_distance("", "") == 0
    assert binary_distance("", "abc") == 3
    assert binary_distance("abc", "") == 3
    assert binary_distance("abc", "ac") == 1, "one deletion"
    assert binary_distance("abc", "a_bc") == 1, "one insertion"
    assert binary_distance("abc", "adc") == 1, "one substitution"
    assert binary_distance("ggbuzgjux{}l", "gbuzgjux{}l") == 1, "one insertion (prepended)"
    assert binary_distance("abcdefgABCDEFG", "ABCDEFGabcdefg") == 14
@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_levenshtein_distances_utf8_with_simple_cases(capabilities_mode: str, device_name: DeviceName):
    if device_name == "gpu_device":
        pytest.skip("CUDA backend does not support custom gaps in UTF-8 Levenshtein distances")
        return
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    unicode_engine = szs.LevenshteinDistancesUTF8(
        capabilities=base_caps if capabilities_mode == "base" else device_scope
    )
    def unicode_distance(a: str, b: str) -> int:
        a_strs = Strs([a])
        b_strs = Strs([b])
        results = unicode_engine(a_strs, b_strs, device=device_scope)
        assert results.shape == (1, 1), "Unicode engine should return a 1x1 matrix"
        return int(results[0, 0])
    assert unicode_distance("hello", "hell") == 1, "no unicode symbols, just ASCII"
    assert unicode_distance("𠜎 𠜱 𠝹 𠱓", "𠜎𠜱𠝹𠱓") == 3, "add 3 whitespaces in Chinese"
    assert unicode_distance("💖", "💗") == 1
    assert unicode_distance("αβγδ", "αγδ") == 1, "insert Beta"
    assert unicode_distance("école", "école") == 2, "etter 'é' as 1 character vs 'e' + '´'"
    assert unicode_distance("façade", "facade") == 1, "'ç' with cedilla vs. plain"
    assert unicode_distance("Schön", "Scho\u0308n") == 2, "'ö' represented as 'o' + '¨'"
    assert unicode_distance("München", "Muenchen") == 2, "German with umlaut vs. transcription"
    assert unicode_distance("こんにちは世界", "こんばんは世界") == 2, "Japanese greetings"
@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_levenshtein_distances_with_custom_gaps(capabilities_mode: str, device_name: DeviceName):
    mismatch: int = 4
    opening: int = 3
    extension: int = 2
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    binary_engine = szs.LevenshteinDistances(
        open=opening,
        extend=extension,
        mismatch=mismatch,
        capabilities=base_caps if capabilities_mode == "base" else device_scope,
    )
    def binary_distance(a: str, b: str) -> int:
        a_strs = Strs([a])
        b_strs = Strs([b])
        results = binary_engine(a_strs, b_strs, device=device_scope)
        assert results.shape == (1, 1), "Binary engine should return a 1x1 matrix"
        return int(results[0, 0])
    assert binary_distance("hello", "hello") == 0
    assert binary_distance("hello", "hell") == opening
    assert binary_distance("", "") == 0
    assert binary_distance("", "abc") == opening + 2 * extension
    assert binary_distance("abc", "") == opening + 2 * extension
    assert binary_distance("abc", "ac") == opening, "one deletion"
    assert binary_distance("abc", "a_bc") == opening, "one insertion"
    assert binary_distance("abc", "adc") == mismatch, "one substitution"
    assert binary_distance("ggbuzgjux{}l", "gbuzgjux{}l") == opening, "one insertion (prepended)"
    assert binary_distance("abcdefgABCDEFG", "ABCDEFGabcdefg") == min(14 * mismatch, 2 * opening + 12 * extension)
@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_levenshtein_distances_utf8_with_custom_gaps(capabilities_mode: str, device_name: DeviceName):
    if device_name == "gpu_device":
        pytest.skip("CUDA backend does not support custom gaps in UTF-8 Levenshtein distances")
        return
    mismatch: int = 4
    opening: int = 3
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    unicode_engine = szs.LevenshteinDistancesUTF8(
        open=opening,
        extend=opening,
        mismatch=mismatch,
        capabilities=base_caps if capabilities_mode == "base" else device_scope,
    )
    def unicode_distance(a: str, b: str) -> int:
        a_strs = Strs([a])
        b_strs = Strs([b])
        results = unicode_engine(a_strs, b_strs, device=device_scope)
        assert results.shape == (1, 1), "Unicode engine should return a 1x1 matrix"
        return int(results[0, 0])
    assert unicode_distance("hello", "hell") == opening, "no unicode symbols, just ASCII"
    assert unicode_distance("𠜎 𠜱 𠝹 𠱓", "𠜎𠜱𠝹𠱓") == 3 * opening, "add 3 whitespaces in Chinese"
    assert unicode_distance("💖", "💗") == 1 * mismatch
    assert unicode_distance("αβγδ", "αγδ") == opening, "insert Beta"
    assert unicode_distance("école", "école") == mismatch + opening, "etter 'é' as 1 character vs 'e' + '´'"
    assert unicode_distance("façade", "facade") == mismatch, "'ç' with cedilla vs. plain"
    assert unicode_distance("Schön", "Scho\u0308n") == mismatch + opening, "'ö' represented as 'o' + '¨'"
    assert unicode_distance("München", "Muenchen") == mismatch + opening, "German with umlaut vs. transcription"
    assert unicode_distance("こんにちは世界", "こんばんは世界") == min(2 * mismatch, 4 * opening), "Japanese greetings"
@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("config", INPUT_SIZE_CONFIGS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_levenshtein_distance_random(
    capabilities_mode: str,
    device_name: DeviceName,
    config: InputSizeConfig,
    seed_value: int,
):
    seed_random_generators(seed_value)
    batch_size, min_len, max_len = generate_string_batches(config)
    a_batch = [get_random_string(length=randint(min_len, max_len)) for _ in range(batch_size)]
    b_batch = [get_random_string(length=randint(min_len, max_len)) for _ in range(batch_size)]
    baselines = np.array([baseline_levenshtein_distance(a, b) for a, b in zip(a_batch, b_batch)])
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.LevenshteinDistances(capabilities=base_caps if capabilities_mode == "base" else device_scope)
    queries, candidates = Strs(a_batch), Strs(b_batch)
    matrix = engine(queries, candidates)
    assert matrix.shape == (batch_size, batch_size)
    np.testing.assert_array_equal(np.diagonal(matrix), baselines, "Edit distances do not match")
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("queries_count", [1, 3, 5])
@pytest.mark.parametrize("candidates_count", [1, 4, 6])
def test_levenshtein_cross_product_matrix(device_name: DeviceName, queries_count: int, candidates_count: int):
    seed_random_generators(42)
    query_strings = [get_random_string(length=randint(4, 20)) for _ in range(queries_count)]
    candidate_strings = [get_random_string(length=randint(4, 20)) for _ in range(candidates_count)]
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.LevenshteinDistances(capabilities=base_caps)
    matrix = engine(Strs(query_strings), Strs(candidate_strings), device=device_scope)
    assert matrix.shape == (queries_count, candidates_count)
    assert matrix.dtype == np.uint64
    reference = np.array(
        [[baseline_levenshtein_distance(q, c) for c in candidate_strings] for q in query_strings],
        dtype=np.uint64,
    )
    for query_index in range(queries_count):
        for candidate_index in range(candidates_count):
            assert (
                matrix[query_index, candidate_index] == reference[query_index, candidate_index]
            ), f"Cross-product mismatch at ({query_index}, {candidate_index})"
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("queries_count", [1, 4, 8])
def test_levenshtein_symmetric_self_similarity(device_name: DeviceName, queries_count: int):
    seed_random_generators(7)
    query_strings = [get_random_string(length=randint(4, 20)) for _ in range(queries_count)]
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.LevenshteinDistances(capabilities=base_caps)
    matrix = engine(Strs(query_strings), device=device_scope)
    assert matrix.shape == (queries_count, queries_count)
    assert matrix.dtype == np.uint64
    assert np.array_equal(np.diagonal(matrix), np.zeros(queries_count, dtype=np.uint64)), "Diagonal must be zero"
    for first_index in range(queries_count):
        for second_index in range(queries_count):
            assert (
                matrix[first_index, second_index] == matrix[second_index, first_index]
            ), f"Matrix must be symmetric at ({first_index}, {second_index})"
    for first_index in range(queries_count):
        for second_index in range(queries_count):
            expected = baseline_levenshtein_distance(query_strings[first_index], query_strings[second_index])
            assert matrix[first_index, second_index] == expected
@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("config", INPUT_SIZE_CONFIGS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_needleman_wunsch_vs_levenshtein_random(
    capabilities_mode: str,
    device_name: DeviceName,
    config: InputSizeConfig,
    seed_value: int,
):
    seed_random_generators(seed_value)
    batch_size, min_len, max_len = generate_string_batches(config)
    a_batch = [get_random_string(length=randint(min_len, max_len)) for _ in range(batch_size)]
    b_batch = [get_random_string(length=randint(min_len, max_len)) for _ in range(batch_size)]
    byte_to_class = (np.arange(256) % 32).astype(np.uint8)
    class_costs = np.full((32, 32), -1, dtype=np.int8)
    np.fill_diagonal(class_costs, 0)
    baselines = np.array([-baseline_levenshtein_distance(a, b) for a, b in zip(a_batch, b_batch)])
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.NeedlemanWunschScores(
        byte_to_class,
        class_costs,
        capabilities=base_caps if capabilities_mode == "base" else device_scope,
        open=-1,
        extend=-1,
    )
    queries, candidates = Strs(a_batch), Strs(b_batch)
    matrix = engine(queries, candidates)
    assert matrix.shape == (batch_size, batch_size)
    np.testing.assert_array_equal(np.diagonal(matrix), baselines, "Edit distances do not match")
@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("batch_size", [1, 7, 33])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_needleman_wunsch_against_affine_gaps(
    capabilities_mode: str,
    device_name: DeviceName,
    batch_size: int,
    seed_value: int,
):
    seed_random_generators(seed_value)
    alphabet = ag.default_proteins_alphabet
    a_batch = [get_random_string(length=randint(5, 50), alphabet=alphabet) for _ in range(batch_size)]
    b_batch = [get_random_string(length=randint(5, 50), alphabet=alphabet) for _ in range(batch_size)]
    baseline = np.array(
        [
            int(
                ag.needleman_wunsch_gotoh_score(
                    a,
                    b,
                    substitution_alphabet=alphabet,
                    substitution_matrix=ag.default_proteins_matrix,
                    gap_opening=ag.default_gap_opening,
                    gap_extension=ag.default_gap_extension,
                )
            )
            for a, b in zip(a_batch, b_batch)
        ],
        dtype=np.int64,
    )
    residue_count = len(alphabet)
    byte_to_class = np.zeros(256, dtype=np.uint8)
    byte_to_class[np.frombuffer(alphabet.encode(), dtype=np.uint8)] = np.arange(1, residue_count + 1, dtype=np.uint8)
    class_costs = np.zeros((32, 32), dtype=np.int8)
    class_costs[1 : residue_count + 1, 1 : residue_count + 1] = ag.default_proteins_matrix[
        :residue_count, :residue_count
    ]
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.NeedlemanWunschScores(
        byte_to_class,
        class_costs,
        capabilities=base_caps if capabilities_mode == "base" else device_scope,
        open=ag.default_gap_opening,
        extend=ag.default_gap_extension,
    )
    matrix = engine(Strs(a_batch), Strs(b_batch), device=device_scope)
    assert matrix.shape == (batch_size, batch_size)
    scores = np.diagonal(matrix)
    if not np.array_equal(scores, baseline):
        idx = int(np.where(scores != baseline)[0][0])
        a, b = a_batch[idx], b_batch[idx]
        aligned_a, aligned_b = ag.needleman_wunsch_gotoh(
            a,
            b,
            substitution_alphabet=alphabet,
            substitution_matrix=ag.default_proteins_matrix,
            gap_open=ag.default_gap_opening,
            gap_extend=ag.default_gap_extension,
        )
        guide_line = "".join("|" if ca == cb else " " for ca, cb in zip(aligned_a, aligned_b))
        pytest.fail(
            "\n".join(
                [
                    f"Needleman-Wunsch mismatch at index {idx}:",
                    f"  a: {a}",
                    f"  b: {b}",
                    f"  szs score:     {int(scores[idx])}",
                    f"  affine_gaps:   {int(baseline[idx])}",
                    "  Alignment (affine_gaps):",
                    f"    {aligned_a}",
                    f"    {guide_line}",
                    f"    {aligned_b}",
                ]
            )
        )
    np.testing.assert_array_equal(scores, baseline)
@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("batch_size", [1, 7, 33])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_smith_waterman_against_affine_gaps(
    capabilities_mode: str,
    device_name: DeviceName,
    batch_size: int,
    seed_value: int,
):
    seed_random_generators(seed_value)
    alphabet = ag.default_proteins_alphabet
    a_batch = [get_random_string(length=randint(5, 50), alphabet=alphabet) for _ in range(batch_size)]
    b_batch = [get_random_string(length=randint(5, 50), alphabet=alphabet) for _ in range(batch_size)]
    baseline = np.array(
        [
            int(
                ag.smith_waterman_gotoh_score(
                    a,
                    b,
                    substitution_alphabet=alphabet,
                    substitution_matrix=ag.default_proteins_matrix,
                    gap_opening=ag.default_gap_opening,
                    gap_extension=ag.default_gap_extension,
                )
            )
            for a, b in zip(a_batch, b_batch)
        ],
        dtype=np.int64,
    )
    residue_count = len(alphabet)
    byte_to_class = np.zeros(256, dtype=np.uint8)
    byte_to_class[np.frombuffer(alphabet.encode(), dtype=np.uint8)] = np.arange(1, residue_count + 1, dtype=np.uint8)
    class_costs = np.zeros((32, 32), dtype=np.int8)
    class_costs[1 : residue_count + 1, 1 : residue_count + 1] = ag.default_proteins_matrix[
        :residue_count, :residue_count
    ]
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.SmithWatermanScores(
        byte_to_class,
        class_costs,
        capabilities=base_caps if capabilities_mode == "base" else device_scope,
        open=ag.default_gap_opening,
        extend=ag.default_gap_extension,
    )
    matrix = engine(Strs(a_batch), Strs(b_batch), device=device_scope)
    assert matrix.shape == (batch_size, batch_size)
    scores = np.diagonal(matrix)
    if not np.array_equal(scores, baseline):
        idx = int(np.where(scores != baseline)[0][0])
        a, b = a_batch[idx], b_batch[idx]
        aligned_a, aligned_b = ag.smith_waterman_gotoh(
            a,
            b,
            substitution_alphabet=alphabet,
            substitution_matrix=ag.default_proteins_matrix,
            gap_open=ag.default_gap_opening,
            gap_extend=ag.default_gap_extension,
        )
        guide_line = "".join("|" if ca == cb else " " for ca, cb in zip(aligned_a, aligned_b))
        pytest.fail(
            "\n".join(
                [
                    f"Smith-Waterman mismatch at index {idx}:",
                    f"  a: {a}",
                    f"  b: {b}",
                    f"  szs score:     {int(scores[idx])}",
                    f"  affine_gaps:   {int(baseline[idx])}",
                    "  Alignment (affine_gaps):",
                    f"    {aligned_a}",
                    f"    {guide_line}",
                    f"    {aligned_b}",
                ]
            )
        )
    np.testing.assert_array_equal(scores, baseline)
LONG_BYTE_STRING = "a" * 1200 + "b" * 50
LONG_UTF8_STRING = "あ" * 600 + "い" * 20
DEGENERATE_BYTE_STRINGS = [
    "",
    "a",
    "b",
    "aaaaaaaaaa",
    "abcdefghijklmnopqrstuvwxyz",
    "zyxwvutsrqponmlkjihgfedcba",
    "1234567890" * 5,
    LONG_BYTE_STRING,
]
DEGENERATE_UTF8_STRINGS = [
    "",
    "x",
    "é",
    "🚀",
    "ああああああああああ",
    "abcXYZ123",
    "こんにちは世界" * 5,
    LONG_UTF8_STRING,
]
DIFFERENTIAL_DEVICE_SCOPES = [
    ("default", szs.DeviceScope()),
    ("cpu_cores_2", szs.DeviceScope(cpu_cores=2)),
]
LONG_PROTEIN_STRING = "ARNDCQEGHI" * 60
DEGENERATE_PROTEIN_STRINGS = [
    "",
    "A",
    "W",
    "AAAAAAAAAA",
    ag.default_proteins_alphabet,
    ag.default_proteins_alphabet[::-1],
    "WWWWWWWWWW",
    LONG_PROTEIN_STRING,
]
@pytest.mark.parametrize("device_label, device_scope", DIFFERENTIAL_DEVICE_SCOPES)
@pytest.mark.parametrize("cost_mode", LEVENSHTEIN_COST_MODES)
def test_levenshtein_distances_backend_differential_degenerate_corpus(cost_mode, device_label, device_scope):
    costs = levenshtein_costs_for_mode(cost_mode)
    queries = Strs(DEGENERATE_BYTE_STRINGS)
    candidates = Strs(DEGENERATE_BYTE_STRINGS)
    def make_engine(capabilities):
        return szs.LevenshteinDistances(capabilities=capabilities, **costs)
    matrices = run_across_engines(make_engine, queries, candidates, device=device_scope)
    if cost_mode == "unit":
        oracle_matrix = np.array(
            [
                [
                    baseline_levenshtein_distance(query.encode(), candidate.encode())
                    for candidate in DEGENERATE_BYTE_STRINGS
                ]
                for query in DEGENERATE_BYTE_STRINGS
            ],
            dtype=np.uint64,
        )
    else:
        oracle_matrix = np.array(
            [
                [
                    affine_gap_edit_distance(
                        query.encode(),
                        candidate.encode(),
                        mismatch=costs["mismatch"],
                        open=costs["open"],
                        extend=costs["extend"],
                    )
                    for candidate in DEGENERATE_BYTE_STRINGS
                ]
                for query in DEGENERATE_BYTE_STRINGS
            ],
            dtype=np.uint64,
        )
    assert_engines_agree(matrices, oracle_matrix, context=f" (cost_mode={cost_mode}, device={device_label})")
    empty_matrices = run_across_engines(make_engine, Strs([]), Strs([]), device=device_scope)
    assert_engines_agree(empty_matrices, np.zeros((0, 0), dtype=np.uint64), context=" (empty batch)")
    self_similarity_matrices = run_across_engines(make_engine, queries, device=device_scope)
    assert_engines_agree(self_similarity_matrices, context=" (self-similarity)")
    assert np.array_equal(
        np.diagonal(self_similarity_matrices[0]), np.zeros(len(DEGENERATE_BYTE_STRINGS), dtype=np.uint64)
    ), "Self-similarity diagonal must be zero regardless of cost mode"
@pytest.mark.parametrize("device_label, device_scope", DIFFERENTIAL_DEVICE_SCOPES)
@pytest.mark.parametrize("cost_mode", LEVENSHTEIN_COST_MODES)
def test_levenshtein_distances_utf8_backend_differential_degenerate_corpus(cost_mode, device_label, device_scope):
    costs = levenshtein_costs_for_mode(cost_mode)
    queries = Strs(DEGENERATE_UTF8_STRINGS)
    candidates = Strs(DEGENERATE_UTF8_STRINGS)
    def make_engine(capabilities):
        return szs.LevenshteinDistancesUTF8(capabilities=capabilities, **costs)
    matrices = run_across_engines(make_engine, queries, candidates, device=device_scope)
    if cost_mode == "unit":
        oracle_matrix = np.array(
            [
                [baseline_levenshtein_distance(list(query), list(candidate)) for candidate in DEGENERATE_UTF8_STRINGS]
                for query in DEGENERATE_UTF8_STRINGS
            ],
            dtype=np.uint64,
        )
    else:
        oracle_matrix = np.array(
            [
                [
                    affine_gap_edit_distance(
                        list(query),
                        list(candidate),
                        mismatch=costs["mismatch"],
                        open=costs["open"],
                        extend=costs["extend"],
                    )
                    for candidate in DEGENERATE_UTF8_STRINGS
                ]
                for query in DEGENERATE_UTF8_STRINGS
            ],
            dtype=np.uint64,
        )
    assert_engines_agree(matrices, oracle_matrix, context=f" (cost_mode={cost_mode}, device={device_label})")
    empty_matrices = run_across_engines(make_engine, Strs([]), Strs([]), device=device_scope)
    assert_engines_agree(empty_matrices, np.zeros((0, 0), dtype=np.uint64), context=" (empty batch)")
    self_similarity_matrices = run_across_engines(make_engine, queries, device=device_scope)
    assert_engines_agree(self_similarity_matrices, context=" (self-similarity)")
    assert np.array_equal(
        np.diagonal(self_similarity_matrices[0]), np.zeros(len(DEGENERATE_UTF8_STRINGS), dtype=np.uint64)
    ), "Self-similarity diagonal must be zero regardless of cost mode"
@pytest.mark.parametrize("device_label, device_scope", DIFFERENTIAL_DEVICE_SCOPES)
@pytest.mark.parametrize("cost_mode", ALIGNMENT_COST_MODES)
def test_needleman_wunsch_backend_differential_degenerate_corpus(cost_mode, device_label, device_scope):
    gap_opening, gap_extension = alignment_costs_for_mode(cost_mode)
    alphabet, byte_to_class, class_costs = protein_substitution_tables()
    queries = Strs(DEGENERATE_PROTEIN_STRINGS)
    candidates = Strs(DEGENERATE_PROTEIN_STRINGS)
    def make_engine(capabilities):
        return szs.NeedlemanWunschScores(
            byte_to_class, class_costs, open=gap_opening, extend=gap_extension, capabilities=capabilities
        )
    matrices = run_across_engines(make_engine, queries, candidates, device=device_scope)
    oracle_matrix = np.array(
        [
            [
                ag.needleman_wunsch_gotoh_score(
                    query,
                    candidate,
                    substitution_alphabet=alphabet,
                    substitution_matrix=ag.default_proteins_matrix,
                    gap_opening=gap_opening,
                    gap_extension=gap_extension,
                )
                for candidate in DEGENERATE_PROTEIN_STRINGS
            ]
            for query in DEGENERATE_PROTEIN_STRINGS
        ],
        dtype=np.int64,
    )
    assert_engines_agree(matrices, oracle_matrix, context=f" (cost_mode={cost_mode}, device={device_label})")
    empty_matrices = run_across_engines(make_engine, Strs([]), Strs([]), device=device_scope)
    assert_engines_agree(empty_matrices, np.zeros((0, 0), dtype=np.int64), context=" (empty batch)")
    self_similarity_matrices = run_across_engines(make_engine, queries, device=device_scope)
    assert_engines_agree(self_similarity_matrices, context=" (self-similarity)")
@pytest.mark.parametrize("device_label, device_scope", DIFFERENTIAL_DEVICE_SCOPES)
@pytest.mark.parametrize("cost_mode", ALIGNMENT_COST_MODES)
def test_smith_waterman_backend_differential_degenerate_corpus(cost_mode, device_label, device_scope):
    gap_opening, gap_extension = alignment_costs_for_mode(cost_mode)
    alphabet, byte_to_class, class_costs = protein_substitution_tables()
    queries = Strs(DEGENERATE_PROTEIN_STRINGS)
    candidates = Strs(DEGENERATE_PROTEIN_STRINGS)
    def make_engine(capabilities):
        return szs.SmithWatermanScores(
            byte_to_class, class_costs, open=gap_opening, extend=gap_extension, capabilities=capabilities
        )
    matrices = run_across_engines(make_engine, queries, candidates, device=device_scope)
    oracle_matrix = np.array(
        [
            [
                ag.smith_waterman_gotoh_score(
                    query,
                    candidate,
                    substitution_alphabet=alphabet,
                    substitution_matrix=ag.default_proteins_matrix,
                    gap_opening=gap_opening,
                    gap_extension=gap_extension,
                )
                for candidate in DEGENERATE_PROTEIN_STRINGS
            ]
            for query in DEGENERATE_PROTEIN_STRINGS
        ],
        dtype=np.int64,
    )
    assert_engines_agree(matrices, oracle_matrix, context=f" (cost_mode={cost_mode}, device={device_label})")
    empty_matrices = run_across_engines(make_engine, Strs([]), Strs([]), device=device_scope)
    assert_engines_agree(empty_matrices, np.zeros((0, 0), dtype=np.int64), context=" (empty batch)")
    self_similarity_matrices = run_across_engines(make_engine, queries, device=device_scope)
    assert_engines_agree(self_similarity_matrices, context=" (self-similarity)")
@pytest.mark.parametrize(
    "engine_cls, oracle_fn",
    [
        (szs.NeedlemanWunschScores, ag.needleman_wunsch_gotoh_score),
        (szs.SmithWatermanScores, ag.smith_waterman_gotoh_score),
    ],
    ids=["needleman_wunsch", "smith_waterman"],
)
def test_alignment_out_buffer_matches_returned_matrix(engine_cls, oracle_fn):
    gap_opening, gap_extension = ag.default_gap_opening, ag.default_gap_extension
    alphabet, byte_to_class, class_costs = protein_substitution_tables()
    engine = engine_cls(byte_to_class, class_costs, open=gap_opening, extend=gap_extension)
    queries = Strs(DEGENERATE_PROTEIN_STRINGS)
    candidates = Strs(DEGENERATE_PROTEIN_STRINGS)
    direct_matrix = engine(queries, candidates)
    out_buffer = np.full(direct_matrix.shape, -123456, dtype=np.int64)
    returned_matrix = engine(queries, candidates, out=out_buffer)
    assert returned_matrix is out_buffer, "out= must be returned as-is, not copied into a new array"
    assert np.array_equal(out_buffer, direct_matrix), "out= buffer must be filled with the same scores as a fresh call"
    oracle_matrix = np.array(
        [
            [
                oracle_fn(
                    query,
                    candidate,
                    substitution_alphabet=alphabet,
                    substitution_matrix=ag.default_proteins_matrix,
                    gap_opening=gap_opening,
                    gap_extension=gap_extension,
                )
                for candidate in DEGENERATE_PROTEIN_STRINGS
            ]
            for query in DEGENERATE_PROTEIN_STRINGS
        ],
        dtype=np.int64,
    )
    assert np.array_equal(out_buffer, oracle_matrix)
def test_levenshtein_distances_pyarrow_input_matches_list_input():
    pyarrow = pytest.importorskip("pyarrow")
    word_list = ["hello", "", "world", "a", LONG_BYTE_STRING]
    try:
        queries_from_arrow = Strs(pyarrow.array(word_list))
    except TypeError:
        pytest.skip("sz.Strs does not accept a raw pyarrow.Array in this build")
    queries_from_list = Strs(word_list)
    candidate_list = ["hello world", "", "a"]
    candidates = Strs(candidate_list)
    engine = szs.LevenshteinDistances()
    matrix_from_arrow = engine(queries_from_arrow, candidates)
    matrix_from_list = engine(queries_from_list, candidates)
    assert np.array_equal(matrix_from_arrow, matrix_from_list)
    oracle_matrix = np.array(
        [
            [baseline_levenshtein_distance(query.encode(), candidate.encode()) for candidate in candidate_list]
            for query in word_list
        ],
        dtype=np.uint64,
    )
    assert np.array_equal(matrix_from_arrow, oracle_matrix)
def test_levenshtein_distances_to_device_round_trip():
    if "cuda" not in szs.__capabilities__:
        pytest.skip("CUDA not available, skipping to_device differential test")
    engine = szs.LevenshteinDistances()
    direct_matrix = engine(Strs(DEGENERATE_BYTE_STRINGS), Strs(DEGENERATE_BYTE_STRINGS))
    device_queries = szs.to_device(Strs(DEGENERATE_BYTE_STRINGS))
    device_candidates = szs.to_device(Strs(DEGENERATE_BYTE_STRINGS))
    via_device_matrix = engine(device_queries, device_candidates)
    assert np.array_equal(direct_matrix, via_device_matrix)
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-x", "-s", __file__]))
