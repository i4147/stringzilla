/**
 *  @brief Haswell (AVX2) backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_uncased_fold/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_HASWELL_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_HASWELL_H_
#include "stringzilla/utf8_uncased_fold/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2")
#endif
/**
 *  @brief Detects bytes in the unsigned range [range_start, range_start + range_length).
 *      AVX2 has no unsigned byte compares, so `(x − start) ≤ limit` is realized as
 *      `min_epu8(x − start, limit) == x − start`: the wrap-around subtraction maps the range
 *      onto [0, limit] and `VPMINUB` + `VPCMPEQB` realize the unsigned `≤` in two single-uop
 *      instructions - cheaper and clearer than the sign-flip `VPXOR` + `VPCMPGTB` alternative.
 */
SZ_HELPER_INLINE __m256i sz_haswell_in_byte_range_(__m256i values_u8x32, sz_u8_t range_start, sz_u8_t range_length) {
    __m256i offsets_u8x32 = _mm256_sub_epi8(values_u8x32, _mm256_set1_epi8((char)range_start));
    return _mm256_cmpeq_epi8(_mm256_min_epu8(offsets_u8x32, _mm256_set1_epi8((char)(range_length - 1))), offsets_u8x32);
}
/** @brief Folds ASCII A-Z to a-z across the whole vector via a masked +0x20 (no `VPBLENDVB` needed). */
SZ_HELPER_INLINE __m256i sz_haswell_fold_ascii_(__m256i source_u8x32) {
    __m256i is_ascii_upper_u8x32 = sz_haswell_in_byte_range_(source_u8x32, 'A', 26);
    return _mm256_add_epi8(source_u8x32, _mm256_and_si256(is_ascii_upper_u8x32, _mm256_set1_epi8(0x20)));
}
/**
 *  @brief Shifts the 32 source bytes right by @p byte_offset lanes, so lane `i` holds byte `i − offset`.
 *      Positions before the chunk receive zeros - safe, because the main loop always advances by whole
 *      characters, so a chunk never starts with a continuation byte that would need its true predecessor.
 *      AVX2 `VPALIGNR` works per 128-bit lane, so a `VPERM2I128` first materializes the cross-lane carry.
 */
SZ_HELPER_INLINE __m256i sz_haswell_previous_bytes_(__m256i source_u8x32, int byte_offset) {
    __m256i carry_u8x32 = _mm256_permute2x128_si256(source_u8x32, source_u8x32, 0x08); 
    return byte_offset == 1 ? _mm256_alignr_epi8(source_u8x32, carry_u8x32, 15)
                            : _mm256_alignr_epi8(source_u8x32, carry_u8x32, 14);
}
/**
 *  @brief Shifts the 32 source bytes left by one lane, so lane `i` holds byte `i + 1`.
 *      Lane 31 receives zero; any 2-byte lead there is trimmed as incomplete before folding anyway.
 */
SZ_HELPER_INLINE __m256i sz_haswell_next_bytes_(__m256i source_u8x32) {
    __m256i carry_u8x32 = _mm256_permute2x128_si256(source_u8x32, source_u8x32, 0x81); 
    return _mm256_alignr_epi8(carry_u8x32, source_u8x32, 1);
}
/** @brief First N bits set; BZHI keeps `n == 32` defined, unlike the `(1 << n) − 1` idiom. */
SZ_HELPER_INLINE sz_u32_t sz_haswell_mask_until_(sz_size_t n) { return (sz_u32_t)_bzhi_u32(0xFFFFFFFFu, (unsigned)n); }
/**
 *  @brief Folds a 32-byte chunk of caseless multi-byte scripts mixed with ASCII.
 *      Folds ASCII A-Z in place and copies everything else, truncating before the first lead of
 *      a case-aware family and trimming incomplete trailing sequences. Stores a full 32-byte
 *      vector and reports how many bytes were consumed: the destination is contractually ≥ 3×
 *      the source, so the overshoot is always in bounds and the next chunk's store (or the
 *      serial tail) overwrites it.
 *  @param is_foreign_lead_mask Lead bytes outside the caseless family: the chunk is truncated
 *      before the first such lead, so Hebrew or CJK text with an embedded Latin or Cyrillic
 *      word still copies its longest caseless prefix vectorized.
 *  @return Bytes consumed and written, or zero if the first character needs another handler.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_haswell_caseless_chunk_( 
    __m256i source_u8x32, sz_u32_t is_two_byte_lead_mask, sz_u32_t is_three_byte_lead_mask,
    sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {
    sz_size_t fold_length = is_foreign_lead_mask ? (sz_size_t)_tzcnt_u32(is_foreign_lead_mask) : 32;
    sz_u32_t incomplete_mask = 
        (is_two_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0)) |
        (is_three_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 2 ? fold_length - 2 : 0));
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)_tzcnt_u32(incomplete_mask);
    if (fold_length == 0) return 0;
    _mm256_storeu_si256((__m256i *)target, sz_haswell_fold_ascii_(source_u8x32));
    return fold_length;
}
/**
 *  @brief Folds a 32-byte chunk of Latin text in place: ASCII, Latin-1 Supplement (C2-C3),
 *      Latin Extended-A/B (C4-C6), and Latin Extended Additional (E1 B8-BB) - the working set
 *      of German, Czech, Vietnamese, and most other Latin-script languages.
 *
 *  Latin Extended folding is parity-based: uppercase codepoints fold to the adjacent codepoint,
 *  and the codepoint's low bit lives in the last byte of its UTF-8 sequence, so the fold is an
 *  in-place masked +1. Where Ice Lake reads per-codepoint deltas from `VPERMB` tables, AVX2
 *  re-derives them arithmetically: within C4 and C5 the foldable continuations form a few
 *  contiguous even-parity or odd-parity runs, so two or three range compares per lead replace the
 *  table. C6 (Latin Ext-B 'ƀ'-'ƿ') is too fragmented for ranges - 16 scattered +1 pairs and 20
 *  irregulars - so its two 64-entry membership bitsets are decomposed into 16-entry `VPSHUFB`
 *  nibble lookups: `bitmap[low_nibble] & (1 << high_quadrant)` tests membership in five ops,
 *  versus ~12 compares for the explicit ranges. Irregular codepoints (those that expand, shrink,
 *  or fold across lead bytes) and foreign E1 sub-families (Georgian, Greek Extended) truncate the
 *  chunk before the character's lead and route one rune to the serial fallback.
 *
 *  All folds are applied across the full vector and a full 32-byte store is issued; lanes at or
 *  beyond the consumed length hold garbage that the next chunk's store overwrites (the destination
 *  is contractually ≥ 3× the source, so the overshoot stays in bounds).
 *
 *  @param is_foreign_lead_mask Lead bytes outside this handler's families: the chunk is truncated
 *      before the first such lead, so mixed-script chunks still fold their longest pure prefix
 *      vectorized instead of degrading to one-rune serial steps per chunk.
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_haswell_latin_chunk_( 
    __m256i source_u8x32, sz_u32_t is_continuation_mask, sz_u32_t is_three_byte_lead_mask,
    sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {
    __m256i previous_bytes_u8x32 = sz_haswell_previous_bytes_(source_u8x32, 1);
    __m256i second_previous_bytes_u8x32 = sz_haswell_previous_bytes_(source_u8x32, 2);
    __m256i next_bytes_u8x32 = sz_haswell_next_bytes_(source_u8x32);
    __m256i is_after_c2_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xC2));
    __m256i is_after_c3_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xC3));
    __m256i is_after_c4_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xC4));
    __m256i is_after_c5_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xC5));
    __m256i is_after_c6_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xC6));
    __m256i is_after_e1_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xE1));
    __m256i is_odd_byte_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(source_u8x32, _mm256_set1_epi8(0x01)),
                                                  _mm256_set1_epi8(0x01));
    __m256i c4_fold_u8x32 = _mm256_and_si256(
        is_after_c4_u8x32, _mm256_or_si256(_mm256_andnot_si256(is_odd_byte_u8x32, 
                                                               sz_haswell_in_byte_range_(source_u8x32, 0x80, 0x38)),
                                           _mm256_and_si256(is_odd_byte_u8x32, 
                                                            sz_haswell_in_byte_range_(source_u8x32, 0xB9, 0x05))));
    __m256i c4_irregular_u8x32 = _mm256_and_si256(
        is_after_c4_u8x32, _mm256_or_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xB0)),
                                           _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xBF))));
    __m256i c5_fold_u8x32 = _mm256_and_si256(
        is_after_c5_u8x32,
        _mm256_or_si256(_mm256_and_si256(is_odd_byte_u8x32, 
                                         _mm256_or_si256(sz_haswell_in_byte_range_(source_u8x32, 0x81, 0x07),
                                                         sz_haswell_in_byte_range_(source_u8x32, 0xB9, 0x05))),
                        _mm256_andnot_si256(is_odd_byte_u8x32, 
                                            sz_haswell_in_byte_range_(source_u8x32, 0x8A, 0x2D))));
    __m256i c5_irregular_u8x32 = _mm256_and_si256(
        is_after_c5_u8x32,
        _mm256_or_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0x89)),
                        _mm256_or_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xB8)),
                                        _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xBF)))));
    __m256i const c6_fold_bitmap_lut_u8x32 = _mm256_setr_epi8( 
        4, 2, 5, 8, 5, 8, 0, 5, 10, 0, 0, 1, 12, 0, 0, 4,      
        4, 2, 5, 8, 5, 8, 0, 5, 10, 0, 0, 1, 12, 0, 0, 4);
    __m256i const c6_irregular_bitmap_lut_u8x32 = _mm256_setr_epi8( 
        2, 9, 8, 2, 2, 0, 7, 10, 0, 5, 1, 0, 2, 2, 5, 3,            
        2, 9, 8, 2, 2, 0, 7, 10, 0, 5, 1, 0, 2, 2, 5, 3);
    __m256i const quadrant_bit_lut_u8x32 = _mm256_setr_epi8( 
        1, 2, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,      
        1, 2, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i low_nibbles_u8x32 = _mm256_and_si256(source_u8x32, _mm256_set1_epi8(0x0F));
    __m256i quadrant_indices_u8x32 = _mm256_and_si256(_mm256_srli_epi16(source_u8x32, 4), _mm256_set1_epi8(0x03));
    __m256i quadrant_bits_u8x32 = _mm256_shuffle_epi8(quadrant_bit_lut_u8x32, quadrant_indices_u8x32);
    __m256i c6_fold_u8x32 = _mm256_and_si256(
        is_after_c6_u8x32, _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(c6_fold_bitmap_lut_u8x32, 
                                                                                  low_nibbles_u8x32),
                                                              quadrant_bits_u8x32),
                                             quadrant_bits_u8x32));
    __m256i c6_irregular_u8x32 = _mm256_and_si256(
        is_after_c6_u8x32,
        _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(c6_irregular_bitmap_lut_u8x32, 
                                                               low_nibbles_u8x32),
                                           quadrant_bits_u8x32),
                          quadrant_bits_u8x32));
    __m256i is_b8_bb_second_u8x32 = sz_haswell_in_byte_range_(source_u8x32, 0xB8, 0x04);
    __m256i foreign_e1_second_u8x32 = _mm256_andnot_si256(is_b8_bb_second_u8x32, is_after_e1_u8x32);
    __m256i e1_latin_third_u8x32 = _mm256_and_si256( 
        _mm256_cmpeq_epi8(second_previous_bytes_u8x32, _mm256_set1_epi8((char)0xE1)),
        sz_haswell_in_byte_range_(previous_bytes_u8x32, 0xB8, 0x04));
    __m256i e1_irregular_u8x32 = _mm256_and_si256(
        _mm256_and_si256(e1_latin_third_u8x32, _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xBA))),
        sz_haswell_in_byte_range_(source_u8x32, 0x96, 0x09));
    sz_u32_t stop_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_or_si256(
        _mm256_or_si256(c4_irregular_u8x32, c5_irregular_u8x32),
        _mm256_or_si256(c6_irregular_u8x32, _mm256_or_si256(e1_irregular_u8x32, foreign_e1_second_u8x32))));
    stop_mask |= is_foreign_lead_mask;
    sz_size_t fold_length = 32;
    if (stop_mask) {
        sz_size_t first_flagged_position = (sz_size_t)_tzcnt_u32(stop_mask);
        while (first_flagged_position && ((is_continuation_mask >> first_flagged_position) & 1))
            --first_flagged_position;
        fold_length = first_flagged_position;
    }
    sz_u32_t is_two_byte_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
        sz_haswell_in_byte_range_(source_u8x32, 0xC2, 0x05));
    sz_u32_t incomplete_mask = 
        (is_two_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0)) |
        (is_three_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 2 ? fold_length - 2 : 0));
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)_tzcnt_u32(incomplete_mask);
    if (fold_length == 0) return 0;
    __m256i folded_u8x32 = sz_haswell_fold_ascii_(source_u8x32);
    __m256i is_latin1_upper_u8x32 = _mm256_andnot_si256(
        _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0x97)),
        _mm256_and_si256(is_after_c3_u8x32, sz_haswell_in_byte_range_(source_u8x32, 0x80, 0x1F)));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_latin1_upper_u8x32, _mm256_set1_epi8(0x20)));
    __m256i is_eszett_second_u8x32 = _mm256_and_si256(is_after_c3_u8x32,
                                                      _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0x9F)));
    __m256i is_eszett_lead_u8x32 = _mm256_and_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xC3)),
                                                    _mm256_cmpeq_epi8(next_bytes_u8x32, _mm256_set1_epi8((char)0x9F)));
    folded_u8x32 = _mm256_sub_epi8(folded_u8x32, _mm256_and_si256(is_eszett_lead_u8x32, _mm256_set1_epi8(0x50)));
    folded_u8x32 = _mm256_sub_epi8(folded_u8x32, _mm256_and_si256(is_eszett_second_u8x32, _mm256_set1_epi8(0x2C)));
    __m256i is_micro_second_u8x32 = _mm256_and_si256(is_after_c2_u8x32,
                                                     _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xB5)));
    __m256i is_micro_lead_u8x32 = _mm256_and_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xC2)),
                                                   _mm256_cmpeq_epi8(next_bytes_u8x32, _mm256_set1_epi8((char)0xB5)));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_micro_lead_u8x32, _mm256_set1_epi8(0x0C)));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_micro_second_u8x32, _mm256_set1_epi8(0x07)));
    __m256i extended_fold_u8x32 = _mm256_or_si256(_mm256_or_si256(c4_fold_u8x32, c5_fold_u8x32), c6_fold_u8x32);
    __m256i e1_fold_u8x32 = _mm256_andnot_si256(is_odd_byte_u8x32, e1_latin_third_u8x32);
    folded_u8x32 = _mm256_add_epi8(
        folded_u8x32, _mm256_and_si256(_mm256_or_si256(extended_fold_u8x32, e1_fold_u8x32), _mm256_set1_epi8(0x01)));
    _mm256_storeu_si256((__m256i *)target, folded_u8x32);
    return fold_length;
}
/**
 *  @brief Folds a 32-byte chunk of basic Cyrillic (D0/D1 leads, U+0400-045F) mixed with ASCII -
 *      the working set of Russian, Ukrainian, Bulgarian, and Serbian text.
 *
 *  The uppercase block maps onto the lowercase block with offsets keyed purely by the second
 *  byte's high nibble, so one `VPSHUFB` over a 16-entry table replaces three range compares
 *  plus three masked adds:
 *
 *  Input Range | Codepoints        | Output           | Transform
 *  D0 80-8F    | Ѐ-Џ (U+0400-040F) | D1 90-9F (ѐ-џ)   | second +0x10, lead D0 → D1
 *  D0 90-9F    | А-П (U+0410-041F) | D0 B0-BF (а-п)   | second +0x20
 *  D0 A0-AF    | Р-Я (U+0420-042F) | D1 80-8F (р-я)   | second −0x20, lead D0 → D1
 *  D0 B0-BF    | а-п lowercase     | unchanged        | high nibble B → offset 0
 *  D1 80-9F    | р-џ lowercase     | unchanged        | only after-D0 lanes take offsets
 *  D1 A0+      | Ext-A (U+0460+)   | → serial         | +1 parity folds, not modeled here
 *
 *  The D0 → D1 lead fixup is a masked +1 (D0 + 1 = D1), keeping the no-`VPBLENDVB` discipline.
 *  Cyrillic Extended-A and foreign-family leads (Russian quotes «» are C2, dashes are E2)
 *  truncate the chunk before the offending lead, so mixed chunks still fold their longest
 *  pure prefix vectorized. A full 32-byte store is issued; lanes at or beyond the consumed
 *  length are overwritten by the next chunk's store (the destination is ≥ 3× the source).
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_haswell_cyrillic_chunk_( 
    __m256i source_u8x32, sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {
    __m256i previous_bytes_u8x32 = sz_haswell_previous_bytes_(source_u8x32, 1);
    __m256i next_bytes_u8x32 = sz_haswell_next_bytes_(source_u8x32);
    __m256i is_after_d0_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xD0));
    __m256i is_after_d1_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xD1));
    sz_u32_t is_extended_second_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_and_si256(is_after_d1_u8x32, sz_haswell_in_byte_range_(source_u8x32, 0xA0, 0x20)));
    sz_u32_t stop_mask = is_foreign_lead_mask | (is_extended_second_mask >> 1);
    sz_size_t fold_length = stop_mask ? (sz_size_t)_tzcnt_u32(stop_mask) : 32;
    sz_u32_t is_cyrillic_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
        sz_haswell_in_byte_range_(source_u8x32, 0xD0, 0x02));
    sz_u32_t incomplete_mask = is_cyrillic_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0);
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)_tzcnt_u32(incomplete_mask);
    if (fold_length == 0) return 0;
    __m256i const cyrillic_offset_lut_u8x32 = _mm256_setr_epi8(        
        0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, (char)0xE0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, (char)0xE0, 0, 0, 0, 0, 0);
    __m256i high_nibbles_u8x32 = _mm256_and_si256(_mm256_srli_epi16(source_u8x32, 4), _mm256_set1_epi8(0x0F));
    __m256i offsets_u8x32 = _mm256_shuffle_epi8(cyrillic_offset_lut_u8x32, high_nibbles_u8x32);
    __m256i folded_u8x32 = sz_haswell_fold_ascii_(source_u8x32);
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_after_d0_u8x32, offsets_u8x32));
    __m256i is_d0_u8x32 = _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xD0));
    __m256i needs_d1_u8x32 = _mm256_and_si256(is_d0_u8x32,
                                              _mm256_or_si256(sz_haswell_in_byte_range_(next_bytes_u8x32, 0x80, 0x10),
                                                              sz_haswell_in_byte_range_(next_bytes_u8x32, 0xA0, 0x10)));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(needs_d1_u8x32, _mm256_set1_epi8(0x01)));
    _mm256_storeu_si256((__m256i *)target, folded_u8x32);
    return fold_length;
}
/**
 *  @brief Folds a 32-byte chunk of basic Greek (CE/CF leads, U+0370-03FF) mixed with ASCII.
 *
 *  Greek CE/CF uppercase → lowercase transformations:
 *
 *  Input Range  | Codepoints        | Output           | Transform
 *  CE 91-9F     | Α-Ο (U+0391-039F) | CE B1-BF (α-ο)   | second +0x20
 *  CE A0-A1     | Π-Ρ (U+03A0-03A1) | CF 80-81 (π-ρ)   | second −0x20, lead CE → CF
 *  CE A3-AB     | Σ-Ϋ (U+03A3-03AB) | CF 83-8B (σ-ϋ)   | second −0x20, lead CE → CF
 *  CE B1-BF     | α-ο lowercase     | unchanged        | —
 *  CF 80-8B     | π-ϋ lowercase     | unchanged        | —
 *  CF 82        | ς final sigma     | CF 83 (σ)        | second +1
 *
 *  CE A2 (U+03A2, unassigned) is deliberately neither folded nor flagged. The fold-side
 *  exclusion set is narrower than the finder's: only sequences whose FOLD is irregular leave
 *  the fast path - CE 80-90 (tonos and accented capitals with non-uniform offsets), CE B0
 *  ('ΰ', which expands to three codepoints), and CF 8C+ (accented lowercase and archaic
 *  symbols like 'ϐ'/'ϑ' that fold onto basic letters) and foreign-family leads truncate the
 *  chunk before the offending lead, so mixed chunks still fold their longest pure prefix
 *  vectorized. The CE → CF lead fixup is a masked +1.
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_haswell_greek_chunk_( 
    __m256i source_u8x32, sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {
    __m256i previous_bytes_u8x32 = sz_haswell_previous_bytes_(source_u8x32, 1);
    __m256i next_bytes_u8x32 = sz_haswell_next_bytes_(source_u8x32);
    __m256i is_after_ce_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xCE));
    __m256i is_after_cf_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xCF));
    __m256i ce_irregular_u8x32 = _mm256_and_si256(
        is_after_ce_u8x32, _mm256_or_si256(sz_haswell_in_byte_range_(source_u8x32, 0x80, 0x11),
                                           _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xB0))));
    __m256i cf_irregular_u8x32 = _mm256_and_si256(is_after_cf_u8x32,
                                                  sz_haswell_in_byte_range_(source_u8x32, 0x8F, 0x31));
    sz_u32_t is_irregular_second_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_or_si256(ce_irregular_u8x32, cf_irregular_u8x32));
    sz_u32_t stop_mask = is_foreign_lead_mask | (is_irregular_second_mask >> 1);
    sz_size_t fold_length = stop_mask ? (sz_size_t)_tzcnt_u32(stop_mask) : 32;
    sz_u32_t is_greek_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_u8x32, 0xCE, 0x02));
    sz_u32_t incomplete_mask = is_greek_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0);
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)_tzcnt_u32(incomplete_mask);
    if (fold_length == 0) return 0;
    __m256i folded_u8x32 = sz_haswell_fold_ascii_(source_u8x32);
    __m256i plus_fold_u8x32 = _mm256_and_si256(is_after_ce_u8x32, sz_haswell_in_byte_range_(source_u8x32, 0x91, 0x0F));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(plus_fold_u8x32, _mm256_set1_epi8(0x20)));
    __m256i minus_second_u8x32 = _mm256_or_si256(sz_haswell_in_byte_range_(source_u8x32, 0xA0, 0x02),
                                                 sz_haswell_in_byte_range_(source_u8x32, 0xA3, 0x09));
    __m256i minus_fold_u8x32 = _mm256_and_si256(is_after_ce_u8x32, minus_second_u8x32);
    folded_u8x32 = _mm256_sub_epi8(folded_u8x32, _mm256_and_si256(minus_fold_u8x32, _mm256_set1_epi8(0x20)));
    __m256i is_ce_u8x32 = _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xCE));
    __m256i needs_cf_u8x32 = _mm256_and_si256(is_ce_u8x32,
                                              _mm256_or_si256(sz_haswell_in_byte_range_(next_bytes_u8x32, 0xA0, 0x02),
                                                              sz_haswell_in_byte_range_(next_bytes_u8x32, 0xA3, 0x09)));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(needs_cf_u8x32, _mm256_set1_epi8(0x01)));
    __m256i is_final_sigma_u8x32 = _mm256_and_si256(is_after_cf_u8x32,
                                                    _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0x82)));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_final_sigma_u8x32, _mm256_set1_epi8(0x01)));
    _mm256_storeu_si256((__m256i *)target, folded_u8x32);
    return fold_length;
}
/**
 *  @brief Folds a 32-byte chunk of Georgian (E1 82/83 leads, U+10A0-10FF) mixed with ASCII -
 *      the working set of Georgian text.
 *
 *  Georgian uppercase (Mtavruli/Asomtavruli) lives in two E1 82/83 ranges that fold by
 *  rewriting all three bytes; the lowercase Mkhedruli letters under E1 82/83 are identity-folding
 *  and copy through unchanged:
 *
 *  Input Range  | Codepoints           | Output             | Transform
 *  E1 82 A0-BF  | Ⴀ-Ⴟ (U+10A0-10BF)    | E2 B4 80-9F (ⴀ-ⴟ)  | lead E1 → E2, second 82 → B4, third −0x20
 *  E1 83 80-85  | Ⴠ-Ⴥ (U+10C0-10C5)    | E2 B4 A0-A5 (ⴠ-ⴥ)  | lead E1 → E2, second 83 → B4, third +0x20
 *  E1 83 87     | Ⴧ (U+10C7)           | E2 B4 A7 (ⴧ)       | lead E1 → E2, second 83 → B4, third +0x20
 *  E1 83 8D     | Ⴭ (U+10CD)           | E2 B4 AD (ⴭ)       | lead E1 → E2, second 83 → B4, third +0x20
 *  E1 82/83 …   | Mkhedruli lowercase  | unchanged          | —
 *
 *  Every fold is a 3-byte → 3-byte rewrite, so it stays in place. The uppercase classification
 *  lives at the third byte, so the masked deltas are derived there and propagated one and two
 *  lanes back (via `next_bytes`) to the second and lead bytes - mirroring the Armenian finder's
 *  `next_bytes` flag propagation. The lead rewrite E1 → E2 is a masked +1; the second rewrites
 *  82 → B4 (+0x32) and 83 → B4 (+0x31) are masked adds, keeping the no-`VPBLENDVB` discipline.
 *
 *  This handler runs AFTER the Latin handler, which already consumes E1 B8-BB (Latin Extended
 *  Additional). Non-Georgian E1 sub-families - Greek Extended (E1 BC-BF) and any other E1
 *  second byte - and foreign-family leads truncate the chunk before the offending lead, so a
 *  mixed chunk still folds its longest pure Georgian prefix vectorized. A full 32-byte store is
 *  issued; lanes at or beyond the consumed length are overwritten by the next chunk's store
 *  (the destination is ≥ 3× the source).
 *
 *  @return Bytes consumed and written, or zero if the first character needs another handler.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_haswell_georgian_chunk_( 
    __m256i source_u8x32, sz_u32_t is_three_byte_lead_mask, sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {
    __m256i previous_bytes_u8x32 = sz_haswell_previous_bytes_(source_u8x32, 1);
    __m256i second_previous_bytes_u8x32 = sz_haswell_previous_bytes_(source_u8x32, 2);
    __m256i next_bytes_u8x32 = sz_haswell_next_bytes_(source_u8x32);
    __m256i is_e1_u8x32 = _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xE1));
    __m256i is_georgian_second_u8x32 = _mm256_or_si256(
        _mm256_cmpeq_epi8(next_bytes_u8x32, _mm256_set1_epi8((char)0x82)),
        _mm256_cmpeq_epi8(next_bytes_u8x32, _mm256_set1_epi8((char)0x83)));
    __m256i foreign_e1_lead_u8x32 = _mm256_andnot_si256(is_georgian_second_u8x32, is_e1_u8x32);
    sz_u32_t stop_mask = is_foreign_lead_mask | (sz_u32_t)_mm256_movemask_epi8(foreign_e1_lead_u8x32);
    sz_size_t fold_length = stop_mask ? (sz_size_t)_tzcnt_u32(stop_mask) : 32;
    sz_u32_t incomplete_mask = is_three_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 2 ? fold_length - 2 : 0);
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)_tzcnt_u32(incomplete_mask);
    if (fold_length == 0) return 0;
    __m256i is_e1_two_back_u8x32 = _mm256_cmpeq_epi8(second_previous_bytes_u8x32, _mm256_set1_epi8((char)0xE1));
    __m256i is_82_one_back_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0x82));
    __m256i is_83_one_back_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0x83));
    __m256i is_82_upper_third_u8x32 = _mm256_and_si256(_mm256_and_si256(is_e1_two_back_u8x32, is_82_one_back_u8x32),
                                                       sz_haswell_in_byte_range_(source_u8x32, 0xA0, 0x20));
    __m256i is_83_upper_range_u8x32 = _mm256_or_si256(
        sz_haswell_in_byte_range_(source_u8x32, 0x80, 0x06),
        _mm256_or_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0x87)),
                        _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0x8D))));
    __m256i is_83_upper_third_u8x32 = _mm256_and_si256(_mm256_and_si256(is_e1_two_back_u8x32, is_83_one_back_u8x32),
                                                       is_83_upper_range_u8x32);
    __m256i folded_u8x32 = sz_haswell_fold_ascii_(source_u8x32);
    folded_u8x32 = _mm256_sub_epi8(folded_u8x32, _mm256_and_si256(is_82_upper_third_u8x32, _mm256_set1_epi8(0x20)));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_83_upper_third_u8x32, _mm256_set1_epi8(0x20)));
    __m256i is_82_upper_second_u8x32 = sz_haswell_next_bytes_(is_82_upper_third_u8x32);
    __m256i is_83_upper_second_u8x32 = sz_haswell_next_bytes_(is_83_upper_third_u8x32);
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_82_upper_second_u8x32, _mm256_set1_epi8(0x32)));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_83_upper_second_u8x32, _mm256_set1_epi8(0x31)));
    __m256i is_upper_lead_u8x32 = sz_haswell_next_bytes_(
        _mm256_or_si256(is_82_upper_second_u8x32, is_83_upper_second_u8x32));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_upper_lead_u8x32, _mm256_set1_epi8(0x01)));
    _mm256_storeu_si256((__m256i *)target, folded_u8x32);
    return fold_length;
}
/**
 *  @brief Folds a 32-byte chunk mixing ASCII with caseless and guarded 3-byte scripts -
 *      CJK with E2 punctuation, German quotes („…“), Korean, and similar real-world blends.
 *
 *  The guarded leads (E2, EA) are case-aware only for specific second bytes, so the chunk
 *  is a fold-ASCII-and-copy as long as every guarded sequence is provably caseless:
 *    - E2 is safe for seconds 80-83 (General Punctuation - quotes, dashes, ellipses);
 *      other E2 blocks (84 Letterlike Kelvin/Angstrom, B0-B3 Glagolitic/Coptic, …) fold;
 *    - EA is safe except seconds 99-9F (Cyrillic Ext-B, Latin Ext-D) and AD-AE (Cherokee
 *      Supplement) - Hangul (EA B0+) passes untouched;
 *    - EF is never safe: fullwidth A-Z lives under EF BC and folds by +0x20.
 *  The first unsafe or foreign-family lead truncates the chunk in place - no walk-back
 *  needed, the flagged lane IS the lead - and one rune goes to the serial fallback.
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_haswell_guarded_chunk_( 
    __m256i source_u8x32, sz_u32_t is_two_byte_lead_mask, sz_u32_t is_three_byte_lead_mask,
    sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {
    __m256i next_bytes_u8x32 = sz_haswell_next_bytes_(source_u8x32);
    __m256i is_e2_u8x32 = _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xE2));
    __m256i is_ea_u8x32 = _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xEA));
    __m256i is_ef_u8x32 = _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xEF));
    __m256i unsafe_e2_u8x32 = _mm256_andnot_si256(sz_haswell_in_byte_range_(next_bytes_u8x32, 0x80, 0x04), is_e2_u8x32);
    __m256i unsafe_ea_u8x32 = _mm256_and_si256(
        is_ea_u8x32, _mm256_or_si256(sz_haswell_in_byte_range_(next_bytes_u8x32, 0x99, 0x07),
                                     sz_haswell_in_byte_range_(next_bytes_u8x32, 0xAD, 0x02)));
    sz_u32_t stop_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_or_si256(_mm256_or_si256(unsafe_e2_u8x32, unsafe_ea_u8x32), is_ef_u8x32));
    stop_mask |= is_foreign_lead_mask;
    sz_size_t fold_length = stop_mask ? (sz_size_t)_tzcnt_u32(stop_mask) : 32;
    sz_u32_t incomplete_mask = 
        (is_two_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0)) |
        (is_three_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 2 ? fold_length - 2 : 0));
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)_tzcnt_u32(incomplete_mask);
    if (fold_length == 0) return 0;
    _mm256_storeu_si256((__m256i *)target, sz_haswell_fold_ascii_(source_u8x32));
    return fold_length;
}
/**
 *  @brief Folds a 32-byte chunk of Armenian (D4-D6 leads) mixed with ASCII, plus the Cyrillic
 *      Supplement (D4 80-AF, U+0500-052F) that shares the D4 lead.
 *
 *  Armenian uppercase spans two lead bytes and folds into three target blocks, and the D4 lead
 *  additionally carries the Cyrillic Supplement, whose uppercase folds by +1 parity:
 *
 *  Input Range  | Codepoints           | Output             | Transform
 *  D4 80-AF     | Ԁ-ԯ (U+0500-052F)    | even +1 (Ԁ→ԁ …)    | even second byte +1 (Cyrillic Supplement)
 *  D4 B1-BF     | Ա-Ձ (U+0531-0541)    | D5 A1-AF (ա-ձ)     | second −0x10, lead D4 → D5
 *  D5 80-8F     | Ղ-Տ (U+0542-054F)    | D5 B0-BF (ղ-տ)     | second +0x30, lead unchanged
 *  D5 90-96     | Ր-Ֆ (U+0550-0556)    | D6 80-86 (ր-ֆ)     | second −0x10, lead D5 → D6
 *  D6 87        | և (U+0587)           | → serial           | folds to "եւ" (4 bytes) - expands
 *
 *  Both −0x10 classes also bump their lead by one block (D4 → D5, D5 → D6), realized as a masked
 *  +1; the +0x30 class leaves its lead unchanged. The Armenian classification lives at the second
 *  byte, so the lead +1 flag propagates one lane back via `next_bytes`, mirroring the case-
 *  insensitive Armenian finder's `..._armenian_fold_ymm_`. The Ech-Yiwn ligature 'և' (D6 87) is
 *  the only D4-D6 codepoint whose fold changes byte length, so its lead truncates the chunk and
 *  routes one rune to the serial fallback. Non-Armenian complex leads (Cyrillic Extension D2-D3,
 *  IPA/Ext-B C7-CD, 4-byte F0+) and foreign-family leads truncate before their lead, so a mixed
 *  chunk still folds its longest pure prefix vectorized. A full 32-byte store is issued; lanes at
 *  or beyond the consumed length are overwritten by the next chunk's store (destination ≥ 3× source).
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_haswell_armenian_chunk_( 
    __m256i source_u8x32, sz_u32_t is_lead_mask, sz_u32_t malformed_lead_mask, sz_ptr_t target) {
    __m256i previous_bytes_u8x32 = sz_haswell_previous_bytes_(source_u8x32, 1);
    __m256i is_after_d4_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xD4));
    __m256i is_after_d5_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xD5));
    __m256i is_after_d6_u8x32 = _mm256_cmpeq_epi8(previous_bytes_u8x32, _mm256_set1_epi8((char)0xD6));
    sz_u32_t expansion_second_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_and_si256(is_after_d6_u8x32, _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0x87))));
    sz_u32_t is_armenian_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
        sz_haswell_in_byte_range_(source_u8x32, 0xD4, 0x03));
    sz_u32_t stop_mask = (is_lead_mask & ~is_armenian_lead_mask) | malformed_lead_mask | (expansion_second_mask >> 1);
    sz_size_t fold_length = stop_mask ? (sz_size_t)_tzcnt_u32(stop_mask) : 32;
    sz_u32_t incomplete_mask = is_armenian_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0);
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)_tzcnt_u32(incomplete_mask);
    if (fold_length == 0) return 0;
    __m256i is_odd_byte_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(source_u8x32, _mm256_set1_epi8(0x01)),
                                                  _mm256_set1_epi8(0x01));
    __m256i is_d4_cyrillic_even_u8x32 = _mm256_andnot_si256(
        is_odd_byte_u8x32, _mm256_and_si256(is_after_d4_u8x32, sz_haswell_in_byte_range_(source_u8x32, 0x80, 0x30)));
    __m256i is_d4_armenian_u8x32 = _mm256_and_si256(is_after_d4_u8x32,
                                                    sz_haswell_in_byte_range_(source_u8x32, 0xB1, 0x4F));
    __m256i is_d5_low_u8x32 = _mm256_and_si256(is_after_d5_u8x32, sz_haswell_in_byte_range_(source_u8x32, 0x80, 0x10));
    __m256i is_d5_high_u8x32 = _mm256_and_si256(is_after_d5_u8x32, sz_haswell_in_byte_range_(source_u8x32, 0x90, 0x07));
    __m256i folded_u8x32 = sz_haswell_fold_ascii_(source_u8x32);
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_d4_cyrillic_even_u8x32, _mm256_set1_epi8(0x01)));
    __m256i is_minus_10_u8x32 = _mm256_or_si256(is_d4_armenian_u8x32, is_d5_high_u8x32);
    folded_u8x32 = _mm256_sub_epi8(folded_u8x32, _mm256_and_si256(is_minus_10_u8x32, _mm256_set1_epi8(0x10)));
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(is_d5_low_u8x32, _mm256_set1_epi8(0x30)));
    __m256i lead_plus_one_u8x32 = sz_haswell_next_bytes_(is_minus_10_u8x32);
    folded_u8x32 = _mm256_add_epi8(folded_u8x32, _mm256_and_si256(lead_plus_one_u8x32, _mm256_set1_epi8(0x01)));
    _mm256_storeu_si256((__m256i *)target, folded_u8x32);
    return fold_length;
}
/**
 *  @brief Folds a 32-byte chunk of supplementary-plane 4-byte sequences (F0-F4 leads) mixed
 *      with ASCII - emoji-rich text and historic scripts.
 *
 *  Every supplementary-plane codepoint WITH case folding (Deseret, Osage, Warang Citi, Adlam,
 *  Garay, Medefaidrin, archaic Latin/Greek extensions, …) lives below U+1F000, i.e. its UTF-8
 *  second byte is < 0x9F under an F0 lead. Sequences whose second byte is ≥ 0x9F - all emoji
 *  and everything in planes 2+ - are caseless and copy through unchanged, with only the mixed
 *  ASCII folded. Folding 4-byte sequences, the complex family's 2-byte leads (C0-C1, C7-CD,
 *  D2-D6: Latin Ext-B tails, IPA, extended Cyrillic, Armenian), and foreign-family leads
 *  truncate the chunk at their lead and route one rune to the serial fallback.
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_haswell_supplementary_chunk_( 
    __m256i source_u8x32, sz_u32_t is_complex_lead_mask, sz_u32_t is_four_byte_lead_mask, sz_u32_t is_foreign_lead_mask,
    sz_ptr_t target) {
    __m256i next_bytes_u8x32 = sz_haswell_next_bytes_(source_u8x32);
    __m256i is_four_byte_lead_u8x32 = sz_haswell_in_byte_range_(source_u8x32, 0xF0, 0x08);
    __m256i folding_four_byte_u8x32 = _mm256_andnot_si256(sz_haswell_in_byte_range_(next_bytes_u8x32, 0x9F, 0x61),
                                                          is_four_byte_lead_u8x32);
    sz_u32_t stop_mask = (is_complex_lead_mask & ~is_four_byte_lead_mask) | is_foreign_lead_mask |
                         (sz_u32_t)_mm256_movemask_epi8(folding_four_byte_u8x32);
    sz_size_t fold_length = stop_mask ? (sz_size_t)_tzcnt_u32(stop_mask) : 32;
    sz_u32_t incomplete_mask = is_four_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 3 ? fold_length - 3 : 0);
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)_tzcnt_u32(incomplete_mask);
    if (fold_length == 0) return 0;
    _mm256_storeu_si256((__m256i *)target, sz_haswell_fold_ascii_(source_u8x32));
    return fold_length;
}
/**
 *  @brief Per-chunk lead-byte classification: the family-presence flags plus every per-family and
 *      per-width lead mask the handler dispatch consumes, kept in one struct so the entrypoint loop
 *      reads as a single classify-then-dispatch step instead of an inlined compare tree.
 */
typedef struct sz_utf8_uncased_fold_haswell_leads_t {
    sz_u8_t lead_families;
    sz_u32_t is_lead_mask;
    sz_u32_t is_continuation_mask;
    sz_u32_t is_two_byte_lead_mask;
    sz_u32_t is_three_byte_lead_mask;
    sz_u32_t is_four_byte_lead_mask;
    sz_u32_t is_caseless_lead_mask;
    sz_u32_t is_latin_lead_mask;
    sz_u32_t is_latin_extended_lead_mask;
    sz_u32_t is_cyrillic_lead_mask;
    sz_u32_t is_greek_lead_mask;
    sz_u32_t is_e1_lead_mask;
    sz_u32_t is_guarded_lead_mask;
    sz_u32_t is_complex_lead_mask;
    sz_u32_t well_formed_lead_mask;
    sz_u32_t malformed_lead_mask;
} sz_utf8_uncased_fold_haswell_leads_t;
/**
 *  @brief Classifies the lead bytes of a non-ASCII 32-byte chunk into folding families.
 *      One range compare per family, each reduced via `VPMOVMSKB`, then OR-folded into the same
 *      flag byte the Ice Lake `VPERMB` LUT produces. The caseless family merges D7-DF and E0 into
 *      one contiguous D7-E0 span.
 */
SZ_HELPER_AUTO sz_utf8_uncased_fold_haswell_leads_t sz_utf8_uncased_fold_haswell_classify_leads_(
    __m256i source_u8x32, sz_u32_t is_non_ascii_mask) {
    sz_utf8_uncased_fold_haswell_leads_t leads;
    leads.is_continuation_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_u8x32, 0x80, 0x40));
    leads.is_lead_mask = is_non_ascii_mask & ~leads.is_continuation_mask;
    leads.is_three_byte_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_u8x32, 0xE0, 0x10));
    leads.is_four_byte_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_u8x32, 0xF0, 0x08));
    leads.is_two_byte_lead_mask = leads.is_lead_mask & ~leads.is_three_byte_lead_mask & ~leads.is_four_byte_lead_mask;
    leads.is_caseless_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_or_si256(sz_haswell_in_byte_range_(source_u8x32, 0xD7, 0x0A),
                        _mm256_or_si256(sz_haswell_in_byte_range_(source_u8x32, 0xE3, 0x07),
                                        sz_haswell_in_byte_range_(source_u8x32, 0xEB, 0x04))));
    leads.is_latin_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_u8x32, 0xC2, 0x02));
    leads.is_latin_extended_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
        sz_haswell_in_byte_range_(source_u8x32, 0xC4, 0x03));
    leads.is_cyrillic_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_u8x32, 0xD0, 0x02));
    leads.is_greek_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_u8x32, 0xCE, 0x02));
    leads.is_e1_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xE1)));
    leads.is_guarded_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_or_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xE2)),
                        _mm256_or_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xEA)),
                                        _mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xEF)))));
    leads.is_complex_lead_mask = leads.is_lead_mask &
                                 ~(leads.is_caseless_lead_mask | leads.is_latin_lead_mask |
                                   leads.is_latin_extended_lead_mask | leads.is_cyrillic_lead_mask |
                                   leads.is_greek_lead_mask | leads.is_e1_lead_mask | leads.is_guarded_lead_mask);
    leads.lead_families = (sz_u8_t)( 
        ((leads.is_caseless_lead_mask != 0) << 0) | ((leads.is_latin_lead_mask != 0) << 1) |
        ((leads.is_latin_extended_lead_mask != 0) << 2) | ((leads.is_cyrillic_lead_mask != 0) << 3) |
        ((leads.is_greek_lead_mask != 0) << 4) | ((leads.is_e1_lead_mask != 0) << 5) |
        ((leads.is_guarded_lead_mask != 0) << 6) | ((leads.is_complex_lead_mask != 0) << 7));
    __m256i second_bytes_u8x32 = sz_haswell_next_bytes_(source_u8x32);
    sz_u32_t e0_bad_second_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_and_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xE0)),
                         sz_haswell_in_byte_range_(second_bytes_u8x32, 0x00, 0xA0)));
    sz_u32_t ed_bad_second_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_and_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xED)),
                         sz_haswell_in_byte_range_(second_bytes_u8x32, 0xA0, 0x60)));
    sz_u32_t f0_bad_second_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_and_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xF0)),
                         sz_haswell_in_byte_range_(second_bytes_u8x32, 0x00, 0x90)));
    sz_u32_t f4_bad_second_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_and_si256(_mm256_cmpeq_epi8(source_u8x32, _mm256_set1_epi8((char)0xF4)),
                         sz_haswell_in_byte_range_(second_bytes_u8x32, 0x90, 0x70)));
    sz_u32_t two_byte_complete_mask = leads.is_two_byte_lead_mask & (leads.is_continuation_mask >> 1);
    sz_u32_t three_byte_complete_mask = leads.is_three_byte_lead_mask & (leads.is_continuation_mask >> 1) &
                                        (leads.is_continuation_mask >> 2);
    sz_u32_t four_byte_complete_mask = leads.is_four_byte_lead_mask & (leads.is_continuation_mask >> 1) &
                                       (leads.is_continuation_mask >> 2) & (leads.is_continuation_mask >> 3);
    sz_u32_t bad_special_mask = e0_bad_second_mask | ed_bad_second_mask | f0_bad_second_mask | f4_bad_second_mask;
    leads.well_formed_lead_mask = (two_byte_complete_mask | three_byte_complete_mask | four_byte_complete_mask) &
                                  ~bad_special_mask;
    leads.malformed_lead_mask = leads.is_lead_mask & ~leads.well_formed_lead_mask;
    return leads;
}
/**
 *  @brief Routes a classified 32-byte chunk through the family handlers and stores the fold.
 *      Dispatch triggers on family PRESENCE, not exclusivity: every handler truncates at the first
 *      lead outside its families, so a mixed chunk still folds its longest matching prefix
 *      vectorized. Without this, one foreign byte - a « quote in Russian, an — dash in German -
 *      poisons ~16 consecutive windows into one-rune serial steps, halving real-corpus throughput.
 *      A handler that cannot fold even the first character returns zero and falls through to the
 *      next family.
 *  @return Bytes consumed and written, or zero if every handler declined the chunk.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_haswell_dispatch_chunk_(__m256i source_u8x32,
                                                                      sz_utf8_uncased_fold_haswell_leads_t const *leads,
                                                                      sz_ptr_t target) {
    sz_u32_t malformed = leads->malformed_lead_mask;
    sz_size_t handled = 0;
    if (leads->lead_families & sz_utf8_fold_lead_caseless_flag_k)
        handled = sz_utf8_uncased_fold_haswell_caseless_chunk_(
            source_u8x32, leads->is_two_byte_lead_mask, leads->is_three_byte_lead_mask,
            (leads->is_lead_mask & ~leads->is_caseless_lead_mask) | malformed, target);
    if (!handled && (leads->lead_families & (sz_utf8_fold_lead_latin_flag_k | sz_utf8_fold_lead_latin_extended_flag_k |
                                             sz_utf8_fold_lead_e1_flag_k)))
        handled = sz_utf8_uncased_fold_haswell_latin_chunk_(
            source_u8x32, leads->is_continuation_mask, leads->is_three_byte_lead_mask,
            (leads->is_lead_mask &
             ~(leads->is_latin_lead_mask | leads->is_latin_extended_lead_mask | leads->is_e1_lead_mask)) |
                malformed,
            target);
    if (!handled && (leads->lead_families & sz_utf8_fold_lead_e1_flag_k))
        handled = sz_utf8_uncased_fold_haswell_georgian_chunk_(
            source_u8x32, leads->is_three_byte_lead_mask, (leads->is_lead_mask & ~leads->is_e1_lead_mask) | malformed,
            target);
    if (!handled && (leads->lead_families & sz_utf8_fold_lead_cyrillic_flag_k))
        handled = sz_utf8_uncased_fold_haswell_cyrillic_chunk_(
            source_u8x32, (leads->is_lead_mask & ~leads->is_cyrillic_lead_mask) | malformed, target);
    if (!handled && (leads->lead_families & sz_utf8_fold_lead_greek_flag_k))
        handled = sz_utf8_uncased_fold_haswell_greek_chunk_(
            source_u8x32, (leads->is_lead_mask & ~leads->is_greek_lead_mask) | malformed, target);
    if (!handled && (leads->lead_families & sz_utf8_fold_lead_guarded_flag_k))
        handled = sz_utf8_uncased_fold_haswell_guarded_chunk_(
            source_u8x32, leads->is_two_byte_lead_mask, leads->is_three_byte_lead_mask,
            (leads->is_lead_mask & ~(leads->is_caseless_lead_mask | leads->is_guarded_lead_mask)) | malformed, target);
    if (!handled && (leads->lead_families & sz_utf8_fold_lead_complex_flag_k))
        handled = sz_utf8_uncased_fold_haswell_armenian_chunk_(source_u8x32, leads->is_lead_mask, malformed, target);
    if (!handled && (leads->lead_families & sz_utf8_fold_lead_complex_flag_k))
        handled = sz_utf8_uncased_fold_haswell_supplementary_chunk_(
            source_u8x32, leads->is_complex_lead_mask, leads->is_four_byte_lead_mask,
            (leads->is_lead_mask & ~leads->is_complex_lead_mask) | malformed, target);
    return handled;
}
/**
 *  @brief Folds a single rune from @p source into @p target for chunks no handler accepted.
 *      With ≥ 32 source bytes still available, a complete (≤ 4-byte) sequence is guaranteed.
 *      Validates with `sz_rune_decode` so a malformed lead - the only reason the vector
 *      handlers decline a multi-byte sequence - copies one byte through unchanged and resyncs,
 *      byte-for-byte with the serial reference.
 *  @param source_end Pointer one past the last readable source byte.
 *  @param rune_length Receives the number of source bytes consumed.
 *  @return Bytes written to @p target (Unicode case folding produces at most 3 runes).
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_haswell_one_rune_(sz_cptr_t source, sz_cptr_t source_end, sz_ptr_t target,
                                                                sz_rune_length_t *rune_length) {
    sz_rune_t rune;
    sz_rune_length_t const parsed_length = sz_rune_decode(source, source_end, &rune);
    if (parsed_length == sz_rune_invalid_k) {
        *target = *source; 
        *rune_length = sz_rune_1byte_k;
        return 1;
    }
    *rune_length = parsed_length;
    sz_rune_t folded_runes[3]; 
    sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
    sz_ptr_t target_ptr = target;
    for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
        target_ptr += sz_rune_encode(folded_runes[rune_index], (sz_u8_t *)target_ptr);
    return (sz_size_t)(target_ptr - target);
}
SZ_API_COMPTIME sz_size_t sz_utf8_uncased_fold_haswell(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target) {
    sz_ptr_t const target_start = target;
    while (source_length >= 32) {
        _mm_prefetch(source + 512, _MM_HINT_T0);
        __m256i source_u8x32 = _mm256_lddqu_si256((__m256i const *)source);
        sz_u32_t is_non_ascii_mask = (sz_u32_t)_mm256_movemask_epi8(source_u8x32);
        if (is_non_ascii_mask == 0) {
            _mm256_storeu_si256((__m256i *)target, sz_haswell_fold_ascii_(source_u8x32));
            target += 32, source += 32, source_length -= 32;
            continue;
        }
        sz_utf8_uncased_fold_haswell_leads_t leads = sz_utf8_uncased_fold_haswell_classify_leads_(source_u8x32,
                                                                                                  is_non_ascii_mask);
        sz_size_t handled = sz_utf8_uncased_fold_haswell_dispatch_chunk_(source_u8x32, &leads, target);
        if (handled) {
            target += handled, source += handled, source_length -= handled;
            continue;
        }
        sz_rune_length_t rune_length;
        target += sz_utf8_uncased_fold_haswell_one_rune_(source, source + source_length, target, &rune_length);
        source += rune_length;
        source_length -= rune_length;
    }
    target += sz_utf8_uncased_fold_serial(source, source_length, target);
    return (sz_size_t)(target - target_start);
}
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif 
#ifdef __cplusplus
}
#endif
#endif 
