/**
 *  @brief Ice Lake (AVX-512) backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_uncased_fold/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_ICELAKE_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_ICELAKE_H_
#include "stringzilla/utf8_uncased_fold/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                                              \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt,popcnt,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                      \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt,popcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
    "lzcnt", "popcnt")
#endif
/**
 *  Helper macros to reduce code duplication in fast-paths.
 *  Detect ASCII uppercase A-Z: returns mask where bytes are in range 0x41-0x5A.
 */
#define sz_icelake_is_ascii_upper_(src_u8x64) \
    _mm512_cmplt_epu8_mask(_mm512_sub_epi8((src_u8x64), a_upper_u8x64), subtract26_u8x64)
/** Apply ASCII case folding (+0x20) to masked positions. */
#define sz_icelake_fold_ascii_(src_u8x64, upper_mask, apply_mask) \
    _mm512_mask_add_epi8((src_u8x64), (upper_mask) & (apply_mask), (src_u8x64), ascii_case_offset_u8x64)
/** Fold ASCII A-Z in source vector within a prefix mask, returning folded result. */
#define sz_icelake_fold_ascii_in_prefix_(src_u8x64, prefix_mask)                                          \
    _mm512_mask_add_epi8((src_u8x64), sz_icelake_is_ascii_upper_(src_u8x64) & (prefix_mask), (src_u8x64), \
                         ascii_case_offset_u8x64)
/**
 *  Georgian uppercase transformation: E1 82/83 XX → E2 B4 YY.
 *  Applies to lead byte positions (sets E2), second byte positions (sets B4),
 *  and adjusts third bytes (-0x20 for 82 sequences, +0x20 for 83 sequences).
 */
#define sz_icelake_transform_georgian_(folded, georgian_leads, is_82_upper, is_83_upper, prefix_mask)                \
    do {                                                                                                             \
        (folded) = _mm512_mask_blend_epi8((georgian_leads), (folded), _mm512_set1_epi8((char)0xE2));                 \
        (folded) = _mm512_mask_blend_epi8((georgian_leads) << 1, (folded), _mm512_set1_epi8((char)0xB4));            \
        (folded) = _mm512_mask_sub_epi8((folded), (is_82_upper) & (prefix_mask), (folded), ascii_case_offset_u8x64); \
        (folded) = _mm512_mask_add_epi8((folded), (is_83_upper) & (prefix_mask), (folded), ascii_case_offset_u8x64); \
    } while (0)
/**
 *  @brief Find the first invalid position within load_mask, returning chunk_size if all valid.
 *      This safely handles the edge case where all 64 bytes are valid (i.e., ctz(0) which is undefined).
 *      The mask `~is_valid | ~load_mask` marks positions that are either invalid OR outside the loaded chunk.
 *      When all loaded bytes are valid, this mask becomes zero, so we return chunk_size instead of calling ctz(0).
 *
 *  @param is_valid Bitmask where bit i is set if byte i passes the validity check.
 *  @param load_mask Bitmask of the bytes actually loaded from the source.
 *  @param chunk_size Number of bytes in the current chunk (returned when all loaded bytes are valid).
 *  @return Index of the first invalid byte, or chunk_size if all loaded bytes are valid.
 */
SZ_HELPER_INLINE sz_size_t sz_icelake_first_invalid_(sz_u64_t is_valid, sz_u64_t load_mask, sz_size_t chunk_size) {
    sz_u64_t invalid_mask = ~is_valid | ~load_mask;
    return invalid_mask ? (sz_size_t)_tzcnt_u64(invalid_mask) : chunk_size;
}
/** @brief OR-reduces all 64 byte lanes of a ZMM register into one byte of accumulated flags. */
SZ_HELPER_INLINE sz_u8_t sz_utf8_fold_icelake_reduce_or_u8_(__m512i flags_u8x64) {
    __m256i upper_u8x32 = _mm512_extracti64x4_epi64(flags_u8x64, 1);
    __m256i or256_u8x32 = _mm256_or_si256(_mm512_castsi512_si256(flags_u8x64), upper_u8x32);
    __m128i or128_u8x16 = _mm_or_si128(_mm256_castsi256_si128(or256_u8x32), _mm256_extracti128_si256(or256_u8x32, 1));
    or128_u8x16 = _mm_or_si128(or128_u8x16, _mm_unpackhi_epi64(or128_u8x16, or128_u8x16));
    sz_u64_t or64 = (sz_u64_t)_mm_cvtsi128_si64(or128_u8x16);
    or64 |= or64 >> 32, or64 |= or64 >> 16, or64 |= or64 >> 8;
    return (sz_u8_t)or64;
}
/**
 *  @brief Folds a chunk containing only caseless multi-byte scripts mixed with ASCII.
 *      Folds ASCII A-Z in place and copies everything else, trimming incomplete trailing sequences.
 *  @return Bytes consumed and written, or zero if the chunk starts with an incomplete sequence.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_icelake_caseless_chunk_( 
    __m512i source_u8x64, __mmask64 load_m64, sz_size_t chunk_size,    
    __mmask64 is_two_byte_lead_m64, __mmask64 is_three_byte_lead_m64, __mmask64 malformed_lead_m64, sz_ptr_t target) {
    __m512i const a_upper_u8x64 = _mm512_set1_epi8('A');
    __m512i const subtract26_u8x64 = _mm512_set1_epi8(26);
    __m512i const ascii_case_offset_u8x64 = _mm512_set1_epi8(0x20);
    __mmask64 incomplete_m64 = (is_two_byte_lead_m64 & ~sz_u64_mask_until_(chunk_size > 1 ? chunk_size - 1 : 0)) |
                               (is_three_byte_lead_m64 & ~sz_u64_mask_until_(chunk_size > 2 ? chunk_size - 2 : 0)) |
                               malformed_lead_m64;
    incomplete_m64 &= load_m64;
    sz_size_t copy_length = incomplete_m64 ? (sz_size_t)_tzcnt_u64(incomplete_m64) : chunk_size;
    if (copy_length == 0) return 0;
    __mmask64 prefix_m64 = sz_u64_mask_until_(copy_length);
    _mm512_mask_storeu_epi8(target, prefix_m64, sz_icelake_fold_ascii_in_prefix_(source_u8x64, prefix_m64));
    return copy_length;
}
/**
 *  @brief Folds a chunk of Latin text in place: ASCII, Latin-1 Supplement (C2-C3),
 *      Latin Extended-A/B (C4-C6), and Latin Extended Additional (E1 B8-BB) - the working set
 *      of German, Czech, Vietnamese, and most other Latin-script languages.
 *
 *  Latin Extended folding is parity-based: uppercase codepoints are even and fold to the next
 *  odd codepoint. The codepoint's low bit lives in the last byte of its UTF-8 sequence, so the
 *  fold is an in-place masked +1. Per-codepoint deltas come from `VPERMB` tables indexed by the
 *  continuation byte's low 6 bits; table entries flag the irregular codepoints (those that
 *  expand, shrink, or fold across blocks), which truncate the chunk and route one rune to the
 *  serial fallback.
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_icelake_latin_chunk_( 
    __m512i source_u8x64, __mmask64 load_m64, sz_size_t chunk_size, 
    __mmask64 is_continuation_m64, __mmask64 is_three_byte_lead_m64, __mmask64 malformed_lead_m64, sz_ptr_t target) {
    __m512i const a_upper_u8x64 = _mm512_set1_epi8('A');
    __m512i const subtract26_u8x64 = _mm512_set1_epi8(26);
    __m512i const ascii_case_offset_u8x64 = _mm512_set1_epi8(0x20);
    __m512i const c4_deltas_lut_u8x64 = _mm512_set_epi8( 
        (char)0x80, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, (char)0x80, 
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,                   
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,                   
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1                    
    );
    __m512i const c5_deltas_lut_u8x64 = _mm512_set_epi8(                  
        (char)0x80, 0, 1, 0, 1, 0, 1, (char)0x80, 0, 1, 0, 1, 0, 1, 0, 1, 
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,                   
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,                   
        0, 1, 0, 1, 0, 1, (char)0x80, 0, 1, 0, 1, 0, 1, 0, 1, 0           
    );
    __m512i const c6_deltas_lut_u8x64 = _mm512_set_epi8( 
        0, 0, 0, 1, 0, 0, 0, 1, (char)0x80, 0, 1, 0, 1, (char)0x80, (char)0x80, 0, 1, (char)0x80, 0, 1, 0, 0,
        (char)0x80, 0, 1, (char)0x80, 0, 1, 0, 1, 0, 1, (char)0x80, 0, (char)0x80, (char)0x80, 0, 0, 0, 1, (char)0x80,
        (char)0x80, 0, (char)0x80, (char)0x80, 0, 1, (char)0x80, (char)0x80, (char)0x80, 0, 0, 1, (char)0x80,
        (char)0x80, 0, 1, (char)0x80, 0, 1, 0, 1, (char)0x80, 0);
    __mmask64 is_c2_m64 = _mm512_cmpeq_epi8_mask(source_u8x64, _mm512_set1_epi8((char)0xC2));
    __mmask64 is_c3_m64 = _mm512_cmpeq_epi8_mask(source_u8x64, _mm512_set1_epi8((char)0xC3));
    __mmask64 is_c4_m64 = _mm512_cmpeq_epi8_mask(source_u8x64, _mm512_set1_epi8((char)0xC4));
    __mmask64 is_c5_m64 = _mm512_cmpeq_epi8_mask(source_u8x64, _mm512_set1_epi8((char)0xC5));
    __mmask64 is_c6_m64 = _mm512_cmpeq_epi8_mask(source_u8x64, _mm512_set1_epi8((char)0xC6));
    __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(source_u8x64, _mm512_set1_epi8((char)0xE1));
    __mmask64 e1_second_positions_m64 = (is_e1_m64 << 1) & load_m64;
    __mmask64 is_b8_bb_second_m64 = e1_second_positions_m64 &
                                    _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_u8x64, _mm512_set1_epi8((char)0xB8)),
                                                           _mm512_set1_epi8(0x04));
    __mmask64 foreign_e1_second_m64 = e1_second_positions_m64 & ~is_b8_bb_second_m64;
    __mmask64 e1_ba_third_positions_m64 =
        ((is_b8_bb_second_m64 & _mm512_cmpeq_epi8_mask(source_u8x64, _mm512_set1_epi8((char)0xBA))) << 1) & load_m64;
    __mmask64 irregular_m64 = e1_ba_third_positions_m64 &
                              _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_u8x64, _mm512_set1_epi8((char)0x96)),
                                                     _mm512_set1_epi8(0x09));
    __mmask64 after_c4_m64 = (is_c4_m64 << 1) & is_continuation_m64;
    __mmask64 after_c5_m64 = (is_c5_m64 << 1) & is_continuation_m64;
    __mmask64 after_c6_m64 = (is_c6_m64 << 1) & is_continuation_m64;
    __m512i c4_deltas_u8x64 = _mm512_maskz_permutexvar_epi8(after_c4_m64, source_u8x64, c4_deltas_lut_u8x64);
    __m512i c5_deltas_u8x64 = _mm512_maskz_permutexvar_epi8(after_c5_m64, source_u8x64, c5_deltas_lut_u8x64);
    __m512i c6_deltas_u8x64 = _mm512_maskz_permutexvar_epi8(after_c6_m64, source_u8x64, c6_deltas_lut_u8x64);
    __m512i extended_deltas_u8x64 = _mm512_ternarylogic_epi64(c4_deltas_u8x64, c5_deltas_u8x64, c6_deltas_u8x64,
                                                              0xFE); 
    irregular_m64 |= _mm512_test_epi8_mask(extended_deltas_u8x64, _mm512_set1_epi8((char)0x80));
    __mmask64 stop_m64 = (irregular_m64 | foreign_e1_second_m64 | malformed_lead_m64) & load_m64;
    sz_size_t fold_length = chunk_size;
    if (stop_m64) {
        sz_size_t first_flagged_position = (sz_size_t)_tzcnt_u64(stop_m64);
        while (first_flagged_position && ((is_continuation_m64 >> first_flagged_position) & 1))
            --first_flagged_position;
        fold_length = first_flagged_position;
    }
    __mmask64 is_two_byte_lead_m64 = is_c2_m64 | is_c3_m64 | is_c4_m64 | is_c5_m64 | is_c6_m64;
    __mmask64 incomplete_m64 = (is_two_byte_lead_m64 & ~sz_u64_mask_until_(fold_length > 1 ? fold_length - 1 : 0)) |
                               (is_three_byte_lead_m64 & ~sz_u64_mask_until_(fold_length > 2 ? fold_length - 2 : 0));
    incomplete_m64 &= sz_u64_mask_until_(fold_length);
    if (incomplete_m64) fold_length = (sz_size_t)_tzcnt_u64(incomplete_m64);
    if (fold_length == 0) return 0;
    __mmask64 prefix_m64 = sz_u64_mask_until_(fold_length);
    __m512i folded_u8x64 = sz_icelake_fold_ascii_in_prefix_(source_u8x64, prefix_m64);
    __mmask64 after_c3_m64 = (is_c3_m64 << 1) & prefix_m64;
    __mmask64 is_latin1_upper_m64 = _mm512_mask_cmplt_epu8_mask(
        after_c3_m64, _mm512_sub_epi8(source_u8x64, _mm512_set1_epi8((char)0x80)), _mm512_set1_epi8(0x1F));
    is_latin1_upper_m64 &= ~_mm512_cmpeq_epi8_mask(source_u8x64, _mm512_set1_epi8((char)0x97));
    folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, is_latin1_upper_m64, folded_u8x64, ascii_case_offset_u8x64);
    __mmask64 is_eszett_second_m64 = after_c3_m64 & _mm512_cmpeq_epi8_mask(source_u8x64, _mm512_set1_epi8((char)0x9F));
    folded_u8x64 = _mm512_mask_set1_epi8(folded_u8x64, is_eszett_second_m64 | (is_eszett_second_m64 >> 1), 's');
    __mmask64 is_micro_second_m64 = ((is_c2_m64 << 1) & prefix_m64) &
                                    _mm512_cmpeq_epi8_mask(source_u8x64, _mm512_set1_epi8((char)0xB5));
    folded_u8x64 = _mm512_mask_mov_epi8(folded_u8x64, is_micro_second_m64 >> 1, _mm512_set1_epi8((char)0xCE));
    folded_u8x64 = _mm512_mask_mov_epi8(folded_u8x64, is_micro_second_m64, _mm512_set1_epi8((char)0xBC));
    __mmask64 extended_positions_m64 = (after_c4_m64 | after_c5_m64 | after_c6_m64) & prefix_m64;
    folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, extended_positions_m64, folded_u8x64, extended_deltas_u8x64);
    __mmask64 e1_third_positions_m64 = (is_b8_bb_second_m64 << 1) & prefix_m64;
    __mmask64 is_even_codepoint_m64 = ~_mm512_test_epi8_mask(source_u8x64, _mm512_set1_epi8(0x01));
    folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, e1_third_positions_m64 & is_even_codepoint_m64, folded_u8x64,
                                        _mm512_set1_epi8(0x01));
    _mm512_mask_storeu_epi8(target, prefix_m64, folded_u8x64);
    return fold_length;
}
SZ_API_COMPTIME sz_size_t sz_utf8_uncased_fold_icelake(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target) {
    sz_u512_vec_t source_vec;
    sz_ptr_t target_start = target;
    __m512i const indices_u8x64 = _mm512_set_epi8(                      
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const a_upper_u8x64 = _mm512_set1_epi8('A');
    __m512i const subtract26_u8x64 = _mm512_set1_epi8(26);
    __m512i const ascii_case_offset_u8x64 = _mm512_set1_epi8(0x20); 
    __m512i const utf8_cont_test_u8x64 = _mm512_set1_epi8((char)0xC0);
    __m512i const utf8_cont_pattern_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const utf8_3byte_test_u8x64 = _mm512_set1_epi8((char)0xF0);
    __m512i const utf8_3byte_pattern_u8x64 = _mm512_set1_epi8((char)0xE0);
    __m512i const utf8_4byte_test_u8x64 = _mm512_set1_epi8((char)0xF8);
    while (source_length) {
        _mm_prefetch(source + 1024, _MM_HINT_T1); 
        _mm_prefetch(source + 512, _MM_HINT_T0);  
        _mm_prefetch(source + 576, _MM_HINT_T0);  
        _mm_prefetch(source + 640, _MM_HINT_T0);  
        sz_size_t chunk_size = sz_min_of_two(source_length, 64);
        __mmask64 load_m64 = sz_u64_mask_until_(chunk_size);
        source_vec.zmm = _mm512_maskz_loadu_epi8(load_m64, source);
        __mmask64 is_non_ascii_m64 = _mm512_movepi8_mask(source_vec.zmm);
        if (is_non_ascii_m64 == 0) {
            _mm512_mask_storeu_epi8(target, load_m64, sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, load_m64));
            target += chunk_size, source += chunk_size, source_length -= chunk_size;
            continue;
        }
        __m512i masked_cont_u8x64 = _mm512_ternarylogic_epi64(source_vec.zmm, utf8_cont_test_u8x64,
                                                              utf8_cont_test_u8x64, 0x80);
        __m512i masked_3byte_u8x64 = _mm512_ternarylogic_epi64(source_vec.zmm, utf8_3byte_test_u8x64,
                                                               utf8_3byte_test_u8x64, 0x80);
        __m512i masked_4byte_u8x64 = _mm512_ternarylogic_epi64(source_vec.zmm, utf8_4byte_test_u8x64,
                                                               utf8_4byte_test_u8x64, 0x80);
        __mmask64 is_cont_m64 = _mm512_cmpeq_epi8_mask(masked_cont_u8x64, utf8_cont_pattern_u8x64);
        __mmask64 is_three_byte_lead_m64 = _mm512_cmpeq_epi8_mask(masked_3byte_u8x64, utf8_3byte_pattern_u8x64);
        __mmask64 is_four_byte_lead_m64 = _mm512_cmpeq_epi8_mask(masked_4byte_u8x64, utf8_3byte_test_u8x64);
        __mmask64 is_lead_m64 = is_non_ascii_m64 & ~is_cont_m64 & load_m64;
        __mmask64 is_two_byte_lead_classifier_m64 = is_lead_m64 & ~is_three_byte_lead_m64 & ~is_four_byte_lead_m64;
        __mmask64 next_is_cont_m64 = is_cont_m64 >> 1;
        __mmask64 next2_is_cont_m64 = is_cont_m64 >> 2;
        __mmask64 next3_is_cont_m64 = is_cont_m64 >> 3;
        __mmask64 is_c0c1_lead_m64 = _mm512_cmplt_epu8_mask(
            _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xC0)), _mm512_set1_epi8(2));
        __mmask64 second_lt_a0_at_lead_m64 = _mm512_cmplt_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xA0)) >> 1;
        __mmask64 second_lt_90_at_lead_m64 = _mm512_cmplt_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x90)) >> 1;
        __mmask64 is_e0_lead_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE0));
        __mmask64 is_ed_lead_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xED));
        __mmask64 is_f0_lead_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xF0));
        __mmask64 is_f4_lead_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xF4));
        __mmask64 bad_special_lead_m64 = (is_e0_lead_m64 & second_lt_a0_at_lead_m64) |
                                         (is_ed_lead_m64 & ~second_lt_a0_at_lead_m64) |
                                         (is_f0_lead_m64 & second_lt_90_at_lead_m64) |
                                         (is_f4_lead_m64 & ~second_lt_90_at_lead_m64);
        __mmask64 well_formed_two_byte_m64 = is_two_byte_lead_classifier_m64 & ~is_c0c1_lead_m64 & next_is_cont_m64;
        __mmask64 well_formed_three_byte_m64 = is_three_byte_lead_m64 & next_is_cont_m64 & next2_is_cont_m64;
        __mmask64 well_formed_four_byte_m64 = is_four_byte_lead_m64 & next_is_cont_m64 & next2_is_cont_m64 &
                                              next3_is_cont_m64;
        __mmask64 well_formed_lead_m64 =
            (well_formed_two_byte_m64 | well_formed_three_byte_m64 | well_formed_four_byte_m64) & ~bad_special_lead_m64;
        __mmask64 malformed_lead_m64 = is_lead_m64 & ~well_formed_lead_m64;
        __m512i const lead_families_lut_u8x64 = _mm512_set_epi8(
            (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
            (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
            0x40, 0x01, 0x01, 0x01, 0x01, 0x40, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x40, 0x20, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
            (char)0x80, 0x08, 0x08,
            0x10, 0x10, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, 0x04, 0x04,
            0x04, 0x02, 0x02, (char)0x80, (char)0x80);
        __m512i lead_families_u8x64 = _mm512_maskz_permutexvar_epi8(is_lead_m64, source_vec.zmm,
                                                                    lead_families_lut_u8x64);
        sz_u8_t lead_families = sz_utf8_fold_icelake_reduce_or_u8_(lead_families_u8x64);
        if (!(lead_families & ~sz_utf8_fold_lead_caseless_flag_k)) {
            __mmask64 is_two_byte_lead_m64 = is_lead_m64 & ~is_three_byte_lead_m64 & ~is_four_byte_lead_m64;
            sz_size_t handled = sz_utf8_uncased_fold_icelake_caseless_chunk_(
                source_vec.zmm, load_m64, chunk_size, is_two_byte_lead_m64, is_three_byte_lead_m64, malformed_lead_m64,
                target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        else if ((lead_families & (sz_utf8_fold_lead_latin_extended_flag_k | sz_utf8_fold_lead_e1_flag_k)) &&
                 !(lead_families & ~(sz_utf8_fold_lead_latin_flag_k | sz_utf8_fold_lead_latin_extended_flag_k |
                                     sz_utf8_fold_lead_e1_flag_k))) {
            sz_size_t handled = sz_utf8_uncased_fold_icelake_latin_chunk_(
                source_vec.zmm, load_m64, chunk_size, is_cont_m64, is_three_byte_lead_m64, malformed_lead_m64, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        {
            __mmask64 is_valid_pure_3byte_m64 = (is_three_byte_lead_m64 | is_cont_m64) & ~malformed_lead_m64;
            if ((is_valid_pure_3byte_m64 & load_m64) == (is_non_ascii_m64 & load_m64) && !is_four_byte_lead_m64) {
                __mmask64 is_e1_e2_m64 = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xE1)), _mm512_set1_epi8(2));
                __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
                __mmask64 is_ef_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEF));
                __mmask64 is_e2_m64 = _kand_mask64(is_e1_e2_m64, _knot_mask64(is_e1_m64));
                __mmask64 e2_second_byte_positions_m64 = is_e2_m64 << 1;
                __mmask64 is_e2_folding_m64 = e2_second_byte_positions_m64 &
                                              ~_mm512_cmplt_epu8_mask(
                                                  _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                                                  _mm512_set1_epi8(0x04)); 
                __mmask64 is_ea_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEA));
                __mmask64 ea_second_byte_positions_m64 = is_ea_m64 << 1;
                __mmask64 is_ea_folding_m64 =
                    ea_second_byte_positions_m64 &
                    (_mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x99)),
                                            _mm512_set1_epi8(0x07)) | 
                     _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xAD)),
                                            _mm512_set1_epi8(0x02))); 
                if (!(is_e1_m64 | is_e2_folding_m64 | is_ea_folding_m64 | is_ef_m64)) {
                    sz_size_t copy_length = chunk_size;
                    __mmask64 leads_in_chunk_m64 = is_three_byte_lead_m64 & load_m64;
                    if (leads_in_chunk_m64) {
                        int last_lead_pos = 63 - (int)_lzcnt_u64(leads_in_chunk_m64);
                        if (last_lead_pos + 3 > (int)copy_length) copy_length = last_lead_pos;
                    }
                    if (copy_length > 0) {
                        __mmask64 copy_m64 = sz_u64_mask_until_(copy_length);
                        _mm512_mask_storeu_epi8(target, copy_m64,
                                                sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, copy_m64));
                        target += copy_length, source += copy_length, source_length -= copy_length;
                        continue;
                    }
                }
            }
        }
        __mmask64 is_latin1_lead_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xC3));
        __mmask64 latin1_second_byte_positions_m64 = is_latin1_lead_m64 << 1;
        sz_size_t latin1_length = 0;
        if (lead_families & sz_utf8_fold_lead_latin_flag_k) {
            __mmask64 is_valid_latin1_mix_m64 =
                (~is_non_ascii_m64 | is_latin1_lead_m64 | latin1_second_byte_positions_m64) & ~malformed_lead_m64;
            latin1_length = sz_icelake_first_invalid_(is_valid_latin1_mix_m64, load_m64, chunk_size);
            latin1_length -= latin1_length && ((is_latin1_lead_m64 >> (latin1_length - 1)) & 1); 
        }
        if (latin1_length >= 2) {
            __mmask64 prefix_m64 = sz_u64_mask_until_(latin1_length);
            __mmask64 latin1_second_bytes_m64 = latin1_second_byte_positions_m64 & prefix_m64;
            __mmask64 is_upper_ascii_m64 = sz_icelake_is_ascii_upper_(source_vec.zmm);
            __mmask64 is_latin1_upper_m64 = _mm512_mask_cmplt_epu8_mask(
                latin1_second_bytes_m64, _mm512_sub_epi8(source_vec.zmm, utf8_cont_pattern_u8x64),
                _mm512_set1_epi8(0x1F));
            is_latin1_upper_m64 ^= _mm512_mask_cmpeq_epi8_mask(is_latin1_upper_m64, source_vec.zmm,
                                                               _mm512_set1_epi8((char)0x97)); 
            __m512i folded_u8x64 = sz_icelake_fold_ascii_(source_vec.zmm, is_upper_ascii_m64 | is_latin1_upper_m64,
                                                          prefix_m64);
            __mmask64 is_eszett_m64 = _mm512_mask_cmpeq_epi8_mask(latin1_second_bytes_m64, source_vec.zmm,
                                                                  _mm512_set1_epi8((char)0x9F));
            folded_u8x64 = _mm512_mask_set1_epi8(folded_u8x64, is_eszett_m64 | (is_eszett_m64 >> 1), 's');
            _mm512_mask_storeu_epi8(target, prefix_m64, folded_u8x64);
            target += latin1_length, source += latin1_length, source_length -= latin1_length;
            continue;
        }
        if (lead_families & sz_utf8_fold_lead_cyrillic_flag_k) {
            __mmask64 is_cyrillic_lead_m64 = _mm512_cmplt_epu8_mask(
                _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xD0)), _mm512_set1_epi8(2));
            __mmask64 is_d0_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD0));
            __mmask64 is_d1_m64 = _kand_mask64(is_cyrillic_lead_m64, _knot_mask64(is_d0_m64));
            __mmask64 cyrillic_second_byte_positions_m64 = is_cyrillic_lead_m64 << 1;
            __mmask64 is_d1_extended_m64 = (is_d1_m64 << 1) &
                                           _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xA0));
            __mmask64 is_valid_cyrillic_mix_m64 = ~is_non_ascii_m64 | is_cyrillic_lead_m64 |
                                                  cyrillic_second_byte_positions_m64;
            is_valid_cyrillic_mix_m64 &= ~is_d1_extended_m64 & ~malformed_lead_m64; 
            sz_size_t cyrillic_length = sz_icelake_first_invalid_(is_valid_cyrillic_mix_m64, load_m64, chunk_size);
            cyrillic_length -= cyrillic_length && ((is_cyrillic_lead_m64 >> (cyrillic_length - 1)) & 1);
            if (cyrillic_length >= 2) {
                __mmask64 prefix_m64 = sz_u64_mask_until_(cyrillic_length);
                __mmask64 is_after_d0_m64 = (is_d0_m64 << 1) & prefix_m64;
                __m512i folded_u8x64 = sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, prefix_m64);
                __mmask64 is_d0_upper1_m64 = _mm512_mask_cmplt_epu8_mask(is_after_d0_m64, source_vec.zmm,
                                                                         _mm512_set1_epi8((char)0x90));
                __mmask64 is_d0_upper2_m64 = _mm512_mask_cmplt_epu8_mask(
                    is_after_d0_m64, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x90)),
                    _mm512_set1_epi8(0x10));
                __mmask64 is_d0_upper3_m64 = _mm512_mask_cmplt_epu8_mask(
                    is_after_d0_m64, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                    _mm512_set1_epi8(0x10));
                folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, is_d0_upper1_m64, folded_u8x64,
                                                    _mm512_set1_epi8(0x10));
                folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, is_d0_upper2_m64, folded_u8x64,
                                                    _mm512_set1_epi8(0x20));
                folded_u8x64 = _mm512_mask_sub_epi8(folded_u8x64, is_d0_upper3_m64, folded_u8x64,
                                                    _mm512_set1_epi8(0x20));
                __mmask64 needs_d1_m64 = ((is_d0_upper1_m64 | is_d0_upper3_m64) >> 1) & (is_d0_m64 & prefix_m64);
                folded_u8x64 = _mm512_mask_mov_epi8(folded_u8x64, needs_d1_m64, _mm512_set1_epi8((char)0xD1));
                _mm512_mask_storeu_epi8(target, prefix_m64, folded_u8x64);
                target += cyrillic_length, source += cyrillic_length, source_length -= cyrillic_length;
                continue;
            }
        }
        if (lead_families & sz_utf8_fold_lead_greek_flag_k) {
            __mmask64 is_ce_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xCE));
            __mmask64 is_cf_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xCF));
            __mmask64 is_greek_lead_m64 = is_ce_m64 | is_cf_m64;
            __mmask64 greek_second_byte_positions_m64 = is_greek_lead_m64 << 1;
            __mmask64 is_ce_problematic_m64 =
                (is_ce_m64 << 1) &
                (_mm512_cmplt_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x91)) | 
                 _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xB0))); 
            __mmask64 is_cf_problematic_m64 = (is_cf_m64 << 1) &
                                              _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x8F));
            __mmask64 is_valid_greek_mix_m64 = ~is_non_ascii_m64 | is_greek_lead_m64 | greek_second_byte_positions_m64;
            is_valid_greek_mix_m64 &= ~(is_ce_problematic_m64 | is_cf_problematic_m64) & ~malformed_lead_m64;
            sz_size_t greek_length = sz_icelake_first_invalid_(is_valid_greek_mix_m64, load_m64, chunk_size);
            greek_length -= greek_length && ((is_greek_lead_m64 >> (greek_length - 1)) & 1);
            if (greek_length >= 2) {
                __mmask64 prefix_m64 = sz_u64_mask_until_(greek_length);
                __mmask64 is_after_ce_m64 = (is_ce_m64 << 1) & prefix_m64;
                __mmask64 is_after_cf_m64 = (is_cf_m64 << 1) & prefix_m64;
                __m512i folded_u8x64 = sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, prefix_m64);
                __mmask64 is_ce_upper1_m64 = _mm512_mask_cmplt_epu8_mask(
                    is_after_ce_m64, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x91)),
                    _mm512_set1_epi8(0x0F));
                __mmask64 is_ce_upper2a_m64 = _mm512_mask_cmplt_epu8_mask(
                    is_after_ce_m64, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                    _mm512_set1_epi8(0x02));
                __mmask64 is_ce_upper2b_m64 = _mm512_mask_cmplt_epu8_mask(
                    is_after_ce_m64, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA3)),
                    _mm512_set1_epi8(0x09));
                __mmask64 is_ce_upper2_m64 = is_ce_upper2a_m64 | is_ce_upper2b_m64;
                folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, is_ce_upper1_m64, folded_u8x64,
                                                    _mm512_set1_epi8(0x20));
                folded_u8x64 = _mm512_mask_sub_epi8(folded_u8x64, is_ce_upper2_m64, folded_u8x64,
                                                    _mm512_set1_epi8(0x20));
                __mmask64 needs_cf_m64 = (is_ce_upper2_m64 >> 1) & (is_ce_m64 & prefix_m64);
                folded_u8x64 = _mm512_mask_mov_epi8(folded_u8x64, needs_cf_m64, _mm512_set1_epi8((char)0xCF));
                __mmask64 is_final_sigma_m64 = is_after_cf_m64 &
                                               _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x82));
                folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, is_final_sigma_m64, folded_u8x64,
                                                    _mm512_set1_epi8(1));
                _mm512_mask_storeu_epi8(target, prefix_m64, folded_u8x64);
                target += greek_length, source += greek_length, source_length -= greek_length;
                continue;
            }
        }
        __mmask64 is_caseless_2byte_m64 = 0;
        if (lead_families & sz_utf8_fold_lead_caseless_flag_k)
            is_caseless_2byte_m64 = _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD7)) &
                                    _mm512_cmple_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xDF));
        if (is_caseless_2byte_m64) {
            __mmask64 is_caseless_second_m64 = is_caseless_2byte_m64 << 1;
            __mmask64 is_valid_caseless_m64 = (~is_non_ascii_m64 | is_caseless_2byte_m64 | is_caseless_second_m64) &
                                              ~malformed_lead_m64;
            sz_size_t caseless_length = sz_icelake_first_invalid_(is_valid_caseless_m64, load_m64, chunk_size);
            caseless_length -= caseless_length && ((is_caseless_2byte_m64 >> (caseless_length - 1)) & 1);
            if (caseless_length >= 2) {
                __mmask64 prefix_m64 = sz_u64_mask_until_(caseless_length);
                _mm512_mask_storeu_epi8(target, prefix_m64,
                                        sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, prefix_m64));
                target += caseless_length, source += caseless_length, source_length -= caseless_length;
                continue;
            }
        }
        __mmask64 is_two_byte_lead_m64 = _mm512_cmpeq_epi8_mask(
            _mm512_and_si512(source_vec.zmm, _mm512_set1_epi8((char)0xE0)), _mm512_set1_epi8((char)0xC0));
        is_two_byte_lead_m64 &= ~is_latin1_lead_m64; 
        __mmask64 two_byte_second_positions_m64 = is_two_byte_lead_m64 << 1;
        __mmask64 is_valid_two_byte_mix_m64 =
            (~is_non_ascii_m64 | is_two_byte_lead_m64 | two_byte_second_positions_m64) & ~malformed_lead_m64;
        sz_size_t two_byte_length = sz_icelake_first_invalid_(is_valid_two_byte_mix_m64, load_m64, chunk_size);
        two_byte_length -= two_byte_length && ((is_two_byte_lead_m64 >> (two_byte_length - 1)) & 1);
        if (two_byte_length >= 2) {
            __mmask64 prefix_m64 = sz_u64_mask_until_(two_byte_length);
            __mmask64 is_char_start_m64 = (~is_non_ascii_m64 | is_two_byte_lead_m64) & prefix_m64;
            sz_size_t character_count = (sz_size_t)_mm_popcnt_u64(is_char_start_m64);
            sz_u512_vec_t char_indices_vec;
            char_indices_vec.zmm = _mm512_maskz_compress_epi8(is_char_start_m64, indices_u8x64);
            if (character_count > 16) {
                sz_u8_t last_char_index = char_indices_vec.u8s[15]; 
                two_byte_length = last_char_index + ((is_two_byte_lead_m64 >> last_char_index) & 1 ? 2 : 1);
                prefix_m64 = sz_u64_mask_until_(two_byte_length);
                is_char_start_m64 &= prefix_m64;
                character_count = 16;
            }
            sz_u512_vec_t first_bytes_vec, second_bytes_vec;
            first_bytes_vec.zmm = _mm512_permutexvar_epi8(char_indices_vec.zmm, source_vec.zmm);
            second_bytes_vec.zmm = _mm512_permutexvar_epi8(_mm512_add_epi8(char_indices_vec.zmm, _mm512_set1_epi8(1)),
                                                           source_vec.zmm);
            __m512i first_wide_u32x16 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(first_bytes_vec.zmm));
            __m512i second_wide_u32x16 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(second_bytes_vec.zmm));
            __mmask16 is_two_byte_char_m16 = (__mmask16)_pext_u64(is_two_byte_lead_m64 & prefix_m64, is_char_start_m64);
            __m512i decoded_u32x16 = _mm512_or_si512(
                _mm512_slli_epi32(_mm512_and_si512(first_wide_u32x16, _mm512_set1_epi32(0x1F)), 6),
                _mm512_and_si512(second_wide_u32x16, _mm512_set1_epi32(0x3F)));
            __m512i codepoints_u32x16 = _mm512_mask_blend_epi32(is_two_byte_char_m16, first_wide_u32x16,
                                                                decoded_u32x16);
            __mmask16 needs_serial_m16 =
                _mm512_cmpeq_epi32_mask(codepoints_u32x16, _mm512_set1_epi32(0x0130)) | 
                _mm512_cmpeq_epi32_mask(codepoints_u32x16, _mm512_set1_epi32(0x0149)) | 
                _mm512_cmpeq_epi32_mask(codepoints_u32x16, _mm512_set1_epi32(0x0178)) | 
                _mm512_cmpeq_epi32_mask(codepoints_u32x16, _mm512_set1_epi32(0x017F)) | 
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0180)),
                                        _mm512_set1_epi32(0x00D0)) | 
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0345)),
                                        _mm512_set1_epi32(0x4C)) | 
                _mm512_cmpeq_epi32_mask(codepoints_u32x16, _mm512_set1_epi32(0x03B0)) |
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x03CF)),
                                        _mm512_set1_epi32(0x31)) | 
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0460)),
                                        _mm512_set1_epi32(0x00D0)) | 
                _mm512_cmpeq_epi32_mask(codepoints_u32x16, _mm512_set1_epi32(0x0587));
            needs_serial_m16 &= (__mmask16)((1u << character_count) - 1);
            if (needs_serial_m16) {
                sz_size_t first_special_index = (sz_size_t)_tzcnt_u64((sz_u64_t)needs_serial_m16);
                if (first_special_index == 0) {
                    sz_rune_t rune;
                    sz_rune_length_t const rune_length = sz_rune_decode(source, source + source_length, &rune);
                    if (rune_length == sz_rune_invalid_k) {
                        *target++ = *source++;
                        source_length -= 1;
                        continue;
                    }
                    sz_rune_t folded_runes[3]; 
                    sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
                    for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
                        target += sz_rune_encode(folded_runes[rune_index], (sz_u8_t *)target);
                    source += rune_length;
                    source_length -= rune_length;
                    continue;
                }
                character_count = first_special_index;
                sz_u8_t last_char_index = char_indices_vec.u8s[character_count - 1];
                two_byte_length = last_char_index + ((is_two_byte_lead_m64 >> last_char_index) & 1 ? 2 : 1);
                prefix_m64 = sz_u64_mask_until_(two_byte_length);
                is_char_start_m64 &= prefix_m64;
                is_two_byte_char_m16 = (__mmask16)_pext_u64(is_two_byte_lead_m64 & prefix_m64, is_char_start_m64);
            }
            __m512i folded_u32x16 = codepoints_u32x16;
            folded_u32x16 = _mm512_mask_add_epi32(
                folded_u32x16,
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0041)),
                                        _mm512_set1_epi32(26)),
                folded_u32x16, _mm512_set1_epi32(0x20));
            folded_u32x16 = _mm512_mask_add_epi32(
                folded_u32x16,
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0410)),
                                        _mm512_set1_epi32(0x20)),
                folded_u32x16, _mm512_set1_epi32(0x20));
            folded_u32x16 = _mm512_mask_add_epi32(
                folded_u32x16,
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0400)),
                                        _mm512_set1_epi32(0x10)),
                folded_u32x16, _mm512_set1_epi32(0x50));
            folded_u32x16 = _mm512_mask_add_epi32(
                folded_u32x16,
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0391)),
                                        _mm512_set1_epi32(0x11)),
                folded_u32x16, _mm512_set1_epi32(0x20));
            folded_u32x16 = _mm512_mask_add_epi32(
                folded_u32x16,
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x03A3)),
                                        _mm512_set1_epi32(0x09)),
                folded_u32x16, _mm512_set1_epi32(0x20));
            folded_u32x16 = _mm512_mask_add_epi32(
                folded_u32x16,
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0531)),
                                        _mm512_set1_epi32(0x26)),
                folded_u32x16, _mm512_set1_epi32(0x30));
            __mmask16 is_odd_m16 = _mm512_test_epi32_mask(codepoints_u32x16, _mm512_set1_epi32(1));
            __mmask16 is_even_m16 = ~is_odd_m16;
            __mmask16 is_latin_even_upper_m16 =
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0100)),
                                        _mm512_set1_epi32(0x30)) |
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0132)),
                                        _mm512_set1_epi32(0x06)) |
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x014A)),
                                        _mm512_set1_epi32(0x2E));
            folded_u32x16 = _mm512_mask_add_epi32(folded_u32x16, is_latin_even_upper_m16 & is_even_m16, folded_u32x16,
                                                  _mm512_set1_epi32(1));
            __mmask16 is_latin_odd_upper_m16 =
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0139)),
                                        _mm512_set1_epi32(0x10)) |
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints_u32x16, _mm512_set1_epi32(0x0179)),
                                        _mm512_set1_epi32(0x06));
            folded_u32x16 = _mm512_mask_add_epi32(folded_u32x16, is_latin_odd_upper_m16 & is_odd_m16, folded_u32x16,
                                                  _mm512_set1_epi32(1));
            folded_u32x16 = _mm512_mask_mov_epi32(folded_u32x16,
                                                  _mm512_cmpeq_epi32_mask(codepoints_u32x16, _mm512_set1_epi32(0x00B5)),
                                                  _mm512_set1_epi32(0x03BC));
            folded_u32x16 = _mm512_mask_mov_epi32(folded_u32x16,
                                                  _mm512_cmpeq_epi32_mask(codepoints_u32x16, _mm512_set1_epi32(0x03C2)),
                                                  _mm512_set1_epi32(0x03C3));
            __m512i new_lead_u32x16 = _mm512_or_si512(_mm512_set1_epi32(0xC0), _mm512_srli_epi32(folded_u32x16, 6));
            __m512i new_second_u32x16 = _mm512_or_si512(_mm512_set1_epi32(0x80),
                                                        _mm512_and_si512(folded_u32x16, _mm512_set1_epi32(0x3F)));
            __mmask16 is_ascii_out_m16 = _mm512_cmplt_epu32_mask(folded_u32x16, _mm512_set1_epi32(0x80));
            new_lead_u32x16 = _mm512_mask_blend_epi32(is_ascii_out_m16, new_lead_u32x16, folded_u32x16);
            __m512i lead_u8x64 = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(new_lead_u32x16));
            __m512i second_u8x64 = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(new_second_u32x16));
            __m512i result_u8x64 = _mm512_mask_expand_epi8(source_vec.zmm, is_char_start_m64, lead_u8x64);
            __m512i second_compressed_u8x64 = _mm512_maskz_compress_epi8((__mmask64)is_two_byte_char_m16, second_u8x64);
            result_u8x64 = _mm512_mask_expand_epi8(result_u8x64, two_byte_second_positions_m64 & prefix_m64,
                                                   second_compressed_u8x64);
            _mm512_mask_storeu_epi8(target, prefix_m64, result_u8x64);
            target += two_byte_length, source += two_byte_length, source_length -= two_byte_length;
            continue;
        }
        {
            __mmask64 is_two_byte_lead_m64 = _mm512_cmplt_epu8_mask(
                _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xC0)),
                _mm512_set1_epi8(0x20)); 
            __mmask64 is_e1_lead_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
            __mmask64 is_e2_lead_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE2));
            __mmask64 is_ef_lead_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEF));
            __mmask64 is_ea_lead_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEA));
            __mmask64 ea_second_byte_positions_m64 = is_ea_lead_m64 << 1;
            __mmask64 is_ea_complex_m64 =
                ea_second_byte_positions_m64 &
                (_mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x99)),
                                        _mm512_set1_epi8(0x07)) | 
                 _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xAD)),
                                        _mm512_set1_epi8(0x02))); 
            __mmask64 has_complex_m64 = (is_two_byte_lead_m64 | is_four_byte_lead_m64 | is_e1_lead_m64 |
                                         is_e2_lead_m64 | is_ea_complex_m64 | is_ef_lead_m64) &
                                        load_m64;
            if (!has_complex_m64) {
                __mmask64 is_valid_m64 = (~is_non_ascii_m64 | is_three_byte_lead_m64 | is_cont_m64) &
                                         ~malformed_lead_m64;
                sz_size_t valid_length = sz_icelake_first_invalid_(is_valid_m64, load_m64, chunk_size);
                if (valid_length >= 1) {
                    __mmask64 all_leads_m64 = is_three_byte_lead_m64 & sz_u64_mask_until_(valid_length);
                    __mmask64 safe_m64 = valid_length >= 3 ? sz_u64_mask_until_(valid_length - 2) : 0;
                    __mmask64 unsafe_m64 = all_leads_m64 & ~safe_m64;
                    if (unsafe_m64) valid_length = (int)_tzcnt_u64(unsafe_m64);
                }
                if (valid_length >= 2) {
                    __mmask64 mask_m64 = sz_u64_mask_until_(valid_length);
                    _mm512_mask_storeu_epi8(target, mask_m64,
                                            sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, mask_m64));
                    target += valid_length, source += valid_length, source_length -= valid_length;
                    continue;
                }
            }
            if (is_e1_lead_m64 && source_length >= 3) {
                __m512i georgian_second_bytes_u8x64 = _mm512_permutexvar_epi8(
                    _mm512_add_epi8(indices_u8x64, _mm512_set1_epi8(1)), source_vec.zmm);
                __mmask64 safe_e1_m64 = is_e1_lead_m64 & (load_m64 >> 1);
                __mmask64 is_82_at_e1_m64 = _mm512_mask_cmpeq_epi8_mask(safe_e1_m64, georgian_second_bytes_u8x64,
                                                                        _mm512_set1_epi8((char)0x82));
                __mmask64 is_83_at_e1_m64 = _mm512_mask_cmpeq_epi8_mask(safe_e1_m64, georgian_second_bytes_u8x64,
                                                                        _mm512_set1_epi8((char)0x83));
                __mmask64 is_georgian_e1_m64 = is_82_at_e1_m64 | is_83_at_e1_m64;
                __mmask64 non_georgian_e1_m64 = safe_e1_m64 & ~is_georgian_e1_m64;
                if (!non_georgian_e1_m64 && is_georgian_e1_m64) {
                    __mmask64 third_pos_82_m64 = is_82_at_e1_m64 << 2;
                    __mmask64 third_pos_83_m64 = is_83_at_e1_m64 << 2;
                    __mmask64 is_82_uppercase_m64 = _mm512_mask_cmplt_epu8_mask(
                        third_pos_82_m64 & load_m64, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                        _mm512_set1_epi8(0x20));
                    __mmask64 is_83_range_m64 = _mm512_mask_cmplt_epu8_mask(
                        third_pos_83_m64 & load_m64, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                        _mm512_set1_epi8(0x06));
                    __mmask64 is_83_c7_m64 = _mm512_mask_cmpeq_epi8_mask(third_pos_83_m64 & load_m64, source_vec.zmm,
                                                                         _mm512_set1_epi8((char)0x87));
                    __mmask64 is_83_cd_m64 = _mm512_mask_cmpeq_epi8_mask(third_pos_83_m64 & load_m64, source_vec.zmm,
                                                                         _mm512_set1_epi8((char)0x8D));
                    __mmask64 is_83_uppercase_m64 = is_83_range_m64 | is_83_c7_m64 | is_83_cd_m64;
                    __mmask64 is_safe_ea_m64 = is_ea_lead_m64 & ~(is_ea_complex_m64 >> 1);
                    __mmask64 is_c2_lead_m64 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xC2));
                    __m512i second_bytes_for_e2_u8x64 = _mm512_permutexvar_epi8(
                        _mm512_add_epi8(indices_u8x64, _mm512_set1_epi8(1)), source_vec.zmm);
                    __mmask64 safe_e2_positions_m64 = is_e2_lead_m64 & (load_m64 >> 1); 
                    __mmask64 is_safe_e2_m64 = safe_e2_positions_m64 &
                                               _mm512_mask_cmplt_epu8_mask(
                                                   safe_e2_positions_m64,
                                                   _mm512_sub_epi8(second_bytes_for_e2_u8x64,
                                                                   _mm512_set1_epi8((char)0x80)),
                                                   _mm512_set1_epi8(0x04)); 
                    __mmask64 is_valid_georgian_mix_m64 = ~is_non_ascii_m64 | is_georgian_e1_m64 | is_safe_e2_m64 |
                                                          is_cont_m64 | is_safe_ea_m64 | is_c2_lead_m64;
                    __mmask64 is_foldable_2byte_m64 = is_two_byte_lead_m64 & ~is_c2_lead_m64;
                    __mmask64 is_unsafe_e2_m64 = is_e2_lead_m64 & ~is_safe_e2_m64;
                    is_valid_georgian_mix_m64 &= ~(is_foldable_2byte_m64 | is_four_byte_lead_m64 | is_ef_lead_m64 |
                                                   is_unsafe_e2_m64) &
                                                 ~malformed_lead_m64;
                    sz_size_t georgian_length = sz_icelake_first_invalid_(is_valid_georgian_mix_m64, load_m64,
                                                                          chunk_size);
                    if (georgian_length >= 1) {
                        __mmask64 prefix_m64 = sz_u64_mask_until_(georgian_length);
                        __mmask64 leads3_in_prefix_m64 = is_three_byte_lead_m64 & prefix_m64;
                        __mmask64 safe3_m64 = georgian_length >= 3 ? sz_u64_mask_until_(georgian_length - 2) : 0;
                        __mmask64 unsafe3_m64 = leads3_in_prefix_m64 & ~safe3_m64;
                        __mmask64 leads2_in_prefix_m64 = is_c2_lead_m64 & prefix_m64;
                        __mmask64 safe2_m64 = georgian_length >= 2 ? sz_u64_mask_until_(georgian_length - 1) : 0;
                        __mmask64 unsafe2_m64 = leads2_in_prefix_m64 & ~safe2_m64;
                        __mmask64 unsafe_m64 = unsafe3_m64 | unsafe2_m64;
                        if (unsafe_m64) georgian_length = (int)_tzcnt_u64(unsafe_m64);
                    }
                    if (georgian_length >= 2) {
                        __mmask64 prefix_m64 = sz_u64_mask_until_(georgian_length);
                        __mmask64 uppercase_leads_m64 = ((is_82_uppercase_m64 | is_83_uppercase_m64) >> 2) &
                                                        is_georgian_e1_m64;
                        uppercase_leads_m64 &= prefix_m64;
                        __m512i folded_u8x64 = source_vec.zmm;
                        folded_u8x64 = _mm512_mask_blend_epi8(uppercase_leads_m64, folded_u8x64,
                                                              _mm512_set1_epi8((char)0xE2));
                        __mmask64 uppercase_second_pos_m64 = uppercase_leads_m64 << 1;
                        folded_u8x64 = _mm512_mask_blend_epi8(uppercase_second_pos_m64, folded_u8x64,
                                                              _mm512_set1_epi8((char)0xB4));
                        __mmask64 prefix_82_upper_m64 = is_82_uppercase_m64 & prefix_m64;
                        __mmask64 prefix_83_upper_m64 = is_83_uppercase_m64 & prefix_m64;
                        folded_u8x64 = _mm512_mask_sub_epi8(folded_u8x64, prefix_82_upper_m64, folded_u8x64,
                                                            _mm512_set1_epi8(0x20));
                        folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, prefix_83_upper_m64, folded_u8x64,
                                                            _mm512_set1_epi8(0x20));
                        folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64,
                                                            sz_icelake_is_ascii_upper_(source_vec.zmm) & prefix_m64,
                                                            folded_u8x64, ascii_case_offset_u8x64);
                        __mmask64 c2_in_prefix_m64 = is_c2_lead_m64 & prefix_m64;
                        __mmask64 c2_second_pos_m64 = c2_in_prefix_m64 << 1;
                        __mmask64 is_micro_second_m64 =
                            c2_second_pos_m64 & _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xB5));
                        __mmask64 is_micro_lead_m64 = is_micro_second_m64 >> 1;
                        folded_u8x64 = _mm512_mask_blend_epi8(is_micro_lead_m64, folded_u8x64,
                                                              _mm512_set1_epi8((char)0xCE));
                        folded_u8x64 = _mm512_mask_blend_epi8(is_micro_second_m64, folded_u8x64,
                                                              _mm512_set1_epi8((char)0xBC));
                        _mm512_mask_storeu_epi8(target, prefix_m64, folded_u8x64);
                        target += georgian_length, source += georgian_length, source_length -= georgian_length;
                        continue;
                    }
                }
            }
            if (is_e1_lead_m64 && source_length >= 3) {
                __m512i latin_ext_second_bytes_u8x64 = _mm512_permutexvar_epi8(
                    _mm512_add_epi8(indices_u8x64, _mm512_set1_epi8(1)), source_vec.zmm);
                __m512i latin_ext_third_bytes_u8x64 = _mm512_permutexvar_epi8(
                    _mm512_add_epi8(indices_u8x64, _mm512_set1_epi8(2)), source_vec.zmm);
                __mmask64 safe_e1_m64 = is_e1_lead_m64 & (load_m64 >> 1);
                __mmask64 is_latin_ext_e1_m64 = _mm512_mask_cmplt_epu8_mask(
                    safe_e1_m64, _mm512_sub_epi8(latin_ext_second_bytes_u8x64, _mm512_set1_epi8((char)0xB8)),
                    _mm512_set1_epi8(0x04)); 
                __mmask64 is_ba_second_m64 = _mm512_mask_cmpeq_epi8_mask(safe_e1_m64, latin_ext_second_bytes_u8x64,
                                                                         _mm512_set1_epi8((char)0xBA));
                __mmask64 is_special_third_m64 = is_ba_second_m64 & _mm512_mask_cmplt_epu8_mask(
                                                                        is_ba_second_m64,
                                                                        _mm512_sub_epi8(latin_ext_third_bytes_u8x64,
                                                                                        _mm512_set1_epi8((char)0x96)),
                                                                        _mm512_set1_epi8(0x09));
                __mmask64 non_latin_ext_e1_m64 = safe_e1_m64 & ~is_latin_ext_e1_m64;
                if (!non_latin_ext_e1_m64 && is_latin_ext_e1_m64 && !is_special_third_m64) {
                    __mmask64 is_valid_latin_ext_m64 = ~is_non_ascii_m64 | is_latin_ext_e1_m64 | is_cont_m64;
                    is_valid_latin_ext_m64 &= ~(is_four_byte_lead_m64 | is_ef_lead_m64) & ~malformed_lead_m64;
                    sz_size_t latin_ext_length = sz_icelake_first_invalid_(is_valid_latin_ext_m64, load_m64,
                                                                           chunk_size);
                    if (latin_ext_length >= 1) {
                        __mmask64 prefix_m64 = sz_u64_mask_until_(latin_ext_length);
                        __mmask64 leads_in_prefix_m64 = is_three_byte_lead_m64 & prefix_m64;
                        __mmask64 safe_m64 = latin_ext_length >= 3 ? sz_u64_mask_until_(latin_ext_length - 2) : 0;
                        __mmask64 unsafe_m64 = leads_in_prefix_m64 & ~safe_m64;
                        if (unsafe_m64) latin_ext_length = (int)_tzcnt_u64(unsafe_m64);
                    }
                    if (latin_ext_length >= 3) {
                        __mmask64 prefix_m64 = sz_u64_mask_until_(latin_ext_length);
                        __mmask64 third_positions_m64 = (is_latin_ext_e1_m64 & prefix_m64) << 2;
                        third_positions_m64 &= prefix_m64;
                        __mmask64 is_even_third_m64 = ~_mm512_test_epi8_mask(source_vec.zmm, _mm512_set1_epi8(0x01)) &
                                                      third_positions_m64;
                        __m512i folded_u8x64 = source_vec.zmm;
                        folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, is_even_third_m64, folded_u8x64,
                                                            _mm512_set1_epi8(0x01));
                        folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64,
                                                            sz_icelake_is_ascii_upper_(source_vec.zmm) & prefix_m64,
                                                            folded_u8x64, ascii_case_offset_u8x64);
                        _mm512_mask_storeu_epi8(target, prefix_m64, folded_u8x64);
                        target += latin_ext_length, source += latin_ext_length, source_length -= latin_ext_length;
                        continue;
                    }
                }
            }
            __mmask64 is_ea_lead_complex_m64 = is_ea_complex_m64 >> 1; 
            __mmask64 is_safe_three_byte_lead_m64 = is_three_byte_lead_m64 & ~is_e1_lead_m64 & ~is_e2_lead_m64 &
                                                    ~is_ea_lead_complex_m64 & ~is_ef_lead_m64;
            __mmask64 is_valid_mixed_m64 = ~is_non_ascii_m64 | is_safe_three_byte_lead_m64 | is_cont_m64;
            is_valid_mixed_m64 &= ~is_four_byte_lead_m64 & ~malformed_lead_m64;
            sz_size_t three_byte_length = sz_icelake_first_invalid_(is_valid_mixed_m64, load_m64, chunk_size);
            if (three_byte_length >= 1) {
                __mmask64 all_leads_m64 = is_three_byte_lead_m64 & sz_u64_mask_until_(three_byte_length);
                __mmask64 safe_leads_m64 = three_byte_length >= 3 ? sz_u64_mask_until_(three_byte_length - 2) : 0;
                __mmask64 unsafe_leads_m64 = all_leads_m64 & ~safe_leads_m64;
                if (unsafe_leads_m64) three_byte_length = (int)_tzcnt_u64(unsafe_leads_m64);
            }
            if (three_byte_length >= 2) {
                __mmask64 prefix_3_m64 = sz_u64_mask_until_(three_byte_length);
                __mmask64 three_byte_leads_in_prefix_m64 = is_three_byte_lead_m64 & prefix_3_m64;
                __mmask64 is_unsafe_e2_m64 = is_e2_lead_m64 & three_byte_leads_in_prefix_m64;
                __mmask64 problematic_leads_m64 = (is_e1_lead_m64 | is_ef_lead_m64 | is_unsafe_e2_m64) &
                                                  three_byte_leads_in_prefix_m64;
                if (!problematic_leads_m64) {
                    _mm512_mask_storeu_epi8(target, prefix_3_m64,
                                            sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, prefix_3_m64));
                    target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                    continue;
                }
                __mmask64 is_e1_in_prefix_m64 = is_e1_lead_m64 & three_byte_leads_in_prefix_m64;
                if (is_e1_in_prefix_m64) {
                    __m512i e1_second_bytes_u8x64 = _mm512_permutexvar_epi8(
                        _mm512_add_epi8(indices_u8x64, _mm512_set1_epi8(1)), source_vec.zmm);
                    __m512i e1_third_bytes_u8x64 = _mm512_permutexvar_epi8(
                        _mm512_add_epi8(indices_u8x64, _mm512_set1_epi8(2)), source_vec.zmm);
                    __mmask64 is_greek_ext_m64 = _mm512_mask_cmplt_epu8_mask(
                        is_e1_in_prefix_m64, _mm512_sub_epi8(e1_second_bytes_u8x64, _mm512_set1_epi8((char)0xBC)),
                        _mm512_set1_epi8(0x04));
                    __mmask64 is_ba_second_m64 = _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix_m64, e1_second_bytes_u8x64,
                                                                             _mm512_set1_epi8((char)0xBA));
                    __mmask64 is_special_third_m64 = _mm512_mask_cmplt_epu8_mask(
                        is_ba_second_m64, _mm512_sub_epi8(e1_third_bytes_u8x64, _mm512_set1_epi8((char)0x96)),
                        _mm512_set1_epi8(0x09));
                    __mmask64 needs_serial_e1_m64 = is_greek_ext_m64 | is_special_third_m64;
                    if (needs_serial_e1_m64) {
                        sz_size_t first_special_byte = (int)_tzcnt_u64(needs_serial_e1_m64);
                        if (first_special_byte == 0) {
                            sz_rune_t rune;
                            sz_rune_length_t const rune_length = sz_rune_decode(source, source + source_length, &rune);
                            if (rune_length == sz_rune_invalid_k) {
                                *target++ = *source++;
                                source_length -= 1;
                                continue;
                            }
                            sz_rune_t folded_runes[3]; 
                            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
                            for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
                                target += sz_rune_encode(folded_runes[rune_index], (sz_u8_t *)target);
                            source += rune_length;
                            source_length -= rune_length;
                            continue;
                        }
                        three_byte_length = first_special_byte;
                        prefix_3_m64 = sz_u64_mask_until_(three_byte_length);
                        three_byte_leads_in_prefix_m64 = is_three_byte_lead_m64 & prefix_3_m64;
                        is_ef_lead_m64 &= prefix_3_m64;
                        is_e1_in_prefix_m64 = is_e1_lead_m64 & three_byte_leads_in_prefix_m64;
                    }
                    __mmask64 is_latin_ext_add_m64 = _mm512_mask_cmplt_epu8_mask(
                        is_e1_in_prefix_m64, _mm512_sub_epi8(e1_second_bytes_u8x64, _mm512_set1_epi8((char)0xB8)),
                        _mm512_set1_epi8(0x04)); 
                    if (is_latin_ext_add_m64) {
                        __mmask64 third_positions_m64 = is_latin_ext_add_m64 << 2;
                        __mmask64 is_even_third_m64 = ~_mm512_test_epi8_mask(source_vec.zmm, _mm512_set1_epi8(0x01));
                        __mmask64 fold_third_m64 = third_positions_m64 & is_even_third_m64 & prefix_3_m64;
                        __mmask64 is_upper_ascii_m64 = sz_icelake_is_ascii_upper_(source_vec.zmm);
                        __m512i folded_u8x64 = _mm512_mask_add_epi8(
                            source_vec.zmm, (fold_third_m64 | is_upper_ascii_m64) & prefix_3_m64, source_vec.zmm,
                            _mm512_set1_epi8(0x01));
                        folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, is_upper_ascii_m64 & prefix_3_m64,
                                                            folded_u8x64, _mm512_set1_epi8(0x1F));
                        _mm512_mask_storeu_epi8(target, prefix_3_m64, folded_u8x64);
                        target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                        continue;
                    }
                    __mmask64 is_82_second_m64 = _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix_m64, e1_second_bytes_u8x64,
                                                                             _mm512_set1_epi8((char)0x82));
                    __mmask64 is_83_second_m64 = _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix_m64, e1_second_bytes_u8x64,
                                                                             _mm512_set1_epi8((char)0x83));
                    __mmask64 is_georgian_second_m64 = is_82_second_m64 | is_83_second_m64;
                    if (is_georgian_second_m64) {
                        __mmask64 third_pos_82_m64 = is_82_second_m64 << 2;
                        __mmask64 third_pos_83_m64 = is_83_second_m64 << 2;
                        __mmask64 is_82_valid_m64 = _mm512_mask_cmplt_epu8_mask(
                            third_pos_82_m64, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                            _mm512_set1_epi8(0x20));
                        __mmask64 is_83_valid_m64 = _mm512_mask_cmplt_epu8_mask(
                            third_pos_83_m64, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                            _mm512_set1_epi8(0x06));
                        __mmask64 georgian_leads_m64 = ((is_82_valid_m64 | is_83_valid_m64) >> 2) & is_e1_in_prefix_m64;
                        if (georgian_leads_m64) {
                            __m512i folded_u8x64 = source_vec.zmm;
                            folded_u8x64 = _mm512_mask_blend_epi8(georgian_leads_m64, folded_u8x64,
                                                                  _mm512_set1_epi8((char)0xE2));
                            __mmask64 georgian_second_pos_m64 = georgian_leads_m64 << 1;
                            folded_u8x64 = _mm512_mask_blend_epi8(georgian_second_pos_m64, folded_u8x64,
                                                                  _mm512_set1_epi8((char)0xB4));
                            folded_u8x64 = _mm512_mask_sub_epi8(folded_u8x64, is_82_valid_m64, folded_u8x64,
                                                                _mm512_set1_epi8(0x20));
                            folded_u8x64 = _mm512_mask_add_epi8(folded_u8x64, is_83_valid_m64, folded_u8x64,
                                                                _mm512_set1_epi8(0x20));
                            folded_u8x64 = _mm512_mask_add_epi8(
                                folded_u8x64, sz_icelake_is_ascii_upper_(source_vec.zmm) & prefix_3_m64, folded_u8x64,
                                ascii_case_offset_u8x64);
                            _mm512_mask_storeu_epi8(target, prefix_3_m64, folded_u8x64);
                            target += three_byte_length, source += three_byte_length,
                                source_length -= three_byte_length;
                            continue;
                        }
                    }
                }
                __mmask64 is_ef_in_prefix_m64 = is_ef_lead_m64 & three_byte_leads_in_prefix_m64;
                if (is_ef_in_prefix_m64) {
                    __m512i ef_second_bytes_u8x64 = _mm512_permutexvar_epi8(
                        _mm512_add_epi8(indices_u8x64, _mm512_set1_epi8(1)), source_vec.zmm);
                    __m512i ef_third_bytes_u8x64 = _mm512_permutexvar_epi8(
                        _mm512_add_epi8(indices_u8x64, _mm512_set1_epi8(2)), source_vec.zmm);
                    __mmask64 is_ef_bc_m64 = _mm512_mask_cmpeq_epi8_mask(is_ef_in_prefix_m64, ef_second_bytes_u8x64,
                                                                         _mm512_set1_epi8((char)0xBC));
                    __mmask64 is_fullwidth_az_m64 = _mm512_mask_cmplt_epu8_mask(
                        is_ef_bc_m64, _mm512_sub_epi8(ef_third_bytes_u8x64, _mm512_set1_epi8((char)0xA1)),
                        _mm512_set1_epi8(0x1A));
                    if (is_fullwidth_az_m64) {
                        __mmask64 third_byte_positions_m64 = is_fullwidth_az_m64 << 2;
                        __mmask64 fold_m64 = (third_byte_positions_m64 | sz_icelake_is_ascii_upper_(source_vec.zmm)) &
                                             prefix_3_m64;
                        __m512i folded_u8x64 = _mm512_mask_add_epi8(source_vec.zmm, fold_m64, source_vec.zmm,
                                                                    ascii_case_offset_u8x64);
                        _mm512_mask_storeu_epi8(target, prefix_3_m64, folded_u8x64);
                        target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                        continue;
                    }
                }
                if (!is_unsafe_e2_m64) {
                    _mm512_mask_storeu_epi8(target, prefix_3_m64,
                                            sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, prefix_3_m64));
                    target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                    continue;
                }
            }
        }
        {
            __mmask64 is_valid_four_byte_only_m64 = (is_four_byte_lead_m64 | is_cont_m64) & ~malformed_lead_m64;
            sz_size_t four_byte_length = sz_icelake_first_invalid_(is_valid_four_byte_only_m64, load_m64, chunk_size);
            if (four_byte_length >= 1) {
                __mmask64 all_leads_m64 = is_four_byte_lead_m64 & sz_u64_mask_until_(four_byte_length);
                __mmask64 safe_leads_m64 = four_byte_length >= 4 ? sz_u64_mask_until_(four_byte_length - 3) : 0;
                __mmask64 unsafe_leads_m64 = all_leads_m64 & ~safe_leads_m64;
                if (unsafe_leads_m64) four_byte_length = (int)_tzcnt_u64(unsafe_leads_m64);
            }
            if (four_byte_length >= 4) {
                __m512i f0_second_bytes_u8x64 = _mm512_permutexvar_epi8(
                    _mm512_add_epi8(indices_u8x64, _mm512_set1_epi8(1)), source_vec.zmm);
                __mmask64 is_emoji_lead_m64 = _mm512_cmpge_epu8_mask(f0_second_bytes_u8x64,
                                                                     _mm512_set1_epi8((char)0x9F));
                __mmask64 prefix_4_m64 = sz_u64_mask_until_(four_byte_length);
                __mmask64 four_byte_leads_in_prefix_m64 = is_four_byte_lead_m64 & prefix_4_m64;
                if ((four_byte_leads_in_prefix_m64 & ~is_emoji_lead_m64) == 0) {
                    _mm512_mask_storeu_epi8(target, prefix_4_m64, source_vec.zmm);
                    target += four_byte_length, source += four_byte_length, source_length -= four_byte_length;
                    continue;
                }
            }
        }
        {
            sz_rune_t rune;
            sz_rune_length_t const rune_length = sz_rune_decode(source, source + source_length, &rune);
            if (rune_length == sz_rune_invalid_k) {
                *target++ = *source++;
                source_length -= 1;
                continue;
            }
            sz_rune_t folded_runes[3]; 
            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
            for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
                target += sz_rune_encode(folded_runes[rune_index], (sz_u8_t *)target);
            source += rune_length;
            source_length -= rune_length;
        }
    }
    return (sz_size_t)(target - target_start);
}
#undef sz_icelake_is_ascii_upper_
#undef sz_icelake_fold_ascii_
#undef sz_icelake_fold_ascii_in_prefix_
#undef sz_icelake_transform_georgian_
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
