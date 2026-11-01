/**
 *  @brief Ice Lake (AVX-512) uncased UTF-8 search, comparison & invariance kernels.
 *  @file include/stringzilla/utf8_uncased/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_ICELAKE_H_
#define STRINGZILLA_UTF8_UNCASED_ICELAKE_H_
#include "stringzilla/find/skylake.h" 
#include "stringzilla/utf8_uncased/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                                  \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif
#pragma region ASCII Uncased Find
/**
 *  @brief Fold a ZMM register using ASCII case folding rules.
 *  @sa sz_utf8_uncased_rune_ascii_invariant_k
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_INLINE __m512i sz_utf8_uncased_search_icelake_ascii_fold_zmm_(__m512i text_u8x64) {
    __m512i const a_upper_u8x64 = _mm512_set1_epi8('A');
    __m512i const range26_u8x64 = _mm512_set1_epi8(26);
    __m512i const x_20_u8x64 = _mm512_set1_epi8(0x20);
    __mmask64 upper_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, a_upper_u8x64), range26_u8x64);
    return _mm512_mask_add_epi8(text_u8x64, upper_m64, text_u8x64, x_20_u8x64);
}
/**
 *  @brief 3-probe ASCII uncased search using XOR + VPTERNLOG + VPTESTNMB.
 *
 *  For needles with folded_slice_length ≤ 3, probes at positions 0, mid, last cover ALL bytes.
 *  Uses parallel loads from different offsets, XOR for difference detection,
 *  VPTERNLOG to combine all 3, and VPTESTNMB to find matches.
 *  No window verification needed since probes cover the entire window.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_icelake_ascii_3probe_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                     
    sz_cptr_t needle, sz_size_t needle_length,                         
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,          
    sz_size_t *matched_length) {
    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_size_t const step = 64 - folded_window_length + 1;
    sz_size_t const offset_second = folded_window_length / 2;
    sz_size_t const offset_last = folded_window_length - 1;
    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);
    sz_u512_vec_t haystack_first_vec, haystack_second_vec, haystack_last_vec;
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr + 64 + offset_last <= haystack_end) {
        haystack_first_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr));
        haystack_second_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_loadu_si512(haystack_ptr + offset_second));
        haystack_last_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_loadu_si512(haystack_ptr + offset_last));
        __m512i diff_first_u8x64 = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_u8x64 = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_last_u8x64 = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);
        __m512i combined_u8x64 = _mm512_ternarylogic_epi64(diff_first_u8x64, diff_second_u8x64, diff_last_u8x64, 0xFE);
        __mmask64 matches_m64 = _mm512_testn_epi8_mask(combined_u8x64, combined_u8x64);
        matches_m64 &= sz_u64_mask_until_(step);
        for (; matches_m64; matches_m64 &= matches_m64 - 1) {
            sz_size_t candidate_offset = (int)_tzcnt_u64(matches_m64);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;
            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               
                haystack, haystack_length,                                                                 
                needle, needle_length,                                                                     
                haystack_candidate_ptr - haystack, folded_window_length,                                   
                needle_metadata->offset_in_unfolded,                                                       
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, 
                matched_length);
            if (match) { return match; }
        }
        haystack_ptr += step;
    }
    sz_size_t remaining = (sz_size_t)(haystack_end - haystack_ptr);
    if (remaining >= folded_window_length) {
        sz_size_t valid_starts = remaining - folded_window_length + 1;
        __mmask64 valid_m64 = sz_u64_mask_until_(valid_starts);
        __mmask64 load_first_m64 = sz_u64_mask_until_(remaining);
        __mmask64 load_second_m64 = (remaining > offset_second) ? sz_u64_mask_until_(remaining - offset_second) : 0;
        __mmask64 load_last_m64 = (remaining > offset_last) ? sz_u64_mask_until_(remaining - offset_last) : 0;
        haystack_first_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_first_m64, haystack_ptr));
        haystack_second_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_second_m64, haystack_ptr + offset_second));
        haystack_last_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_last_m64, haystack_ptr + offset_last));
        __m512i diff_first_u8x64 = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_u8x64 = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_last_u8x64 = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);
        __m512i combined_u8x64 = _mm512_ternarylogic_epi64(diff_first_u8x64, diff_second_u8x64, diff_last_u8x64, 0xFE);
        __mmask64 matches_m64 = _mm512_testn_epi8_mask(combined_u8x64, combined_u8x64);
        matches_m64 &= valid_m64;
        for (; matches_m64; matches_m64 &= matches_m64 - 1) {
            sz_size_t candidate_offset = (int)_tzcnt_u64(matches_m64);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;
            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               
                haystack, haystack_length,                                                                 
                needle, needle_length,                                                                     
                haystack_candidate_ptr - haystack, folded_window_length,                                   
                needle_metadata->offset_in_unfolded,                                                       
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, 
                matched_length);
            if (match) { return match; }
        }
    }
    return SZ_NULL_CHAR;
}
/**
 *  @brief 4-probe ASCII uncased search using XOR + VPTERNLOG + VPTESTNMB.
 *
 *  For needles with folded_slice_length ≥ 4, probes at 4 positions filter candidates quickly.
 *  Uses parallel loads, XOR for difference detection, VPTERNLOG + VPOR to combine,
 *  and VPTESTNMB to find matches. Window verification IS required since probes
 *  don't cover all positions in the folded window.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_icelake_ascii_4probe_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                     
    sz_cptr_t needle, sz_size_t needle_length,                         
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,          
    sz_size_t *matched_length) {
    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_size_t const step = 64 - folded_window_length + 1;
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;
    __mmask16 const folded_window_m16 = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);
    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);
    sz_u512_vec_t haystack_first_vec, haystack_second_vec, haystack_third_vec, haystack_last_vec;
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr + 64 + offset_last <= haystack_end) {
        haystack_first_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr));
        haystack_second_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_loadu_si512(haystack_ptr + offset_second));
        haystack_third_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_loadu_si512(haystack_ptr + offset_third));
        haystack_last_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_loadu_si512(haystack_ptr + offset_last));
        __m512i diff_first_u8x64 = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_u8x64 = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_third_u8x64 = _mm512_xor_si512(haystack_third_vec.zmm, probe_third_vec.zmm);
        __m512i diff_last_u8x64 = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);
        __m512i combined_u8x64 = _mm512_ternarylogic_epi64(diff_first_u8x64, diff_second_u8x64, diff_third_u8x64, 0xFE);
        combined_u8x64 = _mm512_or_si512(combined_u8x64, diff_last_u8x64);
        __mmask64 matches_m64 = _mm512_testn_epi8_mask(combined_u8x64, combined_u8x64);
        matches_m64 &= sz_u64_mask_until_(step);
        for (; matches_m64; matches_m64 &= matches_m64 - 1) {
            sz_size_t candidate_offset = (int)_tzcnt_u64(matches_m64);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;
            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_m16, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(
                sz_utf8_uncased_search_icelake_ascii_fold_zmm_(_mm512_castsi128_si512(haystack_candidate_vec.xmm)));
            __mmask16 window_mismatch_m16 = _mm_mask_cmpneq_epi8_mask(folded_window_m16, haystack_candidate_vec.xmm,
                                                                      needle_window_vec.xmm);
            if (window_mismatch_m16) continue;
            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               
                haystack, haystack_length,                                                                 
                needle, needle_length,                                                                     
                haystack_candidate_ptr - haystack, folded_window_length,                                   
                needle_metadata->offset_in_unfolded,                                                       
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, 
                matched_length);
            if (match) { return match; }
        }
        haystack_ptr += step;
    }
    sz_size_t remaining = (sz_size_t)(haystack_end - haystack_ptr);
    if (remaining >= folded_window_length) {
        sz_size_t valid_starts = remaining - folded_window_length + 1;
        __mmask64 valid_m64 = sz_u64_mask_until_(valid_starts);
        __mmask64 load_first_m64 = sz_u64_mask_until_(remaining);
        __mmask64 load_second_m64 = (remaining > offset_second) ? sz_u64_mask_until_(remaining - offset_second) : 0;
        __mmask64 load_third_m64 = (remaining > offset_third) ? sz_u64_mask_until_(remaining - offset_third) : 0;
        __mmask64 load_last_m64 = (remaining > offset_last) ? sz_u64_mask_until_(remaining - offset_last) : 0;
        haystack_first_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_first_m64, haystack_ptr));
        haystack_second_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_second_m64, haystack_ptr + offset_second));
        haystack_third_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_third_m64, haystack_ptr + offset_third));
        haystack_last_vec.zmm = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_last_m64, haystack_ptr + offset_last));
        __m512i diff_first_u8x64 = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_u8x64 = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_third_u8x64 = _mm512_xor_si512(haystack_third_vec.zmm, probe_third_vec.zmm);
        __m512i diff_last_u8x64 = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);
        __m512i combined_u8x64 = _mm512_ternarylogic_epi64(diff_first_u8x64, diff_second_u8x64, diff_third_u8x64, 0xFE);
        combined_u8x64 = _mm512_or_si512(combined_u8x64, diff_last_u8x64);
        __mmask64 matches_m64 = _mm512_testn_epi8_mask(combined_u8x64, combined_u8x64);
        matches_m64 &= valid_m64;
        for (; matches_m64; matches_m64 &= matches_m64 - 1) {
            sz_size_t candidate_offset = (int)_tzcnt_u64(matches_m64);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;
            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_m16, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(
                sz_utf8_uncased_search_icelake_ascii_fold_zmm_(_mm512_castsi128_si512(haystack_candidate_vec.xmm)));
            __mmask16 window_mismatch_m16 = _mm_mask_cmpneq_epi8_mask(folded_window_m16, haystack_candidate_vec.xmm,
                                                                      needle_window_vec.xmm);
            if (window_mismatch_m16) continue;
            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               
                haystack, haystack_length,                                                                 
                needle, needle_length,                                                                     
                haystack_candidate_ptr - haystack, folded_window_length,                                   
                needle_metadata->offset_in_unfolded,                                                       
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, 
                matched_length);
            if (match) { return match; }
        }
    }
    return SZ_NULL_CHAR;
}
#pragma endregion // ASCII Uncased Find
#pragma region Scripted Uncased Find
/** @brief Folds one ZMM register of haystack text using script-specific rules. */
typedef __m512i (*sz_utf8_uncased_fold_zmm_t)(__m512i text_u8x64);
/**
 *  @brief Flags positions of "danger" characters that fold to a different byte width.
 *  @param load_m64 Bitmask of the bytes actually loaded from the haystack, for tail-safe range checks.
 */
typedef __mmask64 (*sz_utf8_uncased_alarm_zmm_t)(__m512i text_u8x64, __mmask64 load_m64);
/**
 *  @brief Shared scan loop behind all script-specific uncased searches.
 *
 *  Scans the entire haystack from byte 0, looking for the folded window pattern.
 *  When found, verifies the head (backwards) and tail (forwards) using codepoint-by-codepoint
 *  comparison to handle variable-width folding correctly.
 *
 *  Every per-script kernel is a thin wrapper passing its own @p fold and @p alarm callbacks.
 *  The driver is force-inlined into each wrapper, so the callbacks resolve to direct calls
 *  with no indirect branches in the emitted code.
 *
 *  @param fold Script-specific ZMM case-folding callback.
 *  @param alarm Script-specific danger detection callback, or NULL if the script has no
 *      danger characters: the danger branch disappears and the full step is used.
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_INLINE sz_cptr_t sz_utf8_uncased_search_icelake_scripted_( 
    sz_utf8_uncased_fold_zmm_t fold,                                 
    sz_utf8_uncased_alarm_zmm_t alarm,                               
    sz_cptr_t haystack, sz_size_t haystack_length,                   
    sz_cptr_t needle, sz_size_t needle_length,                       
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,        
    sz_size_t *matched_length) {
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");
    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");
    __mmask16 const folded_window_m16 = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;
    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);
    sz_rune_t needle_first_safe_folded_rune = 0;
    if (alarm) sz_rune_decode_unchecked((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune);
    sz_u512_vec_t haystack_vec;
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;
        sz_size_t const chunk_size = available < 64 ? available : 64;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        sz_size_t const step = !alarm ? valid_starts : valid_starts > 2 ? valid_starts - 2 : 1;
        __mmask64 const load_m64 = sz_u64_mask_until_(chunk_size);
        __mmask64 const valid_m64 = sz_u64_mask_until_(valid_starts);
        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_m64, haystack_ptr);
        if (alarm) {
            __mmask64 danger_m64 = alarm(haystack_vec.zmm, load_m64);
            if (danger_m64) {
                sz_cptr_t match = sz_utf8_uncased_search_in_danger_zone_( 
                    haystack, haystack_length,                            
                    needle, needle_length,                                
                    haystack_ptr, chunk_size,                             
                    needle_first_safe_folded_rune,                        
                    needle_metadata->offset_in_unfolded,                  
                    matched_length);
                if (match) return match;
                haystack_ptr += step;
                continue;
            }
        }
        haystack_vec.zmm = fold(haystack_vec.zmm);
        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= valid_m64;
        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = (int)_tzcnt_u64(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;
            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_m16, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(
                fold(_mm512_castsi128_si512(haystack_candidate_vec.xmm)));
            __mmask16 window_mismatch_m16 = _mm_mask_cmpneq_epi8_mask(folded_window_m16, haystack_candidate_vec.xmm,
                                                                      needle_window_vec.xmm);
            if (window_mismatch_m16) continue;
            sz_cptr_t match = sz_utf8_uncased_verify_match_(                             
                haystack, haystack_length,                                               
                needle, needle_length,                                                   
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length, 
                needle_metadata->offset_in_unfolded,                                     
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, 
                matched_length);
            if (match) { return match; }
        }
        haystack_ptr += step;
    }
    if (alarm && haystack_ptr < haystack_end) {
        sz_cptr_t match = sz_utf8_uncased_search_in_danger_zone_(   
            haystack, haystack_length,                              
            needle, needle_length,                                  
            haystack_ptr, (sz_size_t)(haystack_end - haystack_ptr), 
            needle_first_safe_folded_rune,                          
            needle_metadata->offset_in_unfolded,                    
            matched_length);
        if (match) { return match; }
    }
    return SZ_NULL_CHAR;
}
#pragma endregion // Scripted Uncased Find
#pragma region Western European Uncased Find
/**
 *  @brief Fold a ZMM register using Western European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_western_europe_k
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_AUTO __m512i sz_utf8_uncased_search_icelake_western_europe_fold_naively_zmm_(__m512i text_u8x64) {
    __m512i result_u8x64 = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(text_u8x64);
    __m512i const x_20_u8x64 = _mm512_set1_epi8(0x20);
    __m512i const x_73_u8x64 = _mm512_set1_epi8('s');
    __m512i const x_c3_u8x64 = _mm512_set1_epi8((char)0xC3); 
    __m512i const x_9f_u8x64 = _mm512_set1_epi8((char)0x9F); 
    __m512i const x_80_u8x64 = _mm512_set1_epi8((char)0x80); 
    __m512i const x_1f_u8x64 = _mm512_set1_epi8((char)0x1F); 
    __m512i const x_97_u8x64 = _mm512_set1_epi8((char)0x97); 
    __mmask64 is_c3_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c3_u8x64);
    __mmask64 is_after_c3_m64 = is_c3_m64 << 1;
    __mmask64 is_eszett_second_m64 = _mm512_mask_cmpeq_epi8_mask(is_after_c3_m64, text_u8x64, x_9f_u8x64);
    __mmask64 is_eszett_m64 = is_eszett_second_m64 | (is_eszett_second_m64 >> 1);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_eszett_m64, x_73_u8x64);
    __mmask64 is_97_m64 = _mm512_mask_cmpeq_epi8_mask(is_after_c3_m64, text_u8x64, x_97_u8x64);
    __mmask64 is_latin1_upper_m64 = _mm512_mask_cmplt_epu8_mask(is_after_c3_m64 & ~is_eszett_second_m64 & ~is_97_m64,
                                                                _mm512_sub_epi8(text_u8x64, x_80_u8x64), x_1f_u8x64);
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_latin1_upper_m64, result_u8x64, x_20_u8x64);
    return result_u8x64;
}
/**
 *  @brief Fold a ZMM register using Western European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_western_europe_k
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_NOINLINE __m512i sz_utf8_uncased_search_icelake_western_europe_fold_efficiently_zmm_(__m512i text_u8x64) {
    return sz_utf8_uncased_search_icelake_western_europe_fold_naively_zmm_(text_u8x64);
}
/**
 *  @brief Naive alarm function for Western Europe danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E1 BA 9E: 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73) (3 bytes → 2 bytes)
 *  - E2 84 AA: 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B) (3 bytes → 1 byte)
 *  - E2 84 AB: 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5) (3 bytes → 2 bytes)
 *  - EF AC 80-86: Ligatures:
 *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
 *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
 *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
 *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
 *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
 *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
 *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
 *  - C5 BF: 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73) (2 bytes → 1 byte)
 *  - C3 9F: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) (2 bytes → 2 bytes, 1 rune → 2 runes)
 *
 *  Uses 12 CMPEQ operations (5 lead + 7 second byte checks).
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_HELPER_AUTO __mmask64 sz_utf8_uncased_search_icelake_western_europe_alarm_naively_zmm_(__m512i text_u8x64) {
    __m512i const x_e1_u8x64 = _mm512_set1_epi8((char)0xE1);
    __m512i const x_e2_u8x64 = _mm512_set1_epi8((char)0xE2);
    __m512i const x_ef_u8x64 = _mm512_set1_epi8((char)0xEF);
    __m512i const x_c5_u8x64 = _mm512_set1_epi8((char)0xC5);
    __m512i const x_c3_u8x64 = _mm512_set1_epi8((char)0xC3);
    __m512i const x_ba_u8x64 = _mm512_set1_epi8((char)0xBA);
    __m512i const x_84_u8x64 = _mm512_set1_epi8((char)0x84);
    __m512i const x_ac_u8x64 = _mm512_set1_epi8((char)0xAC);
    __m512i const x_bf_u8x64 = _mm512_set1_epi8((char)0xBF);
    __m512i const x_9f_u8x64 = _mm512_set1_epi8((char)0x9F);
    __m512i const x_aa_u8x64 = _mm512_set1_epi8((char)0xAA);
    __m512i const x_ab_u8x64 = _mm512_set1_epi8((char)0xAB);
    __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_e1_u8x64);
    __mmask64 is_e2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_e2_u8x64);
    __mmask64 is_ef_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ef_u8x64);
    __mmask64 is_c5_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c5_u8x64);
    __mmask64 is_c3_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c3_u8x64);
    __mmask64 is_ba_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ba_u8x64);
    __mmask64 is_84_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_84_u8x64);
    __mmask64 is_ac_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ac_u8x64);
    __mmask64 is_bf_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_bf_u8x64);
    __mmask64 is_9f_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_9f_u8x64);
    __mmask64 is_aa_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_aa_u8x64);
    __mmask64 is_ab_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ab_u8x64);
    __mmask64 is_b8_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB8));
    __mmask64 is_e1_ba_danger_m64 = (is_e1_m64 << 1) & is_ba_m64;
    if (is_e1_ba_danger_m64) {
        __mmask64 is_expanding_third_m64 = _mm512_cmplt_epu8_mask(
            _mm512_sub_epi8(text_u8x64, _mm512_set1_epi8((char)0x96)), _mm512_set1_epi8(0x09));
        is_e1_ba_danger_m64 &= is_expanding_third_m64 >> 1;
    }
    return is_e1_ba_danger_m64 |                                             
           ((is_e2_m64 << 1) & is_84_m64 & ((is_aa_m64 | is_ab_m64) >> 1)) | 
           ((is_ef_m64 << 1) & is_ac_m64) |                                  
           ((is_c5_m64 << 1) & is_bf_m64) |                                  
           ((is_c5_m64 << 1) & is_b8_m64) |                                  
           ((is_c3_m64 << 1) & is_9f_m64);                                   
}
/**
 *  @brief Optimized alarm function for Western Europe danger zone detection.
 *
 *  Every Western European danger sequence starts with a 2-byte (lead, second) pair with a UNIQUE
 *  second byte, so the pair test inverts: one VPERMB maps each second byte to the lead that would
 *  make it dangerous (sentinel 0x00 otherwise), another VPERMB materializes the previous byte
 *  in-register, and a single VPCMPEQB matches them up. Two qualifiers make it exact: the second
 *  byte must be a continuation (its low-6-bit VPERMB index aliases ASCII and lead bytes), and the
 *  previous byte must be >= C0 (kills the sentinel matching a 0x00 byte and the zeroed lane 0).
 *  The third-byte refinements (E1 BA only expands for thirds 96-9E, E2 84 only for AA/AB) sit
 *  behind a `pair` branch, so the hot path is 5 port-5 ops instead of the naive 12 CMPEQs.
 *
 *  @param text_u8x64 The text ZMM register to scan for danger bytes.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_HELPER_NOINLINE __mmask64 sz_utf8_uncased_search_icelake_western_europe_alarm_efficiently_zmm_(__m512i text_u8x64,
                                                                                                  __mmask64 load_m64) {
    sz_unused_(load_m64); 
    __m512i const previous_byte_indices_u8x64 = _mm512_set_epi8( 
        62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35,
        34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
        5, 4, 3, 2, 1, 0, 0);
    __m512i const expected_leads_lut_u8x64 = _mm512_set_epi8(
        (char)0xC5, 0, 0, 0, 0, (char)0xE1, 0, (char)0xC5, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, (char)0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        (char)0xC3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (char)0xE2, 0, 0, 0, 0);
    __m512i const x_80_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_u8x64 = _mm512_set1_epi8((char)0x40);
    __m512i const x_c0_u8x64 = _mm512_set1_epi8((char)0xC0);
    __m512i previous_u8x64 = _mm512_maskz_permutexvar_epi8(0xFFFFFFFFFFFFFFFEull, previous_byte_indices_u8x64,
                                                           text_u8x64);
    __m512i expected_leads_u8x64 = _mm512_permutexvar_epi8(text_u8x64, expected_leads_lut_u8x64);
    __mmask64 pair_m64 = _mm512_cmpeq_epi8_mask(previous_u8x64, expected_leads_u8x64);
    pair_m64 &= _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, x_80_u8x64), x_40_u8x64);
    pair_m64 &= _mm512_cmpge_epu8_mask(previous_u8x64, x_c0_u8x64);
    __mmask64 danger_m64 = 0;
    if (pair_m64) {
        __mmask64 is_ba_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xBA));
        __mmask64 is_84_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x84));
        __mmask64 is_expanding_third_m64 = _mm512_cmplt_epu8_mask(
            _mm512_sub_epi8(text_u8x64, _mm512_set1_epi8((char)0x96)), _mm512_set1_epi8(0x09));
        __mmask64 ba_pair_m64 = pair_m64 & is_ba_m64 & (is_expanding_third_m64 >> 1);
        __mmask64 is_aa_or_ab_m64 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(text_u8x64, _mm512_set1_epi8((char)0xFE)),
                                                           _mm512_set1_epi8((char)0xAA));
        __mmask64 kelvin_pair_m64 = pair_m64 & is_84_m64 & (is_aa_or_ab_m64 >> 1);
        danger_m64 = (pair_m64 & ~(is_ba_m64 | is_84_m64)) | ba_pair_m64 | kelvin_pair_m64;
    }
    sz_assert_(danger_m64 == sz_utf8_uncased_search_icelake_western_europe_alarm_naively_zmm_(text_u8x64) &&
               "Efficient Western European alarm must match naive implementation");
    return danger_m64;
}
/**
 *  @brief Western European uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_western_europe_k
 *
 *  Scans the entire haystack from byte 0, looking for the folded window pattern.
 *  When found, verifies the head (backwards) and tail (forwards) using codepoint-by-codepoint
 *  comparison to handle variable-width folding correctly (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)).
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_icelake_western_europe_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                       
    sz_cptr_t needle, sz_size_t needle_length,                           
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,            
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_icelake_scripted_( 
        sz_utf8_uncased_search_icelake_western_europe_fold_efficiently_zmm_,
        sz_utf8_uncased_search_icelake_western_europe_alarm_efficiently_zmm_, 
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}
#pragma endregion // Western European Uncased Find
#pragma region Central European Uncased Find
/**
 *  @brief Fold a ZMM register using Central European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_central_europe_k
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_AUTO __m512i sz_utf8_uncased_search_icelake_central_europe_fold_naively_zmm_(__m512i text_u8x64) {
    __m512i result_u8x64 = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(text_u8x64);
    __m512i const x_20_u8x64 = _mm512_set1_epi8((char)0x20); 
    __m512i const x_01_u8x64 = _mm512_set1_epi8((char)0x01); 
    __m512i const x_c3_u8x64 = _mm512_set1_epi8((char)0xC3);
    __m512i const x_80_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const x_9e_u8x64 = _mm512_set1_epi8((char)0x9E);
    __m512i const x_97_u8x64 = _mm512_set1_epi8((char)0x97);
    __m512i const x_c4_u8x64 = _mm512_set1_epi8((char)0xC4);
    __m512i const x_c5_u8x64 = _mm512_set1_epi8((char)0xC5);
    __mmask64 is_c3_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_c3_u8x64);
    __mmask64 is_after_c3_m64 = is_c3_m64 << 1;
    __mmask64 is_latin1_range_m64 = _mm512_mask_cmpge_epu8_mask(is_after_c3_m64, result_u8x64, x_80_u8x64);
    is_latin1_range_m64 &= _mm512_mask_cmple_epu8_mask(is_after_c3_m64, result_u8x64, x_9e_u8x64);
    __mmask64 is_97_m64 = _mm512_mask_cmpeq_epi8_mask(is_after_c3_m64, result_u8x64, x_97_u8x64);
    __mmask64 fold_latin1_m64 = is_latin1_range_m64 & ~is_97_m64;
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, fold_latin1_m64, result_u8x64, x_20_u8x64);
    __mmask64 is_c4_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_c4_u8x64);
    __mmask64 is_c5_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_c5_u8x64);
    __mmask64 is_after_c4_m64 = is_c4_m64 << 1;
    __mmask64 is_after_c5_m64 = is_c5_m64 << 1;
    __mmask64 is_even_m64 = _mm512_testn_epi8_mask(result_u8x64, x_01_u8x64); 
    __mmask64 is_odd_m64 = ~is_even_m64;
    __m512i const x_81_u8x64 = _mm512_set1_epi8((char)0x81);
    __m512i const x_87_u8x64 = _mm512_set1_epi8((char)0x87);
    __m512i const x_8a_u8x64 = _mm512_set1_epi8((char)0x8A);
    __m512i const x_b6_u8x64 = _mm512_set1_epi8((char)0xB6);
    __m512i const x_b9_u8x64 = _mm512_set1_epi8((char)0xB9);
    __m512i const x_bd_u8x64 = _mm512_set1_epi8((char)0xBD);
    __m512i const x_b7_u8x64 = _mm512_set1_epi8((char)0xB7);
    __mmask64 is_c4_80_b7_m64 = _mm512_mask_cmpge_epu8_mask(is_after_c4_m64, result_u8x64, x_80_u8x64);
    is_c4_80_b7_m64 &= _mm512_mask_cmple_epu8_mask(is_after_c4_m64, result_u8x64, x_b7_u8x64);
    __mmask64 is_c4_b9_bd_m64 = _mm512_mask_cmpge_epu8_mask(is_after_c4_m64, result_u8x64, x_b9_u8x64);
    is_c4_b9_bd_m64 &= _mm512_mask_cmple_epu8_mask(is_after_c4_m64, result_u8x64, x_bd_u8x64);
    __mmask64 is_c5_81_87_m64 = _mm512_mask_cmpge_epu8_mask(is_after_c5_m64, result_u8x64, x_81_u8x64);
    is_c5_81_87_m64 &= _mm512_mask_cmple_epu8_mask(is_after_c5_m64, result_u8x64, x_87_u8x64);
    __mmask64 is_c5_8a_b6_m64 = _mm512_mask_cmpge_epu8_mask(is_after_c5_m64, result_u8x64, x_8a_u8x64);
    is_c5_8a_b6_m64 &= _mm512_mask_cmple_epu8_mask(is_after_c5_m64, result_u8x64, x_b6_u8x64);
    __mmask64 is_c5_b9_bd_m64 = _mm512_mask_cmpge_epu8_mask(is_after_c5_m64, result_u8x64, x_b9_u8x64);
    is_c5_b9_bd_m64 &= _mm512_mask_cmple_epu8_mask(is_after_c5_m64, result_u8x64, x_bd_u8x64);
    __mmask64 fold_latext_m64 = (is_c4_80_b7_m64 & is_even_m64) | (is_c4_b9_bd_m64 & is_odd_m64) |
                                (is_c5_81_87_m64 & is_odd_m64) | (is_c5_8a_b6_m64 & is_even_m64) |
                                (is_c5_b9_bd_m64 & is_odd_m64);
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, fold_latext_m64, result_u8x64, x_01_u8x64);
    return result_u8x64;
}
/**
 *  @brief Fold a ZMM register using Central European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_central_europe_k
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_NOINLINE __m512i sz_utf8_uncased_search_icelake_central_europe_fold_efficiently_zmm_(__m512i text_u8x64) {
    __m512i const a_upper_u8x64 = _mm512_set1_epi8('A');
    __m512i const range26_u8x64 = _mm512_set1_epi8(26);
    __m512i const x_20_u8x64 = _mm512_set1_epi8(0x20);
    __mmask64 is_upper_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, a_upper_u8x64), range26_u8x64);
    __m512i result_u8x64 = _mm512_mask_add_epi8(text_u8x64, is_upper_m64, text_u8x64, x_20_u8x64);
    __m512i const x_c3_u8x64 = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c4_u8x64 = _mm512_set1_epi8((char)0xC4);
    __m512i const x_c5_u8x64 = _mm512_set1_epi8((char)0xC5);
    __m512i const x_80_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_u8x64 = _mm512_set1_epi8((char)0x40);
    __m512i const latin1_deltas_lut_u8x64 = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);
    __m512i const c4_deltas_lut_u8x64 = _mm512_set_epi8(
        0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    __m512i const extended_deltas_lut_u8x64 = _mm512_set_epi8(
        0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0);
    __mmask64 is_c3_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c3_u8x64);
    __mmask64 is_c4_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c4_u8x64);
    __mmask64 is_c5_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c5_u8x64);
    __mmask64 is_continuation_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, x_80_u8x64), x_40_u8x64);
    __m512i latin1_deltas_u8x64 = _mm512_permutexvar_epi8(text_u8x64, latin1_deltas_lut_u8x64);
    __m512i c4_deltas_u8x64 = _mm512_permutexvar_epi8(text_u8x64, c4_deltas_lut_u8x64);
    __m512i extended_deltas_u8x64 = _mm512_permutexvar_epi8(text_u8x64, extended_deltas_lut_u8x64);
    __mmask64 fold_latin1_m64 = (is_c3_m64 << 1) & is_continuation_m64;
    __mmask64 fold_c4_m64 = (is_c4_m64 << 1) & is_continuation_m64;
    __mmask64 fold_extended_m64 = (is_c5_m64 << 1) & is_continuation_m64;
    __m512i latin1_offsets_u8x64 = _mm512_maskz_mov_epi8(fold_latin1_m64, latin1_deltas_u8x64);
    __m512i extended_offsets_u8x64 = _mm512_maskz_mov_epi8(fold_extended_m64, extended_deltas_u8x64);
    __m512i parity_offsets_u8x64 = _mm512_maskz_mov_epi8(fold_c4_m64, c4_deltas_u8x64);
    __m512i offset_u8x64 = _mm512_ternarylogic_epi64(latin1_offsets_u8x64, extended_offsets_u8x64, parity_offsets_u8x64,
                                                     0xFE);
    result_u8x64 = _mm512_add_epi8(result_u8x64, offset_u8x64);
    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_uncased_search_icelake_central_europe_fold_naively_zmm_(text_u8x64),
                                      result_u8x64) == (__mmask64)-1 &&
               "Efficient Central European fold does not match naive implementation");
    return result_u8x64;
}
/**
 *  @brief Naive alarm function for Central Europe danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E2 84 AA: 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B) (3 bytes → 1 byte)
 *  - C3 9F: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) (2 bytes → 2 bytes, 1 rune → 2 runes)
 *  - C4 B0: 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87) (2 bytes → 3 bytes)
 *  - C5 BF: 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73) (2 bytes → 1 byte)
 *  - EF AC 80-86: Ligatures:
 *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
 *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
 *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
 *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
 *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
 *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
 *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
 *
 *  Uses 10 CMPEQ operations (5 lead + 5 second byte checks).
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_HELPER_AUTO __mmask64 sz_utf8_uncased_search_icelake_central_europe_alarm_naively_zmm_(__m512i text_u8x64) {
    __m512i const x_e2_u8x64 = _mm512_set1_epi8((char)0xE2); 
    __m512i const x_c3_u8x64 = _mm512_set1_epi8((char)0xC3); 
    __m512i const x_c4_u8x64 = _mm512_set1_epi8((char)0xC4); 
    __m512i const x_c5_u8x64 = _mm512_set1_epi8((char)0xC5); 
    __m512i const x_ef_u8x64 = _mm512_set1_epi8((char)0xEF); 
    __m512i const x_84_u8x64 = _mm512_set1_epi8((char)0x84); 
    __m512i const x_9f_u8x64 = _mm512_set1_epi8((char)0x9F); 
    __m512i const x_b0_u8x64 = _mm512_set1_epi8((char)0xB0); 
    __m512i const x_bf_u8x64 = _mm512_set1_epi8((char)0xBF); 
    __m512i const x_ac_u8x64 = _mm512_set1_epi8((char)0xAC); 
    __mmask64 is_e2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_e2_u8x64);
    __mmask64 is_c3_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c3_u8x64);
    __mmask64 is_c4_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c4_u8x64);
    __mmask64 is_c5_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c5_u8x64);
    __mmask64 is_ef_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ef_u8x64);
    __mmask64 is_84_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_84_u8x64);
    __mmask64 is_9f_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_9f_u8x64);
    __mmask64 is_b0_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_b0_u8x64);
    __mmask64 is_bf_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_bf_u8x64);
    __mmask64 is_ac_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ac_u8x64);
    __mmask64 is_b8_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB8));
    return ((is_e2_m64 << 1) & is_84_m64) | 
           ((is_c3_m64 << 1) & is_9f_m64) | 
           ((is_c4_m64 << 1) & is_b0_m64) | 
           ((is_c4_m64 << 1) & is_bf_m64) | 
           ((is_c5_m64 << 1) & is_bf_m64) | 
           ((is_c5_m64 << 1) & is_b8_m64) | 
           ((is_ef_m64 << 1) & is_ac_m64);  
}
/**
 *  @brief Optimized alarm function for Central Europe danger zone detection.
 *
 *  Every Central European danger character is a 2-byte (lead, second) pair with a UNIQUE second
 *  byte, so the pair test inverts naturally: one VPERMB maps each second byte to the lead that
 *  would make it dangerous (sentinel 0x00 otherwise), another VPERMB materializes the previous
 *  byte in-register, and a single VPCMPEQB matches them up. Two qualifiers make it exact:
 *  the second byte must be a continuation (its low-6-bit VPERMB index aliases ASCII and lead
 *  bytes), and the previous byte must be >= C0 (kills the sentinel matching a 0x00 byte and the
 *  zeroed lane 0). That is 5 port-5 ops instead of the naive 10 CMPEQs - and far fewer
 *  mask-register ops, which Clang would otherwise pile onto the same ports.
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_HELPER_NOINLINE __mmask64 sz_utf8_uncased_search_icelake_central_europe_alarm_efficiently_zmm_(__m512i text_u8x64,
                                                                                                  __mmask64 load_m64) {
    sz_unused_(load_m64); 
    __m512i const previous_byte_indices_u8x64 = _mm512_set_epi8( 
        62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35,
        34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
        5, 4, 3, 2, 1, 0, 0);
    __m512i const expected_leads_lut_u8x64 = _mm512_set_epi8(
        (char)0xC5, 0, 0, 0, 0, 0, 0, (char)0xC5, 0, 0, 0, 0, 0, 0, 0, (char)0xC4,
        0, 0, 0, (char)0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        (char)0xC3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (char)0xE2, 0, 0, 0, 0);
    __m512i const x_80_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_u8x64 = _mm512_set1_epi8((char)0x40);
    __m512i const x_c0_u8x64 = _mm512_set1_epi8((char)0xC0);
    __m512i previous_u8x64 = _mm512_maskz_permutexvar_epi8(0xFFFFFFFFFFFFFFFEull, previous_byte_indices_u8x64,
                                                           text_u8x64);
    __m512i expected_leads_u8x64 = _mm512_permutexvar_epi8(text_u8x64, expected_leads_lut_u8x64);
    __mmask64 danger_m64 = _mm512_cmpeq_epi8_mask(previous_u8x64, expected_leads_u8x64);
    danger_m64 &= _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, x_80_u8x64), x_40_u8x64);
    danger_m64 &= _mm512_cmpge_epu8_mask(previous_u8x64, x_c0_u8x64);
    __mmask64 is_c4_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xC4));
    __mmask64 is_bf_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xBF));
    danger_m64 |= (is_c4_m64 << 1) & is_bf_m64;
    sz_assert_(danger_m64 == sz_utf8_uncased_search_icelake_central_europe_alarm_naively_zmm_(text_u8x64) &&
               "Efficient Central European alarm must match naive implementation");
    return danger_m64;
}
/**
 *  @brief Central European uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_central_europe_k
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Safe window metadata.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_icelake_central_europe_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                       
    sz_cptr_t needle, sz_size_t needle_length,                           
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,            
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_icelake_scripted_( 
        sz_utf8_uncased_search_icelake_central_europe_fold_efficiently_zmm_,
        sz_utf8_uncased_search_icelake_central_europe_alarm_efficiently_zmm_, 
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}
#pragma endregion // Central European Uncased Find
#pragma region Cyrillic Uncased Find
/**
 *  @brief Fold a ZMM register using Cyrillic case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_cyrillic_k
 *
 *  Handles Basic Cyrillic (D0/D1) and Extended Cyrillic (D2/D3) ranges.
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_AUTO __m512i sz_utf8_uncased_search_icelake_cyrillic_fold_naively_zmm_(__m512i text_u8x64) {
    __m512i result_u8x64 = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(text_u8x64);
    __m512i const x_10_u8x64 = _mm512_set1_epi8((char)0x10); 
    __m512i const x_20_u8x64 = _mm512_set1_epi8((char)0x20); 
    __m512i const x_d0_u8x64 = _mm512_set1_epi8((char)0xD0); 
    __m512i const x_d1_u8x64 = _mm512_set1_epi8((char)0xD1); 
    __m512i const x_80_u8x64 = _mm512_set1_epi8((char)0x80);    
    __m512i const x_90_u8x64 = _mm512_set1_epi8((char)0x90);    
    __m512i const x_a0_u8x64 = _mm512_set1_epi8((char)0xA0);    
    __m512i const x_len16_u8x64 = _mm512_set1_epi8((char)0x10); 
    __mmask64 is_lead_d0_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_d0_u8x64);
    __mmask64 is_after_d0_m64 = is_lead_d0_m64 << 1;
    __mmask64 is_ext_range_m64 = _mm512_mask_cmplt_epu8_mask( 
        is_after_d0_m64, _mm512_sub_epi8(text_u8x64, x_80_u8x64), x_len16_u8x64);
    __mmask64 is_basic1_range_m64 = _mm512_mask_cmplt_epu8_mask( 
        is_after_d0_m64, _mm512_sub_epi8(text_u8x64, x_90_u8x64), x_len16_u8x64);
    __mmask64 is_basic2_range_m64 = _mm512_mask_cmplt_epu8_mask( 
        is_after_d0_m64, _mm512_sub_epi8(text_u8x64, x_a0_u8x64), x_len16_u8x64);
    __mmask64 change_lead_m64 = (is_ext_range_m64 >> 1) | (is_basic2_range_m64 >> 1);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, change_lead_m64, x_d1_u8x64);
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_ext_range_m64, result_u8x64, x_10_u8x64);    
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_basic1_range_m64, result_u8x64, x_20_u8x64); 
    result_u8x64 = _mm512_mask_sub_epi8(result_u8x64, is_basic2_range_m64, result_u8x64, x_20_u8x64); 
    return result_u8x64;
}
/**
 *  @brief Fold a ZMM register using Cyrillic case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_cyrillic_k
 *
 *  This function uses VPSHUFB optimization for Cyrillic case-folding. Why?
 *  Cyrillic has a clean high-nibble pattern: second bytes 8x/9x/Ax map directly
 *  to distinct offsets (+0x10, +0x20, -0x20). A single VPSHUFB lookup replaces
 *  3 range comparisons + 3 masked moves.
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_NOINLINE __m512i sz_utf8_uncased_search_icelake_cyrillic_fold_efficiently_zmm_(__m512i text_u8x64) {
    __m512i result_u8x64 = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(text_u8x64);
    __m512i const x_d0_u8x64 = _mm512_set1_epi8((char)0xD0);
    __m512i const x_d1_u8x64 = _mm512_set1_epi8((char)0xD1);
    __m512i const x_0f_u8x64 = _mm512_set1_epi8(0x0F);
    __mmask64 is_d0_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_d0_u8x64);
    __mmask64 is_after_d0_m64 = is_d0_m64 << 1;
    __m512i const offset_lut_u8x64 = _mm512_set_epi8(
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0);
    __m512i high_nibbles_u8x64 = _mm512_and_si512(_mm512_srli_epi16(text_u8x64, 4), x_0f_u8x64);
    __m512i offsets_u8x64 = _mm512_shuffle_epi8(offset_lut_u8x64, high_nibbles_u8x64);
    offsets_u8x64 = _mm512_maskz_mov_epi8(is_after_d0_m64, offsets_u8x64);
    result_u8x64 = _mm512_add_epi8(result_u8x64, offsets_u8x64);
    __m512i const x_08_u8x64 = _mm512_set1_epi8(0x08);
    __m512i const x_0a_u8x64 = _mm512_set1_epi8(0x0A);
    __mmask64 is_8x_m64 = _mm512_mask_cmpeq_epi8_mask(is_after_d0_m64, high_nibbles_u8x64, x_08_u8x64);
    __mmask64 is_ax_m64 = _mm512_mask_cmpeq_epi8_mask(is_after_d0_m64, high_nibbles_u8x64, x_0a_u8x64);
    __mmask64 change_lead_m64 = ((is_8x_m64 | is_ax_m64) >> 1) & is_d0_m64;
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, change_lead_m64, x_d1_u8x64);
    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_uncased_search_icelake_cyrillic_fold_naively_zmm_(text_u8x64),
                                      result_u8x64) == (__mmask64)-1 &&
               "Efficient Cyrillic fold does not match naive implementation");
    return result_u8x64;
}
/**
 *  @brief Naive alarm function for Cyrillic danger zone detection.
 *
 *  Basic Cyrillic itself never changes byte width when folded, and Extended Cyrillic needles
 *  (D2/D3 leads) are banned at needle-analysis time. The one haystack-side hazard is Cyrillic
 *  Extended-C: 'ᲀ'-'ᲈ' (U+1C80-1C88, E1 B2 80-88) fold INTO basic 2-byte Cyrillic letters
 *  ('в', 'д', 'о', 'с', 'т', 'ъ', 'ѣ'), so a 3-byte haystack character can match a 2-byte
 *  needle character and must go through the serial danger-zone scanner.
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_HELPER_AUTO __mmask64 sz_utf8_uncased_search_icelake_cyrillic_alarm_naively_zmm_(__m512i text_u8x64) {
    __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xE1));
    __mmask64 is_b2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB2));
    __mmask64 is_folding_third_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, _mm512_set1_epi8((char)0x80)),
                                                            _mm512_set1_epi8(0x09));
    return (is_e1_m64 << 1) & is_b2_m64 & (is_folding_third_m64 >> 1);
}
/**
 *  @brief Optimized alarm function for Cyrillic danger zone detection.
 *  @sa sz_utf8_uncased_search_icelake_cyrillic_alarm_naively_zmm_
 *
 *  The E1 B2 pair is absent from virtually all real Cyrillic text, so the third-byte
 *  refinement hides behind a branch and the hot path is two compares.
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @param load_m64 Present for the shared `sz_utf8_uncased_alarm_zmm_t` signature.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_HELPER_NOINLINE __mmask64 sz_utf8_uncased_search_icelake_cyrillic_alarm_efficiently_zmm_(__m512i text_u8x64,
                                                                                            __mmask64 load_m64) {
    sz_unused_(load_m64);
    __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xE1));
    __mmask64 is_b2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB2));
    __mmask64 danger_m64 = (is_e1_m64 << 1) & is_b2_m64;
    if (danger_m64) {
        __mmask64 is_folding_third_m64 = _mm512_cmplt_epu8_mask(
            _mm512_sub_epi8(text_u8x64, _mm512_set1_epi8((char)0x80)), _mm512_set1_epi8(0x09));
        danger_m64 &= is_folding_third_m64 >> 1;
    }
    sz_assert_(danger_m64 == sz_utf8_uncased_search_icelake_cyrillic_alarm_naively_zmm_(text_u8x64) &&
               "Efficient Cyrillic alarm must match naive implementation");
    return danger_m64;
}
/**
 *  @brief Cyrillic uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_cyrillic_k
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Safe window metadata.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_icelake_cyrillic_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                 
    sz_cptr_t needle, sz_size_t needle_length,                     
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,      
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_icelake_scripted_( 
        sz_utf8_uncased_search_icelake_cyrillic_fold_efficiently_zmm_,
        sz_utf8_uncased_search_icelake_cyrillic_alarm_efficiently_zmm_, 
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}
#pragma endregion // Cyrillic Uncased Find
#pragma region Armenian Uncased Find
/**
 *  @brief Fold a ZMM register using Armenian case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_armenian_k
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_AUTO __m512i sz_utf8_uncased_search_icelake_armenian_fold_naively_zmm_(__m512i text_u8x64) {
    __m512i result_u8x64 = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(text_u8x64);
    __m512i const x_d4_u8x64 = _mm512_set1_epi8((char)0xD4);
    __m512i const x_d5_u8x64 = _mm512_set1_epi8((char)0xD5);
    __m512i const x_d6_u8x64 = _mm512_set1_epi8((char)0xD6);
    __m512i const x_b1_u8x64 = _mm512_set1_epi8((char)0xB1);
    __m512i const x_96_u8x64 = _mm512_set1_epi8((char)0x96);
    __m512i const x_30_u8x64 = _mm512_set1_epi8((char)0x30); 
    __m512i const x_10_u8x64 = _mm512_set1_epi8((char)0x10); 
    __mmask64 is_d4_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_d4_u8x64);
    __mmask64 is_d5_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_d5_u8x64);
    __mmask64 is_after_d4_m64 = is_d4_m64 << 1;
    __mmask64 is_after_d5_m64 = is_d5_m64 << 1;
    __mmask64 is_d4_upper_m64 = _mm512_mask_cmpge_epu8_mask(is_after_d4_m64, result_u8x64, x_b1_u8x64);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_d4_upper_m64 >> 1, x_d5_u8x64);          
    result_u8x64 = _mm512_mask_sub_epi8(result_u8x64, is_d4_upper_m64, result_u8x64, x_10_u8x64); 
    __mmask64 is_d5_upper_subset1_m64 = _mm512_mask_cmple_epu8_mask(is_after_d5_m64, result_u8x64,
                                                                    _mm512_set1_epi8((char)0x8F));
    is_d5_upper_subset1_m64 &= _mm512_mask_cmpge_epu8_mask(is_after_d5_m64, result_u8x64, _mm512_set1_epi8((char)0x80));
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_d5_upper_subset1_m64, result_u8x64, x_30_u8x64); 
    __mmask64 is_d5_upper_subset2_m64 = _mm512_mask_cmpge_epu8_mask(is_after_d5_m64, result_u8x64,
                                                                    _mm512_set1_epi8((char)0x90));
    is_d5_upper_subset2_m64 &= _mm512_mask_cmple_epu8_mask(is_after_d5_m64, result_u8x64, x_96_u8x64);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_d5_upper_subset2_m64 >> 1, x_d6_u8x64); 
    result_u8x64 = _mm512_mask_sub_epi8(result_u8x64, is_d5_upper_subset2_m64, result_u8x64,
                                        x_10_u8x64); 
    return result_u8x64;
}
/**
 *  @brief Fold a ZMM register using Armenian case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_armenian_k
 *
 *  This function uses VPTERNLOG optimization for Armenian case-folding. Why?
 *  Armenian has 3 distinct lead bytes (D4/D5/D6) with different sub-ranges.
 *  VPTERNLOG allows building 3 offset vectors in parallel (no dependencies),
 *  then combining them with a single OR operation (imm8=0xFE: A | B | C).
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_NOINLINE __m512i sz_utf8_uncased_search_icelake_armenian_fold_efficiently_zmm_(__m512i text_u8x64) {
    __m512i const a_upper_u8x64 = _mm512_set1_epi8('A');
    __m512i const range26_u8x64 = _mm512_set1_epi8(26);
    __m512i const x_20_u8x64 = _mm512_set1_epi8(0x20);
    __mmask64 is_upper_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, a_upper_u8x64), range26_u8x64);
    __m512i result_u8x64 = _mm512_mask_add_epi8(text_u8x64, is_upper_m64, text_u8x64, x_20_u8x64);
    __m512i const x_d4_u8x64 = _mm512_set1_epi8((char)0xD4);
    __m512i const x_d5_u8x64 = _mm512_set1_epi8((char)0xD5);
    __m512i const x_d6_u8x64 = _mm512_set1_epi8((char)0xD6);
    __m512i const x_b1_u8x64 = _mm512_set1_epi8((char)0xB1);
    __m512i const x_80_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const x_90_u8x64 = _mm512_set1_epi8((char)0x90);
    __m512i const x_0f_u8x64 = _mm512_set1_epi8((char)0x0F); 
    __m512i const x_10_u8x64 = _mm512_set1_epi8((char)0x10); 
    __m512i const x_07_u8x64 = _mm512_set1_epi8((char)0x07); 
    __m512i const x_30_u8x64 = _mm512_set1_epi8((char)0x30); 
    __m512i const x_f0_u8x64 = _mm512_set1_epi8((char)0xF0); 
    __mmask64 is_d4_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_d4_u8x64);
    __mmask64 is_d5_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_d5_u8x64);
    __mmask64 is_after_d4_m64 = is_d4_m64 << 1;
    __mmask64 is_after_d5_m64 = is_d5_m64 << 1;
    __mmask64 is_d4_upper_m64 = _mm512_mask_cmplt_epu8_mask(is_after_d4_m64, _mm512_sub_epi8(result_u8x64, x_b1_u8x64),
                                                            x_0f_u8x64);
    __mmask64 is_d5_subset1_m64 = _mm512_mask_cmplt_epu8_mask(is_after_d5_m64,
                                                              _mm512_sub_epi8(result_u8x64, x_80_u8x64), x_10_u8x64);
    __mmask64 is_d5_subset2_m64 = _mm512_mask_cmplt_epu8_mask(is_after_d5_m64,
                                                              _mm512_sub_epi8(result_u8x64, x_90_u8x64), x_07_u8x64);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_d4_upper_m64 >> 1, x_d5_u8x64);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_d5_subset2_m64 >> 1, x_d6_u8x64);
    __m512i off_d4_u8x64 = _mm512_maskz_mov_epi8(is_d4_upper_m64, x_f0_u8x64);      
    __m512i off_d5_s1_u8x64 = _mm512_maskz_mov_epi8(is_d5_subset1_m64, x_30_u8x64); 
    __m512i off_d5_s2_u8x64 = _mm512_maskz_mov_epi8(is_d5_subset2_m64, x_f0_u8x64); 
    __m512i offset_u8x64 = _mm512_ternarylogic_epi64(off_d4_u8x64, off_d5_s1_u8x64, off_d5_s2_u8x64, 0xFE);
    result_u8x64 = _mm512_add_epi8(result_u8x64, offset_u8x64);
    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_uncased_search_icelake_armenian_fold_naively_zmm_(text_u8x64),
                                      result_u8x64) == (__mmask64)-1 &&
               "Efficient Armenian fold must match naive implementation");
    return result_u8x64;
}
/**
 *  @brief Naive alarm function for Armenian danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - D6 87: 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82) (2 bytes → 4 bytes)
 *  - EF AC 93-97: Armenian ligatures 'ﬓ'..'ﬗ' (U+FB13-U+FB17, EF AC 93-97) → 2 codepoints (4 bytes)
 *
 *  Uses 4 CMPEQ operations (2 lead + 2 second byte checks).
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_HELPER_AUTO __mmask64 sz_utf8_uncased_search_icelake_armenian_alarm_naively_zmm_(__m512i text_u8x64) {
    __m512i const x_d6_u8x64 = _mm512_set1_epi8((char)0xD6); 
    __m512i const x_ef_u8x64 = _mm512_set1_epi8((char)0xEF); 
    __m512i const x_87_u8x64 = _mm512_set1_epi8((char)0x87); 
    __m512i const x_ac_u8x64 = _mm512_set1_epi8((char)0xAC); 
    __mmask64 is_d6_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_d6_u8x64);
    __mmask64 is_ef_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ef_u8x64);
    __mmask64 is_87_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_87_u8x64);
    __mmask64 is_ac_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ac_u8x64);
    __mmask64 ech_yiwn_m64 = (is_d6_m64 << 1) & is_87_m64; 
    __mmask64 ef_ac_m64 = (is_ef_m64 << 1) & is_ac_m64;    
    return ech_yiwn_m64 | ef_ac_m64;
}
/**
 *  @brief Optimized alarm function for Armenian danger zone detection.
 *
 *  Currently delegates to the naive implementation.
 *  Armenian has only 4 CMPEQ ops, so optimization overhead may not be worthwhile.
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_HELPER_NOINLINE __mmask64 sz_utf8_uncased_search_icelake_armenian_alarm_efficiently_zmm_(__m512i text_u8x64,
                                                                                            __mmask64 load_m64) {
    sz_unused_(load_m64); 
    return sz_utf8_uncased_search_icelake_armenian_alarm_naively_zmm_(text_u8x64);
}
/**
 *  @brief Armenian uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_armenian_k
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_icelake_armenian_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                 
    sz_cptr_t needle, sz_size_t needle_length,                     
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,      
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_icelake_scripted_( 
        sz_utf8_uncased_search_icelake_armenian_fold_efficiently_zmm_,
        sz_utf8_uncased_search_icelake_armenian_alarm_efficiently_zmm_, 
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}
#pragma endregion // Armenian Uncased Find
#pragma region Greek Uncased Find
/**
 *  @brief Fold a ZMM register using Greek case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_greek_k
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_AUTO __m512i sz_utf8_uncased_search_icelake_greek_fold_naively_zmm_(__m512i text_u8x64) {
    __m512i result_u8x64 = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(text_u8x64);
    __m512i const x_ce_u8x64 = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_u8x64 = _mm512_set1_epi8((char)0xCF);
    __m512i const x_c2_u8x64 = _mm512_set1_epi8((char)0xC2);
    __m512i const x_b5_u8x64 = _mm512_set1_epi8((char)0xB5);
    __m512i const x_bc_u8x64 = _mm512_set1_epi8((char)0xBC);
    __m512i const x_86_u8x64 = _mm512_set1_epi8((char)0x86);
    __m512i const x_8f_u8x64 = _mm512_set1_epi8((char)0x8F);
    __m512i const x_91_u8x64 = _mm512_set1_epi8((char)0x91);
    __m512i const x_9f_u8x64 = _mm512_set1_epi8((char)0x9F);
    __m512i const x_a0_u8x64 = _mm512_set1_epi8((char)0xA0);
    __m512i const x_a9_u8x64 = _mm512_set1_epi8((char)0xA9);
    __m512i const x_aa_u8x64 = _mm512_set1_epi8((char)0xAA);
    __m512i const x_ab_u8x64 = _mm512_set1_epi8((char)0xAB);
    __m512i const x_82_u8x64 = _mm512_set1_epi8((char)0x82);
    __m512i const x_83_u8x64 = _mm512_set1_epi8((char)0x83);
    __m512i const x_20_u8x64 = _mm512_set1_epi8((char)0x20); 
    __m512i const x_e0_u8x64 = _mm512_set1_epi8((char)0xE0); 
    __m512i const x_01_u8x64 = _mm512_set1_epi8((char)0x01); 
    __mmask64 is_ce_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_ce_u8x64);
    __mmask64 is_cf_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_cf_u8x64);
    __mmask64 is_after_ce_m64 = is_ce_m64 << 1;
    __mmask64 is_after_cf_m64 = is_cf_m64 << 1;
    __mmask64 is_basic1_m64 = _mm512_mask_cmpge_epu8_mask(is_after_ce_m64, result_u8x64, x_91_u8x64);
    is_basic1_m64 &= _mm512_mask_cmple_epu8_mask(is_after_ce_m64, result_u8x64, x_9f_u8x64);
    __mmask64 is_basic2_m64 = _mm512_mask_cmpge_epu8_mask(is_after_ce_m64, result_u8x64, x_a0_u8x64);
    is_basic2_m64 &= _mm512_mask_cmple_epu8_mask(is_after_ce_m64, result_u8x64, x_a9_u8x64);
    __mmask64 is_accented_m64 = _mm512_mask_cmpge_epu8_mask(is_after_ce_m64, result_u8x64, x_86_u8x64);
    is_accented_m64 &= _mm512_mask_cmple_epu8_mask(is_after_ce_m64, result_u8x64, x_8f_u8x64);
    __mmask64 is_86_m64 = is_accented_m64 & _mm512_cmpeq_epi8_mask(result_u8x64, x_86_u8x64);
    __mmask64 is_88_8a_m64 = is_accented_m64 & _mm512_cmpge_epu8_mask(result_u8x64, _mm512_set1_epi8((char)0x88));
    is_88_8a_m64 &= _mm512_cmple_epu8_mask(result_u8x64, _mm512_set1_epi8((char)0x8A));
    __mmask64 is_8c_m64 = is_accented_m64 & _mm512_cmpeq_epi8_mask(result_u8x64, _mm512_set1_epi8((char)0x8C));
    __mmask64 is_8e_8f_m64 = is_accented_m64 & _mm512_cmpge_epu8_mask(result_u8x64, _mm512_set1_epi8((char)0x8E));
    __mmask64 is_dialytika_m64 = _mm512_mask_cmpge_epu8_mask(is_after_ce_m64, result_u8x64, x_aa_u8x64);
    is_dialytika_m64 &= _mm512_mask_cmple_epu8_mask(is_after_ce_m64, result_u8x64, x_ab_u8x64);
    __mmask64 is_final_sigma_m64 = _mm512_mask_cmpeq_epi8_mask(is_after_cf_m64, result_u8x64, x_82_u8x64);
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_basic1_m64, result_u8x64, x_20_u8x64);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_basic2_m64 >> 1, x_cf_u8x64); 
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_basic2_m64, result_u8x64,
                                        x_e0_u8x64); 
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_86_m64, result_u8x64, _mm512_set1_epi8(0x26));
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_88_8a_m64, result_u8x64, _mm512_set1_epi8(0x25));
    __mmask64 change_lead_m64 = (is_8c_m64 >> 1) | (is_8e_8f_m64 >> 1);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, change_lead_m64, x_cf_u8x64);
    result_u8x64 = _mm512_mask_sub_epi8(result_u8x64, is_8e_8f_m64, result_u8x64, x_01_u8x64);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_dialytika_m64 >> 1, x_cf_u8x64);
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_dialytika_m64, result_u8x64, x_e0_u8x64); 
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_final_sigma_m64, x_83_u8x64);
    __mmask64 is_c2_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_c2_u8x64);
    __mmask64 is_micro_second_m64 = (is_c2_m64 << 1) & _mm512_cmpeq_epi8_mask(result_u8x64, x_b5_u8x64);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_micro_second_m64 >> 1, x_ce_u8x64); 
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_micro_second_m64, x_bc_u8x64);      
    return result_u8x64;
}
/**
 *  @brief Fold a ZMM register using Greek case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_greek_k
 *
 *  This function uses VPERMB lookup tables for Greek case-folding. Why?
 *  Every Greek transform is fully determined by the second byte of the CE-led sequence:
 *  five distinct offsets (+0x20, -0x20, +0x26, +0x25, -1) over six sub-ranges, plus a
 *  CE->CF lead promotion for four of them. A continuation byte 0x80-0xBF maps to a unique
 *  VPERMB index through its low 6 bits, so one lookup yields the second-byte delta and one
 *  more yields the lead-promotion flag - replacing ~10 range comparisons and the offset
 *  merging. This also keeps the live-constant count low: GCC re-materializes broadcast
 *  constants inside loops once their number outgrows the register budget.
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_NOINLINE __m512i sz_utf8_uncased_search_icelake_greek_fold_efficiently_zmm_(__m512i text_u8x64) {
    __m512i const a_upper_u8x64 = _mm512_set1_epi8('A');
    __m512i const range26_u8x64 = _mm512_set1_epi8(26);
    __m512i const x_20_u8x64 = _mm512_set1_epi8(0x20);
    __mmask64 is_upper_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, a_upper_u8x64), range26_u8x64);
    __m512i result_u8x64 = _mm512_mask_add_epi8(text_u8x64, is_upper_m64, text_u8x64, x_20_u8x64);
    __m512i const x_ce_u8x64 = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_u8x64 = _mm512_set1_epi8((char)0xCF);
    __m512i const x_c2_u8x64 = _mm512_set1_epi8((char)0xC2);
    __m512i const x_80_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_u8x64 = _mm512_set1_epi8((char)0x40);
    __m512i const x_01_u8x64 = _mm512_set1_epi8(0x01);
    __m512i const second_byte_deltas_lut_u8x64 = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0,
        (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0,
        (char)0xFF, (char)0xFF, 0, 0, 0, 0x25, 0x25, 0x25, 0, 0x26, 0, 0, 0, 0, 0, 0);
    __m512i const lead_promotions_lut_u8x64 = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __mmask64 is_ce_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ce_u8x64);
    __mmask64 is_cf_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_cf_u8x64);
    __mmask64 is_continuation_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, x_80_u8x64), x_40_u8x64);
    __mmask64 fold_positions_m64 = (is_ce_m64 << 1) & is_continuation_m64;
    __m512i second_byte_deltas_u8x64 = _mm512_permutexvar_epi8(text_u8x64, second_byte_deltas_lut_u8x64);
    __m512i lead_promotions_u8x64 = _mm512_permutexvar_epi8(text_u8x64, lead_promotions_lut_u8x64);
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, fold_positions_m64, result_u8x64, second_byte_deltas_u8x64);
    __mmask64 promote_lead_m64 = _mm512_mask_test_epi8_mask(fold_positions_m64, lead_promotions_u8x64,
                                                            lead_promotions_u8x64);
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, promote_lead_m64 >> 1, result_u8x64, x_01_u8x64);
    __mmask64 is_final_sigma_m64 = _mm512_mask_cmpeq_epi8_mask(is_cf_m64 << 1, text_u8x64,
                                                               _mm512_set1_epi8((char)0x82));
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_final_sigma_m64, result_u8x64, x_01_u8x64);
    __mmask64 is_c2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c2_u8x64);
    __mmask64 is_micro_second_m64 = (is_c2_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB5));
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_micro_second_m64 >> 1, x_ce_u8x64);
    result_u8x64 = _mm512_mask_mov_epi8(result_u8x64, is_micro_second_m64, _mm512_set1_epi8((char)0xBC));
    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_uncased_search_icelake_greek_fold_naively_zmm_(text_u8x64),
                                      result_u8x64) == (__mmask64)-1 &&
               "Efficient Greek fold must match naive implementation");
    return result_u8x64;
}
/**
 *  @brief Detect danger zone positions in a ZMM register for Greek text.
 *
 *  Danger characters are those that expand during case folding or have alternative representations:
 *  - CE 90: 'ΐ' (U+0390) expands to 3 codepoints
 *  - CE B0: 'ΰ' (U+03B0) expands to 3 codepoints
 *  - CF 90,91,95,96: Greek symbols 'ϐ', 'ϑ', 'ϕ', 'ϖ'
 *  - CF B0,B1,B5: Greek symbols 'ϰ', 'ϱ', 'ϵ'
 *  - E2 84: Ohm sign 'Ω' prefix
 *  - E1: Polytonic Greek / Extensions
 *  - CD: Combining Diacritical Marks
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_HELPER_AUTO __mmask64 sz_utf8_uncased_search_icelake_greek_alarm_naively_zmm_(__m512i text_u8x64) {
    __m512i const x_ce_u8x64 = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_u8x64 = _mm512_set1_epi8((char)0xCF);
    __m512i const x_e2_u8x64 = _mm512_set1_epi8((char)0xE2);
    __m512i const x_e1_u8x64 = _mm512_set1_epi8((char)0xE1);
    __m512i const x_cd_u8x64 = _mm512_set1_epi8((char)0xCD);
    __m512i const x_90_u8x64 = _mm512_set1_epi8((char)0x90);
    __m512i const x_b0_u8x64 = _mm512_set1_epi8((char)0xB0);
    __m512i const x_91_u8x64 = _mm512_set1_epi8((char)0x91);
    __m512i const x_95_u8x64 = _mm512_set1_epi8((char)0x95);
    __m512i const x_96_u8x64 = _mm512_set1_epi8((char)0x96);
    __m512i const x_b1_u8x64 = _mm512_set1_epi8((char)0xB1);
    __m512i const x_b5_u8x64 = _mm512_set1_epi8((char)0xB5);
    __m512i const x_84_u8x64 = _mm512_set1_epi8((char)0x84);
    __mmask64 is_ce_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ce_u8x64);
    __mmask64 is_cf_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_cf_u8x64);
    __mmask64 is_e2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_e2_u8x64);
    __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_e1_u8x64);
    __mmask64 is_cd_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_cd_u8x64);
    __mmask64 is_90_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_90_u8x64);
    __mmask64 is_b0_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_b0_u8x64);
    __mmask64 is_9x_m64 = is_90_m64 | _mm512_cmpeq_epi8_mask(text_u8x64, x_91_u8x64) | 
                          _mm512_cmpeq_epi8_mask(text_u8x64, x_95_u8x64) |
                          _mm512_cmpeq_epi8_mask(text_u8x64, x_96_u8x64);
    __mmask64 is_bx_m64 = is_b0_m64 | _mm512_cmpeq_epi8_mask(text_u8x64, x_b1_u8x64) |
                          _mm512_cmpeq_epi8_mask(text_u8x64, x_b5_u8x64) |
                          _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB4)); 
    __mmask64 is_84_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_84_u8x64);
    __mmask64 ce_danger_m64 = (is_ce_m64 << 1) & (is_90_m64 | is_b0_m64);
    __mmask64 cf_danger_m64 = (is_cf_m64 << 1) & (is_9x_m64 | is_bx_m64);
    __mmask64 e2_danger_m64 = (is_e2_m64 << 1) & is_84_m64;
    return ce_danger_m64 | cf_danger_m64 | e2_danger_m64 | is_e1_m64 | is_cd_m64;
}
/**
 *  @brief Optimized danger zone detection for Greek text using bit-folded second-byte compares.
 *  @sa sz_utf8_uncased_search_icelake_greek_alarm_naively_zmm_
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_HELPER_NOINLINE __mmask64 sz_utf8_uncased_search_icelake_greek_alarm_efficiently_zmm_(__m512i text_u8x64,
                                                                                         __mmask64 load_m64) {
    sz_unused_(load_m64); 
    __m512i const x_df_u8x64 = _mm512_set1_epi8((char)0xDF);
    __m512i const x_fe_u8x64 = _mm512_set1_epi8((char)0xFE);
    __m512i const x_90_u8x64 = _mm512_set1_epi8((char)0x90);
    __m512i const x_91_u8x64 = _mm512_set1_epi8((char)0x91);
    __m512i const x_95_u8x64 = _mm512_set1_epi8((char)0x95);
    __m512i const x_96_u8x64 = _mm512_set1_epi8((char)0x96);
    __m512i const x_ce_u8x64 = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_u8x64 = _mm512_set1_epi8((char)0xCF);
    __m512i const x_cd_u8x64 = _mm512_set1_epi8((char)0xCD);
    __m512i const x_e1_u8x64 = _mm512_set1_epi8((char)0xE1);
    __m512i const x_e2_u8x64 = _mm512_set1_epi8((char)0xE2);
    __m512i const x_84_u8x64 = _mm512_set1_epi8((char)0x84);
    __m512i folded_u8x64 = _mm512_and_si512(text_u8x64, x_df_u8x64);
    __mmask64 is_90_or_b0_m64 = _mm512_cmpeq_epi8_mask(folded_u8x64, x_90_u8x64);
    __mmask64 is_91_or_b1_m64 = _mm512_cmpeq_epi8_mask(folded_u8x64, x_91_u8x64);
    __mmask64 is_95_or_b5_m64 = _mm512_cmpeq_epi8_mask(folded_u8x64, x_95_u8x64);
    __mmask64 is_96_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_96_u8x64);
    __mmask64 is_b4_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB4));
    __mmask64 is_84_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_84_u8x64);
    __mmask64 is_ce_or_cf_m64 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(text_u8x64, x_fe_u8x64), x_ce_u8x64);
    __mmask64 is_cf_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_cf_u8x64);
    __mmask64 is_cd_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_cd_u8x64);
    __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_e1_u8x64);
    __mmask64 is_e2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_e2_u8x64);
    __mmask64 danger_m64 = ((is_ce_or_cf_m64 << 1) & is_90_or_b0_m64) |
                           ((is_cf_m64 << 1) & (is_91_or_b1_m64 | is_95_or_b5_m64 | is_96_m64 | is_b4_m64)) |
                           ((is_e2_m64 << 1) & is_84_m64) | is_cd_m64 | is_e1_m64;
    sz_assert_(danger_m64 == sz_utf8_uncased_search_icelake_greek_alarm_naively_zmm_(text_u8x64) &&
               "Efficient Greek alarm must match naive implementation");
    return danger_m64;
}
/**
 *  @brief Greek uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_greek_k
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_icelake_greek_( 
    sz_cptr_t haystack, sz_size_t haystack_length,              
    sz_cptr_t needle, sz_size_t needle_length,                  
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,   
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_icelake_scripted_( 
        sz_utf8_uncased_search_icelake_greek_fold_efficiently_zmm_,
        sz_utf8_uncased_search_icelake_greek_alarm_efficiently_zmm_, 
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}
#pragma endregion // Greek Uncased Find
#pragma region Vietnamese Uncased Find
/**
 *  @brief Fold a ZMM register using Vietnamese case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_vietnamese_k
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_AUTO __m512i sz_utf8_uncased_search_icelake_vietnamese_fold_naively_zmm_(__m512i text_u8x64) {
    __m512i result_u8x64 = sz_utf8_uncased_search_icelake_ascii_fold_zmm_(text_u8x64);
    __m512i const x_c3_u8x64 = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c4_u8x64 = _mm512_set1_epi8((char)0xC4);
    __m512i const x_c5_u8x64 = _mm512_set1_epi8((char)0xC5);
    __m512i const x_c6_u8x64 = _mm512_set1_epi8((char)0xC6);
    __m512i const x_e1_u8x64 = _mm512_set1_epi8((char)0xE1);
    __m512i const x_b8_u8x64 = _mm512_set1_epi8((char)0xB8);
    __m512i const x_bb_u8x64 = _mm512_set1_epi8((char)0xBB);
    __m512i const x_ba_u8x64 = _mm512_set1_epi8((char)0xBA);
    __m512i const x_96_u8x64 = _mm512_set1_epi8((char)0x96);
    __m512i const x_9f_u8x64 = _mm512_set1_epi8((char)0x9F);
    __m512i const x_01_u8x64 = _mm512_set1_epi8((char)0x01);
    __m512i const x_20_u8x64 = _mm512_set1_epi8((char)0x20);
    __mmask64 is_c3_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_c3_u8x64);
    __mmask64 is_c4_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_c4_u8x64);
    __mmask64 is_c5_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_c5_u8x64);
    __mmask64 is_c6_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_c6_u8x64);
    __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(result_u8x64, x_e1_u8x64);
    __mmask64 is_after_c3_m64 = is_c3_m64 << 1;
    __mmask64 is_after_c4_m64 = is_c4_m64 << 1;
    __mmask64 is_after_c5_m64 = is_c5_m64 << 1;
    __mmask64 is_after_c6_m64 = is_c6_m64 << 1;
    __mmask64 is_c3_target_m64 = _mm512_mask_cmple_epu8_mask(is_after_c3_m64, result_u8x64,
                                                             _mm512_set1_epi8((char)0x9E)); 
    is_c3_target_m64 &= ~_mm512_cmpeq_epi8_mask(result_u8x64, _mm512_set1_epi8((char)0x97)); 
    is_c3_target_m64 &= _mm512_mask_cmpge_epu8_mask(is_after_c3_m64, result_u8x64, _mm512_set1_epi8((char)0x80));
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_c3_target_m64, result_u8x64, x_20_u8x64);
    __mmask64 is_c4_c5_target_m64 = is_after_c4_m64 | is_after_c5_m64;
    __mmask64 is_even_m64 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(result_u8x64, x_01_u8x64), _mm512_setzero_si512());
    __mmask64 is_odd_m64 = ~is_even_m64;
    __mmask64 is_c4_inverted_range_m64 = is_after_c4_m64 &
                                         _mm512_cmpge_epu8_mask(result_u8x64, _mm512_set1_epi8((char)0xB9)) &
                                         _mm512_cmple_epu8_mask(result_u8x64, _mm512_set1_epi8((char)0xBE));
    __mmask64 is_c5_inverted_range_m64 = is_after_c5_m64 &
                                         _mm512_cmple_epu8_mask(result_u8x64, _mm512_set1_epi8((char)0x88));
    __mmask64 is_inverted_range_m64 = is_c4_inverted_range_m64 | is_c5_inverted_range_m64;
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_c4_c5_target_m64 & is_even_m64 & ~is_inverted_range_m64,
                                        result_u8x64, x_01_u8x64);
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_inverted_range_m64 & is_odd_m64, result_u8x64, x_01_u8x64);
    __mmask64 is_c6_A0_m64 = is_after_c6_m64 & _mm512_cmpeq_epi8_mask(result_u8x64, _mm512_set1_epi8((char)0xA0));
    __mmask64 is_c6_AF_m64 = is_after_c6_m64 & _mm512_cmpeq_epi8_mask(result_u8x64, _mm512_set1_epi8((char)0xAF));
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_c6_A0_m64 | is_c6_AF_m64, result_u8x64, x_01_u8x64);
    __mmask64 is_e1_second_m64 = is_e1_m64 << 1;
    __mmask64 is_valid_second_m64 = _mm512_mask_cmpge_epu8_mask(is_e1_second_m64, result_u8x64, x_b8_u8x64);
    is_valid_second_m64 &= _mm512_mask_cmple_epu8_mask(is_e1_second_m64, result_u8x64, x_bb_u8x64);
    __mmask64 is_e1_third_m64 = is_valid_second_m64 << 1;
    __mmask64 is_ba_second_m64 = is_e1_second_m64 & _mm512_cmpeq_epi8_mask(result_u8x64, x_ba_u8x64);
    __mmask64 is_excluded_third_m64 = (is_ba_second_m64 << 1) &
                                      _mm512_mask_cmpge_epu8_mask(is_e1_third_m64, result_u8x64, x_96_u8x64) &
                                      _mm512_mask_cmple_epu8_mask(is_e1_third_m64, result_u8x64, x_9f_u8x64);
    __mmask64 is_e1_target_m64 = is_e1_third_m64 & is_even_m64 & ~is_excluded_third_m64;
    result_u8x64 = _mm512_mask_add_epi8(result_u8x64, is_e1_target_m64, result_u8x64, x_01_u8x64);
    return result_u8x64;
}
/**
 *  @brief Fold a ZMM register using Vietnamese case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_vietnamese_k
 *
 *  @param text_u8x64 The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_HELPER_NOINLINE __m512i sz_utf8_uncased_search_icelake_vietnamese_fold_efficiently_zmm_(__m512i text_u8x64) {
    __m512i const a_upper_u8x64 = _mm512_set1_epi8('A');
    __m512i const range26_u8x64 = _mm512_set1_epi8(26);
    __m512i const x_20_u8x64 = _mm512_set1_epi8(0x20);
    __m512i const x_01_u8x64 = _mm512_set1_epi8(0x01);
    __mmask64 is_upper_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, a_upper_u8x64), range26_u8x64);
    __m512i result_u8x64 = _mm512_mask_add_epi8(text_u8x64, is_upper_m64, text_u8x64, x_20_u8x64);
    __m512i const fold_flags_lut_u8x64 = _mm512_set_epi8(
        0x00, 0x02, 0x01, 0x02, 0x05, 0x0E, 0x05, 0x07, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03,
        0x40, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x43,
        0x10, 0x33, 0x30, 0x33, 0x30, 0x33, 0x30, 0x33, 0x10, 0x33, 0x20, 0x23, 0x20, 0x23, 0x20, 0x23,
        0x20, 0x23, 0x20, 0x23, 0x20, 0x23, 0x20, 0x21, 0x22, 0x21, 0x22, 0x21, 0x22, 0x21, 0x22, 0x21);
    __m512i const x_c3_u8x64 = _mm512_set1_epi8((char)0xC3);
    __m512i const x_e1_u8x64 = _mm512_set1_epi8((char)0xE1);
    __m512i const x_80_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_u8x64 = _mm512_set1_epi8((char)0x40);
    __m512i const x_02_u8x64 = _mm512_set1_epi8(0x02);
    __m512i const x_04_u8x64 = _mm512_set1_epi8(0x04);
    __m512i const x_08_u8x64 = _mm512_set1_epi8(0x08);
    __m512i const x_10_u8x64 = _mm512_set1_epi8(0x10);
    __mmask64 is_latin_lead_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, x_c3_u8x64), x_04_u8x64);
    __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_e1_u8x64);
    __mmask64 is_continuation_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, x_80_u8x64), x_40_u8x64);
    __mmask64 is_even_m64 = _mm512_testn_epi8_mask(text_u8x64, x_01_u8x64);
    __mmask64 has_bit1_m64 = _mm512_test_epi8_mask(text_u8x64, x_02_u8x64);
    __mmask64 is_after_c3_m64 = (is_latin_lead_m64 & ~is_even_m64 & has_bit1_m64) << 1;
    __mmask64 is_after_c4_m64 = (is_latin_lead_m64 & is_even_m64 & ~has_bit1_m64) << 1;
    __mmask64 is_after_c5_m64 = (is_latin_lead_m64 & ~is_even_m64 & ~has_bit1_m64) << 1;
    __mmask64 is_after_c6_m64 = (is_latin_lead_m64 & is_even_m64 & has_bit1_m64) << 1;
    __mmask64 is_e1_second_m64 = (is_e1_m64 << 1) & is_continuation_m64;
    __m512i fold_flags_u8x64 = _mm512_permutexvar_epi8(text_u8x64, fold_flags_lut_u8x64);
    __mmask64 fold_c3_m64 = _mm512_mask_test_epi8_mask(is_after_c3_m64 & is_continuation_m64, 
                                                       fold_flags_u8x64, x_20_u8x64);
    __mmask64 fold_c4_m64 = _mm512_mask_test_epi8_mask(is_after_c4_m64 & is_continuation_m64, 
                                                       fold_flags_u8x64, x_01_u8x64);
    __mmask64 fold_c5_m64 = _mm512_mask_test_epi8_mask(is_after_c5_m64 & is_continuation_m64, 
                                                       fold_flags_u8x64, x_02_u8x64);
    __mmask64 is_ascii_m64 = _mm512_testn_epi8_mask(text_u8x64, x_80_u8x64);
    __mmask64 fold_malformed_c4_m64 = is_after_c4_m64 & ~is_continuation_m64 & is_even_m64;
    __mmask64 fold_malformed_c5_m64 = is_after_c5_m64 & ~is_continuation_m64 & (is_ascii_m64 ^ is_even_m64);
    __mmask64 fold_c6_m64 = _mm512_mask_test_epi8_mask(is_after_c6_m64 & is_continuation_m64, 
                                                       fold_flags_u8x64, x_40_u8x64);
    __mmask64 is_valid_second_m64 = _mm512_mask_test_epi8_mask(is_e1_second_m64, fold_flags_u8x64, x_04_u8x64);
    __mmask64 is_ba_second_m64 = _mm512_mask_test_epi8_mask(is_e1_second_m64, fold_flags_u8x64, x_08_u8x64);
    __mmask64 is_excluded_third_m64 = _mm512_mask_test_epi8_mask((is_ba_second_m64 << 1) & is_continuation_m64,
                                                                 fold_flags_u8x64, x_10_u8x64);
    __mmask64 fold_e1_m64 = (is_valid_second_m64 << 1) & is_even_m64 & ~is_excluded_third_m64;
    __mmask64 fold_plus_one_m64 = fold_c4_m64 | fold_c5_m64 | fold_malformed_c4_m64 | fold_malformed_c5_m64 |
                                  fold_c6_m64 | fold_e1_m64;
    __m512i offset_u8x64 = _mm512_maskz_mov_epi8(fold_c3_m64, x_20_u8x64);
    offset_u8x64 = _mm512_mask_mov_epi8(offset_u8x64, fold_plus_one_m64, x_01_u8x64);
    result_u8x64 = _mm512_add_epi8(result_u8x64, offset_u8x64);
    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_uncased_search_icelake_vietnamese_fold_naively_zmm_(text_u8x64),
                                      result_u8x64) == (__mmask64)-1 &&
               "Efficient Vietnamese fold must match naive implementation");
    return result_u8x64;
}
/**
 *  @brief Naive alarm function for Vietnamese danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E1 BA 96-9F: Latin Extensions like 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
 *  - C3 9F: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
 *  - C5 BF: 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
 *  - EF AC 80-86: Ligatures:
 *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
 *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
 *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
 *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
 *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
 *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
 *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
 *  - E2 84 AA: 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
 *
 *  Uses ~11 CMPEQ + 2 range operations.
 *  Note: Returns danger positions shifted to the START of multi-byte sequences.
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @param load_m64 Mask of valid bytes in the ZMM register.
 *  @return Bitmask of positions where danger characters are detected (at sequence start).
 */
SZ_HELPER_AUTO __mmask64 sz_utf8_uncased_search_icelake_vietnamese_alarm_naively_zmm_(__m512i text_u8x64,
                                                                                      __mmask64 load_m64) {
    __m512i const x_e1_u8x64 = _mm512_set1_epi8((char)0xE1);
    __m512i const x_c3_u8x64 = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c5_u8x64 = _mm512_set1_epi8((char)0xC5);
    __m512i const x_ef_u8x64 = _mm512_set1_epi8((char)0xEF);
    __m512i const x_e2_u8x64 = _mm512_set1_epi8((char)0xE2);
    __m512i const x_ba_u8x64 = _mm512_set1_epi8((char)0xBA);
    __m512i const x_9f_u8x64 = _mm512_set1_epi8((char)0x9F);
    __m512i const x_bf_u8x64 = _mm512_set1_epi8((char)0xBF);
    __m512i const x_ac_u8x64 = _mm512_set1_epi8((char)0xAC);
    __m512i const x_84_u8x64 = _mm512_set1_epi8((char)0x84);
    __m512i const x_96_u8x64 = _mm512_set1_epi8((char)0x96);
    __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_e1_u8x64);
    __mmask64 is_c3_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c3_u8x64);
    __mmask64 is_c5_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c5_u8x64);
    __mmask64 is_ef_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ef_u8x64);
    __mmask64 is_e2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_e2_u8x64);
    if (!(is_e1_m64 | is_c3_m64 | is_c5_m64 | is_ef_m64 | is_e2_m64)) return 0;
    __mmask64 ba_second_m64 = (is_e1_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, x_ba_u8x64);
    __mmask64 bad_third_m64 = (ba_second_m64 << 1) & _mm512_mask_cmpge_epu8_mask(load_m64, text_u8x64, x_96_u8x64) &
                              _mm512_mask_cmple_epu8_mask(load_m64, text_u8x64, x_9f_u8x64);
    __mmask64 sharp_s_m64 = (is_c3_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, x_9f_u8x64);
    __mmask64 long_s_m64 = (is_c5_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, x_bf_u8x64);
    __mmask64 ligature_m64 = (is_ef_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, x_ac_u8x64);
    __mmask64 kelvin_m64 = (is_e2_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, x_84_u8x64);
    return (bad_third_m64 >> 2) | (sharp_s_m64 >> 1) | (long_s_m64 >> 1) | (ligature_m64 >> 1) | (kelvin_m64 >> 1);
}
/**
 *  @brief Optimized alarm function for Vietnamese danger zone detection.
 *
 *  Reduces port 5 pressure from ~12 to ~11 operations by:
 *  - E1/E2 consecutive: 2 CMPEQ → 1 CMPLT + 1 VPTESTNMB (p0)
 *
 *  Port summary: 11 p5 ops + 1 p0 op (vs 12 p5 originally)
 *
 *  @param text_u8x64 The haystack ZMM register.
 *  @param load_m64 Mask of valid bytes in the ZMM register.
 *  @return Bitmask of positions where danger characters are detected (at sequence start).
 */
SZ_HELPER_NOINLINE __mmask64 sz_utf8_uncased_search_icelake_vietnamese_alarm_efficiently_zmm_(__m512i text_u8x64,
                                                                                              __mmask64 load_m64) {
    __m512i const x_e1_u8x64 = _mm512_set1_epi8((char)0xE1);
    __m512i const x_c3_u8x64 = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c5_u8x64 = _mm512_set1_epi8((char)0xC5);
    __m512i const x_ef_u8x64 = _mm512_set1_epi8((char)0xEF);
    __m512i const x_ba_u8x64 = _mm512_set1_epi8((char)0xBA);
    __m512i const x_9f_u8x64 = _mm512_set1_epi8((char)0x9F);
    __m512i const x_bf_u8x64 = _mm512_set1_epi8((char)0xBF);
    __m512i const x_ac_u8x64 = _mm512_set1_epi8((char)0xAC);
    __m512i const x_84_u8x64 = _mm512_set1_epi8((char)0x84);
    __m512i const x_96_u8x64 = _mm512_set1_epi8((char)0x96);
    __m512i const x_02_u8x64 = _mm512_set1_epi8(0x02);
    __m512i off_e1_u8x64 = _mm512_sub_epi8(text_u8x64, x_e1_u8x64);
    __mmask64 is_e1_or_e2_m64 = _mm512_cmplt_epu8_mask(off_e1_u8x64, x_02_u8x64);
    __mmask64 is_e1_m64 = is_e1_or_e2_m64 & _mm512_testn_epi8_mask(off_e1_u8x64, off_e1_u8x64); 
    __mmask64 is_e2_m64 = is_e1_or_e2_m64 & ~is_e1_m64;
    __mmask64 is_c3_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c3_u8x64);
    __mmask64 is_c5_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_c5_u8x64);
    __mmask64 is_ef_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, x_ef_u8x64);
    if (!(is_e1_or_e2_m64 | is_c3_m64 | is_c5_m64 | is_ef_m64)) return 0;
    __mmask64 ba_second_m64 = (is_e1_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, x_ba_u8x64);
    __mmask64 bad_third_m64 = (ba_second_m64 << 1) & _mm512_mask_cmpge_epu8_mask(load_m64, text_u8x64, x_96_u8x64) &
                              _mm512_mask_cmple_epu8_mask(load_m64, text_u8x64, x_9f_u8x64);
    __mmask64 sharp_s_m64 = (is_c3_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, x_9f_u8x64);
    __mmask64 long_s_m64 = (is_c5_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, x_bf_u8x64);
    __mmask64 ligature_m64 = (is_ef_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, x_ac_u8x64);
    __mmask64 kelvin_m64 = (is_e2_m64 << 1) & _mm512_cmpeq_epi8_mask(text_u8x64, x_84_u8x64);
    __mmask64 danger_m64 = (bad_third_m64 >> 2) | (sharp_s_m64 >> 1) | (long_s_m64 >> 1) | (ligature_m64 >> 1) |
                           (kelvin_m64 >> 1);
    sz_assert_(danger_m64 == sz_utf8_uncased_search_icelake_vietnamese_alarm_naively_zmm_(text_u8x64, load_m64) &&
               "Efficient Vietnamese alarm must match naive implementation");
    return danger_m64;
}
/**
 *  @brief Vietnamese uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_vietnamese_k
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_icelake_vietnamese_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                   
    sz_cptr_t needle, sz_size_t needle_length,                       
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,        
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_icelake_scripted_( 
        sz_utf8_uncased_search_icelake_vietnamese_fold_efficiently_zmm_,
        sz_utf8_uncased_search_icelake_vietnamese_alarm_efficiently_zmm_, 
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}
#pragma endregion // Vietnamese Uncased Find
#pragma region Georgian Uncased Find
/**
 *  @brief Detect danger zones in Georgian text (Mtavruli, Asomtavruli, Nuskhuri).
 *  @sa sz_utf8_uncased_rune_safe_georgian_k
 *
 *  Georgian Mkhedruli is caseless, so we only need to detect non-Mkhedruli Georgian scripts
 *  that require special handling via serial fallback:
 *  - Mtavruli (E1 B2 xx): Modern uppercase, folds to Mkhedruli
 *  - Asomtavruli (E1 82 A0-E5): Historical uppercase, folds to Nuskhuri
 *  - Nuskhuri (E2 B4 xx): Ecclesiastical script
 *
 *  All Georgian scripts use 3-byte UTF-8, so no length changes during folding.
 */
SZ_HELPER_AUTO __mmask64 sz_utf8_uncased_search_icelake_georgian_alarm_zmm_(__m512i text_u8x64, __mmask64 load_m64) {
    sz_unused_(load_m64); 
    __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xE1));
    __mmask64 is_e2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xE2));
    __mmask64 is_b2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB2)); 
    __mmask64 is_82_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x82)); 
    __mmask64 is_b4_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB4)); 
    __mmask64 mtavruli_m64 = (is_e1_m64 << 1) & is_b2_m64;
    __mmask64 after_e1_82_m64 = (is_e1_m64 << 1) & is_82_m64;
    __m512i offset_a0_u8x64 = _mm512_add_epi8(text_u8x64, _mm512_set1_epi8(0x60)); 
    __mmask64 in_a0_e5_m64 = _mm512_cmplt_epu8_mask(offset_a0_u8x64, _mm512_set1_epi8(0x46));
    __mmask64 asomtavruli_m64 = (after_e1_82_m64 << 1) & in_a0_e5_m64;
    __mmask64 nuskhuri_m64 = (is_e2_m64 << 1) & is_b4_m64;
    return (mtavruli_m64 >> 1) | (asomtavruli_m64 >> 2) | (nuskhuri_m64 >> 1);
}
/**
 *  @brief Optimized alarm function for Georgian danger zone detection.
 *  @sa sz_utf8_uncased_search_icelake_georgian_alarm_zmm_
 *
 *  Every Georgian danger sequence starts with a 2-byte (lead, second) pair with a UNIQUE second
 *  byte, so the pair test inverts: one VPERMB maps each second byte to the lead that would make
 *  it dangerous (sentinel 0x00 otherwise), another VPERMB materializes the previous byte
 *  in-register, and a single VPCMPEQB matches them up. Two qualifiers make it exact: the second
 *  byte must be a continuation (its low-6-bit VPERMB index aliases ASCII and lead bytes), and the
 *  previous byte must be >= C0 (kills the sentinel matching a 0x00 byte and the zeroed lane 0).
 *  The Asomtavruli third-byte range check sits behind a `pair` branch, keeping the hot path at
 *  5 port-5 ops. The reference `sz_utf8_uncased_search_icelake_georgian_alarm_zmm_` keeps
 *  its name unsuffixed because the probe harness references it directly.
 */
SZ_HELPER_NOINLINE __mmask64 sz_utf8_uncased_search_icelake_georgian_alarm_efficiently_zmm_(__m512i text_u8x64,
                                                                                            __mmask64 load_m64) {
    sz_unused_(load_m64); 
    __m512i const previous_byte_indices_u8x64 = _mm512_set_epi8( 
        62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35,
        34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
        5, 4, 3, 2, 1, 0, 0);
    __m512i const expected_leads_lut_u8x64 = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (char)0xE2, 0, (char)0xE1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (char)0xE1, 0, 0);
    __m512i const x_80_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_u8x64 = _mm512_set1_epi8((char)0x40);
    __m512i const x_c0_u8x64 = _mm512_set1_epi8((char)0xC0);
    __m512i previous_u8x64 = _mm512_maskz_permutexvar_epi8(0xFFFFFFFFFFFFFFFEull, previous_byte_indices_u8x64,
                                                           text_u8x64);
    __m512i expected_leads_u8x64 = _mm512_permutexvar_epi8(text_u8x64, expected_leads_lut_u8x64);
    __mmask64 pair_m64 = _mm512_cmpeq_epi8_mask(previous_u8x64, expected_leads_u8x64);
    pair_m64 &= _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, x_80_u8x64), x_40_u8x64);
    pair_m64 &= _mm512_cmpge_epu8_mask(previous_u8x64, x_c0_u8x64);
    __mmask64 danger_m64 = 0;
    if (pair_m64) {
        __mmask64 is_82_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x82));
        __mmask64 in_a0_e5_m64 = _mm512_cmplt_epu8_mask(_mm512_add_epi8(text_u8x64, _mm512_set1_epi8(0x60)),
                                                        _mm512_set1_epi8(0x46));
        __mmask64 asomtavruli_pair_m64 = pair_m64 & is_82_m64 & (in_a0_e5_m64 >> 1);
        danger_m64 = ((pair_m64 & ~is_82_m64) >> 1) | (asomtavruli_pair_m64 >> 1);
    }
    sz_assert_(danger_m64 == sz_utf8_uncased_search_icelake_georgian_alarm_zmm_(text_u8x64, load_m64) &&
               "Efficient Georgian alarm must match the reference implementation");
    return danger_m64;
}
/**
 *  @brief Fold Georgian text - only ASCII A-Z needs folding.
 *  @sa sz_utf8_uncased_rune_safe_georgian_k
 *
 *  Georgian Mkhedruli is caseless, so Georgian characters pass through unchanged.
 *  Only ASCII uppercase letters need folding for mixed Latin text.
 */
SZ_HELPER_NOINLINE __m512i sz_utf8_uncased_search_icelake_georgian_fold_zmm_(__m512i text_u8x64) {
    __m512i offset_a_u8x64 = _mm512_sub_epi8(text_u8x64, _mm512_set1_epi8('A'));
    __mmask64 is_upper_m64 = _mm512_cmple_epu8_mask(offset_a_u8x64, _mm512_set1_epi8(25));
    return _mm512_mask_add_epi8(text_u8x64, is_upper_m64, text_u8x64, _mm512_set1_epi8(0x20));
}
/**
 *  @brief Georgian uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_georgian_k
 *
 *  This is the fastest non-ASCII kernel because Mkhedruli is caseless - no Georgian
 *  folding needed in the hot path. Only ASCII A-Z folding for mixed text.
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_icelake_georgian_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                 
    sz_cptr_t needle, sz_size_t needle_length,                     
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,      
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_icelake_scripted_( 
        sz_utf8_uncased_search_icelake_georgian_fold_zmm_,
        sz_utf8_uncased_search_icelake_georgian_alarm_efficiently_zmm_, 
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}
#pragma endregion // Georgian Uncased Find
SZ_API_COMPTIME sz_cptr_t sz_utf8_uncased_search_icelake( 
    sz_cptr_t haystack, sz_size_t haystack_length,        
    sz_cptr_t needle, sz_size_t needle_length,            
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {
    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }
    int const is_unknown = needle_metadata->kernel_id == sz_utf8_uncased_rune_unknown_k;
    int const known_agnostic = needle_metadata->kernel_id == sz_utf8_uncased_rune_invariant_k;
    if (known_agnostic || (is_unknown && sz_utf8_find_cased_icelake(needle, needle_length) == SZ_NULL_CHAR)) {
        sz_cptr_t result = sz_find_skylake(haystack, haystack_length, needle, needle_length);
        *matched_length = result ? needle_length : 0;
        return result;
    }
    if (is_unknown) {
        sz_utf8_uncased_needle_metadata_(needle, needle_length, needle_metadata);
        if (needle_metadata->kernel_id == sz_utf8_uncased_rune_fallback_serial_k)
            return sz_utf8_uncased_search_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                 matched_length);
    }
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_ascii_invariant_k) {
        if (needle_metadata->folded_slice_length <= 3)
            return sz_utf8_uncased_search_icelake_ascii_3probe_( 
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
        else
            return sz_utf8_uncased_search_icelake_ascii_4probe_( 
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    }
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_western_europe_k)
        return sz_utf8_uncased_search_icelake_western_europe_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_central_europe_k)
        return sz_utf8_uncased_search_icelake_central_europe_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_greek_k)
        return sz_utf8_uncased_search_icelake_greek_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_armenian_k)
        return sz_utf8_uncased_search_icelake_armenian_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_vietnamese_k)
        return sz_utf8_uncased_search_icelake_vietnamese_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_cyrillic_k)
        return sz_utf8_uncased_search_icelake_cyrillic_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_georgian_k)
        return sz_utf8_uncased_search_icelake_georgian_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    needle_metadata->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
    return sz_utf8_uncased_search_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                         matched_length);
}
SZ_API_COMPTIME sz_cptr_t sz_utf8_find_cased_icelake(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *text_cursor = (sz_u8_t const *)str;
    __m512i const a_upper_u8x64 = _mm512_set1_epi8('A');
    __m512i const a_lower_u8x64 = _mm512_set1_epi8('a');
    __m512i const range26_u8x64 = _mm512_set1_epi8(26);
    __m512i const x80_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const xc0_u8x64 = _mm512_set1_epi8((char)0xC0);
    __m512i const xc3_u8x64 = _mm512_set1_epi8((char)0xC3);
    __m512i const xe0_u8x64 = _mm512_set1_epi8((char)0xE0);
    __m512i const xf0_u8x64 = _mm512_set1_epi8((char)0xF0);
    __m512i const xf8_u8x64 = _mm512_set1_epi8((char)0xF8);
    while (length) {
        sz_size_t block_length = sz_min_of_two(length, 61);
        __mmask64 lead_m64 = sz_u64_mask_until_(block_length);
        __mmask64 load_m64 = sz_u64_clamp_mask_until_(length); 
        __m512i text_u8x64 = _mm512_maskz_loadu_epi8(load_m64, text_cursor);
        __mmask64 is_upper_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, a_upper_u8x64), range26_u8x64);
        __mmask64 is_lower_m64 = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, a_lower_u8x64), range26_u8x64);
        if (is_upper_m64 | is_lower_m64) return sz_utf8_find_cased_serial((sz_cptr_t)text_cursor, length);
        __mmask64 is_non_ascii_m64 = _mm512_movepi8_mask(text_u8x64) & lead_m64;
        if (is_non_ascii_m64) {
            __mmask64 is_two_m64 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(text_u8x64, xe0_u8x64), xc0_u8x64) &
                                   lead_m64;
            __mmask64 is_three_m64 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(text_u8x64, xf0_u8x64), xe0_u8x64) &
                                     lead_m64;
            __mmask64 is_four_m64 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(text_u8x64, xf8_u8x64), xf0_u8x64) &
                                    lead_m64;
            if (is_four_m64) {
                __mmask64 after_f0_m64 = is_four_m64 << 1;
                __mmask64 is_90_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x90));
                __mmask64 is_91_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x91));
                __mmask64 is_96_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x96));
                __mmask64 is_9d_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x9D));
                __mmask64 is_9e_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x9E));
                if (after_f0_m64 & (is_90_m64 | is_91_m64 | is_96_m64 | is_9d_m64 | is_9e_m64))
                    return sz_utf8_find_cased_serial((sz_cptr_t)text_cursor, length);
            }
            if (is_two_m64) {
                __mmask64 is_bicameral_m64 = _mm512_cmplt_epu8_mask( 
                    _mm512_sub_epi8(text_u8x64, xc3_u8x64), _mm512_set1_epi8(0x14));
                __mmask64 is_c2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xC2)) & is_two_m64;
                if (is_c2_m64) {
                    __mmask64 after_c2_m64 = is_c2_m64 << 1;
                    __mmask64 is_b5_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB5));
                    if (after_c2_m64 & is_b5_m64) return sz_utf8_find_cased_serial((sz_cptr_t)text_cursor, length);
                }
                if (is_bicameral_m64 & is_two_m64) return sz_utf8_find_cased_serial((sz_cptr_t)text_cursor, length);
            }
            if (is_three_m64) {
                __mmask64 is_e1_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xE1));
                if (is_e1_m64 & is_three_m64) return sz_utf8_find_cased_serial((sz_cptr_t)text_cursor, length);
                __mmask64 is_ef_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xEF));
                if (is_ef_m64 & is_three_m64) return sz_utf8_find_cased_serial((sz_cptr_t)text_cursor, length);
                __mmask64 is_e2_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xE2)) & is_three_m64;
                if (is_e2_m64) {
                    __mmask64 after_e2_m64 = is_e2_m64 << 1;
                    __mmask64 e2_second_safe_m64 = _mm512_cmplt_epu8_mask( 
                        _mm512_sub_epi8(text_u8x64, x80_u8x64), _mm512_set1_epi8(0x04));
                    if (after_e2_m64 & ~e2_second_safe_m64)
                        return sz_utf8_find_cased_serial((sz_cptr_t)text_cursor, length);
                }
                __mmask64 is_ea_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xEA)) & is_three_m64;
                if (is_ea_m64) {
                    __mmask64 after_ea_m64 = is_ea_m64 << 1;
                    __mmask64 is_99_range_m64 = _mm512_cmplt_epu8_mask( 
                        _mm512_sub_epi8(text_u8x64, _mm512_set1_epi8((char)0x99)), _mm512_set1_epi8(0x07));
                    __mmask64 is_ad_range_m64 = _mm512_cmplt_epu8_mask( 
                        _mm512_sub_epi8(text_u8x64, _mm512_set1_epi8((char)0xAC)), _mm512_set1_epi8(0x03));
                    if (after_ea_m64 & (is_99_range_m64 | is_ad_range_m64))
                        return sz_utf8_find_cased_serial((sz_cptr_t)text_cursor, length);
                }
            }
        }
        text_cursor += block_length;
        length -= block_length;
    }
    return SZ_NULL_CHAR;
}
SZ_API_COMPTIME sz_ordering_t sz_utf8_uncased_order_icelake(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                            sz_size_t b_length) {
    return sz_utf8_uncased_order_serial(a, a_length, b, b_length);
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
