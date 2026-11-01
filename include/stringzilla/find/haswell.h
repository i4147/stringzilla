/**
 *  @brief Haswell (AVX2) backend for substring & byte-set search.
 *  @file include/stringzilla/find/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_HASWELL_H_
#define STRINGZILLA_FIND_HASWELL_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#include "stringzilla/find/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,lzcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "lzcnt")
#endif
SZ_API_COMPTIME sz_cptr_t sz_find_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    int matches_mask;
    sz_u256_vec_t haystack_vec, needle_vec;
    needle_vec.ymm = _mm256_set1_epi8(needle[0]);
    while (haystack_length >= 32) {
        haystack_vec.ymm = _mm256_lddqu_si256((__m256i const *)haystack);
        matches_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(haystack_vec.ymm, needle_vec.ymm));
        if (matches_mask) return haystack + (int)_tzcnt_u32(matches_mask);
        haystack += 32, haystack_length -= 32;
    }
    return sz_find_byte_serial(haystack, haystack_length, needle);
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    int matches_mask;
    sz_u256_vec_t haystack_vec, needle_vec;
    needle_vec.ymm = _mm256_set1_epi8(needle[0]);
    while (haystack_length >= 32) {
        haystack_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + haystack_length - 32));
        matches_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(haystack_vec.ymm, needle_vec.ymm));
        if (matches_mask) return haystack + haystack_length - 1 - (int)_lzcnt_u32(matches_mask);
        haystack_length -= 32;
    }
    return sz_rfind_byte_serial(haystack, haystack_length, needle);
}
SZ_API_COMPTIME sz_cptr_t sz_find_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length) {
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_haswell(haystack, haystack_length, needle);
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(needle[offset_first]);
    n_mid_vec.ymm = _mm256_set1_epi8(needle[offset_mid]);
    n_last_vec.ymm = _mm256_set1_epi8(needle[offset_last]);
    for (; haystack_length >= needle_length + 32; haystack += 32, haystack_length -= 32) {
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + offset_first));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + offset_mid));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + offset_last));
        matches_vec.i32 = 
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches_vec.u32) {
            int potential_offset = (int)_tzcnt_u32(matches_vec.u32);
            if (sz_equal_haswell(haystack + potential_offset, needle, needle_length))
                return haystack + potential_offset;
            matches_vec.u32 &= matches_vec.u32 - 1;
        }
    }
    return sz_find_serial(haystack, haystack_length, needle, needle_length);
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length) {
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_haswell(haystack, haystack_length, needle);
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(needle[offset_first]);
    n_mid_vec.ymm = _mm256_set1_epi8(needle[offset_mid]);
    n_last_vec.ymm = _mm256_set1_epi8(needle[offset_last]);
    sz_cptr_t haystack_cursor;
    for (; haystack_length >= needle_length + 32; haystack_length -= 32) {
        haystack_cursor = haystack + haystack_length - needle_length - 32 + 1;
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack_cursor + offset_first));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack_cursor + offset_mid));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack_cursor + offset_last));
        matches_vec.i32 = 
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches_vec.u32) {
            int potential_offset = (int)_lzcnt_u32(matches_vec.u32);
            if (sz_equal_haswell(haystack + haystack_length - needle_length - potential_offset, needle, needle_length))
                return haystack + haystack_length - needle_length - potential_offset;
            matches_vec.u32 &= ~(1u << (31 - potential_offset));
        }
    }
    return sz_rfind_serial(haystack, haystack_length, needle, needle_length);
}
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_haswell(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    sz_u128_vec_t byte_mask_vec;
    sz_u128_vec_t filter_low_vec, filter_high_vec;
    sz_u128_vec_t low_evens_vec, high_evens_vec;
    sz_u128_vec_t low_odds_vec, high_odds_vec;
    sz_u128_vec_t evens_xmm_vec, odds_xmm_vec;
    sz_u256_vec_t filter_even_vec, filter_odd_vec;
    sz_u256_vec_t text_vec;
    sz_u256_vec_t matches_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec;
    byte_mask_vec.xmm = _mm_set1_epi16(0x00ff);
    filter_low_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter));
    filter_high_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter) + 1);
    low_evens_vec.xmm = _mm_and_si128(filter_low_vec.xmm, byte_mask_vec.xmm);
    high_evens_vec.xmm = _mm_and_si128(filter_high_vec.xmm, byte_mask_vec.xmm);
    low_odds_vec.xmm = _mm_srli_epi16(filter_low_vec.xmm, 8);
    high_odds_vec.xmm = _mm_srli_epi16(filter_high_vec.xmm, 8);
    evens_xmm_vec.xmm = _mm_packus_epi16(low_evens_vec.xmm, high_evens_vec.xmm);
    odds_xmm_vec.xmm = _mm_packus_epi16(low_odds_vec.xmm, high_odds_vec.xmm);
    filter_even_vec.ymm = _mm256_set_m128i(evens_xmm_vec.xmm, evens_xmm_vec.xmm);
    filter_odd_vec.ymm = _mm256_set_m128i(odds_xmm_vec.xmm, odds_xmm_vec.xmm);
    bitmask_lookup_vec.ymm = _mm256_set_epi8(                       
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, 
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);
    while (length >= 32) {
        text_vec.ymm = _mm256_lddqu_si256((__m256i const *)text);
        lower_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, _mm256_set1_epi8(0x0f));
        bitmask_vec.ymm = _mm256_shuffle_epi8(bitmask_lookup_vec.ymm, lower_nibbles_vec.ymm);
        higher_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), _mm256_set1_epi8(0x0f));
        bitset_even_vec.ymm = _mm256_shuffle_epi8(filter_even_vec.ymm, higher_nibbles_vec.ymm);
        bitset_odd_vec.ymm = _mm256_shuffle_epi8(filter_odd_vec.ymm, higher_nibbles_vec.ymm);
        __m256i take_first_u8x32 = _mm256_cmpgt_epi8(_mm256_set1_epi8(8), lower_nibbles_vec.ymm);
        bitset_even_vec.ymm = _mm256_blendv_epi8(bitset_odd_vec.ymm, bitset_even_vec.ymm, take_first_u8x32);
        matches_vec.ymm = _mm256_and_si256(bitset_even_vec.ymm, bitmask_vec.ymm);
        matches_vec.ymm = _mm256_cmpeq_epi8(matches_vec.ymm, _mm256_setzero_si256());
        int matches_mask = ~_mm256_movemask_epi8(matches_vec.ymm);
        if (matches_mask) {
            int offset = (int)_tzcnt_u32(matches_mask);
            return text + offset;
        }
        else { text += 32, length -= 32; }
    }
    return sz_find_byteset_serial(text, length, filter);
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_haswell(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    sz_u128_vec_t byte_mask_vec;
    sz_u128_vec_t filter_low_vec, filter_high_vec;
    sz_u128_vec_t low_evens_vec, high_evens_vec;
    sz_u128_vec_t low_odds_vec, high_odds_vec;
    sz_u128_vec_t evens_xmm_vec, odds_xmm_vec;
    sz_u256_vec_t filter_even_vec, filter_odd_vec;
    sz_u256_vec_t text_vec;
    sz_u256_vec_t matches_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec;
    byte_mask_vec.xmm = _mm_set1_epi16(0x00ff);
    filter_low_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter));
    filter_high_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter) + 1);
    low_evens_vec.xmm = _mm_and_si128(filter_low_vec.xmm, byte_mask_vec.xmm);
    high_evens_vec.xmm = _mm_and_si128(filter_high_vec.xmm, byte_mask_vec.xmm);
    low_odds_vec.xmm = _mm_srli_epi16(filter_low_vec.xmm, 8);
    high_odds_vec.xmm = _mm_srli_epi16(filter_high_vec.xmm, 8);
    evens_xmm_vec.xmm = _mm_packus_epi16(low_evens_vec.xmm, high_evens_vec.xmm);
    odds_xmm_vec.xmm = _mm_packus_epi16(low_odds_vec.xmm, high_odds_vec.xmm);
    filter_even_vec.ymm = _mm256_set_m128i(evens_xmm_vec.xmm, evens_xmm_vec.xmm);
    filter_odd_vec.ymm = _mm256_set_m128i(odds_xmm_vec.xmm, odds_xmm_vec.xmm);
    bitmask_lookup_vec.ymm = _mm256_set_epi8(                       
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, 
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);
    while (length >= 32) {
        sz_cptr_t const window = text + length - 32;
        text_vec.ymm = _mm256_lddqu_si256((__m256i const *)window);
        lower_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, _mm256_set1_epi8(0x0f));
        bitmask_vec.ymm = _mm256_shuffle_epi8(bitmask_lookup_vec.ymm, lower_nibbles_vec.ymm);
        higher_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), _mm256_set1_epi8(0x0f));
        bitset_even_vec.ymm = _mm256_shuffle_epi8(filter_even_vec.ymm, higher_nibbles_vec.ymm);
        bitset_odd_vec.ymm = _mm256_shuffle_epi8(filter_odd_vec.ymm, higher_nibbles_vec.ymm);
        __m256i take_first_u8x32 = _mm256_cmpgt_epi8(_mm256_set1_epi8(8), lower_nibbles_vec.ymm);
        bitset_even_vec.ymm = _mm256_blendv_epi8(bitset_odd_vec.ymm, bitset_even_vec.ymm, take_first_u8x32);
        matches_vec.ymm = _mm256_and_si256(bitset_even_vec.ymm, bitmask_vec.ymm);
        matches_vec.ymm = _mm256_cmpeq_epi8(matches_vec.ymm, _mm256_setzero_si256());
        int matches_mask = ~_mm256_movemask_epi8(matches_vec.ymm);
        if (matches_mask) {
            int offset = 31 - (int)_lzcnt_u32((sz_u32_t)matches_mask);
            return window + offset;
        }
        else { length -= 32; }
    }
    return sz_rfind_byteset_serial(text, length, filter);
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
