/**
 *  @brief Ice Lake (AVX-512 VBMI+VAES) backend for substring & byte-set search.
 *  @file include/stringzilla/find/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_ICELAKE_H_
#define STRINGZILLA_FIND_ICELAKE_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#include "stringzilla/find/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                                        \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
    "lzcnt")
#endif
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_icelake(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    sz_u512_vec_t filter_even_vec, filter_odd_vec;
    __m256i filter_u8x32 = _mm256_lddqu_si256((__m256i const *)filter);
    filter_even_vec.zmm = _mm512_broadcast_i32x4(_mm256_castsi256_si128( 
        _mm256_maskz_compress_epi8(0x55555555, filter_u8x32)));
    filter_odd_vec.zmm = _mm512_broadcast_i32x4(_mm256_castsi256_si128( 
        _mm256_maskz_compress_epi8(0xaaaaaaaa, filter_u8x32)));
    sz_u512_vec_t text_vec;
    sz_u512_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u512_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u512_vec_t bitmask_vec, bitmask_lookup_vec;
    bitmask_lookup_vec.zmm = _mm512_set_epi8(                       
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, 
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, 
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, 
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);
    while (length) {
        sz_size_t load_length = sz_min_of_two(length, 64);
        __mmask64 load_m64 = sz_u64_mask_until_(load_length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(load_m64, text);
        lower_nibbles_vec.zmm = _mm512_and_si512(text_vec.zmm, _mm512_set1_epi8(0x0f));
        bitmask_vec.zmm = _mm512_shuffle_epi8(bitmask_lookup_vec.zmm, lower_nibbles_vec.zmm);
        higher_nibbles_vec.zmm = _mm512_and_si512(_mm512_srli_epi16(text_vec.zmm, 4), _mm512_set1_epi8(0x0f));
        bitset_even_vec.zmm = _mm512_shuffle_epi8(filter_even_vec.zmm, higher_nibbles_vec.zmm);
        bitset_odd_vec.zmm = _mm512_shuffle_epi8(filter_odd_vec.zmm, higher_nibbles_vec.zmm);
        __mmask64 take_first_m64 = _mm512_cmplt_epi8_mask(lower_nibbles_vec.zmm, _mm512_set1_epi8(8));
        bitset_even_vec.zmm = _mm512_mask_blend_epi8(take_first_m64, bitset_odd_vec.zmm, bitset_even_vec.zmm);
        __mmask64 matches_m64 = _mm512_mask_test_epi8_mask(load_m64, bitset_even_vec.zmm, bitmask_vec.zmm);
        if (matches_m64) {
            int offset = (int)_tzcnt_u64(matches_m64);
            return text + offset;
        }
        else { text += load_length, length -= load_length; }
    }
    return SZ_NULL_CHAR;
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_icelake(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    sz_u512_vec_t filter_even_vec, filter_odd_vec;
    __m256i filter_u8x32 = _mm256_lddqu_si256((__m256i const *)filter);
    filter_even_vec.zmm = _mm512_broadcast_i32x4(_mm256_castsi256_si128( 
        _mm256_maskz_compress_epi8(0x55555555, filter_u8x32)));
    filter_odd_vec.zmm = _mm512_broadcast_i32x4(_mm256_castsi256_si128( 
        _mm256_maskz_compress_epi8(0xaaaaaaaa, filter_u8x32)));
    sz_u512_vec_t text_vec;
    sz_u512_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u512_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u512_vec_t bitmask_vec, bitmask_lookup_vec;
    bitmask_lookup_vec.zmm = _mm512_set_epi8(                       
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, 
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, 
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, 
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);
    while (length) {
        sz_size_t load_length = sz_min_of_two(length, 64);
        sz_cptr_t window = text + length - load_length;
        __mmask64 load_m64 = sz_u64_mask_until_(load_length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(load_m64, window);
        lower_nibbles_vec.zmm = _mm512_and_si512(text_vec.zmm, _mm512_set1_epi8(0x0f));
        bitmask_vec.zmm = _mm512_shuffle_epi8(bitmask_lookup_vec.zmm, lower_nibbles_vec.zmm);
        higher_nibbles_vec.zmm = _mm512_and_si512(_mm512_srli_epi16(text_vec.zmm, 4), _mm512_set1_epi8(0x0f));
        bitset_even_vec.zmm = _mm512_shuffle_epi8(filter_even_vec.zmm, higher_nibbles_vec.zmm);
        bitset_odd_vec.zmm = _mm512_shuffle_epi8(filter_odd_vec.zmm, higher_nibbles_vec.zmm);
        __mmask64 take_first_m64 = _mm512_cmplt_epi8_mask(lower_nibbles_vec.zmm, _mm512_set1_epi8(8));
        bitset_even_vec.zmm = _mm512_mask_blend_epi8(take_first_m64, bitset_odd_vec.zmm, bitset_even_vec.zmm);
        sz_u64_t matches_mask = _cvtmask64_u64(
            _mm512_mask_test_epi8_mask(load_m64, bitset_even_vec.zmm, bitmask_vec.zmm));
        if (matches_mask) {
            int offset = 63 - (int)_lzcnt_u64(matches_mask); 
            return window + offset;
        }
        else { length -= load_length; }
    }
    return SZ_NULL_CHAR;
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
