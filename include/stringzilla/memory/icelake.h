/**
 *  @brief AVX-512 VBMI backend for hardware-accelerated memory operations on Ice Lake and newer x86 CPUs.
 *  @file include/stringzilla/memory/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_ICELAKE_H_
#define STRINGZILLA_MEMORY_ICELAKE_H_
#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                      \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "bmi", "bmi2")
#endif
SZ_API_COMPTIME void sz_lookup_icelake(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                       char const lut[sz_at_least_(256)]) {
    if (length <= 128) {
        sz_u8_t const *lut_u8 = (sz_u8_t const *)lut;
        sz_u8_t const *source_u8 = (sz_u8_t const *)source;
        sz_u8_t *target_u8 = (sz_u8_t *)target;
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
            target_u8[byte_index] = lut_u8[source_u8[byte_index]];
        return;
    }
    sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; 
    sz_size_t tail_length = (sz_size_t)(target + length) % 64;    
    __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
    __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);
    sz_u512_vec_t lut_0_to_63_vec, lut_64_to_127_vec, lut_128_to_191_vec, lut_192_to_255_vec;
    lut_0_to_63_vec.zmm = _mm512_loadu_si512((lut));
    lut_64_to_127_vec.zmm = _mm512_loadu_si512((lut + 64));
    lut_128_to_191_vec.zmm = _mm512_loadu_si512((lut + 128));
    lut_192_to_255_vec.zmm = _mm512_loadu_si512((lut + 192));
    __mmask64 high_bit_mask_m64;
    sz_u512_vec_t source_vec, low_half_vec, high_half_vec, result_vec;
    if (head_length) {
        source_vec.zmm = _mm512_maskz_loadu_epi8(head_mask_m64, source);
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        high_bit_mask_m64 = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask_m64, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_mask_storeu_epi8(target, head_mask_m64, result_vec.zmm);
        source += head_length, target += head_length, length -= head_length;
    }
    while (length >= 64) {
        source_vec.zmm = _mm512_loadu_si512(source);
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        high_bit_mask_m64 = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask_m64, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_store_si512(target, result_vec.zmm); 
        source += 64, target += 64, length -= 64;
    }
    if (tail_length) {
        source_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask_m64, source);
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        high_bit_mask_m64 = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask_m64, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_mask_storeu_epi8(target, tail_mask_m64, result_vec.zmm);
        source += tail_length, target += tail_length, length -= tail_length;
    }
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
