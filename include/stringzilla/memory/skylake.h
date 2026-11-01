/**
 *  @brief AVX-512 backend for hardware-accelerated memory operations on Skylake and newer x86 CPUs.
 *  @file include/stringzilla/memory/skylake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_SKYLAKE_H_
#define STRINGZILLA_MEMORY_SKYLAKE_H_
#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_SKYLAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,evex512"))), \
                             apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#endif
SZ_API_COMPTIME void sz_fill_skylake(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    __m512i value_u8x64 = _mm512_set1_epi8(value);
    if (length <= 64) {
        __mmask64 mask_m64 = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8(target, mask_m64, value_u8x64);
    }
    else {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; 
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;    
        sz_size_t body_length = length - head_length - tail_length;   
        __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);
        _mm512_mask_storeu_epi8(target, head_mask_m64, value_u8x64);
        for (target += head_length; body_length >= 64; target += 64, body_length -= 64)
            _mm512_store_si512(target, value_u8x64);
        _mm512_mask_storeu_epi8(target, tail_mask_m64, value_u8x64);
    }
}
SZ_API_COMPTIME void sz_copy_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    int const is_huge = length >= 1ull * 1024ull * 1024ull;
    if (length <= 64) {
        __mmask64 mask_m64 = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8(target, mask_m64, _mm512_maskz_loadu_epi8(mask_m64, source));
    }
    else if ((sz_size_t)target % 64 == 0 && (sz_size_t)source % 64 == 0 && !is_huge) {
        for (; length >= 64; target += 64, source += 64, length -= 64)
            _mm512_store_si512(target, _mm512_load_si512(source));
        __mmask64 mask_m64 = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8(target, mask_m64, _mm512_maskz_loadu_epi8(mask_m64, source));
    }
    else if (!is_huge) {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; 
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;    
        sz_size_t body_length = length - head_length - tail_length;   
        __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);
        _mm512_mask_storeu_epi8(target, head_mask_m64, _mm512_maskz_loadu_epi8(head_mask_m64, source));
        for (target += head_length, source += head_length; body_length >= 64;
             target += 64, source += 64, body_length -= 64)
            _mm512_store_si512(target, _mm512_loadu_si512(source)); 
        _mm512_mask_storeu_epi8(target, tail_mask_m64, _mm512_maskz_loadu_epi8(tail_mask_m64, source));
    }
    else {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64;
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;
        sz_size_t body_length = length - head_length - tail_length;
        __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);
        _mm512_mask_storeu_epi8(target, head_mask_m64, _mm512_maskz_loadu_epi8(head_mask_m64, source));
        _mm512_mask_storeu_epi8(target + head_length + body_length, tail_mask_m64,
                                _mm512_maskz_loadu_epi8(tail_mask_m64, source + head_length + body_length));
        for (target += head_length, source += head_length; 
             body_length >= 128;                           
             target += 64, source += 64, body_length -= 128) {
            _mm512_stream_si512((__m512i *)(target), _mm512_loadu_si512(source));
            _mm512_stream_si512((__m512i *)(target + body_length - 64), _mm512_loadu_si512(source + body_length - 64));
        }
        if (body_length >= 64) _mm512_stream_si512((__m512i *)target, _mm512_loadu_si512(source));
        _mm_sfence();
    }
}
SZ_API_COMPTIME void sz_move_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target == source) return; 
    if (length <= 64) {
        __mmask64 mask_m64 = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8(target, mask_m64, _mm512_maskz_loadu_epi8(mask_m64, source));
    }
    else if (length <= 128) {
        sz_size_t last_length = length - 64;
        __mmask64 mask_m64 = sz_u64_mask_until_(last_length);
        __m512i source0_u8x64 = _mm512_loadu_epi8(source);
        __m512i source1_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, source + 64);
        _mm512_storeu_epi8(target, source0_u8x64);
        _mm512_mask_storeu_epi8(target + 64, mask_m64, source1_u8x64);
    }
    else if (length <= 192) {
        sz_size_t last_length = length - 128;
        __mmask64 mask_m64 = sz_u64_mask_until_(last_length);
        __m512i source0_u8x64 = _mm512_loadu_epi8(source);
        __m512i source1_u8x64 = _mm512_loadu_epi8(source + 64);
        __m512i source2_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, source + 128);
        _mm512_storeu_epi8(target, source0_u8x64);
        _mm512_storeu_epi8(target + 64, source1_u8x64);
        _mm512_mask_storeu_epi8(target + 128, mask_m64, source2_u8x64);
    }
    else if (length <= 256) {
        sz_size_t last_length = length - 192;
        __mmask64 mask_m64 = sz_u64_mask_until_(last_length);
        __m512i source0_u8x64 = _mm512_loadu_epi8(source);
        __m512i source1_u8x64 = _mm512_loadu_epi8(source + 64);
        __m512i source2_u8x64 = _mm512_loadu_epi8(source + 128);
        __m512i source3_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, source + 192);
        _mm512_storeu_epi8(target, source0_u8x64);
        _mm512_storeu_epi8(target + 64, source1_u8x64);
        _mm512_storeu_epi8(target + 128, source2_u8x64);
        _mm512_mask_storeu_epi8(target + 192, mask_m64, source3_u8x64);
    }
    else if (target + length < source || target >= source + length) { sz_copy_skylake(target, source, length); }
    else {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; 
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;    
        sz_size_t body_length = length - head_length - tail_length;   
        __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);
        int const left_to_right_traversal = source > target;
        if (left_to_right_traversal) {
            _mm512_mask_storeu_epi8(target, head_mask_m64, _mm512_maskz_loadu_epi8(head_mask_m64, source));
            for (target += head_length, source += head_length; body_length >= 64;
                 target += 64, source += 64, body_length -= 64)
                _mm512_store_si512(target, _mm512_loadu_si512(source));
            _mm512_mask_storeu_epi8(target, tail_mask_m64, _mm512_maskz_loadu_epi8(tail_mask_m64, source));
        }
        else {
            _mm512_mask_storeu_epi8(target + head_length + body_length, tail_mask_m64,
                                    _mm512_maskz_loadu_epi8(tail_mask_m64, source + head_length + body_length));
            for (; body_length >= 64; body_length -= 64)
                _mm512_store_si512(target + head_length + body_length - 64,
                                   _mm512_loadu_si512(source + head_length + body_length - 64));
            _mm512_mask_storeu_epi8(target, head_mask_m64, _mm512_maskz_loadu_epi8(head_mask_m64, source));
        }
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
