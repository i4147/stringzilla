/**
 *  @brief NEON backend for hardware-accelerated memory operations on 64-bit Arm CPUs.
 *  @file include/stringzilla/memory/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_NEON_H_
#define STRINGZILLA_MEMORY_NEON_H_
#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif
SZ_API_COMPTIME void sz_copy_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (length < 16) {
        if (length) sz_copy_serial(target, source, length);
        return;
    }
    for (; length >= 64; target += 64, source += 64, length -= 64) {
        vst1q_u8((sz_u8_t *)(target + 0), vld1q_u8((sz_u8_t const *)(source + 0)));
        vst1q_u8((sz_u8_t *)(target + 16), vld1q_u8((sz_u8_t const *)(source + 16)));
        vst1q_u8((sz_u8_t *)(target + 32), vld1q_u8((sz_u8_t const *)(source + 32)));
        vst1q_u8((sz_u8_t *)(target + 48), vld1q_u8((sz_u8_t const *)(source + 48)));
    }
    for (; length >= 16; target += 16, source += 16, length -= 16)
        vst1q_u8((sz_u8_t *)target, vld1q_u8((sz_u8_t const *)source));
    if (length) {
        sz_size_t const back_step = 16 - length;
        vst1q_u8((sz_u8_t *)(target - back_step), vld1q_u8((sz_u8_t const *)(source - back_step)));
    }
}
SZ_API_COMPTIME void sz_move_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target < source || target >= source + length) {
        for (; length >= 64; target += 64, source += 64, length -= 64) {
            vst1q_u8((sz_u8_t *)(target + 0), vld1q_u8((sz_u8_t const *)(source + 0)));
            vst1q_u8((sz_u8_t *)(target + 16), vld1q_u8((sz_u8_t const *)(source + 16)));
            vst1q_u8((sz_u8_t *)(target + 32), vld1q_u8((sz_u8_t const *)(source + 32)));
            vst1q_u8((sz_u8_t *)(target + 48), vld1q_u8((sz_u8_t const *)(source + 48)));
        }
        for (; length >= 16; target += 16, source += 16, length -= 16)
            vst1q_u8((sz_u8_t *)target, vld1q_u8((sz_u8_t const *)source));
        while (length) { *target++ = *source++, --length; }
    }
    else {
        target += length;
        source += length;
        sz_u128_vec_t source_vec;
        while (length >= 16) {
            target -= 16, source -= 16, length -= 16;
            source_vec.u8x16 = vld1q_u8((sz_u8_t const *)source);
            vst1q_u8((sz_u8_t *)target, source_vec.u8x16);
        }
        while (length) {
            target -= 1, source -= 1, length -= 1;
            *target = *source;
        }
    }
}
SZ_API_COMPTIME void sz_fill_neon(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    uint8x16_t fill_u8x16 = vdupq_n_u8(value); 
    if (length < 16) {
        if (length) sz_fill_serial(target, length, value);
        return;
    }
    for (; length >= 64; target += 64, length -= 64) {
        vst1q_u8((sz_u8_t *)(target + 0), fill_u8x16);
        vst1q_u8((sz_u8_t *)(target + 16), fill_u8x16);
        vst1q_u8((sz_u8_t *)(target + 32), fill_u8x16);
        vst1q_u8((sz_u8_t *)(target + 48), fill_u8x16);
    }
    for (; length >= 16; target += 16, length -= 16) vst1q_u8((sz_u8_t *)target, fill_u8x16);
    if (length) vst1q_u8((sz_u8_t *)(target - (16 - length)), fill_u8x16);
}
SZ_API_COMPTIME void sz_lookup_neon(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                    char const lut[sz_at_least_(256)]) {
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }
    sz_size_t head_length = (16 - ((sz_size_t)target % 16)) % 16; 
    sz_size_t tail_length = (sz_size_t)(target + length) % 16;    
    sz_size_t body_length = length - head_length - tail_length;
    uint8x16x4_t lut_0_to_63_vec, lut_64_to_127_vec, lut_128_to_191_vec, lut_192_to_255_vec;
    lut_0_to_63_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 0));
    lut_64_to_127_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 64));
    lut_128_to_191_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 128));
    lut_192_to_255_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 192));
    sz_u128_vec_t source_vec;
    sz_u128_vec_t lookup_0_to_63_vec, lookup_64_to_127_vec, lookup_128_to_191_vec, lookup_192_to_255_vec;
    sz_u128_vec_t blended_0_to_255_vec;
    for (; head_length; target += 1, source += 1, head_length -= 1) *target = lut[*(sz_u8_t const *)source];
    for (; body_length >= 16; source += 16, target += 16, body_length -= 16) {
        source_vec.u8x16 = vld1q_u8((sz_u8_t const *)source);
        lookup_0_to_63_vec.u8x16 = vqtbl4q_u8(lut_0_to_63_vec, source_vec.u8x16);
        lookup_64_to_127_vec.u8x16 = vqtbl4q_u8(lut_64_to_127_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0x40)));
        lookup_128_to_191_vec.u8x16 = vqtbl4q_u8(lut_128_to_191_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0x80)));
        lookup_192_to_255_vec.u8x16 = vqtbl4q_u8(lut_192_to_255_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0xc0)));
        blended_0_to_255_vec.u8x16 = vorrq_u8( 
            vorrq_u8(lookup_0_to_63_vec.u8x16, lookup_64_to_127_vec.u8x16),
            vorrq_u8(lookup_128_to_191_vec.u8x16, lookup_192_to_255_vec.u8x16));
        vst1q_u8((sz_u8_t *)target, blended_0_to_255_vec.u8x16);
    }
    for (; tail_length; target += 1, source += 1, tail_length -= 1) *target = lut[*(sz_u8_t const *)source];
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
