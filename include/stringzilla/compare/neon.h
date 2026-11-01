/**
 *  @brief NEON backend for string comparison utilities.
 *  @file include/stringzilla/compare/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_NEON_H_
#define STRINGZILLA_COMPARE_NEON_H_
#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"
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
SZ_API_COMPTIME sz_ordering_t sz_order_neon(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    return sz_order_serial(a, a_length, b, b_length);
}
SZ_API_COMPTIME sz_bool_t sz_equal_neon(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    if (length < 16) return sz_equal_serial(a, b, length);
    sz_u128_vec_t a_vec, b_vec;
    sz_size_t offset = 0;
    do {
        a_vec.u8x16 = vld1q_u8((sz_u8_t const *)(a + offset));
        b_vec.u8x16 = vld1q_u8((sz_u8_t const *)(b + offset));
        uint8x16_t matches_u8x16 = vceqq_u8(a_vec.u8x16, b_vec.u8x16);
        if (vminvq_u8(matches_u8x16) != 255) return sz_false_k; 
        offset += 16;
    } while (offset + 16 <= length);
    a_vec.u8x16 = vld1q_u8((sz_u8_t const *)(a + length - 16));
    b_vec.u8x16 = vld1q_u8((sz_u8_t const *)(b + length - 16));
    uint8x16_t matches_u8x16 = vceqq_u8(a_vec.u8x16, b_vec.u8x16);
    if (vminvq_u8(matches_u8x16) != 255) return sz_false_k;
    return sz_true_k;
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
