/**
 *  @brief WebAssembly SIMD128 backend for compare.
 *  @file include/stringzilla/compare/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_V128_H_
#define STRINGZILLA_COMPARE_V128_H_
#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif
SZ_API_COMPTIME sz_ordering_t sz_order_v128(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    return sz_order_serial(a, a_length, b, b_length);
}
SZ_API_COMPTIME sz_bool_t sz_equal_v128(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    if (length < 16) return sz_equal_serial(a, b, length);
    sz_size_t offset = 0;
    while (offset + 64 <= length) {
        v128_t diff_at_0_u8x16 = wasm_v128_xor(wasm_v128_load(a + offset + 0), wasm_v128_load(b + offset + 0));
        v128_t diff_at_16_u8x16 = wasm_v128_xor(wasm_v128_load(a + offset + 16), wasm_v128_load(b + offset + 16));
        v128_t diff_at_32_u8x16 = wasm_v128_xor(wasm_v128_load(a + offset + 32), wasm_v128_load(b + offset + 32));
        v128_t diff_at_48_u8x16 = wasm_v128_xor(wasm_v128_load(a + offset + 48), wasm_v128_load(b + offset + 48));
        v128_t any_diff_u8x16 = wasm_v128_or(wasm_v128_or(diff_at_0_u8x16, diff_at_16_u8x16),
                                             wasm_v128_or(diff_at_32_u8x16, diff_at_48_u8x16));
        if (wasm_v128_any_true(any_diff_u8x16)) return sz_false_k;
        offset += 64;
    }
    while (offset + 16 <= length) {
        if (wasm_v128_any_true(wasm_v128_xor(wasm_v128_load(a + offset), wasm_v128_load(b + offset))))
            return sz_false_k;
        offset += 16;
    }
    if (wasm_v128_any_true(wasm_v128_xor(wasm_v128_load(a + length - 16), wasm_v128_load(b + length - 16))))
        return sz_false_k;
    return sz_true_k;
}
#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif 
#ifdef __cplusplus
}
#endif
#endif 
