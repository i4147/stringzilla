/**
 *  @brief WebAssembly relaxed-SIMD backend for memory (level above SIMD128).
 *  @file include/stringzilla/memory/v128relaxed.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_V128RELAXED_H_
#define STRINGZILLA_MEMORY_V128RELAXED_H_
#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"
#include "stringzilla/memory/v128.h" 
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif
SZ_API_COMPTIME void sz_copy_v128relaxed(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_copy_v128(target, source, length);
}
SZ_API_COMPTIME void sz_move_v128relaxed(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_move_v128(target, source, length);
}
SZ_API_COMPTIME void sz_fill_v128relaxed(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    sz_fill_v128(target, length, value);
}
SZ_API_COMPTIME void sz_lookup_v128relaxed(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                           char const lut[sz_at_least_(256)]) {
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }
    v128_t lut_u8x16s[16];
    for (sz_size_t sub_table_index = 0; sub_table_index < 16; ++sub_table_index)
        lut_u8x16s[sub_table_index] = wasm_v128_load(lut + sub_table_index * 16);
    v128_t const low_nibble_mask_u8x16 = wasm_i8x16_splat(0x0F);
    v128_t const bit4_u8x16 = wasm_i8x16_splat(0x10);
    v128_t const bit5_u8x16 = wasm_i8x16_splat(0x20);
    v128_t const bit6_u8x16 = wasm_i8x16_splat(0x40);
    v128_t const bit7_u8x16 = wasm_i8x16_splat((sz_i8_t)0x80);
    while (length >= 16) {
        v128_t source_u8x16 = wasm_v128_load(source);
        v128_t low_nibble_index_u8x16 = wasm_v128_and(source_u8x16, low_nibble_mask_u8x16);
        v128_t sub_table_u8x16s[16];
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; ++sub_table_index)
            sub_table_u8x16s[sub_table_index] = wasm_i8x16_relaxed_swizzle(lut_u8x16s[sub_table_index],
                                                                           low_nibble_index_u8x16);
        v128_t select_bit4_u8x16 = wasm_i8x16_eq(wasm_v128_and(source_u8x16, bit4_u8x16), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 2)
            sub_table_u8x16s[sub_table_index] = wasm_v128_bitselect(
                sub_table_u8x16s[sub_table_index], sub_table_u8x16s[sub_table_index + 1], select_bit4_u8x16);
        v128_t select_bit5_u8x16 = wasm_i8x16_eq(wasm_v128_and(source_u8x16, bit5_u8x16), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 4)
            sub_table_u8x16s[sub_table_index] = wasm_v128_bitselect(
                sub_table_u8x16s[sub_table_index], sub_table_u8x16s[sub_table_index + 2], select_bit5_u8x16);
        v128_t select_bit6_u8x16 = wasm_i8x16_eq(wasm_v128_and(source_u8x16, bit6_u8x16), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 8)
            sub_table_u8x16s[sub_table_index] = wasm_v128_bitselect(
                sub_table_u8x16s[sub_table_index], sub_table_u8x16s[sub_table_index + 4], select_bit6_u8x16);
        v128_t select_bit7_u8x16 = wasm_i8x16_eq(wasm_v128_and(source_u8x16, bit7_u8x16), wasm_i8x16_splat(0));
        v128_t result_u8x16 = wasm_v128_bitselect(sub_table_u8x16s[0], sub_table_u8x16s[8], select_bit7_u8x16);
        wasm_v128_store(target, result_u8x16);
        source += 16, target += 16, length -= 16;
    }
    if (length) sz_lookup_serial(target, length, source, lut);
}
#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif 
#ifdef __cplusplus
}
#endif
#endif 
