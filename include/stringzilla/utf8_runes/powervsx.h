/**
 *  @brief POWER VSX backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_runes/powervsx.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_POWERVSX_H_
#define STRINGZILLA_UTF8_RUNES_POWERVSX_H_
#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif
SZ_API_COMPTIME sz_size_t sz_utf8_count_powervsx(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;
    __vector unsigned char const header_mask_u8x16 = vec_splats((unsigned char)0xC0);
    __vector unsigned char const continuation_pattern_u8x16 = vec_splats((unsigned char)0x80);
    __vector unsigned char const one_byte_u8x16 = vec_splats((unsigned char)1);
    sz_size_t const block_windows = (sz_size_t)1000000000; 
    while (length >= 16) {
        sz_size_t windows = length / 16;
        if (windows > block_windows) windows = block_windows;
        __vector unsigned int start_count_u32x4 = vec_splats((unsigned int)0);
        for (sz_size_t window_index = 0; window_index < windows; ++window_index, text_u8 += 16) {
            __vector unsigned char text_window_u8x16 = vec_xl(0, text_u8);
            __vector unsigned char header_result_u8x16 = vec_and(text_window_u8x16, header_mask_u8x16);
            __vector unsigned char continuation_mask_u8x16 = (__vector unsigned char)vec_cmpeq(
                header_result_u8x16, continuation_pattern_u8x16);
            __vector unsigned char start_mask_u8x16 = vec_andc(one_byte_u8x16, continuation_mask_u8x16);
            start_count_u32x4 = vec_msum(start_mask_u8x16, one_byte_u8x16,
                                         start_count_u32x4); 
        }
        char_count += (sz_size_t)start_count_u32x4[0] + start_count_u32x4[1] + start_count_u32x4[2] +
                      start_count_u32x4[3];
        length -= windows * 16;
    }
    if (length) char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}
SZ_HELPER_INLINE sz_u32_t sz_utf8_iterate_movemask_powervsx_(__vector unsigned char boolean_lanes_u8x16) {
    __vector unsigned char const gather_indices_u8x16 = {120, 112, 104, 96, 88, 80, 72, 64,
                                                         56,  48,  40,  32, 24, 16, 8,  0};
    __vector unsigned long long const gathered_mask_u64x2 = (__vector unsigned long long)vec_vbpermq(
        boolean_lanes_u8x16, gather_indices_u8x16);
#if SZ_IS_BIG_ENDIAN_
    return (sz_u32_t)(gathered_mask_u64x2[0] & 0xFFFFull);
#else
    return (sz_u32_t)(gathered_mask_u64x2[1] & 0xFFFFull);
#endif
}
/**
 *  @brief Locate the start byte of the n-th code-point (0-indexed), mirroring `sz_utf8_seek_serial`.
 *         Per tile a code-point-start bitmask `~continuation_mask` is popcounted to skip whole tiles; the
 *         n-th start's lane comes from `sz_u32_nth_set_bit`. The `< 16` tail defers to the serial reference.
 */
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    __vector unsigned char const header_mask_u8x16 = vec_splats((unsigned char)0xC0);
    __vector unsigned char const continuation_pattern_u8x16 = vec_splats((unsigned char)0x80);
    while (length >= 16) {
        __vector unsigned char text_window_u8x16 = vec_xl(0, text_u8);
        __vector unsigned char header_result_u8x16 = vec_and(text_window_u8x16, header_mask_u8x16);
        __vector unsigned char continuation_mask_u8x16 = (__vector unsigned char)vec_cmpeq(header_result_u8x16,
                                                                                           continuation_pattern_u8x16);
        sz_u32_t start_bits = (~sz_utf8_iterate_movemask_powervsx_(continuation_mask_u8x16)) & 0xFFFFu;
        sz_size_t const start_count = (sz_size_t)sz_u32_popcount(start_bits);
        if (n >= start_count) {
            n -= start_count, text_u8 += 16, length -= 16;
            continue;
        }
        return (sz_cptr_t)(text_u8 + sz_u32_nth_set_bit(start_bits, n));
    }
    return sz_utf8_seek_serial((sz_cptr_t)text_u8, length, n);
}
#if !SZ_IS_BIG_ENDIAN_
#pragma region Shared SIMD leaf substrate
/** @brief  Left-pack shuffle table: row `mask` holds the ascending byte indices (in [0,8)) of the set bits of an
 *          8-bit mask, padded with 0xFF. The VSX `vec_perm`-based stand-in for `vpcompressb` / BMI2 `pext`; two rows
 *          (low byte, high byte of a 16-bit lane mask) left-pack one 16-lane register's start offsets in two permutes
 *          stitched by `popcount(low8)`. 2 KB of `.rodata`, read straight (no per-call materialization). */
static sz_u8_t const sz_utf8_compress8_powervsx_[256][8] = {{255, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 255, 255, 255, 255, 255, 255, 255},
                                                            {1, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 255, 255, 255, 255, 255, 255},
                                                            {2, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 255, 255, 255, 255, 255, 255},
                                                            {1, 2, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 255, 255, 255, 255, 255},
                                                            {3, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 3, 255, 255, 255, 255, 255, 255},
                                                            {1, 3, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 3, 255, 255, 255, 255, 255},
                                                            {2, 3, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 3, 255, 255, 255, 255, 255},
                                                            {1, 2, 3, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 3, 255, 255, 255, 255},
                                                            {4, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 4, 255, 255, 255, 255, 255, 255},
                                                            {1, 4, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 4, 255, 255, 255, 255, 255},
                                                            {2, 4, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 4, 255, 255, 255, 255, 255},
                                                            {1, 2, 4, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 4, 255, 255, 255, 255},
                                                            {3, 4, 255, 255, 255, 255, 255, 255},
                                                            {0, 3, 4, 255, 255, 255, 255, 255},
                                                            {1, 3, 4, 255, 255, 255, 255, 255},
                                                            {0, 1, 3, 4, 255, 255, 255, 255},
                                                            {2, 3, 4, 255, 255, 255, 255, 255},
                                                            {0, 2, 3, 4, 255, 255, 255, 255},
                                                            {1, 2, 3, 4, 255, 255, 255, 255},
                                                            {0, 1, 2, 3, 4, 255, 255, 255},
                                                            {5, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 5, 255, 255, 255, 255, 255, 255},
                                                            {1, 5, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 5, 255, 255, 255, 255, 255},
                                                            {2, 5, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 5, 255, 255, 255, 255, 255},
                                                            {1, 2, 5, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 5, 255, 255, 255, 255},
                                                            {3, 5, 255, 255, 255, 255, 255, 255},
                                                            {0, 3, 5, 255, 255, 255, 255, 255},
                                                            {1, 3, 5, 255, 255, 255, 255, 255},
                                                            {0, 1, 3, 5, 255, 255, 255, 255},
                                                            {2, 3, 5, 255, 255, 255, 255, 255},
                                                            {0, 2, 3, 5, 255, 255, 255, 255},
                                                            {1, 2, 3, 5, 255, 255, 255, 255},
                                                            {0, 1, 2, 3, 5, 255, 255, 255},
                                                            {4, 5, 255, 255, 255, 255, 255, 255},
                                                            {0, 4, 5, 255, 255, 255, 255, 255},
                                                            {1, 4, 5, 255, 255, 255, 255, 255},
                                                            {0, 1, 4, 5, 255, 255, 255, 255},
                                                            {2, 4, 5, 255, 255, 255, 255, 255},
                                                            {0, 2, 4, 5, 255, 255, 255, 255},
                                                            {1, 2, 4, 5, 255, 255, 255, 255},
                                                            {0, 1, 2, 4, 5, 255, 255, 255},
                                                            {3, 4, 5, 255, 255, 255, 255, 255},
                                                            {0, 3, 4, 5, 255, 255, 255, 255},
                                                            {1, 3, 4, 5, 255, 255, 255, 255},
                                                            {0, 1, 3, 4, 5, 255, 255, 255},
                                                            {2, 3, 4, 5, 255, 255, 255, 255},
                                                            {0, 2, 3, 4, 5, 255, 255, 255},
                                                            {1, 2, 3, 4, 5, 255, 255, 255},
                                                            {0, 1, 2, 3, 4, 5, 255, 255},
                                                            {6, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 6, 255, 255, 255, 255, 255, 255},
                                                            {1, 6, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 6, 255, 255, 255, 255, 255},
                                                            {2, 6, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 6, 255, 255, 255, 255, 255},
                                                            {1, 2, 6, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 6, 255, 255, 255, 255},
                                                            {3, 6, 255, 255, 255, 255, 255, 255},
                                                            {0, 3, 6, 255, 255, 255, 255, 255},
                                                            {1, 3, 6, 255, 255, 255, 255, 255},
                                                            {0, 1, 3, 6, 255, 255, 255, 255},
                                                            {2, 3, 6, 255, 255, 255, 255, 255},
                                                            {0, 2, 3, 6, 255, 255, 255, 255},
                                                            {1, 2, 3, 6, 255, 255, 255, 255},
                                                            {0, 1, 2, 3, 6, 255, 255, 255},
                                                            {4, 6, 255, 255, 255, 255, 255, 255},
                                                            {0, 4, 6, 255, 255, 255, 255, 255},
                                                            {1, 4, 6, 255, 255, 255, 255, 255},
                                                            {0, 1, 4, 6, 255, 255, 255, 255},
                                                            {2, 4, 6, 255, 255, 255, 255, 255},
                                                            {0, 2, 4, 6, 255, 255, 255, 255},
                                                            {1, 2, 4, 6, 255, 255, 255, 255},
                                                            {0, 1, 2, 4, 6, 255, 255, 255},
                                                            {3, 4, 6, 255, 255, 255, 255, 255},
                                                            {0, 3, 4, 6, 255, 255, 255, 255},
                                                            {1, 3, 4, 6, 255, 255, 255, 255},
                                                            {0, 1, 3, 4, 6, 255, 255, 255},
                                                            {2, 3, 4, 6, 255, 255, 255, 255},
                                                            {0, 2, 3, 4, 6, 255, 255, 255},
                                                            {1, 2, 3, 4, 6, 255, 255, 255},
                                                            {0, 1, 2, 3, 4, 6, 255, 255},
                                                            {5, 6, 255, 255, 255, 255, 255, 255},
                                                            {0, 5, 6, 255, 255, 255, 255, 255},
                                                            {1, 5, 6, 255, 255, 255, 255, 255},
                                                            {0, 1, 5, 6, 255, 255, 255, 255},
                                                            {2, 5, 6, 255, 255, 255, 255, 255},
                                                            {0, 2, 5, 6, 255, 255, 255, 255},
                                                            {1, 2, 5, 6, 255, 255, 255, 255},
                                                            {0, 1, 2, 5, 6, 255, 255, 255},
                                                            {3, 5, 6, 255, 255, 255, 255, 255},
                                                            {0, 3, 5, 6, 255, 255, 255, 255},
                                                            {1, 3, 5, 6, 255, 255, 255, 255},
                                                            {0, 1, 3, 5, 6, 255, 255, 255},
                                                            {2, 3, 5, 6, 255, 255, 255, 255},
                                                            {0, 2, 3, 5, 6, 255, 255, 255},
                                                            {1, 2, 3, 5, 6, 255, 255, 255},
                                                            {0, 1, 2, 3, 5, 6, 255, 255},
                                                            {4, 5, 6, 255, 255, 255, 255, 255},
                                                            {0, 4, 5, 6, 255, 255, 255, 255},
                                                            {1, 4, 5, 6, 255, 255, 255, 255},
                                                            {0, 1, 4, 5, 6, 255, 255, 255},
                                                            {2, 4, 5, 6, 255, 255, 255, 255},
                                                            {0, 2, 4, 5, 6, 255, 255, 255},
                                                            {1, 2, 4, 5, 6, 255, 255, 255},
                                                            {0, 1, 2, 4, 5, 6, 255, 255},
                                                            {3, 4, 5, 6, 255, 255, 255, 255},
                                                            {0, 3, 4, 5, 6, 255, 255, 255},
                                                            {1, 3, 4, 5, 6, 255, 255, 255},
                                                            {0, 1, 3, 4, 5, 6, 255, 255},
                                                            {2, 3, 4, 5, 6, 255, 255, 255},
                                                            {0, 2, 3, 4, 5, 6, 255, 255},
                                                            {1, 2, 3, 4, 5, 6, 255, 255},
                                                            {0, 1, 2, 3, 4, 5, 6, 255},
                                                            {7, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 7, 255, 255, 255, 255, 255, 255},
                                                            {1, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 7, 255, 255, 255, 255, 255},
                                                            {2, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 7, 255, 255, 255, 255, 255},
                                                            {1, 2, 7, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 7, 255, 255, 255, 255},
                                                            {3, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 3, 7, 255, 255, 255, 255, 255},
                                                            {1, 3, 7, 255, 255, 255, 255, 255},
                                                            {0, 1, 3, 7, 255, 255, 255, 255},
                                                            {2, 3, 7, 255, 255, 255, 255, 255},
                                                            {0, 2, 3, 7, 255, 255, 255, 255},
                                                            {1, 2, 3, 7, 255, 255, 255, 255},
                                                            {0, 1, 2, 3, 7, 255, 255, 255},
                                                            {4, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 4, 7, 255, 255, 255, 255, 255},
                                                            {1, 4, 7, 255, 255, 255, 255, 255},
                                                            {0, 1, 4, 7, 255, 255, 255, 255},
                                                            {2, 4, 7, 255, 255, 255, 255, 255},
                                                            {0, 2, 4, 7, 255, 255, 255, 255},
                                                            {1, 2, 4, 7, 255, 255, 255, 255},
                                                            {0, 1, 2, 4, 7, 255, 255, 255},
                                                            {3, 4, 7, 255, 255, 255, 255, 255},
                                                            {0, 3, 4, 7, 255, 255, 255, 255},
                                                            {1, 3, 4, 7, 255, 255, 255, 255},
                                                            {0, 1, 3, 4, 7, 255, 255, 255},
                                                            {2, 3, 4, 7, 255, 255, 255, 255},
                                                            {0, 2, 3, 4, 7, 255, 255, 255},
                                                            {1, 2, 3, 4, 7, 255, 255, 255},
                                                            {0, 1, 2, 3, 4, 7, 255, 255},
                                                            {5, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 5, 7, 255, 255, 255, 255, 255},
                                                            {1, 5, 7, 255, 255, 255, 255, 255},
                                                            {0, 1, 5, 7, 255, 255, 255, 255},
                                                            {2, 5, 7, 255, 255, 255, 255, 255},
                                                            {0, 2, 5, 7, 255, 255, 255, 255},
                                                            {1, 2, 5, 7, 255, 255, 255, 255},
                                                            {0, 1, 2, 5, 7, 255, 255, 255},
                                                            {3, 5, 7, 255, 255, 255, 255, 255},
                                                            {0, 3, 5, 7, 255, 255, 255, 255},
                                                            {1, 3, 5, 7, 255, 255, 255, 255},
                                                            {0, 1, 3, 5, 7, 255, 255, 255},
                                                            {2, 3, 5, 7, 255, 255, 255, 255},
                                                            {0, 2, 3, 5, 7, 255, 255, 255},
                                                            {1, 2, 3, 5, 7, 255, 255, 255},
                                                            {0, 1, 2, 3, 5, 7, 255, 255},
                                                            {4, 5, 7, 255, 255, 255, 255, 255},
                                                            {0, 4, 5, 7, 255, 255, 255, 255},
                                                            {1, 4, 5, 7, 255, 255, 255, 255},
                                                            {0, 1, 4, 5, 7, 255, 255, 255},
                                                            {2, 4, 5, 7, 255, 255, 255, 255},
                                                            {0, 2, 4, 5, 7, 255, 255, 255},
                                                            {1, 2, 4, 5, 7, 255, 255, 255},
                                                            {0, 1, 2, 4, 5, 7, 255, 255},
                                                            {3, 4, 5, 7, 255, 255, 255, 255},
                                                            {0, 3, 4, 5, 7, 255, 255, 255},
                                                            {1, 3, 4, 5, 7, 255, 255, 255},
                                                            {0, 1, 3, 4, 5, 7, 255, 255},
                                                            {2, 3, 4, 5, 7, 255, 255, 255},
                                                            {0, 2, 3, 4, 5, 7, 255, 255},
                                                            {1, 2, 3, 4, 5, 7, 255, 255},
                                                            {0, 1, 2, 3, 4, 5, 7, 255},
                                                            {6, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 6, 7, 255, 255, 255, 255, 255},
                                                            {1, 6, 7, 255, 255, 255, 255, 255},
                                                            {0, 1, 6, 7, 255, 255, 255, 255},
                                                            {2, 6, 7, 255, 255, 255, 255, 255},
                                                            {0, 2, 6, 7, 255, 255, 255, 255},
                                                            {1, 2, 6, 7, 255, 255, 255, 255},
                                                            {0, 1, 2, 6, 7, 255, 255, 255},
                                                            {3, 6, 7, 255, 255, 255, 255, 255},
                                                            {0, 3, 6, 7, 255, 255, 255, 255},
                                                            {1, 3, 6, 7, 255, 255, 255, 255},
                                                            {0, 1, 3, 6, 7, 255, 255, 255},
                                                            {2, 3, 6, 7, 255, 255, 255, 255},
                                                            {0, 2, 3, 6, 7, 255, 255, 255},
                                                            {1, 2, 3, 6, 7, 255, 255, 255},
                                                            {0, 1, 2, 3, 6, 7, 255, 255},
                                                            {4, 6, 7, 255, 255, 255, 255, 255},
                                                            {0, 4, 6, 7, 255, 255, 255, 255},
                                                            {1, 4, 6, 7, 255, 255, 255, 255},
                                                            {0, 1, 4, 6, 7, 255, 255, 255},
                                                            {2, 4, 6, 7, 255, 255, 255, 255},
                                                            {0, 2, 4, 6, 7, 255, 255, 255},
                                                            {1, 2, 4, 6, 7, 255, 255, 255},
                                                            {0, 1, 2, 4, 6, 7, 255, 255},
                                                            {3, 4, 6, 7, 255, 255, 255, 255},
                                                            {0, 3, 4, 6, 7, 255, 255, 255},
                                                            {1, 3, 4, 6, 7, 255, 255, 255},
                                                            {0, 1, 3, 4, 6, 7, 255, 255},
                                                            {2, 3, 4, 6, 7, 255, 255, 255},
                                                            {0, 2, 3, 4, 6, 7, 255, 255},
                                                            {1, 2, 3, 4, 6, 7, 255, 255},
                                                            {0, 1, 2, 3, 4, 6, 7, 255},
                                                            {5, 6, 7, 255, 255, 255, 255, 255},
                                                            {0, 5, 6, 7, 255, 255, 255, 255},
                                                            {1, 5, 6, 7, 255, 255, 255, 255},
                                                            {0, 1, 5, 6, 7, 255, 255, 255},
                                                            {2, 5, 6, 7, 255, 255, 255, 255},
                                                            {0, 2, 5, 6, 7, 255, 255, 255},
                                                            {1, 2, 5, 6, 7, 255, 255, 255},
                                                            {0, 1, 2, 5, 6, 7, 255, 255},
                                                            {3, 5, 6, 7, 255, 255, 255, 255},
                                                            {0, 3, 5, 6, 7, 255, 255, 255},
                                                            {1, 3, 5, 6, 7, 255, 255, 255},
                                                            {0, 1, 3, 5, 6, 7, 255, 255},
                                                            {2, 3, 5, 6, 7, 255, 255, 255},
                                                            {0, 2, 3, 5, 6, 7, 255, 255},
                                                            {1, 2, 3, 5, 6, 7, 255, 255},
                                                            {0, 1, 2, 3, 5, 6, 7, 255},
                                                            {4, 5, 6, 7, 255, 255, 255, 255},
                                                            {0, 4, 5, 6, 7, 255, 255, 255},
                                                            {1, 4, 5, 6, 7, 255, 255, 255},
                                                            {0, 1, 4, 5, 6, 7, 255, 255},
                                                            {2, 4, 5, 6, 7, 255, 255, 255},
                                                            {0, 2, 4, 5, 6, 7, 255, 255},
                                                            {1, 2, 4, 5, 6, 7, 255, 255},
                                                            {0, 1, 2, 4, 5, 6, 7, 255},
                                                            {3, 4, 5, 6, 7, 255, 255, 255},
                                                            {0, 3, 4, 5, 6, 7, 255, 255},
                                                            {1, 3, 4, 5, 6, 7, 255, 255},
                                                            {0, 1, 3, 4, 5, 6, 7, 255},
                                                            {2, 3, 4, 5, 6, 7, 255, 255},
                                                            {0, 2, 3, 4, 5, 6, 7, 255},
                                                            {1, 2, 3, 4, 5, 6, 7, 255},
                                                            {0, 1, 2, 3, 4, 5, 6, 7}};
/** @brief  Reduce the 16 byte-lane booleans (0x00/0xFF) of @p boolean_lanes_u8x16 into a 16-bit lane mask
 *          (bit `i` <=> lane `i`, lowest-addressed byte → bit 0), the VSX `vpmovmskb` via `vec_vbpermq`. Mirror of
 *          @ref sz_utf8_iterate_movemask_powervsx_ but kept local to the decode path for clarity at the call sites. */
SZ_HELPER_INLINE sz_u32_t sz_utf8_movemask16_powervsx_(__vector unsigned char boolean_lanes_u8x16) {
    __vector unsigned char const gather_indices_u8x16 = {120, 112, 104, 96, 88, 80, 72, 64,
                                                         56,  48,  40,  32, 24, 16, 8,  0};
    __vector unsigned long long const gathered_mask_u64x2 = (__vector unsigned long long)vec_vbpermq(
        boolean_lanes_u8x16, gather_indices_u8x16);
    return (sz_u32_t)(gathered_mask_u64x2[1] & 0xFFFFull);
}
/** @brief  Combine the four per-register VSX movemasks of @p booleans_u8x16 into one 64-bit lane mask: register `r` →
 *          bits [16*r, 16*r+16). The VSX twin of the NEON `mask_combine` and the Ice Lake native `__mmask64`. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_mask_combine_powervsx_( 
    __vector unsigned char const *booleans_u8x16) {
    sz_u64_t mask = (sz_u64_t)sz_utf8_movemask16_powervsx_(booleans_u8x16[0]);
    mask |= (sz_u64_t)sz_utf8_movemask16_powervsx_(booleans_u8x16[1]) << 16;
    mask |= (sz_u64_t)sz_utf8_movemask16_powervsx_(booleans_u8x16[2]) << 32;
    mask |= (sz_u64_t)sz_utf8_movemask16_powervsx_(booleans_u8x16[3]) << 48;
    return mask;
}
/** @brief  Gather 16 bytes from the 64-byte window held in @p regs_u8x16 at the byte offsets @p index_u8x16 (lanes
 *          in [0,64)), gather-free. VSX `vec_perm` addresses only a 32-byte register pair, so the low half
 *          `(regs_u8x16[0],regs_u8x16[1])` and high half `(regs_u8x16[2],regs_u8x16[3])` are each permuted and the
 *          per-lane `index_u8x16 >= 32` predicate selects the half. Lanes whose offset runs past the window read
 *          the zero padding (the value is discarded downstream). */
SZ_HELPER_INLINE __vector unsigned char sz_utf8_gather16_powervsx_( 
    __vector unsigned char const *regs_u8x16, __vector unsigned char index_u8x16) {
    __vector unsigned char const half_select_threshold_u8x16 = vec_splats((unsigned char)32);
    __vector unsigned char const gathered_low_u8x16 = vec_perm(regs_u8x16[0], regs_u8x16[1], index_u8x16);
    __vector unsigned char const gathered_high_u8x16 = vec_perm(regs_u8x16[2], regs_u8x16[3],
                                                                vec_sub(index_u8x16, half_select_threshold_u8x16));
    __vector bool char const select_high_mask_u8x16 = vec_cmpge(index_u8x16, half_select_threshold_u8x16);
    return vec_sel(gathered_low_u8x16, gathered_high_u8x16, select_high_mask_u8x16);
}
/**
 *  @brief  Left-pack the emitted start byte-offsets of one 16-lane register @p emit16 into @p packed (ascending),
 *          appending after @p produced already-packed offsets and returning the new count. The VSX `vpcompressb`
 *          analogue: split @p emit16 into low8 / high8, `vec_perm` the byte-index identity vector by the matching
 *          2 KB shuffle-LUT rows, add the register base @p base, and stitch the two halves by `popcount(low8)`.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_compress_starts_powervsx_( 
    sz_u32_t emit16, int base, sz_u8_t *packed, sz_size_t produced) {
    sz_u32_t const low8 = emit16 & 0xFFu;
    sz_u32_t const high8 = (emit16 >> 8) & 0xFFu;
    __vector unsigned char const byte_iota_u8x16 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    __vector unsigned char const register_base_u8x16 = vec_splats((unsigned char)base);
    __vector unsigned char const eight_u8x16 = vec_splats((unsigned char)8);
    __vector unsigned char const low_byte_shuffle_lut_u8x16 = vec_xl(0, sz_utf8_compress8_powervsx_[low8]);
    __vector unsigned char const low_byte_packed_u8x16 = vec_add(
        vec_perm(byte_iota_u8x16, byte_iota_u8x16, low_byte_shuffle_lut_u8x16), register_base_u8x16);
    __vector unsigned char const high_byte_shuffle_lut_u8x16 = vec_xl(0, sz_utf8_compress8_powervsx_[high8]);
    __vector unsigned char const high_byte_packed_u8x16 = vec_add(
        vec_add(vec_perm(byte_iota_u8x16, byte_iota_u8x16, high_byte_shuffle_lut_u8x16), eight_u8x16),
        register_base_u8x16);
    sz_u128_vec_t low_bytes_vec, high_bytes_vec;
    low_bytes_vec.vsx_u8 = low_byte_packed_u8x16;
    high_bytes_vec.vsx_u8 = high_byte_packed_u8x16;
    int const low_count = sz_u32_popcount(low8);
    int const high_count = sz_u32_popcount(high8);
    for (int i = 0; i < low_count; ++i) packed[produced + (sz_size_t)i] = low_bytes_vec.u8s[i];
    for (int i = 0; i < high_count; ++i) packed[produced + (sz_size_t)low_count + (sz_size_t)i] = high_bytes_vec.u8s[i];
    return produced + (sz_size_t)low_count + (sz_size_t)high_count;
}
/**
 *  @brief  Decode the emitted start lanes @p emit_starts of a classified 64-byte window (held in the four
 *          @p regs_u8x16 16-byte registers) into sequential UTF-32 runes, the VSX sibling of
 *          @ref sz_utf8_rune_drain_icelake_. The start byte-offsets are
 *          left-packed by the shuffle-LUT, then a single 16-lane-block loop gathers the lead + up to three trailing
 *          bytes via @ref sz_utf8_gather16_powervsx_ and width-blends each codepoint; the wider 3rd/4th trailing
 *          bytes are gathered and blended CONDITIONALLY (@p has_three / @p has_four sibling `if`s). Every
 *          @p ill_formed start (its maximal ill-formed subpart) is overwritten with U+FFFD after the blend, and the
 *          resume cursor reads the last emitted start's offset + its compacted @p consumed_length (the maximal-subpart
 *          length), so an ill-formed trailing lane never skips bytes owed their own next U+FFFD.
 *  @return Number of runes emitted; sets @p consumed_bytes to the byte span they cover (the resume cursor delta).
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_powervsx_( 
    __vector unsigned char const *regs_u8x16, sz_u64_t emit_starts, sz_u64_t ill_formed, sz_u8_t const *consumed_length,
    int has_three, int has_four, sz_size_t emit_count, sz_rune_t *runes, sz_size_t capacity,
    sz_size_t *consumed_bytes) {
    sz_u512_vec_t packed_offsets_vec;
    sz_size_t packed = 0;
    for (int reg = 0; reg < 4; ++reg)
        packed = sz_utf8_compress_starts_powervsx_((sz_u32_t)((emit_starts >> (reg * 16)) & 0xFFFFu), reg * 16,
                                                   packed_offsets_vec.u8s, packed);
    sz_size_t const want = emit_count < capacity ? emit_count : capacity;
    sz_size_t produced = 0;
    __vector unsigned char const one_byte_u8x16 = vec_splats((unsigned char)1);
    __vector unsigned char const two_byte_u8x16 = vec_splats((unsigned char)2);
    __vector unsigned char const three_byte_u8x16 = vec_splats((unsigned char)3);
    for (sz_size_t block_start = 0; block_start < want; block_start += 16) {
        sz_u128_vec_t index_lanes_vec;
        for (int lane = 0; lane < 16; ++lane) {
            sz_size_t const source = block_start + (sz_size_t)lane;
            index_lanes_vec.u8s[lane] = source < want ? packed_offsets_vec.u8s[source] : (sz_u8_t)0xFF;
        }
        __vector unsigned char const gather_index_u8x16 = index_lanes_vec.vsx_u8;
        __vector unsigned char const lead_byte_u8x16 = sz_utf8_gather16_powervsx_(regs_u8x16, gather_index_u8x16);
        __vector unsigned char const continuation_byte_1_u8x16 = sz_utf8_gather16_powervsx_(
            regs_u8x16, vec_add(gather_index_u8x16, one_byte_u8x16));
        __vector unsigned char const continuation_byte_2_u8x16 =
            has_three ? sz_utf8_gather16_powervsx_(regs_u8x16, vec_add(gather_index_u8x16, two_byte_u8x16))
                      : vec_splats((unsigned char)0);
        __vector unsigned char const continuation_byte_3_u8x16 =
            has_four ? sz_utf8_gather16_powervsx_(regs_u8x16, vec_add(gather_index_u8x16, three_byte_u8x16))
                     : vec_splats((unsigned char)0);
        __vector unsigned char const selector_quarter_u8x16[4] = {
            {0x00, 0x10, 0x10, 0x10, 0x01, 0x10, 0x10, 0x10, 0x02, 0x10, 0x10, 0x10, 0x03, 0x10, 0x10, 0x10},
            {0x04, 0x10, 0x10, 0x10, 0x05, 0x10, 0x10, 0x10, 0x06, 0x10, 0x10, 0x10, 0x07, 0x10, 0x10, 0x10},
            {0x08, 0x10, 0x10, 0x10, 0x09, 0x10, 0x10, 0x10, 0x0a, 0x10, 0x10, 0x10, 0x0b, 0x10, 0x10, 0x10},
            {0x0c, 0x10, 0x10, 0x10, 0x0d, 0x10, 0x10, 0x10, 0x0e, 0x10, 0x10, 0x10, 0x0f, 0x10, 0x10, 0x10}};
        __vector unsigned char const zero_u8x16 = vec_splats((unsigned char)0);
        for (int quarter = 0; quarter < 4; ++quarter) {
            sz_size_t const out_base = block_start + (sz_size_t)quarter * 4;
            if (out_base >= want) break;
            __vector unsigned char const selector_u8x16 = selector_quarter_u8x16[quarter];
            __vector unsigned int const lead_u32x4 = (__vector unsigned int)vec_perm(lead_byte_u8x16, zero_u8x16,
                                                                                     selector_u8x16);
            __vector unsigned int const continuation_byte_1_u32x4 = (__vector unsigned int)vec_perm(
                continuation_byte_1_u8x16, zero_u8x16, selector_u8x16);
            __vector unsigned int const continuation_byte_2_u32x4 = (__vector unsigned int)vec_perm(
                continuation_byte_2_u8x16, zero_u8x16, selector_u8x16);
            __vector unsigned int const continuation_byte_3_u32x4 = (__vector unsigned int)vec_perm(
                continuation_byte_3_u8x16, zero_u8x16, selector_u8x16);
            __vector unsigned int const mask_word_3f_u32x4 = vec_splats((unsigned int)0x3F);
            __vector unsigned int const mask_word_c0_u32x4 = vec_splats((unsigned int)0xC0);
            __vector unsigned int const mask_word_e0_u32x4 = vec_splats((unsigned int)0xE0);
            __vector unsigned int const mask_word_f0_u32x4 = vec_splats((unsigned int)0xF0);
            __vector bool int const two_byte_mask_u32x4 = vec_and(vec_cmpge(lead_u32x4, mask_word_c0_u32x4),
                                                                  vec_cmplt(lead_u32x4, mask_word_e0_u32x4));
            __vector unsigned int const two_byte_codepoint_u32x4 = vec_or(
                vec_sl(vec_and(lead_u32x4, vec_splats((unsigned int)0x1F)), vec_splats((unsigned int)6)),
                vec_and(continuation_byte_1_u32x4, mask_word_3f_u32x4));
            __vector unsigned int decoded_codepoint_u32x4 = vec_sel(lead_u32x4, two_byte_codepoint_u32x4,
                                                                    two_byte_mask_u32x4);
            if (has_three) {
                __vector bool int const three_byte_mask_u32x4 = vec_and(vec_cmpge(lead_u32x4, mask_word_e0_u32x4),
                                                                        vec_cmplt(lead_u32x4, mask_word_f0_u32x4));
                __vector unsigned int const three_byte_codepoint_u32x4 = vec_or(
                    vec_or(vec_sl(vec_and(lead_u32x4, vec_splats((unsigned int)0x0F)), vec_splats((unsigned int)12)),
                           vec_sl(vec_and(continuation_byte_1_u32x4, mask_word_3f_u32x4), vec_splats((unsigned int)6))),
                    vec_and(continuation_byte_2_u32x4, mask_word_3f_u32x4));
                decoded_codepoint_u32x4 = vec_sel(decoded_codepoint_u32x4, three_byte_codepoint_u32x4,
                                                  three_byte_mask_u32x4);
            }
            if (has_four) { 
                __vector bool int const four_byte_mask_u32x4 = vec_cmpge(lead_u32x4, mask_word_f0_u32x4);
                __vector unsigned int const four_byte_codepoint_u32x4 = vec_or(
                    vec_or(
                        vec_sl(vec_and(lead_u32x4, vec_splats((unsigned int)0x07)), vec_splats((unsigned int)18)),
                        vec_sl(vec_and(continuation_byte_1_u32x4, mask_word_3f_u32x4), vec_splats((unsigned int)12))),
                    vec_or(vec_sl(vec_and(continuation_byte_2_u32x4, mask_word_3f_u32x4), vec_splats((unsigned int)6)),
                           vec_and(continuation_byte_3_u32x4, mask_word_3f_u32x4)));
                decoded_codepoint_u32x4 = vec_sel(decoded_codepoint_u32x4, four_byte_codepoint_u32x4,
                                                  four_byte_mask_u32x4);
            }
            sz_u128_vec_t lane_values_vec;
            lane_values_vec.vsx_u32 = decoded_codepoint_u32x4;
            sz_size_t lanes = want - out_base;
            if (lanes > 4) lanes = 4;
            for (sz_size_t lane = 0; lane < lanes; ++lane) {
                sz_size_t const flat = out_base + lane;
                int const offset = packed_offsets_vec.u8s[flat];
                sz_rune_t value = (sz_rune_t)lane_values_vec.u32s[lane];
                if ((ill_formed >> offset) & 1) value = (sz_rune_t)sz_rune_replacement_k;
                runes[flat] = value;
            }
            produced += lanes;
        }
    }
    int const last_offset = packed_offsets_vec.u8s[produced - 1];
    *consumed_bytes = (sz_size_t)last_offset + (sz_size_t)consumed_length[last_offset];
    return produced;
}
#pragma endregion Shared SIMD leaf substrate
/**
 *  @brief  Decode one window of @p text into dense UTF-32 @p runes by the uniform "ASCII fast lane → classify →
 *          per-lane well-formed + orphan promotion → compress emitted starts → gather → width-blend → blend
 *          U+FFFD" path, the VSX sibling of @ref sz_utf8_decode_once_icelake_. The decode is TOTAL: clean and
 *          dirty bytes are handled in-vector, one U+FFFD per maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C),
 *          bit-exact with @ref sz_utf8_decode_serial. The step declines (`*runes_unpacked == 0`, cursor
 *          unchanged) ONLY when the first lead's declared sequence crosses the window edge (a boundary truncation),
 *          which the public entry finalizes without a serial re-decode.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_decode_once_powervsx_( 
    sz_cptr_t text, sz_size_t length,                   
    sz_rune_t *runes, sz_size_t runes_capacity,         
    sz_size_t *runes_unpacked) {
    sz_size_t const chunk = length < 64 ? length : 64;
    sz_u512_vec_t staging_vec;
    for (sz_size_t index = 0; index < 64; ++index) staging_vec.u8s[index] = index < chunk ? (sz_u8_t)text[index] : 0;
    __vector unsigned char input_register_u8x16[4];
    input_register_u8x16[0] = vec_xl(0, staging_vec.u8s + 0);
    input_register_u8x16[1] = vec_xl(0, staging_vec.u8s + 16);
    input_register_u8x16[2] = vec_xl(0, staging_vec.u8s + 32);
    input_register_u8x16[3] = vec_xl(0, staging_vec.u8s + 48);
    __vector unsigned char const high_bit_mask_u8x16 = vec_splats((unsigned char)0x80);
    __vector unsigned char nonascii_mask_u8x16[4];
    for (int reg = 0; reg < 4; ++reg)
        nonascii_mask_u8x16[reg] = (__vector unsigned char)vec_cmpge(input_register_u8x16[reg], high_bit_mask_u8x16);
    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(chunk);
    sz_u64_t const nonascii = sz_utf8_mask_combine_powervsx_(nonascii_mask_u8x16) & loaded_mask;
    if (nonascii == 0) {
        sz_size_t const want = chunk < runes_capacity ? chunk : runes_capacity;
        for (sz_size_t index = 0; index < want; ++index) runes[index] = (sz_rune_t)staging_vec.u8s[index];
        *runes_unpacked = want;
        return text + want;
    }
    {
        __vector unsigned char four_or_mask_u8x16[4];
        __vector unsigned char const byte_f0_gate_u8x16 = vec_splats((unsigned char)0xF0);
        for (int reg = 0; reg < 4; ++reg)
            four_or_mask_u8x16[reg] = (__vector unsigned char)vec_cmpge(input_register_u8x16[reg], byte_f0_gate_u8x16);
        sz_u64_t const four_or_bits = sz_utf8_mask_combine_powervsx_(four_or_mask_u8x16) & loaded_mask;
        if (four_or_bits == 0) {
            __vector unsigned char continuation_clean_u8x16[4], ascii_clean_u8x16[4], lead2_clean_u8x16[4],
                lead3_clean_u8x16[4];
            __vector unsigned char const continuation_mask_u8x16 = vec_splats((unsigned char)0xC0);
            __vector unsigned char const continuation_pattern_u8x16 = vec_splats((unsigned char)0x80);
            __vector unsigned char const byte_c2_u8x16 = vec_splats((unsigned char)0xC2);
            __vector unsigned char const byte_e0_u8x16 = vec_splats((unsigned char)0xE0);
            __vector unsigned char const byte_f0_u8x16 = vec_splats((unsigned char)0xF0);
            for (int reg = 0; reg < 4; ++reg) {
                __vector unsigned char const register_value_u8x16 = input_register_u8x16[reg];
                continuation_clean_u8x16[reg] = (__vector unsigned char)vec_cmpeq(
                    vec_and(register_value_u8x16, continuation_mask_u8x16), continuation_pattern_u8x16);
                ascii_clean_u8x16[reg] = (__vector unsigned char)vec_cmplt(register_value_u8x16,
                                                                           continuation_pattern_u8x16);
                lead2_clean_u8x16[reg] = vec_and(
                    (__vector unsigned char)vec_cmpge(register_value_u8x16, byte_c2_u8x16),
                    (__vector unsigned char)vec_cmplt(register_value_u8x16, byte_e0_u8x16));
                lead3_clean_u8x16[reg] = (__vector unsigned char)vec_cmpeq(vec_and(register_value_u8x16, byte_f0_u8x16),
                                                                           byte_e0_u8x16);
            }
            sz_u64_t const continuation_clean_bits = sz_utf8_mask_combine_powervsx_(continuation_clean_u8x16) &
                                                     loaded_mask;
            sz_u64_t const ascii_clean_bits = sz_utf8_mask_combine_powervsx_(ascii_clean_u8x16) & loaded_mask;
            sz_u64_t const lead2_clean_bits = sz_utf8_mask_combine_powervsx_(lead2_clean_u8x16) & loaded_mask;
            sz_u64_t const lead3_clean_bits = sz_utf8_mask_combine_powervsx_(lead3_clean_u8x16) & loaded_mask;
            sz_u64_t const starts_clean_bits = loaded_mask & ~continuation_clean_bits;
            sz_u64_t const owe1_clean_bits = lead2_clean_bits | lead3_clean_bits; 
            sz_size_t decodable_end_clean = chunk;
            if (chunk && ((owe1_clean_bits >> (chunk - 1)) & 1u)) decodable_end_clean = chunk - 1;
            else if (chunk >= 2 && ((lead3_clean_bits >> (chunk - 2)) & 1u)) decodable_end_clean = chunk - 2;
            sz_u64_t const decodable_clean = sz_u64_mask_until_serial_(decodable_end_clean);
            sz_u64_t const starts_decodable_clean = starts_clean_bits & decodable_clean;
            sz_u64_t const continuation_decodable_clean = continuation_clean_bits & decodable_clean;
            sz_u64_t const owe1_decodable_clean = owe1_clean_bits & decodable_clean;
            sz_u64_t const lead3_decodable_clean = lead3_clean_bits & decodable_clean;
            sz_u64_t const covered_clean_bits = (owe1_clean_bits << 1) | (lead3_clean_bits << 2); 
            int const has_three_clean = lead3_clean_bits != 0;
            sz_u64_t bounds_bad_clean = 0;
            if (has_three_clean) {
                __vector unsigned char e0_mask_u8x16[4], ed_mask_u8x16[4], n1_lt_a0_mask_u8x16[4],
                    n1_ge_a0_mask_u8x16[4];
                __vector unsigned char const byte_ed_u8x16 = vec_splats((unsigned char)0xED);
                __vector unsigned char const byte_a0_u8x16 = vec_splats((unsigned char)0xA0);
                for (int reg = 0; reg < 4; ++reg) {
                    __vector unsigned char const register_value_u8x16 = input_register_u8x16[reg];
                    e0_mask_u8x16[reg] = (__vector unsigned char)vec_cmpeq(register_value_u8x16, byte_e0_u8x16);
                    ed_mask_u8x16[reg] = (__vector unsigned char)vec_cmpeq(register_value_u8x16, byte_ed_u8x16);
                    n1_lt_a0_mask_u8x16[reg] = (__vector unsigned char)vec_cmplt(register_value_u8x16, byte_a0_u8x16);
                    n1_ge_a0_mask_u8x16[reg] = (__vector unsigned char)vec_cmpge(register_value_u8x16, byte_a0_u8x16);
                }
                sz_u64_t const e0_bits = sz_utf8_mask_combine_powervsx_(e0_mask_u8x16);
                sz_u64_t const ed_bits = sz_utf8_mask_combine_powervsx_(ed_mask_u8x16);
                sz_u64_t const n1_lt_a0_bits = sz_utf8_mask_combine_powervsx_(n1_lt_a0_mask_u8x16);
                sz_u64_t const n1_ge_a0_bits = sz_utf8_mask_combine_powervsx_(n1_ge_a0_mask_u8x16);
                bounds_bad_clean = ((e0_bits & (n1_lt_a0_bits >> 1)) | (ed_bits & (n1_ge_a0_bits >> 1))) &
                                   decodable_clean;
            }
            int const clean_window =
                ((starts_decodable_clean & ~(ascii_clean_bits | lead2_clean_bits | lead3_clean_bits)) == 0) &&
                ((owe1_decodable_clean & ~(continuation_clean_bits >> 1)) == 0) &&
                ((lead3_decodable_clean & ~(continuation_clean_bits >> 2)) == 0) &&
                ((continuation_decodable_clean & ~covered_clean_bits) == 0) && (bounds_bad_clean == 0);
            if (clean_window && starts_decodable_clean) {
                sz_u512_vec_t consumed_length_clean_vec;
                for (int lane = 0; lane < 64; ++lane) {
                    sz_u8_t length_here = 1;
                    length_here = (sz_u8_t)(length_here + (sz_u8_t)((lead2_clean_bits >> lane) & 1));
                    length_here = (sz_u8_t)(length_here + (sz_u8_t)(((lead3_clean_bits >> lane) & 1) << 1));
                    consumed_length_clean_vec.u8s[lane] = length_here;
                }
                __vector unsigned long long const starts_word_u64x2 = {starts_decodable_clean, 0ull};
                sz_size_t const emit_count_clean = (sz_size_t)vec_extract(vec_popcnt(starts_word_u64x2), 0);
                sz_size_t consumed_clean = 0;
                sz_size_t const produced_clean = sz_utf8_rune_drain_powervsx_(
                    input_register_u8x16, starts_decodable_clean, 0, consumed_length_clean_vec.u8s, has_three_clean, 0,
                    emit_count_clean, runes, runes_capacity, &consumed_clean);
                *runes_unpacked = produced_clean;
                return text + consumed_clean;
            }
        }
    }
    __vector unsigned char const length_lookup_table_u8x16 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    __vector unsigned char const continuation_mask_u8x16 = vec_splats((unsigned char)0xC0);
    __vector unsigned char const continuation_pattern_u8x16 = vec_splats((unsigned char)0x80);
    __vector unsigned char const low_nibble_mask_u8x16 = vec_splats((unsigned char)0x0F);
    __vector unsigned char continuation_mask_per_register_u8x16[4], length_2_mask_u8x16[4], length_3_mask_u8x16[4],
        length_4_mask_u8x16[4];
    __vector unsigned char sequence_length_per_register_u8x16[4];
    for (int reg = 0; reg < 4; ++reg) {
        __vector unsigned char const register_value_u8x16 = input_register_u8x16[reg];
        continuation_mask_per_register_u8x16[reg] = (__vector unsigned char)vec_cmpeq(
            vec_and(register_value_u8x16, continuation_mask_u8x16), continuation_pattern_u8x16);
        __vector unsigned char const high_nibble_u8x16 = vec_and(
            vec_sr(register_value_u8x16, vec_splats((unsigned char)4)), low_nibble_mask_u8x16);
        __vector unsigned char const sequence_length_u8x16 = vec_perm(length_lookup_table_u8x16,
                                                                      length_lookup_table_u8x16, high_nibble_u8x16);
        sequence_length_per_register_u8x16[reg] = sequence_length_u8x16;
        length_2_mask_u8x16[reg] = (__vector unsigned char)vec_cmpeq(sequence_length_u8x16,
                                                                     vec_splats((unsigned char)2));
        length_3_mask_u8x16[reg] = (__vector unsigned char)vec_cmpeq(sequence_length_u8x16,
                                                                     vec_splats((unsigned char)3));
        length_4_mask_u8x16[reg] = (__vector unsigned char)vec_cmpeq(sequence_length_u8x16,
                                                                     vec_splats((unsigned char)4));
    }
    sz_u64_t const continuation_bits = sz_utf8_mask_combine_powervsx_(continuation_mask_per_register_u8x16) &
                                       loaded_mask;
    sz_u64_t const starts_bits = loaded_mask & ~continuation_bits;
    sz_u64_t const length_ge_two = sz_utf8_mask_combine_powervsx_(length_2_mask_u8x16) |
                                   sz_utf8_mask_combine_powervsx_(length_3_mask_u8x16) |
                                   sz_utf8_mask_combine_powervsx_(length_4_mask_u8x16);
    sz_u64_t const length_ge_three = sz_utf8_mask_combine_powervsx_(length_3_mask_u8x16) |
                                     sz_utf8_mask_combine_powervsx_(length_4_mask_u8x16);
    sz_u64_t const length_ge_four = sz_utf8_mask_combine_powervsx_(length_4_mask_u8x16);
    sz_u64_t const length_ge_two_starts = length_ge_two & starts_bits;
    sz_u64_t const length_ge_three_starts = length_ge_three & starts_bits;
    sz_u64_t const length_ge_four_starts = length_ge_four & starts_bits;
    __vector unsigned char const byte_iota_u8x16 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    __vector unsigned char const chunk_size_u8x16 = vec_splats((unsigned char)chunk);
    __vector unsigned char overrun_mask_u8x16[4];
    for (int reg = 0; reg < 4; ++reg) {
        __vector unsigned char const global_lane_offset_u8x16 = vec_add(byte_iota_u8x16,
                                                                        vec_splats((unsigned char)(reg * 16)));
        __vector unsigned char const sequence_end_offset_u8x16 = vec_add(global_lane_offset_u8x16,
                                                                         sequence_length_per_register_u8x16[reg]);
        overrun_mask_u8x16[reg] = (__vector unsigned char)vec_cmpgt(sequence_end_offset_u8x16, chunk_size_u8x16);
    }
    sz_u64_t const overruns = sz_utf8_mask_combine_powervsx_(overrun_mask_u8x16) & starts_bits;
    sz_size_t const decodable_end = overruns ? (sz_size_t)sz_u64_ctz(overruns) : chunk;
    sz_u64_t const decodable_mask = sz_u64_mask_until_serial_(decodable_end);
    __vector unsigned char const successor_iota_u8x16 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    sz_u64_t bad_lead_bits = 0, overlong_or_surrogate_or_range_bits = 0;
    int const has_three = (length_ge_three_starts) != 0;
    int const has_four = (length_ge_four_starts) != 0;
    {
        __vector unsigned char bad_lead_mask_u8x16[4];
        __vector unsigned char overlong_surrogate_range_mask_u8x16[4];
        __vector unsigned char const byte_c2_u8x16 = vec_splats((unsigned char)0xC2);
        __vector unsigned char const byte_f4_u8x16 = vec_splats((unsigned char)0xF4);
        __vector unsigned char const byte_a0_u8x16 = vec_splats((unsigned char)0xA0);
        __vector unsigned char const byte_90_u8x16 = vec_splats((unsigned char)0x90);
        __vector unsigned char const byte_e0_u8x16 = vec_splats((unsigned char)0xE0);
        __vector unsigned char const byte_ed_u8x16 = vec_splats((unsigned char)0xED);
        __vector unsigned char const byte_f0_u8x16 = vec_splats((unsigned char)0xF0);
        for (int reg = 0; reg < 4; ++reg) {
            __vector unsigned char const register_value_u8x16 = input_register_u8x16[reg];
            __vector unsigned char const successor_register_u8x16 = (reg < 3) ? input_register_u8x16[reg + 1]
                                                                              : input_register_u8x16[3];
            __vector unsigned char const next_byte_u8x16 = vec_perm(register_value_u8x16, successor_register_u8x16,
                                                                    successor_iota_u8x16);
            __vector bool char const length_2_mask_u8x16 = vec_cmpeq(sequence_length_per_register_u8x16[reg],
                                                                     vec_splats((unsigned char)2));
            __vector bool char const length_4_mask_u8x16 = vec_cmpeq(sequence_length_per_register_u8x16[reg],
                                                                     vec_splats((unsigned char)4));
            __vector unsigned char const lead_lt_c2_mask_u8x16 = (__vector unsigned char)vec_cmplt(register_value_u8x16,
                                                                                                   byte_c2_u8x16);
            __vector unsigned char const lead_gt_f4_mask_u8x16 = (__vector unsigned char)vec_cmpgt(register_value_u8x16,
                                                                                                   byte_f4_u8x16);
            bad_lead_mask_u8x16[reg] = vec_or(
                vec_and((__vector unsigned char)length_2_mask_u8x16, lead_lt_c2_mask_u8x16),
                vec_and((__vector unsigned char)length_4_mask_u8x16, lead_gt_f4_mask_u8x16));
            __vector unsigned char const e0_bad_mask_u8x16 = vec_and(
                (__vector unsigned char)vec_cmpeq(register_value_u8x16, byte_e0_u8x16),
                (__vector unsigned char)vec_cmplt(next_byte_u8x16, byte_a0_u8x16));
            __vector unsigned char const ed_bad_mask_u8x16 = vec_and(
                (__vector unsigned char)vec_cmpeq(register_value_u8x16, byte_ed_u8x16),
                (__vector unsigned char)vec_cmpge(next_byte_u8x16, byte_a0_u8x16));
            __vector unsigned char const f0_bad_mask_u8x16 = vec_and(
                (__vector unsigned char)vec_cmpeq(register_value_u8x16, byte_f0_u8x16),
                (__vector unsigned char)vec_cmplt(next_byte_u8x16, byte_90_u8x16));
            __vector unsigned char const f4_bad_mask_u8x16 = vec_and(
                (__vector unsigned char)vec_cmpeq(register_value_u8x16, byte_f4_u8x16),
                (__vector unsigned char)vec_cmpge(next_byte_u8x16, byte_90_u8x16));
            overlong_surrogate_range_mask_u8x16[reg] = vec_or(vec_or(e0_bad_mask_u8x16, ed_bad_mask_u8x16),
                                                              vec_or(f0_bad_mask_u8x16, f4_bad_mask_u8x16));
        }
        bad_lead_bits = sz_utf8_mask_combine_powervsx_(bad_lead_mask_u8x16) & starts_bits;
        overlong_or_surrogate_or_range_bits = sz_utf8_mask_combine_powervsx_(overlong_surrogate_range_mask_u8x16) &
                                              starts_bits;
    }
    sz_u64_t const cont1 = starts_bits & (continuation_bits >> 1);
    sz_u64_t const cont2 = starts_bits & (continuation_bits >> 2);
    sz_u64_t const cont3 = starts_bits & (continuation_bits >> 3);
    sz_u64_t const b1_range_ok = starts_bits & ~overlong_or_surrogate_or_range_bits;
    sz_u64_t const first_ok = cont1 & b1_range_ok & ~bad_lead_bits;
    sz_u64_t const length_one_bits = starts_bits & ~length_ge_two_starts;
    sz_u64_t const length_two_bits = length_ge_two_starts & ~length_ge_three_starts;
    sz_u64_t const wf1 = length_one_bits;
    sz_u64_t const wf2 = length_two_bits & ~bad_lead_bits & cont1;
    sz_u64_t const wf3 = (length_ge_three_starts & ~length_ge_four_starts) & b1_range_ok & cont1 & cont2;
    sz_u64_t const wf4 = length_ge_four_starts & ~bad_lead_bits & b1_range_ok & cont1 & cont2 & cont3;
    sz_u64_t const well_formed = (wf1 | wf2 | wf3 | wf4) & decodable_mask;
    sz_u64_t const step2 = length_ge_two_starts & first_ok;
    sz_u64_t const step3 = step2 & length_ge_three_starts & cont2;
    sz_u64_t const step4 = step3 & length_ge_four_starts & cont3;
    sz_u64_t const covered = ((step2 & decodable_mask) << 1) | ((step3 & decodable_mask) << 2) |
                             ((step4 & decodable_mask) << 3);
    sz_u64_t const orphan = continuation_bits & decodable_mask & ~covered;
    sz_u64_t const emit_starts = (starts_bits | orphan) & decodable_mask;
    sz_size_t const emit_count = (sz_size_t)sz_u64_popcount(emit_starts);
    if (emit_count == 0) { return *runes_unpacked = 0, text; } 
    sz_u64_t const ill_formed = emit_starts & ~well_formed;
    sz_u512_vec_t consumed_length_vec;
    for (int lane = 0; lane < 64; ++lane) {
        sz_u8_t length_here = 1;
        length_here = (sz_u8_t)(length_here + (sz_u8_t)((step2 >> lane) & 1));
        length_here = (sz_u8_t)(length_here + (sz_u8_t)((step3 >> lane) & 1));
        length_here = (sz_u8_t)(length_here + (sz_u8_t)((step4 >> lane) & 1));
        consumed_length_vec.u8s[lane] = length_here;
    }
    sz_size_t consumed = 0;
    sz_size_t const produced = sz_utf8_rune_drain_powervsx_(input_register_u8x16, emit_starts, ill_formed,
                                                            consumed_length_vec.u8s, has_three, has_four, emit_count,
                                                            runes, runes_capacity, &consumed);
    *runes_unpacked = produced;
    return text + consumed;
}
#pragma region Windowed engine leaf substrate
/** @brief  The decoded 64-byte window for the POWER VSX backend, mirroring @ref sz_utf8_rune_window_neon_t. The
 *          64 bytes live as four `__vector unsigned char` quarters (`window[0]` is lanes [0, 16), ...); the
 *          per-lane byte-domain codepoint halves `high`/`low` share that shape. Masks are `sz_u64_t` (one bit per
 *          byte-lane, quarter `q` at bit positions [16*q, 16*q+16)). */
typedef struct sz_utf8_rune_window_powervsx_t {
    __vector unsigned char window[4]; /**< Raw input bytes for lanes [16*q, 16*q+16). */
    __vector unsigned char high[4];   /**< Per-lane `codepoint >> 8`. */
    __vector unsigned char low[4];    /**< Per-lane `codepoint & 0xFF`. */
    sz_u64_t continuation;            /**< Bit `i` is set when lane `i` is a continuation byte `10xxxxxx`. */
    sz_u64_t codepoint_starts;  /**< Bit `i` is set when lane `i` begins a codepoint (loaded, non-continuation). */
    sz_u64_t two_byte_starts;   /**< Bit `i` is set when lane `i` is a 2-byte lead `110xxxxx`. */
    sz_u64_t three_byte_starts; /**< Bit `i` is set when lane `i` is a 3-byte lead `1110xxxx`. */
    sz_u64_t four_byte_starts;  /**< Bit `i` is set when lane `i` is a 4-byte lead `11110xxx`. */
    sz_size_t loaded;           /**< Number of bytes actually loaded (at most 64). */
} sz_utf8_rune_window_powervsx_t;
/** @brief  Per-byte logical right shift by @p shift keeping the low @p keep bits, the VSX twin of `srl8_`; VSX
 *          `vec_sr` takes a per-lane vector shift count, so no immediate switch is needed. */
SZ_HELPER_INLINE __vector unsigned char sz_utf8_srl8_powervsx_(__vector unsigned char value_u8x16, unsigned char shift,
                                                               unsigned char keep) {
    return vec_and(vec_sr(value_u8x16, vec_splats(shift)), vec_splats(keep));
}
/** @brief  Masked 64-byte load into four quarters; bytes [loaded, 64) read as zero, staged through a zeroed
 *          `sz_u512_vec_t` byte buffer so no read runs past `text + loaded` (the sanctioned union tail idiom).
 *          Mirrors @ref sz_utf8_load_window_neon_. */
SZ_HELPER_AUTO void sz_utf8_load_window_powervsx_(sz_u8_t const *text, sz_size_t loaded,
                                                  __vector unsigned char *out_u8x16) {
    if (loaded >= 64) {
        out_u8x16[0] = vec_xl(0, text + 0);
        out_u8x16[1] = vec_xl(0, text + 16);
        out_u8x16[2] = vec_xl(0, text + 32);
        out_u8x16[3] = vec_xl(0, text + 48);
        return;
    }
    sz_u512_vec_t window_vec;
    for (sz_size_t index = 0; index < 64; ++index) window_vec.u8s[index] = 0;
    for (sz_size_t index = 0; index < loaded; ++index) window_vec.u8s[index] = text[index];
    out_u8x16[0] = vec_xl(0, window_vec.u8s + 0);
    out_u8x16[1] = vec_xl(0, window_vec.u8s + 16);
    out_u8x16[2] = vec_xl(0, window_vec.u8s + 32);
    out_u8x16[3] = vec_xl(0, window_vec.u8s + 48);
}
/** @brief  Forward neighbours `next1[i] = window[i+1]`, `next2[i] = window[i+2]`, `next3[i] = window[i+3]` over all
 *          64 lanes, wrapping modulo 64 to match the NEON `vextq` quarters and the Ice Lake `vpermb`. Each quarter
 *          `r` is stitched to its successor `(r+1) & 3` by one `vec_perm` per distance over the register pair
 *          `{window[r], window[(r+1)&3]}`, indexed by `{k, k+1, ..., k+15}` (index `>= 16` reads the successor).
 *          Requires the zero-padded window so wrapped lanes past `loaded` are deterministic zeros. */
SZ_HELPER_AUTO void sz_utf8_forward_neighbours_powervsx_( 
    __vector unsigned char const *window_u8x16, __vector unsigned char *next1_u8x16,
    __vector unsigned char *next2_u8x16, __vector unsigned char *next3_u8x16) {
    __vector unsigned char const index_next1_u8x16 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    __vector unsigned char const index_next2_u8x16 = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    __vector unsigned char const index_next3_u8x16 = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    for (int quarter = 0; quarter < 4; ++quarter) {
        __vector unsigned char const here_u8x16 = window_u8x16[quarter];
        __vector unsigned char const successor_u8x16 = window_u8x16[(quarter + 1) & 3];
        next1_u8x16[quarter] = vec_perm(here_u8x16, successor_u8x16, index_next1_u8x16);
        next2_u8x16[quarter] = vec_perm(here_u8x16, successor_u8x16, index_next2_u8x16);
        next3_u8x16[quarter] = vec_perm(here_u8x16, successor_u8x16, index_next3_u8x16);
    }
}
/** @brief  Load up to 64 bytes (masked tail) and decode every lane into byte-domain halves, the VSX twin of
 *          @ref sz_utf8_rune_decode_window_neon_, bit-identical on every lane. */
SZ_HELPER_AUTO sz_utf8_rune_window_powervsx_t sz_utf8_rune_decode_window_powervsx_( 
    sz_u8_t const *text, sz_size_t available) {
    sz_utf8_rune_window_powervsx_t result;
    result.loaded = available < 64 ? available : 64;
    __vector unsigned char window_u8x16[4];
    sz_utf8_load_window_powervsx_(text, result.loaded, window_u8x16);
    for (int quarter = 0; quarter < 4; ++quarter) result.window[quarter] = window_u8x16[quarter];
    __vector unsigned char next1_u8x16[4], next2_u8x16[4], next3_u8x16[4];
    sz_utf8_forward_neighbours_powervsx_(window_u8x16, next1_u8x16, next2_u8x16, next3_u8x16);
    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(result.loaded);
    __vector unsigned char continuation_bool_u8x16[4], two_byte_bool_u8x16[4], three_byte_bool_u8x16[4],
        four_byte_bool_u8x16[4];
    __vector unsigned char const mask_continuation_u8x16 = vec_splats((unsigned char)0xC0),
                                 pattern_continuation_u8x16 = vec_splats((unsigned char)0x80);
    __vector unsigned char const mask_two_u8x16 = vec_splats((unsigned char)0xE0),
                                 pattern_two_u8x16 = vec_splats((unsigned char)0xC0);
    __vector unsigned char const mask_three_u8x16 = vec_splats((unsigned char)0xF0),
                                 pattern_three_u8x16 = vec_splats((unsigned char)0xE0);
    __vector unsigned char const mask_four_u8x16 = vec_splats((unsigned char)0xF8),
                                 pattern_four_u8x16 = vec_splats((unsigned char)0xF0);
    for (int quarter = 0; quarter < 4; ++quarter) {
        __vector unsigned char const here_u8x16 = window_u8x16[quarter];
        continuation_bool_u8x16[quarter] = (__vector unsigned char)vec_cmpeq(
            vec_and(here_u8x16, mask_continuation_u8x16), pattern_continuation_u8x16);
        two_byte_bool_u8x16[quarter] = (__vector unsigned char)vec_cmpeq(vec_and(here_u8x16, mask_two_u8x16),
                                                                         pattern_two_u8x16);
        three_byte_bool_u8x16[quarter] = (__vector unsigned char)vec_cmpeq(vec_and(here_u8x16, mask_three_u8x16),
                                                                           pattern_three_u8x16);
        four_byte_bool_u8x16[quarter] = (__vector unsigned char)vec_cmpeq(vec_and(here_u8x16, mask_four_u8x16),
                                                                          pattern_four_u8x16);
    }
    result.continuation = sz_utf8_mask_combine_powervsx_(continuation_bool_u8x16) & loaded_mask;
    result.codepoint_starts = loaded_mask & ~result.continuation;
    result.two_byte_starts = sz_utf8_mask_combine_powervsx_(two_byte_bool_u8x16) & loaded_mask;
    result.three_byte_starts = sz_utf8_mask_combine_powervsx_(three_byte_bool_u8x16) & loaded_mask;
    result.four_byte_starts = sz_utf8_mask_combine_powervsx_(four_byte_bool_u8x16) & loaded_mask;
    __vector unsigned char const low_five_bits_u8x16 = vec_splats((unsigned char)0x1F);
    __vector unsigned char const low_six_bits_u8x16 = vec_splats((unsigned char)0x3F);
    __vector unsigned char const low_two_bits_u8x16 = vec_splats((unsigned char)0x03);
    __vector unsigned char const low_four_bits_u8x16 = vec_splats((unsigned char)0x0F);
    __vector unsigned char const shift_six_u8x16 = vec_splats((unsigned char)6);
    __vector unsigned char const shift_four_u8x16 = vec_splats((unsigned char)4);
    for (int quarter = 0; quarter < 4; ++quarter) {
        __vector unsigned char const here_u8x16 = window_u8x16[quarter];
        __vector unsigned char const next_byte_u8x16 = next1_u8x16[quarter];
        __vector unsigned char const next_byte2_u8x16 = next2_u8x16[quarter];
        __vector unsigned char const high_two_u8x16 = sz_utf8_srl8_powervsx_(vec_and(here_u8x16, low_five_bits_u8x16),
                                                                             2, 0x07);
        __vector unsigned char const low_two_u8x16 = vec_or(
            vec_sl(vec_and(here_u8x16, low_two_bits_u8x16), shift_six_u8x16),
            vec_and(next_byte_u8x16, low_six_bits_u8x16));
        __vector unsigned char const high_three_u8x16 = vec_or(
            vec_sl(vec_and(here_u8x16, low_four_bits_u8x16), shift_four_u8x16),
            sz_utf8_srl8_powervsx_(next_byte_u8x16, 2, 0x0F));
        __vector unsigned char const low_three_u8x16 = vec_or(
            vec_sl(vec_and(next_byte_u8x16, low_two_bits_u8x16), shift_six_u8x16),
            vec_and(next_byte2_u8x16, low_six_bits_u8x16));
        __vector bool char const three_select_u8x16 = (__vector bool char)three_byte_bool_u8x16[quarter];
        result.high[quarter] = vec_sel(high_two_u8x16, high_three_u8x16, three_select_u8x16);
        result.low[quarter] = vec_sel(low_two_u8x16, low_three_u8x16, three_select_u8x16);
    }
    return result;
}
/** @brief  One nibble-cascade stage with a sub-256 selector: `result[lane] = table[selector[lane]*16 +
 *          within[lane]]` when `selector < tile_count`, else 0, the VSX twin of @ref sz_utf8_rune_cascade_stage_neon_.
 *          Each 16-byte row is one selector value: it is `vec_perm`-gathered by the nibble @p within over its own
 *          register pair `{row, row}` and blended in for the lanes whose @p selector equals that row. Selectors past
 *          the table match no row, so those lanes stay 0 (the NEON `< tile_count` clamp). Gather-free. */
SZ_HELPER_AUTO __vector unsigned char sz_utf8_rune_cascade_stage_powervsx_( 
    sz_u8_t const *table, int tile_count, __vector unsigned char selector_u8x16, __vector unsigned char within_u8x16) {
    __vector unsigned char const within_nibble_u8x16 = vec_and(within_u8x16, vec_splats((unsigned char)0x0F));
    __vector unsigned char result_u8x16 = vec_splats((unsigned char)0);
    for (int tile = 0; tile < tile_count; ++tile) {
        __vector unsigned char const row_u8x16 = vec_xl(0, table + tile * 16);
        __vector unsigned char const gathered_u8x16 = vec_perm(row_u8x16, row_u8x16, within_nibble_u8x16);
        __vector bool char const here_u8x16 = vec_cmpeq(selector_u8x16, vec_splats((unsigned char)tile));
        result_u8x16 = vec_sel(result_u8x16, gathered_u8x16, here_u8x16);
    }
    return result_u8x16;
}
/** @brief  256-entry byte LUT addressed by a per-lane byte index in [0, 256): `result[lane] = group_base[index[lane]]`,
 *          the VSX twin of @ref sz_utf8_rune_lut256_neon_. VSX `vec_perm` reaches only 32 bytes, so the 256-byte
 *          table is read by a bounded scalar L1 walk staged through a `sz_u128_vec_t` union. */
SZ_HELPER_AUTO __vector unsigned char sz_utf8_rune_lut256_powervsx_(sz_u8_t const *group_base,
                                                                    __vector unsigned char index_u8x16) {
    sz_u128_vec_t index_vec, result_vec;
    index_vec.vsx_u8 = index_u8x16;
    for (int lane = 0; lane < 16; ++lane) result_vec.u8s[lane] = group_base[index_vec.u8s[lane]];
    return result_vec.vsx_u8;
}
/** @brief  Class byte per lane from a page-compressed flat table, the VSX twin of @ref sz_utf8_rune_flat_lookup_neon_:
 *          `page_lut[high]` selects one 256-byte page, then `flat[page*256 + low]` is read per lane. VSX has no
 *          gather, so the whole lookup is a bounded scalar L1 walk over the fused index; lanes whose page reaches
 *          @p page_count return 0. */
SZ_HELPER_AUTO __vector unsigned char sz_utf8_rune_flat_lookup_powervsx_( 
    sz_u8_t const *page_lut, sz_u8_t const *flat, int page_count, __vector unsigned char high_bytes_u8x16,
    __vector unsigned char low_bytes_u8x16) {
    sz_u128_vec_t high_vec, low_vec, result_vec;
    high_vec.vsx_u8 = high_bytes_u8x16;
    low_vec.vsx_u8 = low_bytes_u8x16;
    for (int lane = 0; lane < 16; ++lane) {
        int const page = page_lut[high_vec.u8s[lane]];
        result_vec.u8s[lane] = page < page_count ? flat[page * 256 + low_vec.u8s[lane]] : (sz_u8_t)0;
    }
    return result_vec.vsx_u8;
}
/** @brief  POWER VSX forward drain, the `vpcompressb`-free twin of @ref sz_utf8_rune_drain_forward_neon_, bit-exact
 *          with it. Emits one (start, length) per set boundary lane (ascending), honoring @p capacity and the
 *          carried previous boundary @p previous_io. The set boundary lanes are left-packed to ascending byte
 *          offsets by the shuffle-LUT compaction (no scalar `ctz` walk); that compaction's return value is the
 *          boundary count, so no popcount is needed either. */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_forward_powervsx_( 
    sz_u64_t boundary, sz_size_t base, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *previous_io) {
    sz_size_t previous = *previous_io;
    if (boundary == 0 || produced >= capacity) {
        *previous_io = previous;
        return produced;
    }
    sz_u8_t indices[64];
    sz_size_t boundary_count = 0;
    for (int reg = 0; reg < 4; ++reg)
        boundary_count = sz_utf8_compress_starts_powervsx_((sz_u32_t)((boundary >> (reg * 16)) & 0xFFFFu), reg * 16,
                                                           indices, boundary_count);
    sz_size_t emitted = 0;
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const position = base + (sz_size_t)indices[emitted];
        starts[produced] = previous;
        lengths[produced] = position - previous;
        previous = position;
        ++produced, ++emitted;
    }
    *previous_io = previous;
    return produced;
}
#pragma endregion Windowed engine leaf substrate
#endif 
SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_powervsx( 
    sz_cptr_t text, sz_size_t length,              
    sz_rune_t *runes, sz_size_t runes_capacity,    
    sz_size_t *runes_unpacked) {
#if !SZ_IS_BIG_ENDIAN_
    sz_cptr_t cursor = text;
    sz_cptr_t const end = text + length;
    sz_size_t runes_written = 0;
    while (runes_written < runes_capacity && cursor < end) {
        sz_size_t step_unpacked = 0;
        sz_cptr_t next = sz_utf8_decode_once_powervsx_(cursor, (sz_size_t)(end - cursor), runes + runes_written,
                                                       runes_capacity - runes_written, &step_unpacked);
        if (step_unpacked) {
            runes_written += step_unpacked;
            cursor = next;
            continue;
        }
        if (sz_utf8_incomplete_tail_(cursor, end)) break;
        runes[runes_written++] = (sz_rune_t)sz_rune_replacement_k;
        cursor += sz_utf8_maximal_subpart_(cursor, end);
    }
    *runes_unpacked = runes_written;
    return cursor;
#else
    return sz_utf8_decode_serial(text, length, runes, runes_capacity, runes_unpacked);
#endif
}
#pragma region Multistep newline / whitespace iteration
#pragma endregion Multistep newline / whitespace iteration
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
