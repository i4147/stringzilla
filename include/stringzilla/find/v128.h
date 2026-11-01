/**
 *  @brief WebAssembly SIMD128 backend for find.
 *  @file include/stringzilla/find/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_V128_H_
#define STRINGZILLA_FIND_V128_H_
#include "stringzilla/types.h"
#include "stringzilla/find/serial.h"
#include "stringzilla/compare.h" 
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif
/**
 *  @brief Return a pointer to the first matching byte in a 16-byte block, or NULL if none.
 *  @param match_u8x16 A 16-byte comparison result (0xFF where matched, 0x00 otherwise).
 *  @param block_start Pointer to the first byte of the block.
 *  @return Pointer to the first match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_INLINE sz_cptr_t sz_locate_first_v128_(v128_t match_u8x16, sz_cptr_t block_start) {
    sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_u8x16);
    if (!matches) return SZ_NULL_CHAR;
    return block_start + sz_u32_ctz(matches);
}
/**
 *  @brief Return a pointer to the last matching byte in a 16-byte block, or NULL if none.
 *  @param match_u8x16 A 16-byte comparison result (0xFF where matched, 0x00 otherwise).
 *  @param block_start Pointer to the first byte of the block.
 *  @return Pointer to the last match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_INLINE sz_cptr_t sz_locate_last_v128_(v128_t match_u8x16, sz_cptr_t block_start) {
    sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_u8x16);
    if (!matches) return SZ_NULL_CHAR;
    return block_start + (31 - sz_u32_clz(matches));
}
SZ_API_COMPTIME sz_cptr_t sz_find_byte_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    v128_t needle_u8x16 = wasm_i8x16_splat(*(sz_i8_t const *)needle);
    while (haystack_length >= 64) {
        v128_t match_at_0_u8x16 = wasm_i8x16_eq(wasm_v128_load(haystack + 0), needle_u8x16);
        v128_t match_at_16_u8x16 = wasm_i8x16_eq(wasm_v128_load(haystack + 16), needle_u8x16);
        v128_t match_at_32_u8x16 = wasm_i8x16_eq(wasm_v128_load(haystack + 32), needle_u8x16);
        v128_t match_at_48_u8x16 = wasm_i8x16_eq(wasm_v128_load(haystack + 48), needle_u8x16);
        v128_t any_match_u8x16 = wasm_v128_or(wasm_v128_or(match_at_0_u8x16, match_at_16_u8x16),
                                              wasm_v128_or(match_at_32_u8x16, match_at_48_u8x16));
        if (wasm_v128_any_true(any_match_u8x16)) {
            sz_cptr_t found;
            if ((found = sz_locate_first_v128_(match_at_0_u8x16, haystack + 0))) return found;
            if ((found = sz_locate_first_v128_(match_at_16_u8x16, haystack + 16))) return found;
            if ((found = sz_locate_first_v128_(match_at_32_u8x16, haystack + 32))) return found;
            if ((found = sz_locate_first_v128_(match_at_48_u8x16, haystack + 48))) return found;
        }
        haystack += 64, haystack_length -= 64;
    }
    while (haystack_length >= 16) {
        sz_cptr_t found = sz_locate_first_v128_(wasm_i8x16_eq(wasm_v128_load(haystack), needle_u8x16), haystack);
        if (found) return found;
        haystack += 16, haystack_length -= 16;
    }
    return sz_find_byte_serial(haystack, haystack_length, needle);
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    v128_t needle_u8x16 = wasm_i8x16_splat(*(sz_i8_t const *)needle);
    while (haystack_length >= 64) {
        sz_cptr_t window_start = haystack + haystack_length - 64;
        v128_t match_at_0_u8x16 = wasm_i8x16_eq(wasm_v128_load(window_start + 0), needle_u8x16);
        v128_t match_at_16_u8x16 = wasm_i8x16_eq(wasm_v128_load(window_start + 16), needle_u8x16);
        v128_t match_at_32_u8x16 = wasm_i8x16_eq(wasm_v128_load(window_start + 32), needle_u8x16);
        v128_t match_at_48_u8x16 = wasm_i8x16_eq(wasm_v128_load(window_start + 48), needle_u8x16);
        v128_t any_match_u8x16 = wasm_v128_or(wasm_v128_or(match_at_0_u8x16, match_at_16_u8x16),
                                              wasm_v128_or(match_at_32_u8x16, match_at_48_u8x16));
        if (wasm_v128_any_true(any_match_u8x16)) {
            sz_cptr_t found;
            if ((found = sz_locate_last_v128_(match_at_48_u8x16, window_start + 48))) return found;
            if ((found = sz_locate_last_v128_(match_at_32_u8x16, window_start + 32))) return found;
            if ((found = sz_locate_last_v128_(match_at_16_u8x16, window_start + 16))) return found;
            if ((found = sz_locate_last_v128_(match_at_0_u8x16, window_start + 0))) return found;
        }
        haystack_length -= 64;
    }
    while (haystack_length >= 16) {
        sz_cptr_t window_start = haystack + haystack_length - 16;
        sz_cptr_t found = sz_locate_last_v128_(wasm_i8x16_eq(wasm_v128_load(window_start), needle_u8x16), window_start);
        if (found) return found;
        haystack_length -= 16;
    }
    return sz_rfind_byte_serial(haystack, haystack_length, needle);
}
/**
 *  @brief Compute a per-lane match mask for bytes present in `set`, split across two 16-byte half-tables.
 *  @param haystack_u8x16 The 16-byte input register.
 *  @param set_top_u8x16 Top half of the byteset (byte indices 0..15).
 *  @param set_bottom_u8x16 Bottom half of the byteset (byte indices 16..31).
 *  @return 0xFF per lane where the byte belongs to the set, 0x00 otherwise.
 */
SZ_HELPER_AUTO v128_t sz_find_byteset_match_v128_(v128_t haystack_u8x16, v128_t set_top_u8x16,
                                                  v128_t set_bottom_u8x16) {
    v128_t byte_index_u8x16 = wasm_u8x16_shr(haystack_u8x16, 3); 
    v128_t bit_table_u8x16 = wasm_i8x16_make(1, 2, 4, 8, 16, 32, 64, (sz_i8_t)128, 0, 0, 0, 0, 0, 0, 0, 0);
    v128_t byte_mask_u8x16 = wasm_i8x16_swizzle(bit_table_u8x16, wasm_v128_and(haystack_u8x16, wasm_i8x16_splat(7)));
    v128_t matches_top_u8x16 = wasm_i8x16_swizzle(set_top_u8x16, byte_index_u8x16);
    v128_t matches_bottom_u8x16 = wasm_i8x16_swizzle(set_bottom_u8x16,
                                                     wasm_i8x16_sub(byte_index_u8x16, wasm_i8x16_splat(16)));
    v128_t matches_u8x16 = wasm_v128_or(matches_top_u8x16, matches_bottom_u8x16);
    return wasm_i8x16_ne(wasm_v128_and(matches_u8x16, byte_mask_u8x16), wasm_i8x16_splat(0));
}
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set) {
    v128_t set_top_u8x16 = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_u8x16 = wasm_v128_load(&set->_u8s[16]);
    while (haystack_length >= 64) {
        v128_t match_at_0_u8x16 = sz_find_byteset_match_v128_(wasm_v128_load(haystack + 0), set_top_u8x16,
                                                              set_bottom_u8x16);
        v128_t match_at_16_u8x16 = sz_find_byteset_match_v128_(wasm_v128_load(haystack + 16), set_top_u8x16,
                                                               set_bottom_u8x16);
        v128_t match_at_32_u8x16 = sz_find_byteset_match_v128_(wasm_v128_load(haystack + 32), set_top_u8x16,
                                                               set_bottom_u8x16);
        v128_t match_at_48_u8x16 = sz_find_byteset_match_v128_(wasm_v128_load(haystack + 48), set_top_u8x16,
                                                               set_bottom_u8x16);
        v128_t any_match_u8x16 = wasm_v128_or(wasm_v128_or(match_at_0_u8x16, match_at_16_u8x16),
                                              wasm_v128_or(match_at_32_u8x16, match_at_48_u8x16));
        if (wasm_v128_any_true(any_match_u8x16)) {
            sz_cptr_t found;
            if ((found = sz_locate_first_v128_(match_at_0_u8x16, haystack + 0))) return found;
            if ((found = sz_locate_first_v128_(match_at_16_u8x16, haystack + 16))) return found;
            if ((found = sz_locate_first_v128_(match_at_32_u8x16, haystack + 32))) return found;
            if ((found = sz_locate_first_v128_(match_at_48_u8x16, haystack + 48))) return found;
        }
        haystack += 64, haystack_length -= 64;
    }
    for (; haystack_length >= 16; haystack += 16, haystack_length -= 16) {
        v128_t match_u8x16 = sz_find_byteset_match_v128_(wasm_v128_load(haystack), set_top_u8x16, set_bottom_u8x16);
        sz_cptr_t found = sz_locate_first_v128_(match_u8x16, haystack);
        if (found) return found;
    }
    return sz_find_byteset_serial(haystack, haystack_length, set);
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_v128(sz_cptr_t haystack, sz_size_t haystack_length,
                                                sz_byteset_t const *set) {
    v128_t set_top_u8x16 = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_u8x16 = wasm_v128_load(&set->_u8s[16]);
    while (haystack_length >= 64) {
        sz_cptr_t window_start = haystack + haystack_length - 64;
        v128_t match_at_0_u8x16 = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 0), set_top_u8x16,
                                                              set_bottom_u8x16);
        v128_t match_at_16_u8x16 = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 16), set_top_u8x16,
                                                               set_bottom_u8x16);
        v128_t match_at_32_u8x16 = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 32), set_top_u8x16,
                                                               set_bottom_u8x16);
        v128_t match_at_48_u8x16 = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 48), set_top_u8x16,
                                                               set_bottom_u8x16);
        v128_t any_match_u8x16 = wasm_v128_or(wasm_v128_or(match_at_0_u8x16, match_at_16_u8x16),
                                              wasm_v128_or(match_at_32_u8x16, match_at_48_u8x16));
        if (wasm_v128_any_true(any_match_u8x16)) {
            sz_cptr_t found;
            if ((found = sz_locate_last_v128_(match_at_48_u8x16, window_start + 48))) return found;
            if ((found = sz_locate_last_v128_(match_at_32_u8x16, window_start + 32))) return found;
            if ((found = sz_locate_last_v128_(match_at_16_u8x16, window_start + 16))) return found;
            if ((found = sz_locate_last_v128_(match_at_0_u8x16, window_start + 0))) return found;
        }
        haystack_length -= 64;
    }
    for (; haystack_length >= 16; haystack_length -= 16) {
        sz_cptr_t window_start = haystack + haystack_length - 16;
        v128_t match_u8x16 = sz_find_byteset_match_v128_(wasm_v128_load(window_start), set_top_u8x16, set_bottom_u8x16);
        sz_cptr_t found = sz_locate_last_v128_(match_u8x16, window_start);
        if (found) return found;
    }
    return sz_rfind_byteset_serial(haystack, haystack_length, set);
}
/**
 *  @brief Candidate mask for a substring window: lanes where the first/mid/last needle bytes all line up.
 *  @param haystack_start Base pointer for the 16-byte window.
 *  @param offset_first Offset of the first anomaly byte in the needle.
 *  @param offset_mid Offset of the mid anomaly byte in the needle.
 *  @param offset_last Offset of the last anomaly byte in the needle.
 *  @param needle_first_u8x16 Broadcasted first-anomaly needle byte.
 *  @param needle_mid_u8x16 Broadcasted mid-anomaly needle byte.
 *  @param needle_last_u8x16 Broadcasted last-anomaly needle byte.
 *  @return 0xFF per lane where all three needle bytes match.
 */
SZ_HELPER_INLINE v128_t sz_find_substr_match_v128_(                                                
    sz_cptr_t haystack_start, sz_size_t offset_first, sz_size_t offset_mid, sz_size_t offset_last, 
    v128_t needle_first_u8x16, v128_t needle_mid_u8x16, v128_t needle_last_u8x16) {
    return wasm_v128_and(                                                                     
        wasm_v128_and(                                                                        
            wasm_i8x16_eq(wasm_v128_load(haystack_start + offset_first), needle_first_u8x16), 
            wasm_i8x16_eq(wasm_v128_load(haystack_start + offset_mid), needle_mid_u8x16)),    
        wasm_i8x16_eq(wasm_v128_load(haystack_start + offset_last), needle_last_u8x16));
}
/**
 *  @brief Walk a window's candidates low-to-high, returning the first that verifies via `sz_equal`.
 *  @param match_u8x16 The 16-byte candidate mask.
 *  @param window_start Base pointer for this 16-byte window.
 *  @param needle The full needle to verify against.
 *  @param needle_length Length of `needle` in bytes.
 *  @return Pointer to the first verified match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_AUTO sz_cptr_t sz_locate_substr_first_v128_( 
    v128_t match_u8x16, sz_cptr_t window_start, sz_cptr_t needle, sz_size_t needle_length) {
    sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_u8x16);
    while (matches) {
        int candidate_offset = sz_u32_ctz(matches);
        if (sz_equal_v128(window_start + candidate_offset, needle, needle_length))
            return window_start + candidate_offset;
        matches &= matches - 1; 
    }
    return SZ_NULL_CHAR;
}
/**
 *  @brief Walk a window's candidates high-to-low, returning the last that verifies via `sz_equal`.
 *  @param match_u8x16 The 16-byte candidate mask.
 *  @param window_start Base pointer for this 16-byte window.
 *  @param needle The full needle to verify against.
 *  @param needle_length Length of `needle` in bytes.
 *  @return Pointer to the last verified match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_AUTO sz_cptr_t sz_locate_substr_last_v128_( 
    v128_t match_u8x16, sz_cptr_t window_start, sz_cptr_t needle, sz_size_t needle_length) {
    sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_u8x16);
    while (matches) {
        int candidate_offset = 31 - sz_u32_clz(matches);
        if (sz_equal_v128(window_start + candidate_offset, needle, needle_length))
            return window_start + candidate_offset;
        matches &= ~((sz_u32_t)1 << candidate_offset); 
    }
    return SZ_NULL_CHAR;
}
SZ_API_COMPTIME sz_cptr_t sz_find_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length) {
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_v128(haystack, haystack_length, needle);
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    v128_t needle_first_u8x16 = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_first]);
    v128_t needle_mid_u8x16 = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_mid]);
    v128_t needle_last_u8x16 = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_last]);
    while (haystack_length >= needle_length + 64) {
        v128_t match_at_0_u8x16 = sz_find_substr_match_v128_(haystack + 0, offset_first, offset_mid, offset_last,
                                                             needle_first_u8x16, needle_mid_u8x16, needle_last_u8x16);
        v128_t match_at_16_u8x16 = sz_find_substr_match_v128_(haystack + 16, offset_first, offset_mid, offset_last,
                                                              needle_first_u8x16, needle_mid_u8x16, needle_last_u8x16);
        v128_t match_at_32_u8x16 = sz_find_substr_match_v128_(haystack + 32, offset_first, offset_mid, offset_last,
                                                              needle_first_u8x16, needle_mid_u8x16, needle_last_u8x16);
        v128_t match_at_48_u8x16 = sz_find_substr_match_v128_(haystack + 48, offset_first, offset_mid, offset_last,
                                                              needle_first_u8x16, needle_mid_u8x16, needle_last_u8x16);
        v128_t any_match_u8x16 = wasm_v128_or(wasm_v128_or(match_at_0_u8x16, match_at_16_u8x16),
                                              wasm_v128_or(match_at_32_u8x16, match_at_48_u8x16));
        if (wasm_v128_any_true(any_match_u8x16)) {
            sz_cptr_t found;
            if ((found = sz_locate_substr_first_v128_(match_at_0_u8x16, haystack + 0, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_first_v128_(match_at_16_u8x16, haystack + 16, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_first_v128_(match_at_32_u8x16, haystack + 32, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_first_v128_(match_at_48_u8x16, haystack + 48, needle, needle_length)))
                return found;
        }
        haystack += 64, haystack_length -= 64;
    }
    for (; haystack_length >= needle_length + 16; haystack += 16, haystack_length -= 16) {
        v128_t match_u8x16 = sz_find_substr_match_v128_( 
            haystack, offset_first, offset_mid, offset_last, needle_first_u8x16, needle_mid_u8x16, needle_last_u8x16);
        sz_cptr_t found = sz_locate_substr_first_v128_(match_u8x16, haystack, needle, needle_length);
        if (found) return found;
    }
    return sz_find_serial(haystack, haystack_length, needle, needle_length);
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                        sz_size_t needle_length) {
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_v128(haystack, haystack_length, needle);
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    v128_t needle_first_u8x16 = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_first]);
    v128_t needle_mid_u8x16 = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_mid]);
    v128_t needle_last_u8x16 = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_last]);
    while (haystack_length >= needle_length + 64) {
        sz_cptr_t window_start = haystack + haystack_length - needle_length - 15; 
        v128_t match_at_0_u8x16 = sz_find_substr_match_v128_(window_start - 0, offset_first, offset_mid, offset_last,
                                                             needle_first_u8x16, needle_mid_u8x16, needle_last_u8x16);
        v128_t match_at_16_u8x16 = sz_find_substr_match_v128_(window_start - 16, offset_first, offset_mid, offset_last,
                                                              needle_first_u8x16, needle_mid_u8x16, needle_last_u8x16);
        v128_t match_at_32_u8x16 = sz_find_substr_match_v128_(window_start - 32, offset_first, offset_mid, offset_last,
                                                              needle_first_u8x16, needle_mid_u8x16, needle_last_u8x16);
        v128_t match_at_48_u8x16 = sz_find_substr_match_v128_(window_start - 48, offset_first, offset_mid, offset_last,
                                                              needle_first_u8x16, needle_mid_u8x16, needle_last_u8x16);
        v128_t any_match_u8x16 = wasm_v128_or(wasm_v128_or(match_at_0_u8x16, match_at_16_u8x16),
                                              wasm_v128_or(match_at_32_u8x16, match_at_48_u8x16));
        if (wasm_v128_any_true(any_match_u8x16)) {
            sz_cptr_t found;
            if ((found = sz_locate_substr_last_v128_(match_at_0_u8x16, window_start - 0, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_last_v128_(match_at_16_u8x16, window_start - 16, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_last_v128_(match_at_32_u8x16, window_start - 32, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_last_v128_(match_at_48_u8x16, window_start - 48, needle, needle_length)))
                return found;
        }
        haystack_length -= 64;
    }
    for (; haystack_length >= needle_length + 16; haystack_length -= 16) {
        sz_cptr_t window_start = haystack + haystack_length - needle_length - 15;
        v128_t match_u8x16 = sz_find_substr_match_v128_( 
            window_start, offset_first, offset_mid, offset_last, needle_first_u8x16, needle_mid_u8x16,
            needle_last_u8x16);
        sz_cptr_t found = sz_locate_substr_last_v128_(match_u8x16, window_start, needle, needle_length);
        if (found) return found;
    }
    return sz_rfind_serial(haystack, haystack_length, needle, needle_length);
}
#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif 
#ifdef __cplusplus
}
#endif
#endif 
