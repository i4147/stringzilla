/**
 *  @brief LoongArch LASX (256-bit) backend for find.
 *  @file include/stringzilla/find/lasx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_LASX_H_
#define STRINGZILLA_FIND_LASX_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#include "stringzilla/find/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_LASX
/**
 *  @brief Produce an AVX2-style 32-bit movemask from a LASX 256-bit comparison result.
 *      Bit `i` reflects byte `i` of the comparison (0xFF → set, 0x00 → clear).
 *
 *  See `compare/lasx.h` for the rationale: `__lasx_xvmskltz_b` packs the sign bit of each byte into a
 *  per-128-bit-lane 16-bit mask (word 0 = low lane, word 4 = high lane). Recombining matches AVX2's
 *  `_mm256_movemask_epi8` byte ordering, so `ctz`/`clz` index bytes identically to the Haswell backend.
 *
 *  @param sign_extended_u8x32 A 256-bit comparison result vector (0xFF where matched, 0x00 otherwise).
 *  @return 32-bit movemask where bit `i` is set when byte `i` matched.
 */
SZ_HELPER_INLINE sz_u32_t sz_xvmovemask_b_find_lasx_(__m256i sign_extended_u8x32) {
    __m256i collected_u32x8 = __lasx_xvmskltz_b(sign_extended_u8x32);
    unsigned int low = __lasx_xvpickve2gr_wu(collected_u32x8, 0);
    unsigned int high = __lasx_xvpickve2gr_wu(collected_u32x8, 4);
    return (low & 0xFFFFu) | ((high & 0xFFFFu) << 16);
}
/**
 *  @brief Produce an SSE-style 16-bit movemask from an LSX 128-bit comparison result.
 *      Used to cover the sub-32-byte head/tail that a 256-bit-only loop would otherwise hand to the serial path.
 *
 *  @param sign_extended_u8x16 A 128-bit comparison result vector (0xFF where matched, 0x00 otherwise).
 *  @return Low 16 bits of the movemask.
 */
SZ_HELPER_INLINE sz_u32_t sz_vmovemask_b_find_lsx_(__m128i sign_extended_u8x16) {
    return __lsx_vpickve2gr_wu(__lsx_vmskltz_b(sign_extended_u8x16), 0) & 0xFFFFu;
}
SZ_API_COMPTIME sz_cptr_t sz_find_byte_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u256_vec_t haystack_vec, needle_vec, matches_vec;
    needle_vec.lasx = __lasx_xvldrepl_b(needle, 0);
    while (haystack_length >= 32) {
        haystack_vec.lasx = __lasx_xvld(haystack, 0);
        matches_vec.lasx = __lasx_xvseq_b(haystack_vec.lasx, needle_vec.lasx);
        if (__lasx_xbnz_v(matches_vec.lasx)) {
            __m256i first_match_indices_u8x32 = __lasx_xvfrstpi_b(matches_vec.lasx, matches_vec.lasx, 0);
            unsigned int low_lane_index = __lasx_xvpickve2gr_wu(first_match_indices_u8x32, 0) & 0xFF;
            if (low_lane_index < 16) return haystack + low_lane_index;
            return haystack + 16 + (__lasx_xvpickve2gr_wu(first_match_indices_u8x32, 4) & 0xFF);
        }
        haystack += 32, haystack_length -= 32;
    }
    if (haystack_length >= 16) {
        sz_u128_vec_t haystack128_vec, needle128_vec, matches128_vec;
        needle128_vec.lsx = __lsx_vldrepl_b(needle, 0);
        haystack128_vec.lsx = __lsx_vld(haystack, 0);
        matches128_vec.lsx = __lsx_vseq_b(haystack128_vec.lsx, needle128_vec.lsx);
        if (__lsx_bnz_v(matches128_vec.lsx)) {
            __m128i first_match_indices_u8x16 = __lsx_vfrstpi_b(matches128_vec.lsx, matches128_vec.lsx, 0);
            return haystack + (__lsx_vpickve2gr_wu(first_match_indices_u8x16, 0) & 0xFF);
        }
        haystack += 16, haystack_length -= 16;
    }
    return sz_find_byte_serial(haystack, haystack_length, needle);
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u256_vec_t haystack_vec, needle_vec, matches_vec;
    needle_vec.lasx = __lasx_xvldrepl_b(needle, 0);
    while (haystack_length >= 32) {
        haystack_vec.lasx = __lasx_xvld(haystack + haystack_length - 32, 0);
        matches_vec.lasx = __lasx_xvseq_b(haystack_vec.lasx, needle_vec.lasx);
        if (__lasx_xbnz_v(matches_vec.lasx))
            return haystack + haystack_length - 1 - sz_u32_clz(sz_xvmovemask_b_find_lasx_(matches_vec.lasx));
        haystack_length -= 32;
    }
    if (haystack_length >= 16) {
        sz_u128_vec_t haystack128_vec, needle128_vec, matches128_vec;
        needle128_vec.lsx = __lsx_vldrepl_b(needle, 0);
        haystack128_vec.lsx = __lsx_vld(haystack + haystack_length - 16, 0);
        matches128_vec.lsx = __lsx_vseq_b(haystack128_vec.lsx, needle128_vec.lsx);
        if (__lsx_bnz_v(matches128_vec.lsx))
            return haystack + haystack_length - 1 - (sz_u32_clz(sz_vmovemask_b_find_lsx_(matches128_vec.lsx)) - 16);
        haystack_length -= 16;
    }
    return sz_rfind_byte_serial(haystack, haystack_length, needle);
}
SZ_API_COMPTIME sz_cptr_t sz_find_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length) {
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_lasx(haystack, haystack_length, needle);
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t haystack_first_vec, haystack_mid_vec, haystack_last_vec, needle_first_vec, needle_mid_vec,
        needle_last_vec;
    needle_first_vec.lasx = __lasx_xvldrepl_b(needle + offset_first, 0);
    needle_mid_vec.lasx = __lasx_xvldrepl_b(needle + offset_mid, 0);
    needle_last_vec.lasx = __lasx_xvldrepl_b(needle + offset_last, 0);
    for (; haystack_length >= needle_length + 32; haystack += 32, haystack_length -= 32) {
        haystack_first_vec.lasx = __lasx_xvld(haystack + offset_first, 0);
        haystack_mid_vec.lasx = __lasx_xvld(haystack + offset_mid, 0);
        haystack_last_vec.lasx = __lasx_xvld(haystack + offset_last, 0);
        __m256i first_matches_u8x32 = __lasx_xvseq_b(haystack_first_vec.lasx, needle_first_vec.lasx);
        __m256i mid_matches_u8x32 = __lasx_xvseq_b(haystack_mid_vec.lasx, needle_mid_vec.lasx);
        __m256i last_matches_u8x32 = __lasx_xvseq_b(haystack_last_vec.lasx, needle_last_vec.lasx);
        __m256i all_matches_u8x32 = __lasx_xvand_v(__lasx_xvand_v(first_matches_u8x32, mid_matches_u8x32),
                                                   last_matches_u8x32);
        if (__lasx_xbnz_v(all_matches_u8x32)) {
            matches_vec.u32 = sz_xvmovemask_b_find_lasx_(all_matches_u8x32);
            while (matches_vec.u32) {
                int potential_offset = sz_u32_ctz(matches_vec.u32);
                if (sz_equal_lasx(haystack + potential_offset, needle, needle_length))
                    return haystack + potential_offset;
                matches_vec.u32 &= matches_vec.u32 - 1;
            }
        }
    }
    {
        sz_u128_vec_t haystack_first128_vec, haystack_mid128_vec, haystack_last128_vec, needle_first128_vec,
            needle_mid128_vec, needle_last128_vec;
        needle_first128_vec.lsx = __lsx_vldrepl_b(needle + offset_first, 0);
        needle_mid128_vec.lsx = __lsx_vldrepl_b(needle + offset_mid, 0);
        needle_last128_vec.lsx = __lsx_vldrepl_b(needle + offset_last, 0);
        for (; haystack_length >= needle_length + 16; haystack += 16, haystack_length -= 16) {
            haystack_first128_vec.lsx = __lsx_vld(haystack + offset_first, 0);
            haystack_mid128_vec.lsx = __lsx_vld(haystack + offset_mid, 0);
            haystack_last128_vec.lsx = __lsx_vld(haystack + offset_last, 0);
            __m128i first_matches_u8x16 = __lsx_vseq_b(haystack_first128_vec.lsx, needle_first128_vec.lsx);
            __m128i mid_matches_u8x16 = __lsx_vseq_b(haystack_mid128_vec.lsx, needle_mid128_vec.lsx);
            __m128i last_matches_u8x16 = __lsx_vseq_b(haystack_last128_vec.lsx, needle_last128_vec.lsx);
            __m128i all_matches_u8x16 = __lsx_vand_v(__lsx_vand_v(first_matches_u8x16, mid_matches_u8x16),
                                                     last_matches_u8x16);
            if (__lsx_bnz_v(all_matches_u8x16)) {
                sz_u32_t matches = sz_vmovemask_b_find_lsx_(all_matches_u8x16);
                while (matches) {
                    int potential_offset = sz_u32_ctz(matches);
                    if (sz_equal_lasx(haystack + potential_offset, needle, needle_length))
                        return haystack + potential_offset;
                    matches &= matches - 1;
                }
            }
        }
    }
    return sz_find_serial(haystack, haystack_length, needle, needle_length);
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                        sz_size_t needle_length) {
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_lasx(haystack, haystack_length, needle);
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t haystack_first_vec, haystack_mid_vec, haystack_last_vec, needle_first_vec, needle_mid_vec,
        needle_last_vec;
    needle_first_vec.lasx = __lasx_xvldrepl_b(needle + offset_first, 0);
    needle_mid_vec.lasx = __lasx_xvldrepl_b(needle + offset_mid, 0);
    needle_last_vec.lasx = __lasx_xvldrepl_b(needle + offset_last, 0);
    sz_cptr_t haystack_reversed;
    for (; haystack_length >= needle_length + 32; haystack_length -= 32) {
        haystack_reversed = haystack + haystack_length - needle_length - 32 + 1;
        haystack_first_vec.lasx = __lasx_xvld(haystack_reversed + offset_first, 0);
        haystack_mid_vec.lasx = __lasx_xvld(haystack_reversed + offset_mid, 0);
        haystack_last_vec.lasx = __lasx_xvld(haystack_reversed + offset_last, 0);
        __m256i first_matches_u8x32 = __lasx_xvseq_b(haystack_first_vec.lasx, needle_first_vec.lasx);
        __m256i mid_matches_u8x32 = __lasx_xvseq_b(haystack_mid_vec.lasx, needle_mid_vec.lasx);
        __m256i last_matches_u8x32 = __lasx_xvseq_b(haystack_last_vec.lasx, needle_last_vec.lasx);
        __m256i all_matches_u8x32 = __lasx_xvand_v(__lasx_xvand_v(first_matches_u8x32, mid_matches_u8x32),
                                                   last_matches_u8x32);
        if (__lasx_xbnz_v(all_matches_u8x32)) {
            matches_vec.u32 = sz_xvmovemask_b_find_lasx_(all_matches_u8x32);
            while (matches_vec.u32) {
                int potential_offset = sz_u32_clz(matches_vec.u32);
                if (sz_equal_lasx(haystack + haystack_length - needle_length - potential_offset, needle, needle_length))
                    return haystack + haystack_length - needle_length - potential_offset;
                matches_vec.u32 &= ~(1u << (31 - potential_offset));
            }
        }
    }
    {
        sz_u128_vec_t haystack_first128_vec, haystack_mid128_vec, haystack_last128_vec, needle_first128_vec,
            needle_mid128_vec, needle_last128_vec;
        needle_first128_vec.lsx = __lsx_vldrepl_b(needle + offset_first, 0);
        needle_mid128_vec.lsx = __lsx_vldrepl_b(needle + offset_mid, 0);
        needle_last128_vec.lsx = __lsx_vldrepl_b(needle + offset_last, 0);
        for (; haystack_length >= needle_length + 16; haystack_length -= 16) {
            sz_cptr_t haystack_reversed128 = haystack + haystack_length - needle_length - 16 + 1;
            haystack_first128_vec.lsx = __lsx_vld(haystack_reversed128 + offset_first, 0);
            haystack_mid128_vec.lsx = __lsx_vld(haystack_reversed128 + offset_mid, 0);
            haystack_last128_vec.lsx = __lsx_vld(haystack_reversed128 + offset_last, 0);
            __m128i first_matches_u8x16 = __lsx_vseq_b(haystack_first128_vec.lsx, needle_first128_vec.lsx);
            __m128i mid_matches_u8x16 = __lsx_vseq_b(haystack_mid128_vec.lsx, needle_mid128_vec.lsx);
            __m128i last_matches_u8x16 = __lsx_vseq_b(haystack_last128_vec.lsx, needle_last128_vec.lsx);
            __m128i all_matches_u8x16 = __lsx_vand_v(__lsx_vand_v(first_matches_u8x16, mid_matches_u8x16),
                                                     last_matches_u8x16);
            if (__lsx_bnz_v(all_matches_u8x16)) {
                sz_u32_t matches = sz_vmovemask_b_find_lsx_(all_matches_u8x16);
                while (matches) {
                    int potential_offset = (int)sz_u32_clz(matches) - 16;
                    if (sz_equal_lasx(haystack + haystack_length - needle_length - potential_offset, needle,
                                      needle_length))
                        return haystack + haystack_length - needle_length - potential_offset;
                    matches &= ~(1u << (15 - potential_offset));
                }
            }
        }
    }
    return sz_rfind_serial(haystack, haystack_length, needle, needle_length);
}
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_lasx(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    sz_u256_vec_t filter_even_vec, filter_odd_vec;
    sz_u256_vec_t text_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec, matches_vec;
    sz_u8_t even_pairs[32], odd_pairs[32];
    sz_u8_t const *filter_bytes = &filter->_u8s[0];
    for (sz_size_t j = 0; j < 16; ++j) {
        even_pairs[j] = even_pairs[j + 16] = filter_bytes[2 * j];
        odd_pairs[j] = odd_pairs[j + 16] = filter_bytes[2 * j + 1];
    }
    filter_even_vec.lasx = __lasx_xvld(even_pairs, 0);
    filter_odd_vec.lasx = __lasx_xvld(odd_pairs, 0);
    sz_u8_t bitmask_table[32];
    for (sz_size_t j = 0; j < 16; ++j) bitmask_table[j] = bitmask_table[j + 16] = (sz_u8_t)(1u << (j & 7));
    bitmask_lookup_vec.lasx = __lasx_xvld(bitmask_table, 0);
    __m256i const zero_u8x32 = __lasx_xvreplgr2vr_b(0);
    __m256i const nibble_mask_u8x32 = __lasx_xvreplgr2vr_b(0x0F);
    __m256i const eight_u8x32 = __lasx_xvreplgr2vr_b(8);
    while (length >= 32) {
        text_vec.lasx = __lasx_xvld(text, 0);
        lower_nibbles_vec.lasx = __lasx_xvand_v(text_vec.lasx, nibble_mask_u8x32);
        bitmask_vec.lasx = __lasx_xvshuf_b(zero_u8x32, bitmask_lookup_vec.lasx, lower_nibbles_vec.lasx);
        higher_nibbles_vec.lasx = __lasx_xvand_v(__lasx_xvsrli_b(text_vec.lasx, 4), nibble_mask_u8x32);
        bitset_even_vec.lasx = __lasx_xvshuf_b(zero_u8x32, filter_even_vec.lasx, higher_nibbles_vec.lasx);
        bitset_odd_vec.lasx = __lasx_xvshuf_b(zero_u8x32, filter_odd_vec.lasx, higher_nibbles_vec.lasx);
        __m256i use_odd_table_mask_u8x32 = __lasx_xvsle_bu(eight_u8x32, lower_nibbles_vec.lasx);
        bitset_even_vec.lasx = __lasx_xvbitsel_v(bitset_even_vec.lasx, bitset_odd_vec.lasx, use_odd_table_mask_u8x32);
        matches_vec.lasx = __lasx_xvand_v(bitset_even_vec.lasx, bitmask_vec.lasx);
        if (__lasx_xbnz_v(matches_vec.lasx)) {
            __m256i in_set_u8x32 = __lasx_xvslt_bu(zero_u8x32, matches_vec.lasx);
            __m256i first_match_indices_u8x32 = __lasx_xvfrstpi_b(in_set_u8x32, in_set_u8x32, 0);
            unsigned int low_lane_index = __lasx_xvpickve2gr_wu(first_match_indices_u8x32, 0) & 0xFF;
            if (low_lane_index < 16) return text + low_lane_index;
            return text + 16 + (__lasx_xvpickve2gr_wu(first_match_indices_u8x32, 4) & 0xFF);
        }
        text += 32, length -= 32;
    }
    if (length >= 16) {
        sz_u128_vec_t filter_even128_vec, filter_odd128_vec, bitmask_lookup128_vec;
        sz_u128_vec_t text128_vec, lower_nibbles128_vec, higher_nibbles128_vec;
        sz_u128_vec_t bitmask128_vec, bitset_even128_vec, bitset_odd128_vec, matches128_vec;
        filter_even128_vec.lsx = __lsx_vld(even_pairs, 0);
        filter_odd128_vec.lsx = __lsx_vld(odd_pairs, 0);
        bitmask_lookup128_vec.lsx = __lsx_vld(bitmask_table, 0);
        __m128i const zero_u8x16 = __lsx_vreplgr2vr_b(0);
        __m128i const nibble_mask_u8x16 = __lsx_vreplgr2vr_b(0x0F);
        __m128i const eight_u8x16 = __lsx_vreplgr2vr_b(8);
        text128_vec.lsx = __lsx_vld(text, 0);
        lower_nibbles128_vec.lsx = __lsx_vand_v(text128_vec.lsx, nibble_mask_u8x16);
        bitmask128_vec.lsx = __lsx_vshuf_b(zero_u8x16, bitmask_lookup128_vec.lsx, lower_nibbles128_vec.lsx);
        higher_nibbles128_vec.lsx = __lsx_vand_v(__lsx_vsrli_b(text128_vec.lsx, 4), nibble_mask_u8x16);
        bitset_even128_vec.lsx = __lsx_vshuf_b(zero_u8x16, filter_even128_vec.lsx, higher_nibbles128_vec.lsx);
        bitset_odd128_vec.lsx = __lsx_vshuf_b(zero_u8x16, filter_odd128_vec.lsx, higher_nibbles128_vec.lsx);
        __m128i use_odd_table_mask_u8x16 = __lsx_vsle_bu(eight_u8x16, lower_nibbles128_vec.lsx);
        bitset_even128_vec.lsx = __lsx_vbitsel_v(bitset_even128_vec.lsx, bitset_odd128_vec.lsx,
                                                 use_odd_table_mask_u8x16);
        matches128_vec.lsx = __lsx_vand_v(bitset_even128_vec.lsx, bitmask128_vec.lsx);
        if (__lsx_bnz_v(matches128_vec.lsx)) {
            __m128i in_set_u8x16 = __lsx_vslt_bu(zero_u8x16, matches128_vec.lsx);
            __m128i first_match_indices_u8x16 = __lsx_vfrstpi_b(in_set_u8x16, in_set_u8x16, 0);
            return text + (__lsx_vpickve2gr_wu(first_match_indices_u8x16, 0) & 0xFF);
        }
        text += 16, length -= 16;
    }
    return sz_find_byteset_serial(text, length, filter);
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_lasx(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    sz_u256_vec_t filter_even_vec, filter_odd_vec;
    sz_u256_vec_t text_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec, matches_vec;
    sz_u8_t even_pairs[32], odd_pairs[32];
    sz_u8_t const *filter_bytes = &filter->_u8s[0];
    for (sz_size_t j = 0; j < 16; ++j) {
        even_pairs[j] = even_pairs[j + 16] = filter_bytes[2 * j];
        odd_pairs[j] = odd_pairs[j + 16] = filter_bytes[2 * j + 1];
    }
    filter_even_vec.lasx = __lasx_xvld(even_pairs, 0);
    filter_odd_vec.lasx = __lasx_xvld(odd_pairs, 0);
    sz_u8_t bitmask_table[32];
    for (sz_size_t j = 0; j < 16; ++j) bitmask_table[j] = bitmask_table[j + 16] = (sz_u8_t)(1u << (j & 7));
    bitmask_lookup_vec.lasx = __lasx_xvld(bitmask_table, 0);
    __m256i const zero_u8x32 = __lasx_xvreplgr2vr_b(0);
    __m256i const nibble_mask_u8x32 = __lasx_xvreplgr2vr_b(0x0F);
    __m256i const eight_u8x32 = __lasx_xvreplgr2vr_b(8);
    while (length >= 32) {
        sz_cptr_t const window = text + length - 32;
        text_vec.lasx = __lasx_xvld(window, 0);
        lower_nibbles_vec.lasx = __lasx_xvand_v(text_vec.lasx, nibble_mask_u8x32);
        bitmask_vec.lasx = __lasx_xvshuf_b(zero_u8x32, bitmask_lookup_vec.lasx, lower_nibbles_vec.lasx);
        higher_nibbles_vec.lasx = __lasx_xvand_v(__lasx_xvsrli_b(text_vec.lasx, 4), nibble_mask_u8x32);
        bitset_even_vec.lasx = __lasx_xvshuf_b(zero_u8x32, filter_even_vec.lasx, higher_nibbles_vec.lasx);
        bitset_odd_vec.lasx = __lasx_xvshuf_b(zero_u8x32, filter_odd_vec.lasx, higher_nibbles_vec.lasx);
        __m256i use_odd_table_mask_u8x32 = __lasx_xvsle_bu(eight_u8x32, lower_nibbles_vec.lasx);
        bitset_even_vec.lasx = __lasx_xvbitsel_v(bitset_even_vec.lasx, bitset_odd_vec.lasx, use_odd_table_mask_u8x32);
        matches_vec.lasx = __lasx_xvand_v(bitset_even_vec.lasx, bitmask_vec.lasx);
        if (__lasx_xbnz_v(matches_vec.lasx)) {
            __m256i in_set_u8x32 = __lasx_xvslt_bu(zero_u8x32, matches_vec.lasx);
            sz_u32_t matches_mask = sz_xvmovemask_b_find_lasx_(in_set_u8x32);
            return window + (31 - sz_u32_clz(matches_mask)); 
        }
        length -= 32;
    }
    return sz_rfind_byteset_serial(text, length, filter);
}
#endif 
#ifdef __cplusplus
}
#endif
#endif 
