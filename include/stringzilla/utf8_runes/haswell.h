/**
 *  @brief Haswell (AVX2) backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_runes/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_HASWELL_H_
#define STRINGZILLA_UTF8_RUNES_HASWELL_H_
#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,bmi,bmi2,popcnt")
#endif
/** @brief  Mask for `_mm256_maskstore_epi64` selecting the low `count` (0..4) of four 64-bit lanes. */
SZ_HELPER_INLINE __m256i sz_mm256_store_mask_epi64_(sz_size_t count) {
    return _mm256_cmpgt_epi64(_mm256_set1_epi64x((long long)count), _mm256_setr_epi64x(0, 1, 2, 3));
}
SZ_API_COMPTIME sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length) {
    sz_u256_vec_t start_threshold_vec;
    start_threshold_vec.ymm = _mm256_set1_epi8((char)-65);
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;
    sz_u256_vec_t text_vec;
    while (length >= 32) {
        text_vec.ymm = _mm256_loadu_si256((__m256i const *)text_u8);
        sz_u32_t start_byte_mask = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpgt_epi8(text_vec.ymm, start_threshold_vec.ymm));
        char_count += _mm_popcnt_u32(start_byte_mask);
        text_u8 += 32;
        length -= 32;
    }
    char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u256_vec_t start_threshold_vec;
    start_threshold_vec.ymm = _mm256_set1_epi8((char)-65); 
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u256_vec_t text_vec;
    while (length >= 32) {
        text_vec.ymm = _mm256_loadu_si256((__m256i const *)text_u8);
        sz_u32_t start_byte_mask = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpgt_epi8(text_vec.ymm, start_threshold_vec.ymm));
        sz_size_t start_byte_count = _mm_popcnt_u32(start_byte_mask);
        if (n >= start_byte_count) {
            n -= start_byte_count;
            text_u8 += 32;
            length -= 32;
            continue;
        }
        sz_u32_t deposited_bits = _pdep_u32((sz_u32_t)1 << n, start_byte_mask);
        int byte_offset = (int)_tzcnt_u32(deposited_bits);
        return (sz_cptr_t)(text_u8 + byte_offset);
    }
    return sz_utf8_seek_serial((sz_cptr_t)text_u8, length, n);
}
#pragma region Shared SIMD leaf substrate
/** @brief  The decoded 64-byte window for the AVX2 backend. The 64 bytes live as two `__m256i` halves (`*_lo` =
 *          lanes [0, 32), `*_hi` = lanes [32, 64)); the per-lane byte-domain codepoint halves `high`/`low` share that
 *          shape. Masks are `sz_u64_t` (`vpmovmskb` per half, OR-combined) rather than the Ice Lake `__mmask64`. Field
 *          names and semantics match @ref sz_utf8_rune_window_t so the portable rule algebra is unchanged. */
typedef struct sz_utf8_rune_window_haswell_t {
    __m256i window_lo;          /**< Raw input bytes for lanes [0, 32). */
    __m256i window_hi;          /**< Raw input bytes for lanes [32, 64). */
    __m256i high_lo;            /**< Per-lane `codepoint >> 8` for lanes [0, 32). */
    __m256i high_hi;            /**< Per-lane `codepoint >> 8` for lanes [32, 64). */
    __m256i low_lo;             /**< Per-lane `codepoint & 0xFF` for lanes [0, 32). */
    __m256i low_hi;             /**< Per-lane `codepoint & 0xFF` for lanes [32, 64). */
    sz_u64_t continuation;      /**< Bit `i` => lane `i` is a continuation byte `10xxxxxx`. */
    sz_u64_t codepoint_starts;  /**< Bit `i` => lane `i` begins a codepoint (loaded, non-continuation). */
    sz_u64_t two_byte_starts;   /**< Bit `i` => lane `i` is a 2-byte lead `110xxxxx`. */
    sz_u64_t three_byte_starts; /**< Bit `i` => lane `i` is a 3-byte lead `1110xxxx`. */
    sz_u64_t four_byte_starts;  /**< Bit `i` => lane `i` is a 4-byte lead `11110xxx`. */
    sz_size_t loaded;           /**< Number of bytes actually loaded (<= 64). */
} sz_utf8_rune_window_haswell_t;
/** @brief  Per-byte logical right shift by @p shift keeping the low @p keep bits — the AVX2 twin of `srl8_`. */
SZ_HELPER_INLINE __m256i sz_utf8_srl8_haswell_(__m256i value_u8x32, int shift, sz_u8_t keep) {
    return _mm256_and_si256(_mm256_srli_epi16(value_u8x32, shift), _mm256_set1_epi8((char)keep));
}
/** @brief  Combine two per-half `vpmovmskb` results into one 64-bit lane mask. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_mask_combine_haswell_(__m256i low_half_u8x32, __m256i high_half_u8x32) {
    sz_u64_t const low_bits = (sz_u32_t)_mm256_movemask_epi8(low_half_u8x32);
    sz_u64_t const high_bits = (sz_u32_t)_mm256_movemask_epi8(high_half_u8x32);
    return low_bits | (high_bits << 32);
}
/** @brief  Masked 64-byte load into two halves; bytes [loaded, 64) read as zero (the AVX2 stand-in for
 *          `_mm512_maskz_loadu_epi8`). A small stack staging union covers the partial tail so we never read past
 *          `text + loaded`. */
SZ_HELPER_AUTO void sz_utf8_load_window_haswell_( 
    sz_u8_t const *text, sz_size_t loaded, __m256i *out_low_u8x32, __m256i *out_high_u8x32) {
    if (loaded >= 64) {
        *out_low_u8x32 = _mm256_loadu_si256((__m256i const *)(text + 0));
        *out_high_u8x32 = _mm256_loadu_si256((__m256i const *)(text + 32));
        return;
    }
    sz_u512_vec_t staging_vec;
    staging_vec.ymms[0] = _mm256_setzero_si256();
    staging_vec.ymms[1] = _mm256_setzero_si256();
    for (sz_size_t i = 0; i < loaded; ++i) staging_vec.u8s[i] = text[i];
    *out_low_u8x32 = staging_vec.ymms[0];
    *out_high_u8x32 = staging_vec.ymms[1];
}
/** @brief  Forward neighbours `next1[i] = window[i+1]`, `next2[i] = window[i+2]` over all 64 lanes, with the lanes
 *          past the window WRAPPING modulo 64 to match Ice Lake's `_mm512_permutexvar_epi8` (so `next1[63]==window[0]`,
 *          `next2[62]==window[0]`, `next2[63]==window[1]`). AVX2 has no 32-byte byte permute, so each 128-bit lane is
 *          fed its following bytes via `_mm256_permute2x128_si256` (to rotate in the successor 128-bit block, the
 *          window head wrapping in after `window_hi`) then `_mm256_alignr_epi8` to shift across the lane boundary. */
SZ_HELPER_INLINE void sz_utf8_forward_neighbours_haswell_( 
    __m256i window_low_u8x32, __m256i window_high_u8x32,   
    __m256i *next_byte_1_low_u8x32, __m256i *next_byte_1_high_u8x32, __m256i *next_byte_2_low_u8x32,
    __m256i *next_byte_2_high_u8x32) {
    __m256i const successor_low_u8x32 = _mm256_permute2x128_si256(window_low_u8x32, window_high_u8x32, 0x21);
    *next_byte_1_low_u8x32 = _mm256_alignr_epi8(successor_low_u8x32, window_low_u8x32, 1);
    *next_byte_2_low_u8x32 = _mm256_alignr_epi8(successor_low_u8x32, window_low_u8x32, 2);
    __m256i const successor_high_u8x32 = _mm256_permute2x128_si256(window_high_u8x32, window_low_u8x32, 0x21);
    *next_byte_1_high_u8x32 = _mm256_alignr_epi8(successor_high_u8x32, window_high_u8x32, 1);
    *next_byte_2_high_u8x32 = _mm256_alignr_epi8(successor_high_u8x32, window_high_u8x32, 2);
}
/** @brief  Expand a 32-bit lane mask into a 32-byte select vector (byte `i` = 0xFF when bit `i` is set) to drive
 *          `_mm256_blendv_epi8` in place of Ice Lake's `_mm512_mask_blend_epi8`. In-register: broadcast the mask,
 *          route the right mask byte to each output byte via `vpshufb`, isolate the per-lane bit, then `cmpeq`. */
SZ_HELPER_INLINE __m256i sz_utf8_byte_mask_from_bits_haswell_(sz_u32_t bits) {
    __m256i const mask_broadcast_u32x8 = _mm256_set1_epi32((int)bits);
    __m256i const byte_router_shuffle_u8x32 = _mm256_setr_epi8( 
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,         
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    __m256i const mask_spread_u8x32 = _mm256_shuffle_epi8(mask_broadcast_u32x8, byte_router_shuffle_u8x32);
    __m256i const bit_select_isolation_u8x32 = _mm256_setr_epi8(              
        1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128, 
        1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128);
    __m256i const bit_isolated_u8x32 = _mm256_and_si256(mask_spread_u8x32, bit_select_isolation_u8x32);
    return _mm256_cmpeq_epi8(bit_isolated_u8x32, bit_select_isolation_u8x32);
}
/** @brief  Load up to 64 bytes (masked tail) and decode every lane into byte-domain halves — the AVX2 twin of
 *          @ref sz_utf8_rune_decode_window_, bit-identical to it on every lane. */
SZ_HELPER_AUTO sz_utf8_rune_window_haswell_t sz_utf8_rune_decode_window_haswell_( 
    sz_u8_t const *text, sz_size_t available) {
    sz_utf8_rune_window_haswell_t result;
    result.loaded = available < 64 ? available : 64;
    __m256i window_bytes_low_u8x32, window_bytes_high_u8x32;
    sz_utf8_load_window_haswell_(text, result.loaded, &window_bytes_low_u8x32, &window_bytes_high_u8x32);
    result.window_lo = window_bytes_low_u8x32, result.window_hi = window_bytes_high_u8x32;
    __m256i next_byte_1_low_u8x32, next_byte_1_high_u8x32, next_byte_2_low_u8x32, next_byte_2_high_u8x32;
    sz_utf8_forward_neighbours_haswell_(window_bytes_low_u8x32, window_bytes_high_u8x32, &next_byte_1_low_u8x32,
                                        &next_byte_1_high_u8x32, &next_byte_2_low_u8x32, &next_byte_2_high_u8x32);
    sz_u64_t const loaded_mask = result.loaded >= 64 ? ~(sz_u64_t)0 : (((sz_u64_t)1 << result.loaded) - 1);
    __m256i const continuation_mask_u8x32 = _mm256_set1_epi8((char)0xC0);
    __m256i const is_continuation_low_u8x32 = _mm256_cmpeq_epi8(
        _mm256_and_si256(window_bytes_low_u8x32, continuation_mask_u8x32), _mm256_set1_epi8((char)0x80));
    __m256i const is_continuation_high_u8x32 = _mm256_cmpeq_epi8(
        _mm256_and_si256(window_bytes_high_u8x32, continuation_mask_u8x32), _mm256_set1_epi8((char)0x80));
    result.continuation = sz_utf8_mask_combine_haswell_(is_continuation_low_u8x32, is_continuation_high_u8x32) &
                          loaded_mask;
    result.codepoint_starts = loaded_mask & ~result.continuation;
    __m256i const two_byte_lead_mask_u8x32 = _mm256_set1_epi8((char)0xE0);
    result.two_byte_starts = sz_utf8_mask_combine_haswell_(
                                 _mm256_cmpeq_epi8(_mm256_and_si256(window_bytes_low_u8x32, two_byte_lead_mask_u8x32),
                                                   _mm256_set1_epi8((char)0xC0)),
                                 _mm256_cmpeq_epi8(_mm256_and_si256(window_bytes_high_u8x32, two_byte_lead_mask_u8x32),
                                                   _mm256_set1_epi8((char)0xC0))) &
                             loaded_mask;
    __m256i const three_byte_lead_mask_u8x32 = _mm256_set1_epi8((char)0xF0);
    result.three_byte_starts =
        sz_utf8_mask_combine_haswell_(
            _mm256_cmpeq_epi8(_mm256_and_si256(window_bytes_low_u8x32, three_byte_lead_mask_u8x32),
                              _mm256_set1_epi8((char)0xE0)),
            _mm256_cmpeq_epi8(_mm256_and_si256(window_bytes_high_u8x32, three_byte_lead_mask_u8x32),
                              _mm256_set1_epi8((char)0xE0))) &
        loaded_mask;
    __m256i const four_byte_lead_mask_u8x32 = _mm256_set1_epi8((char)0xF8);
    result.four_byte_starts = sz_utf8_mask_combine_haswell_(
                                  _mm256_cmpeq_epi8(_mm256_and_si256(window_bytes_low_u8x32, four_byte_lead_mask_u8x32),
                                                    _mm256_set1_epi8((char)0xF0)),
                                  _mm256_cmpeq_epi8(
                                      _mm256_and_si256(window_bytes_high_u8x32, four_byte_lead_mask_u8x32),
                                      _mm256_set1_epi8((char)0xF0))) &
                              loaded_mask;
    __m256i const mask_five_low_bits_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const mask_six_low_bits_u8x32 = _mm256_set1_epi8(0x3F);
    __m256i const mask_two_low_bits_u8x32 = _mm256_set1_epi8(0x03);
    __m256i const mask_four_low_bits_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const high_two_byte_low_u8x32 = sz_utf8_srl8_haswell_(
        _mm256_and_si256(window_bytes_low_u8x32, mask_five_low_bits_u8x32), 2, 0x07);
    __m256i const high_two_byte_high_u8x32 = sz_utf8_srl8_haswell_(
        _mm256_and_si256(window_bytes_high_u8x32, mask_five_low_bits_u8x32), 2, 0x07);
    __m256i const low_two_byte_low_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(window_bytes_low_u8x32, mask_two_low_bits_u8x32), 6),
        _mm256_and_si256(next_byte_1_low_u8x32, mask_six_low_bits_u8x32));
    __m256i const low_two_byte_high_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(window_bytes_high_u8x32, mask_two_low_bits_u8x32), 6),
        _mm256_and_si256(next_byte_1_high_u8x32, mask_six_low_bits_u8x32));
    __m256i const high_three_byte_low_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(window_bytes_low_u8x32, mask_four_low_bits_u8x32), 4),
        sz_utf8_srl8_haswell_(next_byte_1_low_u8x32, 2, 0x0F));
    __m256i const high_three_byte_high_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(window_bytes_high_u8x32, mask_four_low_bits_u8x32), 4),
        sz_utf8_srl8_haswell_(next_byte_1_high_u8x32, 2, 0x0F));
    __m256i const low_three_byte_low_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(next_byte_1_low_u8x32, mask_two_low_bits_u8x32), 6),
        _mm256_and_si256(next_byte_2_low_u8x32, mask_six_low_bits_u8x32));
    __m256i const low_three_byte_high_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(next_byte_1_high_u8x32, mask_two_low_bits_u8x32), 6),
        _mm256_and_si256(next_byte_2_high_u8x32, mask_six_low_bits_u8x32));
    __m256i const is_three_byte_low_select_u8x32 = sz_utf8_byte_mask_from_bits_haswell_(
        (sz_u32_t)(result.three_byte_starts & 0xFFFFFFFFu));
    __m256i const is_three_byte_high_select_u8x32 = sz_utf8_byte_mask_from_bits_haswell_(
        (sz_u32_t)((result.three_byte_starts >> 32) & 0xFFFFFFFFu));
    result.high_lo = _mm256_blendv_epi8(high_two_byte_low_u8x32, high_three_byte_low_u8x32,
                                        is_three_byte_low_select_u8x32);
    result.high_hi = _mm256_blendv_epi8(high_two_byte_high_u8x32, high_three_byte_high_u8x32,
                                        is_three_byte_high_select_u8x32);
    result.low_lo = _mm256_blendv_epi8(low_two_byte_low_u8x32, low_three_byte_low_u8x32,
                                       is_three_byte_low_select_u8x32);
    result.low_hi = _mm256_blendv_epi8(low_two_byte_high_u8x32, low_three_byte_high_u8x32,
                                       is_three_byte_high_select_u8x32);
    return result;
}
/** @brief  One nibble-cascade stage with a sub-256 selector: `result[lane] = table[selector[lane]*16 + within[lane]]`.
 *          Each of @p tile_count 16-byte rows is one selector value; @p within_u8x32 is the addressing nibble
 *          (caller-masked to `[0,16)`). The AVX2 stand-in for VBMI `vpermi2b`: there is no 32-byte byte permute, so
 *          each resident row is broadcast and shuffled by @p within_u8x32, then blended in for the lanes whose
 *          @p selector_u8x32 picks that row. In-register — only `vbroadcasti128`/`vpshufb`/`vpcmpeqb`/`vpblendvb`.
 *
 *  ! Cost scales with @p tile_count, not with the window: every resident row is shuffled and blended on the shuffle
 *  ! port. The BMP classifiers use @ref sz_utf8_rune_flat_lookup_haswell_ instead for that reason. */
SZ_HELPER_AUTO __m256i sz_utf8_rune_cascade_stage_haswell_( 
    sz_u8_t const *table, int tile_count, __m256i selector_u8x32, __m256i within_u8x32) {
    __m256i result_u8x32 = _mm256_setzero_si256();
    for (int tile = 0; tile < tile_count; ++tile) {
        __m256i const lut_row_broadcast_u8x32 = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((__m128i const *)(table + tile * 16)));
        __m256i const lut_row_picked_u8x32 = _mm256_shuffle_epi8(lut_row_broadcast_u8x32, within_u8x32);
        __m256i const selector_match_u8x32 = _mm256_cmpeq_epi8(selector_u8x32, _mm256_set1_epi8((char)tile));
        result_u8x32 = _mm256_blendv_epi8(result_u8x32, lut_row_picked_u8x32, selector_match_u8x32);
    }
    return result_u8x32;
}
/** @brief  256-entry byte LUT addressed by a per-lane byte index in `[0,256)`: `result[lane] = group_base[index[lane]]`.
 *          Two-stage `vpshufb` over 16 resident rows — the low nibble shuffles within a row, the high nibble selects
 *          the row. The AVX2 twin of the substrate `lut256` leaf. */
SZ_HELPER_INLINE __m256i sz_utf8_rune_lut256_haswell_(sz_u8_t const *group_base, __m256i index_u8x32) {
    __m256i const index_within_low_u8x32 = _mm256_and_si256(index_u8x32, _mm256_set1_epi8(0x0F));
    __m256i const index_selector_high_u8x32 = _mm256_and_si256(_mm256_srli_epi16(index_u8x32, 4),
                                                               _mm256_set1_epi8(0x0F));
    return sz_utf8_rune_cascade_stage_haswell_(group_base, 16, index_selector_high_u8x32, index_within_low_u8x32);
}
/** @brief  64-entry byte LUT addressed by a per-lane index in `[0,64)`: a four-row cascade over the 64-byte table.
 *          The bounded twin of @ref sz_utf8_rune_lut256_haswell_ for callers whose table is only four rows wide -
 *          e.g. a 128-entry property table read as two 64-byte halves - so the loads never run past the array, and
 *          only a quarter of the shuffle/blend work is issued. @p group_base must point to at least 64 valid bytes. */
SZ_HELPER_INLINE __m256i sz_utf8_rune_lut64_haswell_(sz_u8_t const *group_base, __m256i index_u8x32) {
    __m256i const index_within_low_u8x32 = _mm256_and_si256(index_u8x32, _mm256_set1_epi8(0x0F));
    __m256i const index_selector_high_u8x32 = _mm256_and_si256(_mm256_srli_epi16(index_u8x32, 4),
                                                               _mm256_set1_epi8(0x03));
    return sz_utf8_rune_cascade_stage_haswell_(group_base, 4, index_selector_high_u8x32, index_within_low_u8x32);
}
/** @brief  16-entry byte LUT addressed by a per-lane index in `[0,16)`: one broadcast row and one `vpshufb`, with
 *          no cascade at all, since a single row covers the whole index range. The bounded twin of
 *          @ref sz_utf8_rune_lut256_haswell_ for nibble-wide tables. @p group_base must point to at least 16 valid
 *          bytes, and @p index_u8x32 must have bit 7 clear on every lane or `vpshufb` zeroes that lane. */
SZ_HELPER_INLINE __m256i sz_utf8_rune_lut16_haswell_(sz_u8_t const *group_base, __m256i index_u8x32) {
    __m256i const lut_row_broadcast_u8x32 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)group_base));
    return _mm256_shuffle_epi8(lut_row_broadcast_u8x32, index_u8x32);
}
/** @brief  Narrow four `u32x8` vectors, each carrying one class byte in the low byte of every dword, into a single
 *          `u8x32` in ascending lane order. The companion of the flat-leaf `vpgatherdd`, which resolves eight lanes
 *          per gather and so needs four gathers per 32-lane window. `packus` saturates per 128-bit half, leaving the
 *          dwords interleaved, so a final `vpermd` restores the order. */
SZ_HELPER_INLINE __m256i sz_utf8_rune_pack4_u32_to_u8_haswell_( 
    __m256i first_u32x8, __m256i second_u32x8, __m256i third_u32x8, __m256i fourth_u32x8) {
    __m256i const first_second_u16x16 = _mm256_packus_epi32(first_u32x8, second_u32x8);
    __m256i const third_fourth_u16x16 = _mm256_packus_epi32(third_u32x8, fourth_u32x8);
    __m256i const interleaved_u8x32 = _mm256_packus_epi16(first_second_u16x16, third_fourth_u16x16);
    return _mm256_permutevar8x32_epi32(interleaved_u8x32, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}
/** @brief  Class byte per lane from a page-compressed flat table: `page_lut[high]` selects a 256-byte page, then
 *          `flat[page * 256 + low]` is fetched with four `vpgatherdd`. Unlike the nibble cascade this scales with the
 *          window, not the table: the cascade scanned every 16-byte tile of every stage on the shuffle port, while the
 *          gather issues on the load ports and leaves the shuffle port to the decode. @p flat must extend four bytes
 *          past its last index, since the dword gather over-reads three. */
SZ_HELPER_AUTO __m256i sz_utf8_rune_flat_lookup_haswell_( 
    sz_u8_t const *page_lut, sz_u8_t const *flat, __m256i high_bytes_u8x32, __m256i low_bytes_u8x32) {
    __m256i const page_indices_u8x32 = sz_utf8_rune_lut256_haswell_(page_lut, high_bytes_u8x32);
    __m128i const page_indices_low_u8x16 = _mm256_castsi256_si128(page_indices_u8x32);
    __m128i const page_indices_high_u8x16 = _mm256_extracti128_si256(page_indices_u8x32, 1);
    __m128i const low_bytes_low_u8x16 = _mm256_castsi256_si128(low_bytes_u8x32);
    __m128i const low_bytes_high_u8x16 = _mm256_extracti128_si256(low_bytes_u8x32, 1);
    __m128i const page_quarters_u8x16[4] = {page_indices_low_u8x16, _mm_srli_si128(page_indices_low_u8x16, 8),
                                            page_indices_high_u8x16, _mm_srli_si128(page_indices_high_u8x16, 8)};
    __m128i const low_quarters_u8x16[4] = {low_bytes_low_u8x16, _mm_srli_si128(low_bytes_low_u8x16, 8),
                                           low_bytes_high_u8x16, _mm_srli_si128(low_bytes_high_u8x16, 8)};
    __m256i gathered_u32x8[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        __m256i const flat_indices_u32x8 = _mm256_add_epi32(
            _mm256_slli_epi32(_mm256_cvtepu8_epi32(page_quarters_u8x16[quarter]), 8),
            _mm256_cvtepu8_epi32(low_quarters_u8x16[quarter]));
        gathered_u32x8[quarter] = _mm256_and_si256(_mm256_i32gather_epi32((int const *)flat, flat_indices_u32x8, 1),
                                                   _mm256_set1_epi32(0xFF));
    }
    return sz_utf8_rune_pack4_u32_to_u8_haswell_(gathered_u32x8[0], gathered_u32x8[1], gathered_u32x8[2],
                                                 gathered_u32x8[3]);
}
#pragma endregion Shared SIMD leaf substrate
#pragma region Drains
/** @brief  Left-pack the set lane indices (in [0, 64), ascending) of a 64-bit @p mask into @p out[0..popcount).
 *          BMI2 path: `tzcnt` pulls the lowest set bit's index, `blsr` clears it. On Intel Haswell `tzcnt`/`blsr`
 *          are single-uop and beat a `vpshufb` left-pack LUT; the LUT only wins where `pext`/`blsr` is microcoded
 *          (AMD pre-Zen3). */
SZ_HELPER_AUTO void sz_utf8_unpack_indices_haswell_(sz_u64_t mask, sz_u8_t *out) {
    while (mask) {
        *out++ = (sz_u8_t)(int)_tzcnt_u64(mask);
        mask = _blsr_u64(mask); 
    }
}
/** @brief  AVX2 forward drain — the `vpcompressb`-free twin of @ref sz_utf8_rune_drain_forward_. Emits one
 *          (start, length) per set boundary lane (ascending), honoring @p capacity and the carried previous-boundary
 *          via @p previous_io; bit-exact with the Ice Lake leaf. The set lanes are index-unpacked once (BMI2), then
 *          streamed in waves of four u64 positions (`vpmovzxbq` widen + `base`, segment starts via `vpermq` shift +
 *          `vpblendd` carry-seat, lengths via `vpsubq`), with a scalar tail for the final partial wave (no AVX2
 *          masked store). */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_forward_haswell_( 
    sz_u64_t boundary, sz_size_t base, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *previous_io) {
    sz_size_t const boundary_count = (sz_size_t)_mm_popcnt_u64(boundary);
    sz_u64_t previous = (sz_u64_t)*previous_io;
    if (boundary_count == 0 || produced >= capacity) {
        *previous_io = (sz_size_t)previous;
        return produced;
    }
    sz_u8_t indices[64 + 8];
    sz_utf8_unpack_indices_haswell_(boundary, indices);
    _mm_storel_epi64((__m128i *)(indices + 64), _mm_setzero_si128());
    __m256i const base_address_u64x4 = _mm256_set1_epi64x((long long)base);
    sz_size_t emitted = 0;
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const wave = sz_min_of_three(boundary_count - emitted, capacity - produced, 4);
        __m128i const indices_bytes_u8x16 = _mm_loadl_epi64((__m128i const *)(indices + emitted));
        __m256i const positions_u64x4 = _mm256_add_epi64(_mm256_cvtepu8_epi64(indices_bytes_u8x16), base_address_u64x4);
        __m256i const previous_start_u64x4 = _mm256_set1_epi64x((long long)previous);
        __m256i const positions_shifted_u64x4 = _mm256_permute4x64_epi64(positions_u64x4, _MM_SHUFFLE(2, 1, 0, 0));
        __m256i const segment_starts_u64x4 = _mm256_blend_epi32(positions_shifted_u64x4, previous_start_u64x4, 0x03);
        __m256i const segment_lengths_u64x4 = _mm256_sub_epi64(positions_u64x4, segment_starts_u64x4);
        if (produced + 4 <= capacity && wave == 4) {
            _mm256_storeu_si256((__m256i *)(starts + produced), segment_starts_u64x4);
            _mm256_storeu_si256((__m256i *)(lengths + produced), segment_lengths_u64x4);
        }
        else {
            sz_size_t tail_starts[4], tail_lengths[4];
            _mm256_storeu_si256((__m256i *)tail_starts, segment_starts_u64x4);
            _mm256_storeu_si256((__m256i *)tail_lengths, segment_lengths_u64x4);
            for (sz_size_t lane = 0; lane < wave; ++lane)
                starts[produced + lane] = tail_starts[lane], lengths[produced + lane] = tail_lengths[lane];
        }
        produced += wave, emitted += wave;
        previous = (sz_u64_t)(starts[produced - 1] + lengths[produced - 1]);
    }
    *previous_io = (sz_size_t)previous;
    return produced;
}
#pragma endregion Drains
#pragma region Total vectorized decode
/** @brief  Low @p count bits set within a 32-bit lane mask (0 for `count==0`, all-ones for `count>=32`). The 32-bit
 *          twin of @ref sz_u64_mask_until_serial_, used to bound a sub-window of an AVX2 32-byte window. */
SZ_HELPER_INLINE sz_u32_t sz_u32_mask_until_serial_(sz_size_t count) {
    return count >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << count) - 1u);
}
/**
 *  @brief  Gather one byte per 32-bit lane from a 32-byte window addressed by the 8 per-dword byte @p offsets_u32x8
 *          (each in `[0, 32)`), entirely in-register (NO `vpgatherdd`). The window's two 16-byte halves are each
 *          broadcast into both 128-bit lanes (@p window_dup_lo_u8x32 / @p window_dup_hi_u8x32); a `vpshufb` per half
 *          routes `window[offset & 15]` into the low byte of every dword, and the high offset bit blends the two
 *          halves. Lanes pointing past lane 31 read the wrapped low half, but the caller only consults lanes whose
 *          offset is a real emitted start.
 */
SZ_HELPER_INLINE __m256i sz_utf8_rune_gather8_window_haswell_( 
    __m256i window_dup_lo_u8x32, __m256i window_dup_hi_u8x32, __m256i offsets_u32x8) {
    __m256i const offset_within_u32x8 = _mm256_and_si256(offsets_u32x8, _mm256_set1_epi32(0x0F));
    __m256i const shuffle_control_u8x32 = _mm256_or_si256(offset_within_u32x8, _mm256_set1_epi32((int)0x80808000u));
    __m256i const window_low_picked_u8x32 = _mm256_shuffle_epi8(window_dup_lo_u8x32, shuffle_control_u8x32);
    __m256i const window_high_picked_u8x32 = _mm256_shuffle_epi8(window_dup_hi_u8x32, shuffle_control_u8x32);
    __m256i const offset_high_bit_select_u32x8 = _mm256_cmpeq_epi32(
        _mm256_and_si256(offsets_u32x8, _mm256_set1_epi32(0x10)), _mm256_set1_epi32(0x10));
    return _mm256_blendv_epi8(window_low_picked_u8x32, window_high_picked_u8x32, offset_high_bit_select_u32x8);
}
/**
 *  @brief  Left-pack the set-bit positions of a 32-bit lane @p mask into @p out as a dense, ascending array of
 *          byte-offsets in [0, 32), returning the count - the AVX2 `vpcompressb`-free start-compaction shared with the
 *          NEON / LASX / PowerVSX backends. Replaces a scalar `ctz` walk with a 2 KB shuffle-LUT (`leftpack8`) keyed by
 *          each 8-bit sub-mask: for every 16-lane half the mask splits into a low and a high byte, each `vpshufb` over
 *          the LUT row of its set-bit positions, the high half offset by +8, the two stitched at `popcount(low8)` via a
 *          gap-shift `vpshufb` (no scalar per-lane index walk). The half offset `h*16` is added in vector; one loop over
 *          the two halves.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_leftpack_offsets_haswell_(sz_u32_t mask, sz_u8_t *out) {
    static sz_u8_t const leftpack8[256 * 8] = {
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 
        0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x04, 0x80, 0x80, 0x80, 0x80, 
        0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x04, 0x80, 0x80, 0x80, 
        0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x05, 0x80, 0x80, 0x80, 0x80, 
        0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x05, 0x80, 0x80, 0x80, 
        0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x04, 0x05, 0x80, 0x80, 0x80, 
        0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x80, 0x80, 
        0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x06, 0x80, 0x80, 0x80, 
        0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x04, 0x06, 0x80, 0x80, 0x80, 
        0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x04, 0x06, 0x80, 0x80, 
        0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x05, 0x06, 0x80, 0x80, 0x80, 
        0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x80, 0x80, 
        0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 
        0x02, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x80, 0x80, 
        0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 
        0x02, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x80, 
        0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x07, 0x80, 0x80, 0x80, 
        0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x04, 0x07, 0x80, 0x80, 0x80, 
        0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x04, 0x07, 0x80, 0x80, 
        0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x05, 0x07, 0x80, 0x80, 0x80, 
        0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x05, 0x07, 0x80, 0x80, 
        0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 
        0x02, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x04, 0x05, 0x07, 0x80, 0x80, 
        0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 
        0x02, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x80, 
        0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x02, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x02, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x06, 0x07, 0x80, 0x80, 
        0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x02, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x04, 0x06, 0x07, 0x80, 0x80, 
        0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 
        0x02, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x04, 0x06, 0x07, 0x80, 
        0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x01, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x02, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x01, 0x02, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x05, 0x06, 0x07, 0x80, 0x80, 
        0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x01, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 
        0x02, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 
        0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x80, 
        0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 
        0x00, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x01, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x01, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 
        0x02, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x02, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 
        0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 
        0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x80, 
        0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 
        0x00, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 
        0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 
        0x00, 0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 
        0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 
        0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 
    };
    __m128i const lane_iota_u8x16 = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m128i const constant_eight_u8x16 = _mm_set1_epi8(8);
    sz_size_t produced = 0;
    for (int half = 0; half < 2; ++half) {
        sz_u32_t const half_mask = (mask >> (half * 16)) & 0xFFFFu;
        if (half_mask == 0) continue;
        sz_u32_t const low8 = half_mask & 0xFFu;
        sz_u32_t const high8 = (half_mask >> 8) & 0xFFu;
        int const count_low = (int)_mm_popcnt_u64(low8);
        int const count = (int)_mm_popcnt_u64(half_mask);
        __m128i const packed_offsets_low_u8x16 = _mm_loadl_epi64((__m128i const *)(leftpack8 + low8 * 8));
        __m128i const packed_offsets_high_raw_u8x16 = _mm_loadl_epi64((__m128i const *)(leftpack8 + high8 * 8));
        __m128i const packed_offsets_high_adjusted_u8x16 = _mm_add_epi8(packed_offsets_high_raw_u8x16,
                                                                        _mm_set1_epi8(8));
        __m128i const packed_halves_u8x16 = _mm_unpacklo_epi64(packed_offsets_low_u8x16,
                                                               packed_offsets_high_adjusted_u8x16);
        __m128i const greater_equal_count_u8x16 = _mm_cmpgt_epi8(lane_iota_u8x16, _mm_set1_epi8((char)(count_low - 1)));
        __m128i const gap_adjustment_u8x16 = _mm_and_si128(
            greater_equal_count_u8x16, _mm_sub_epi8(constant_eight_u8x16, _mm_set1_epi8((char)count_low)));
        __m128i const stitch_source_u8x16 = _mm_add_epi8(lane_iota_u8x16, gap_adjustment_u8x16);
        __m128i const stitched_offsets_u8x16 = _mm_shuffle_epi8(packed_halves_u8x16, stitch_source_u8x16);
        __m128i const global_offsets_u8x16 = _mm_add_epi8(stitched_offsets_u8x16, _mm_set1_epi8((char)(half * 16)));
        sz_u128_vec_t lanes_vec;
        _mm_storeu_si128((__m128i *)lanes_vec.u8s, global_offsets_u8x16);
        for (int lane = 0; lane < count; ++lane) out[produced + (sz_size_t)lane] = lanes_vec.u8s[lane];
        produced += (sz_size_t)count;
    }
    return produced;
}
/**
 *  @brief  Decode the dense emitted-start lanes of one classified AVX2 window into sequential UTF-32 runes, the
 *          `vpcompressb`-free twin of @ref sz_utf8_rune_drain_icelake_. The set lanes of @p emit_starts are left-packed
 *          once into a dense ascending byte-offset array (BMI2 `tzcnt`/`blsr`, @ref sz_utf8_unpack_indices_haswell_),
 *          and each block of up to 8 starts loads its offsets to dwords, gathers the lead and trailing bytes from the
 *          in-register window, width-blends 1/2/3/4-byte lanes branchlessly, and — only when @p has_ill — overwrites
 *          ill-formed lanes with U+FFFD (their per-emitted-lane flag gathered with the SAME packed offsets out of
 *          @p ill_formed_lanes_u8x32).
 *  @return Number of runes emitted; sets @p last_off_out to the last emitted start's window byte-offset (the caller
 *          turns it into the resume cursor by adding that lane's maximal-subpart length).
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_haswell_(                           
    __m256i window_u8x32, __m256i ill_formed_lanes_u8x32, sz_u32_t emit_starts, 
    int has_three, int has_four, int has_ill,                                   
    sz_size_t emit_count, sz_rune_t *runes, sz_size_t capacity, sz_u8_t *last_off_out) {
    sz_u512_vec_t offsets_vec;
    sz_utf8_unpack_indices_haswell_(emit_starts, offsets_vec.u8s);
    __m256i const window_low_broadcast_u8x32 = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(window_u8x32));
    __m256i const window_high_broadcast_u8x32 = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(window_u8x32, 1));
    sz_size_t const want = emit_count < capacity ? emit_count : capacity;
    sz_size_t produced = 0;
    for (sz_size_t block_start = 0; block_start < want; block_start += 8) {
        __m128i const offsets_bytes_u8x16 = _mm_loadl_epi64((__m128i const *)(offsets_vec.u8s + block_start));
        __m256i const offsets_u32x8 = _mm256_cvtepu8_epi32(offsets_bytes_u8x16);
        __m256i const lead_byte_u32x8 = sz_utf8_rune_gather8_window_haswell_(
            window_low_broadcast_u8x32, window_high_broadcast_u8x32, offsets_u32x8);
        __m256i const second_byte_u32x8 = sz_utf8_rune_gather8_window_haswell_(
            window_low_broadcast_u8x32, window_high_broadcast_u8x32,
            _mm256_add_epi32(offsets_u32x8, _mm256_set1_epi32(1)));
        __m256i const is_greater_equal_0xc0_u32x8 = _mm256_cmpgt_epi32(lead_byte_u32x8, _mm256_set1_epi32(0xC0 - 1));
        __m256i const is_greater_equal_0xe0_u32x8 = _mm256_cmpgt_epi32(lead_byte_u32x8, _mm256_set1_epi32(0xE0 - 1));
        __m256i const is_greater_equal_0xf0_u32x8 = _mm256_cmpgt_epi32(lead_byte_u32x8, _mm256_set1_epi32(0xF0 - 1));
        __m256i const is_two_byte_u32x8 = _mm256_andnot_si256(is_greater_equal_0xe0_u32x8, is_greater_equal_0xc0_u32x8);
        __m256i const is_three_byte_u32x8 = _mm256_andnot_si256(is_greater_equal_0xf0_u32x8,
                                                                is_greater_equal_0xe0_u32x8);
        __m256i const two_byte_codepoints_u32x8 = _mm256_or_si256(
            _mm256_slli_epi32(_mm256_and_si256(lead_byte_u32x8, _mm256_set1_epi32(0x1F)), 6),
            _mm256_and_si256(second_byte_u32x8, _mm256_set1_epi32(0x3F)));
        __m256i codepoints_u32x8 = _mm256_blendv_epi8(lead_byte_u32x8, two_byte_codepoints_u32x8, is_two_byte_u32x8);
        __m256i third_byte_u32x8 = _mm256_setzero_si256(); 
        if (has_three) {
            third_byte_u32x8 = sz_utf8_rune_gather8_window_haswell_(
                window_low_broadcast_u8x32, window_high_broadcast_u8x32,
                _mm256_add_epi32(offsets_u32x8, _mm256_set1_epi32(2)));
            __m256i const three_byte_codepoints_u32x8 = _mm256_or_si256(
                _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(lead_byte_u32x8, _mm256_set1_epi32(0x0F)), 12),
                                _mm256_slli_epi32(_mm256_and_si256(second_byte_u32x8, _mm256_set1_epi32(0x3F)), 6)),
                _mm256_and_si256(third_byte_u32x8, _mm256_set1_epi32(0x3F)));
            codepoints_u32x8 = _mm256_blendv_epi8(codepoints_u32x8, three_byte_codepoints_u32x8, is_three_byte_u32x8);
        }
        if (has_four) { 
            __m256i const fourth_byte_u32x8 = sz_utf8_rune_gather8_window_haswell_(
                window_low_broadcast_u8x32, window_high_broadcast_u8x32,
                _mm256_add_epi32(offsets_u32x8, _mm256_set1_epi32(3)));
            __m256i const four_byte_codepoints_u32x8 = _mm256_or_si256(
                _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(lead_byte_u32x8, _mm256_set1_epi32(0x07)), 18),
                                _mm256_slli_epi32(_mm256_and_si256(second_byte_u32x8, _mm256_set1_epi32(0x3F)), 12)),
                _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(third_byte_u32x8, _mm256_set1_epi32(0x3F)), 6),
                                _mm256_and_si256(fourth_byte_u32x8, _mm256_set1_epi32(0x3F))));
            codepoints_u32x8 = _mm256_blendv_epi8(codepoints_u32x8, four_byte_codepoints_u32x8,
                                                  is_greater_equal_0xf0_u32x8);
        }
        if (has_ill) {
            __m256i const ill_formed_low_broadcast_u8x32 = _mm256_broadcastsi128_si256(
                _mm256_castsi256_si128(ill_formed_lanes_u8x32));
            __m256i const ill_formed_high_broadcast_u8x32 = _mm256_broadcastsi128_si256(
                _mm256_extracti128_si256(ill_formed_lanes_u8x32, 1));
            __m256i const ill_formed_flags_u32x8 = sz_utf8_rune_gather8_window_haswell_(
                ill_formed_low_broadcast_u8x32, ill_formed_high_broadcast_u8x32, offsets_u32x8);
            __m256i const ill_formed_detected_u32x8 = _mm256_cmpgt_epi32(ill_formed_flags_u32x8,
                                                                         _mm256_setzero_si256());
            codepoints_u32x8 = _mm256_blendv_epi8(codepoints_u32x8, _mm256_set1_epi32((int)sz_rune_replacement_k),
                                                  ill_formed_detected_u32x8);
        }
        sz_size_t lanes = want - produced;
        if (lanes > 8) lanes = 8;
        if (lanes == 8) { _mm256_storeu_si256((__m256i *)(runes + produced), codepoints_u32x8); }
        else {
            sz_rune_t tail[8];
            _mm256_storeu_si256((__m256i *)tail, codepoints_u32x8);
            for (sz_size_t lane = 0; lane < lanes; ++lane) runes[produced + lane] = tail[lane];
        }
        produced += lanes;
    }
    *last_off_out = offsets_vec.u8s[produced - 1];
    return produced;
}
/**
 *  @brief  Decode one 32-byte window of @p text into dense UTF-32 @p runes by the uniform "classify → per-lane
 *          well-formed + orphan promotion → compact emitted starts → in-register gather → width-blend → blend
 *          U+FFFD" path - the AVX2 twin of @ref sz_utf8_decode_once_icelake_, bit-exact with the serial
 *          reference (one U+FFFD per maximal ill-formed subpart, Unicode 17.0 §3.9 / W3C). Pure-ASCII windows take
 *          a `vpmovzxbd` widen lane. The step declines (`*runes_unpacked == 0`, cursor unchanged) ONLY when the
 *          first lead's declared sequence crosses the window edge (a boundary truncation), which the public entry
 *          finalizes without a serial re-decode. The decode is TOTAL: no decline-to-serial.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_decode_once_haswell_( 
    sz_cptr_t text, sz_size_t length,                  
    sz_rune_t *runes, sz_size_t runes_capacity,        
    sz_size_t *runes_unpacked) {
    sz_size_t const chunk = length < 32 ? length : 32;
    sz_u32_t const load_mask = sz_u32_mask_until_serial_(chunk);
    __m256i window_u8x32; 
    if (chunk >= 32) { window_u8x32 = _mm256_loadu_si256((__m256i const *)text); }
    else {
        sz_u256_vec_t staging_vec;
        staging_vec.ymm = _mm256_setzero_si256();
        for (sz_size_t i = 0; i < chunk; ++i) staging_vec.u8s[i] = (sz_u8_t)text[i];
        window_u8x32 = staging_vec.ymm;
    }
    if (((sz_u32_t)_mm256_movemask_epi8(window_u8x32) & load_mask) == 0) {
        sz_size_t const runes_to_unpack = chunk < runes_capacity ? chunk : runes_capacity;
        __m128i const window_low_half_u8x16 = _mm256_castsi256_si128(window_u8x32);
        __m128i const window_high_half_u8x16 = _mm256_extracti128_si256(window_u8x32, 1);
        sz_rune_t ascii_runes[32];
        _mm256_storeu_si256((__m256i *)(ascii_runes + 0), _mm256_cvtepu8_epi32(window_low_half_u8x16));
        _mm256_storeu_si256((__m256i *)(ascii_runes + 8),
                            _mm256_cvtepu8_epi32(_mm_srli_si128(window_low_half_u8x16, 8)));
        _mm256_storeu_si256((__m256i *)(ascii_runes + 16), _mm256_cvtepu8_epi32(window_high_half_u8x16));
        _mm256_storeu_si256((__m256i *)(ascii_runes + 24),
                            _mm256_cvtepu8_epi32(_mm_srli_si128(window_high_half_u8x16, 8)));
        for (sz_size_t i = 0; i < runes_to_unpack; ++i) runes[i] = ascii_runes[i];
        *runes_unpacked = runes_to_unpack;
        return text + runes_to_unpack;
    }
    sz_u32_t const starts_bits =
        (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(window_u8x32, _mm256_set1_epi8((char)-65))) & load_mask;
    sz_u32_t const continuation_bits = load_mask & ~starts_bits;
    {
        __m256i const signed_bias_u8x32 = _mm256_set1_epi8((char)0x80);
        __m256i const window_biased_u8x32 = _mm256_add_epi8(window_u8x32, signed_bias_u8x32);
        sz_u32_t const four_or_invalid_bits = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(
                                                  window_biased_u8x32,
                                                  _mm256_add_epi8(_mm256_set1_epi8((char)0xEF), signed_bias_u8x32))) &
                                              load_mask;
        if (four_or_invalid_bits == 0) {
            sz_u32_t const high_bit_bits = (sz_u32_t)_mm256_movemask_epi8(window_u8x32) & load_mask;
            sz_u32_t const ascii_bits = load_mask & ~high_bit_bits;
            sz_u32_t const lead2_bits =
                (sz_u32_t)_mm256_movemask_epi8(_mm256_and_si256(
                    _mm256_cmpgt_epi8(window_biased_u8x32,
                                      _mm256_add_epi8(_mm256_set1_epi8((char)0xC1), signed_bias_u8x32)),
                    _mm256_cmpgt_epi8(_mm256_add_epi8(_mm256_set1_epi8((char)0xE0), signed_bias_u8x32),
                                      window_biased_u8x32))) &
                load_mask;
            sz_u32_t const lead3_bits = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                                            _mm256_and_si256(window_u8x32, _mm256_set1_epi8((char)0xF0)),
                                            _mm256_set1_epi8((char)0xE0))) &
                                        load_mask;
            sz_u32_t const owe1_bits = lead2_bits | lead3_bits; 
            sz_size_t decodable_end = chunk;
            if (chunk && ((owe1_bits >> (chunk - 1)) & 1u)) { decodable_end = chunk - 1; }
            else if (chunk >= 2 && ((lead3_bits >> (chunk - 2)) & 1u)) { decodable_end = chunk - 2; }
            sz_u32_t const decodable = sz_u32_mask_until_serial_(decodable_end);
            sz_u32_t const starts_decodable = starts_bits & decodable;
            sz_u32_t const cont_decodable = continuation_bits & decodable;
            sz_u32_t const owe1_decodable = owe1_bits & decodable;
            sz_u32_t const lead3_decodable = lead3_bits & decodable;
            sz_u32_t const covered_bits = (owe1_bits << 1) | (lead3_bits << 2);
            int const clean_has_three = lead3_bits != 0;
            sz_u32_t bounds_bad = 0;
            if (clean_has_three) {
                sz_u32_t const e0_bits = (sz_u32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(window_u8x32, _mm256_set1_epi8((char)0xE0)));
                sz_u32_t const ed_bits = (sz_u32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(window_u8x32, _mm256_set1_epi8((char)0xED)));
                sz_u32_t const next_lt_0xa0_bits = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(
                    _mm256_add_epi8(_mm256_set1_epi8((char)0xA0), signed_bias_u8x32), window_biased_u8x32));
                sz_u32_t const next_ge_0xa0_bits = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(
                    window_biased_u8x32, _mm256_add_epi8(_mm256_set1_epi8((char)0x9F), signed_bias_u8x32)));
                bounds_bad = ((e0_bits & (next_lt_0xa0_bits >> 1)) | (ed_bits & (next_ge_0xa0_bits >> 1))) & decodable;
            }
            int const clean = ((starts_decodable & ~(ascii_bits | lead2_bits | lead3_bits)) == 0) &&
                              ((owe1_decodable & ~(continuation_bits >> 1)) == 0) &&
                              ((lead3_decodable & ~(continuation_bits >> 2)) == 0) &&
                              ((cont_decodable & ~covered_bits) == 0) && (bounds_bad == 0);
            if (clean && starts_decodable) {
                sz_size_t const emit_count = (sz_size_t)_mm_popcnt_u32(starts_decodable);
                sz_u8_t last_off = 0;
                sz_size_t const produced = sz_utf8_rune_drain_haswell_(window_u8x32, _mm256_setzero_si256(),
                                                                       starts_decodable, clean_has_three, 0, 0,
                                                                       emit_count, runes, runes_capacity, &last_off);
                sz_size_t const last_length = (sz_size_t)1 + ((lead2_bits >> last_off) & 1u) +
                                              (((lead3_bits >> last_off) & 1u) << 1);
                *runes_unpacked = produced;
                return text + (sz_size_t)last_off + last_length;
            }
        }
    }
    __m256i const length_lut_u8x32 = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4));
    __m256i const byte_high_nibble_u8x32 = _mm256_and_si256(_mm256_srli_epi16(window_u8x32, 4), _mm256_set1_epi8(0x0F));
    __m256i const sequence_lengths_u8x32 = _mm256_shuffle_epi8(length_lut_u8x32, byte_high_nibble_u8x32);
    sz_u32_t const len_ge_two =
        (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(sequence_lengths_u8x32, _mm256_set1_epi8(1))) & starts_bits;
    sz_u32_t const len_ge_three =
        (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(sequence_lengths_u8x32, _mm256_set1_epi8(2))) & starts_bits;
    sz_u32_t const len_ge_four =
        (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(sequence_lengths_u8x32, _mm256_set1_epi8(3))) & starts_bits;
    __m256i const lane_iota_u8x32 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                                                     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m256i const sequence_end_u8x32 = _mm256_add_epi8(lane_iota_u8x32, sequence_lengths_u8x32);
    __m256i const signed_bias_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const overrun_detected_u8x32 = _mm256_cmpgt_epi8(
        _mm256_add_epi8(sequence_end_u8x32, signed_bias_u8x32),
        _mm256_add_epi8(_mm256_set1_epi8((char)chunk), signed_bias_u8x32));
    sz_u32_t const overruns_bits = (sz_u32_t)_mm256_movemask_epi8(overrun_detected_u8x32) & starts_bits;
    sz_size_t const decodable_end = overruns_bits ? (sz_size_t)_tzcnt_u64(overruns_bits) : chunk;
    sz_u32_t const decodable_mask = sz_u32_mask_until_serial_(decodable_end);
    __m256i const is_length_two_u8x32 = _mm256_cmpeq_epi8(sequence_lengths_u8x32, _mm256_set1_epi8(2));
    __m256i const is_length_four_u8x32 = _mm256_cmpeq_epi8(sequence_lengths_u8x32, _mm256_set1_epi8(4));
    __m256i const is_less_than_0xc2_u8x32 = _mm256_cmpgt_epi8(
        _mm256_add_epi8(_mm256_set1_epi8((char)0xC2), signed_bias_u8x32),
        _mm256_add_epi8(window_u8x32, signed_bias_u8x32));
    __m256i const is_greater_than_0xf4_u8x32 = _mm256_cmpgt_epi8(
        _mm256_add_epi8(window_u8x32, signed_bias_u8x32),
        _mm256_add_epi8(_mm256_set1_epi8((char)0xF4), signed_bias_u8x32));
    sz_u32_t const bad_lead_bits = (sz_u32_t)_mm256_movemask_epi8(_mm256_or_si256(
                                       _mm256_and_si256(is_length_two_u8x32, is_less_than_0xc2_u8x32),
                                       _mm256_and_si256(is_length_four_u8x32, is_greater_than_0xf4_u8x32))) &
                                   starts_bits;
    sz_u32_t overlong_or_surrogate_or_range_bits = 0;
    __m256i const is_lead_0xe0_u8x32 = _mm256_cmpeq_epi8(window_u8x32, _mm256_set1_epi8((char)0xE0));
    __m256i const is_lead_0xed_u8x32 = _mm256_cmpeq_epi8(window_u8x32, _mm256_set1_epi8((char)0xED));
    __m256i const is_lead_0xf0_u8x32 = _mm256_cmpeq_epi8(window_u8x32, _mm256_set1_epi8((char)0xF0));
    __m256i const is_lead_0xf4_u8x32 = _mm256_cmpeq_epi8(window_u8x32, _mm256_set1_epi8((char)0xF4));
    __m256i const is_special_lead_u8x32 = _mm256_or_si256(_mm256_or_si256(is_lead_0xe0_u8x32, is_lead_0xed_u8x32),
                                                          _mm256_or_si256(is_lead_0xf0_u8x32, is_lead_0xf4_u8x32));
    if (_mm256_movemask_epi8(is_special_lead_u8x32)) {
        __m256i const window_successor_u8x32 = _mm256_permute2x128_si256(window_u8x32, window_u8x32, 0x01);
        __m256i const next_byte_u8x32 = _mm256_alignr_epi8(window_successor_u8x32, window_u8x32, 1);
        __m256i const next_byte_biased_u8x32 = _mm256_add_epi8(next_byte_u8x32, signed_bias_u8x32);
        __m256i const is_next_byte_lt_0xa0_u8x32 = _mm256_cmpgt_epi8(
            _mm256_add_epi8(_mm256_set1_epi8((char)0xA0), signed_bias_u8x32), next_byte_biased_u8x32);
        __m256i const is_next_byte_ge_0xa0_u8x32 = _mm256_cmpgt_epi8(
            next_byte_biased_u8x32, _mm256_add_epi8(_mm256_set1_epi8((char)0x9F), signed_bias_u8x32));
        __m256i const is_next_byte_lt_0x90_u8x32 = _mm256_cmpgt_epi8(
            _mm256_add_epi8(_mm256_set1_epi8((char)0x90), signed_bias_u8x32), next_byte_biased_u8x32);
        __m256i const is_next_byte_ge_0x90_u8x32 = _mm256_cmpgt_epi8(
            next_byte_biased_u8x32, _mm256_add_epi8(_mm256_set1_epi8((char)0x8F), signed_bias_u8x32));
        __m256i const range_violation_u8x32 = _mm256_or_si256(
            _mm256_or_si256(_mm256_and_si256(is_lead_0xe0_u8x32, is_next_byte_lt_0xa0_u8x32),
                            _mm256_and_si256(is_lead_0xed_u8x32, is_next_byte_ge_0xa0_u8x32)),
            _mm256_or_si256(_mm256_and_si256(is_lead_0xf0_u8x32, is_next_byte_lt_0x90_u8x32),
                            _mm256_and_si256(is_lead_0xf4_u8x32, is_next_byte_ge_0x90_u8x32)));
        overlong_or_surrogate_or_range_bits = (sz_u32_t)_mm256_movemask_epi8(range_violation_u8x32) & starts_bits;
    }
    sz_u32_t const cont1 = starts_bits & (continuation_bits >> 1);
    sz_u32_t const cont2 = starts_bits & (continuation_bits >> 2);
    sz_u32_t const cont3 = starts_bits & (continuation_bits >> 3);
    sz_u32_t const b1_range_ok = starts_bits & ~overlong_or_surrogate_or_range_bits;
    sz_u32_t const first_ok = cont1 & b1_range_ok & ~bad_lead_bits;
    sz_u32_t const length_one_bits = starts_bits & ~len_ge_two;
    sz_u32_t const length_two_bits = len_ge_two & ~len_ge_three;
    sz_u32_t const length_three_bits = len_ge_three & ~len_ge_four;
    sz_u32_t const length_four_bits = len_ge_four;
    sz_u32_t const wf1 = length_one_bits;
    sz_u32_t const wf2 = length_two_bits & ~bad_lead_bits & cont1;
    sz_u32_t const wf3 = length_three_bits & b1_range_ok & cont1 & cont2;
    sz_u32_t const wf4 = length_four_bits & ~bad_lead_bits & b1_range_ok & cont1 & cont2 & cont3;
    sz_u32_t const well_formed = (wf1 | wf2 | wf3 | wf4) & decodable_mask;
    sz_u32_t const step2 = len_ge_two & first_ok;
    sz_u32_t const step3 = step2 & len_ge_three & cont2;
    sz_u32_t const step4 = step3 & len_ge_four & cont3;
    sz_u32_t const covered = ((step2 & decodable_mask) << 1) | ((step3 & decodable_mask) << 2) |
                             ((step4 & decodable_mask) << 3);
    sz_u32_t const orphan = continuation_bits & decodable_mask & ~covered;
    sz_u32_t const emit_starts = (starts_bits | orphan) & decodable_mask;
    sz_size_t const emit_count = (sz_size_t)_mm_popcnt_u64(emit_starts);
    if (emit_count == 0) { return *runes_unpacked = 0, text; } 
    sz_u32_t const ill_formed = emit_starts & ~well_formed;
    int const has_ill = ill_formed != 0;
    __m256i const ill_formed_lanes_u8x32 = has_ill ? sz_utf8_byte_mask_from_bits_haswell_(ill_formed)
                                                   : _mm256_setzero_si256();
    int const has_three = (len_ge_three | len_ge_four) != 0;
    int const has_four = len_ge_four != 0;
    sz_u8_t last_off = 0;
    sz_size_t const produced = sz_utf8_rune_drain_haswell_(window_u8x32, ill_formed_lanes_u8x32, emit_starts, has_three,
                                                           has_four, has_ill, emit_count, runes, runes_capacity,
                                                           &last_off);
    sz_size_t const last_length = (sz_size_t)1 + ((step2 >> last_off) & 1u) + ((step3 >> last_off) & 1u) +
                                  ((step4 >> last_off) & 1u);
    *runes_unpacked = produced;
    return text + (sz_size_t)last_off + last_length;
}
/**
 *  @brief  Decode UTF-8 into dense UTF-32 @p runes, TOTAL in-vector over 32-byte AVX2 windows (NO decline-to-serial).
 *          Each window classifies all lanes, promotes orphan continuation bytes, compacts the emitted starts, gathers
 *          their bytes in-register, and width-blends; ill-formed leads/orphans collapse to one U+FFFD per maximal
 *          ill-formed subpart (Unicode 17.0 §3.9 / W3C), bit-exact with @ref sz_utf8_decode_serial. A pure-ASCII
 *          window widens with `vpmovzxbd`. Gather-free on the hot path: `vbroadcasti128` + `vpshufb` window reads and
 *          BMI2 `pext`/`tzcnt`/`blsr` compaction, never a `vpgatherdd`.
 */
SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_haswell( 
    sz_cptr_t text, sz_size_t length,             
    sz_rune_t *runes, sz_size_t runes_capacity,   
    sz_size_t *runes_unpacked) {
    sz_cptr_t cursor = text;
    sz_cptr_t const end = text + length;
    sz_size_t runes_written = 0;
    while (runes_written < runes_capacity && cursor < end) {
        sz_size_t step_unpacked = 0;
        sz_cptr_t next = sz_utf8_decode_once_haswell_(cursor, (sz_size_t)(end - cursor), runes + runes_written,
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
}
#pragma endregion Total vectorized decode
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
