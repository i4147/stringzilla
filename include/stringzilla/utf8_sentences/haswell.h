/**
 *  @brief Haswell (AVX2) backend for UAX-29 sentence boundaries.
 *  @file include/stringzilla/utf8_sentences/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_SENTENCES_HASWELL_H_
#define STRINGZILLA_UTF8_SENTENCES_HASWELL_H_
#include "stringzilla/types.h"
#include "stringzilla/utf8_sentences/tables.h"
#include "stringzilla/utf8_sentences/serial.h"
#include "stringzilla/utf8_runes/haswell.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt,lzcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "popcnt", "lzcnt")
#endif
#pragma region UAX 29 Sentence Boundaries forward kernel
#pragma region In register vectorized classifier
/** @brief  Sentence_Break class byte for thirty-two BMP codepoints (per-lane high = cp>>8, low = cp&0xFF): the
 *          `bmp_page_lut_` page LUT selects one of the 57 distinct 256-byte pages, then `flat_bmp_` is fetched by
 *          `vpgatherdd`. Bit-exact with `sz_rune_sentence_break_property` over the whole BMP. */
SZ_HELPER_AUTO __m256i sz_utf8_sentence_break_bmp_class_haswell_(__m256i high_bytes_u8x32, __m256i low_bytes_u8x32) {
    return sz_utf8_rune_flat_lookup_haswell_(sz_utf8_sentence_break_bmp_page_lut_, sz_utf8_sentence_break_flat_bmp_,
                                             high_bytes_u8x32, low_bytes_u8x32);
}
/** @brief  Sentence_Break class byte for thirty-two ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble
 *          cascade). Per-lane bytes: @p plane_u8x32 = (offset>>16)&0xFF (low nibble meaningful), @p high_u8x32 =
 *          (offset>>8)&0xFF, @p low_u8x32 = offset&0xFF. Bit-exact with `sz_rune_sentence_break_property` over all
 *          astral. */
SZ_HELPER_AUTO __m256i sz_utf8_sentence_break_astral_class_haswell_(__m256i plane_u8x32, __m256i high_u8x32,
                                                                    __m256i low_u8x32) {
    __m256i const low_nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const n4_u8x32 = _mm256_and_si256(plane_u8x32, low_nibble_mask_u8x32);
    __m256i const n3_u8x32 = _mm256_and_si256(_mm256_srli_epi16(high_u8x32, 4), low_nibble_mask_u8x32);
    __m256i const stage1_index_u8x32 = _mm256_or_si256(_mm256_slli_epi16(n4_u8x32, 4), n3_u8x32);
    __m256i const page_u8x32 = sz_utf8_rune_lut256_haswell_(sz_utf8_sentence_break_haswell_astral_stage1_,
                                                            stage1_index_u8x32);
    __m256i const n2_u8x32 = _mm256_and_si256(high_u8x32, low_nibble_mask_u8x32);
    __m256i const leaf2_lo_u8x32 = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_sentence_break_haswell_astral_stage2_lo_, sz_utf8_sentence_break_haswell_astral_stage2_lo_count_k / 16,
        page_u8x32, n2_u8x32);
    __m256i const n1_u8x32 = _mm256_and_si256(_mm256_srli_epi16(low_u8x32, 4), low_nibble_mask_u8x32);
    __m256i const leaf_lo_u8x32 = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_sentence_break_haswell_astral_stage3_lo_, sz_utf8_sentence_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2_lo_u8x32, n1_u8x32);
    __m256i const leaf_hi_u8x32 = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_sentence_break_haswell_astral_stage3_hi_, sz_utf8_sentence_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2_lo_u8x32, n1_u8x32);
    __m256i const n0_u8x32 = _mm256_and_si256(low_u8x32, low_nibble_mask_u8x32);
    __m256i const leaf_group_u8x32 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(leaf_lo_u8x32, 4), low_nibble_mask_u8x32),
        _mm256_slli_epi16(leaf_hi_u8x32, 4));
    __m256i const leaf_low_nibble_u8x32 = _mm256_and_si256(leaf_lo_u8x32, low_nibble_mask_u8x32);
    __m256i const stage4_lut_index_u8x32 = _mm256_or_si256(_mm256_slli_epi16(leaf_low_nibble_u8x32, 4), n0_u8x32);
    __m256i result_u8x32 = _mm256_setzero_si256();
    for (int group = 0; group < (int)sz_utf8_sentence_break_haswell_astral_leaf_groups_k; ++group) {
        __m256i const value_u8x32 = sz_utf8_rune_lut256_haswell_(
            sz_utf8_sentence_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index_u8x32);
        __m256i const here_u8x32 = _mm256_cmpeq_epi8(leaf_group_u8x32, _mm256_set1_epi8((char)group));
        result_u8x32 = _mm256_blendv_epi8(result_u8x32, value_u8x32, here_u8x32);
    }
    return result_u8x32;
}
/** @brief  Per-byte-lane Sentence_Break class for one decoded window half, fully in-register and zero-scalar - the
 *          AVX2 twin of @ref sz_utf8_sentence_break_classify_window_icelake_. The decoded window only carries the
 *          2-/3-byte (high, low) reconstruction; this leaf rebuilds the ASCII (`low_u8x32 = raw_u8x32`,
 *          `high_u8x32 = 0`) and 4-byte (`high_u8x32`/`low_u8x32` from the four-byte formula) codepoint bytes before
 *          the cascade, exactly as the icelake driver reconstructs them. BMP lanes go through the BMP cascade;
 *          4-byte lanes are routed by reconstructed plane through the astral cascade. The class on non-codepoint-start
 *          lanes is irrelevant (the dense compaction only reads start lanes), so those lanes are never selected. */
SZ_HELPER_AUTO __m256i sz_utf8_sentence_break_classify_half_haswell_( 
    __m256i window_high_u8x32, __m256i window_low_u8x32, __m256i raw_u8x32, __m256i next1_u8x32, __m256i next2_u8x32,
    __m256i next3_u8x32, sz_u32_t four_byte_bits) {
    __m256i const low_two_bits_u8x32 = _mm256_set1_epi8(0x03);
    __m256i const low_four_bits_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const low_six_bits_u8x32 = _mm256_set1_epi8(0x3F);
    __m256i const below_c0_u8x32 = _mm256_cmpeq_epi8(_mm256_min_epu8(raw_u8x32, _mm256_set1_epi8((char)0xC0)),
                                                     raw_u8x32);
    __m256i const at_least_f8_u8x32 = _mm256_cmpeq_epi8(_mm256_max_epu8(raw_u8x32, _mm256_set1_epi8((char)0xF8)),
                                                        raw_u8x32);
    __m256i const raw_select_u8x32 = _mm256_andnot_si256(_mm256_cmpeq_epi8(raw_u8x32, _mm256_set1_epi8((char)0xC0)),
                                                         _mm256_or_si256(below_c0_u8x32, at_least_f8_u8x32));
    __m256i low_u8x32 = _mm256_blendv_epi8(window_low_u8x32, raw_u8x32, raw_select_u8x32);
    __m256i high_u8x32 = _mm256_andnot_si256(raw_select_u8x32, window_high_u8x32);
    if (four_byte_bits) {
        __m256i const four_low_u8x32 = _mm256_or_si256(
            _mm256_slli_epi16(_mm256_and_si256(next2_u8x32, low_two_bits_u8x32), 6),
            _mm256_and_si256(next3_u8x32, low_six_bits_u8x32));
        __m256i const four_high_u8x32 = _mm256_or_si256(
            _mm256_slli_epi16(_mm256_and_si256(next1_u8x32, low_four_bits_u8x32), 4),
            sz_utf8_srl8_haswell_(next2_u8x32, 2, 0x0F));
        __m256i const four_select_u8x32 = sz_utf8_byte_mask_from_bits_haswell_(four_byte_bits);
        low_u8x32 = _mm256_blendv_epi8(low_u8x32, four_low_u8x32, four_select_u8x32);
        high_u8x32 = _mm256_blendv_epi8(high_u8x32, four_high_u8x32, four_select_u8x32);
        __m256i const plane_u8x32 = _mm256_or_si256(
            _mm256_slli_epi16(_mm256_and_si256(raw_u8x32, _mm256_set1_epi8(0x07)), 2),
            sz_utf8_srl8_haswell_(next1_u8x32, 4, 0x03));
        __m256i const plane_nonzero_u8x32 = _mm256_andnot_si256(_mm256_cmpeq_epi8(plane_u8x32, _mm256_setzero_si256()),
                                                                four_select_u8x32);
        __m256i const plane_le_16_u8x32 = _mm256_cmpeq_epi8(_mm256_max_epu8(plane_u8x32, _mm256_set1_epi8(0x10)),
                                                            _mm256_set1_epi8(0x10));
        __m256i const is_astral_lanes_u8x32 = _mm256_and_si256(plane_nonzero_u8x32, plane_le_16_u8x32);
        __m256i const is_overrange_lanes_u8x32 = _mm256_andnot_si256(plane_le_16_u8x32, plane_nonzero_u8x32);
        __m256i const plane_off_u8x32 = _mm256_sub_epi8(_mm256_and_si256(is_astral_lanes_u8x32, plane_u8x32),
                                                        _mm256_set1_epi8(1));
        __m256i const bmp_u8x32 = sz_utf8_sentence_break_bmp_class_haswell_(high_u8x32, low_u8x32);
        __m256i const astral_u8x32 = sz_utf8_sentence_break_astral_class_haswell_(plane_off_u8x32, high_u8x32,
                                                                                  low_u8x32);
        __m256i const classed_u8x32 = _mm256_blendv_epi8(bmp_u8x32, astral_u8x32, is_astral_lanes_u8x32);
        return _mm256_andnot_si256(is_overrange_lanes_u8x32, classed_u8x32);
    }
    return sz_utf8_sentence_break_bmp_class_haswell_(high_u8x32, low_u8x32);
}
/** @brief  Reconstruct the BMP (2-/3-byte) codepoint high/low bytes for one 32-lane half from edge-masked forward
 *          neighbours, matching the rune-window decode's arithmetic but with `next1_u8x32`/`next2_u8x32` already
 *          zeroed past the loaded edge. Lanes that are neither a 2- nor a 3-byte lead keep `low_u8x32 = raw_u8x32`,
 *          `high_u8x32 = 0` (the classifier re-seats raw / 4-byte lanes anyway). Used by the sentence driver so a
 *          truncated trailing multi-byte lead reads its missing continuations as zero, exactly like serial / icelake
 *          (no mod-64 wrap aliasing). */
SZ_HELPER_AUTO void sz_utf8_sentence_break_bmp_highlow_haswell_( 
    __m256i raw_u8x32, __m256i next1_u8x32, __m256i next2_u8x32, sz_u32_t two_byte_bits, sz_u32_t three_byte_bits,
    __m256i *out_high_u8x32, __m256i *out_low_u8x32) {
    __m256i const low_two_bits_u8x32 = _mm256_set1_epi8(0x03);
    __m256i const low_four_bits_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const low_five_bits_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const low_six_bits_u8x32 = _mm256_set1_epi8(0x3F);
    __m256i const two_high_u8x32 = sz_utf8_srl8_haswell_(_mm256_and_si256(raw_u8x32, low_five_bits_u8x32), 2, 0x07);
    __m256i const two_low_u8x32 = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(raw_u8x32, low_two_bits_u8x32), 6),
                                                  _mm256_and_si256(next1_u8x32, low_six_bits_u8x32));
    __m256i const three_high_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(raw_u8x32, low_four_bits_u8x32), 4),
        sz_utf8_srl8_haswell_(next1_u8x32, 2, 0x0F));
    __m256i const three_low_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(next1_u8x32, low_two_bits_u8x32), 6),
        _mm256_and_si256(next2_u8x32, low_six_bits_u8x32));
    __m256i const two_select_u8x32 = sz_utf8_byte_mask_from_bits_haswell_(two_byte_bits);
    __m256i const three_select_u8x32 = sz_utf8_byte_mask_from_bits_haswell_(three_byte_bits);
    __m256i high_u8x32 = _mm256_setzero_si256();
    __m256i low_u8x32 = raw_u8x32;
    high_u8x32 = _mm256_blendv_epi8(high_u8x32, two_high_u8x32, two_select_u8x32);
    low_u8x32 = _mm256_blendv_epi8(low_u8x32, two_low_u8x32, two_select_u8x32);
    high_u8x32 = _mm256_blendv_epi8(high_u8x32, three_high_u8x32, three_select_u8x32);
    low_u8x32 = _mm256_blendv_epi8(low_u8x32, three_low_u8x32, three_select_u8x32);
    *out_high_u8x32 = high_u8x32, *out_low_u8x32 = low_u8x32;
}
/** @brief  Third forward neighbour `next3[i] = window[i+3]` over all 64 lanes with mod-64 wrap, the AVX2 twin of
 *          icelake's `_mm512_permutexvar_epi8(lane_identity+3)`. Same idiom as the substrate `forward_neighbours_`. */
SZ_HELPER_INLINE void sz_utf8_sentence_break_next3_haswell_(__m256i window_lo_u8x32, __m256i window_hi_u8x32,
                                                            __m256i *next3_lo_u8x32, __m256i *next3_hi_u8x32) {
    __m256i const low_successor_u8x32 = _mm256_permute2x128_si256(window_lo_u8x32, window_hi_u8x32, 0x21);
    *next3_lo_u8x32 = _mm256_alignr_epi8(low_successor_u8x32, window_lo_u8x32, 3);
    __m256i const high_successor_u8x32 = _mm256_permute2x128_si256(window_hi_u8x32, window_lo_u8x32, 0x21);
    *next3_hi_u8x32 = _mm256_alignr_epi8(high_successor_u8x32, window_hi_u8x32, 3);
}
#pragma endregion In register vectorized classifier
#pragma region Dense compaction and scatter
/** @brief  Build the per-class membership frame from the dense class byte stream with AVX2 compares: each class is one
 *          `vpcmpeqb` per 32-lane half OR-combined to a u64, the AVX2 twin of the icelake fifteen-`vpcmpeqb` build (no
 *          scalar pass). The dense stream is at most 64 lanes, held as two `__m256i`. */
SZ_HELPER_AUTO sz_utf8_sentence_break_frame_t sz_utf8_sentence_break_frame_haswell_(sz_u8_t const *dense_classes,
                                                                                    sz_u64_t valid) {
    __m256i const dense_lo_u8x32 = _mm256_loadu_si256((__m256i const *)(dense_classes + 0));
    __m256i const dense_hi_u8x32 = _mm256_loadu_si256((__m256i const *)(dense_classes + 32));
    sz_utf8_sentence_break_frame_t frame;
    for (int cls = 0; cls < 15; ++cls) {
        __m256i const value_u8x32 = _mm256_set1_epi8((char)cls);
        frame.by_class[cls] = sz_utf8_mask_combine_haswell_(_mm256_cmpeq_epi8(dense_lo_u8x32, value_u8x32),
                                                            _mm256_cmpeq_epi8(dense_hi_u8x32, value_u8x32)) &
                              valid;
    }
    return frame;
}
/** @brief  Run the portable rule engine over a dense class stream, building the frame with AVX2 compares first. */
SZ_HELPER_INLINE sz_utf8_sentence_break_window_t sz_utf8_sentence_break_decide_dense_haswell_( 
    sz_u8_t const *dense_classes, sz_size_t count, sz_utf8_sentence_break_carry_t *carry, sz_bool_t more_text) {
    sz_u64_t const valid = (count >= 64) ? ~0ull : ((1ull << count) - 1);
    sz_utf8_sentence_break_frame_t const frame = sz_utf8_sentence_break_frame_haswell_(dense_classes, valid);
    return sz_utf8_sentence_break_decide_block_(&frame, dense_classes, count, carry, more_text);
}
/** @brief  Largest byte prefix of the window whose codepoints are all fully loaded — the AVX2 twin of the icelake
 *          driver's effective-window<64 trim. Never below 1 when the window is non-empty. */
SZ_HELPER_AUTO sz_size_t sz_utf8_sentence_break_complete_limit_haswell_(sz_utf8_rune_window_haswell_t window,
                                                                        sz_u8_t const *bytes_after,
                                                                        sz_bool_t more_text) {
    sz_size_t const loaded = window.loaded;
    if (!more_text) return loaded;
    sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);
    sz_u64_t const start_bytes = window.codepoint_starts & valid;
    if (!start_bytes) return loaded;
    sz_u64_t const two = window.two_byte_starts & valid;
    sz_u64_t const three = window.three_byte_starts & valid;
    sz_u64_t const four = window.four_byte_starts & valid;
    sz_u64_t const straddle = ((two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                               (three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                               (four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                              valid;
    sz_size_t limit = straddle ? (sz_size_t)_tzcnt_u64(straddle) : loaded;
    if ((bytes_after[0] & 0xC0) == 0x80) {
        sz_size_t const last_lead = (sz_size_t)(63 - (int)_lzcnt_u64(start_bytes));
        if (last_lead < limit) limit = last_lead;
    }
    return limit > 0 ? limit : loaded;
}
#pragma endregion Dense compaction and scatter
#pragma region Forward driver
/**
 *  @brief  Forward UAX-29 sentence segmentation kernel (Haswell AVX2). Bit-exact with `sz_utf8_sentences_serial` and
 *          `sz_utf8_sentences_icelake`: an AVX2 window/classify/dense-compaction front-end feeds the shared portable
 *          rule engine @ref sz_utf8_sentence_break_decide_block_, whose dense breaks are scattered back to byte lanes.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_sentences_haswell(         
    sz_cptr_t text, sz_size_t length,                        
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, 
    sz_size_t sentences_capacity, sz_size_t *bytes_consumed) {
    sz_size_t sentences = 0;
    if (length == 0 || sentences_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t sentence_start = 0;
    sz_size_t position = 0;
    sz_utf8_sentence_break_carry_t carry;
    carry.have_prev = 0;
    carry.prev_raw = carry.prev_eff = carry.prev_prev_eff = (sz_u8_t)sz_sentence_break_other_k;
    carry.in_shadow = carry.shadow_aterm = carry.shadow_saw_sp = 0;
    carry.sb8_pending = 0;
    sz_size_t sb8_pending_position = 0;
    int sb8_pending_active = 0;
    while (position < length) {
        sz_utf8_rune_window_haswell_t const window = sz_utf8_rune_decode_window_haswell_(text_u8 + position,
                                                                                         length - position);
        sz_size_t const loaded = window.loaded;
        sz_u64_t const lead_continuation = (position == 0) ? 1ull : 0ull;
        sz_u64_t const start_bytes = window.codepoint_starts | lead_continuation;
        sz_bool_t const more_text = (position + loaded < length) ? sz_true_k : sz_false_k;
        __m256i next1_lo_u8x32, next1_hi_u8x32, next2_lo_u8x32, next2_hi_u8x32, next3_lo_u8x32, next3_hi_u8x32;
        sz_utf8_forward_neighbours_haswell_(window.window_lo, window.window_hi, &next1_lo_u8x32, &next1_hi_u8x32,
                                            &next2_lo_u8x32, &next2_hi_u8x32);
        sz_utf8_sentence_break_next3_haswell_(window.window_lo, window.window_hi, &next3_lo_u8x32, &next3_hi_u8x32);
        sz_u64_t const keep1 = sz_u64_mask_until_serial_(loaded >= 1 ? loaded - 1 : 0);
        sz_u64_t const keep2 = sz_u64_mask_until_serial_(loaded >= 2 ? loaded - 2 : 0);
        sz_u64_t const keep3 = sz_u64_mask_until_serial_(loaded >= 3 ? loaded - 3 : 0);
        next1_lo_u8x32 = _mm256_and_si256(next1_lo_u8x32, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)keep1));
        next1_hi_u8x32 = _mm256_and_si256(next1_hi_u8x32,
                                          sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(keep1 >> 32)));
        next2_lo_u8x32 = _mm256_and_si256(next2_lo_u8x32, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)keep2));
        next2_hi_u8x32 = _mm256_and_si256(next2_hi_u8x32,
                                          sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(keep2 >> 32)));
        next3_lo_u8x32 = _mm256_and_si256(next3_lo_u8x32, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)keep3));
        next3_hi_u8x32 = _mm256_and_si256(next3_hi_u8x32,
                                          sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(keep3 >> 32)));
        __m256i high_lo_u8x32, high_hi_u8x32, low_lo_u8x32, low_hi_u8x32;
        sz_utf8_sentence_break_bmp_highlow_haswell_(window.window_lo, next1_lo_u8x32, next2_lo_u8x32,
                                                    (sz_u32_t)window.two_byte_starts,
                                                    (sz_u32_t)window.three_byte_starts, &high_lo_u8x32, &low_lo_u8x32);
        sz_utf8_sentence_break_bmp_highlow_haswell_(
            window.window_hi, next1_hi_u8x32, next2_hi_u8x32, (sz_u32_t)(window.two_byte_starts >> 32),
            (sz_u32_t)(window.three_byte_starts >> 32), &high_hi_u8x32, &low_hi_u8x32);
        __m256i const classes_lo_u8x32 = sz_utf8_sentence_break_classify_half_haswell_(
            high_lo_u8x32, low_lo_u8x32, window.window_lo, next1_lo_u8x32, next2_lo_u8x32, next3_lo_u8x32,
            (sz_u32_t)window.four_byte_starts);
        __m256i const classes_hi_u8x32 = sz_utf8_sentence_break_classify_half_haswell_(
            high_hi_u8x32, low_hi_u8x32, window.window_hi, next1_hi_u8x32, next2_hi_u8x32, next3_hi_u8x32,
            (sz_u32_t)(window.four_byte_starts >> 32));
        sz_size_t const complete_limit = sz_utf8_sentence_break_complete_limit_haswell_(
            window, text_u8 + position + loaded, more_text);
        sz_u8_t class_bytes[64];
        _mm256_storeu_si256((__m256i *)(class_bytes + 0), classes_lo_u8x32);
        _mm256_storeu_si256((__m256i *)(class_bytes + 32), classes_hi_u8x32);
        sz_u64_t const complete_mask = sz_u64_mask_until_serial_(complete_limit);
        sz_u64_t const dense_start_lanes = start_bytes & complete_mask;
        sz_size_t const dense_count = (sz_size_t)_mm_popcnt_u64(dense_start_lanes);
        sz_u8_t dense_classes[64];
        {
            sz_u64_t remaining = dense_start_lanes;
            sz_size_t dense_index = 0;
            while (remaining) {
                sz_size_t const lane = (sz_size_t)_tzcnt_u64(remaining);
                remaining = _blsr_u64(remaining);
                dense_classes[dense_index++] = class_bytes[lane];
            }
        }
        sz_utf8_sentence_break_carry_t carry_full = carry;
        sz_utf8_sentence_break_window_t const win = sz_utf8_sentence_break_decide_dense_haswell_(
            dense_classes, dense_count, &carry_full, more_text);
        if (sb8_pending_active && win.sb8_resolution != 0) {
            if (win.sb8_resolution == 1) {
                if (sentences == sentences_capacity) {
                    if (bytes_consumed) *bytes_consumed = sentence_start;
                    return sentences;
                }
                sentence_starts[sentences] = sentence_start;
                sentence_lengths[sentences] = sb8_pending_position - sentence_start;
                ++sentences;
                sentence_start = sb8_pending_position;
            }
            sb8_pending_active = 0;
        }
        sz_size_t const dense_adv = win.resolved;
        sz_u64_t const dense_breaks = win.breaks & sz_u64_mask_until_serial_(dense_adv);
        sz_u64_t boundary_lanes = _pdep_u64(dense_breaks, dense_start_lanes);
        if (!carry.have_prev) boundary_lanes &= ~1ull;
        sz_size_t byte_adv;
        if (dense_adv >= dense_count) { byte_adv = complete_limit ? complete_limit : loaded; }
        else {
            sz_u64_t const upto = _pdep_u64((1ull << dense_adv), dense_start_lanes);
            byte_adv = (sz_size_t)_tzcnt_u64(upto);
        }
        sentences = sz_utf8_rune_drain_forward_haswell_(boundary_lanes, position, sentence_starts, sentence_lengths,
                                                        sentences, sentences_capacity, &sentence_start);
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }
        if (dense_adv >= dense_count) {
            carry = carry_full;
            position += byte_adv;
        }
        else if (dense_adv > 0) {
            sz_utf8_sentence_break_carry_t carry_to_edge = carry;
            sz_utf8_sentence_break_decide_dense_haswell_(dense_classes, dense_adv, &carry_to_edge, sz_true_k);
            carry = carry_to_edge;
            position += byte_adv;
        }
        else {
            if (!sb8_pending_active) {
                sb8_pending_active = 1;
                sb8_pending_position = position;
            }
            carry = carry_full;
            carry.sb8_pending = 1;
            position += complete_limit ? complete_limit : loaded;
        }
    }
    if (sb8_pending_active) {
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }
        sentence_starts[sentences] = sentence_start;
        sentence_lengths[sentences] = sb8_pending_position - sentence_start;
        ++sentences;
        sentence_start = sb8_pending_position;
    }
    if (sentences == sentences_capacity) {
        if (bytes_consumed) *bytes_consumed = sentence_start;
        return sentences;
    }
    sentence_starts[sentences] = sentence_start;
    sentence_lengths[sentences] = length - sentence_start;
    ++sentences;
    if (bytes_consumed) *bytes_consumed = length;
    return sentences;
}
#pragma endregion Forward driver
#pragma endregion UAX 29 Sentence Boundaries forward kernel
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
