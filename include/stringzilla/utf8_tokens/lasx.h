/**
 *  @brief LoongArch LASX backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/lasx.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_LASX_H_
#define STRINGZILLA_UTF8_TOKENS_LASX_H_
#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/lasx.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_LASX
/** @brief  Peel the tile's first `emit_count` matches with a `__lasx_xvperm_w` left-pack, 4 lanes per sub-block.
 *  Each sub-block gathers its set `(position+lane, length)` pairs to the front (same dword-index table as
 *  `sz_utf8_iterate_peel_haswell_`) and element-stores `min(popcount, remaining)` at the advancing cursor. */
SZ_HELPER_AUTO void sz_utf8_iterate_peel_lasx_(                                
    sz_u32_t start_bits, sz_u32_t two_byte_starts, sz_u32_t three_byte_starts, 
    sz_size_t emit_count, sz_size_t position,                                  
    sz_size_t *match_offsets, sz_size_t *match_lengths) {
    static sz_u32_t const compact_lut[16][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 0, 0, 0, 0},
        {4, 5, 0, 0, 0, 0, 0, 0}, {0, 1, 4, 5, 0, 0, 0, 0}, {2, 3, 4, 5, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 0, 0},
        {6, 7, 0, 0, 0, 0, 0, 0}, {0, 1, 6, 7, 0, 0, 0, 0}, {2, 3, 6, 7, 0, 0, 0, 0}, {0, 1, 2, 3, 6, 7, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0}, {0, 1, 4, 5, 6, 7, 0, 0}, {2, 3, 4, 5, 6, 7, 0, 0}, {0, 1, 2, 3, 4, 5, 6, 7},
    };
    static sz_u64_t const lane_ramp[4] = {0, 1, 2, 3};
    __m256i const lane_ramp_u64x4 = __lasx_xvld(lane_ramp, 0);
    sz_size_t emitted = 0;
    for (sz_size_t sub_block = 0; sub_block < 8 && emitted < emit_count; ++sub_block) {
        sz_u32_t const submask = (start_bits >> (sub_block * 4)) & 0xFu;
        if (!submask) continue;
        sz_size_t const base_lane = sub_block * 4;
        sz_u32_t const two_byte_sub = (two_byte_starts >> base_lane) & 0xFu;
        sz_u32_t const three_byte_sub = (three_byte_starts >> base_lane) & 0xFu;
        sz_u32_t packed_lengths = 0;
        for (sz_size_t lane_in_block = 0; lane_in_block < 4; ++lane_in_block) {
            sz_u32_t const match_length = 1u + ((two_byte_sub >> lane_in_block) & 1u) +
                                          2u * ((three_byte_sub >> lane_in_block) & 1u);
            packed_lengths |= match_length << (lane_in_block * 8);
        }
        __m256i const lengths_u8x32 = __lasx_xvinsgr2vr_w(__lasx_xvreplgr2vr_b(0), (int)packed_lengths, 0);
        __m256i const offsets_u64x4 = __lasx_xvadd_d(__lasx_xvreplgr2vr_d((long long)(position + base_lane)),
                                                     lane_ramp_u64x4);
        __m256i const lengths_u64x4 = __lasx_vext2xv_du_bu(lengths_u8x32);
        __m256i const permutation_u32x8 = __lasx_xvld(compact_lut[submask], 0);
        __m256i const packed_offsets_u64x4 = __lasx_xvperm_w(offsets_u64x4, permutation_u32x8);
        __m256i const packed_lengths_u64x4 = __lasx_xvperm_w(lengths_u64x4, permutation_u32x8);
        sz_size_t const taken = sz_min_of_two((sz_size_t)sz_u32_popcount(submask), emit_count - emitted);
        sz_utf8_iterate_store_group_lasx_(packed_offsets_u64x4, taken, match_offsets + emitted);
        sz_utf8_iterate_store_group_lasx_(packed_lengths_u64x4, taken, match_lengths + emitted);
        emitted += taken;
    }
}
SZ_API_COMPTIME sz_size_t sz_utf8_newlines_lasx(        
    sz_cptr_t text, sz_size_t length,                   
    sz_size_t *match_offsets, sz_size_t *match_lengths, 
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;
    __m256i newline_u8x32 = __lasx_xvreplgr2vr_b('\n'), vertical_tab_u8x32 = __lasx_xvreplgr2vr_b('\v'),
            form_feed_u8x32 = __lasx_xvreplgr2vr_b('\f'), carriage_return_u8x32 = __lasx_xvreplgr2vr_b('\r'),
            lead_c2_u8x32 = __lasx_xvreplgr2vr_b((char)0xC2), x_85_u8x32 = __lasx_xvreplgr2vr_b((char)0x85),
            lead_e2_u8x32 = __lasx_xvreplgr2vr_b((char)0xE2), byte_80_u8x32 = __lasx_xvreplgr2vr_b((char)0x80),
            x_a8_u8x32 = __lasx_xvreplgr2vr_b((char)0xA8), x_a9_u8x32 = __lasx_xvreplgr2vr_b((char)0xA9);
    while (position + 32 <= length && count < matches_capacity) {
        __m256i window_u8x32 = __lasx_xvld(text_u8 + position, 0);
        sz_u32_t newline_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, newline_u8x32));
        sz_u32_t carriage_return_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, carriage_return_u8x32));
        sz_u32_t one_byte_mask = newline_mask |
                                 sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, vertical_tab_u8x32)) |
                                 sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, form_feed_u8x32)) |
                                 carriage_return_mask;
        sz_u32_t lead_c2_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, lead_c2_u8x32));
        sz_u32_t x_85_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_85_u8x32));
        sz_u32_t lead_e2_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, lead_e2_u8x32));
        sz_u32_t byte_80_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, byte_80_u8x32));
        sz_u32_t x_a8_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_a8_u8x32));
        sz_u32_t x_a9_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_a9_u8x32));
        sz_u32_t nel_mask = lead_c2_mask & (x_85_mask >> 1);                       
        sz_u32_t lead_e280_mask = lead_e2_mask & (byte_80_mask >> 1);              
        sz_u32_t line_para_mask = lead_e280_mask & ((x_a8_mask | x_a9_mask) >> 2); 
        sz_u32_t crlf_mask = carriage_return_mask & (newline_mask >> 1);
        sz_u32_t lf_of_crlf_mask = newline_mask & (carriage_return_mask << 1);
        sz_u32_t two_byte_starts = crlf_mask | nel_mask;
        sz_u32_t three_byte_starts = line_para_mask;
        sz_u32_t start_bits = (one_byte_mask | nel_mask | line_para_mask) & ~lf_of_crlf_mask;
        start_bits &= 0x3FFFFFFFu; 
        if (position != 0 && text_u8[position - 1] == '\r') start_bits &= ~(newline_mask & 1u);
        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_lasx_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                       match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { 
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 30;
    }
    if (position != 0 && position < length && text_u8[position - 1] == '\r' && text_u8[position] == '\n') ++position;
    count += sz_utf8_newlines_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                      match_offsets + count, match_lengths + count, matches_capacity - count,
                                      bytes_consumed);
    return count;
}
SZ_API_COMPTIME sz_size_t sz_utf8_whitespaces_lasx(     
    sz_cptr_t text, sz_size_t length,                   
    sz_size_t *match_offsets, sz_size_t *match_lengths, 
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;
    __m256i x_20_u8x32 = __lasx_xvreplgr2vr_b(' '), x_08_u8x32 = __lasx_xvreplgr2vr_b((char)0x08),
            x_0e_u8x32 = __lasx_xvreplgr2vr_b((char)0x0E), lead_c2_u8x32 = __lasx_xvreplgr2vr_b((char)0xC2),
            x_85_u8x32 = __lasx_xvreplgr2vr_b((char)0x85), x_a0_u8x32 = __lasx_xvreplgr2vr_b((char)0xA0),
            x_e1_u8x32 = __lasx_xvreplgr2vr_b((char)0xE1), lead_e2_u8x32 = __lasx_xvreplgr2vr_b((char)0xE2),
            x_e3_u8x32 = __lasx_xvreplgr2vr_b((char)0xE3), x_9a_u8x32 = __lasx_xvreplgr2vr_b((char)0x9A),
            byte_80_u8x32 = __lasx_xvreplgr2vr_b((char)0x80), x_81_u8x32 = __lasx_xvreplgr2vr_b((char)0x81),
            x_8a_u8x32 = __lasx_xvreplgr2vr_b((char)0x8A), x_a8_u8x32 = __lasx_xvreplgr2vr_b((char)0xA8),
            x_a9_u8x32 = __lasx_xvreplgr2vr_b((char)0xA9), x_af_u8x32 = __lasx_xvreplgr2vr_b((char)0xAF),
            x_9f_u8x32 = __lasx_xvreplgr2vr_b((char)0x9F);
    while (position + 32 <= length && count < matches_capacity) {
        __m256i window_u8x32 = __lasx_xvld(text_u8 + position, 0);
        __m256i tab_lower_bound_u8x32 = __lasx_xvslt_b(x_08_u8x32, window_u8x32);
        __m256i carriage_return_upper_bound_u8x32 = __lasx_xvslt_b(window_u8x32, x_0e_u8x32);
        __m256i one_byte_cmp_u8x32 = __lasx_xvor_v(
            __lasx_xvseq_b(window_u8x32, x_20_u8x32),
            __lasx_xvand_v(tab_lower_bound_u8x32, carriage_return_upper_bound_u8x32));
        sz_u32_t one_byte_mask = sz_xvmovemask_b_utf8_lasx_(one_byte_cmp_u8x32);
        sz_u32_t lead_c2_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, lead_c2_u8x32));
        sz_u32_t x_85_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_85_u8x32));
        sz_u32_t x_a0_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_a0_u8x32));
        sz_u32_t two_byte_starts = lead_c2_mask & ((x_85_mask | x_a0_mask) >> 1);
        sz_u32_t x_e1_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_e1_u8x32));
        sz_u32_t lead_e2_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, lead_e2_u8x32));
        sz_u32_t x_e3_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_e3_u8x32));
        sz_u32_t x_9a_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_9a_u8x32));
        sz_u32_t byte_80_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, byte_80_u8x32));
        sz_u32_t x_81_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_81_u8x32));
        sz_u32_t x_a8_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_a8_u8x32));
        sz_u32_t x_a9_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_a9_u8x32));
        sz_u32_t x_af_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_af_u8x32));
        sz_u32_t x_9f_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window_u8x32, x_9f_u8x32));
        __m256i x_80_ge_cmp_u8x32 = __lasx_xvsle_bu(byte_80_u8x32, window_u8x32);
        __m256i x_8a_le_cmp_u8x32 = __lasx_xvsle_bu(window_u8x32, x_8a_u8x32);
        sz_u32_t x_8a_range_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvand_v(x_80_ge_cmp_u8x32, x_8a_le_cmp_u8x32));
        sz_u32_t lead_e280_mask = lead_e2_mask & (byte_80_mask >> 1);                      
        sz_u32_t ogham_mask = x_e1_mask & (x_9a_mask >> 1) & (byte_80_mask >> 2);          
        sz_u32_t range_e280_mask = lead_e280_mask & (x_8a_range_mask >> 2);                
        sz_u32_t nnbsp_mask = lead_e280_mask & (x_af_mask >> 2);                           
        sz_u32_t mmsp_mask = lead_e2_mask & (x_81_mask >> 1) & (x_9f_mask >> 2);           
        sz_u32_t line_mask = lead_e280_mask & (x_a8_mask >> 2);                            
        sz_u32_t para_mask = lead_e280_mask & (x_a9_mask >> 2);                            
        sz_u32_t ideographic_mask = x_e3_mask & (byte_80_mask >> 1) & (byte_80_mask >> 2); 
        sz_u32_t three_byte_starts = ogham_mask | range_e280_mask | nnbsp_mask | mmsp_mask | line_mask | para_mask |
                                     ideographic_mask;
        sz_u32_t start_bits = (one_byte_mask | two_byte_starts | three_byte_starts) & 0x3FFFFFFFu; 
        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_lasx_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                       match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { 
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 30;
    }
    count += sz_utf8_whitespaces_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                         match_offsets + count, match_lengths + count, matches_capacity - count,
                                         bytes_consumed);
    return count;
}
#pragma endregion Multistep Newline &Whitespace Iteration
#endif 
#ifdef __cplusplus
}
#endif
#endif 
