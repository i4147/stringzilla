/**
 *  @brief LoongArch LASX (256-bit) backend for hash.
 *  @file include/stringzilla/hash/lasx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_LASX_H_
#define STRINGZILLA_HASH_LASX_H_
#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/compare.h" 
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_LASX
SZ_API_COMPTIME sz_u64_t sz_bytesum_lasx(sz_cptr_t text, sz_size_t length) {
    if (length <= 32) { return sz_bytesum_serial(text, length); }
    else {
        __m256i const zero_u8x32 = __lasx_xvreplgr2vr_b(0);
        __m256i long_sums_u64x4 = zero_u8x32;   
        __m256i short_sums_u16x16 = zero_u8x32; 
        sz_size_t blocks_since_flush = 0;
        for (; length >= 32; text += 32, length -= 32) {
            __m256i text_u8x32 = __lasx_xvld(text, 0);
            short_sums_u16x16 = __lasx_xvadd_h(short_sums_u16x16,
                                               __lasx_xvhaddw_hu_bu(text_u8x32, text_u8x32)); 
            if (++blocks_since_flush == 120) {
                __m256i widened_u32x8 = __lasx_xvhaddw_wu_hu(short_sums_u16x16, short_sums_u16x16);
                long_sums_u64x4 = __lasx_xvadd_d(long_sums_u64x4, __lasx_xvhaddw_du_wu(widened_u32x8, widened_u32x8));
                short_sums_u16x16 = zero_u8x32;
                blocks_since_flush = 0;
            }
        }
        {
            __m256i widened_u32x8 = __lasx_xvhaddw_wu_hu(short_sums_u16x16, short_sums_u16x16);
            long_sums_u64x4 = __lasx_xvadd_d(long_sums_u64x4, __lasx_xvhaddw_du_wu(widened_u32x8, widened_u32x8));
        }
        __m256i pairwise_sums_u64x4 = __lasx_xvhaddw_q_d(long_sums_u64x4, long_sums_u64x4);
        sz_u64_t result = (sz_u64_t)__lasx_xvpickve2gr_du(pairwise_sums_u64x4, 0) +
                          (sz_u64_t)__lasx_xvpickve2gr_du(pairwise_sums_u64x4, 2);
        if (length) result += sz_bytesum_serial(text, length);
        return result;
    }
}
/** @brief  Pre-computed vpaes-style nibble tables for the LASX AES round (see derivation above). */
SZ_HELPER_INLINE sz_u8_t const *sz_aes_lasx_tables_(void) {
    static sz_align_(64) sz_u8_t const tables[8 * 16] = {
        0x00,
        0x01,
        0x20,
        0x21,
        0x46,
        0x47,
        0x66,
        0x67,
        0x4C,
        0x4D,
        0x6C,
        0x6D,
        0x0A,
        0x0B,
        0x2A,
        0x2B, 
        0x00,
        0x3C,
        0xD5,
        0xE9,
        0x34,
        0x08,
        0xE1,
        0xDD,
        0xE5,
        0xD9,
        0x30,
        0x0C,
        0xD1,
        0xED,
        0x04,
        0x38, 
        0x00,
        0x1F,
        0xB2,
        0xAD,
        0xAB,
        0xB4,
        0x19,
        0x06,
        0x36,
        0x29,
        0x84,
        0x9B,
        0x9D,
        0x82,
        0x2F,
        0x30, 
        0x00,
        0x52,
        0x3E,
        0x6C,
        0x65,
        0x37,
        0x5B,
        0x09,
        0x60,
        0x32,
        0x5E,
        0x0C,
        0x05,
        0x57,
        0x3B,
        0x69, 
        0x00,
        0x00,
        0x01,
        0x04,
        0x02,
        0x08,
        0x05,
        0x0A,
        0x03,
        0x0E,
        0x09,
        0x07,
        0x06,
        0x0D,
        0x0B,
        0x0C, 
        0x01,
        0x02,
        0x04,
        0x08,
        0x03,
        0x06,
        0x0C,
        0x0B,
        0x05,
        0x0A,
        0x07,
        0x0E,
        0x0F,
        0x0D,
        0x09,
        0x01, 
        0x00,
        0x01,
        0x09,
        0x0E,
        0x0D,
        0x0B,
        0x07,
        0x06,
        0x0F,
        0x02,
        0x0C,
        0x05,
        0x0A,
        0x04,
        0x03,
        0x08, 
        0x00,
        0x05,
        0x0A,
        0x0F,
        0x04,
        0x09,
        0x0E,
        0x03,
        0x08,
        0x0D,
        0x02,
        0x07,
        0x0C,
        0x01,
        0x06,
        0x0B, 
    };
    return &tables[0];
}
/** @brief  `_mm_shuffle_epi8`-equivalent for indices in 0..15 (top bits clear). */
SZ_HELPER_INLINE __m128i sz_lsx_pshufb_(__m128i table_u8x16, __m128i indices_u8x16) {
    return __lsx_vshuf_b(indices_u8x16, table_u8x16, indices_u8x16);
}
/** @brief  Lane-wise GF(16) multiply (poly 0x13) of two nibble vectors via log/antilog tables. */
SZ_HELPER_AUTO __m128i sz_lsx_gf16_mul_(__m128i factor_a_u8x16, __m128i factor_b_u8x16, __m128i gf16_log_u8x16,
                                        __m128i gf16_exp_u8x16, __m128i zero_u8x16) {
    __m128i log_sum_u8x16 = __lsx_vadd_b(sz_lsx_pshufb_(gf16_log_u8x16, factor_a_u8x16),
                                         sz_lsx_pshufb_(gf16_log_u8x16, factor_b_u8x16)); 
    __m128i fifteen_u8x16 = __lsx_vreplgr2vr_b(15);
    __m128i wraps_mask_u8x16 = __lsx_vsle_b(fifteen_u8x16, log_sum_u8x16); 
    __m128i folded_log_u8x16 = __lsx_vsub_b(log_sum_u8x16,
                                            __lsx_vand_v(wraps_mask_u8x16, fifteen_u8x16)); 
    __m128i antilog_product_u8x16 = sz_lsx_pshufb_(gf16_exp_u8x16, folded_log_u8x16);
    __m128i zero_factor_mask_u8x16 = __lsx_vor_v(__lsx_vseq_b(factor_a_u8x16, zero_u8x16),
                                                 __lsx_vseq_b(factor_b_u8x16, zero_u8x16));
    return __lsx_vandn_v(zero_factor_mask_u8x16, antilog_product_u8x16); 
}
/**
 *  @brief AES SubBytes on the ShiftRows-permuted state via tower-field GF((2^4)^2) inversion + affine.
 *  @return The S-box output (with the `^ 0x63` AES affine constant already applied).
 */
SZ_HELPER_AUTO __m128i sz_emulate_aes_subbytes_lasx_( 
    __m128i shifted_state_u8x16, __m128i zero_u8x16, __m128i low_nibble_mask_u8x16, __m128i input_transform_low_u8x16,
    __m128i input_transform_high_u8x16, __m128i sbox_output_low_u8x16, __m128i sbox_output_high_u8x16,
    __m128i gf16_log_u8x16, __m128i gf16_exp_u8x16, __m128i gf16_inverse_u8x16) {
    __m128i state_low_nibbles_u8x16 = __lsx_vand_v(shifted_state_u8x16, low_nibble_mask_u8x16);
    __m128i state_high_nibbles_u8x16 = __lsx_vand_v(__lsx_vsrli_b(shifted_state_u8x16, 4), low_nibble_mask_u8x16);
    __m128i tower_basis_u8x16 = __lsx_vxor_v(sz_lsx_pshufb_(input_transform_low_u8x16, state_low_nibbles_u8x16),
                                             sz_lsx_pshufb_(input_transform_high_u8x16, state_high_nibbles_u8x16));
    __m128i tower_high_u8x16 = __lsx_vand_v(__lsx_vsrli_b(tower_basis_u8x16, 4), low_nibble_mask_u8x16);
    __m128i tower_low_u8x16 = __lsx_vand_v(tower_basis_u8x16, low_nibble_mask_u8x16);
    __m128i tower_low_xor_high_u8x16 = __lsx_vxor_v(tower_low_u8x16, tower_high_u8x16);
    __m128i tower_high_squared_u8x16 = sz_lsx_gf16_mul_(tower_high_u8x16, tower_high_u8x16, gf16_log_u8x16,
                                                        gf16_exp_u8x16, zero_u8x16);
    __m128i tower_nu_u8x16 = __lsx_vreplgr2vr_b(0x08); 
    __m128i inverse_denominator_u8x16 = __lsx_vxor_v(
        sz_lsx_gf16_mul_(tower_high_squared_u8x16, tower_nu_u8x16, gf16_log_u8x16, gf16_exp_u8x16, zero_u8x16),
        sz_lsx_gf16_mul_(tower_low_u8x16, tower_low_xor_high_u8x16, gf16_log_u8x16, gf16_exp_u8x16, zero_u8x16));
    __m128i denominator_inverse_u8x16 = sz_lsx_pshufb_(gf16_inverse_u8x16, inverse_denominator_u8x16);
    __m128i inverted_high_u8x16 = sz_lsx_gf16_mul_(tower_high_u8x16, denominator_inverse_u8x16, gf16_log_u8x16,
                                                   gf16_exp_u8x16, zero_u8x16);
    __m128i inverted_low_u8x16 = sz_lsx_gf16_mul_(tower_low_xor_high_u8x16, denominator_inverse_u8x16, gf16_log_u8x16,
                                                  gf16_exp_u8x16, zero_u8x16);
    __m128i inverted_nibbles_u8x16 = __lsx_vor_v(__lsx_vslli_b(inverted_high_u8x16, 4), inverted_low_u8x16);
    __m128i inverted_high_nibbles_u8x16 = __lsx_vand_v(__lsx_vsrli_b(inverted_nibbles_u8x16, 4), low_nibble_mask_u8x16);
    __m128i inverted_low_nibbles_u8x16 = __lsx_vand_v(inverted_nibbles_u8x16, low_nibble_mask_u8x16);
    __m128i sbox_output_u8x16 = __lsx_vxor_v(sz_lsx_pshufb_(sbox_output_low_u8x16, inverted_low_nibbles_u8x16),
                                             sz_lsx_pshufb_(sbox_output_high_u8x16, inverted_high_nibbles_u8x16));
    return __lsx_vxor_v(sbox_output_u8x16, __lsx_vreplgr2vr_b(0x63)); 
}
/**
 *  @brief AES MixColumns over the four 4-byte columns of the S-box output.
 *  @return `c[j] ^ (col_base^col_rot1^col_rot2^col_rot3) ^ xtime(c[j] ^ c[j+1])` per column.
 */
SZ_HELPER_AUTO __m128i sz_emulate_aes_mixcolumns_lasx_(__m128i sbox_output_u8x16) {
    static sz_align_(16) sz_u8_t const rot1_bytes[16] = {1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12};
    static sz_align_(16) sz_u8_t const rot2_bytes[16] = {2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13};
    static sz_align_(16) sz_u8_t const rot3_bytes[16] = {3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14};
    __m128i col_base_u8x16 = sbox_output_u8x16;
    __m128i col_rot1_u8x16 = sz_lsx_pshufb_(sbox_output_u8x16, __lsx_vld(rot1_bytes, 0));
    __m128i col_rot2_u8x16 = sz_lsx_pshufb_(sbox_output_u8x16, __lsx_vld(rot2_bytes, 0));
    __m128i col_rot3_u8x16 = sz_lsx_pshufb_(sbox_output_u8x16, __lsx_vld(rot3_bytes, 0));
    __m128i column_rotation_sum_u8x16 = __lsx_vxor_v(__lsx_vxor_v(col_base_u8x16, col_rot1_u8x16),
                                                     __lsx_vxor_v(col_rot2_u8x16, col_rot3_u8x16));
    __m128i adjacent_column_xor_u8x16 = __lsx_vxor_v(col_base_u8x16, col_rot1_u8x16);
    __m128i column_xtime_u8x16 = __lsx_vxor_v(
        __lsx_vslli_b(adjacent_column_xor_u8x16, 1),
        __lsx_vand_v(__lsx_vslti_b(adjacent_column_xor_u8x16, 0), __lsx_vreplgr2vr_b(0x1b)));
    return __lsx_vxor_v(__lsx_vxor_v(col_base_u8x16, column_rotation_sum_u8x16), column_xtime_u8x16);
}
/**
 *  @brief Emulates a single `_mm_aesenc_si128` round on LoongArch LSX (128-bit lanes).
 *  @return Result of `MixColumns(SubBytes(ShiftRows(state))) ^ round_key`, bit-identical to
 *          `sz_emulate_aesenc_si128_serial_`.
 */
SZ_HELPER_AUTO __m128i sz_emulate_aesenc_lasx_(__m128i state_u8x16, __m128i round_key_u8x16) {
    sz_u8_t const *tables = sz_aes_lasx_tables_();
    __m128i zero_u8x16 = __lsx_vreplgr2vr_b(0);
    __m128i low_nibble_mask_u8x16 = __lsx_vreplgr2vr_b(0x0F);
    __m128i input_transform_low_u8x16 = __lsx_vld(tables + 0, 0),
            input_transform_high_u8x16 = __lsx_vld(tables + 16, 0);
    __m128i sbox_output_low_u8x16 = __lsx_vld(tables + 32, 0), sbox_output_high_u8x16 = __lsx_vld(tables + 48, 0);
    __m128i gf16_log_u8x16 = __lsx_vld(tables + 64, 0), gf16_exp_u8x16 = __lsx_vld(tables + 80, 0);
    __m128i gf16_inverse_u8x16 = __lsx_vld(tables + 96, 0), shift_rows_indices_u8x16 = __lsx_vld(tables + 112, 0);
    __m128i shifted_state_u8x16 = sz_lsx_pshufb_(state_u8x16, shift_rows_indices_u8x16);
    __m128i sbox_output_u8x16 = sz_emulate_aes_subbytes_lasx_(
        shifted_state_u8x16, zero_u8x16, low_nibble_mask_u8x16, input_transform_low_u8x16, input_transform_high_u8x16,
        sbox_output_low_u8x16, sbox_output_high_u8x16, gf16_log_u8x16, gf16_exp_u8x16, gf16_inverse_u8x16);
    __m128i mixed_u8x16 = sz_emulate_aes_mixcolumns_lasx_(sbox_output_u8x16);
    return __lsx_vxor_v(mixed_u8x16, round_key_u8x16);
}
/** @brief  Load 16 bytes from an `sz_u128_vec_t`-style buffer into an LSX register. */
SZ_HELPER_INLINE __m128i sz_lsx_load128_(void const *pointer) { return __lsx_vld(pointer, 0); }
/** @brief  Store an LSX register into a 16-byte buffer. */
SZ_HELPER_INLINE void sz_lsx_store128_(void *pointer, __m128i value_u8x16) { __lsx_vst(value_u8x16, pointer, 0); }
SZ_HELPER_AUTO void sz_hash_state_short_init_lasx_(sz_hash_state_aligned_for_short_t *state, sz_u64_t seed) {
    state->key.u64s[0] = seed, state->key.u64s[1] = seed;
    sz_u64_t const *pi = sz_hash_pi_constants_();
    state->aes.u64s[0] = seed ^ pi[0], state->aes.u64s[1] = seed ^ pi[1];
    state->sum.u64s[0] = seed ^ pi[8], state->sum.u64s[1] = seed ^ pi[9];
}
SZ_HELPER_AUTO void sz_hash_state_short_update_lasx_(sz_hash_state_aligned_for_short_t *state,
                                                     sz_u128_vec_t block_vec) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    __m128i aes_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state->aes), sz_lsx_load128_(&block_vec));
    sz_lsx_store128_(&state->aes, aes_u8x16);
    state->sum = sz_emulate_shuffle_epi8_serial_(state->sum, shuffle);
    state->sum.u64s[0] += block_vec.u64s[0], state->sum.u64s[1] += block_vec.u64s[1];
}
SZ_HELPER_AUTO sz_u64_t sz_hash_state_short_finalize_lasx_(sz_hash_state_aligned_for_short_t const *state,
                                                           sz_size_t length) {
    sz_u128_vec_t key_with_length_vec = state->key;
    key_with_length_vec.u64s[0] += length;
    __m128i mixed_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state->sum), sz_lsx_load128_(&state->aes));
    __m128i mixed_in_register_u8x16 = sz_emulate_aesenc_lasx_(
        sz_emulate_aesenc_lasx_(mixed_u8x16, sz_lsx_load128_(&key_with_length_vec)), mixed_u8x16);
    sz_u128_vec_t result_vec;
    sz_lsx_store128_(&result_vec, mixed_in_register_u8x16);
    return result_vec.u64s[0];
}
SZ_API_COMPTIME void sz_hash_state_init_lasx(sz_hash_state_t *state, sz_u64_t seed);
SZ_HELPER_AUTO sz_hash_state_aligned_t sz_hash_state_load_lasx_(sz_hash_state_t const *packed);
SZ_HELPER_AUTO void sz_hash_state_store_lasx_(sz_hash_state_t *packed, sz_hash_state_aligned_t const *state);
SZ_HELPER_AUTO void sz_hash_state_update_lasx_(sz_hash_state_aligned_t *state);
SZ_HELPER_AUTO sz_u64_t sz_hash_state_finalize_lasx_(sz_hash_state_aligned_t state);
SZ_API_COMPTIME SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_lasx(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_lasx_(&state, seed);
        sz_u128_vec_t data_vec;
        data_vec.u64s[0] = data_vec.u64s[1] = 0;
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) data_vec.u8s[byte_index] = start[byte_index];
        sz_hash_state_short_update_lasx_(&state, data_vec);
        return sz_hash_state_short_finalize_lasx_(&state, length);
    }
    else if (length <= 32) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_lasx_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.lsx = __lsx_vld(start, 0);
        data1_vec.lsx = __lsx_vld(start + length - 16, 0);
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));
        sz_hash_state_short_update_lasx_(&state, data0_vec);
        sz_hash_state_short_update_lasx_(&state, data1_vec);
        return sz_hash_state_short_finalize_lasx_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_lasx_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.lsx = __lsx_vld(start, 0);
        data1_vec.lsx = __lsx_vld(start + 16, 0);
        data2_vec.lsx = __lsx_vld(start + length - 16, 0);
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));
        sz_hash_state_short_update_lasx_(&state, data0_vec);
        sz_hash_state_short_update_lasx_(&state, data1_vec);
        sz_hash_state_short_update_lasx_(&state, data2_vec);
        return sz_hash_state_short_finalize_lasx_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_lasx_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.lsx = __lsx_vld(start, 0);
        data1_vec.lsx = __lsx_vld(start + 16, 0);
        data2_vec.lsx = __lsx_vld(start + 32, 0);
        data3_vec.lsx = __lsx_vld(start + length - 16, 0);
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));
        sz_hash_state_short_update_lasx_(&state, data0_vec);
        sz_hash_state_short_update_lasx_(&state, data1_vec);
        sz_hash_state_short_update_lasx_(&state, data2_vec);
        sz_hash_state_short_update_lasx_(&state, data3_vec);
        return sz_hash_state_short_finalize_lasx_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_aligned_t state;
        sz_hash_state_init_lasx((sz_hash_state_t *)&state, seed);
        for (; state.ins_length + 64 < length; state.ins_length += 64) {
            __lasx_xvst(__lasx_xvld(start + state.ins_length, 0), state.ins.u8s, 0);
            __lasx_xvst(__lasx_xvld(start + state.ins_length + 32, 0), state.ins.u8s + 32, 0);
            sz_hash_state_update_lasx_(&state);
        }
        __m256i const zero_u8x32 = __lasx_xvreplgr2vr_b(0);
        __lasx_xvst(zero_u8x32, state.ins.u8s, 0);
        __lasx_xvst(zero_u8x32, state.ins.u8s + 32, 0);
        for (sz_size_t byte_index = 0; state.ins_length < length; ++byte_index, ++state.ins_length)
            state.ins.u8s[byte_index] = (sz_u8_t)start[state.ins_length];
        return sz_hash_state_finalize_lasx_(state);
    }
}
SZ_API_COMPTIME void sz_hash_state_init_lasx(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_serial(state, seed);
}
/**
 *  @brief Loads the packed public state into the aligned internal twin (LASX: 2x `__lasx_xvld` per 64-byte field).
 */
SZ_HELPER_AUTO sz_hash_state_aligned_t sz_hash_state_load_lasx_(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state;
    __lasx_xvst(__lasx_xvld(packed->aes, 0), state.aes.u8s, 0);
    __lasx_xvst(__lasx_xvld(packed->aes + 32, 0), state.aes.u8s + 32, 0);
    __lasx_xvst(__lasx_xvld(packed->sum, 0), state.sum.u8s, 0);
    __lasx_xvst(__lasx_xvld(packed->sum + 32, 0), state.sum.u8s + 32, 0);
    __lasx_xvst(__lasx_xvld(packed->ins, 0), state.ins.u8s, 0);
    __lasx_xvst(__lasx_xvld(packed->ins + 32, 0), state.ins.u8s + 32, 0);
    sz_lsx_store128_(&state.key, sz_lsx_load128_(packed->key));
    state.ins_length = packed->ins_length;
    return state;
}
/** @brief Stores the aligned internal twin back into the packed public state. */
SZ_HELPER_AUTO void sz_hash_state_store_lasx_(sz_hash_state_t *packed, sz_hash_state_aligned_t const *state) {
    __lasx_xvst(__lasx_xvld(state->aes.u8s, 0), packed->aes, 0);
    __lasx_xvst(__lasx_xvld(state->aes.u8s + 32, 0), packed->aes + 32, 0);
    __lasx_xvst(__lasx_xvld(state->sum.u8s, 0), packed->sum, 0);
    __lasx_xvst(__lasx_xvld(state->sum.u8s + 32, 0), packed->sum + 32, 0);
    __lasx_xvst(__lasx_xvld(state->ins.u8s, 0), packed->ins, 0);
    __lasx_xvst(__lasx_xvld(state->ins.u8s + 32, 0), packed->ins + 32, 0);
    sz_lsx_store128_(packed->key, state->key.lsx);
    packed->ins_length = state->ins_length;
}
/**
 *  @brief Absorbs the buffered 64-byte block into the aligned state (four 128-bit lanes), in place.
 *  @param state Pointer to the aligned hash state whose `ins` lanes are consumed.
 */
SZ_HELPER_AUTO void sz_hash_state_update_lasx_(sz_hash_state_aligned_t *state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        sz_u128_vec_t *aes_vec = &state->aes.u128s[lane_index];
        sz_u128_vec_t *sum_vec = &state->sum.u128s[lane_index];
        sz_u128_vec_t const *ins_vec = &state->ins.u128s[lane_index];
        sz_lsx_store128_(aes_vec, sz_emulate_aesenc_lasx_(sz_lsx_load128_(aes_vec), sz_lsx_load128_(ins_vec)));
        *sum_vec = sz_emulate_shuffle_epi8_serial_(*sum_vec, shuffle);
        sum_vec->u64s[0] += ins_vec->u64s[0], sum_vec->u64s[1] += ins_vec->u64s[1];
    }
}
/**
 *  @brief Finalizes the full 512-bit hash state and returns a 64-bit digest.
 *  @param state The hash state, taken by value.
 *  @return 64-bit hash value derived by folding the four AES lanes together with the key.
 */
SZ_HELPER_AUTO sz_u64_t sz_hash_state_finalize_lasx_(sz_hash_state_aligned_t state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    sz_u128_vec_t key_with_length_vec;
    key_with_length_vec.u64s[0] = state.key.u64s[0] + state.ins_length;
    key_with_length_vec.u64s[1] = state.key.u64s[1];
    __m128i aes0_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state.aes.u128s[0]),
                                                 sz_lsx_load128_(&state.ins.u128s[0]));
    __m128i aes1_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state.aes.u128s[1]),
                                                 sz_lsx_load128_(&state.ins.u128s[1]));
    __m128i aes2_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state.aes.u128s[2]),
                                                 sz_lsx_load128_(&state.ins.u128s[2]));
    __m128i aes3_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state.aes.u128s[3]),
                                                 sz_lsx_load128_(&state.ins.u128s[3]));
    sz_u128_vec_t sum0_vec = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[0], shuffle);
    sz_u128_vec_t sum1_vec = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[1], shuffle);
    sz_u128_vec_t sum2_vec = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[2], shuffle);
    sz_u128_vec_t sum3_vec = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[3], shuffle);
    sum0_vec.u64s[0] += state.ins.u128s[0].u64s[0], sum0_vec.u64s[1] += state.ins.u128s[0].u64s[1];
    sum1_vec.u64s[0] += state.ins.u128s[1].u64s[0], sum1_vec.u64s[1] += state.ins.u128s[1].u64s[1];
    sum2_vec.u64s[0] += state.ins.u128s[2].u64s[0], sum2_vec.u64s[1] += state.ins.u128s[2].u64s[1];
    sum3_vec.u64s[0] += state.ins.u128s[3].u64s[0], sum3_vec.u64s[1] += state.ins.u128s[3].u64s[1];
    __m128i mixed0_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&sum0_vec), aes0_u8x16);
    __m128i mixed1_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&sum1_vec), aes1_u8x16);
    __m128i mixed2_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&sum2_vec), aes2_u8x16);
    __m128i mixed3_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&sum3_vec), aes3_u8x16);
    __m128i mixed01_u8x16 = sz_emulate_aesenc_lasx_(mixed0_u8x16, mixed1_u8x16);
    __m128i mixed23_u8x16 = sz_emulate_aesenc_lasx_(mixed2_u8x16, mixed3_u8x16);
    __m128i mixed_u8x16 = sz_emulate_aesenc_lasx_(mixed01_u8x16, mixed23_u8x16);
    __m128i mixed_in_register_u8x16 = sz_emulate_aesenc_lasx_(
        sz_emulate_aesenc_lasx_(mixed_u8x16, sz_lsx_load128_(&key_with_length_vec)), mixed_u8x16);
    sz_u128_vec_t result_vec;
    sz_lsx_store128_(&result_vec, mixed_in_register_u8x16);
    return result_vec.u64s[0];
}
SZ_API_COMPTIME void sz_hash_state_update_lasx(sz_hash_state_t *packed, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_aligned_t state = sz_hash_state_load_lasx_(packed);
    __m256i const zero_u8x32 = __lasx_xvreplgr2vr_b(0);
    while (length) {
        sz_size_t progress_in_block = state.ins_length % 64;
        if (progress_in_block == 0 && state.ins_length != 0) {
            sz_hash_state_update_lasx_(&state);
            __lasx_xvst(zero_u8x32, state.ins.u8s, 0);
            __lasx_xvst(zero_u8x32, state.ins.u8s + 32, 0);
        }
        sz_size_t to_copy = sz_min_of_two(length, 64 - progress_in_block);
        state.ins_length += to_copy;
        length -= to_copy;
        while (to_copy--) state.ins.u8s[progress_in_block++] = (sz_u8_t)*text++;
    }
    sz_hash_state_store_lasx_(packed, &state);
}
SZ_API_COMPTIME sz_u64_t sz_hash_state_digest_lasx(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state = sz_hash_state_load_lasx_(packed);
    sz_size_t length = state.ins_length;
    if (length > 64) return sz_hash_state_finalize_lasx_(state);
    sz_hash_state_aligned_for_short_t minimal_state;
    minimal_state.key = state.key;
    minimal_state.aes = state.aes.u128s[0];
    minimal_state.sum = state.sum.u128s[0];
    if (length <= 16) {
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[0]);
        return sz_hash_state_short_finalize_lasx_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[1]);
        return sz_hash_state_short_finalize_lasx_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[1]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[2]);
        return sz_hash_state_short_finalize_lasx_(&minimal_state, length);
    }
    else {
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[1]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[2]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[3]);
        return sz_hash_state_short_finalize_lasx_(&minimal_state, length);
    }
}
SZ_API_COMPTIME void sz_fill_random_lasx(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_constants = sz_hash_pi_constants_();
    sz_u128_vec_t input_vec, pi_vec, key_vec, generated_vec;
    for (sz_size_t lane_index = 0; length; ++lane_index) {
        input_vec.u64s[0] = input_vec.u64s[1] = nonce + lane_index;
        pi_vec = ((sz_u128_vec_t const *)pi_constants)[lane_index % 4];
        key_vec.u64s[0] = nonce ^ pi_vec.u64s[0];
        key_vec.u64s[1] = nonce ^ pi_vec.u64s[1];
        __m128i gen_u8x16 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&input_vec), sz_lsx_load128_(&key_vec));
        sz_lsx_store128_(&generated_vec, gen_u8x16);
        for (sz_size_t byte_index = 0; byte_index < 16 && length; ++byte_index, --length)
            *text++ = generated_vec.u8s[byte_index];
    }
}
/** @brief  Lane-wise SHA-256 lowercase-sigma0: `ROTR(x,7) ^ ROTR(x,18) ^ SHR(x,3)` over 4x u32. */
SZ_HELPER_INLINE __m128i sz_sha256_sigma0_lower_lasx_(__m128i words_u32x4) {
    return __lsx_vxor_v(__lsx_vxor_v(__lsx_vrotri_w(words_u32x4, 7), __lsx_vrotri_w(words_u32x4, 18)),
                        __lsx_vsrli_w(words_u32x4, 3));
}
/** @brief  Lane-wise SHA-256 lowercase-sigma1: `ROTR(x,17) ^ ROTR(x,19) ^ SHR(x,10)` over 4x u32. */
SZ_HELPER_INLINE __m128i sz_sha256_sigma1_lower_lasx_(__m128i words_u32x4) {
    return __lsx_vxor_v(__lsx_vxor_v(__lsx_vrotri_w(words_u32x4, 17), __lsx_vrotri_w(words_u32x4, 19)),
                        __lsx_vsrli_w(words_u32x4, 10));
}
SZ_HELPER_AUTO void sz_sha256_process_block_lasx_(sz_u32_t hash[sz_at_least_(8)],
                                                  sz_u8_t const block[sz_at_least_(SZ_SHA256_BLOCK_LENGTH)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    sz_align_(16) sz_u32_t message_schedule[64];
    static
        sz_align_(16) sz_u8_t const byte_reverse_indices[16] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12};
    __m128i byte_reverse_u8x16 = __lsx_vld(byte_reverse_indices, 0);
    for (sz_size_t group_index = 0; group_index < 4; ++group_index) {
        __m128i raw_u8x16 = __lsx_vld(block + group_index * 16, 0);
        __lsx_vst(__lsx_vshuf_b(raw_u8x16, raw_u8x16, byte_reverse_u8x16), &message_schedule[group_index * 4], 0);
    }
    __m128i const low_pair_mask_u32x4 = __lsx_vinsgr2vr_d(__lsx_vreplgr2vr_d(-1), 0, 1); 
    for (sz_size_t word_index = 16; word_index < 64; word_index += 4) {
        __m128i window_16_u32x4 = __lsx_vld(&message_schedule[word_index - 16], 0); 
        __m128i window_15_u32x4 = __lsx_vld(&message_schedule[word_index - 15], 0); 
        __m128i window_7_u32x4 = __lsx_vld(&message_schedule[word_index - 7], 0);   
        __m128i base_u32x4 = __lsx_vadd_w(__lsx_vadd_w(window_16_u32x4, sz_sha256_sigma0_lower_lasx_(window_15_u32x4)),
                                          window_7_u32x4);
        __m128i prev_pair_u32x4 = __lsx_vand_v(__lsx_vld(&message_schedule[word_index - 2], 0), low_pair_mask_u32x4);
        __m128i schedule_u32x4 = __lsx_vadd_w(base_u32x4, sz_sha256_sigma1_lower_lasx_(prev_pair_u32x4));
        __m128i new_pair_u32x4 = __lsx_vbsll_v(schedule_u32x4, 8); 
        schedule_u32x4 = __lsx_vadd_w(schedule_u32x4, sz_sha256_sigma1_lower_lasx_(new_pair_u32x4));
        __lsx_vst(schedule_u32x4, &message_schedule[word_index], 0);
    }
    sz_u32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3];
    sz_u32_t e = hash[4], f = hash[5], g = hash[6], h = hash[7], temp1, temp2;
    for (sz_size_t round_index = 0; round_index < 64; ++round_index) {
        temp1 = h + sz_sha256_sigma1_(e) + sz_sha256_ch_(e, f, g) + round_constants[round_index] +
                message_schedule[round_index];
        temp2 = sz_sha256_sigma0_(a) + sz_sha256_maj_(a, b, c);
        h = g, g = f, f = e;
        e = d + temp1;
        d = c, c = b, b = a;
        a = temp1 + temp2;
    }
    hash[0] += a, hash[1] += b, hash[2] += c, hash[3] += d;
    hash[4] += e, hash[5] += f, hash[6] += g, hash[7] += h;
}
SZ_API_COMPTIME void sz_sha256_state_init_lasx(sz_sha256_state_t *state) { sz_sha256_state_init_serial(state); }
SZ_API_COMPTIME void sz_sha256_state_update_lasx(sz_sha256_state_t *state_ptr, sz_cptr_t data, sz_size_t length) {
    sz_u8_t const *input = (sz_u8_t const *)data;
    sz_size_t const current_block_index = state_ptr->block_length / SZ_SHA256_BLOCK_LENGTH;
    sz_size_t const final_block_index = (state_ptr->block_length + length) / SZ_SHA256_BLOCK_LENGTH;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state_ptr->block_length + length) % SZ_SHA256_BLOCK_LENGTH == 0;
    state_ptr->total_length += length;
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state_ptr->block_length, ++input) state_ptr->block[state_ptr->block_length] = *input;
        return;
    }
    sz_size_t const head_length = (SZ_SHA256_BLOCK_LENGTH - state_ptr->block_length) % SZ_SHA256_BLOCK_LENGTH;
    sz_size_t const tail_length = (state_ptr->block_length + length) % SZ_SHA256_BLOCK_LENGTH;
    sz_size_t const body_length = length - head_length - tail_length;
    sz_align_(32) sz_u32_t hash[8];
    hash[0] = state_ptr->hash[0], hash[1] = state_ptr->hash[1], hash[2] = state_ptr->hash[2],
    hash[3] = state_ptr->hash[3], hash[4] = state_ptr->hash[4], hash[5] = state_ptr->hash[5],
    hash[6] = state_ptr->hash[6], hash[7] = state_ptr->hash[7];
    if (head_length) {
        for (sz_size_t byte_index = 0; byte_index < head_length; ++byte_index)
            state_ptr->block[state_ptr->block_length++] = input[byte_index];
        sz_sha256_process_block_lasx_(hash, state_ptr->block);
        state_ptr->block_length = 0;
        input += head_length;
    }
    for (sz_size_t processed = 0; processed < body_length;
         processed += SZ_SHA256_BLOCK_LENGTH, input += SZ_SHA256_BLOCK_LENGTH)
        sz_sha256_process_block_lasx_(hash, input);
    for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
        state_ptr->block[byte_index] = input[byte_index];
    state_ptr->block_length = tail_length;
    state_ptr->hash[0] = hash[0], state_ptr->hash[1] = hash[1], state_ptr->hash[2] = hash[2],
    state_ptr->hash[3] = hash[3], state_ptr->hash[4] = hash[4], state_ptr->hash[5] = hash[5],
    state_ptr->hash[6] = hash[6], state_ptr->hash[7] = hash[7];
}
SZ_API_COMPTIME void sz_sha256_state_digest_lasx(sz_sha256_state_t const *state_ptr,
                                                 sz_u8_t digest[sz_at_least_(SZ_SHA256_DIGEST_LENGTH)]) {
    sz_sha256_state_t state = *state_ptr;
    state.block[state.block_length++] = 0x80;
    if (state.block_length > 56) {
        for (sz_size_t byte_index = state.block_length; byte_index < SZ_SHA256_BLOCK_LENGTH; ++byte_index)
            state.block[byte_index] = 0;
        sz_sha256_process_block_lasx_(state.hash, state.block);
        state.block_length = 0;
    }
    for (sz_size_t byte_index = state.block_length; byte_index < 56; ++byte_index) state.block[byte_index] = 0;
    state.block_length = 56;
    sz_u64_t bit_length = state.total_length * 8;
    state.block[56] = (sz_u8_t)(bit_length >> 56);
    state.block[57] = (sz_u8_t)(bit_length >> 48);
    state.block[58] = (sz_u8_t)(bit_length >> 40);
    state.block[59] = (sz_u8_t)(bit_length >> 32);
    state.block[60] = (sz_u8_t)(bit_length >> 24);
    state.block[61] = (sz_u8_t)(bit_length >> 16);
    state.block[62] = (sz_u8_t)(bit_length >> 8);
    state.block[63] = (sz_u8_t)(bit_length >> 0);
    sz_sha256_process_block_lasx_(state.hash, state.block);
    for (sz_size_t word_index = 0; word_index < 8; ++word_index) {
        digest[word_index * 4 + 0] = (sz_u8_t)(state.hash[word_index] >> 24);
        digest[word_index * 4 + 1] = (sz_u8_t)(state.hash[word_index] >> 16);
        digest[word_index * 4 + 2] = (sz_u8_t)(state.hash[word_index] >> 8);
        digest[word_index * 4 + 3] = (sz_u8_t)(state.hash[word_index] >> 0);
    }
}
#endif 
#ifdef __cplusplus
}
#endif
#endif 
