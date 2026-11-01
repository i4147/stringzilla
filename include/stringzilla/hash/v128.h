/**
 *  @brief WebAssembly SIMD128 backend for hash.
 *  @file include/stringzilla/hash/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_V128_H_
#define STRINGZILLA_HASH_V128_H_
#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/compare.h"     
#include "stringzilla/memory/v128.h" 
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif
SZ_API_COMPTIME sz_u64_t sz_bytesum_v128(sz_cptr_t text, sz_size_t length) {
    sz_u128_vec_t sum16_vec, sum64_vec;
    sum16_vec.v128 = wasm_u64x2_splat(0); 
    sum64_vec.v128 = wasm_u64x2_splat(0); 
    while (length >= 16) {
        sz_size_t blocks = length / 16;
        if (blocks > 128) blocks = 128;
        length -= blocks * 16;
        for (sz_size_t block_index = 0; block_index < blocks; ++block_index, text += 16) {
            v128_t vec_u8x16 = wasm_v128_load(text);
            sum16_vec.v128 = wasm_i16x8_add(sum16_vec.v128, wasm_u16x8_extadd_pairwise_u8x16(vec_u8x16));
        }
        v128_t pair32_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum16_vec.v128);
        sum64_vec.v128 = wasm_i64x2_add( 
            sum64_vec.v128,
            wasm_i64x2_add(wasm_u64x2_extend_low_u32x4(pair32_u32x4), wasm_u64x2_extend_high_u32x4(pair32_u32x4)));
        sum16_vec.v128 = wasm_u64x2_splat(0);
    }
    sz_u64_t sum = sum64_vec.u64s[0] + sum64_vec.u64s[1];
    while (length--) sum += *(sz_u8_t const *)text++;
    return sum;
}
/** @brief Evaluate a GF(2)-linear byte map `x -> low_table[x&0xF] ^ high_table[x>>4]` across all 16 lanes. */
SZ_HELPER_INLINE v128_t sz_aes_linear_v128_(v128_t low_table_u8x16, v128_t high_table_u8x16, v128_t x_u8x16) {
    v128_t low_nibbles_u8x16 = wasm_v128_and(x_u8x16, wasm_i8x16_splat((sz_i8_t)0x0F));
    v128_t high_nibbles_u8x16 = wasm_u8x16_shr(x_u8x16, 4);
    return wasm_v128_xor(wasm_i8x16_swizzle(low_table_u8x16, low_nibbles_u8x16),
                         wasm_i8x16_swizzle(high_table_u8x16, high_nibbles_u8x16));
}
/** @brief GF(2^4) (modulus `x^4+x+1`) lane-wise multiply via log/antilog swizzles with zero masking. */
SZ_HELPER_AUTO v128_t sz_aes_gf4_mul_v128_(v128_t a_u8x16, v128_t b_u8x16) {
    static sz_align_(16) sz_u8_t const log_table[16] = {0x00, 0x00, 0x01, 0x04, 0x02, 0x08, 0x05, 0x0a,
                                                        0x03, 0x0e, 0x09, 0x07, 0x06, 0x0d, 0x0b, 0x0c};
    static sz_align_(16) sz_u8_t const exp_lo_table[16] = {0x01, 0x02, 0x04, 0x08, 0x03, 0x06, 0x0c, 0x0b,
                                                           0x05, 0x0a, 0x07, 0x0e, 0x0f, 0x0d, 0x09, 0x01};
    static sz_align_(16) sz_u8_t const exp_hi_table[16] = {0x02, 0x04, 0x08, 0x03, 0x06, 0x0c, 0x0b, 0x05,
                                                           0x0a, 0x07, 0x0e, 0x0f, 0x0d, 0x09, 0x00, 0x00};
    v128_t log_u8x16 = wasm_v128_load(log_table);
    v128_t exp_lo_u8x16 = wasm_v128_load(exp_lo_table);
    v128_t exp_hi_u8x16 = wasm_v128_load(exp_hi_table);
    v128_t zero_u8x16 = wasm_i8x16_splat(0);
    v128_t sum_u8x16 = wasm_i8x16_add(wasm_i8x16_swizzle(log_u8x16, a_u8x16), wasm_i8x16_swizzle(log_u8x16, b_u8x16));
    v128_t product_u8x16 = wasm_v128_xor(
        wasm_i8x16_swizzle(exp_lo_u8x16, sum_u8x16),
        wasm_i8x16_swizzle(exp_hi_u8x16, wasm_i8x16_sub(sum_u8x16, wasm_i8x16_splat(16))));
    v128_t any_zero_u8x16 = wasm_v128_or(wasm_i8x16_eq(a_u8x16, zero_u8x16), wasm_i8x16_eq(b_u8x16, zero_u8x16));
    return wasm_v128_andnot(product_u8x16, any_zero_u8x16);
}
/**
 *  @brief Bit-exact `_mm_aesenc_si128` for one round, identical to `sz_emulate_aesenc_si128_serial_`.
 *  @return `MixColumns(SubBytes(ShiftRows(state))) ^ round_key`, computed with the vpaes tower field.
 */
SZ_HELPER_AUTO sz_u128_vec_t sz_emulate_aesenc_v128_(sz_u128_vec_t state_vec, sz_u128_vec_t round_key_vec) {
    static sz_align_(16) sz_u8_t const fwd_lo[16] = {0x00, 0x01, 0x20, 0x21, 0x46, 0x47, 0x66, 0x67,
                                                     0x4c, 0x4d, 0x6c, 0x6d, 0x0a, 0x0b, 0x2a, 0x2b};
    static sz_align_(16) sz_u8_t const fwd_hi[16] = {0x00, 0x3c, 0xd5, 0xe9, 0x34, 0x08, 0xe1, 0xdd,
                                                     0xe5, 0xd9, 0x30, 0x0c, 0xd1, 0xed, 0x04, 0x38};
    static sz_align_(16) sz_u8_t const inv_lo[16] = {0x00, 0x01, 0x5c, 0x5d, 0xe0, 0xe1, 0xbc, 0xbd,
                                                     0x50, 0x51, 0x0c, 0x0d, 0xb0, 0xb1, 0xec, 0xed};
    static sz_align_(16) sz_u8_t const inv_hi[16] = {0x00, 0xa2, 0x02, 0xa0, 0xb8, 0x1a, 0xba, 0x18,
                                                     0xdb, 0x79, 0xd9, 0x7b, 0x63, 0xc1, 0x61, 0xc3};
    static sz_align_(16) sz_u8_t const aff_lo[16] = {0x63, 0x7c, 0x5d, 0x42, 0x1f, 0x00, 0x21, 0x3e,
                                                     0x9b, 0x84, 0xa5, 0xba, 0xe7, 0xf8, 0xd9, 0xc6};
    static sz_align_(16) sz_u8_t const aff_hi[16] = {0x00, 0xf1, 0xe3, 0x12, 0xc7, 0x36, 0x24, 0xd5,
                                                     0x8f, 0x7e, 0x6c, 0x9d, 0x48, 0xb9, 0xab, 0x5a};
    static sz_align_(16) sz_u8_t const gf4_inv[16] = {0x00, 0x01, 0x09, 0x0e, 0x0d, 0x0b, 0x07, 0x06,
                                                      0x0f, 0x02, 0x0c, 0x05, 0x0a, 0x04, 0x03, 0x08};
    static sz_align_(16) sz_u8_t const gf4_sqr[16] = {0x00, 0x01, 0x04, 0x05, 0x03, 0x02, 0x07, 0x06,
                                                      0x0c, 0x0d, 0x08, 0x09, 0x0f, 0x0e, 0x0b, 0x0a};
    static sz_align_(16) sz_u8_t const gf4_sqr_n[16] = {0x00, 0x08, 0x06, 0x0e, 0x0b, 0x03, 0x0d, 0x05,
                                                        0x0a, 0x02, 0x0c, 0x04, 0x01, 0x09, 0x07, 0x0f};
    static sz_align_(16) sz_u8_t const shift_rows[16] = {0x00, 0x05, 0x0a, 0x0f, 0x04, 0x09, 0x0e, 0x03,
                                                         0x08, 0x0d, 0x02, 0x07, 0x0c, 0x01, 0x06, 0x0b};
    static sz_align_(16) sz_u8_t const next_col[16] = {0x01, 0x02, 0x03, 0x00, 0x05, 0x06, 0x07, 0x04,
                                                       0x09, 0x0a, 0x0b, 0x08, 0x0d, 0x0e, 0x0f, 0x0c};
    v128_t state_u8x16 = state_vec.v128;
    v128_t low_nibble_mask_u8x16 = wasm_i8x16_splat((sz_i8_t)0x0F);
    v128_t mapped_u8x16 = sz_aes_linear_v128_(wasm_v128_load(fwd_lo), wasm_v128_load(fwd_hi), state_u8x16);
    v128_t high_nibble_u8x16 = wasm_u8x16_shr(mapped_u8x16, 4);                   
    v128_t low_nibble_u8x16 = wasm_v128_and(mapped_u8x16, low_nibble_mask_u8x16); 
    v128_t high_sqr_n_u8x16 = wasm_i8x16_swizzle(wasm_v128_load(gf4_sqr_n), high_nibble_u8x16);
    v128_t high_low_product_u8x16 = sz_aes_gf4_mul_v128_(high_nibble_u8x16, low_nibble_u8x16);
    v128_t low_sqr_u8x16 = wasm_i8x16_swizzle(wasm_v128_load(gf4_sqr), low_nibble_u8x16);
    v128_t d_u8x16 = wasm_v128_xor(wasm_v128_xor(high_sqr_n_u8x16, high_low_product_u8x16), low_sqr_u8x16);
    v128_t d_inv_u8x16 = wasm_i8x16_swizzle(wasm_v128_load(gf4_inv), d_u8x16);
    v128_t high_inverse_u8x16 = sz_aes_gf4_mul_v128_(high_nibble_u8x16, d_inv_u8x16);
    v128_t low_inverse_u8x16 = sz_aes_gf4_mul_v128_(wasm_v128_xor(high_nibble_u8x16, low_nibble_u8x16), d_inv_u8x16);
    v128_t inverted_u8x16 = wasm_v128_or(wasm_i8x16_shl(high_inverse_u8x16, 4),
                                         low_inverse_u8x16); 
    v128_t back_u8x16 = sz_aes_linear_v128_(wasm_v128_load(inv_lo), wasm_v128_load(inv_hi), inverted_u8x16);
    v128_t subbed_u8x16 = sz_aes_linear_v128_(wasm_v128_load(aff_lo), wasm_v128_load(aff_hi), back_u8x16);
    v128_t shifted_u8x16 = wasm_i8x16_swizzle(subbed_u8x16, wasm_v128_load(shift_rows));
    v128_t col_table_u8x16 = wasm_v128_load(next_col);
    v128_t col1_u8x16 = wasm_i8x16_swizzle(shifted_u8x16, col_table_u8x16);
    v128_t col2_u8x16 = wasm_i8x16_swizzle(col1_u8x16, col_table_u8x16);
    v128_t col3_u8x16 = wasm_i8x16_swizzle(col2_u8x16, col_table_u8x16);
    v128_t row_xor_u8x16 = wasm_v128_xor(wasm_v128_xor(shifted_u8x16, col1_u8x16),
                                         wasm_v128_xor(col2_u8x16, col3_u8x16));
    v128_t diff_u8x16 = wasm_v128_xor(shifted_u8x16, col1_u8x16); 
    v128_t need_reduce_u8x16 = wasm_i8x16_shr(diff_u8x16, 7);     
    v128_t reduce_u8x16 = wasm_v128_and(need_reduce_u8x16, wasm_i8x16_splat((sz_i8_t)0x1B));
    v128_t xtime_u8x16 = wasm_v128_xor(wasm_i8x16_shl(diff_u8x16, 1), reduce_u8x16);
    v128_t mixed_u8x16 = wasm_v128_xor(wasm_v128_xor(shifted_u8x16, row_xor_u8x16), xtime_u8x16);
    sz_u128_vec_t result_vec;
    result_vec.v128 = wasm_v128_xor(mixed_u8x16, round_key_vec.v128);
    return result_vec;
}
/** @brief Vectorized counterpart of `sz_emulate_shuffle_epi8_serial_` using one `wasm_i8x16_swizzle`. */
SZ_HELPER_INLINE sz_u128_vec_t sz_emulate_shuffle_epi8_v128_(sz_u128_vec_t state_vec, v128_t order_u8x16) {
    sz_u128_vec_t result_vec;
    result_vec.v128 = wasm_i8x16_swizzle(state_vec.v128, order_u8x16);
    return result_vec;
}
#pragma region Hash with SIMD128 AES
SZ_HELPER_AUTO void sz_hash_state_short_init_v128_(sz_hash_state_aligned_for_short_t *state, sz_u64_t seed) {
    sz_hash_state_short_init_serial_(state, seed);
}
SZ_HELPER_AUTO void sz_hash_state_short_update_v128_(sz_hash_state_aligned_for_short_t *state,
                                                     sz_u128_vec_t block_vec) {
    v128_t shuffle_u8x16 = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    state->aes = sz_emulate_aesenc_v128_(state->aes, block_vec);
    state->sum = sz_emulate_shuffle_epi8_v128_(state->sum, shuffle_u8x16);
    state->sum.v128 = wasm_i64x2_add(state->sum.v128, block_vec.v128);
}
SZ_HELPER_AUTO sz_u64_t sz_hash_state_short_finalize_v128_(sz_hash_state_aligned_for_short_t const *state,
                                                           sz_size_t length) {
    sz_u128_vec_t key_with_length_vec = state->key;
    key_with_length_vec.u64s[0] += length;
    sz_u128_vec_t mixed_vec = sz_emulate_aesenc_v128_(state->sum, state->aes);
    sz_u128_vec_t mixed_in_register_vec = sz_emulate_aesenc_v128_(
        sz_emulate_aesenc_v128_(mixed_vec, key_with_length_vec), mixed_vec);
    return mixed_in_register_vec.u64s[0];
}
/** @brief Loads the packed public state into the aligned internal twin (4x `wasm_v128_load` per 64-byte field). */
SZ_HELPER_AUTO sz_hash_state_aligned_t sz_hash_state_load_v128_(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state;
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        sz_size_t const offset = lane_index * 16;
        state.aes.u128s[lane_index].v128 = wasm_v128_load(packed->aes + offset);
        state.sum.u128s[lane_index].v128 = wasm_v128_load(packed->sum + offset);
        state.ins.u128s[lane_index].v128 = wasm_v128_load(packed->ins + offset);
    }
    state.key.v128 = wasm_v128_load(packed->key);
    state.ins_length = packed->ins_length;
    return state;
}
/** @brief Stores the aligned internal twin back into the packed public state. */
SZ_HELPER_AUTO void sz_hash_state_store_v128_(sz_hash_state_t *packed, sz_hash_state_aligned_t const *state) {
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        sz_size_t const offset = lane_index * 16;
        wasm_v128_store(packed->aes + offset, state->aes.u128s[lane_index].v128);
        wasm_v128_store(packed->sum + offset, state->sum.u128s[lane_index].v128);
        wasm_v128_store(packed->ins + offset, state->ins.u128s[lane_index].v128);
    }
    wasm_v128_store(packed->key, state->key.v128);
    packed->ins_length = state->ins_length;
}
SZ_HELPER_AUTO void sz_hash_state_update_v128_(sz_hash_state_aligned_t *state) {
    v128_t shuffle_u8x16 = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        sz_u128_vec_t ins_vec = state->ins.u128s[lane_index];
        state->aes.u128s[lane_index] = sz_emulate_aesenc_v128_(state->aes.u128s[lane_index], ins_vec);
        state->sum.u128s[lane_index] = sz_emulate_shuffle_epi8_v128_(state->sum.u128s[lane_index], shuffle_u8x16);
        state->sum.u128s[lane_index].v128 = wasm_i64x2_add(state->sum.u128s[lane_index].v128, ins_vec.v128);
    }
}
SZ_HELPER_AUTO sz_u64_t sz_hash_state_finalize_v128_(sz_hash_state_aligned_t state) {
    v128_t shuffle_u8x16 = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    sz_u128_vec_t key_with_length_vec;
    key_with_length_vec.u64s[0] = state.key.u64s[0] + state.ins_length;
    key_with_length_vec.u64s[1] = state.key.u64s[1];
    sz_u128_vec_t ins0_vec = state.ins.u128s[0], ins1_vec = state.ins.u128s[1];
    sz_u128_vec_t ins2_vec = state.ins.u128s[2], ins3_vec = state.ins.u128s[3];
    sz_u128_vec_t aes0_vec = sz_emulate_aesenc_v128_(state.aes.u128s[0], ins0_vec);
    sz_u128_vec_t aes1_vec = sz_emulate_aesenc_v128_(state.aes.u128s[1], ins1_vec);
    sz_u128_vec_t aes2_vec = sz_emulate_aesenc_v128_(state.aes.u128s[2], ins2_vec);
    sz_u128_vec_t aes3_vec = sz_emulate_aesenc_v128_(state.aes.u128s[3], ins3_vec);
    sz_u128_vec_t sum0_vec = sz_emulate_shuffle_epi8_v128_(state.sum.u128s[0], shuffle_u8x16);
    sz_u128_vec_t sum1_vec = sz_emulate_shuffle_epi8_v128_(state.sum.u128s[1], shuffle_u8x16);
    sz_u128_vec_t sum2_vec = sz_emulate_shuffle_epi8_v128_(state.sum.u128s[2], shuffle_u8x16);
    sz_u128_vec_t sum3_vec = sz_emulate_shuffle_epi8_v128_(state.sum.u128s[3], shuffle_u8x16);
    sum0_vec.v128 = wasm_i64x2_add(sum0_vec.v128, ins0_vec.v128);
    sum1_vec.v128 = wasm_i64x2_add(sum1_vec.v128, ins1_vec.v128);
    sum2_vec.v128 = wasm_i64x2_add(sum2_vec.v128, ins2_vec.v128);
    sum3_vec.v128 = wasm_i64x2_add(sum3_vec.v128, ins3_vec.v128);
    sz_u128_vec_t mixed0_vec = sz_emulate_aesenc_v128_(sum0_vec, aes0_vec);
    sz_u128_vec_t mixed1_vec = sz_emulate_aesenc_v128_(sum1_vec, aes1_vec);
    sz_u128_vec_t mixed2_vec = sz_emulate_aesenc_v128_(sum2_vec, aes2_vec);
    sz_u128_vec_t mixed3_vec = sz_emulate_aesenc_v128_(sum3_vec, aes3_vec);
    sz_u128_vec_t mixed01_vec = sz_emulate_aesenc_v128_(mixed0_vec, mixed1_vec);
    sz_u128_vec_t mixed23_vec = sz_emulate_aesenc_v128_(mixed2_vec, mixed3_vec);
    sz_u128_vec_t mixed_vec = sz_emulate_aesenc_v128_(mixed01_vec, mixed23_vec);
    sz_u128_vec_t mixed_in_register_vec = sz_emulate_aesenc_v128_(
        sz_emulate_aesenc_v128_(mixed_vec, key_with_length_vec), mixed_vec);
    return mixed_in_register_vec.u64s[0];
}
SZ_API_COMPTIME SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_v128(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_v128_(&state, seed);
        sz_u128_vec_t data_vec;
        data_vec.v128 = sz_load_partial_v128_(start, length);
        sz_hash_state_short_update_v128_(&state, data_vec);
        return sz_hash_state_short_finalize_v128_(&state, length);
    }
    else if (length <= 32) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_v128_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));
        sz_hash_state_short_update_v128_(&state, data0_vec);
        sz_hash_state_short_update_v128_(&state, data1_vec);
        return sz_hash_state_short_finalize_v128_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_v128_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + 16);
        data2_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));
        sz_hash_state_short_update_v128_(&state, data0_vec);
        sz_hash_state_short_update_v128_(&state, data1_vec);
        sz_hash_state_short_update_v128_(&state, data2_vec);
        return sz_hash_state_short_finalize_v128_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_v128_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + 16);
        data2_vec.v128 = wasm_v128_load(start + 32);
        data3_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));
        sz_hash_state_short_update_v128_(&state, data0_vec);
        sz_hash_state_short_update_v128_(&state, data1_vec);
        sz_hash_state_short_update_v128_(&state, data2_vec);
        sz_hash_state_short_update_v128_(&state, data3_vec);
        return sz_hash_state_short_finalize_v128_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_aligned_t state;
        sz_hash_state_init_serial((sz_hash_state_t *)&state, seed);
        for (; state.ins_length + 64 < length; state.ins_length += 64) {
            for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index)
                state.ins.u128s[lane_index].v128 = wasm_v128_load(start + state.ins_length + lane_index * 16);
            sz_hash_state_update_v128_(&state);
        }
        v128_t const zero_u64x2 = wasm_u64x2_splat(0);
        for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) state.ins.u128s[lane_index].v128 = zero_u64x2;
        sz_size_t const tail_length = length - state.ins_length;
        for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
            state.ins.u8s[byte_index] = start[state.ins_length + byte_index];
        state.ins_length = length;
        return sz_hash_state_finalize_v128_(state);
    }
}
SZ_API_COMPTIME void sz_hash_state_init_v128(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_serial(state, seed);
}
SZ_API_COMPTIME void sz_hash_state_update_v128(sz_hash_state_t *packed, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_aligned_t state = sz_hash_state_load_v128_(packed);
    v128_t const zero_u64x2 = wasm_u64x2_splat(0);
    sz_size_t buffered = state.ins_length % 64;
    if (buffered == 0 && state.ins_length) buffered = 64;
    while (length) {
        if (buffered == 64) { 
            sz_hash_state_update_v128_(&state);
            for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) state.ins.u128s[lane_index].v128 = zero_u64x2;
            buffered = 0;
        }
        sz_size_t const take = sz_min_of_two(length, (sz_size_t)64 - buffered);
        for (sz_size_t byte_index = 0; byte_index < take; ++byte_index)
            state.ins.u8s[buffered + byte_index] = text[byte_index];
        buffered += take, text += take, length -= take, state.ins_length += take;
    }
    sz_hash_state_store_v128_(packed, &state);
}
SZ_API_COMPTIME sz_u64_t sz_hash_state_digest_v128(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state = sz_hash_state_load_v128_(packed);
    sz_size_t length = state.ins_length;
    if (length > 64) return sz_hash_state_finalize_v128_(state);
    sz_hash_state_aligned_for_short_t minimal_state;
    minimal_state.key = state.key;
    minimal_state.aes = state.aes.u128s[0];
    minimal_state.sum = state.sum.u128s[0];
    if (length <= 16) {
        sz_hash_state_short_update_v128_(&minimal_state, state.ins.u128s[0]);
        return sz_hash_state_short_finalize_v128_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_state_short_update_v128_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_v128_(&minimal_state, state.ins.u128s[1]);
        return sz_hash_state_short_finalize_v128_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_state_short_update_v128_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_v128_(&minimal_state, state.ins.u128s[1]);
        sz_hash_state_short_update_v128_(&minimal_state, state.ins.u128s[2]);
        return sz_hash_state_short_finalize_v128_(&minimal_state, length);
    }
    else {
        sz_hash_state_short_update_v128_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_v128_(&minimal_state, state.ins.u128s[1]);
        sz_hash_state_short_update_v128_(&minimal_state, state.ins.u128s[2]);
        sz_hash_state_short_update_v128_(&minimal_state, state.ins.u128s[3]);
        return sz_hash_state_short_finalize_v128_(&minimal_state, length);
    }
}
SZ_API_COMPTIME void sz_fill_random_v128(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_constants = sz_hash_pi_constants_();
    sz_u128_vec_t input_vec, pi_vec, key_vec, generated_vec;
    for (sz_size_t lane_index = 0; length; ++lane_index) {
        input_vec.v128 = wasm_u64x2_splat(nonce + lane_index); 
        pi_vec = ((sz_u128_vec_t const *)pi_constants)[lane_index % 4];
        key_vec.v128 = wasm_v128_xor(pi_vec.v128, wasm_u64x2_splat(nonce)); 
        generated_vec = sz_emulate_aesenc_v128_(input_vec, key_vec);
        if (length >= 16) { wasm_v128_store(text, generated_vec.v128), text += 16, length -= 16; }
        else { sz_store_partial_v128_(text, generated_vec.v128, length), length = 0; }
    }
}
#pragma region Multi Seed Hashing
/**
 *  @brief Replays prepared text-lanes through the SIMD128 minimal AES state for a single seed.
 *  @return 64-bit hash, bit-identical to `sz_hash_v128(text, length, seed)` (hence to serial).
 *  @sa sz_hash_multiseed_replay_serial_, sz_hash_multiseed_prepare_serial_
 */
SZ_HELPER_AUTO sz_u64_t sz_hash_multiseed_replay_v128_(sz_u512_vec_t const *text_lanes_vec, sz_size_t text_lanes_count,
                                                       sz_size_t length, sz_u64_t seed) {
    sz_align_(16) sz_hash_state_aligned_for_short_t state;
    sz_hash_state_short_init_v128_(&state, seed);
    for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
        sz_hash_state_short_update_v128_(&state, text_lanes_vec->u128s[lane_index]);
    return sz_hash_state_short_finalize_v128_(&state, length);
}
/**
 *  @brief Hashes one `text` under many `seeds`, sharing a single normalization pass for short inputs.
 *
 *  The branchy load + de-interleave of a `<= 64` byte input depends only on `(text, length)`, so it is
 *  done exactly once (`sz_hash_multiseed_prepare_serial_`) and replayed through the cheap SIMD128 AES
 *  rounds per seed — amortizing the input-load/tail cost that dominates short-string hashing. On a
 *  128-bit register each seed's minimal state is a full vector, so seeds are not lane-packed; the win is
 *  the shared normalization plus the faster v128 AES emulation. Bit-identical to `sz_hash_multiseed_serial`.
 */
SZ_API_COMPTIME void sz_hash_multiseed_v128(sz_cptr_t text, sz_size_t length,             
                                            sz_u64_t const *seeds, sz_size_t seeds_count, 
                                            sz_u64_t *hashes) {
    if (seeds_count == 0) return;
    if (seeds_count == 1) {
        hashes[0] = sz_hash_v128(text, length, seeds[0]);
        return;
    }
    if (length <= 64) {
        sz_u512_vec_t text_lanes_vec;
        sz_size_t const text_lanes_count = sz_hash_multiseed_prepare_serial_(text, length, &text_lanes_vec);
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_multiseed_replay_v128_(&text_lanes_vec, text_lanes_count, length,
                                                                seeds[seed_index]);
    }
    else {
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_v128(text, length, seeds[seed_index]);
    }
}
#pragma endregion // Multi Seed Hashing
#pragma endregion // Hash with SIMD128 AES
/** @brief 32-bit lane-wise rotate-right (no native WASM rotate; built from two shifts and an OR). */
SZ_HELPER_INLINE v128_t sz_sha256_rotr_v128_(v128_t x_u32x4, int count) {
    return wasm_v128_or(wasm_u32x4_shr(x_u32x4, count), wasm_i32x4_shl(x_u32x4, 32 - count));
}
/** @brief 4-wide `sigma0` of the message schedule: ROTR(x,7) ^ ROTR(x,18) ^ SHR(x,3). */
SZ_HELPER_INLINE v128_t sz_sha256_sigma0_lower_v128_(v128_t x_u32x4) {
    return wasm_v128_xor(wasm_v128_xor(sz_sha256_rotr_v128_(x_u32x4, 7), sz_sha256_rotr_v128_(x_u32x4, 18)),
                         wasm_u32x4_shr(x_u32x4, 3));
}
/** @brief 4-wide `sigma1` of the message schedule: ROTR(x,17) ^ ROTR(x,19) ^ SHR(x,10). */
SZ_HELPER_INLINE v128_t sz_sha256_sigma1_lower_v128_(v128_t x_u32x4) {
    return wasm_v128_xor(wasm_v128_xor(sz_sha256_rotr_v128_(x_u32x4, 17), sz_sha256_rotr_v128_(x_u32x4, 19)),
                         wasm_u32x4_shr(x_u32x4, 10));
}
/**
 *  @brief Process one 64-byte block with a SIMD message schedule, bit-exact with the serial reference.
 *
 *  The 64-round compression is sequential in `a..h`, so it stays scalar; the gain is the message
 *  schedule, computed four words per `i32x4` (the Gueron-Krasnov layout). `W[i]` needs `sigma1(W[i-2])`,
 *  so within a group of four the upper two lanes depend on the lower two just computed — handled with a
 *  two-phase `sigma1` and a final lane blend. The input words are loaded big-endian via one shuffle.
 */
SZ_HELPER_AUTO void sz_sha256_process_block_v128_(sz_u32_t hash[sz_at_least_(8)],
                                                  sz_u8_t const block[sz_at_least_(SZ_SHA256_BLOCK_LENGTH)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    sz_align_(16) sz_u32_t w[64];
    for (sz_size_t group_index = 0; group_index < 4; ++group_index) {
        v128_t loaded_u8x16 = wasm_v128_load(block + group_index * 16);
        wasm_v128_store(&w[group_index * 4], wasm_i8x16_shuffle(loaded_u8x16, loaded_u8x16, 3, 2, 1, 0, 7, 6, 5, 4, 11,
                                                                10, 9, 8, 15, 14, 13, 12));
    }
    for (sz_size_t word_index = 16; word_index < 64; word_index += 4) {
        v128_t w_m16_u32x4 = wasm_v128_load(&w[word_index - 16]);
        v128_t w_m15_u32x4 = wasm_v128_load(&w[word_index - 15]);
        v128_t w_m7_u32x4 = wasm_v128_load(&w[word_index - 7]);
        v128_t w_m2_u32x4 = wasm_v128_load(
            &w[word_index - 2]); 
        v128_t partial_u32x4 = wasm_i32x4_add(wasm_i32x4_add(w_m16_u32x4, sz_sha256_sigma0_lower_v128_(w_m15_u32x4)),
                                              w_m7_u32x4);
        v128_t t0_u32x4 = wasm_i32x4_add(partial_u32x4, sz_sha256_sigma1_lower_v128_(w_m2_u32x4));
        v128_t w_hi_u32x4 = wasm_i32x4_shuffle(t0_u32x4, t0_u32x4, 0, 1, 0, 1); 
        v128_t t1_u32x4 = wasm_i32x4_add(partial_u32x4, sz_sha256_sigma1_lower_v128_(w_hi_u32x4));
        wasm_v128_store(&w[word_index], wasm_i32x4_shuffle(t0_u32x4, t1_u32x4, 0, 1, 6, 7));
    }
    sz_u32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3];
    sz_u32_t e = hash[4], f = hash[5], g = hash[6], h = hash[7];
    for (sz_size_t round_index = 0; round_index < 64; ++round_index) {
        sz_u32_t temp1 = h + sz_sha256_sigma1_(e) + sz_sha256_ch_(e, f, g) + round_constants[round_index] +
                         w[round_index];
        sz_u32_t temp2 = sz_sha256_sigma0_(a) + sz_sha256_maj_(a, b, c);
        h = g, g = f, f = e;
        e = d + temp1;
        d = c, c = b, b = a;
        a = temp1 + temp2;
    }
    hash[0] += a, hash[1] += b, hash[2] += c, hash[3] += d;
    hash[4] += e, hash[5] += f, hash[6] += g, hash[7] += h;
}
SZ_API_COMPTIME void sz_sha256_state_init_v128(sz_sha256_state_t *state) { sz_sha256_state_init_serial(state); }
SZ_API_COMPTIME void sz_sha256_state_update_v128(sz_sha256_state_t *state_ptr, sz_cptr_t data, sz_size_t length) {
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
    for (sz_size_t word_index = 0; word_index < 8; ++word_index) hash[word_index] = state_ptr->hash[word_index];
    if (head_length) {
        for (sz_size_t byte_index = 0; byte_index < head_length; ++byte_index)
            state_ptr->block[state_ptr->block_length++] = input[byte_index];
        sz_sha256_process_block_v128_(hash, state_ptr->block);
        state_ptr->block_length = 0;
        input += head_length;
    }
    for (sz_size_t processed = 0; processed < body_length;
         processed += SZ_SHA256_BLOCK_LENGTH, input += SZ_SHA256_BLOCK_LENGTH)
        sz_sha256_process_block_v128_(hash, input);
    for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
        state_ptr->block[byte_index] = input[byte_index];
    state_ptr->block_length = tail_length;
    for (sz_size_t word_index = 0; word_index < 8; ++word_index) state_ptr->hash[word_index] = hash[word_index];
}
SZ_API_COMPTIME void sz_sha256_state_digest_v128(sz_sha256_state_t const *state,
                                                 sz_u8_t digest[sz_at_least_(SZ_SHA256_DIGEST_LENGTH)]) {
    sz_sha256_state_digest_serial(state, digest);
}
#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif 
#ifdef __cplusplus
}
#endif
#endif 
