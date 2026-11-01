/**
 *  @brief WebAssembly relaxed-SIMD backend for hash (level above SIMD128).
 *  @file include/stringzilla/hash/v128relaxed.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_V128RELAXED_H_
#define STRINGZILLA_HASH_V128RELAXED_H_
#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/hash/v128.h" 
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif
SZ_API_COMPTIME sz_u64_t sz_bytesum_v128relaxed(sz_cptr_t text, sz_size_t length) {
    v128_t const ones_u8x16 = wasm_i8x16_splat(1);
    v128_t const bit7_u8x16 = wasm_i8x16_splat((sz_i8_t)0x80);
    sz_u128_vec_t dot32_vec, high8_vec, high64_vec;
    dot32_vec.v128 = wasm_u64x2_splat(0);  
    high64_vec.v128 = wasm_u64x2_splat(0); 
    sz_i64_t signed_sum = 0;
    while (length >= 16) {
        sz_size_t blocks = length / 16;
        if (blocks > 255) blocks = 255;
        length -= blocks * 16;
        high8_vec.v128 = wasm_u64x2_splat(0); 
        for (sz_size_t block_index = 0; block_index < blocks; ++block_index, text += 16) {
            v128_t block_u8x16 = wasm_v128_load(text);
            dot32_vec.v128 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(block_u8x16, ones_u8x16, dot32_vec.v128);
            v128_t high_u8x16 = wasm_i8x16_eq(wasm_v128_and(block_u8x16, bit7_u8x16), bit7_u8x16);
            high8_vec.v128 = wasm_i8x16_sub(high8_vec.v128, high_u8x16);
        }
        v128_t high16_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(high8_vec.v128);
        v128_t high32_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(high16_u16x8);
        high64_vec.v128 = wasm_i64x2_add( 
            high64_vec.v128,
            wasm_i64x2_add(wasm_u64x2_extend_low_u32x4(high32_u32x4), wasm_u64x2_extend_high_u32x4(high32_u32x4)));
    }
    signed_sum += (sz_i64_t)dot32_vec.i32s[0] + dot32_vec.i32s[1] + dot32_vec.i32s[2] + dot32_vec.i32s[3];
    sz_u64_t high_count = high64_vec.u64s[0] + high64_vec.u64s[1];
    sz_u64_t sum = (sz_u64_t)(signed_sum + (sz_i64_t)(high_count * 256));
    while (length--) sum += *(sz_u8_t const *)text++;
    return sum;
}
#pragma region Hash with relaxed SIMD AES
/** @brief `relaxed_swizzle` counterpart of `sz_aes_linear_v128_`. */
SZ_HELPER_INLINE v128_t sz_aes_linear_v128relaxed_(v128_t low_table_u8x16, v128_t high_table_u8x16, v128_t x_u8x16) {
    v128_t low_nibbles_u8x16 = wasm_v128_and(x_u8x16, wasm_i8x16_splat((sz_i8_t)0x0F));
    v128_t high_nibbles_u8x16 = wasm_u8x16_shr(x_u8x16, 4);
    return wasm_v128_xor(wasm_i8x16_relaxed_swizzle(low_table_u8x16, low_nibbles_u8x16),
                         wasm_i8x16_relaxed_swizzle(high_table_u8x16, high_nibbles_u8x16));
}
/** @brief `relaxed_swizzle` counterpart of `sz_aes_gf4_mul_v128_`. */
SZ_HELPER_AUTO v128_t sz_aes_gf4_mul_v128relaxed_(v128_t a_u8x16, v128_t b_u8x16) {
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
    v128_t sum_u8x16 = wasm_i8x16_add(wasm_i8x16_relaxed_swizzle(log_u8x16, a_u8x16),
                                      wasm_i8x16_relaxed_swizzle(log_u8x16, b_u8x16));
    v128_t product_u8x16 = wasm_v128_xor(
        wasm_i8x16_swizzle(exp_lo_u8x16, sum_u8x16),
        wasm_i8x16_swizzle(exp_hi_u8x16, wasm_i8x16_sub(sum_u8x16, wasm_i8x16_splat(16))));
    v128_t any_zero_u8x16 = wasm_v128_or(wasm_i8x16_eq(a_u8x16, zero_u8x16), wasm_i8x16_eq(b_u8x16, zero_u8x16));
    return wasm_v128_andnot(product_u8x16, any_zero_u8x16);
}
/** @brief `relaxed_swizzle` counterpart of `sz_emulate_aesenc_v128_` (bit-exact with the serial round). */
SZ_HELPER_AUTO sz_u128_vec_t sz_emulate_aesenc_v128relaxed_(sz_u128_vec_t state_vec, sz_u128_vec_t round_key_vec) {
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
    v128_t mapped_u8x16 = sz_aes_linear_v128relaxed_(wasm_v128_load(fwd_lo), wasm_v128_load(fwd_hi), state_u8x16);
    v128_t high_nibble_u8x16 = wasm_u8x16_shr(mapped_u8x16, 4);
    v128_t low_nibble_u8x16 = wasm_v128_and(mapped_u8x16, low_nibble_mask_u8x16);
    v128_t high_sqr_n_u8x16 = wasm_i8x16_relaxed_swizzle(wasm_v128_load(gf4_sqr_n), high_nibble_u8x16);
    v128_t high_low_product_u8x16 = sz_aes_gf4_mul_v128relaxed_(high_nibble_u8x16, low_nibble_u8x16);
    v128_t low_sqr_u8x16 = wasm_i8x16_relaxed_swizzle(wasm_v128_load(gf4_sqr), low_nibble_u8x16);
    v128_t d_u8x16 = wasm_v128_xor(wasm_v128_xor(high_sqr_n_u8x16, high_low_product_u8x16), low_sqr_u8x16);
    v128_t d_inv_u8x16 = wasm_i8x16_relaxed_swizzle(wasm_v128_load(gf4_inv), d_u8x16);
    v128_t high_inverse_u8x16 = sz_aes_gf4_mul_v128relaxed_(high_nibble_u8x16, d_inv_u8x16);
    v128_t low_inverse_u8x16 = sz_aes_gf4_mul_v128relaxed_(wasm_v128_xor(high_nibble_u8x16, low_nibble_u8x16),
                                                           d_inv_u8x16);
    v128_t inverted_u8x16 = wasm_v128_or(wasm_i8x16_shl(high_inverse_u8x16, 4), low_inverse_u8x16);
    v128_t back_u8x16 = sz_aes_linear_v128relaxed_(wasm_v128_load(inv_lo), wasm_v128_load(inv_hi), inverted_u8x16);
    v128_t subbed_u8x16 = sz_aes_linear_v128relaxed_(wasm_v128_load(aff_lo), wasm_v128_load(aff_hi), back_u8x16);
    v128_t shifted_u8x16 = wasm_i8x16_relaxed_swizzle(subbed_u8x16, wasm_v128_load(shift_rows));
    v128_t col_table_u8x16 = wasm_v128_load(next_col);
    v128_t col1_u8x16 = wasm_i8x16_relaxed_swizzle(shifted_u8x16, col_table_u8x16);
    v128_t col2_u8x16 = wasm_i8x16_relaxed_swizzle(col1_u8x16, col_table_u8x16);
    v128_t col3_u8x16 = wasm_i8x16_relaxed_swizzle(col2_u8x16, col_table_u8x16);
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
/** @brief `relaxed_swizzle` counterpart of `sz_emulate_shuffle_epi8_v128_` (order indices are in [0,15]). */
SZ_HELPER_INLINE sz_u128_vec_t sz_emulate_shuffle_epi8_v128relaxed_(sz_u128_vec_t state_vec, v128_t order_u8x16) {
    sz_u128_vec_t result_vec;
    result_vec.v128 = wasm_i8x16_relaxed_swizzle(state_vec.v128, order_u8x16);
    return result_vec;
}
SZ_HELPER_AUTO void sz_hash_state_short_update_v128relaxed_(sz_hash_state_aligned_for_short_t *state,
                                                            sz_u128_vec_t block_vec) {
    v128_t shuffle_u8x16 = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    state->aes = sz_emulate_aesenc_v128relaxed_(state->aes, block_vec);
    state->sum = sz_emulate_shuffle_epi8_v128relaxed_(state->sum, shuffle_u8x16);
    state->sum.v128 = wasm_i64x2_add(state->sum.v128, block_vec.v128);
}
SZ_HELPER_AUTO sz_u64_t sz_hash_state_short_finalize_v128relaxed_(sz_hash_state_aligned_for_short_t const *state,
                                                                  sz_size_t length) {
    sz_u128_vec_t key_with_length_vec = state->key;
    key_with_length_vec.u64s[0] += length;
    sz_u128_vec_t mixed_vec = sz_emulate_aesenc_v128relaxed_(state->sum, state->aes);
    sz_u128_vec_t mixed_in_register_vec = sz_emulate_aesenc_v128relaxed_(
        sz_emulate_aesenc_v128relaxed_(mixed_vec, key_with_length_vec), mixed_vec);
    return mixed_in_register_vec.u64s[0];
}
/** @brief Loads the packed public state into the aligned internal twin (4x `wasm_v128_load` per 64-byte field). */
SZ_HELPER_AUTO sz_hash_state_aligned_t sz_hash_state_load_v128relaxed_(sz_hash_state_t const *packed) {
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
SZ_HELPER_AUTO void sz_hash_state_store_v128relaxed_(sz_hash_state_t *packed, sz_hash_state_aligned_t const *state) {
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        sz_size_t const offset = lane_index * 16;
        wasm_v128_store(packed->aes + offset, state->aes.u128s[lane_index].v128);
        wasm_v128_store(packed->sum + offset, state->sum.u128s[lane_index].v128);
        wasm_v128_store(packed->ins + offset, state->ins.u128s[lane_index].v128);
    }
    wasm_v128_store(packed->key, state->key.v128);
    packed->ins_length = state->ins_length;
}
SZ_HELPER_AUTO void sz_hash_state_update_v128relaxed_(sz_hash_state_aligned_t *state) {
    v128_t shuffle_u8x16 = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        sz_u128_vec_t ins_vec = state->ins.u128s[lane_index];
        state->aes.u128s[lane_index] = sz_emulate_aesenc_v128relaxed_(state->aes.u128s[lane_index], ins_vec);
        state->sum.u128s[lane_index] = sz_emulate_shuffle_epi8_v128relaxed_(state->sum.u128s[lane_index],
                                                                            shuffle_u8x16);
        state->sum.u128s[lane_index].v128 = wasm_i64x2_add(state->sum.u128s[lane_index].v128, ins_vec.v128);
    }
}
SZ_HELPER_AUTO sz_u64_t sz_hash_state_finalize_v128relaxed_(sz_hash_state_aligned_t state) {
    v128_t shuffle_u8x16 = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    sz_u128_vec_t key_with_length_vec;
    key_with_length_vec.u64s[0] = state.key.u64s[0] + state.ins_length;
    key_with_length_vec.u64s[1] = state.key.u64s[1];
    sz_u128_vec_t ins0_vec = state.ins.u128s[0], ins1_vec = state.ins.u128s[1];
    sz_u128_vec_t ins2_vec = state.ins.u128s[2], ins3_vec = state.ins.u128s[3];
    sz_u128_vec_t aes0_vec = sz_emulate_aesenc_v128relaxed_(state.aes.u128s[0], ins0_vec);
    sz_u128_vec_t aes1_vec = sz_emulate_aesenc_v128relaxed_(state.aes.u128s[1], ins1_vec);
    sz_u128_vec_t aes2_vec = sz_emulate_aesenc_v128relaxed_(state.aes.u128s[2], ins2_vec);
    sz_u128_vec_t aes3_vec = sz_emulate_aesenc_v128relaxed_(state.aes.u128s[3], ins3_vec);
    sz_u128_vec_t sum0_vec = sz_emulate_shuffle_epi8_v128relaxed_(state.sum.u128s[0], shuffle_u8x16);
    sz_u128_vec_t sum1_vec = sz_emulate_shuffle_epi8_v128relaxed_(state.sum.u128s[1], shuffle_u8x16);
    sz_u128_vec_t sum2_vec = sz_emulate_shuffle_epi8_v128relaxed_(state.sum.u128s[2], shuffle_u8x16);
    sz_u128_vec_t sum3_vec = sz_emulate_shuffle_epi8_v128relaxed_(state.sum.u128s[3], shuffle_u8x16);
    sum0_vec.v128 = wasm_i64x2_add(sum0_vec.v128, ins0_vec.v128);
    sum1_vec.v128 = wasm_i64x2_add(sum1_vec.v128, ins1_vec.v128);
    sum2_vec.v128 = wasm_i64x2_add(sum2_vec.v128, ins2_vec.v128);
    sum3_vec.v128 = wasm_i64x2_add(sum3_vec.v128, ins3_vec.v128);
    sz_u128_vec_t mixed0_vec = sz_emulate_aesenc_v128relaxed_(sum0_vec, aes0_vec);
    sz_u128_vec_t mixed1_vec = sz_emulate_aesenc_v128relaxed_(sum1_vec, aes1_vec);
    sz_u128_vec_t mixed2_vec = sz_emulate_aesenc_v128relaxed_(sum2_vec, aes2_vec);
    sz_u128_vec_t mixed3_vec = sz_emulate_aesenc_v128relaxed_(sum3_vec, aes3_vec);
    sz_u128_vec_t mixed01_vec = sz_emulate_aesenc_v128relaxed_(mixed0_vec, mixed1_vec);
    sz_u128_vec_t mixed23_vec = sz_emulate_aesenc_v128relaxed_(mixed2_vec, mixed3_vec);
    sz_u128_vec_t mixed_vec = sz_emulate_aesenc_v128relaxed_(mixed01_vec, mixed23_vec);
    sz_u128_vec_t mixed_in_register_vec = sz_emulate_aesenc_v128relaxed_(
        sz_emulate_aesenc_v128relaxed_(mixed_vec, key_with_length_vec), mixed_vec);
    return mixed_in_register_vec.u64s[0];
}
SZ_API_COMPTIME SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_v128relaxed(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_serial_(&state, seed);
        sz_u128_vec_t data_vec;
        data_vec.v128 = sz_load_partial_v128_(start, length);
        sz_hash_state_short_update_v128relaxed_(&state, data_vec);
        return sz_hash_state_short_finalize_v128relaxed_(&state, length);
    }
    else if (length <= 32) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));
        sz_hash_state_short_update_v128relaxed_(&state, data0_vec);
        sz_hash_state_short_update_v128relaxed_(&state, data1_vec);
        return sz_hash_state_short_finalize_v128relaxed_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + 16);
        data2_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));
        sz_hash_state_short_update_v128relaxed_(&state, data0_vec);
        sz_hash_state_short_update_v128relaxed_(&state, data1_vec);
        sz_hash_state_short_update_v128relaxed_(&state, data2_vec);
        return sz_hash_state_short_finalize_v128relaxed_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + 16);
        data2_vec.v128 = wasm_v128_load(start + 32);
        data3_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));
        sz_hash_state_short_update_v128relaxed_(&state, data0_vec);
        sz_hash_state_short_update_v128relaxed_(&state, data1_vec);
        sz_hash_state_short_update_v128relaxed_(&state, data2_vec);
        sz_hash_state_short_update_v128relaxed_(&state, data3_vec);
        return sz_hash_state_short_finalize_v128relaxed_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_aligned_t state;
        sz_hash_state_init_serial((sz_hash_state_t *)&state, seed);
        for (; state.ins_length + 64 < length; state.ins_length += 64) {
            for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index)
                state.ins.u128s[lane_index].v128 = wasm_v128_load(start + state.ins_length + lane_index * 16);
            sz_hash_state_update_v128relaxed_(&state);
        }
        v128_t const zero_u64x2 = wasm_u64x2_splat(0);
        for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) state.ins.u128s[lane_index].v128 = zero_u64x2;
        sz_size_t const tail_length = length - state.ins_length;
        for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
            state.ins.u8s[byte_index] = start[state.ins_length + byte_index];
        state.ins_length = length;
        return sz_hash_state_finalize_v128relaxed_(state);
    }
}
SZ_API_COMPTIME void sz_hash_state_init_v128relaxed(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_serial(state, seed);
}
SZ_API_COMPTIME void sz_hash_state_update_v128relaxed(sz_hash_state_t *packed, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_aligned_t state = sz_hash_state_load_v128relaxed_(packed);
    v128_t const zero_u64x2 = wasm_u64x2_splat(0);
    sz_size_t buffered = state.ins_length % 64;
    if (buffered == 0 && state.ins_length) buffered = 64;
    while (length) {
        if (buffered == 64) { 
            sz_hash_state_update_v128relaxed_(&state);
            for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) state.ins.u128s[lane_index].v128 = zero_u64x2;
            buffered = 0;
        }
        sz_size_t const take = sz_min_of_two(length, (sz_size_t)64 - buffered);
        for (sz_size_t byte_index = 0; byte_index < take; ++byte_index)
            state.ins.u8s[buffered + byte_index] = text[byte_index];
        buffered += take, text += take, length -= take, state.ins_length += take;
    }
    sz_hash_state_store_v128relaxed_(packed, &state);
}
SZ_API_COMPTIME sz_u64_t sz_hash_state_digest_v128relaxed(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state = sz_hash_state_load_v128relaxed_(packed);
    sz_size_t length = state.ins_length;
    if (length > 64) return sz_hash_state_finalize_v128relaxed_(state);
    sz_hash_state_aligned_for_short_t minimal_state;
    minimal_state.key = state.key;
    minimal_state.aes = state.aes.u128s[0];
    minimal_state.sum = state.sum.u128s[0];
    if (length <= 16) {
        sz_hash_state_short_update_v128relaxed_(&minimal_state, state.ins.u128s[0]);
        return sz_hash_state_short_finalize_v128relaxed_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_state_short_update_v128relaxed_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_v128relaxed_(&minimal_state, state.ins.u128s[1]);
        return sz_hash_state_short_finalize_v128relaxed_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_state_short_update_v128relaxed_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_v128relaxed_(&minimal_state, state.ins.u128s[1]);
        sz_hash_state_short_update_v128relaxed_(&minimal_state, state.ins.u128s[2]);
        return sz_hash_state_short_finalize_v128relaxed_(&minimal_state, length);
    }
    else {
        sz_hash_state_short_update_v128relaxed_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_v128relaxed_(&minimal_state, state.ins.u128s[1]);
        sz_hash_state_short_update_v128relaxed_(&minimal_state, state.ins.u128s[2]);
        sz_hash_state_short_update_v128relaxed_(&minimal_state, state.ins.u128s[3]);
        return sz_hash_state_short_finalize_v128relaxed_(&minimal_state, length);
    }
}
SZ_API_COMPTIME void sz_fill_random_v128relaxed(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_constants = sz_hash_pi_constants_();
    sz_u128_vec_t input_vec, pi_vec, key_vec, generated_vec;
    for (sz_size_t lane_index = 0; length; ++lane_index) {
        input_vec.v128 = wasm_u64x2_splat(nonce + lane_index);
        pi_vec = ((sz_u128_vec_t const *)pi_constants)[lane_index % 4];
        key_vec.v128 = wasm_v128_xor(pi_vec.v128, wasm_u64x2_splat(nonce));
        generated_vec = sz_emulate_aesenc_v128relaxed_(input_vec, key_vec);
        if (length >= 16) { wasm_v128_store(text, generated_vec.v128), text += 16, length -= 16; }
        else { sz_store_partial_v128_(text, generated_vec.v128, length), length = 0; }
    }
}
/**
 *  @brief Replays prepared text-lanes through the relaxed-SIMD minimal AES state for a single seed.
 *  @return 64-bit hash, bit-identical to `sz_hash_v128relaxed(text, length, seed)` (hence to serial).
 */
SZ_HELPER_AUTO sz_u64_t sz_hash_multiseed_replay_v128relaxed_(sz_u512_vec_t const *text_lanes_vec,
                                                              sz_size_t text_lanes_count, sz_size_t length,
                                                              sz_u64_t seed) {
    sz_align_(16) sz_hash_state_aligned_for_short_t state;
    sz_hash_state_short_init_serial_(&state, seed);
    for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
        sz_hash_state_short_update_v128relaxed_(&state, text_lanes_vec->u128s[lane_index]);
    return sz_hash_state_short_finalize_v128relaxed_(&state, length);
}
SZ_API_COMPTIME void sz_hash_multiseed_v128relaxed(sz_cptr_t text, sz_size_t length,             
                                                   sz_u64_t const *seeds, sz_size_t seeds_count, 
                                                   sz_u64_t *hashes) {
    if (seeds_count == 0) return;
    if (seeds_count == 1) {
        hashes[0] = sz_hash_v128relaxed(text, length, seeds[0]);
        return;
    }
    if (length <= 64) {
        sz_u512_vec_t text_lanes_vec;
        sz_size_t const text_lanes_count = sz_hash_multiseed_prepare_serial_(text, length, &text_lanes_vec);
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_multiseed_replay_v128relaxed_(&text_lanes_vec, text_lanes_count, length,
                                                                       seeds[seed_index]);
    }
    else {
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_v128relaxed(text, length, seeds[seed_index]);
    }
}
#pragma endregion // Hash with relaxed SIMD AES
#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif 
#ifdef __cplusplus
}
#endif
#endif 
