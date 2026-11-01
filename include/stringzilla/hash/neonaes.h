/**
 *  @brief NEON + AES backend for string hashing and checksums.
 *  @file include/stringzilla/hash/neonaes.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_NEONAES_H_
#define STRINGZILLA_HASH_NEONAES_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#include "stringzilla/hash/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_NEONAES
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd+crypto+aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd+crypto+aes")
#endif
/**
 *  @brief Emulates the Intel's AES-NI `AESENC` instruction on Arm NEON.
 *  @see "Emulating x86 AES Intrinsics on ARMv8-A" by Michael Brase:
 *       https://blog.michaelbrase.com/2018/05/08/emulating-x86-aes-intrinsics-on-armv8-a/
 */
SZ_HELPER_AUTO uint8x16_t sz_emulate_aesenc_u8x16_neon_(uint8x16_t state_u8x16, uint8x16_t round_key_u8x16) {
    return veorq_u8(vaesmcq_u8(vaeseq_u8(state_u8x16, vdupq_n_u8(0))), round_key_u8x16);
}
/**
 *  @brief Emulates the Intel's AES-NI `AESENC` instruction on Arm NEON, operating on u64x2 registers.
 *
 *  @param state_u64x2 128-bit AES state represented as two 64-bit lanes.
 *  @param round_key_u64x2 128-bit round key represented as two 64-bit lanes.
 *  @return AES-encrypted 128-bit result as two 64-bit lanes.
 */
SZ_HELPER_INLINE uint64x2_t sz_emulate_aesenc_u64x2_neon_(uint64x2_t state_u64x2, uint64x2_t round_key_u64x2) {
    return vreinterpretq_u64_u8(               
        sz_emulate_aesenc_u8x16_neon_(         
            vreinterpretq_u8_u64(state_u64x2), 
            vreinterpretq_u8_u64(round_key_u64x2)));
}
/**
 *  @brief Initializes the minimal hash state using NEON with the given seed.
 *
 *  @param state Pointer to the minimal hash state to initialize.
 *  @param seed 64-bit seed value for the hash.
 */
SZ_HELPER_AUTO void sz_hash_state_short_init_neon_(sz_hash_state_aligned_for_short_t *state, sz_u64_t seed) {
    uint64x2_t seed_u64x2 = vdupq_n_u64(seed);
    state->key.u64x2 = seed_u64x2;
    sz_u64_t const *pi = sz_hash_pi_constants_();
    uint64x2_t const pi0_u64x2 = vld1q_u64(pi);
    uint64x2_t const pi1_u64x2 = vld1q_u64(pi + 8);
    uint64x2_t aes_state_key_u64x2 = veorq_u64(seed_u64x2, pi0_u64x2);
    uint64x2_t sum_state_key_u64x2 = veorq_u64(seed_u64x2, pi1_u64x2);
    state->aes.u64x2 = aes_state_key_u64x2;
    state->sum.u64x2 = sum_state_key_u64x2;
}
/**
 *  @brief Finalizes the minimal hash state and returns a 64-bit digest.
 *
 *  @param state Pointer to the minimal hash state to finalize.
 *  @param length Total number of bytes that were hashed.
 *  @return 64-bit hash digest.
 */
SZ_HELPER_AUTO sz_u64_t sz_hash_state_short_finalize_neon_(sz_hash_state_aligned_for_short_t const *state,
                                                           sz_size_t length) {
    uint64x2_t key_with_length_u64x2 = vaddq_u64(state->key.u64x2, vsetq_lane_u64(length, vdupq_n_u64(0), 0));
    uint8x16_t mixed_u8x16 = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16, state->aes.u8x16);
    uint8x16_t final_mixed_u8x16 = sz_emulate_aesenc_u8x16_neon_(
        sz_emulate_aesenc_u8x16_neon_(mixed_u8x16, vreinterpretq_u8_u64(key_with_length_u64x2)), mixed_u8x16);
    return vgetq_lane_u64(vreinterpretq_u64_u8(final_mixed_u8x16), 0);
}
/**
 *  @brief Feeds one 16-byte block into the minimal hash state.
 *
 *  @param state Pointer to the minimal hash state to update.
 *  @param block_u8x16 16-byte input block as a NEON register.
 */
SZ_HELPER_AUTO void sz_hash_state_short_update_neon_(sz_hash_state_aligned_for_short_t *state, uint8x16_t block_u8x16) {
    uint8x16_t const order_u8x16 = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    state->aes.u8x16 = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16, block_u8x16);
    uint8x16_t sum_shuffled_u8x16 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2), order_u8x16);
    state->sum.u64x2 = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled_u8x16), vreinterpretq_u64_u8(block_u8x16));
}
SZ_API_COMPTIME void sz_hash_state_init_neonaes(sz_hash_state_t *state, sz_u64_t seed) {
    uint64x2_t seed_u64x2 = vdupq_n_u64(seed);
    vst1q_u64((sz_u64_t *)state->key, seed_u64x2);
    sz_u64_t const *pi = sz_hash_pi_constants_();
    for (int lane_index = 0; lane_index < 4; ++lane_index)
        vst1q_u8(state->aes + lane_index * 16,
                 vreinterpretq_u8_u64(veorq_u64(seed_u64x2, vld1q_u64(pi + lane_index * 2))));
    for (int lane_index = 0; lane_index < 4; ++lane_index)
        vst1q_u8(state->sum + lane_index * 16,
                 vreinterpretq_u8_u64(veorq_u64(seed_u64x2, vld1q_u64(pi + lane_index * 2 + 8))));
    uint8x16_t zeros_u8x16 = vdupq_n_u8(0);
    for (int lane_index = 0; lane_index < 4; ++lane_index) vst1q_u8(state->ins + lane_index * 16, zeros_u8x16);
    state->ins_length = 0;
}
/** @brief Loads the packed public state into the aligned internal twin (NEON: 4x `vld1q_u8` per 64-byte field). */
SZ_HELPER_AUTO sz_hash_state_aligned_t sz_hash_state_load_neonaes_(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state;
    for (int lane_index = 0; lane_index < 4; ++lane_index) {
        state.aes.u8x16s[lane_index] = vld1q_u8(packed->aes + lane_index * 16);
        state.sum.u8x16s[lane_index] = vld1q_u8(packed->sum + lane_index * 16);
        state.ins.u8x16s[lane_index] = vld1q_u8(packed->ins + lane_index * 16);
    }
    state.key.u8x16 = vld1q_u8(packed->key);
    state.ins_length = packed->ins_length;
    return state;
}
/** @brief Stores the aligned internal twin back into the packed public state (NEON: 4x `vst1q_u8` per field). */
SZ_HELPER_AUTO void sz_hash_state_store_neonaes_(sz_hash_state_t *packed, sz_hash_state_aligned_t const *state) {
    for (int lane_index = 0; lane_index < 4; ++lane_index) {
        vst1q_u8(packed->aes + lane_index * 16, state->aes.u8x16s[lane_index]);
        vst1q_u8(packed->sum + lane_index * 16, state->sum.u8x16s[lane_index]);
        vst1q_u8(packed->ins + lane_index * 16, state->ins.u8x16s[lane_index]);
    }
    vst1q_u8(packed->key, state->key.u8x16);
    packed->ins_length = state->ins_length;
}
/**
 *  @brief Absorbs the buffered 64-byte block into the aligned state (four 128-bit lanes), in place.
 *  @param state Pointer to the aligned hash state whose `ins` lanes are consumed.
 */
SZ_HELPER_AUTO void sz_hash_state_update_neonaes_(sz_hash_state_aligned_t *state) {
    uint8x16_t const order_u8x16 = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    state->aes.u8x16s[0] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[0], state->ins.u8x16s[0]);
    uint8x16_t sum_shuffled_0_u8x16 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[0]), order_u8x16);
    state->sum.u64x2s[0] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled_0_u8x16), state->ins.u64x2s[0]);
    state->aes.u8x16s[1] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[1], state->ins.u8x16s[1]);
    uint8x16_t sum_shuffled_1_u8x16 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[1]), order_u8x16);
    state->sum.u64x2s[1] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled_1_u8x16), state->ins.u64x2s[1]);
    state->aes.u8x16s[2] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[2], state->ins.u8x16s[2]);
    uint8x16_t sum_shuffled_2_u8x16 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[2]), order_u8x16);
    state->sum.u64x2s[2] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled_2_u8x16), state->ins.u64x2s[2]);
    state->aes.u8x16s[3] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[3], state->ins.u8x16s[3]);
    uint8x16_t sum_shuffled_3_u8x16 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[3]), order_u8x16);
    state->sum.u64x2s[3] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled_3_u8x16), state->ins.u64x2s[3]);
}
/**
 *  @brief Finalizes the full internal hash state and returns a 64-bit digest.
 *
 *  @param state The internal hash state to finalize.
 *  @return 64-bit hash digest.
 */
SZ_HELPER_AUTO sz_u64_t sz_hash_state_finalize_neonaes_(sz_hash_state_aligned_t state) {
    uint64x2_t key_with_length_u64x2 = vaddq_u64(state.key.u64x2, vsetq_lane_u64(state.ins_length, vdupq_n_u64(0), 0));
    uint8x16_t const order_u8x16 = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    uint8x16_t aes_0_u8x16 = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[0], state.ins.u8x16s[0]);
    uint8x16_t aes_1_u8x16 = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[1], state.ins.u8x16s[1]);
    uint8x16_t aes_2_u8x16 = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[2], state.ins.u8x16s[2]);
    uint8x16_t aes_3_u8x16 = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[3], state.ins.u8x16s[3]);
    uint64x2_t sum_0_u64x2 = vaddq_u64(
        vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[0]), order_u8x16)), state.ins.u64x2s[0]);
    uint64x2_t sum_1_u64x2 = vaddq_u64(
        vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[1]), order_u8x16)), state.ins.u64x2s[1]);
    uint64x2_t sum_2_u64x2 = vaddq_u64(
        vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[2]), order_u8x16)), state.ins.u64x2s[2]);
    uint64x2_t sum_3_u64x2 = vaddq_u64(
        vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[3]), order_u8x16)), state.ins.u64x2s[3]);
    uint8x16_t mixed_0_u8x16 = sz_emulate_aesenc_u8x16_neon_(vreinterpretq_u8_u64(sum_0_u64x2), aes_0_u8x16);
    uint8x16_t mixed_1_u8x16 = sz_emulate_aesenc_u8x16_neon_(vreinterpretq_u8_u64(sum_1_u64x2), aes_1_u8x16);
    uint8x16_t mixed_2_u8x16 = sz_emulate_aesenc_u8x16_neon_(vreinterpretq_u8_u64(sum_2_u64x2), aes_2_u8x16);
    uint8x16_t mixed_3_u8x16 = sz_emulate_aesenc_u8x16_neon_(vreinterpretq_u8_u64(sum_3_u64x2), aes_3_u8x16);
    uint8x16_t mixed_01_u8x16 = sz_emulate_aesenc_u8x16_neon_(mixed_0_u8x16, mixed_1_u8x16);
    uint8x16_t mixed_23_u8x16 = sz_emulate_aesenc_u8x16_neon_(mixed_2_u8x16, mixed_3_u8x16);
    uint8x16_t mixed_u8x16 = sz_emulate_aesenc_u8x16_neon_(mixed_01_u8x16, mixed_23_u8x16);
    uint8x16_t final_mixed_u8x16 = sz_emulate_aesenc_u8x16_neon_(
        sz_emulate_aesenc_u8x16_neon_(mixed_u8x16, vreinterpretq_u8_u64(key_with_length_u64x2)), mixed_u8x16);
    return vgetq_lane_u64(vreinterpretq_u64_u8(final_mixed_u8x16), 0);
}
SZ_API_COMPTIME void sz_hash_state_update_neonaes(sz_hash_state_t *packed, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_aligned_t state = sz_hash_state_load_neonaes_(packed);
    uint8x16_t const zeros_u8x16 = vdupq_n_u8(0);
    while (length) {
        sz_size_t progress_in_block = state.ins_length % 64;
        if (progress_in_block == 0 && state.ins_length != 0) {
            sz_hash_state_update_neonaes_(&state);
            for (int lane_index = 0; lane_index < 4; ++lane_index) state.ins.u8x16s[lane_index] = zeros_u8x16;
        }
        sz_size_t to_copy = sz_min_of_two(length, 64 - progress_in_block);
        state.ins_length += to_copy;
        length -= to_copy;
        while (to_copy--) state.ins.u8s[progress_in_block++] = (sz_u8_t)*text++;
    }
    sz_hash_state_store_neonaes_(packed, &state);
}
SZ_API_COMPTIME sz_u64_t sz_hash_state_digest_neonaes(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state = sz_hash_state_load_neonaes_(packed);
    sz_size_t length = state.ins_length;
    if (length > 64) return sz_hash_state_finalize_neonaes_(state);
    sz_hash_state_aligned_for_short_t minimal_state;
    minimal_state.key = state.key;
    minimal_state.aes = state.aes.u128s[0];
    minimal_state.sum = state.sum.u128s[0];
    if (length <= 16) {
        sz_hash_state_short_update_neon_(&minimal_state, state.ins.u128s[0].u8x16);
        return sz_hash_state_short_finalize_neon_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_state_short_update_neon_(&minimal_state, state.ins.u128s[0].u8x16);
        sz_hash_state_short_update_neon_(&minimal_state, state.ins.u128s[1].u8x16);
        return sz_hash_state_short_finalize_neon_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_state_short_update_neon_(&minimal_state, state.ins.u128s[0].u8x16);
        sz_hash_state_short_update_neon_(&minimal_state, state.ins.u128s[1].u8x16);
        sz_hash_state_short_update_neon_(&minimal_state, state.ins.u128s[2].u8x16);
        return sz_hash_state_short_finalize_neon_(&minimal_state, length);
    }
    else {
        sz_hash_state_short_update_neon_(&minimal_state, state.ins.u128s[0].u8x16);
        sz_hash_state_short_update_neon_(&minimal_state, state.ins.u128s[1].u8x16);
        sz_hash_state_short_update_neon_(&minimal_state, state.ins.u128s[2].u8x16);
        sz_hash_state_short_update_neon_(&minimal_state, state.ins.u128s[3].u8x16);
        return sz_hash_state_short_finalize_neon_(&minimal_state, length);
    }
}
SZ_API_COMPTIME SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_neonaes(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_neon_(&state, seed);
        sz_u128_vec_t data_vec;
        if (length == 16) { data_vec.u8x16 = vld1q_u8((sz_u8_t const *)text); } 
        else {
            data_vec.u8x16 = vdupq_n_u8(0);
            for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
                data_vec.u8s[byte_index] = text[byte_index]; 
        }
        sz_hash_state_short_update_neon_(&state, data_vec.u8x16);
        return sz_hash_state_short_finalize_neon_(&state, length);
    }
    else if (length <= 32) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_neon_(&state, seed);
        sz_u128_vec_t data_0_vec, data_1_vec;
        data_0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 0));
        data_1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + length - 16));
        sz_hash_shift_in_register_serial_(&data_1_vec, (int)(32 - length)); 
        sz_hash_state_short_update_neon_(&state, data_0_vec.u8x16);
        sz_hash_state_short_update_neon_(&state, data_1_vec.u8x16);
        return sz_hash_state_short_finalize_neon_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_neon_(&state, seed);
        sz_u128_vec_t data_0_vec, data_1_vec, data_2_vec;
        data_0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 0));
        data_1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 16));
        data_2_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + length - 16));
        sz_hash_shift_in_register_serial_(&data_2_vec, (int)(48 - length)); 
        sz_hash_state_short_update_neon_(&state, data_0_vec.u8x16);
        sz_hash_state_short_update_neon_(&state, data_1_vec.u8x16);
        sz_hash_state_short_update_neon_(&state, data_2_vec.u8x16);
        return sz_hash_state_short_finalize_neon_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_neon_(&state, seed);
        sz_u128_vec_t data_0_vec, data_1_vec, data_2_vec, data_3_vec;
        data_0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 0));
        data_1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 16));
        data_2_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 32));
        data_3_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + length - 16));
        sz_hash_shift_in_register_serial_(&data_3_vec, (int)(64 - length)); 
        sz_hash_state_short_update_neon_(&state, data_0_vec.u8x16);
        sz_hash_state_short_update_neon_(&state, data_1_vec.u8x16);
        sz_hash_state_short_update_neon_(&state, data_2_vec.u8x16);
        sz_hash_state_short_update_neon_(&state, data_3_vec.u8x16);
        return sz_hash_state_short_finalize_neon_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_aligned_t state;
        sz_hash_state_init_neonaes((sz_hash_state_t *)&state, seed);
        for (; state.ins_length + 64 < length; state.ins_length += 64) {
            state.ins.u8x16s[0] = vld1q_u8((sz_u8_t const *)(text + state.ins_length + 0));
            state.ins.u8x16s[1] = vld1q_u8((sz_u8_t const *)(text + state.ins_length + 16));
            state.ins.u8x16s[2] = vld1q_u8((sz_u8_t const *)(text + state.ins_length + 32));
            state.ins.u8x16s[3] = vld1q_u8((sz_u8_t const *)(text + state.ins_length + 48));
            sz_hash_state_update_neonaes_(&state);
        }
        uint8x16_t const zeros_u8x16 = vdupq_n_u8(0);
        state.ins.u8x16s[0] = zeros_u8x16;
        state.ins.u8x16s[1] = zeros_u8x16;
        state.ins.u8x16s[2] = zeros_u8x16;
        state.ins.u8x16s[3] = zeros_u8x16;
        for (sz_size_t byte_index = 0; state.ins_length < length; ++byte_index, ++state.ins_length)
            state.ins.u8s[byte_index] = (sz_u8_t)text[state.ins_length];
        return sz_hash_state_finalize_neonaes_(state);
    }
}
/**
 *  @brief Splits a short (<= 64B) input into up to four 128-bit text-lanes using NEON loads.
 *         Mirrors the loading ladder of `sz_hash_neonaes`: full `vld1q_u8` loads for complete lanes
 *         and one overlapping load + in-register shift for the partial tail; byte-by-byte only for
 *         the `< 16` case, where a 16-byte NEON load could read past the input.
 *  @return The number of populated text-lanes (1..4).
 */
SZ_HELPER_AUTO sz_size_t sz_hash_multiseed_prepare_neon_(sz_cptr_t text, sz_size_t length,
                                                         sz_u512_vec_t *text_lanes_vec) {
    if (length <= 16) {
        sz_u128_vec_t lane_vec;
        if (length == 16) { lane_vec.u8x16 = vld1q_u8((sz_u8_t const *)text); }
        else {
            lane_vec.u8x16 = vdupq_n_u8(0);
            for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
                lane_vec.u8s[byte_index] = text[byte_index];
        }
        text_lanes_vec->u128s[0] = lane_vec;
        return 1;
    }
    sz_size_t const text_lanes_count = sz_size_divide_round_up(length, 16);
    for (sz_size_t lane_index = 0; lane_index + 1 < text_lanes_count; ++lane_index)
        text_lanes_vec->u128s[lane_index].u8x16 = vld1q_u8((sz_u8_t const *)(text + lane_index * 16));
    sz_align_(16) static sz_u8_t const lane_iota[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8x16_t const tail_u8x16 = vld1q_u8((sz_u8_t const *)(text + length - 16));
    uint8x16_t const shift_u8x16 = vdupq_n_u8((sz_u8_t)(text_lanes_count * 16 - length));
    text_lanes_vec->u128s[text_lanes_count - 1].u8x16 = vqtbl1q_u8(tail_u8x16,
                                                                   vaddq_u8(vld1q_u8(lane_iota), shift_u8x16));
    return text_lanes_count;
}
SZ_API_COMPTIME void sz_hash_multiseed_neonaes(sz_cptr_t text, sz_size_t length,             
                                               sz_u64_t const *seeds, sz_size_t seeds_count, 
                                               sz_u64_t *hashes) {
    if (seeds_count == 0) return;
    if (seeds_count == 1) {
        hashes[0] = sz_hash_neonaes(text, length, seeds[0]);
        return;
    }
    if (length > 64) {
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_neonaes(text, length, seeds[seed_index]);
        return;
    }
    sz_u512_vec_t text_lanes_vec;
    sz_size_t const text_lanes_count = sz_hash_multiseed_prepare_neon_(text, length, &text_lanes_vec);
    sz_size_t seed_index = 0;
    for (; seed_index + 2 <= seeds_count; seed_index += 2) {
        sz_hash_state_aligned_for_short_t state0, state1;
        sz_hash_state_short_init_neon_(&state0, seeds[seed_index + 0]);
        sz_hash_state_short_init_neon_(&state1, seeds[seed_index + 1]);
        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index) {
            sz_hash_state_short_update_neon_(&state0, text_lanes_vec.u128s[lane_index].u8x16);
            sz_hash_state_short_update_neon_(&state1, text_lanes_vec.u128s[lane_index].u8x16);
        }
        hashes[seed_index + 0] = sz_hash_state_short_finalize_neon_(&state0, length);
        hashes[seed_index + 1] = sz_hash_state_short_finalize_neon_(&state1, length);
    }
    if (seed_index < seeds_count) {
        sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_neon_(&state, seeds[seed_index]);
        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
            sz_hash_state_short_update_neon_(&state, text_lanes_vec.u128s[lane_index].u8x16);
        hashes[seed_index] = sz_hash_state_short_finalize_neon_(&state, length);
    }
}
SZ_API_COMPTIME void sz_fill_random_neonaes(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_pointer = sz_hash_pi_constants_();
    if (length <= 16) {
        uint64x2_t input_u64x2 = vdupq_n_u64(nonce);
        uint64x2_t pi_u64x2 = vld1q_u64(pi_pointer);
        uint64x2_t key_u64x2 = veorq_u64(vdupq_n_u64(nonce), pi_u64x2);
        uint64x2_t generated_u64x2 = sz_emulate_aesenc_u64x2_neon_(input_u64x2, key_u64x2);
        if (length >= 16) {
            vst1q_u8((sz_u8_t *)text, vreinterpretq_u8_u64(generated_u64x2)); 
        }
        else
            for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
                text[byte_index] = ((sz_u8_t *)&generated_u64x2)[byte_index]; 
    }
    else if (length <= 32) {
        uint64x2_t inputs_u64x2[2], pis_u64x2[2], keys_u64x2[2], generated_u64x2[2];
        inputs_u64x2[0] = vdupq_n_u64(nonce + 0);
        inputs_u64x2[1] = vdupq_n_u64(nonce + 1);
        pis_u64x2[0] = vld1q_u64(pi_pointer + 0);
        pis_u64x2[1] = vld1q_u64(pi_pointer + 2);
        keys_u64x2[0] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[0]);
        keys_u64x2[1] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[1]);
        generated_u64x2[0] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[0], keys_u64x2[0]);
        generated_u64x2[1] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[1], keys_u64x2[1]);
        vst1q_u64((sz_u64_t *)(text), generated_u64x2[0]);
        if (length >= 32) {
            vst1q_u8((sz_u8_t *)(text + 16), vreinterpretq_u8_u64(generated_u64x2[1])); 
        }
        else
            for (sz_size_t byte_index = 16; byte_index < length; ++byte_index)
                text[byte_index] = ((sz_u8_t *)&generated_u64x2[1])[byte_index - 16]; 
    }
    else if (length <= 48) {
        uint64x2_t inputs_u64x2[3], pis_u64x2[3], keys_u64x2[3], generated_u64x2[3];
        inputs_u64x2[0] = vdupq_n_u64(nonce);
        inputs_u64x2[1] = vdupq_n_u64(nonce + 1);
        inputs_u64x2[2] = vdupq_n_u64(nonce + 2);
        pis_u64x2[0] = vld1q_u64(pi_pointer + 0);
        pis_u64x2[1] = vld1q_u64(pi_pointer + 2);
        pis_u64x2[2] = vld1q_u64(pi_pointer + 4);
        keys_u64x2[0] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[0]);
        keys_u64x2[1] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[1]);
        keys_u64x2[2] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[2]);
        generated_u64x2[0] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[0], keys_u64x2[0]);
        generated_u64x2[1] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[1], keys_u64x2[1]);
        generated_u64x2[2] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[2], keys_u64x2[2]);
        vst1q_u64((sz_u64_t *)(text + 0), generated_u64x2[0]);
        vst1q_u64((sz_u64_t *)(text + 16), generated_u64x2[1]);
        if (length >= 48) {
            vst1q_u8((sz_u8_t *)(text + 32), vreinterpretq_u8_u64(generated_u64x2[2])); 
        }
        else
            for (sz_size_t byte_index = 32; byte_index < length; ++byte_index)
                text[byte_index] = ((sz_u8_t *)generated_u64x2)[byte_index]; 
    }
    else {
        uint64x2_t inputs_u64x2[4], pis_u64x2[4], keys_u64x2[4], generated_u64x2[4];
        inputs_u64x2[0] = vdupq_n_u64(nonce + 0);
        inputs_u64x2[1] = vdupq_n_u64(nonce + 1);
        inputs_u64x2[2] = vdupq_n_u64(nonce + 2);
        inputs_u64x2[3] = vdupq_n_u64(nonce + 3);
        pis_u64x2[0] = vld1q_u64(pi_pointer + 0);
        pis_u64x2[1] = vld1q_u64(pi_pointer + 2);
        pis_u64x2[2] = vld1q_u64(pi_pointer + 4);
        pis_u64x2[3] = vld1q_u64(pi_pointer + 6);
        keys_u64x2[0] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[0]);
        keys_u64x2[1] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[1]);
        keys_u64x2[2] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[2]);
        keys_u64x2[3] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[3]);
        sz_size_t byte_index = 0;
        uint64x2_t const increment_u64x2 = vdupq_n_u64(4);
        for (; byte_index + 64 <= length; byte_index += 64) {
            generated_u64x2[0] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[0], keys_u64x2[0]);
            generated_u64x2[1] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[1], keys_u64x2[1]);
            generated_u64x2[2] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[2], keys_u64x2[2]);
            generated_u64x2[3] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[3], keys_u64x2[3]);
            vst1q_u64((sz_u64_t *)(text + byte_index + 0), generated_u64x2[0]);
            vst1q_u64((sz_u64_t *)(text + byte_index + 16), generated_u64x2[1]);
            vst1q_u64((sz_u64_t *)(text + byte_index + 32), generated_u64x2[2]);
            vst1q_u64((sz_u64_t *)(text + byte_index + 48), generated_u64x2[3]);
            inputs_u64x2[0] = vaddq_u64(inputs_u64x2[0], increment_u64x2);
            inputs_u64x2[1] = vaddq_u64(inputs_u64x2[1], increment_u64x2);
            inputs_u64x2[2] = vaddq_u64(inputs_u64x2[2], increment_u64x2);
            inputs_u64x2[3] = vaddq_u64(inputs_u64x2[3], increment_u64x2);
        }
        {
            generated_u64x2[0] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[0], keys_u64x2[0]);
            generated_u64x2[1] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[1], keys_u64x2[1]);
            generated_u64x2[2] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[2], keys_u64x2[2]);
            generated_u64x2[3] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[3], keys_u64x2[3]);
            sz_size_t generated_index = 0;
            for (; byte_index + 16 <= length; byte_index += 16, generated_index += 16)
                vst1q_u8((sz_u8_t *)(text + byte_index),
                         vld1q_u8((sz_u8_t const *)generated_u64x2 + generated_index)); 
            for (; byte_index < length; ++byte_index, ++generated_index)
                text[byte_index] = ((sz_u8_t *)generated_u64x2)[generated_index]; 
        }
    }
}
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
