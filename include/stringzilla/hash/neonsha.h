/**
 *  @brief NEON + SHA2 backend for string hashing and checksums.
 *  @file include/stringzilla/hash/neonsha.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_NEONSHA_H_
#define STRINGZILLA_HASH_NEONSHA_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#include "stringzilla/hash/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_NEONSHA
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd+crypto+sha2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd+crypto+sha2")
#endif
/**
 *  @brief Process a single 512-bit (64-byte) block of data using SHA256.
 *  @param hash Pointer to 8x 32-bit hash values, modified in place.
 *  @param block Pointer to 64-byte message block.
 */
SZ_HELPER_AUTO void sz_sha256_process_block_neon_(sz_u32_t hash[sz_at_least_(8)],
                                                  sz_u8_t const block[sz_at_least_(SZ_SHA256_BLOCK_LENGTH)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    uint32x4x4_t k_batch0 = vld1q_u32_x4(&round_constants[0]);  
    uint32x4x4_t k_batch1 = vld1q_u32_x4(&round_constants[16]); 
    uint32x4x4_t k_batch2 = vld1q_u32_x4(&round_constants[32]); 
    uint32x4x4_t k_batch3 = vld1q_u32_x4(&round_constants[48]); 
    uint32x4_t k0 = k_batch0.val[0];
    uint32x4_t k1 = k_batch0.val[1];
    uint32x4_t k2 = k_batch0.val[2];
    uint32x4_t k3 = k_batch0.val[3];
    uint32x4_t k4 = k_batch1.val[0];
    uint32x4_t k5 = k_batch1.val[1];
    uint32x4_t k6 = k_batch1.val[2];
    uint32x4_t k7 = k_batch1.val[3];
    uint32x4_t k8 = k_batch2.val[0];
    uint32x4_t k9 = k_batch2.val[1];
    uint32x4_t k10 = k_batch2.val[2];
    uint32x4_t k11 = k_batch2.val[3];
    uint32x4_t k12 = k_batch3.val[0];
    uint32x4_t k13 = k_batch3.val[1];
    uint32x4_t k14 = k_batch3.val[2];
    uint32x4_t k15 = k_batch3.val[3];
    uint32x4_t state0 = vld1q_u32(&hash[0]); 
    uint32x4_t state1 = vld1q_u32(&hash[4]); 
    uint32x4_t state0_saved = state0;
    uint32x4_t state1_saved = state1;
    uint32x4_t msg0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(&block[0])));
    uint32x4_t msg1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(&block[16])));
    uint32x4_t msg2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(&block[32])));
    uint32x4_t msg3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(&block[48])));
    uint32x4_t scratch_0, scratch_1;
    scratch_0 = vaddq_u32(msg0, k0);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    scratch_0 = vaddq_u32(msg1, k1);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    scratch_0 = vaddq_u32(msg2, k2);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    scratch_0 = vaddq_u32(msg3, k3);
    msg0 = vsha256su0q_u32(msg0, msg1);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg0 = vsha256su1q_u32(msg0, msg2, msg3);
    scratch_0 = vaddq_u32(msg0, k4);
    msg1 = vsha256su0q_u32(msg1, msg2);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg1 = vsha256su1q_u32(msg1, msg3, msg0);
    scratch_0 = vaddq_u32(msg1, k5);
    msg2 = vsha256su0q_u32(msg2, msg3);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg2 = vsha256su1q_u32(msg2, msg0, msg1);
    scratch_0 = vaddq_u32(msg2, k6);
    msg3 = vsha256su0q_u32(msg3, msg0);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg3 = vsha256su1q_u32(msg3, msg1, msg2);
    scratch_0 = vaddq_u32(msg3, k7);
    msg0 = vsha256su0q_u32(msg0, msg1);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg0 = vsha256su1q_u32(msg0, msg2, msg3);
    scratch_0 = vaddq_u32(msg0, k8);
    msg1 = vsha256su0q_u32(msg1, msg2);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg1 = vsha256su1q_u32(msg1, msg3, msg0);
    scratch_0 = vaddq_u32(msg1, k9);
    msg2 = vsha256su0q_u32(msg2, msg3);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg2 = vsha256su1q_u32(msg2, msg0, msg1);
    scratch_0 = vaddq_u32(msg2, k10);
    msg3 = vsha256su0q_u32(msg3, msg0);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg3 = vsha256su1q_u32(msg3, msg1, msg2);
    scratch_0 = vaddq_u32(msg3, k11);
    msg0 = vsha256su0q_u32(msg0, msg1);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg0 = vsha256su1q_u32(msg0, msg2, msg3);
    scratch_0 = vaddq_u32(msg0, k12);
    msg1 = vsha256su0q_u32(msg1, msg2);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg1 = vsha256su1q_u32(msg1, msg3, msg0);
    scratch_0 = vaddq_u32(msg1, k13);
    msg2 = vsha256su0q_u32(msg2, msg3);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg2 = vsha256su1q_u32(msg2, msg0, msg1);
    scratch_0 = vaddq_u32(msg2, k14);
    msg3 = vsha256su0q_u32(msg3, msg0);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg3 = vsha256su1q_u32(msg3, msg1, msg2);
    scratch_0 = vaddq_u32(msg3, k15);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    state0 = vaddq_u32(state0, state0_saved);
    state1 = vaddq_u32(state1, state1_saved);
    vst1q_u32(&hash[0], state0);
    vst1q_u32(&hash[4], state1);
}
SZ_API_COMPTIME void sz_sha256_state_init_neonsha(sz_sha256_state_t *state) {
    sz_u32_t const *initial_hash = sz_sha256_initial_hash_();
    vst1q_u32(&state->hash[0], vld1q_u32(&initial_hash[0]));
    vst1q_u32(&state->hash[4], vld1q_u32(&initial_hash[4]));
    state->block_length = 0, state->total_length = 0;
}
SZ_API_COMPTIME void sz_sha256_state_update_neonsha(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *input_cursor = (sz_u8_t const *)text;
    sz_size_t const current_block_index = state->block_length / SZ_SHA256_BLOCK_LENGTH;
    sz_size_t const final_block_index = (state->block_length + length) / SZ_SHA256_BLOCK_LENGTH;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state->block_length + length) % SZ_SHA256_BLOCK_LENGTH == 0;
    state->total_length += length;
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state->block_length, ++input_cursor)
            state->block[state->block_length] = *input_cursor;
        return;
    }
    sz_size_t const head_length = (SZ_SHA256_BLOCK_LENGTH - state->block_length) % SZ_SHA256_BLOCK_LENGTH;
    sz_size_t const tail_length = (state->block_length + length) % SZ_SHA256_BLOCK_LENGTH;
    sz_size_t const body_length = length - head_length - tail_length;
    sz_align_(16) sz_u32_t hash[8];
    vst1q_u32(&hash[0], vld1q_u32(&state->hash[0]));
    vst1q_u32(&hash[4], vld1q_u32(&state->hash[4]));
    if (head_length) {
        for (sz_size_t byte_index = 0; byte_index < head_length; ++byte_index)
            state->block[state->block_length++] = input_cursor[byte_index];
        sz_sha256_process_block_neon_(hash, state->block);
        state->block_length = 0;
        input_cursor += head_length;
    }
    for (sz_size_t processed = 0; processed < body_length;
         processed += SZ_SHA256_BLOCK_LENGTH, input_cursor += SZ_SHA256_BLOCK_LENGTH)
        sz_sha256_process_block_neon_(hash, input_cursor);
    for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
        state->block[byte_index] = input_cursor[byte_index];
    state->block_length = tail_length;
    vst1q_u32(&state->hash[0], vld1q_u32(&hash[0]));
    vst1q_u32(&state->hash[4], vld1q_u32(&hash[4]));
}
SZ_API_COMPTIME void sz_sha256_state_digest_neonsha(sz_sha256_state_t const *state,
                                                    sz_u8_t digest[sz_at_least_(SZ_SHA256_DIGEST_LENGTH)]) {
    sz_sha256_state_t local_state = *state;
    local_state.block[local_state.block_length++] = 0x80;
    if (local_state.block_length > 56) {
        sz_size_t remaining = SZ_SHA256_BLOCK_LENGTH - local_state.block_length;
        sz_size_t vec_bytes = (remaining / 16) * 16;
        uint8x16_t zeros_u8x16 = vdupq_n_u8(0);
        for (sz_size_t byte_index = 0; byte_index < vec_bytes; byte_index += 16)
            vst1q_u8(&local_state.block[local_state.block_length + byte_index], zeros_u8x16);
        for (sz_size_t byte_index = vec_bytes; byte_index < remaining; ++byte_index)
            local_state.block[local_state.block_length + byte_index] = 0;
        sz_sha256_process_block_neon_(local_state.hash, local_state.block);
        local_state.block_length = 0;
    }
    sz_size_t remaining = 56 - local_state.block_length;
    sz_size_t vec_bytes = (remaining / 16) * 16;
    uint8x16_t zeros_u8x16 = vdupq_n_u8(0);
    for (sz_size_t byte_index = 0; byte_index < vec_bytes; byte_index += 16)
        vst1q_u8(&local_state.block[local_state.block_length + byte_index], zeros_u8x16);
    for (sz_size_t byte_index = vec_bytes; byte_index < remaining; ++byte_index)
        local_state.block[local_state.block_length + byte_index] = 0;
    local_state.block_length = 56;
    sz_u64_t bit_length = local_state.total_length * 8;
    local_state.block[56] = (sz_u8_t)(bit_length >> 56);
    local_state.block[57] = (sz_u8_t)(bit_length >> 48);
    local_state.block[58] = (sz_u8_t)(bit_length >> 40);
    local_state.block[59] = (sz_u8_t)(bit_length >> 32);
    local_state.block[60] = (sz_u8_t)(bit_length >> 24);
    local_state.block[61] = (sz_u8_t)(bit_length >> 16);
    local_state.block[62] = (sz_u8_t)(bit_length >> 8);
    local_state.block[63] = (sz_u8_t)(bit_length >> 0);
    sz_sha256_process_block_neon_(local_state.hash, local_state.block);
    for (sz_size_t lane_index = 0; lane_index < 8; ++lane_index) {
        digest[lane_index * 4 + 0] = (sz_u8_t)(local_state.hash[lane_index] >> 24);
        digest[lane_index * 4 + 1] = (sz_u8_t)(local_state.hash[lane_index] >> 16);
        digest[lane_index * 4 + 2] = (sz_u8_t)(local_state.hash[lane_index] >> 8);
        digest[lane_index * 4 + 3] = (sz_u8_t)(local_state.hash[lane_index] >> 0);
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
