/**
 *  @brief Goldmont (SHA-NI) backend for string hashing and checksums.
 *  @file include/stringzilla/hash/goldmont.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_GOLDMONT_H_
#define STRINGZILLA_HASH_GOLDMONT_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#include "stringzilla/hash/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_GOLDMONT
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse3,ssse3,sse4.1,sha"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("sse3", "ssse3", "sse4.1", "sha")
#endif
/**
 *  @brief Process a single 512-bit (64-byte) block of data using SHA256 with SHA-NI intrinsics.
 *  @param hash Pointer to 8x 32-bit hash values, modified in place.
 *  @param block Pointer to 64-byte message block.
 */
SZ_HELPER_AUTO void sz_sha256_process_block_goldmont_(sz_u32_t hash[sz_at_least_(8)],
                                                      sz_u8_t const block[sz_at_least_(SZ_SHA256_BLOCK_LENGTH)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    __m128i const bswap_mask_u8x16 = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    __m128i msg0_u32x4 = _mm_shuffle_epi8(_mm_lddqu_si128((__m128i const *)&block[0]), bswap_mask_u8x16);
    __m128i msg1_u32x4 = _mm_shuffle_epi8(_mm_lddqu_si128((__m128i const *)&block[16]), bswap_mask_u8x16);
    __m128i msg2_u32x4 = _mm_shuffle_epi8(_mm_lddqu_si128((__m128i const *)&block[32]), bswap_mask_u8x16);
    __m128i msg3_u32x4 = _mm_shuffle_epi8(_mm_lddqu_si128((__m128i const *)&block[48]), bswap_mask_u8x16);
    __m128i state0_u32x4 = _mm_lddqu_si128((__m128i const *)&hash[0]); 
    __m128i state1_u32x4 = _mm_lddqu_si128((__m128i const *)&hash[4]); 
    __m128i perm_u32x4 = _mm_shuffle_epi32(state0_u32x4, 0xB1);        
    state1_u32x4 = _mm_shuffle_epi32(state1_u32x4, 0x1B);              
    state0_u32x4 = _mm_alignr_epi8(perm_u32x4, state1_u32x4, 8);       
    state1_u32x4 = _mm_blend_epi16(state1_u32x4, perm_u32x4, 0xF0);    
    __m128i state0_saved_u32x4 = state0_u32x4;
    __m128i state1_saved_u32x4 = state1_u32x4;
    __m128i round_input_u32x4 = _mm_add_epi32(msg0_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[0]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg1_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[4]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg0_u32x4 = _mm_sha256msg1_epu32(msg0_u32x4, msg1_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg2_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[8]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg1_u32x4 = _mm_sha256msg1_epu32(msg1_u32x4, msg2_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg3_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[12]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg0_u32x4 = _mm_add_epi32(msg0_u32x4, _mm_alignr_epi8(msg3_u32x4, msg2_u32x4, 4));
    msg0_u32x4 = _mm_sha256msg2_epu32(msg0_u32x4, msg3_u32x4);
    msg2_u32x4 = _mm_sha256msg1_epu32(msg2_u32x4, msg3_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg0_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[16]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg1_u32x4 = _mm_add_epi32(msg1_u32x4, _mm_alignr_epi8(msg0_u32x4, msg3_u32x4, 4));
    msg1_u32x4 = _mm_sha256msg2_epu32(msg1_u32x4, msg0_u32x4);
    msg3_u32x4 = _mm_sha256msg1_epu32(msg3_u32x4, msg0_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg1_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[20]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg2_u32x4 = _mm_add_epi32(msg2_u32x4, _mm_alignr_epi8(msg1_u32x4, msg0_u32x4, 4));
    msg2_u32x4 = _mm_sha256msg2_epu32(msg2_u32x4, msg1_u32x4);
    msg0_u32x4 = _mm_sha256msg1_epu32(msg0_u32x4, msg1_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg2_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[24]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg3_u32x4 = _mm_add_epi32(msg3_u32x4, _mm_alignr_epi8(msg2_u32x4, msg1_u32x4, 4));
    msg3_u32x4 = _mm_sha256msg2_epu32(msg3_u32x4, msg2_u32x4);
    msg1_u32x4 = _mm_sha256msg1_epu32(msg1_u32x4, msg2_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg3_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[28]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg0_u32x4 = _mm_add_epi32(msg0_u32x4, _mm_alignr_epi8(msg3_u32x4, msg2_u32x4, 4));
    msg0_u32x4 = _mm_sha256msg2_epu32(msg0_u32x4, msg3_u32x4);
    msg2_u32x4 = _mm_sha256msg1_epu32(msg2_u32x4, msg3_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg0_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[32]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg1_u32x4 = _mm_add_epi32(msg1_u32x4, _mm_alignr_epi8(msg0_u32x4, msg3_u32x4, 4));
    msg1_u32x4 = _mm_sha256msg2_epu32(msg1_u32x4, msg0_u32x4);
    msg3_u32x4 = _mm_sha256msg1_epu32(msg3_u32x4, msg0_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg1_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[36]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg2_u32x4 = _mm_add_epi32(msg2_u32x4, _mm_alignr_epi8(msg1_u32x4, msg0_u32x4, 4));
    msg2_u32x4 = _mm_sha256msg2_epu32(msg2_u32x4, msg1_u32x4);
    msg0_u32x4 = _mm_sha256msg1_epu32(msg0_u32x4, msg1_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg2_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[40]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg3_u32x4 = _mm_add_epi32(msg3_u32x4, _mm_alignr_epi8(msg2_u32x4, msg1_u32x4, 4));
    msg3_u32x4 = _mm_sha256msg2_epu32(msg3_u32x4, msg2_u32x4);
    msg1_u32x4 = _mm_sha256msg1_epu32(msg1_u32x4, msg2_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg3_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[44]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg0_u32x4 = _mm_add_epi32(msg0_u32x4, _mm_alignr_epi8(msg3_u32x4, msg2_u32x4, 4));
    msg0_u32x4 = _mm_sha256msg2_epu32(msg0_u32x4, msg3_u32x4);
    msg2_u32x4 = _mm_sha256msg1_epu32(msg2_u32x4, msg3_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg0_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[48]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg1_u32x4 = _mm_add_epi32(msg1_u32x4, _mm_alignr_epi8(msg0_u32x4, msg3_u32x4, 4));
    msg1_u32x4 = _mm_sha256msg2_epu32(msg1_u32x4, msg0_u32x4);
    msg3_u32x4 = _mm_sha256msg1_epu32(msg3_u32x4, msg0_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg1_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[52]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg2_u32x4 = _mm_add_epi32(msg2_u32x4, _mm_alignr_epi8(msg1_u32x4, msg0_u32x4, 4));
    msg2_u32x4 = _mm_sha256msg2_epu32(msg2_u32x4, msg1_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg2_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[56]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    msg3_u32x4 = _mm_add_epi32(msg3_u32x4, _mm_alignr_epi8(msg2_u32x4, msg1_u32x4, 4));
    msg3_u32x4 = _mm_sha256msg2_epu32(msg3_u32x4, msg2_u32x4);
    round_input_u32x4 = _mm_add_epi32(msg3_u32x4, _mm_lddqu_si128((__m128i const *)&round_constants[60]));
    state1_u32x4 = _mm_sha256rnds2_epu32(state1_u32x4, state0_u32x4, round_input_u32x4);
    round_input_u32x4 = _mm_shuffle_epi32(round_input_u32x4, 0x0E);
    state0_u32x4 = _mm_sha256rnds2_epu32(state0_u32x4, state1_u32x4, round_input_u32x4);
    state0_u32x4 = _mm_add_epi32(state0_u32x4, state0_saved_u32x4);
    state1_u32x4 = _mm_add_epi32(state1_u32x4, state1_saved_u32x4);
    perm_u32x4 = _mm_shuffle_epi32(state0_u32x4, 0x1B);             
    state1_u32x4 = _mm_shuffle_epi32(state1_u32x4, 0xB1);           
    state0_u32x4 = _mm_blend_epi16(perm_u32x4, state1_u32x4, 0xF0); 
    state1_u32x4 = _mm_alignr_epi8(state1_u32x4, perm_u32x4, 8);    
    _mm_storeu_si128((__m128i *)&hash[0], state0_u32x4);
    _mm_storeu_si128((__m128i *)&hash[4], state1_u32x4);
}
SZ_API_COMPTIME void sz_sha256_state_init_goldmont(sz_sha256_state_t *state_ptr) {
    sz_u32_t const *initial_hash = sz_sha256_initial_hash_();
    _mm_storeu_si128((__m128i *)&state_ptr->hash[0], _mm_lddqu_si128((__m128i const *)&initial_hash[0]));
    _mm_storeu_si128((__m128i *)&state_ptr->hash[4], _mm_lddqu_si128((__m128i const *)&initial_hash[4]));
    state_ptr->block_length = 0, state_ptr->total_length = 0;
}
SZ_API_COMPTIME void sz_sha256_state_update_goldmont(sz_sha256_state_t *state_ptr, sz_cptr_t data, sz_size_t length) {
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
    sz_align_(16) sz_u32_t hash[8];
    _mm_store_si128((__m128i *)&hash[0], _mm_lddqu_si128((__m128i const *)&state_ptr->hash[0]));
    _mm_store_si128((__m128i *)&hash[4], _mm_lddqu_si128((__m128i const *)&state_ptr->hash[4]));
    if (head_length) {
        for (sz_size_t byte_index = 0; byte_index < head_length; ++byte_index)
            state_ptr->block[state_ptr->block_length++] = input[byte_index];
        sz_sha256_process_block_goldmont_(hash, state_ptr->block);
        state_ptr->block_length = 0;
        input += head_length;
    }
    for (sz_size_t processed = 0; processed < body_length;
         processed += SZ_SHA256_BLOCK_LENGTH, input += SZ_SHA256_BLOCK_LENGTH)
        sz_sha256_process_block_goldmont_(hash, input);
    for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
        state_ptr->block[byte_index] = input[byte_index];
    state_ptr->block_length = tail_length;
    _mm_storeu_si128((__m128i *)&state_ptr->hash[0], _mm_load_si128((__m128i const *)&hash[0]));
    _mm_storeu_si128((__m128i *)&state_ptr->hash[4], _mm_load_si128((__m128i const *)&hash[4]));
}
SZ_API_COMPTIME void sz_sha256_state_digest_goldmont(sz_sha256_state_t const *state_ptr,
                                                     sz_u8_t digest[sz_at_least_(SZ_SHA256_DIGEST_LENGTH)]) {
    sz_sha256_state_t state = *state_ptr;
    state.block[state.block_length++] = 0x80;
    if (state.block_length > 56) {
        sz_size_t remaining = SZ_SHA256_BLOCK_LENGTH - state.block_length;
        sz_size_t xmm_bytes = (remaining / 16) * 16;
        for (sz_size_t byte_index = 0; byte_index < xmm_bytes; byte_index += 16)
            _mm_storeu_si128((__m128i *)&state.block[state.block_length + byte_index], _mm_setzero_si128());
        for (sz_size_t byte_index = xmm_bytes; byte_index < remaining; ++byte_index)
            state.block[state.block_length + byte_index] = 0;
        sz_sha256_process_block_goldmont_(state.hash, state.block);
        state.block_length = 0;
    }
    sz_size_t remaining = 56 - state.block_length;
    sz_size_t xmm_bytes = (remaining / 16) * 16;
    for (sz_size_t byte_index = 0; byte_index < xmm_bytes; byte_index += 16)
        _mm_storeu_si128((__m128i *)&state.block[state.block_length + byte_index], _mm_setzero_si128());
    for (sz_size_t byte_index = xmm_bytes; byte_index < remaining; ++byte_index)
        state.block[state.block_length + byte_index] = 0;
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
    sz_sha256_process_block_goldmont_(state.hash, state.block);
    for (sz_size_t lane_index = 0; lane_index < 8; ++lane_index) {
        digest[lane_index * 4 + 0] = (sz_u8_t)(state.hash[lane_index] >> 24);
        digest[lane_index * 4 + 1] = (sz_u8_t)(state.hash[lane_index] >> 16);
        digest[lane_index * 4 + 2] = (sz_u8_t)(state.hash[lane_index] >> 8);
        digest[lane_index * 4 + 3] = (sz_u8_t)(state.hash[lane_index] >> 0);
    }
}
SZ_API_COMPTIME void sz_sha256_multistate_update_goldmont(sz_sha256_state_t *states, sz_sequence_t const *texts) {
    sz_size_t const lanes_count = texts->count;
    for (sz_size_t lane_index = 0; lane_index != lanes_count; ++lane_index)
        sz_sha256_state_update_goldmont(&states[lane_index], texts->get_start(texts->handle, lane_index),
                                        texts->get_length(texts->handle, lane_index));
}
SZ_API_COMPTIME void sz_sha256_multistate_digest_goldmont(sz_sha256_state_t const *states, sz_size_t states_count,
                                                          sz_u8_t *digests) {
    for (sz_size_t lane_index = 0; lane_index != states_count; ++lane_index)
        sz_sha256_state_digest_goldmont(&states[lane_index], &digests[lane_index * SZ_SHA256_DIGEST_LENGTH]);
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
