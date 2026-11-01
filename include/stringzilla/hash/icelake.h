/**
 *  @brief Ice Lake (AVX-512 + VAES) backend for string hashing and checksums.
 *  @file include/stringzilla/hash/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_ICELAKE_H_
#define STRINGZILLA_HASH_ICELAKE_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#include "stringzilla/hash/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                                   \
    __attribute__((                                                                                             \
        target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vnni,bmi,bmi2,aes,vaes,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                  \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vnni,bmi,bmi2,aes,vaes"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vnni", "bmi", "bmi2", \
                   "aes", "vaes")
#endif
SZ_API_COMPTIME sz_u64_t sz_bytesum_icelake(sz_cptr_t text, sz_size_t length) {
    int const is_huge = length >= 1ull * 1024ull * 1024ull;
    sz_u512_vec_t text_vec, sums_vec;
    if (length <= 16) {
        __mmask16 mask_m16 = sz_u16_mask_until_(length);
        text_vec.xmms[0] = _mm_maskz_loadu_epi8(mask_m16, text);
        sums_vec.xmms[0] = _mm_sad_epu8(text_vec.xmms[0], _mm_setzero_si128());
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_vec.xmms[0]);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_vec.xmms[0], 1);
        return low + high;
    }
    else if (length <= 32) {
        __mmask32 mask_m32 = sz_u32_mask_until_(length);
        text_vec.ymms[0] = _mm256_maskz_loadu_epi8(mask_m32, text);
        sums_vec.ymms[0] = _mm256_sad_epu8(text_vec.ymms[0], _mm256_setzero_si256());
        __m128i low_u64x2 = _mm256_castsi256_si128(sums_vec.ymms[0]);
        __m128i high_u64x2 = _mm256_extracti128_si256(sums_vec.ymms[0], 1);
        __m128i sums_u64x2 = _mm_add_epi64(low_u64x2, high_u64x2);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_u64x2);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_u64x2, 1);
        return low + high;
    }
    else if (length <= 64) {
        __mmask64 mask_m64 = sz_u64_mask_until_(length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(mask_m64, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        return _mm512_reduce_add_epi64(sums_vec.zmm);
    }
    else if (!is_huge) {
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64; 
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;    
        sz_size_t body_length = length - head_length - tail_length; 
        sz_assert_(body_length % 64 == 0 && head_length < 64 && tail_length < 64);
        __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);
        sz_u512_vec_t zeros_vec, ones_vec;
        zeros_vec.zmm = _mm512_setzero_si512();
        ones_vec.zmm = _mm512_set1_epi8(1);
        sz_u512_vec_t text_reversed_vec, sums_reversed_vec;
        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask_m64, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, zeros_vec.zmm);
        text_reversed_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask_m64, text + head_length + body_length);
        sums_reversed_vec.zmm = _mm512_dpbusds_epi32(zeros_vec.zmm, text_reversed_vec.zmm, ones_vec.zmm);
        for (text += head_length; body_length >= 128; text += 64, body_length -= 128) {
            text_reversed_vec.zmm = _mm512_load_si512((__m512i *)(text + body_length - 64));
            sums_reversed_vec.zmm = _mm512_dpbusds_epi32(sums_reversed_vec.zmm, text_reversed_vec.zmm, ones_vec.zmm);
            text_vec.zmm = _mm512_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, zeros_vec.zmm));
        }
        if (body_length >= 64) {
            sz_assert_(body_length == 64);
            text_vec.zmm = _mm512_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, zeros_vec.zmm));
        }
        return _mm512_reduce_add_epi64(sums_vec.zmm) + _mm512_reduce_add_epi32(sums_reversed_vec.zmm);
    }
    else {
        sz_u512_vec_t text_reversed_vec, sums_reversed_vec;
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64;
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;
        sz_size_t body_length = length - head_length - tail_length;
        __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask_m64, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        text_reversed_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask_m64, text + head_length + body_length);
        sums_reversed_vec.zmm = _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512());
        for (text += head_length; body_length >= 128; text += 64, body_length -= 128) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
            text_reversed_vec.zmm = _mm512_stream_load_si512((__m512i *)(text + body_length - 64));
            sums_reversed_vec.zmm = _mm512_add_epi64(sums_reversed_vec.zmm,
                                                     _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512()));
        }
        if (body_length >= 64) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        }
        return _mm512_reduce_add_epi64(_mm512_add_epi64(sums_vec.zmm, sums_reversed_vec.zmm));
    }
}
SZ_API_COMPTIME SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_icelake(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length), start);
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 32) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 16), start + 16);
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 32), start + 32);
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data2_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 32));
        data3_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 48), start + 48);
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data2_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data3_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_aligned_t state;
        sz_hash_state_init_skylake((sz_hash_state_t *)&state, seed);
        __m512i const order_u8x64 = _mm512_load_si512((__m512i const *)sz_hash_u8x16x4_shuffle_());
        for (; state.ins_length + 64 < length; state.ins_length += 64) {
            state.ins.zmm = _mm512_loadu_epi8(start + state.ins_length);
            state.aes.zmm = _mm512_aesenc_epi128(state.aes.zmm, state.ins.zmm);
            state.sum.zmm = _mm512_add_epi64(_mm512_shuffle_epi8(state.sum.zmm, order_u8x64), state.ins.zmm);
        }
        state.ins.zmm = _mm512_maskz_loadu_epi8(sz_u64_mask_until_(length - state.ins_length),
                                                start + state.ins_length);
        state.ins_length = length;
        return sz_hash_state_finalize_westmere_(&state);
    }
}
SZ_API_COMPTIME void sz_hash_state_init_icelake(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_skylake(state, seed);
}
/** @brief Loads the packed public state into the aligned twin (one `_mm512_loadu_si512` per 64-byte field). */
SZ_HELPER_AUTO sz_hash_state_aligned_t sz_hash_state_load_icelake_(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state;
    state.aes.zmm = _mm512_loadu_si512((__m512i const *)packed->aes);
    state.sum.zmm = _mm512_loadu_si512((__m512i const *)packed->sum);
    state.ins.zmm = _mm512_loadu_si512((__m512i const *)packed->ins);
    state.key.xmm = _mm_lddqu_si128((__m128i const *)packed->key);
    state.ins_length = packed->ins_length;
    return state;
}
/** @brief Stores the aligned twin back into the packed public state (one `_mm512_storeu_si512` per field). */
SZ_HELPER_AUTO void sz_hash_state_store_icelake_(sz_hash_state_t *packed, sz_hash_state_aligned_t const *state) {
    _mm512_storeu_si512((__m512i *)packed->aes, state->aes.zmm);
    _mm512_storeu_si512((__m512i *)packed->sum, state->sum.zmm);
    _mm512_storeu_si512((__m512i *)packed->ins, state->ins.zmm);
    _mm_storeu_si128((__m128i *)packed->key, state->key.xmm);
    packed->ins_length = state->ins_length;
}
/** @brief Absorbs the buffered 64-byte block into the aligned state with a single VAES `VAESENC` over four lanes. */
SZ_HELPER_AUTO void sz_hash_state_update_icelake_(sz_hash_state_aligned_t *state) {
    __m512i const order_u8x64 = _mm512_load_si512((__m512i const *)sz_hash_u8x16x4_shuffle_());
    state->aes.zmm = _mm512_aesenc_epi128(state->aes.zmm, state->ins.zmm);
    state->sum.zmm = _mm512_add_epi64(_mm512_shuffle_epi8(state->sum.zmm, order_u8x64), state->ins.zmm);
}
SZ_API_COMPTIME void sz_hash_state_update_icelake(sz_hash_state_t *state_ptr, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_aligned_t state = sz_hash_state_load_icelake_(state_ptr);
    sz_size_t buffered = state.ins_length % 64;
    if (buffered == 0 && state.ins_length) buffered = 64;
    sz_align_(64) static sz_u8_t const lane_iota[64] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                                        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                                        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
    __m512i const lane_iota_u8x64 = _mm512_load_si512((__m512i const *)lane_iota);
    while (length) {
        if (buffered == 64) { 
            sz_hash_state_update_icelake_(&state);
            state.ins.zmm = _mm512_setzero_si512();
            buffered = 0;
        }
        sz_size_t const to_copy = sz_min_of_two(length, (sz_size_t)64 - buffered);
        __mmask64 const copy_mask_m64 = _cvtu64_mask64(sz_u64_mask_until_(to_copy));
        __m512i const incoming_u8x64 = _mm512_maskz_loadu_epi8(copy_mask_m64, text);
        __m512i const slide_u8x64 = _mm512_sub_epi8(lane_iota_u8x64, _mm512_set1_epi8((char)buffered));
        __m512i const shifted_u8x64 = _mm512_permutexvar_epi8(slide_u8x64, incoming_u8x64);
        state.ins.zmm = _mm512_mask_blend_epi8(_cvtu64_mask64(sz_u64_mask_until_(to_copy) << buffered), state.ins.zmm,
                                               shifted_u8x64);
        buffered += to_copy, text += to_copy, length -= to_copy, state.ins_length += to_copy;
    }
    sz_hash_state_store_icelake_(state_ptr, &state);
}
SZ_API_COMPTIME sz_u64_t sz_hash_state_digest_icelake(sz_hash_state_t const *state) {
    return sz_hash_state_digest_westmere(state);
}
SZ_API_COMPTIME void sz_fill_random_icelake(sz_ptr_t output, sz_size_t length, sz_u64_t nonce) {
    if (length <= 16) {
        __m128i input_u8x16 = _mm_set1_epi64x(nonce);
        __m128i pi_u8x16 = _mm_load_si128((__m128i const *)sz_hash_pi_constants_());
        __m128i key_u8x16 = _mm_xor_si128(_mm_set1_epi64x(nonce), pi_u8x16);
        __m128i generated_u8x16 = _mm_aesenc_si128(input_u8x16, key_u8x16);
        __mmask16 store_mask_m16 = sz_u16_mask_until_(length);
        _mm_mask_storeu_epi8((void *)output, store_mask_m16, generated_u8x16);
    }
    else if (length <= 32) {
        __m256i input_u8x32 = _mm256_set_epi64x(nonce + 1, nonce + 1, nonce, nonce);
        __m256i pi_u8x32 = _mm256_load_si256((__m256i const *)sz_hash_pi_constants_());
        __m256i key_u8x32 = _mm256_xor_si256(_mm256_set1_epi64x(nonce), pi_u8x32);
        __m256i generated_u8x32 = _mm256_aesenc_epi128(input_u8x32, key_u8x32);
        __mmask32 store_mask_m32 = sz_u32_mask_until_(length);
        _mm256_mask_storeu_epi8((void *)output, store_mask_m32, generated_u8x32);
    }
    else if (length <= 64) {
        __m512i input_u8x64 = _mm512_set_epi64(         
            nonce + 3, nonce + 3, nonce + 2, nonce + 2, 
            nonce + 1, nonce + 1, nonce, nonce);
        __m512i pi_u8x64 = _mm512_load_si512((__m512i const *)sz_hash_pi_constants_());
        __m512i key_u8x64 = _mm512_xor_si512(_mm512_set1_epi64(nonce), pi_u8x64);
        __m512i generated_u8x64 = _mm512_aesenc_epi128(input_u8x64, key_u8x64);
        __mmask64 store_mask_m64 = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8((void *)output, store_mask_m64, generated_u8x64);
    }
    else {
        __m512i const increment_u64x8 = _mm512_set1_epi64(4);
        __m512i input_u8x64 = _mm512_set_epi64(         
            nonce + 3, nonce + 3, nonce + 2, nonce + 2, 
            nonce + 1, nonce + 1, nonce, nonce);
        __m512i const pi_u8x64 = _mm512_load_si512((__m512i const *)sz_hash_pi_constants_());
        __m512i const key_u8x64 = _mm512_xor_si512(_mm512_set1_epi64(nonce), pi_u8x64);
        sz_size_t byte_index = 0;
        for (; byte_index + 64 <= length; byte_index += 64) {
            __m512i generated_u8x64 = _mm512_aesenc_epi128(input_u8x64, key_u8x64);
            _mm512_storeu_epi8((void *)(output + byte_index), generated_u8x64);
            input_u8x64 = _mm512_add_epi64(input_u8x64, increment_u64x8);
        }
        __m512i generated_u8x64 = _mm512_aesenc_epi128(input_u8x64, key_u8x64);
        __mmask64 store_mask_m64 = sz_u64_mask_until_(length - byte_index);
        _mm512_mask_storeu_epi8((void *)(output + byte_index), store_mask_m64, generated_u8x64);
    }
}
/**
 *  @brief A wider parallel analog of `sz_hash_state_aligned_for_short_t`, which is not used for computing individual hashes,
 *         but for parallel hashing of @b short 4x separate strings under 16 bytes long.
 *         Useful for higher-level Database and Machine Learning operations.
 */
typedef struct sz_hash_state_aligned_for_short_x4_t {
    sz_u512_vec_t aes_vec;
    sz_u512_vec_t sum_vec;
    sz_u512_vec_t key_vec;
} sz_hash_state_aligned_for_short_x4_t;
/**
 *  @brief Initializes the 4-wide parallel minimal hash state using VAES and AVX-512 intrinsics.
 *  @param state Pointer to the 4-wide minimal hash state to initialize.
 *  @param seed 64-bit seed XOR-ed with Pi constants replicated across all four 128-bit lanes.
 */
SZ_HELPER_AUTO void sz_hash_state_short_x4_init_icelake_(sz_hash_state_aligned_for_short_x4_t *state, sz_u64_t seed) {
    __m512i seed_u64x8 = _mm512_set1_epi64(seed);
    state->key_vec.zmm = seed_u64x8; 
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m512i pi0_u8x64 = _mm512_load_si512((__m512i const *)(pi));
    __m512i pi1_u8x64 = _mm512_load_si512((__m512i const *)(pi + 8));
    pi0_u8x64 = _mm512_shuffle_i64x2(pi0_u8x64, pi0_u8x64, 0);
    pi1_u8x64 = _mm512_shuffle_i64x2(pi1_u8x64, pi1_u8x64, 0);
    __m512i aes_state_key_u8x64 = _mm512_xor_si512(seed_u64x8, pi0_u8x64);
    __m512i sum_state_key_u8x64 = _mm512_xor_si512(seed_u64x8, pi1_u8x64);
    state->aes_vec.zmm = aes_state_key_u8x64;
    state->sum_vec.zmm = sum_state_key_u8x64;
}
/**
 *  @brief Finalizes the 4-wide parallel minimal hash state, returning four 64-bit digests packed in a 256-bit vector.
 *  @param state Pointer to the (const) 4-wide minimal hash state.
 *  @param length0 Total byte count for the first 128-bit lane.
 *  @param length1 Total byte count for the second 128-bit lane.
 *  @param length2 Total byte count for the third 128-bit lane.
 *  @param length3 Total byte count for the fourth 128-bit lane.
 *  @return 256-bit vector containing four 64-bit hash values (one per lane).
 */
SZ_HELPER_AUTO __m256i sz_hash_state_short_x4_finalize_icelake_(sz_hash_state_aligned_for_short_x4_t const *state, 
                                                                sz_size_t length0, sz_size_t length1, sz_size_t length2,
                                                                sz_size_t length3) {
    __m512i const padded_lengths_u64x8 = _mm512_set_epi64(0, length3, 0, length2, 0, length1, 0, length0);
    __m512i key_with_length_u64x8 = _mm512_add_epi64(state->key_vec.zmm, padded_lengths_u64x8);
    __m512i mixed_u8x64 = _mm512_aesenc_epi128(state->sum_vec.zmm, state->aes_vec.zmm);
    __m512i mixed_in_register_u8x64 = _mm512_aesenc_epi128(_mm512_aesenc_epi128(mixed_u8x64, key_with_length_u64x8),
                                                           mixed_u8x64);
    return _mm512_castsi512_si256(
        _mm512_permutexvar_epi64(_mm512_set_epi64(0, 0, 0, 0, 6, 4, 2, 0), mixed_in_register_u8x64));
}
/**
 *  @brief Absorbs four 128-bit blocks (one per hash lane) into the 4-wide minimal hash state.
 *  @param state Pointer to the 4-wide minimal hash state.
 *  @param blocks_u8x64 512-bit register containing four 128-bit data blocks, one per lane.
 */
SZ_HELPER_AUTO void sz_hash_state_short_x4_update_icelake_(sz_hash_state_aligned_for_short_x4_t *state,
                                                           __m512i blocks_u8x64) {
    __m512i const order_u8x64 = _mm512_load_si512((__m512i const *)sz_hash_u8x16x4_shuffle_());
    state->aes_vec.zmm = _mm512_aesenc_epi128(state->aes_vec.zmm, blocks_u8x64);
    state->sum_vec.zmm = _mm512_add_epi64(_mm512_shuffle_epi8(state->sum_vec.zmm, order_u8x64), blocks_u8x64);
}
/**
 *  @brief Initializes the 4-wide parallel minimal state with four @b distinct seeds (one per lane).
 *         Unlike `sz_hash_state_short_x4_init_icelake_`, which seeds all four lanes identically to hash
 *         four different strings, this seeds each lane differently to hash one string under four seeds.
 *  @param state Pointer to the 4-wide minimal hash state to initialize.
 *  @param seeds_u64x8 Four seeds spread as `[s0,s0,s1,s1,s2,s2,s3,s3]` across the 512-bit register.
 */
SZ_HELPER_AUTO void sz_hash_multiseed_x4_init_icelake_(sz_hash_state_aligned_for_short_x4_t *state,
                                                       __m512i seeds_u64x8) {
    state->key_vec.zmm = seeds_u64x8;
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m512i pi0_u8x64 = _mm512_shuffle_i64x2(_mm512_load_si512((__m512i const *)(pi)), 
                                             _mm512_load_si512((__m512i const *)(pi)), 0);
    __m512i pi1_u8x64 = _mm512_shuffle_i64x2(_mm512_load_si512((__m512i const *)(pi + 8)), 
                                             _mm512_load_si512((__m512i const *)(pi + 8)), 0);
    state->aes_vec.zmm = _mm512_xor_si512(seeds_u64x8, pi0_u8x64);
    state->sum_vec.zmm = _mm512_xor_si512(seeds_u64x8, pi1_u8x64);
}
/**
 *  @brief Finalizes the 4-wide parallel minimal state, folding in the input length.
 *  @param state Pointer to the (const) 4-wide minimal hash state.
 *  @param lengths_u64x8 The length addend broadcast into each seed-lane's key low half, i.e.
 *                 `[0, length, 0, length, 0, length, 0, length]`. Seed-independent, so the caller
 *                 builds it once and reuses it across all seed groups.
 *  @return 256-bit vector with four 64-bit hashes, one per lane.
 */
SZ_HELPER_AUTO __m256i sz_hash_multiseed_x4_finalize_icelake_(sz_hash_state_aligned_for_short_x4_t const *state,
                                                              __m512i lengths_u64x8) {
    __m512i key_with_length_u64x8 = _mm512_add_epi64(state->key_vec.zmm, lengths_u64x8);
    __m512i mixed_u8x64 = _mm512_aesenc_epi128(state->sum_vec.zmm, state->aes_vec.zmm);
    __m512i mixed_in_register_u8x64 = _mm512_aesenc_epi128(_mm512_aesenc_epi128(mixed_u8x64, key_with_length_u64x8),
                                                           mixed_u8x64);
    return _mm512_castsi512_si256(
        _mm512_permutexvar_epi64(_mm512_set_epi64(0, 0, 0, 0, 6, 4, 2, 0), mixed_in_register_u8x64));
}
SZ_API_COMPTIME void sz_hash_multiseed_icelake(sz_cptr_t text, sz_size_t length,             
                                               sz_u64_t const *seeds, sz_size_t seeds_count, 
                                               sz_u64_t *hashes) {
    if (seeds_count == 0) return;
    if (seeds_count == 1) {
        hashes[0] = sz_hash_icelake(text, length, seeds[0]);
        return;
    }
    if (length > 64) {
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_icelake(text, length, seeds[seed_index]);
        return;
    }
    sz_u512_vec_t text_lanes_vec;
    text_lanes_vec.zmm = _mm512_maskz_loadu_epi8(sz_u64_mask_until_(length), text);
    sz_size_t const text_lanes_count = length <= 16 ? 1 : sz_size_divide_round_up(length, 16);
    __m512i broadcast_text_lanes_u8x64[4];
    for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
        broadcast_text_lanes_u8x64[lane_index] = _mm512_broadcast_i32x4(text_lanes_vec.xmms[lane_index]);
    __m512i const lengths_u64x8 = _mm512_set_epi64(0, (sz_i64_t)length, 0, (sz_i64_t)length, 
                                                   0, (sz_i64_t)length, 0, (sz_i64_t)length);
    __m512i const seed_spread_u64x8 = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
    sz_size_t seed_index = 0;
    for (; seed_index + 4 <= seeds_count; seed_index += 4) {
        sz_hash_state_aligned_for_short_x4_t state;
        __m512i const seeds_u64x8 = _mm512_permutexvar_epi64(
            seed_spread_u64x8, _mm512_castsi256_si512(_mm256_loadu_si256((__m256i const *)(seeds + seed_index))));
        sz_hash_multiseed_x4_init_icelake_(&state, seeds_u64x8);
        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
            sz_hash_state_short_x4_update_icelake_(&state, broadcast_text_lanes_u8x64[lane_index]);
        _mm256_storeu_si256((__m256i *)(hashes + seed_index),
                            sz_hash_multiseed_x4_finalize_icelake_(&state, lengths_u64x8));
    }
    if (seed_index < seeds_count) {
        __mmask8 const seed_mask_m8 = (__mmask8)((1u << (seeds_count - seed_index)) - 1);
        sz_hash_state_aligned_for_short_x4_t state;
        __m512i const seeds_u64x8 = _mm512_permutexvar_epi64(
            seed_spread_u64x8, _mm512_castsi256_si512(_mm256_maskz_loadu_epi64(seed_mask_m8, seeds + seed_index)));
        sz_hash_multiseed_x4_init_icelake_(&state, seeds_u64x8);
        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
            sz_hash_state_short_x4_update_icelake_(&state, broadcast_text_lanes_u8x64[lane_index]);
        _mm256_mask_storeu_epi64(hashes + seed_index, seed_mask_m8,
                                 sz_hash_multiseed_x4_finalize_icelake_(&state, lengths_u64x8));
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
