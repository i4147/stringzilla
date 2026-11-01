/**
 *  @brief AVX2 backend for hardware-accelerated memory operations on Haswell and newer x86 CPUs.
 *  @file include/stringzilla/memory/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_HASWELL_H_
#define STRINGZILLA_MEMORY_HASWELL_H_
#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif
SZ_API_COMPTIME void sz_fill_haswell(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    char value_char = *(char *)&value;
    __m256i value_u8x32 = _mm256_set1_epi8(value_char);
    if (length <= 32) sz_fill_serial(target, length, value);
    else {
        sz_size_t head_length = (32 - ((sz_size_t)target % 32)) % 32; 
        sz_size_t tail_length = (sz_size_t)(target + length) % 32;    
        sz_size_t body_length = length - head_length - tail_length;   
        sz_u16_t value16 = (sz_u16_t)value * 0x0101u;
        sz_u32_t value32 = (sz_u32_t)value16 * 0x00010001u;
        sz_u64_t value64 = (sz_u64_t)value32 * 0x0000000100000001ull;
        if (head_length & 1) *(sz_u8_t *)target = value, target++, head_length--;
        if (head_length & 2) *(sz_u16_t *)target = value16, target += 2, head_length -= 2;
        if (head_length & 4) *(sz_u32_t *)target = value32, target += 4, head_length -= 4;
        if (head_length & 8) *(sz_u64_t *)target = value64, target += 8, head_length -= 8;
        if (head_length & 16)
            _mm_store_si128((__m128i *)target, _mm_set1_epi8(value_char)), target += 16, head_length -= 16;
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");
        for (; body_length >= 32; target += 32, body_length -= 32) _mm256_store_si256((__m256i *)target, value_u8x32);
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");
        if (tail_length & 16)
            _mm_store_si128((__m128i *)target, _mm_set1_epi8(value_char)), target += 16, tail_length -= 16;
        if (tail_length & 8) *(sz_u64_t *)target = value64, target += 8, tail_length -= 8;
        if (tail_length & 4) *(sz_u32_t *)target = value32, target += 4, tail_length -= 4;
        if (tail_length & 2) *(sz_u16_t *)target = value16, target += 2, tail_length -= 2;
        if (tail_length & 1) *(sz_u8_t *)target = value, target++, tail_length--;
    }
}
SZ_API_COMPTIME void sz_copy_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    int is_huge = length > 1ull * 1024ull * 1024ull;
    if (length < 8) {
        while (length--) *(target++) = *(source++);
    }
    else if (length <= 16) {
        sz_u64_t source_first_word = sz_u64_load(source).u64;
        sz_u64_t source_second_word = sz_u64_load(source + length - 8).u64;
        sz_u64_store(target, source_first_word);
        sz_u64_store(target + length - 8, source_second_word);
    }
    else if (length <= 32) {
        sz_u128_vec_t source_first_vec, source_second_vec;
        source_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(source));
        source_second_vec.xmm = _mm_lddqu_si128((__m128i const *)(source + length - 16));
        _mm_storeu_si128((__m128i *)(target), source_first_vec.xmm);
        _mm_storeu_si128((__m128i *)(target + length - 16), source_second_vec.xmm);
    }
    else if (length <= 64) {
        sz_u256_vec_t source_first_vec, source_second_vec;
        source_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source));
        source_second_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source + length - 32));
        _mm256_storeu_si256((__m256i *)(target), source_first_vec.ymm);
        _mm256_storeu_si256((__m256i *)(target + length - 32), source_second_vec.ymm);
    }
    else if ((sz_size_t)target % 32 == 0 && (sz_size_t)source % 32 == 0 && !is_huge) {
        for (; length >= 32; target += 32, source += 32, length -= 32)
            _mm256_store_si256((__m256i *)target, _mm256_load_si256((__m256i const *)source));
        if (length) sz_copy_serial(target, source, length);
    }
    else {
        sz_size_t head_length = (32 - ((sz_size_t)target % 32)) % 32; 
        sz_size_t tail_length = (sz_size_t)(target + length) % 32;    
        sz_size_t body_length = length - head_length - tail_length;   
        if (head_length & 1) *(sz_u8_t *)target = *(sz_u8_t *)source, target++, source++, head_length--;
        if (head_length & 2) sz_u16_store(target, sz_u16_load(source).u16), target += 2, source += 2, head_length -= 2;
        if (head_length & 4) sz_u32_store(target, sz_u32_load(source).u32), target += 4, source += 4, head_length -= 4;
        if (head_length & 8) sz_u64_store(target, sz_u64_load(source).u64), target += 8, source += 8, head_length -= 8;
        if (head_length & 16)
            _mm_store_si128((__m128i *)target, _mm_lddqu_si128((__m128i const *)source)), target += 16, source += 16,
                head_length -= 16;
        sz_assert_(head_length == 0 && "The head length should be zero after the head copy.");
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");
        if (!is_huge) {
            for (; body_length >= 32; target += 32, source += 32, body_length -= 32)
                _mm256_store_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
        }
        else {
            sz_size_t tails_bytes_skipped = 0;
            for (; body_length >= 64; target += 32, source += 32, body_length -= 64, tails_bytes_skipped += 32) {
                _mm256_store_si256((__m256i *)(target), _mm256_lddqu_si256((__m256i const *)(source)));
                _mm256_store_si256((__m256i *)(target + body_length - 32),
                                   _mm256_lddqu_si256((__m256i const *)(source + body_length - 32)));
            }
            if (body_length) {
                sz_assert_(body_length == 32 && "The only remaining body length should be 32 bytes.");
                _mm256_store_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
                target += 32, source += 32, body_length -= 32;
            }
            target += tails_bytes_skipped;
            source += tails_bytes_skipped;
        }
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");
        if (tail_length & 16)
            _mm_store_si128((__m128i *)target, _mm_lddqu_si128((__m128i const *)source)), target += 16, source += 16,
                tail_length -= 16;
        if (tail_length & 8) sz_u64_store(target, sz_u64_load(source).u64), target += 8, source += 8, tail_length -= 8;
        if (tail_length & 4) sz_u32_store(target, sz_u32_load(source).u32), target += 4, source += 4, tail_length -= 4;
        if (tail_length & 2) sz_u16_store(target, sz_u16_load(source).u16), target += 2, source += 2, tail_length -= 2;
        if (tail_length & 1) *(sz_u8_t *)target = *(sz_u8_t *)source, target++, source++, tail_length--;
        sz_assert_(tail_length == 0 && "The tail length should be zero after the tail copy.");
    }
}
SZ_API_COMPTIME void sz_move_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (length < 8) {
        if (target < source)
            while (length--) *(target++) = *(source++);
        else {
            target += length, source += length;
            while (length--) *(--target) = *(--source);
        }
    }
    else if (length <= 16) {
        sz_u64_t source_first_word = sz_u64_load(source).u64;
        sz_u64_t source_second_word = sz_u64_load(source + length - 8).u64;
        sz_u64_store(target, source_first_word);
        sz_u64_store(target + length - 8, source_second_word);
    }
    else if (length <= 32) {
        sz_u128_vec_t source_first_vec, source_second_vec;
        source_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(source));
        source_second_vec.xmm = _mm_lddqu_si128((__m128i const *)(source + length - 16));
        _mm_storeu_si128((__m128i *)(target), source_first_vec.xmm);
        _mm_storeu_si128((__m128i *)(target + length - 16), source_second_vec.xmm);
    }
    else if (length <= 64) {
        sz_u256_vec_t source_first_vec, source_second_vec;
        source_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source));
        source_second_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source + length - 32));
        _mm256_storeu_si256((__m256i *)(target), source_first_vec.ymm);
        _mm256_storeu_si256((__m256i *)(target + length - 32), source_second_vec.ymm);
    }
    else if (target < source || target >= source + length) {
        for (; length >= 32; target += 32, source += 32, length -= 32)
            _mm256_storeu_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
        while (length--) *(target++) = *(source++);
    }
    else {
        for (target += length, source += length; length >= 32; length -= 32)
            _mm256_storeu_si256((__m256i *)(target -= 32), _mm256_lddqu_si256((__m256i const *)(source -= 32)));
        while (length--) *(--target) = *(--source);
    }
}
SZ_API_COMPTIME void sz_lookup_haswell(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                       char const lut[sz_at_least_(256)]) {
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }
    sz_u256_vec_t lut_0_to_15_vec, lut_16_to_31_vec, lut_32_to_47_vec, lut_48_to_63_vec, 
        lut_64_to_79_vec, lut_80_to_95_vec, lut_96_to_111_vec, lut_112_to_127_vec,       
        lut_128_to_143_vec, lut_144_to_159_vec, lut_160_to_175_vec, lut_176_to_191_vec,  
        lut_192_to_207_vec, lut_208_to_223_vec, lut_224_to_239_vec, lut_240_to_255_vec;
    lut_0_to_15_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut)));
    lut_16_to_31_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 16)));
    lut_32_to_47_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 32)));
    lut_48_to_63_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 48)));
    lut_64_to_79_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 64)));
    lut_80_to_95_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 80)));
    lut_96_to_111_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 96)));
    lut_112_to_127_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 112)));
    lut_128_to_143_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 128)));
    lut_144_to_159_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 144)));
    lut_160_to_175_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 160)));
    lut_176_to_191_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 176)));
    lut_192_to_207_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 192)));
    lut_208_to_223_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 208)));
    lut_224_to_239_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 224)));
    lut_240_to_255_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 240)));
    sz_u256_vec_t not_first_bit_vec, not_second_bit_vec, not_third_bit_vec, not_fourth_bit_vec;
    sz_u256_vec_t source_vec, source_low_nibble_vec;
    sz_u256_vec_t blended_0_to_31_vec, blended_32_to_63_vec, blended_64_to_95_vec, blended_96_to_127_vec,
        blended_128_to_159_vec, blended_160_to_191_vec, blended_192_to_223_vec, blended_224_to_255_vec;
    while (length >= 32) {
        source_vec.ymm = _mm256_lddqu_si256((__m256i const *)source);
        source_low_nibble_vec.ymm = _mm256_and_si256(source_vec.ymm, _mm256_set1_epi8((char)0x0F));
        not_fourth_bit_vec.ymm = _mm256_cmpeq_epi8( 
            _mm256_and_si256(_mm256_set1_epi8((char)0x10), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8(                             
            _mm256_shuffle_epi8(lut_16_to_31_vec.ymm, source_low_nibble_vec.ymm), 
            _mm256_shuffle_epi8(lut_0_to_15_vec.ymm, source_low_nibble_vec.ymm),  
            not_fourth_bit_vec.ymm);
        blended_32_to_63_vec.ymm = _mm256_blendv_epi8(                            
            _mm256_shuffle_epi8(lut_48_to_63_vec.ymm, source_low_nibble_vec.ymm), 
            _mm256_shuffle_epi8(lut_32_to_47_vec.ymm, source_low_nibble_vec.ymm), 
            not_fourth_bit_vec.ymm);
        blended_64_to_95_vec.ymm = _mm256_blendv_epi8(                            
            _mm256_shuffle_epi8(lut_80_to_95_vec.ymm, source_low_nibble_vec.ymm), 
            _mm256_shuffle_epi8(lut_64_to_79_vec.ymm, source_low_nibble_vec.ymm), 
            not_fourth_bit_vec.ymm);
        blended_96_to_127_vec.ymm = _mm256_blendv_epi8(                             
            _mm256_shuffle_epi8(lut_112_to_127_vec.ymm, source_low_nibble_vec.ymm), 
            _mm256_shuffle_epi8(lut_96_to_111_vec.ymm, source_low_nibble_vec.ymm),  
            not_fourth_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8(                            
            _mm256_shuffle_epi8(lut_144_to_159_vec.ymm, source_low_nibble_vec.ymm), 
            _mm256_shuffle_epi8(lut_128_to_143_vec.ymm, source_low_nibble_vec.ymm), 
            not_fourth_bit_vec.ymm);
        blended_160_to_191_vec.ymm = _mm256_blendv_epi8(                            
            _mm256_shuffle_epi8(lut_176_to_191_vec.ymm, source_low_nibble_vec.ymm), 
            _mm256_shuffle_epi8(lut_160_to_175_vec.ymm, source_low_nibble_vec.ymm), 
            not_fourth_bit_vec.ymm);
        blended_192_to_223_vec.ymm = _mm256_blendv_epi8(                            
            _mm256_shuffle_epi8(lut_208_to_223_vec.ymm, source_low_nibble_vec.ymm), 
            _mm256_shuffle_epi8(lut_192_to_207_vec.ymm, source_low_nibble_vec.ymm), 
            not_fourth_bit_vec.ymm);
        blended_224_to_255_vec.ymm = _mm256_blendv_epi8(                            
            _mm256_shuffle_epi8(lut_240_to_255_vec.ymm, source_low_nibble_vec.ymm), 
            _mm256_shuffle_epi8(lut_224_to_239_vec.ymm, source_low_nibble_vec.ymm), 
            not_fourth_bit_vec.ymm);
        not_third_bit_vec.ymm = _mm256_cmpeq_epi8( 
            _mm256_and_si256(_mm256_set1_epi8((char)0x20), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( 
            blended_32_to_63_vec.ymm,                 
            blended_0_to_31_vec.ymm,                  
            not_third_bit_vec.ymm);
        blended_64_to_95_vec.ymm = _mm256_blendv_epi8( 
            blended_96_to_127_vec.ymm,                 
            blended_64_to_95_vec.ymm,                  
            not_third_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8( 
            blended_160_to_191_vec.ymm,                  
            blended_128_to_159_vec.ymm,                  
            not_third_bit_vec.ymm);
        blended_192_to_223_vec.ymm = _mm256_blendv_epi8( 
            blended_224_to_255_vec.ymm,                  
            blended_192_to_223_vec.ymm,                  
            not_third_bit_vec.ymm);
        not_second_bit_vec.ymm = _mm256_cmpeq_epi8( 
            _mm256_and_si256(_mm256_set1_epi8((char)0x40), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( 
            blended_64_to_95_vec.ymm,                 
            blended_0_to_31_vec.ymm,                  
            not_second_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8( 
            blended_192_to_223_vec.ymm,                  
            blended_128_to_159_vec.ymm,                  
            not_second_bit_vec.ymm);
        not_first_bit_vec.ymm = _mm256_cmpeq_epi8( 
            _mm256_and_si256(_mm256_set1_epi8((char)0x80), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( 
            blended_128_to_159_vec.ymm,               
            blended_0_to_31_vec.ymm,                  
            not_first_bit_vec.ymm);
        _mm256_storeu_si256((__m256i *)target, blended_0_to_31_vec.ymm);
        source += 32, target += 32, length -= 32;
    }
    if (length) sz_lookup_serial(target, length, source, lut);
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
