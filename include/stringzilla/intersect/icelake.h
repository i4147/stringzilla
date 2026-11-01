/**
 *  @brief Ice Lake (AVX-512 VBMI+VAES) backend for set intersection.
 *  @file include/stringzilla/intersect/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/intersect.h
 */
#ifndef STRINGZILLA_INTERSECT_ICELAKE_H_
#define STRINGZILLA_INTERSECT_ICELAKE_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#include "stringzilla/memory.h"  
#include "stringzilla/hash.h"    
#include "stringzilla/intersect/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                                   \
    __attribute__((                                                                                             \
        target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vnni,bmi,bmi2,aes,vaes,sha,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                      \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vnni,bmi,bmi2,aes,vaes,sha"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vnni", "bmi", "bmi2", \
                   "aes", "vaes", "sha")
#endif
/**
 *  @brief Checks whether any two of the four 64-bit integers in a 256-bit vector are equal (i.e., have collisions).
 *      Used to detect slot collisions before attempting a vectorized scatter into the hash table.
 *
 *  @param values_u64x4 A 256-bit vector holding four 64-bit values [a, b, c, d].
 *  @return Non-zero if at least two of the four values are identical, zero otherwise.
 */
SZ_HELPER_AUTO int sz_u64x4_contains_collisions_haswell_(__m256i values_u64x4) {
    __m256i cmp1_u64x4 = _mm256_cmpeq_epi64(values_u64x4, _mm256_permute4x64_epi64(values_u64x4, 0xB1));
    __m256i cmp2_u64x4 = _mm256_cmpeq_epi64(values_u64x4, _mm256_permute4x64_epi64(values_u64x4, 0x4E));
    __m256i cmp3_u64x4 = _mm256_cmpeq_epi64(values_u64x4, _mm256_permute4x64_epi64(values_u64x4, 0x1B));
    __m256i cmp_u64x4 = _mm256_or_si256(_mm256_or_si256(cmp1_u64x4, cmp2_u64x4), cmp3_u64x4);
    int matches_mask = _mm256_movemask_epi8(cmp_u64x4);
    return matches_mask;
}
SZ_API_COMPTIME sz_status_t sz_sequence_intersect_icelake(                          
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence,      
    sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_count_ptr, 
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions) {
    sz_sequence_t const *small_sequence, *large_sequence;
    sz_sorted_idx_t *small_positions, *large_positions;
    if (first_sequence->count <= second_sequence->count) {
        small_sequence = first_sequence, large_sequence = second_sequence;
        small_positions = first_positions, large_positions = second_positions;
    }
    else {
        small_sequence = second_sequence, large_sequence = first_sequence;
        small_positions = second_positions, large_positions = first_positions;
    }
    if (small_sequence->count == 0) {
        *intersection_count_ptr = 0;
        return sz_success_k;
    }
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }
    sz_size_t const hash_table_slots = sz_size_bit_ceil(small_sequence->count) * (1u << SZ_SEQUENCE_INTERSECT_BUDGET);
    sz_size_t const bytes_per_entry = sizeof(sz_size_t) + sizeof(sz_u64_t);
    sz_size_t *table_positions = (sz_size_t *)alloc->allocate(hash_table_slots * bytes_per_entry, alloc);
    if (!table_positions) return sz_bad_alloc_k;
    sz_u64_t *table_hashes = (sz_u64_t *)(table_positions + hash_table_slots);
    sz_fill((sz_ptr_t)table_positions, hash_table_slots * bytes_per_entry, 0xFF);
    sz_u64_t const empty_slot = ~(sz_u64_t)0;
    sz_size_t const consumed_flag = (sz_size_t)1 << (sizeof(sz_size_t) * 8 - 1);
    sz_hash_state_aligned_for_short_x4_t batch_hashes_states_initial;
    sz_hash_state_short_x4_init_icelake_(&batch_hashes_states_initial, seed);
    sz_size_t count_longer = 0;
    for (sz_size_t small_position = 0; small_position < small_sequence->count;) {
        sz_string_view_t batch[4];
        sz_u256_vec_t batch_positions_vec;
        sz_size_t batch_size;
        for (batch_size = 0; batch_size < 4 && small_position < small_sequence->count; ++small_position) {
            sz_size_t length = small_sequence->get_length(small_sequence->handle, small_position);
            if (length > 16) {
                count_longer++;
                continue;
            }
            sz_cptr_t str = small_sequence->get_start(small_sequence->handle, small_position);
            batch[batch_size].start = str;
            batch[batch_size].length = length;
            batch_positions_vec.u64s[batch_size] = small_position;
            ++batch_size;
        }
        if (batch_size != 4) {
            for (sz_size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
                sz_cptr_t const str = batch[batch_index].start;
                sz_size_t const length = batch[batch_index].length;
                sz_u64_t const hash = sz_hash(str, length, seed);
                sz_size_t hash_slot = hash & (hash_table_slots - 1);
                while (table_hashes[hash_slot] != empty_slot) hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
                table_hashes[hash_slot] = hash;
                table_positions[hash_slot] = batch_positions_vec.u64s[batch_index];
            }
        }
        else {
            sz_u256_vec_t batch_hashes_vec;
            sz_u512_vec_t batch_prefixes_vec;
            batch_prefixes_vec.zmm = _mm512_castsi128_si512(
                _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[0].length), batch[0].start));
            batch_prefixes_vec.zmm = _mm512_inserti64x2(
                batch_prefixes_vec.zmm, _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[1].length), batch[1].start), 1);
            batch_prefixes_vec.zmm = _mm512_inserti64x2(
                batch_prefixes_vec.zmm, _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[2].length), batch[2].start), 2);
            batch_prefixes_vec.zmm = _mm512_inserti64x2(
                batch_prefixes_vec.zmm, _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[3].length), batch[3].start), 3);
            sz_hash_state_aligned_for_short_x4_t batch_hashes_states = batch_hashes_states_initial;
            sz_hash_state_short_x4_update_icelake_(&batch_hashes_states, batch_prefixes_vec.zmm);
            batch_hashes_vec.ymm = sz_hash_state_short_x4_finalize_icelake_(
                &batch_hashes_states, batch[0].length, batch[1].length, batch[2].length, batch[3].length);
            sz_assert_(batch_hashes_vec.u64s[0] == sz_hash(batch[0].start, batch[0].length, seed));
            sz_assert_(batch_hashes_vec.u64s[1] == sz_hash(batch[1].start, batch[1].length, seed));
            sz_assert_(batch_hashes_vec.u64s[2] == sz_hash(batch[2].start, batch[2].length, seed));
            sz_assert_(batch_hashes_vec.u64s[3] == sz_hash(batch[3].start, batch[3].length, seed));
            sz_u256_vec_t batch_slots_vec, existing_hashes_vec;
            batch_slots_vec.ymm = _mm256_and_si256(batch_hashes_vec.ymm, _mm256_set1_epi64x(hash_table_slots - 1));
            int const has_slot_collisions = sz_u64x4_contains_collisions_haswell_(batch_slots_vec.ymm);
            existing_hashes_vec.ymm = _mm256_mmask_i64gather_epi64(_mm256_setzero_si256(), 0xFF, batch_slots_vec.ymm,
                                                                   table_hashes, 8);
            int const all_empty = _mm256_testc_si256(existing_hashes_vec.ymm, _mm256_set1_epi64x(-1));
            if (all_empty && !has_slot_collisions) {
                _mm256_mask_i64scatter_epi64(table_hashes, 0xFF, batch_slots_vec.ymm, batch_hashes_vec.ymm, 8);
                _mm256_mask_i64scatter_epi64(table_positions, 0xFF, batch_slots_vec.ymm, batch_positions_vec.ymm, 8);
            }
            else {
                for (sz_size_t batch_index = 0; batch_index < 4; ++batch_index) {
                    sz_size_t hash_slot = batch_slots_vec.u64s[batch_index] & (hash_table_slots - 1);
                    while (table_hashes[hash_slot] != empty_slot) hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
                    table_hashes[hash_slot] = batch_hashes_vec.u64s[batch_index];
                    table_positions[hash_slot] = batch_positions_vec.u64s[batch_index];
                }
            }
        }
    }
    sz_size_t intersection_count = 0;
    for (sz_size_t large_position = 0; large_position < large_sequence->count;) {
        sz_string_view_t batch[4];
        sz_u256_vec_t batch_positions_vec;
        sz_size_t batch_size;
        for (batch_size = 0; batch_size < 4 && large_position < large_sequence->count; ++large_position) {
            sz_size_t length = large_sequence->get_length(large_sequence->handle, large_position);
            if (length > 16) {
                count_longer++;
                continue;
            }
            sz_cptr_t str = large_sequence->get_start(large_sequence->handle, large_position);
            batch[batch_size].start = str;
            batch[batch_size].length = length;
            batch_positions_vec.u64s[batch_size] = large_position;
            ++batch_size;
        }
        if (batch_size != 4) {
            for (sz_size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
                sz_cptr_t const str = batch[batch_index].start;
                sz_size_t const length = batch[batch_index].length;
                sz_u64_t const hash = sz_hash(str, length, seed);
                sz_size_t hash_slot = hash & (hash_table_slots - 1);
                for (; table_hashes[hash_slot] != empty_slot; hash_slot = (hash_slot + 1) & (hash_table_slots - 1)) {
                    sz_u64_t small_hash = table_hashes[hash_slot];
                    if (small_hash != hash) continue;
                    sz_size_t const stored = table_positions[hash_slot];
                    sz_size_t const small_position = stored & ~consumed_flag;
                    sz_size_t const small_length = small_sequence->get_length(small_sequence->handle, small_position);
                    if (length != small_length) continue;
                    sz_cptr_t const small_str = small_sequence->get_start(small_sequence->handle, small_position);
                    sz_bool_t const same = sz_equal(str, small_str, length);
                    if (same != sz_true_k) continue;
                    if (stored & consumed_flag) break;
                    small_positions[intersection_count] = small_position;
                    large_positions[intersection_count] = batch_positions_vec.u64s[batch_index];
                    ++intersection_count;
                    table_positions[hash_slot] = small_position | consumed_flag;
                    break;
                }
            }
        }
        else {
            sz_u256_vec_t batch_hashes_vec;
            sz_u512_vec_t batch_prefixes_vec;
            batch_prefixes_vec.zmm = _mm512_castsi128_si512(
                _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[0].length), batch[0].start));
            batch_prefixes_vec.zmm = _mm512_inserti64x2(
                batch_prefixes_vec.zmm, _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[1].length), batch[1].start), 1);
            batch_prefixes_vec.zmm = _mm512_inserti64x2(
                batch_prefixes_vec.zmm, _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[2].length), batch[2].start), 2);
            batch_prefixes_vec.zmm = _mm512_inserti64x2(
                batch_prefixes_vec.zmm, _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[3].length), batch[3].start), 3);
            sz_hash_state_aligned_for_short_x4_t batch_hashes_states = batch_hashes_states_initial;
            sz_hash_state_short_x4_update_icelake_(&batch_hashes_states, batch_prefixes_vec.zmm);
            batch_hashes_vec.ymm = sz_hash_state_short_x4_finalize_icelake_(
                &batch_hashes_states, batch[0].length, batch[1].length, batch[2].length, batch[3].length);
            sz_assert_(batch_hashes_vec.u64s[0] == sz_hash(batch[0].start, batch[0].length, seed));
            sz_assert_(batch_hashes_vec.u64s[1] == sz_hash(batch[1].start, batch[1].length, seed));
            sz_assert_(batch_hashes_vec.u64s[2] == sz_hash(batch[2].start, batch[2].length, seed));
            sz_assert_(batch_hashes_vec.u64s[3] == sz_hash(batch[3].start, batch[3].length, seed));
            sz_u256_vec_t batch_slots_vec, existing_hashes_vec;
            batch_slots_vec.ymm = _mm256_and_si256(batch_hashes_vec.ymm, _mm256_set1_epi64x(hash_table_slots - 1));
            existing_hashes_vec.ymm = _mm256_mmask_i64gather_epi64(_mm256_setzero_si256(), 0xFF, batch_slots_vec.ymm,
                                                                   table_hashes, 8);
            int const same_hashes = _mm256_movemask_epi8(
                _mm256_cmpeq_epi64(existing_hashes_vec.ymm, batch_hashes_vec.ymm));
            int const nulled_hashes = _mm256_movemask_epi8(
                _mm256_cmpeq_epi64(existing_hashes_vec.ymm, _mm256_set1_epi64x(-1)));
            for (sz_size_t batch_index = 0; batch_index < 4; ++batch_index) {
                sz_cptr_t const str = batch[batch_index].start;
                sz_size_t const length = batch[batch_index].length;
                sz_u64_t const hash = batch_hashes_vec.u64s[batch_index];
                int const same_hash = (same_hashes >> (8 * batch_index)) & 0xFF;
                int const nulled_hash = (nulled_hashes >> (8 * batch_index)) & 0xFF;
                if (nulled_hash) continue;
                sz_size_t hash_slot = batch_slots_vec.u64s[batch_index];
                if (same_hash) {
                    sz_size_t const stored = table_positions[hash_slot];
                    sz_size_t const small_position = stored & ~consumed_flag;
                    sz_size_t const small_length = small_sequence->get_length(small_sequence->handle, small_position);
                    if (length == small_length) {
                        sz_cptr_t const small_str = small_sequence->get_start(small_sequence->handle, small_position);
                        sz_bool_t const same = sz_equal(str, small_str, length);
                        if (same == sz_true_k) {
                            if (stored & consumed_flag) continue;
                            small_positions[intersection_count] = small_position;
                            large_positions[intersection_count] = batch_positions_vec.u64s[batch_index];
                            ++intersection_count;
                            table_positions[hash_slot] = small_position | consumed_flag;
                            continue;
                        }
                    }
                    hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
                }
                for (; table_hashes[hash_slot] != empty_slot; hash_slot = (hash_slot + 1) & (hash_table_slots - 1)) {
                    sz_u64_t small_hash = table_hashes[hash_slot];
                    if (small_hash != hash) continue;
                    sz_size_t const stored = table_positions[hash_slot];
                    sz_size_t const small_position = stored & ~consumed_flag;
                    sz_size_t const small_length = small_sequence->get_length(small_sequence->handle, small_position);
                    if (length != small_length) continue;
                    sz_cptr_t const small_str = small_sequence->get_start(small_sequence->handle, small_position);
                    sz_bool_t const same = sz_equal(str, small_str, length);
                    if (same != sz_true_k) continue;
                    if (stored & consumed_flag) break;
                    small_positions[intersection_count] = small_position;
                    large_positions[intersection_count] = batch_positions_vec.u64s[batch_index];
                    ++intersection_count;
                    table_positions[hash_slot] = small_position | consumed_flag;
                    break;
                }
            }
        }
    }
    if (count_longer) {
        sz_fill((sz_ptr_t)table_positions, hash_table_slots * bytes_per_entry, 0xFF);
        for (sz_size_t small_position = 0; small_position < small_sequence->count; ++small_position) {
            sz_size_t const length = small_sequence->get_length(small_sequence->handle, small_position);
            if (length <= 16) continue; 
            sz_cptr_t const str = small_sequence->get_start(small_sequence->handle, small_position);
            sz_u64_t const hash = sz_hash(str, length, seed);
            sz_size_t hash_slot = hash & (hash_table_slots - 1);
            while (table_hashes[hash_slot] != empty_slot) hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
            table_hashes[hash_slot] = hash;
            table_positions[hash_slot] = small_position;
        }
        for (sz_size_t large_position = 0; large_position < large_sequence->count; ++large_position) {
            sz_size_t const length = large_sequence->get_length(large_sequence->handle, large_position);
            if (length <= 16) continue; 
            sz_cptr_t const str = large_sequence->get_start(large_sequence->handle, large_position);
            sz_u64_t const hash = sz_hash(str, length, seed);
            sz_size_t hash_slot = hash & (hash_table_slots - 1);
            for (; table_hashes[hash_slot] != empty_slot; hash_slot = (hash_slot + 1) & (hash_table_slots - 1)) {
                sz_u64_t small_hash = table_hashes[hash_slot];
                if (small_hash != hash) continue;
                sz_size_t const stored = table_positions[hash_slot];
                sz_size_t const small_position = stored & ~consumed_flag;
                sz_size_t const small_length = small_sequence->get_length(small_sequence->handle, small_position);
                if (length != small_length) continue;
                sz_cptr_t const small_str = small_sequence->get_start(small_sequence->handle, small_position);
                sz_bool_t const same = sz_equal(str, small_str, length);
                if (same != sz_true_k) continue;
                if (stored & consumed_flag) break;
                small_positions[intersection_count] = small_position;
                large_positions[intersection_count] = large_position;
                ++intersection_count;
                table_positions[hash_slot] = small_position | consumed_flag;
                break;
            }
        }
    }
    alloc->free(table_positions, hash_table_slots * bytes_per_entry, alloc);
    *intersection_count_ptr = intersection_count;
    return sz_success_k;
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
