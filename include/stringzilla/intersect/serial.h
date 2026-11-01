/**
 *  @brief Serial backend for set intersection.
 *  @file include/stringzilla/intersect/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/intersect.h
 */
#ifndef STRINGZILLA_INTERSECT_SERIAL_H_
#define STRINGZILLA_INTERSECT_SERIAL_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#include "stringzilla/memory.h"  
#include "stringzilla/hash.h"    
#ifdef __cplusplus
extern "C" {
#endif
SZ_API_COMPTIME sz_status_t sz_sequence_intersect_serial(                           
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
    sz_size_t *const table_positions = (sz_size_t *)alloc->allocate(hash_table_slots * bytes_per_entry, alloc);
    if (!table_positions) return sz_bad_alloc_k;
    sz_u64_t *const table_hashes = (sz_u64_t *)(table_positions + hash_table_slots);
    sz_fill((sz_ptr_t)table_positions, hash_table_slots * bytes_per_entry, 0xFF);
    sz_u64_t const empty_slot = ~(sz_u64_t)0;
    sz_size_t const consumed_flag = (sz_size_t)1 << (sizeof(sz_size_t) * 8 - 1);
    for (sz_size_t small_position = 0; small_position < small_sequence->count; ++small_position) {
        sz_cptr_t const str = small_sequence->get_start(small_sequence->handle, small_position);
        sz_size_t const length = small_sequence->get_length(small_sequence->handle, small_position);
        sz_u64_t const hash = sz_hash(str, length, seed);
        sz_size_t hash_slot = hash & (hash_table_slots - 1);
        while (table_hashes[hash_slot] != empty_slot) hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
        table_hashes[hash_slot] = hash;
        table_positions[hash_slot] = small_position;
    }
    sz_size_t intersection_count = 0;
    for (sz_size_t large_position = 0; large_position < large_sequence->count; ++large_position) {
        sz_cptr_t const str = large_sequence->get_start(large_sequence->handle, large_position);
        sz_size_t const length = large_sequence->get_length(large_sequence->handle, large_position);
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
    alloc->free(table_positions, hash_table_slots * bytes_per_entry, alloc);
    *intersection_count_ptr = intersection_count;
    return sz_success_k;
}
#ifdef __cplusplus
}
#endif
#endif 
