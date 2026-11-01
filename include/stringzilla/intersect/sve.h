/**
 *  @brief SVE backend for set intersection.
 *  @file include/stringzilla/intersect/sve.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/intersect.h
 */
#ifndef STRINGZILLA_INTERSECT_SVE_H_
#define STRINGZILLA_INTERSECT_SVE_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#include "stringzilla/memory.h"  
#include "stringzilla/hash.h"    
#include "stringzilla/intersect/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif
SZ_API_COMPTIME sz_status_t sz_sequence_intersect_sve(sz_sequence_t const *first_sequence,
                                                      sz_sequence_t const *second_sequence, 
                                                      sz_memory_allocator_t *alloc, sz_u64_t seed,
                                                      sz_size_t *intersection_count_ptr,
                                                      sz_sorted_idx_t *first_positions,
                                                      sz_sorted_idx_t *second_positions) {
    return sz_sequence_intersect_serial(     
        first_sequence, second_sequence,     
        alloc, seed, intersection_count_ptr, 
        first_positions, second_positions);
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
