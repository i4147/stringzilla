/**
 *  @brief Serial backend for sorting string collections.
 *  @file include/stringzilla/sort/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/sort.h
 */
#ifndef STRINGZILLA_SORT_SERIAL_H_
#define STRINGZILLA_SORT_SERIAL_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h"                  
#include "stringzilla/memory.h"                   
#include "stringzilla/utf8_uncased_fold/serial.h" 
#ifdef __cplusplus
extern "C" {
#endif
#pragma region Generic Public Helpers
/**
 *  @brief Quadratic complexity @b stable insertion sort adjust for our @b argsort usecase.
 *         Needs no extra memory and is used as a fallback for small inputs.
 */
SZ_API_COMPTIME void sz_sequence_argsort_with_insertion(sz_sequence_t const *sequence, sz_sorted_idx_t *order) {
    for (sz_size_t element_index = 1; element_index < sequence->count; ++element_index) {
        sz_sorted_idx_t current_idx = order[element_index];
        sz_size_t position_index = element_index;
        while (position_index > 0) {
            sz_sorted_idx_t previous_idx = order[position_index - 1];
            sz_cptr_t previous_start = sequence->get_start(sequence->handle, previous_idx);
            sz_cptr_t current_start = sequence->get_start(sequence->handle, current_idx);
            sz_size_t previous_length = sequence->get_length(sequence->handle, previous_idx);
            sz_size_t current_length = sequence->get_length(sequence->handle, current_idx);
            sz_ordering_t ordering = sz_order(previous_start, previous_length, current_start, current_length);
            if (ordering != sz_greater_k) break;
            order[position_index] = order[position_index - 1];
            --position_index;
        }
        order[position_index] = current_idx;
    }
}
/**
 *  @brief Quadratic complexity @b stable insertion sort adjust for our @b pgram-sorting usecase.
 *         Needs no extra memory and is used as a fallback for small inputs.
 */
SZ_API_COMPTIME void sz_pgrams_sort_with_insertion(sz_pgram_t *pgrams, sz_size_t count, sz_sorted_idx_t *order) {
    for (sz_size_t element_index = 1; element_index < count; ++element_index) {
        sz_pgram_t current_key = pgrams[element_index];
        sz_sorted_idx_t current_idx = order[element_index];
        sz_size_t position_index = element_index;
        while (position_index > 0 && pgrams[position_index - 1] > current_key) {
            pgrams[position_index] = pgrams[position_index - 1];
            order[position_index] = order[position_index - 1];
            --position_index;
        }
        pgrams[position_index] = current_key;
        order[position_index] = current_idx;
    }
#if SZ_DEBUG
    for (sz_size_t element_index = 1; element_index < count; ++element_index)
        sz_assert_(pgrams[element_index - 1] <= pgrams[element_index] &&
                   "The pgrams should be sorted in ascending order.");
#endif
}
#pragma endregion // Generic Public Helpers
#pragma region Generic Internal Helpers
/**
 *  @brief Convenience macro for of conditional swap of "pgrams" and their indices for a sorting network.
 *  @see https://en.wikipedia.org/wiki/Sorting_network
 */
#define sz_sequence_sorting_network_conditional_swap_(i, j)    \
    do {                                                       \
        if (pgrams[i] > pgrams[j]) {                           \
            sz_swap_(sz_pgram_t, pgrams[i], pgrams[j]);        \
            sz_swap_(sz_sorted_idx_t, offsets[i], offsets[j]); \
        }                                                      \
    } while (0)
/**
 *  @brief Sorting network for 2 elements is just a single compare-swap.
 */
SZ_HELPER_INLINE void sz_sequence_sorting_network_2x_(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {
    sz_sequence_sorting_network_conditional_swap_(0, 1);
}
/**
 *  @brief Sorting network for 3 elements.
 *
 *  The network uses 3 compare-swap operations:
 *
 *      Stage 1: (0, 1)
 *      Stage 2: (0, 2)
 *      Stage 3: (1, 2)
 */
SZ_HELPER_INLINE void sz_sequence_sorting_network_3x_(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {
    sz_sequence_sorting_network_conditional_swap_(0, 1);
    sz_sequence_sorting_network_conditional_swap_(0, 2);
    sz_sequence_sorting_network_conditional_swap_(1, 2);
#if SZ_DEBUG
    for (sz_size_t element_index = 1; element_index < 3; ++element_index)
        sz_assert_(pgrams[element_index - 1] <= pgrams[element_index] && "Sorting network for 3 elements failed.");
#endif
}
/**
 *  @brief Sorting network for 4 elements.
 *
 *  The network uses 5 compare-swap operations:
 *
 *      Stage 1: (0, 1) and (2, 3)
 *      Stage 2: (0, 2)
 *      Stage 3: (1, 3)
 *      Stage 4: (1, 2)
 */
SZ_HELPER_AUTO void sz_sequence_sorting_network_4x_(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {
    sz_sequence_sorting_network_conditional_swap_(0, 1);
    sz_sequence_sorting_network_conditional_swap_(2, 3);
    sz_sequence_sorting_network_conditional_swap_(0, 2);
    sz_sequence_sorting_network_conditional_swap_(1, 3);
    sz_sequence_sorting_network_conditional_swap_(1, 2);
#if SZ_DEBUG
    for (sz_size_t element_index = 1; element_index < 4; ++element_index)
        sz_assert_(pgrams[element_index - 1] <= pgrams[element_index] && "Sorting network for 4 elements failed.");
#endif
}
/**
 *  @brief A scalar sorting network for 8 elements that reorders both the pgrams
 *         and their corresponding offsets in only 19 comparisons, the most efficient
 *         variant currently known.
 *
 *  The network consists of 6 stages with the following compare-swap pairs:
 *
 *      Stage 1: (0,1), (2,3), (4,5), (6,7)
 *      Stage 2: (0,2), (1,3), (4,6), (5,7)
 *      Stage 3: (1,2), (5,6)
 *      Stage 4: (0,4), (1,5), (2,6), (3,7)
 *      Stage 5: (2,4), (3,5)
 *      Stage 6: (1,2), (3,4), (5,6)
 */
SZ_HELPER_AUTO void sz_sequence_sorting_network_8x_(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {
    sz_sequence_sorting_network_conditional_swap_(0, 1);
    sz_sequence_sorting_network_conditional_swap_(2, 3);
    sz_sequence_sorting_network_conditional_swap_(4, 5);
    sz_sequence_sorting_network_conditional_swap_(6, 7);
    sz_sequence_sorting_network_conditional_swap_(0, 2);
    sz_sequence_sorting_network_conditional_swap_(1, 3);
    sz_sequence_sorting_network_conditional_swap_(4, 6);
    sz_sequence_sorting_network_conditional_swap_(5, 7);
    sz_sequence_sorting_network_conditional_swap_(1, 2);
    sz_sequence_sorting_network_conditional_swap_(5, 6);
    sz_sequence_sorting_network_conditional_swap_(0, 4);
    sz_sequence_sorting_network_conditional_swap_(1, 5);
    sz_sequence_sorting_network_conditional_swap_(2, 6);
    sz_sequence_sorting_network_conditional_swap_(3, 7);
    sz_sequence_sorting_network_conditional_swap_(2, 4);
    sz_sequence_sorting_network_conditional_swap_(3, 5);
    sz_sequence_sorting_network_conditional_swap_(1, 2);
    sz_sequence_sorting_network_conditional_swap_(3, 4);
    sz_sequence_sorting_network_conditional_swap_(5, 6);
#if SZ_DEBUG
    for (sz_size_t element_index = 1; element_index < 8; ++element_index)
        sz_assert_(pgrams[element_index - 1] <= pgrams[element_index] &&
                   "The sorting network must sort the pgrams in ascending order.");
#endif
}
#undef sz_sequence_sorting_network_conditional_swap_
/**
 *  @brief Stable in-place ascending sort of an @p order slice by original index. Restores stable order
 *      within a terminal run of byte-identical (or fold-identical) strings, whose pgrams are all equal
 *      and carry no further ordering information.
 *
 *  Uses an iterative QuickSort with a median-of-three pivot and an insertion-sort base case. A plain
 *  insertion sort would be quadratic on large runs of one repeated string (e.g. a very common word),
 *  so the indices - which are distinct integers - get the same log-linear treatment as the pgrams.
 */
SZ_HELPER_AUTO void sz_order_indices_ascending_(sz_sorted_idx_t *order, sz_size_t count) {
    sz_size_t stack[2 * 64];
    sz_size_t stack_size = 0;
    sz_size_t range_start = 0, range_end = count;
    for (;;) {
        if (range_end - range_start < 32) {
            for (sz_size_t element_index = range_start + 1; element_index < range_end; ++element_index) {
                sz_sorted_idx_t const current_idx = order[element_index];
                sz_size_t position_index = element_index;
                while (position_index > range_start && order[position_index - 1] > current_idx) {
                    order[position_index] = order[position_index - 1];
                    --position_index;
                }
                order[position_index] = current_idx;
            }
            if (!stack_size) break;
            range_end = stack[--stack_size], range_start = stack[--stack_size];
            continue;
        }
        sz_size_t const middle = range_start + (range_end - range_start) / 2;
        sz_size_t const last = range_end - 1;
        if (order[middle] < order[range_start]) sz_swap_(sz_sorted_idx_t, order[middle], order[range_start]);
        if (order[last] < order[range_start]) sz_swap_(sz_sorted_idx_t, order[last], order[range_start]);
        if (order[last] < order[middle]) sz_swap_(sz_sorted_idx_t, order[last], order[middle]);
        sz_swap_(sz_sorted_idx_t, order[middle], order[last]);
        sz_sorted_idx_t const pivot = order[last];
        sz_size_t store = range_start;
        for (sz_size_t scan = range_start; scan < last; ++scan)
            if (order[scan] < pivot) {
                sz_swap_(sz_sorted_idx_t, order[scan], order[store]);
                ++store;
            }
        sz_swap_(sz_sorted_idx_t, order[store], order[last]);
        sz_size_t const left_start = range_start, left_end = store;
        sz_size_t const right_start = store + 1, right_end = range_end;
        if (left_end - left_start > right_end - right_start) {
            stack[stack_size++] = left_start, stack[stack_size++] = left_end;
            range_start = right_start, range_end = right_end;
        }
        else {
            stack[stack_size++] = right_start, stack[stack_size++] = right_end;
            range_start = left_start, range_end = left_end;
        }
    }
}
#pragma endregion // Generic Internal Helpers
/**
 *  @brief Exports the next N-gram (pgram) slice for each string in the given range, storing the results
 *      as byte-reversed integers so that simple integer comparisons yield lexicographic ordering.
 *
 *  @param sequence The sequence of strings to export pgrams from.
 *  @param global_pgrams Output array of pgram integers, indexed by position in the sequence.
 *  @param global_order Current permutation of sequence indices (identity on the first call).
 *  @param start_in_sequence First index (inclusive) in the sequence range to process.
 *  @param end_in_sequence One-past-the-last index in the sequence range to process.
 *  @param start_character Byte offset into each string from which to begin exporting.
 *  @param reverse If set, exports the bitwise-complement of each pgram, so that an ascending integer
 *      sort of the complemented keys yields a descending lexicographic order of the strings.
 */
SZ_HELPER_AUTO void sz_sequence_argsort_serial_export_byte_window_(             
    sz_sequence_t const *const sequence,                                        
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t const *const global_order, 
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,         
    sz_size_t const start_character, sz_bool_t const reverse) {
    sz_size_t const pgram_capacity = sizeof(sz_pgram_t) - 1;
    sz_pgram_t const reverse_mask = reverse ? ~(sz_pgram_t)0 : (sz_pgram_t)0;
    for (sz_size_t sequence_index = start_in_sequence; sequence_index < end_in_sequence; ++sequence_index) {
        sz_sorted_idx_t const partial_order_index = global_order[sequence_index];
        if (SZ_DEBUG && start_character == 0)
            sz_assert_(partial_order_index == sequence_index && "At start this must be an identity permutation.");
        sz_cptr_t const source_str = sequence->get_start(sequence->handle, partial_order_index);
        sz_size_t const length = sequence->get_length(sequence->handle, partial_order_index);
        sz_size_t const remaining_length = length > start_character ? length - start_character : 0;
        sz_size_t const exported_length = remaining_length > pgram_capacity ? pgram_capacity : remaining_length;
        sz_pgram_t *target_pgram = &global_pgrams[sequence_index];
        sz_ptr_t target_str = (sz_ptr_t)target_pgram;
#if !SZ_IS_BIG_ENDIAN_ && SZ_IS_64BIT_
        if (remaining_length >= sizeof(sz_pgram_t)) {
            sz_u64_t const loaded = sz_u64_load(source_str + start_character).u64;
            *target_pgram = (sz_pgram_t)((sz_u64_bytes_reverse(loaded) & ~(sz_u64_t)0xFF) | (sz_u64_t)pgram_capacity);
        }
        else {
            *target_pgram = 0;
            for (sz_size_t character_index = 0; character_index < exported_length; ++character_index)
                target_str[character_index] = source_str[character_index + start_character];
            target_str[pgram_capacity] = (char)exported_length;
            *target_pgram = sz_u64_bytes_reverse(*target_pgram);
        }
#else
        *target_pgram = 0;
        for (sz_size_t character_index = 0; character_index < exported_length; ++character_index)
            target_str[character_index] = source_str[character_index + start_character];
        target_str[pgram_capacity] = (char)exported_length;
#if !SZ_IS_BIG_ENDIAN_
        *target_pgram = sz_u32_bytes_reverse(*target_pgram);
#endif
#endif
        sz_assert_(                                                
            (length <= start_character) == (*target_pgram == 0) && 
            "We can have a zero value if only the string is shorter than other strings at this position.");
        *target_pgram ^= reverse_mask;
    }
    if (SZ_DEBUG && start_character == 0 && !reverse)
        for (sz_size_t sequence_index = start_in_sequence + 1; sequence_index < end_in_sequence; ++sequence_index) {
            sz_pgram_t const previous_pgram = global_pgrams[sequence_index - 1];
            sz_pgram_t const current_pgram = global_pgrams[sequence_index];
            sz_cptr_t const previous_str = sequence->get_start(sequence->handle, sequence_index - 1);
            sz_size_t const previous_length = sequence->get_length(sequence->handle, sequence_index - 1);
            sz_cptr_t const current_str = sequence->get_start(sequence->handle, sequence_index);
            sz_size_t const current_length = sequence->get_length(sequence->handle, sequence_index);
            sz_ordering_t const ordering = sz_order(                                               
                previous_str, previous_length > pgram_capacity ? pgram_capacity : previous_length, 
                current_str, current_length > pgram_capacity ? pgram_capacity : current_length);
            sz_assert_(                                                        
                (previous_pgram < current_pgram) == (ordering == sz_less_k) && 
                "The exported pgrams should be in the same order as the original strings.");
        }
}
/**
 *  @brief Picks the "pivot" value for the QuickSort algorithm's partitioning step using Robert Sedgewick's method,
 *      the median of three elements - the first, the middle, and the last element of the given range.
 *
 *  @param pgrams Pointer to the array of pgrams to choose a pivot from.
 *  @param count Number of pgrams in the array.
 *  @return Pointer to the chosen pivot pgram within the array.
 */
SZ_HELPER_AUTO sz_pgram_t const *sz_sequence_partitioning_pivot_(sz_pgram_t const *pgrams, sz_size_t count) {
    sz_size_t const middle_offset = count / 2;
    sz_pgram_t const *first_pgram = &pgrams[0];
    sz_pgram_t const *middle_pgram = &pgrams[middle_offset];
    sz_pgram_t const *last_pgram = &pgrams[count - 1];
    if (*first_pgram < *middle_pgram) {
        if (*middle_pgram < *last_pgram) { return middle_pgram; }
        else if (*first_pgram < *last_pgram) { return last_pgram; }
        else { return first_pgram; }
    }
    else {
        if (*first_pgram < *last_pgram) { return first_pgram; }
        else if (*middle_pgram < *last_pgram) { return last_pgram; }
        else { return middle_pgram; }
    }
}
/**
 *  @brief The most important part of the QuickSort algorithm partitioning the elements around the pivot.
 *
 *  The classical variant uses the normal 2-way partitioning, but it will scatter the values equal to the pivot
 *  into the left and right partitions. Instead we use the Dutch National Flag @b 3-way partitioning, outputting
 *  the range of values equal to the pivot.
 *
 *  @param global_pgrams Pgram array to partition in place.
 *  @param global_order Corresponding order array, permuted in sync with `global_pgrams`.
 *  @param start_in_sequence First index (inclusive) of the range to partition.
 *  @param end_in_sequence One-past-the-last index of the range to partition.
 *  @param first_pivot_offset Receives the index of the first element equal to the pivot.
 *  @param last_pivot_offset Receives the index of the last element equal to the pivot.
 *  @see https://en.wikipedia.org/wiki/Dutch_national_flag_problem
 */
SZ_HELPER_AUTO void sz_sequence_argsort_serial_3way_partition_(           
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order, 
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,   
    sz_size_t *first_pivot_offset, sz_size_t *last_pivot_offset) {
    sz_size_t const count = end_in_sequence - start_in_sequence;
    if (count <= 4) {
        sz_pgram_t *const pgrams = global_pgrams + start_in_sequence;
        sz_sorted_idx_t *const offsets = global_order + start_in_sequence;
        if (count == 2) { sz_sequence_sorting_network_2x_(pgrams, offsets); }
        else if (count == 3) { sz_sequence_sorting_network_3x_(pgrams, offsets); }
        else if (count == 4) { sz_sequence_sorting_network_4x_(pgrams, offsets); }
        *first_pivot_offset = start_in_sequence;
        *last_pivot_offset = end_in_sequence;
        return;
    }
    sz_pgram_t const pivot_pgram = *sz_sequence_partitioning_pivot_(global_pgrams + start_in_sequence, count);
    sz_size_t partitioning_progress = start_in_sequence; 
    sz_size_t smaller_offset = start_in_sequence;        
    sz_size_t greater_offset = end_in_sequence - 1;      
    while (partitioning_progress <= greater_offset) {
        if (global_pgrams[partitioning_progress] < pivot_pgram) {
            sz_swap_(sz_sorted_idx_t, global_order[partitioning_progress], global_order[smaller_offset]);
            sz_swap_(sz_pgram_t, global_pgrams[partitioning_progress], global_pgrams[smaller_offset]);
            ++partitioning_progress;
            ++smaller_offset;
        }
        else if (global_pgrams[partitioning_progress] > pivot_pgram) {
            sz_swap_(sz_sorted_idx_t, global_order[partitioning_progress], global_order[greater_offset]);
            sz_swap_(sz_pgram_t, global_pgrams[partitioning_progress], global_pgrams[greater_offset]);
            --greater_offset;
        }
        else { ++partitioning_progress; }
    }
    *first_pivot_offset = smaller_offset;
    *last_pivot_offset = greater_offset;
}
/**
 *  @brief Recursive Quick-Sort implementation backing both the `sz_sequence_argsort` and `sz_pgrams_sort`,
 *      and using the `sz_sequence_argsort_serial_3way_partition_` under the hood.
 *
 *  @param global_pgrams Pgram array to sort in place.
 *  @param global_order Corresponding order array, permuted in sync with `global_pgrams`.
 *  @param start_in_sequence First index (inclusive) of the range to sort.
 *  @param end_in_sequence One-past-the-last index of the range to sort.
 *  @param top_count Global cut-off: partitions starting at or beyond this index are left unsorted.
 *      Pass 0 to fully sort the range. With the complement trick the wanted elements (smallest, or
 *      largest under reverse) always fall in `[0, top_count)`, so one cut-off serves both directions.
 */
SZ_API_COMPTIME void sz_sequence_argsort_serial_quicksort_pgrams_(        
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order, 
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence, sz_size_t const top_count) {
    sz_size_t range_start = start_in_sequence, range_end = end_in_sequence;
    for (;;) {
        sz_size_t first_pivot_index, last_pivot_index;
        sz_sequence_argsort_serial_3way_partition_( 
            global_pgrams, global_order,            
            range_start, range_end,                 
            &first_pivot_index, &last_pivot_index);
        sz_size_t const left_start = range_start, left_end = first_pivot_index;
        sz_size_t const right_start = last_pivot_index + 1, right_end = range_end;
        int const left_live = left_start < left_end;
        int const right_live = right_start < right_end && (top_count == 0 || right_start < top_count);
        if (left_live && right_live && (left_end - left_start) <= (right_end - right_start)) {
            sz_sequence_argsort_serial_quicksort_pgrams_(global_pgrams, global_order, left_start, left_end, top_count);
            range_start = right_start, range_end = right_end;
            continue;
        }
        if (left_live && right_live) {
            sz_sequence_argsort_serial_quicksort_pgrams_(global_pgrams, global_order, right_start, right_end, top_count);
            range_start = left_start, range_end = left_end;
            continue;
        }
        if (left_live) {
            range_start = left_start, range_end = left_end;
            continue;
        }
        if (right_live) {
            range_start = right_start, range_end = right_end;
            continue;
        }
        return;
    }
}
/**
 *  @brief One pending equal-prefix group. Window walks used to recurse once per pgram, so a shared prefix of
 *      a few thousand bytes overflowed a small thread stack. Pending groups are disjoint, so `count` frames suffice.
 */
typedef struct sz_argsort_window_frame_t {
    sz_size_t start;
    sz_size_t end;
    sz_size_t skip;
} sz_argsort_window_frame_t;
/** @brief Sorts one pgram window. SIMD backends close their scratch buffers over @p context. */
typedef void (*sz_argsort_range_sorter_t)(sz_pgram_t *pgrams, sz_sorted_idx_t *order, sz_size_t start, sz_size_t end,
                                         sz_size_t top_count, void *context);
SZ_HELPER_AUTO void sz_argsort_serial_sort_range_(sz_pgram_t *pgrams, sz_sorted_idx_t *order, sz_size_t start,
                                                  sz_size_t end, sz_size_t top_count, void *context) {
    sz_unused_(context);
    sz_sequence_argsort_serial_quicksort_pgrams_(pgrams, order, start, end, top_count);
}
/**
 *  @brief Iterative form of the old byte-window recursion. @p stack must hold at least `end - start` frames.
 */
SZ_HELPER_AUTO void sz_argsort_walk_byte_windows_(                                
    sz_sequence_t const *const sequence,                                          
    sz_pgram_t *const pgrams, sz_sorted_idx_t *const order,                       
    sz_size_t const start, sz_size_t const end, sz_size_t const start_character,  
    sz_size_t const top_count, sz_bool_t const reverse,                           
    sz_argsort_range_sorter_t const sort_range, void *const context,              
    sz_argsort_window_frame_t *const stack) {
    sz_size_t stack_size = 1;
    stack[0].start = start, stack[0].end = end, stack[0].skip = start_character;
    sz_size_t const pgram_capacity = sizeof(sz_pgram_t) - 1;
    while (stack_size) {
        sz_argsort_window_frame_t const frame = stack[--stack_size];
        sz_sequence_argsort_serial_export_byte_window_(sequence, pgrams, order, frame.start, frame.end, frame.skip,
                                                       reverse);
        sort_range(pgrams, order, frame.start, frame.end, top_count, context);
        sz_size_t nested_start = frame.start, nested_end = frame.start;
        while (nested_end != frame.end) {
            if (top_count != 0 && nested_start >= top_count) break;
            sz_pgram_t const current_pgram = pgrams[nested_start];
            while (nested_end != frame.end && current_pgram == pgrams[nested_end]) ++nested_end;
            sz_pgram_t const length_source = reverse ? ~current_pgram : current_pgram;
            sz_cptr_t const length_str = (sz_cptr_t)&length_source;
#if !SZ_IS_BIG_ENDIAN_
            sz_size_t const current_pgram_length = (sz_size_t)(sz_u8_t)length_str[0];
#else
            sz_size_t const current_pgram_length = (sz_size_t)(sz_u8_t)length_str[pgram_capacity];
#endif
            int const has_multiple_strings = nested_end - nested_start > 1;
            int const has_more_characters = current_pgram_length == pgram_capacity;
            if (has_multiple_strings && has_more_characters) {
                stack[stack_size].start = nested_start;
                stack[stack_size].end = nested_end;
                stack[stack_size].skip = frame.skip + pgram_capacity;
                ++stack_size;
            }
            else if (has_multiple_strings)
                sz_order_indices_ascending_(order + nested_start, nested_end - nested_start);
            nested_start = nested_end;
        }
    }
}
SZ_API_COMPTIME sz_status_t sz_sequence_argsort_serial(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                       sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse) {
    for (sz_size_t sequence_index = 0; sequence_index != sequence->count; ++sequence_index)
        order[sequence_index] = sequence_index;
    if (sequence->count < 2) return sz_success_k;
    if (sequence->count <= 32 && !reverse) {
        sz_sequence_argsort_with_insertion(sequence, order);
        return sz_success_k;
    }
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }
    sz_size_t const count = sequence->count;
    sz_size_t memory_usage = count * (sizeof(sz_pgram_t) + sizeof(sz_argsort_window_frame_t));
    sz_pgram_t *pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!pgrams) return sz_bad_alloc_k;
    sz_argsort_window_frame_t *frames = (sz_argsort_window_frame_t *)(pgrams + count);
    sz_argsort_walk_byte_windows_(sequence, pgrams, order, 0, count, 0, top_count, reverse,
                                  sz_argsort_serial_sort_range_, SZ_NULL, frames);
    alloc->free(pgrams, memory_usage, alloc);
    return sz_success_k;
}
SZ_API_COMPTIME sz_status_t sz_pgrams_sort_serial(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                                  sz_sorted_idx_t *order) {
    sz_unused_(alloc);
    for (sz_size_t pgram_index = 0; pgram_index != count; ++pgram_index) order[pgram_index] = pgram_index;
    sz_sequence_argsort_serial_quicksort_pgrams_((sz_pgram_t *)pgrams, order, 0, count, 0);
    return sz_success_k;
}
#pragma region Case Insensitive
/** @brief Bit-width of one packed folded code-point: `code-point + 1` for any Unicode scalar (<= U+10FFFF). */
#define sz_argsort_casefold_field_bits_ ((sz_size_t)21)
/** @brief Number of folded code-points packed into one pgram (3 on 64-bit, 1 on 32-bit). */
#define sz_argsort_casefold_fields_(pgram_type) ((sz_size_t)(sizeof(pgram_type) * 8) / sz_argsort_casefold_field_bits_)
/**
 *  @brief Uncased counterpart of `sz_sequence_argsort_serial_export_byte_window_`.
 *      Packs the folded @b code-points of each string at recursion depth @p folded_skip_count into a pgram -
 *      three 21-bit fields on 64-bit targets, MSB-first, each holding `code-point + 1` so a zero field means
 *      "no code-point here". Because `0` sorts below every real `code-point + 1`, a string that ends sorts
 *      before a longer one sharing its prefix - no length byte and no byte-reversal are needed.
 *
 *      Like the byte sort's `start_character`, @p folded_skip_count is the shared recursion depth, only
 *      measured in @b folded code-points. The export is @b stateless: it re-folds each string from the start,
 *      drops the first @p folded_skip_count folded code-points, and emits the next window's worth. Folding is
 *      counted per folded code-point, so one-to-many expansions (e.g. ß→ss) need no special carry. This trades
 *      a per-string heap cursor for re-folding the shared prefix - cheap in the common case of short strings
 *      that diverge early (the prefix is a handful of code-points), and only quadratic for pathologically deep
 *      shared prefixes.
 *
 *      Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint is treated
 *      as a single literal field equal to its raw byte value (so it sorts by byte value, like any one-byte
 *      unit) and processing resyncs at the next byte. Valid input folds and sorts byte-identically to before.
 *
 *  @param folded_skip_count Number of leading folded code-points to skip (recursion depth x fields-per-pgram).
 *  @param reverse Whether to export complemented keys for descending order.
 */
SZ_HELPER_AUTO void sz_sequence_argsort_serial_export_casefold_window_(         
    sz_sequence_t const *const sequence,                                        
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t const *const global_order, 
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,         
    sz_size_t const folded_skip_count, sz_bool_t const reverse) {
    sz_size_t const fields_per_pgram = sz_argsort_casefold_fields_(sz_pgram_t);
    sz_pgram_t const reverse_mask = reverse ? ~(sz_pgram_t)0 : (sz_pgram_t)0;
    for (sz_size_t sequence_index = start_in_sequence; sequence_index < end_in_sequence; ++sequence_index) {
        sz_sorted_idx_t const original_index = global_order[sequence_index];
        if (SZ_DEBUG && folded_skip_count == 0)
            sz_assert_(original_index == sequence_index && "At start this must be an identity permutation.");
        sz_cptr_t const source_str = sequence->get_start(sequence->handle, original_index);
        sz_size_t const source_length = sequence->get_length(sequence->handle, original_index);
        sz_pgram_t key = 0;
        sz_size_t produced = 0, skipped = 0, raw_position = 0;
        while (produced < fields_per_pgram && raw_position < source_length) {
            sz_rune_t source_rune;
            sz_rune_length_t const source_rune_length = sz_rune_decode(source_str + raw_position,
                                                                       source_str + source_length, &source_rune);
            if (source_rune_length == sz_rune_invalid_k) {
                ++raw_position;
                if (skipped < folded_skip_count) {
                    ++skipped;
                    continue;
                }
                sz_size_t const shift = (fields_per_pgram - 1 - produced) * sz_argsort_casefold_field_bits_;
                sz_u8_t const literal_byte = (sz_u8_t)source_str[raw_position - 1];
                key |= (sz_pgram_t)((sz_rune_t)literal_byte + 1) << shift;
                ++produced;
                continue;
            }
            raw_position += (sz_size_t)source_rune_length;
            sz_rune_t folded_runes[3];
            sz_size_t const folded_runes_count = sz_unicode_fold_codepoint_(source_rune, folded_runes);
            for (sz_size_t rune_index = 0; rune_index < folded_runes_count && produced < fields_per_pgram;
                 ++rune_index) {
                if (skipped < folded_skip_count) {
                    ++skipped;
                    continue;
                }
                sz_size_t const shift = (fields_per_pgram - 1 - produced) * sz_argsort_casefold_field_bits_;
                key |= (sz_pgram_t)(folded_runes[rune_index] + 1) << shift;
                ++produced;
            }
        }
        global_pgrams[sequence_index] = key ^ reverse_mask;
    }
}
/**
 *  @brief Iterative form of the old case-fold window recursion. @p stack must hold at least `end - start` frames.
 */
SZ_HELPER_AUTO void sz_argsort_walk_casefold_windows_(                              
    sz_sequence_t const *const sequence,                                            
    sz_pgram_t *const pgrams, sz_sorted_idx_t *const order,                         
    sz_size_t const start, sz_size_t const end, sz_size_t const folded_skip_count,  
    sz_size_t const top_count, sz_bool_t const reverse,                             
    sz_argsort_range_sorter_t const sort_range, void *const context,                
    sz_argsort_window_frame_t *const stack) {
    sz_size_t stack_size = 1;
    stack[0].start = start, stack[0].end = end, stack[0].skip = folded_skip_count;
    sz_size_t const fields_per_pgram = sz_argsort_casefold_fields_(sz_pgram_t);
    sz_pgram_t const lowest_field_mask = ((sz_pgram_t)1 << sz_argsort_casefold_field_bits_) - 1;
    while (stack_size) {
        sz_argsort_window_frame_t const frame = stack[--stack_size];
        sz_sequence_argsort_serial_export_casefold_window_(sequence, pgrams, order, frame.start, frame.end, frame.skip,
                                                           reverse);
        sort_range(pgrams, order, frame.start, frame.end, top_count, context);
        sz_size_t nested_start = frame.start, nested_end = frame.start;
        while (nested_end != frame.end) {
            if (top_count != 0 && nested_start >= top_count) break;
            sz_pgram_t const current_pgram = pgrams[nested_start];
            while (nested_end != frame.end && current_pgram == pgrams[nested_end]) ++nested_end;
            sz_pgram_t const decoded_pgram = reverse ? ~current_pgram : current_pgram;
            int const has_multiple_strings = nested_end - nested_start > 1;
            int const has_more_characters = (decoded_pgram & lowest_field_mask) != 0;
            if (has_multiple_strings && has_more_characters) {
                stack[stack_size].start = nested_start;
                stack[stack_size].end = nested_end;
                stack[stack_size].skip = frame.skip + fields_per_pgram;
                ++stack_size;
            }
            else if (has_multiple_strings)
                sz_order_indices_ascending_(order + nested_start, nested_end - nested_start);
            nested_start = nested_end;
        }
    }
}
SZ_API_COMPTIME sz_status_t sz_sequence_argsort_uncased_serial(  
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc, 
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse) {
    sz_size_t const count = sequence->count;
    for (sz_size_t sequence_index = 0; sequence_index != count; ++sequence_index)
        order[sequence_index] = sequence_index;
    if (count < 2) return sz_success_k;
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }
    sz_size_t const memory_usage = count * (sizeof(sz_pgram_t) + sizeof(sz_argsort_window_frame_t));
    sz_pgram_t *pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!pgrams) return sz_bad_alloc_k;
    sz_argsort_window_frame_t *frames = (sz_argsort_window_frame_t *)(pgrams + count);
    sz_argsort_walk_casefold_windows_(sequence, pgrams, order, 0, count, 0, top_count, reverse,
                                      sz_argsort_serial_sort_range_, SZ_NULL, frames);
    alloc->free(pgrams, memory_usage, alloc);
    return sz_success_k;
}
#pragma endregion // Case Insensitive
/**
 *  @brief Helper function similar to `std::set_union` over pairs of integers and their original indices.
 *      Merges two sorted pgram arrays into a single sorted result, deduplicating equal keys by keeping the
 *      entry from the first array (stable with respect to the first input).
 *
 *  @param first_pgrams Sorted pgram array from the first input.
 *  @param first_indices Corresponding index array for the first input.
 *  @param first_count Number of elements in the first input.
 *  @param second_pgrams Sorted pgram array from the second input.
 *  @param second_indices Corresponding index array for the second input.
 *  @param second_count Number of elements in the second input.
 *  @param result_pgrams Output array for merged pgrams; must hold at least `first_count + second_count` entries.
 *  @param result_indices Output array for merged indices; must hold at least `first_count + second_count` entries.
 *  @see https://en.cppreference.com/w/cpp/algorithm/set_union
 */
SZ_HELPER_AUTO void sz_pgrams_union_serial_(                                                        
    sz_pgram_t const *first_pgrams, sz_sorted_idx_t const *first_indices, sz_size_t first_count,    
    sz_pgram_t const *second_pgrams, sz_sorted_idx_t const *second_indices, sz_size_t second_count, 
    sz_pgram_t *result_pgrams, sz_sorted_idx_t *result_indices) {
    sz_pgram_t const *const first_end = first_pgrams + first_count;
    sz_pgram_t const *const second_end = second_pgrams + second_count;
    sz_pgram_t *const merged_begin = result_pgrams;
    while (first_pgrams < first_end && second_pgrams < second_end) {
        if (*first_pgrams < *second_pgrams) {
            *result_pgrams++ = *first_pgrams++;
            *result_indices++ = *first_indices++;
        }
        else if (*second_pgrams < *first_pgrams) {
            *result_pgrams++ = *second_pgrams++;
            *result_indices++ = *second_indices++;
        }
        else {
            *result_pgrams++ = *first_pgrams;
            *result_indices++ = *first_indices;
            ++first_pgrams;
            ++first_indices;
        }
    }
    while (first_pgrams < first_end) {
        *result_pgrams++ = *first_pgrams++;
        *result_indices++ = *first_indices++;
    }
    while (second_pgrams < second_end) {
        *result_pgrams++ = *second_pgrams++;
        *result_indices++ = *second_indices++;
    }
    if (SZ_DEBUG)
        for (sz_size_t pgram_index = 1; pgram_index < first_count + second_count; ++pgram_index)
            sz_assert_(merged_begin[pgram_index - 1] <= merged_begin[pgram_index] &&
                       "The merged pgrams must be in ascending order.");
}
#ifdef __cplusplus
}
#endif
#endif 
