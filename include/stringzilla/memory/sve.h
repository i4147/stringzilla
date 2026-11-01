/**
 *  @brief SVE backend for hardware-accelerated memory operations on Arm v9 CPUs.
 *  @file include/stringzilla/memory/sve.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_SVE_H_
#define STRINGZILLA_MEMORY_SVE_H_
#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"
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
SZ_API_COMPTIME void sz_fill_sve(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    svuint8_t value_u8x = svdup_u8(value);
    sz_size_t vector_length = svcntb(); 
    if (length <= vector_length) {
        svbool_t mask_b8x = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
        svst1_u8(mask_b8x, (sz_u8_t *)target, value_u8x);
    }
    else {
        sz_size_t head_length = vector_length - ((sz_size_t)target % vector_length);
        sz_size_t tail_length = (sz_size_t)(target + length) % vector_length;
        sz_size_t body_length = length - head_length - tail_length;
        svbool_t head_mask_b8x = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)head_length);
        svst1_u8(head_mask_b8x, (sz_u8_t *)target, value_u8x);
        target += head_length;
        for (; body_length >= vector_length; target += vector_length, body_length -= vector_length)
            svst1_u8(svptrue_b8(), (sz_u8_t *)target, value_u8x);
        svbool_t tail_mask_b8x = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)tail_length);
        svst1_u8(tail_mask_b8x, (sz_u8_t *)target, value_u8x);
    }
}
SZ_API_COMPTIME void sz_copy_sve(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_size_t vector_length = svcntb(); 
    if (length <= vector_length) {
        svbool_t mask_b8x = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
        svuint8_t source_u8x = svld1_u8(mask_b8x, (sz_u8_t *)source);
        svst1_u8(mask_b8x, (sz_u8_t *)target, source_u8x);
    }
    else if (length <= 2 * vector_length) {
        svbool_t mask_first_b8x = svptrue_b8();
        svbool_t mask_second_b8x = svwhilelt_b8((sz_u64_t)vector_length, (sz_u64_t)length);
        svuint8_t source_first_u8x = svld1_u8(mask_first_b8x, (sz_u8_t *)(source));
        svuint8_t source_second_u8x = svld1_u8(mask_second_b8x, (sz_u8_t *)(source + vector_length));
        svst1_u8(mask_first_b8x, (sz_u8_t *)(target), source_first_u8x);
        svst1_u8(mask_second_b8x, (sz_u8_t *)(target + vector_length), source_second_u8x);
    }
    else {
        for (; length >= vector_length; source += vector_length, target += vector_length, length -= vector_length) {
            svuint8_t source_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)source);
            svst1_u8(svptrue_b8(), (sz_u8_t *)target, source_u8x);
        }
        if (length) {
            svbool_t mask_b8x = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
            svuint8_t source_u8x = svld1_u8(mask_b8x, (sz_u8_t const *)source);
            svst1_u8(mask_b8x, (sz_u8_t *)target, source_u8x);
        }
    }
}
SZ_API_COMPTIME void sz_move_sve(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_size_t vector_length = svcntb(); 
    if (length <= vector_length) {
        svbool_t mask_b8x = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
        svuint8_t source_u8x = svld1_u8(mask_b8x, (sz_u8_t *)source);
        svst1_u8(mask_b8x, (sz_u8_t *)target, source_u8x);
    }
    else if (length <= 2 * vector_length) {
        svbool_t mask_first_b8x = svptrue_b8();
        svbool_t mask_second_b8x = svwhilelt_b8((sz_u64_t)vector_length, (sz_u64_t)length);
        svuint8_t source_first_u8x = svld1_u8(mask_first_b8x, (sz_u8_t *)(source));
        svuint8_t source_second_u8x = svld1_u8(mask_second_b8x, (sz_u8_t *)(source + vector_length));
        svst1_u8(mask_first_b8x, (sz_u8_t *)(target), source_first_u8x);
        svst1_u8(mask_second_b8x, (sz_u8_t *)(target + vector_length), source_second_u8x);
    }
    else {
        int const overlapping = (target > source && target < source + length);
        if (overlapping) {
            source += length;
            target += length;
            for (; length >= vector_length; length -= vector_length) {
                source -= vector_length;
                target -= vector_length;
                svuint8_t source_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)source);
                svst1_u8(svptrue_b8(), (sz_u8_t *)target, source_u8x);
            }
            if (length) {
                source -= length;
                target -= length;
                svbool_t mask_b8x = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
                svuint8_t source_u8x = svld1_u8(mask_b8x, (sz_u8_t const *)source);
                svst1_u8(mask_b8x, (sz_u8_t *)target, source_u8x);
            }
        }
        else {
            for (; length >= vector_length; source += vector_length, target += vector_length, length -= vector_length) {
                svuint8_t source_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)source);
                svst1_u8(svptrue_b8(), (sz_u8_t *)target, source_u8x);
            }
            if (length) {
                svbool_t mask_b8x = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
                svuint8_t source_u8x = svld1_u8(mask_b8x, (sz_u8_t const *)source);
                svst1_u8(mask_b8x, (sz_u8_t *)target, source_u8x);
            }
        }
    }
}
SZ_API_COMPTIME void sz_lookup_sve(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                   char const lut[sz_at_least_(256)]) {
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }
    sz_size_t const vector_length = svcntb();
    svbool_t const row_predicate_b8x = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)16);
    svuint8_t const row_0_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x00));
    svuint8_t const row_1_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x10));
    svuint8_t const row_2_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x20));
    svuint8_t const row_3_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x30));
    svuint8_t const row_4_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x40));
    svuint8_t const row_5_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x50));
    svuint8_t const row_6_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x60));
    svuint8_t const row_7_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x70));
    svuint8_t const row_8_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x80));
    svuint8_t const row_9_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x90));
    svuint8_t const row_10_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xA0));
    svuint8_t const row_11_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xB0));
    svuint8_t const row_12_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xC0));
    svuint8_t const row_13_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xD0));
    svuint8_t const row_14_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xE0));
    svuint8_t const row_15_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xF0));
    for (sz_size_t byte_index = 0; byte_index < length; byte_index += vector_length) {
        svbool_t const active_b8x = svwhilelt_b8_u64(byte_index, length);
        svuint8_t const source_u8x = svld1_u8(active_b8x, (sz_u8_t const *)(source + byte_index));
        svuint8_t const column_u8x = svand_n_u8_x(active_b8x, source_u8x, 0x0F);
        svuint8_t const high_nibble_u8x = svlsr_n_u8_x(active_b8x, source_u8x, 4);
        svuint8_t result_u8x = svtbl_u8(row_0_u8x, column_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 1), svtbl_u8(row_1_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 2), svtbl_u8(row_2_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 3), svtbl_u8(row_3_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 4), svtbl_u8(row_4_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 5), svtbl_u8(row_5_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 6), svtbl_u8(row_6_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 7), svtbl_u8(row_7_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 8), svtbl_u8(row_8_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 9), svtbl_u8(row_9_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 10), svtbl_u8(row_10_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 11), svtbl_u8(row_11_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 12), svtbl_u8(row_12_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 13), svtbl_u8(row_13_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 14), svtbl_u8(row_14_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 15), svtbl_u8(row_15_u8x, column_u8x),
                              result_u8x);
        svst1_u8(active_b8x, (sz_u8_t *)(target + byte_index), result_u8x);
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
