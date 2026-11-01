/**
 *  @brief IBM Power VSX backend for memory.
 *  @file include/stringzilla/memory/powervsx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_POWERVSX_H_
#define STRINGZILLA_MEMORY_POWERVSX_H_
#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif
SZ_API_COMPTIME void sz_copy_powervsx(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    for (; length >= 16; target += 16, source += 16, length -= 16)
        vec_xst(vec_xl(0, (unsigned char const *)source), 0, (unsigned char *)target);
    if (length) sz_copy_serial(target, source, length);
}
SZ_API_COMPTIME void sz_move_powervsx(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target < source || target >= source + length) {
        sz_copy_powervsx(target, source, length);
    }
    else {
        target += length, source += length;
        while (length >= 16) {
            target -= 16, source -= 16, length -= 16;
            vec_xst(vec_xl(0, (unsigned char const *)source), 0, (unsigned char *)target);
        }
        while (length) {
            target -= 1, source -= 1, length -= 1;
            *target = *source;
        }
    }
}
SZ_API_COMPTIME void sz_fill_powervsx(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    __vector unsigned char fill_u8x16 = vec_splats(value);
    for (; length >= 16; target += 16, length -= 16) vec_xst(fill_u8x16, 0, (unsigned char *)target);
    if (length) sz_fill_serial(target, length, value);
}
SZ_API_COMPTIME void sz_lookup_powervsx(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                        char const lut[sz_at_least_(256)]) {
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }
    __vector unsigned char lut_u8x16[16];
    for (sz_size_t sub_table_index = 0; sub_table_index < 16; ++sub_table_index)
        lut_u8x16[sub_table_index] = vec_xl(0, (unsigned char const *)(lut + sub_table_index * 16));
    __vector unsigned char const low_nibble_mask_u8x16 = vec_splats((unsigned char)0x0F);
    __vector unsigned char const bit4_mask_u8x16 = vec_splats((unsigned char)0x10);
    __vector unsigned char const bit5_mask_u8x16 = vec_splats((unsigned char)0x20);
    __vector unsigned char const bit6_mask_u8x16 = vec_splats((unsigned char)0x40);
    __vector unsigned char const bit7_mask_u8x16 = vec_splats((unsigned char)0x80);
    sz_size_t byte_index = 0;
    for (; byte_index + 16 <= length; byte_index += 16) {
        __vector unsigned char source_u8x16 = vec_xl(0, (unsigned char const *)(source + byte_index));
        __vector unsigned char low_nibble_index_u8x16 = vec_and(source_u8x16, low_nibble_mask_u8x16);
        __vector unsigned char permuted_u8x16[16];
        for (sz_size_t chunk_index = 0; chunk_index < 16; ++chunk_index)
            permuted_u8x16[chunk_index] = vec_perm(lut_u8x16[chunk_index], lut_u8x16[chunk_index],
                                                   low_nibble_index_u8x16);
        __vector unsigned char select_bit4_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(source_u8x16, bit4_mask_u8x16), bit4_mask_u8x16);
        for (sz_size_t chunk_index = 0; chunk_index < 16; chunk_index += 2)
            permuted_u8x16[chunk_index] = vec_sel(permuted_u8x16[chunk_index], permuted_u8x16[chunk_index + 1],
                                                  select_bit4_u8x16);
        __vector unsigned char select_bit5_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(source_u8x16, bit5_mask_u8x16), bit5_mask_u8x16);
        for (sz_size_t chunk_index = 0; chunk_index < 16; chunk_index += 4)
            permuted_u8x16[chunk_index] = vec_sel(permuted_u8x16[chunk_index], permuted_u8x16[chunk_index + 2],
                                                  select_bit5_u8x16);
        __vector unsigned char select_bit6_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(source_u8x16, bit6_mask_u8x16), bit6_mask_u8x16);
        for (sz_size_t chunk_index = 0; chunk_index < 16; chunk_index += 8)
            permuted_u8x16[chunk_index] = vec_sel(permuted_u8x16[chunk_index], permuted_u8x16[chunk_index + 4],
                                                  select_bit6_u8x16);
        __vector unsigned char select_bit7_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(source_u8x16, bit7_mask_u8x16), bit7_mask_u8x16);
        __vector unsigned char result_u8x16 = vec_sel(permuted_u8x16[0], permuted_u8x16[8], select_bit7_u8x16);
        vec_xst(result_u8x16, 0, (unsigned char *)(target + byte_index));
    }
    if (byte_index < length) sz_lookup_serial(target + byte_index, length - byte_index, source + byte_index, lut);
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
