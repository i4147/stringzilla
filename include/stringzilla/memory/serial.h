/**
 *  @brief Serial backend for hardware-accelerated memory operations.
 *  @file include/stringzilla/memory/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_SERIAL_H_
#define STRINGZILLA_MEMORY_SERIAL_H_
#include "stringzilla/types.h"
#ifdef __cplusplus
extern "C" {
#endif
SZ_API_COMPTIME void sz_lookup_serial(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                      char const lut[sz_at_least_(256)]) {
    sz_u8_t const *lut_u8 = (sz_u8_t const *)lut;
    sz_u8_t const *source_u8 = (sz_u8_t const *)source;
    sz_u8_t *target_u8 = (sz_u8_t *)target;
    sz_u8_t const *source_end = source_u8 + length;
    for (; source_u8 != source_end; ++source_u8, ++target_u8) *target_u8 = lut_u8[*source_u8];
}
#if defined(_MSC_VER) && defined(SZ_OVERRIDE_LIBC) && SZ_OVERRIDE_LIBC
#pragma optimize("", off)
#endif
SZ_API_COMPTIME void sz_fill_serial(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    sz_ptr_t end = target + length;
    if (length < SZ_SWAR_THRESHOLD)
        while (target != end) *(target++) = value;
    else {
        sz_u64_t value64 = (sz_u64_t)value * 0x0101010101010101ull;
        while ((sz_size_t)target & 7ull) *(target++) = value;
        while (target + 8 <= end) *(sz_u64_t *)target = value64, target += 8;
        while (target != end) *(target++) = value;
    }
}
#if defined(_MSC_VER) && defined(SZ_OVERRIDE_LIBC) && SZ_OVERRIDE_LIBC
#pragma optimize("", on)
#endif
SZ_API_COMPTIME void sz_copy_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    while (length >= 8) {
        sz_u64_store(target, sz_u64_load(source).u64);
        target += 8, source += 8, length -= 8;
    }
    while (length--) *(target++) = *(source++);
}
SZ_API_COMPTIME void sz_move_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target < source || target >= source + length) {
        while (length >= 8) {
            sz_u64_store(target, sz_u64_load(source).u64);
            target += 8, source += 8, length -= 8;
        }
        while (length--) *(target++) = *(source++);
    }
    else {
        target += length, source += length;
        while (length >= 8) {
            target -= 8, source -= 8, length -= 8;
            sz_u64_store(target, sz_u64_load(source).u64);
        }
        while (length--) *(--target) = *(--source);
    }
}
#ifdef __cplusplus
}
#endif
#endif 
