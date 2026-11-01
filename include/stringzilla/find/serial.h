/**
 *  @brief Serial backend for substring & byte-set search.
 *  @file include/stringzilla/find/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_SERIAL_H_
#define STRINGZILLA_FIND_SERIAL_H_
#include "stringzilla/types.h"
#include "stringzilla/compare.h" 
#ifdef __cplusplus
extern "C" {
#endif
/**
 *  @brief Chooses the offsets of the most interesting characters in a search needle.
 *
 *  Search throughput can significantly deteriorate if we are matching the wrong characters.
 *  Say the needle is "aXaYa", and we are comparing the first, second, and last character.
 *  If we use SIMD and compare many offsets at a time, comparing against "a" in every register is a waste.
 *
 *  Similarly, dealing with UTF-8 inputs, we know that the lower bits of each character code carry more information.
 *  Cyrillic alphabet, for example, falls into [0x0410, 0x042F] code range for uppercase [А, Я], and
 *  into [0x0430, 0x044F] for lowercase [а, я]. Scanning through a text written in Russian, half of the
 *  bytes will carry absolutely no value and will be equal to 0x04.
 *
 *  @param start Pointer to the needle bytes.
 *  @param length Length of the needle in bytes.
 *  @param first Output offset of the first anomalous byte.
 *  @param second Output offset of the second anomalous byte.
 *  @param third Output offset of the third anomalous byte.
 */
SZ_HELPER_AUTO void sz_locate_needle_anomalies_( 
    sz_cptr_t start, sz_size_t length,           
    sz_size_t *first, sz_size_t *second, sz_size_t *third) {
    *first = 0;
    *second = length / 2;
    *third = length - 1;
    int has_duplicates =                   
        start[*first] == start[*second] || 
        start[*first] == start[*third] ||  
        start[*second] == start[*third];
    if (length > 3 && has_duplicates) {
        while (start[*second] == start[*first] && *second + 1 < *third) ++(*second);
        while ((start[*third] == start[*second] || start[*third] == start[*first]) && *third > (*second + 1))
            --(*third);
    }
    if (length > 8) {
        sz_u8_t const *start_u8 = (sz_u8_t const *)start;
        sz_size_t vibrant_first = *first, vibrant_second = *second, vibrant_third = *third;
        while ((start_u8[vibrant_second] > 191 || start_u8[vibrant_second] == start_u8[vibrant_third]) &&
               (vibrant_second + 1 < vibrant_third))
            ++vibrant_second;
        if (start_u8[vibrant_second] < 191) { *second = vibrant_second; }
        else { vibrant_second = *second; }
        while ((start_u8[vibrant_first] > 191 || start_u8[vibrant_first] == start_u8[vibrant_second] ||
                start_u8[vibrant_first] == start_u8[vibrant_third]) &&
               (vibrant_first + 1 < vibrant_second))
            ++vibrant_first;
        if (start_u8[vibrant_first] < 191) { *first = vibrant_first; }
    }
}
/** @brief  Number of byte values present in @p set - four branchless word popcounts. */
SZ_HELPER_INLINE sz_size_t sz_byteset_population_serial_(sz_byteset_t const *set) {
    return (sz_size_t)(sz_u64_popcount(set->_u64s[0]) + sz_u64_popcount(set->_u64s[1]) +
                       sz_u64_popcount(set->_u64s[2]) + sz_u64_popcount(set->_u64s[3]));
}
/** @brief  Unpack the member byte values of @p set into @p members in ascending order; the caller has already
 *          sized the destination from @ref sz_byteset_population_serial_. Shared by the ISA back-ends whose
 *          small-set fast paths broadcast the members as a needle segment. */
SZ_HELPER_INLINE void sz_byteset_members_serial_(sz_byteset_t const *set, sz_u8_t *members) {
    sz_size_t filled = 0;
    for (sz_size_t word_index = 0; word_index != 4; ++word_index)
        for (sz_u64_t word_bits = set->_u64s[word_index]; word_bits; word_bits &= word_bits - 1)
            members[filled++] = (sz_u8_t)(word_index * 64 + sz_u64_ctz(word_bits));
}
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    for (sz_cptr_t const end = text + length; text != end; ++text)
        if (sz_byteset_contains(set, *text)) return text;
    return SZ_NULL_CHAR;
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    sz_cptr_t const end = text;
    for (text += length; text != end;)
        if (sz_byteset_contains(set, *(text -= 1))) return text;
    return SZ_NULL_CHAR;
}
SZ_API_COMPTIME sz_cptr_t sz_find_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    if (!haystack_length) return SZ_NULL_CHAR;
    sz_u8_t const *haystack_cursor = (sz_u8_t const *)haystack;
    sz_u8_t const *const needle_u8 = (sz_u8_t const *)needle;
    sz_u8_t const *const haystack_end = haystack_cursor + haystack_length;
#if !SZ_IS_BIG_ENDIAN_       
#if !SZ_USE_MISALIGNED_LOADS 
    for (; ((sz_size_t)haystack_cursor & 7ull) && haystack_cursor < haystack_end; ++haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
#endif
    sz_u64_vec_t haystack_vec, needle_vec, match_vec;
    match_vec.u64 = 0;
    needle_vec.u64 = (sz_u64_t)*needle_u8 * 0x0101010101010101ull;
    for (; haystack_cursor + 8 <= haystack_end; haystack_cursor += 8) {
        haystack_vec.u64 = *(sz_u64_t const *)haystack_cursor;
        match_vec = sz_u64_each_byte_equal_(haystack_vec, needle_vec);
        if (match_vec.u64) return (sz_cptr_t)(haystack_cursor + sz_u64_ctz(match_vec.u64) / 8);
    }
#endif
    for (; haystack_cursor < haystack_end; ++haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
    return SZ_NULL_CHAR;
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    if (!haystack_length) return SZ_NULL_CHAR;
    sz_u8_t const *const haystack_start = (sz_u8_t const *)haystack;
    sz_u8_t const *const needle_u8 = (sz_u8_t const *)needle;
    sz_u8_t const *haystack_cursor = haystack_start + haystack_length - 1;
#if !SZ_IS_BIG_ENDIAN_       
#if !SZ_USE_MISALIGNED_LOADS 
    for (; ((sz_size_t)(haystack_cursor + 1) & 7ull) && haystack_cursor >= haystack_start; --haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
#endif
    sz_u64_vec_t haystack_vec, needle_vec, match_vec;
    needle_vec.u64 = (sz_u64_t)*needle_u8 * 0x0101010101010101ull;
    for (; haystack_cursor >= haystack_start + 7; haystack_cursor -= 8) {
        haystack_vec.u64 = *(sz_u64_t const *)(haystack_cursor - 7);
        match_vec = sz_u64_each_byte_equal_(haystack_vec, needle_vec);
        if (match_vec.u64) return (sz_cptr_t)(haystack_cursor - sz_u64_clz(match_vec.u64) / 8);
    }
#endif
    for (; haystack_cursor >= haystack_start; --haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
    return SZ_NULL_CHAR;
}
/**
 *  @brief 2-byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 2-byte group signifies a match.
 */
SZ_HELPER_INLINE sz_u64_vec_t sz_u64_each_2byte_equal_(sz_u64_vec_t a_vec, sz_u64_vec_t b_vec) {
    sz_u64_vec_t vec_vec;
    vec_vec.u64 = ~(a_vec.u64 ^ b_vec.u64);
    vec_vec.u64 = ((vec_vec.u64 & 0x7FFF7FFF7FFF7FFFull) + 0x0001000100010001ull) &
                  ((vec_vec.u64 & 0x8000800080008000ull));
    return vec_vec;
}
SZ_HELPER_NOINLINE sz_cptr_t sz_find_1byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length) {
    sz_unused_(needle_length); 
    return sz_find_byte_serial(haystack, haystack_length, needle);
}
SZ_HELPER_NOINLINE sz_cptr_t sz_rfind_1byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                    sz_size_t needle_length) {
    sz_unused_(needle_length); 
    return sz_rfind_byte_serial(haystack, haystack_length, needle);
}
/**
 *  @brief Find the first occurrence of a @b two-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_find_2byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length) {
    sz_assert_(haystack_length >= 2 && "The haystack is too short.");
    sz_unused_(needle_length); 
    sz_cptr_t const haystack_end = haystack + haystack_length;
#if SZ_IS_BIG_ENDIAN_
    for (; haystack + 2 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) == 2) return haystack;
    return SZ_NULL_CHAR;
#endif
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)haystack & 7ull) && haystack + 2 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) == 2) return haystack;
#endif
    sz_u64_vec_t haystack_even_vec, haystack_odd_vec, needle_vec, matches_even_vec, matches_odd_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0], needle_vec.u8s[1] = needle[1];
    needle_vec.u64 *= 0x0001000100010001ull; 
    for (; haystack + 9 <= haystack_end; haystack += 8) {
        haystack_even_vec.u64 = *(sz_u64_t *)haystack;
        haystack_odd_vec.u64 = (haystack_even_vec.u64 >> 8) | ((sz_u64_t)haystack[8] << 56);
        matches_even_vec = sz_u64_each_2byte_equal_(haystack_even_vec, needle_vec);
        matches_odd_vec = sz_u64_each_2byte_equal_(haystack_odd_vec, needle_vec);
        matches_even_vec.u64 >>= 8;
        if (matches_even_vec.u64 + matches_odd_vec.u64) {
            sz_u64_t match_indicators = matches_even_vec.u64 | matches_odd_vec.u64;
            return haystack + sz_u64_ctz(match_indicators) / 8;
        }
    }
    for (; haystack + 2 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) == 2) return haystack;
    return SZ_NULL_CHAR;
}
/**
 *  @brief 4-byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 4-byte group signifies a match.
 */
SZ_HELPER_INLINE sz_u64_vec_t sz_u64_each_4byte_equal_(sz_u64_vec_t a_vec, sz_u64_vec_t b_vec) {
    sz_u64_vec_t vec_vec;
    vec_vec.u64 = ~(a_vec.u64 ^ b_vec.u64);
    vec_vec.u64 = ((vec_vec.u64 & 0x7FFFFFFF7FFFFFFFull) + 0x0000000100000001ull) &
                  ((vec_vec.u64 & 0x8000000080000000ull));
    return vec_vec;
}
/**
 *  @brief Find the first occurrence of a @b four-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_find_4byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length) {
    sz_assert_(haystack_length >= 4 && "The haystack is too short.");
    sz_unused_(needle_length); 
    sz_cptr_t const haystack_end = haystack + haystack_length;
#if SZ_IS_BIG_ENDIAN_
    for (; haystack + 4 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) +
                (haystack[3] == needle[3]) ==
            4)
            return haystack;
    return SZ_NULL_CHAR;
#endif
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)haystack & 7ull) && haystack + 4 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) +
                (haystack[3] == needle[3]) ==
            4)
            return haystack;
#endif
    sz_u64_vec_t haystack0_vec, haystack1_vec, haystack2_vec, haystack3_vec, needle_vec;
    sz_u64_vec_t matches0_vec, matches1_vec, matches2_vec, matches3_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0], needle_vec.u8s[1] = needle[1], needle_vec.u8s[2] = needle[2],
    needle_vec.u8s[3] = needle[3];
    needle_vec.u64 *= 0x0000000100000001ull; 
    sz_u64_t haystack_page_current, haystack_page_next;
    for (; haystack + sizeof(sz_u64_t) + sizeof(sz_u32_t) <= haystack_end; haystack += sizeof(sz_u64_t)) {
        haystack_page_current = *(sz_u64_t *)haystack;
        haystack_page_next = *(sz_u32_t *)(haystack + 8);
        haystack0_vec.u64 = (haystack_page_current);
        haystack1_vec.u64 = (haystack_page_current >> 8) | (haystack_page_next << 56);
        haystack2_vec.u64 = (haystack_page_current >> 16) | (haystack_page_next << 48);
        haystack3_vec.u64 = (haystack_page_current >> 24) | (haystack_page_next << 40);
        matches0_vec = sz_u64_each_4byte_equal_(haystack0_vec, needle_vec);
        matches1_vec = sz_u64_each_4byte_equal_(haystack1_vec, needle_vec);
        matches2_vec = sz_u64_each_4byte_equal_(haystack2_vec, needle_vec);
        matches3_vec = sz_u64_each_4byte_equal_(haystack3_vec, needle_vec);
        if (matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64) {
            matches0_vec.u64 >>= 24;
            matches1_vec.u64 >>= 16;
            matches2_vec.u64 >>= 8;
            sz_u64_t match_indicators = matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64;
            return haystack + sz_u64_ctz(match_indicators) / 8;
        }
    }
    for (; haystack + 4 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) +
                (haystack[3] == needle[3]) ==
            4)
            return haystack;
    return SZ_NULL_CHAR;
}
/**
 *  @brief 3-byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 3-byte group signifies a match.
 */
SZ_HELPER_INLINE sz_u64_vec_t sz_u64_each_3byte_equal_(sz_u64_vec_t a_vec, sz_u64_vec_t b_vec) {
    sz_u64_vec_t vec_vec;
    vec_vec.u64 = ~(a_vec.u64 ^ b_vec.u64);
    vec_vec.u64 = ((vec_vec.u64 & 0xFFFF7FFFFF7FFFFFull) + 0x0000000001000001ull) &
                  ((vec_vec.u64 & 0x0000800000800000ull));
    return vec_vec;
}
/**
 *  @brief Find the first occurrence of a @b three-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_find_3byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length) {
    sz_assert_(haystack_length >= 3 && "The haystack is too short.");
    sz_unused_(needle_length); 
    sz_cptr_t const haystack_end = haystack + haystack_length;
#if SZ_IS_BIG_ENDIAN_
    for (; haystack + 3 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) == 3) return haystack;
    return SZ_NULL_CHAR;
#endif
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)haystack & 7ull) && haystack + 3 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) == 3) return haystack;
#endif
    sz_u64_vec_t haystack0_vec, haystack1_vec, haystack2_vec, haystack3_vec, haystack4_vec;
    sz_u64_vec_t matches0_vec, matches1_vec, matches2_vec, matches3_vec, matches4_vec;
    sz_u64_vec_t needle_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0], needle_vec.u8s[1] = needle[1], needle_vec.u8s[2] = needle[2];
    needle_vec.u64 *= 0x0000000001000001ull; 
    sz_u64_t haystack_page_current, haystack_page_next;
    for (; haystack + sizeof(sz_u64_t) + sizeof(sz_u16_t) <= haystack_end; haystack += sizeof(sz_u64_t)) {
        haystack_page_current = *(sz_u64_t *)haystack;
        haystack_page_next = *(sz_u16_t *)(haystack + 8);
        haystack0_vec.u64 = (haystack_page_current);
        haystack1_vec.u64 = (haystack_page_current >> 8) | (haystack_page_next << 56);
        haystack2_vec.u64 = (haystack_page_current >> 16) | (haystack_page_next << 48);
        haystack3_vec.u64 = (haystack_page_current >> 24) | (haystack_page_next << 40);
        haystack4_vec.u64 = (haystack_page_current >> 32) | (haystack_page_next << 32);
        matches0_vec = sz_u64_each_3byte_equal_(haystack0_vec, needle_vec);
        matches1_vec = sz_u64_each_3byte_equal_(haystack1_vec, needle_vec);
        matches2_vec = sz_u64_each_3byte_equal_(haystack2_vec, needle_vec);
        matches3_vec = sz_u64_each_3byte_equal_(haystack3_vec, needle_vec);
        matches4_vec = sz_u64_each_3byte_equal_(haystack4_vec, needle_vec);
        if (matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64 | matches4_vec.u64) {
            matches0_vec.u64 >>= 16;
            matches1_vec.u64 >>= 8;
            matches3_vec.u64 <<= 8;
            matches4_vec.u64 <<= 16;
            sz_u64_t match_indicators = matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64 |
                                        matches4_vec.u64;
            return haystack + sz_u64_ctz(match_indicators) / 8;
        }
    }
    for (; haystack + 3 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) == 3) return haystack;
    return SZ_NULL_CHAR;
}
/**
 *  @brief Find the last occurrence of a @b two-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_rfind_2byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                    sz_size_t needle_length) {
    sz_assert_(haystack_length >= 2 && "The haystack is too short.");
    sz_unused_(needle_length); 
    sz_cptr_t last = haystack + haystack_length - 2;
#if SZ_IS_BIG_ENDIAN_
    for (; last >= haystack; --last)
        if ((last[0] == needle[0]) + (last[1] == needle[1]) == 2) return last;
    return SZ_NULL_CHAR;
#endif
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)(last + 1) & 7ull) && last >= haystack; --last)
        if ((last[0] == needle[0]) + (last[1] == needle[1]) == 2) return last;
#endif
    sz_u64_vec_t haystack_even_vec, haystack_odd_vec, needle_vec, matches_even_vec, matches_odd_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0], needle_vec.u8s[1] = needle[1];
    needle_vec.u64 *= 0x0001000100010001ull; 
    for (; last >= haystack + 7; last -= 8) {
        sz_cptr_t const window = last - 7;
        haystack_even_vec.u64 = *(sz_u64_t *)window;
        haystack_odd_vec.u64 = (haystack_even_vec.u64 >> 8) | ((sz_u64_t)(sz_u8_t)window[8] << 56);
        matches_even_vec = sz_u64_each_2byte_equal_(haystack_even_vec, needle_vec);
        matches_odd_vec = sz_u64_each_2byte_equal_(haystack_odd_vec, needle_vec);
        matches_even_vec.u64 >>= 8;
        if (matches_even_vec.u64 + matches_odd_vec.u64) {
            sz_u64_t match_indicators = matches_even_vec.u64 | matches_odd_vec.u64;
            return window + (63 - sz_u64_clz(match_indicators)) / 8;
        }
    }
    for (; last >= haystack; --last)
        if ((last[0] == needle[0]) + (last[1] == needle[1]) == 2) return last;
    return SZ_NULL_CHAR;
}
/**
 *  @brief Find the last occurrence of a @b three-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_rfind_3byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                    sz_size_t needle_length) {
    sz_assert_(haystack_length >= 3 && "The haystack is too short.");
    sz_unused_(needle_length); 
    sz_cptr_t last = haystack + haystack_length - 3;
#if SZ_IS_BIG_ENDIAN_
    for (; last >= haystack; --last)
        if ((last[0] == needle[0]) + (last[1] == needle[1]) + (last[2] == needle[2]) == 3) return last;
    return SZ_NULL_CHAR;
#endif
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)(last + 1) & 7ull) && last >= haystack; --last)
        if ((last[0] == needle[0]) + (last[1] == needle[1]) + (last[2] == needle[2]) == 3) return last;
#endif
    sz_u64_vec_t haystack0_vec, haystack1_vec, haystack2_vec, haystack3_vec, haystack4_vec;
    sz_u64_vec_t matches0_vec, matches1_vec, matches2_vec, matches3_vec, matches4_vec;
    sz_u64_vec_t needle_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0], needle_vec.u8s[1] = needle[1], needle_vec.u8s[2] = needle[2];
    needle_vec.u64 *= 0x0000000001000001ull; 
    sz_u64_t haystack_page_current, haystack_page_next;
    for (; last >= haystack + 7; last -= 8) {
        sz_cptr_t const window = last - 7;
        haystack_page_current = *(sz_u64_t *)window;
        haystack_page_next = *(sz_u16_t *)(window + 8);
        haystack0_vec.u64 = (haystack_page_current);
        haystack1_vec.u64 = (haystack_page_current >> 8) | (haystack_page_next << 56);
        haystack2_vec.u64 = (haystack_page_current >> 16) | (haystack_page_next << 48);
        haystack3_vec.u64 = (haystack_page_current >> 24) | (haystack_page_next << 40);
        haystack4_vec.u64 = (haystack_page_current >> 32) | (haystack_page_next << 32);
        matches0_vec = sz_u64_each_3byte_equal_(haystack0_vec, needle_vec);
        matches1_vec = sz_u64_each_3byte_equal_(haystack1_vec, needle_vec);
        matches2_vec = sz_u64_each_3byte_equal_(haystack2_vec, needle_vec);
        matches3_vec = sz_u64_each_3byte_equal_(haystack3_vec, needle_vec);
        matches4_vec = sz_u64_each_3byte_equal_(haystack4_vec, needle_vec);
        if (matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64 | matches4_vec.u64) {
            matches0_vec.u64 >>= 16;
            matches1_vec.u64 >>= 8;
            matches3_vec.u64 <<= 8;
            matches4_vec.u64 <<= 16;
            sz_u64_t match_indicators = matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64 |
                                        matches4_vec.u64;
            return window + (63 - sz_u64_clz(match_indicators)) / 8;
        }
    }
    for (; last >= haystack; --last)
        if ((last[0] == needle[0]) + (last[1] == needle[1]) + (last[2] == needle[2]) == 3) return last;
    return SZ_NULL_CHAR;
}
/**
 *  @brief Find the last occurrence of a @b four-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_rfind_4byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                    sz_size_t needle_length) {
    sz_assert_(haystack_length >= 4 && "The haystack is too short.");
    sz_unused_(needle_length); 
    sz_cptr_t last = haystack + haystack_length - 4;
#if SZ_IS_BIG_ENDIAN_
    for (; last >= haystack; --last)
        if ((last[0] == needle[0]) + (last[1] == needle[1]) + (last[2] == needle[2]) + (last[3] == needle[3]) == 4)
            return last;
    return SZ_NULL_CHAR;
#endif
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)(last + 1) & 7ull) && last >= haystack; --last)
        if ((last[0] == needle[0]) + (last[1] == needle[1]) + (last[2] == needle[2]) + (last[3] == needle[3]) == 4)
            return last;
#endif
    sz_u64_vec_t haystack0_vec, haystack1_vec, haystack2_vec, haystack3_vec, needle_vec;
    sz_u64_vec_t matches0_vec, matches1_vec, matches2_vec, matches3_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0], needle_vec.u8s[1] = needle[1], needle_vec.u8s[2] = needle[2],
    needle_vec.u8s[3] = needle[3];
    needle_vec.u64 *= 0x0000000100000001ull; 
    sz_u64_t haystack_page_current, haystack_page_next;
    for (; last >= haystack + 7; last -= 8) {
        sz_cptr_t const window = last - 7;
        haystack_page_current = *(sz_u64_t *)window;
        haystack_page_next = (sz_u64_t)(*(sz_u16_t *)(window + 8)) | ((sz_u64_t)(sz_u8_t)window[10] << 16);
        haystack0_vec.u64 = (haystack_page_current);
        haystack1_vec.u64 = (haystack_page_current >> 8) | (haystack_page_next << 56);
        haystack2_vec.u64 = (haystack_page_current >> 16) | (haystack_page_next << 48);
        haystack3_vec.u64 = (haystack_page_current >> 24) | (haystack_page_next << 40);
        matches0_vec = sz_u64_each_4byte_equal_(haystack0_vec, needle_vec);
        matches1_vec = sz_u64_each_4byte_equal_(haystack1_vec, needle_vec);
        matches2_vec = sz_u64_each_4byte_equal_(haystack2_vec, needle_vec);
        matches3_vec = sz_u64_each_4byte_equal_(haystack3_vec, needle_vec);
        if (matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64) {
            matches0_vec.u64 >>= 24;
            matches1_vec.u64 >>= 16;
            matches2_vec.u64 >>= 8;
            sz_u64_t match_indicators = matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64;
            return window + (63 - sz_u64_clz(match_indicators)) / 8;
        }
    }
    for (; last >= haystack; --last)
        if ((last[0] == needle[0]) + (last[1] == needle[1]) + (last[2] == needle[2]) + (last[3] == needle[3]) == 4)
            return last;
    return SZ_NULL_CHAR;
}
/**
 *  @brief Boyer-Moore-Horspool algorithm for exact matching of patterns up to @b 256-bytes long.
 *         Uses the Raita heuristic to match the first two, the last, and the middle character of the pattern.
 *
 *  @param haystack The haystack bytes.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle The needle bytes.
 *  @param needle_length Length of the needle in bytes (must be <= 256).
 *  @return Pointer to first match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_find_horspool_upto_256bytes_serial_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                   
    sz_cptr_t needle, sz_size_t needle_length) {
    sz_assert_(needle_length <= 256 && "The pattern is too long.");
    sz_assert_(haystack_length >= needle_length && "The haystack is too short.");
    union {
        sz_u8_t jumps[256];
        sz_u64_vec_t vecs_vec[64];
    } bad_shift_table;
    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    sz_u8_t const *needle_u8 = (sz_u8_t const *)needle;
    {
        sz_u64_vec_t needle_length_vec;
        needle_length_vec.u64 = ((sz_u8_t)(needle_length - 1)) * 0x0101010101010101ull; 
        for (sz_size_t byte_index = 0; byte_index != 64; ++byte_index)
            bad_shift_table.vecs_vec[byte_index].u64 = needle_length_vec.u64;
        for (sz_size_t byte_index = 0; byte_index + 1 < needle_length; ++byte_index)
            bad_shift_table.jumps[needle_u8[byte_index]] = (sz_u8_t)(needle_length - byte_index - 1);
    }
    sz_u32_vec_t haystack_vec, needle_vec;
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    needle_vec.u8s[0] = needle_u8[offset_first];
    needle_vec.u8s[1] = needle_u8[offset_first + 1];
    needle_vec.u8s[2] = needle_u8[offset_mid];
    needle_vec.u8s[3] = needle_u8[offset_last];
    for (sz_size_t byte_index = 0; byte_index <= haystack_length - needle_length;) {
        haystack_vec.u8s[0] = haystack_u8[byte_index + offset_first];
        haystack_vec.u8s[1] = haystack_u8[byte_index + offset_first + 1];
        haystack_vec.u8s[2] = haystack_u8[byte_index + offset_mid];
        haystack_vec.u8s[3] = haystack_u8[byte_index + offset_last];
        if (haystack_vec.u32 == needle_vec.u32 &&
            sz_equal_serial((sz_cptr_t)haystack_u8 + byte_index, needle, needle_length))
            return (sz_cptr_t)haystack_u8 + byte_index;
        byte_index += bad_shift_table.jumps[haystack_u8[byte_index + needle_length - 1]];
    }
    return SZ_NULL_CHAR;
}
/**
 *  @brief Boyer-Moore-Horspool algorithm for @b reverse-order exact matching of patterns up to @b 256-bytes long.
 *         Uses the Raita heuristic to match the first two, the last, and the middle character of the pattern.
 *
 *  @param haystack The haystack bytes.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle The needle bytes.
 *  @param needle_length Length of the needle in bytes (must be <= 256).
 *  @return Pointer to last match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_rfind_horspool_upto_256bytes_serial_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                    
    sz_cptr_t needle, sz_size_t needle_length) {
    sz_assert_(needle_length <= 256 && "The pattern is too long.");
    sz_assert_(haystack_length >= needle_length && "The haystack is too short.");
    union {
        sz_u8_t jumps[256];
        sz_u64_vec_t vecs_vec[64];
    } bad_shift_table;
    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    sz_u8_t const *needle_u8 = (sz_u8_t const *)needle;
    {
        sz_u64_vec_t needle_length_vec;
        needle_length_vec.u64 = ((sz_u8_t)(needle_length - 1)) * 0x0101010101010101ull; 
        for (sz_size_t byte_index = 0; byte_index != 64; ++byte_index)
            bad_shift_table.vecs_vec[byte_index].u64 = needle_length_vec.u64;
        for (sz_size_t byte_index = 0; byte_index + 1 < needle_length; ++byte_index)
            bad_shift_table.jumps[needle_u8[needle_length - byte_index - 1]] = (sz_u8_t)(needle_length - byte_index -
                                                                                         1);
    }
    sz_u32_vec_t haystack_vec, needle_vec;
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    needle_vec.u8s[0] = needle_u8[offset_first];
    needle_vec.u8s[1] = needle_u8[offset_first + 1];
    needle_vec.u8s[2] = needle_u8[offset_mid];
    needle_vec.u8s[3] = needle_u8[offset_last];
    for (sz_size_t skip_index = 0; skip_index <= haystack_length - needle_length;) {
        sz_size_t byte_index = haystack_length - needle_length - skip_index;
        haystack_vec.u8s[0] = haystack_u8[byte_index + offset_first];
        haystack_vec.u8s[1] = haystack_u8[byte_index + offset_first + 1];
        haystack_vec.u8s[2] = haystack_u8[byte_index + offset_mid];
        haystack_vec.u8s[3] = haystack_u8[byte_index + offset_last];
        if (haystack_vec.u32 == needle_vec.u32 &&
            sz_equal_serial((sz_cptr_t)haystack_u8 + byte_index, needle, needle_length))
            return (sz_cptr_t)haystack_u8 + byte_index;
        skip_index += bad_shift_table.jumps[haystack_u8[byte_index]];
    }
    return SZ_NULL_CHAR;
}
/**
 *  @brief Exact substring search helper function, that finds the first occurrence of a prefix of the needle
 *         using a given search function, and then verifies the remaining part of the needle.
 *
 *  @param haystack Pointer to the haystack.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the needle.
 *  @param needle_length Length of the needle in bytes.
 *  @param find_prefix Function used to search for the prefix.
 *  @param prefix_length Length of the prefix to search for.
 *  @return Pointer to first match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_AUTO sz_cptr_t sz_find_with_prefix_( 
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length, sz_find_t find_prefix,
    sz_size_t prefix_length) {
    sz_size_t suffix_length = needle_length - prefix_length;
    while (1) {
        sz_cptr_t found = find_prefix(haystack, haystack_length, needle, prefix_length);
        if (!found) return SZ_NULL_CHAR;
        sz_size_t remaining = haystack_length - (found - haystack);
        if (remaining < needle_length) return SZ_NULL_CHAR;
        if (sz_equal_serial(found + prefix_length, needle + prefix_length, suffix_length)) return found;
        haystack = found + 1;
        haystack_length = remaining - 1;
    }
    return SZ_NULL_CHAR;
}
/**
 *  @brief Exact reverse-order substring search helper function, that finds the last occurrence of a suffix of the
 *         needle using a given search function, and then verifies the remaining part of the needle.
 *
 *  @param haystack Pointer to the haystack.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the needle.
 *  @param needle_length Length of the needle in bytes.
 *  @param find_suffix Function used to search for the suffix.
 *  @param suffix_length Length of the suffix to search for.
 *  @return Pointer to last match start, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_AUTO sz_cptr_t sz_rfind_with_suffix_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_size_t needle_length, sz_find_t find_suffix,
                                               sz_size_t suffix_length) {
    sz_size_t prefix_length = needle_length - suffix_length;
    while (1) {
        sz_cptr_t found = find_suffix(haystack, haystack_length, needle + prefix_length, suffix_length);
        if (!found) return SZ_NULL_CHAR;
        sz_size_t remaining = found - haystack;
        if (remaining < prefix_length) return SZ_NULL_CHAR;
        if (sz_equal_serial(found - prefix_length, needle, prefix_length)) return found - prefix_length;
        if (remaining <= needle_length) return SZ_NULL_CHAR;
        haystack_length = remaining - 1;
    }
    return SZ_NULL_CHAR;
}
SZ_HELPER_NOINLINE sz_cptr_t sz_find_over_4bytes_serial_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                         sz_cptr_t needle, sz_size_t needle_length) {
    return sz_find_with_prefix_(haystack, haystack_length, needle, needle_length, (sz_find_t)sz_find_4byte_serial_, 4);
}
SZ_HELPER_NOINLINE sz_cptr_t sz_find_horspool_over_256bytes_serial_( 
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length) {
    return sz_find_with_prefix_(haystack, haystack_length, needle, needle_length,
                                sz_find_horspool_upto_256bytes_serial_, 256);
}
SZ_HELPER_NOINLINE sz_cptr_t sz_rfind_horspool_over_256bytes_serial_( 
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length) {
    return sz_rfind_with_suffix_(haystack, haystack_length, needle, needle_length,
                                 sz_rfind_horspool_upto_256bytes_serial_, 256);
}
SZ_API_COMPTIME sz_cptr_t sz_find_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length) {
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    sz_find_t backends[] = {
        sz_find_1byte_serial_,
        sz_find_2byte_serial_,
        sz_find_3byte_serial_,
        sz_find_4byte_serial_,
        sz_find_over_4bytes_serial_,
        sz_find_horspool_upto_256bytes_serial_,
        sz_find_horspool_over_256bytes_serial_,
    };
    return backends[
        (needle_length > 1) + (needle_length > 2) + (needle_length > 3) +
        (needle_length > 4) +
        (needle_length > 8) + (needle_length > 256)](haystack, haystack_length, needle, needle_length);
}
SZ_API_COMPTIME sz_cptr_t sz_rfind_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length) {
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    sz_find_t backends[] = {
        sz_rfind_1byte_serial_,
        sz_rfind_2byte_serial_,
        sz_rfind_3byte_serial_,
        sz_rfind_4byte_serial_,
        sz_rfind_horspool_upto_256bytes_serial_,
        sz_rfind_horspool_over_256bytes_serial_,
    };
    return backends[
        (needle_length > 1) + (needle_length > 2) + (needle_length > 3) +
        (needle_length > 4) + (needle_length > 256)](haystack, haystack_length, needle, needle_length);
}
#ifdef __cplusplus
}
#endif
#endif 
