/**
 *  @brief Uncased UTF-8 search, comparison & invariance checks: serial scaffolding.
 *  @file include/stringzilla/utf8_uncased/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_SERIAL_H_
#define STRINGZILLA_UTF8_UNCASED_SERIAL_H_
#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"
#include "stringzilla/utf8_uncased_fold/serial.h" 
#include "stringzilla/find/serial.h"              
#ifdef __cplusplus
extern "C" {
#endif
/**
 *  @brief Iterator state for streaming through folded UTF-8 runes.
 *  Handles one-to-many case folding expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) transparently.
 */
typedef struct {
    sz_cptr_t ptr;           
    sz_cptr_t end;           
    sz_rune_t pending[4];    
    sz_size_t pending_count; 
    sz_size_t pending_idx;   
} sz_utf8_folded_iter_t;
/** @brief Initialize a folded rune iterator. */
SZ_HELPER_AUTO void sz_utf8_folded_iter_init_(sz_utf8_folded_iter_t *iterator, sz_cptr_t string, sz_size_t length) {
    iterator->ptr = string;
    iterator->end = string + length;
    iterator->pending_count = 0;
    iterator->pending_idx = 0;
}
/**
 *  @brief Get next folded rune. Returns `sz_false_k` when exhausted.
 *  Malformed UTF-8 is handled losslessly: a byte that does not begin a well-formed codepoint is emitted as a
 *  single literal byte (tagged so it compares byte-for-byte and never collides with a real folded codepoint) and
 *  the iterator resyncs by one byte, never reading past `end`.
 */
SZ_HELPER_AUTO sz_bool_t sz_utf8_folded_iter_next_(sz_utf8_folded_iter_t *it, sz_rune_t *out_rune) {
    if (it->pending_idx >= it->pending_count) {
        if (it->ptr >= it->end) return sz_false_k;
        sz_u8_t lead = *(sz_u8_t const *)it->ptr;
        if (lead < 0x80) {
            *out_rune = sz_ascii_fold_(lead);
            it->ptr++;
            it->pending_count = 0; 
            it->pending_idx = 0;   
            return sz_true_k;
        }
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode(it->ptr, it->end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            *out_rune = sz_rune_malformed_byte_(lead);
            it->ptr++;
            it->pending_count = 0;
            it->pending_idx = 0;
            return sz_true_k;
        }
        it->ptr += rune_length;
        it->pending[0] = 0xFFFFFFFFu;
        it->pending[1] = 0xFFFFFFFEu;
        it->pending[2] = 0xFFFFFFFDu;
        it->pending[3] = 0xFFFFFFFCu;
        it->pending_count = sz_unicode_fold_codepoint_(rune, it->pending);
        it->pending_idx = 0;
    }
    *out_rune = it->pending[it->pending_idx++];
    return sz_true_k;
}
/**
 *  @brief Reverse iterator state for streaming through folded UTF-8 runes backwards.
 * Handles one-to-many case folding expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) transparently
 * in reverse order.
 */
typedef struct {
    sz_cptr_t ptr;           
    sz_cptr_t start;         
    sz_rune_t pending[4];    
    sz_size_t pending_count; 
    sz_size_t pending_idx;   
} sz_utf8_folded_reverse_iter_t;
/** @brief Initialize a reverse folded rune iterator. Iterates from end towards start. */
SZ_HELPER_AUTO void sz_utf8_folded_reverse_iter_init_(sz_utf8_folded_reverse_iter_t *it, sz_cptr_t start,
                                                      sz_cptr_t end) {
    it->ptr = end;
    it->start = start;
    it->pending_count = 0;
    it->pending_idx = 0;
}
/**
 *  @brief Get previous folded rune (walking backwards). Returns `sz_false_k` when exhausted.
 * When a codepoint folds to multiple runes (like 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)), returns them in
 * reverse order ('s', then 's'). Malformed UTF-8 is handled losslessly and byte-identically to the forward
 * iterator: a byte that does not begin/end a well-formed codepoint is emitted as a single tagged literal byte and
 * the iterator resyncs by one byte, so the backward rune stream is exactly the reverse of the forward stream.
 */
SZ_HELPER_AUTO sz_bool_t sz_utf8_folded_reverse_iter_prev_(sz_utf8_folded_reverse_iter_t *it, sz_rune_t *out_rune) {
    if (it->pending_idx < it->pending_count) {
        *out_rune = it->pending[it->pending_count - 1 - it->pending_idx];
        it->pending_idx++;
        return sz_true_k;
    }
    if (it->ptr <= it->start) return sz_false_k;
    sz_cptr_t const sequence_end = it->ptr;
    sz_u8_t const last_byte = *(sz_u8_t const *)(sequence_end - 1);
    if (last_byte < 0x80) {
        it->ptr = sequence_end - 1;
        *out_rune = sz_ascii_fold_(last_byte);
        it->pending_count = 0;
        it->pending_idx = 0;
        return sz_true_k;
    }
    sz_cptr_t candidate = sequence_end - 1;
    for (sz_size_t back = 0; back < 3 && candidate > it->start && (*(sz_u8_t const *)candidate & 0xC0) == 0x80; ++back)
        candidate--;
    sz_rune_t rune;
    sz_rune_length_t const rune_length = sz_rune_decode(candidate, sequence_end, &rune);
    if (rune_length == sz_rune_invalid_k || candidate + rune_length != sequence_end) {
        it->ptr = sequence_end - 1;
        *out_rune = sz_rune_malformed_byte_(last_byte);
        it->pending_count = 0;
        it->pending_idx = 0;
        return sz_true_k;
    }
    it->ptr = candidate;
    it->pending[0] = 0xFFFFFFFFu;
    it->pending[1] = 0xFFFFFFFEu;
    it->pending[2] = 0xFFFFFFFDu;
    it->pending[3] = 0xFFFFFFFCu;
    it->pending_count = sz_unicode_fold_codepoint_(rune, it->pending);
    it->pending_idx = 1; 
    *out_rune = it->pending[it->pending_count - 1];
    return sz_true_k;
}
#pragma region Case Invariance & Ordering
/**
 *  @brief Internal helper: checks if a single Unicode codepoint is case-agnostic.
 *
 *  A codepoint is case-agnostic if ALL of the following are true:
 *  1. It folds to exactly itself (no transformation, no expansion)
 *  2. It does NOT belong to any bicameral (cased) script
 *  3. It does NOT appear in any case fold expansion as a target character
 *
 *  The third condition is critical. Consider 'ʾ' (U+02BE, CA BE):
 *  - It has no case variant and folds to itself
 *  - However, 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
 *  - A needle containing 'ʾ' must match at position 1 of the folded expansion of 'ẚ'
 *  - Binary search cannot handle this - it only sees 'ẚ' as a 3-byte sequence (E1 BA 9A)
 *  - Therefore 'ʾ' must NOT be treated as case-agnostic
 *
 *  This function implements the check via explicit range exclusions for all bicameral
 *  scripts and all Unicode blocks containing case fold expansion target characters.
 *
 *  @param rune Unicode codepoint to check.
 *  @return sz_true_k if the codepoint is case-agnostic, sz_false_k otherwise.
 *
 *  @warning This is an internal function. Use sz_utf8_find_cased_serial() for string checking.
 *  @see sz_utf8_find_cased_serial
 *  @see sz_unicode_fold_codepoint_
 */
SZ_HELPER_AUTO sz_bool_t sz_rune_is_uncased_(sz_rune_t rune) {
    sz_rune_t folded_runes[3];
    sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
    if (folded_count != 1 || folded_runes[0] != rune) return sz_false_k;
    if (rune >= 0x0041 && rune <= 0x005A) return sz_false_k; 
    if (rune >= 0x0061 && rune <= 0x007A) return sz_false_k; 
    if (rune >= 0x00C0 && rune <= 0x00FF) return sz_false_k; 
    if (rune >= 0x0100 && rune <= 0x024F) return sz_false_k; 
    if (rune >= 0x0250 && rune <= 0x02AF) return sz_false_k; 
    if (rune >= 0x02B0 && rune <= 0x02FF) return sz_false_k; 
    if (rune >= 0x0300 && rune <= 0x036F) return sz_false_k; 
    if (rune >= 0x0370 && rune <= 0x03FF) return sz_false_k; 
    if (rune >= 0x0400 && rune <= 0x04FF) return sz_false_k; 
    if (rune >= 0x0500 && rune <= 0x052F) return sz_false_k; 
    if (rune >= 0x0531 && rune <= 0x0587) return sz_false_k; 
    if (rune >= 0x10A0 && rune <= 0x10FF) return sz_false_k; 
    if (rune >= 0x13A0 && rune <= 0x13FD) return sz_false_k; 
    if (rune >= 0x1C80 && rune <= 0x1C8F) return sz_false_k; 
    if (rune >= 0x1C90 && rune <= 0x1CBF) return sz_false_k; 
    if (rune == 0x1D79 || rune == 0x1D7D || rune == 0x1D8E)
        return sz_false_k; 
    if (rune >= 0x1E00 && rune <= 0x1EFF) return sz_false_k; 
    if (rune >= 0x1F00 && rune <= 0x1FFF) return sz_false_k; 
    if (rune == 0x214E) return sz_false_k;                   
    if (rune >= 0x2170 && rune <= 0x217F) return sz_false_k; 
    if (rune == 0x2184) return sz_false_k;                   
    if (rune >= 0x24D0 && rune <= 0x24E9) return sz_false_k; 
    if (rune >= 0x2C00 && rune <= 0x2C5F) return sz_false_k; 
    if (rune >= 0x2C60 && rune <= 0x2C7F) return sz_false_k; 
    if (rune >= 0x2C80 && rune <= 0x2CFF) return sz_false_k; 
    if (rune >= 0x2D00 && rune <= 0x2D2F) return sz_false_k; 
    if (rune >= 0x2DE0 && rune <= 0x2DFF) return sz_false_k; 
    if (rune >= 0xA640 && rune <= 0xA69F) return sz_false_k; 
    if (rune >= 0xA720 && rune <= 0xA7FF) return sz_false_k; 
    if (rune >= 0xAB30 && rune <= 0xAB6F) return sz_false_k; 
    if (rune >= 0xAB70 && rune <= 0xABBF) return sz_false_k; 
    if (rune >= 0xFB00 && rune <= 0xFB06) return sz_false_k; 
    if (rune >= 0xFB13 && rune <= 0xFB17) return sz_false_k; 
    if (rune >= 0xFF21 && rune <= 0xFF5A) return sz_false_k; 
    if (rune >= 0x10400 && rune <= 0x1044F) return sz_false_k; 
    if (rune >= 0x104B0 && rune <= 0x104FF) return sz_false_k; 
    if (rune >= 0x10570 && rune <= 0x105BF) return sz_false_k; 
    if (rune >= 0x10780 && rune <= 0x107BF) return sz_false_k; 
    if (rune >= 0x10C80 && rune <= 0x10CFF) return sz_false_k; 
    if (rune >= 0x10D70 && rune <= 0x10D85) return sz_false_k; 
    if (rune >= 0x118A0 && rune <= 0x118FF) return sz_false_k; 
    if (rune >= 0x16E40 && rune <= 0x16E9F) return sz_false_k; 
    if (rune >= 0x16EBB && rune <= 0x16ED3) return sz_false_k; 
    if (rune >= 0x1DF00 && rune <= 0x1DFFF) return sz_false_k; 
    if (rune >= 0x1E000 && rune <= 0x1E02F) return sz_false_k; 
    if (rune >= 0x1E030 && rune <= 0x1E08F) return sz_false_k; 
    if (rune >= 0x1E900 && rune <= 0x1E95F) return sz_false_k; 
    return sz_true_k;
}
SZ_API_COMPTIME sz_cptr_t sz_utf8_find_cased_serial(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *text_cursor = (sz_u8_t const *)str;
    sz_u8_t const *text_end = text_cursor + length;
    while (text_cursor < text_end) {
        sz_u8_t lead = *text_cursor;
        if (lead < 0x80) {
            if ((lead >= 'A' && lead <= 'Z') || (lead >= 'a' && lead <= 'z')) return (sz_cptr_t)text_cursor;
            text_cursor++;
            continue;
        }
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode((sz_cptr_t)text_cursor, (sz_cptr_t)text_end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            text_cursor++;
            continue;
        }
        if (sz_rune_is_uncased_(rune) == sz_false_k) return (sz_cptr_t)text_cursor;
        text_cursor += rune_length;
    }
    return SZ_NULL_CHAR;
}
SZ_API_COMPTIME sz_ordering_t sz_utf8_uncased_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                           sz_size_t b_length) {
    sz_utf8_folded_iter_t a_iterator, b_iterator;
    sz_utf8_folded_iter_init_(&a_iterator, a, a_length);
    sz_utf8_folded_iter_init_(&b_iterator, b, b_length);
    sz_rune_t a_rune = 0,
              b_rune = 0; 
    for (;;) {
        sz_bool_t pulled_from_a = sz_utf8_folded_iter_next_(&a_iterator, &a_rune);
        sz_bool_t pulled_from_b = sz_utf8_folded_iter_next_(&b_iterator, &b_rune);
        if (!pulled_from_a && !pulled_from_b) return sz_equal_k;
        if (!pulled_from_a) return sz_less_k;
        if (!pulled_from_b) return sz_greater_k;
        if (a_rune != b_rune) return sz_order_scalars_(a_rune, b_rune);
    }
}
#pragma endregion // Case Invariance & Ordering
/** @brief  Pops the lowest candidate position from @p matches and returns its bit index - the shared scalar
 *          walk behind every ISA probe filter, so the vector kernels never materialize their own bit scans. */
SZ_HELPER_INLINE sz_size_t sz_utf8_uncased_pop_candidate_(sz_u64_t *matches) {
    sz_size_t const position = (sz_size_t)sz_u64_ctz(*matches);
    *matches &= *matches - 1;
    return position;
}
/**
 *  Per-codepoint Latin Extended-A fold deltas after a C4/C5 lead, indexed by the continuation
 *  byte's low 6 bits (`text & 0x3F`). Entry value is the in-place add: 0 = identity, 1 = fold by
 *  +1. The cross-block irregulars that the case-fold tables flag with 0x80 ('İ' C4 B0, 'Ŀ' C4 BF,
 *  'Ÿ' C5 B8, 'ſ' C5 BF) are 0 here because the alarm routes them to the danger-zone handler, so
 *  the fold leaves them untouched. Same parity that the explicit range checks used to compute, but
 *  resolved in one `vqtbl4q_u8` per lead family. Verified against the serial reference in tests.
 */
static sz_u8_t const sz_utf8_uncased_central_c4_deltas_lut_[64] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 
};
static sz_u8_t const sz_utf8_uncased_central_c5_deltas_lut_[64] = {
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 
};
/**
 *  Monotonic-Greek second-byte fold deltas after a CE lead, indexed by `text & 0x3F`. The deltas
 *  the per-rule range checks used to assemble, resolved in one `vqtbl4q_u8`:
 *  'Ά' (86) +0x26, 'Έ'-'Ί' (88-8A) +0x25, 'Ύ'/'Ώ' (8E-8F) −1, 'Α'-'Ο' (91-9F) +0x20,
 *  'Π'-'Ω' (A0-A9) and 'Ϊ'/'Ϋ' (AA-AB) −0x20. 'Ό' (8C) keeps its byte (lead-only change).
 */
static sz_u8_t const sz_utf8_uncased_greek_ce_deltas_lut_[64] = {
    0,    0,    0,    0,    0,    0,    0x26, 0,
    0x25, 0x25, 0x25, 0,    0,    0,    0xFF, 0xFF, 
    0,    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 
    0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0,
    0xE0, 0xE0, 0xE0, 0xE0, 0,    0,    0,    0, 
    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0, 
};
/**
 *  Lead-promotion flags (+1, CE → CF) for the second-byte classes whose lowercase lands in the CF
 *  block: 'Ό' (8C), 'Ύ'/'Ώ' (8E-8F), 'Π'-'Ω' (A0-A9), 'Ϊ'/'Ϋ' (AA-AB). 'Α'-'Ο' (91-9F) stay
 *  under CE, so they do not promote. Propagated one lane back through `next_bytes` onto the lead.
 */
static sz_u8_t const sz_utf8_uncased_greek_ce_promotes_lut_[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
};
#pragma region Substring Search
/**
 *  @brief Verify head region uncasedly (backward iteration).
 *
 *  Walks backward from needle_end/haystack_end, comparing folded runes.
 *  Returns true if needle region exhausts (matched), with haystack bytes consumed.
 *
 *  @param needle_start Start of needle head region.
 *  @param needle_end End of needle head region (where safe window begins).
 *  @param haystack_start Start of haystack (lower bound for backward scan).
 *  @param haystack_end End of haystack head region (where safe window was found).
 *  @param match_length Haystack bytes consumed by this match.
 */
SZ_HELPER_AUTO sz_bool_t sz_utf8_uncased_verify_head_(sz_cptr_t needle_start, sz_cptr_t needle_end,
                                                      sz_cptr_t haystack_start, sz_cptr_t haystack_end,
                                                      sz_size_t *match_length) {
    if (needle_end <= needle_start) {
        *match_length = 0;
        return sz_true_k;
    }
    sz_utf8_folded_reverse_iter_t needle_riter, haystack_riter;
    sz_utf8_folded_reverse_iter_init_(&needle_riter, needle_start, needle_end);
    sz_utf8_folded_reverse_iter_init_(&haystack_riter, haystack_start, haystack_end);
    sz_rune_t needle_rune = 0, haystack_rune = 0;
    for (;;) {
        sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_rune);
        if (!have_needle) {
            *match_length = (sz_size_t)(haystack_end - haystack_riter.ptr);
            return sz_true_k;
        }
        sz_bool_t have_haystack = sz_utf8_folded_reverse_iter_prev_(&haystack_riter, &haystack_rune);
        if (!have_haystack) return sz_false_k;
        if (needle_rune != haystack_rune) return sz_false_k;
    }
}
/**
 *  @brief Verify tail region uncasedly (forward iteration).
 *
 *  Walks forward, comparing folded runes. Returns true if needle exhausts.
 *
 *  @param needle_start Start of needle tail region.
 *  @param needle_end End of needle tail region (= needle + needle_length).
 *  @param haystack_start Start of haystack tail region.
 *  @param haystack_end End of haystack (upper bound for forward scan).
 *  @param match_length Haystack bytes consumed by this match.
 */
SZ_HELPER_AUTO sz_bool_t sz_utf8_uncased_verify_tail_(sz_cptr_t needle_start, sz_cptr_t needle_end,
                                                      sz_cptr_t haystack_start, sz_cptr_t haystack_end,
                                                      sz_size_t *match_length) {
    sz_size_t needle_length = (sz_size_t)(needle_end - needle_start);
    if (needle_length == 0) {
        *match_length = 0;
        return sz_true_k;
    }
    sz_utf8_folded_iter_t needle_iter, haystack_iter;
    sz_utf8_folded_iter_init_(&needle_iter, needle_start, needle_length);
    sz_utf8_folded_iter_init_(&haystack_iter, haystack_start, (sz_size_t)(haystack_end - haystack_start));
    sz_rune_t needle_rune = 0, haystack_rune = 0;
    for (;;) {
        sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_rune);
        if (!have_needle) {
            *match_length = (sz_size_t)(haystack_iter.ptr - haystack_start);
            return sz_true_k;
        }
        sz_bool_t have_haystack = sz_utf8_folded_iter_next_(&haystack_iter, &haystack_rune);
        if (!have_haystack) return sz_false_k;
        if (needle_rune != haystack_rune) return sz_false_k;
    }
}
/**
 *  @brief Verify a complete match around a SIMD-detected window.
 *
 *  Verifies two regions: "head" (before window) and "tail" (after window).
 *  It's important to note that the middle part may still be in part unprocessed, if its larger
 *  than the "folded slice" of the needle. We handle it as part of the "tail" and the `needle_tail_bytes`
 *  must be calculated accordingly.
 *
 *  @param haystack Haystack start pointer, arbitrary case.
 *  @param haystack_length Haystack length in bytes.
 *  @param needle Needle start pointer, arbitrary case.
 *  @param needle_length Needle length in bytes.
 *  @param haystack_matched_offset Start offset of matched safe window in haystack in bytes.
 *  @param haystack_matched_length Length of matched safe window in haystack in bytes.
 *  @param needle_head_bytes Start of matched safe window in needle in bytes.
 *  @param needle_tail_bytes Number of bytes in the needle remaining after the matched part.
 *  @param match_length Total length of the verified match in haystack bytes.
 *  @return Match start pointer, or SZ_NULL_CHAR if validation fails.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_verify_match_(                   
    sz_cptr_t haystack, sz_size_t haystack_length,                        
    sz_cptr_t needle, sz_size_t needle_length,                            
    sz_size_t haystack_matched_offset, sz_size_t haystack_matched_length, 
    sz_size_t needle_head_bytes, sz_size_t needle_tail_bytes,             
    sz_size_t *match_length) {
    sz_cptr_t needle_end = needle + needle_length;
    sz_cptr_t haystack_end = haystack + haystack_length;
    sz_size_t head_match_length = 0;
    if (needle_head_bytes)
        if (!sz_utf8_uncased_verify_head_(                    
                needle, needle + needle_head_bytes,           
                haystack, haystack + haystack_matched_offset, 
                &head_match_length))
            return SZ_NULL_CHAR;
    sz_size_t tail_match_length = 0;
    sz_cptr_t haystack_tail_start = haystack + haystack_matched_offset + haystack_matched_length;
    if (needle_tail_bytes)
        if (!sz_utf8_uncased_verify_tail_(                              
                needle + needle_length - needle_tail_bytes, needle_end, 
                haystack_tail_start, haystack_end,                      
                &tail_match_length))
            return SZ_NULL_CHAR;
    *match_length = head_match_length + haystack_matched_length + tail_match_length;
    return haystack + haystack_matched_offset - head_match_length;
}
/**
 *  @brief Hash-free uncased search for needles that fold to exactly 1 rune.
 *      Examples: 'a', 'A', 'б', 'Б' (but NOT 'ß' (U+00DF, C3 9F) → "ss" = 2 runes).
 *
 *  Single-pass algorithm: parses each source rune, folds it, checks if it produces
 *  exactly one rune matching the target. No iterator overhead, no verification needed.
 *
 *  @param haystack Pointer to the haystack string to search within.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle_folded The single folded rune to search for.
 *  @param match_length Output: length of the matched rune in haystack bytes on success.
 *  @return Pointer to the first matching rune, or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_1folded_serial_( 
    sz_cptr_t haystack, sz_size_t haystack_length,               
    sz_rune_t needle_folded, sz_size_t *match_length) {
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length;
    sz_rune_t haystack_folded_runes[3] = {~needle_folded};
    while (haystack < haystack_end) {
        haystack_rune_length = sz_rune_decode(haystack, haystack_end, &haystack_rune);
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[0] = sz_rune_malformed_byte_((sz_u8_t)*haystack);
            haystack_folded_runes[1] = ~needle_folded;
            haystack_folded_runes[2] = ~needle_folded;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes); }
        sz_u32_t has_match =                              
            (haystack_folded_runes[0] == needle_folded) + 
            (haystack_folded_runes[1] == needle_folded) + 
            (haystack_folded_runes[2] == needle_folded);
        if (has_match) {
            *match_length = haystack_rune_length;
            return haystack;
        }
        haystack += haystack_rune_length;
    }
    *match_length = 0;
    return SZ_NULL_CHAR;
}
/**
 *  @brief Search a "danger zone" region using 1-folded candidate search + validation.
 *
 *  When SIMD kernels detect potentially problematic bytes (ligatures, Greek Extended, etc.),
 *  they fall back to this serial search within the affected chunk. This function:
 *  1. Extracts the first folded rune from the needle's safe window
 *  2. Searches for candidates matching that rune
 *  3. Validates each candidate using the full verification pipeline
 *
 *  @param haystack Full haystack string, arbitrary case.
 *  @param haystack_length Full haystack length in bytes.
 *  @param needle Full needle string, arbitrary case.
 *  @param needle_length Full needle length.
 *  @param danger_cursor Start of the danger zone region to search.
 *  @param danger_length Length of the danger zone region in bytes.
 *  @param needle_first_safe_folded_rune The first rune of the safe window, folded.
 *  @param needle_first_safe_folded_rune_offset Offset of the safe window within the needle.
 *  @param match_length Haystack bytes consumed by the match.
 *  @return Pointer to match start, or SZ_NULL_CHAR if not found in this region.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_in_danger_zone_( 
    sz_cptr_t haystack, sz_size_t haystack_length,               
    sz_cptr_t needle, sz_size_t needle_length,                   
    sz_cptr_t danger_cursor, sz_size_t danger_length,            
    sz_rune_t needle_first_safe_folded_rune,                     
    sz_size_t needle_first_safe_folded_rune_offset,              
    sz_size_t *match_length) {
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_cptr_t const danger_end = sz_min_of_two(danger_cursor + danger_length, haystack_end);
    while (danger_cursor < danger_end) {
        sz_u8_t lead_byte = *(sz_u8_t const *)danger_cursor;
        if ((lead_byte & 0xC0) == 0x80) {
            danger_cursor++;
            continue;
        }
        sz_rune_t haystack_rune;
        sz_rune_length_t haystack_rune_length;
        sz_rune_t haystack_folded_runes[3] = {~needle_first_safe_folded_rune};
        haystack_rune_length = sz_rune_decode(danger_cursor, haystack_end, &haystack_rune);
        sz_size_t haystack_folded_runes_count;
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[0] = sz_rune_malformed_byte_((sz_u8_t)*danger_cursor);
            haystack_folded_runes_count = 1;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { haystack_folded_runes_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes); }
        if (haystack_folded_runes[0] == needle_first_safe_folded_rune) {
            sz_cptr_t match = sz_utf8_uncased_verify_match_( 
                haystack, haystack_length,                   
                needle, needle_length,                       
                danger_cursor - haystack, 0,                 
                needle_first_safe_folded_rune_offset,
                needle_length - needle_first_safe_folded_rune_offset, 
                match_length);
            if (match) return match;
            else { goto consider_second_haystack_folded_rune; } 
        }
    consider_second_haystack_folded_rune:
        if (haystack_folded_runes_count > 1 && haystack_folded_runes[1] == needle_first_safe_folded_rune) {
            sz_cptr_t haystack_match_start = 0, haystack_match_end = 0;
            sz_rune_t needle_riter_rune = 0, haystack_riter_rune = 0;
            sz_utf8_folded_reverse_iter_t needle_riter, haystack_riter;
            sz_utf8_folded_reverse_iter_init_(&needle_riter, needle, needle + needle_first_safe_folded_rune_offset);
            sz_utf8_folded_reverse_iter_init_(&haystack_riter, haystack, danger_cursor);
            {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (have_needle && needle_riter_rune != haystack_folded_runes[0])
                    goto consider_third_haystack_folded_rune;
            }
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (!have_needle) {
                    haystack_match_start = haystack_riter.ptr;
                    break;
                }
                sz_bool_t have_haystack = sz_utf8_folded_reverse_iter_prev_(&haystack_riter, &haystack_riter_rune);
                if (!have_haystack) goto consider_third_haystack_folded_rune;
                if (needle_riter_rune != haystack_riter_rune) goto consider_third_haystack_folded_rune;
            }
            sz_rune_t needle_iter_rune = 0, haystack_iter_rune = 0;
            sz_utf8_folded_iter_t needle_iter, haystack_iter;
            sz_utf8_folded_iter_init_(&needle_iter, needle + needle_first_safe_folded_rune_offset,
                                      needle_length - needle_first_safe_folded_rune_offset);
            sz_utf8_folded_iter_init_(&haystack_iter, danger_cursor + haystack_rune_length,
                                      haystack_end - (danger_cursor + haystack_rune_length));
            {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                sz_assert_(have_needle && needle_iter_rune == needle_first_safe_folded_rune);
            }
            if (haystack_folded_runes_count == 3) {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                if (have_needle && needle_iter_rune != haystack_folded_runes[2])
                    goto consider_third_haystack_folded_rune;
            }
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                if (!have_needle) {
                    haystack_match_end = haystack_iter.ptr;
                    break;
                }
                sz_bool_t have_haystack = sz_utf8_folded_iter_next_(&haystack_iter, &haystack_iter_rune);
                if (!have_haystack) goto consider_third_haystack_folded_rune;
                if (needle_iter_rune != haystack_iter_rune) goto consider_third_haystack_folded_rune;
            }
            if (haystack_match_start != 0 && haystack_match_end != 0) {
                *match_length = (sz_size_t)(haystack_match_end - haystack_match_start);
                return haystack_match_start;
            }
        }
    consider_third_haystack_folded_rune:
        if (haystack_folded_runes_count > 2 && haystack_folded_runes[2] == needle_first_safe_folded_rune) {
            sz_cptr_t haystack_match_start = 0, haystack_match_end = 0;
            sz_rune_t needle_riter_rune = 0, haystack_riter_rune = 0;
            sz_utf8_folded_reverse_iter_t needle_riter, haystack_riter;
            sz_utf8_folded_reverse_iter_init_(&needle_riter, needle, needle + needle_first_safe_folded_rune_offset);
            sz_utf8_folded_reverse_iter_init_(&haystack_riter, haystack, danger_cursor);
            {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (have_needle && needle_riter_rune != haystack_folded_runes[1])
                    goto consider_following_haystack_runes;
                have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (have_needle && needle_riter_rune != haystack_folded_runes[0])
                    goto consider_following_haystack_runes;
            }
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (!have_needle) {
                    haystack_match_start = haystack_riter.ptr;
                    break;
                }
                sz_bool_t have_haystack = sz_utf8_folded_reverse_iter_prev_(&haystack_riter, &haystack_riter_rune);
                if (!have_haystack) goto consider_following_haystack_runes;
                if (needle_riter_rune != haystack_riter_rune) goto consider_following_haystack_runes;
            }
            sz_rune_t needle_iter_rune = 0, haystack_iter_rune = 0;
            sz_utf8_folded_iter_t needle_iter, haystack_iter;
            sz_utf8_folded_iter_init_(&needle_iter, needle + needle_first_safe_folded_rune_offset,
                                      needle_length - needle_first_safe_folded_rune_offset);
            sz_utf8_folded_iter_init_(&haystack_iter, danger_cursor + haystack_rune_length,
                                      haystack_end - (danger_cursor + haystack_rune_length));
            {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                sz_assert_(have_needle && needle_iter_rune == needle_first_safe_folded_rune);
            }
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                if (!have_needle) {
                    haystack_match_end = haystack_iter.ptr;
                    break;
                }
                sz_bool_t have_haystack = sz_utf8_folded_iter_next_(&haystack_iter, &haystack_iter_rune);
                if (!have_haystack) goto consider_following_haystack_runes;
                if (needle_iter_rune != haystack_iter_rune) goto consider_following_haystack_runes;
            }
            if (haystack_match_start != 0 && haystack_match_end != 0) {
                *match_length = (sz_size_t)(haystack_match_end - haystack_match_start);
                return haystack_match_start;
            }
        }
    consider_following_haystack_runes:
        danger_cursor += haystack_rune_length;
    }
    return SZ_NULL_CHAR;
}
/**
 *  @brief Hash-free uncased search for needles that fold to exactly 2 runes.
 *      Examples: 'ab', 'AB', 'ß' (U+00DF) → "ss", 'ﬁ' (U+FB01) → "fi".
 *
 *  Single-pass sliding window over the folded rune stream. Handles expansions
 *  by buffering folded runes from each source and tracking source boundaries.
 *
 *  @param haystack Pointer to the haystack string to search within.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param first_needle_folded First folded rune of the 2-rune needle.
 *  @param second_needle_folded Second folded rune of the 2-rune needle.
 *  @param match_length Output: length of the matched region in haystack bytes on success.
 *  @return Pointer to the first match, or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_2folded_serial_( 
    sz_cptr_t haystack, sz_size_t haystack_length,               
    sz_rune_t first_needle_folded, sz_rune_t second_needle_folded, sz_size_t *match_length) {
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length;
    sz_cptr_t first_history_source_begin = haystack;
    sz_rune_t haystack_folded_runes[4] = {~first_needle_folded};
    while (haystack < haystack_end) {
        haystack_rune_length = sz_rune_decode(haystack, haystack_end, &haystack_rune);
        sz_rune_t sentinel = ~second_needle_folded;
        haystack_folded_runes[2] = sentinel;
        haystack_folded_runes[3] = sentinel;
        sz_size_t folded_count;
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[1] = sz_rune_malformed_byte_((sz_u8_t)*haystack);
            folded_count = 1;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { folded_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes + 1); }
        sz_u32_t has_match_f0 = first_needle_folded == haystack_folded_runes[0];
        sz_u32_t has_match_f1 = first_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_f2 = first_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s1 = second_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_s2 = second_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s3 = second_needle_folded == haystack_folded_runes[3];
        sz_u32_t match_at_01 = has_match_f0 * has_match_s1;
        sz_u32_t match_at_12 = has_match_f1 * has_match_s2;
        sz_u32_t match_at_23 = has_match_f2 * has_match_s3;
        sz_u32_t has_match = match_at_01 + match_at_12 + match_at_23;
        if (has_match) {
            sz_cptr_t match_begin = match_at_01 ? first_history_source_begin : haystack;
            sz_cptr_t match_end = haystack + haystack_rune_length;
            *match_length = (sz_size_t)(match_end - match_begin);
            return match_begin;
        }
        haystack_folded_runes[0] = haystack_folded_runes[folded_count];
        first_history_source_begin = haystack;
        haystack += haystack_rune_length;
    }
    *match_length = 0;
    return SZ_NULL_CHAR;
}
/**
 *  @brief Hash-free uncased search for needles that fold to exactly 3 runes.
 *      Examples: 'abc', 'ABC', "aß" → "ass", "ﬁa" (U+FB01) → "fia".
 *
 *  Single-pass sliding window of 3 folded runes over the haystack's folded stream.
 *  Handles expansions by buffering folded runes and tracking source boundaries.
 *
 *  @param haystack Pointer to the haystack string to search within.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param first_needle_folded First folded rune of the 3-rune needle.
 *  @param second_needle_folded Second folded rune of the 3-rune needle.
 *  @param third_needle_folded Third folded rune of the 3-rune needle.
 *  @param match_length Output: length of the matched region in haystack bytes on success.
 *  @return Pointer to the first match, or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_3folded_serial_( 
    sz_cptr_t haystack, sz_size_t haystack_length,               
    sz_rune_t first_needle_folded, sz_rune_t second_needle_folded, sz_rune_t third_needle_folded,
    sz_size_t *match_length) {
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length;
    sz_cptr_t history_source_begin[2] = {haystack, haystack};
    sz_rune_t haystack_folded_runes[5] = {~first_needle_folded, ~second_needle_folded, 0, 0, 0};
    while (haystack < haystack_end) {
        haystack_rune_length = sz_rune_decode(haystack, haystack_end, &haystack_rune);
        sz_rune_t sentinel = ~third_needle_folded;
        haystack_folded_runes[3] = sentinel;
        haystack_folded_runes[4] = sentinel;
        sz_size_t folded_count;
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[2] = sz_rune_malformed_byte_((sz_u8_t)*haystack);
            folded_count = 1;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { folded_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes + 2); }
        sz_u32_t has_match_f0 = first_needle_folded == haystack_folded_runes[0];
        sz_u32_t has_match_f1 = first_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_f2 = first_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s1 = second_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_s2 = second_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s3 = second_needle_folded == haystack_folded_runes[3];
        sz_u32_t has_match_t2 = third_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_t3 = third_needle_folded == haystack_folded_runes[3];
        sz_u32_t has_match_t4 = third_needle_folded == haystack_folded_runes[4];
        sz_u32_t match_at_012 = has_match_f0 * has_match_s1 * has_match_t2;
        sz_u32_t match_at_123 = has_match_f1 * has_match_s2 * has_match_t3;
        sz_u32_t match_at_234 = has_match_f2 * has_match_s3 * has_match_t4;
        sz_u32_t has_match = match_at_012 + match_at_123 + match_at_234;
        if (has_match) {
            sz_cptr_t match_begin;
            if (match_at_012) match_begin = history_source_begin[0];
            else if (match_at_123) match_begin = history_source_begin[1];
            else match_begin = haystack;
            sz_cptr_t match_end = haystack + haystack_rune_length;
            *match_length = (sz_size_t)(match_end - match_begin);
            return match_begin;
        }
        if (folded_count >= 2) {
            haystack_folded_runes[0] = haystack_folded_runes[folded_count];
            haystack_folded_runes[1] = haystack_folded_runes[folded_count + 1];
            history_source_begin[0] = haystack;
            history_source_begin[1] = haystack;
        }
        else {
            sz_assert_(folded_count == 1);
            haystack_folded_runes[0] = haystack_folded_runes[1];
            haystack_folded_runes[1] = haystack_folded_runes[2];
            history_source_begin[0] = history_source_begin[1];
            history_source_begin[1] = haystack;
        }
        haystack += haystack_rune_length;
    }
    *match_length = 0;
    return SZ_NULL_CHAR;
}
SZ_API_COMPTIME sz_cptr_t sz_utf8_uncased_search_serial( 
    sz_cptr_t haystack, sz_size_t haystack_length,       
    sz_cptr_t needle, sz_size_t needle_length,           
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *match_length) {
    sz_unused_(needle_metadata); 
    if (needle_length == 0) {
        *match_length = 0;
        return haystack;
    }
    if (sz_utf8_find_cased_serial(needle, needle_length) == SZ_NULL_CHAR) {
        sz_cptr_t result = sz_find_serial(haystack, haystack_length, needle, needle_length);
        if (result) {
            *match_length = needle_length;
            return result;
        }
        *match_length = 0;
        return SZ_NULL_CHAR;
    }
    if (needle_length <= 12) {
        sz_rune_t folded_runes[4]; 
        sz_size_t folded_count = 0;
        sz_utf8_folded_iter_t iter;
        sz_utf8_folded_iter_init_(&iter, needle, needle_length);
        sz_rune_t rune;
        while (folded_count < 4 && sz_utf8_folded_iter_next_(&iter, &rune)) folded_runes[folded_count++] = rune;
        switch (folded_count) {
        case 1:
            return sz_utf8_uncased_search_1folded_serial_( 
                haystack, haystack_length,                 
                folded_runes[0], match_length);
        case 2:
            return sz_utf8_uncased_search_2folded_serial_( 
                haystack, haystack_length,                 
                folded_runes[0], folded_runes[1], match_length);
        case 3:
            return sz_utf8_uncased_search_3folded_serial_( 
                haystack, haystack_length,                 
                folded_runes[0], folded_runes[1], folded_runes[2], match_length);
        default: break; 
        }
    }
    sz_size_t const ring_capacity = 32;
    sz_rune_t needle_runes[32];
    sz_size_t needle_prefix_count = 0, needle_total_count = 0;
    sz_u64_t needle_hash = 0;
    {
        sz_utf8_folded_iter_t needle_iter;
        sz_utf8_folded_iter_init_(&needle_iter, needle, needle_length);
        sz_rune_t rune;
        while (needle_prefix_count < ring_capacity && sz_utf8_folded_iter_next_(&needle_iter, &rune)) {
            needle_runes[needle_prefix_count++] = rune;
            needle_hash = needle_hash * 257 + rune;
        }
        needle_total_count = needle_prefix_count;
        while (sz_utf8_folded_iter_next_(&needle_iter, &rune)) needle_total_count++;
    }
    if (!needle_prefix_count) {
        *match_length = 0;
        return SZ_NULL_CHAR;
    }
    sz_u64_t hash_multiplier = 1;
    for (sz_size_t lane_index = 1; lane_index < needle_prefix_count; ++lane_index) hash_multiplier *= 257;
    sz_rune_t window_runes[32];
    sz_cptr_t window_sources[32];     
    sz_size_t window_skip_counts[32]; 
    sz_size_t ring_head = 0;
    sz_u64_t window_hash = 0;
    sz_utf8_folded_iter_t haystack_iter;
    sz_utf8_folded_iter_init_(&haystack_iter, haystack, haystack_length);
    sz_cptr_t window_start = haystack;
    sz_cptr_t current_source = haystack;
    sz_size_t current_skip = 0;
    sz_size_t window_count = 0;
    while (window_count < needle_prefix_count) {
        sz_cptr_t pre_advance_cursor = haystack_iter.ptr;
        sz_rune_t rune;
        if (!sz_utf8_folded_iter_next_(&haystack_iter, &rune)) break;
        window_runes[window_count] = rune;
        if (haystack_iter.pending_idx <= 1 || haystack_iter.pending_count == 0) {
            current_source = pre_advance_cursor;
            current_skip = 0;
        }
        window_sources[window_count] = current_source;
        window_skip_counts[window_count] = current_skip;
        window_hash = window_hash * 257 + rune;
        window_count++;
        if (haystack_iter.pending_idx > 0 && haystack_iter.pending_idx < haystack_iter.pending_count)
            current_skip = haystack_iter.pending_idx;
    }
    if (window_count < needle_prefix_count) {
        *match_length = 0;
        return SZ_NULL_CHAR;
    }
    sz_cptr_t window_end = haystack_iter.ptr;
    for (;;) {
        if (window_hash == needle_hash) {
            sz_size_t first_segment = needle_prefix_count - ring_head;
            sz_size_t mismatches = 0;
            for (sz_size_t lane_index = 0; lane_index < first_segment; ++lane_index)
                mismatches += window_runes[ring_head + lane_index] != needle_runes[lane_index];
            for (sz_size_t lane_index = 0; lane_index < ring_head; ++lane_index)
                mismatches += window_runes[lane_index] != needle_runes[first_segment + lane_index];
            if (!mismatches) {
                sz_size_t skip_runes = window_skip_counts[ring_head];
                if (needle_total_count <= ring_capacity) {
                    *match_length = (sz_size_t)(window_end - window_start);
                    return window_start;
                }
                sz_utf8_folded_iter_t verify_haystack_iter;
                sz_utf8_folded_iter_init_(&verify_haystack_iter, window_start,
                                          (sz_size_t)(haystack + haystack_length - window_start));
                sz_rune_t skip_rune;
                for (sz_size_t skip_index = 0; skip_index < skip_runes; ++skip_index)
                    sz_utf8_folded_iter_next_(&verify_haystack_iter, &skip_rune);
                sz_utf8_folded_iter_t verify_needle_iter;
                sz_utf8_folded_iter_init_(&verify_needle_iter, needle, needle_length);
                sz_rune_t needle_rune_v, haystack_rune_v;
                sz_bool_t match_ok = sz_true_k;
                while (sz_utf8_folded_iter_next_(&verify_needle_iter, &needle_rune_v)) {
                    if (!sz_utf8_folded_iter_next_(&verify_haystack_iter, &haystack_rune_v) ||
                        needle_rune_v != haystack_rune_v) {
                        match_ok = sz_false_k;
                        break;
                    }
                }
                if (match_ok) {
                    *match_length = (sz_size_t)(verify_haystack_iter.ptr - window_start);
                    return window_start;
                }
            }
        }
        sz_cptr_t pre_advance_cursor = haystack_iter.ptr;
        sz_rune_t new_rune;
        if (!sz_utf8_folded_iter_next_(&haystack_iter, &new_rune)) break;
        window_hash -= window_runes[ring_head] * hash_multiplier;
        window_hash = window_hash * 257 + new_rune;
        sz_size_t next_head = ring_head + 1;
        next_head = next_head == needle_prefix_count ? 0 : next_head;
        window_runes[ring_head] = new_rune;
        if (haystack_iter.pending_idx <= 1 || haystack_iter.pending_count == 0) {
            current_source = pre_advance_cursor;
            current_skip = 0;
        }
        window_sources[ring_head] = current_source;
        window_skip_counts[ring_head] = current_skip;
        if (haystack_iter.pending_idx > 0 && haystack_iter.pending_idx < haystack_iter.pending_count)
            current_skip = haystack_iter.pending_idx;
        ring_head = next_head;
        window_start = window_sources[ring_head];
        window_end = haystack_iter.ptr;
    }
    *match_length = 0;
    return SZ_NULL_CHAR;
}
#pragma endregion // Substring Search
#pragma region Character Safety Profiles
/**
 *  @brief Determine safety profile for a character across all script contexts.
 *
 *  This function encodes the contextual safety rules from the ASCII selector
 *  and applies them consistently to all paths that include ASCII.
 *
 *  @param rune The decoded codepoint.
 *  @param rune_bytes UTF-8 byte length of this codepoint (1-4).
 *  @param prev_rune Previous codepoint (0 if at start).
 *  @param next_rune Next codepoint (0 if at end).
 *  @param prev_prev_rune Codepoint before prev_rune (0 if prev is at start).
 *  @param next_next_rune Codepoint after next_rune (0 if next is at end).
 *  @param safety_profiles Safety flags for each script path.
 *  @return The primary fast path preferred for this rune.
 *
 *  @note Using 0 for boundary markers is safe even though NUL (U+0000) is a valid
 *        codepoint in StringZilla's length-based strings. This works because:
 *        1. NUL is valid ASCII (< 0x80), so boundary and actual NUL are treated identically
 *        2. Ligature checks use inequality (lower_prev != 'f'), and 0 never matches letters
 *        3. NUL doesn't participate in any Unicode case folding or ligature expansions
 *
 *  @note The neighbor-of-neighbor context (prev_prev, next_next) enables position-1 and
 *        position-N-2 detection for the 's' rule: if prev_prev==0 && prev!=0, we're at
 *        position 1; if next_next==0 && next!=0, we're at position N-2.
 */
SZ_HELPER_AUTO sz_utf8_uncased_rune_safety_profile_t sz_utf8_uncased_rune_safety_profile_( 
    sz_rune_t rune, sz_size_t rune_bytes,                                                  
    sz_rune_t prev_rune, sz_rune_t next_rune,                                              
    sz_rune_t prev_prev_rune, sz_rune_t next_next_rune,                                    
    unsigned int *safety_profiles) {
    unsigned safety = 0;
    unsigned int western_group = 
        (1 << sz_utf8_uncased_rune_safe_western_europe_k);
    unsigned int central_viet_group =                       
        (1 << sz_utf8_uncased_rune_safe_central_europe_k) | 
        (1 << sz_utf8_uncased_rune_safe_vietnamese_k);
    unsigned int strict_ascii_group =                   
        (1 << sz_utf8_uncased_rune_ascii_invariant_k) | 
        (1 << sz_utf8_uncased_rune_safe_cyrillic_k) |   
        (1 << sz_utf8_uncased_rune_safe_greek_k) |      
        (1 << sz_utf8_uncased_rune_safe_armenian_k) |   
        (1 << sz_utf8_uncased_rune_safe_georgian_k);
    sz_rune_t lower = (rune >= 'A' && rune <= 'Z') ? (rune + 0x20) : rune;
    sz_rune_t lower_prev = (prev_rune >= 'A' && prev_rune <= 'Z') ? (prev_rune + 0x20) : prev_rune;
    sz_rune_t lower_next = (next_rune >= 'A' && next_rune <= 'Z') ? (next_rune + 0x20) : next_rune;
    sz_bool_t prev_ascii = (prev_rune != 0 && prev_rune < 0x80) ? sz_true_k : sz_false_k;
    sz_bool_t next_ascii = (next_rune != 0 && next_rune < 0x80) ? sz_true_k : sz_false_k;
    sz_bool_t at_start = (prev_rune == 0) ? sz_true_k : sz_false_k;
    sz_bool_t at_end = (next_rune == 0) ? sz_true_k : sz_false_k;
    sz_bool_t at_pos_1 = (prev_rune != 0 && prev_prev_rune == 0) ? sz_true_k : sz_false_k;
    sz_bool_t at_pos_n_minus_2 = (next_rune != 0 && next_next_rune == 0) ? sz_true_k : sz_false_k;
    if (rune < 0x80) {
        if (lower >= 'a' && lower <= 'z') {
            switch (lower) {
            case 'b':
            case 'c':
            case 'd':
            case 'e':
            case 'g':
            case 'm':
            case 'o':
            case 'p':
            case 'q':
            case 'r':
            case 'u':
            case 'v':
            case 'x':
            case 'z': safety |= strict_ascii_group | central_viet_group | western_group; break;
            case 'k': safety |= central_viet_group | western_group; break;
            case 'a':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;
            case 'h':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;
            case 'j':
                if (at_end == sz_false_k && next_ascii)
                    safety |= strict_ascii_group | central_viet_group | western_group;
                break;
            case 'w':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;
            case 'y':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;
            case 'n':
                if (at_start == sz_false_k && prev_ascii) {
                    safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k) | 
                              (1 << sz_utf8_uncased_rune_safe_cyrillic_k) |   
                              (1 << sz_utf8_uncased_rune_safe_greek_k);       
                    safety |= central_viet_group | western_group;
                }
                break;
            case 'i':
                if (at_start == sz_false_k && at_end == sz_false_k && next_ascii && lower_prev != 'f')
                    safety |= strict_ascii_group | central_viet_group | western_group;
                break;
            case 'l':
                if (at_start == sz_false_k && lower_prev != 'f') safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;
            case 't':
                if (at_start == sz_false_k && at_end == sz_false_k && next_ascii && lower_prev != 's')
                    safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;
            case 'f':
                if (at_start == sz_false_k && at_end == sz_false_k && prev_ascii && next_ascii && lower_prev != 'f' &&
                    lower_next != 'f' && lower_next != 'i' && lower_next != 'l')
                    safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;
            case 's':
                if (at_start == sz_false_k && at_end == sz_false_k && prev_ascii && next_ascii && lower_prev != 's' &&
                    lower_next != 's' && lower_next != 't')
                    safety |= central_viet_group;
                if (at_start == sz_false_k && at_end == sz_false_k && 
                    !(at_pos_1 == sz_true_k && lower_prev == 's') &&  
                    !(at_pos_n_minus_2 == sz_true_k && lower_next == 's'))
                    safety |= western_group;
                break;
            default:
                safety |= strict_ascii_group | central_viet_group | western_group;
                break;
            }
        }
        else {
            safety |= strict_ascii_group | central_viet_group | western_group;
        }
        *safety_profiles = safety;
        return sz_utf8_uncased_rune_ascii_invariant_k;
    }
    if (rune_bytes == 2) {
        sz_u8_t lead = (rune >> 6) | 0xC0;     
        sz_u8_t second = (rune & 0x3F) | 0x80; 
        if (lead == 0xC2 || lead == 0xC3) {
            if (rune == 0x00E5) {
            }
            else if (rune == 0x00DF) {
                safety |= western_group;
            }
            else if (rune == 0x00B5) {
                safety |= (1 << sz_utf8_uncased_rune_safe_greek_k);
            }
            else { safety |= western_group | central_viet_group; }
        }
        if (lead == 0xC4 || lead == 0xC5) {
            if (rune != 0x0130 && rune != 0x0149 && rune != 0x017F) { safety |= central_viet_group; }
        }
        if (lead == 0xC6) { safety |= (1 << sz_utf8_uncased_rune_safe_vietnamese_k); }
        if ((lead == 0xD0 && second >= 0x80 && second <= 0xBF) || 
            (lead == 0xD1 && second >= 0x80 && second <= 0x9F)) { 
            safety |= (1 << sz_utf8_uncased_rune_safe_cyrillic_k);
        }
        if (lead == 0xCE) {
            if ((second >= 0x86 && second <= 0x8F) && second != 0x87 && second != 0x8B && second != 0x8D &&
                second != 0x90) {
                safety |= (1 << sz_utf8_uncased_rune_safe_greek_k);
            }
            if (second >= 0x91 && second <= 0xA9) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            if (second >= 0xAA && second <= 0xAB) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            if (second >= 0xAC && second <= 0xAF) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            if (second >= 0xB1 && second <= 0xBF) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
        }
        if (lead == 0xCF) {
            if (second >= 0x80 && second <= 0x89) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            if (second >= 0x8A && second <= 0x8E) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
        }
        {
            sz_bool_t is_armenian_range = sz_false_k;
            sz_bool_t armenian_safe = sz_true_k;
            if ((lead == 0xD4 && second >= 0xB1 && second <= 0xBF) ||
                (lead == 0xD5 && second >= 0x80 && second <= 0x96) ||
                (lead == 0xD5 && second >= 0xA1 && second <= 0xBF) ||
                (lead == 0xD6 && second >= 0x80 && second <= 0x86)) {
                is_armenian_range = sz_true_k;
                sz_rune_t lower_prev_arm = prev_rune;
                sz_rune_t lower_next_arm = next_rune;
                if (prev_rune >= 0x0531 && prev_rune <= 0x0556) lower_prev_arm = prev_rune + 0x30;
                if (next_rune >= 0x0531 && next_rune <= 0x0556) lower_next_arm = next_rune + 0x30;
                switch (rune) {
                case 0x0565: 
                case 0x0535: 
                    if (at_start || lower_prev_arm == 0x0574 || lower_next_arm == 0x0582) armenian_safe = sz_false_k;
                    break;
                case 0x0582: 
                    if (at_start || at_end || lower_prev_arm == 0x0565) armenian_safe = sz_false_k;
                    break;
                case 0x0574: 
                case 0x0544: 
                    if (at_end || lower_next_arm == 0x0576 || lower_next_arm == 0x0565 || lower_next_arm == 0x056B ||
                        lower_next_arm == 0x056D)
                        armenian_safe = sz_false_k;
                    break;
                case 0x0576: 
                case 0x0546: 
                    if (at_start || lower_prev_arm == 0x0574 || lower_prev_arm == 0x057E) armenian_safe = sz_false_k;
                    break;
                case 0x056B: 
                case 0x053B: 
                    if (at_start || lower_prev_arm == 0x0574) armenian_safe = sz_false_k;
                    break;
                case 0x057E: 
                case 0x054E: 
                    if (at_end || lower_next_arm == 0x0576) armenian_safe = sz_false_k;
                    break;
                case 0x056D: 
                case 0x053D: 
                    if (at_start || lower_prev_arm == 0x0574) armenian_safe = sz_false_k;
                    break;
                default: break;
                }
            }
            if (is_armenian_range && armenian_safe) { safety |= (1 << sz_utf8_uncased_rune_safe_armenian_k); }
        }
        if (sz_rune_is_uncased_(rune)) safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k);
        *safety_profiles = safety;
        if (rune >= 0x0080 && rune <= 0x00FF) return sz_utf8_uncased_rune_safe_western_europe_k; 
        if (rune >= 0x0100 && rune <= 0x024F) return sz_utf8_uncased_rune_safe_central_europe_k; 
        if (rune >= 0x0370 && rune <= 0x03FF) return sz_utf8_uncased_rune_safe_greek_k;          
        if (rune >= 0x0400 && rune <= 0x04FF) return sz_utf8_uncased_rune_safe_cyrillic_k;       
        if (rune >= 0x0530 && rune <= 0x058F) return sz_utf8_uncased_rune_safe_armenian_k;       
        return sz_utf8_uncased_rune_invariant_k;
    }
    if (rune_bytes == 3) {
        sz_u8_t lead = (rune >> 12) | 0xE0;
        sz_u8_t second = ((rune >> 6) & 0x3F) | 0x80;
        sz_u8_t third = (rune & 0x3F) | 0x80;
        if (lead == 0xE1 && (second >= 0xB8 && second <= 0xBB)) {
            if (second == 0xBA && third >= 0x96 && third <= 0x9F) {
            }
            else { safety |= (1 << sz_utf8_uncased_rune_safe_vietnamese_k); }
        }
        if (lead == 0xE1 && second == 0x83 && third >= 0x90) { safety |= (1 << sz_utf8_uncased_rune_safe_georgian_k); }
        if (sz_rune_is_uncased_(rune)) safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k);
        *safety_profiles = safety;
        if (rune >= 0x10D0 && rune <= 0x10FF) return sz_utf8_uncased_rune_safe_georgian_k; 
        if (rune >= 0x1E00 && rune <= 0x1EFF)
            return sz_utf8_uncased_rune_safe_vietnamese_k; 
        return sz_utf8_uncased_rune_invariant_k;
    }
    if (sz_rune_is_uncased_(rune)) safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k);
    *safety_profiles = safety;
    return sz_utf8_uncased_rune_invariant_k;
}
/**
 *  @brief Compute diversity score for a byte sequence.
 *
 *  Uses a 256-bit bitmap to efficiently count distinct byte values.
 *  Higher scores indicate more diverse byte values, which lead to better
 *  filtering during SIMD search (fewer false positives).
 *
 *  @param data Pointer to byte sequence.
 *  @param length Length of byte sequence.
 *  @return Count of distinct byte values (0-256).
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_probe_diversity_score_(sz_u8_t const *data, sz_size_t length) {
    if (length <= 1) return length;
    sz_u64_t seen[4] = {0, 0, 0, 0}; 
    sz_size_t distinct = 0;
    for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) {
        sz_u8_t byte = data[byte_index];
        sz_size_t word = byte >> 6;                
        sz_u64_t bit = (sz_u64_t)1 << (byte & 63); 
        if (!(seen[word] & bit)) {
            seen[word] |= bit;
            ++distinct;
        }
    }
    return distinct;
}
/**
 *  @brief Find the "best safe window" in the needle for each script path.
 *
 *  The objective is as follows. For a given needle, find a slice, that when folded fits into 16 bytes
 *  and where all characters are "safe" with respect to a certain path. If no such path can be found,
 *  an empty result is returned. It might be the case for a search query like "s" or "n", that by itself
 *  isn't safe for any path given the number of Unicode characters expanding into multiple 's'- or 'n'-containing
 *  sequences. The selected safe folded slice will never begin mid-character in the needle, so if it starts with
 *  an 'ŉ' (U+0149, C5 89), we can't choose - 'n' (6E) - the second half of its folded sequence as a starting point.
 *
 *  The algorithm is as follows. Iterate through the arbitrary-case "ŉEeDlE_WITH_LONG_SUFFIX", unpacking runes.
 *  For each input rune, perform folding, expanding into a sequence, like:
 *
 *      'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E).
 *
 *  Continue unpacking the rest, until we reach a 16-byte limit, like:
 *
 *      ʼ     n  e  e  d  l  e  _  w  i  t  h  _  l  o  n  g
 *      CA BC 6E 45 45 44 4C 45 5F 57 49 54 48 5F 4C 4F 4E 47
 *
 *  At this point, we need to trim it to make sure - its characters satisfy boundary conditions.
 *  Assuming at the next step we'll move the iterator to the next input rune to point to 'E' (U+0045) input character,
 *  we only trim from the end. But also invalidate the whole starting position if a bad character is chosen at start.
 *  For safe window starting position we can have multiple length variants, assuming different safe paths can have
 *  different rules for the last symbol in the safe sequence.
 *
 *  Once we have safe window for a certain script, we evaluate its diversity score - the number of distinct byte
 *  values in the folded window. The more diverse - the better! We keep track of best seen window for each script.
 *
 *  We also track not only the safety with respect to a certain profile, but also applicability. For example,
 *  the needle "xyz" is safe with respect to the Western European path, as well as Central European, Vietnamese,
 *  and potentially others. But it's pure ASCII. We shouldn't pay the cost of complex Vietnamese case-folding of
 *  triple-byte Latin extensions for just "xyz". So we must invalidate the "safe path" if its just "safe", but
 *  not ideal.
 *
 *  In the end, we'll have up to 7 best safe windows, one per script path.
 *  The heuristic is:
 *
 *  - Prefer ASCII, if there is an ASCII-safe path at least 4 bytes wide with at least 4 distinct byte values.
 *    It's only one subtraction, a comparison, and a masked addition. Cheapest of all kernels.
 *  - Pick the most diverse variant from all others, if ASCII variant isn't good enough.
 *
 *  We then identify the four "probe" positions within the <= 16 byte folded safe window, one more than
 *  in exact substring search kernels with Raita heuristics:
 *
 *  1. implicit at `refined->folded_slice[0]`
 *  2. stored in `refined->probe_second` - targets last byte of 2nd character when 4+ chars
 *  3. stored in `refined->probe_third` - targets last byte of 3rd character when 4+ chars
 *  4. implicit at `refined->folded_slice[refined->folded_slice_length - 1]`
 *
 *  By aiming at the last byte of each UTF-8 codepoint we maximize diversity, as in a Russian text almost
 *  all letters will have the same first byte, but mostly different second byte. The same is true for many
 *  other languages. For short strings (< 4 bytes), probes will necessarily overlap - this is expected.
 *  The function also sets `offset_in_unfolded` and `length_in_unfolded` to track where the
 *  selected folded slice came from in the original unfolded input.
 *
 *  @param needle Pointer to needle string (original, not folded).
 *  @param needle_length Length in bytes.
 *  @param refined Output metadata structure to populate.
 */
SZ_HELPER_AUTO void sz_utf8_uncased_needle_metadata_(sz_cptr_t needle, sz_size_t needle_length, 
                                                     sz_utf8_uncased_needle_metadata_t *refined) {
    typedef struct {
        sz_size_t start_offset;   
        sz_size_t input_length;   
        sz_u8_t folded_bytes[16]; 
        sz_size_t folded_length;  
        sz_bool_t applicable;     
        sz_bool_t broken;         
        sz_size_t diversity;      
    } script_window_t_;
    sz_size_t const num_scripts = 9;
    script_window_t_ best[9];
    for (sz_size_t script_index = 0; script_index < num_scripts; ++script_index) {
        best[script_index].start_offset = 0;
        best[script_index].input_length = 0;
        best[script_index].folded_length = 0;
        best[script_index].applicable = sz_false_k;
        best[script_index].broken = sz_false_k;
        best[script_index].diversity = 0;
    }
    if (needle_length == 0) {
        refined->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
        refined->offset_in_unfolded = 0;
        refined->length_in_unfolded = 0;
        refined->folded_slice_length = 0;
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }
    if (sz_utf8_find_malformed(needle, needle_length) != SZ_NULL_CHAR) {
        refined->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
        refined->offset_in_unfolded = 0;
        refined->length_in_unfolded = 0;
        refined->folded_slice_length = 0;
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }
    sz_u8_t const *needle_start = (sz_u8_t const *)needle;
    sz_u8_t const *needle_end = needle_start + needle_length;
    for (sz_u8_t const *needle_cursor = needle_start; needle_cursor < needle_end;) {
        script_window_t_ current[9];
        for (sz_size_t script_index = 0; script_index < num_scripts; ++script_index) {
            current[script_index].start_offset = (sz_size_t)(needle_cursor - needle_start);
            current[script_index].input_length = 0;
            current[script_index].folded_length = 0;
            current[script_index].applicable = sz_false_k;
            current[script_index].broken = sz_false_k;
            current[script_index].diversity = 0;
        }
        sz_rune_t prev_prev_rune = 0;
        sz_rune_t prev_rune = 0;
        sz_u8_t const *position = needle_cursor;
        sz_bool_t any_active = sz_true_k;
        while (position < needle_end && any_active) {
            sz_rune_t rune;
            sz_rune_length_t const rune_bytes = sz_rune_decode_unchecked((sz_cptr_t)position, &rune);
            if (position + rune_bytes > needle_end) break; 
            sz_rune_t next_rune = 0;
            sz_rune_length_t next_bytes = sz_rune_invalid_k;
            if (position + rune_bytes < needle_end) {
                next_bytes = sz_rune_decode_unchecked((sz_cptr_t)(position + rune_bytes), &next_rune);
                if (position + rune_bytes + next_bytes > needle_end) next_rune = 0;
            }
            sz_rune_t next_next_rune = 0;
            if (next_rune != 0 && position + rune_bytes + next_bytes < needle_end) {
                sz_rune_length_t const next_next_bytes = sz_rune_decode_unchecked(
                    (sz_cptr_t)(position + rune_bytes + next_bytes), &next_next_rune);
                if (position + rune_bytes + next_bytes + next_next_bytes > needle_end) next_next_rune = 0;
            }
            unsigned safety_mask = 0;
            sz_utf8_uncased_rune_safety_profile_t primary_script = sz_utf8_uncased_rune_safety_profile_( 
                rune, rune_bytes, prev_rune, next_rune, prev_prev_rune, next_next_rune, &safety_mask);
            sz_rune_t folded_runes[4];
            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
            sz_u8_t folded_utf8[16];
            sz_size_t folded_utf8_length = 0;
            for (sz_size_t rune_index = 0; rune_index < folded_count; ++rune_index) {
                folded_utf8_length += sz_rune_encode(folded_runes[rune_index], folded_utf8 + folded_utf8_length);
            }
            any_active = sz_false_k;
            for (sz_size_t script_index = 1; script_index < num_scripts; ++script_index) {
                if (current[script_index].broken) continue;
                sz_bool_t is_safe = (safety_mask & (1u << script_index)) ? sz_true_k : sz_false_k;
                if (is_safe && current[script_index].folded_length + folded_utf8_length <= 16) {
                    for (sz_size_t byte_index = 0; byte_index < folded_utf8_length; ++byte_index) {
                        current[script_index].folded_bytes[current[script_index].folded_length + byte_index] =
                            folded_utf8[byte_index];
                    }
                    current[script_index].folded_length += folded_utf8_length;
                    current[script_index].input_length += rune_bytes;
                    if (primary_script == script_index) { current[script_index].applicable = sz_true_k; }
                    any_active = sz_true_k;
                }
                else {
                    current[script_index].broken = sz_true_k;
                }
            }
            prev_prev_rune = prev_rune;
            prev_rune = rune;
            position += rune_bytes;
        }
        for (sz_size_t script_index = 1; script_index < num_scripts; ++script_index) {
            if (!current[script_index].applicable || current[script_index].folded_length == 0) continue;
            current[script_index].diversity = sz_utf8_probe_diversity_score_(current[script_index].folded_bytes,
                                                                             current[script_index].folded_length);
            if (current[script_index].diversity > best[script_index].diversity ||
                (current[script_index].diversity == best[script_index].diversity &&
                 current[script_index].folded_length > best[script_index].folded_length)) {
                best[script_index] = current[script_index];
            }
        }
        sz_rune_t skip_rune;
        sz_rune_length_t const skip_length = sz_rune_decode_unchecked((sz_cptr_t)needle_cursor, &skip_rune);
        needle_cursor += skip_length;
    }
    sz_size_t chosen_script = 0;
    sz_size_t best_diversity = 0;
    if (best[sz_utf8_uncased_rune_ascii_invariant_k].applicable &&
        best[sz_utf8_uncased_rune_ascii_invariant_k].folded_length >= 4 &&
        best[sz_utf8_uncased_rune_ascii_invariant_k].diversity >= 4) {
        chosen_script = sz_utf8_uncased_rune_ascii_invariant_k;
    }
    else {
        for (sz_size_t script_index = 1; script_index < num_scripts; ++script_index) {
            if (best[script_index].applicable && best[script_index].diversity > best_diversity) {
                best_diversity = best[script_index].diversity;
                chosen_script = script_index;
            }
        }
    }
    if (chosen_script == 0) {
        refined->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
        refined->offset_in_unfolded = 0;
        refined->length_in_unfolded = 0;
        refined->folded_slice_length = 0;
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }
    refined->kernel_id = (sz_u8_t)chosen_script;
    refined->offset_in_unfolded = best[chosen_script].start_offset;
    refined->length_in_unfolded = best[chosen_script].input_length;
    refined->folded_slice_length = (sz_u8_t)best[chosen_script].folded_length;
    for (sz_size_t byte_index = 0; byte_index < best[chosen_script].folded_length; ++byte_index) {
        refined->folded_slice[byte_index] = best[chosen_script].folded_bytes[byte_index];
    }
    sz_size_t folded_length = best[chosen_script].folded_length;
    if (folded_length == 0) {
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }
    sz_size_t char_ends[16];
    sz_size_t char_count = 0;
    for (sz_size_t byte_index = 0; byte_index < folded_length; ++byte_index) {
        sz_u8_t next = (byte_index + 1 < folded_length) ? refined->folded_slice[byte_index + 1]
                                                        : 0xC0; 
        if ((next & 0xC0) != 0x80) {                            
            if (char_count < 16) char_ends[char_count++] = byte_index;
        }
    }
    if (char_count >= 4) {
        refined->probe_second = (sz_u8_t)char_ends[1];
        refined->probe_third = (sz_u8_t)char_ends[2];
    }
    else if (folded_length <= 3) {
        refined->probe_second = (folded_length > 1) ? 1 : 0;
        refined->probe_third = (folded_length > 1) ? 1 : 0;
    }
    else {
        sz_u8_t byte_first = refined->folded_slice[0];
        sz_u8_t byte_last = refined->folded_slice[folded_length - 1];
        sz_size_t probe_second = folded_length / 3;
        sz_size_t probe_third = (folded_length * 2) / 3;
        for (sz_size_t byte_index = 1; byte_index < folded_length - 1; ++byte_index) {
            if (refined->folded_slice[byte_index] != byte_first && refined->folded_slice[byte_index] != byte_last) {
                probe_second = byte_index;
                break;
            }
        }
        sz_u8_t byte_second = refined->folded_slice[probe_second];
        for (sz_size_t byte_index = probe_second + 1; byte_index < folded_length - 1; ++byte_index) {
            if (refined->folded_slice[byte_index] != byte_first && refined->folded_slice[byte_index] != byte_last &&
                refined->folded_slice[byte_index] != byte_second) {
                probe_third = byte_index;
                break;
            }
        }
        if (probe_second == 0) probe_second = 1;
        if (probe_third >= folded_length - 1) probe_third = folded_length - 2;
        if (probe_third <= probe_second && probe_second + 1 < folded_length - 1) probe_third = probe_second + 1;
        refined->probe_second = (sz_u8_t)probe_second;
        refined->probe_third = (sz_u8_t)probe_third;
    }
}
#pragma endregion // Character Safety Profiles
#ifdef __cplusplus
}
#endif
#endif 
