/**
 *  @brief Serial backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_uncased_fold/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_SERIAL_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_SERIAL_H_
#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 *  @brief Branchless ASCII case fold - converts A-Z to a-z.
 *  Uses unsigned subtraction trick: (c - 'A') <= 25 is true only for uppercase letters.
 */
SZ_HELPER_AUTO sz_u8_t sz_ascii_fold_(sz_u8_t c) { return c + (((sz_u8_t)(c - 'A') <= 25u) * 0x20); }
/**
 *  @brief Folded-rune representation of a byte that does not begin a well-formed codepoint.
 *
 *  A malformed byte folds to itself and is matched/compared byte-for-byte, never as a Unicode codepoint.
 *  Tagging it above the valid Unicode range (0x10FFFF) keeps it distinct from every real folded rune, so a
 *  lone malformed byte 0xFC can only match another malformed 0xFC - never the valid rune U+00FC ('ü'). Two
 *  equal malformed bytes still produce equal tagged runes, preserving byte-for-byte matching.
 */
SZ_HELPER_INLINE sz_rune_t sz_rune_malformed_byte_(sz_u8_t byte) { return 0x80000000u | (sz_rune_t)byte; }
/**
 *  Bit flags describing which UTF-8 lead-byte families occur in a chunk, shared by every back-end.
 *  Each family shares one folding strategy, so the union of flags picks the chunk handler in a
 *  single dispatch - instead of sequentially probing per-script fast paths, which degrades on
 *  mixed-script text. The flags themselves are ISA-neutral; only the way a back-end derives them
 *  differs - a `VPERMB` lookup on Ice Lake, `vqtbl4q_u8` on NEON, a compare tree on Haswell.
 */
enum sz_utf8_fold_lead_family_t {
    sz_utf8_fold_lead_caseless_flag_k = 1 << 0,       
    sz_utf8_fold_lead_latin_flag_k = 1 << 1,          
    sz_utf8_fold_lead_latin_extended_flag_k = 1 << 2, 
    sz_utf8_fold_lead_cyrillic_flag_k = 1 << 3,       
    sz_utf8_fold_lead_greek_flag_k = 1 << 4,          
    sz_utf8_fold_lead_e1_flag_k = 1 << 5,             
    sz_utf8_fold_lead_guarded_flag_k = 1 << 6,        
    sz_utf8_fold_lead_complex_flag_k = 1 << 7,        
};
/**
 *  Lead-byte family table indexed by the low 6 bits of the lead byte: leads 0xC0-0xFF map onto
 *  indices 0x00-0x3F injectively, and ASCII/continuation bytes are masked out before the lookup.
 *  Byte-for-byte the same 64 values as the Ice Lake `lead_families_lut`, but laid out in
 *  ascending index order: `vqtbl4q_u8` reads its `uint8x16x4_t` table in memory order, whereas
 *  `_mm512_set_epi8` lists lanes 0x3F → 0x00.
 */
static sz_u8_t const sz_utf8_fold_lead_families_lut_[64] = {
    0x80, 0x80, 0x02, 0x02, 0x04, 0x04, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x10, 0x10,
    0x08, 0x08, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x20, 0x40, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x40, 0x01, 0x01, 0x01, 0x01, 0x40,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
};
/**
 *  Per-codepoint deltas for 2-byte Latin Extended sequences, indexed by the continuation byte's
 *  low 6 bits: 0x00 = identity, 0x01 = fold by +1, 0x80 = irregular (serial fallback). The same
 *  64 values as the Ice Lake `c4_deltas_lut`, but in ascending index order: `vqtbl4q_u8` reads
 *  its table in memory order, whereas `_mm512_set_epi8` lists lanes 0x3F → 0x00.
 *  Generated from Unicode full case folding; verified against the serial reference in tests.
 */
static sz_u8_t const sz_utf8_fold_c4_deltas_lut_[64] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
    0x80, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0x80,
};
static sz_u8_t const sz_utf8_fold_c5_deltas_lut_[64] = {
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0x80, 1, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,    
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,    
    1, 0, 1, 0, 1, 0, 1, 0, 0x80, 1, 0, 1, 0, 1, 0, 0x80,
};
static sz_u8_t const sz_utf8_fold_c6_deltas_lut_[64] = {
    0, 0x80, 1, 0, 1, 0, 0x80, 1, 0, 0x80, 0x80, 1, 0, 0, 0x80, 0x80, 
    0x80, 1, 0, 0x80, 0x80, 0, 0x80, 0x80, 1, 0, 0, 0, 0x80, 0x80, 0, 0x80, 
    1, 0, 1, 0, 1, 0, 0x80, 1, 0, 0x80, 0, 0, 1, 0, 0x80, 1, 
    0, 0x80, 0x80, 1, 0, 1, 0, 0x80, 1, 0, 0, 0, 1, 0, 0, 0, 
};
/** @brief  Character boundary of the first stop lane in a 64-byte superchunk: takes the lowest set bit of
 *          @p stop_lanes and walks back over continuation bytes so the consumed prefix ends on a boundary.
 *          Shared u64 mask math for the windowed ISA fold handlers; landing on byte 0 routes one rune to the
 *          serial fallback. */
SZ_HELPER_INLINE sz_size_t sz_utf8_fold_stop_boundary_serial_(sz_u64_t stop_lanes, sz_cptr_t source) {
    sz_size_t first_flagged_position = (sz_size_t)sz_u64_ctz(stop_lanes);
    while (first_flagged_position && ((sz_u8_t)source[first_flagged_position] & 0xC0) == 0x80) --first_flagged_position;
    return first_flagged_position;
}
/**  Helper macro for readable assertions - use for SIMD implementation reference */
#define sz_is_in_range_(x, low, high) ((x) >= (low) && (x) <= (high))
/**
 *  @brief Fold a Unicode codepoint to its case-folded form (Unicode 17.0).
 *
 *  Optimization strategy:
 *  - Single-comparison range checks: `(sz_u32_t)(rune - base) <= size` instead of two comparisons
 *  - Combined upper+lower ranges: check both cases, apply offset only for uppercase (branchless)
 *  - Combined even/odd ranges: check full range, apply +1 only for uppercase parity
 *  - Hierarchical by UTF-8 byte width for early exit on common cases
 *  - Per-section switches for irregular mappings (better compiler optimization)
 *
 *  Each range check includes an assertion with traditional bounds for SIMD implementation reference.
 */
SZ_HELPER_AUTO sz_size_t sz_unicode_fold_codepoint_(sz_rune_t rune, sz_rune_t *folded) {
    if (rune <= 0x7F) {
        if ((sz_u32_t)(rune - 0x41) <= 25) { 
            sz_assert_(sz_is_in_range_(rune, 0x0041, 0x005A));
            folded[0] = rune + 0x20;
            return 1;
        }
        folded[0] = rune;
        return 1; 
    }
    if (rune <= 0x7FF) {
        if ((sz_u32_t)(rune - 0x0410) <= 0x3F) {
            sz_assert_(sz_is_in_range_(rune, 0x0410, 0x044F));
            folded[0] = rune + ((rune <= 0x042F) * 0x20);
            return 1;
        } 
        if ((sz_u32_t)(rune - 0x00C0) <= 0x3E) {
            sz_assert_(sz_is_in_range_(rune, 0x00C0, 0x00FE));
            if ((rune | 0x20) == 0xF7) {
                folded[0] = rune;
                return 1;
            } 
            if (rune == 0x00DF) {
                folded[0] = 0x0073;
                folded[1] = 0x0073;
                return 2;
            }
            folded[0] = rune + ((rune <= 0x00DE) * 0x20);
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0391) <= 0x10) {
            sz_assert_(sz_is_in_range_(rune, 0x0391, 0x03A1));
            folded[0] = rune + 0x20;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x03A3) <= 0x08) {
            sz_assert_(sz_is_in_range_(rune, 0x03A3, 0x03AB));
            folded[0] = rune + 0x20;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0400) <= 0x0F) {
            sz_assert_(sz_is_in_range_(rune, 0x0400, 0x040F));
            folded[0] = rune + 0x50;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0531) <= 0x25) {
            sz_assert_(sz_is_in_range_(rune, 0x0531, 0x0556));
            folded[0] = rune + 0x30;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0388) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x0388, 0x038A));
            folded[0] = rune + 0x25;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x03FD) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x03FD, 0x03FF));
            folded[0] = rune - 130;
            return 1;
        }
        sz_u32_t is_even = ((rune & 1) == 0);
        if ((sz_u32_t)(rune - 0x0100) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0100, 0x012E));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0132) <= 0x04 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0132, 0x0136));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0139) <= 0x0E && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0139, 0x0147));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x014A) <= 0x2C && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x014A, 0x0176));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0179) <= 0x04 && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0179, 0x017D));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x01CD) <= 0x0E && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01CD, 0x01DB));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x01DE) <= 0x10 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01DE, 0x01EE));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x01F8) <= 0x06 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01F8, 0x01FE));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0200) <= 0x1E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0200, 0x021E));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0222) <= 0x10 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0222, 0x0232));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0246) <= 0x08 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0246, 0x024E));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0370) <= 0x02 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0370, 0x0372));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x03D8) <= 0x16 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x03D8, 0x03EE));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0460) <= 0x20 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0460, 0x0480));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x048A) <= 0x34 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x048A, 0x04BE));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x04C1) <= 0x0C && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x04C1, 0x04CD));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x04D0) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x04D0, 0x04FE));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x0500) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0500, 0x052E));
            folded[0] = rune + 1;
            return 1;
        }
        switch (rune) {
        case 0x00B5: folded[0] = 0x03BC; return 1; 
        case 0x0178: folded[0] = 0x00FF; return 1; 
        case 0x017F:
            folded[0] = 0x0073;
            return 1; 
        case 0x0181: folded[0] = 0x0253; return 1; 
        case 0x0182: folded[0] = 0x0183; return 1; 
        case 0x0184: folded[0] = 0x0185; return 1; 
        case 0x0186: folded[0] = 0x0254; return 1; 
        case 0x0187: folded[0] = 0x0188; return 1; 
        case 0x0189: folded[0] = 0x0256; return 1; 
        case 0x018A: folded[0] = 0x0257; return 1; 
        case 0x018B: folded[0] = 0x018C; return 1; 
        case 0x018E: folded[0] = 0x01DD; return 1; 
        case 0x018F: folded[0] = 0x0259; return 1; 
        case 0x0190: folded[0] = 0x025B; return 1; 
        case 0x0191: folded[0] = 0x0192; return 1; 
        case 0x0193: folded[0] = 0x0260; return 1; 
        case 0x0194: folded[0] = 0x0263; return 1; 
        case 0x0196: folded[0] = 0x0269; return 1; 
        case 0x0197: folded[0] = 0x0268; return 1; 
        case 0x0198: folded[0] = 0x0199; return 1; 
        case 0x019C: folded[0] = 0x026F; return 1; 
        case 0x019D: folded[0] = 0x0272; return 1; 
        case 0x019F: folded[0] = 0x0275; return 1; 
        case 0x01A0: folded[0] = 0x01A1; return 1; 
        case 0x01A2: folded[0] = 0x01A3; return 1; 
        case 0x01A4: folded[0] = 0x01A5; return 1; 
        case 0x01A6: folded[0] = 0x0280; return 1; 
        case 0x01A7: folded[0] = 0x01A8; return 1; 
        case 0x01A9: folded[0] = 0x0283; return 1; 
        case 0x01AC: folded[0] = 0x01AD; return 1; 
        case 0x01AE: folded[0] = 0x0288; return 1; 
        case 0x01AF: folded[0] = 0x01B0; return 1; 
        case 0x01B1: folded[0] = 0x028A; return 1; 
        case 0x01B2: folded[0] = 0x028B; return 1; 
        case 0x01B3: folded[0] = 0x01B4; return 1; 
        case 0x01B5: folded[0] = 0x01B6; return 1; 
        case 0x01B7: folded[0] = 0x0292; return 1; 
        case 0x01B8: folded[0] = 0x01B9; return 1; 
        case 0x01BC:
            folded[0] = 0x01BD;
            return 1; 
        case 0x01C4: folded[0] = 0x01C6; return 1; 
        case 0x01C5: folded[0] = 0x01C6; return 1; 
        case 0x01C7: folded[0] = 0x01C9; return 1; 
        case 0x01C8: folded[0] = 0x01C9; return 1; 
        case 0x01CA: folded[0] = 0x01CC; return 1; 
        case 0x01CB: folded[0] = 0x01CC; return 1; 
        case 0x01F1: folded[0] = 0x01F3; return 1; 
        case 0x01F2:
            folded[0] = 0x01F3;
            return 1; 
        case 0x01F4: folded[0] = 0x01F5; return 1; 
        case 0x01F6: folded[0] = 0x0195; return 1; 
        case 0x01F7: folded[0] = 0x01BF; return 1; 
        case 0x0220: folded[0] = 0x019E; return 1; 
        case 0x023A: folded[0] = 0x2C65; return 1; 
        case 0x023B: folded[0] = 0x023C; return 1; 
        case 0x023D: folded[0] = 0x019A; return 1; 
        case 0x023E: folded[0] = 0x2C66; return 1; 
        case 0x0241: folded[0] = 0x0242; return 1; 
        case 0x0243: folded[0] = 0x0180; return 1; 
        case 0x0244: folded[0] = 0x0289; return 1; 
        case 0x0245:
            folded[0] = 0x028C;
            return 1; 
        case 0x0345: folded[0] = 0x03B9; return 1; 
        case 0x0376: folded[0] = 0x0377; return 1; 
        case 0x037F: folded[0] = 0x03F3; return 1; 
        case 0x0386: folded[0] = 0x03AC; return 1; 
        case 0x038C: folded[0] = 0x03CC; return 1; 
        case 0x038E: folded[0] = 0x03CD; return 1; 
        case 0x038F: folded[0] = 0x03CE; return 1; 
        case 0x03C2: folded[0] = 0x03C3; return 1; 
        case 0x03CF: folded[0] = 0x03D7; return 1; 
        case 0x03D0: folded[0] = 0x03B2; return 1; 
        case 0x03D1: folded[0] = 0x03B8; return 1; 
        case 0x03D5: folded[0] = 0x03C6; return 1; 
        case 0x03D6: folded[0] = 0x03C0; return 1; 
        case 0x03F0: folded[0] = 0x03BA; return 1; 
        case 0x03F1: folded[0] = 0x03C1; return 1; 
        case 0x03F4: folded[0] = 0x03B8; return 1; 
        case 0x03F5: folded[0] = 0x03B5; return 1; 
        case 0x03F7: folded[0] = 0x03F8; return 1; 
        case 0x03F9: folded[0] = 0x03F2; return 1; 
        case 0x03FA:
            folded[0] = 0x03FB;
            return 1; 
        case 0x04C0: folded[0] = 0x04CF; return 1; 
        }
        switch (rune) {
        case 0x0130:
            folded[0] = 0x0069;
            folded[1] = 0x0307;
            return 2;
        case 0x0149:
            folded[0] = 0x02BC;
            folded[1] = 0x006E;
            return 2;
        case 0x01F0:
            folded[0] = 0x006A;
            folded[1] = 0x030C;
            return 2;
        case 0x0390:
            folded[0] = 0x03B9;
            folded[1] = 0x0308;
            folded[2] = 0x0301;
            return 3;
        case 0x03B0:
            folded[0] = 0x03C5;
            folded[1] = 0x0308;
            folded[2] = 0x0301;
            return 3;
        case 0x0587:
            folded[0] = 0x0565;
            folded[1] = 0x0582;
            return 2;
        }
        folded[0] = rune;
        return 1; 
    }
    if (rune <= 0xFFFF) {
        if ((sz_u32_t)(rune - 0x10A0) <= 0x25) {
            sz_assert_(sz_is_in_range_(rune, 0x10A0, 0x10C5));
            folded[0] = rune + 0x1C60;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1C90) <= 0x2A) {
            sz_assert_(sz_is_in_range_(rune, 0x1C90, 0x1CBA));
            folded[0] = rune - 0xBC0;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1CBD) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x1CBD, 0x1CBF));
            folded[0] = rune - 0xBC0;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x13F8) <= 0x05) {
            sz_assert_(sz_is_in_range_(rune, 0x13F8, 0x13FD));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xAB70) <= 0x4F) {
            sz_assert_(sz_is_in_range_(rune, 0xAB70, 0xABBF));
            folded[0] = rune - 0x97D0;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F08) <= 0x07) { 
            sz_assert_(sz_is_in_range_(rune, 0x1F08, 0x1F0F));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F18) <= 0x05) { 
            sz_assert_(sz_is_in_range_(rune, 0x1F18, 0x1F1D));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F28) <= 0x07) { 
            sz_assert_(sz_is_in_range_(rune, 0x1F28, 0x1F2F));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F38) <= 0x07) { 
            sz_assert_(sz_is_in_range_(rune, 0x1F38, 0x1F3F));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F48) <= 0x05) { 
            sz_assert_(sz_is_in_range_(rune, 0x1F48, 0x1F4D));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F68) <= 0x07) { 
            sz_assert_(sz_is_in_range_(rune, 0x1F68, 0x1F6F));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1FC8) <= 0x03) {
            sz_assert_(sz_is_in_range_(rune, 0x1FC8, 0x1FCB));
            folded[0] = rune - 86;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x2160) <= 0x0F) {
            sz_assert_(sz_is_in_range_(rune, 0x2160, 0x216F));
            folded[0] = rune + 0x10;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x24B6) <= 0x19) {
            sz_assert_(sz_is_in_range_(rune, 0x24B6, 0x24CF));
            folded[0] = rune + 0x1A;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x2C00) <= 0x2F) {
            sz_assert_(sz_is_in_range_(rune, 0x2C00, 0x2C2F));
            folded[0] = rune + 0x30;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xFF21) <= 0x19) {
            sz_assert_(sz_is_in_range_(rune, 0xFF21, 0xFF3A));
            folded[0] = rune + 0x20;
            return 1;
        }
        sz_u32_t is_even = ((rune & 1) == 0);
        if ((sz_u32_t)(rune - 0x1E00) <= 0x94 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x1E00, 0x1E94));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1EA0) <= 0x5E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x1EA0, 0x1EFE));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x2C80) <= 0x62 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x2C80, 0x2CE2));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA640) <= 0x2C && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0xA640, 0xA66C));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA680) <= 0x1A && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0xA680, 0xA69A));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA722) <= 0x0C && is_even) { 
            sz_assert_(sz_is_in_range_(rune, 0xA722, 0xA72E));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA732) <= 0x3C && is_even) { 
            sz_assert_(sz_is_in_range_(rune, 0xA732, 0xA76E));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA77E) <= 0x08 && is_even) { 
            sz_assert_(sz_is_in_range_(rune, 0xA77E, 0xA786));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA790) <= 0x02 && is_even) { 
            sz_assert_(sz_is_in_range_(rune, 0xA790, 0xA792));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA796) <= 0x12 && is_even) { 
            sz_assert_(sz_is_in_range_(rune, 0xA796, 0xA7A8));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA7B4) <= 0x0E && is_even) { 
            sz_assert_(sz_is_in_range_(rune, 0xA7B4, 0xA7C2));
            folded[0] = rune + 1;
            return 1;
        }
        switch (rune) {
        case 0x10C7: folded[0] = 0x2D27; return 1; 
        case 0x10CD:
            folded[0] = 0x2D2D;
            return 1; 
        case 0x1C80: folded[0] = 0x0432; return 1; 
        case 0x1C81: folded[0] = 0x0434; return 1; 
        case 0x1C82: folded[0] = 0x043E; return 1; 
        case 0x1C83: folded[0] = 0x0441; return 1; 
        case 0x1C84: folded[0] = 0x0442; return 1; 
        case 0x1C85: folded[0] = 0x0442; return 1; 
        case 0x1C86: folded[0] = 0x044A; return 1; 
        case 0x1C87: folded[0] = 0x0463; return 1; 
        case 0x1C88: folded[0] = 0xA64B; return 1; 
        case 0x1C89:
            folded[0] = 0x1C8A;
            return 1; 
        case 0x1E9B:
            folded[0] = 0x1E61;
            return 1; 
        case 0x1F59: folded[0] = 0x1F51; return 1; 
        case 0x1F5B: folded[0] = 0x1F53; return 1; 
        case 0x1F5D: folded[0] = 0x1F55; return 1; 
        case 0x1F5F: folded[0] = 0x1F57; return 1; 
        case 0x1FB8: folded[0] = 0x1FB0; return 1; 
        case 0x1FB9: folded[0] = 0x1FB1; return 1; 
        case 0x1FBA: folded[0] = 0x1F70; return 1; 
        case 0x1FBB: folded[0] = 0x1F71; return 1; 
        case 0x1FBE: folded[0] = 0x03B9; return 1; 
        case 0x1FD8: folded[0] = 0x1FD0; return 1; 
        case 0x1FD9: folded[0] = 0x1FD1; return 1; 
        case 0x1FDA: folded[0] = 0x1F76; return 1; 
        case 0x1FDB: folded[0] = 0x1F77; return 1; 
        case 0x1FE8: folded[0] = 0x1FE0; return 1; 
        case 0x1FE9: folded[0] = 0x1FE1; return 1; 
        case 0x1FEA: folded[0] = 0x1F7A; return 1; 
        case 0x1FEB: folded[0] = 0x1F7B; return 1; 
        case 0x1FEC: folded[0] = 0x1FE5; return 1; 
        case 0x1FF8: folded[0] = 0x1F78; return 1; 
        case 0x1FF9: folded[0] = 0x1F79; return 1; 
        case 0x1FFA: folded[0] = 0x1F7C; return 1; 
        case 0x1FFB:
            folded[0] = 0x1F7D;
            return 1; 
        case 0x2126: folded[0] = 0x03C9; return 1; 
        case 0x212A: folded[0] = 0x006B; return 1; 
        case 0x212B: folded[0] = 0x00E5; return 1; 
        case 0x2132: folded[0] = 0x214E; return 1; 
        case 0x2183:
            folded[0] = 0x2184;
            return 1; 
        case 0x2C60: folded[0] = 0x2C61; return 1; 
        case 0x2C62: folded[0] = 0x026B; return 1; 
        case 0x2C63: folded[0] = 0x1D7D; return 1; 
        case 0x2C64: folded[0] = 0x027D; return 1; 
        case 0x2C67: folded[0] = 0x2C68; return 1; 
        case 0x2C69: folded[0] = 0x2C6A; return 1; 
        case 0x2C6B: folded[0] = 0x2C6C; return 1; 
        case 0x2C6D: folded[0] = 0x0251; return 1; 
        case 0x2C6E: folded[0] = 0x0271; return 1; 
        case 0x2C6F: folded[0] = 0x0250; return 1; 
        case 0x2C70: folded[0] = 0x0252; return 1; 
        case 0x2C72: folded[0] = 0x2C73; return 1; 
        case 0x2C75: folded[0] = 0x2C76; return 1; 
        case 0x2C7E: folded[0] = 0x023F; return 1; 
        case 0x2C7F:
            folded[0] = 0x0240;
            return 1; 
        case 0x2CEB: folded[0] = 0x2CEC; return 1; 
        case 0x2CED: folded[0] = 0x2CEE; return 1; 
        case 0x2CF2:
            folded[0] = 0x2CF3;
            return 1; 
        case 0xA779: folded[0] = 0xA77A; return 1; 
        case 0xA77B: folded[0] = 0xA77C; return 1; 
        case 0xA77D: folded[0] = 0x1D79; return 1; 
        case 0xA78B: folded[0] = 0xA78C; return 1; 
        case 0xA78D: folded[0] = 0x0265; return 1; 
        case 0xA7AA: folded[0] = 0x0266; return 1; 
        case 0xA7AB: folded[0] = 0x025C; return 1; 
        case 0xA7AC: folded[0] = 0x0261; return 1; 
        case 0xA7AD: folded[0] = 0x026C; return 1; 
        case 0xA7AE: folded[0] = 0x026A; return 1; 
        case 0xA7B0: folded[0] = 0x029E; return 1; 
        case 0xA7B1: folded[0] = 0x0287; return 1; 
        case 0xA7B2: folded[0] = 0x029D; return 1; 
        case 0xA7B3: folded[0] = 0xAB53; return 1; 
        case 0xA7C4: folded[0] = 0xA794; return 1; 
        case 0xA7C5: folded[0] = 0x0282; return 1; 
        case 0xA7C6: folded[0] = 0x1D8E; return 1; 
        case 0xA7C7: folded[0] = 0xA7C8; return 1; 
        case 0xA7C9: folded[0] = 0xA7CA; return 1; 
        case 0xA7CB: folded[0] = 0x0264; return 1; 
        case 0xA7CC: folded[0] = 0xA7CD; return 1; 
        case 0xA7CE: folded[0] = 0xA7CF; return 1; 
        case 0xA7D0: folded[0] = 0xA7D1; return 1; 
        case 0xA7D2: folded[0] = 0xA7D3; return 1; 
        case 0xA7D4: folded[0] = 0xA7D5; return 1; 
        case 0xA7D6: folded[0] = 0xA7D7; return 1; 
        case 0xA7D8: folded[0] = 0xA7D9; return 1; 
        case 0xA7DA: folded[0] = 0xA7DB; return 1; 
        case 0xA7DC: folded[0] = 0x019B; return 1; 
        case 0xA7F5: folded[0] = 0xA7F6; return 1; 
        }
        switch (rune) {
        case 0x1E96:
            folded[0] = 0x0068;
            folded[1] = 0x0331;
            return 2;
        case 0x1E97:
            folded[0] = 0x0074;
            folded[1] = 0x0308;
            return 2;
        case 0x1E98:
            folded[0] = 0x0077;
            folded[1] = 0x030A;
            return 2;
        case 0x1E99:
            folded[0] = 0x0079;
            folded[1] = 0x030A;
            return 2;
        case 0x1E9A:
            folded[0] = 0x0061;
            folded[1] = 0x02BE;
            return 2;
        case 0x1E9E:
            folded[0] = 0x0073;
            folded[1] = 0x0073;
            return 2;
        case 0x1F50:
            folded[0] = 0x03C5;
            folded[1] = 0x0313;
            return 2;
        case 0x1F52:
            folded[0] = 0x03C5;
            folded[1] = 0x0313;
            folded[2] = 0x0300;
            return 3;
        case 0x1F54:
            folded[0] = 0x03C5;
            folded[1] = 0x0313;
            folded[2] = 0x0301;
            return 3;
        case 0x1F56:
            folded[0] = 0x03C5;
            folded[1] = 0x0313;
            folded[2] = 0x0342;
            return 3;
        case 0x1F80:
            folded[0] = 0x1F00;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F81:
            folded[0] = 0x1F01;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F82:
            folded[0] = 0x1F02;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F83:
            folded[0] = 0x1F03;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F84:
            folded[0] = 0x1F04;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F85:
            folded[0] = 0x1F05;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F86:
            folded[0] = 0x1F06;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F87:
            folded[0] = 0x1F07;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F88:
            folded[0] = 0x1F00;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F89:
            folded[0] = 0x1F01;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F8A:
            folded[0] = 0x1F02;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F8B:
            folded[0] = 0x1F03;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F8C:
            folded[0] = 0x1F04;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F8D:
            folded[0] = 0x1F05;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F8E:
            folded[0] = 0x1F06;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F8F:
            folded[0] = 0x1F07;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F90:
            folded[0] = 0x1F20;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F91:
            folded[0] = 0x1F21;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F92:
            folded[0] = 0x1F22;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F93:
            folded[0] = 0x1F23;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F94:
            folded[0] = 0x1F24;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F95:
            folded[0] = 0x1F25;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F96:
            folded[0] = 0x1F26;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F97:
            folded[0] = 0x1F27;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F98:
            folded[0] = 0x1F20;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F99:
            folded[0] = 0x1F21;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F9A:
            folded[0] = 0x1F22;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F9B:
            folded[0] = 0x1F23;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F9C:
            folded[0] = 0x1F24;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F9D:
            folded[0] = 0x1F25;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F9E:
            folded[0] = 0x1F26;
            folded[1] = 0x03B9;
            return 2;
        case 0x1F9F:
            folded[0] = 0x1F27;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FA0:
            folded[0] = 0x1F60;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FA1:
            folded[0] = 0x1F61;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FA2:
            folded[0] = 0x1F62;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FA3:
            folded[0] = 0x1F63;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FA4:
            folded[0] = 0x1F64;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FA5:
            folded[0] = 0x1F65;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FA6:
            folded[0] = 0x1F66;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FA7:
            folded[0] = 0x1F67;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FA8:
            folded[0] = 0x1F60;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FA9:
            folded[0] = 0x1F61;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FAA:
            folded[0] = 0x1F62;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FAB:
            folded[0] = 0x1F63;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FAC:
            folded[0] = 0x1F64;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FAD:
            folded[0] = 0x1F65;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FAE:
            folded[0] = 0x1F66;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FAF:
            folded[0] = 0x1F67;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FB2:
            folded[0] = 0x1F70;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FB3:
            folded[0] = 0x03B1;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FB4:
            folded[0] = 0x03AC;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FB6:
            folded[0] = 0x03B1;
            folded[1] = 0x0342;
            return 2;
        case 0x1FB7:
            folded[0] = 0x03B1;
            folded[1] = 0x0342;
            folded[2] = 0x03B9;
            return 3;
        case 0x1FBC:
            folded[0] = 0x03B1;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FC2:
            folded[0] = 0x1F74;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FC3:
            folded[0] = 0x03B7;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FC4:
            folded[0] = 0x03AE;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FC6:
            folded[0] = 0x03B7;
            folded[1] = 0x0342;
            return 2;
        case 0x1FC7:
            folded[0] = 0x03B7;
            folded[1] = 0x0342;
            folded[2] = 0x03B9;
            return 3;
        case 0x1FCC:
            folded[0] = 0x03B7;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FD2:
            folded[0] = 0x03B9;
            folded[1] = 0x0308;
            folded[2] = 0x0300;
            return 3;
        case 0x1FD3:
            folded[0] = 0x03B9;
            folded[1] = 0x0308;
            folded[2] = 0x0301;
            return 3;
        case 0x1FD6:
            folded[0] = 0x03B9;
            folded[1] = 0x0342;
            return 2;
        case 0x1FD7:
            folded[0] = 0x03B9;
            folded[1] = 0x0308;
            folded[2] = 0x0342;
            return 3;
        case 0x1FE2:
            folded[0] = 0x03C5;
            folded[1] = 0x0308;
            folded[2] = 0x0300;
            return 3;
        case 0x1FE3:
            folded[0] = 0x03C5;
            folded[1] = 0x0308;
            folded[2] = 0x0301;
            return 3;
        case 0x1FE4:
            folded[0] = 0x03C1;
            folded[1] = 0x0313;
            return 2;
        case 0x1FE6:
            folded[0] = 0x03C5;
            folded[1] = 0x0342;
            return 2;
        case 0x1FE7:
            folded[0] = 0x03C5;
            folded[1] = 0x0308;
            folded[2] = 0x0342;
            return 3;
        case 0x1FF2:
            folded[0] = 0x1F7C;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FF3:
            folded[0] = 0x03C9;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FF4:
            folded[0] = 0x03CE;
            folded[1] = 0x03B9;
            return 2;
        case 0x1FF6:
            folded[0] = 0x03C9;
            folded[1] = 0x0342;
            return 2;
        case 0x1FF7:
            folded[0] = 0x03C9;
            folded[1] = 0x0342;
            folded[2] = 0x03B9;
            return 3;
        case 0x1FFC:
            folded[0] = 0x03C9;
            folded[1] = 0x03B9;
            return 2;
        case 0xFB00:
            folded[0] = 0x0066;
            folded[1] = 0x0066;
            return 2;
        case 0xFB01:
            folded[0] = 0x0066;
            folded[1] = 0x0069;
            return 2;
        case 0xFB02:
            folded[0] = 0x0066;
            folded[1] = 0x006C;
            return 2;
        case 0xFB03:
            folded[0] = 0x0066;
            folded[1] = 0x0066;
            folded[2] = 0x0069;
            return 3;
        case 0xFB04:
            folded[0] = 0x0066;
            folded[1] = 0x0066;
            folded[2] = 0x006C;
            return 3;
        case 0xFB05:
            folded[0] = 0x0073;
            folded[1] = 0x0074;
            return 2;
        case 0xFB06:
            folded[0] = 0x0073;
            folded[1] = 0x0074;
            return 2;
        case 0xFB13:
            folded[0] = 0x0574;
            folded[1] = 0x0576;
            return 2;
        case 0xFB14:
            folded[0] = 0x0574;
            folded[1] = 0x0565;
            return 2;
        case 0xFB15:
            folded[0] = 0x0574;
            folded[1] = 0x056B;
            return 2;
        case 0xFB16:
            folded[0] = 0x057E;
            folded[1] = 0x0576;
            return 2;
        case 0xFB17:
            folded[0] = 0x0574;
            folded[1] = 0x056D;
            return 2;
        }
        folded[0] = rune;
        return 1; 
    }
    if ((sz_u32_t)(rune - 0x10400) <= 0x27) {
        sz_assert_(sz_is_in_range_(rune, 0x10400, 0x10427));
        folded[0] = rune + 0x28;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x104B0) <= 0x23) {
        sz_assert_(sz_is_in_range_(rune, 0x104B0, 0x104D3));
        folded[0] = rune + 0x28;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x10570) <= 0x0A) { 
        sz_assert_(sz_is_in_range_(rune, 0x10570, 0x1057A));
        folded[0] = rune + 0x27;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x1057C) <= 0x0E) { 
        sz_assert_(sz_is_in_range_(rune, 0x1057C, 0x1058A));
        folded[0] = rune + 0x27;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x1058C) <= 0x06) { 
        sz_assert_(sz_is_in_range_(rune, 0x1058C, 0x10592));
        folded[0] = rune + 0x27;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x10C80) <= 0x32) {
        sz_assert_(sz_is_in_range_(rune, 0x10C80, 0x10CB2));
        folded[0] = rune + 0x40;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x10D50) <= 0x15) {
        sz_assert_(sz_is_in_range_(rune, 0x10D50, 0x10D65));
        folded[0] = rune + 0x20;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x118A0) <= 0x1F) {
        sz_assert_(sz_is_in_range_(rune, 0x118A0, 0x118BF));
        folded[0] = rune + 0x20;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x16E40) <= 0x1F) {
        sz_assert_(sz_is_in_range_(rune, 0x16E40, 0x16E5F));
        folded[0] = rune + 0x20;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x16EA0) <= 0x18) {
        sz_assert_(sz_is_in_range_(rune, 0x16EA0, 0x16EB8));
        folded[0] = rune + 0x1B;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x1E900) <= 0x21) {
        sz_assert_(sz_is_in_range_(rune, 0x1E900, 0x1E921));
        folded[0] = rune + 0x22;
        return 1;
    }
    switch (rune) {
    case 0x10594: folded[0] = 0x105BB; return 1; 
    case 0x10595: folded[0] = 0x105BC; return 1; 
    }
    folded[0] = rune;
    return 1; 
}
/**
 *  @brief Helper function performing case-folding under the constraint, that no output may be incomplete.
 *
 *  @param source Pointer to the source UTF-8 data, must be valid UTF-8.
 *  @param source_length Length of the source data in bytes.
 *  @param destination Pointer to the destination buffer.
 *  @param destination_length Length of the destination buffer in bytes.
 *  @param codepoints_consumed Number of codepoints read from source.
 *  @param codepoints_exported Number of codepoints written to destination.
 *  @param bytes_consumed Number of bytes read from source.
 *  @param bytes_exported Number of bytes written to destination.
 */
SZ_HELPER_AUTO void sz_utf8_uncased_fold_upto_(                     
    sz_cptr_t source, sz_size_t source_length,                      
    sz_ptr_t destination, sz_size_t destination_length,             
    sz_size_t *codepoints_consumed, sz_size_t *codepoints_exported, 
    sz_size_t *bytes_consumed, sz_size_t *bytes_exported) {
    sz_u8_t const *const source_start = (sz_u8_t const *)source;
    sz_u8_t const *const source_limit = source_start + source_length;
    sz_u8_t *const destination_start = (sz_u8_t *)destination;
    sz_u8_t *const destination_limit = destination_start + destination_length;
    sz_u8_t const *source_ptr = source_start;
    sz_u8_t *destination_ptr = destination_start;
    sz_size_t codepoints_read = 0;
    sz_size_t codepoints_written = 0;
    while (source_ptr < source_limit && destination_ptr < destination_limit) {
        while (source_ptr < source_limit && destination_ptr < destination_limit && *source_ptr < 0x80) {
            *destination_ptr++ = sz_ascii_fold_(*source_ptr++);
            codepoints_read++;
            codepoints_written++;
        }
        if (source_ptr >= source_limit || destination_ptr >= destination_limit) break;
        sz_rune_t source_rune;
        sz_rune_length_t const source_rune_length = sz_rune_decode((sz_cptr_t)source_ptr, (sz_cptr_t)source_limit,
                                                                   &source_rune);
        if (source_rune_length == sz_rune_invalid_k) {
            *destination_ptr++ = *source_ptr++; 
            codepoints_read++, codepoints_written++;
            continue;
        }
        sz_rune_t target_runes[3];
        sz_size_t target_runes_count = sz_unicode_fold_codepoint_(source_rune, target_runes);
        sz_u8_t target_bytes[12]; 
        sz_size_t target_bytes_count = 0;
        for (sz_size_t rune_index = 0; rune_index < target_runes_count; ++rune_index)
            target_bytes_count += sz_rune_encode(target_runes[rune_index], target_bytes + target_bytes_count);
        if (destination_ptr + target_bytes_count > destination_limit) break;
        for (sz_size_t byte_index = 0; byte_index < target_bytes_count; ++byte_index)
            *destination_ptr++ = target_bytes[byte_index];
        source_ptr += source_rune_length;
        codepoints_read++;
        codepoints_written += target_runes_count;
    }
    if (codepoints_consumed) *codepoints_consumed = codepoints_read;
    if (codepoints_exported) *codepoints_exported = codepoints_written;
    if (bytes_consumed) *bytes_consumed = (sz_size_t)(source_ptr - source_start);
    if (bytes_exported) *bytes_exported = (sz_size_t)(destination_ptr - destination_start);
}
SZ_API_COMPTIME sz_size_t sz_utf8_uncased_fold_serial(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    sz_u8_t const *source_ptr = (sz_u8_t const *)source;
    sz_u8_t const *source_end = source_ptr + source_length;
    sz_u8_t *destination_ptr = (sz_u8_t *)destination;
    while (source_ptr < source_end) {
        while (source_ptr < source_end && *source_ptr < 0x80) *destination_ptr++ = sz_ascii_fold_(*source_ptr++);
        if (source_ptr >= source_end) break;
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode((sz_cptr_t)source_ptr, (sz_cptr_t)source_end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            *destination_ptr++ = *source_ptr++;
            continue;
        }
        source_ptr += rune_length;
        sz_rune_t folded_runes[3]; 
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
            destination_ptr += sz_rune_encode(folded_runes[rune_index], destination_ptr);
    }
    return (sz_size_t)(destination_ptr - (sz_u8_t *)destination);
}
#ifdef __cplusplus
}
#endif
#endif 
