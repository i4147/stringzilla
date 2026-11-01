/**
 *  @brief RISC-V Vector backend for UTF-8 case folding (delegates to serial).
 *  @file include/stringzilla/utf8_uncased_fold/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_RVV_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_RVV_H_
#include "stringzilla/utf8_uncased_fold/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if SZ_USE_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif
static sz_u8_t const sz_utf8_fold_latin_c456_deltas_rvv_[192] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,       
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,       
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,       
    0x80, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0x80, 
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0x80, 1, 0, 1, 0, 1, 0,    
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,       
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,       
    1, 0, 1, 0, 1, 0, 1, 0, 0x80, 1, 0, 1, 0, 1, 0, 0x80, 
    0, 0x80, 1, 0, 1, 0, 0x80, 1, 0, 0x80, 0x80, 1, 0, 0, 0x80, 0x80,       
    0x80, 1, 0, 0x80, 0x80, 0, 0x80, 0x80, 1, 0, 0, 0, 0x80, 0x80, 0, 0x80, 
    1, 0, 1, 0, 1, 0, 0x80, 1, 0, 0x80, 0, 0, 1, 0, 0x80, 1,                
    0, 0x80, 0x80, 1, 0, 1, 0, 0x80, 1, 0, 0, 0, 1, 0, 0, 0};
static sz_u8_t const *const sz_utf8_fold_latin_c4_deltas_rvv_ = sz_utf8_fold_latin_c456_deltas_rvv_ + 0;
static sz_u8_t const *const sz_utf8_fold_latin_c5_deltas_rvv_ = sz_utf8_fold_latin_c456_deltas_rvv_ + 64;
static sz_u8_t const *const sz_utf8_fold_latin_c6_deltas_rvv_ = sz_utf8_fold_latin_c456_deltas_rvv_ + 128;
SZ_HELPER_INLINE vuint8m8_t sz_utf8_fold_ascii_rvv_(vuint8m8_t source_u8m8, sz_size_t vector_length) {
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t lowered_u8m8 = __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length);
    return __riscv_vmerge_vvm_u8m8(source_u8m8, lowered_u8m8, is_upper_b1, vector_length);
}
SZ_HELPER_AUTO sz_size_t sz_utf8_fold_trim_incomplete_(sz_u8_t const *source_ptr, sz_size_t vector_length,
                                                       sz_size_t remaining) {
    if (vector_length >= remaining) return vector_length;
    sz_size_t boundary = vector_length;
    while (boundary && (source_ptr[boundary - 1] & 0xC0) == 0x80) --boundary; 
    if (!boundary) return vector_length;
    sz_u8_t lead = source_ptr[boundary - 1];
    sz_size_t needed = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    return ((boundary - 1) + needed > vector_length) ? (boundary - 1) : vector_length;
}
SZ_HELPER_AUTO sz_size_t sz_utf8_fold_latin_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                       sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    sz_u8_t next_carry = (vector_length < remaining) ? source_ptr[vector_length] : 0;
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);
    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t after_c2_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xC2, vector_length);
    vbool1_t after_c3_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xC3, vector_length);
    vbool1_t after_c456_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(previous_u8m8, 0xC4, vector_length), 2,
                                                       vector_length);
    vbool1_t delta_lanes_b1 = __riscv_vmand_mm_b1(after_c456_b1, is_continuation_b1, vector_length);
    vuint8m8_t low6_u8m8 = __riscv_vand_vx_u8m8(source_u8m8, 0x3F, vector_length);
    vuint8m8_t family_base_u8m8 = __riscv_vmul_vx_u8m8(__riscv_vsub_vx_u8m8(previous_u8m8, 0xC4, vector_length), 64,
                                                       vector_length);
    vuint8m8_t lut_index_u8m8 = __riscv_vadd_vv_u8m8(family_base_u8m8, low6_u8m8, vector_length);
    vuint8m8_t delta_u8m8 = __riscv_vmv_v_x_u8m8(0, vector_length);
    delta_u8m8 = __riscv_vluxei8_v_u8m8_mu(delta_lanes_b1, delta_u8m8, sz_utf8_fold_latin_c456_deltas_rvv_,
                                           lut_index_u8m8, vector_length);
    vbool1_t is_irregular_b1 = __riscv_vmsne_vx_u8m8_b1(__riscv_vand_vx_u8m8(delta_u8m8, 0x80, vector_length), 0,
                                                        vector_length);
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    vbool1_t latin1_range_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0x80, vector_length), 0x1F,
                                                         vector_length);
    vbool1_t is_latin1_upper_b1 = __riscv_vmandn_mm_b1(__riscv_vmand_mm_b1(after_c3_b1, latin1_range_b1, vector_length),
                                                       __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0x97, vector_length),
                                                       vector_length);
    folded_u8m8 = __riscv_vmerge_vvm_u8m8(folded_u8m8, __riscv_vadd_vx_u8m8(folded_u8m8, 0x20, vector_length),
                                          is_latin1_upper_b1, vector_length);
    vbool1_t eszett_lead_b1 = __riscv_vmand_mm_b1(__riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xC3, vector_length),
                                                  __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x9F, vector_length),
                                                  vector_length);
    vbool1_t eszett_second_b1 = __riscv_vmand_mm_b1(
        after_c3_b1, __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0x9F, vector_length), vector_length);
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(
        folded_u8m8, 's', __riscv_vmor_mm_b1(eszett_lead_b1, eszett_second_b1, vector_length), vector_length);
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(
        folded_u8m8, 0xCE,
        __riscv_vmand_mm_b1(__riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xC2, vector_length),
                            __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0xB5, vector_length), vector_length),
        vector_length);
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(
        folded_u8m8, 0xBC,
        __riscv_vmand_mm_b1(after_c2_b1, __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xB5, vector_length), vector_length),
        vector_length);
    folded_u8m8 = __riscv_vadd_vv_u8m8(folded_u8m8, delta_u8m8, vector_length);
    vbool1_t is_non_ascii_b1 = __riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length);
    vbool1_t is_lead_b1 = __riscv_vmandn_mm_b1(is_non_ascii_b1, is_continuation_b1, vector_length);
    vbool1_t is_family_lead_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0xC2, vector_length), 4,
                                                           vector_length);
    vbool1_t is_foreign_lead_b1 = __riscv_vmandn_mm_b1(is_lead_b1, is_family_lead_b1, vector_length);
    vbool1_t next_is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(next_u8m8, 0xC0, vector_length),
                                                                0x80, vector_length);
    vbool1_t malformed_family_lead_b1 = __riscv_vmandn_mm_b1(is_family_lead_b1, next_is_continuation_b1, vector_length);
    vbool1_t stop_b1 = __riscv_vmor_mm_b1(
        __riscv_vmor_mm_b1(is_foreign_lead_b1, malformed_family_lead_b1, vector_length), is_irregular_b1,
        vector_length);
    long first_stop = __riscv_vfirst_m_b1(stop_b1, vector_length);
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed; 
        *needs_serial = 1;
    }
    else {
        consumed = vector_length;
        if (vector_length < remaining && source_ptr[vector_length - 1] >= 0xC0) --consumed;
        *needs_serial = 0;
    }
    if (consumed) __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, consumed);
    return consumed;
}
SZ_HELPER_AUTO sz_size_t sz_utf8_fold_cyrillic_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                          sz_u8_t *destination_ptr, int *needs_serial) {
    static sz_u8_t const second_byte_offsets[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, 0xE0, 0, 0, 0, 0, 0};
    sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    sz_u8_t next_carry = (vector_length < remaining) ? source_ptr[vector_length] : 0;
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t is_d0_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD0, vector_length);
    vbool1_t is_d1_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD1, vector_length);
    vbool1_t after_d0_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xD0, vector_length);
    vbool1_t is_lead_b1 = __riscv_vmandn_mm_b1(__riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length),
                                               is_continuation_b1, vector_length);
    vbool1_t is_family_lead_b1 = __riscv_vmor_mm_b1(is_d0_b1, is_d1_b1, vector_length);
    vbool1_t is_foreign_lead_b1 = __riscv_vmandn_mm_b1(is_lead_b1, is_family_lead_b1, vector_length);
    vbool1_t next_is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(next_u8m8, 0xC0, vector_length),
                                                                0x80, vector_length);
    vbool1_t malformed_family_lead_b1 = __riscv_vmandn_mm_b1(is_family_lead_b1, next_is_continuation_b1, vector_length);
    vbool1_t is_extended_b1 = __riscv_vmand_mm_b1(is_d1_b1, __riscv_vmsgeu_vx_u8m8_b1(next_u8m8, 0xA0, vector_length),
                                                  vector_length);
    vbool1_t stop_b1 = __riscv_vmor_mm_b1(
        __riscv_vmor_mm_b1(is_foreign_lead_b1, malformed_family_lead_b1, vector_length), is_extended_b1, vector_length);
    vuint8m8_t offset_u8m8 = __riscv_vmv_v_x_u8m8(0, vector_length);
    offset_u8m8 = __riscv_vluxei8_v_u8m8_mu(after_d0_b1, offset_u8m8, second_byte_offsets,
                                            __riscv_vsrl_vx_u8m8(source_u8m8, 4, vector_length), vector_length);
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    folded_u8m8 = __riscv_vadd_vv_u8m8(folded_u8m8, offset_u8m8, vector_length);
    vbool1_t next_80_8f_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0x80, vector_length), 0x10,
                                                       vector_length);
    vbool1_t next_a0_af_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0xA0, vector_length), 0x10,
                                                       vector_length);
    vbool1_t needs_d1_b1 = __riscv_vmand_mm_b1(
        is_d0_b1, __riscv_vmor_mm_b1(next_80_8f_b1, next_a0_af_b1, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(needs_d1_b1, folded_u8m8, folded_u8m8, 1, vector_length);
    long first_stop = __riscv_vfirst_m_b1(stop_b1, vector_length);
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed;
        *needs_serial = 1;
    }
    else {
        consumed = sz_utf8_fold_trim_incomplete_(source_ptr, vector_length, remaining);
        *needs_serial = 0;
    }
    if (consumed) __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, consumed);
    return consumed;
}
SZ_HELPER_AUTO sz_size_t sz_utf8_fold_greek_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                       sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    sz_u8_t next_carry = (vector_length < remaining) ? source_ptr[vector_length] : 0;
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t is_ce_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xCE, vector_length);
    vbool1_t is_cf_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xCF, vector_length);
    vbool1_t after_ce_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xCE, vector_length);
    vbool1_t after_cf_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xCF, vector_length);
    vbool1_t is_lead_b1 = __riscv_vmandn_mm_b1(__riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length),
                                               is_continuation_b1, vector_length);
    vbool1_t is_family_lead_b1 = __riscv_vmor_mm_b1(is_ce_b1, is_cf_b1, vector_length);
    vbool1_t is_foreign_lead_b1 = __riscv_vmandn_mm_b1(is_lead_b1, is_family_lead_b1, vector_length);
    vbool1_t next_is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(next_u8m8, 0xC0, vector_length),
                                                                0x80, vector_length);
    vbool1_t malformed_family_lead_b1 = __riscv_vmandn_mm_b1(is_family_lead_b1, next_is_continuation_b1, vector_length);
    vbool1_t ce_excluded_b1 = __riscv_vmand_mm_b1(
        is_ce_b1,
        __riscv_vmor_mm_b1(__riscv_vmsltu_vx_u8m8_b1(next_u8m8, 0x91, vector_length),
                           __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0xB0, vector_length), vector_length),
        vector_length);
    vbool1_t cf_excluded_b1 = __riscv_vmand_mm_b1(is_cf_b1, __riscv_vmsgeu_vx_u8m8_b1(next_u8m8, 0x8F, vector_length),
                                                  vector_length);
    vbool1_t stop_b1 = __riscv_vmor_mm_b1(
        __riscv_vmor_mm_b1(is_foreign_lead_b1, malformed_family_lead_b1, vector_length),
        __riscv_vmor_mm_b1(ce_excluded_b1, cf_excluded_b1, vector_length), vector_length);
    vbool1_t in_promote_a0_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0xA0, vector_length), 0x02,
                                                          vector_length);
    vbool1_t in_promote_a3_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0xA3, vector_length), 0x09,
                                                          vector_length);
    vbool1_t next_promote_a0_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0xA0, vector_length), 0x02,
                                                            vector_length);
    vbool1_t next_promote_a3_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0xA3, vector_length), 0x09,
                                                            vector_length);
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    vbool1_t basic_upper_b1 = __riscv_vmand_mm_b1(
        after_ce_b1,
        __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0x91, vector_length), 0x0F, vector_length),
        vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(basic_upper_b1, folded_u8m8, folded_u8m8, 0x20, vector_length);
    vbool1_t promoting_upper_b1 = __riscv_vmand_mm_b1(
        after_ce_b1, __riscv_vmor_mm_b1(in_promote_a0_b1, in_promote_a3_b1, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(promoting_upper_b1, folded_u8m8, folded_u8m8, 0xE0, vector_length); 
    vbool1_t final_sigma_b1 = __riscv_vmand_mm_b1(
        after_cf_b1, __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0x82, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(final_sigma_b1, folded_u8m8, folded_u8m8, 0x01, vector_length);
    vbool1_t promotes_lead_b1 = __riscv_vmand_mm_b1(
        is_ce_b1, __riscv_vmor_mm_b1(next_promote_a0_b1, next_promote_a3_b1, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(promotes_lead_b1, folded_u8m8, folded_u8m8, 0x01, vector_length); 
    long first_stop = __riscv_vfirst_m_b1(stop_b1, vector_length);
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed;
        *needs_serial = 1;
    }
    else {
        consumed = sz_utf8_fold_trim_incomplete_(source_ptr, vector_length, remaining);
        *needs_serial = 0;
    }
    if (consumed) __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, consumed);
    return consumed;
}
SZ_HELPER_AUTO sz_size_t sz_utf8_fold_armenian_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                          sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    sz_u8_t next_carry = (vector_length < remaining) ? source_ptr[vector_length] : 0;
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t is_d4_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD4, vector_length);
    vbool1_t is_d5_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD5, vector_length);
    vbool1_t is_d6_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD6, vector_length);
    vbool1_t after_d4_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xD4, vector_length);
    vbool1_t after_d5_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xD5, vector_length);
    vbool1_t is_lead_b1 = __riscv_vmandn_mm_b1(__riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length),
                                               is_continuation_b1, vector_length);
    vbool1_t is_family_b1 = __riscv_vmor_mm_b1(__riscv_vmor_mm_b1(is_d4_b1, is_d5_b1, vector_length), is_d6_b1,
                                               vector_length);
    vbool1_t is_foreign_lead_b1 = __riscv_vmandn_mm_b1(is_lead_b1, is_family_b1, vector_length);
    vbool1_t next_is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(next_u8m8, 0xC0, vector_length),
                                                                0x80, vector_length);
    vbool1_t malformed_family_lead_b1 = __riscv_vmandn_mm_b1(is_family_b1, next_is_continuation_b1, vector_length);
    vbool1_t d4_stop_b1 = __riscv_vmand_mm_b1(is_d4_b1, __riscv_vmsltu_vx_u8m8_b1(next_u8m8, 0xB1, vector_length),
                                              vector_length);
    vbool1_t ligature_stop_b1 = __riscv_vmand_mm_b1(is_d6_b1, __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x87, vector_length),
                                                    vector_length);
    vbool1_t stop_b1 = __riscv_vmor_mm_b1(
        __riscv_vmor_mm_b1(is_foreign_lead_b1, malformed_family_lead_b1, vector_length),
        __riscv_vmor_mm_b1(d4_stop_b1, ligature_stop_b1, vector_length), vector_length);
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    vbool1_t d4_second_b1 = __riscv_vmand_mm_b1(
        after_d4_b1, __riscv_vmsgeu_vx_u8m8_b1(source_u8m8, 0xB1, vector_length), vector_length);
    vbool1_t d5_plus30_b1 = __riscv_vmand_mm_b1(
        after_d5_b1,
        __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0x80, vector_length), 0x10, vector_length),
        vector_length);
    vbool1_t d5_minus10_b1 = __riscv_vmand_mm_b1(
        after_d5_b1,
        __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0x90, vector_length), 0x07, vector_length),
        vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(__riscv_vmor_mm_b1(d4_second_b1, d5_minus10_b1, vector_length), folded_u8m8,
                                          folded_u8m8, 0xF0, vector_length); 
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(d5_plus30_b1, folded_u8m8, folded_u8m8, 0x30, vector_length);
    vbool1_t promotes_d4_b1 = __riscv_vmand_mm_b1(is_d4_b1, __riscv_vmsgeu_vx_u8m8_b1(next_u8m8, 0xB1, vector_length),
                                                  vector_length);
    vbool1_t promotes_d5_b1 = __riscv_vmand_mm_b1(
        is_d5_b1, __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0x90, vector_length), 0x07, vector_length),
        vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(__riscv_vmor_mm_b1(promotes_d4_b1, promotes_d5_b1, vector_length),
                                          folded_u8m8, folded_u8m8, 0x01, vector_length);
    long first_stop = __riscv_vfirst_m_b1(stop_b1, vector_length);
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed;
        *needs_serial = 1;
    }
    else {
        consumed = sz_utf8_fold_trim_incomplete_(source_ptr, vector_length, remaining);
        *needs_serial = 0;
    }
    if (consumed) __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, consumed);
    return consumed;
}
SZ_HELPER_AUTO sz_size_t sz_utf8_fold_georgian_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                          sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    sz_u8_t carry1 = (vector_length < remaining) ? source_ptr[vector_length] : 0;
    sz_u8_t carry2 = (vector_length + 1 < remaining) ? source_ptr[vector_length + 1] : 0;
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, carry1, vector_length);
    vuint8m8_t next_next_u8m8 = __riscv_vslide1down_vx_u8m8(next_u8m8, carry2, vector_length);
    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t is_e1_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xE1, vector_length);
    vbool1_t is_lead_b1 = __riscv_vmandn_mm_b1(__riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length),
                                               is_continuation_b1, vector_length);
    vbool1_t is_foreign_lead_b1 = __riscv_vmandn_mm_b1(is_lead_b1, is_e1_b1, vector_length);
    vbool1_t is_82_lead_b1 = __riscv_vmand_mm_b1(is_e1_b1, __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x82, vector_length),
                                                 vector_length);
    vbool1_t is_83_lead_b1 = __riscv_vmand_mm_b1(is_e1_b1, __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x83, vector_length),
                                                 vector_length);
    vbool1_t is_foreign_e1_b1 = __riscv_vmandn_mm_b1(
        is_e1_b1, __riscv_vmor_mm_b1(is_82_lead_b1, is_83_lead_b1, vector_length), vector_length);
    vbool1_t third_is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(
        __riscv_vand_vx_u8m8(next_next_u8m8, 0xC0, vector_length), 0x80, vector_length);
    vbool1_t malformed_family_lead_b1 = __riscv_vmandn_mm_b1(
        __riscv_vmor_mm_b1(is_82_lead_b1, is_83_lead_b1, vector_length), third_is_continuation_b1, vector_length);
    vbool1_t stop_b1 = __riscv_vmor_mm_b1(__riscv_vmor_mm_b1(is_foreign_lead_b1, is_foreign_e1_b1, vector_length),
                                          malformed_family_lead_b1, vector_length);
    vbool1_t is_82_upper_lead_b1 = __riscv_vmand_mm_b1(
        is_82_lead_b1, __riscv_vmsgeu_vx_u8m8_b1(next_next_u8m8, 0xA0, vector_length), vector_length);
    vbool1_t third_83_range_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_next_u8m8, 0x80, vector_length),
                                                           0x06, vector_length);
    vbool1_t third_83_extra_b1 = __riscv_vmor_mm_b1(__riscv_vmseq_vx_u8m8_b1(next_next_u8m8, 0x87, vector_length),
                                                    __riscv_vmseq_vx_u8m8_b1(next_next_u8m8, 0x8D, vector_length),
                                                    vector_length);
    vbool1_t is_83_upper_lead_b1 = __riscv_vmand_mm_b1(
        is_83_lead_b1, __riscv_vmor_mm_b1(third_83_range_b1, third_83_extra_b1, vector_length), vector_length);
    vuint8m8_t flag82_lead_u8m8 = __riscv_vmerge_vxm_u8m8(__riscv_vmv_v_x_u8m8(0, vector_length), 1,
                                                          is_82_upper_lead_b1, vector_length);
    vuint8m8_t flag83_lead_u8m8 = __riscv_vmerge_vxm_u8m8(__riscv_vmv_v_x_u8m8(0, vector_length), 1,
                                                          is_83_upper_lead_b1, vector_length);
    vuint8m8_t flag82_second_u8m8 = __riscv_vslide1up_vx_u8m8(flag82_lead_u8m8, 0, vector_length);
    vuint8m8_t flag83_second_u8m8 = __riscv_vslide1up_vx_u8m8(flag83_lead_u8m8, 0, vector_length);
    vuint8m8_t flag82_third_u8m8 = __riscv_vslide1up_vx_u8m8(flag82_second_u8m8, 0, vector_length);
    vuint8m8_t flag83_third_u8m8 = __riscv_vslide1up_vx_u8m8(flag83_second_u8m8, 0, vector_length);
    vbool1_t is_upper_lead_b1 = __riscv_vmor_mm_b1(is_82_upper_lead_b1, is_83_upper_lead_b1, vector_length);
    vbool1_t is_upper_second_b1 = __riscv_vmsne_vx_u8m8_b1(
        __riscv_vor_vv_u8m8(flag82_second_u8m8, flag83_second_u8m8, vector_length), 0, vector_length);
    vbool1_t is_82_upper_third_b1 = __riscv_vmsne_vx_u8m8_b1(flag82_third_u8m8, 0, vector_length);
    vbool1_t is_83_upper_third_b1 = __riscv_vmsne_vx_u8m8_b1(flag83_third_u8m8, 0, vector_length);
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(folded_u8m8, 0xE2, is_upper_lead_b1, vector_length); 
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(folded_u8m8, 0xB4, is_upper_second_b1,
                                          vector_length); 
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(is_82_upper_third_b1, folded_u8m8, folded_u8m8, 0xE0,
                                          vector_length); 
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(is_83_upper_third_b1, folded_u8m8, folded_u8m8, 0x20,
                                          vector_length); 
    long first_stop = __riscv_vfirst_m_b1(stop_b1, vector_length);
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed;
        *needs_serial = 1;
    }
    else {
        consumed = sz_utf8_fold_trim_incomplete_(source_ptr, vector_length, remaining);
        *needs_serial = 0;
    }
    if (consumed) __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, consumed);
    return consumed;
}
SZ_API_COMPTIME sz_size_t sz_utf8_uncased_fold_rvv(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    sz_u8_t const *source_ptr = (sz_u8_t const *)source;
    sz_u8_t const *source_end = source_ptr + source_length;
    sz_u8_t *destination_ptr = (sz_u8_t *)destination;
    while (source_ptr < source_end) {
        sz_size_t ascii_bytes = (sz_size_t)(source_end - source_ptr);
        while (ascii_bytes) {
            sz_size_t vector_length = __riscv_vsetvl_e8m8(ascii_bytes);
            vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
            long first_non_ascii = __riscv_vfirst_m_b1(__riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length),
                                                       vector_length);
            sz_size_t ascii_run = first_non_ascii < 0 ? vector_length : (sz_size_t)first_non_ascii;
            if (ascii_run)
                __riscv_vse8_v_u8m8(destination_ptr, sz_utf8_fold_ascii_rvv_(source_u8m8, vector_length), ascii_run);
            source_ptr += ascii_run;
            destination_ptr += ascii_run;
            ascii_bytes -= ascii_run;
            if (first_non_ascii >= 0) break; 
        }
        if (source_ptr >= source_end) break;
        int needs_serial = 1;
        sz_size_t consumed = 0;
        sz_size_t remaining = (sz_size_t)(source_end - source_ptr);
        sz_u8_t lead = source_ptr[0];
        if (lead >= 0xC2 && lead <= 0xC6)
            consumed = sz_utf8_fold_latin_strip_rvv_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xD0 || lead == 0xD1)
            consumed = sz_utf8_fold_cyrillic_strip_rvv_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xCE || lead == 0xCF)
            consumed = sz_utf8_fold_greek_strip_rvv_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xD4 || lead == 0xD5 || lead == 0xD6)
            consumed = sz_utf8_fold_armenian_strip_rvv_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xE1)
            consumed = sz_utf8_fold_georgian_strip_rvv_(source_ptr, remaining, destination_ptr, &needs_serial);
        source_ptr += consumed;
        destination_ptr += consumed;
        if (!needs_serial && consumed) continue; 
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
