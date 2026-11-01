/**
 *  @brief RISC-V Vector (RVV 1.0) uncased UTF-8 search, comparison & invariance backend.
 *  @file include/stringzilla/utf8_uncased/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_RVV_H_
#define STRINGZILLA_UTF8_UNCASED_RVV_H_
#include "stringzilla/find/rvv.h"              
#include "stringzilla/utf8_uncased_fold/rvv.h" 
#include "stringzilla/utf8_uncased/serial.h"
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
SZ_API_COMPTIME sz_cptr_t sz_utf8_find_cased_rvv(sz_cptr_t str, sz_size_t length);
#pragma region Substring Search
/** @brief Folds a length-preserving strip of haystack bytes in place using script-specific rules. */
typedef void (*sz_utf8_uncased_fold_strip_rvv_t)(sz_u8_t const *source_ptr, sz_size_t vector_length,
                                                 sz_u8_t *destination_ptr);
/** @brief Scans a strip for length-changing fold characters; returns the offset of the first, or -1. */
typedef long (*sz_utf8_uncased_alarm_strip_rvv_t)(sz_u8_t const *source_ptr, sz_size_t vector_length);
#pragma region Per Script Fold Strips
SZ_HELPER_NOINLINE void sz_utf8_uncased_fold_ascii_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vector_length,
                                                              sz_u8_t *destination_ptr) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, vector_length);
}
SZ_HELPER_NOINLINE void sz_utf8_uncased_fold_western_europe_strip_rvv_(sz_u8_t const *source_ptr,
                                                                       sz_size_t vector_length,
                                                                       sz_u8_t *destination_ptr) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    sz_u8_t next_carry = source_ptr[vector_length]; 
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);
    vbool1_t after_c3_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xC3, vector_length);
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    vbool1_t latin1_range_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0x80, vector_length), 0x1F,
                                                         vector_length);
    vbool1_t is_latin1_upper_b1 = __riscv_vmandn_mm_b1(__riscv_vmand_mm_b1(after_c3_b1, latin1_range_b1, vector_length),
                                                       __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0x97, vector_length),
                                                       vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(is_latin1_upper_b1, folded_u8m8, folded_u8m8, 0x20, vector_length);
    vbool1_t eszett_lead_b1 = __riscv_vmand_mm_b1(__riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xC3, vector_length),
                                                  __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x9F, vector_length),
                                                  vector_length);
    vbool1_t eszett_second_b1 = __riscv_vmand_mm_b1(
        after_c3_b1, __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0x9F, vector_length), vector_length);
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(
        folded_u8m8, 's', __riscv_vmor_mm_b1(eszett_lead_b1, eszett_second_b1, vector_length), vector_length);
    __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, vector_length);
}
SZ_HELPER_NOINLINE void sz_utf8_uncased_fold_central_europe_strip_rvv_(sz_u8_t const *source_ptr,
                                                                       sz_size_t vector_length,
                                                                       sz_u8_t *destination_ptr) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t after_c3_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xC3, vector_length);
    vbool1_t after_c45_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(previous_u8m8, 0xC4, vector_length), 1,
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
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(is_latin1_upper_b1, folded_u8m8, folded_u8m8, 0x20, vector_length);
    vbool1_t delta_lanes_b1 = __riscv_vmand_mm_b1(after_c45_b1, is_continuation_b1, vector_length);
    vuint8m8_t low6_u8m8 = __riscv_vand_vx_u8m8(source_u8m8, 0x3F, vector_length);
    vuint8m8_t family_base_u8m8 = __riscv_vmul_vx_u8m8(__riscv_vsub_vx_u8m8(previous_u8m8, 0xC4, vector_length), 64,
                                                       vector_length);
    vuint8m8_t lut_index_u8m8 = __riscv_vadd_vv_u8m8(family_base_u8m8, low6_u8m8, vector_length);
    vuint8m8_t delta_u8m8 = __riscv_vmv_v_x_u8m8(0, vector_length);
    delta_u8m8 = __riscv_vluxei8_v_u8m8_mu(delta_lanes_b1, delta_u8m8, sz_utf8_fold_latin_c456_deltas_rvv_,
                                           lut_index_u8m8, vector_length);
    delta_u8m8 = __riscv_vand_vx_u8m8(delta_u8m8, 0x01, vector_length);
    folded_u8m8 = __riscv_vadd_vv_u8m8(folded_u8m8, delta_u8m8, vector_length);
    __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, vector_length);
}
SZ_HELPER_NOINLINE void sz_utf8_uncased_fold_cyrillic_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vector_length,
                                                                 sz_u8_t *destination_ptr) {
    static sz_u8_t const second_byte_offsets[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, 0xE0, 0, 0, 0, 0, 0};
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    sz_u8_t next_carry = source_ptr[vector_length];
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);
    vbool1_t after_d0_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xD0, vector_length);
    vbool1_t is_d0_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD0, vector_length);
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    vuint8m8_t offset_u8m8 = __riscv_vmv_v_x_u8m8(0, vector_length);
    offset_u8m8 = __riscv_vluxei8_v_u8m8_mu(after_d0_b1, offset_u8m8, second_byte_offsets,
                                            __riscv_vsrl_vx_u8m8(source_u8m8, 4, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vv_u8m8(folded_u8m8, offset_u8m8, vector_length);
    vbool1_t next_80_8f_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0x80, vector_length), 0x10,
                                                       vector_length);
    vbool1_t next_a0_af_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0xA0, vector_length), 0x10,
                                                       vector_length);
    vbool1_t needs_d1_b1 = __riscv_vmand_mm_b1(
        is_d0_b1, __riscv_vmor_mm_b1(next_80_8f_b1, next_a0_af_b1, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(needs_d1_b1, folded_u8m8, folded_u8m8, 1, vector_length);
    __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, vector_length);
}
static sz_u8_t const sz_utf8_uncased_greek_ce_table_rvv_[128] = {
    0,    0,    0,    0,    0,    0,    0x26, 0,
    0x25, 0x25, 0x25, 0,    0,    0,    0xFF, 0xFF, 
    0,    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 
    0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0,
    0xE0, 0xE0, 0xE0, 0xE0, 0,    0,    0,    0, 
    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0, 
    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    1,    0,    1,    1, 
    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0, 
    1,    1,    1,    1,    1,    1,    1,    1,
    1,    1,    1,    1,    0,    0,    0,    0, 
    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0, 
};
SZ_HELPER_NOINLINE void sz_utf8_uncased_fold_greek_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vector_length,
                                                              sz_u8_t *destination_ptr) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t after_ce_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xCE, vector_length);
    vbool1_t after_cf_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xCF, vector_length);
    vbool1_t after_c2_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xC2, vector_length);
    vbool1_t after_ce_cont_b1 = __riscv_vmand_mm_b1(after_ce_b1, is_continuation_b1, vector_length);
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    vuint8m8_t low6_u8m8 = __riscv_vand_vx_u8m8(source_u8m8, 0x3F, vector_length);
    vuint8m8_t ce_delta_u8m8 = __riscv_vmv_v_x_u8m8(0, vector_length);
    ce_delta_u8m8 = __riscv_vluxei8_v_u8m8_mu(after_ce_cont_b1, ce_delta_u8m8, sz_utf8_uncased_greek_ce_table_rvv_,
                                              low6_u8m8, vector_length);
    vuint8m8_t promote_second_u8m8 = __riscv_vmv_v_x_u8m8(0, vector_length);
    promote_second_u8m8 = __riscv_vluxei8_v_u8m8_mu(after_ce_cont_b1, promote_second_u8m8,
                                                    sz_utf8_uncased_greek_ce_table_rvv_,
                                                    __riscv_vadd_vx_u8m8(low6_u8m8, 64, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vv_u8m8(folded_u8m8, ce_delta_u8m8, vector_length);
    vbool1_t final_sigma_b1 = __riscv_vmand_mm_b1(
        after_cf_b1, __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0x82, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(final_sigma_b1, folded_u8m8, folded_u8m8, 0x01, vector_length);
    vbool1_t micro_second_b1 = __riscv_vmand_mm_b1(
        after_c2_b1, __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xB5, vector_length), vector_length);
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(folded_u8m8, 0xBC, micro_second_b1, vector_length);
    vuint8m8_t micro_second_byte_u8m8 = __riscv_vmerge_vxm_u8m8(__riscv_vmv_v_x_u8m8(0, vector_length), 1,
                                                                micro_second_b1, vector_length);
    vuint8m8_t promote_lead_u8m8 = __riscv_vslide1down_vx_u8m8(promote_second_u8m8, 0, vector_length);
    vuint8m8_t micro_lead_u8m8 = __riscv_vslide1down_vx_u8m8(micro_second_byte_u8m8, 0, vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(__riscv_vmsne_vx_u8m8_b1(promote_lead_u8m8, 0, vector_length), folded_u8m8,
                                          folded_u8m8, 0x01, vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(__riscv_vmsne_vx_u8m8_b1(micro_lead_u8m8, 0, vector_length), folded_u8m8,
                                          folded_u8m8, 0x0C, vector_length);
    __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, vector_length);
}
SZ_HELPER_NOINLINE void sz_utf8_uncased_fold_armenian_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vector_length,
                                                                 sz_u8_t *destination_ptr) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    sz_u8_t next_carry = source_ptr[vector_length];
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);
    vbool1_t is_d4_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD4, vector_length);
    vbool1_t is_d5_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD5, vector_length);
    vbool1_t after_d4_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xD4, vector_length);
    vbool1_t after_d5_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xD5, vector_length);
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
    __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, vector_length);
}
SZ_HELPER_NOINLINE void sz_utf8_uncased_fold_vietnamese_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vector_length,
                                                                   sz_u8_t *destination_ptr) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vuint8m8_t previous2_u8m8 = __riscv_vslide1up_vx_u8m8(previous_u8m8, 0, vector_length);
    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t after_c3_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xC3, vector_length);
    vbool1_t after_c456_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(previous_u8m8, 0xC4, vector_length), 2,
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
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(is_latin1_upper_b1, folded_u8m8, folded_u8m8, 0x20, vector_length);
    vbool1_t delta_lanes_b1 = __riscv_vmand_mm_b1(after_c456_b1, is_continuation_b1, vector_length);
    vuint8m8_t low6_u8m8 = __riscv_vand_vx_u8m8(source_u8m8, 0x3F, vector_length);
    vuint8m8_t family_base_u8m8 = __riscv_vmul_vx_u8m8(__riscv_vsub_vx_u8m8(previous_u8m8, 0xC4, vector_length), 64,
                                                       vector_length);
    vuint8m8_t lut_index_u8m8 = __riscv_vadd_vv_u8m8(family_base_u8m8, low6_u8m8, vector_length);
    vuint8m8_t delta_u8m8 = __riscv_vmv_v_x_u8m8(0, vector_length);
    delta_u8m8 = __riscv_vluxei8_v_u8m8_mu(delta_lanes_b1, delta_u8m8, sz_utf8_fold_latin_c456_deltas_rvv_,
                                           lut_index_u8m8, vector_length);
    delta_u8m8 = __riscv_vand_vx_u8m8(delta_u8m8, 0x01, vector_length);
    folded_u8m8 = __riscv_vadd_vv_u8m8(folded_u8m8, delta_u8m8, vector_length);
    vbool1_t after_e1_pair_b1 = __riscv_vmand_mm_b1(
        __riscv_vmseq_vx_u8m8_b1(previous2_u8m8, 0xE1, vector_length),
        __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(previous_u8m8, 0xB8, vector_length), 0x04, vector_length),
        vector_length);
    vbool1_t third_even_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0x01, vector_length), 0,
                                                      vector_length);
    vbool1_t fold_e1_b1 = __riscv_vmand_mm_b1(after_e1_pair_b1, third_even_b1, vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(fold_e1_b1, folded_u8m8, folded_u8m8, 0x01, vector_length);
    __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, vector_length);
}
#pragma endregion // Per Script Fold Strips
#pragma region Per Script Alarm Strips
SZ_HELPER_INLINE vuint8m8_t sz_utf8_uncased_eq_byte_(vuint8m8_t bytes_u8m8, sz_u8_t value, sz_size_t vector_length) {
    return __riscv_vmerge_vxm_u8m8(__riscv_vmv_v_x_u8m8(0, vector_length), 1,
                                   __riscv_vmseq_vx_u8m8_b1(bytes_u8m8, value, vector_length), vector_length);
}
SZ_HELPER_INLINE vuint8m8_t sz_utf8_uncased_in_range_byte_(vuint8m8_t bytes_u8m8, sz_u8_t start, sz_u8_t length,
                                                           sz_size_t vector_length) {
    vbool1_t in_range_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(bytes_u8m8, start, vector_length), length,
                                                     vector_length);
    return __riscv_vmerge_vxm_u8m8(__riscv_vmv_v_x_u8m8(0, vector_length), 1, in_range_b1, vector_length);
}
SZ_HELPER_INLINE long sz_utf8_uncased_alarm_to_lead_(vuint8m8_t danger_at_second_u8m8, sz_size_t vector_length) {
    vuint8m8_t flag_at_lead_u8m8 = __riscv_vslide1down_vx_u8m8(danger_at_second_u8m8, 0, vector_length);
    return __riscv_vfirst_m_b1(__riscv_vmsne_vx_u8m8_b1(flag_at_lead_u8m8, 0, vector_length), vector_length);
}
SZ_HELPER_NOINLINE long sz_utf8_uncased_alarm_western_europe_strip_rvv_(sz_u8_t const *source_ptr,
                                                                        sz_size_t vector_length) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, source_ptr[vector_length], vector_length);
    vuint8m8_t after_c3_u8m8 = sz_utf8_uncased_eq_byte_(previous_u8m8, 0xC3, vector_length);
    vuint8m8_t after_c5_u8m8 = sz_utf8_uncased_eq_byte_(previous_u8m8, 0xC5, vector_length);
    vuint8m8_t danger_u8m8 = __riscv_vand_vv_u8m8(
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xBA, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xE1, vector_length), vector_length),
        sz_utf8_uncased_in_range_byte_(next_u8m8, 0x96, 0x09, vector_length), vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(
            __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0x84, vector_length),
                                 sz_utf8_uncased_eq_byte_(previous_u8m8, 0xE2, vector_length), vector_length),
            __riscv_vor_vv_u8m8(sz_utf8_uncased_eq_byte_(next_u8m8, 0xAA, vector_length),
                                sz_utf8_uncased_eq_byte_(next_u8m8, 0xAB, vector_length), vector_length),
            vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xAC, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xEF, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(after_c5_u8m8, sz_utf8_uncased_eq_byte_(source_u8m8, 0xBF, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(after_c5_u8m8, sz_utf8_uncased_eq_byte_(source_u8m8, 0xB8, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(after_c3_u8m8, sz_utf8_uncased_eq_byte_(source_u8m8, 0x9F, vector_length), vector_length),
        vector_length);
    return sz_utf8_uncased_alarm_to_lead_(danger_u8m8, vector_length);
}
SZ_HELPER_NOINLINE long sz_utf8_uncased_alarm_central_europe_strip_rvv_(sz_u8_t const *source_ptr,
                                                                        sz_size_t vector_length) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vuint8m8_t after_c3_u8m8 = sz_utf8_uncased_eq_byte_(previous_u8m8, 0xC3, vector_length);
    vuint8m8_t after_c4_u8m8 = sz_utf8_uncased_eq_byte_(previous_u8m8, 0xC4, vector_length);
    vuint8m8_t after_c5_u8m8 = sz_utf8_uncased_eq_byte_(previous_u8m8, 0xC5, vector_length);
    vuint8m8_t danger_u8m8 = __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0x84, vector_length),
                                                  sz_utf8_uncased_eq_byte_(previous_u8m8, 0xE2, vector_length),
                                                  vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(after_c3_u8m8, sz_utf8_uncased_eq_byte_(source_u8m8, 0x9F, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(after_c4_u8m8, sz_utf8_uncased_eq_byte_(source_u8m8, 0xB0, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(after_c4_u8m8, sz_utf8_uncased_eq_byte_(source_u8m8, 0xBF, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(after_c5_u8m8, sz_utf8_uncased_eq_byte_(source_u8m8, 0xBF, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(after_c5_u8m8, sz_utf8_uncased_eq_byte_(source_u8m8, 0xB8, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xAC, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xEF, vector_length), vector_length),
        vector_length);
    return sz_utf8_uncased_alarm_to_lead_(danger_u8m8, vector_length);
}
SZ_HELPER_NOINLINE long sz_utf8_uncased_alarm_cyrillic_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vector_length) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, source_ptr[vector_length], vector_length);
    vuint8m8_t danger_u8m8 = __riscv_vand_vv_u8m8(
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xB2, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xE1, vector_length), vector_length),
        sz_utf8_uncased_in_range_byte_(next_u8m8, 0x80, 0x09, vector_length), vector_length);
    return sz_utf8_uncased_alarm_to_lead_(danger_u8m8, vector_length);
}
SZ_HELPER_NOINLINE long sz_utf8_uncased_alarm_greek_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vector_length) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vuint8m8_t after_ce_u8m8 = sz_utf8_uncased_eq_byte_(previous_u8m8, 0xCE, vector_length);
    vuint8m8_t after_cf_u8m8 = sz_utf8_uncased_eq_byte_(previous_u8m8, 0xCF, vector_length);
    vuint8m8_t second_90_b0_a2_u8m8 = __riscv_vor_vv_u8m8(
        __riscv_vor_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0x90, vector_length),
                            sz_utf8_uncased_eq_byte_(source_u8m8, 0xB0, vector_length), vector_length),
        sz_utf8_uncased_eq_byte_(source_u8m8, 0xA2, vector_length), vector_length);
    vuint8m8_t danger_u8m8 = __riscv_vand_vv_u8m8(after_ce_u8m8, second_90_b0_a2_u8m8, vector_length);
    vuint8m8_t second_9x_u8m8 = __riscv_vor_vv_u8m8(
        __riscv_vor_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0x90, vector_length),
                            sz_utf8_uncased_eq_byte_(source_u8m8, 0x91, vector_length), vector_length),
        __riscv_vor_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0x95, vector_length),
                            sz_utf8_uncased_eq_byte_(source_u8m8, 0x96, vector_length), vector_length),
        vector_length);
    vuint8m8_t second_bx_u8m8 = __riscv_vor_vv_u8m8(
        __riscv_vor_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xB0, vector_length),
                            sz_utf8_uncased_eq_byte_(source_u8m8, 0xB1, vector_length), vector_length),
        __riscv_vor_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xB4, vector_length),
                            sz_utf8_uncased_eq_byte_(source_u8m8, 0xB5, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(after_cf_u8m8, __riscv_vor_vv_u8m8(second_9x_u8m8, second_bx_u8m8, vector_length),
                             vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0x84, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xE2, vector_length), vector_length),
        vector_length);
    long lead_danger = sz_utf8_uncased_alarm_to_lead_(danger_u8m8, vector_length);
    vbool1_t lead_blanket_b1 = __riscv_vmor_mm_b1(__riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xE1, vector_length),
                                                  __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xCD, vector_length),
                                                  vector_length);
    long blanket_danger = __riscv_vfirst_m_b1(lead_blanket_b1, vector_length);
    if (lead_danger < 0) return blanket_danger;
    if (blanket_danger < 0) return lead_danger;
    return lead_danger < blanket_danger ? lead_danger : blanket_danger;
}
SZ_HELPER_NOINLINE long sz_utf8_uncased_alarm_armenian_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vector_length) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vuint8m8_t danger_u8m8 = __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0x87, vector_length),
                                                  sz_utf8_uncased_eq_byte_(previous_u8m8, 0xD6, vector_length),
                                                  vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xAC, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xEF, vector_length), vector_length),
        vector_length);
    return sz_utf8_uncased_alarm_to_lead_(danger_u8m8, vector_length);
}
SZ_HELPER_NOINLINE long sz_utf8_uncased_alarm_vietnamese_strip_rvv_(sz_u8_t const *source_ptr,
                                                                    sz_size_t vector_length) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, source_ptr[vector_length], vector_length);
    vuint8m8_t danger_u8m8 = __riscv_vand_vv_u8m8(
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xBA, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xE1, vector_length), vector_length),
        sz_utf8_uncased_in_range_byte_(next_u8m8, 0x96, 0x0A, vector_length), vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0x9F, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xC3, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xBF, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xC5, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xAC, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xEF, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0x84, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xE2, vector_length), vector_length),
        vector_length);
    vbool1_t after_c456_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(previous_u8m8, 0xC4, vector_length), 2,
                                                       vector_length);
    vuint8m8_t low6_u8m8 = __riscv_vand_vx_u8m8(source_u8m8, 0x3F, vector_length);
    vuint8m8_t family_base_u8m8 = __riscv_vmul_vx_u8m8(__riscv_vsub_vx_u8m8(previous_u8m8, 0xC4, vector_length), 64,
                                                       vector_length);
    vuint8m8_t lut_index_u8m8 = __riscv_vadd_vv_u8m8(family_base_u8m8, low6_u8m8, vector_length);
    vuint8m8_t delta_u8m8 = __riscv_vmv_v_x_u8m8(0, vector_length);
    delta_u8m8 = __riscv_vluxei8_v_u8m8_mu(after_c456_b1, delta_u8m8, sz_utf8_fold_latin_c456_deltas_rvv_,
                                           lut_index_u8m8, vector_length);
    vuint8m8_t irregular_u8m8 = sz_utf8_uncased_eq_byte_(__riscv_vand_vx_u8m8(delta_u8m8, 0x80, vector_length), 0x80,
                                                         vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(danger_u8m8, irregular_u8m8, vector_length);
    return sz_utf8_uncased_alarm_to_lead_(danger_u8m8, vector_length);
}
SZ_HELPER_NOINLINE long sz_utf8_uncased_alarm_georgian_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vector_length) {
    vector_length = __riscv_vsetvl_e8m8(
        vector_length); 
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, source_ptr[vector_length], vector_length);
    vuint8m8_t after_e1_u8m8 = sz_utf8_uncased_eq_byte_(previous_u8m8, 0xE1, vector_length);
    vuint8m8_t danger_u8m8 = __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xB2, vector_length),
                                                  after_e1_u8m8, vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(__riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0x82, vector_length),
                                                  after_e1_u8m8, vector_length),
                             sz_utf8_uncased_in_range_byte_(next_u8m8, 0xA0, 0x46, vector_length), vector_length),
        vector_length);
    danger_u8m8 = __riscv_vor_vv_u8m8(
        danger_u8m8,
        __riscv_vand_vv_u8m8(sz_utf8_uncased_eq_byte_(source_u8m8, 0xB4, vector_length),
                             sz_utf8_uncased_eq_byte_(previous_u8m8, 0xE2, vector_length), vector_length),
        vector_length);
    return sz_utf8_uncased_alarm_to_lead_(danger_u8m8, vector_length);
}
#pragma endregion // Per Script Alarm Strips
#pragma region Scripted Driver
SZ_HELPER_AUTO sz_u8_t const *sz_utf8_uncased_load_padded_rvv_(sz_cptr_t source, sz_size_t length, sz_u8_t *buffer,
                                                               sz_size_t buffer_capacity) {
    for (sz_size_t byte_index = 0; byte_index < buffer_capacity; ++byte_index) buffer[byte_index] = 0;
    for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) buffer[byte_index] = (sz_u8_t)source[byte_index];
    return buffer;
}
SZ_HELPER_INLINE sz_cptr_t sz_utf8_uncased_search_rvv_scripted_( 
    sz_utf8_uncased_fold_strip_rvv_t fold,                       
    sz_utf8_uncased_alarm_strip_rvv_t alarm,                     
    sz_cptr_t haystack, sz_size_t haystack_length,               
    sz_cptr_t needle, sz_size_t needle_length,                   
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,    
    sz_size_t *matched_length) {
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");
    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in one register");
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;
    sz_u8_t const probe_first = needle_metadata->folded_slice[0];
    sz_u8_t const probe_second = needle_metadata->folded_slice[offset_second];
    sz_u8_t const probe_third = needle_metadata->folded_slice[offset_third];
    sz_u8_t const probe_last = needle_metadata->folded_slice[offset_last];
    sz_rune_t needle_first_safe_folded_rune = 0;
    if (alarm) sz_rune_decode_unchecked((sz_cptr_t)needle_metadata->folded_slice, &needle_first_safe_folded_rune);
    enum { strip_capacity_k = 4096, fold_pad_k = 8 };
    sz_u8_t source_buffer[strip_capacity_k + fold_pad_k];
    sz_u8_t folded_buffer[strip_capacity_k + fold_pad_k];
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;
        sz_size_t chunk_request = available < strip_capacity_k ? available : strip_capacity_k;
        sz_size_t chunk_size = __riscv_vsetvl_e8m8(chunk_request);
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        sz_size_t const step = !alarm                                ? valid_starts
                               : valid_starts > folded_window_length ? valid_starts - (folded_window_length - 1)
                                                                     : 1;
        sz_u8_t const *source = sz_utf8_uncased_load_padded_rvv_(haystack_ptr, chunk_size, source_buffer,
                                                                 strip_capacity_k + fold_pad_k);
        if (alarm) {
            long danger_offset = alarm(source, chunk_size);
            if (danger_offset >= 0) {
                sz_cptr_t match = sz_utf8_uncased_search_in_danger_zone_( 
                    haystack, haystack_length,                            
                    needle, needle_length,                                
                    haystack_ptr, chunk_size,                             
                    needle_first_safe_folded_rune,                        
                    needle_metadata->offset_in_unfolded,                  
                    matched_length);
                if (match) return match;
                haystack_ptr += step;
                continue;
            }
        }
        fold(source, chunk_size, folded_buffer);
        sz_size_t position = 0;
        while (position < valid_starts) {
            sz_size_t vector_length = __riscv_vsetvl_e8m8(valid_starts - position);
            vuint8m8_t first_view_u8m8 = __riscv_vle8_v_u8m8(folded_buffer + position, vector_length);
            vuint8m8_t second_view_u8m8 = __riscv_vle8_v_u8m8(folded_buffer + position + offset_second, vector_length);
            vuint8m8_t third_view_u8m8 = __riscv_vle8_v_u8m8(folded_buffer + position + offset_third, vector_length);
            vuint8m8_t last_view_u8m8 = __riscv_vle8_v_u8m8(folded_buffer + position + offset_last, vector_length);
            vbool1_t match_mask_b1 = __riscv_vmseq_vx_u8m8_b1(first_view_u8m8, probe_first, vector_length);
            match_mask_b1 = __riscv_vmand_mm_b1(
                match_mask_b1, __riscv_vmseq_vx_u8m8_b1(second_view_u8m8, probe_second, vector_length), vector_length);
            match_mask_b1 = __riscv_vmand_mm_b1(
                match_mask_b1, __riscv_vmseq_vx_u8m8_b1(third_view_u8m8, probe_third, vector_length), vector_length);
            match_mask_b1 = __riscv_vmand_mm_b1(
                match_mask_b1, __riscv_vmseq_vx_u8m8_b1(last_view_u8m8, probe_last, vector_length), vector_length);
            for (long match_index = __riscv_vfirst_m_b1(match_mask_b1, vector_length); match_index >= 0;
                 match_index = __riscv_vfirst_m_b1(match_mask_b1, vector_length)) {
                sz_size_t const candidate_offset = position + (sz_size_t)match_index;
                sz_cptr_t const haystack_candidate_ptr = haystack_ptr + candidate_offset;
                int window_ok = 1;
                for (sz_size_t byte_index = 0; byte_index < folded_window_length; ++byte_index)
                    if (folded_buffer[candidate_offset + byte_index] != needle_metadata->folded_slice[byte_index]) {
                        window_ok = 0;
                        break;
                    }
                if (window_ok) {
                    sz_cptr_t match = sz_utf8_uncased_verify_match_(             
                        haystack, haystack_length,                               
                        needle, needle_length,                                   
                        haystack_candidate_ptr - haystack, folded_window_length, 
                        needle_metadata->offset_in_unfolded,                     
                        needle_length - needle_metadata->offset_in_unfolded -    
                            needle_metadata->length_in_unfolded,                 
                        matched_length);
                    if (match) return match;
                }
                match_mask_b1 = __riscv_vmandn_mm_b1(match_mask_b1, __riscv_vmsif_m_b1(match_mask_b1, vector_length),
                                                     vector_length);
            }
            position += vector_length;
        }
        haystack_ptr += step;
    }
    if (alarm && haystack_ptr < haystack_end) {
        sz_cptr_t match = sz_utf8_uncased_search_in_danger_zone_(   
            haystack, haystack_length,                              
            needle, needle_length,                                  
            haystack_ptr, (sz_size_t)(haystack_end - haystack_ptr), 
            needle_first_safe_folded_rune,                          
            needle_metadata->offset_in_unfolded,                    
            matched_length);
        if (match) return match;
    }
    return SZ_NULL_CHAR;
}
#pragma endregion // Scripted Driver
#pragma region Per Script Kernels
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_rvv_ascii_( 
    sz_cptr_t haystack, sz_size_t haystack_length,          
    sz_cptr_t needle, sz_size_t needle_length,              
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_uncased_search_rvv_scripted_(sz_utf8_uncased_fold_ascii_strip_rvv_,
                                                (sz_utf8_uncased_alarm_strip_rvv_t)SZ_NULL, haystack, haystack_length,
                                                needle, needle_length, needle_metadata, matched_length);
}
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_rvv_western_europe_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                   
    sz_cptr_t needle, sz_size_t needle_length,                       
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_uncased_search_rvv_scripted_(
        sz_utf8_uncased_fold_western_europe_strip_rvv_, sz_utf8_uncased_alarm_western_europe_strip_rvv_, haystack,
        haystack_length, needle, needle_length, needle_metadata, matched_length);
}
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_rvv_central_europe_( 
    sz_cptr_t haystack, sz_size_t haystack_length,                   
    sz_cptr_t needle, sz_size_t needle_length,                       
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_uncased_search_rvv_scripted_(
        sz_utf8_uncased_fold_central_europe_strip_rvv_, sz_utf8_uncased_alarm_central_europe_strip_rvv_, haystack,
        haystack_length, needle, needle_length, needle_metadata, matched_length);
}
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_rvv_cyrillic_( 
    sz_cptr_t haystack, sz_size_t haystack_length,             
    sz_cptr_t needle, sz_size_t needle_length,                 
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_uncased_search_rvv_scripted_(sz_utf8_uncased_fold_cyrillic_strip_rvv_,
                                                sz_utf8_uncased_alarm_cyrillic_strip_rvv_, haystack, haystack_length,
                                                needle, needle_length, needle_metadata, matched_length);
}
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_rvv_greek_( 
    sz_cptr_t haystack, sz_size_t haystack_length,          
    sz_cptr_t needle, sz_size_t needle_length,              
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_uncased_search_rvv_scripted_(sz_utf8_uncased_fold_greek_strip_rvv_,
                                                sz_utf8_uncased_alarm_greek_strip_rvv_, haystack, haystack_length,
                                                needle, needle_length, needle_metadata, matched_length);
}
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_rvv_armenian_( 
    sz_cptr_t haystack, sz_size_t haystack_length,             
    sz_cptr_t needle, sz_size_t needle_length,                 
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_uncased_search_rvv_scripted_(sz_utf8_uncased_fold_armenian_strip_rvv_,
                                                sz_utf8_uncased_alarm_armenian_strip_rvv_, haystack, haystack_length,
                                                needle, needle_length, needle_metadata, matched_length);
}
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_rvv_vietnamese_( 
    sz_cptr_t haystack, sz_size_t haystack_length,               
    sz_cptr_t needle, sz_size_t needle_length,                   
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_uncased_search_rvv_scripted_(sz_utf8_uncased_fold_vietnamese_strip_rvv_,
                                                sz_utf8_uncased_alarm_vietnamese_strip_rvv_, haystack, haystack_length,
                                                needle, needle_length, needle_metadata, matched_length);
}
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_rvv_georgian_( 
    sz_cptr_t haystack, sz_size_t haystack_length,             
    sz_cptr_t needle, sz_size_t needle_length,                 
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_uncased_search_rvv_scripted_(sz_utf8_uncased_fold_ascii_strip_rvv_,
                                                sz_utf8_uncased_alarm_georgian_strip_rvv_, haystack, haystack_length,
                                                needle, needle_length, needle_metadata, matched_length);
}
#pragma endregion // Per Script Kernels
SZ_API_COMPTIME sz_cptr_t sz_utf8_uncased_search_rvv( 
    sz_cptr_t haystack, sz_size_t haystack_length,    
    sz_cptr_t needle, sz_size_t needle_length,        
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {
    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }
    int const is_unknown = needle_metadata->kernel_id == sz_utf8_uncased_rune_unknown_k;
    int const known_agnostic = needle_metadata->kernel_id == sz_utf8_uncased_rune_invariant_k;
    if (known_agnostic || (is_unknown && sz_utf8_find_cased_rvv(needle, needle_length) == SZ_NULL_CHAR)) {
        sz_cptr_t result = sz_find_rvv(haystack, haystack_length, needle, needle_length);
        *matched_length = result ? needle_length : 0;
        return result;
    }
    if (is_unknown) {
        sz_utf8_uncased_needle_metadata_(needle, needle_length, needle_metadata);
        if (needle_metadata->kernel_id == sz_utf8_uncased_rune_fallback_serial_k)
            return sz_utf8_uncased_search_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                 matched_length);
    }
    switch (needle_metadata->kernel_id) {
    case sz_utf8_uncased_rune_ascii_invariant_k:
        return sz_utf8_uncased_search_rvv_ascii_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_western_europe_k:
        return sz_utf8_uncased_search_rvv_western_europe_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_central_europe_k:
        return sz_utf8_uncased_search_rvv_central_europe_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_cyrillic_k:
        return sz_utf8_uncased_search_rvv_cyrillic_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_greek_k:
        return sz_utf8_uncased_search_rvv_greek_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_armenian_k:
        return sz_utf8_uncased_search_rvv_armenian_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_vietnamese_k:
        return sz_utf8_uncased_search_rvv_vietnamese_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_georgian_k:
        return sz_utf8_uncased_search_rvv_georgian_( 
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    default: break;
    }
    needle_metadata->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
    return sz_utf8_uncased_search_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                         matched_length);
}
#pragma endregion // Substring Search
SZ_API_COMPTIME sz_cptr_t sz_utf8_find_cased_rvv(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *cursor = (sz_u8_t const *)str;
    sz_u8_t const *end = cursor + length;
    while (cursor < end) {
        sz_size_t remaining = (sz_size_t)(end - cursor);
        sz_size_t skip = 0;
        while (skip < remaining) {
            sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining - skip);
            vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(cursor + skip, vector_length);
            vbool1_t is_letter_b1 = __riscv_vmsleu_vx_u8m8_b1(
                __riscv_vsub_vx_u8m8(__riscv_vor_vx_u8m8(bytes_u8m8, 0x20, vector_length), 0x61, vector_length), 0x19,
                vector_length);
            vbool1_t is_non_ascii_b1 = __riscv_vmsgtu_vx_u8m8_b1(bytes_u8m8, 0x7F, vector_length);
            long first = __riscv_vfirst_m_b1(__riscv_vmor_mm_b1(is_letter_b1, is_non_ascii_b1, vector_length),
                                             vector_length);
            if (first < 0) {
                skip += vector_length;
                continue;
            }
            skip += (sz_size_t)first;
            break;
        }
        cursor += skip;
        if (cursor >= end) break;
        if (*cursor < 0x80) return (sz_cptr_t)cursor; 
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode((sz_cptr_t)cursor, (sz_cptr_t)end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            cursor++;
            continue;
        }
        if (sz_rune_is_uncased_(rune) == sz_false_k) return (sz_cptr_t)cursor;
        cursor += rune_length;
    }
    return SZ_NULL_CHAR;
}
SZ_API_COMPTIME sz_ordering_t sz_utf8_uncased_order_rvv(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                        sz_size_t b_length) {
    return sz_utf8_uncased_order_serial(a, a_length, b, b_length);
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
