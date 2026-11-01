/**
 *  @brief Parallel string similarity scores in C++ - CPU backends hub.
 *  @file include/stringzillas/similarities.hpp
 *  @author Ash Vardanian
 *
 *  Includes core APIs, defined as the following template objects:
 *
 *  - `sz::levenshtein_distance` & `sz::levenshtein_distance_utf8` for Levenshtein edit-scores.
 *  - `sz::needleman_wunsch_score` for weighted Needleman-Wunsch @b (NW) global alignment.
 *  - `sz::smith_waterman_score` for weighted Smith-Waterman @b (SW) local alignment.
 *
 *  Also includes their batch-capable and parallel versions:
 *
 *  - `sz::levenshtein_distances` & `sz::levenshtein_distances_utf8` for Levenshtein edit-scores.
 *  - `sz::needleman_wunsch_scores` for weighted Needleman-Wunsch global alignment.
 *  - `sz::smith_waterman_scores` for weighted Smith-Waterman local alignment.
 *
 *  This is a thin hub aggregating the per-backend headers. The ISA-agnostic template core and
 *  the serial backend aliases live in `similarities/serial.hpp`; CPU SIMD specializations live
 *  in their own per-ISA files. GPU backends are aggregated by `similarities.cuh`.
 *
 *  @sa similarities/serial.hpp
 *  @sa similarities/haswell.hpp
 *  @sa similarities/icelake.hpp
 */
#ifndef STRINGZILLAS_SIMILARITIES_HPP_
#define STRINGZILLAS_SIMILARITIES_HPP_
#include "stringzillas/similarities/serial.hpp"  
#include "stringzillas/similarities/haswell.hpp" 
#include "stringzillas/similarities/icelake.hpp" 
#include "stringzillas/similarities/neon.hpp"    
#include "stringzillas/similarities/rvv.hpp"     
namespace ashvardanian {
namespace stringzillas {
} 
} 
#if SZ_DYNAMIC_DISPATCH
namespace ashvardanian {
namespace stringzillas {
extern template status_t levenshtein_distance<char, linear_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, linear_gap_costs_t, sz_cap_serial_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, size_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, affine_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, affine_gap_costs_t, sz_cap_serial_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, size_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_cap_serial_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, size_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<affine_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<affine_gap_costs_t, sz_cap_serial_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, size_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_cap_serial_k>::
operator()<forkunion_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                                 forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_cap_serial_k>::
operator()<forkunion_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                                 forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t
smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_cap_serial_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_cap_serial_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
#if SZ_USE_ICELAKE
extern template status_t levenshtein_distance<char, linear_gap_costs_t, sz_caps_sil_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, linear_gap_costs_t, sz_caps_sil_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, size_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, affine_gap_costs_t, sz_caps_sil_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, affine_gap_costs_t, sz_caps_sil_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, size_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sil_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sil_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sil_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sil_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sil_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sil_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sil_k>::operator()<
    dummy_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                      dummy_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sil_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sil_k>::operator()<
    dummy_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                      dummy_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sil_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
#endif 
#if SZ_USE_HASWELL
extern template status_t levenshtein_distance<char, linear_gap_costs_t, sz_caps_sh_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, linear_gap_costs_t, sz_caps_sh_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, affine_gap_costs_t, sz_caps_sh_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, affine_gap_costs_t, sz_caps_sh_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sh_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sh_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sh_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sh_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sh_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sh_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sh_k>::operator()<
    dummy_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                      dummy_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sh_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sh_k>::operator()<
    dummy_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                      dummy_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sh_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
#endif 
#if SZ_USE_NEON
extern template status_t levenshtein_distance<char, linear_gap_costs_t, sz_caps_sn_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, linear_gap_costs_t, sz_caps_sn_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, affine_gap_costs_t, sz_caps_sn_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, affine_gap_costs_t, sz_caps_sn_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sn_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sn_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sn_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sn_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sn_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sn_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sn_k>::operator()<
    dummy_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                      dummy_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sn_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sn_k>::operator()<
    dummy_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                      dummy_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sn_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
#endif 
#if SZ_USE_RVV
extern template status_t levenshtein_distance<char, linear_gap_costs_t, sz_caps_sr_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, linear_gap_costs_t, sz_caps_sr_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, affine_gap_costs_t, sz_caps_sr_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance<char, affine_gap_costs_t, sz_caps_sr_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sr_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sr_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sr_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sr_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sr_k>::operator()<dummy_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sr_k>::operator()<forkunion_executor_t>(
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sr_k>::operator()<
    dummy_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                      dummy_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sr_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sr_k>::operator()<
    dummy_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                      dummy_executor_t &, cpu_specs_t const &) const noexcept;
extern template status_t smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sr_k>::operator()<
    forkunion_executor_t>(span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t,
                          forkunion_executor_t &, cpu_specs_t const &) const noexcept;
#endif 
} 
} 
#endif 
#endif 
