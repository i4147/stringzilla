/**
 *  @file c/stringzillas/levenshtein_icelake.cpp
 *  @brief Ice Lake (AVX-512) single-pair scorer instantiations for parallel Levenshtein distances.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include "stringzillas/similarities.hpp" 
#if SZ_USE_ICELAKE
namespace ashvardanian {
namespace stringzillas {
template status_t                                                                            
levenshtein_distance<char, linear_gap_costs_t, sz_caps_sil_k>::operator()<dummy_executor_t>( 
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                                
levenshtein_distance<char, linear_gap_costs_t, sz_caps_sil_k>::operator()<forkunion_executor_t>( 
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                            
levenshtein_distance<char, affine_gap_costs_t, sz_caps_sil_k>::operator()<dummy_executor_t>( 
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                                
levenshtein_distance<char, affine_gap_costs_t, sz_caps_sil_k>::operator()<forkunion_executor_t>( 
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                           
levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sil_k>::operator()<dummy_executor_t>( 
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                               
levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sil_k>::operator()<forkunion_executor_t>( 
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
} 
} 
#endif 
