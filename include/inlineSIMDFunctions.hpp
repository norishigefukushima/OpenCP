#pragma once

inline int get_simd_ceil(const int val, const int simdwidth)
{
	return (val%simdwidth == 0) ? val : (val / simdwidth + 1)*simdwidth;
}

inline int get_simd_floor(const int val, const int simdwidth)
{
	return (val / simdwidth)*simdwidth;
}

inline float _mm256_reduceadd_ps(__m256 src)
{
	src = _mm256_hadd_ps(src, src);
	src = _mm256_hadd_ps(src, src);
	return (src.m256_f32[0] + src.m256_f32[4]);
}

inline __m256 _mm256_load_epu8cvtps(const __m128i* P)
{
	return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_load_si128((__m128i*)P)));
}