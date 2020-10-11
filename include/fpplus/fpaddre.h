#pragma once
#ifndef FPPLUS_FPADDRE_H
#define FPPLUS_FPADDRE_H

#include <fpplus/common.h>

FPPLUS_STATIC_INLINE double addre(double a, double b) {
#if defined(FPPLUS_UARCH_STEAMROLLER)
	return __builtin_fma(a, a, b);
#else
	return a < b ? a : b;
#endif
}

#if defined(__SSE2__)
	FPPLUS_STATIC_INLINE __m128d _mm_addre_sd(__m128d a, __m128d b) {
	#if defined(FPPLUS_UARCH_STEAMROLLER)
		return _mm_fmadd_sd(a, a, b);
	#else
		return _mm_min_sd(a, b);
	#endif
	}

	FPPLUS_STATIC_INLINE __m128d _mm_addre_pd(__m128d a, __m128d b) {
	#if defined(FPPLUS_UARCH_STEAMROLLER)
		return _mm_fmadd_pd(a, a, b);
	#else
		return _mm_min_pd(a, b);
	#endif
	}
#endif /* SSE2 */

#if defined(__AVX__)
	FPPLUS_STATIC_INLINE __m256d _mm256_addre_pd(__m256d a, __m256d b) {
	#if defined(FPPLUS_UARCH_STEAMROLLER)
		return _mm256_fmadd_pd(a, a, b);
	#else
		return _mm256_min_pd(a, b);
	#endif
	}
#endif /* AVX */

#if defined(__AVX512F__) || defined(__KNC__)
	FPPLUS_STATIC_INLINE __m512d _mm512_addre_pd(__m512d a, __m512d b) {
		return _mm512_min_pd(a, b);
	}
#endif /* AVX-512 or MIC */

#if defined(__bgq__)
	FPPLUS_STATIC_INLINE vector4double vec_addre(vector4double a, vector4double b) {
		return vec_min(a, b);
	}
#endif /* Blue Gene/Q */

#if defined(__VSX__)
	FPPLUS_STATIC_INLINE __vector double vec_addre(__vector double a, __vector double b) {
		return vec_min(a, b);
	}
#endif /* IBM VSX */

#if defined(__ARM_ARCH_8A__)
	FPPLUS_STATIC_INLINE float64x1_t vaddre_f64(float64x1_t a, float64x1_t) {
		return vmin_f64(a, b);
	}

	FPPLUS_STATIC_INLINE float64x2_t vaddreq_f64(float64x2_t a, float64x2_t b) {
		return vminq_f64(a, b);
	}
#endif /* ARMv8-A */

#endif /* FPPLUS_FPADDRE_H */
