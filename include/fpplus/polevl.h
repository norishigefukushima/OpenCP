#pragma once
#ifndef FPPLUS_POLEVL_H
#define FPPLUS_POLEVL_H

#include <fpplus/eft.h>

inline static double muladd_horner15(double x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8, double c9, double c10, double c11, double c12, double c13, double c14, double c15) {
	double y = c15;
	y = y * x + c14;
	y = y * x + c13;
	y = y * x + c12;
	y = y * x + c11;
	y = y * x + c10;
	y = y * x + c9;
	y = y * x + c8;
	y = y * x + c7;
	y = y * x + c6;
	y = y * x + c5;
	y = y * x + c4;
	y = y * x + c3;
	y = y * x + c2;
	y = y * x + c1;
	y = y * x + c0;
	return y;
}

inline static double fma_horner15(double x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8, double c9, double c10, double c11, double c12, double c13, double c14, double c15) {
	double y = c15;
	y = __builtin_fma(y, x, c14);
	y = __builtin_fma(y, x, c13);
	y = __builtin_fma(y, x, c12);
	y = __builtin_fma(y, x, c11);
	y = __builtin_fma(y, x, c10);
	y = __builtin_fma(y, x, c9);
	y = __builtin_fma(y, x, c8);
	y = __builtin_fma(y, x, c7);
	y = __builtin_fma(y, x, c6);
	y = __builtin_fma(y, x, c5);
	y = __builtin_fma(y, x, c4);
	y = __builtin_fma(y, x, c3);
	y = __builtin_fma(y, x, c2);
	y = __builtin_fma(y, x, c1);
	y = __builtin_fma(y, x, c0);
	return y;
}

inline static double complensated_horner15(double x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8, double c9, double c10, double c11, double c12, double c13, double c14, double c15) {
	double addc, mulc;

	double y = efadd(efmul(c15, x, &mulc), c14, &addc);
	double yc = addc + mulc;

	y = efadd(efmul(y, x, &mulc), c13, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c12, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c11, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c10, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c9, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c8, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c7, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c6, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c5, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c4, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c3, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c2, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c1, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	y = efadd(efmul(y, x, &mulc), c0, &addc);
	yc = __builtin_fma(yc, x, addc + mulc);

	return y + yc;
}

#ifdef __MIC__

inline static __m512d _mm512_muladd_horner15_pd(__m512d x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8, double c9, double c10, double c11, double c12, double c13, double c14, double c15) {
	__m512d y = _mm512_set1_pd(c15);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c14),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c13),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c12),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c11),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c10),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c9),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c8),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c7),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c6),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c5),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c4),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c3),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c2),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c1),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_add_round_pd(
			_mm512_mul_round_pd(y, x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
			_mm512_set1_pd(c0),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	return y;
}

inline static __m512d _mm512_fma_horner15_pd(__m512d x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8, double c9, double c10, double c11, double c12, double c13, double c14, double c15) {
	__m512d y = _mm512_set1_pd(c15);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c14), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c13), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c12), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c11), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c10), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c9), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c8), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c7), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c6), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c5), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c4), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c3), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c1), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = _mm512_fmadd_round_pd(y, x, _mm512_set1_pd(c0), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	return y;
}

inline static __m512d _mm512_comp_horner15_pd(__m512d x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8, double c9, double c10, double c11, double c12, double c13, double c14, double c15) {
	__m512d addc, mulc;

	__m512d y = _mm512_efadd_pd(_mm512_efmul_pd(_mm512_set1_pd(c15), x, &mulc), _mm512_set1_pd(c14), &addc);
	__m512d yc = _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c13), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c12), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c11), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c10), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c9), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c8), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c7), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c6), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c5), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c4), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c3), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c2), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c1), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	y = _mm512_efadd_pd(_mm512_efmul_pd(y, x, &mulc), _mm512_set1_pd(c0), &addc);
	yc = _mm512_fmadd_round_pd(yc, x, _mm512_add_round_pd(addc, mulc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	return _mm512_add_round_pd(y, yc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

#endif

#endif /* FPPLUS_EFT_H */
