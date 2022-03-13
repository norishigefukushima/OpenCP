#pragma once
#ifndef FPPLUS_EFT_H
#define FPPLUS_EFT_H

#include <fpplus/common.h>

/**
 * @defgroup EFT Error-free transforms
 */

/**
 * @ingroup EFT
 * @brief Error-free addition.
 * @details Computes @a s and @a e such that
 *     - s = a + b rounded to the nearest double
 *     - a + b = s + e exactly
 *
 * The algorith is due to @cite Knuth1997. Implementation follows @cite Shewchuk1997, Theorem 7.
 *
 * @par	Computational complexity
 *     <table>
 *         <tr><th>Operation</th><th>Count (default ISA)</th><th>Count (with ADD3)</th></tr>
 *         <tr><td>FP ADD</td><td>6</td><td>1</td></tr>
 *         <tr><td>FP ADD3</td><td></td><td>1</td></tr>
 *     </table>
 *
 * @param[in] a - addend, the first floating-point number to be added.
 * @param[in] b - augend, the second floating-point number to be added.
 * @param[out] e - the roundoff error in floating-point addition.
 * @return The sum @a s of @a and @b rounded to the nearest double-precision number (result of normal floating-point addition).
 *
 * @post @f$ s = \circ(a + b) @f$
 * @post @f$ s + e = a + b @f$
 */
FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
double twosum(
	double a,
	double b,
	double FPPLUS_NONNULL_POINTER(e))
{
#if defined(__CUDA_ARCH__)
	/* CUDA-specific version */
	const double sum = __dadd_rn(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const double b_virtual = __dsub_rn(sum, a);
	const double a_virtual = __dsub_rn(sum, b_virtual);
	const double b_roundoff = __dsub_rn(b, b_virtual);
	const double a_roundoff = __dsub_rn(a, a_virtual);
	*e = __dadd_rn(a_roundoff, b_roundoff);
#else
	*e = addre(a, b);
#endif
	/* End of CUDA-specific version */
#else
	/* Generic version */
	const double sum = a + b;
#if FPPLUS_USE_FPADDRE == 0
	const double b_virtual = sum - a;
	const double a_virtual = sum - b_virtual;
	const double b_roundoff = b - b_virtual;
	const double a_roundoff = a - a_virtual;
	*e = a_roundoff + b_roundoff;
#else
	*e = addre(a, b);
#endif
	/* End of generic version */
#endif
	return sum;
}

/**
 * @ingroup EFT
 * @brief Fast error-free addition of ordered, in magnitude, values.
 * @details Computes @a s and @p e such that
 *     - s = a + b rounded to the nearest double
 *     - a + b = s + e exactly
 *
 * The algorith is due to @cite Dekker1971. Implementation follows @cite Shewchuk1997, Theorem 6.
 *
 * @par	Computational complexity
 *     <table>
 *         <tr><th>Operation</th><th>Count (default ISA)</th><th>Count (with ADD3)</th></tr>
 *         <tr><td>FP ADD</td><td>3</td><td>1</td></tr>
 *         <tr><td>FP ADD3</td><td></td><td>1</td></tr>
 *     </table>
 *
 * @param[in] a - addend, the first floating-point number to be added. Must be not smaller in magnitude than @p b.
 * @param[in] b - augend, the second floating-point number to be added. Must be not larger in magnitude than @p a.
 * @param[out] e - the roundoff error in floating-point addition.
 * @return The sum @a s of @p a and @p b rounded to the nearest double-precision number (result of normal floating-point addition).
 *
 * @pre @f$ |a| >= |b| @f$
 * @post @f$ s = \circ(a + b) @f$
 * @post @f$ s + e = a + b @f$
 */
FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
double twosumfast(
	double a,
	double b,
	double FPPLUS_NONNULL_POINTER(e))
{
#if defined(__CUDA_ARCH__)
	/* CUDA-specific version */
	const double sum = __dadd_rn(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const double b_virtual = __dsub_rn(sum, a);
	*e = __dsub_rn(b, b_virtual);
#else
	*e = addre(a, b);
#endif
	/* End of CUDA-specific version */
#else
	/* Generic version */
	const double sum = a + b;
#if FPPLUS_USE_FPADDRE == 0
	const double b_virtual = sum - a;
	*e = b - b_virtual;
#else
	*e = addre(a, b);
#endif
	/* End of generic version */
#endif
	return sum;
}

/**
 * @ingroup EFT
 * @brief Error-free multiplication.
 * @details Computes @a p and @p e such that
 *     - p = a * b rounded to the nearest double
 *     - a * b = p + e exactly
 *
 * The implementation follows @cite QD2000, Algorithm 7.
 *
 * @par	Computational complexity
 *     <table>
 *         <tr><th>Operation</th><th>Count</th></tr>
 *         <tr><td>FP MUL</td><td>1</td></tr>
 *         <tr><td>FP FMA</td><td>1</td></tr>
 *     </table>
 *
 * @param[in] a - multiplicand, the first floating-point number to be multiplied.
 * @param[in] b - multiplier, the second floating-point number to be multiplied.
 * @param[out] e - the roundoff error in floating-point multiplication.
 * @return The product @a p of @p a and @p b rounded to the nearest double-precision number (result of normal floating-point multiplication).
 *
 * @post @f$ p = \circ(a \times b) @f$
 * @post @f$ p + e = a \times b @f$
 */
FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
double efmul(
	double a,
	double b,
	double FPPLUS_NONNULL_POINTER(e))
{
#if defined(__CUDA_ARCH__)
	/* CUDA-specific version */
	const double product = __dmul_rn(a, b);
	*e = __fma_rn(a, b, -product);
	/* End of CUDA-specific version */
#else
	/* Generic version */
	const double x = a * b;
#if defined(__GNUC__)
	*e = __builtin_fma(a, b, -product);
#else
	*e = fma(a, b, -x);
#endif
	/* End of generic version */
#endif
	return x;
}

/**
 * @ingroup EFT
 * @brief Error-free fused multiply-add.
 * @details Computes @a m, @p e_high and @p e_low such that
 *     - m = a * b + c rounded to the nearest double
 *     - a * b + c = m + e_high + e_low exactly
 *     - |e_high + e_low| <= 0.5 ulp(m)
 *
 * The implementation follows @cite BoldoMuller2005, Property 11 with the second enhancement in Section 5.4.
 *
 * @note The mantissa bits in e_high and e_low may overlap. If you want normalized error, i.e. |e_low| <= 0.5 ulp(u_high), add
 *     <code>e_high = efaddord(e_high, e_low, &e_low)</code>
 * after a call to effma.
 *
 * @par	Computational complexity
 *     <table>
 *         <tr><th>Operation</th><th>Count</th></tr>
 *         <tr><td>FP ADD</td><td>13</td></tr>
 *         <tr><td>FP MUL</td><td>1</td></tr>
 *         <tr><td>FP FMA</td><td>2</td></tr>
 *     </table>
 *
 * @param[in] a - multiplicand, the first floating-point number to be multiplied.
 * @param[in] b - multiplier, the second floating-point number to be multiplied.
 * @param[in] c - augend, the floating-point number to be added to the intermediate product.
 * @param[out] e_high - the high part of the roundoff error in floating-point fused multiply-add operation.
 * @param[out] e_low - the low part of the roundoff error in floating-point fused multiply-add operation.
 * @return The result @a m of @p a * @p b + @p c rounded to the nearest double-precision number (result of normal fused multiply-add).
 *
 * @post @f$ m = \circ(a \times b + c) @f$
 * @post @f$ m + e_{high} + e_{low} = a \times b + c @f$
 * @post @f$ | e_{high} + e_{low} | \leq = \frac{1}{2} \mathrm{ulp}(m) @f$
 */
FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
double effma(
	double a,
	double b,
	double c,
	double FPPLUS_NONNULL_POINTER(e_high),
	double FPPLUS_NONNULL_POINTER(e_low))
{
#if defined(__CUDA_ARCH__)
	/* CUDA-specific version */
	const double mac = __fma_rn(a, b, c);
	double u2;
	const double u1 = efmul(a, b, &u2);
	const double alpha1 = efadd(c, u2, e_low);
	double beta2;
	const double beta1 = efadd(u1, alpha1, &beta2);
	*e_high = __dadd_rn(__dsub_rn(beta1, mac), beta2);
	/* End of CUDA-specific version */
#else
	/* Generic version */
#if defined(__GNUC__)
	const double mac = __builtin_fma(a, b, c);
#else
	const double mac = fma(a, b, c);
#endif
	double u2;
	const double u1 = efmul(a, b, &u2);
	const double alpha1 = twosum(c, u2, e_low);
	double beta2;
	const double beta1 = twosum(u1, alpha1, &beta2);
	*e_high = (beta1 - mac) + beta2;
	/* End of generic version */
#endif
	return mac;
}

#if defined(__AVX__) && (defined(__FMA__) || defined(__FMA4__) || defined(__AVX2__))

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m128d _mm_efadd_sd(
	__m128d a,
	__m128d b,
	__m128d FPPLUS_NONNULL_POINTER(e))
{
	const __m128d sum = _mm_add_sd(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const __m128d b_virtual = _mm_sub_sd(sum, a);
	const __m128d a_virtual = _mm_sub_sd(sum, b_virtual);
	const __m128d b_roundoff = _mm_sub_sd(b, b_virtual);
	const __m128d a_roundoff = _mm_sub_sd(a, a_virtual);
	*e = _mm_add_sd(a_roundoff, b_roundoff);
#else
	*e = _mm_addre_sd(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m128d _mm_efadd_pd(
	__m128d a,
	__m128d b,
	__m128d FPPLUS_NONNULL_POINTER(e))
{
	const __m128d sum = _mm_add_pd(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const __m128d b_virtual = _mm_sub_pd(sum, a);
	const __m128d a_virtual = _mm_sub_pd(sum, b_virtual);
	const __m128d b_roundoff = _mm_sub_pd(b, b_virtual);
	const __m128d a_roundoff = _mm_sub_pd(a, a_virtual);
	*e = _mm_add_pd(a_roundoff, b_roundoff);
#else
	*e = _mm_addre_pd(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m128d _mm_efaddord_sd(
	__m128d a,
	__m128d b,
	__m128d FPPLUS_NONNULL_POINTER(e))
{
	const __m128d sum = _mm_add_sd(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const __m128d b_virtual = _mm_sub_sd(sum, a);
	*e = _mm_sub_sd(b, b_virtual);
#else
	*e = _mm_addre_sd(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m128d _mm_efaddord_pd(
	__m128d a,
	__m128d b,
	__m128d FPPLUS_NONNULL_POINTER(e))
{
	const __m128d sum = _mm_add_pd(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const __m128d b_virtual = _mm_sub_pd(sum, a);
	*e = _mm_sub_pd(b, b_virtual);
#else
	*e = _mm_addre_pd(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m128d _mm_efmul_sd(
	__m128d a,
	__m128d b,
	__m128d FPPLUS_NONNULL_POINTER(e))
{
	const __m128d product = _mm_mul_sd(a, b);
#if defined(__FMA__) || defined(__AVX2__)
	*e = _mm_fmsub_sd(a, b, product);
#else
	*e = _mm_msub_sd(a, b, product);
#endif
	return product;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m128d _mm_efmul_pd(
	__m128d a,
	__m128d b,
	__m128d FPPLUS_NONNULL_POINTER(e))
{
	const __m128d product = _mm_mul_pd(a, b);
#if defined(__FMA__) || defined(__AVX2__)
	*e = _mm_fmsub_pd(a, b, product);
#else
	*e = _mm_msub_pd(a, b, product);
#endif
	return product;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m128d _mm_effma_pd(
	__m128d a,
	__m128d b,
	__m128d c,
	__m128d FPPLUS_NONNULL_POINTER(e_high),
	__m128d FPPLUS_NONNULL_POINTER(e_low))
{
#if defined(__FMA__) || defined(__AVX2__)
	const __m128d mac = _mm_fmadd_pd(a, b, c);
#else
	const __m128d mac = _mm_macc_pd(a, b, c);
#endif
	__m128d u2;
	const __m128d u1 = _mm_efmul_pd(a, b, &u2);
	const __m128d alpha1 = _mm_efadd_pd(c, u2, e_low);
	__m128d beta2;
	const __m128d beta1 = _mm_efadd_pd(u1, alpha1, &beta2);
	*e_high = _mm_add_pd(_mm_sub_pd(beta1, mac), beta2);
	return mac;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m256d _mm256_twosum_pd(
	__m256d a,
	__m256d b,
	__m256d FPPLUS_NONNULL_POINTER(e))
{
	const __m256d x = _mm256_add_pd(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const __m256d b_virtual = _mm256_sub_pd(x, a);
	const __m256d a_virtual = _mm256_sub_pd(x, b_virtual);
	const __m256d b_roundoff = _mm256_sub_pd(b, b_virtual);
	const __m256d a_roundoff = _mm256_sub_pd(a, a_virtual);
	*e = _mm256_add_pd(a_roundoff, b_roundoff);
#else
	*e = _mm256_addre_pd(a, b);
#endif
	return x;
}

//a-b
FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m256d _mm256_twonsum_pd(
	__m256d a,
	__m256d b,
	__m256d FPPLUS_NONNULL_POINTER(e))
{
	const __m256d x = _mm256_sub_pd(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const __m256d b_virtual = _mm256_sub_pd(x, a);
	const __m256d a_virtual = _mm256_sub_pd(x, b_virtual);
	const __m256d b_roundoff = _mm256_add_pd(b, b_virtual);
	const __m256d a_roundoff = _mm256_sub_pd(a, a_virtual);
	*e = _mm256_sub_pd(a_roundoff, b_roundoff);
#else
	*e = _mm256_addre_pd(a, b);
#endif
	return x;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m256d _mm256_twosumfast_pd(
	__m256d a,
	__m256d b,
	__m256d FPPLUS_NONNULL_POINTER(e))
{
	const __m256d x = _mm256_add_pd(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const __m256d b_virtual = _mm256_sub_pd(x, a);
	*e = _mm256_sub_pd(b, b_virtual);
#else
	*e = _mm256_addre_pd(a, b);
#endif
	return x;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m256d _mm256_twoproduct_pd(
	__m256d a,
	__m256d b,
	__m256d FPPLUS_NONNULL_POINTER(e))
{
	const __m256d x = _mm256_mul_pd(a, b);
#if defined(__FMA__) || defined(__AVX2__)
	*e = _mm256_fmsub_pd(a, b, x);
#else
	*e = _mm256_msub_pd(a, b, product);
#endif
	return x;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m256d _mm256_effma_pd(
	__m256d a,
	__m256d b,
	__m256d c,
	__m256d FPPLUS_NONNULL_POINTER(e_high),
	__m256d FPPLUS_NONNULL_POINTER(e_low))
{
#if defined(__FMA__) || defined(__AVX2__)
	const __m256d mac = _mm256_fmadd_pd(a, b, c);
#else
	const __m256d mac = _mm256_macc_pd(a, b, c);
#endif
	__m256d u2;
	const __m256d u1 = _mm256_twoproduct_pd(a, b, &u2);
	const __m256d alpha1 = _mm256_twosum_pd(c, u2, e_low);
	__m256d beta2;
	const __m256d beta1 = _mm256_twosum_pd(u1, alpha1, &beta2);
	*e_high = _mm256_add_pd(_mm256_sub_pd(beta1, mac), beta2);
	return mac;
}

#endif /* AVX */

#if defined(__AVX512F__) || defined(__KNC__)

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m512d _mm512_efadd_pd(
	__m512d a,
	__m512d b,
	__m512d FPPLUS_NONNULL_POINTER(e))
{
	const __m512d sum = _mm512_add_round_pd(a, b, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
#if FPPLUS_USE_FPADDRE == 0
	const __m512d b_virtual = _mm512_sub_round_pd(sum, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	const __m512d a_virtual = _mm512_sub_round_pd(sum, b_virtual, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	const __m512d b_roundoff = _mm512_sub_round_pd(b, b_virtual, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	const __m512d a_roundoff = _mm512_sub_round_pd(a, a_virtual, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	*e = _mm512_add_round_pd(a_roundoff, b_roundoff, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
	*e = _mm512_addre_pd(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m512d _mm512_efaddord_pd(
	__m512d a,
	__m512d b,
	__m512d FPPLUS_NONNULL_POINTER(e))
{
	const __m512d sum = _mm512_add_round_pd(a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#if FPPLUS_USE_FPADDRE == 0
	const __m512d b_virtual = _mm512_sub_round_pd(sum, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	*e = _mm512_sub_round_pd(b, b_virtual, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
	*e = _mm512_addre_pd(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m512d _mm512_efmul_pd(
	__m512d a,
	__m512d b,
	__m512d FPPLUS_NONNULL_POINTER(e))
{
	const __m512d product = _mm512_mul_round_pd(a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	*e = _mm512_fmsub_round_pd(a, b, product, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	return product;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__m512d _mm512_effma_pd(
	__m512d a,
	__m512d b,
	__m512d c,
	__m512d FPPLUS_NONNULL_POINTER(e_high),
	__m512d FPPLUS_NONNULL_POINTER(e_low))
{
	const __m512d mac = _mm512_fmadd_round_pd(a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	__m512d u2;
	const __m512d u1 = _mm512_efmul_pd(a, b, &u2);
	const __m512d alpha1 = _mm512_efadd_pd(c, u2, e_low);
	__m512d beta2;
	const __m512d beta1 = _mm512_efadd_pd(u1, alpha1, &beta2);
	*e_high = _mm512_add_round_pd(
		_mm512_sub_round_pd(beta1, mac, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
		beta2,
		_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	return mac;
}

#endif /* Intel KNC or AVX-512 */

#if defined(__bgq__)

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
vector4double vec_efadd(
	vector4double a,
	vector4double b,
	vector4double FPPLUS_NONNULL_POINTER(e))
{
	const vector4double sum = vec_add(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const vector4double b_virtual = vec_sub(sum, a);
	const vector4double a_virtual = vec_sub(sum, b_virtual);
	const vector4double b_roundoff = vec_sub(b, b_virtual);
	const vector4double a_roundoff = vec_sub(a, a_virtual);
	*e = vec_add(a_roundoff, b_roundoff);
#else
	*e = vec_addre(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
vector4double vec_efaddord(
	vector4double a,
	vector4double b,
	vector4double FPPLUS_NONNULL_POINTER(e))
{
	const vector4double sum = vec_add(a, b);
#if FPPLUS_USE_FPADDRE == 0
	const vector4double b_virtual = vec_sub(sum, a);
	*e = vec_sub(b, b_virtual);
#else
	*e = vec_addre(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
vector4double vec_efmul(
	vector4double a,
	vector4double b,
	vector4double FPPLUS_NONNULL_POINTER(e))
{
	const vector4double product = vec_mul(a, b);
	*e = vec_msub(a, b, product);
	return product;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
vector4double vec_effma(
	vector4double a,
	vector4double b,
	vector4double c,
	vector4double FPPLUS_NONNULL_POINTER(e_high),
	vector4double FPPLUS_NONNULL_POINTER(e_low))
{
	const vector4double mac = vec_madd(a, b, c);
	vector4double u2;
	const vector4double u1 = vec_efmul(a, b, &u2);
	const vector4double alpha1 = vec_efadd(c, u2, e_low);
	vector4double beta2;
	const vector4double beta1 = vec_efadd(u1, alpha1, &beta2);
	*e_high = vec_add(vec_sub(beta1, mac), beta2);
	return mac;
}

#endif /* Blue Gene/Q */

#if defined(__VSX__)

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__vector double vec_efadd(
	__vector double a,
	__vector double b,
	__vector double FPPLUS_NONNULL_POINTER(e))
{
	const __vector double sum = vec_add(a, b);
#ifndef FPPLUS_EMULATE_FPADDRE
	const __vector double b_virtual = vec_sub(sum, a);
	const __vector double a_virtual = vec_sub(sum, b_virtual);
	const __vector double b_roundoff = vec_sub(b, b_virtual);
	const __vector double a_roundoff = vec_sub(a, a_virtual);
	*e = vec_add(a_roundoff, b_roundoff);
#else
	*e = vec_addre(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__vector double vec_efaddord(
	__vector double a,
	__vector double b,
	__vector double FPPLUS_NONNULL_POINTER(e))
{
	const __vector double sum = vec_add(a, b);
#ifndef FPPLUS_EMULATE_FPADDRE
	const __vector double b_virtual = vec_sub(sum, a);
	*e = vec_sub(b, b_virtual);
#else
	*e = vec_addre(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__vector double vec_efmul(
	__vector double a,
	__vector double b,
	__vector double FPPLUS_NONNULL_POINTER(e))
{
	const __vector double product = vec_mul(a, b);
	*e = vec_msub(a, b, product);
	return product;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
__vector double vec_effma(
	__vector double a,
	__vector double b,
	__vector double c,
	__vector double FPPLUS_NONNULL_POINTER(e_high),
	__vector double FPPLUS_NONNULL_POINTER(e_low))
{
	const __vector double mac = vec_madd(a, b, c);
	__vector double u2;
	const __vector double u1 = vec_efmul(a, b, &u2);
	const __vector double alpha1 = vec_efadd(c, u2, e_low);
	__vector double beta2;
	const __vector double beta1 = vec_efadd(u1, alpha1, &beta2);
	*e_high = vec_add(vec_sub(beta1, mac), beta2);
	return mac;
}

#endif /* IBM VSX */

#if defined(__ARM_ARCH_8A__)

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
float64x1_t vefadd_f64(
	float64x1_t a,
	float64x1_t b,
	float64x1_t FPPLUS_NONNULL_POINTER(e))
{
	const float64x1_t sum = vadd_f64(a, b);
#ifndef FPPLUS_EMULATE_FPADDRE
	const float64x1_t b_virtual = vsub_f64(sum, a);
	const float64x1_t a_virtual = vsub_f64(sum, b_virtual);
	const float64x1_t b_roundoff = vsub_f64(b, b_virtual);
	const float64x1_t a_roundoff = vsub_f64(a, a_virtual);
	*e = vadd_f64(a_roundoff, b_roundoff);
#else
	*e = vaddre_f64(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
float64x1_t vefaddord_f64(
	float64x1_t a,
	float64x1_t b,
	float64x1_t FPPLUS_NONNULL_POINTER(e))
{
	const float64x1_t sum = vadd_f64(a, b);
#ifndef FPPLUS_EMULATE_FPADDRE
	const float64x1_t b_virtual = vsub_f64(sum, a);
	*e = vsub_f64(b, b_virtual);
#else
	*e = vaddre_f64(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
float64x1_t vefmulq_f64(
	float64x1_t a,
	float64x1_t b,
	float64x1_t FPPLUS_NONNULL_POINTER(e))
{
	const float64x1_t product = vmul_f64(a, b);
	*e = vfma_f64(vneg_f64(product), a, b);
	return product;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
float64x1_t veffma_f64(
	float64x1_t a,
	float64x1_t b,
	float64x1_t c,
	float64x1_t FPPLUS_NONNULL_POINTER(e_high),
	float64x1_t FPPLUS_NONNULL_POINTER(e_low))
{
	const float64x1_t mac = vfma_f64(a, b, c);
	float64x1_t u2;
	const float64x1_t u1 = vefmul_f64(a, b, &u2);
	const float64x1_t alpha1 = vefadd_f64(c, u2, e_low);
	float64x1_t beta2;
	const float64x1_t beta1 = vefadd_f64(u1, alpha1, &beta2);
	*e_high = vadd_f64(vsub_f64(beta1, mac), beta2);
	return mac;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
float64x2_t vefaddq_f64(
	float64x2_t a,
	float64x2_t b,
	float64x2_t FPPLUS_NONNULL_POINTER(e))
{
	const float64x2_t sum = vaddq_f64(a, b);
#ifndef FPPLUS_EMULATE_FPADDRE
	const float64x2_t b_virtual = vsubq_f64(sum, a);
	const float64x2_t a_virtual = vsubq_f64(sum, b_virtual);
	const float64x2_t b_roundoff = vsubq_f64(b, b_virtual);
	const float64x2_t a_roundoff = vsubq_f64(a, a_virtual);
	*e = vaddq_f64(a_roundoff, b_roundoff);
#else
	*e = vaddreq_f64(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
float64x2_t vefaddordq_f64(
	float64x2_t a,
	float64x2_t b,
	float64x2_t FPPLUS_NONNULL_POINTER(e))
{
	const float64x2_t sum = vaddq_f64(a, b);
#ifndef FPPLUS_EMULATE_FPADDRE
	const float64x2_t b_virtual = vsubq_f64(sum, a);
	*e = vsubq_f64(b, b_virtual);
#else
	*e = vaddreq_f64(a, b);
#endif
	return sum;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
float64x2_t vefmulq_f64(
	float64x2_t a,
	float64x2_t b,
	float64x2_t FPPLUS_NONNULL_POINTER(e))
{
	const float64x2_t product = vmulq_f64(a, b);
	*e = vfmaq_f64(vnegq_f64(product), a, b);
	return product;
}

FPPLUS_STATIC_INLINE FPPLUS_NONNULL_POINTER_ARGUMENTS
float64x2_t veffmaq_f64(
	float64x2_t a,
	float64x2_t b,
	float64x2_t c,
	float64x2_t FPPLUS_NONNULL_POINTER(e_high),
	float64x2_t FPPLUS_NONNULL_POINTER(e_low))
{
	const float64x2_t mac = vfmaq_f64(a, b, c);
	float64x2_t u2;
	const float64x2_t u1 = vefmulq_f64(a, b, &u2);
	const float64x2_t alpha1 = vefaddq_f64(c, u2, e_low);
	float64x2_t beta2;
	const float64x2_t beta1 = vefaddq_f64(u1, alpha1, &beta2);
	*e_high = vaddq_f64(vsubq_f64(beta1, mac), beta2);
	return mac;
}

#endif /* ARMv8-A */

#endif /* FPPLUS_EFT_H */
