/**
 * \file gaussian_conv_vyv.c
 * \brief Vliet-Young-Verbeek approximate Gaussian convolution
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * Copyright (c) 2012-2013, Pascal Getreuer
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under, at your option, the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version, or the terms of the
 * simplified BSD license.
 *
 * You should have received a copy of these licenses along this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
//#include <complex>
#include "complex_arith.h"
#include "gaussian_conv.h"

/** \brief Number of newton iterations used to determine q */
#define YVY_NUM_NEWTON_ITERATIONS       6

/**
 * \brief Compute the variance of the impulse response
 * \param poles0    unscaled pole locations
 * \param q         rescaling parameter
 * \param K         number of poles
 * \return variance achieved by poles = poles0^(1/q)
 * \ingroup vyv_gaussian
 */
static double variance(const complex4c *poles0, int K, double q)
{
	complex4c sum = { 0, 0 };
	int k;

	for (k = 0; k < K; ++k)
	{
		complex4c z = c_real_pow(poles0[k], 1 / q), denom = z;
		denom.real -= 1;
		/* Compute sum += z / (z - 1)^2. */
		sum = c_add(sum, c_div(z, c_mul(denom, denom)));
	}

	return 2 * sum.real;
}

/**
 * \brief Derivative of variance with respect to q
 * \param poles0    unscaled pole locations
 * \param q         rescaling parameter
 * \param K         number of poles
 * \return derivative of variance with respect to q
 * \ingroup vyv_gaussian
 *
 * This function is used by compute_q() in solving for q.
 */
static double dq_variance(const complex4c *poles0, int K, double q)
{
	complex4c sum = { 0, 0 };
	int k;

	for (k = 0; k < K; ++k)
	{
		complex4c z = c_real_pow(poles0[k], 1 / q), w = z, denom = z;
		w.real += 1;
		denom.real -= 1;
		/* Compute sum += z log(z) (z + 1) / (z - 1)^3 */
		sum = c_add(sum, c_div(c_mul(c_mul(z, c_log(z)), w),
			c_real_pow(denom, 3)));
	}

	return (2 / q) * sum.real;
}

/**
 * \brief Compute q for a desired sigma using Newton's method
 * \param poles0    unscaled pole locations
 * \param K         number of poles
 * \param sigma     the desired sigma
 * \param q0        initial estimate of q
 * \return refined value of q
 * \ingroup vyv_gaussian
 *
 * This routine uses Newton's method to solve for the value of q so that the
 * filter achieves the specified variance,
 * \f[ \operatorname{var}(h) = \sum_{k=1}^K \frac{2 d_k^{1/q}}
 {(d_k^{1/q} - 1)^2} = \sigma^2, \f]
 * where the \f$ d_k \f$ are the unscaled pole locations.
 */
static double compute_q(const complex4c *poles0, int K,
	double sigma, double q0)
{
	double sigma2 = sigma * sigma;
	double q = q0;
	int i;

	for (i = 0; i < YVY_NUM_NEWTON_ITERATIONS; ++i)
		q -= (variance(poles0, K, q) - sigma2)
		/ dq_variance(poles0, K, q);

	return q;
}

/**
 * \brief Expand pole product
 * \param c         resulting filter coefficients
 * \param poles     pole locations
 * \param K         number of poles
 * \ingroup vyv_gaussian
 *
 * This routine expands the product to obtain the filter coefficients:
 * \f[ \prod_{k=0}^{K-1}\frac{\mathrm{poles}[k]-1}{\mathrm{poles}[k]-z^{-1}}
 = \frac{c[0]}{1+\sum_{k=1}^K c[k] z^{-k}}. \f]
 */
static void expand_pole_product(double *c, const complex4c *poles, int K)
{
	complex4c denom[VYV_MAX_K + 1];
	int k, j;

	assert(K <= VYV_MAX_K);
	denom[0] = poles[0];
	denom[1] = make_complex(-1, 0);

	for (k = 1; k < K; ++k)
	{
		denom[k + 1] = c_neg(denom[k]);

		for (j = k; j > 0; --j)
			denom[j] = c_sub(c_mul(denom[j], poles[k]), denom[j - 1]);

		denom[0] = c_mul(denom[0], poles[k]);
	}

	for (k = 1; k <= K; ++k)
		c[k] = c_div(denom[k], denom[0]).real;

	for (c[0] = 1, k = 1; k <= K; ++k)
		c[0] += c[k];

	return;
}

/**
 * \brief Precomputations for Vliet-Young-Verbeek Gaussian approximation
 * \param c         coefficients structure to populate
 * \param sigma     the standard deviation of the Gaussian in pixels
 * \param K         filter order = 3, 4, or 5
 * \param tol       accuracy for initialization on the left boundary
 *
 * This routine precomputes the filter coefficients for Vliet-Young-Verbeek
 * approximate Gaussian convolution for use in vyv_gaussian_conv() or
 * vyv_gaussian_conv_image(). The filter coefficients are computed by the
 * following steps:
 *
 * 1. For the specified sigma value, compute_q() solves for the value of q
 *    so that the filter achieves the correct variance,
 * \f[ \operatorname{var}(h) = \sum_{k=1}^K \frac{2 d_k^{1/q}}
 {(d_k^{1/q} - 1)^2} = \sigma^2, \f]
 *    where the \f$ d_k \f$ are the unscaled pole locations.
 * 2. The pole locations are scaled, \f$ \mathrm{poles}[k] = d_k^{1/q} \f$.
 * 3. The filter is algebraically rearranged by expand_pole_product() as
 * \f[ \prod_{k=0}^{K-1}\frac{\mathrm{poles}[k]-1}{\mathrm{poles}[k]-z^{-1}}
 = \frac{c[0]}{1+\sum_{k=1}^K c[k] z^{-k}}. \f]
 *
 * For handling the right boundary, the routine precomputes the inverse
 * to the linear system
 * \f[ u_{N-m} = b_0 q_{N-m} - \sum_{k=1}^K a_k \Tilde{u}_{N-m+k},
 \quad m=1,\ldots,K. \f]
 * The inverse is stored in matrix `c->M`, ordered such that
 * \f[ u_{N-K+m}=\sum_{n=0}^{K-1}M(m,n)\,q_{N-K+m},\quad m=0,\ldots,K-1. \f]
 */

template <typename srcType>
void vyv_precomp_(vyv_coeffs<srcType> *c, srcType sigma, int K, srcType tol)
{
	/* Optimized unscaled pole locations. */
	__declspec(align(16)) static const complex4c poles0[VYV_MAX_K - VYV_MIN_K + 1][5] = {
		{ { 1.4165, 1.00829 }, { 1.4165, -1.00829 }, { 1.86543, 0 } },
		{ { 1.13228, 1.28114 }, { 1.13228, -1.28114 },
		{ 1.78534, 0.46763 }, { 1.78534, -0.46763 } },
		{ { 0.8643, 1.45389 }, { 0.8643, -1.45389 },
		{ 1.61433, 0.83134 }, { 1.61433, -0.83134 }, { 1.87504, 0 } }
	};
	__declspec(align(16)) complex4c poles[VYV_MAX_K];
	double q;
	__declspec(align(16)) double filter[VYV_MAX_K + 1];
	__declspec(align(16)) double A[VYV_MAX_K * VYV_MAX_K], inv_A[VYV_MAX_K * VYV_MAX_K];
	int i, j, matrix_size;

	assert(c && sigma > 0 && VYV_VALID_K(K) && tol > 0);

	/* Make a crude initial estimate of q. */
	q = sigma / 2;
	/* Compute an accurate value of q using Newton's method. */
	q = compute_q(poles0[K - VYV_MIN_K], K, sigma, q);

	for (i = 0; i < K; ++i)
		poles[i] = c_real_pow(poles0[K - VYV_MIN_K][i], 1 / q);

	/* Compute the filter coefficients b_0, a_1, ..., a_K. */
	expand_pole_product(filter, poles, K);

	/* Compute matrix for handling the right boundary. */
	for (i = 0, matrix_size = K * K; i < matrix_size; ++i)
		A[i] = 0.0;

	for (i = 0; i < K; ++i)
		for (A[i + K * i] = 1.0, j = 1; j <= K; ++j)
			A[i + K * extension(K, i + j)] += filter[j];

	invert_matrix(inv_A, A, K);

	for (i = 0; i < matrix_size; ++i)
		inv_A[i] *= filter[0];

	/* Store precomputations in coeffs struct. */
	for (i = 0; i <= K; ++i)
		c->filter[i] = (srcType)filter[i];

	for (i = 0; i < matrix_size; ++i)
		c->M[i] = (srcType)inv_A[i];

	c->K = K;
	c->sigma = (srcType)sigma;
	c->tol = tol;
	c->max_iter = (int)(10.0 * sigma);
	return;
}

void vyv_precomp(vyv_coeffs<float> *c, float sigma, int K, float tol)
{
	vyv_precomp_<float>(c, sigma, K, tol);
}
void vyv_precomp(vyv_coeffs<double> *c, double sigma, int K, double tol)
{
	vyv_precomp_<double>(c, sigma, K, tol);
}
/**
 * \brief Gaussian convolution Vliet-Young-Verbeek approximation
 * \param c         vyv_coeffs created by vyv_precomp()
 * \param dest      output convolved data
 * \param src       data to be convolved, modified in-place if src = dest
 * \param N         number of samples
 * \param stride    stride between successive samples
 *
 * This routine performs Vliet-Young-Verbeek recursive filtering Gaussian
 * convolution approximation. The Gaussian is approximated by a causal filter
 * followed by an anticausal filter,
 * \f[ H(z) = G(z) G(z^{-1}), \quad
 G(z) = \frac{b_0}{1 + a_1 z^{-1} + \cdots + a_K z^{-K}}. \f]
 *
 * The array \c c.filter holds the filter coefficients \f$ b_0, a_k \f$,
 * which are precomputed by vyv_precomp().
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result).
 *
 * \note When the #num typedef is set to single-precision arithmetic,
 * results may be inaccurate for large values of sigma.
 */

template <typename srcType>
void vyv_gaussian_conv(const vyv_coeffs<srcType> c, srcType *dest, const srcType *src, const int N, const int stride)
{
	const int stride_2 = stride * 2;
	const int stride_3 = stride * 3;
	const int stride_4 = stride * 4;
	const int stride_5 = stride * 5;
	const int stride_N = stride * N;
	srcType q[VYV_MAX_K];
	long i;
	int m, n;

	assert(dest && src && N > 0 && stride != 0);

	if (N <= 4)
	{   /* Special case for very short signals. */
		gaussian_short_conv(dest, src, N, stride, c.sigma);
		return;
	}


	/* Handle the left boundary. */
	init_recursive_filter_<srcType>(q, src, N, stride, c.filter, 0, c.filter, c.K, (srcType)1.0, c.tol, c.max_iter);

	for (m = 0; m < c.K; ++m)
		dest[stride * m] = q[m];

	/* The following applies the causal recursive filter according to the
	   filter order c.K. The loops implement the pseudocode

	   For n = K, ..., N - 1,
	   dest(n) = filter(0) src(n) - \sum_{k=1}^K dest(n - k)

	   Variable i = stride * n is the offset to the nth sample. */

	switch (c.K)
	{
	case 3:
		for (i = stride_3; i < stride_N; i += stride)
			dest[i] = c.filter[0] * src[i]
			- c.filter[1] * dest[i - stride]
			- c.filter[2] * dest[i - stride_2]
			- c.filter[3] * dest[i - stride_3];
		break;
	case 4:
		for (i = stride_4; i < stride_N; i += stride)
			dest[i] = c.filter[0] * src[i]
			- c.filter[1] * dest[i - stride]
			- c.filter[2] * dest[i - stride_2]
			- c.filter[3] * dest[i - stride_3]
			- c.filter[4] * dest[i - stride_4];
		break;
	case 5:
		for (i = stride_5; i < stride_N; i += stride)
			dest[i] = c.filter[0] * src[i]
			- c.filter[1] * dest[i - stride]
			- c.filter[2] * dest[i - stride_2]
			- c.filter[3] * dest[i - stride_3]
			- c.filter[4] * dest[i - stride_4]
			- c.filter[5] * dest[i - stride_5];
		break;
	}

	/* Handle the right boundary by multiplying matrix c.M with last K
	   samples dest(N - K - m), m = 0, ..., K - 1. */

	/* Copy last K samples into array q, q(m) = dest(N - K + m). */
	for (m = 0; m < c.K; ++m)
		q[m] = dest[stride_N - stride * (c.K - m)];

	/* Perform matrix multiplication,
	   dest(N - K + m) = \sum_{n=0}^{K-1} M(m, n) q(n). */
	for (m = 0; m < c.K; ++m)
	{
		srcType accum = (srcType)0.0;

		for (n = 0; n < c.K; ++n)
			accum += c.M[m + c.K*n] * q[n];

		dest[stride_N - stride * (c.K - m)] = accum;
	}

	/* The following applies the anticausal filter, implementing the
	   pseudocode

	   For n = N - K - 1, ..., 0,
	   dest(n) = filter(0) dest(n) - \sum_{k=1}^K dest(n + k)

	   Variable i = stride * n is the offset to the nth sample. */

	switch (c.K)
	{
	case 3:
		for (i = stride_N - stride_4; i >= 0; i -= stride)
			dest[i] = c.filter[0] * dest[i]
			- c.filter[1] * dest[i + stride]
			- c.filter[2] * dest[i + stride_2]
			- c.filter[3] * dest[i + stride_3];
		break;
	case 4:

		for (i = stride_N - stride_5; i >= 0; i -= stride)
			dest[i] = c.filter[0] * dest[i]
			- c.filter[1] * dest[i + stride]
			- c.filter[2] * dest[i + stride_2]
			- c.filter[3] * dest[i + stride_3]
			- c.filter[4] * dest[i + stride_4];
		break;

	case 5:
		for (i = stride_N - stride * 6; i >= 0; i -= stride)
			dest[i] = c.filter[0] * dest[i]
			- c.filter[1] * dest[i + stride]
			- c.filter[2] * dest[i + stride_2]
			- c.filter[3] * dest[i + stride_3]
			- c.filter[4] * dest[i + stride_4]
			- c.filter[5] * dest[i + stride_5];
		break;
	}

	return;
}

template <typename srcType>
void vyv_gaussian_conv_w(const vyv_coeffs<srcType> c, srcType *dest, const srcType *src, const int N)
{
	//const vyv_coeffs<T> c = c_;
	__declspec(align(16)) srcType q[VYV_MAX_K];
	long i;
	int m, n;

	//assert(dest && src && N > 0 && stride != 0);

	if (N <= 4)
	{   /* Special case for very short signals. */
		gaussian_short_conv(dest, src, N, 1, c.sigma);
		return;
	}


	/* Handle the left boundary. */
	init_recursive_filter_<srcType>(q, src, N, 1, c.filter, 0, c.filter, c.K, (srcType)1.0, c.tol, c.max_iter);

	for (m = 0; m < c.K; ++m)
		dest[m] = q[m];

	/* The following applies the causal recursive filter according to the
	filter order c.K. The loops implement the pseudocode

	For n = K, ..., N - 1,
	dest(n) = filter(0) src(n) - \sum_{k=1}^K dest(n - k)

	Variable i = stride * n is the offset to the nth sample. */

	switch (c.K)
	{
	case 3:
		for (i = 3; i < N; i++)
			dest[i] = c.filter[0] * src[i]
			- c.filter[1] * dest[i - 1]
			- c.filter[2] * dest[i - 2]
			- c.filter[3] * dest[i - 3];
		break;
	case 4:
		for (i = 4; i < N; i++)
			dest[i] = c.filter[0] * src[i]
			- c.filter[1] * dest[i - 1]
			- c.filter[2] * dest[i - 2]
			- c.filter[3] * dest[i - 3]
			- c.filter[4] * dest[i - 4];
		break;
	case 5:
		for (i = 5; i < N; i++)
			dest[i] = c.filter[0] * src[i]
			- c.filter[1] * dest[i - 1]
			- c.filter[2] * dest[i - 2]
			- c.filter[3] * dest[i - 3]
			- c.filter[4] * dest[i - 4]
			- c.filter[5] * dest[i - 5];
		break;
	}

	/* Handle the right boundary by multiplying matrix c.M with last K
	samples dest(N - K - m), m = 0, ..., K - 1. */

	/* Copy last K samples into array q, q(m) = dest(N - K + m). */
	for (m = 0; m < c.K; ++m)
		q[m] = dest[N - (c.K - m)];

	/* Perform matrix multiplication,
	dest(N - K + m) = \sum_{n=0}^{K-1} M(m, n) q(n). */
	for (m = 0; m < c.K; ++m)
	{
		srcType accum = (srcType)0.0;

		for (n = 0; n < c.K; ++n)
			accum += c.M[m + c.K*n] * q[n];

		dest[N - (c.K - m)] = accum;
	}

	/* The following applies the anticausal filter, implementing the
	pseudocode

	For n = N - K - 1, ..., 0,
	dest(n) = filter(0) dest(n) - \sum_{k=1}^K dest(n + k)

	Variable i = stride * n is the offset to the nth sample. */

	switch (c.K)
	{
	case 3:
		for (i = N - 4; i >= 0; i--)
			dest[i] = c.filter[0] * dest[i]
			- c.filter[1] * dest[i + 1]
			- c.filter[2] * dest[i + 2]
			- c.filter[3] * dest[i + 3];
		break;
	case 4:

		for (i = N - 5; i >= 0; i--)
			dest[i] = c.filter[0] * dest[i]
			- c.filter[1] * dest[i + 1]
			- c.filter[2] * dest[i + 2]
			- c.filter[3] * dest[i + 3]
			- c.filter[4] * dest[i + 4];
		break;

	case 5:
		for (i = N - 6; i >= 0; i--)
			dest[i] = c.filter[0] * dest[i]
			- c.filter[1] * dest[i + 1]
			- c.filter[2] * dest[i + 2]
			- c.filter[3] * dest[i + 3]
			- c.filter[4] * dest[i + 4]
			- c.filter[5] * dest[i + 5];
		break;
	}
	return;
}

//#define USE_SSE 1
template <>
void vyv_gaussian_conv_w<float>(const vyv_coeffs<float> c, float *dest, const float *src, const int N)
{
	//const vyv_coeffs<T> c = c_;
	__declspec(align(16)) float q[VYV_MAX_K];
	long i;
	int m, n;

	//assert(dest && src && N > 0 && stride != 0);

	if (N <= 4)
	{   /* Special case for very short signals. */
		gaussian_short_conv(dest, src, N, 1, c.sigma);
		return;
	}

	/* Handle the left boundary. */
	init_recursive_filter_<float>(q, src, N, 1, c.filter, 0, c.filter, c.K, 1.f, c.tol, c.max_iter);

	for (m = 0; m < c.K; ++m)
		dest[m] = q[m];

	/* The following applies the causal recursive filter according to the
	filter order c.K. The loops implement the pseudocode

	For n = K, ..., N - 1,
	dest(n) = filter(0) src(n) - \sum_{k=1}^K dest(n - k)

	Variable i = stride * n is the offset to the nth sample. */

#ifdef USE_SSE
	__m128 coeff = _mm_loadu_ps(&c.filter[c.K-3]);
	coeff = _mm_shuffle_ps(coeff, coeff, _MM_SHUFFLE(0, 1, 2, 3));//reverse
#endif
	
	switch (c.K)
	{
	case 3:
		for (i = 3; i < N; i++)
		{
			dest[i] = c.filter[0] * src[i]
				- c.filter[1] * dest[i - 1]
				- c.filter[2] * dest[i - 2]
				- c.filter[3] * dest[i - 3];
		}
		break;
	case 4:

		for (i = 4; i < N; i++)
		{
#ifdef USE_SSE
			dest[i] = c.filter[0] * src[i] - _mm_cvtss_f32(_mm_dp_ps(coeff, _mm_loadu_ps((dest + i - 4)), 0xFF));
#else
			dest[i] = c.filter[0] * src[i]
				- c.filter[1] * dest[i - 1]
				- c.filter[2] * dest[i - 2]
				- c.filter[3] * dest[i - 3]
				- c.filter[4] * dest[i - 4];
#endif
		}
		break;
	case 5:
		for (i = 5; i < N; i++)
			dest[i] = c.filter[0] * src[i]
			- c.filter[1] * dest[i - 1]
			- c.filter[2] * dest[i - 2]
			- c.filter[3] * dest[i - 3]
			- c.filter[4] * dest[i - 4]
			- c.filter[5] * dest[i - 5];
		break;
	}

	/* Handle the right boundary by multiplying matrix c.M with last K
	samples dest(N - K - m), m = 0, ..., K - 1. */

	/* Copy last K samples into array q, q(m) = dest(N - K + m). */
	for (m = 0; m < c.K; ++m)
		q[m] = dest[N - (c.K - m)];

	/* Perform matrix multiplication,
	dest(N - K + m) = \sum_{n=0}^{K-1} M(m, n) q(n). */
	for (m = 0; m < c.K; ++m)
	{
		float accum = 0.f;

		for (n = 0; n < c.K; ++n)
			accum += c.M[m + c.K*n] * q[n];

		dest[N - (c.K - m)] = accum;
	}

	/* The following applies the anticausal filter, implementing the
	pseudocode

	For n = N - K - 1, ..., 0,
	dest(n) = filter(0) dest(n) - \sum_{k=1}^K dest(n + k)

	Variable i = stride * n is the offset to the nth sample. */

#ifdef USE_SSE
	coeff = _mm_shuffle_ps(coeff, coeff, _MM_SHUFFLE(0, 1, 2, 3));//reverse
#endif
	switch (c.K)
	{
	case 3:
		for (i = N - 4; i >= 0; i--)
			dest[i] = c.filter[0] * dest[i]
			- c.filter[1] * dest[i + 1]
			- c.filter[2] * dest[i + 2]
			- c.filter[3] * dest[i + 3];
		break;
	case 4:

		for (i = N - 5; i >= 0; i--)
		{
#ifdef USE_SSE
			dest[i] = c.filter[0] * dest[i] - _mm_cvtss_f32(_mm_dp_ps(coeff, _mm_loadu_ps((dest + i + 1)), 0xFF));
#else
			dest[i] = c.filter[0] * dest[i]
				- c.filter[1] * dest[i + 1]
				- c.filter[2] * dest[i + 2]
				- c.filter[3] * dest[i + 3]
				- c.filter[4] * dest[i + 4];
#endif
		}
		break;

	case 5:
		for (i = N - 6; i >= 0; i--)
			dest[i] = c.filter[0] * dest[i]
			- c.filter[1] * dest[i + 1]
			- c.filter[2] * dest[i + 2]
			- c.filter[3] * dest[i + 3]
			- c.filter[4] * dest[i + 4]
			- c.filter[5] * dest[i + 5];
		break;
	}

	return;
}

/**
 * \brief 2D Gaussian convolution Vliet-Young-Verbeek approximation
 * \param c             vyv_coeffs created by vyv_precomp()
 * \param dest          destination image data
 * \param src           source image data, modified in-place if src = dest
 * \param width         image width
 * \param height        image height
 * \param num_channels  number of image channels
 *
 * Similar to vyv_gaussian_conv(), this routine approximates 2D Gaussian
 * convolution with Vliet-Young-Verbeek recursive filtering.
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result).
 *
 * \note When the #num typedef is set to single-precision arithmetic,
 * results may be inaccurate for large values of sigma.
 */
template <typename srcType>
void vyv_gaussian_conv_image_(vyv_coeffs<srcType> c, srcType *dest, const srcType *src, const int width, const int height, const int num_channels)
{
	const int num_pixels = (width)* (height);
	int x, y, channel;

	//assert(dest && src && num_pixels > 0);

	/* Loop over the image channels. */
	for (channel = 0; channel < num_channels; ++channel)
	{
		srcType *dest_y = dest;
		const srcType *src_y = src;

		/* Filter each row of the channel. */
		{
			//CalcTime t("width");
			for (y = 0; y < height; ++y)
			{
				//vyv_gaussian_conv<T>(c, dest_y, src_y, width, 1);
				vyv_gaussian_conv_w<srcType>(c, dest_y, src_y, width);

				dest_y += width;
				src_y += width;
			}
		}

		/* Filter each column of the channel. */
		{
			//CalcTime t("height");
			for (x = 0; x < width; ++x)
				vyv_gaussian_conv<srcType>(c, dest + x, dest + x, height, width);
		}
		dest += num_pixels;
		src += num_pixels;
	}

	return;
}

void vyv_gaussian_conv_image(vyv_coeffs<float> c, float *dest, const float *src, const int width, const int height, const int num_channels)
{
	vyv_gaussian_conv_image_<float>(c, dest, src, width, height, num_channels);
}
void vyv_gaussian_conv_image(vyv_coeffs<double> c, double *dest, const double *src, const int width, const int height, const int num_channels)
{
	vyv_gaussian_conv_image_<double>(c, dest, src, width, height, num_channels);
}
