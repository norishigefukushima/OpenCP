/**
 * \file filter_util.c
 * \brief Filtering utility functions
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
 * You should have received a copy of these licenses along with this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "gaussian_conv.h"
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/** \brief Maximum possible value of q in init_recursive_filter() */
#define MAX_Q       7

/**
 * \brief Compute taps of the impulse response of a causal recursive filter
 * \param h     destination array of size N
 * \param N     number of taps to compute (beginning from n = 0)
 * \param b     numerator coefficients
 * \param p     largest delay of the numerator
 * \param a     denominator coefficients
 * \param q     largest delay of the denominator
 *
 * Computes taps \f$ h_0, \ldots, h_{N-1} \f$ of the impulse response of the
 * recursive filter
 * \f[ H(z) = \frac{b[0] + b[1]z^{-1} + \cdots + b[p]z^{-p}}
 *            {1 + a[1]z^{-1} + \cdots + a[q]z^{-q}}, \f]
 * or equivalently
 * \f[ \begin{aligned} h[n] = & b[0]\delta_n + b[1]\delta_{n-1}+\cdots +
 * b[p]\delta_{n-p}\\ & -a[1]h[n-1]-\cdots -a[q]h[n-q], \end{aligned} \f]
 * for n = 0, ..., N-1 where \f$ \delta \f$ is the unit impulse. In the
 * denominator coefficient array, element a[0] is not used.
 */
template<typename T>
void recursive_filter_impulse(T *h, long N,  const T *b, int p, const T *a, int q)
{
    long m, n;
    
    //assert(h && N > 0 && b && p >= 0 && a && q > 0);
    
    for (n = 0; n < N; ++n)
    {
        h[n] = (n <= p) ? b[n] : 0;
        
        for (m = 1; m <= q && m <= n; ++m)
            h[n] -= a[m] * h[n - m];
    }
    
    return;
}

/**
 * \brief Initialize a causal recursive filter with boundary extension
 * \param dest      destination array with size of at least q
 * \param src       input signal of length N
 * \param N         length of src
 * \param stride    the stride between successive samples of src
 * \param b         numerator coefficients
 * \param p         largest delay of the numerator
 * \param a         denominator coefficients
 * \param q         largest delay of the denominator
 * \param sum       the L^1 norm of the impulse response
 * \param tol       accuracy tolerance
 * \param max_iter  maximum number of samples to use for approximation
 *
 * This routine initializes a recursive filter,
 * \f[  \begin{aligned} u_n &= b_0 f_n + b_1 f_{n-1} + \cdots + b_p f_{n-p}
&\quad - a_1 u_{n-1} - a_2 u_{n-2} - \cdots - a_q u_{n-q}, \end{aligned} \f]
 * with boundary extension by approximating the infinite sum
 * \f[ u_m=\sum_{n=-m}^\infty h_{n+m}\Tilde{f}_{-n} \approx \sum_{n=-m}^{k-1}
           h_{n+m} \Tilde{f}_{-n}, \quad m = 0, \ldots, q - 1. \f]
 */
template<typename T>
void init_recursive_filter_(T *dest, const T *src, long N, long stride, const T *b, int p, const T *a, int q, T sum, T tol, int max_iter)
{
	__declspec(align(16)) T h[MAX_Q + 1];
    long n;
    int m;
    
	//assert(dest && src && N > 0 && stride != 0 && b && p >= 0 && a && 0 < q && q <= MAX_Q        && tol > 0 && max_iter > 0);
    
    /* Compute the first q taps of the impulse response, h_0, ..., h_{q-1} */
    recursive_filter_impulse(h, q, b, p, a, q);
    
    /* Compute dest_m = sum_{n=1}^m h_{m-n} src_n, m = 0, ..., q-1 */
    for (m = 0; m < q; ++m)
        for (dest[m] = 0, n = 1; n <= m; ++n)
            dest[m] += h[m - n] * src[stride * extension(N, n)];

    for (n = 0; n < max_iter; ++n)
    {
        T cur = src[stride * extension(N, -n)];
        
        /* dest_m = dest_m + h_{n+m} src_{-n} */
        for (m = 0; m < q; ++m)
            dest[m] += h[m] * cur;
        
        sum -= fabs(h[0]);
        
        if (sum <= tol)
            break;
        
        /* Compute the next impulse response tap, h_{n+q} */
        h[q] = (n + q <= p) ? b[n + q] : 0;
        
        for (m = 1; m <= q; ++m)
            h[q] -= a[m] * h[q - m];
        
        /* Shift the h array for the next iteration */
        for (m = 0; m < q; ++m)
            h[m] = h[m + 1];
    }
    
    return;
}

void init_recursive_filter(float *dest, const float *src, long N, long stride, const float *b, int p, const float *a, int q, float sum, float tol, int max_iter)
{
	init_recursive_filter_<float>(dest, src, N, stride, b, p, a, q, sum, tol, max_iter);
}

void init_recursive_filter(double *dest, const double *src, long N, long stride, const double *b, int p, const double *a, int q, double sum, double tol, int max_iter)
{
	init_recursive_filter_<double>(dest, src, N, stride, b, p, a, q, sum, tol, max_iter);
}

/**
* \brief Invert matrix through QR decomposition
* \param inv_A pointer to memory for holding the result
* \param A pointer to column-major matrix data
* \param N the number of dimensions
* \return 1 on success, 0 on failure
*
* The input data is overwritten during the computation. \c inv_A
* should be allocated before calling this function with space for at least
* N^2 doubles. Matrices are represented in column-major format, meaning
*    A(i,j) = A[i + N*j], 0 <= i, j < N.
*/
int invert_matrix(double *inv_A, double *A, int N)
{
	double *c = NULL, *d = NULL, *col_j, *col_k, *inv_col_k;
	double temp, scale, sum;
	int i, j, k, success = 0;

	//assert(inv_A && A && N > 0);

	if (!(c = (double *)malloc(sizeof(double) * N))
		|| !(d = (double *)malloc(sizeof(double) * N)))
		goto fail;

	for (k = 0, col_k = A; k < N - 1; ++k, col_k += N)
	{
		scale = 0.0;

		for (i = k; i < N; ++i)
			if ((temp = fabs(col_k[i])) > scale)
				scale = temp;

		if (scale == 0.0)
			goto fail; /* Singular matrix */

		for (i = k; i < N; ++i)
			col_k[i] /= scale;

		for (sum = 0.0, i = k; i < N; ++i)
			sum += col_k[i] * col_k[i];

		temp = (col_k[k] >= 0.0) ? sqrt(sum) : -sqrt(sum);
		col_k[k] += temp;
		c[k] = temp * col_k[k];
		d[k] = -scale * temp;

		for (j = k + 1, col_j = col_k + N; j < N; ++j, col_j += N)
		{
			for (scale = 0.0, i = k; i < N; ++i)
				scale += col_k[i] * col_j[i];

			scale /= c[k];

			for (i = k; i < N; ++i)
				col_j[i] -= scale * col_k[i];
		}
	}

	d[N - 1] = col_k[k];

	if (d[N - 1] == 0.0)
		goto fail; /* Singular matrix */

	for (k = 0, inv_col_k = inv_A; k < N; ++k, inv_col_k += N)
	{
		for (i = 0; i < N; ++i)
			inv_col_k[i] = -A[k] * A[i] / c[0];

		inv_col_k[k] += 1.0;

		for (j = 1, col_j = A + N; j < N - 1; ++j, col_j += N)
		{
			for (scale = 0.0, i = j; i < N; ++i)
				scale += col_j[i] * inv_col_k[i];

			scale /= c[j];

			for (i = j; i < N; ++i)
				inv_col_k[i] -= scale * col_j[i];
		}

		inv_col_k[j] /= d[N - 1];

		for (i = N - 2; i >= 0; --i)
		{
			for (sum = 0.0, j = i + 1, col_j = A + N*(i + 1);
				j < N; ++j, col_j += N)
				sum += col_j[i] * inv_col_k[j];

			inv_col_k[i] = (inv_col_k[i] - sum) / d[i];
		}
	}

	success = 1; /* Finished successfully */
fail: /* Clean up */
	free(d);
	free(c);
	return success;
}

/**
* \brief Acklam's algorithm for the inverse complementary error function
* \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
* \brief Inverse complementary error function \f$\mathrm{erfc}^{-1}(x)\f$
* \param x     input argument
*
* Reference: P.J. Acklam, "An algorithm for computing the inverse normal
* cumulative distribution function," 2010, online at
* http://home.online.no/~pjacklam/notes/invnorm/
*/
double inverfc_acklam(double x)
{
	static const double a[] = { -3.969683028665376e1, 2.209460984245205e2,
		-2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1,
		2.506628277459239 };
	static const double b[] = { -5.447609879822406e1, 1.615858368580409e2,
		-1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1 };
	static const double c[] = { -7.784894002430293e-3, -3.223964580411365e-1,
		-2.400758277161838, -2.549732539343734, 4.374664141464968,
		2.938163982698783 };
	static const double d[] = { 7.784695709041462e-3, 3.224671290700398e-1,
		2.445134137142996, 3.754408661907416 };
	double y, e, u;

	x /= 2.0;

	if (0.02425 <= x && x <= 0.97575)
	{
		double q = x - 0.5;
		double r = q * q;
		y = (((((a[0] * r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q
			/ (((((b[0] * r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1);
	}
	else
	{
		double q = sqrt(-2.0 * log((x > 0.97575) ? (1.0 - x) : x));
		y = (((((c[0] * q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
			/ ((((d[0] * q + d[1])*q + d[2])*q + d[3])*q + 1);

		if (x > 0.97575)
			y = -y;
	}

	e = 0.5 * erfc_cody(-y / M_SQRT2) - x;
	u = e * M_SQRT2PI * exp(0.5 * y * y);
	y -= u / (1.0 + 0.5 * y * u);
	return -y / M_SQRT2;
}


/**
* \file erfc_cody.c
* \brief W.J. Cody's approximation of the complementary error function (erfc)
* \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
*
* Copyright (c) 2012, Pascal Getreuer
* All rights reserved.
*
* This program is free software: you can redistribute it and/or modify it
* under, at your option, the terms of the GNU General Public License as
* published by the Free Software Foundation, either version 3 of the
* License, or (at your option) any later version, or the terms of the
* simplified BSD license.
*
* You should have received a copy of these licenses along with this program.
* If not, see <http://www.gnu.org/licenses/> and
* <http://www.opensource.org/licenses/bsd-license.html>.
*/

/**
* \brief Evaluate rational polynomial for erfc approximation
* \param P     numerator coefficients
* \param Q     denominator coefficients
* \param N     order
* \param x     x-value at which to evaluate the rational polynomial
*
* Evaluates rational polynomial
* \f[ \frac{P[N+1] x^{N+1} + P[0] x^N + \cdots + P[N-1] x + P[N]}
{x^{N+1} + Q[0] x^N + \cdots + Q[N-1] x + Q[N]}. \f]
*/
static double erfc_cody_rpoly(const double *P,
	const double *Q, int N, double x)
{
	double xnum = P[N + 1] * x, xden = x;
	int n;

	for (n = 0; n < N; ++n)
	{
		xnum = (xnum + P[n]) * x;
		xden = (xden + Q[n]) * x;
	}

	return (xnum + P[N]) / (xden + Q[N]);
}

/**
* \brief Complementary error function
*
* Based on the public domain NETLIB (Fortran) code by W. J. Cody
* Applied Mathematics Division
* Argonne National Laboratory
* Argonne, IL 60439
*
* From the original documentation:
* The main computation evaluates near-minimax approximations from "Rational
* Chebyshev approximations for the error function" by W. J. Cody, Math.
* Comp., 1969, PP. 631-638. This transportable program uses rational
* functions that theoretically approximate erf(x) and erfc(x) to at least 18
* significant decimal digits. The accuracy achieved depends on the
* arithmetic system, the compiler, the intrinsic functions, and proper
* selection of the machine-dependent constants.
*/
double erfc_cody(double x)
{
	static const double P1[5] = { 3.16112374387056560e0,
		1.13864154151050156e2, 3.77485237685302021e2,
		3.20937758913846947e3, 1.85777706184603153e-1 };
	static const double Q1[4] = { 2.36012909523441209e1,
		2.44024637934444173e2, 1.28261652607737228e3,
		2.84423683343917062e3 };
	static const double P2[9] = { 5.64188496988670089e-1,
		8.88314979438837594e0, 6.61191906371416295e1,
		2.98635138197400131e2, 8.81952221241769090e2,
		1.71204761263407058e3, 2.05107837782607147e3,
		1.23033935479799725e3, 2.15311535474403846e-8 };
	static const double Q2[8] = { 1.57449261107098347e1,
		1.17693950891312499e2, 5.37181101862009858e2,
		1.62138957456669019e3, 3.29079923573345963e3,
		4.36261909014324716e3, 3.43936767414372164e3,
		1.23033935480374942e3 };
	static const double P3[6] = { 3.05326634961232344e-1,
		3.60344899949804439e-1, 1.25781726111229246e-1,
		1.60837851487422766e-2, 6.58749161529837803e-4,
		1.63153871373020978e-2 };
	static const double Q3[5] = { 2.56852019228982242e0,
		1.87295284992346047e0, 5.27905102951428412e-1,
		6.05183413124413191e-2, 2.33520497626869185e-3 };
	double y, result;

	y = fabs(x);

	if (y <= 0.46875)
		return 1 - x * erfc_cody_rpoly(P1, Q1, 3, (y > 1.11e-16) ? y*y : 0);
	else if (y <= 4)
		result = exp(-y*y) * erfc_cody_rpoly(P2, Q2, 7, y);
	else if (y >= 26.543)
		result = 0;
	else
		result = exp(-y*y) * ((M_1_SQRTPI
		- y*y * erfc_cody_rpoly(P3, Q3, 4, 1.0 / (y*y))) / y);

	return (x < 0) ? (2 - result) : result;
}
