/**
 * \file gaussian_conv_deriche.c
 * \brief Deriche's approximation of Gaussian convolution
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
#include <stdlib.h>
#include <string.h>
#include "complex_arith.h"

#ifndef M_SQRT2PI
/** \brief The constant sqrt(2 pi) */
#define M_SQRT2PI   2.50662827463100050241576528481104525
#endif

template <typename T>
static void make_filter(T *result_b, T *result_a, const complex4c *alpha, const complex4c *beta, int K, T sigma);

/**
 * \brief Precompute coefficients for Deriche's Gaussian approximation
 * \param c         deriche_coeffs pointer to hold precomputed coefficients
 * \param sigma     Gaussian standard deviation
 * \param K         filter order = 2, 3, or 4
 * \param tol       accuracy for filter initialization at the boundaries
 *
 * This routine precomputes the recursive filter coefficients for Deriche's
 * Gaussian convolution approximation. The recursive filters are created
 * through the following steps:
 *
 * 1. First, the causal filter is represented in the form
 * \f[ H^{(K)}(z) = \frac{1}{\sqrt{2\pi\sigma^2}}\sum_{k=1}^K \frac{\alpha_k}
                    {1 - \mathrm{e}^{-\lambda_k/\sigma} z^{-1}}, \f]
 *    where the \f$ \alpha_k \f$ and \f$ \lambda_k \f$ are Deriche's optimized
 *    parameters and \f$ \sigma \f$ is the specified sigma value.
 * 2. Setting \f$ \beta_k = -\mathrm{e}^{-\lambda_k/\sigma} \f$, the filter is
      algebraically rearranged by make_filter() to the form
 * \f[ \frac{1}{\sqrt{2\pi\sigma^2}}\sum_{k=0}^{K-1}\frac{\alpha_k}{1+\beta_k
 z^{-1}}=\frac{\sum_{k=0}^{K-1}b_k z^{-k}}{1+\sum_{k=1}^{K}a_k z^{-k}} \f]
 *    to obtain the numerator and denominator coefficients for the causal filter.
 * 3. The anticausal filter is determined according to
 * \f[ \frac{\sum_{k=1}^K b^-_k z^k}{1+\sum_{k=1}^K a_k z^k}= H^{(K)}(z^{-1})-
h_0^{(K)}=\frac{\sum_{k=1}^K(b^+_k-a_k b^+_0)z^k}{1+\sum_{k=1}^K a_k z^k}. \f]
 */
template<typename T>
void deriche_precomp_(deriche_coeffs<T> *c, T sigma, int K, T tol)
{
    /* Deriche's optimized filter parameters. */
    static const complex4c alpha[DERICHE_MAX_K - DERICHE_MIN_K + 1][4] = {
        {{0.48145, 0.971}, {0.48145, -0.971}},
        {{-0.44645, 0.5105}, {-0.44645, -0.5105}, {1.898, 0}},
        {{0.84, 1.8675}, {0.84, -1.8675},
            {-0.34015, -0.1299}, {-0.34015, 0.1299}}
        };
    static const complex4c lambda[DERICHE_MAX_K - DERICHE_MIN_K + 1][4] = {
        {{1.26, 0.8448}, {1.26, -0.8448}},
        {{1.512, 1.475}, {1.512, -1.475}, {1.556, 0}},
        {{1.783, 0.6318}, {1.783, -0.6318},
            {1.723, 1.997}, {1.723, -1.997}}
        };
    complex4c beta[DERICHE_MAX_K];
    
    int k;
    double accum, accum_denom = 1.0;
    
    assert(c && sigma > 0 && DERICHE_VALID_K(K) && 0 < tol && tol < 1);
    
    for (k = 0; k < K; ++k)
    {
        double temp = exp(-lambda[K - DERICHE_MIN_K][k].real / sigma);
        beta[k] = make_complex(
            -temp * cos(lambda[K - DERICHE_MIN_K][k].imag / sigma),
            temp * sin(lambda[K - DERICHE_MIN_K][k].imag / sigma));
    }
    
    /* Compute the causal filter coefficients */
    make_filter<T>(c->b_causal, c->a, alpha[K - DERICHE_MIN_K], beta, K, sigma);
    
    /* Numerator coefficients of the anticausal filter */
    c->b_anticausal[0] = (T)(0.0);
    
    for (k = 1; k < K; ++k)
        c->b_anticausal[k] = c->b_causal[k] - c->a[k] * c->b_causal[0];
    
    c->b_anticausal[K] = -c->a[K] * c->b_causal[0];
    
    /* Impulse response sums */
    for (k = 1; k <= K; ++k)
        accum_denom += c->a[k];
    
    for (k = 0, accum = 0.0; k < K; ++k)
        accum += c->b_causal[k];
    
    c->sum_causal = (T)(accum / accum_denom);
    
    for (k = 1, accum = 0.0; k <= K; ++k)
        accum += c->b_anticausal[k];
    
    c->sum_anticausal = (T)(accum / accum_denom);
    
    c->sigma = (T)sigma;
    c->K = K;
    c->tol = tol;
    c->max_iter = (int)ceil(10.0 * sigma);
    return;
}

void deriche_precomp(deriche_coeffs<float> *c, float sigma, int K, float tol)
{
	deriche_precomp_<float>(c, sigma, K, tol);
}

void deriche_precomp(deriche_coeffs<double> *c, double sigma, int K, double tol)
{
	deriche_precomp_<double>(c, sigma, K, tol);
}
/**
 * \brief Make Deriche filter from alpha and beta coefficients
 * \param result_b      resulting numerator filter coefficients
 * \param result_a      resulting denominator filter coefficients
 * \param alpha, beta   input coefficients
 * \param K             number of terms
 * \param sigma         Gaussian sigma parameter
 * \ingroup deriche_gaussian
 *
 * This routine performs the algebraic rearrangement
 * \f[ \frac{1}{\sqrt{2\pi\sigma^2}}\sum_{k=0}^{K-1}\frac{\alpha_k}{1+\beta_k
z^{-1}}=\frac{\sum_{k=0}^{K-1}b_k z^{-k}}{1+\sum_{k=1}^{K}a_k z^{-k}} \f]
 * to obtain the numerator and denominator coefficients for the causal filter
 * in Deriche's Gaussian approximation.
 *
 * The routine initializes b/a as the 0th term,
 * \f[ \frac{b(z)}{a(z)} = \frac{\alpha_0}{1 + \beta_0 z^{-1}}, \f]
 * then the kth term is added according to
 * \f[ \frac{b(z)}{a(z)}\leftarrow\frac{b(z)}{a(z)}+\frac{\alpha_k}{1+\beta_k
z^{-1}}=\frac{b(z)(1+\beta_kz^{-1})+a(z)\alpha_k}{a(z)(1+\beta_kz^{-1})}. \f]
 */
template<typename T>
static void make_filter(T *result_b, T *result_a, const complex4c *alpha, const complex4c *beta, int K, T sigma)
{
    const double denom = sigma * M_SQRT2PI;
    complex4c b[DERICHE_MAX_K], a[DERICHE_MAX_K + 1];
    int k, j;
        
    b[0] = alpha[0];    /* Initialize b/a = alpha[0] / (1 + beta[0] z^-1) */
    a[0] = make_complex(1, 0);
    a[1] = beta[0];
    
    for (k = 1; k < K; ++k)
    {   /* Add kth term, b/a += alpha[k] / (1 + beta[k] z^-1) */
        b[k] = c_mul(beta[k], b[k-1]);
        
        for (j = k - 1; j > 0; --j)
            b[j] = c_add(b[j], c_mul(beta[k], b[j - 1]));
        
        for (j = 0; j <= k; ++j)
            b[j] = c_add(b[j], c_mul(alpha[k], a[j]));
        
        a[k + 1] = c_mul(beta[k], a[k]);
        
        for (j = k; j > 0; --j)
            a[j] = c_add(a[j], c_mul(beta[k], a[j - 1]));
    }
    
    for (k = 0; k < K; ++k)
    {
        result_b[k] = (T)(b[k].real / denom);
        result_a[k + 1] = (T)a[k + 1].real;
    }
    
    return;
}

/**
 * \brief Deriche Gaussian convolution
 * \param c         coefficients precomputed by deriche_precomp()
 * \param dest      output convolved data
 * \param buffer    workspace array with space for at least 2 * N elements
 * \param src       input, overwritten if src = dest
 * \param N         number of samples
 * \param stride    stride between successive samples
 *
 * This routine performs Deriche's recursive filtering approximation of
 * Gaussian convolution. The Gaussian is approximated by summing the
 * responses of a causal filter and an anticausal filter. The causal
 * filter has the form
 * \f[ H^{(K)}(z) = \frac{\sum_{k=0}^{K-1} b^+_k z^{-k}}
                    {1 + \sum_{k=1}^K a_k z^{-k}}, \f]
 * where K is the filter order (2, 3, or 4). The anticausal form is
 * the spatial reversal of the causal filter minus the sample at n = 0,
 * \f$ H^{(K)}(z^{-1}) - h_0^{(K)}. \f$
 *
 * The filter coefficients \f$ a_k, b^+_k, b^-_k \f$ correspond to the
 * variables `c.a`, `c.b_causal`, and `c.b_anticausal`, which are precomputed
 * by the routine deriche_precomp().
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result). However, the `buffer` array
 * must be distinct from `src`.
 *
 * \note When the #num typedef is set to single-precision arithmetic,
 * results may be inaccurate for large values of sigma.
 */

template<typename T>
void deriche_gaussian_conv_(deriche_coeffs<T> c, T *dest, T *buffer, const T *src, long N, long stride)
{
    const long stride_2 = stride * 2;
    const long stride_3 = stride * 3;
    const long stride_4 = stride * 4;
    const long stride_N = stride * N;
    T *y_causal, *y_anticausal;
    long i, n;
    
    assert(dest && buffer && src && buffer != src && N > 0 && stride != 0);
    
    if (N <= 4)
    {   /* Special case for very short signals. */
        gaussian_short_conv(dest, src, N, stride, c.sigma);
        return;
    }
    
    /* Divide buffer into two buffers each of length N. */
    y_causal = buffer;
    y_anticausal = buffer + N;
    
    /* Initialize the causal filter on the left boundary. */
    init_recursive_filter<T>(y_causal, src, N, stride,
        c.b_causal, c.K - 1, c.a, c.K, c.sum_causal, c.tol, c.max_iter);
    
    /* The following filters the interior samples according to the filter
       order c.K. The loops below implement the pseudocode
       
       For n = K, ..., N - 1,
           y^+(n) = \sum_{k=0}^{K-1} b^+_k src(n - k)
                    - \sum_{k=1}^K a_k y^+(n - k)
       
       Variable i tracks the offset to the nth sample of src, it is
       updated together with n such that i = stride * n. */
    switch (c.K)
    {
    case 2:
        for (n = 2, i = stride_2; n < N; ++n, i += stride)
            y_causal[n] = c.b_causal[0] * src[i]
                + c.b_causal[1] * src[i - stride]
                - c.a[1] * y_causal[n - 1]
                - c.a[2] * y_causal[n - 2];
        break;
    case 3:
        for (n = 3, i = stride_3; n < N; ++n, i += stride)
            y_causal[n] = c.b_causal[0] * src[i]
                + c.b_causal[1] * src[i - stride]
                + c.b_causal[2] * src[i - stride_2]
                - c.a[1] * y_causal[n - 1]
                - c.a[2] * y_causal[n - 2]
                - c.a[3] * y_causal[n - 3];
        break;
    case 4:
        for (n = 4, i = stride_4; n < N; ++n, i += stride)
            y_causal[n] = c.b_causal[0] * src[i]
                + c.b_causal[1] * src[i - stride]
                + c.b_causal[2] * src[i - stride_2]
                + c.b_causal[3] * src[i - stride_3]
                - c.a[1] * y_causal[n - 1]
                - c.a[2] * y_causal[n - 2]
                - c.a[3] * y_causal[n - 3]
                - c.a[4] * y_causal[n - 4];
        break;
    }
    
    /* Initialize the anticausal filter on the right boundary. */
    init_recursive_filter(y_anticausal, src + stride_N - stride, N, -stride,
        c.b_anticausal, c.K, c.a, c.K, c.sum_anticausal, c.tol, c.max_iter);
    
    /* Similar to the causal filter code above, the following implements
       the pseudocode
       
       For n = K, ..., N - 1,
           y^-(n) = \sum_{k=1}^K b^-_k src(N - n - 1 - k)
                    - \sum_{k=1}^K a_k y^-(n - k)
     
       Variable i is updated such that i = stride * (N - n - 1). */
    switch (c.K)
    {
    case 2:
        for (n = 2, i = stride_N - stride_3; n < N; ++n, i -= stride)
            y_anticausal[n] = c.b_anticausal[1] * src[i + stride]
                + c.b_anticausal[2] * src[i + stride_2]
                - c.a[1] * y_anticausal[n - 1]
                - c.a[2] * y_anticausal[n - 2];
        break;
    case 3:
        for (n = 3, i = stride_N - stride_4; n < N; ++n, i -= stride)
            y_anticausal[n] = c.b_anticausal[1] * src[i + stride]
                + c.b_anticausal[2] * src[i + stride_2]
                + c.b_anticausal[3] * src[i + stride_3]
                - c.a[1] * y_anticausal[n - 1]
                - c.a[2] * y_anticausal[n - 2]
                - c.a[3] * y_anticausal[n - 3];
        break;
    case 4:
        for (n = 4, i = stride_N - stride * 5; n < N; ++n, i -= stride)
            y_anticausal[n] = c.b_anticausal[1] * src[i + stride]
                + c.b_anticausal[2] * src[i + stride_2]
                + c.b_anticausal[3] * src[i + stride_3]
                + c.b_anticausal[4] * src[i + stride_4]
                - c.a[1] * y_anticausal[n - 1]
                - c.a[2] * y_anticausal[n - 2]
                - c.a[3] * y_anticausal[n - 3]
                - c.a[4] * y_anticausal[n - 4];
        break;
    }
    
    /* Sum the causal and anticausal responses to obtain the final result. */
    for (n = 0, i = 0; n < N; ++n, i += stride)
        dest[i] = y_causal[n] + y_anticausal[N - n - 1];
    
    return;
}

void deriche_gaussian_conv(deriche_coeffs<float> c, float *dest, float *buffer, const float *src, long N, long stride)
{
	deriche_gaussian_conv_<float>(c, dest, buffer, src, N, stride);
}

void deriche_gaussian_conv(deriche_coeffs<double> c, double *dest, double *buffer, const double *src, long N, long stride)
{
	deriche_gaussian_conv_<double>(c, dest, buffer, src, N, stride);
}

/**
 * \brief Deriche Gaussian 2D convolution
 * \param c             coefficients precomputed by deriche_precomp()
 * \param dest          output convolved data
 * \param buffer        array with at least 2*max(width,height) elements
 * \param src           input image, overwritten if src = dest
 * \param width         image width
 * \param height        image height
 * \param num_channels  number of image channels
 *
 * Similar to deriche_gaussian_conv(), this routine approximates 2D Gaussian
 * convolution with Deriche recursive filtering.
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result). However, the buffer array
 * must be distinct from `src`.
 *
 * \note When the #num typedef is set to single-precision arithmetic,
 * results may be inaccurate for large values of sigma.
 */
template<typename T>
void deriche_gaussian_conv_image_(deriche_coeffs<T> c, T *dest, T *buffer, const T *src, int width, int height, int num_channels)
{
    long num_pixels = ((long)width) * ((long)height);
    int x, y, channel;
    
    assert(dest && buffer && src && num_pixels > 0);
    
    /* Loop over the image channels. */
    for (channel = 0; channel < num_channels; ++channel)
    {
        T *dest_y = dest;
        const T *src_y = src;
        
        /* Filter each row of the channel. */
        for (y = 0; y < height; ++y)
        {
            deriche_gaussian_conv(c,
                dest_y, buffer, src_y, width, 1);
            dest_y += width;
            src_y += width;
        }
        
        /* Filter each column of the channel. */
        for (x = 0; x < width; ++x)
            deriche_gaussian_conv(c,
                dest + x, buffer, dest + x, height, width);
        
        dest += num_pixels;
        src += num_pixels;
    }
    
    return;
}

void deriche_gaussian_conv_image(deriche_coeffs<float> c, float *dest, float *buffer, const float *src, int width, int height, int num_channels)
{
	deriche_gaussian_conv_image_<float>(c, dest, buffer, src, width, height, num_channels);
}
void deriche_gaussian_conv_image(deriche_coeffs<double> c, double *dest, double *buffer, const double *src, int width, int height, int num_channels)
{
	deriche_gaussian_conv_image_<double>(c, dest, buffer, src, width, height, num_channels);
}