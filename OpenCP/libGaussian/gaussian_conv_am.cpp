/**
 * \file gaussian_conv_am.c
 * \brief Alvarez-Mazorra approximate Gaussian convolution
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * Copyright (c) 2011-2013, Pascal Getreuer
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

#include <assert.h>
#include <math.h>
#include "gaussian_conv.h"

/**
 * \brief Handling of the left boundary for Alvarez-Mazorra
 * \param data          signal data
 * \param N             number of elements
 * \param stride        stride between successive samples
 * \param nu            filter parameter nu
 * \param num_terms     number of terms to use to approximate infinite sum
 * \return the sum approximating the first filtered sample value
 *
 * This routine approximates the infinite sum
 * \f$ u_0 = \sum_{j=0}^\infty \nu^j \Tilde{x}_j \f$
 * by adding the first `num_terms` terms.
 */
template<typename T>
inline T am_left_boundary_(const T *data, long N, long stride, T nu, long num_terms)
{
    T h = 1, accum = data[0];
    long m;
    
    for (m = 1; m < num_terms; ++m)
    {
        h *= nu;
        accum += h * data[stride * extension(N, -m)];
    }
    
    return accum;
}

/**
 * \brief Gaussian convolution with Alvarez-Mazorra
 * \param dest              output convolved data
 * \param src               input data, modified in-place if src = dest
 * \param N                 number of elements
 * \param stride            stride between successive samples
 * \param sigma             standard deviation of the Gaussian in pixels
 * \param K                 number of passes (larger implies better accuracy)
 * \param tol               accuracy in evaluating left boundary sum
 * \param use_adjusted_q    if nonzero, use proposed regression for q
 *
 * Implements the fast approximate Gaussian convolution algorithm of Alvarez
 * and Mazorra, where the Gaussian is approximated by K passes of a first-
 * order causal filter and a first-order anticausal filter,
 * \f[ H(z) = \left(\nu/\lambda\right)^K \left( \frac{1}{1 - \nu z^{-1}}
\frac{1}{1 - \nu z} \right)^K. \f]
 * Boundaries are handled with half-sample symmetric extension, and `tol`
 * specifies the accuracy for approximating an infinite sum on the left
 * boundary.
 *
 * Gaussian convolution is approached as approximating the heat equation and
 * each timestep is performed with an efficient recursive computation.  Using
 * more steps yields a more accurate approximation of the Gaussian.
 * Reasonable values for the parameters are `K` = 4, `tol` = 1e-3.
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result).
 */
template<typename T>
void am_gaussian_conv_(T *dest, const T *src, long N, long stride,
    double sigma, int K, T tol, bool use_adjusted_q)
{
    const long stride_N = stride * N;
    double q, lambda, dnu;
    T nu, scale;
    long i, num_terms;
    int pass;
    
    assert(dest && src && N > 0 && stride != 0 && sigma > 0
        && K > 0 && tol > 0);
    
    if (use_adjusted_q)  /* Use a regression on q for improved accuracy. */
        q = sigma * (1.0 + (0.3165 * K + 0.5695)
                / ((K + 0.7818) * (K + 0.7818)));
    else                /* Use q = sigma as in the original A-M method. */
        q = sigma;
    
    /* Precompute the filter coefficient nu. */
    lambda = (q * q) / (2.0 * K);
    dnu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
    nu = (T)dnu;
    /* For handling the left boundary, determine the number of terms needed to
       approximate the sum with accuracy tol. */
    num_terms = (long)ceil(log((1.0 - dnu)*tol) / log(dnu));
    /* Precompute the constant scale factor. */
    scale = (T)(pow(dnu / lambda, K));
    
    /* Copy src to dest and multiply by the constant scale factor. */
    for (i = 0; i < stride_N; i += stride)
        dest[i] = src[i] * scale;
    
    /* Perform K passes of filtering. */
    for (pass = 0; pass < K; ++pass)
    {
        /* Initialize the recursive filter on the left boundary. */
        dest[0] = am_left_boundary_<T>(dest, N, stride, nu, num_terms);
        
        /* This loop applies the causal filter, implementing the pseudocode
           
           For n = 1, ..., N - 1
               dest(n) = dest(n) + nu dest(n - 1)
           
           Variable i = stride * n is the offset to the nth sample.  */
        for (i = stride; i < stride_N; i += stride)
            dest[i] += nu * dest[i - stride];
        
        /* Handle the right boundary. */
        i -= stride;
        dest[i] /= (1 - nu);
        
        /* Similarly, this loop applies the anticausal filter as
           
           For n = N - 1, ..., 1
               dest(n - 1) = dest(n - 1) + nu dest(n) */
        for (; i > 0; i -= stride)
            dest[i - stride] += nu * dest[i];
    }
    
    return;
}

void am_gaussian_conv(float *dest, const float *src, long N, long stride, float sigma, int K, float tol, bool use_adjusted_q)
{
	am_gaussian_conv_<float>(dest, src, N, stride, sigma, K, tol, use_adjusted_q);
}

void am_gaussian_conv(double *dest, const double *src, long N, long stride, double sigma, int K, double tol, bool use_adjusted_q)
{
	am_gaussian_conv_<double>(dest, src, N, stride, sigma, K, tol, use_adjusted_q);
}
/**
 * \brief 2D Gaussian convolution with Alvarez-Mazorra
 * \param dest                          output convolved data
 * \param src                           input, modified in-place if src = dest
 * \param width, height, num_channels   the image dimensions
 * \param sigma                         Gaussian standard deviation
 * \param K                             number of passes
 * \param tol                           accuracy for left boundary sum
 * \param use_adjusted_q                if nonzero, use proposed q
 *
 * Similar to am_gaussian_conv(), this routine approximates 2D Gaussian
 * convolution with Alvarez-Mazorra recursive filtering.
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result).
 */

template<typename T>
void am_gaussian_conv_image_(T *dest, const T *src, int width, int height, int num_channels, T sigma, int K, T tol, bool use_adjusted_q)
{
    const long num_pixels = ((long)width) * ((long)height);
    int x, y, channel;
    
	assert(dest && src && num_pixels > 0 && sigma > 0 && K > 0 && tol > 0);
    
    /* Loop over the image channels. */
    for (channel = 0; channel < num_channels; ++channel)
    {
        T *dest_y = dest;
        const T *src_y = src;
        
        /* Filter each row of the channel. */
        for (y = 0; y < height; ++y)
        {
			am_gaussian_conv_<T>(dest_y, src_y, width, 1, sigma, K, tol, use_adjusted_q);
            dest_y += width;
            src_y += width;
        }
        
        /* Filter each column of the channel. */
        for (x = 0; x < width; ++x)
            am_gaussian_conv_<T>(dest + x, dest + x, height, width,
                sigma, K, tol, use_adjusted_q);
        
        dest += num_pixels;
        src += num_pixels;
    }
    
    return;
}

void am_gaussian_conv_image(float *dest, const float *src, int width, int height, int num_channels, float sigma, int K, float tol, bool use_adjusted_q)
{
	am_gaussian_conv_image_<float>(dest, src, width, height, num_channels, sigma, K, tol, use_adjusted_q);
}

void am_gaussian_conv_image(double *dest, const double *src, int width, int height, int num_channels, double sigma, int K, double tol, bool use_adjusted_q)
{
	am_gaussian_conv_image_<double>(dest, src, width, height, num_channels, sigma, K, tol, use_adjusted_q);
}