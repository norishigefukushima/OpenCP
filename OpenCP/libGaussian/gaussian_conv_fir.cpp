/**
 * \file gaussian_conv_fir.c
 * \brief Gaussian convolution using FIR filters
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

/**
 * \brief Construct truncated Gaussian filter for FIR convolution
 * \param sigma     Gaussian standard deviation
 * \param r         radius of the filter
 * \return pointer to filter, or NULL on failure
 * \ingroup fir_gaussian
 *
 * This routine constructs a truncated Gaussian \f$ g^\text{trunc}\f$
 * for a specified radius \c r,
 * \f[ g^\text{trunc}_n = G_\sigma(n) / s(r), \quad
       s(r) = \sum_{n=-r}^{r} G_\sigma(n). \f]
 *
 * The filter should be released by the caller
 * with \c free().
 */
template <typename T>
static T *make_g_trunc(T sigma, long r)
{
    T *g_trunc = NULL;
    
    if ((g_trunc = (T *)malloc(sizeof(T) * (r + 1))))
    {
        T accum = g_trunc[0] = 1.0;
        long m;
        
        for (m = 1; m <= r; ++m)
        {
            T temp = m / sigma;
            g_trunc[m] = (T)exp(-0.5 * temp * temp);
            accum += (T)(2.0 * g_trunc[m]);
        }
        
        /* Normalize such that the filter has unit sum. */
        for (m = 0; m <= r; ++m)
            g_trunc[m] /= accum;
    }
    return g_trunc;
}

/**
 * \brief Convolution with a symmetric filter
 * \param dest      destination (must be distinct from src)
 * \param src       source signal
 * \param N         signal length
 * \param stride    stride between successive src and dest samples
 * \param h         symmetric filter, an array of length r + 1
 * \param r         radius of filter h
 * \ingroup fir_gaussian
 *
 * This routine computes the convolution of \c src and \c h according to
 * \f[ \mathrm{dest[stride*n]} = \sum_{|m| \le r} h_{|m|} \,
 \mathrm{src[stride*(n-m)]}, \f]
 * where \c src is extrapolated with half-sample symmetry.
 */
template <typename T>
static void conv_sym(T *dest, const T *src, long N, long stride, const T *h, long r)
{
    long n;
    
    for (n = 0; n < N; ++n)
    {
        T accum = h[0] * src[stride * n];
        long m;
        
        /* Compute \sum_m h_m ( src(n - m) + src(n + m) ). */
        for (m = 1; m <= r; ++m)
            accum += h[m] * (src[stride * extension(N, n - m)]
                + src[stride * extension(N, n + m)]);
        
        dest[stride * n] = accum;
    }
    
    return;
}

/**
 * \brief Precompute filter coefficients for FIR filtering
 * \param c         fir_coeffs pointer to hold precomputed coefficients
 * \param sigma     Gaussian standard deviation
 * \param tol       filter accuracy (smaller tol implies larger filter)
 *
 * This routine calls make_g_trunc() to construct the truncated Gaussian FIR
 * filter. The radius of the filter is determined such that
 * \f[ \lVert g - g^\text{trunc} \rVert_1 \le \mathit{tol}. \f]
 * By Young's inequality, the error between convolution using this truncated
 * FIR filter and exact Gaussian convolution is bounded,
 * \f[ \lVert g * \Tilde{f} - g^\text{trunc} * \Tilde{f} \rVert_\infty
       \le \mathit{tol} \lVert f \rVert_\infty. \f]
 */

int fir_precomp(fir_coeffs<float> *c, float sigma, float tol)
{
    assert(c && sigma > 0.0 && 0.0 < tol && tol < 1.0);
    c->radius = (long)ceil(M_SQRT2 * sigma * inverfc(0.5 * tol));
    return (c->g_trunc = make_g_trunc(sigma, c->radius)) ? 1 : 0;
}

int fir_precomp(fir_coeffs<double> *c, double sigma, double tol)
{
	assert(c && sigma > 0.0 && 0.0 < tol && tol < 1.0);
	c->radius = (long)ceil(M_SQRT2 * sigma * inverfc(0.5 * tol));
	return (c->g_trunc = make_g_trunc(sigma, c->radius)) ? 1 : 0;
}

/**
 * \brief FIR Gaussian convolution
 * \param c         fir_coeffs created by fir_precomp()
 * \param dest      output convolved data (must be distinct from src)
 * \param src       data to be convolved
 * \param N         number of samples
 * \param stride    stride between successive samples
 *
 * This routine approximates 1D Gaussian convolution with the FIR filter. The
 * computation itself is performed by conv_sym().
 *
 * \note The computation is out-of-place, `src` and `dest` must be distinct.
 */

template <typename T>
void fir_gaussian_conv_(fir_coeffs<T> c, T *dest, const T *src, long N, long stride)
{
    assert(c.g_trunc && dest && src && dest != src && N > 0 && stride != 0);
    conv_sym(dest, src, N, stride, c.g_trunc, c.radius);
    return;
}

void fir_gaussian_conv(fir_coeffs<float> c, float *dest, const float *src, long N, long stride)
{
	fir_gaussian_conv_<float>(c, dest, src, N, stride);
}

void fir_gaussian_conv(fir_coeffs<double> c, double *dest, const double *src, long N, long stride)
{
	fir_gaussian_conv_<double>(c, dest, src, N, stride);
}

/**
 * \brief FIR filtering approximation of 2D Gaussian convolution
 * \param c             fir_coeffs created by fir_precomp()
 * \param dest          destination image
 * \param buffer        array with at least width samples
 * \param src           source image, must be distinct from dest
 * \param width         image width
 * \param height        image height
 * \param num_channels  number of image channels
 *
 * Similar to fir_gaussian_conv(), this routine performs 2D FIR-based
 * Gaussian convolution.
 */
template <typename T>
void fir_gaussian_conv_image_(fir_coeffs<T> c, T *dest, T *buffer, const T *src, int width, int height, int num_channels)
{
    const long num_pixels = ((long)width) * ((long)height);
    int x, y, channel;
    
    assert(c.g_trunc && dest && buffer
        && src && dest != src && num_pixels > 0);
    
    /* Loop over the image channels. */
    for (channel = 0; channel < num_channels; ++channel)
    {
        T *dest_y = dest;
        const T *src_y = src;
        
        /* Filter each column of the channel. */
        for (x = 0; x < width; ++x)
            conv_sym(dest + x, src + x, height, width, c.g_trunc, c.radius);
        
        /* Filter each row of the channel. */
        for (y = 0; y < height; ++y)
        {
            conv_sym(buffer, dest_y, width, 1, c.g_trunc, c.radius);
            memcpy(dest_y, buffer, sizeof(T) * width);
            dest_y += width;
            src_y += width;
        }
        
        dest += num_pixels;
        src += num_pixels;
    }
    
    return;
}

void fir_gaussian_conv_image(fir_coeffs<float> c, float *dest, float *buffer, const float *src, int width, int height, int num_channels)
{
	fir_gaussian_conv_image_<float>(c, dest, buffer, src, width, height, num_channels);
}

void fir_gaussian_conv_image(fir_coeffs<double> c, double *dest, double *buffer, const double *src, int width, int height, int num_channels)
{
	fir_gaussian_conv_image_<double>(c, dest, buffer, src, width, height, num_channels);
}
/**
 * \brief Release memory associated with fir_coeffs struct
 * \param c    fir_coeffs created by fir_precomp()
 */
template <typename T>
void fir_free_(fir_coeffs<T> *c)
{
    if (c && c->g_trunc)
    {
        free(c->g_trunc);
        c->g_trunc = NULL;
    }
    
    return;
}

void fir_free(fir_coeffs<float> *c)
{
	fir_free_<float>(c);
}
void fir_free(fir_coeffs<double> *c)
{
	fir_free_<double>(c);
}
