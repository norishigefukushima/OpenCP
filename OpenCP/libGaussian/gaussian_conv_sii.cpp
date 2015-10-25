/**
 * \file gaussian_conv_sii.c
 * \brief Gaussian convolution using stacked integral images
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
#include <stdio.h>

#ifndef M_PI
/** \brief The constant pi */
#define M_PI        3.14159265358979323846264338327950288
#endif

/**
 * \brief Precompute filter coefficients for SII Gaussian convolution
 * \param c         sii_coeffs pointer to hold precomputed coefficients
 * \param sigma     Gaussian standard deviation
 * \param K         number of boxes = 3, 4, or 5
 * \return 1 on success, 0 on failure
 *
 * This routine reads Elboher and Werman's optimal SII radii and weights for
 * reference standard deviation \f$ \sigma_0 = 100/\pi \f$ from a table and
 * scales them to the specified value of sigma.
 */
template<typename T>
void sii_precomp_(sii_coeffs<T> &c, const T sigma, const int K)
{
	/* Elboher and Werman's optimal radii and weights. */
	const double sigma0 = 100.0 / M_PI;
	static const short radii0[SII_MAX_K - SII_MIN_K + 1][SII_MAX_K] =
	{ { 76, 46, 23, 0, 0 },
	{ 82, 56, 37, 19, 0 },
	{ 85, 61, 44, 30, 16 } };
	static const float weights0[SII_MAX_K - SII_MIN_K + 1][SII_MAX_K] =
	{ { 0.1618f, 0.5502f, 0.9495f, 0, 0 },
	{ 0.0976f, 0.3376f, 0.6700f, 0.9649f, 0 },
	{ 0.0739f, 0.2534f, 0.5031f, 0.7596f, 0.9738f } };

	const int i = K - SII_MIN_K;
	double sum;
	int k;

	assert(sigma > 0 && SII_VALID_K(K));
	c.K = K;

	for (k = 0, sum = 0; k < K; ++k)
	{
		c.radii[k] = (long)(radii0[i][k] * (sigma / sigma0) + 0.5);
		sum += weights0[i][k] * (2 * c.radii[k] + 1);
	}

	for (k = 0; k < K; ++k)
		c.weights[k] = (T)(weights0[i][k] / sum);

	return;
}

void sii_precomp(sii_coeffs<float> &c, const float sigma, const int K)
{
	sii_precomp_<float>(c, sigma, K);
}
void sii_precomp(sii_coeffs<double> &c, const double sigma, const int K)
{
	sii_precomp_<double>(c, sigma, K);
}
/**
 * \brief Determines the buffer size needed for SII Gaussian convolution
 * \param c     sii_coeffs created by sii_precomp()
 * \param N     number of samples
 * \return required buffer size in units of num samples
 *
 * This routine determines the minimum size of the buffer needed for use in
 * sii_gaussian_conv() or sii_gaussian_conv_image(). This size is the length
 * of the signal (or in 2D, max(width, height)) plus the twice largest box
 * radius, for padding.
 */
template<typename T>
long sii_buffer_size_(sii_coeffs<T> c, long N)
{
    long pad = c.radii[0] + 1;
    return N + 2 * pad;
}

long sii_buffer_size(sii_coeffs<float> c, long N)
{
	return sii_buffer_size_<float>(c, N);
}

long sii_buffer_size(sii_coeffs<double> c, long N)
{
	return sii_buffer_size_<double>(c, N);
}
/**
 * \brief Gaussian convolution SII approximation
 * \param c         sii_coeffs created by sii_precomp()
 * \param dest      output convolved data
 * \param buffer    array with space for sii_buffer_size() samples
 * \param src       input, modified in-place if src = dest
 * \param N         number of samples
 * \param stride    stride between successive samples
 *
 * This routine performs stacked integral images approximation of Gaussian
 * convolution with half-sample symmetric boundary handling. The buffer array
 * is used to store the cumulative sum of the input, and must have space for
 * at least sii_buffer_size(c,N) samples.
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result). However, the buffer array
 * must be distinct from `src` and `dest`.
 */
template<typename T>
void sii_gaussian_conv_(sii_coeffs<T> c, T *dest, T *buffer, const T *src, long N, long stride)
{
    T accum;
    int pad, n;
    int k;
    
	//assert(dest && buffer && src && dest != buffer && src != buffer && N > 0 && stride != 0)
    
    pad = c.radii[0] + 1;
    buffer += pad;
    
    /* Compute cumulative sum of src over n = -pad,..., N + pad - 1. */
    for (n = -pad, accum = 0; n < N + pad; ++n)
    {
        accum += src[stride * extension(N, n)];
        buffer[n] = accum;
    }
    
    /* Compute stacked box filters. */
    for (n = 0; n < N; ++n, dest += stride)
    {
        accum = c.weights[0] * (buffer[n + c.radii[0]]
            - buffer[n - c.radii[0] - 1]);
        
        for (k = 1; k < c.K; ++k)
            accum += c.weights[k] * (buffer[n + c.radii[k]]
                - buffer[n - c.radii[k] - 1]);
        
        *dest = accum;
    }
    
    return;
}

void sii_gaussian_conv(sii_coeffs<float> c, float *dest, float *buffer, const float *src, long N, long stride)
{
	sii_gaussian_conv_<float>(c, dest, buffer, src, N, stride);
}
void sii_gaussian_conv(sii_coeffs<double> c, double *dest, double *buffer, const double *src, long N, long stride)
{
	sii_gaussian_conv_<double>(c, dest, buffer, src, N, stride);
}
/**
 * \brief 2D Gaussian convolution SII approximation
 * \param c             sii_coeffs created by sii_precomp()
 * \param dest          output convolved data
 * \param buffer        array with space for sii_buffer_size() samples
 * \param src           image to be convolved, overwritten if src = dest
 * \param width         image width
 * \param height        image height
 * \param num_channels  number of image channels
 *
 * Similar to sii_gaussian_conv(), this routine approximates 2D Gaussian
 * convolution with stacked integral images. The buffer array must have space
 * for at least sii_buffer_size(c,max(width,height)) samples.
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result). However, the buffer array
 * must be distinct from `src` and `dest`.
 */
template<typename T>
void sii_gaussian_conv_image_(sii_coeffs<T> c, T *dest, T *buffer, const T *src, int width, int height, int num_channels)
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
			sii_gaussian_conv(c, dest_y, buffer, src_y, width, 1);
            dest_y += width;
            src_y += width;
        }
        
        /* Filter each column of the channel. */
		for (x = 0; x < width; ++x)
			sii_gaussian_conv(c, dest + x, buffer, dest + x, height, width);
        
        dest += num_pixels;
        src += num_pixels;
    }
    
    return;
}

void sii_gaussian_conv_image(sii_coeffs<float> c, float *dest, float *buffer, const float *src, int width, int height, int num_channels)
{
	sii_gaussian_conv_image_<float>(c, dest, buffer, src, width, height, num_channels);
}

void sii_gaussian_conv_image(sii_coeffs<double> c, double *dest, double *buffer, const double *src, int width, int height, int num_channels)
{
	sii_gaussian_conv_image_<double>(c, dest, buffer, src, width, height, num_channels);
}