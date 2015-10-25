/**
 * \file gaussian_conv_box.c
 * \brief Box filtering approximation of Gaussian convolution
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
 * \brief Perform one pass of box filtering
 * \param dest          destination array
 * \param dest_stride   stride between successive samples of dest
 * \param src_stride    src array (must be distinct from dest)
 * \param src_stride    stride between successive samples of src
 * \param N             number of samples
 * \param r             radius of the box filter
 * \ingroup box_gaussian
 *
 * Performs one pass of box filtering with radius r (diameter 2r+1) according
 * to the recursive filter
 * \f[ H(z) = \frac{1}{2r+1} \frac{z^r - z^{-r-1}}{1 - z^{-1}}. \f]
 *
 * \note The computation is out-of-place, src and dest must be distinct.
 */
template <typename T>
static void box_filter(T *dest, long dest_stride, const T *src, long src_stride, long N, long r)
{
	long n;
	T accum = 0;

	assert(dest && src && dest != src && N > 0 && r >= 0);

	/* Initialize the filter on the left boundary by directly computing
	   dest(0) = accum = \sum_{n=-r}^r src(n). */
	for (n = -r; n <= r; ++n)
		accum += src[src_stride * extension(N, n)];

	dest[0] = accum;

	/* Filter the interior samples. */
	for (n = 1; n < N; ++n)
	{
		/* Update accum: add sample src(n + r) and remove src(n - r - 1). */
		accum += src[src_stride * extension(N, n + r)]
			- src[src_stride * extension(N, n - r - 1)];
		dest[dest_stride * n] = accum;
	}

	return;
}

/**
 * \brief Box filtering approximation of Gaussian convolution
 * \param dest_data     destination array
 * \param buffer_data   array with space for at least N samples
 * \param src           input, overwritten if src = dest_data
 * \param N             number of samples
 * \param stride        stride between successive samples
 * \param sigma         Gaussian standard deviation in pixels
 * \param K             number of box filter passes
 *
 * \note This routine can only approximate Gaussian convolution for a
 * quantized set of \f$ \sigma \f$ values. The input argument `sigma` is
 * rounded to the closest supported value.
 *
 * This routine performs iterated box filtering approximation of Gaussian
 * convolution.  Well's approximation
 * \f$ \sigma^2 = \tfrac{1}{12} K \bigl((2r+1)^2 - 1\bigr) \f$
 * is used to select the box filter radius.
 *
 * The filtering itself is performed by calling the box_filter() function
 * `K` times. Since box_filter() requires that the source and destination
 * arrays are distinct, the iteration alternates the roles of the dest and
 * buffer arrays. For example the iteration pseudocode for `K` = 4 is
 \verbatim
 buffer <- box_filter(src)
 dest   <- box_filter(buffer)
 buffer <- box_filter(dest)
 dest   <- box_filter(buffer)
 dest   <- dest * scale
 \endverbatim
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result). However, `buffer_data` must
 * be distinct from `dest_data`.
 */

template <typename T>
void box_gaussian_conv_(T *dest_data, T *buffer_data, const T *src, long N, long stride, T sigma, int K)
{
	struct
	{
		T *data;
		long stride;
	} dest, buffer, cur, next;
	T scale;
	long r;
	int step;

	assert(dest_data && buffer_data && src && dest_data != buffer_data && N > 0 && sigma > 0 && K > 0);

	/* Compute the box radius according to Wells' formula. */
	r = (long)(0.5 * sqrt((12.0 * sigma * sigma) / K + 1.0));
	scale = (T)(1.0 / (double)pow(2.0*r + 1.0, K));

	dest.data = dest_data;
	dest.stride = stride;
	buffer.data = buffer_data;
	buffer.stride = (buffer_data == src) ? stride : 1;

	/* Here we decide whether dest or buffer should be the first output array.
	   If K is even, then buffer is the better choice so that the result is in
	   dest after K iterations, e.g. for K = 4 (as in the function comment),

	   src -> buffer -> dest -> buffer -> dest.

	   If K is odd, we would like to choose dest, e.g. for K = 3,

	   src -> dest -> buffer -> dest.

	   However, if src and dest point to the same memory (i.e., in-place
	   computation), then we must select buffer as the first output array. */
	if (buffer_data == src || (dest_data != src && K % 2 == 1))
		next = dest;
	else
		next = buffer;

	/* Perform the first step of box filtering. */
	box_filter(next.data, next.stride, src, stride, N, r);

	/* Perform another (K - 1) steps of box filtering, alternating the roles
	   of the dest and buffer arrays. */
	for (step = 1; step < K; ++step)
	{
		cur = next;
		next = (cur.data == buffer_data) ? dest : buffer;
		box_filter(next.data, next.stride, cur.data, cur.stride, N, r);
	}

	/* Multiply by the constant scale factor. */
	if (next.data != dest_data)
	{
		long n, i;

		for (n = i = 0; n < N; ++n, i += stride)
			dest_data[i] = buffer_data[n] * scale;
	}
	else
	{
		long i, i_end = stride * N;

		for (i = 0; i < i_end; i += stride)
			dest_data[i] *= scale;
	}

	return;
}

void box_gaussian_conv(float *dest_data, float *buffer_data, const float *src, long N, long stride, float sigma, int K)
{
	box_gaussian_conv_<float>(dest_data, buffer_data, src, N, stride, sigma, K);
}
void box_gaussian_conv(double *dest_data, double *buffer_data, const double *src, long N, long stride, double sigma, int K)
{
	box_gaussian_conv_<double>(dest_data, buffer_data, src, N, stride, sigma, K);
}
/**
 * \brief Box filtering approximation of 2D Gaussian convolution
 * \param dest          destination image
 * \param buffer        array with at least max(width,height) samples
 * \param src           input, overwritten if src = dest
 * \param width         image width
 * \param height        image height
 * \param num_channels  number of image channels
 * \param sigma         Gaussian standard deviation in pixels
 * \param K             number of box filter passes
 *
 * Similar to box_gaussian_conv(), this routine approximates 2D Gaussian
 * convolution with box filtering.
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result). However, `buffer` must be
 * distinct from `dest`.
 */
template <typename T>
void box_gaussian_conv_image_(T *dest, T *buffer, const T *src, int width, int height, int num_channels, T sigma, int K)
{
	const long num_pixels = ((long)width) * ((long)height);
	int x, y, channel;

	assert(dest && buffer && src && dest != buffer && num_pixels > 0 && sigma > 0 && K > 0);

	/* Loop over the image channels. */
	for (channel = 0; channel < num_channels; ++channel)
	{
		T *dest_y = dest;
		const T *src_y = src;

		/* Filter each column of the channel. */
		for (y = 0; y < height; ++y)
		{
			box_gaussian_conv(dest_y, buffer, src_y,
				width, 1, sigma, K);
			dest_y += width;
			src_y += width;
		}

		/* Filter each row of the channel. */
		for (x = 0; x < width; ++x)
			box_gaussian_conv(dest + x, buffer, dest + x,
			height, width, sigma, K);

		dest += num_pixels;
		src += num_pixels;
	}
	return;
}

void box_gaussian_conv_image(float *dest, float *buffer, const float *src, int width, int height, int num_channels, float sigma, int K)
{
	box_gaussian_conv_image_<float>(dest, buffer, src, width, height, num_channels, sigma, K);
}

void box_gaussian_conv_image(double *dest, double *buffer, const double *src, int width, int height, int num_channels, double sigma, int K)
{
	box_gaussian_conv_image_<double>(dest, buffer, src, width, height, num_channels, sigma, K);
}