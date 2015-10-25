/**
 * \file gaussian_conv_ebox.c
 * \brief Gaussian convolution with extended box filters
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

/**
 * \brief Precompute coefficients for extended box filtering
 * \param c         ebox_coeffs pointer to hold precomputed coefficients
 * \param sigma     Gaussian standard deviation
 * \param K         number of filtering passes
 *
 * This routine precomputes the coefficients for extended box filtering
 * Gaussian convolution approximation.
 */
template <typename T>
void ebox_precomp_(ebox_coeffs<T> *c, T sigma, const int K)
{
    double alpha;
    
    assert(c && sigma > 0 && K > 0);
    
    /* Set the parameters according to Gwosdek et al. */
    c->r = (long)(0.5 * sqrt((12.0 * sigma * sigma) / K + 1.0) - 0.5);
    alpha = (2 * c->r + 1) * (c->r * (c->r + 1) - 3.0 * sigma * sigma / K)
        / (6.0 * (sigma * sigma / K - (c->r + 1) * (c->r + 1)));
    c->c_1 = (T)(alpha / (2.0 * (alpha + c->r) + 1));
    c->c_2 = (T)((1.0 - alpha) / (2.0 * (alpha + c->r) + 1));
    c->K = K;
    return;
}

void ebox_precomp(ebox_coeffs<float> *c, float sigma, const int K)
{
	ebox_precomp_<float>(c, sigma, K);
}
void ebox_precomp(ebox_coeffs<double> *c, double sigma, const int K)
{
	ebox_precomp_<double>(c, sigma, K);
}
/**
 * \brief Perform one pass of extended box filtering
 * \param dest          destination array
 * \param dest_stride   stride between successive samples of dest
 * \param src_stride    src array (must be distinct from dest)
 * \param src_stride    stride between successive samples of src
 * \param N             number of samples
 * \param r             radius of the inner box
 * \param c_1           weight of the outer box
 * \param c_2           weight of the inner box
 * \ingroup ebox_gaussian
 *
 * This routine performs one pass of extended box filtering with inner box
 * radius `r`,
 * \f[ u_n = u_{n-1}+c_1(f_{n+r+1}-f_{n-r-2})+c_2(f_{n+r}-f_{n-r-1}). \f]
 *
 * \note The computation is out-of-place, `src` and `dest` must be distinct.
 */
template <typename T>
static void ebox_filter(T *dest, long dest_stride, const T *src, long src_stride, long N, long r, T c_1, T c_2)
{
    long n;
    T accum = 0;
    
    assert(dest && src && dest != src && N > 0 && r >= 0);
    
    for (n = -r; n <= r; ++n)
        accum += src[src_stride * extension(N, n)];
    
    dest[0] = accum = c_1 * (src[src_stride * extension(N, r + 1)]
        + src[src_stride * extension(N, -r - 1)])
        + (c_1 + c_2) * accum;
    
    for (n = 1; n < N; ++n)
    {
        accum += c_1 * (src[src_stride * extension(N, n + r + 1)]
            - src[src_stride * extension(N, n - r - 2)])
            + c_2 * (src[src_stride * extension(N, n + r)]
            - src[src_stride * extension(N, n - r - 1)]);
        dest[dest_stride * n] = accum;
    }
    
    return;
}

/**
 * \brief Extended box filtering approximation of Gaussian convolution
 * \param c             ebox_coeffs created by ebox_precomp()
 * \param dest_data     destination array
 * \param buffer_data   array with space for at least N samples
 * \param src           input, overwritten if src = dest_data
 * \param N             number of samples
 * \param stride        stride between successive samples
 *
 * This routine performs the extended box filtering approximation of Gaussian
 * convolution by multiple passes of the extended box filter. The filtering
 * itself is performed by ebox_filter().
 *
 * Since ebox_filter() requires that the source and destination are distinct,
 * the iteration alternates the roles of two arrays. See box_gaussian_conv()
 * for more detailed discussion.
 *
 * The convolution can be performed in-place by setting `src` = `dest_data`
 * (the source array is overwritten with the result). However, `buffer_data`
 * must be distinct from `dest_data`.
 */
template <typename T>
void ebox_gaussian_conv_(ebox_coeffs<T> c, T *dest_data, T *buffer_data, const T *src, long N, long stride)
{
    struct
    {
        T *data;
        long stride;
    } dest, buffer, cur, next;
    int step;
    
    assert(dest_data && buffer_data && src
        && dest_data != buffer_data && N > 0);
    
    dest.data = dest_data;
    dest.stride = stride;
    buffer.data = buffer_data;
    buffer.stride = (buffer_data == src) ? stride : 1;
    
    next = (buffer_data == src || (dest_data != src && c.K % 2 == 1))
        ? dest : buffer;
    /* Perform the first filtering pass. */
    ebox_filter(next.data, next.stride, src, stride,
        N, c.r, c.c_1, c.c_2);
    
    /* Perform (K - 1) filtering passes, alternating the roles of the dest
       and buffer arrays. */
    for (step = 1; step < c.K; ++step)
    {
        cur = next;
        next = (cur.data == buffer_data) ? dest : buffer;
        ebox_filter(next.data, next.stride, cur.data, cur.stride,
            N, c.r, c.c_1, c.c_2);
    }
    
    /* If necessary, copy the result to the destination array. */
    if (next.data != dest_data)
    {
        if (stride == 1)
            memcpy(dest_data, buffer_data, sizeof(T) * N);
        else
        {
            long n, i;
            
            for (n = i = 0; n < N; ++n, i += stride)
                dest_data[i] = buffer_data[n];
        }
    }
    
    return;
}

void ebox_gaussian_conv(ebox_coeffs<float> c, float *dest_data, float *buffer_data, const float *src, long N, long stride)
{
	ebox_gaussian_conv_<float>(c, dest_data, buffer_data, src, N, stride);
}
void ebox_gaussian_conv(ebox_coeffs<double> c, double *dest_data, double *buffer_data, const double *src, long N, long stride)
{
	ebox_gaussian_conv_<double>(c, dest_data, buffer_data, src, N, stride);
}
/**
 * \brief Extended box filtering approximation of 2D Gaussian convolution
 * \param c             ebox_coeffs created by ebox_precomp()
 * \param dest          destination image
 * \param buffer        array with at least max(width,height) samples
 * \param src           input image, overwritten if src = dest
 * \param width         image width
 * \param height        image height
 * \param num_channels  number of image channels
 *
 * Similar to ebox_gaussian_conv(), this routine approximates 2D Gaussian
 * convolution with extended box filtering.
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result). However, `buffer` must be
 * be distinct from `dest`.
 */
template <typename T>
void ebox_gaussian_conv_image_(ebox_coeffs<T> c, T *dest, T *buffer, const T *src, int width, int height, int num_channels)
{
    const long num_pixels = ((long)width) * ((long)height);
    int x, y, channel;
    
    assert(dest && buffer && src && dest != buffer && num_pixels > 0);
    
    /* Loop over the image channels. */
    for (channel = 0; channel < num_channels; ++channel)
    {
        T *dest_y = dest;
        const T *src_y = src;
        
        /* Filter each row of the channel. */
        for (y = 0; y < height; ++y)
        {
            ebox_gaussian_conv(c, dest_y, buffer, src_y, width, 1);
            dest_y += width;
            src_y += width;
        }
        
        /* Filter each column of the channel. */
        for (x = 0; x < width; ++x)
            ebox_gaussian_conv(c, dest + x, buffer, dest + x, height, width);
        
        dest += num_pixels;
        src += num_pixels;
    }
    
    return;
}

void ebox_gaussian_conv_image(ebox_coeffs<float> c, float *dest, float *buffer, const float *src, int width, int height, int num_channels)
{
	ebox_gaussian_conv_image_<float>(c, dest, buffer, src, width, height, num_channels);
}
void ebox_gaussian_conv_image(ebox_coeffs<double> c, double *dest, double *buffer, const double *src, int width, int height, int num_channels)
{
	ebox_gaussian_conv_image_<double>(c, dest, buffer, src, width, height, num_channels);
}

