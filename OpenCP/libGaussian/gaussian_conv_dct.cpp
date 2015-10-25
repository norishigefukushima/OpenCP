/**
 * \file gaussian_conv_dct.c
 * \brief Gaussian convolution via DCT
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

#include "gaussian_conv.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
/** \brief The constant pi */
#define M_PI        3.14159265358979323846264338327950288
#endif

/**
 * \brief DCT precomputations for Gaussian convolution
 * \param c         dct_coeffs pointer to hold precomputations
 * \param dest      output convolved data
 * \param src       data to be convolved, modified in-place if src = dest
 * \param N         number of samples
 * \param stride    stride between successive samples
 * \param sigma     standard deviation of the Gaussian in pixels
 * \return          1 on success, 0 on failure
 *
 * This routine performs precomputations for 1D DCT-based Gaussian
 * convolution, to be used in dct_gaussian_conv(). FFTW transform plans are
 * prepared for the forward and inverse DCT-II transform and the value
 * \f$ \alpha = (\sigma\pi/N)^2 / 2 \f$ is precomputed.
 *
 * The convolution can be performed in-place by setting src = dest (the
 * source array is overwritten with the result).
 *
 * \note For use of the #num typedef with FFTW's APIs, the macro
 * `FFT(functionname)` expands to `fftw_functionname` if #num is double or
 * `fftwf_functionname` if #num is float.
 */



int dct_precomp(dct_coeffs<float> *c, float *dest, const float *src, long N, long stride, float sigma)
{
	const fftw_r2r_kind dct_2 = FFTW_REDFT10, dct_3 = FFTW_REDFT01;
	double temp;
	int length = N;

	assert(c && dest && src && N > 0 && stride != 0 && sigma > 0);
	c->forward_plan = c->inverse_plan = NULL;



	if (!(c->forward_plan = FFTF(plan_many_r2r)(1, &length, 1, (float *)src,
		NULL, stride, 0, dest, NULL, stride, 0, &dct_2,
		FFTW_ESTIMATE | ((src != dest) ? FFTW_PRESERVE_INPUT : 0)))
		|| !(c->inverse_plan = FFTF(plan_many_r2r)(1, &length, 1, dest,
		NULL, stride, 0, dest, NULL, stride, 0, &dct_3, FFTW_ESTIMATE)))
	{
		dct_free(c);
		return 0;
	}



	c->dest = dest;
	c->src = src;
	c->conv_type = dct_coeffs<float>::DCT_GAUSSIAN_1D;
	temp = (sigma * M_PI) / N;
	c->dims.one.alpha = (float)(temp * temp / 2);
	c->dims.one.N = N;
	c->dims.one.stride = stride;
	return 1;
}
int dct_precomp(dct_coeffs<double> *c, double *dest, const double *src, long N, long stride, double sigma)
{
	const fftw_r2r_kind dct_2 = FFTW_REDFT10, dct_3 = FFTW_REDFT01;
	double temp;
	int length = N;

	assert(c && dest && src && N > 0 && stride != 0 && sigma > 0);
	c->forward_plan = c->inverse_plan = NULL;



	if (!(c->forward_plan = FFTD(plan_many_r2r)(1, &length, 1, (double *)src,
		NULL, stride, 0, dest, NULL, stride, 0, &dct_2,
		FFTW_ESTIMATE | ((src != dest) ? FFTW_PRESERVE_INPUT : 0)))
		|| !(c->inverse_plan = FFTD(plan_many_r2r)(1, &length, 1, dest,
		NULL, stride, 0, dest, NULL, stride, 0, &dct_3, FFTW_ESTIMATE)))
	{
		dct_free(c);
		return 0;
	}


	c->dest = dest;
	c->src = src;
	c->conv_type = dct_coeffs<double>::DCT_GAUSSIAN_1D;
	temp = (sigma * M_PI) / N;
	c->dims.one.alpha = (double)(temp * temp / 2);
	c->dims.one.N = N;
	c->dims.one.stride = stride;
	return 1;
}

/**
 * \brief DCT precomputations for 2D Gaussian convolution
 * \param c             dct_coeffs pointer to hold precomputations
 * \param dest          output convolved image
 * \param src           input image, modified in-place if src = dest
 * \param width         image width
 * \param height        image height
 * \param num_channels  number of image channels
 * \param sigma         standard deviation of the Gaussian in pixels
 * \return              1 on success, 0 on failure
 *
 * Similar to dct_precomp(), this routine performs precomputations for 2D
 * DCT-based Gaussian convolution, to be used in dct_gaussian_conv().
 *
 * The convolution can be performed in-place by setting `src` = `dest` (the
 * source array is overwritten with the result).
 */

int dct_precomp_image(dct_coeffs<float> *c, float *dest, const float *src, int width, int height, int num_channels, float sigma)
{
	const fftw_r2r_kind dct_2[] = { FFTW_REDFT10, FFTW_REDFT10 };
	const fftw_r2r_kind dct_3[] = { FFTW_REDFT01, FFTW_REDFT01 };
	const int dist = width * height;
	double temp;
	int size[2];

	assert(c && dest && src && width > 0 && height > 0 && sigma > 0);
	size[1] = width;
	size[0] = height;
	c->forward_plan = c->inverse_plan = NULL;

	if (!(c->forward_plan = FFTF(plan_many_r2r)(2, size, num_channels,
		(float *)src, NULL, 1, dist, dest, NULL, 1, dist, dct_2,
		FFTW_ESTIMATE | ((src != dest) ? FFTW_PRESERVE_INPUT : 0)))
		|| !(c->inverse_plan = FFTF(plan_many_r2r)(2, size, num_channels,
		dest, NULL, 1, dist, dest, NULL, 1, dist, dct_3, FFTW_ESTIMATE)))
	{
		dct_free(c);
		return 0;
	}

	c->dest = dest;
	c->src = src;
	c->conv_type = dct_coeffs<float>::DCT_GAUSSIAN_IMAGE;
	temp = (sigma * M_PI) / width;
	c->dims.image.alpha_x = (float)(temp * temp / 2);
	temp = (sigma * M_PI) / height;
	c->dims.image.alpha_y = (float)(temp * temp / 2);
	c->dims.image.width = width;
	c->dims.image.height = height;
	c->dims.image.num_channels = num_channels;
	return 1;
}
int dct_precomp_image(dct_coeffs<double> *c, double *dest, const double *src, int width, int height, int num_channels, double sigma)
{
	const fftw_r2r_kind dct_2[] = { FFTW_REDFT10, FFTW_REDFT10 };
	const fftw_r2r_kind dct_3[] = { FFTW_REDFT01, FFTW_REDFT01 };
	const int dist = width * height;
	double temp;
	int size[2];

	assert(c && dest && src && width > 0 && height > 0 && sigma > 0);
	size[1] = width;
	size[0] = height;
	c->forward_plan = c->inverse_plan = NULL;

	if (!(c->forward_plan = FFTD(plan_many_r2r)(2, size, num_channels,
		(double *)src, NULL, 1, dist, dest, NULL, 1, dist, dct_2,
		FFTW_ESTIMATE | ((src != dest) ? FFTW_PRESERVE_INPUT : 0)))
		|| !(c->inverse_plan = FFTD(plan_many_r2r)(2, size, num_channels,
		dest, NULL, 1, dist, dest, NULL, 1, dist, dct_3, FFTW_ESTIMATE)))
	{
		dct_free(c);
		return 0;
	}

	c->dest = dest;
	c->src = src;
	c->conv_type = dct_coeffs<double>::DCT_GAUSSIAN_IMAGE;
	temp = (sigma * M_PI) / width;
	c->dims.image.alpha_x = (double)(temp * temp / 2);
	temp = (sigma * M_PI) / height;
	c->dims.image.alpha_y = (double)(temp * temp / 2);
	c->dims.image.width = width;
	c->dims.image.height = height;
	c->dims.image.num_channels = num_channels;
	return 1;
}
/**
 * \brief Perform DCT-based Gaussian convolution
 * \param c     dct_coeffs created by dct_precomp() or dct_precomp_image()
 *
 * This routine performs 1D and 2D Gaussian convolution with symmetric
 * boundary handling using DCT transforms,
 * \f[ G_\sigma\!\stackrel{\text{sym}}{*}\!f=\mathcal{C}_\mathrm{2e}^{-1}\bigl
 (\mathcal{C}_\mathrm{1e}(G_\sigma)\cdot\mathcal{C}_\mathrm{2e}(f)\bigr), \f]
 * where \f$ \mathcal{C}_\mathrm{1e} \f$ and \f$ \mathcal{C}_\mathrm{2e} \f$
 * denote respectively the DCT-I and DCT-II transforms of the same period
 * length.
 *
 * In one dimension, the DCT-I transform of the Gaussian
 * \f$ \mathcal{C}_\mathrm{1e}(G_\sigma) \f$ is
 * \f[ \mathcal{C}_\mathrm{1e}(G_\sigma)_k =
 \exp\bigl(-2\pi^2\sigma^2(\tfrac{k}{2N})^2\bigr). \f]
 * The DCT-II transforms of the signal are performed using the FFTW library,
 * and the plans are prepared by dct_precomp() or dct_precomp_image().
 */


void dct_gaussian_conv(dct_coeffs<float> c)
{
	assert(c.forward_plan && c.inverse_plan);

	/* Perform the forward DCT-II transform. */
	FFTF(execute)(c.forward_plan);

	/* Perform spectral domain multiplication with the DCT-I transform of the
	Gaussian, which is exp(-alpha n^2) for the nth mode. */
	if (c.conv_type == dct_coeffs<float>::DCT_GAUSSIAN_1D)
	{   /* Multiplication in one dimension. */
		float denom = 2.f * c.dims.one.N;
		long n;

		for (n = 0; n < c.dims.one.N; ++n)
		{
			*c.dest *= ((float)exp(-c.dims.one.alpha * n * n)) / denom;
			c.dest += c.dims.one.stride;
		}
	}
	else
	{   /* Multiplication in two dimensions. */
		float denom = 4 * ((float)c.dims.image.width)
			* ((float)c.dims.image.height);
		int x, y, channel;

		for (channel = 0; channel < c.dims.image.num_channels; ++channel)
			for (y = 0; y < c.dims.image.height; ++y)
			{
				for (x = 0; x < c.dims.image.width; ++x)
					c.dest[x] *= ((float)exp(
					-c.dims.image.alpha_x * x * x
					- c.dims.image.alpha_y * y * y)) / denom;

				c.dest += c.dims.image.width;
			}
	}

	/* Perform the inverse DCT-II transform. */
	FFTF(execute)(c.inverse_plan);
	return;
}
void dct_gaussian_conv(dct_coeffs<double> c)
{
	assert(c.forward_plan && c.inverse_plan);

	/* Perform the forward DCT-II transform. */
	FFTD(execute)(c.forward_plan);

	/* Perform spectral domain multiplication with the DCT-I transform of the
	Gaussian, which is exp(-alpha n^2) for the nth mode. */
	if (c.conv_type == dct_coeffs<double>::DCT_GAUSSIAN_1D)
	{   /* Multiplication in one dimension. */
		float denom = 2.f * c.dims.one.N;
		long n;

		for (n = 0; n < c.dims.one.N; ++n)
		{
			*c.dest *= ((double)exp(-c.dims.one.alpha * n * n)) / denom;
			c.dest += c.dims.one.stride;
		}
	}
	else
	{   /* Multiplication in two dimensions. */
		double denom = 4 * ((double)c.dims.image.width)
			* ((double)c.dims.image.height);
		int x, y, channel;

		for (channel = 0; channel < c.dims.image.num_channels; ++channel)
			for (y = 0; y < c.dims.image.height; ++y)
			{
				for (x = 0; x < c.dims.image.width; ++x)
					c.dest[x] *= ((double)exp(
					-c.dims.image.alpha_x * x * x
					- c.dims.image.alpha_y * y * y)) / denom;

				c.dest += c.dims.image.width;
			}
	}

	/* Perform the inverse DCT-II transform. */
	FFTD(execute)(c.inverse_plan);
	return;
}
/**
 * \brief Release FFTW plans associated with a dct_coeffs struct
 * \param c     dct_coeffs created by dct_precomp() or dct_precomp_image()
 */

void dct_free(dct_coeffs<float> *c)
{
	assert(c);

	if (c->inverse_plan)
		FFTF(destroy_plan)(c->inverse_plan);
	if (c->forward_plan)
		FFTF(destroy_plan)(c->forward_plan);

	FFTF(cleanup)();
	c->forward_plan = c->inverse_plan = NULL;
	return;
}
void dct_free(dct_coeffs<double> *c)
{
	assert(c);

	if (c->inverse_plan)
		FFTD(destroy_plan)(c->inverse_plan);
	if (c->forward_plan)
		FFTD(destroy_plan)(c->forward_plan);

	FFTD(cleanup)();
	c->forward_plan = c->inverse_plan = NULL;
	return;
}