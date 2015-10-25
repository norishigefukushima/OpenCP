/**
 * \file gaussian_short_conv.c
 * \brief Gaussian convolution for short signals (N <= 4)
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
 * You should have received a copy of these licenses along this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "gaussian_conv.h"
#include <assert.h>
#include <math.h>

#ifndef M_PI_4
/** \brief The constant pi/4 */
#define M_PI_4      0.78539816339744830961566084581987572
#endif

/* Define coefficients for computing short DCTs */
#ifndef M_1_SQRT2
/** \brief The constant 1/sqrt(2) */
#define M_1_SQRT2   0.70710678118654752440084436210484904
#endif
#ifndef M_SQRT3
/** \brief The constant sqrt(3) */
#define M_SQRT3     1.73205080756887729352744634150587237
#endif
#ifndef M_COSPI_8
/** \brief The constant cos(pi/8) = sqrt(sqrt(2) + 2)/2 */
#define M_COSPI_8   0.92387953251128675612818318939678829
#endif
#ifndef M_COSPI3_8
/** \brief The constant cos(pi 3/8) = sqrt(2 - sqrt(2))/2 */
#define M_COSPI3_8  0.38268343236508977172845998403039887
#endif


/**
 * \brief Gaussian convolution on a short signal (N <= 4)
 * \param dest      output convolved data
 * \param src       data to be convolved, modified in-place if src=dest
 * \param N         number of samples (0 < N <= 4)
 * \param stride    stride between successive samples
 * \param sigma     Gaussian standard deviation
 *
 * This routine performs DCT-based Gaussian convolution for very short
 * signals, `N` <= 4.  Some of the Gaussian convolution implementations
 * cannot handle such short signals, and this routine is used as a fallback.
 *
 * Since the signals are of very short lengths, we compute the DCTs by
 * direct formulas rather than with FFTW.
 */
template<typename T>
void gaussian_short_conv_(T *dest, const T *src, const int N, const int stride, T sigma)
{
    double alpha = (2.0 * M_PI_4 * M_PI_4) * (sigma * sigma);
    
    assert(dest && src && 0 < N && N <= 4 && stride != 0 && sigma > 0);
    
    switch (N)
    {
    case 1:
        dest[0] = src[0];
        break;
    case 2:
        {
            double F0 = 0.5 * (src[0] + src[stride]);
            double F1 = 0.5 * (src[0] - src[stride]);
            F1 *= exp(-alpha);
            dest[0] = (T)(F0 + F1);
            dest[stride] = (T)(F0 - F1);
        }
        break;
    case 3:
        {
            double F0 = (src[0] + src[stride] + src[stride * 2]) / 3.0;
            double F1 = ((0.5*M_SQRT3) * (src[0] - src[stride * 2])) / 3.0;
            double F2 = (0.5*(src[0] + src[stride * 2]) - src[stride]) / 3.0;
            F1 *= exp(-alpha);
            F2 *= exp(-4.0 * alpha);
            dest[0] = (T)(F0 + M_SQRT3 * F1 + F2);
            dest[stride] = (T)(F0 - 2.0 * F2);
            dest[stride * 2] = (T)(F0 - M_SQRT3 * F1 + F2);
        }
        break;
    case 4:
        {
            double F0 = (src[0] + src[stride]
                + src[stride * 2] + src[stride * 3]) / 4.0;
            double F1 = (M_COSPI_8 * (src[0] - src[stride * 3])
                + M_COSPI3_8 * (src[stride] - src[stride * 2])) / 4.0;
            double F2 = (M_1_SQRT2 * (src[0] - src[stride]
                + src[stride * 2] - src[stride * 3])) / 4.0;
            double F3 = (M_COSPI3_8 * (src[0] - src[stride * 3])
                - M_COSPI_8 * (src[stride] - src[stride * 2])) / 4.0;
            F1 *= exp(-alpha);
            F2 *= exp(-4.0 * alpha);
            F3 *= exp(-9.0 * alpha);
            dest[0] = (T)(F0 + 2.0 *
                (M_COSPI_8 * F1 + M_1_SQRT2 * F2 + M_COSPI3_8 * F3));
            dest[stride] = (T)(F0 + 2.0 *
                (M_COSPI3_8 * F1 - M_1_SQRT2 * F2 - M_COSPI_8 * F3));
            dest[stride * 2] = (T)(F0 + 2.0 *
                (-M_COSPI3_8 * F1 - M_1_SQRT2 * F2 + M_COSPI_8 * F3));
            dest[stride * 3] = (T)(F0 + 2.0 *
                (-M_COSPI_8 * F1 + M_1_SQRT2 * F2 - M_COSPI3_8 * F3));
        }
        break;
    }
    
    return;
}

void gaussian_short_conv(float *dest, const float *src, const int N, const int stride, const float sigma)
{
	gaussian_short_conv_<float>(dest, src, N, stride, sigma);
}

void gaussian_short_conv(double *dest, const double *src, const int N, const int stride, const double sigma)
{
	gaussian_short_conv_<double>(dest, src, N, stride, sigma);
}