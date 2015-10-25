#pragma once
#include <fftw3.h>
/**
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

//util
#ifdef __GNUC__
__attribute__((pure, unused))
#endif
/**
* \brief Half-sample symmetric boundary extension
* \param N     signal length
* \param n     requested sample, possibly outside {0,...,`N`-1}
* \return reflected sample in {0,...,`N`-1}
*
* This function is used for boundary handling. Suppose that `src` is an array
* of length `N`, then `src[extension(N, n)]` evaluates the symmetric
* extension of `src` at location `n`.
*
* Half-sample symmetric extension is implemented by the pseudocode
\verbatim
repeat
if n < 0, reflect n over -1/2
if n >= N, reflect n over N - 1/2
until 0 <= n < N
\endverbatim
* The loop is necessary as some `n` require multiple reflections to bring
* them into the domain {0,...,`N`-1}.
*
* This function is used by all of the Gaussian convolution algorithms
* included in this work except for DCT-based convolution (where symmetric
* boundary handling is performed implicitly by the transform). For FIR, box,
* extended box, SII, and Deriche filtering, this function could be replaced
* to apply some other boundary extension (e.g., periodic or constant
* extrapolation) without any further changes. However, Alvarez-Mazorra and
* Vliet-Young-Verbeek are hard-coded for symmetric extension on the right
* boundary, and would require specific modification to change the handling
* on the right boundary.
*
* \par A note on efficiency
* This function is a computational bottleneck, as it is used within core
* filtering loops. As a small optimization, we encourage inlining by defining
* the function as `static`. We refrain from further optimization since this
* is a pedagogical implementation, and code readability is more important.
* Ideally, filtering routines should take advantage of algorithm-specific
* properties such as exploiting sequential sample locations (to update the
* extension cheaply) and samples that are provably in the interior (where
* boundary checks may omitted be entirely).
*/
static long extension(long N, long n)
{
	while (1)
		if (n < 0)
			n = -1 - n;         /* Reflect over n = -1/2.    */
		else if (n >= N)
			n = 2 * N - 1 - n;  /* Reflect over n = N - 1/2. */
		else
			break;

		return n;
}

template<typename T>
void recursive_filter_impulse(T *h, long N, const T *b, int p, const T *a, int q);

template<typename T>
void init_recursive_filter(T *dest, const T *src, long N, long stride, const T *b, int p, const T *a, int q, T sum, T tol, int max_iter);
void init_recursive_filter(double *dest, const double *src, long N, long stride, const double *b, int p, const double *a, int q, double sum, double tol, int max_iter);
void init_recursive_filter(float *dest, const float *src, long N, long stride, const float *b, int p, const float *a, int q, float sum, float tol, int max_iter);

int invert_matrix(double *inv_A, double *A, int N);


#ifndef M_1_SQRTPI
/** \brief The constant \f$ 1/\pi \f$ */
#define M_1_SQRTPI  0.564189583547756286948
#endif

double erfc_cody(double x);

/** \brief Short alias of erfc_cody() */

#ifndef M_SQRT2
/** \brief The constant \f$ \sqrt{2} \f$ */
#define M_SQRT2     1.41421356237309504880168872420969808
#endif
#ifndef M_SQRT2PI
/** \brief The constant \f$ \sqrt{2 \pi} \f$ */
#define M_SQRT2PI   2.50662827463100050241576528481104525
#endif

double inverfc_acklam(double x);

/** \brief Short alias of inverfc_acklam() */
#define inverfc(x)  inverfc_acklam(x)

/** \def FFT
* For use with the FFTW libary, the macro `FFT(functionname)` is defined
* such that it expands to
*    `fftwf_functionname`  if #num is single,
* or
*    `fftw_functionname`   if #num is double.
*/
/** \brief Token-pasting macro */
#define _FFTW_CONCAT(A,B)    A ## B


/** \def IMAGEIO_NUM
* For use with imageio.c, define `IMAGEIO_NUM` to be either `IMAGEIO_SINGLE`
* or `IMAGEIO_DOUBLE`, depending on whether `NUM_SINGLE` is defined.
*/
#ifdef NUM_SINGLE
#define IMAGEIO_NUM     IMAGEIO_SINGLE
#else
#define IMAGEIO_NUM     IMAGEIO_DOUBLE
#endif

//dct
//dct
/**
* \file gaussian_conv_dct.h
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

/**
* \defgroup dct_gaussian DCT-based Gaussian convolution
* \brief Convolution via multiplication in the DCT domain.
*
* Via the convolution-multiplication property, discrete cosine transforms
* (DCTs) are an effective way to implement Gaussian convolution. We follow
* Martucci's use of DCTs to perform convolution with half-sample symmetric
* boundary handling,
* \f[ G_\sigma\!\stackrel{\text{sym}}{*}\!f=\mathcal{C}_\mathrm{2e}^{-1}\bigl
(\mathcal{C}_\mathrm{1e}(G_\sigma)\cdot\mathcal{C}_\mathrm{2e}(f)\bigr), \f]
* where \f$ \mathcal{C}_\mathrm{1e} \f$ and \f$ \mathcal{C}_\mathrm{2e} \f$
* denote respectively the DCT-I and DCT-II transforms of the same period
* length. This DCT-based convolution is equivalent to (but is faster and
* uses less memory) than FFT-based convolution with the symmetrized signal
* \f$ (f_0,\ldots,f_{N-1},f_{N-1},\ldots,f_0) \f$.
*
* The FFTW library is used to compute the DCT transforms.
*
* The process to use these functions is the following:
*    -# dct_precomp() or dct_precomp_image() to set up the convolution
*    -# dct_gaussian_conv() to perform the convolution itself (it may
*       be called multiple times if desired)
*    -# dct_free() to clean up
*
* \par Example
\code
dct_coeffs c;

dct_precomp(&c, dest, src, N, stride, sigma);
dct_gaussian_conv(c);
dct_free(&c);
\endcode
*
* \par Reference
*  - S. Martucci, "Symmetric convolution and the discrete sine and cosine
*    transforms," IEEE Transactions on Signal Processing SP-42,
*    pp. 1038-1051, 1994. http://dx.doi.org/10.1109/78.295213
*
* \{
*/

/** \brief FFTW plans and coefficients for DCT-based Gaussian convolution */

template <typename T>
struct dct_coeffs
{
	;
};

template <>
struct dct_coeffs<float>
{
#define FFTF(S)      _FFTW_CONCAT(fftwf_,S)
	FFTF(plan) forward_plan;     /**< forward DCT plan   */
	FFTF(plan) inverse_plan;     /**< inverse DCT plan   */
	float *dest;                  /**< destination array  */
	const float *src;             /**< source array       */

	enum
	{
		DCT_GAUSSIAN_1D,
		DCT_GAUSSIAN_IMAGE
	} conv_type;                /**< flag to switch between 1D vs. 2D  */
	union
	{
		struct
		{
			float alpha;          /**< exponent coefficient              */
			long N;             /**< signal length                     */
			long stride;        /**< stride between successive samples */
		} one;                  /**< 1D convolution parameters         */
		struct
		{
			float alpha_x;        /**< exponent coefficient              */
			float alpha_y;        /**< exponent coefficient              */
			int width;          /**< image width                       */
			int height;         /**< image height                      */
			int num_channels;   /**< number of image channels          */
		} image;                /**< 2D convolution parameters         */
	} dims;
};

template <>
struct dct_coeffs<double>
{
#define FFTD(S)      _FFTW_CONCAT(fftw_,S)
	FFTD(plan) forward_plan;     /**< forward DCT plan   */
	FFTD(plan) inverse_plan;     /**< inverse DCT plan   */
	double *dest;                  /**< destination array  */
	const double *src;             /**< source array       */

	enum
	{
		DCT_GAUSSIAN_1D,
		DCT_GAUSSIAN_IMAGE
	} conv_type;                /**< flag to switch between 1D vs. 2D  */
	union
	{
		struct
		{
			double alpha;          /**< exponent coefficient              */
			long N;             /**< signal length                     */
			long stride;        /**< stride between successive samples */
		} one;                  /**< 1D convolution parameters         */
		struct
		{
			double alpha_x;        /**< exponent coefficient              */
			double alpha_y;        /**< exponent coefficient              */
			int width;          /**< image width                       */
			int height;         /**< image height                      */
			int num_channels;   /**< number of image channels          */
		} image;                /**< 2D convolution parameters         */
	} dims;
};


int dct_precomp(dct_coeffs<double> *c, double *dest, const double *src, long N, long stride, double sigma);
int dct_precomp(dct_coeffs<float> *c, float *dest, const float *src, long N, long stride, float sigma);
int dct_precomp_image(dct_coeffs<double> *c, double *dest, const double *src, int width, int height, int num_channels, double sigma);
int dct_precomp_image(dct_coeffs<float> *c, float *dest, const float *src, int width, int height, int num_channels, float sigma);

void dct_free(dct_coeffs<float> *c);
void dct_free(dct_coeffs<double> *c);
void dct_gaussian_conv(dct_coeffs<float> c);
void dct_gaussian_conv(dct_coeffs<double> c);
//am
/**
* \defgroup am_gaussian Alvarez-Mazorra Gaussian convolution
* \brief A first-order recursive filter approximation, computed in-place.
*
* This code implements Alvarez and Mazorra's recursive filter approximation
* of Gaussian convolution. The Gaussian is approximated by a cascade of
* first-order recursive filters,
* \f[ H(z) = \left(\nu/\lambda\right)^K
\left( \frac{1}{1 - \nu z^{-1}} \frac{1}{1 - \nu z} \right)^K. \f]
*
* \par References
*  - Alvarez, Mazorra, "Signal and Image Restoration using Shock Filters and
*    Anisotropic Diffusion," SIAM J. on Numerical Analysis, vol. 31, no. 2,
*    pp. 590-605, 1994. http://www.jstor.org/stable/2158018
*
* \{
*/
void am_gaussian_conv(float *dest, const float *src, long N, long stride, double sigma, int K, float tol, bool use_adjusted_q);
void am_gaussian_conv_(double *dest, const double *src, long N, long stride, double sigma, int K, double tol, bool use_adjuste);

void am_gaussian_conv_image(double *dest, const double *src, int width, int height, int num_channels, double sigma, int K, double tol, bool use_adjusted_q);
void am_gaussian_conv_image(float *dest,  const float *src,  int width, int height, int num_channels, float sigma,  int K, float tol, bool use_adjusted_q);

//box
/**
* \defgroup box_gaussian Box filter Gaussian convolution
* \brief A fast low-accuracy approximation of Gaussian convolution.
*
* This code implements the basic iterated box filter approximation of
* Gaussian convolution as developed by Wells. This approach is based on the
* efficient recursive implementation of the box filter as
* \f[ H(z) = \frac{1}{2r + 1}\frac{z^r - z^{-r-1}}{1 - z^{-1}}, \f]
* where r is the box radius.
*
* While box filtering is very efficient, it has the limitation that only a
* quantized set of \f$ \sigma \f$ values can be approximated because the box
* radius r is integer. \ref ebox_gaussian and \ref sii_gaussian are
* extensions of box filtering that allow \f$ \sigma \f$ to vary continuously.
*
* \par Example
\code
num *buffer;

buffer = (num *)malloc(sizeof(num) * N);
box_gaussian_conv(dest, buffer, src, N, stride, sigma, K);
free(buffer);
\endcode
*
* \par Reference
*  - W.M. Wells, "Efficient synthesis of Gaussian filters by cascaded
*    uniform filters," IEEE Transactions on Pattern Analysis and Machine
*    Intelligence, vol. 8, no. 2, pp. 234-239, 1986.
*    http://dx.doi.org/10.1109/TPAMI.1986.4767776
*
* \{
*/


void box_gaussian_conv(float *dest, float *buffer, const float *src, long N, long stride, float sigma, int K);
void box_gaussian_conv_image(float *dest, float *buffer, const float *src, int width, int height, int num_channels, float sigma, int K);
void box_gaussian_conv(double *dest, double *buffer, const double *src, long N, long stride, double sigma, int K);
void box_gaussian_conv_image(double *dest, double *buffer, const double *src, int width, int height, int num_channels, double sigma, int K);

//ebox
/**
* \defgroup ebox_gaussian Extended box filter Gaussian convolution
* \brief An improvement of box filtering with continuous selection of sigma.
*
* This code implements the extended box filter approximation of Gaussian
* convolution proposed by Gwosdek, Grewenig, Bruhn, and Weickert. The
* extended box filter is the recursive filter
* \f[ u_n = u_{n-1}+c_1(f_{n+r+1}-f_{n-r-2})+c_2(f_{n+r}-f_{n-r-1}). \f]
*
* The process to use these functions is the following:
*    -# ebox_precomp() to precompute coefficients for the convolution
*    -# ebox_gaussian_conv() or ebox_gaussian_conv_image() to perform
*       the convolution itself (may be called multiple times if desired)
*
* \par Example
\code
ebox_coeffs c;
num *buffer;

ebox_precomp(&c, sigma, K);
buffer = (num *)malloc(sizeof(num) * N);
ebox_gaussian_conv(c, dest, buffer, src, N, stride);
free(buffer);
\endcode
*
* \par Reference
*  - P. Gwosdek, S. Grewenig, A. Bruhn, J. Weickert, "Theoretical
*    Foundations of Gaussian Convolution by Extended Box Filtering,"
*    International Conference on Scale Space and Variational Methods in
*    Computer Vision, pp. 447-458, 2011.
*    http://dx.doi.org/10.1007/978-3-642-24785-9_38
*
* \{
*/

/** \brief Coefficients for extended box filter Gaussian approximation */
template <typename T>
struct ebox_coeffs
{
	T c_1;        /**< Outer box weight           */
	T c_2;        /**< Inner box weight           */
	long r;         /**< Inner box radius           */
	int K;          /**< Number of filtering passes */
};
void ebox_precomp(ebox_coeffs<double> *c, double sigma, const int K);
void ebox_precomp(ebox_coeffs<float> *c, float sigma, const int K);
void ebox_gaussian_conv(ebox_coeffs<double> c, double *dest_data, double *buffer_data, const double *src, long N, long stride);
void ebox_gaussian_conv(ebox_coeffs<float> c, float *dest_data, float *buffer_data, const float *src, long N, long stride);
void ebox_gaussian_conv_image(ebox_coeffs<double> c, double *dest, double *buffer, const double *src, int width, int height, int num_channels);
void ebox_gaussian_conv_image(ebox_coeffs<float> c, float *dest, float *buffer, const float *src, int width, int height, int num_channels);
//fir
/**
* \defgroup fir_gaussian FIR Gaussian convolution
* \brief Approximation by finite impulse response filtering.
*
* This code approximates Gaussian convolution with finite impulse response
* (FIR) filtering. By truncating the Gaussian to a finite support, Gaussian
* convolution is approximated by
* \f[ H(z) = \tfrac{1}{s(r)}\sum_{n=-r}^r G_\sigma(n) z^{-n}, \quad
s(r) = \sum_{n=-r}^{r} G_\sigma(n). \f]
*
* The process to use these functions is the following:
*    -# fir_precomp() to precompute filter coefficients for the convolution
*    -# fir_gaussian_conv() or fir_gaussian_conv_image() to perform
*       the convolution itself (may be called multiple times if desired)
*    -# fir_free() to clean up
*
* \par Example
\code
fir_coeffs c;

fir_precomp(&c, sigma, tol);
fir_gaussian_conv(c, dest, src, N, stride);
fir_free(&c);
\endcode
*
* \{
*/

//for
/** \brief Coefficients for FIR Gaussian approximation */
template <typename T>
struct fir_coeffs
{
	T *g_trunc;   /**< FIR filter coefficients            */
	long radius;    /**< The radius of the filter's support */
};

void fir_gaussian_conv_image(fir_coeffs<double> c, double *dest, double *buffer, const double *src, int width, int height, int num_channels);
void fir_gaussian_conv_image(fir_coeffs<float> c, float *dest, float *buffer, const float *src, int width, int height, int num_channels);
void fir_free(fir_coeffs<double> *c);
void fir_free(fir_coeffs<float> *c);
int fir_precomp(fir_coeffs<float> *c, float sigma, float tol);
int fir_precomp(fir_coeffs<double> *c, double sigma, double tol);
//sii
/**
* \defgroup sii_gaussian Stacked integral images Gaussian convolution
* \brief A sum of box filter responses, rather than a cascade.
*
* This code implements stacked integral images Gaussian convolution
* approximation introduced by Bhatia, Snyder, and Bilbro and refined by
* Elboher and Werman. The Gaussian is approximated as
* \f[ u_n = \sum_{k=1}^K w_k (s_{n+r_k} - s_{n-r_k-1}), \quad
s_n = \sum_{m\le n} f_n, \f]
* where the kth term of the sum is effectively a box filter of radius
* \f$ r_k \f$.
*
* The process to use these functions is the following:
*    -# sii_precomp() to precompute coefficients for the convolution
*    -# sii_gaussian_conv() or sii_gaussian_conv_image() to perform
*       the convolution itself (may be called multiple times if desired)
*
* The function sii_buffer_size() should be used to determine the minimum
* required buffer size.
*
* \par Example
\code
sii_coeffs c;
num *buffer;

sii_precomp(&c, sigma, K);
buffer = (num *)malloc(sizeof(num) * sii_buffer_size(c, N));
sii_gaussian_conv(c, dest, buffer, src, N, stride);
free(buffer);
\endcode
*
* \par References
*  - A. Bhatia, W.E. Snyder, G. Bilbro, "Stacked Integral Image," IEEE
*    International Conference on Robotics and Automation (ICRA),
*    pp. 1530-1535, 2010. http://dx.doi.org/10.1109/ROBOT.2010.5509400
*  - E. Elboher, M. Werman, "Efficient and Accurate Gaussian Image Filtering
*    Using Running Sums," Computing Research Repository,
*    vol. abs/1107.4958, 2011. http://arxiv.org/abs/1107.4958
*
* \{
*/

//sii
/** \brief Minimum SII filter order */
#define SII_MIN_K       3
/** \brief Maximum SII filter order */
#define SII_MAX_K       5
/** \brief Test whether a given K value is a valid SII filter order */
#define SII_VALID_K(K)  (SII_MIN_K <= (K) && (K) <= SII_MAX_K)

/** \brief Parameters for stacked integral images Gaussian approximation */
template<typename T>
struct sii_coeffs
{
	__declspec(align(16)) T weights[SII_MAX_K];     /**< Box weights     */
	__declspec(align(16)) int radii[SII_MAX_K];      /**< Box radii       */
	int K;                      /**< Number of boxes */
};

void sii_precomp(sii_coeffs<double> &c, const double sigma, const int K);
void sii_precomp(sii_coeffs<float> &c, const float sigma, const int K);

long sii_buffer_size(sii_coeffs<float> c, long N);
long sii_buffer_size(sii_coeffs<double> c, long N);

void sii_gaussian_conv(sii_coeffs<double> c, double *dest, double *buffer, const double *src, long N, long stride);
void sii_gaussian_conv(sii_coeffs<float> c, float *dest, float *buffer, const float *src, long N, long stride);

void sii_gaussian_conv_image(sii_coeffs<double> c, double *dest, double *buffer, const double *src, int width, int height, int num_channels);
void sii_gaussian_conv_image(sii_coeffs<float> c, float *dest, float *buffer, const float *src, int width, int height, int num_channels);

//short conv
void gaussian_short_conv(double *dest, const double *src, const int N, const int stride, const double sigma);
void gaussian_short_conv( float *dest, const  float *src, const int N, const int stride, const  float sigma);

//deriche
/**
* \defgroup deriche_gaussian Deriche Gaussian convolution
* \brief An accurate recursive filter approximation, computed out-of-place.
*
* Deriche uses a sum of causal and an anticausal recursive filters to
* approximate the Gaussian. The causal filter has the form
* \f[ H^{(K)}(z) = \frac{1}{\sqrt{2\pi\sigma^2}}\sum_{k=1}^K \frac{\alpha_k}
{1-\mathrm{e}^{-\lambda_k/\sigma} z^{-1}}
= \frac{\sum_{k=0}^{K-1}b_k z^{-k}}{1+\sum_{k=1}^{K}a_k z^{-k}}, \f]
* where K is the filter order (2, 3, or 4). The anticausal form is
* the spatial reversal of the causal filter minus the sample at n = 0,
* \f$ H^{(K)}(z^{-1}) - h_0^{(K)}. \f$
*
* The process to use these functions is the following:
*    -# deriche_precomp() to precompute coefficients for the convolution
*    -# deriche_gaussian_conv() or deriche_gaussian_conv_image() to perform
*       the convolution itself (may be called multiple times if desired)
*
* \par Example
\code
deriche_coeffs c;
num *buffer;

deriche_precomp(&c, sigma, K, tol);
buffer = (num *)malloc(sizeof(num) * 2 * N);
deriche_gaussian_conv(c, dest, buffer, src, N, stride);
free(buffer);
\endcode
*
* \note When the #num typedef is set to single-precision arithmetic,
* deriche_gaussian_conv() may be inaccurate for large values of sigma.
*
* \par Reference
*  - R. Deriche, "Recursively Implementing the Gaussian and its
*    Derivatives," INRIA Research Report 1893, France, 1993.
*    http://hal.inria.fr/docs/00/07/47/78/PDF/RR-1893.pdf
*
* \{
*/

/** \brief Minimum Deriche filter order */
#define DERICHE_MIN_K       2
/** \brief Maximum Deriche filter order */
#define DERICHE_MAX_K       4
/** \brief Test whether a given K value is a valid Deriche filter order */
#define DERICHE_VALID_K(K)  (DERICHE_MIN_K <= (K) && (K) <= DERICHE_MAX_K)

/**
* \brief Coefficients for Deriche Gaussian approximation
*
* The deriche_coeffs struct stores the filter coefficients for the causal and
* anticausal recursive filters of order `K`. This struct allows to precompute
* these filter coefficients separately from actually performing the filtering
* so that filtering may be performed multiple times using the same
* precomputed coefficients.
*
* This coefficients struct is precomputed by deriche_precomp() and then used
* by deriche_gaussian_conv() or deriche_gaussian_conv_image().
*/
template<typename T>
struct deriche_coeffs
{
	__declspec(align(16)) T a[DERICHE_MAX_K + 1];             /**< Denominator coeffs          */
	__declspec(align(16)) T b_causal[DERICHE_MAX_K];          /**< Causal numerator            */
	__declspec(align(16)) T b_anticausal[DERICHE_MAX_K + 1];  /**< Anticausal numerator        */
	T sum_causal;                       /**< Causal filter sum           */
	T sum_anticausal;                   /**< Anticausal filter sum       */
	T sigma;                            /**< Gaussian standard deviation */
	int K;                                /**< Filter order = 2, 3, or 4   */
	T tol;                              /**< Boundary accuracy           */
	int max_iter;
};

void deriche_precomp(deriche_coeffs<double> *c, double sigma, int K, double tol);
void deriche_precomp(deriche_coeffs<float> *c, float sigma, int K, float tol);

void deriche_gaussian_conv(deriche_coeffs<double> c, double *dest, double *buffer, const double *src, long N, long stride);
void deriche_gaussian_conv(deriche_coeffs<float> c, float *dest, float *buffer, const float *src, long N, long stride);

void deriche_gaussian_conv_image(deriche_coeffs<double> c, double *dest, double *buffer, const double *src, int width, int height, int num_channels);
void deriche_gaussian_conv_image(deriche_coeffs<float> c, float *dest, float *buffer, const float *src, int width, int height, int num_channels);

//vyv
/**
* \defgroup vyv_gaussian Vliet-Young-Verbeek Gaussian convolution
* \brief An accurate recursive filter approximation, computed in-place.
*
* This code implements the recursive filter approximation of Gaussian
* convolution proposed by Vliet, Young, and Verbeek. The Gaussian is
* approximated by a cascade of a causal filter and an anticausal filter,
* \f[ H(z) = G(z) G(z^{-1}), \quad
G(z) = \frac{b_0}{1 + a_1 z^{-1} + \cdots + a_K z^{-K}}. \f]
*
* The process to use these functions is the following:
*    -# vyv_precomp() to precompute coefficients for the convolution
*    -# vyv_gaussian_conv() or vyv_gaussian_conv_image() to perform
*       the convolution itself (may be called multiple times if desired)
*
* \par Example
\code
vyv_coeffs c;

vyv_precomp(&c, sigma, K, tol);
vyv_gaussian_conv(c, dest, src, N, stride);
\endcode
*
* \note When the #num typedef is set to single-precision arithmetic,
* vyv_gaussian_conv() may be inaccurate for large values of sigma.
*
* \par References
*  - I.T. Young, L.J. van Vliet, "Recursive implementation of the Gaussian
*    filter," Signal Processing, vol. 44, no. 2, pp. 139-151, 1995.
*    http://dx.doi.org/10.1016/0165-1684(95)00020-E
*  - L.J. van Vliet, I.T. Young, P.W. Verbeek, "Recursive Gaussian
*    derivative filters," Proceedings of the 14th International Conference
*    on Pattern Recognition, vol. 1, pp. 509-514, 1998.
*    http://dx.doi.org/10.1109/ICPR.1998.711192
*
* \{
*/

/** \brief Minimum valid VYV filter order. */
#define VYV_MIN_K       3
/** \brief Maximum valid VYV filter order. */
#define VYV_MAX_K       5
/** \brief Test whether a given K value is a valid VYV filter order. */
#define VYV_VALID_K(K)  (VYV_MIN_K <= (K) && (K) <= VYV_MAX_K)

/**
* \brief Coefficients for Vliet-Young-Verbeek Gaussian approximation.
*
* The vyv_coeffs struct stores the coefficients for the recursive filter of
* order K. This struct allows to precompute these filter coefficients
* separately from actually performing the filtering so that filtering may be
* performed multiple times using the same precomputed coefficients.
*
* This coefficients struct is precomputed by vyv_precomp() and then used
* by vyv_gaussian_conv() or vyv_gaussian_conv_image().
*/
template <typename T>
struct vyv_coeffs
{
	__declspec(align(16)) T filter[VYV_MAX_K + 1];     /**< Recursive filter coefficients       */
	__declspec(align(16)) T M[VYV_MAX_K * VYV_MAX_K];  /**< Matrix for handling right boundary  */
	T sigma;                     /**< Gaussian standard deviation         */
	T tol;                       /**< Boundary accuracy                   */
	int K;                         /**< Filter order                        */
	int max_iter;                 /**< Max iterations for left boundary    */
};



template <typename T>
void vyv_precomp_(vyv_coeffs<T> *c, T sigma, int K, T tol);
void vyv_precomp(vyv_coeffs<double> *c, double sigma, int K, double tol);
void vyv_precomp(vyv_coeffs<float> *c, float sigma, int K, float tol);


template <typename T>
void vyv_gaussian_conv(const vyv_coeffs<T> c, T *dest, const T *src, const int N, const int stride);

void vyv_gaussian_conv_image(vyv_coeffs<double> c, double *dest, const double *src, const int width, const int height, const int num_channels);
void vyv_gaussian_conv_image(vyv_coeffs<float> c, float *dest, const float *src, const int width, const int height, const int num_channels);
