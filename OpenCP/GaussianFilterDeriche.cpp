#include "opencp.hpp"
#include <opencv2/core/internal.hpp>

//switch for double or float
#define NUM_SINGLE

#ifdef NUM_SINGLE
typedef float num;
#else
typedef double num;
#endif

/** \brief Minimum Deriche filter order */
#define DERICHE_MIN_K       2
/** \brief Maximum Deriche filter order */
#define DERICHE_MAX_K       4
/** \brief Test whether a given K value is a valid Deriche filter order */
#define DERICHE_VALID_K(K)  (DERICHE_MIN_K <= (K) && (K) <= DERICHE_MAX_K)
typedef struct deriche_coeffs
{
    num a[DERICHE_MAX_K + 1];             /**< Denominator coeffs          */
    num b_causal[DERICHE_MAX_K];          /**< Causal numerator            */
    num b_anticausal[DERICHE_MAX_K + 1];  /**< Anticausal numerator        */
    num sum_causal;                       /**< Causal filter sum           */
    num sum_anticausal;                   /**< Anticausal filter sum       */
    num sigma;                            /**< Gaussian standard deviation */
    int K;                                /**< Filter order = 2, 3, or 4   */
    num tol;                              /**< Boundary accuracy           */
    int max_iter;
} deriche_coeffs;

void deriche_precomp(deriche_coeffs *c, double sigma, int K, num tol);
void deriche_gaussian_conv(deriche_coeffs c,
    num *dest, num *buffer, const num *src, long N, long stride);
void deriche_gaussian_conv_image(deriche_coeffs c,
    num *dest, num *buffer, const num *src,
    int width, int height, int num_channels);


/** \brief Complex double data type. */
typedef struct _complex_type
{
    double real;    /**< real part      */
    double imag;    /**< imaginary part */
} _complex_type;

/** \brief Short alias for _complex_type */
#define complex     _complex_type

static complex make_complex(double a, double b)
{
    complex z;
    z.real = a;
    z.imag = b;
    return z;
}
static complex c_add(complex w, complex z)
{
    complex result;
    result.real = w.real + z.real;
    result.imag = w.imag + z.imag;
    return result;
}

static complex c_mul(complex w, complex z)
{
    complex result;
    result.real = w.real * z.real - w.imag * z.imag;
    result.imag = w.real * z.imag + w.imag * z.real;
    return result;
}

static complex c_div(complex w, complex z)
{
    complex result;
    
    /* For accuracy, choose the formula with the smaller value of |ratio|. */
    if (fabs(z.real) >= fabs(z.imag))
    {
        double ratio = z.imag / z.real;
        double denom = z.real + z.imag * ratio;
        result.real = (w.real + w.imag * ratio) / denom;
        result.imag = (w.imag - w.real * ratio) / denom;
    }
    else
    {
        double ratio = z.real / z.imag;
        double denom = z.real * ratio + z.imag;
        result.real = (w.real * ratio + w.imag) / denom;
        result.imag = (w.imag * ratio - w.real) / denom;
    }
    
    return result;
}

static void make_filter(num *result_b, num *result_a,
    const complex *alpha, const complex *beta, int K, double sigma);
void deriche_precomp(deriche_coeffs *c, double sigma, int K, num tol)
{
    /* Deriche's optimized filter parameters. */
    static const complex alpha[DERICHE_MAX_K - DERICHE_MIN_K + 1][4] = {
        {{0.48145, 0.971}, {0.48145, -0.971}},
        {{-0.44645, 0.5105}, {-0.44645, -0.5105}, {1.898, 0}},
        {{0.84, 1.8675}, {0.84, -1.8675},
            {-0.34015, -0.1299}, {-0.34015, 0.1299}}
        };
    static const complex lambda[DERICHE_MAX_K - DERICHE_MIN_K + 1][4] = {
        {{1.26, 0.8448}, {1.26, -0.8448}},
        {{1.512, 1.475}, {1.512, -1.475}, {1.556, 0}},
        {{1.783, 0.6318}, {1.783, -0.6318},
            {1.723, 1.997}, {1.723, -1.997}}
        };
    complex beta[DERICHE_MAX_K];
    
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
    make_filter(c->b_causal, c->a, alpha[K - DERICHE_MIN_K], beta, K, sigma);
    
    /* Numerator coefficients of the anticausal filter */
    c->b_anticausal[0] = (num)(0.0);
    
    for (k = 1; k < K; ++k)
        c->b_anticausal[k] = c->b_causal[k] - c->a[k] * c->b_causal[0];
    
    c->b_anticausal[K] = -c->a[K] * c->b_causal[0];
    
    /* Impulse response sums */
    for (k = 1; k <= K; ++k)
        accum_denom += c->a[k];
    
    for (k = 0, accum = 0.0; k < K; ++k)
        accum += c->b_causal[k];
    
    c->sum_causal = (num)(accum / accum_denom);
    
    for (k = 1, accum = 0.0; k <= K; ++k)
        accum += c->b_anticausal[k];
    
    c->sum_anticausal = (num)(accum / accum_denom);
    
    c->sigma = (num)sigma;
    c->K = K;
    c->tol = tol;
    c->max_iter = (int)ceil(10 * sigma);
    return;
}

#ifndef M_SQRT2PI
/** \brief The constant sqrt(2 pi) */
#define M_SQRT2PI   2.50662827463100050241576528481104525
#endif

void recursive_filter_impulse(num *h, long N,
    const num *b, int p, const num *a, int q)
{
    long m, n;
    
    assert(h && N > 0 && b && p >= 0 && a && q > 0);
    
    for (n = 0; n < N; ++n)
    {
        h[n] = (n <= p) ? b[n] : 0;
        
        for (m = 1; m <= q && m <= n; ++m)
            h[n] -= a[m] * h[n - m];
    }
    
    return;
}

/** \brief Maximum possible value of q in init_recursive_filter() */
#define MAX_Q       7

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

void init_recursive_filter(num *dest, const num *src, long N, long stride,
    const num *b, int p, const num *a, int q,
    num sum, num tol, long max_iter)
{
    num h[MAX_Q + 1];
    long n;
    int m;
    
    assert(dest && src && N > 0 && stride != 0
        && b && p >= 0 && a && 0 < q && q <= MAX_Q
        && tol > 0 && max_iter > 0);
    
    /* Compute the first q taps of the impulse response, h_0, ..., h_{q-1} */
    recursive_filter_impulse(h, q, b, p, a, q);
    
    /* Compute dest_m = sum_{n=1}^m h_{m-n} src_n, m = 0, ..., q-1 */
    for (m = 0; m < q; ++m)
        for (dest[m] = 0, n = 1; n <= m; ++n)
            dest[m] += h[m - n] * src[stride * extension(N, n)];

    for (n = 0; n < max_iter; ++n)
    {
        num cur = src[stride * extension(N, -n)];
        
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
static void make_filter(num *result_b, num *result_a,
    const complex *alpha, const complex *beta, int K, double sigma)
{
    const double denom = sigma * M_SQRT2PI;
    complex b[DERICHE_MAX_K], a[DERICHE_MAX_K + 1];
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
        result_b[k] = (num)(b[k].real / denom);
        result_a[k + 1] = (num)a[k + 1].real;
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
void deriche_gaussian_conv(deriche_coeffs c, num *dest, num *buffer, const num *src, int N, int stride)
{
    const long stride_2 = stride * 2;
    const long stride_3 = stride * 3;
    const long stride_4 = stride * 4;
    const long stride_N = stride * N;
    num *y_causal, *y_anticausal;
    long i, n;
    
    assert(dest && buffer && src && buffer != src && N > 0 && stride != 0);
        
    /* Divide buffer into two buffers each of length N. */
    y_causal = buffer;
    y_anticausal = buffer + N;
    
    /* Initialize the causal filter on the left boundary. */
    init_recursive_filter(y_causal, src, N, stride,
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
void deriche_gaussian_conv_image(deriche_coeffs c, num *dest, num *buffer, const num *src, int width, int height, int num_channels)
{
    long num_pixels = ((long)width) * ((long)height);
    int x, y, channel;
    
    assert(dest && buffer && src && num_pixels > 0);
    
    /* Loop over the image channels. */
    for (channel = 0; channel < num_channels; ++channel)
    {
        num *dest_y = dest;
        const num *src_y = src;
        
        /* Filter each row of the channel. */
        for (y = 0; y < height; ++y)
        {
            deriche_gaussian_conv(c, dest_y, buffer, src_y, width, 1);
            dest_y += width;
            src_y += width;
        }
        
        /* Filter each column of the channel. */
        for (x = 0; x < width; ++x)
		{
            deriche_gaussian_conv(c, dest + x, buffer, dest + x, height, width);
		}
        
        dest += num_pixels;
        src += num_pixels;
    }
    
    return;
}


void GaussianBlurDeriche(InputArray src_, OutputArray dest, float sigma, int K)
 {
	 Mat src = src_.getMat();
	 Mat srcf;
	 if(src.depth()!=CV_32F) src.convertTo(srcf,CV_32F);
	 else srcf = src;

	 if(src.channels()==1)
	 {	 
		 deriche_coeffs c;


		 if (!DERICHE_VALID_K(K))
		 {
			 fprintf(stderr, "Error: K=%d is invalid for Deriche\n", K);
		 }

		// printf("Deriche recursive filtering,"
			 //" K=%d, tol=%g boundary accuracy\n", param.K, param.tol);

		 deriche_precomp(&c, sigma, K, (float)1e-6);

		 Mat buffer(src.cols*2*src.rows,1,CV_32F);
		 deriche_gaussian_conv_image(c, srcf.ptr<float>(0), buffer.ptr<float>(0), srcf.ptr<float>(0),
			 src.cols, src.rows, 1);
	 }
	 else if (src.channels()==3)
	 {/*
		 vector<Mat> plane;
		 split(srcf,plane);
		// cvtColorBGR2PLANE(src,plane);
		 gaussianiir2d(plane[0].ptr<float>(0),src.cols,src.rows, sigma, numsteps);
		 gaussianiir2d(plane[1].ptr<float>(0),src.cols,src.rows, sigma, numsteps);
		 gaussianiir2d(plane[2].ptr<float>(0),src.cols,src.rows, sigma, numsteps);

		 merge(plane,dest);*/
	 }

	 if(src.depth()!=CV_32F)srcf.convertTo(dest,src.type(),1.0,0.5);
	 else srcf.copyTo(dest);
 }