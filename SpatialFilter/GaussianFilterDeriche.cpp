#include "stdafx.h"
#include "spatialfilter/SpatialFilter.hpp"

#define USE_FMA_DERICHE

//optimization float OK
//optimization double OK

using namespace std;
using namespace cv;

namespace cp
{
#define CV_SQRT2PI 2.50662827463100050241576528481104525

	template<typename Type>
	void computeDericheCoefficients(Type* dest_b, Type* dest_a, const complex<double>* alpha, const complex<double>* beta, const int order, const Type sigma)
	{
		const double denominator = sigma * CV_SQRT2PI;
		complex<double> b[DERICHE_ORDER_MAX], a[DERICHE_ORDER_MAX + 1];

		b[0] = alpha[0]; //Initialize b/a = alpha[0] / (1 + beta[0] z^-1)
		a[0] = complex<double>(1.0, 0.0);
		a[1] = beta[0];

		for (int k = 1; k < order; ++k)
		{
			//Add kth term, b/a += alpha[k] / (1 + beta[k] z^-1)
			b[k] = beta[k] * b[k - 1];

			for (int j = k - 1; j > 0; --j)
			{
				b[j] += beta[k] * b[j - 1];
			}

			for (int j = 0; j <= k; ++j)
			{
				b[j] += alpha[k] * a[j];
			}

			a[k + 1] = beta[k] * a[k];

			for (int j = k; j > 0; --j)
			{
				a[j] += beta[k] * a[j - 1];
			}
		}

		for (int k = 0; k < order; ++k)
		{
			dest_b[k] = (Type)(b[k].real() / denominator);
			dest_a[k + 1] = (Type)a[k + 1].real();
		}

		return;
	}

#pragma region DERICHE_Naive
	template<typename Type>
	GaussianFilterDERICHE<Type>::GaussianFilterDERICHE(cv::Size imgSize, Type sigma, int order)
		: SpatialFilterBase(imgSize, cp::typeToCVDepth<Type>()), order(order), sigma(sigma)
	{
		truncate_r = (int)ceil(4.0 * sigma);

		if (inter.depth() != CV_32F || inter.size() != imgSize)
		{
			inter = Mat::zeros(imgSize, depth);
			buf = Mat::zeros(imgSize, depth);
		}

		fh = new Type[truncate_r + order];
		bh = new Type[truncate_r + order];

		complex<double> beta[DERICHE_ORDER_MAX];

		//optimized filter parameters for Deriche's IIR filter
		const complex<double> alpha[DERICHE_ORDER_MAX - DERICHE_ORDER_MIN + 1][4] =
		{
			{ {  0.48145, 0.9710 },{  0.48145, -0.9710 } },
			{ { -0.44645, 0.5105 },{ -0.44645, -0.5105 },{  1.898  ,  0.0000 } },
			{ {  0.84000, 1.8675 },{  0.84000, -1.8675 },{ -0.34015, -0.1299 },{ -0.34015, 0.1299 } }
		};
		const complex<double> lambda[DERICHE_ORDER_MAX - DERICHE_ORDER_MIN + 1][4] =
		{
			{ { 1.260, 0.8448 },{ 1.260, -0.8448 } },
			{ { 1.512, 1.4750 },{ 1.512, -1.4750 },{ 1.556,	0.000 } },
			{ { 1.783, 0.6318 },{ 1.783, -0.6318 },{ 1.723,	1.997 },{ 1.723, -1.997 } }
		};

		for (int i = 0; i < order; ++i)
		{
			double temp = exp(-lambda[order - DERICHE_ORDER_MIN][i].real() / sigma);
			beta[i] = complex<double>(
				-temp * cos(lambda[order - DERICHE_ORDER_MIN][i].imag() / sigma),
				temp * sin(lambda[order - DERICHE_ORDER_MIN][i].imag() / sigma));
		}

		//Compute causal filter coefficients
		computeDericheCoefficients<Type>(fb, a, alpha[order - DERICHE_ORDER_MIN], beta, order, sigma);

		//compute anti-causal
		bb[0] = (Type)(0.0);
		for (int i = 1; i < order; ++i)
		{
			bb[i] = fb[i] - a[i] * fb[0];
		}
		bb[order] = -a[order] * fb[0];

		for (int n = 0; n < truncate_r + order; ++n)
		{
			fh[n] = (n < order) ? fb[n] : 0;

			for (int m = 1; m <= order && m <= n; ++m)
			{
				fh[n] -= a[m] * fh[n - m];
			}

			bh[n] = (n <= order) ? bb[n] : 0;

			for (int m = 1; m <= order && m <= n; ++m)
			{
				bh[n] -= a[m] * bh[n - m];
			}
		}
	}

	template<typename Type>
	GaussianFilterDERICHE<Type>::~GaussianFilterDERICHE()
	{
		delete[] fh;
		fh = nullptr;
		delete[] bh;
		bh = nullptr;
	}

	template<typename Type>
	void GaussianFilterDERICHE<Type>::horizontalFilter(const cv::Mat& src, cv::Mat& dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		for (int y = 0; y < height; ++y)
		{
			const Type* srcPtr = src.ptr<Type>(y);
			Type* bufPtr = buf.ptr<Type>(y);
			Type* dstPtr = dst.ptr<Type>(y);

			//boundary processing
			for (int j = 0; j < order; ++j)
			{
				bufPtr[j] = (Type)0.0;
				for (int i = -j; i < truncate_r; ++i)
				{
					bufPtr[j] += fh[j + i] * srcPtr[ref_lborder(-i, borderType)];
				}
			}
			for (int i = 0; i < order; ++i)
			{
				dstPtr[i] = bufPtr[i];
			}

			switch (order)
			{
			case 2:
				for (int x = 2; x < width; ++x)
				{
					bufPtr[x] = fb[0] * srcPtr[x]
						+ fb[1] * srcPtr[x - 1]
						- a[1] * bufPtr[x - 1]
						- a[2] * bufPtr[x - 2];

					dstPtr[x] = bufPtr[x];
				}
				break;
			case 3:
				for (int x = 3; x < width; ++x)
				{
					bufPtr[x] = fb[0] * srcPtr[x]
						+ fb[1] * srcPtr[x - 1]
						+ fb[2] * srcPtr[x - 2]
						- a[1] * bufPtr[x - 1]
						- a[2] * bufPtr[x - 2]
						- a[3] * bufPtr[x - 3];

					dstPtr[x] = bufPtr[x];
				}

				break;
			case 4:
				for (int x = 4; x < width; ++x)
				{
					bufPtr[x] = fb[0] * srcPtr[x]
						+ fb[1] * srcPtr[x - 1]
						+ fb[2] * srcPtr[x - 2]
						+ fb[3] * srcPtr[x - 3]
						- a[1] * bufPtr[x - 1]
						- a[2] * bufPtr[x - 2]
						- a[3] * bufPtr[x - 3]
						- a[4] * bufPtr[x - 4];

					dstPtr[x] = bufPtr[x];
				}
				break;
			}
		}

		for (int y = height - 1; y >= 0; --y)
		{
			const Type* srcPtr = src.ptr<Type>(y);
			Type* bufPtr = buf.ptr<Type>(y);
			Type* dstPtr = dst.ptr<Type>(y);

			//boundary processing
			for (int j = 0; j < order; ++j)
			{
				bufPtr[width - 1 - j] = (Type)0.0;
				for (int i = -j; i < truncate_r; ++i)
				{
					bufPtr[width - 1 - j] += bh[j + i] * srcPtr[ref_rborder(width - 1 + i, width, borderType)];
				}
			}
			for (int i = 0; i < order; ++i)
			{
				dstPtr[width - 1 - i] += bufPtr[width - 1 - i];
			}

			switch (order)
			{
			case 2:
				for (int x = width - 3; x >= 0; --x)
				{
					bufPtr[x] = bb[1] * srcPtr[x + 1]
						+ bb[2] * srcPtr[x + 2]
						- a[1] * bufPtr[x + 1]
						- a[2] * bufPtr[x + 2];

					dstPtr[x] += bufPtr[x];
				}
				break;
			case 3:
				for (int x = width - 4; x >= 0; --x)
				{
					bufPtr[x] = bb[1] * srcPtr[x + 1]
						+ bb[2] * srcPtr[x + 2]
						+ bb[3] * srcPtr[x + 3]
						- a[1] * bufPtr[x + 1]
						- a[2] * bufPtr[x + 2]
						- a[3] * bufPtr[x + 3];

					dstPtr[x] += bufPtr[x];
				}
				break;
			case 4:
				for (int x = width - 5; x >= 0; --x)
				{
					bufPtr[x] = bb[1] * srcPtr[x + 1]
						+ bb[2] * srcPtr[x + 2]
						+ bb[3] * srcPtr[x + 3]
						+ bb[4] * srcPtr[x + 4]
						- a[1] * bufPtr[x + 1]
						- a[2] * bufPtr[x + 2]
						- a[3] * bufPtr[x + 3]
						- a[4] * bufPtr[x + 4];

					dstPtr[x] += bufPtr[x];
				}
				break;
			}
		}
	}

	template<typename Type>
	void GaussianFilterDERICHE<Type>::verticalFilter(const cv::Mat& src, cv::Mat& dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		int offset[DERICHE_ORDER_MAX + 1];
		for (int i = 0; i <= order; ++i)
		{
			offset[i] = i * width;
		}

		const Type* srcPtr = src.ptr<Type>();
		Type* bufPtr = buf.ptr<Type>();
		Type* dstPtr = dst.ptr<Type>();
		for (int x = 0; x < width; ++x)
		{
			//boundary processing
			for (int j = 0; j < order; ++j)
			{
				bufPtr[offset[j] + x] = (Type)0.0;
				for (int i = -j; i < truncate_r; ++i)
				{
					bufPtr[offset[j] + x] += fh[j + i] * srcPtr[ref_tborder(-i, width, borderType) + x];
				}
			}
			for (int i = 0; i < order; ++i)
			{
				dstPtr[offset[i] + x] = bufPtr[offset[i] + x];
			}
		}

		switch (order)
		{
		case 2:
			for (int y = 2; y < height; ++y)
			{
				srcPtr = src.ptr<Type>(y);
				bufPtr = buf.ptr<Type>(y);
				dstPtr = dst.ptr<Type>(y);
				for (int x = 0; x < width; ++x)
				{
					bufPtr[x] = fb[0] * srcPtr[x]
						+ fb[1] * srcPtr[x - offset[1]]
						- a[1] * bufPtr[x - offset[1]]
						- a[2] * bufPtr[x - offset[2]];

					dstPtr[x] = bufPtr[x];
				}
			}
			break;
		case 3:
			for (int y = 3; y < height; ++y)
			{
				srcPtr = src.ptr<Type>(y);
				bufPtr = buf.ptr<Type>(y);
				dstPtr = dst.ptr<Type>(y);
				for (int x = 0; x < width; ++x)
				{
					bufPtr[x] = fb[0] * srcPtr[x]
						+ fb[1] * srcPtr[x - offset[1]]
						+ fb[2] * srcPtr[x - offset[2]]
						- a[1] * bufPtr[x - offset[1]]
						- a[2] * bufPtr[x - offset[2]]
						- a[3] * bufPtr[x - offset[3]];

					dstPtr[x] = bufPtr[x];
				}
			}
			break;
		case 4:
			for (int y = 4; y < height; ++y)
			{
				srcPtr = src.ptr<Type>(y);
				bufPtr = buf.ptr<Type>(y);
				dstPtr = dst.ptr<Type>(y);
				for (int x = 0; x < width; ++x)
				{
					bufPtr[x] = fb[0] * srcPtr[x]
						+ fb[1] * srcPtr[x - offset[1]]
						+ fb[2] * srcPtr[x - offset[2]]
						+ fb[3] * srcPtr[x - offset[3]]
						- a[1] * bufPtr[x - offset[1]]
						- a[2] * bufPtr[x - offset[2]]
						- a[3] * bufPtr[x - offset[3]]
						- a[4] * bufPtr[x - offset[4]];

					dstPtr[x] = bufPtr[x];
				}
			}
			break;
		}

		srcPtr = src.ptr<Type>();
		bufPtr = buf.ptr<Type>();
		dstPtr = dst.ptr<Type>();
		for (int x = width - 1; x >= 0; --x)
		{
			//boundary processing
			for (int j = 0; j < order; ++j)
			{
				bufPtr[width * (height - 1 - j) + x] = 0;
				for (int i = -j; i < truncate_r; ++i)
				{
					bufPtr[width * (height - 1 - j) + x] += bh[j + i] * srcPtr[ref_bborder(height - 1 + i, width, height, borderType) + x];
				}
			}
			for (int i = 0; i < order; ++i)
			{
				dstPtr[width * (height - 1 - i) + x] += bufPtr[width * (height - 1 - i) + x];
			}
		}

		switch (order)
		{
		case 2:
			for (int y = height - 3; y >= 0; --y)
			{
				srcPtr = src.ptr<Type>(y);
				bufPtr = buf.ptr<Type>(y);
				dstPtr = dst.ptr<Type>(y);
				for (int x = width - 1; x >= 0; --x)
				{
					bufPtr[x] = bb[1] * srcPtr[x + offset[1]]
						+ bb[2] * srcPtr[x + offset[2]]
						- a[1] * bufPtr[x + offset[1]]
						- a[2] * bufPtr[x + offset[2]];

					dstPtr[x] += bufPtr[x];
				}
			}
			break;
		case 3:
			for (int y = height - 4; y >= 0; --y)
			{
				srcPtr = src.ptr<Type>(y);
				bufPtr = buf.ptr<Type>(y);
				dstPtr = dst.ptr<Type>(y);
				for (int x = width - 1; x >= 0; --x)
				{
					bufPtr[x] = bb[1] * srcPtr[x + offset[1]]
						+ bb[2] * srcPtr[x + offset[2]]
						+ bb[3] * srcPtr[x + offset[3]]
						- a[1] * bufPtr[x + offset[1]]
						- a[2] * bufPtr[x + offset[2]]
						- a[3] * bufPtr[x + offset[3]];

					dstPtr[x] += bufPtr[x];
				}
			}
			break;
		case 4:
			for (int y = height - 5; y >= 0; --y)
			{
				srcPtr = src.ptr<Type>(y);
				bufPtr = buf.ptr<Type>(y);
				dstPtr = dst.ptr<Type>(y);
				for (int x = width - 1; x >= 0; --x)
				{
					bufPtr[x] = bb[1] * srcPtr[x + offset[1]]
						+ bb[2] * srcPtr[x + offset[2]]
						+ bb[3] * srcPtr[x + offset[3]]
						+ bb[4] * srcPtr[x + offset[4]]
						- a[1] * bufPtr[x + offset[1]]
						- a[2] * bufPtr[x + offset[2]]
						- a[3] * bufPtr[x + offset[3]]
						- a[4] * bufPtr[x + offset[4]];

					dstPtr[x] += bufPtr[x];
				}
			}
			break;
		}
	}

	template<class Type>
	void GaussianFilterDERICHE<Type>::body(const cv::Mat& _src, cv::Mat& dst, const int borderType)
	{
		this->borderType = borderType;

		Mat src;
		if (_src.depth() == depth)
			src = _src;
		else
			_src.convertTo(src, depth);

		if (dst.size() != imgSize || dst.depth() != depth)
			dst.create(imgSize, depth);

		horizontalFilter(src, inter);
		verticalFilter(inter, dst);
	}

	template class GaussianFilterDERICHE<float>;
	template class GaussianFilterDERICHE<double>;

#pragma endregion


#pragma region DERICHE_32F_AVX

	void GaussianFilterDERICHE_AVX_32F::allocBuffer()
	{
		truncate_r = (int)ceil(4.0 * sigma);

		if (inter.depth() != CV_32F || inter.size() != imgSize)
			inter.create(imgSize, depth);

		//inter = Mat::zeros(imgSize, depth);

		buf = (__m256**)_mm_malloc((gf_order + 1) * sizeof(__m256*), AVX_ALIGNMENT);
		for (int i = 0; i <= gf_order; ++i)
		{
			buf[i] = (__m256*)_mm_malloc(imgSize.width / 8 * sizeof(__m256), AVX_ALIGNMENT);
		}

		_mm_free(fh);
		_mm_free(bh);
		fh = (float*)_mm_malloc((truncate_r + gf_order) * sizeof(float), AVX_ALIGNMENT);
		bh = (float*)_mm_malloc((truncate_r + gf_order) * sizeof(float), AVX_ALIGNMENT);

		complex<double> beta[DERICHE_ORDER_MAX];

		//optimized filter parameters for Deriche's IIR filter
		const complex<double> alpha[DERICHE_ORDER_MAX - DERICHE_ORDER_MIN + 1][4] =
		{
			{ {  0.48145, 0.9710 },{  0.48145, -0.9710 } },
			{ { -0.44645, 0.5105 },{ -0.44645, -0.5105 },{  1.898  ,  0.0000 } },
			{ {  0.84000, 1.8675 },{  0.84000, -1.8675 },{ -0.34015, -0.1299 },{ -0.34015, 0.1299 } }
		};
		const complex<double> lambda[DERICHE_ORDER_MAX - DERICHE_ORDER_MIN + 1][4] =
		{
			{ { 1.260, 0.8448 },{ 1.260, -0.8448 } },
			{ { 1.512, 1.4750 },{ 1.512, -1.4750 },{ 1.556,	0.000 } },
			{ { 1.783, 0.6318 },{ 1.783, -0.6318 },{ 1.723,	1.997 },{ 1.723, -1.997 } }
		};

		for (int i = 0; i < gf_order; ++i)
		{
			double temp = exp(-lambda[gf_order - DERICHE_ORDER_MIN][i].real() / sigma);
			beta[i] = complex<double>(
				-temp * cos(lambda[gf_order - DERICHE_ORDER_MIN][i].imag() / sigma),
				temp * sin(lambda[gf_order - DERICHE_ORDER_MIN][i].imag() / sigma)
				);
		}

		//Compute causal filter coefficients
		computeDericheCoefficients<float>(fb, a, alpha[gf_order - DERICHE_ORDER_MIN], beta, gf_order, sigma);

		//compute anti-causal
		bb[0] = 0.f;
		for (int i = 1; i < gf_order; ++i)
		{
			bb[i] = fb[i] - a[i] * fb[0];
		}
		bb[gf_order] = -a[gf_order] * fb[0];

		for (int n = 0; n < truncate_r + gf_order; ++n)
		{
			fh[n] = (n < gf_order) ? fb[n] : 0.f;

			for (int m = 1; m <= gf_order && m <= n; ++m)
			{
				fh[n] -= a[m] * fh[n - m];
			}

			bh[n] = (n <= gf_order) ? bb[n] : 0.f;

			for (int m = 1; m <= gf_order && m <= n; ++m)
			{
				bh[n] -= a[m] * bh[n - m];
			}
		}
	}

	GaussianFilterDERICHE_AVX_32F::GaussianFilterDERICHE_AVX_32F(cv::Size imgSize, float sigma, int order)
		: SpatialFilterBase(imgSize, CV_32F)
	{
		this->gf_order = clipOrder(order, SpatialFilterAlgorithm::IIR_DERICHE);
		this->sigma = sigma;
		allocBuffer();
	}

	GaussianFilterDERICHE_AVX_32F::GaussianFilterDERICHE_AVX_32F(const int dest_depth)
	{
		this->dest_depth = dest_depth;
		this->depth = CV_32F;
	}

	GaussianFilterDERICHE_AVX_32F::~GaussianFilterDERICHE_AVX_32F()
	{
		for (int i = 0; i <= gf_order; ++i)
		{
			_mm_free(buf[i]);
		}
		_mm_free(buf);

		_mm_free(fh);
		_mm_free(bh);
	}

	void GaussianFilterDERICHE_AVX_32F::horizontalFilterVLoadSetTransposeStore(const cv::Mat& src, cv::Mat& dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		__m256 prev_input[DERICHE_ORDER_MAX];
		__m256 patch[8];
		__m256 patch_t[8];

		//forward direction
		for (int y = 0; y < height; y += 8)
		{
			const float* srcPtr = src.ptr<float>(y);
			float* dstPtr = dst.ptr<float>(y);

			//boundary processing
			for (int j = 0; j < gf_order; ++j)
			{
				patch[j] = _mm256_setzero_ps();
				for (int i = -j; i < truncate_r; ++i)
				{
					int refx = ref_lborder(-i, borderType);
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width + refx], srcPtr[6 * width + refx], srcPtr[5 * width + refx], srcPtr[4 * width + refx], srcPtr[3 * width + refx], srcPtr[2 * width + refx], srcPtr[width + refx], srcPtr[refx]);

#ifdef USE_FMA_DERICHE
					patch[j] = _mm256_fmadd_ps(_mm256_set1_ps(fh[j + i]), prev_input[0], patch[j]);
#else 
					patch[j] = _mm256_add_ps(patch[j], _mm256_mul_ps(_mm256_set1_ps(fh[j + i]), prev_input[0]));
#endif
				}
			}

			switch (gf_order)
			{
			case 2:
			{
				//fast 8 row
				++srcPtr;
				prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				++srcPtr;
				for (int i = gf_order; i < 8; ++i)
				{
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
								_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2]));
#endif
					++srcPtr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dstPtr, patch_t, width);
				dstPtr += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
								_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[6]));
#endif
					++srcPtr;

					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
								_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[7]));
#endif
					++srcPtr;

					for (int i = 2; i < 8; ++i)
					{
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
									_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
										_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
								_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2]));
#endif
						++srcPtr;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dstPtr, patch_t, width);
					dstPtr += 8;
				}
				break;
			}
			case 3:
			{
				//fast 8 row
				++srcPtr;
				prev_input[1] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				++srcPtr;
				prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				++srcPtr;
				for (int i = gf_order; i < 8; ++i)
				{
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i - 3],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
									_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i - 3]));
#endif
					++srcPtr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dstPtr, patch_t, width);
				dstPtr += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
									_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[5]));
#endif
					++srcPtr;

					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
									_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[6]));
#endif

					++srcPtr;

					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[2] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[1],
									_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[7]));
#endif
					++srcPtr;

					for (int i = 3; i < 8; ++i)
					{
						prev_input[2] = prev_input[1];
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i - 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
												_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
								_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2])),
								_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i - 3]));
#endif
						++srcPtr;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dstPtr, patch_t, width);
					dstPtr += 8;
				}
				break;
			}
			case 4:
			{
				//fast 8 row
				++srcPtr;
				prev_input[2] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				++srcPtr;
				prev_input[1] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				++srcPtr;
				prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				++srcPtr;
				for (int i = gf_order; i < 8; ++i)
				{
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[i - 4],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i - 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i - 3])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[i - 4]));
#endif
					++srcPtr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dstPtr, patch_t, width);
				dstPtr += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[4],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[5],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[4]));
#endif
					++srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[5]));
#endif
					++srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[2] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[1],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[6]));
#endif
					++srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[3] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[2],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[7]));
#endif
					++srcPtr;

					for (int i = 4; i < 8; ++i)
					{
						prev_input[3] = prev_input[2];
						prev_input[2] = prev_input[1];
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[i - 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i - 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
													_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
														_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
								_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
								_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2])),
								_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i - 3])),
								_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[i - 4]));
#endif

						++srcPtr;
					}
					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dstPtr, patch_t, width);
					dstPtr += 8;
				}
				break;
			}
			}
		}

		for (int y = height - 8; y >= 0; y -= 8)
		{
			const float* srcPtr = src.ptr<float>(y);
			float* dstPtr = dst.ptr<float>(y) + width - 8;

			//boundary processing
			for (int j = 0; j < gf_order; ++j)
			{
				patch[7 - j] = _mm256_setzero_ps();
				for (int i = -j; i < truncate_r; ++i)
				{
					int refx = ref_rborder(width - 1 + i, width, borderType);
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width + refx], srcPtr[6 * width + refx], srcPtr[5 * width + refx], srcPtr[4 * width + refx], srcPtr[3 * width + refx], srcPtr[2 * width + refx], srcPtr[width + refx], srcPtr[refx]);

#ifdef USE_FMA_DERICHE
					patch[7 - j] = _mm256_fmadd_ps(_mm256_set1_ps(bh[j + i]), prev_input[0], patch[7 - j]);
#else 
					patch[7 - j] = _mm256_add_ps(patch[7 - j], _mm256_mul_ps(_mm256_set1_ps(bh[j + i]), prev_input[0]));
#endif
				}
			}

			switch (gf_order)
			{
			case 2:
			{
				//last 8 row
				srcPtr += width - 1;
				prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				--srcPtr;
				for (int i = 5; i >= 0; --i)
				{
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
								_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2]));
#endif
					--srcPtr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_addstorepatch_ps(dstPtr, patch_t, width);
				dstPtr -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[7] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
								_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[1]));
#endif
					--srcPtr;

					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[6] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
								_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[0]));
#endif
					--srcPtr;

					for (int i = 5; i >= 0; --i)
					{
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
										_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
								_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2]));
#endif

						--srcPtr;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_addstorepatch_ps(dstPtr, patch_t, width);
					dstPtr -= 8;
				}
				break;
			}
			case 3:
			{
				//last 8 row
				srcPtr += width - 1;
				prev_input[1] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				--srcPtr;
				prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				--srcPtr;
				for (int i = 4; i >= 0; --i)
				{
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i + 3],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i + 3]));
#endif
					--srcPtr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_addstorepatch_ps(dstPtr, patch_t, width);
				dstPtr -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[7] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[2]));
#endif

					--srcPtr;

					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[6] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[1]));
#endif
					--srcPtr;

					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[5] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[6],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[0]));
#endif
					--srcPtr;

					for (int i = 4; i >= 0; --i)
					{
						prev_input[2] = prev_input[1];
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i + 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
												_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
								_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2])),
								_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i + 3]));
#endif

						--srcPtr;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_addstorepatch_ps(dstPtr, patch_t, width);
					dstPtr -= 8;
				}
				break;
			}
			case 4:
			{
				//last 8 row
				srcPtr += width - 1;
				prev_input[2] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				--srcPtr;
				prev_input[1] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				--srcPtr;
				prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);
				--srcPtr;
				for (int i = 3; i >= 0; --i)
				{
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[i + 4],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i + 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i + 3])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[i + 4]));
#endif				
					--srcPtr;
				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_addstorepatch_ps(dstPtr, patch_t, width);
				dstPtr -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[7] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[3],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[2],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[3]));
#endif
					--srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[6] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[2]));
#endif
					--srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[5] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[6],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[1]));
#endif
					--srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

					patch[4] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[5],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[0]));
#endif
					--srcPtr;

					for (int i = 3; i >= 0; --i)
					{
						prev_input[3] = prev_input[2];
						prev_input[2] = prev_input[1];
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_set_ps(srcPtr[7 * width], srcPtr[6 * width], srcPtr[5 * width], srcPtr[4 * width], srcPtr[3 * width], srcPtr[2 * width], srcPtr[width], srcPtr[0]);

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[i + 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i + 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
													_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
														_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
								_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2])),
								_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i + 3])),
								_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[i + 4]));
#endif
						--srcPtr;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_addstorepatch_ps(dstPtr, patch_t, width);
					dstPtr -= 8;
				}
				break;
			}
			default:
				break;
			}
		}
	}

	void GaussianFilterDERICHE_AVX_32F::horizontalFilterVLoadGatherTransposeStore(const cv::Mat& src, cv::Mat& dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		__m256 prev_input[DERICHE_ORDER_MAX];
		__m256 patch[8];
		__m256 patch_t[8];

		const __m256i mm_offset = _mm256_set_epi32(7 * width, 6 * width, 5 * width, 4 * width, 3 * width, 2 * width, width, 0);

		//forward direction
		for (int y = 0; y < height; y += 8)
		{
			const float* srcPtr = src.ptr<float>(y);
			float* dstPtr = dst.ptr<float>(y);

			//boundary processing
			for (int j = 0; j < gf_order; ++j)
			{
				patch[j] = _mm256_setzero_ps();
				for (int i = -j; i < truncate_r; ++i)
				{
					float* s = (float*)(srcPtr + ref_lborder(-i, borderType));
					prev_input[0] = _mm256_i32gather_ps(s, mm_offset, sizeof(float));
#ifdef USE_FMA_DERICHE
					patch[j] = _mm256_fmadd_ps(_mm256_set1_ps(fh[j + i]), prev_input[0], patch[j]);
#else 
					patch[j] = _mm256_add_ps(patch[j], _mm256_mul_ps(_mm256_set1_ps(fh[j + i]), prev_input[0]));
#endif
				}
			}

			switch (gf_order)
			{
			case 2:
			{
				//fast 8 row
				++srcPtr;
				prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));
				++srcPtr;
				for (int i = gf_order; i < 8; ++i)
				{
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
								_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2]));
#endif
					++srcPtr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dstPtr, patch_t, width);
				dstPtr += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));
					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
								_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[6]));
#endif
					++srcPtr;

					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
								_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[7]));
#endif
					++srcPtr;

					for (int i = 2; i < 8; ++i)
					{
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
									_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
										_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
								_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2]));
#endif
						++srcPtr;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dstPtr, patch_t, width);
					dstPtr += 8;
				}
				break;
			}
			case 3:
			{
				//fast 8 row
				++srcPtr;
				prev_input[1] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));
				++srcPtr;
				prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));
				++srcPtr;
				for (int i = gf_order; i < 8; ++i)
				{
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i - 3],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
									_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i - 3]));
#endif
					++srcPtr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dstPtr, patch_t, width);
				dstPtr += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
									_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[5]));
#endif
					++srcPtr;

					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
									_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[6]));
#endif

					++srcPtr;

					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[2] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[1],
									_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[7]));
#endif
					++srcPtr;

					for (int i = 3; i < 8; ++i)
					{
						prev_input[2] = prev_input[1];
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i - 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
												_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
								_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2])),
								_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i - 3]));
#endif
						++srcPtr;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dstPtr, patch_t, width);
					dstPtr += 8;
				}
				break;
			}
			case 4:
			{
				//fast 8 row
				++srcPtr;
				prev_input[2] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));;
				++srcPtr;
				prev_input[1] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));;
				++srcPtr;
				prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));;
				++srcPtr;
				for (int i = gf_order; i < 8; ++i)
				{
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));;

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[i - 4],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i - 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i - 3])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[i - 4]));
#endif
					++srcPtr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dstPtr, patch_t, width);
				dstPtr += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[4],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[5],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[4]));
#endif
					++srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[5]));
#endif
					++srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[2] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[1],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[6]));
#endif
					++srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[3] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[2],
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[7]));
#endif
					++srcPtr;

					for (int i = 4; i < 8; ++i)
					{
						prev_input[3] = prev_input[2];
						prev_input[2] = prev_input[1];
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[i - 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i - 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i - 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i - 1],
											_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), prev_input[3],
												_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), prev_input[2],
													_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), prev_input[1],
														_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]))))))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
								_mm256_mul_ps(_mm256_set1_ps(fb[0]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(fb[1]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(fb[2]), prev_input[2])),
								_mm256_mul_ps(_mm256_set1_ps(fb[3]), prev_input[3])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i - 2])),
								_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i - 3])),
								_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[i - 4]));
#endif

						++srcPtr;
					}
					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dstPtr, patch_t, width);
					dstPtr += 8;
				}
				break;
			}
			}
		}

		//backward direction
		for (int y = height - 8; y >= 0; y -= 8)
		{
			const float* srcPtr = src.ptr<float>(y);
			float* dstPtr = dst.ptr<float>(y) + width - 8;

			//boundary processing
			for (int j = 0; j < gf_order; ++j)
			{
				patch[7 - j] = _mm256_setzero_ps();
				for (int i = -j; i < truncate_r; ++i)
				{
					float* s = (float*)(srcPtr + ref_rborder(width - 1 + i, width, borderType));
					prev_input[0] = _mm256_i32gather_ps(s, mm_offset, sizeof(float));
#ifdef USE_FMA_DERICHE
					patch[7 - j] = _mm256_fmadd_ps(_mm256_set1_ps(bh[j + i]), prev_input[0], patch[7 - j]);
#else 
					patch[7 - j] = _mm256_add_ps(patch[7 - j], _mm256_mul_ps(_mm256_set1_ps(bh[j + i]), prev_input[0]));
#endif
				}
			}

			switch (gf_order)
			{
			case 2:
			{
				//last 8 row
				srcPtr += width - 1;
				prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));
				--srcPtr;
				for (int i = 5; i >= 0; --i)
				{
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
								_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2]));
#endif
					--srcPtr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_addstorepatch_ps(dstPtr, patch_t, width);
				dstPtr -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[7] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
								_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[1]));
#endif
					--srcPtr;

					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[6] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
								_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
									_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[0]));
#endif
					--srcPtr;

					for (int i = 5; i >= 0; --i)
					{
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
										_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
								_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2]));
#endif

						--srcPtr;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_addstorepatch_ps(dstPtr, patch_t, width);
					dstPtr -= 8;
				}
				break;
			}
			case 3:
			{
				//last 8 row
				srcPtr += width - 1;
				prev_input[1] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));
				--srcPtr;
				prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));
				--srcPtr;
				for (int i = 4; i >= 0; --i)
				{
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i + 3],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i + 3]));
#endif
					--srcPtr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_addstorepatch_ps(dstPtr, patch_t, width);
				dstPtr -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[7] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[2]));
#endif

					--srcPtr;

					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[6] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[1]));
#endif
					--srcPtr;

					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[5] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[6],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
											_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[0]));
#endif
					--srcPtr;

					for (int i = 4; i >= 0; --i)
					{
						prev_input[2] = prev_input[1];
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i + 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
												_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
								_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2])),
								_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i + 3]));
#endif

						--srcPtr;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_addstorepatch_ps(dstPtr, patch_t, width);
					dstPtr -= 8;
				}
				break;
			}
			case 4:
			{
				//last 8 row
				srcPtr += width - 1;
				prev_input[2] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));
				--srcPtr;
				prev_input[1] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));
				--srcPtr;
				prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));
				--srcPtr;
				for (int i = 3; i >= 0; --i)
				{
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[i] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[i + 4],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i + 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i + 3])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[i + 4]));
#endif				
					--srcPtr;
				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_addstorepatch_ps(dstPtr, patch_t, width);
				dstPtr -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[7] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[3],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[2],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[0],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[3]));
#endif
					--srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[6] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[7],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[2]));
#endif
					--srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[5] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[6],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[1]));
#endif
					--srcPtr;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

					patch[4] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[5],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[0]));
#endif
					--srcPtr;

					for (int i = 3; i >= 0; --i)
					{
						prev_input[3] = prev_input[2];
						prev_input[2] = prev_input[1];
						prev_input[1] = prev_input[0];
						prev_input[0] = _mm256_i32gather_ps(srcPtr, mm_offset, sizeof(float));

						patch[i] =
#ifdef USE_FMA_DERICHE
							_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), patch[i + 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), patch[i + 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), patch[i + 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), patch[i + 1],
											_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), prev_input[3],
												_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), prev_input[2],
													_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), prev_input[1],
														_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]))))))));
#else 
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(bb[1]), prev_input[0]),
								_mm256_mul_ps(_mm256_set1_ps(bb[2]), prev_input[1])),
								_mm256_mul_ps(_mm256_set1_ps(bb[3]), prev_input[2])),
								_mm256_mul_ps(_mm256_set1_ps(bb[4]), prev_input[3])),
								_mm256_mul_ps(_mm256_set1_ps(a[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(a[2]), patch[i + 2])),
								_mm256_mul_ps(_mm256_set1_ps(a[3]), patch[i + 3])),
								_mm256_mul_ps(_mm256_set1_ps(a[4]), patch[i + 4]));
#endif
						--srcPtr;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_addstorepatch_ps(dstPtr, patch_t, width);
					dstPtr -= 8;
				}
				break;
			}
			default:
				break;
			}
		}
	}

	void GaussianFilterDERICHE_AVX_32F::verticalFiler(const cv::Mat& src, cv::Mat& dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		__m256 accum[DERICHE_ORDER_MAX];
		int offset[DERICHE_ORDER_MAX + 1];

		for (int i = 0; i <= gf_order; ++i)
		{
			offset[i] = i * width;
		}

		//forward direction
		const float* srcPtr = src.ptr<float>(0);
		float* dstPtr = dst.ptr<float>(0);
		for (int x = 0; x < width; x += 8)
		{
			//boundary processing
			for (int j = 0; j < gf_order; ++j)
			{
				accum[j] = _mm256_setzero_ps();
				for (int i = -j; i < truncate_r; ++i)
				{
#ifdef USE_FMA_DERICHE
					
					accum[j] = _mm256_fmadd_ps(_mm256_set1_ps(fh[j + i]), *(__m256*) & srcPtr[ref_tborder(-i, width, borderType) + x], accum[j]);
#else 
					accum[j] = _mm256_add_ps(accum[j], _mm256_mul_ps(_mm256_set1_ps(fh[j + i]), *(__m256*) & srcPtr[ref_tborder(-i, width, borderType) + x]));
#endif
				}
			}
			for (int j = 0; j < gf_order; ++j)
			{
				*(__m256*)& dstPtr[offset[j] + x] = accum[j];
			}
		}

		switch (gf_order)
		{
		case 2:
			for (int y = 2; y < height; ++y)
			{
				srcPtr = src.ptr<float>(y);
				dstPtr = dst.ptr<float>(y);
				for (int x = 0; x < width; x += 8)
				{
					*(__m256*)(dstPtr + x) =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), *(__m256*)(dstPtr + x - offset[2]),
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), *(__m256*)(dstPtr + x - offset[1]),
								_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), *(__m256*)(srcPtr + x - offset[1]),
									_mm256_mul_ps(_mm256_set1_ps(fb[0]), *(__m256*)(srcPtr + x)))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), *(__m256*)(srcPtr + x)),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), *(__m256*)(srcPtr + x - offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), *(__m256*)(dstPtr + x - offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), *(__m256*)(dstPtr + x - offset[2])));
#endif

				}
			}
			break;

		case 3:
			for (int y = 3; y < height; ++y)
			{
				srcPtr = src.ptr<float>(y);
				dstPtr = dst.ptr<float>(y);
				for (int x = 0; x < width; x += 8)
				{
					*(__m256*)(dstPtr + x) =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), *(__m256*)(dstPtr + x - offset[3]),
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), *(__m256*)(dstPtr + x - offset[2]),
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), *(__m256*)(dstPtr + x - offset[1]),
									_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), *(__m256*)(srcPtr + x - offset[2]),
										_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), *(__m256*)(srcPtr + x - offset[1]),
											_mm256_mul_ps(_mm256_set1_ps(fb[0]), *(__m256*)(srcPtr + x)))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), *(__m256*)(srcPtr + x)),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), *(__m256*)(srcPtr + x - offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), *(__m256*)(srcPtr + x - offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), *(__m256*)(dstPtr + x - offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), *(__m256*)(dstPtr + x - offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), *(__m256*)(dstPtr + x - offset[3])));
#endif

				}
			}
			break;

		case 4:
			for (int y = 4; y < height; ++y)
			{
				srcPtr = src.ptr<float>(y);
				dstPtr = dst.ptr<float>(y);
				for (int x = 0; x < width; x += 8)
				{
					*(__m256*)(dstPtr + x) =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), *(__m256*)(dstPtr + x - offset[4]),
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), *(__m256*)(dstPtr + x - offset[3]),
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), *(__m256*)(dstPtr + x - offset[2]),
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), *(__m256*)(dstPtr + x - offset[1]),
										_mm256_fmadd_ps(_mm256_set1_ps(fb[3]), *(__m256*)(srcPtr + x - offset[3]),
											_mm256_fmadd_ps(_mm256_set1_ps(fb[2]), *(__m256*)(srcPtr + x - offset[2]),
												_mm256_fmadd_ps(_mm256_set1_ps(fb[1]), *(__m256*)(srcPtr + x - offset[1]),
													_mm256_mul_ps(_mm256_set1_ps(fb[0]), *(__m256*)(srcPtr + x)))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(fb[0]), *(__m256*)(srcPtr + x)),
							_mm256_mul_ps(_mm256_set1_ps(fb[1]), *(__m256*)(srcPtr + x - offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(fb[2]), *(__m256*)(srcPtr + x - offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(fb[3]), *(__m256*)(srcPtr + x - offset[3]))),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), *(__m256*)(dstPtr + x - offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), *(__m256*)(dstPtr + x - offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), *(__m256*)(dstPtr + x - offset[3]))),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), *(__m256*)(dstPtr + x - offset[4])));
#endif
				}
			}
			break;
		}

		//backward direction
		srcPtr = src.ptr<float>();
		dstPtr = dst.ptr<float>();
		for (int x = width - 8; x >= 0; x -= 8)
		{
			//boundary processing
			const int X = x >> 3;
			for (int j = 0; j < gf_order; ++j)
			{
				buf[gf_order - j][X] = _mm256_setzero_ps();
				for (int i = -j; i < truncate_r; ++i)
				{
#ifdef USE_FMA_DERICHE
					
					buf[gf_order - j][X] = _mm256_fmadd_ps(_mm256_set1_ps(bh[j + i]), *(__m256*) & srcPtr[ref_bborder(height - 1 + i, width, height, borderType) + x], buf[gf_order - j][X]);
#else 
					buf[order - j][X] = _mm256_add_ps(buf[order - j][X], _mm256_mul_ps(_mm256_set1_ps(bh[j + i]), *(__m256*) & srcPtr[ref_bborder(height - 1 + i, width, height, borderType) + x]));
#endif
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				*(__m256*)& dstPtr[width * (height - 1 - i) + x] = _mm256_add_ps(*(__m256*) & dstPtr[width * (height - 1 - i) + x], buf[gf_order - i][X]);
			}
		}

		switch (gf_order)
		{
		case 2:
			for (int y = height - 3; y >= 0; --y)
			{
				srcPtr = src.ptr<float>(y);
				dstPtr = dst.ptr<float>(y);
				for (int x = width - 8; x >= 0; x -= 8)
				{
					const int X = x >> 3;
					buf[0][X] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), buf[2][X],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), buf[1][X],
								_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), *(__m256*)(srcPtr + x + offset[2]),
									_mm256_mul_ps(_mm256_set1_ps(bb[1]), *(__m256*)(srcPtr + x + offset[1])))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(bb[1]), *(__m256*)(srcPtr + x + offset[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), *(__m256*)(srcPtr + x + offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), buf[1][X])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), buf[2][X]));
#endif
					* (__m256*)(dstPtr + x) = _mm256_add_ps(*(__m256*)(dstPtr + x), buf[0][X]);
				}
				__m256* tmp = buf[2];
				for (int i = 2; i > 0; --i)
				{
					buf[i] = buf[i - 1];
				}
				buf[0] = tmp;
			}
			break;
		case 3:
			for (int y = height - 4; y >= 0; --y)
			{
				srcPtr = src.ptr<float>(y);
				dstPtr = dst.ptr<float>(y);
				for (int x = width - 8; x >= 0; x -= 8)
				{
					const int X = x >> 3;
					buf[0][X] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), buf[3][X],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), buf[2][X],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), buf[1][X],
									_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), *(__m256*)(srcPtr + x + offset[3]),
										_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), *(__m256*)(srcPtr + x + offset[2]),
											_mm256_mul_ps(_mm256_set1_ps(bb[1]), *(__m256*)(srcPtr + x + offset[1])))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), *(__m256*)(srcPtr + x + offset[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), *(__m256*)(srcPtr + x + offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), *(__m256*)(srcPtr + x + offset[3]))),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), buf[1][X])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), buf[2][X])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), buf[3][X]));
#endif
					* (__m256*)(dstPtr + x) = _mm256_add_ps(*(__m256*)(dstPtr + x), buf[0][X]);
				}
				__m256* tmp = buf[3];
				for (int i = 3; i > 0; --i)
				{
					buf[i] = buf[i - 1];
				}
				buf[0] = tmp;
			}
			break;
		case 4:
			for (int y = height - 5; y >= 0; --y)
			{
				srcPtr = src.ptr<float>(y);
				dstPtr = dst.ptr<float>(y);
				for (int x = width - 8; x >= 0; x -= 8)
				{
					const int X = x >> 3;
					buf[0][X] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_ps(_mm256_set1_ps(a[4]), buf[4][X],
							_mm256_fnmadd_ps(_mm256_set1_ps(a[3]), buf[3][X],
								_mm256_fnmadd_ps(_mm256_set1_ps(a[2]), buf[2][X],
									_mm256_fnmadd_ps(_mm256_set1_ps(a[1]), buf[1][X],
										_mm256_fmadd_ps(_mm256_set1_ps(bb[4]), *(__m256*)(srcPtr + x + offset[4]),
											_mm256_fmadd_ps(_mm256_set1_ps(bb[3]), *(__m256*)(srcPtr + x + offset[3]),
												_mm256_fmadd_ps(_mm256_set1_ps(bb[2]), *(__m256*)(srcPtr + x + offset[2]),
													_mm256_mul_ps(_mm256_set1_ps(bb[1]), *(__m256*)(srcPtr + x + offset[1])))))))));
#else 
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(
							_mm256_mul_ps(_mm256_set1_ps(bb[1]), *(__m256*)(srcPtr + x + offset[1])),
							_mm256_mul_ps(_mm256_set1_ps(bb[2]), *(__m256*)(srcPtr + x + offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(bb[3]), *(__m256*)(srcPtr + x + offset[3]))),
							_mm256_mul_ps(_mm256_set1_ps(bb[4]), *(__m256*)(srcPtr + x + offset[4]))),
							_mm256_mul_ps(_mm256_set1_ps(a[1]), buf[1][X])),
							_mm256_mul_ps(_mm256_set1_ps(a[2]), buf[2][X])),
							_mm256_mul_ps(_mm256_set1_ps(a[3]), buf[3][X])),
							_mm256_mul_ps(_mm256_set1_ps(a[4]), buf[4][X]));
#endif
					* (__m256*)(dstPtr + x) = _mm256_add_ps(*(__m256*)(dstPtr + x), buf[0][X]);
				}
				__m256* tmp = buf[4];
				for (int i = 4; i > 0; --i)
				{
					buf[i] = buf[i - 1];
				}
				buf[0] = tmp;
			}
			break;
		}
	}

	void GaussianFilterDERICHE_AVX_32F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		this->borderType = borderType;

		CV_Assert(src.cols % 8 == 0);
		CV_Assert(src.rows % 8 == 0);
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);

		dst.create(imgSize, dest_depth);
		inter.create(imgSize, CV_32F);

		if (dest_depth == CV_32F)
		{
			if (src.depth() == CV_32F)
			{
				horizontalFilterVLoadGatherTransposeStore(src, inter);
				verticalFiler(inter, dst);
			}
			else
			{
				src.convertTo(inter2, CV_32F);
				horizontalFilterVLoadGatherTransposeStore(inter2, inter);
				verticalFiler(inter, dst);
			}
		}
		else
		{
			inter2.create(imgSize, CV_32F);
			if (src.depth() == CV_32F)
			{
				horizontalFilterVLoadGatherTransposeStore(src, inter);
				verticalFiler(inter, inter2);
				inter2.convertTo(dst, dest_depth);
			}
			else
			{
				src.convertTo(inter, CV_32F);
				horizontalFilterVLoadGatherTransposeStore(inter, inter2);
				verticalFiler(inter2, inter);
				inter.convertTo(dst, dest_depth);
			}
		}
	}

	void GaussianFilterDERICHE_AVX_32F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		int corder = clipOrder(order, SpatialFilterAlgorithm::IIR_DERICHE);
		
		if (this->sigma != sigma || this->gf_order != corder || this->imgSize != src.size())
		{
			this->sigma = sigma;
			this->gf_order = corder;
			this->imgSize = src.size();
			allocBuffer();
		}

		body(src, dst, borderType);
	}

#pragma endregion


#pragma region DERICHE_64F_AVX

	void GaussianFilterDERICHE_AVX_64F::allocBuffer()
	{
		truncate_r = (int)ceil(4.0 * sigma);

		if (inter.depth() != CV_64F || inter.size() != imgSize)
			inter.create(imgSize, depth);

		buf = (__m256d**)_mm_malloc((gf_order + 1) * sizeof(__m256d*), AVX_ALIGNMENT);
		for (int i = 0; i <= gf_order; ++i)
		{
			buf[i] = (__m256d*)_mm_malloc(imgSize.width / 4 * sizeof(__m256d), AVX_ALIGNMENT);
		}

		fh = (double*)_mm_malloc((truncate_r + gf_order) * sizeof(double), AVX_ALIGNMENT);
		bh = (double*)_mm_malloc((truncate_r + gf_order) * sizeof(double), AVX_ALIGNMENT);

		complex<double> beta[DERICHE_ORDER_MAX];

		//optimized filter parameters for Deriche's IIR filter
		const complex<double> alpha[DERICHE_ORDER_MAX - DERICHE_ORDER_MIN + 1][4] =
		{
			{ {  0.48145, 0.9710 },{  0.48145, -0.9710 } },
			{ { -0.44645, 0.5105 },{ -0.44645, -0.5105 },{  1.898  ,  0.0000 } },
			{ {  0.84000, 1.8675 },{  0.84000, -1.8675 },{ -0.34015, -0.1299 },{ -0.34015, 0.1299 } }
		};
		const complex<double> lambda[DERICHE_ORDER_MAX - DERICHE_ORDER_MIN + 1][4] =
		{
			{ { 1.260, 0.8448 },{ 1.260, -0.8448 } },
			{ { 1.512, 1.4750 },{ 1.512, -1.4750 },{ 1.556,	0.000 } },
			{ { 1.783, 0.6318 },{ 1.783, -0.6318 },{ 1.723,	1.997 },{ 1.723, -1.997 } }
		};

		for (int i = 0; i < gf_order; ++i)
		{
			double temp = exp(-lambda[gf_order - DERICHE_ORDER_MIN][i].real() / sigma);
			beta[i] = complex<double>(
				-temp * cos(lambda[gf_order - DERICHE_ORDER_MIN][i].imag() / sigma),
				temp * sin(lambda[gf_order - DERICHE_ORDER_MIN][i].imag() / sigma));
		}

		//compute causal filter coefficients
		computeDericheCoefficients<double>(fb, a, alpha[gf_order - DERICHE_ORDER_MIN], beta, gf_order, sigma);

		//compute anti-causal
		bb[0] = 0;
		for (int i = 1; i < gf_order; ++i)
		{
			bb[i] = fb[i] - a[i] * fb[0];
		}
		bb[gf_order] = -a[gf_order] * fb[0];

		for (int n = 0; n < truncate_r + gf_order; ++n)
		{
			fh[n] = (n < gf_order) ? fb[n] : 0;

			for (int m = 1; m <= gf_order && m <= n; ++m)
			{
				fh[n] -= a[m] * fh[n - m];
			}

			bh[n] = (n <= gf_order) ? bb[n] : 0;

			for (int m = 1; m <= gf_order && m <= n; ++m)
			{
				bh[n] -= a[m] * bh[n - m];
			}
		}
	}

	GaussianFilterDERICHE_AVX_64F::GaussianFilterDERICHE_AVX_64F(cv::Size imgSize, double sigma, int order)
		: SpatialFilterBase(imgSize, CV_64F)
	{
		this->gf_order = clipOrder(order, SpatialFilterAlgorithm::IIR_DERICHE);
		this->sigma = sigma;
		allocBuffer();
	}

	GaussianFilterDERICHE_AVX_64F::GaussianFilterDERICHE_AVX_64F(const int dest_depth)
	{
		this->dest_depth = dest_depth;
		this->depth = CV_64F;
	}

	GaussianFilterDERICHE_AVX_64F::~GaussianFilterDERICHE_AVX_64F()
	{
		for (int i = 0; i <= gf_order; ++i)
		{
			_mm_free(buf[i]);
		}
		_mm_free(buf);

		_mm_free(fh);
		_mm_free(bh);
	}

	void GaussianFilterDERICHE_AVX_64F::horizontalFilter(const cv::Mat& _src, cv::Mat& _dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		int i, j, x, y;
		__m256d prev_input[DERICHE_ORDER_MAX];
		__m256d patch[4];
		__m256d patch_t[4];

		//forward direction
		for (y = 0; y < height; y += 4)
		{
			const double* src = _src.ptr<double>(y);
			double* dst = _dst.ptr<double>(y);

			//boundary processing
			for (j = 0; j < gf_order; ++j)
			{
				for (patch[j] = _mm256_setzero_pd(), i = -j; i < truncate_r; ++i)
				{
					int refx = ref_lborder(-i, borderType);
					prev_input[0] = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);
#ifdef USE_FMA_DERICHE
					patch[j] = _mm256_fmadd_pd(_mm256_set1_pd(fh[j + i]), prev_input[0], patch[j]);
#else 
					patch[j] = _mm256_add_pd(patch[j], _mm256_mul_pd(_mm256_set1_pd(fh[j + i]), prev_input[0]));
#endif
				}
			}

			switch (gf_order)
			{
			case 2:
			{
				++src;
				prev_input[1] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				++src;
				prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				++src;

				patch[2] =
#ifdef USE_FMA_DERICHE
					_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[0],
						_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[1],
							_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
								_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))));
#else 
					_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
						_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
						_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
						_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[1])),
						_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[0]));
#endif
				prev_input[1] = prev_input[0];
				prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				++src;

				patch[3] =
#ifdef USE_FMA_DERICHE
					_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[1],
						_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[2],
							_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
								_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))));
#else 
					_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
						_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
						_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
						_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[2])),
						_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[1]));
#endif
				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst += 4;

				//IIR filtering
				for (x = 4; x < width; x += 4)
				{
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					++src;

					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[3],
								_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
									_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[2]));
#endif
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					++src;

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[0],
								_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
									_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[3]));
#endif
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					++src;

					patch[2] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[1],
								_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
									_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[0]));
#endif
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					++src;

					patch[3] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[2],
								_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
									_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[1]));
#endif
					_mm256_transpose4_pd(patch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst += 4;
				}
				break;
			}
			case 3:
			{
				++src;
				prev_input[2] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				++src;
				prev_input[1] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				++src;
				prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				++src;

				patch[3] =
#ifdef USE_FMA_DERICHE
					_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[0],
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[2],
								_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), prev_input[2],
									_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
										_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))))));
#else 
					_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
						_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
						_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
						_mm256_mul_pd(_mm256_set1_pd(fb[2]), prev_input[2])),
						_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[2])),
						_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[1])),
						_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[0]));
#endif
				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst += 4;

				//IIR filtering
				for (x = 4; x < width; x += 4)
				{
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					++src;

					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[3],
									_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), prev_input[2],
										_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
											_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))))));

#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(fb[2]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[1]));
#endif
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					++src;

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[0],
									_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), prev_input[2],
										_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
											_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(fb[2]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[2]));
#endif
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					++src;

					patch[2] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[1],
									_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), prev_input[2],
										_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
											_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(fb[2]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[3]));
#endif
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					++src;

					patch[3] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[2],
									_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), prev_input[2],
										_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
											_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(fb[2]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[0]));
#endif
					_mm256_transpose4_pd(patch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst += 4;
				}
				break;
			}
			case 4:
			{
				prev_input[3] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				++src;
				prev_input[2] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				++src;
				prev_input[1] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				++src;
				prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				++src;

				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst += 4;

				//IIR filtering
				for (x = 4; x < width; x += 4)
				{
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);

					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[4]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[2],
									_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[3],
										_mm256_fmadd_pd(_mm256_set1_pd(fb[3]), prev_input[3],
											_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), prev_input[2],
												_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
													_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(fb[2]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(fb[3]), prev_input[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[4]), patch[0]));
#endif
					++src;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[4]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[3],
									_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[0],
										_mm256_fmadd_pd(_mm256_set1_pd(fb[3]), prev_input[3],
											_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), prev_input[2],
												_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
													_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(fb[2]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(fb[3]), prev_input[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[4]), patch[1]));
#endif
					++src;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);

					patch[2] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[4]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[0],
									_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[1],
										_mm256_fmadd_pd(_mm256_set1_pd(fb[3]), prev_input[3],
											_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), prev_input[2],
												_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
													_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(fb[2]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(fb[3]), prev_input[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[4]), patch[2]));
#endif
					++src;

					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);

					patch[3] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[4]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[1],
									_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[2],
										_mm256_fmadd_pd(_mm256_set1_pd(fb[3]), prev_input[3],
											_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), prev_input[2],
												_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), prev_input[1],
													_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]))))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(fb[2]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(fb[3]), prev_input[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[4]), patch[3]));
#endif
					++src;

					_mm256_transpose4_pd(patch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst += 4;
				}
				break;
			}
			}
		}

		//backward direction
		for (y = height - 4; y >= 0; y -= 4)
		{
			const double* src = _src.ptr<double>(y);
			double* dst = _dst.ptr<double>(y) + width - 4;

			//boundary processing
			for (j = 0; j < gf_order; ++j)
			{
				for (patch[3 - j] = _mm256_setzero_pd(), i = -j; i < truncate_r; ++i)
				{
					int refx = ref_rborder(width - 1 + i, width, borderType);
					prev_input[0] = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);
#ifdef USE_FMA_DERICHE
					patch[3 - j] = _mm256_fmadd_pd(_mm256_set1_pd(bh[j + i]), prev_input[0], patch[3 - j]);
#else 
					patch[3 - j] = _mm256_add_pd(patch[3 - j], _mm256_mul_pd(_mm256_set1_pd(bh[j + i]), prev_input[0]));
#endif
				}
			}

			switch (gf_order)
			{
			case 2:
			{
				//last 4
				src += width - 1;
				prev_input[1] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				--src;
				prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				--src;

				patch[1] =
#ifdef USE_FMA_DERICHE
					_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[3],
						_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[2],
							_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
								_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))));

#else 
					_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
						_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
						_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
						_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[2])),
						_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[3]));
#endif
				prev_input[1] = prev_input[0];
				prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				--src;

				patch[0] =
#ifdef USE_FMA_DERICHE
					_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[2],
						_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[1],
							_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
								_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))));
#else 
					_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
						_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
						_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
						_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[1])),
						_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[2]));
#endif
				_mm256_transpose4_pd(patch, patch_t);
				_mm256_addstorepatch_pd(dst, patch_t, width);
				dst -= 4;

				//IIR filtering
				for (x = width - 8; x >= 0; x -= 4)
				{
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[3] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[0],
								_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
									_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[1]));
#endif
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[2] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[3],
								_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
									_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[0]));
#endif
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[2],
								_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
									_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[3]));
#endif
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[1],
								_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
									_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[2]));
#endif
					_mm256_transpose4_pd(patch, patch_t);
					_mm256_addstorepatch_pd(dst, patch_t, width);
					dst -= 4;
				}
				break;
			}
			case 3:
			{
				//last 4
				src += width - 1;
				prev_input[2] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				--src;
				prev_input[1] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				--src;
				prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				--src;

				patch[0] =
#ifdef USE_FMA_DERICHE
					_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[3],
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[1],
								_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), prev_input[2],
									_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
										_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))))));
#else 
					_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
						_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
						_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
						_mm256_mul_pd(_mm256_set1_pd(bb[3]), prev_input[2])),
						_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[1])),
						_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[2])),
						_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[3]));
#endif
				_mm256_transpose4_pd(patch, patch_t);
				_mm256_addstorepatch_pd(dst, patch_t, width);
				dst -= 4;

				//IIR filtering
				for (x = width - 8; x >= 0; x -= 4)
				{
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[3] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[0],
									_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), prev_input[2],
										_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
											_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[3]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[2]));
#endif
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[2] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[3],
									_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), prev_input[2],
										_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
											_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[3]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[1]));
#endif
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[2],
									_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), prev_input[2],
										_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
											_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[3]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[0]));
#endif
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[1],
									_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), prev_input[2],
										_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
											_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[3]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[3]));
#endif
					_mm256_transpose4_pd(patch, patch_t);
					_mm256_addstorepatch_pd(dst, patch_t, width);
					dst -= 4;
				}
				break;
			}
			case 4:
			{
				//last 4 row
				src += width - 1;
				prev_input[2] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				--src;
				prev_input[1] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				--src;
				prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				--src;

				_mm256_transpose4_pd(patch, patch_t);
				_mm256_addstorepatch_pd(dst, patch_t, width);
				dst -= 4;

				//IIR filtering
				for (x = width - 8; x >= 0; x -= 4)
				{
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[3] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[4]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[1],
									_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[0],
										_mm256_fmadd_pd(_mm256_set1_pd(bb[4]), prev_input[3],
											_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), prev_input[2],
												_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
													_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[3]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(bb[4]), prev_input[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[4]), patch[3]));
#endif
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[2] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[4]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[0],
									_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[3],
										_mm256_fmadd_pd(_mm256_set1_pd(bb[4]), prev_input[3],
											_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), prev_input[2],
												_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
													_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[3]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(bb[4]), prev_input[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[4]), patch[2]));
#endif
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[1] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[4]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[3],
									_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[2],
										_mm256_fmadd_pd(_mm256_set1_pd(bb[4]), prev_input[3],
											_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), prev_input[2],
												_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
													_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[3]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(bb[4]), prev_input[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(a[4]), patch[1]));
#endif
					prev_input[3] = prev_input[2];
					prev_input[2] = prev_input[1];
					prev_input[1] = prev_input[0];
					prev_input[0] = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					--src;

					patch[0] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[4]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), patch[2],
									_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), patch[1],
										_mm256_fmadd_pd(_mm256_set1_pd(bb[4]), prev_input[3],
											_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), prev_input[2],
												_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), prev_input[1],
													_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]))))))));
#else 
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), prev_input[0]),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), prev_input[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[3]), prev_input[2])),
							_mm256_mul_pd(_mm256_set1_pd(bb[4]), prev_input[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(a[4]), patch[0]));
#endif
					_mm256_transpose4_pd(patch, patch_t);
					_mm256_addstorepatch_pd(dst, patch_t, width);
					dst -= 4;
				}
				break;
			}
			default:
				break;
			}
		}
	}

	void GaussianFilterDERICHE_AVX_64F::verticalFilter(const cv::Mat& src, cv::Mat& dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		__m256d accum[DERICHE_ORDER_MAX];
		int offset[DERICHE_ORDER_MAX + 1];

		for (int i = 0; i <= gf_order; ++i)
		{
			offset[i] = i * width;
		}

		const double* srcPtr = src.ptr<double>(0);
		double* dstPtr = dst.ptr<double>(0);

		//forward direction
		for (int x = 0; x < width; x += 4)
		{
			//boundary processing
			for (int j = 0; j < gf_order; ++j)
			{
				accum[j] = _mm256_setzero_pd();
				for (int i = -j; i < truncate_r; ++i)
				{
#ifdef USE_FMA_DERICHE
					accum[j] = _mm256_fmadd_pd(_mm256_set1_pd(fh[j + i]), *(__m256d*) & srcPtr[ref_tborder(-i, width, borderType) + x], accum[j]);
#else
					accum[j] = _mm256_add_pd(accum[j], _mm256_mul_pd(_mm256_set1_pd(fh[j + i]), *(__m256d*) & srcPtr[ref_tborder(-i, width, borderType) + x]));
#endif
				}
			}
			for (int j = 0; j < gf_order; ++j)
			{
				*(__m256d*)& dstPtr[offset[j] + x] = accum[j];
			}
		}

		switch (gf_order)
		{
		case 2:
			for (int y = 2; y < height; ++y)
			{
				srcPtr = src.ptr<double>(y);
				dstPtr = dst.ptr<double>(y);
				for (int x = 0; x < width; x += 4)
				{
					*(__m256d*)(dstPtr + x) =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), *(__m256d*)(dstPtr + x - offset[2]),
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), *(__m256d*)(dstPtr + x - offset[1]),
								_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), *(__m256d*)(srcPtr + x - offset[1]),
									_mm256_mul_pd(_mm256_set1_pd(fb[0]), *(__m256d*)(srcPtr + x)))));

#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), *(__m256d*)(srcPtr + x)),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), *(__m256d*)(srcPtr + x - offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), *(__m256d*)(dstPtr + x - offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), *(__m256d*)(dstPtr + x - offset[2])));
#endif
				}
			}
			break;
		case 3:
			for (int y = 3; y < height; ++y)
			{
				srcPtr = src.ptr<double>(y);
				dstPtr = dst.ptr<double>(y);
				for (int x = 0; x < width; x += 4)
				{
					*(__m256d*)(dstPtr + x) =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), *(__m256d*)(dstPtr + x - offset[3]),
							_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), *(__m256d*)(dstPtr + x - offset[2]),
								_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), *(__m256d*)(dstPtr + x - offset[1]),
									_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), *(__m256d*)(srcPtr + x - offset[2]),
										_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), *(__m256d*)(srcPtr + x - offset[1]),
											_mm256_mul_pd(_mm256_set1_pd(fb[0]), *(__m256d*)(srcPtr + x)))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), *(__m256d*)(srcPtr + x)),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), *(__m256d*)(srcPtr + x - offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(fb[2]), *(__m256d*)(srcPtr + x - offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), *(__m256d*)(dstPtr + x - offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), *(__m256d*)(dstPtr + x - offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), *(__m256d*)(dstPtr + x - offset[3])));
#endif
				}
			}
			break;
		case 4:
			for (int y = 4; y < height; ++y)
			{
				srcPtr = src.ptr<double>(y);
				dstPtr = dst.ptr<double>(y);
				for (int x = 0; x < width; x += 4)
				{
					*(__m256d*)(dstPtr + x) =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[4]), *(__m256d*)(dstPtr + x - offset[4]),
							_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), *(__m256d*)(dstPtr + x - offset[3]),
								_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), *(__m256d*)(dstPtr + x - offset[2]),
									_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), *(__m256d*)(dstPtr + x - offset[1]),
										_mm256_fmadd_pd(_mm256_set1_pd(fb[3]), *(__m256d*)(srcPtr + x - offset[3]),
											_mm256_fmadd_pd(_mm256_set1_pd(fb[2]), *(__m256d*)(srcPtr + x - offset[2]),
												_mm256_fmadd_pd(_mm256_set1_pd(fb[1]), *(__m256d*)(srcPtr + x - offset[1]),
													_mm256_mul_pd(_mm256_set1_pd(fb[0]), *(__m256d*)(srcPtr + x)))))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(fb[0]), *(__m256d*)(srcPtr + x)),
							_mm256_mul_pd(_mm256_set1_pd(fb[1]), *(__m256d*)(srcPtr + x - offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(fb[2]), *(__m256d*)(srcPtr + x - offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(fb[3]), *(__m256d*)(srcPtr + x - offset[3]))),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), *(__m256d*)(dstPtr + x - offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), *(__m256d*)(dstPtr + x - offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), *(__m256d*)(dstPtr + x - offset[3]))),
							_mm256_mul_pd(_mm256_set1_pd(a[4]), *(__m256d*)(dstPtr + x - offset[4])));
#endif					
				}
			}
			break;
		}

		srcPtr = src.ptr<double>(0);
		dstPtr = dst.ptr<double>(0);
		//backward direction
		for (int x = width - 4, vx = width / 4 - 1; x >= 0; x -= 4, --vx)
		{
			//boundary processing
			for (int j = 0; j < gf_order; ++j)
			{
				buf[gf_order - j][vx] = _mm256_setzero_pd();
				for (int i = -j; i < truncate_r; ++i)
				{
#ifdef USE_FMA_DERICHE
					buf[gf_order - j][vx] = _mm256_fmadd_pd(_mm256_set1_pd(bh[j + i]), *(__m256d*) & srcPtr[ref_bborder(height - 1 + i, width, height, borderType) + x], buf[gf_order - j][vx]);
#else
					buf[order - j][vx] = _mm256_add_pd(buf[order - j][vx], _mm256_mul_pd(_mm256_set1_pd(bh[j + i]), *(__m256d*) & srcPtr[ref_bborder(height - 1 + i, width, height, borderType) + x]));
#endif
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				*(__m256d*)& dstPtr[width * (height - 1 - i) + x] = _mm256_add_pd(*(__m256d*) & dstPtr[width * (height - 1 - i) + x], buf[gf_order - i][vx]);
			}
		}

		switch (gf_order)
		{
		case 2:
			for (int y = height - 3; y >= 0; --y)
			{
				srcPtr = src.ptr<double>(y);
				dstPtr = dst.ptr<double>(y);
				for (int x = width - 4, vx = width / 4 - 1; x >= 0; x -= 4, --vx)
				{
					buf[0][vx] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), buf[2][vx],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), buf[1][vx],
								_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), *(__m256d*)(srcPtr + x + offset[2]),
									_mm256_mul_pd(_mm256_set1_pd(bb[1]), *(__m256d*)(srcPtr + x + offset[1])))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), *(__m256d*)(srcPtr + x + offset[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), *(__m256d*)(srcPtr + x + offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), buf[1][vx])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), buf[2][vx]));
#endif
					* (__m256d*)(dstPtr + x) = _mm256_add_pd(*(__m256d*)(dstPtr + x), buf[0][vx]);
				}
				__m256d* tmp = buf[2];
				for (int i = 2; i > 0; --i)
				{
					buf[i] = buf[i - 1];
				}
				buf[0] = tmp;
			}
			break;
		case 3:
			for (int y = height - 4; y >= 0; --y)
			{
				srcPtr = src.ptr<double>(y);
				dstPtr = dst.ptr<double>(y);
				for (int x = width - 4, vx = width / 4 - 1; x >= 0; x -= 4, --vx)
				{
					buf[0][vx] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), buf[3][vx],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), buf[2][vx],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), buf[1][vx],
									_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), *(__m256d*)(srcPtr + x + offset[3]),
										_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), *(__m256d*)(srcPtr + x + offset[2]),
											_mm256_mul_pd(_mm256_set1_pd(bb[1]), *(__m256d*)(srcPtr + x + offset[1])))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), *(__m256d*)(srcPtr + x + offset[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), *(__m256d*)(srcPtr + x + offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(bb[3]), *(__m256d*)(srcPtr + x + offset[3]))),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), buf[1][vx])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), buf[2][vx])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), buf[3][vx]));
#endif
					* (__m256d*)(dstPtr + x) = _mm256_add_pd(*(__m256d*)(dstPtr + x), buf[0][vx]);
				}
				__m256d* tmp = buf[3];
				for (int i = 3; i > 0; --i)
				{
					buf[i] = buf[i - 1];
				}
				buf[0] = tmp;
			}
			break;
		case 4:
			for (int y = height - 5; y >= 0; --y)
			{
				srcPtr = src.ptr<double>(y);
				dstPtr = dst.ptr<double>(y);
				for (int x = width - 4, vx = width / 4 - 1; x >= 0; x -= 4, --vx)
				{
					buf[0][vx] =
#ifdef USE_FMA_DERICHE
						_mm256_fnmadd_pd(_mm256_set1_pd(a[4]), buf[4][vx],
							_mm256_fnmadd_pd(_mm256_set1_pd(a[3]), buf[3][vx],
								_mm256_fnmadd_pd(_mm256_set1_pd(a[2]), buf[2][vx],
									_mm256_fnmadd_pd(_mm256_set1_pd(a[1]), buf[1][vx],
										_mm256_fmadd_pd(_mm256_set1_pd(bb[4]), *(__m256d*)(srcPtr + x + offset[4]),
											_mm256_fmadd_pd(_mm256_set1_pd(bb[3]), *(__m256d*)(srcPtr + x + offset[3]),
												_mm256_fmadd_pd(_mm256_set1_pd(bb[2]), *(__m256d*)(srcPtr + x + offset[2]),
													_mm256_mul_pd(_mm256_set1_pd(bb[1]), *(__m256d*)(srcPtr + x + offset[1])))))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(
							_mm256_mul_pd(_mm256_set1_pd(bb[1]), *(__m256d*)(srcPtr + x + offset[1])),
							_mm256_mul_pd(_mm256_set1_pd(bb[2]), *(__m256d*)(srcPtr + x + offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(bb[3]), *(__m256d*)(srcPtr + x + offset[3]))),
							_mm256_mul_pd(_mm256_set1_pd(bb[4]), *(__m256d*)(srcPtr + x + offset[4]))),
							_mm256_mul_pd(_mm256_set1_pd(a[1]), buf[1][vx])),
							_mm256_mul_pd(_mm256_set1_pd(a[2]), buf[2][vx])),
							_mm256_mul_pd(_mm256_set1_pd(a[3]), buf[3][vx])),
							_mm256_mul_pd(_mm256_set1_pd(a[4]), buf[4][vx]));
#endif
					* (__m256d*)(dstPtr + x) = _mm256_add_pd(*(__m256d*)(dstPtr + x), buf[0][vx]);
				}
				__m256d* tmp = buf[4];
				for (int i = 4; i > 0; --i)
				{
					buf[i] = buf[i - 1];
				}
				buf[0] = tmp;
			}
			break;
		}
	}

	void GaussianFilterDERICHE_AVX_64F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		this->borderType = borderType;

		CV_Assert(src.cols % 4 == 0);
		CV_Assert(src.rows % 4 == 0);
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F || src.depth() == CV_64F);

		dst.create(imgSize, dest_depth);
		inter.create(imgSize, CV_64F);

		if (dest_depth == CV_64F)
		{
			if (src.depth() == CV_64F)
			{
				horizontalFilter(src, inter);
				verticalFilter(inter, dst);
			}
			else
			{
				src.convertTo(inter2, CV_64F);
				horizontalFilter(inter2, inter);
				verticalFilter(inter, dst);
			}
		}
		else
		{
			inter2.create(imgSize, CV_64F);
			if (src.depth() == CV_64F)
			{
				horizontalFilter(src, inter);
				verticalFilter(inter, inter2);
				inter2.convertTo(dst, dest_depth);
			}
			else
			{
				src.convertTo(inter, CV_64F);
				horizontalFilter(inter, inter2);
				verticalFilter(inter2, inter);
				inter.convertTo(dst, dest_depth);
			}
		}
	}

	void GaussianFilterDERICHE_AVX_64F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		int corder = clipOrder(order, SpatialFilterAlgorithm::IIR_DERICHE);
		if (this->sigma != sigma || this->gf_order != corder || this->imgSize != src.size())
		{
			this->sigma = sigma;
			this->gf_order = corder;
			this->imgSize = src.size();
			allocBuffer();
		}

		body(src, dst, borderType);
	}

#pragma endregion

}
