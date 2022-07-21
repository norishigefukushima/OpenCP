#include "multiscalefilter/MultiScaleFilter.hpp"
#include "multiscalefilter/FourierSeriesExpansion.h"
#include <search1D.hpp>

//#define AMD_OPTIMIZATION
#define USE_GATHER8U
#define MASKSTORE
//#define MASKSTORELASTLINEAR

#ifdef AMD_OPTIMIZATION
inline __m256 _mm256_i32gather_ps(float* s, __m256i index, int val)
{
	return _mm256_setr_ps(s[((int*)&index)[0]], s[((int*)&index)[1]], s[((int*)&index)[2]], s[((int*)&index)[3]], s[((int*)&index)[4]], s[((int*)&index)[5]], s[((int*)&index)[6]], s[((int*)&index)[7]]);
}

#define _mm256_permute4x64_ps(src,imm8) (_mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(src), imm8)))
//for _MM_SHUFFLE(3, 1, 2, 0) 
/*
inline __m256 _mm256_permute4x64_ps(__m256 src, const int imm8)
{
	//perm128x1, shuffle_pd(1,0.5) x2
	__m256 tmp = _mm256_permute2f128_ps(src, src, 0x01);
	__m256d tm2 = _mm256_shuffle_pd(_mm256_castps_pd(src), _mm256_castps_pd(tmp), 0b1100);
	__m256 ret = _mm256_castpd_ps(_mm256_shuffle_pd(tm2, tm2, 0b0110));
	return ret;
}
*/
/*
inline __m256 _mm256_permute4x64_ps(__m256 src, const int imm8)
{
	//perm128x1 shuffle_ps(1,0.5) x2, blend(1,0.33) x1
	__m256 tmp = _mm256_permute2f128_ps(src, src, 0x01);
	__m256 rt1 = _mm256_shuffle_ps(src, tmp, _MM_SHUFFLE(1, 0, 1, 0));
	__m256 rt2 = _mm256_shuffle_ps(tmp, src, _MM_SHUFFLE(3, 2, 3, 2));
	__m256 ret = _mm256_blend_ps(rt1, rt2, 0b11110000);
	return ret;
}*/
#else
#define _mm256_permute4x64_ps(src,imm8) (_mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(src), imm8)))
//#define _mm256_permute4x64_ps(src,imm8) (_mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(src), imm8)))
#endif 


//#define USE_SLIDING_DCT
#ifdef USE_SLIDING_DCT
#pragma comment(lib, "ConstantTimeBF.lib")
//#include <E:/Github/Sliding-DCT-Gaussian-Filtering/ConstantTimeBF/include/SpatialFilter.hpp>
#include <C:/Users/ckv14073/Desktop/lab\Project_folder/Sliding-DCT-Gaussian-Filtering/ConstantTimeBF/include/SpatialFilter.hpp>
#endif

using namespace cv;
using namespace std;

inline __m128i get_storemask1(const int width, const int simdwidth)
{
	__m128i ret;
	const int WIDTH = get_simd_floor(width, simdwidth);
	if ((width - WIDTH) == 2) ret = _mm_cmpeq_epi32(_mm_setr_epi32(0, 1, 1, 1), _mm_setzero_si128());
	if ((width - WIDTH) == 4) ret = _mm_cmpeq_epi32(_mm_setr_epi32(0, 0, 1, 1), _mm_setzero_si128());
	if ((width - WIDTH) == 6) ret = _mm_cmpeq_epi32(_mm_setr_epi32(0, 0, 0, 1), _mm_setzero_si128());
	return ret;
}

inline void get_storemask2(const int width, __m256i& maskl, __m256i& maskr, const int unit)
{
	const int WIDTH = get_simd_floor(width, unit);
	if ((width - WIDTH) == 1)
	{
		maskl = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 1, 1, 1, 1, 1, 1), _mm256_setzero_si256());
		maskr = _mm256_cmpeq_epi32(_mm256_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1), _mm256_setzero_si256());
	}
	if ((width - WIDTH) == 2)
	{
		maskl = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1), _mm256_setzero_si256());
		maskr = _mm256_cmpeq_epi32(_mm256_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1), _mm256_setzero_si256());
	}
	if ((width - WIDTH) == 3)
	{
		maskl = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 1, 1), _mm256_setzero_si256());
		maskr = _mm256_cmpeq_epi32(_mm256_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1), _mm256_setzero_si256());
	}
	if ((width - WIDTH) == 4)
	{
		maskl = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0), _mm256_setzero_si256());
		maskr = _mm256_cmpeq_epi32(_mm256_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1), _mm256_setzero_si256());
	}
	if ((width - WIDTH) == 5)
	{
		maskl = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0), _mm256_setzero_si256());
		maskr = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 1, 1, 1, 1, 1, 1), _mm256_setzero_si256());
	}
	if ((width - WIDTH) == 6)
	{
		maskl = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0), _mm256_setzero_si256());
		maskr = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1), _mm256_setzero_si256());
	}
	if ((width - WIDTH) == 7)
	{
		maskl = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0), _mm256_setzero_si256());
		maskr = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 1, 1), _mm256_setzero_si256());
	}
}

namespace cp
{
#pragma region MultiScaleFilter
	void MultiScaleFilter::initRangeTable(const float sigma, const float boost)
	{
		rangeTable.resize(rangeMax);

		for (int i = 0; i < rangeMax; i++)
		{
			rangeTable[i] = getGaussianRangeWeight(float(i), sigma, boost);
		}
	}

	void MultiScaleFilter::remap(const Mat& src, Mat& dest, const float g, const float sigma_range, const float boost)
	{
		if (src.data != dest.data) dest.create(src.size(), CV_32F);

		if (windowType == 0)
		{
			const float* s = src.ptr<float>();
			float* d = dest.ptr<float>();
			const int size = src.size().area();
			const int SIZE32 = get_simd_floor(size, 32);
			const int SIZE8 = get_simd_floor(size, 8);
			const __m256 mg = _mm256_set1_ps(g);
			float* rptr = &rangeTable[0];
			const float coeff = float(1.0 / (-2.0 * sigma_range * sigma_range));
			const __m256 mcoeff = _mm256_set1_ps(coeff);
			const __m256 mdetail = _mm256_set1_ps(boost);

			__m256 ms, subsg;
			for (int i = 0; i < SIZE32; i += 32)
			{
				ms = _mm256_loadu_ps(s + i);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

				ms = _mm256_loadu_ps(s + i + 8);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

				ms = _mm256_loadu_ps(s + i + 16);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

				ms = _mm256_loadu_ps(s + i + 24);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				//_mm256_storeu_ps(d + i, _mm256_fnmadd_ps(_mm256_i32gather_ps(rptr, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), subsg, ms));
			}
			for (int i = SIZE32; i < SIZE8; i += 8)
			{
				ms = _mm256_loadu_ps(s + i);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				//_mm256_storeu_ps(d + i, _mm256_fnmadd_ps(_mm256_i32gather_ps(rptr, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), subsg, ms));
			}

			for (int i = SIZE8; i < size; i++)
				//for (int i = 0; i < size; i++)
			{
				const float x = s[i] - g;
				//d[i] = s[i] - x * range[saturate_cast<uchar>(abs(x))];
				d[i] = x * boost * exp(x * x * coeff) + s[i];
			}
		}
		else
		{
			for (int y = 0; y < src.rows; y++)
			{
				const float* s = src.ptr<float>(y);
				float* d = dest.ptr<float>(y);
				for (int x = 0; x < src.cols; x++)
				{
					d[x] = getremapCoefficient(s[x], g, windowType, sigma_range, Salpha, Sbeta, sigma_range, boost);
				}
			}
		}
	}

	void MultiScaleFilter::remapAdaptive(const Mat& src, Mat& dest, const float g, const Mat& sigma_range, const Mat& boost)
	{
		if (src.data != dest.data) dest.create(src.size(), CV_32F);

		if (windowType == 0)
		{
			const float* sigmaptr = sigma_range.ptr<float>();
			const float* boostptr = boost.ptr<float>();

			const float* s = src.ptr<float>();
			float* d = dest.ptr<float>();
			const int size = src.size().area();
			const int SIZE32 = get_simd_floor(size, 32);
			const int SIZE8 = get_simd_floor(size, 8);
			const __m256 mg = _mm256_set1_ps(g);
			float* rptr = &rangeTable[0];

			for (int i = 0; i < SIZE32; i += 32)
			{
				__m256 msgma = _mm256_loadu_ps(sigmaptr + i);
				__m256 mcoeff = _mm256_rcpnr_ps(_mm256_mul_ps(_mm256_set1_ps(-2.f), _mm256_mul_ps(msgma, msgma)));
				__m256 mdetail = _mm256_loadu_ps(boostptr + i);
				__m256 ms = _mm256_loadu_ps(s + i);
				__m256 subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

				msgma = _mm256_loadu_ps(sigmaptr + i + 8);
				mcoeff = _mm256_rcpnr_ps(_mm256_mul_ps(_mm256_set1_ps(-2.f), _mm256_mul_ps(msgma, msgma)));
				mdetail = _mm256_loadu_ps(boostptr + i + 8);
				ms = _mm256_loadu_ps(s + i + 8);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

				msgma = _mm256_loadu_ps(sigmaptr + i + 16);
				mcoeff = _mm256_rcpnr_ps(_mm256_mul_ps(_mm256_set1_ps(-2.f), _mm256_mul_ps(msgma, msgma)));
				mdetail = _mm256_loadu_ps(boostptr + i + 16);
				ms = _mm256_loadu_ps(s + i + 16);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

				msgma = _mm256_loadu_ps(sigmaptr + i + 24);
				mcoeff = _mm256_rcpnr_ps(_mm256_mul_ps(_mm256_set1_ps(-2.f), _mm256_mul_ps(msgma, msgma)));
				mdetail = _mm256_loadu_ps(boostptr + i + 24);
				ms = _mm256_loadu_ps(s + i + 24);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
			}
			for (int i = SIZE32; i < SIZE8; i += 8)
			{
				const __m256 msgma = _mm256_loadu_ps(sigmaptr + i);
				const __m256 mcoeff = _mm256_rcpnr_ps(_mm256_mul_ps(_mm256_set1_ps(-2.f), _mm256_mul_ps(msgma, msgma)));
				const __m256 mdetail = _mm256_loadu_ps(boostptr + i);

				const __m256 ms = _mm256_loadu_ps(s + i);
				__m256 subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				//_mm256_storeu_ps(d + i, _mm256_fnmadd_ps(_mm256_i32gather_ps(rptr, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), subsg, ms));
			}
			for (int i = SIZE32; i < size; i++)
				//for (int i = 0; i < size; i++)
			{
				const float coeff = 1.f / (-2.f * sigmaptr[i] * sigmaptr[i]);
				const float boost = boostptr[i];
				const float x = s[i] - g;

				//d[i] = s[i] - x * range[saturate_cast<uchar>(abs(x))];
				d[i] = s[i] + x * boost * exp(x * x * coeff);
			}
		}
		else
		{
			/*for (int y = 0; y < src.rows; y++)
			{
				const float* s = src.ptr<float>(y);
				float* d = dest.ptr<float>(y);
				for (int x = 0; x < src.cols; x++)
				{
					d[x] = getremapCoefficient(s[x], g, windowType, sigma_range, Salpha, Sbeta, sigma_range, detail_param, detail_param);
				}
			}*/
		}
	}

	void MultiScaleFilter::rangeDescope(const Mat& src)
	{
		if (rangeDescopeMethod == RangeDescopeMethod::MINMAX)
		{
			double minv, maxv;
			cv::minMaxLoc(src, &minv, &maxv);
			intensityMin = float(minv);
			intensityMax = float(maxv);
			intensityRange = intensityMax - intensityMin;
		}
		else
		{
			intensityMin = 0.f;
			intensityMax = 255.f;
			intensityRange = intensityMax - intensityMin;
		}
	}

	float* MultiScaleFilter::generateWeight(int r, const float sigma, float& evenratio, float& oddratio)
	{
		const int D = 2 * r + 1;
		float* w = (float*)_mm_malloc(D * sizeof(float), AVX_ALIGN);
		const float coeff = float(-1.0 / (2.0 * sigma * sigma));
		float total = 0.f;
		for (int i = 0; i < D; i++)
		{
			int x = i - r;
			const float v = exp(x * x * coeff);
			total += v;
			w[i] = v;
		}
		for (int i = 0; i < D; i++)
		{
			w[i] /= total;
		}

		float even = 0.f;
		float odd = 0.f;
		for (int i = 0; i < D; i += 2)
		{
			even += w[i];
		}
		for (int i = 1; i < D; i += 2)
		{
			odd += w[i];
		}
		evenratio = 1.f / even;
		oddratio = 1.f / odd;
		return w;
	}

#pragma region pyramid up/down
	void MultiScaleFilter::allocSpaceWeight(const float sigma)
	{
		_mm_free(GaussWeight);
		radius = getGaussianRadius(sigma);
		GaussWeight = generateWeight(radius, sigma, evenratio, oddratio);
	}

	void MultiScaleFilter::freeSpaceWeight()
	{
		_mm_free(GaussWeight);
		GaussWeight = nullptr;
	}

	void MultiScaleFilter::GaussDownFull(const Mat& src, Mat& dest, const float sigma, const int borderType)
	{
		Mat temp;
		const int r = getGaussianRadius(sigma);
		GaussianBlur(src, temp, Size(2 * r + 1, 2 * r + 1), sigma, sigma, borderType);
		resize(temp, dest, Size(), 0.5, 0.5, INTER_NEAREST);
	}

	void MultiScaleFilter::GaussDown(const Mat& src, Mat& dest)
	{
		CV_Assert(src.depth() == CV_32F);
		const Size size = src.size();
		dest.create(size / 2, CV_32F);

		const int D = 2 * radius + 1;
		const int rs = radius >> 1;
		Mat im; cv::copyMakeBorder(src, im, radius, radius, radius, radius, borderType);
		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}

		const int width = src.cols;
		const int height = src.rows;

		const int linesize = im.cols;
		float* linebuff = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		memset(linebuff, 0, sizeof(float) * linesize);

		const float* sptr = im.ptr<float>();
		float* dptr = dest.ptr<float>();
		const int IMCOLS = get_simd_floor(im.cols, 8);
		const int WIDTH = get_simd_floor(width, 8);

		for (int j = 0; j < height; j += 2)
		{
			//v filter
			for (int i = 0; i < IMCOLS; i += 8)
			{
				const float* s = sptr + i;
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(s)); s += im.cols;
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(s), sum); s += im.cols;
				}
				_mm256_storeu_ps(linebuff + i, sum);
			}
			for (int i = IMCOLS; i < im.cols; i++)
			{
				const float* s = sptr + i;
				float sum = GaussWeight[0] * *s; s += im.cols;
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * *s; s += im.cols;
				}
				linebuff[i] = sum;
			}
			sptr += 2 * im.cols;

			//h filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(linebuff + i));
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(linebuff + i + k), sum);
				}
				sum = _mm256_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 0, 2, 0));
				sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum));
			}
			for (int i = WIDTH; i < width; i += 2)
			{
				float sum = GaussWeight[0] * linebuff[i];
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * linebuff[i + k];
				}
				dptr[i >> 1] = sum;
			}
			dptr += dest.cols;
		}

		_mm_free(linebuff);
		_mm_free(W);
	}

	template<int D>
	void MultiScaleFilter::GaussDown(const Mat& src, Mat& dest, float* linebuff)
	{
		CV_Assert(src.depth() == CV_32F);
		const Size size = src.size();
		dest.create(size / 2, CV_32F);

		//const int D = 2 * radius + 1;
		const int rs = radius >> 1;
		Mat im; cv::copyMakeBorder(src, im, radius, radius, radius, radius, borderType);
		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}

		const int width = src.cols;
		const int height = src.rows;

		const float* sptr = im.ptr<float>();
		float* dptr = dest.ptr<float>();
		const int IMCOLS = get_simd_floor(im.cols, 8);
		const int WIDTH = get_simd_floor(width, 8);

		for (int j = 0; j < height; j += 2)
		{
			//v filter
			for (int i = 0; i < IMCOLS; i += 8)
			{
				const float* s = sptr + i;
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(s)); s += im.cols;
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(s), sum); s += im.cols;
				}
				_mm256_storeu_ps(linebuff + i, sum);
			}
			for (int i = IMCOLS; i < im.cols; i++)
			{
				const float* s = sptr + i;
				float sum = GaussWeight[0] * *s; s += im.cols;
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * *s; s += im.cols;
				}
				linebuff[i] = sum;
			}
			sptr += 2 * im.cols;

			//h filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(linebuff + i));
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(linebuff + i + k), sum);
				}
				sum = _mm256_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 0, 2, 0));
				sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum));
			}
			for (int i = WIDTH; i < width; i += 2)
			{
				float sum = GaussWeight[0] * linebuff[i];
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * linebuff[i + k];
				}
				dptr[i >> 1] = sum;
			}
			dptr += dest.cols;
		}

		_mm_free(W);
	}

	void MultiScaleFilter::GaussDownIgnoreBoundary(const Mat& src, Mat& dest)
	{
		CV_Assert(src.depth() == CV_32F);
		const Size size = src.size();
		dest.create(size / 2, CV_32F);

		const int D = 2 * radius + 1;
		const int rs = radius >> 1;
		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}

		const int width = src.cols;
		const int height = src.rows;

		const int linesize = src.cols;
		float* linebuff = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		memset(linebuff, 0, sizeof(float) * linesize);

		const float* sptr = src.ptr<float>();
		float* dptr = dest.ptr<float>(rs, rs);
		const int hend = width - 2 * radius;
		const int vend = height - 2 * radius;
		const int WIDTH = get_simd_floor(width, 8);
		const int HEND = get_simd_floor(hend, 8);

		for (int j = 0; j < vend; j += 2)
		{
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* s = sptr + i;
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(s));
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(s), sum); s += width;
				}
				_mm256_storeu_ps(linebuff + i, sum);
			}
			for (int i = WIDTH; i < width; i++)
			{
				const float* s = sptr + i;
				float sum = GaussWeight[0] * *s;
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * *s;
					s += width;
				}
				linebuff[i] = sum;
			}
			sptr += 2 * width;

			//h filter
			for (int i = 0; i < HEND; i += 8)
			{
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(linebuff + i));
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(linebuff + i + k), sum);
				}
				sum = _mm256_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 0, 2, 0));
				sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum));
			}
			for (int i = HEND; i < hend; i += 2)
			{
				float sum = GaussWeight[0] * linebuff[i];
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * linebuff[i + k];
				}
				dptr[i >> 1] = sum;
			}
			dptr += dest.cols;
		}

		_mm_free(linebuff);
		_mm_free(W);
	}

	template<int D>
	void MultiScaleFilter::GaussDownIgnoreBoundary(const Mat& src, Mat& dest, float* linebuff)
	{
		CV_Assert(src.depth() == CV_32F);
		const Size size = src.size();
		dest.create(size / 2, CV_32F);

		//const int D = 2 * radius + 1;
		const int rs = radius >> 1;
		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}

		const int width = src.cols;
		const int height = src.rows;

		const int linesize = src.cols;
		//memset(linebuff, 0, sizeof(float) * linesize);

		const float* sptr = src.ptr<float>();
		float* dptr = dest.ptr<float>(rs, rs);
		const int hend = width - 2 * radius;
		const int vend = height - 2 * radius;
		const int WIDTH = get_simd_floor(width, 8);
		const int HEND = get_simd_floor(hend, 8);

		for (int j = 0; j < vend; j += 2)
		{
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* s = sptr + i;
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(s));
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(s), sum); s += width;
				}
				_mm256_storeu_ps(linebuff + i, sum);
			}
			for (int i = WIDTH; i < width; i++)
			{
				const float* s = sptr + i;
				float sum = GaussWeight[0] * *s;
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * *s;
					s += width;
				}
				linebuff[i] = sum;
			}
			sptr += 2 * width;

			//h filter
			for (int i = 0; i < HEND; i += 8)
			{
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(linebuff + i));
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(linebuff + i + k), sum);
				}
				sum = _mm256_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 0, 2, 0));
				sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum));
			}
			for (int i = HEND; i < hend; i += 2)
			{
				float sum = GaussWeight[0] * linebuff[i];
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * linebuff[i + k];
				}
				dptr[i >> 1] = sum;
			}
			dptr += dest.cols;
		}
		_mm_free(W);
	}

	void MultiScaleFilter::GaussUpFull(const Mat& src, Mat& dest, const float sigma, const int borderType)
	{
		dest = Mat::zeros(src.size() * 2, CV_32F);
		Mat wmap = Mat::zeros(src.size() * 2, CV_32F);
		const int r = getGaussianRadius(sigma);
		Size ksize = Size(2 * r + 1, 2 * r + 1);
		for (int j = 0; j < dest.rows; j += 2)
		{
			for (int i = 0; i < dest.cols; i += 2)
			{
				dest.at<float>(j, i) = src.at<float>(j >> 1, i >> 1);
				wmap.at<float>(j, i) = 1.f;
			}
		}
		cv::GaussianBlur(wmap, wmap, ksize, sigma, sigma, borderType);
		cv::GaussianBlur(dest, dest, ksize, sigma, sigma, borderType);
		divide(dest, wmap, dest);
	}

	void MultiScaleFilter::GaussUp(const Mat& src, Mat& dest)
	{
		CV_Assert(src.depth() == CV_32F);
		dest.create(src.size() * 2, src.type());

		__m256* GW = (__m256*)_mm_malloc(sizeof(__m256) * (2 * radius + 1), AVX_ALIGN);
		for (int i = 0; i < 2 * radius + 1; i++)
		{
			GW[i] = _mm256_set1_ps(GaussWeight[i]);
		}
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const int rs = radius >> 1;
		const int D = 2 * rs + 1;
		const int D2 = 2 * D;
		Mat im; cv::copyMakeBorder(src, im, rs, rs, rs, rs, borderType);
		const int step = im.cols;
		float* linee = (float*)_mm_malloc(sizeof(float) * im.cols, AVX_ALIGN);
		float* lineo = (float*)_mm_malloc(sizeof(float) * im.cols, AVX_ALIGN);
		const int IMCOLS = get_simd_floor(im.cols, 8);
		const int WIDTH = get_simd_floor(src.cols, 8);
		const __m256i mask = get_simd_residualmask_epi32(im.cols);
		for (int j = 0; j < dest.rows; j += 2)
		{
			float* sptr = im.ptr<float>(j >> 1);
			//v filter
			for (int i = 0; i < IMCOLS; i += 8)
			{
				float* si = sptr + i;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo, moddratio));
			}
			{
				float* si = sptr + IMCOLS;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_maskstore_ps(linee + IMCOLS, mask, _mm256_mul_ps(sume, mevenratio));
				_mm256_maskstore_ps(lineo + IMCOLS, mask, _mm256_mul_ps(sumo, moddratio));
			}
			/*for (int i = IMCOLS; i < im.cols; i++)
			{
				float* si = sptr + i;
				float sume = GaussWeight[0] * *si; si += step;
				float sumo = 0.f;
				for (int k = 2; k < D2; k += 2)
				{
					sume += GaussWeight[k] * *si;
					sumo += GaussWeight[k - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}*/

			// h filter
			float* deptr = dest.ptr<float>(j);
			float* doptr = dest.ptr<float>(j + 1);
			for (int i = 0; i < WIDTH; i += 8)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}
				_mm256_store_interleave_ps(deptr + 2 * i, _mm256_mul_ps(mevenratio, sumee), _mm256_mul_ps(moddratio, sumoe));
				_mm256_store_interleave_ps(doptr + 2 * i, _mm256_mul_ps(mevenratio, sumeo), _mm256_mul_ps(moddratio, sumoo));
			}
			for (int i = WIDTH; i < src.cols; i++)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				float sumee = GaussWeight[0] * *sie++;
				float sumoe = 0.f;
				float sumeo = GaussWeight[0] * *sio++;
				float sumoo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sumee += GaussWeight[K] * *sie;
					sumoe += GaussWeight[K - 1] * *sie++;
					sumeo += GaussWeight[K] * *sio;
					sumoo += GaussWeight[K - 1] * *sio++;
				}
				const int I = i << 1;
				deptr[I + 0] = sumee * evenratio;
				deptr[I + 1] = sumoe * oddratio;
				doptr[I + 0] = sumeo * evenratio;
				doptr[I + 1] = sumoo * oddratio;
			}
		}
		_mm_free(linee);
		_mm_free(lineo);
		_mm_free(GW);
	}

	void MultiScaleFilter::GaussUpIgnoreBoundary(const Mat& src, Mat& dest)
	{
		CV_Assert(src.depth() == CV_32F);
		dest.create(src.size() * 2, src.type());

		__m256* GW = (__m256*)_mm_malloc(sizeof(__m256) * (2 * radius + 1), AVX_ALIGN);
		for (int i = 0; i < 2 * radius + 1; i++)
		{
			GW[i] = _mm256_set1_ps(GaussWeight[i]);
		}
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const int rs = radius >> 1;
		const int D = 2 * rs + 1;
		const int D2 = 2 * D;

		const int step = src.cols;
		float* linee = (float*)_mm_malloc(sizeof(float) * src.cols, AVX_ALIGN);
		float* lineo = (float*)_mm_malloc(sizeof(float) * src.cols, AVX_ALIGN);
		const int hend = src.cols - 2 * rs;
		const int HEND = get_simd_floor(hend, 8);
		const int WIDTH = get_simd_floor(src.cols, 8);
		for (int j = radius; j < dest.rows - radius; j += 2)
		{
			const float* sptr = src.ptr<float>((j - radius) >> 1);
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* si = sptr + i;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo, moddratio));

			}
			for (int i = WIDTH; i < src.cols; i++)
			{
				const float* si = sptr + i;
				float sume = GaussWeight[0] * *si; si += step;
				float sumo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sume += GaussWeight[K] * *si;
					sumo += GaussWeight[K - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}

			// h filter
			float* deptr = dest.ptr<float>(j, radius);
			float* doptr = dest.ptr<float>(j + 1, radius);
			for (int i = 0; i < HEND; i += 8)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}
				_mm256_store_interleave_ps(deptr + 2 * i, _mm256_mul_ps(sumee, mevenratio), _mm256_mul_ps(sumoe, moddratio));
				_mm256_store_interleave_ps(doptr + 2 * i, _mm256_mul_ps(sumeo, mevenratio), _mm256_mul_ps(sumoo, moddratio));
			}
			for (int i = HEND; i < hend; i++)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				float sumee = GaussWeight[0] * *sie++;
				float sumoe = 0.f;
				float sumeo = GaussWeight[0] * *sio++;
				float sumoo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sumee += GaussWeight[K] * *sie;
					sumoe += GaussWeight[K - 1] * *sie++;
					sumeo += GaussWeight[K] * *sio;
					sumoo += GaussWeight[K - 1] * *sio++;
				}
				const int I = i << 1;
				deptr[I + 0] = sumee * evenratio;
				deptr[I + 1] = sumoe * oddratio;
				doptr[I + 0] = sumeo * evenratio;
				doptr[I + 1] = sumoo * oddratio;
			}
		}
		_mm_free(linee);
		_mm_free(lineo);
		_mm_free(GW);
	}

	template<bool isAdd>
	void MultiScaleFilter::GaussUpAddFull(const Mat& src, const cv::Mat& addsubsrc, Mat& dest, const float sigma, const int borderType)
	{
		Mat temp = addsubsrc.clone();
		GaussUpFull(src, dest, sigma, borderType);
		if (isAdd) add(temp, dest, dest);
		else subtract(temp, dest, dest);
		return;
	}

	template<bool isAdd>
	void MultiScaleFilter::GaussUpAdd(const Mat& src, const cv::Mat& addsubsrc, Mat& dest)
	{
		CV_Assert(src.depth() == CV_32F);

		dest.create(src.size() * 2, src.type());

		__m256* GW = (__m256*)_mm_malloc(sizeof(__m256) * (2 * radius + 1), AVX_ALIGN);
		for (int i = 0; i < 2 * radius + 1; i++)
		{
			GW[i] = _mm256_set1_ps(GaussWeight[i]);
		}
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const int r = radius >> 1;
		const int D = 2 * r + 1;
		const int D2 = 2 * D;
		Mat im; cv::copyMakeBorder(src, im, r, r, r, r, borderType);
		const int step = im.cols;
		float* linee = (float*)_mm_malloc(sizeof(float) * im.cols, AVX_ALIGN);
		float* lineo = (float*)_mm_malloc(sizeof(float) * im.cols, AVX_ALIGN);
		const int IMCOLS = get_simd_floor(im.cols, 8);
		const int WIDTH = get_simd_floor(src.cols, 8);
		for (int j = 0; j < dest.rows; j += 2)
		{
			float* sptr = im.ptr<float>(j >> 1);
			//v filter
			for (int i = 0; i < IMCOLS; i += 8)
			{
				float* si = sptr + i;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					sume = _mm256_fmadd_ps(GW[k], _mm256_loadu_ps(si), sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], _mm256_loadu_ps(si), sumo);
					si += step;
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo, moddratio));
			}
			for (int i = IMCOLS; i < im.cols; i++)
			{
				float* si = sptr + i;
				float sume = GaussWeight[0] * *si; si += step;
				float sumo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sume += GaussWeight[K] * *si;
					sumo += GaussWeight[K - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}

			// h filter
			float* deptr = dest.ptr<float>(j);
			float* doptr = dest.ptr<float>(j + 1);
			const float* daeptr = addsubsrc.ptr<float>(j);
			const float* daoptr = addsubsrc.ptr<float>(j + 1);
			for (int i = 0; i < WIDTH; i += 8)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}

				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);
				if constexpr (isAdd)
				{
					_mm256_storeu_ps((float*)(deptr + 2 * i + 0), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(deptr + 2 * i + 8), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)));
				}
				else
				{
					_mm256_storeu_ps((float*)(deptr + 2 * i + 0), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(deptr + 2 * i + 8), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)));
				}

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);
				if constexpr (isAdd)
				{
					_mm256_storeu_ps((float*)(doptr + 2 * i + 0), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(doptr + 2 * i + 8), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)));
				}
				else
				{
					_mm256_storeu_ps((float*)(doptr + 2 * i + 0), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(doptr + 2 * i + 8), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)));
				}
			}
			for (int i = WIDTH; i < src.cols; i++)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				float sumee = GaussWeight[0] * *sie++;
				float sumoe = 0.f;
				float sumeo = GaussWeight[0] * *sio++;
				float sumoo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sumee += GaussWeight[K] * *sie;
					sumoe += GaussWeight[K - 1] * *sie++;
					sumeo += GaussWeight[K] * *sio;
					sumoo += GaussWeight[K - 1] * *sio++;
				}
				const int I = i << 1;
				if constexpr (isAdd)
				{
					dest.at<float>(j + 0, I + 0) = addsubsrc.at<float>(j + 0, I + 0) + sumee * evenratio;
					dest.at<float>(j + 0, I + 1) = addsubsrc.at<float>(j + 0, I + 1) + sumoe * oddratio;
					dest.at<float>(j + 1, I + 0) = addsubsrc.at<float>(j + 1, I + 0) + sumeo * evenratio;
					dest.at<float>(j + 1, I + 1) = addsubsrc.at<float>(j + 1, I + 1) + sumoo * oddratio;
				}
				else
				{
					dest.at<float>(j + 0, I + 0) = addsubsrc.at<float>(j + 0, I + 0) - sumee * evenratio;
					dest.at<float>(j + 0, I + 1) = addsubsrc.at<float>(j + 0, I + 1) - sumoe * oddratio;
					dest.at<float>(j + 1, I + 0) = addsubsrc.at<float>(j + 1, I + 0) - sumeo * evenratio;
					dest.at<float>(j + 1, I + 1) = addsubsrc.at<float>(j + 1, I + 1) - sumoo * oddratio;
				}
			}
		}
		_mm_free(linee);
		_mm_free(lineo);
		_mm_free(GW);
	}

	template<bool isAdd, int D, int D2>
	void MultiScaleFilter::GaussUpAdd(const Mat& src, const cv::Mat& addsubsrc, Mat& dest)
	{
		CV_Assert(src.depth() == CV_32F);

		dest.create(src.size() * 2, src.type());

		__m256* GW = (__m256*)_mm_malloc(sizeof(__m256) * (2 * radius + 1), AVX_ALIGN);
		for (int i = 0; i < 2 * radius + 1; i++)
		{
			GW[i] = _mm256_set1_ps(GaussWeight[i]);
		}
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const int r = radius >> 1;
		//const int D = 2 * r + 1;
		//const int D2 = 2 * D;
		Mat im; cv::copyMakeBorder(src, im, r, r, r, r, borderType);
		const int step = im.cols;
		float* linee = (float*)_mm_malloc(sizeof(float) * im.cols, AVX_ALIGN);
		float* lineo = (float*)_mm_malloc(sizeof(float) * im.cols, AVX_ALIGN);
		const int IMCOLS = get_simd_floor(im.cols, 8);
		const int WIDTH = get_simd_floor(src.cols, 8);
		for (int j = 0; j < dest.rows; j += 2)
		{
			float* sptr = im.ptr<float>(j >> 1);
			//v filter
			for (int i = 0; i < IMCOLS; i += 8)
			{
				float* si = sptr + i;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					sume = _mm256_fmadd_ps(GW[k], _mm256_loadu_ps(si), sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], _mm256_loadu_ps(si), sumo);
					si += step;
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo, moddratio));
			}
			for (int i = IMCOLS; i < im.cols; i++)
			{
				float* si = sptr + i;
				float sume = GaussWeight[0] * *si; si += step;
				float sumo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sume += GaussWeight[K] * *si;
					sumo += GaussWeight[K - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}

			// h filter
			float* deptr = dest.ptr<float>(j);
			float* doptr = dest.ptr<float>(j + 1);
			const float* daeptr = addsubsrc.ptr<float>(j);
			const float* daoptr = addsubsrc.ptr<float>(j + 1);
			for (int i = 0; i < WIDTH; i += 8)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}

				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);
				if constexpr (isAdd)
				{
					_mm256_storeu_ps((float*)(deptr + 2 * i + 0), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(deptr + 2 * i + 8), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)));
				}
				else
				{
					_mm256_storeu_ps((float*)(deptr + 2 * i + 0), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(deptr + 2 * i + 8), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)));
				}

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);
				if constexpr (isAdd)
				{
					_mm256_storeu_ps((float*)(doptr + 2 * i + 0), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(doptr + 2 * i + 8), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)));
				}
				else
				{
					_mm256_storeu_ps((float*)(doptr + 2 * i + 0), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(doptr + 2 * i + 8), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)));
				}
			}
			for (int i = WIDTH; i < src.cols; i++)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				float sumee = GaussWeight[0] * *sie++;
				float sumoe = 0.f;
				float sumeo = GaussWeight[0] * *sio++;
				float sumoo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sumee += GaussWeight[K] * *sie;
					sumoe += GaussWeight[K - 1] * *sie++;
					sumeo += GaussWeight[K] * *sio;
					sumoo += GaussWeight[K - 1] * *sio++;
				}
				const int I = i << 1;
				if constexpr (isAdd)
				{
					dest.at<float>(j + 0, I + 0) = addsubsrc.at<float>(j + 0, I + 0) + sumee * evenratio;
					dest.at<float>(j + 0, I + 1) = addsubsrc.at<float>(j + 0, I + 1) + sumoe * oddratio;
					dest.at<float>(j + 1, I + 0) = addsubsrc.at<float>(j + 1, I + 0) + sumeo * evenratio;
					dest.at<float>(j + 1, I + 1) = addsubsrc.at<float>(j + 1, I + 1) + sumoo * oddratio;
				}
				else
				{
					dest.at<float>(j + 0, I + 0) = addsubsrc.at<float>(j + 0, I + 0) - sumee * evenratio;
					dest.at<float>(j + 0, I + 1) = addsubsrc.at<float>(j + 0, I + 1) - sumoe * oddratio;
					dest.at<float>(j + 1, I + 0) = addsubsrc.at<float>(j + 1, I + 0) - sumeo * evenratio;
					dest.at<float>(j + 1, I + 1) = addsubsrc.at<float>(j + 1, I + 1) - sumoo * oddratio;
				}
			}
		}
		_mm_free(linee);
		_mm_free(lineo);
		_mm_free(GW);
	}


	template<bool isAdd>
	void MultiScaleFilter::GaussUpAddIgnoreBoundary(const Mat& src, const cv::Mat& addsubsrc, Mat& dest)
	{
		CV_Assert(src.depth() == CV_32F);
		dest.create(src.size() * 2, src.type());

		__m256* GW = (__m256*)_mm_malloc(sizeof(__m256) * (2 * radius + 1), AVX_ALIGN);
		for (int i = 0; i < 2 * radius + 1; i++)
		{
			GW[i] = _mm256_set1_ps(GaussWeight[i]);
		}
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const int rs = radius >> 1;
		const int D2 = 2 * (2 * rs + 1);

		const int step = src.cols;
		float* linee = (float*)_mm_malloc(sizeof(float) * src.cols, AVX_ALIGN);
		float* lineo = (float*)_mm_malloc(sizeof(float) * src.cols, AVX_ALIGN);
		const int hend = src.cols - 2 * rs;
		const int HEND = get_simd_floor(hend, 8);
		const int WIDTH = get_simd_floor(src.cols, 8);
		for (int j = radius; j < dest.rows - radius; j += 2)
		{
			const float* sptr = src.ptr<float>((j - radius) >> 1);
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* si = sptr + i;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo, moddratio));

			}
			for (int i = WIDTH; i < src.cols; i++)
			{
				const float* si = sptr + i;
				float sume = GaussWeight[0] * *si; si += step;
				float sumo = 0.f;
				for (int k = 2; k < D2; k += 2)
				{
					sume += GaussWeight[k] * *si;
					sumo += GaussWeight[k - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}

			// h filter
			float* deptr = dest.ptr<float>(j, radius);
			float* doptr = dest.ptr<float>(j + 1, radius);
			const float* daeptr = addsubsrc.ptr<float>(j, radius);
			const float* daoptr = addsubsrc.ptr<float>(j + 1, radius);
			for (int i = 0; i < HEND; i += 8)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}
				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);
				if constexpr (isAdd)
				{
					_mm256_storeu_ps((float*)(deptr + 2 * i + 0), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(deptr + 2 * i + 8), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)));
				}
				else
				{
					_mm256_storeu_ps((float*)(deptr + 2 * i + 0), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(deptr + 2 * i + 8), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)));
				}

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);
				if constexpr (isAdd)
				{
					_mm256_storeu_ps((float*)(doptr + 2 * i + 0), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(doptr + 2 * i + 8), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)));
				}
				else
				{
					_mm256_storeu_ps((float*)(doptr + 2 * i + 0), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(doptr + 2 * i + 8), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)));
				}
			}
			for (int i = HEND; i < hend; i++)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				float sumee = GaussWeight[0] * *sie++;
				float sumoe = 0.f;
				float sumeo = GaussWeight[0] * *sio++;
				float sumoo = 0.f;
				for (int k = 2; k < D2; k += 2)
				{
					sumee += GaussWeight[k] * *sie;
					sumoe += GaussWeight[k - 1] * *sie++;
					sumeo += GaussWeight[k] * *sio;
					sumoo += GaussWeight[k - 1] * *sio++;
				}
				const int I = i << 1;
				if constexpr (isAdd)
				{
					deptr[I + 0] = daeptr[I + 0] + sumee * evenratio;
					deptr[I + 1] = daeptr[I + 1] + sumoe * oddratio;
					doptr[I + 0] = daoptr[I + 0] + sumeo * evenratio;
					doptr[I + 1] = daoptr[I + 1] + sumoo * oddratio;
				}
				else
				{

					deptr[I + 0] = daeptr[I + 0] - sumee * evenratio;
					deptr[I + 1] = daeptr[I + 1] - sumoe * oddratio;
					doptr[I + 0] = daoptr[I + 0] - sumeo * evenratio;
					doptr[I + 1] = daoptr[I + 1] - sumoo * oddratio;
				}
			}
		}
		_mm_free(linee);
		_mm_free(lineo);
		_mm_free(GW);
	}

	template<bool isAdd, int D2>
	void MultiScaleFilter::GaussUpAddIgnoreBoundary(const Mat& src, const cv::Mat& addsubsrc, Mat& dest, float* linee, float* lineo)
	{
		CV_Assert(src.depth() == CV_32F);
		dest.create(src.size() * 2, src.type());

		__m256* GW = (__m256*)_mm_malloc(sizeof(__m256) * (2 * radius + 1), AVX_ALIGN);
		for (int i = 0; i < 2 * radius + 1; i++)
		{
			GW[i] = _mm256_set1_ps(GaussWeight[i]);
		}
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const int rs = radius >> 1;
		//const int D = 2 * rs + 1;
		//const int D2 = 2 * D;

		const int step = src.cols;
		//float* linee = (float*)_mm_malloc(sizeof(float) * src.cols, AVX_ALIGN);
		//float* lineo = (float*)_mm_malloc(sizeof(float) * src.cols, AVX_ALIGN);
		const int hend = src.cols - 2 * rs;
		const int HEND = get_simd_floor(hend, 8);
		const int WIDTH = get_simd_floor(src.cols, 8);
		for (int j = radius; j < dest.rows - radius; j += 2)
		{
			const float* sptr = src.ptr<float>((j - radius) >> 1);
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* si = sptr + i;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo, moddratio));

			}
			for (int i = WIDTH; i < src.cols; i++)
			{
				const float* si = sptr + i;
				float sume = GaussWeight[0] * *si; si += step;
				float sumo = 0.f;
				for (int k = 2; k < D2; k += 2)
				{
					sume += GaussWeight[k] * *si;
					sumo += GaussWeight[k - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}

			// h filter
			float* deptr = dest.ptr<float>(j, radius);
			float* doptr = dest.ptr<float>(j + 1, radius);
			const float* daeptr = addsubsrc.ptr<float>(j, radius);
			const float* daoptr = addsubsrc.ptr<float>(j + 1, radius);
			for (int i = 0; i < HEND; i += 8)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}
				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);
				if constexpr (isAdd)
				{
					_mm256_storeu_ps((float*)(deptr + 2 * i + 0), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(deptr + 2 * i + 8), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)));
				}
				else
				{
					_mm256_storeu_ps((float*)(deptr + 2 * i + 0), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(deptr + 2 * i + 8), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)));
				}

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);
				if constexpr (isAdd)
				{
					_mm256_storeu_ps((float*)(doptr + 2 * i + 0), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(doptr + 2 * i + 8), _mm256_fmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)));
				}
				else
				{
					_mm256_storeu_ps((float*)(doptr + 2 * i + 0), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)));
					_mm256_storeu_ps((float*)(doptr + 2 * i + 8), _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)));
				}
			}
			for (int i = HEND; i < hend; i++)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				float sumee = GaussWeight[0] * *sie++;
				float sumoe = 0.f;
				float sumeo = GaussWeight[0] * *sio++;
				float sumoo = 0.f;
				for (int k = 2; k < D2; k += 2)
				{
					sumee += GaussWeight[k] * *sie;
					sumoe += GaussWeight[k - 1] * *sie++;
					sumeo += GaussWeight[k] * *sio;
					sumoo += GaussWeight[k - 1] * *sio++;
				}
				const int I = i << 1;
				if constexpr (isAdd)
				{
					deptr[I + 0] = daeptr[I + 0] + sumee * evenratio;
					deptr[I + 1] = daeptr[I + 1] + sumoe * oddratio;
					doptr[I + 0] = daoptr[I + 0] + sumeo * evenratio;
					doptr[I + 1] = daoptr[I + 1] + sumoo * oddratio;
				}
				else
				{

					deptr[I + 0] = daeptr[I + 0] - sumee * evenratio;
					deptr[I + 1] = daeptr[I + 1] - sumoe * oddratio;
					doptr[I + 0] = daoptr[I + 0] - sumeo * evenratio;
					doptr[I + 1] = daoptr[I + 1] - sumoo * oddratio;
				}
			}
		}
		//_mm_free(linee);
		//_mm_free(lineo);
		_mm_free(GW);
	}

#pragma endregion

#pragma region pyramid bullding
	void MultiScaleFilter::buildGaussianPyramid(const Mat& src, vector<Mat>& destPyramid, const int level, const float sigma)
	{
		destPyramid.resize(level + 1);
		src.copyTo(destPyramid[0]);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			GaussDownIgnoreBoundary(src, destPyramid[1]);
			for (int l = 1; l < level; l++)
			{
				GaussDownIgnoreBoundary(destPyramid[l], destPyramid[l + 1]);
			}
		}
		else if (pyramidComputeMethod == Fast)
		{
			GaussDown(src, destPyramid[1]);
			for (int l = 1; l < level; l++)
			{
				GaussDown(destPyramid[l], destPyramid[l + 1]);
			}
		}
		else if (pyramidComputeMethod == Full)
		{
			GaussDownFull(src, destPyramid[1], sigma, borderType);
			for (int l = 1; l < level; l++)
			{
				GaussDownFull(destPyramid[l], destPyramid[l + 1], sigma, borderType);
			}
		}
		else if (pyramidComputeMethod == OpenCV)
		{
			cv::buildPyramid(destPyramid[0], destPyramid, level, borderType);
		}
	}

	void MultiScaleFilter::buildGaussianLaplacianPyramid(const Mat& src, vector<Mat>& GaussianPyramid, vector<Mat>& LaplacianPyramid, const int level, const float sigma)
	{
		GaussianPyramid.resize(level + 1);
		if (src.data != GaussianPyramid[0].data)	src.copyTo(GaussianPyramid[0]);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			float* linebuff = (float*)_mm_malloc(sizeof(float) * src.cols, AVX_ALIGN);
			float* linebuff2 = (float*)_mm_malloc(sizeof(float) * src.cols, AVX_ALIGN);
			if (radius == 2)
			{
				GaussDownIgnoreBoundary<5>(src, GaussianPyramid[1], linebuff);
				GaussUpAddIgnoreBoundary<false, 6>(GaussianPyramid[1], src, LaplacianPyramid[0], linebuff, linebuff2);
			}
			else if (radius == 4)
			{
				GaussDownIgnoreBoundary<9>(src, GaussianPyramid[1], linebuff);
				GaussUpAddIgnoreBoundary<false, 10>(GaussianPyramid[1], src, LaplacianPyramid[0], linebuff, linebuff2);
			}
			else
			{
				GaussDownIgnoreBoundary(src, GaussianPyramid[1]);
				GaussUpAddIgnoreBoundary<false>(GaussianPyramid[1], src, LaplacianPyramid[0]);
			}

			for (int l = 1; l < level; l++)
			{
				if (radius == 2)
				{
					GaussDownIgnoreBoundary<5>(GaussianPyramid[l], GaussianPyramid[l + 1], linebuff);
					GaussUpAddIgnoreBoundary<false, 6>(GaussianPyramid[l + 1], GaussianPyramid[l], LaplacianPyramid[l], linebuff, linebuff2);
				}
				else if (radius == 4)
				{
					GaussDownIgnoreBoundary<9>(GaussianPyramid[l], GaussianPyramid[l + 1], linebuff);
					GaussUpAddIgnoreBoundary<false, 10>(GaussianPyramid[l + 1], GaussianPyramid[l], LaplacianPyramid[l], linebuff, linebuff2);
				}
				else
				{
					GaussDownIgnoreBoundary(GaussianPyramid[l], GaussianPyramid[l + 1]);
					GaussUpAddIgnoreBoundary<false>(GaussianPyramid[l + 1], GaussianPyramid[l], LaplacianPyramid[l]);
				}

			}
			_mm_free(linebuff);
			_mm_free(linebuff2);
		}
		else if (pyramidComputeMethod == Fast)
		{
			GaussDown(src, GaussianPyramid[1]);
			GaussUpAdd<false>(GaussianPyramid[1], src, LaplacianPyramid[0]);
			for (int l = 1; l < level; l++)
			{
				GaussDown(GaussianPyramid[l], GaussianPyramid[l + 1]);
				GaussUpAdd <false>(GaussianPyramid[l + 1], GaussianPyramid[l], LaplacianPyramid[l]);
			}
		}
		else if (pyramidComputeMethod == Full)
		{
			GaussDownFull(src, GaussianPyramid[1], sigma, borderType);
			GaussUpAddFull<false>(GaussianPyramid[1], src, LaplacianPyramid[0], sigma, borderType);
			for (int l = 1; l < level; l++)
			{
				GaussDownFull(GaussianPyramid[l], GaussianPyramid[l + 1], sigma, borderType);
				GaussUpAddFull<false>(GaussianPyramid[l + 1], GaussianPyramid[l], LaplacianPyramid[l], sigma, borderType);
			}
		}
		else if (pyramidComputeMethod == OpenCV)
		{
			buildPyramid(src, GaussianPyramid, level, borderType);
			for (int i = 0; i < level; i++)
			{
				Mat temp;
				pyrUp(GaussianPyramid[i + 1], temp, GaussianPyramid[i].size(), borderType);
				subtract(GaussianPyramid[i], temp, LaplacianPyramid[i]);
			}
		}
	}


	void MultiScaleFilter::buildLaplacianPyramid(const Mat& src, vector<Mat>& destPyramid, const int level, const float sigma)
	{
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);
		//destPyramid[0].create(src.size(), CV_32F);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			GaussDownIgnoreBoundary(src, destPyramid[1]);
			GaussUpAddIgnoreBoundary <false>(destPyramid[1], src, destPyramid[0]);
			for (int l = 1; l < level; l++)
			{
				GaussDownIgnoreBoundary(destPyramid[l], destPyramid[l + 1]);
				GaussUpAddIgnoreBoundary <false>(destPyramid[l + 1], destPyramid[l], destPyramid[l]);
			}
		}
		else if (pyramidComputeMethod == Fast)
		{
			GaussDown(src, destPyramid[1]);
			GaussUpAdd<false>(destPyramid[1], src, destPyramid[0]);
			for (int l = 1; l < level; l++)
			{
				GaussDown(destPyramid[l], destPyramid[l + 1]);
				GaussUpAdd <false>(destPyramid[l + 1], destPyramid[l], destPyramid[l]);
			}
		}
		else if (pyramidComputeMethod == Full)
		{
			GaussDownFull(src, destPyramid[1], sigma, borderType);
			GaussUpAddFull<false>(destPyramid[1], src, destPyramid[0], sigma, borderType);
			for (int l = 1; l < level; l++)
			{
				GaussDownFull(destPyramid[l], destPyramid[l + 1], sigma, borderType);
				GaussUpAddFull<false>(destPyramid[l + 1], destPyramid[l], destPyramid[l], sigma, borderType);
			}
		}
		else if (pyramidComputeMethod == OpenCV)
		{
			buildPyramid(src, destPyramid, level, borderType);
			for (int i = 0; i < level; i++)
			{
				Mat temp;
				pyrUp(destPyramid[i + 1], temp, destPyramid[i].size(), borderType);
				subtract(destPyramid[i], temp, destPyramid[i]);
			}
		}
	}

	template<int D, int d, int d2>
	void MultiScaleFilter::buildLaplacianPyramid(const Mat& src, vector<Mat>& destPyramid, const int level, const float sigma, float* linebuff)
	{
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);
		//destPyramid[0].create(src.size(), CV_32F);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			GaussDownIgnoreBoundary<D>(src, destPyramid[1], linebuff);
			GaussUpAddIgnoreBoundary<false, d2>(destPyramid[1], src, destPyramid[0], linebuff, linebuff + destPyramid[1].cols);
			for (int l = 1; l < level; l++)
			{
				GaussDownIgnoreBoundary<D>(destPyramid[l], destPyramid[l + 1], linebuff);
				GaussUpAddIgnoreBoundary<false, d2>(destPyramid[l + 1], destPyramid[l], destPyramid[l], linebuff, linebuff + destPyramid[1].cols);
			}
		}
		else if (pyramidComputeMethod == Fast)
		{
			GaussDown<D>(src, destPyramid[1], linebuff);
			GaussUpAdd<false, d, d2>(destPyramid[1], src, destPyramid[0]);
			for (int l = 1; l < level; l++)
			{
				GaussDown<D>(destPyramid[l], destPyramid[l + 1], linebuff);
				GaussUpAdd<false, d, d2>(destPyramid[l + 1], destPyramid[l], destPyramid[l]);
			}
		}
		else if (pyramidComputeMethod == Full)
		{
			GaussDownFull(src, destPyramid[1], sigma, borderType);
			GaussUpAddFull<false>(destPyramid[1], src, destPyramid[0], sigma, borderType);
			for (int l = 1; l < level; l++)
			{
				GaussDownFull(destPyramid[l], destPyramid[l + 1], sigma, borderType);
				GaussUpAddFull<false>(destPyramid[l + 1], destPyramid[l], destPyramid[l], sigma, borderType);
			}
		}
		else if (pyramidComputeMethod == OpenCV)
		{
			buildPyramid(src, destPyramid, level, borderType);
			for (int i = 0; i < level; i++)
			{
				Mat temp;
				pyrUp(destPyramid[i + 1], temp, destPyramid[i].size(), borderType);
				subtract(destPyramid[i], temp, destPyramid[i]);
			}
		}
	}

	void MultiScaleFilter::buildLaplacianPyramid(const vector<Mat>& GaussianPyramid, vector<Mat>& destPyramid, const int level, const float sigma)
	{
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			for (int l = 0; l < level; l++)
			{
				GaussUpAddIgnoreBoundary<false>(GaussianPyramid[l + 1], GaussianPyramid[l], destPyramid[l]);
			}
		}
		else if (pyramidComputeMethod == Fast)
		{
			for (int l = 0; l < level; l++)
			{
				GaussUpAdd<false>(GaussianPyramid[l + 1], GaussianPyramid[l], destPyramid[l]);
			}
		}
		else if (pyramidComputeMethod == Full)
		{
			for (int l = 0; l < level; l++)
			{
				GaussUpAddFull<false>(GaussianPyramid[l + 1], GaussianPyramid[l], destPyramid[l], sigma, borderType);
			}
		}
		else if (pyramidComputeMethod == OpenCV)
		{
			for (int l = 0; l < level; l++)
			{
				Mat temp;
				pyrUp(GaussianPyramid[l + 1], temp, destPyramid[l].size(), borderType);
				subtract(destPyramid[l], temp, destPyramid[l]);
			}
		}
	}

	void MultiScaleFilter::collapseLaplacianPyramid(vector<Mat>& LaplacianPyramid, Mat& dest)
	{
		const int level = (int)LaplacianPyramid.size();

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			for (int i = level - 1; i > 1; i--)
			{
				GaussUpAddIgnoreBoundary<true>(LaplacianPyramid[i], LaplacianPyramid[i - 1], LaplacianPyramid[i - 1]);
			}
			GaussUpAddIgnoreBoundary<true>(LaplacianPyramid[1], LaplacianPyramid[0], dest);
		}
		else if (pyramidComputeMethod == Fast)
		{
			for (int i = level - 1; i > 1; i--)
			{
				GaussUpAdd<true>(LaplacianPyramid[i], LaplacianPyramid[i - 1], LaplacianPyramid[i - 1]);
			}
			GaussUpAdd<true>(LaplacianPyramid[1], LaplacianPyramid[0], dest);
		}
		else if (pyramidComputeMethod == Full)
		{
			for (int i = level - 1; i > 1; i--)
			{
				GaussUpAddFull<true>(LaplacianPyramid[i], LaplacianPyramid[i - 1], LaplacianPyramid[i - 1], sigma_space, borderType);
			}
			GaussUpAddFull<true>(LaplacianPyramid[1], LaplacianPyramid[0], dest, sigma_space, borderType);
		}
		else if (pyramidComputeMethod == OpenCV)
		{
			Mat ret;
			cv::pyrUp(LaplacianPyramid[level - 1], ret, LaplacianPyramid[level - 2].size(), borderType);
			for (int i = level - 2; i != 0; i--)
			{
				cv::add(ret, LaplacianPyramid[i], ret);
				cv::pyrUp(ret, ret, LaplacianPyramid[i - 1].size(), borderType);
			}
			cv::add(ret, LaplacianPyramid[0], dest);
		}

	}
#pragma endregion

	void MultiScaleFilter::buildGaussianStack(Mat& src, vector<Mat>& GaussianStack, const float sigma_s, const int level)
	{
		GaussianStack.resize(level + 1);

		src.convertTo(GaussianStack[0], CV_32F);

#pragma omp parallel for
		for (int i = 1; i <= level; i++)
		{
			//const float sigma_l = sigma_s * i;
			const float sigma_l = (float)getPyramidSigma(sigma_s, i);
			const int r = (int)ceil(sigma_l * 3.f);
			const Size ksize(2 * r + 1, 2 * r + 1);
#ifdef USE_SLIDING_DCT
			gf::SpatialFilterSlidingDCT5_AVX_32F sf(gf::DCT_COEFFICIENTS::FULL_SEARCH_NOOPT);
			sf.filter(GaussianStack[0], GaussianStack[i], sigma_l, 2);
#else
			GaussianBlur(GaussianStack[0], GaussianStack[i], ksize, sigma_l);
#endif
		}
	}

	void MultiScaleFilter::buildDoGStack(Mat& src, vector<Mat>& ImageStack, const float sigma_s, const int level)
	{
		if (ImageStack.size() != level + 1) ImageStack.resize(level + 1);
		for (int l = 0; l <= level; l++)ImageStack[l].create(src.size(), CV_32F);

		//spatialGradient
		//gf::SpatialFilter sf(gf::SpatialFilterAlgorithm::SlidingDCT5_AVX, CV_32F);
#ifdef USE_SLIDING_DCT
		gf::SpatialFilterSlidingDCT5_AVX_32F sf(gf::DCT_COEFFICIENTS::FULL_SEARCH_NOOPT);
#endif
		Mat prev = src;
		//#pragma omp parallel for
		for (int l = 1; l <= level; l++)
		{
			const float sigma_l = (float)getPyramidSigma(sigma_s, l);
			const int r = (int)ceil(sigma_l * 3.f);
			const Size ksize(2 * r + 1, 2 * r + 1);

#ifdef USE_SLIDING_DCT
			sf.filter(src, DoGStack[l], sigma_l, 2);
#else
			GaussianBlur(src, ImageStack[l], ksize, sigma_l);//DoG is not separable
#endif	
			subtract(prev, ImageStack[l], ImageStack[l - 1]);
			prev = ImageStack[l];
		}
		/*
		buildGaussianStack(src, DoGStack, sigma_s, level);
		for (int i = 0; i <= level - 1; i++)
		{
			subtract(DoGStack[i], DoGStack[i + 1], DoGStack[i]);
		}
		*/
	}

	void MultiScaleFilter::collapseDoGStack(vector<Mat>& ImageStack, Mat& dest)
	{
		const int level = (int)ImageStack.size();
		dest.create(ImageStack[0].size(), CV_32F);

		AutoBuffer<float*> ptr(level);
		for (int l = 0; l < level; l++)
		{
			ptr[l] = ImageStack[l].ptr<float>();
		}
		float* dptr = dest.ptr<float>();
		const int size = ImageStack[0].rows * ImageStack[0].cols;
		const int simd_end = get_simd_floor(size, 32);
#pragma omp parallel for
		for (int i = 0; i < simd_end; i += 32)
		{
			__m256 sum0 = _mm256_loadu_ps(ptr[0] + i);
			__m256 sum1 = _mm256_loadu_ps(ptr[0] + i + 8);
			__m256 sum2 = _mm256_loadu_ps(ptr[0] + i + 16);
			__m256 sum3 = _mm256_loadu_ps(ptr[0] + i + 24);

			for (int l = 1; l < level; l++)
			{
				const float* idx = ptr[l] + i;
				sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(idx));
				sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(idx + 8));
				sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(idx + 16));
				sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(idx + 24));
			}
			_mm256_storeu_ps(dptr + i, sum0);
			_mm256_storeu_ps(dptr + i + 8, sum1);
			_mm256_storeu_ps(dptr + i + 16, sum2);
			_mm256_storeu_ps(dptr + i + 24, sum3);
		}

		for (int i = simd_end; i < size; i++)
		{
			dptr[i] = ptr[0][i];
			for (int l = 1; l < level; l++)
			{
				dptr[i] += ptr[l][i];
			}
		}

		/*DoGStack[0].copyTo(dest);
		for (int i = 1; i < level; i++)
		{
			dest += DoGStack[i];
		}*/
	}


	void MultiScaleFilter::body(const Mat& src, Mat& dest)
	{
		dest.create(src.size(), src.type());
		if (src.channels() == 1)
		{
			gray(src, dest);
		}
		else
		{
			bool flag = true;
			if (flag)
			{
				Mat gim;;
				cv::cvtColor(src, gim, COLOR_BGR2YUV);
				vector<Mat> vsrc;
				split(gim, vsrc);
				gray(vsrc[0], vsrc[0]);
				merge(vsrc, dest);
				cv::cvtColor(dest, dest, COLOR_YUV2BGR);
			}
			else
			{
				vector<Mat> vsrc;
				vector<Mat> vdst(3);
				split(src, vsrc);
				gray(vsrc[0], vdst[0]);
				gray(vsrc[1], vdst[1]);
				gray(vsrc[2], vdst[2]);
				merge(vdst, dest);
			}
		}
	}

	void MultiScaleFilter::gray(const Mat& src, Mat& dest)
	{
		if (scalespaceMethod == Pyramid)
		{
			pyramid(src, dest);
		}
		else
		{
			dog(src, dest);
		}
	}


	void MultiScaleFilter::showPyramid(string wname, vector<Mat>& pyramid, bool isShowLevel)
	{
		namedWindow(wname);
		const int size = (int)pyramid.size();
		if (size == 1)
		{
			imshow(wname, pyramid[0]);
			return;
		}

		Mat show;
		Mat expandv = Mat::zeros(pyramid[1].size(), pyramid[1].type());
		vconcat(pyramid[1], expandv, expandv);
		hconcat(pyramid[0], expandv, show);
		int vindex = pyramid[1].rows;
		for (int l = 2; l < size; l++)
		{
			Mat roi = show(Rect(pyramid[0].cols, vindex, pyramid[l].cols, pyramid[l].rows));
			pyramid[l].copyTo(roi);
			vindex += pyramid[l].rows;
		}
		Mat show8u;
		show.convertTo(show8u, CV_8U);
		if (show8u.channels() == 1)cvtColor(show8u, show8u, COLOR_GRAY2BGR);
		if (isShowLevel)
		{
			const string font = "Consolas";
			const int fontSize = 24;
			const Scalar fontColor = Scalar(255, 255, 255, 100);
			for (int l = 0; l < size; l++)
			{
				if (l == 0)
				{
					cv::addText(show8u, to_string(l), Point(pyramid[0].cols - fontSize, pyramid[0].rows - fontSize), font, fontSize, fontColor);
				}
				else if (l == 1)
				{
					cv::addText(show8u, to_string(l), Point(show.cols - fontSize, pyramid[1].rows + fontSize), font, fontSize, fontColor);
				}
				else
				{
					int vindex = pyramid[1].rows;
					for (int n = 2; n <= l; n++)
						vindex += pyramid[n].rows;

					cv::addText(show8u, to_string(l), Point(pyramid[0].cols + pyramid[l].rows, vindex), font, fontSize, fontColor);
				}
			}
		}
		imshow(wname, show8u);
	}

	void MultiScaleFilter::drawRemap(bool isWait, Size size)
	{
		if (rangeTable.empty())
		{
			cout << "do not initialized" << endl;
			return;
		}

		cp::Plot pt(size);
		string wname = "remap function";
		cv::namedWindow(wname);
		static int x = 128; cv::createTrackbar("x", wname, &x, 255);
		pt.setPlotSymbolALL(0);
		pt.setPlotTitle(0, "x");
		pt.setPlotTitle(1, "remap");
		pt.setPlotTitle(2, "2x");

		int key = 0;
		while (key != 'q')
		{
			for (int i = 0; i < 256; i++)
			{
				float v = i - (i - x) * rangeTable[abs(i - x)];//gaussian detail enhance

				pt.push_back(i, 2 * i - x, 2);
				pt.push_back(i, v, 1);
				pt.push_back(i, i, 0);
			}
			pt.plot(wname, false);
			key = waitKey(1);
			if (!isWait)break;
			pt.clear();
		}
	}

#pragma region getter/setter
	void MultiScaleFilter::setAdaptive(const bool adaptiveMethod, cv::Mat& adaptiveSigmaMap, cv::Mat& adaptiveBoostMap, const int level)
	{
		this->adaptiveMethod = adaptiveMethod ? AdaptiveMethod::ADAPTIVE : AdaptiveMethod::FIX;
		if (this->adaptiveMethod == AdaptiveMethod::FIX)return;

		this->adaptiveSigmaMap.resize(level);
		this->adaptiveBoostMap.resize(level);
		this->adaptiveSigmaMap[0] = adaptiveSigmaMap;
		this->adaptiveBoostMap[0] = adaptiveBoostMap;
		//adaptiveSigmaMap.copyTo(this->adaptiveSigmaMap[0]);
		//adaptiveBoostMap.copyTo(this->adaptiveBoostMap[0]);
		for (int l = 0; l < level - 1; l++)
		{
			resize(this->adaptiveSigmaMap[l], this->adaptiveSigmaMap[l + 1], Size(), 0.5, 0.5, INTER_NEAREST);
			resize(this->adaptiveBoostMap[l], this->adaptiveBoostMap[l + 1], Size(), 0.5, 0.5, INTER_NEAREST);
		}
	}

	std::string MultiScaleFilter::getAdaptiveName()
	{
		string ret = "";
		if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)ret = "ADAPTIVE";
		else ret = "FIX";

		return ret;
	}

	void MultiScaleFilter::setPyramidComputeMethod(const PyramidComputeMethod scaleSpaceMethod)
	{
		pyramidComputeMethod = scaleSpaceMethod;
	}

	void MultiScaleFilter::setRangeDescopeMethod(RangeDescopeMethod scaleSpaceMethod)
	{
		rangeDescopeMethod = scaleSpaceMethod;
	}

	std::string MultiScaleFilter::getRangeDescopeMethod()
	{
		string ret = "";
		if (rangeDescopeMethod == RangeDescopeMethod::FULL) ret = "FULL";
		if (rangeDescopeMethod == RangeDescopeMethod::MINMAX) ret = "MINMAX";
		if (rangeDescopeMethod == RangeDescopeMethod::LOCAL) ret = "LOCAL";
		return ret;
	}

	cv::Size MultiScaleFilter::getLayerSize(const int level)
	{
		if (layerSize.empty())return Size(0, 0);
		if (layerSize.size() < level)return Size(0, 0);
		return layerSize[level];
	}

	std::string MultiScaleFilter::getScaleSpaceName()
	{
		string ret = "";
		if (scalespaceMethod == ScaleSpace::Pyramid)ret = "Pyramid";
		if (scalespaceMethod == ScaleSpace::DoG)ret = "DoG";

		return ret;
	}

	std::string MultiScaleFilter::getPyramidComputeName()
	{
		string ret = "";
		if (pyramidComputeMethod == PyramidComputeMethod::IgnoreBoundary)ret = "IgnoreBoundary";
		if (pyramidComputeMethod == PyramidComputeMethod::Fast)ret = "Fast";
		if (pyramidComputeMethod == PyramidComputeMethod::Full)ret = "Full";
		if (pyramidComputeMethod == PyramidComputeMethod::OpenCV)ret = "OpenCV";

		return ret;
	}

#pragma endregion

#pragma endregion


#pragma region MultiScaleGaussianFilter

	void MultiScaleGaussianFilter::pyramid(const Mat& src, Mat& dest)
	{
		ImageStack.resize(level + 1);
		initRangeTable(sigma_range, boost);

		if (src.depth() == CV_8U) src.convertTo(ImageStack[0], CV_32F);
		else src.copyTo(ImageStack[0]);

		buildGaussianPyramid(ImageStack[0], ImageStack, level, sigma_space);
		buildLaplacianPyramid(ImageStack[0], ImageStack, level, sigma_space);

		for (int i = 0; i < ImageStack.size() - 1; i++)
		{
			remap(ImageStack[i], ImageStack[i], 0.f, sigma_range, boost);
		}

		collapseLaplacianPyramid(ImageStack, ImageStack[0]);//override srcf for saving memory	

		ImageStack[0].convertTo(dest, src.type());
	}

	void MultiScaleGaussianFilter::dog(const Mat& src, Mat& dest)
	{
		initRangeTable(sigma_range, boost);

		Mat srcf;
		if (src.depth() == CV_8U) src.convertTo(srcf, CV_32F);
		else srcf = src.clone();

		buildDoGStack(srcf, ImageStack, sigma_space, level);
		for (int i = 0; i < ImageStack.size() - 1; i++)
		{
			remap(ImageStack[i], ImageStack[i], 0.f, sigma_range, boost);
		}
		collapseDoGStack(ImageStack, srcf);//override srcf for saving memory	

		srcf.convertTo(dest, src.type());
	}

	void MultiScaleGaussianFilter::filter(const Mat& src, Mat& dest, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);
		this->pyramidComputeMethod = Fast;

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->boost = boost;
		this->level = level;
		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);

		freeSpaceWeight();
	}

#pragma endregion

#pragma region MultiScaleBilateralFilter

	void MultiScaleBilateralFilter::buildDoBFStack(const Mat& src, vector<Mat>& DoBFStack, const float sigma_r, const float sigma_s, const int level)
	{
		if (DoBFStack.size() != level + 1) DoBFStack.resize(level + 1);
		for (int l = 0; l <= level; l++)DoBFStack[l].create(src.size(), CV_32F);

		Mat prev = src;
		for (int l = 1; l <= level; l++)
		{
			const float sigma_l = (float)getPyramidSigma(sigma_s, l);
			const int r = getGaussianRadius(sigma_l);
			const Size ksize(2 * r + 1, 2 * r + 1);

			cv::bilateralFilter(src, DoBFStack[l], 2 * r + 1, sigma_r, sigma_l, borderType);

			subtract(prev, DoBFStack[l], DoBFStack[l - 1]);
			prev = DoBFStack[l];
		}
	}

	void MultiScaleBilateralFilter::pyramid(const Mat& src, Mat& dest)
	{
		ImageStack.resize(level + 1);
		initRangeTable(sigma_range, boost);

		if (src.depth() == CV_8U) src.convertTo(ImageStack[0], CV_32F);
		else src.copyTo(ImageStack[0]);

		buildGaussianPyramid(ImageStack[0], ImageStack, level, sigma_space);
		buildLaplacianPyramid(ImageStack[0], ImageStack, level, sigma_space);


		if (pyramidComputeMethod) cv::buildPyramid(ImageStack[0], ImageStack, level, borderType);
		else buildGaussianPyramid(ImageStack[0], ImageStack, level, sigma_space);

		buildLaplacianPyramid(ImageStack[0], ImageStack, level, 1.f);

		for (int i = 0; i < ImageStack.size() - 1; i++)
		{
			remap(ImageStack[i], ImageStack[i], 0.f, sigma_range, boost);
		}

		collapseLaplacianPyramid(ImageStack, ImageStack[0]);//override srcf for saving memory	

		ImageStack[0].convertTo(dest, src.type());
	}

	void MultiScaleBilateralFilter::dog(const Mat& src, Mat& dest)
	{
		//cout << "BF DoG" << endl;
		ImageStack.resize(level + 1);
		initRangeTable(sigma_range, boost);

		if (src.depth() == CV_8U) src.convertTo(ImageStack[0], CV_32F);
		else src.copyTo(ImageStack[0]);

		buildDoBFStack(src, ImageStack, sigma_range, sigma_space, level);
		for (int i = 0; i < ImageStack.size() - 1; i++)
		{
			remap(ImageStack[i], ImageStack[i], 0.f, sigma_range, boost);
		}

		collapseDoGStack(ImageStack, ImageStack[0]);//override srcf for saving memory	

		ImageStack[0].convertTo(dest, src.type());
	}

	void MultiScaleBilateralFilter::filter(const Mat& src, Mat& dest, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);
		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->boost = boost;
		this->level = level;
		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);

		freeSpaceWeight();
	}

#pragma endregion


#pragma region LocalMultiScaleFilter
	void LocalMultiScaleFilter::pyramid(const Mat& src_, Mat& dest_)
	{
		//rangeDescope(src_);

		//const bool isDebug = true;
		const bool isDebug = false;

		Mat src, dest;
		//const int r2 = (int)pow(2, level+1) * radius;
		//print_debug2(r, r2);
		const int r_ = (int)pow(2, level - 1) * radius * 3;
		const int r = (get_simd_ceil(src_.cols + 2 * r_, (int)pow(2, level)) - src_.cols + 2 * r_) / 2;
		copyMakeBorder(src_, src, r, r, r, r, borderType);

		initRangeTable(sigma_range, boost);
		Mat srcf;
		if (src.depth() == CV_8U) src.convertTo(srcf, CV_32F);
		else srcf = src;

		const int lowr = 3 * radius;
		const int r_pad0 = lowr * (int)pow(2, level - 1);

		AutoBuffer<int> widthl(level + 1);
		AutoBuffer<int> heightl(level + 1);
		AutoBuffer<int> r_pad_gfpyl(level + 1);
		AutoBuffer<int> r_pad_localpyl(level + 1);

		AutoBuffer<int> ampl(level + 1);
		AutoBuffer<int> block_sizel(level + 1);
		AutoBuffer<int> patch_sizel(level + 1);
		for (int l = 0; l <= level; l++)
		{
			heightl[l] = src.rows / (int)pow(2, l);
			widthl[l] = src.cols / (int)pow(2, l);

			const int ww = (src.cols + 2 * r_pad0) / (int)pow(2, l);
			r_pad_gfpyl[l] = (ww - widthl[l]) / 2;
			r_pad_localpyl[l] = lowr * (int)pow(2, l);
			ampl[l] = (int)pow(2, l);
			block_sizel[l] = (int)pow(2, l + 1);
			patch_sizel[l] = (block_sizel[l] + 2 * lowr) * (int)pow(2, l);
		}
		copyMakeBorder(srcf, border, r_pad0, r_pad0, r_pad0, r_pad0, borderType);

		//(1) build Gaussian pyramid
		buildGaussianPyramid(border, ImageStack, level, sigma_space);

		if (pyramidComputeMethod == IgnoreBoundary && adaptiveMethod == AdaptiveMethod::ADAPTIVE)
		{
			if (adaptiveBoostBorder.size() != level)adaptiveBoostBorder.resize(level);
			if (adaptiveSigmaBorder.size() != level)adaptiveSigmaBorder.resize(level);
			const bool borderEach = false;
			if (borderEach)
			{
				for (int l = 0; l < level; l++)
				{
					int rr = (ImageStack[l].cols - adaptiveBoostMap[l].cols) / 2;
					cv::copyMakeBorder(adaptiveBoostMap[l], adaptiveBoostBorder[l], rr, rr, rr, rr, borderType);
					cv::copyMakeBorder(adaptiveSigmaMap[l], adaptiveSigmaBorder[l], rr, rr, rr, rr, borderType);
				}
			}
			else
			{
				cv::copyMakeBorder(adaptiveBoostMap[0], adaptiveBoostBorder[0], r_pad0 + r, r_pad0 + r, r_pad0 + r, r_pad0 + r, borderType);
				cv::copyMakeBorder(adaptiveSigmaMap[0], adaptiveSigmaBorder[0], r_pad0 + r, r_pad0 + r, r_pad0 + r, r_pad0 + r, borderType);
				//print_debug2(srcf.size(), adaptiveBoostMap[0].size());
				//print_debug2(border.size(), adaptiveBoostBorder[0].size());
				for (int l = 0; l < level - 1; l++)
				{
					resize(adaptiveBoostBorder[l], adaptiveBoostBorder[l + 1], Size(), 0.5, 0.5, INTER_NEAREST);
					resize(adaptiveSigmaBorder[l], adaptiveSigmaBorder[l + 1], Size(), 0.5, 0.5, INTER_NEAREST);
				}
			}
		}

		if (isDebug)
		{
			showPyramid("GaussPy", ImageStack);
			cout << "-------------------------------------------------------" << endl;
			for (int l = 0; l < level; l++)
			{
				print_debug2(l, ImageStack[l].size());
				print_debug2(widthl[l], heightl[l]);
				print_debug5(patch_sizel[l], block_sizel[l], r_pad0, r_pad_gfpyl[l], r_pad_localpyl[l]);

				vector<Mat> llp;
				Mat rm(patch_sizel[l], patch_sizel[l], CV_32F);
				buildLaplacianPyramid(rm, llp, l + 1, sigma_space);
				print_debug2(l, llp[l].size());
				cout << endl;
			}
		}

		if (LaplacianPyramid.size() != ImageStack.size())LaplacianPyramid.resize(ImageStack.size());
		LaplacianPyramid[0].create(src.size(), ImageStack[0].depth());
		for (int l = 1; l < LaplacianPyramid.size(); l++)
		{
			LaplacianPyramid[l].create(LaplacianPyramid[l - 1].size() / 2, ImageStack[0].depth());
			//LaplacianPyramid[l].create(GaussianPyramid[l].size(), GaussianPyramid[0].depth());
		}

		//(2) build Laplacian pyramid (0 to level)
		for (int l = 0; l <= level; l++)
		{
			const int height = heightl[l];
			const int width = widthl[l];
			const int r_pad_gfpy = r_pad_gfpyl[l];
			const int r_pad_localpy = r_pad_localpyl[l];
			const int amp = ampl[l];
			const int block_size = block_sizel[l];
			const int patch_size = patch_sizel[l];
			//const int width = src.cols / (int)pow(2, l);
			//const int r_pad_py = (DoGStack[level-1].cols - src.cols / (int)pow(2, level-1)) / 2;
			if (l == level)
			{
				ImageStack[l](Rect(r_pad_gfpy, r_pad_gfpy, width, height)).copyTo(LaplacianPyramid[l]);
			}
			else
			{
				if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
#pragma omp parallel for schedule (dynamic)
					for (int j = 0; j < height; j += block_size)
					{
						vector<Mat> llp;
						Mat rm(patch_size, patch_size, CV_32F);
						float* linebuff = (float*)_mm_malloc(sizeof(float) * (patch_size), AVX_ALIGN);
						for (int i = 0; i < width; i += block_size)
						{
							//generating pyramid from 0 to l+1
							const Mat rect = ImageStack[0](Rect(i* amp + r_pad0 - r_pad_localpy, j* amp + r_pad0 - r_pad_localpy, patch_size, patch_size)).clone();
							for (int n = 0; n < block_size; n++)
							{
								for (int m = 0; m < block_size; m++)
								{
									const float g = ImageStack[l].at<float>(r_pad_gfpy + j + n, r_pad_gfpy + i + m);
									const float sigma = adaptiveSigmaBorder[l].at<float>(r_pad_gfpy + j + n, r_pad_gfpy + i + m);
									const float boost = adaptiveBoostBorder[l].at<float>(r_pad_gfpy + j + n, r_pad_gfpy + i + m);
									//const float sigma = sigma_range;
									//const float boost = detail_param;
									remap(rect, rm, g, sigma, boost);
									if (radius == 2)
									{
										//const int D = 2 * r + 1;
										//const int d = 2 * rs + 1;
										//const int d2 = 2 * D;
										buildLaplacianPyramid<5, 3, 6>(rm, llp, l + 1, sigma_space, linebuff);
									}
									else if (radius == 4)
									{
										buildLaplacianPyramid<9, 5, 10>(rm, llp, l + 1, sigma_space, linebuff);
									}
									else
									{
										buildLaplacianPyramid(rm, llp, l + 1, sigma_space);
									}
									//buildLaplacianPyramid(rm, llp, l + 1, sigma_space);
									LaplacianPyramid[l].at<float>(j + n, i + m) = llp[l].at<float>(lowr + n, lowr + m);
								}
							}
						}
						_mm_free(linebuff);
					}
				}
				else
				{
#pragma omp parallel for schedule (dynamic)
					for (int j = 0; j < height; j += block_size)
					{
						vector<Mat> llp;
						Mat rm(patch_size, patch_size, CV_32F);
						float* linebuff = (float*)_mm_malloc(sizeof(float) * (patch_size), AVX_ALIGN);
						for (int i = 0; i < width; i += block_size)
						{
							//generating pyramid from 0 to l+1
							const Mat rect = ImageStack[0](Rect(i* amp + r_pad0 - r_pad_localpy, j* amp + r_pad0 - r_pad_localpy, patch_size, patch_size)).clone();
							for (int n = 0; n < block_size; n++)
							{
								for (int m = 0; m < block_size; m++)
								{
									const float g = ImageStack[l].at<float>(r_pad_gfpy + j + n, r_pad_gfpy + i + m);
									remap(rect, rm, g, sigma_range, boost);
									if (radius == 2)
									{
										buildLaplacianPyramid<5, 3, 6>(rm, llp, l + 1, sigma_space, linebuff);
									}
									else if (radius == 4)
									{
										buildLaplacianPyramid<9, 5, 10>(rm, llp, l + 1, sigma_space, linebuff);
									}
									else
									{
										buildLaplacianPyramid(rm, llp, l + 1, sigma_space);
									}
									//buildLaplacianPyramid(rm, llp, l + 1, sigma_space);
									LaplacianPyramid[l].at<float>(j + n, i + m) = llp[l].at<float>(lowr + n, lowr + m);
								}
							}
						}
						//showPyramid("lLP", llp, false); showPyramid("LP", LaplacianPyramid); waitKey(1);
						_mm_free(linebuff);
					}
				}
			}
		}
		if (isDebug)showPyramid("Laplacian Pyramid Paris2011", LaplacianPyramid);
		collapseLaplacianPyramid(LaplacianPyramid, srcf);//override srcf for saving memory	
		srcf.convertTo(dest, src.type());
		dest(Rect(r, r, src_.cols, src_.rows)).copyTo(dest_);
	}

	void LocalMultiScaleFilter::dog(const Mat& src, Mat& dest)
	{
		initRangeTable(sigma_range, boost);

		Mat srcf;
		if (src.depth() == CV_32F)
		{
			srcf = src;
		}
		else
		{
			src.convertTo(srcf, CV_32F);
		}

		const float sigma_lmax = (float)getPyramidSigma(sigma_space, level);
		const int rmax = (int)ceil(sigma_lmax * 3.f);
		const Size ksizemax(2 * rmax + 1, 2 * rmax + 1);

		const int r_pad = (int)pow(2, level + 1);//2^(level+1)
		vector<Mat>  LaplacianStack(level + 1);

		//(1) build Gaussian stack
		buildGaussianStack(srcf, ImageStack, sigma_space, level);

		for (int i = 0; i < level; i++)
		{
			LaplacianStack[i].create(ImageStack[0].size(), CV_32F);
		}
		Mat im;
		copyMakeBorder(srcf, im, rmax, rmax, rmax, rmax, borderType);

		//(2) build DoG stack (0 to level-1)
		for (int l = 0; l < level; l++)
		{
			const float sigma_l = (float)getPyramidSigma(sigma_space, l);
			const float sigma_lp = (float)getPyramidSigma(sigma_space, l + 1);
			const int r = (int)ceil(sigma_lp * 3.f);
			const Size ksize(2 * r + 1, 2 * r + 1);
			AutoBuffer<float> weight(ksize.area());
			AutoBuffer<int> index(ksize.area());
			setDoGKernel(weight, index, im.cols, ksize, sigma_l, sigma_lp);

#pragma omp parallel for schedule (dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				for (int i = 0; i < src.cols; i++)
				{
					const float g = ImageStack[l].at<float>(j, i);
					LaplacianStack[l].at<float>(j, i) = getRemapDoGCoeffLn(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
					//if(l==0)LaplacianStack[l].at<float>(j, i) = getDoGCoeffLnNoremap(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
					//else LaplacianStack[l].at<float>(j, i) = getDoGCoeffLn(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
				}
			}
		}
		//(2) the last level is a copy of the last level DoG
		ImageStack[level].copyTo(LaplacianStack[level]);

		//(3) collapseDoG
		if (srcf.depth() == CV_32F)
		{
			collapseDoGStack(LaplacianStack, dest);
		}
		else
		{
			collapseDoGStack(LaplacianStack, srcf);//override srcf for saving memory	
			srcf.convertTo(dest, src.type());
		}
	}

	void LocalMultiScaleFilter::filter(const Mat& src, Mat& dest, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->level = level;
		this->boost = boost;
		this->scalespaceMethod = scaleSpaceMethod;
		body(src, dest);

		freeSpaceWeight();
	}

	void LocalMultiScaleFilter::setDoGKernel(float* weight, int* index, const int index_step, Size ksize, const float sigma1, const float sigma2)
	{
		CV_Assert(sigma2 > sigma1);

		const int r = ksize.width / 2;
		int count = 0;
		if (sigma1 == 0.f)
		{
			float sum2 = 0.f;
			const float coeff2 = float(1.0 / (-2.0 * sigma2 * sigma2));
			for (int j = -r; j <= r; j++)
			{
				for (int i = -r; i <= r; i++)
				{
					const float dist = float(j * j + i * i);
					const float v2 = exp(dist * coeff2);
					weight[count] = v2;
					index[count] = j * index_step + i;
					count++;
					sum2 += v2;
				}
			}
			sum2 = 1.f / sum2;
			for (int i = 0; i < ksize.area(); i++)
			{
				weight[i] = 0.f - weight[i] * sum2;
			}
			weight[ksize.area() / 2] = 1.f + weight[ksize.area() / 2];
		}
		else
		{
			AutoBuffer<float> buff(ksize.area());
			float sum1 = 0.f;
			float sum2 = 0.f;
			const float coeff1 = float(1.0 / (-2.0 * sigma1 * sigma1));
			const float coeff2 = float(1.0 / (-2.0 * sigma2 * sigma2));
			for (int j = -r; j <= r; j++)
			{
				for (int i = -r; i <= r; i++)
				{
					float dist = float(j * j + i * i);
					float v1 = exp(dist * coeff1);
					float v2 = exp(dist * coeff2);
					weight[count] = v1;
					buff[count] = v2;
					index[count] = j * index_step + i;
					count++;
					sum1 += v1;
					sum2 += v2;
				}
			}
			sum1 = 1.f / sum1;
			sum2 = 1.f / sum2;
			for (int i = 0; i < ksize.area(); i++)
			{
				weight[i] = weight[i] * sum1 - buff[i] * sum2;
			}
		}
	}

	float LocalMultiScaleFilter::getDoGCoeffLnNoremap(Mat& src, const float g, const int y, const int x, const int size, int* index, float* weight)
	{
		float* sptr = src.ptr<float>(y, x);
		const int simd_size = get_simd_floor(size, 8);

		const __m256 mg = _mm256_set1_ps(g);
		__m256 msum = _mm256_setzero_ps();
		for (int i = 0; i < simd_size; i += 8)
		{
			__m256i idx = _mm256_load_si256((const __m256i*)index);
			const __m256 ms = _mm256_i32gather_ps(sptr, idx, sizeof(float));
			msum = _mm256_fmadd_ps(_mm256_load_ps(weight), ms, msum);
			weight += 8;
			index += 8;
		}
		float sum = _mm256_reduceadd_ps(msum);

		for (int i = simd_size; i < size; i++)
		{
			const float s = sptr[*index];
			sum += *weight * s;
			weight++;
			index++;
		}

		return sum;
	}

	float LocalMultiScaleFilter::getRemapDoGCoeffLn(Mat& src, const float g, const int y, const int x, const int size, int* index, float* weight)
	{
		float* sptr = src.ptr<float>(y, x);
		float* rptr = &rangeTable[0];
		const int simd_size = get_simd_floor(size, 8);
		//cout << "size : " << size << endl;
		//cout << "simd_size : " << simd_size << endl;

		const __m256 mg = _mm256_set1_ps(g);
		__m256 msum = _mm256_setzero_ps();
		for (int i = 0; i < simd_size; i += 8)
		{
			__m256i idx = _mm256_load_si256((const __m256i*)index);
			const __m256 ms = _mm256_i32gather_ps(sptr, idx, sizeof(float));
			const __m256 subsg = _mm256_sub_ps(ms, mg);
			const __m256 md = _mm256_fmadd_ps(_mm256_i32gather_ps(rptr, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), subsg, ms);
			msum = _mm256_fmadd_ps(_mm256_load_ps(weight), md, msum);
			weight += 8;
			index += 8;
		}

		float sum = _mm256_reduceadd_ps(msum);

		//for (int i = 0; i < size; i++)
		for (int i = simd_size; i < size; i++)
		{
			const float s = sptr[*index];
			//float d = s - (s - g) * range[saturate_cast<uchar>(abs(s - g))];
			float d = s + (s - g) * rangeTable[saturate_cast<uchar>(abs(s - g))];
			//float d = s - (s - g) * -exp((s - g) * (s - g) / (-2.0 * 30.f * 30.f));
			sum += *weight * d;
			weight++;
			index++;
		}

		return sum;
	}
#pragma endregion

#pragma region LocalMultiScaleFilterFull

	void LocalMultiScaleFilterFull::pyramid(const Mat& src_, Mat& dest_)
	{
		//rangeDescope(src_);

		Mat src, dest;
		const int r = (int)pow(2, level) * 4;
		copyMakeBorder(src_, src, r, r, r, r, borderType);

		//const bool isDebug = true;
		const bool isDebug = false;

		initRangeTable(sigma_range, boost);

		Mat srcf;
		if (src.depth() == CV_8U) src.convertTo(srcf, CV_32F);
		else srcf = src;

		//(1) build Gaussian pyramid
		ImageStack.resize(level + 1);
		srcf.copyTo(ImageStack[0]);
		buildGaussianPyramid(ImageStack[0], ImageStack, level, sigma_space);

		if (isDebug) showPyramid("GaussPy", ImageStack);

		vector<Mat> LaplacianPyramid(ImageStack.size());
		LaplacianPyramid[0].create(src.size(), ImageStack[0].depth());
		for (int l = 1; l < LaplacianPyramid.size(); l++)
		{
			LaplacianPyramid[l].create(LaplacianPyramid[l - 1].size() / 2, ImageStack[0].depth());
		}

		//(2) build Laplacian pyramid (0 to level)
		for (int l = 0; l <= level; l++)
		{
			const int height = src.rows / (int)pow(2, l);
			const int width = src.cols / (int)pow(2, l);

			if (l == level)
			{
				ImageStack[l](Rect(0, 0, width, height)).copyTo(LaplacianPyramid[l]);
			}
			else
			{
				if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
#pragma omp parallel for schedule (dynamic)
					for (int j = 0; j < height; j++)
					{
						vector<Mat> llp;
						Mat rm(srcf.size(), CV_32F);
						for (int i = 0; i < width; i++)
						{
							const float g = ImageStack[l].at<float>(j, i);
							const float sigma = adaptiveSigmaMap[l].at<float>(j, i);
							const float boost = adaptiveBoostMap[l].at<float>(j, i);
							remap(srcf, rm, g, sigma, boost);
							buildLaplacianPyramid(rm, llp, l + 1, sigma_space);
							LaplacianPyramid[l].at<float>(j, i) = llp[l].at<float>(j, i);
						}
					}
				}
				else
				{
#pragma omp parallel for schedule (dynamic)
					for (int j = 0; j < height; j++)
					{
						vector<Mat> llp;
						Mat rm(src.size(), CV_32F);
						//Mat rm = srcf.clone();
						for (int i = 0; i < width; i++)
						{
							const float g = ImageStack[l].at<float>(j, i);
							remap(srcf, rm, g, sigma_range, boost);
							buildLaplacianPyramid(rm, llp, l + 1, sigma_space);
							LaplacianPyramid[l].at<float>(j, i) = llp[l].at<float>(j, i);
						}
					}
				}
			}

			if (isDebug)showPyramid("Laplacian Pyramid Paris2011", LaplacianPyramid);
			collapseLaplacianPyramid(LaplacianPyramid, ImageStack[0]);//override srcf for saving memory	

			ImageStack[0].convertTo(dest, src.type());
			dest(Rect(r, r, src_.cols, src_.rows)).copyTo(dest_);
		}
	}

	void LocalMultiScaleFilterFull::dog(const Mat& src, Mat& dest)
	{
		initRangeTable(sigma_range, boost);

		Mat srcf;
		if (src.depth() == CV_32F)
		{
			srcf = src;
		}
		else
		{
			src.convertTo(srcf, CV_32F);
		}

		const float sigma_lmax = (float)getPyramidSigma(sigma_space, level);
		const int rmax = (int)ceil(sigma_lmax * 3.f);
		const Size ksizemax(2 * rmax + 1, 2 * rmax + 1);

		const int r_pad = (int)pow(2, level + 1);//2^(level+1)
		vector<Mat>  LaplacianStack(level + 1);

		//(1) build Gaussian stack
		buildGaussianStack(srcf, ImageStack, sigma_space, level);

		for (int i = 0; i < level; i++)
		{
			LaplacianStack[i].create(ImageStack[0].size(), CV_32F);
		}
		Mat im;
		copyMakeBorder(srcf, im, rmax, rmax, rmax, rmax, BORDER_DEFAULT);

		//(2) build DoG stack (0 to level-1)
		for (int l = 0; l < level; l++)
		{
			const float sigma_l = (float)getPyramidSigma(sigma_space, l);
			const float sigma_lp = (float)getPyramidSigma(sigma_space, l + 1);
			const int r = (int)ceil(sigma_lp * 3.f);
			const Size ksize(2 * r + 1, 2 * r + 1);
			AutoBuffer<float> weight(ksize.area());
			AutoBuffer<int> index(ksize.area());
			setDoGKernel(weight, index, im.cols, ksize, sigma_l, sigma_lp);

#pragma omp parallel for schedule (dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				for (int i = 0; i < src.cols; i++)
				{
					const float g = ImageStack[0].at<float>(j, i);
					LaplacianStack[l].at<float>(j, i) = getDoGCoeffLn(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
					//if(l==0)LaplacianStack[l].at<float>(j, i) = getDoGCoeffLnNoremap(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
					//else LaplacianStack[l].at<float>(j, i) = getDoGCoeffLn(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
				}
			}
		}
		//(2) the last level is a copy of the last level DoG
		ImageStack[level].copyTo(LaplacianStack[level]);

		//(3) collapseDoG
		if (srcf.depth() == CV_32F)
		{
			collapseDoGStack(LaplacianStack, dest);
		}
		else
		{
			collapseDoGStack(LaplacianStack, srcf);//override srcf for saving memory	
			srcf.convertTo(dest, src.type());
		}
	}

	void LocalMultiScaleFilterFull::filter(const Mat& src, Mat& dest, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);
		this->pyramidComputeMethod = Fast;

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->level = level;
		this->boost = boost;
		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);

		freeSpaceWeight();
	}

	void LocalMultiScaleFilterFull::setDoGKernel(float* weight, int* index, const int index_step, Size ksize, const float sigma1, const float sigma2)
	{
		CV_Assert(sigma2 > sigma1);

		const int r = ksize.width / 2;
		int count = 0;
		if (sigma1 == 0.f)
		{
			float sum2 = 0.f;
			const float coeff2 = float(1.0 / (-2.0 * sigma2 * sigma2));
			for (int j = -r; j <= r; j++)
			{
				for (int i = -r; i <= r; i++)
				{
					const float dist = float(j * j + i * i);
					const float v2 = exp(dist * coeff2);
					weight[count] = v2;
					index[count] = j * index_step + i;
					count++;
					sum2 += v2;
				}
			}
			sum2 = 1.f / sum2;
			for (int i = 0; i < ksize.area(); i++)
			{
				weight[i] = 0.f - weight[i] * sum2;
			}
			weight[ksize.area() / 2] = 1.f + weight[ksize.area() / 2];
		}
		else
		{
			AutoBuffer<float> buff(ksize.area());
			float sum1 = 0.f;
			float sum2 = 0.f;
			const float coeff1 = float(1.0 / (-2.0 * sigma1 * sigma1));
			const float coeff2 = float(1.0 / (-2.0 * sigma2 * sigma2));
			for (int j = -r; j <= r; j++)
			{
				for (int i = -r; i <= r; i++)
				{
					float dist = float(j * j + i * i);
					float v1 = exp(dist * coeff1);
					float v2 = exp(dist * coeff2);
					weight[count] = v1;
					buff[count] = v2;
					index[count] = j * index_step + i;
					count++;
					sum1 += v1;
					sum2 += v2;
				}
			}
			sum1 = 1.f / sum1;
			sum2 = 1.f / sum2;
			for (int i = 0; i < ksize.area(); i++)
			{
				weight[i] = weight[i] * sum1 - buff[i] * sum2;
			}
		}
	}

	float LocalMultiScaleFilterFull::getDoGCoeffLnNoremap(Mat& src, const float g, const int y, const int x, const int size, int* index, float* weight)
	{
		float* sptr = src.ptr<float>(y, x);
		const int simd_size = get_simd_floor(size, 8);

		const __m256 mg = _mm256_set1_ps(g);
		__m256 msum = _mm256_setzero_ps();
		for (int i = 0; i < simd_size; i += 8)
		{
			__m256i idx = _mm256_load_si256((const __m256i*)index);
			const __m256 ms = _mm256_i32gather_ps(sptr, idx, sizeof(float));
			msum = _mm256_fmadd_ps(_mm256_load_ps(weight), ms, msum);
			weight += 8;
			index += 8;
		}
		float sum = _mm256_reduceadd_ps(msum);

		for (int i = simd_size; i < size; i++)
		{
			const float s = sptr[*index];
			sum += *weight * s;
			weight++;
			index++;
		}

		return sum;
	}

	float LocalMultiScaleFilterFull::getDoGCoeffLn(Mat& src, const float g, const int y, const int x, const int size, int* index, float* weight)
	{
		float* sptr = src.ptr<float>(y, x);
		float* rptr = &rangeTable[0];
		const int simd_size = get_simd_floor(size, 8);
		//cout << "size : " << size << endl;
		//cout << "simd_size : " << simd_size << endl;

		const __m256 mg = _mm256_set1_ps(g);
		__m256 msum = _mm256_setzero_ps();
		for (int i = 0; i < simd_size; i += 8)
		{
			__m256i idx = _mm256_load_si256((const __m256i*)index);
			const __m256 ms = _mm256_i32gather_ps(sptr, idx, sizeof(float));
			const __m256 subsg = _mm256_sub_ps(ms, mg);
			const __m256 md = _mm256_fnmadd_ps(_mm256_i32gather_ps(rptr, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), subsg, ms);
			msum = _mm256_fmadd_ps(_mm256_load_ps(weight), md, msum);
			weight += 8;
			index += 8;
		}
		float sum = _mm256_reduceadd_ps(msum);

		for (int i = simd_size; i < size; i++)
		{
			const float s = sptr[*index];
			//float d = s - (s - g) * range[saturate_cast<uchar>(abs(s - g))];
			float d = s - (s - g) * rangeTable[saturate_cast<uchar>(abs(s - g))];
			//float d = s - (s - g) * -exp((s - g) * (s - g) / (-2.0 * 30.f * 30.f));
			sum += *weight * d;
			weight++;
			index++;
		}

		return sum;
	}
#pragma endregion


#pragma region FastLLFReference

	float FastLLFReference::getTau(const int k)
	{
		const float delta = intensityRange / (order - 1);
		return float(k * delta + intensityMin);
	}

	void FastLLFReference::blendLaplacianLinear(const vector<vector<Mat>>& LaplacianPyramid, vector<Mat>& GaussianPyramid, vector<Mat>& destPyramid, const int order)
	{
		const int level = (int)GaussianPyramid.size();
		destPyramid.resize(level);
		AutoBuffer<const float*> lptr(order);
		for (int l = 0; l < level - 1; l++)
		{
			destPyramid[l].create(GaussianPyramid[l].size(), CV_32F);
			float* g = GaussianPyramid[l].ptr<float>();
			float* d = destPyramid[l].ptr<float>();
			for (int k = 0; k < order; k++)
			{
				lptr[k] = LaplacianPyramid[k][l].ptr<float>();
			}

			for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
			{
				float alpha;
				int high, low;
				getLinearIndex(g[i], low, high, alpha, order, intensityMin, intensityMax);
				d[i] = alpha * lptr[low][i] + (1.f - alpha) * lptr[high][i];
			}
		}
	}

	void FastLLFReference::pyramid(const Mat& src, Mat& dest)
	{
		pyramidComputeMethod = PyramidComputeMethod::Full;

		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);

		if (src.depth() == CV_32F) src.copyTo(GaussianPyramid[0]);
		else src.convertTo(GaussianPyramid[0], CV_32F);

		//(1) build Gaussian Pyramid
		buildGaussianPyramid(GaussianPyramid[0], GaussianPyramid, level, sigma_space);

		//(2) build Laplacian Pyramids
		LaplacianPyramidOrder.resize(order);
		for (int n = 0; n < order; n++)
		{
			LaplacianPyramidOrder[n].resize(level + 1);

			//(2)-1 Remap Input Image
			if (adaptiveMethod == AdaptiveMethod::FIX) remap(GaussianPyramid[0], LaplacianPyramidOrder[n][0], getTau(n), sigma_range, boost);
			else remapAdaptive(GaussianPyramid[0], LaplacianPyramidOrder[n][0], getTau(n), adaptiveSigmaMap[0], adaptiveBoostMap[0]);

			//(2)-2 Build Remapped Laplacian Pyramids
			buildLaplacianPyramid(LaplacianPyramidOrder[n][0], LaplacianPyramidOrder[n], level, sigma_space);
		}

		//(3) interpolate Laplacian Pyramid from Remapped Laplacian Pyramids
		blendLaplacianLinear(LaplacianPyramidOrder, GaussianPyramid, LaplacianPyramid, order);
		//set last level
		LaplacianPyramid[level] = GaussianPyramid[level];

		//(4) collapse Laplacian Pyramid
		collapseLaplacianPyramid(LaplacianPyramid, dest);
	}

	void FastLLFReference::filter(const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod, const int interpolationMethod)
	{
		allocSpaceWeight(sigma_space);

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->level = level;
		this->boost = boost;
		this->scalespaceMethod = scaleSpaceMethod;

		this->order = order;
		body(src, dest);

		freeSpaceWeight();
	}

#pragma endregion

#pragma region LocalMultiScaleFilterInterpolation

	void LocalMultiScaleFilterInterpolation::initRangeTableInteger(const float sigma, const float boost)
	{
		const int intensityRange2 = get_simd_ceil((int)intensityRange, order - 1);
		const int tableSize = intensityRange2 + 1;
		integerSampleTable = (float*)_mm_malloc(sizeof(float) * tableSize, AVX_ALIGN);
		int rem = intensityRange2 - (int)intensityRange;

		intensityRange = float(intensityRange2);
		intensityMax += (float)rem;
		for (int i = 0; i < tableSize; i++)
		{
			integerSampleTable[i] = getGaussianRangeWeight(float(i), sigma_range, boost);
		}
	}

	float LocalMultiScaleFilterInterpolation::getTau(const int k)
	{
#if 1
		const float delta = intensityRange / (order - 1);
		return float(k * delta + intensityMin);
#else
		const float intensityRange = float(intensityMax - intensityMin);
		const float delta = intensityRange / (order - 2);
		return float(k * delta + intensityMin - delta);
#endif
	}


	template<bool isUseTable, int D>
	void LocalMultiScaleFilterInterpolation::remapGaussDownIgnoreBoundary(const Mat& src, Mat& remapIm, Mat& dest, const float g, const float sigma_range, const float boost)
	{
		CV_Assert(src.depth() == CV_32F);
		const Size size = src.size();
		dest.create(size / 2, CV_32F);
		remapIm.create(size, CV_32F);

		//const int D = 2 * radius + 1;
		const int rs = radius >> 1;
		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}

		const int width = src.cols;
		const int height = src.rows;

#pragma region remap top
		const __m256 mg = _mm256_set1_ps(g);
		const float coeff = float(1.0 / (-2.0 * sigma_range * sigma_range));
		const __m256 mcoeff = _mm256_set1_ps(coeff);
		const __m256 mdetail = _mm256_set1_ps(boost);

		//splat
		{
			const float* sptr = src.ptr<float>();
			float* d = remapIm.ptr<float>();
			const int size = width * (D - 1);
			const int REMAPSIZE32 = get_simd_floor(size, 32);
			const int REMAPSIZE8 = get_simd_ceil(size, 8);
			if constexpr (isUseTable)
			{
				//float* rt = &rangeTable[0];
				float* rt = integerSampleTable;
				for (int i = 0; i < REMAPSIZE32; i += 32)
				{
					__m256 ms = _mm256_loadu_ps(sptr + i);
					__m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 8);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 16);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 24);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
				}
				for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
				{
					const __m256 ms = _mm256_loadu_ps(sptr + i);
					const __m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
				}
			}
			else
			{
				for (int i = 0; i < REMAPSIZE32; i += 32)
				{
					__m256 ms = _mm256_loadu_ps(sptr + i);
					__m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 8);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 16);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 24);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				}
				for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
				{
					const __m256 ms = _mm256_loadu_ps(sptr + i);
					const __m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				}
			}
		}

#pragma endregion

		const int linesize = src.cols;
		float* linebuff = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		//memset(linebuff, 0, sizeof(float) * linesize);

		const float* sptr = remapIm.ptr<float>();
		float* dptr = dest.ptr<float>(rs, rs);
		const int hend = width - 2 * radius;
		const int vend = height - 2 * radius;
		const int WIDTH = get_simd_floor(width, 8);

		const int HEND32 = get_simd_floor(hend, 32);
		const int HEND = get_simd_floor(hend, 8);
		const __m128i maskhend = get_storemask1(hend, 8);

		for (int j = 0; j < vend; j += 2)
		{
			//remap line
			{
				const float* sptr = src.ptr<float>(j + D - 1);
				float* d = remapIm.ptr<float>(j + D - 1);
				const int size = 2 * width;
				const int REMAPSIZE32 = get_simd_floor(size, 32);
				const int REMAPSIZE8 = get_simd_ceil(size, 8);
				if constexpr (isUseTable)
				{
					//float* rt = &rangeTable[0];
					float* rt = integerSampleTable;
					for (int i = 0; i < REMAPSIZE32; i += 32)
					{
						__m256 ms = _mm256_loadu_ps(sptr + i);
						__m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 8);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 16);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 24);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
					}
					for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
					{
						const __m256 ms = _mm256_loadu_ps(sptr + i);
						const __m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
					}
				}
				else
				{
					for (int i = 0; i < REMAPSIZE32; i += 32)
					{
						__m256 ms = _mm256_loadu_ps(sptr + i);
						__m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 8);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 16);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 24);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
					}
					for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
					{
						const __m256 ms = _mm256_loadu_ps(sptr + i);
						const __m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
					}
				}
			}
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* s = sptr + i;
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(s));
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(s), sum); s += width;
				}
				_mm256_storeu_ps(linebuff + i, sum);
			}
			for (int i = WIDTH; i < width; i++)
			{
				const float* s = sptr + i;
				float sum = GaussWeight[0] * *s;
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * *s;
					s += width;
				}
				linebuff[i] = sum;
			}
			sptr += 2 * width;

			//h filter
			for (int i = 0; i < HEND32; i += 32)
			{
				float* lb0 = linebuff + i;
				float* lb1 = linebuff + i + 8;
				float* lb2 = linebuff + i + 16;
				float* lb3 = linebuff + i + 24;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				__m256 sum1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb1++));
				__m256 sum2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb2++));
				__m256 sum3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb3++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
					sum1 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb1++), sum1);
					sum2 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb2++), sum2);
					sum3 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb3++), sum3);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum0));

				sum1 = _mm256_shuffle_ps(sum1, sum1, _MM_SHUFFLE(2, 0, 2, 0));
				sum1 = _mm256_permute4x64_ps(sum1, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 8) >> 1), _mm256_castps256_ps128(sum1));

				sum2 = _mm256_shuffle_ps(sum2, sum2, _MM_SHUFFLE(2, 0, 2, 0));
				sum2 = _mm256_permute4x64_ps(sum2, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 16) >> 1), _mm256_castps256_ps128(sum2));

				sum3 = _mm256_shuffle_ps(sum3, sum3, _MM_SHUFFLE(2, 0, 2, 0));
				sum3 = _mm256_permute4x64_ps(sum3, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 24) >> 1), _mm256_castps256_ps128(sum3));
			}
			for (int i = HEND32; i < HEND; i += 8)
			{
				float* lb0 = linebuff + i;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum0));
			}
#ifdef MASKSTORE
			//last
			{
				float* lb0 = linebuff + HEND;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_maskstore_ps(dptr + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum0));
			}
#else
			for (int i = HEND; i < hend; i += 2)
			{
				float sum = GaussWeight[0] * linebuff[i];
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * linebuff[i + k];
				}
				dptr[i >> 1] = sum;
			}
#endif
			dptr += dest.cols;
		}

		_mm_free(linebuff);
		_mm_free(W);
	}

	template<bool isUseTable>
	void LocalMultiScaleFilterInterpolation::remapGaussDownIgnoreBoundary(const Mat& src, Mat& remapIm, Mat& dest, const float g, const float sigma_range, const float boost)
	{
		CV_Assert(src.depth() == CV_32F);
		const Size size = src.size();
		dest.create(size / 2, CV_32F);
		remapIm.create(size, CV_32F);

		const int D = 2 * radius + 1;
		const int rs = radius >> 1;
		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}

		const int width = src.cols;
		const int height = src.rows;

#pragma region remap top
		const __m256 mg = _mm256_set1_ps(g);
		const float coeff = float(1.0 / (-2.0 * sigma_range * sigma_range));
		const __m256 mcoeff = _mm256_set1_ps(coeff);
		const __m256 mdetail = _mm256_set1_ps(boost);

		//splat
		{
			const float* sptr = src.ptr<float>();
			float* d = remapIm.ptr<float>();
			const int size = width * (D - 1);
			const int REMAPSIZE32 = get_simd_floor(size, 32);
			const int REMAPSIZE8 = get_simd_ceil(size, 8);
			if constexpr (isUseTable)
			{
				//float* rt = &rangeTable[0];
				float* rt = integerSampleTable;
				for (int i = 0; i < REMAPSIZE32; i += 32)
				{
					__m256 ms = _mm256_loadu_ps(sptr + i);
					__m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 8);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 16);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 24);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
				}
				for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
				{
					const __m256 ms = _mm256_loadu_ps(sptr + i);
					const __m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
				}
			}
			else
			{
				for (int i = 0; i < REMAPSIZE32; i += 32)
				{
					__m256 ms = _mm256_loadu_ps(sptr + i);
					__m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 8);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 16);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 24);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				}
				for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
				{
					const __m256 ms = _mm256_loadu_ps(sptr + i);
					const __m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				}
			}
		}

#pragma endregion

		const int linesize = src.cols;
		float* linebuff = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		//memset(linebuff, 0, sizeof(float) * linesize);

		const float* sptr = remapIm.ptr<float>();
		float* dptr = dest.ptr<float>(rs, rs);
		const int hend = width - 2 * radius;
		const int vend = height - 2 * radius;
		const int WIDTH = get_simd_floor(width, 8);

		const int HEND32 = get_simd_floor(hend, 32);
		const int HEND = get_simd_floor(hend, 8);
		const __m128i maskhend = get_storemask1(hend, 8);

		for (int j = 0; j < vend; j += 2)
		{
			//remap line
			{
				const float* sptr = src.ptr<float>(j + D - 1);
				float* d = remapIm.ptr<float>(j + D - 1);
				const int size = 2 * width;
				const int REMAPSIZE32 = get_simd_floor(size, 32);
				const int REMAPSIZE8 = get_simd_ceil(size, 8);
				if constexpr (isUseTable)
				{
					//float* rt = &rangeTable[0];
					float* rt = integerSampleTable;
					for (int i = 0; i < REMAPSIZE32; i += 32)
					{
						__m256 ms = _mm256_loadu_ps(sptr + i);
						__m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 8);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 16);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 24);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
					}
					for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
					{
						const __m256 ms = _mm256_loadu_ps(sptr + i);
						const __m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
					}
				}
				else
				{
					for (int i = 0; i < REMAPSIZE32; i += 32)
					{
						__m256 ms = _mm256_loadu_ps(sptr + i);
						__m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 8);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 16);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 24);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
					}
					for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
					{
						const __m256 ms = _mm256_loadu_ps(sptr + i);
						const __m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
					}
				}
			}
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* s = sptr + i;
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(s));
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(s), sum); s += width;
				}
				_mm256_storeu_ps(linebuff + i, sum);
			}
			for (int i = WIDTH; i < width; i++)
			{
				const float* s = sptr + i;
				float sum = GaussWeight[0] * *s;
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * *s;
					s += width;
				}
				linebuff[i] = sum;
			}
			sptr += 2 * width;

			//h filter
			for (int i = 0; i < HEND32; i += 32)
			{
				float* lb0 = linebuff + i;
				float* lb1 = linebuff + i + 8;
				float* lb2 = linebuff + i + 16;
				float* lb3 = linebuff + i + 24;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				__m256 sum1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb1++));
				__m256 sum2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb2++));
				__m256 sum3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb3++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
					sum1 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb1++), sum1);
					sum2 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb2++), sum2);
					sum3 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb3++), sum3);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum0));

				sum1 = _mm256_shuffle_ps(sum1, sum1, _MM_SHUFFLE(2, 0, 2, 0));
				sum1 = _mm256_permute4x64_ps(sum1, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 8) >> 1), _mm256_castps256_ps128(sum1));

				sum2 = _mm256_shuffle_ps(sum2, sum2, _MM_SHUFFLE(2, 0, 2, 0));
				sum2 = _mm256_permute4x64_ps(sum2, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 16) >> 1), _mm256_castps256_ps128(sum2));

				sum3 = _mm256_shuffle_ps(sum3, sum3, _MM_SHUFFLE(2, 0, 2, 0));
				sum3 = _mm256_permute4x64_ps(sum3, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 24) >> 1), _mm256_castps256_ps128(sum3));
			}
			for (int i = HEND32; i < HEND; i += 8)
			{
				float* lb0 = linebuff + i;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum0));
			}
#ifdef MASKSTORE
			//last
			{
				float* lb0 = linebuff + HEND;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_maskstore_ps(dptr + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum0));
			}
#else
			for (int i = HEND; i < hend; i += 2)
			{
				float sum = GaussWeight[0] * linebuff[i];
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * linebuff[i + k];
				}
				dptr[i >> 1] = sum;
			}
#endif
			dptr += dest.cols;
		}

		_mm_free(linebuff);
		_mm_free(W);
	}


	void LocalMultiScaleFilterInterpolation::remapAdaptiveGaussDownIgnoreBoundary(const Mat& src, Mat& remapIm, Mat& dest, const float g, const Mat& sigma_range, const Mat& boost)
	{
		CV_Assert(src.depth() == CV_32F);
		const Size size = src.size();
		dest.create(size / 2, CV_32F);
		remapIm.create(size, CV_32F);

		const int D = 2 * radius + 1;
		const int rs = radius >> 1;
		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}

		const int width = src.cols;
		const int height = src.rows;

#pragma region remap top
		const __m256 mg = _mm256_set1_ps(g);

		//splat
		{
			const float* sptr = src.ptr<float>();
			const float* asmap = sigma_range.ptr<float>();
			const float* abmap = boost.ptr<float>();
			float* d = remapIm.ptr<float>();
			const int SIZE = get_simd_ceil(width * (D - 1), 8);

			for (int i = 0; i < SIZE; i += 8)
			{
				const __m256 msgma = _mm256_loadu_ps(asmap + i);
				__m256 mcoeff = _mm256_rcpnr_ps(_mm256_mul_ps(_mm256_set1_ps(-2.f), _mm256_mul_ps(msgma, msgma)));
				const __m256 mdetail = _mm256_loadu_ps(abmap + i);
				__m256 ms = _mm256_loadu_ps(sptr + i);
				__m256 subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
			}
		}

#pragma endregion

		const int linesize = src.cols;
		float* linebuff = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		memset(linebuff, 0, sizeof(float) * linesize);

		const float* sptr = remapIm.ptr<float>();
		float* dptr = dest.ptr<float>(rs, rs);
		const int hend = width - 2 * radius;
		const int vend = height - 2 * radius;
		const int WIDTH = get_simd_floor(width, 8);
		const int HEND = get_simd_floor(hend, 8);

		for (int j = 0; j < vend; j += 2)
		{
			//remap line
			{
				const float* sptr = src.ptr<float>(j + D - 1);
				const float* asmap = sigma_range.ptr<float>(j + D - 1);
				const float* abmap = boost.ptr<float>(j + D - 1);
				float* d = remapIm.ptr<float>(j + D - 1);
				const int SIZE = get_simd_floor(width * 2, 8);
				for (int i = 0; i < SIZE; i += 8)
				{
					const __m256 msgma = _mm256_loadu_ps(asmap + i);
					const __m256 mcoeff = _mm256_rcpnr_ps(_mm256_mul_ps(_mm256_set1_ps(-2.f), _mm256_mul_ps(msgma, msgma)));
					const __m256 mdetail = _mm256_loadu_ps(abmap + i);
					__m256 ms = _mm256_loadu_ps(sptr + i);
					__m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				}
				for (int i = SIZE; i < width * 2; i++)
				{
					const float sigma = asmap[i];
					const float coeff = 1.f / (-2.f * sigma * sigma);
					const float detail = abmap[i];
					float s = sptr[i];
					float subsg = s - g;
					d[i] = subsg * (detail * exp(subsg * subsg * coeff)) + s;
				}
			}
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* s = sptr + i;
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(s));
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(s), sum); s += width;
				}
				_mm256_storeu_ps(linebuff + i, sum);
			}
			for (int i = WIDTH; i < width; i++)
			{
				const float* s = sptr + i;
				float sum = GaussWeight[0] * *s;
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * *s;
					s += width;
				}
				linebuff[i] = sum;
			}
			sptr += 2 * width;

			//h filter
			for (int i = 0; i < HEND; i += 8)
			{
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(linebuff + i));
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(linebuff + i + k), sum);
				}
				sum = _mm256_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 0, 2, 0));
				sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum));
			}
			for (int i = HEND; i < hend; i += 2)
			{
				float sum = GaussWeight[0] * linebuff[i];
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * linebuff[i + k];
				}
				dptr[i >> 1] = sum;
			}
			dptr += dest.cols;
		}

		_mm_free(linebuff);
		_mm_free(W);
	}


	template<bool isInit, int interpolation, int D2>
	void LocalMultiScaleFilterInterpolation::GaussUpSubProductSumIgnoreBoundary(const Mat& src, const cv::Mat& subsrc, const Mat& GaussianPyramid, Mat& dest, const float g)
	{
		CV_Assert(src.depth() == CV_32F);
		dest.create(src.size() * 2, src.type());

		__m256* GW = (__m256*)_mm_malloc(sizeof(__m256) * (2 * radius + 1), AVX_ALIGN);
		for (int i = 0; i < 2 * radius + 1; i++)
		{
			GW[i] = _mm256_set1_ps(GaussWeight[i]);
		}
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const int rs = radius >> 1;
		const int D = 2 * rs + 1;
		//const int D2 = 2 * D;

		const int step = src.cols;

		float* linebuff = (float*)_mm_malloc(sizeof(float) * (src.cols * 2 + 8), AVX_ALIGN);
		float* linee = linebuff;
		float* lineo = linebuff + src.cols;

		const int hend = src.cols - 2 * rs;
		const int HEND8 = get_simd_floor(hend, 8);
		const int WIDTH32 = get_simd_floor(src.cols, 32);
		const int WIDTH8 = get_simd_floor(src.cols, 8);
		const __m256i maskwidth = get_simd_residualmask_epi32(src.cols);

		__m256i maskhendL, maskhendR;
		get_storemask2(hend, maskhendL, maskhendR, 8);

		const float delta = intensityRange / (order - 1);
		const float idelta = 1.f / delta;
		const __m256 mg = _mm256_set1_ps(g);
		const __m256 mgmax = _mm256_set1_ps(intensityMax - delta);
		const __m256 mgmin = _mm256_set1_ps(intensityMin + delta);
		const __m256 midelta = _mm256_set1_ps(idelta);
		const __m256 mcubicalpha = _mm256_set1_ps(cubicAlpha);
		const __m256 mone = _mm256_set1_ps(1.f);
		const __m256 mtwo = _mm256_set1_ps(2.f);
		const __m256 mtwoalpha = _mm256_set1_ps(2.f + cubicAlpha);
		const __m256 mnthreealpha = _mm256_set1_ps(-(3.f + cubicAlpha));
		const __m256 mmfouralpha = _mm256_set1_ps(-4.f * cubicAlpha);
		const __m256 mmfivealpha = _mm256_set1_ps(-5.f * cubicAlpha);
		const __m256 meightalpha = _mm256_set1_ps(8.f * cubicAlpha);

		for (int j = radius; j < dest.rows - radius; j += 2)
		{
			const float* sptr = src.ptr<float>((j - radius) >> 1);
			//v filter
			for (int i = 0; i < WIDTH32; i += 32)
			{
				const float* si = sptr + i;
				__m256 sume0 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si));
				__m256 sumo0 = _mm256_setzero_ps();
				__m256 sume1 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 8));
				__m256 sumo1 = _mm256_setzero_ps();
				__m256 sume2 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 16));
				__m256 sumo2 = _mm256_setzero_ps();
				__m256 sume3 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 24));
				__m256 sumo3 = _mm256_setzero_ps();
				si += step;
				for (int k = 2; k < D2; k += 2)
				{
					__m256 ms = _mm256_loadu_ps(si);
					sume0 = _mm256_fmadd_ps(GW[k], ms, sume0);
					sumo0 = _mm256_fmadd_ps(GW[k - 1], ms, sumo0);

					ms = _mm256_loadu_ps(si + 8);
					sume1 = _mm256_fmadd_ps(GW[k], ms, sume1);
					sumo1 = _mm256_fmadd_ps(GW[k - 1], ms, sumo1);

					ms = _mm256_loadu_ps(si + 16);
					sume2 = _mm256_fmadd_ps(GW[k], ms, sume2);
					sumo2 = _mm256_fmadd_ps(GW[k - 1], ms, sumo2);

					ms = _mm256_loadu_ps(si + 24);
					sume3 = _mm256_fmadd_ps(GW[k], ms, sume3);
					sumo3 = _mm256_fmadd_ps(GW[k - 1], ms, sumo3);

					si += step;
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume0, mevenratio));
				_mm256_storeu_ps(linee + i + 8, _mm256_mul_ps(sume1, mevenratio));
				_mm256_storeu_ps(linee + i + 16, _mm256_mul_ps(sume2, mevenratio));
				_mm256_storeu_ps(linee + i + 24, _mm256_mul_ps(sume3, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo0, moddratio));
				_mm256_storeu_ps(lineo + i + 8, _mm256_mul_ps(sumo1, moddratio));
				_mm256_storeu_ps(lineo + i + 16, _mm256_mul_ps(sumo2, moddratio));
				_mm256_storeu_ps(lineo + i + 24, _mm256_mul_ps(sumo3, moddratio));
			}
			for (int i = WIDTH32; i < WIDTH8; i += 8)
			{
				const float* si = sptr + i;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo, moddratio));
			}
#ifdef MASKSTORE
			{
				const float* si = sptr + WIDTH8;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_maskstore_ps(linee + WIDTH8, maskwidth, _mm256_mul_ps(sume, mevenratio));
				_mm256_maskstore_ps(lineo + WIDTH8, maskwidth, _mm256_mul_ps(sumo, moddratio));
			}
#else
			for (int i = WIDTH8; i < src.cols; i++)
			{
				const float* si = sptr + i;
				float sume = GaussWeight[0] * *si; si += step;
				float sumo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sume += GaussWeight[K] * *si;
					sumo += GaussWeight[K - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}
#endif

			// h filter
			float* deptr = dest.ptr<float>(j, radius);
			float* doptr = dest.ptr<float>(j + 1, radius);
			const float* gpye = GaussianPyramid.ptr<float>(j, radius);
			const float* gpyo = GaussianPyramid.ptr<float>(j + 1, radius);
			const float* daeptr = subsrc.ptr<float>(j, radius);
			const float* daoptr = subsrc.ptr<float>(j + 1, radius);

			for (int i = 0; i < HEND8; i += 8)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}

				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);

				__m256 w;
				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(deptr + 2 * i + 0, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0))));
				else _mm256_storeu_ps(deptr + 2 * i + 0, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)), _mm256_loadu_ps(deptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(deptr + 2 * i + 8, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8))));
				else _mm256_storeu_ps(deptr + 2 * i + 8, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)), _mm256_loadu_ps(deptr + 2 * i + 8)));

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(doptr + 2 * i + 0, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0))));
				else _mm256_storeu_ps(doptr + 2 * i + 0, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)), _mm256_loadu_ps(doptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(doptr + 2 * i + 8, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8))));
				else _mm256_storeu_ps(doptr + 2 * i + 8, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)), _mm256_loadu_ps(doptr + 2 * i + 8)));
			}
#ifdef MASKSTORELASTLINEAR
			if (HEND8 != hend)
			{
				const int i = HEND8;
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}

				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);

				__m256 w;
				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(deptr + 2 * i + 0, maskhendL, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0))));
				else _mm256_maskstore_ps(deptr + 2 * i + 0, maskhendL, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)), _mm256_loadu_ps(deptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(deptr + 2 * i + 8, maskhendR, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8))));
				else _mm256_maskstore_ps(deptr + 2 * i + 8, maskhendR, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)), _mm256_loadu_ps(deptr + 2 * i + 8)));

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(doptr + 2 * i + 0, maskhendL, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0))));
				else _mm256_maskstore_ps(doptr + 2 * i + 0, maskhendL, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)), _mm256_loadu_ps(doptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(doptr + 2 * i + 8, maskhendR, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8))));
				else _mm256_maskstore_ps(doptr + 2 * i + 8, maskhendR, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)), _mm256_loadu_ps(doptr + 2 * i + 8)));
			}
#else
			for (int i = HEND8; i < hend; i++)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				float sumee = GaussWeight[0] * *sie++;
				float sumoe = 0.f;
				float sumeo = GaussWeight[0] * *sio++;
				float sumoo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sumee += GaussWeight[K] * *sie;
					sumoe += GaussWeight[K - 1] * *sie++;
					sumeo += GaussWeight[K] * *sio;
					sumoo += GaussWeight[K - 1] * *sio++;
				}
				const int I = i << 1;
				float w;

				if constexpr (interpolation == 0) w = (abs((gpye[I + 0] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpye[I + 0] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpye[I + 0];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) deptr[I + 0] = w * (daeptr[I + 0] - sumee * evenratio);
				else deptr[I + 0] += w * (daeptr[I + 0] - sumee * evenratio);

				if constexpr (interpolation == 0) w = (abs((gpye[I + 1] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpye[I + 1] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpye[I + 1];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) deptr[I + 1] = w * (daeptr[I + 1] - sumoe * oddratio);
				else deptr[I + 1] += w * (daeptr[I + 1] - sumoe * oddratio);

				if constexpr (interpolation == 0) w = (abs((gpyo[I + 0] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpyo[I + 0] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpyo[I + 0];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) doptr[I + 0] = w * (daoptr[I + 0] - sumeo * evenratio);
				else doptr[I + 0] += w * (daoptr[I + 0] - sumeo * evenratio);

				if constexpr (interpolation == 0) w = (abs((gpyo[I + 1] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpyo[I + 1] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpyo[I + 1];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) doptr[I + 1] = w * (daoptr[I + 1] - sumoo * oddratio);
				else doptr[I + 1] += w * (daoptr[I + 1] - sumoo * oddratio);
			}
#endif
		}

		_mm_free(linebuff);
		_mm_free(GW);
	}

	template<bool isInit, int interpolation>
	void LocalMultiScaleFilterInterpolation::GaussUpSubProductSumIgnoreBoundary(const Mat& src, const cv::Mat& subsrc, const Mat& GaussianPyramid, Mat& dest, const float g)
	{
		CV_Assert(src.depth() == CV_32F);
		dest.create(src.size() * 2, src.type());

		__m256* GW = (__m256*)_mm_malloc(sizeof(__m256) * (2 * radius + 1), AVX_ALIGN);
		for (int i = 0; i < 2 * radius + 1; i++)
		{
			GW[i] = _mm256_set1_ps(GaussWeight[i]);
		}
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const int rs = radius >> 1;
		const int D = 2 * rs + 1;
		const int D2 = 2 * D;

		const int step = src.cols;

		float* linebuff = (float*)_mm_malloc(sizeof(float) * (src.cols * 2 + 8), AVX_ALIGN);
		float* linee = linebuff;
		float* lineo = linebuff + src.cols;

		const int hend = src.cols - 2 * rs;
		const int HEND8 = get_simd_floor(hend, 8);
		const int WIDTH32 = get_simd_floor(src.cols, 32);
		const int WIDTH8 = get_simd_floor(src.cols, 8);
		const __m256i maskwidth = get_simd_residualmask_epi32(src.cols);

		__m256i maskhendL, maskhendR;
		get_storemask2(hend, maskhendL, maskhendR, 8);

		const float delta = intensityRange / (order - 1);
		const float idelta = 1.f / delta;
		const __m256 mg = _mm256_set1_ps(g);
		const __m256 mgmax = _mm256_set1_ps(intensityMax - delta);
		const __m256 mgmin = _mm256_set1_ps(intensityMin + delta);
		const __m256 midelta = _mm256_set1_ps(idelta);
		const __m256 mcubicalpha = _mm256_set1_ps(cubicAlpha);
		const __m256 mone = _mm256_set1_ps(1.f);
		const __m256 mtwo = _mm256_set1_ps(2.f);
		const __m256 mtwoalpha = _mm256_set1_ps(2.f + cubicAlpha);
		const __m256 mnthreealpha = _mm256_set1_ps(-(3.f + cubicAlpha));
		const __m256 mmfouralpha = _mm256_set1_ps(-4.f * cubicAlpha);
		const __m256 mmfivealpha = _mm256_set1_ps(-5.f * cubicAlpha);
		const __m256 meightalpha = _mm256_set1_ps(8.f * cubicAlpha);

		for (int j = radius; j < dest.rows - radius; j += 2)
		{
			const float* sptr = src.ptr<float>((j - radius) >> 1);
			//v filter
			for (int i = 0; i < WIDTH32; i += 32)
			{
				const float* si = sptr + i;
				__m256 sume0 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si));
				__m256 sumo0 = _mm256_setzero_ps();
				__m256 sume1 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 8));
				__m256 sumo1 = _mm256_setzero_ps();
				__m256 sume2 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 16));
				__m256 sumo2 = _mm256_setzero_ps();
				__m256 sume3 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 24));
				__m256 sumo3 = _mm256_setzero_ps();
				si += step;
				for (int k = 2; k < D2; k += 2)
				{
					__m256 ms = _mm256_loadu_ps(si);
					sume0 = _mm256_fmadd_ps(GW[k], ms, sume0);
					sumo0 = _mm256_fmadd_ps(GW[k - 1], ms, sumo0);

					ms = _mm256_loadu_ps(si + 8);
					sume1 = _mm256_fmadd_ps(GW[k], ms, sume1);
					sumo1 = _mm256_fmadd_ps(GW[k - 1], ms, sumo1);

					ms = _mm256_loadu_ps(si + 16);
					sume2 = _mm256_fmadd_ps(GW[k], ms, sume2);
					sumo2 = _mm256_fmadd_ps(GW[k - 1], ms, sumo2);

					ms = _mm256_loadu_ps(si + 24);
					sume3 = _mm256_fmadd_ps(GW[k], ms, sume3);
					sumo3 = _mm256_fmadd_ps(GW[k - 1], ms, sumo3);

					si += step;
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume0, mevenratio));
				_mm256_storeu_ps(linee + i + 8, _mm256_mul_ps(sume1, mevenratio));
				_mm256_storeu_ps(linee + i + 16, _mm256_mul_ps(sume2, mevenratio));
				_mm256_storeu_ps(linee + i + 24, _mm256_mul_ps(sume3, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo0, moddratio));
				_mm256_storeu_ps(lineo + i + 8, _mm256_mul_ps(sumo1, moddratio));
				_mm256_storeu_ps(lineo + i + 16, _mm256_mul_ps(sumo2, moddratio));
				_mm256_storeu_ps(lineo + i + 24, _mm256_mul_ps(sumo3, moddratio));
			}
			for (int i = WIDTH32; i < WIDTH8; i += 8)
			{
				const float* si = sptr + i;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo, moddratio));
			}
#ifdef MASKSTORE
			{
				const float* si = sptr + WIDTH8;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_maskstore_ps(linee + WIDTH8, maskwidth, _mm256_mul_ps(sume, mevenratio));
				_mm256_maskstore_ps(lineo + WIDTH8, maskwidth, _mm256_mul_ps(sumo, moddratio));
			}
#else
			for (int i = WIDTH8; i < src.cols; i++)
			{
				const float* si = sptr + i;
				float sume = GaussWeight[0] * *si; si += step;
				float sumo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sume += GaussWeight[K] * *si;
					sumo += GaussWeight[K - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}
#endif

			// h filter
			float* deptr = dest.ptr<float>(j, radius);
			float* doptr = dest.ptr<float>(j + 1, radius);
			const float* gpye = GaussianPyramid.ptr<float>(j, radius);
			const float* gpyo = GaussianPyramid.ptr<float>(j + 1, radius);
			const float* daeptr = subsrc.ptr<float>(j, radius);
			const float* daoptr = subsrc.ptr<float>(j + 1, radius);

			for (int i = 0; i < HEND8; i += 8)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}

				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);

				__m256 w;
				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(deptr + 2 * i + 0, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0))));
				else _mm256_storeu_ps(deptr + 2 * i + 0, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)), _mm256_loadu_ps(deptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(deptr + 2 * i + 8, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8))));
				else _mm256_storeu_ps(deptr + 2 * i + 8, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)), _mm256_loadu_ps(deptr + 2 * i + 8)));

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(doptr + 2 * i + 0, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0))));
				else _mm256_storeu_ps(doptr + 2 * i + 0, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)), _mm256_loadu_ps(doptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(doptr + 2 * i + 8, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8))));
				else _mm256_storeu_ps(doptr + 2 * i + 8, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)), _mm256_loadu_ps(doptr + 2 * i + 8)));
			}
#ifdef MASKSTORE0
			//last
			{
				const int i = HEND8;
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}

				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);

				__m256 w;
				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(deptr + 2 * i + 0, maskhendL, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0))));
				else _mm256_maskstore_ps(deptr + 2 * i + 0, maskhendL, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)), _mm256_loadu_ps(deptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(deptr + 2 * i + 8, maskhendR, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8))));
				else _mm256_maskstore_ps(deptr + 2 * i + 8, maskhendR, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)), _mm256_loadu_ps(deptr + 2 * i + 8)));

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(doptr + 2 * i + 0, maskhendL, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0))));
				else _mm256_maskstore_ps(doptr + 2 * i + 0, maskhendL, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)), _mm256_loadu_ps(doptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(doptr + 2 * i + 8, maskhendR, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8))));
				else _mm256_maskstore_ps(doptr + 2 * i + 8, maskhendR, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)), _mm256_loadu_ps(doptr + 2 * i + 8)));
			}
#else
			for (int i = HEND8; i < hend; i++)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				float sumee = GaussWeight[0] * *sie++;
				float sumoe = 0.f;
				float sumeo = GaussWeight[0] * *sio++;
				float sumoo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sumee += GaussWeight[K] * *sie;
					sumoe += GaussWeight[K - 1] * *sie++;
					sumeo += GaussWeight[K] * *sio;
					sumoo += GaussWeight[K - 1] * *sio++;
				}
				const int I = i << 1;
				float w;

				if constexpr (interpolation == 0) w = (abs((gpye[I + 0] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpye[I + 0] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpye[I + 0];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) deptr[I + 0] = w * (daeptr[I + 0] - sumee * evenratio);
				else deptr[I + 0] += w * (daeptr[I + 0] - sumee * evenratio);

				if constexpr (interpolation == 0) w = (abs((gpye[I + 1] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpye[I + 1] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpye[I + 1];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) deptr[I + 1] = w * (daeptr[I + 1] - sumoe * oddratio);
				else deptr[I + 1] += w * (daeptr[I + 1] - sumoe * oddratio);

				if constexpr (interpolation == 0) w = (abs((gpyo[I + 0] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpyo[I + 0] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpyo[I + 0];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) doptr[I + 0] = w * (daoptr[I + 0] - sumeo * evenratio);
				else doptr[I + 0] += w * (daoptr[I + 0] - sumeo * evenratio);

				if constexpr (interpolation == 0) w = (abs((gpyo[I + 1] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpyo[I + 1] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpyo[I + 1];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) doptr[I + 1] = w * (daoptr[I + 1] - sumoo * oddratio);
				else doptr[I + 1] += w * (daoptr[I + 1] - sumoo * oddratio);
			}
#endif
		}

		_mm_free(linebuff);
		_mm_free(GW);
	}


	//for parallel
	void LocalMultiScaleFilterInterpolation::buildRemapLaplacianPyramidEachOrder(const Mat& src, vector<Mat>& destPyramid, const int level, const float sigma, const float g, const float sigma_range, const float boost)
	{
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);
		//destPyramid[0].create(src.size(), CV_32F);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			remapGaussDownIgnoreBoundary<false>(src, destPyramid[0], destPyramid[1], g, sigma_range, boost);
			GaussUpAddIgnoreBoundary <false>(destPyramid[1], destPyramid[0], destPyramid[0]);
			for (int l = 1; l < level; l++)
			{
				GaussDownIgnoreBoundary(destPyramid[l], destPyramid[l + 1]);
				GaussUpAddIgnoreBoundary <false>(destPyramid[l + 1], destPyramid[l], destPyramid[l]);
			}
		}
		else if (pyramidComputeMethod == Fast)
		{
			cout << "not supported: buildRemapLaplacianPyramid" << endl;
			GaussDown(src, destPyramid[1]);
			GaussUpAdd<false>(destPyramid[1], src, destPyramid[0]);
			for (int l = 1; l < level; l++)
			{
				GaussDown(destPyramid[l], destPyramid[l + 1]);
				GaussUpAdd <false>(destPyramid[l + 1], destPyramid[l], destPyramid[l]);
			}
		}
		else if (pyramidComputeMethod == Full)
		{
			cout << "not supported: buildRemapLaplacianPyramid" << endl;
			GaussDownFull(src, destPyramid[1], sigma, borderType);
			GaussUpAddFull<false>(destPyramid[1], src, destPyramid[0], sigma, borderType);
			for (int l = 1; l < level; l++)
			{
				GaussDownFull(destPyramid[l], destPyramid[l + 1], sigma, borderType);
				GaussUpAddFull<false>(destPyramid[l + 1], destPyramid[l], destPyramid[l], sigma, borderType);
			}
		}
		else if (pyramidComputeMethod == OpenCV)
		{
			cout << "not supported: buildRemapLaplacianPyramid" << endl;
			buildPyramid(src, destPyramid, level, borderType);
			for (int i = 0; i < level; i++)
			{
				Mat temp;
				pyrUp(destPyramid[i + 1], temp, destPyramid[i].size(), borderType);
				subtract(destPyramid[i], temp, destPyramid[i]);
			}
		}
	}

	//for serial
	template<bool isInit>
	void LocalMultiScaleFilterInterpolation::buildRemapLaplacianPyramid(const std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& LaplacianPyramid, vector<Mat>& destPyramid, const int level, const float sigma, const float g, const float sigma_range, const float boost)
	{
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);
		if (LaplacianPyramid.size() != level + 1) LaplacianPyramid.resize(level + 1);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			if (isUseTable)
			{
				if (radius == 2) remapGaussDownIgnoreBoundary<true, 5>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
				else if (radius == 4) remapGaussDownIgnoreBoundary<true, 9>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
				else remapGaussDownIgnoreBoundary<true>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
			}
			else
			{
				if (radius == 2) remapGaussDownIgnoreBoundary<false, 5>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
				else if (radius == 4) remapGaussDownIgnoreBoundary<false, 9>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
				else remapGaussDownIgnoreBoundary<false>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
			}

			//const int rs = radius >> 1;
			//const int D = 2 * rs + 1;
			//const int D2 = 2 * D;
			if (interpolation_method == 0)
			{
				if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 0, 6>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 0, 10>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else GaussUpSubProductSumIgnoreBoundary<isInit, 0>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			}
			if (interpolation_method == 1)
			{
				if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 1, 6>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 1, 10>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else GaussUpSubProductSumIgnoreBoundary<isInit, 1>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			}
			if (interpolation_method == 2)
			{
				if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 2, 6>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 2, 10>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else GaussUpSubProductSumIgnoreBoundary<isInit, 2>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			}

			float* linebuff = (float*)_mm_malloc(sizeof(float) * LaplacianPyramid[1].cols, AVX_ALIGN);
			for (int l = 1; l < level; l++)
			{
				if (radius == 2)  GaussDownIgnoreBoundary<5>(LaplacianPyramid[l], LaplacianPyramid[l + 1], linebuff);
				else if (radius == 4) GaussDownIgnoreBoundary<9>(LaplacianPyramid[l], LaplacianPyramid[l + 1], linebuff);
				else GaussDownIgnoreBoundary(LaplacianPyramid[l], LaplacianPyramid[l + 1]);

				if (interpolation_method == 0)
				{
					if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 0, 6>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 0, 10>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else  GaussUpSubProductSumIgnoreBoundary<isInit, 0>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
				}
				if (interpolation_method == 1)
				{
					if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 1, 6>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 1, 10>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else GaussUpSubProductSumIgnoreBoundary<isInit, 1>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
				}
				if (interpolation_method == 2)
				{
					if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 2, 6>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 2, 10>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else GaussUpSubProductSumIgnoreBoundary<isInit, 2>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
				}
			}
			_mm_free(linebuff);
		}
	}

	template<bool isInit>
	void LocalMultiScaleFilterInterpolation::buildRemapAdaptiveLaplacianPyramid(const std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& LaplacianPyramid, vector<Mat>& destPyramid, const int level, const float sigma, const float g, const Mat& sigma_range, const Mat& boost)
	{
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);
		if (LaplacianPyramid.size() != level + 1) LaplacianPyramid.resize(level + 1);
		//destPyramid[0].create(src.size(), CV_32F);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			remapAdaptiveGaussDownIgnoreBoundary(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
			if (interpolation_method == 0) GaussUpSubProductSumIgnoreBoundary<isInit, 0>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			if (interpolation_method == 1) GaussUpSubProductSumIgnoreBoundary<isInit, 1>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			if (interpolation_method == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 2>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			for (int l = 1; l < level; l++)
			{
				GaussDownIgnoreBoundary(LaplacianPyramid[l], LaplacianPyramid[l + 1]);
				if (interpolation_method == 0) GaussUpSubProductSumIgnoreBoundary<isInit, 0>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
				if (interpolation_method == 1) GaussUpSubProductSumIgnoreBoundary<isInit, 1>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
				if (interpolation_method == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 2>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
			}
		}
	}


	void LocalMultiScaleFilterInterpolation::blendLaplacianNearest(const vector<vector<Mat>>& LaplacianPyramid, vector<Mat>& GaussianPyramid, vector<Mat>& destPyramid, const int order)//order is no effect now
	{
		const int level = (int)GaussianPyramid.size();
		if (order == 256)
		{
			for (int l = 0; l < level - 1; l++)
			{
				float* s = GaussianPyramid[l].ptr<float>();
				float* d = destPyramid[l].ptr<float>();
				for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
				{
					const int c = saturate_cast<uchar>(s[i]);
					d[i] = LaplacianPyramid[c][l].at<float>(i);
				}
			}
		}
		else
		{
			AutoBuffer<const float*> lptr(order);
			const float idelta = (order - 1) / intensityRange;
			for (int l = 0; l < level - 1; l++)
			{
				float* g = GaussianPyramid[l].ptr<float>();
				float* d = destPyramid[l].ptr<float>();
				for (int k = 0; k < order; k++)
				{
					lptr[k] = LaplacianPyramid[k][l].ptr<float>();
				}
				if (isParallel)
				{
#pragma omp parallel for //schedule (dynamic)
					for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
					{
						const int c = min(order - 1, (int)saturate_cast<uchar>((g[i] - intensityMin) * idelta));
						//const int c = min(order - 1, int(g[i] * istep+0.5));
						d[i] = lptr[c][i];
					}
				}
				else
				{
					for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
					{
						const int c = min(order - 1, (int)saturate_cast<uchar>((g[i] - intensityMin) * idelta));
						d[i] = lptr[c][i];
					}
				}
			}
		}
	}

	void LocalMultiScaleFilterInterpolation::blendLaplacianLinear(const vector<vector<Mat>>& LaplacianPyramid, vector<Mat>& GaussianPyramid, vector<Mat>& destPyramid, const int order)
	{
		const int level = (int)GaussianPyramid.size();
		AutoBuffer<const float*> lptr(order);
		for (int l = 0; l < level - 1; l++)
		{
			float* g = GaussianPyramid[l].ptr<float>();
			float* d = destPyramid[l].ptr<float>();
			for (int k = 0; k < order; k++)
			{
				lptr[k] = LaplacianPyramid[k][l].ptr<float>();
			}

			if (isParallel)
			{
#pragma omp parallel for //schedule (dynamic)
				for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
				{
					float alpha;
					int high, low;
					getLinearIndex(g[i], low, high, alpha, order, intensityMin, intensityMax);
					d[i] = alpha * lptr[low][i] + (1.f - alpha) * lptr[high][i];
				}
			}
			else
			{
				for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
				{
					float alpha;
					int high, low;
					getLinearIndex(g[i], low, high, alpha, order, intensityMin, intensityMax);
					d[i] = alpha * lptr[low][i] + (1.f - alpha) * lptr[high][i];
				}
			}
		}
	}

	void LocalMultiScaleFilterInterpolation::blendLaplacianCubic(const vector<vector<Mat>>& LaplacianPyramid, vector<Mat>& GaussianPyramid, vector<Mat>& destPyramid, const int order)
	{
		const int level = (int)GaussianPyramid.size();
		AutoBuffer<const float*> lptr(order);
		for (int l = 0; l < level - 1; l++)
		{
			float* g = GaussianPyramid[l].ptr<float>();
			float* d = destPyramid[l].ptr<float>();
			for (int k = 0; k < order; k++)
			{
				lptr[k] = LaplacianPyramid[k][l].ptr<float>();
			}
			if (isParallel)
			{
#pragma omp parallel for //schedule (dynamic)
				for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
				{
					d[i] = getCubicInterpolation(g[i], order, lptr, i, cubicAlpha, intensityMin, intensityMax);
				}
			}
			else
			{
				for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
				{
					d[i] = getCubicInterpolation(g[i], order, lptr, i, cubicAlpha, intensityMin, intensityMax);
				}
			}
		}
	}

	template<int interpolation>
	void LocalMultiScaleFilterInterpolation::productSumLaplacianPyramid(const std::vector<cv::Mat>& LaplacianPyramid, std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int order, const float g)
	{
		const int level = (int)GaussianPyramid.size();

		for (int l = 0; l < level - 1; l++)
		{
			const float* lpy = LaplacianPyramid[l].ptr<float>();
			float* gpy = GaussianPyramid[l].ptr<float>();
			float* d = destPyramid[l].ptr<float>();

			//const float delta = intensityRange / (order - 2);
			const float delta = intensityRange / (order - 1);
			const float idelta = 1.f / delta;
			if (isParallel)
			{
#pragma omp parallel for //schedule (dynamic)
				for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
				{
					;
				}
			}
			else
			{
				//__m256 milinearstep = _mm256_set1_ps(istep);
				//__m256 mlinearstepk = _mm256_set1_ps(g);
				/*for (int i = 0; i < GaussianPyramid[l].size().area(); i += 8)
				{
					//const float w = max(0.f, 1.f - abs(gpy[i] - g) * istep);//hat
					//d[i] += w * lpy[i];
					__m256 w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(milinearstep, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpy + i + 0), mlinearstepk)), _mm256_set1_ps(1.f)));
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(w, _mm256_loadu_ps(lpy + i), _mm256_loadu_ps(d + i)));
				}*/
				for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
				{
					float w;
					if constexpr (interpolation == 0)
					{
						//const int c = min(order - 1, (int)saturate_cast<uchar>(g[i] * istep));
						w = (abs(gpy[i] - g) * idelta < 0.5f) ? 1.f : 0.f;
					}
					else if constexpr (interpolation == 1)
					{
						w = max(0.f, 1.f - abs(gpy[i] - g) * idelta);//hat
					}
					else //cv::INTER_CUBIC
					{
						w = getCubicCoeff((gpy[i] - g) * idelta, cubicAlpha);
						if (gpy[i] < intensityMin + delta) w = max(0.f, 1.f - abs(gpy[i] - g) * idelta);//hat
						if (gpy[i] > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy[i] - g) * idelta);//hat
					}
					d[i] += w * lpy[i];
				}
			}
		}
	}


	void LocalMultiScaleFilterInterpolation::pyramidParallel(const Mat& src, Mat& dest)
	{
		initRangeTable(sigma_range, boost);

		remapIm.resize(threadMax);

		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);

		const int gfRadius = getGaussianRadius(sigma_space);
		const int lowr = 2 * gfRadius + gfRadius;
		const int r_pad0 = lowr * (int)pow(2, level - 1);

		Mat smap, bmap;
		if (pyramidComputeMethod == IgnoreBoundary)
		{
			if (src.depth() == CV_32F)
			{
				copyMakeBorder(src, GaussianPyramid[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
			}
			else
			{
				copyMakeBorder(src, border, r_pad0, r_pad0, r_pad0, r_pad0, borderType);
				border.convertTo(GaussianPyramid[0], CV_32F);
			}
			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
				adaptiveSigmaBorder.resize(1);
				adaptiveBoostBorder.resize(1);
				cv::copyMakeBorder(adaptiveSigmaMap[0], adaptiveSigmaBorder[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
				cv::copyMakeBorder(adaptiveBoostMap[0], adaptiveBoostBorder[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
				smap = adaptiveSigmaBorder[0];
				bmap = adaptiveBoostBorder[0];
			}
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				src.copyTo(GaussianPyramid[0]);
			}
			else
			{
				src.convertTo(GaussianPyramid[0], CV_32F);
			}
			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
				smap = adaptiveSigmaMap[0];
				bmap = adaptiveBoostMap[0];
			}
		}

		//(1) build Gaussian Pyramid
		{
			//cp::Timer t("(1) build Gaussian Pyramid");
			buildGaussianPyramid(GaussianPyramid[0], GaussianPyramid, level, sigma_space);
		}

		//(2) build Laplacian Pyramid
		LaplacianPyramid.resize(order);
		{
			//cp::Timer t("(2) build Laplacian Pyramid");
			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
#pragma omp parallel for schedule(dynamic)
				for (int n = 0; n < order; n++)
				{
					const int tidx = omp_get_thread_num();
					remapAdaptive(GaussianPyramid[0], remapIm[tidx], getTau(n), smap, bmap);
					buildLaplacianPyramid(remapIm[tidx], LaplacianPyramid[n], level, sigma_space);
				}
			}
			else
			{
#pragma omp parallel for schedule(dynamic)
				for (int n = 0; n < order; n++)
				{
					const int tidx = omp_get_thread_num();
#if 0
					float* linebuff = (float*)_mm_malloc(sizeof(float) * GaussianPyramid[0].cols, AVX_ALIGN);
					remap(GaussianPyramid[0], remapIm[tidx], (float)(step * n), sigma_range, detail_param);
					if (radius == 2)
					{
						buildLaplacianPyramid<5, 3, 6>(remapIm[tidx], LaplacianPyramid[n], level, sigma_space, linebuff);
					}
					else if (radius == 4)
					{
						buildLaplacianPyramid<9, 5, 10>(remapIm[tidx], LaplacianPyramid[n], level, sigma_space, linebuff);
					}
					else
					{
						buildLaplacianPyramid(remapIm[tidx], LaplacianPyramid[n], level, sigma_space);
					}
					_mm_free(linebuff);
#else
					buildRemapLaplacianPyramidEachOrder(GaussianPyramid[0], LaplacianPyramid[n], level, sigma_space, getTau(n), sigma_range, boost);
#endif
				}
			}

			if (interpolation_method == cv::INTER_LINEAR)
			{
				blendLaplacianLinear(LaplacianPyramid, GaussianPyramid, GaussianPyramid, order);//orverride destnation pyramid for saving memory
			}
			else if (interpolation_method == cv::INTER_NEAREST)
			{
				blendLaplacianNearest(LaplacianPyramid, GaussianPyramid, GaussianPyramid, order);//orverride destnation pyramid for saving memory
			}
			else if (interpolation_method == cv::INTER_CUBIC)
			{
				blendLaplacianCubic(LaplacianPyramid, GaussianPyramid, GaussianPyramid, order);//orverride destnation pyramid for saving memory
			}
		}

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			collapseLaplacianPyramid(GaussianPyramid, GaussianPyramid[0]);
			if (src.depth() == CV_32F)
			{
				GaussianPyramid[0](Rect(r_pad0, r_pad0, src.cols, src.rows)).copyTo(dest);
			}
			else
			{
				GaussianPyramid[0](Rect(r_pad0, r_pad0, src.cols, src.rows)).convertTo(dest, src.type());
			}
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				collapseLaplacianPyramid(GaussianPyramid, dest);
			}
			else
			{
				Mat srcf;
				collapseLaplacianPyramid(GaussianPyramid, srcf);//override srcf for saving memory	
				srcf.convertTo(dest, src.type());
			}
		}
		//showPyramid("Laplacian Pyramid fast", GaussianPyramid);
	}

	void LocalMultiScaleFilterInterpolation::pyramidSerial(const Mat& src, Mat& dest)
	{
		layerSize.resize(level + 1);

		//initRangeTable(sigma_range, boost);
		if (isUseTable) initRangeTableInteger(sigma_range, boost);

		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);

		const int gfRadius = getGaussianRadius(sigma_space);
		const int lowr = 2 * gfRadius + gfRadius;
		const int r_pad0 = lowr * (int)pow(2, level - 1);

		Mat smap, bmap;
		if (pyramidComputeMethod == IgnoreBoundary)
		{
			if (src.depth() == CV_8U)
			{
				src.convertTo(GaussianPyramid[0], CV_32F);
				//src8u = src;
			}
			else
			{
				src.copyTo(GaussianPyramid[0]);
			}

			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
				smap = adaptiveSigmaMap[0];
				bmap = adaptiveBoostMap[0];
			}
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				src.copyTo(GaussianPyramid[0]);
			}
			else
			{
				src.convertTo(GaussianPyramid[0], CV_32F);
			}
			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
				smap = adaptiveSigmaMap[0];
				bmap = adaptiveBoostMap[0];
			}
		}

		//(1) build Gaussian Pyramid
		{
			//cp::Timer t("(1) build Gaussian Pyramid");
			buildGaussianPyramid(GaussianPyramid[0], GaussianPyramid, level, sigma_space);
			ImageStack.resize(GaussianPyramid.size());
			for (int i = 0; i < GaussianPyramid.size() - 1; i++)
			{
				ImageStack[i].create(GaussianPyramid[i].size(), CV_32F);
			}
			ImageStack[level] = GaussianPyramid[level];
		}

		//(2) build Laplacian Pyramid
		LaplacianPyramid.resize(1);
		{
			//cp::Timer t("(2) build Laplacian Pyramid");
			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
				buildRemapAdaptiveLaplacianPyramid<true>(GaussianPyramid, LaplacianPyramid[0], ImageStack, level, sigma_space, getTau(0), smap, bmap);
				for (int n = 1; n < order; n++)
				{
					buildRemapAdaptiveLaplacianPyramid<false>(GaussianPyramid, LaplacianPyramid[0], ImageStack, level, sigma_space, getTau(n), smap, bmap);
				}
			}
			else
			{
				const bool test = false;
				if (test)
				{
					for (int n = 0; n < order; n++)
					{
						buildRemapLaplacianPyramidEachOrder(GaussianPyramid[0], LaplacianPyramid[0], level, sigma_space, getTau(n), sigma_range, boost);
						if (interpolation_method == 0) productSumLaplacianPyramid<0>(LaplacianPyramid[0], GaussianPyramid, ImageStack, order, getTau(n));
						if (interpolation_method == 1) productSumLaplacianPyramid<1>(LaplacianPyramid[0], GaussianPyramid, ImageStack, order, getTau(n));
						if (interpolation_method == 2) productSumLaplacianPyramid<2>(LaplacianPyramid[0], GaussianPyramid, ImageStack, order, getTau(n));
					}
				}
				else
				{
					buildRemapLaplacianPyramid<true>(GaussianPyramid, LaplacianPyramid[0], ImageStack, level, sigma_space, getTau(0), sigma_range, boost);
					for (int n = 1; n < order; n++)
					{
#if 0
						float* linebuff = (float*)_mm_malloc(sizeof(float) * GaussianPyramid[0].cols, AVX_ALIGN);
						remap(GaussianPyramid[0], remapIm[tidx], (float)(step * n), sigma_range, detail_param);
						if (radius == 2)
						{
							buildLaplacianPyramid<5, 3, 6>(remapIm[tidx], LaplacianPyramid[n], level, sigma_space, linebuff);
						}
						else if (radius == 4)
						{
							buildLaplacianPyramid<9, 5, 10>(remapIm[tidx], LaplacianPyramid[n], level, sigma_space, linebuff);
						}
						else
						{
							buildLaplacianPyramid(remapIm[tidx], LaplacianPyramid[n], level, sigma_space);
						}
						_mm_free(linebuff);
#else
						buildRemapLaplacianPyramid<false>(GaussianPyramid, LaplacianPyramid[0], ImageStack, level, sigma_space, getTau(n), sigma_range, boost);
#endif
					}
				}
			}
		}

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			collapseLaplacianPyramid(ImageStack, dest);
			//collapseLaplacianPyramid(GaussianPyramid, dest);
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				collapseLaplacianPyramid(GaussianPyramid, dest);
			}
			else
			{
				Mat srcf;
				collapseLaplacianPyramid(GaussianPyramid, srcf);//override srcf for saving memory	
				srcf.convertTo(dest, src.type());
			}
		}
		//showPyramid("Laplacian Pyramid fast", GaussianPyramid);
		if (isUseTable)
		{
			_mm_free(integerSampleTable);
			integerSampleTable = nullptr;
		}
	}

	void LocalMultiScaleFilterInterpolation::pyramid(const Mat& src, Mat& dest)
	{
		rangeDescope(src);

		if (isParallel) pyramidParallel(src, dest);
		else pyramidSerial(src, dest);
	}

	void LocalMultiScaleFilterInterpolation::dog(const Mat& src, Mat& dest)
	{
		initRangeTable(sigma_range, boost);
		remapIm.resize(omp_get_max_threads());

		Mat srcf;
		if (src.depth() == CV_32F)
		{
			srcf = src;
		}
		else
		{
			src.convertTo(srcf, CV_32F);
		}

		//(1) build Gaussian stack
		{
			//merged in next step for parallelization
			//cp::Timer t("(1) build DoG");
			//buildGaussianStack(srcf, GaussianStack, sigma_space, level);
		}

		//(2) build DoG stack
		DoGStackLayer.resize(order);
		const float step = 255.f / order;
		{
			//cp::Timer t("(2) build DoG stack");
#pragma omp parallel for schedule(dynamic)
			for (int n = -1; n < order; n++)
			{
				if (n == -1)
				{
					buildGaussianStack(srcf, GaussianStack, sigma_space, level);
				}
				else
				{
					const int tidx = omp_get_thread_num();
					remap(srcf, remapIm[tidx], (float)(step * n), sigma_range, boost);
					buildDoGStack(remapIm[tidx], DoGStackLayer[n], sigma_space, level);
				}
			}

			if (interpolation_method == cv::INTER_LINEAR)
			{
				blendLaplacianLinear(DoGStackLayer, GaussianStack, GaussianStack, order);//orverride destnation pyramid for saving memory and copy lastlevel
			}
			else if (interpolation_method == cv::INTER_NEAREST)
			{
				blendLaplacianNearest(DoGStackLayer, GaussianStack, GaussianStack, order);//orverride destnation pyramid for saving memory
			}
			else if (interpolation_method == cv::INTER_CUBIC)
			{
				blendLaplacianCubic(DoGStackLayer, GaussianStack, GaussianStack, order);//orverride destnation pyramid for saving memory
			}
		}

		if (srcf.depth() == CV_32F)
		{
			collapseDoGStack(GaussianStack, dest);
		}
		else
		{
			collapseDoGStack(GaussianStack, srcf);//override srcf for saving memory	
			srcf.convertTo(dest, src.type());
		}
	}

	void LocalMultiScaleFilterInterpolation::filter(const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod, const int interpolationMethod)
	{
		allocSpaceWeight(sigma_space);

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->level = level;
		this->boost = boost;
		this->scalespaceMethod = scaleSpaceMethod;

		this->interpolation_method = interpolationMethod;
		this->order = order;
		body(src, dest);

		freeSpaceWeight();
	}


	void LocalMultiScaleFilterInterpolation::setCubicAlpha(const float alpha)
	{
		this->cubicAlpha = alpha;
	}

	void LocalMultiScaleFilterInterpolation::setComputeScheduleMethod(const bool useTable)
	{
		this->isUseTable = useTable;
	}

	string LocalMultiScaleFilterInterpolation::getComputeScheduleName()
	{
		string ret = "";
		if (this->isUseTable)ret = "IntegerSampledTable";
		else ret = "Compute";
		return ret;
	}

	void LocalMultiScaleFilterInterpolation::setIsParallel(const bool flag)
	{
		isParallel = flag;
	}
#pragma endregion



#pragma region Fourier
	class ComputeCompressiveT_Fast32F : public cp::Search1D32F
	{
		int cbf_order;
		float sigma_range;
		float Salpha, Sbeta, Ssigma;
		int windowType;
		int Imin = 0;
		int Imax = 0;
		int Irange = 0;
		std::vector<float> ideal;
		int window_type;
		int integration_interval;

		template <typename type, int windowType>
		float getArbitraryRangeKernelErrorEven(const type T, const int order, const type sigma_range, const int Imin, const int Imax, std::vector<type>& ideal, const int integration_interval, int window_type,
			float Salpha, float Sbeta, float Ssigma)
		{
			std::vector<type> alpha(order + 1);
			std::vector<type> beta(order + 1);
			//const type gaussRangeCoeff = (type)(-0.5 / (sigma_range * sigma_range));
			const type omega = (float)(CV_2PI / T);

			type alphaSum = 1.f;
			{
				FourierDecomposition Fourier(T, sigma_range, Sbeta, Salpha, Ssigma, 0, windowType);
				const double normal = 2.0 / Fourier.init(0, T / 2, integration_interval);
				for (int n = 1; n < order + 1; n++)
				{
					FourierDecomposition Fourier(T, sigma_range, Sbeta, Salpha, Ssigma, n, windowType);
					if (windowType == GAUSS)
					{
						alpha[n] = (type)(normal * Fourier.ct(0, T / 2, integration_interval));
						alphaSum += alpha[n];
					}
					else if (windowType == S_TONE)
					{
						beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
					}
					else if (windowType == HAT)
					{
						beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
					}
					else if (windowType == SMOOTH_HAT)
					{
						beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
					}
				}
			}

			for (int n = 1; n < order + 1; ++n)
			{
				alpha[n] /= alphaSum;
			}
			double mse = 0.0;

			std::vector<type> cosTable(order + 1);
			std::vector<type> sinTable(order + 1);

			const type t = (type)0.0;
			//const type t = (type)(Irange * 0.5 + Imin);
			//const type t = (type)(Imax); 
			//const type t = (type)((Imax - Imin) * 0.5 + Imin);
			//const type t = (type)(Imin);

			for (int n = 1; n < order + 1; n++)
			{
				cosTable[n] = (type)cos(omega * n * t);
				sinTable[n] = (type)sin(omega * n * t);
			}
#if 1
			for (int i = 0; i <= Irange; i++)
			{
				const type omegai = (type)(omega * i);
				type s = (type)(1.0 / alphaSum);
				for (int n = 1; n < order + 1; n++)
				{
					if constexpr (windowType == GAUSS) s += alpha[n] * (cosTable[n] * cos(omegai * n) + sinTable[n] * sin(omegai * n));
					else if constexpr (windowType == S_TONE) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == SMOOTH_HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
				}

				if constexpr (windowType == GAUSS)
				{
					type wr = s;
					type we = ideal[(int)abs(i - t)];
					double sub = double(wr - we);
					mse += sub * sub;
				}
				else if constexpr (windowType == S_TONE)
				{
					type wr = s + Imin;
					type we = getSToneCurve<float>((float)i, float(Imin), Ssigma, Sbeta, Salpha);
					mse += (type)((we - wr) * (we - wr));
				}
				else if constexpr (windowType == HAT)
				{
					type wr = s + Imin;
					type we = (type)((i - Imin) * std::max(0.0, 1.0 - abs((i - Imin) / sigma_range)));
					mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == SMOOTH_HAT)
				{
					type wr = s + Imin;
					//we = (i - Imin) * getSmoothingHat(i, Imin, sigma_range, 5);
					type we = getSmoothingHat(float(i), float(Imin), sigma_range, 5);
					mse += (we - wr) * (we - wr);
				}
			}
#else
			cp::Plot pt;
			pt.setPlotTitle(0, "ideal");
			pt.setPlotTitle(1, "approx");
			pt.setPlotTitle(2, "diff");

			print_debug(alphaSum);

			for (int i = 0; i <= 255; i++)
			{
				const type omegai = (type)(omega * i);
				type s = (type)(1.0 / alphaSum);
				for (int n = 1; n < cbf_order + 1; n++)
				{
					//if constexpr (windowType == GAUSS) s += alpha[n] * (cosTable[n] * sin(omegai * n) - sinTable[n] * cos(omegai * n));
					if constexpr (windowType == GAUSS) s += alpha[n] * (cosTable[n] * cos(omegai * n) + sinTable[n] * sin(omegai * n));
					else if constexpr (windowType == S_TONE) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == SMOOTH_HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
				}

				if constexpr (windowType == GAUSS)
				{
					type wr = s;
					//type we = (i - t) * ideal[abs(i - t)];
					type we = ideal[abs(i - t)];
					pt.push_back(i, we, 0);
					pt.push_back(i, wr, 1);
					pt.push_back(i, (we - wr) * 20, 2);

					if (i <= Irange)
						mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == S_TONE)
				{
					type wr = s + Imin;
					type we = getSToneCurve<float>((float)i, float(Imin), Ssigma, Sbeta, Salpha);
					mse += (type)((we - wr) * (we - wr));
				}
				else if constexpr (windowType == HAT)
				{
					type wr = s + Imin;
					type we = (type)((i - Imin) * std::max(0.0, 1.0 - abs((i - Imin) / sigma_range)));
					mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == SMOOTH_HAT)
				{
					type wr = s + Imin;
					//we = (i - Imin) * getSmoothingHat(i, Imin, sigma_range, 5);
					type we = getSmoothingHat(float(i), float(Imin), sigma_range, 5);
					mse += (we - wr) * (we - wr);
				}
			}
			print_debug4(Imin, Imax, T, sqrt(mse));
			pt.plot();
#endif

			//mse = sqrt(mse);
			return float(mse);
		}

		template <typename type, int windowType>
		float getArbitraryRangeKernelErrorOdd(const type T, const int cbf_order, const type sigma_range, const int Imin, const int Imax, std::vector<type>& ideal, const int integration_interval, int window_type,
			float Salpha, float Sbeta, float Ssigma)
		{
			std::vector<type> alpha(cbf_order + 1);
			std::vector<type> beta(cbf_order + 1);
			const type omega = (float)(CV_2PI / T);

			type alphaSum = 0.f;
			for (int n = 1; n < cbf_order + 1; n++)
			{
				FourierDecomposition Fourier(T, sigma_range, Sbeta, Salpha, Ssigma, n, windowType);
				if (windowType == GAUSS)
				{
					alpha[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
					alphaSum += alpha[n];
				}
				else if (windowType == S_TONE)
				{
					beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
				}
				else if (windowType == HAT)
				{
					beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
				}
				else if (windowType == SMOOTH_HAT)
				{
					beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
				}
			}

			std::vector<type> cosTable(cbf_order + 1);
			std::vector<type> sinTable(cbf_order + 1);
			const int t = 0;
			//const type t = (type)(Irange * 0.5 + Imin);
			//const type t = (type)(Imax); 
			//const type t = (type)(Imin);

			for (int n = 1; n < cbf_order + 1; n++)
			{
				cosTable[n] = (type)cos(omega * n * t);
				sinTable[n] = (type)sin(omega * n * t);
			}

			type mse = (type)0.0;

#if 1
			for (int i = 0; i <= Irange; i++)
				//for (int i = 0; i <= 1; i++)
			{
				const type omegai = (type)(omega * i);
				type s = (type)0.0;
				for (int n = 1; n < cbf_order + 1; n++)
				{
					if constexpr (windowType == GAUSS) s += alpha[n] * (cosTable[n] * sin(omegai * n) - sinTable[n] * cos(omegai * n));
					else if constexpr (windowType == S_TONE) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == SMOOTH_HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
				}

				if constexpr (windowType == GAUSS)
				{
					type wr = s;
					type we = (i - t) * ideal[abs(i - t)];
					mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == S_TONE)
				{
					type wr = s + Imin;
					type we = getSToneCurve<float>((float)i, float(Imin), Ssigma, Sbeta, Salpha);
					mse += (type)((we - wr) * (we - wr));
				}
				else if constexpr (windowType == HAT)
				{
					type wr = s + Imin;
					type we = (type)((i - Imin) * std::max(0.0, 1.0 - abs((i - Imin) / sigma_range)));
					mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == SMOOTH_HAT)
				{
					type wr = s + Imin;
					//we = (i - Imin) * getSmoothingHat(i, Imin, sigma_range, 5);
					type we = getSmoothingHat(float(i), float(Imin), sigma_range, 5);
					mse += (we - wr) * (we - wr);
				}
			}
#else
			cp::Plot pt;
			pt.setPlotTitle(0, "ideal");
			pt.setPlotTitle(1, "approx");
			for (int i = 0; i <= 255; i++)
			{
				const type omegai = (type)(omega * i);
				type s = (type)(0.0);
				for (int n = 1; n < cbf_order + 1; n++)
				{
					if constexpr (windowType == GAUSS) s += alpha[n] * (cosTable[n] * sin(omegai * n) - sinTable[n] * cos(omegai * n));
					else if constexpr (windowType == S_TONE) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == SMOOTH_HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
				}

				if constexpr (windowType == GAUSS)
				{
					type wr = s;
					type we = (i - t) * ideal[abs(i - t)];
					pt.push_back(i, we, 0);
					pt.push_back(i, wr, 1);

					if (i <= Irange)
						mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == S_TONE)
				{
					type wr = s + Imin;
					type we = getSToneCurve<float>((float)i, float(Imin), Ssigma, Sbeta, Salpha);
					mse += (type)((we - wr) * (we - wr));
				}
				else if constexpr (windowType == HAT)
				{
					type wr = s + Imin;
					type we = (type)((i - Imin) * std::max(0.0, 1.0 - abs((i - Imin) / sigma_range)));
					mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == SMOOTH_HAT)
				{
					type wr = s + Imin;
					//we = (i - Imin) * getSmoothingHat(i, Imin, sigma_range, 5);
					type we = getSmoothingHat(float(i), float(Imin), sigma_range, 5);
					mse += (we - wr) * (we - wr);
				}
			}
			print_debug4(Imin, Imax, T, sqrt(mse));
			pt.plot();
#endif

			//mse = sqrt(mse);
			return mse;
		}

		float getError(float a)
		{
			float ret = 0.f;
			const bool isEven = true;
			if (isEven)
			{
				if (windowType == GAUSS) ret = getArbitraryRangeKernelErrorEven<float, GAUSS>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == S_TONE) ret = getArbitraryRangeKernelErrorEven<float, S_TONE>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == HAT) ret = getArbitraryRangeKernelErrorEven<float, HAT>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == SMOOTH_HAT) ret = getArbitraryRangeKernelErrorEven<float, SMOOTH_HAT>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
			}
			else
			{
				if (windowType == GAUSS) ret = getArbitraryRangeKernelErrorOdd<float, GAUSS>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == S_TONE) ret = getArbitraryRangeKernelErrorOdd<float, S_TONE>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == HAT) ret = getArbitraryRangeKernelErrorOdd<float, HAT>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == SMOOTH_HAT) ret = getArbitraryRangeKernelErrorOdd<float, SMOOTH_HAT>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
			}
			return ret;
		}

	public:
		ComputeCompressiveT_Fast32F(int Irange, int Imin, int Imax, int cbf_order, float sigma_range, int integration_interval, int window_type, float Salpha, float Sbeta, float Ssigma, int windowType)
		{
			this->Irange = Irange;
			this->cbf_order = cbf_order;
			this->sigma_range = sigma_range;
			this->Imin = Imin;
			this->Imax = Imax;
			this->integration_interval = integration_interval;
			this->window_type = window_type;
			this->Salpha = Salpha;
			this->Sbeta = Sbeta;
			this->Ssigma = Ssigma;
			this->windowType = windowType;

#if 1
			ideal.resize(256);
#else
			ideal.resize(Irange + 1);
#endif
			for (int i = 0; i < ideal.size(); i++)
				//for (int i = 0; i < 256; i++)
			{
				ideal[i] = (float)getRangeKernelFunction(double(i), sigma_range, window_type);
			}
		}
	};

	double getOptimalT_32F(const int cbf_order, const double sigma_range, const int Imin, const int Imax, float Salpha, float Sbeta, float Ssigma, int windowType,
		int integral_interval = 100, double serch_T_min = 0.0, double serch_T_max = 5.0, double search_T_diff_min = 0.01, double search_iteration_max = 20)
	{
		const int Irange = Imax - Imin;

		ComputeCompressiveT_Fast32F ct(Irange, Imin, Imax, cbf_order, (float)sigma_range, integral_interval, windowType, Salpha, Sbeta, Ssigma, windowType);
		double ret = (double)ct.goldenSectionSearch((float)serch_T_min, (float)serch_T_max, (float)search_T_diff_min, (int)search_iteration_max);
		//double ret = (double)ct.linearSearch(serch_T_min, serch_T_max, 0.001);
		return ret;
	}

	inline double dfCompressive(double x, const int K, const double Irange, const double sigma_range)
	{
		const double s = sigma_range / Irange;
		const double kappa = (2 * K + 1) * CV_PI;
		const double psi = kappa * s / x;
		const double phi = (x - 1.0) / s;
		return (-kappa * exp(-phi * phi) + psi * psi * exp(-psi * psi));
	}

	double computeCompressiveT_ClosedForm(int order, double sigma_range, const double intensityRange)
	{
		double x, diff;

		double x1 = 1.0, x2 = 15.0;
		int loop = 20;
		for (int i = 0; i < loop; ++i)
		{
			x = (x1 + x2) / 2.0;
			diff = dfCompressive(x, order, intensityRange, sigma_range);
			((0.0 <= diff) ? x2 : x1) = x;
		}
		return x;
	}

	inline __m256 getAdaptiveAlpha(__m256 coeff, __m256 base, __m256 sigma, __m256 boost)
	{
		__m256 a = _mm256_mul_ps(coeff, sigma);
		a = _mm256_exp_ps((_mm256_mul_ps(_mm256_set1_ps(-0.5), _mm256_mul_ps(a, a))));
		return _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(sigma, sigma), sigma), boost), base), a);
	}

	inline float getAdaptiveAlpha(float coeff, float base, float sigma, float boost)
	{
		float a = coeff * sigma;
		a = exp(-0.5f * a * a);
		return sigma * sigma * sigma * boost * base * a;
	}
#pragma endregion

#pragma region LocalMultiScaleFilterFourierReference

	void LocalMultiScaleFilterFourierReference::initRangeFourier(const int order, const float sigma_range, const float boost)
	{
		if (alpha.size() != order)
		{
			alpha.resize(order);
			beta.resize(order);
		}

		if (omega.size() != order) omega.resize(order);

		T = float(intensityRange * computeCompressiveT_ClosedForm(order, sigma_range, intensityRange));
		//T = intensityRange * getOptimalT_32F(1.0, order, sigma_range, 0, (int)intensityRange, Salpha, Sbeta, sigma_range, windowType, 100, 0, 5.0, 0.001, 20, 0);

		for (int k = 0; k < order; k++)
		{
			omega[k] = float(CV_2PI / (double)T * (double)(k + 1));
			const double coeff_kT = omega[k] * sigma_range;
			alpha[k] = float(2.0 * exp(-0.5 * coeff_kT * coeff_kT) * sqrt(CV_2PI) * sigma_range / T);
		}
	}

	void LocalMultiScaleFilterFourierReference::remapCos(const cv::Mat& src, cv::Mat& dest, const float omega)
	{
		dest.create(src.size(), CV_32F);

		if (isSIMD)
		{
			const __m256 momega = _mm256_set1_ps(omega);
			const __m256* msrc = (__m256*)src.ptr<float>();
			__m256* mdest = (__m256*)dest.ptr<float>();
			const int SIZE = src.size().area() / 8;

			for (int i = 0; i < SIZE; i++)
			{
				*(mdest++) = _mm256_cos_ps(_mm256_mul_ps(momega, *msrc++));
			}
		}
		else
		{
			const float* msrc = src.ptr<float>();
			float* mdest = dest.ptr<float>();
			const int SIZE = src.size().area();

			for (int i = 0; i < SIZE; i++)
			{
				mdest[i] = cos(omega * msrc[i]);
			}
		}
	}

	void LocalMultiScaleFilterFourierReference::remapSin(const cv::Mat& src, cv::Mat& dest, const float omega)
	{
		dest.create(src.size(), CV_32F);

		if (isSIMD)
		{
			const __m256 momega = _mm256_set1_ps(omega);
			const __m256* msrc = (__m256*)src.ptr<float>();
			__m256* mdest = (__m256*)dest.ptr<float>();
			const int SIZE = src.size().area() / 8;

			for (int i = 0; i < SIZE; i++)
			{
				*(mdest++) = _mm256_sin_ps(_mm256_mul_ps(momega, *msrc++));
			}
		}
		else
		{
			const float* msrc = src.ptr<float>();
			float* mdest = dest.ptr<float>();
			const int SIZE = src.size().area();

			for (int i = 0; i < SIZE; i++)
			{
				mdest[i] = sin(omega * msrc[i]);
			}
		}
	}

	void LocalMultiScaleFilterFourierReference::productSumPyramidLayer(const cv::Mat& srccos, const cv::Mat& srcsin, const cv::Mat gauss, cv::Mat& dest, const float omega, const float alpha, const float sigma, const float boost)
	{
		dest.create(srccos.size(), CV_32F);

		if (isSIMD)
		{
			const int SIZE = srccos.size().area() / 8;

			const __m256* scptr = (__m256*)srccos.ptr<float>();
			const __m256* ssptr = (__m256*)srcsin.ptr<float>();
			const __m256* gptr = (__m256*)gauss.ptr<float>();
			__m256* dptr = (__m256*)dest.ptr<float>();

			__m256 malpha = _mm256_set1_ps(-sigma * sigma * omega * alpha * boost);
			const __m256 momega_k = _mm256_set1_ps(omega);
			for (int i = 0; i < SIZE; i++)
			{
				const __m256 ms = _mm256_mul_ps(momega_k, *gptr++);
				const __m256 msin = _mm256_sin_ps(ms);
				const __m256 mcos = _mm256_cos_ps(ms);
				*(dptr) = _mm256_fmadd_ps(malpha, _mm256_fmsub_ps(msin, *(scptr++), _mm256_mul_ps(mcos, *(ssptr++))), *(dptr));
				dptr++;
			}
		}
		else
		{
			const int SIZE = srccos.size().area();

			const float* cosptr = srccos.ptr<float>();
			const float* sinptr = srcsin.ptr<float>();
			const float* gptr = gauss.ptr<float>();
			float* dptr = dest.ptr<float>();

			const float lalpha = -sigma * sigma * omega * alpha * boost;
			for (int i = 0; i < SIZE; i++)
			{
				const float ms = omega * gptr[i];
				dptr[i] += lalpha * (sin(ms) * cosptr[i] - cos(ms) * (sinptr[i]));
			}
		}
	}

	void LocalMultiScaleFilterFourierReference::productSumAdaptivePyramidLayer(const cv::Mat& srccos, const cv::Mat& srcsin, const cv::Mat gauss, cv::Mat& dest, const float omega, const float alpha, const Mat& sigma, const Mat& boost)
	{
		dest.create(srccos.size(), CV_32F);

		if (isSIMD)
		{
			const int SIZE = srccos.size().area() / 8;

			const __m256* scptr = (__m256*)srccos.ptr<float>();
			const __m256* ssptr = (__m256*)srcsin.ptr<float>();
			const __m256* gptr = (__m256*)gauss.ptr<float>();
			__m256* dptr = (__m256*)dest.ptr<float>();

			const float base = -float(2.0 * sqrt(CV_2PI) * omega / T);
			const __m256 mbase = _mm256_set1_ps(base);//for adaptive
			__m256* adaptiveSigma = (__m256*)sigma.ptr<float>();
			__m256* adaptiveBoost = (__m256*)boost.ptr<float>();
			const __m256 momega_k = _mm256_set1_ps(omega);

			for (int i = 0; i < SIZE; i++)
			{
				__m256 malpha = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma++, *adaptiveBoost++);
				const __m256 ms = _mm256_mul_ps(momega_k, *gptr++);
				const __m256 msin = _mm256_sin_ps(ms);
				const __m256 mcos = _mm256_cos_ps(ms);
				*(dptr) = _mm256_fmadd_ps(malpha, _mm256_fmsub_ps(msin, *(scptr++), _mm256_mul_ps(mcos, *(ssptr++))), *(dptr));
				dptr++;
			}
		}
		else
		{
			const int SIZE = srccos.size().area();

			const float* scptr = srccos.ptr<float>();
			const float* ssptr = srcsin.ptr<float>();
			const float* gptr = gauss.ptr<float>();
			float* dptr = dest.ptr<float>();

			const float base = -float(2.0 * sqrt(CV_2PI) * omega / T);
			const float* adaptiveSigma = sigma.ptr<float>();
			const float* adaptiveBoost = boost.ptr<float>();

			for (int i = 0; i < SIZE; i++)
			{
				float lalpha = getAdaptiveAlpha(omega, base, adaptiveSigma[i], adaptiveBoost[i]);
				const float ms = omega * gptr[i];
				dptr[i] += lalpha * (sin(ms) * scptr[i] - cos(ms) * (ssptr[i]));
			}
		}
	}

	void LocalMultiScaleFilterFourierReference::pyramid(const Mat& src, Mat& dest)
	{
		pyramidComputeMethod = PyramidComputeMethod::Full;

		//alloc buffer
		GaussianPyramid.resize(level + 1);
		LaplacianPyramid.resize(level + 1);
		FourierPyramidCos.resize(level + 1);
		FourierPyramidSin.resize(level + 1);
		if (src.depth() == CV_8U) src.convertTo(GaussianPyramid[0], CV_32F);
		else src.copyTo(GaussianPyramid[0]);
		layerSize.resize(level + 1);
		for (int l = 0; l < level + 1; l++) layerSize[l] = GaussianPyramid[l].size();

		//compute alpha, omega, T
		initRangeFourier(order, sigma_range, boost);

		//Build Gaussian Pyramid
		buildGaussianPyramid(GaussianPyramid[0], GaussianPyramid, level, sigma_space);

		//Build Laplacian Pyramid for DC
		buildLaplacianPyramid(GaussianPyramid, LaplacianPyramid, level, sigma_space);
		for (int k = 0; k < order; k++)
		{
			remapCos(src, FourierPyramidCos[0], omega[k]);
			remapSin(src, FourierPyramidSin[0], omega[k]);

			buildLaplacianPyramid(FourierPyramidCos[0], FourierPyramidCos, level, sigma_space);
			buildLaplacianPyramid(FourierPyramidSin[0], FourierPyramidSin, level, sigma_space);

			if (adaptiveMethod == AdaptiveMethod::FIX)
			{
				for (int l = 0; l < level; l++)
				{
					productSumPyramidLayer(FourierPyramidCos[l], FourierPyramidSin[l], GaussianPyramid[l], LaplacianPyramid[l], omega[k], alpha[k], sigma_range, boost);
				}
			}
			else
			{
				for (int l = 0; l < level; l++)
				{
					productSumAdaptivePyramidLayer(FourierPyramidCos[l], FourierPyramidSin[l], GaussianPyramid[l], LaplacianPyramid[l], omega[k], alpha[k], adaptiveSigmaMap[l], adaptiveBoostMap[l]);
				}
			}
		}
		//set last level
		LaplacianPyramid[level] = GaussianPyramid[level];

		//collapse Laplacian Pyramid
		collapseLaplacianPyramid(LaplacianPyramid, dest);
	}

	void LocalMultiScaleFilterFourierReference::filter(const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);

		this->order = order;

		this->sigma_space = sigma_space;
		this->level = max(level, 1);
		this->sigma_range = sigma_range;
		this->boost = boost;

		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);

		freeSpaceWeight();
	}

#pragma endregion

#pragma region LocalMultiScaleFilterFourier

	template<typename Type>
	void LocalMultiScaleFilterFourier::kernelPlot(const int window_type, const int order, const int R, const double boost, const double sigma_range, float Salpha, float Sbeta, float Ssigma, const int Imin, const int Imax, const int Irange, const Type T,
		Type* sinTable, Type* cosTable, std::vector<Type>& alpha, std::vector<Type>& beta, int windowType, const std::string wname, const cv::Size windowSize)
	{
		cp::Plot pt(windowSize);
		pt.setPlotTitle(0, "Ideal");
		pt.setPlotTitle(1, "y=xf(x)");
		pt.setPlotTitle(2, "y=x");
		pt.setPlotTitle(3, "y=boost*x");
		pt.setPlotSymbolALL(0);
		pt.setPlotLineWidthALL(2);
		pt.setKey(cp::Plot::KEY::LEFT_TOP);

		namedWindow(wname);
		createTrackbar("t", wname, &kernel_plotting_t, 255);
		createTrackbar("amp_pow", wname, &kernel_plotting_amp, 100);
		const int t = kernel_plotting_t;
		double error = 0.0;

		if (kernel_plotting_amp != 0) pt.setPlotTitle(4, "diff");

		for (int s = 0; s <= R; s++)
		{
			Type wr = (Type)0.0;

			for (int k = 0; k < order; ++k)
			{
				float* ct = &cosTable[256 * k];
				float* st = &sinTable[256 * k];
				double omega = CV_2PI / T * (double)(k + 1);
				const double lalpha = sigma_range * sigma_range * omega * alpha[k] * boost;

				switch (windowType)
				{
				case GAUSS:
					wr += Type(lalpha * (st[s] * ct[t] - ct[s] * st[t])); break;
				case S_TONE:
					wr += Type(beta[k] * (sinTable[256 * k + s] * cosTable[256 * k + t] - cosTable[256 * k + s] * sinTable[256 * k + t])); break;
				case HAT:
					wr += Type(alpha[k] * (sinTable[256 * k + s] * cosTable[256 * k + t] - cosTable[256 * k + s] * sinTable[256 * k + t])); break;
				case SMOOTH_HAT:
					wr += Type(alpha[k] * (sinTable[256 * k + s] * cosTable[256 * k + t] - cosTable[256 * k + s] * sinTable[256 * k + t])); break;
				}
			}

			double ideal_value = 0.0;
			double apprx_value = 0.0;
			switch (windowType)
			{
			case GAUSS:
				ideal_value = s + double(s - t) * boost * getRangeKernelFunction(double(s - t), sigma_range, window_type);
				apprx_value = s + double(wr);
				break;

			case S_TONE:
				ideal_value = getSToneCurve(float(s), float(t), Ssigma, Sbeta, Salpha);
				apprx_value = t + wr;
				pt.push_back(s, t + wr, 1);
				break;
			case HAT:
				ideal_value = s + (s - t) * std::max(0.0, 1.0 - abs((float)(s - t) / sigma_range));
				apprx_value = s + wr;
				break;
			case SMOOTH_HAT:
				//v = s + (s - t) * getSmoothingHat(s, t, sigma_range, 5);
				ideal_value = s + getSmoothingHat(float(s), float(t), float(sigma_range), 10);
				apprx_value = s + wr;
				break;
			}

			pt.push_back(s, ideal_value, 0);//ideal
			pt.push_back(s, apprx_value, 1);//approx
			pt.push_back(s, s, 2);
			pt.push_back(s, boost * (s - t) + s, 3);
			error += (ideal_value - apprx_value) * (ideal_value - apprx_value);
			if (kernel_plotting_amp != 0) pt.push_back(s, (ideal_value - apprx_value) * pow(10.0, kernel_plotting_amp), 4);
		}
		error = 20.0 * std::log10(R / sqrt(error / (R + 1)));

		//pt.setYRange(0, 1);
		pt.setXRange(0, R + 1);
		pt.setYRange(-128, 256 + 128);
		pt.setIsDrawMousePosition(true);
		pt.setGrid(2);
		//pt.plot(wname, false, "", format("err. total: %2.2f, err. current: %2.2f, (min,max)=(%d  %d)", error, errort, Imin, Imax));
		pt.plot(wname, false, "", cv::format("Kernel Err: %6.2lf dB (min,max)=(%d, %d)", error, Imin, Imax));
	}

	void LocalMultiScaleFilterFourier::initRangeFourier(const int order, const float sigma_range, const float boost)
	{
		bool isRecompute = true;
		if (preorder == order && presigma_range == sigma_range && predetail_param == boost && preIntensityMin == intensityMin && preIntensityRange == intensityRange && preperiodMethod == periodMethod)isRecompute = false;
		if (!isRecompute)return;

		//recomputing flag setting
		preorder = order;
		presigma_range = sigma_range;
		predetail_param = boost;
		preIntensityMin = intensityMin;
		preIntensityRange = intensityRange;
		preperiodMethod = periodMethod;

		//alloc
		if (alpha.size() != order)
		{
			alpha.resize(order);
			beta.resize(order);
		}
		if (omega.size() != order) omega.resize(order);

		//compute T
		//static int rangeMax = 255;//if use opt
		//int rangeMax = 255;
		switch (periodMethod)
		{
			//Using the derivative of a Gaussian function
		case GAUSS_DIFF:
			//cout << "Gauss Diff" << endl;
			T = float(intensityRange * computeCompressiveT_ClosedForm(order, sigma_range, intensityRange));
			break;
			//minimizing the squared error
		case OPTIMIZE:
			//cout << "Optimize" << endl;
			T = float(intensityRange * getOptimalT_32F(order, sigma_range, (int)intensityMin, (int)intensityMax, Salpha, Sbeta, sigma_range, windowType, 100, 1.0, 20.0, 0.001, 20));
			break;
		case PRE_SET:
			//static int T_ = 4639;//4639, 1547, 3093, 6186
			int T_ = 4639;//4639, 1547, 3093, 6186
			//cv::createTrackbar("T", "", &T_, 20000);
			T = T_ * 0.1f;
			break;
		}
		//cout << "T : " << T << endl;

		//compute omega and alpha
		if (periodMethod == GAUSS_DIFF)
		{
			for (int k = 0; k < order; k++)
			{
				omega[k] = float(CV_2PI / (double)T * (double)(k + 1));
				const double coeff_kT = omega[k] * sigma_range;
				switch (windowType)
				{
				case GAUSS:
					alpha[k] = float(2.0 * exp(-0.5 * coeff_kT * coeff_kT) * sqrt(CV_2PI) * sigma_range / T);
					//alpha[k] = exp(-0.5 * coeff_kT * coeff_kT);
					break;

				default:
				{
					FourierDecomposition Fourier(T, sigma_range, Sbeta, Salpha, sigma_range, k + 1, windowType);
					alpha[k] = float(4.0 * Fourier.st(0, T / 2, 100) / T);
					beta[k] = float(4.0 * Fourier.st(0, T / 2, 100) / T);
					break;
				}
				}
			}
		}
		else
		{
			FourierDecomposition Fourier(T, sigma_range, 0, 0, 0, 0, windowType);
			double a0 = 2.0 / Fourier.init(0, T / 2, 100);
			double alphaSum = a0;
			for (int k = 0; k < order; k++)
			{
				omega[k] = float(CV_2PI / (double)T * (double)(k + 1));
				switch (windowType)
				{
				case GAUSS:
				{
					FourierDecomposition Fourier(T, sigma_range, 0, 0, 0, k + 1, windowType);
					alpha[k] = float(a0 * Fourier.st(0, T / 2, 100));
					alphaSum += double(alpha[k]);
					break;
				}
				default:
				{
					FourierDecomposition Fourier(T, sigma_range, Sbeta, Salpha, sigma_range, k + 1, windowType);
					alpha[k] = float(4.0 * Fourier.st(0, T / 2, 100) / T);
					beta[k] = float(4.0 * Fourier.st(0, T / 2, 100) / T);
					break;
				}
				}
			}

			alphaSum *= sqrt(CV_2PI);
			for (int k = 0; k < order; k++)
			{
				alpha[k] = float(alpha[k] / alphaSum);
			}
		}


		//compute cos/sin table
		if (isUseFourierTable0)
		{
			_mm_free(sinTable);
			_mm_free(cosTable);
			sinTable = (float*)_mm_malloc(sizeof(float) * FourierTableSize * order, AVX_ALIGN);
			cosTable = (float*)_mm_malloc(sizeof(float) * FourierTableSize * order, AVX_ALIGN);

			const int TABLESIZE = get_simd_floor(FourierTableSize, 8);
			const __m256 m8 = _mm256_set1_ps(8.f);
			for (int k = 0; k < order; k++)
			{
				float* sinptr = sinTable + FourierTableSize * k;
				float* cosptr = cosTable + FourierTableSize * k;

				const __m256 momega_k = _mm256_set1_ps(omega[k]);
				__m256 unit = _mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);
				for (int i = 0; i < TABLESIZE; i += 8)
				{
					__m256 base = _mm256_mul_ps(momega_k, unit);
					_mm256_store_ps(sinptr + i, _mm256_sin_ps(base));
					_mm256_store_ps(cosptr + i, _mm256_cos_ps(base));
					unit = _mm256_add_ps(unit, m8);
				}
				for (int i = TABLESIZE; i < FourierTableSize; i++)
				{
					sinptr[i] = sin(omega[k] * i);
					cosptr[i] = cos(omega[k] * i);
				}
			}
		}

		//cout << alphaSum << endl;
		if (false)//normalize test
		{
			float alphaSum = 1.0;
			for (int k = 0; k < order; k++)
			{
				alphaSum += alpha[k];
				//print_debug2(k, alpha[k]);
			}

			cout << "before " << alphaSum * sqrt(CV_2PI) * sigma_range / T << "," << T << endl;
			for (int k = 0; k < order; k++)
			{
				alpha[k] = alpha[k] / alphaSum;
			}
			alphaSum = 1.f / alphaSum;
			cout << "vv " << alphaSum << endl;
			for (int k = 0; k < order; k++)
			{
				alphaSum += alpha[k];
			}
			cout << "normal " << alphaSum << endl;
		}
	}

	void LocalMultiScaleFilterFourier::allocImageBuffer(const int order, const int level)
	{
		if (ImageStack.size() != level + 1)
		{
			ImageStack.resize(0);
			ImageStack.resize(level + 1);
			LaplacianPyramid.resize(level + 1);
		}

		if (isParallel)
		{
			const int OoT = max(order, threadMax);
			if (FourierPyramidCos.size() != OoT)
			{
				FourierPyramidCos.resize(OoT);
				FourierPyramidSin.resize(OoT);
				destEachOrder.resize(OoT);
				for (int i = 0; i < OoT; i++)
				{
					FourierPyramidCos[i].resize(level + 1);
					FourierPyramidSin[i].resize(level + 1);
					destEachOrder[i].resize(level);
				}
			}
		}
		else
		{
			if (FourierPyramidCos.size() != 1)
			{
				FourierPyramidCos.resize(1);
				FourierPyramidSin.resize(1);
				destEachOrder.resize(1);
				for (int i = 0; i < 1; i++)
				{
					destEachOrder[i].resize(level);
				}
			}
		}
	}

	template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseFourierTableLevel, int D, int D2>
	void LocalMultiScaleFilterFourier::buildLaplacianFourierPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidCos, vector<Mat>& FourierPyramidSin)
	{
		const int rs = radius >> 1;
		//const int D = 2 * radius + 1;
		//const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidCos.size() != level + 1)FourierPyramidCos.resize(level + 1);
		if (FourierPyramidSin.size() != level + 1)FourierPyramidSin.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidCos[0].create(imSize, CV_32F);
		FourierPyramidSin[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidCos[l].create(pySize, CV_32F);
			FourierPyramidSin[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidCos[level].create(pySize, CV_32F);
			FourierPyramidSin[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* linebuffer = (float*)_mm_malloc(sizeof(float) * linesize * 4, AVX_ALIGN);;
		float* spcosline_e = linebuffer + 0 * linesize;
		float* spsinline_e = linebuffer + 1 * linesize;
		float* spcosline_o = linebuffer + 2 * linesize;
		float* spsinline_o = linebuffer + 3 * linesize;

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 meven_ratio = _mm256_set1_ps(evenratio);
		const __m256 modd__ratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = sigma_range * sigma_range * omega[k] * alpha[k] * boost;
		__m256 malpha_k = _mm256_set1_ps(alphak);
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);
		const float base = float(2.0 * sqrt(CV_2PI) * omega[k] / T);
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			//l=0
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* splatCos = (__m256*)FourierPyramidCos[0].ptr<float>();
				__m256* splatSin = (__m256*)FourierPyramidSin[0].ptr<float>();
				const int SIZEREMAP = width * (D - 1) / 8;
				const int SIZEREMAP32 = SIZEREMAP / 4;
				const int SIZEREMAP8 = (SIZEREMAP * 8 - SIZEREMAP32 * 32) / 8;
				if (isUseFourierTable0)
				{
#ifdef USE_GATHER8U
					const __m64* sptr = (__m64*)(src8u.ptr<uchar>());
					const float* stable = &sinTable[FourierTableSize * k];
					const float* ctable = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZEREMAP32; ++i)
					{
						__m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					for (int i = 0; i < SIZEREMAP8; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
#else
					const __m256* gptr = (__m256*)GaussianPyramid[0].ptr<float>();
					const float* stable = &sinTable[FourierTableSize * k];
					const float* ctable = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZEREMAP32; ++i)
					{
						__m256i idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					for (int i = 0; i < SIZEREMAP8; ++i)
					{
						__m256i idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
#endif
				}
				else
				{
					__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
					for (int i = 0; i < SIZEREMAP32; i++)
					{
						__m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
					for (int i = 0; i < SIZEREMAP8; i++)
					{
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_cos = FourierPyramidCos[0].ptr<float>();
			float* sfpy_sin = FourierPyramidSin[0].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[1].ptr<float>(rs, rs);
			float* dfpyn_sin = FourierPyramidSin[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH32 = get_simd_floor(width, 32);
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND32 = get_simd_floor(hend, 32);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL32 = get_simd_floor(hendl, 32);
			const int HENDL = get_simd_floor(hendl, 8);

			const int SIZEREMAP = 2 * width / 8;
			const int SIZEREMAP32 = SIZEREMAP / 4;
			const int SIZEREMAP8 = (SIZEREMAP * 8 - SIZEREMAP32 * 32) / 8;

			const __m128i maskhend = get_storemask1(hend, 8);
			const __m256i maskhendl = get_simd_residualmask_epi32(hendl);
			__m256i maskhendll, maskhendlr;
			get_storemask2(hendl, maskhendll, maskhendlr, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* splatCos = (__m256*)(sfpy_cos + (D - 1) * width);
					__m256* splatSin = (__m256*)(sfpy_sin + (D - 1) * width);
					if (isUseFourierTable0)
					{
#ifdef USE_GATHER8U
						const __m64* sptr = (__m64*)(src8u.ptr<uchar>(j + D - 1));
						const float* stable = &sinTable[FourierTableSize * k];
						const float* ctable = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZEREMAP32; ++i)
						{
							__m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
						for (int i = 0; i < SIZEREMAP8; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
#else
						__m256* gptr = (__m256*)GaussianPyramid[0].ptr<float>(j + D - 1);
						const float* stable = &sinTable[FourierTableSize * k];
						const float* ctable = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZEREMAP32; i++)
						{
							__m256i idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
						for (int i = 0; i < SIZEREMAP8; i++)
						{
							const __m256i idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
#endif
					}
					else
					{
						__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
						for (int i = 0; i < SIZEREMAP32; i++)
						{
							__m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
						for (int i = 0; i < SIZEREMAP8; ++i)
						{
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
					}
				}

				//v filter
				__m256* spc = (__m256*)spcosline_e;
				__m256* sps = (__m256*)spsinline_e;
				for (int i = 0; i < WIDTH32; i += 32)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc));
					__m256 sumc1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 8));
					__m256 sumc2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 16));
					__m256 sumc3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 24));
					__m256 sums0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss));
					__m256 sums1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 8));
					__m256 sums2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 16));
					__m256 sums3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 24));
					ss += width;
					sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc0 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc0);
						sumc1 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 8), sumc1);
						sumc2 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 16), sumc2);
						sumc3 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 24), sumc3);
						sums0 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums0);
						sums1 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 8), sums1);
						sums2 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 16), sums2);
						sums3 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 24), sums3);
						sc += width;
						ss += width;
					}
					*spc++ = sumc0;
					*spc++ = sumc1;
					*spc++ = sumc2;
					*spc++ = sumc3;
					*sps++ = sums0;
					*sps++ = sums1;
					*sps++ = sums2;
					*sps++ = sums3;
				}
				for (int i = WIDTH32; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					*spc++ = sumc;
					*sps++ = sums;
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spcosline_e[i] = sumc;
					spsinline_e[i] = sums;
				}
				sfpy_cos += 2 * width;
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND32; i += 32)
				{
					float* cosi0 = spcosline_e + i;
					float* sini0 = spsinline_e + i;
					float* cosi1 = spcosline_e + i + 8;
					float* sini1 = spsinline_e + i + 8;
					float* cosi2 = spcosline_e + i + 16;
					float* sini2 = spsinline_e + i + 16;
					float* cosi3 = spcosline_e + i + 24;
					float* sini3 = spsinline_e + i + 24;
					__m256 sum0 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi0++), _mm256_loadu_ps(sini0++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum1 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi1++), _mm256_loadu_ps(sini1++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum2 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi2++), _mm256_loadu_ps(sini2++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum3 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi3++), _mm256_loadu_ps(sini3++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum0 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi0++), _mm256_loadu_ps(sini0++), _MM_SHUFFLE(2, 0, 2, 0)), sum0);
						sum1 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi1++), _mm256_loadu_ps(sini1++), _MM_SHUFFLE(2, 0, 2, 0)), sum1);
						sum2 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi2++), _mm256_loadu_ps(sini2++), _MM_SHUFFLE(2, 0, 2, 0)), sum2);
						sum3 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi3++), _mm256_loadu_ps(sini3++), _MM_SHUFFLE(2, 0, 2, 0)), sum3);
					}
					sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum0));

					sum1 = _mm256_permute4x64_ps(sum1, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 8) >> 1), _mm256_castps256_ps128(sum1));
					_mm_storeu_ps(dfpyn_sin + ((i + 8) >> 1), _mm256_castps256hi_ps128(sum1));

					sum2 = _mm256_permute4x64_ps(sum2, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 16) >> 1), _mm256_castps256_ps128(sum2));
					_mm_storeu_ps(dfpyn_sin + ((i + 16) >> 1), _mm256_castps256hi_ps128(sum2));

					sum3 = _mm256_permute4x64_ps(sum3, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 24) >> 1), _mm256_castps256_ps128(sum3));
					_mm_storeu_ps(dfpyn_sin + ((i + 24) >> 1), _mm256_castps256hi_ps128(sum3));
				}
				for (int i = HEND32; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;
					float* sini = spsinline_e + i;
#if 1
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum));
#else
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sumc = _mm256_shuffle_ps(sumc, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sumc));
#endif
				}
#ifdef MASKSTORE
				//last 
				{
					float* cosi = spcosline_e + HEND;
					float* sini = spsinline_e + HEND;
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_maskstore_ps(dfpyn_cos + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum));
					_mm_maskstore_ps(dfpyn_sin + (HEND >> 1), maskhend, _mm256_castps256hi_ps128(sum));
				}
#else 
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
					dfpyn_sin[i >> 1] = sums;
				}
#endif
				dfpyn_cos += widths;
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region Laplacian0
			float* stable = nullptr;
			float* ctable = nullptr;
			if constexpr (isUseFourierTable0)
			{
				stable = &sinTable[FourierTableSize * k];
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[1].ptr<float>(0, rs);
			sfpy_sin = FourierPyramidSin[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				__m256* spce = (__m256*)(spcosline_e + rs);
				__m256* spco = (__m256*)(spcosline_o + rs);
				__m256* spse = (__m256*)(spsinline_e + rs);
				__m256* spso = (__m256*)(spsinline_o + rs);
				for (int i = 0; i < HENDL32; i += 32)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					__m256 sumce0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc));
					__m256 sumse0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss));
					__m256 sumco0 = _mm256_setzero_ps();
					__m256 sumso0 = _mm256_setzero_ps();
					__m256 sumce1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 8));
					__m256 sumse1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 8));
					__m256 sumco1 = _mm256_setzero_ps();
					__m256 sumso1 = _mm256_setzero_ps();
					__m256 sumce2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 16));
					__m256 sumse2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 16));
					__m256 sumco2 = _mm256_setzero_ps();
					__m256 sumso2 = _mm256_setzero_ps();
					__m256 sumce3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 24));
					__m256 sumse3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 24));
					__m256 sumco3 = _mm256_setzero_ps();
					__m256 sumso3 = _mm256_setzero_ps();
					ss += widths;
					sc += widths;
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						__m256 msc = _mm256_loadu_ps(sc);
						sumce0 = _mm256_fmadd_ps(W[m], msc, sumce0);
						sumco0 = _mm256_fmadd_ps(W[m - 1], msc, sumco0);
						msc = _mm256_loadu_ps(sc + 8);
						sumce1 = _mm256_fmadd_ps(W[m], msc, sumce1);
						sumco1 = _mm256_fmadd_ps(W[m - 1], msc, sumco1);
						msc = _mm256_loadu_ps(sc + 16);
						sumce2 = _mm256_fmadd_ps(W[m], msc, sumce2);
						sumco2 = _mm256_fmadd_ps(W[m - 1], msc, sumco2);
						msc = _mm256_loadu_ps(sc + 24);
						sumce3 = _mm256_fmadd_ps(W[m], msc, sumce3);
						sumco3 = _mm256_fmadd_ps(W[m - 1], msc, sumco3);
						sc += widths;
						//sin
						__m256 mss = _mm256_loadu_ps(ss);
						sumse0 = _mm256_fmadd_ps(W[m], mss, sumse0);
						sumso0 = _mm256_fmadd_ps(W[m - 1], mss, sumso0);
						mss = _mm256_loadu_ps(ss + 8);
						sumse1 = _mm256_fmadd_ps(W[m], mss, sumse1);
						sumso1 = _mm256_fmadd_ps(W[m - 1], mss, sumso1);
						mss = _mm256_loadu_ps(ss + 16);
						sumse2 = _mm256_fmadd_ps(W[m], mss, sumse2);
						sumso2 = _mm256_fmadd_ps(W[m - 1], mss, sumso2);
						mss = _mm256_loadu_ps(ss + 24);
						sumse3 = _mm256_fmadd_ps(W[m], mss, sumse3);
						sumso3 = _mm256_fmadd_ps(W[m - 1], mss, sumso3);
						ss += widths;
					}
					*spce++ = _mm256_mul_ps(meven_ratio, sumce0);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce1);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce2);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce3);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco0);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco1);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco2);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco3);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse0);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse1);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse2);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse3);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso0);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso1);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso2);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso3);
				}
				for (int i = HENDL32; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						//sin
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					*spce++ = _mm256_mul_ps(meven_ratio, sumce);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso);
				}
#ifdef MASKSTORE
				{
					float* sc = sfpy_cos + HENDL;
					float* ss = sfpy_sin + HENDL;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						//sin
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_maskstore_ps(spcosline_e + HENDL + rs, maskhendl, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_maskstore_ps(spcosline_o + HENDL + rs, maskhendl, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_maskstore_ps(spsinline_e + HENDL + rs, maskhendl, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_maskstore_ps(spsinline_o + HENDL + rs, maskhendl, _mm256_mul_ps(modd__ratio, sumso));
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumco = 0.f;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumse += GaussWeight[m] * *ss;
						sumco += GaussWeight[m - 1] * *sc;
						sumso += GaussWeight[m - 1] * *ss;
						sc += widths;
						ss += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}
#endif

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 msin, mcos;
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#ifdef MASKSTORE
				//last 
				{
					float* cosie = spcosline_e + HENDL;
					float* cosio = spcosline_o + HENDL;
					float* sinie = spsinline_e + HENDL;
					float* sinio = spsinline_o + HENDL;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = (HENDL << 1);
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 msin, mcos;
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * (sin(os) * sumcee - cos(os) * sumsee);
					}
					else
					{
						dste[I] += alphak * evenratio * (sin(os) * sumcee - cos(os) * sumsee);
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * (sin(os) * sumcoe - cos(os) * sumsoe);
					}
					else
					{
						dste[I + 1] += alphak * oddratio * (sin(os) * sumcoe - cos(os) * sumsoe);
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * (sin(os) * sumceo - cos(os) * sumseo);
					}
					else
					{
						dsto[I] += alphak * evenratio * (sin(os) * sumceo - cos(os) * sumseo);
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * (sin(os) * sumcoo - cos(os) * sumsoo);
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * (sin(os) * sumcoo - cos(os) * sumsoo);
					}
				}
#endif
				sfpy_cos += widths;
				sfpy_sin += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			const __m128i maskhend = get_storemask1(hend, 8);
			const __m256i maskhendl = get_simd_residualmask_epi32(hendl);
			__m256i maskhendll, maskhendlr;
			get_storemask2(hendl, maskhendll, maskhendlr, 8);

#pragma region GaussianLevel
			float* sfpy_cos = FourierPyramidCos[l].ptr<float>();
			float* sfpy_sin = FourierPyramidSin[l].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[l + 1].ptr<float>(rs, rs);
			float* dfpyn_sin = FourierPyramidSin[l + 1].ptr<float>(rs, rs);
			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spcosline_e[i] = sumc;
					spsinline_e[i] = sums;
				}
				sfpy_cos += 2 * width;
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;
					float* sini = spsinline_e + i;
#if 1
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum));
#else
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sumc = _mm256_shuffle_ps(sumc, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sumc));
#endif
				}
#ifdef MASKSTORE
				{
					float* cosi = spcosline_e + HEND;
					float* sini = spsinline_e + HEND;
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_maskstore_ps(dfpyn_cos + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum));
					_mm_maskstore_ps(dfpyn_sin + (HEND >> 1), maskhend, _mm256_castps256hi_ps128(sum));
				}
#else
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
					dfpyn_sin[i >> 1] = sums;
				}
#endif
				dfpyn_cos += widths;
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region LaplacianLevel
			float* stable = nullptr;
			float* ctable = nullptr;
			if constexpr (isUseFourierTableLevel)
			{
				stable = &sinTable[FourierTableSize * k];
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[l + 1].ptr<float>(0, rs);
			sfpy_sin = FourierPyramidSin[l + 1].ptr<float>(0, rs);
			float* ppye_cos = FourierPyramidCos[l].ptr<float>(radius, radius);
			float* ppye_sin = FourierPyramidSin[l].ptr<float>(radius, radius);
			float* ppyo_cos = FourierPyramidCos[l].ptr<float>(radius + 1, radius);
			float* ppyo_sin = FourierPyramidSin[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(modd__ratio, sumso));
				}
#ifdef MASKSTORE
				{
					float* sc = sfpy_cos + HENDL;
					float* ss = sfpy_sin + HENDL;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_maskstore_ps(spcosline_e + rs + HENDL, maskhendl, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_maskstore_ps(spsinline_e + rs + HENDL, maskhendl, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_maskstore_ps(spcosline_o + rs + HENDL, maskhendl, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_maskstore_ps(spsinline_o + rs + HENDL, maskhendl, _mm256_mul_ps(modd__ratio, sumso));
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumco = 0.f;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumse += GaussWeight[m] * *ss;
						sumco += GaussWeight[m - 1] * *sc;
						sumso += GaussWeight[m - 1] * *ss;
						sc += widths;
						ss += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spsinline_e[i + rs] = sumse * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}
#endif

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}
					const int I = i << 1;
					//even line
					__m256 cos0, cos1, sin0, sin1;
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 msin, mcos;
					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}

				}
#ifdef MASKSTORE
				if (hendl != HENDL)//last
				{
					float* cosie = spcosline_e + HENDL;
					float* cosio = spcosline_o + HENDL;
					float* sinie = spsinline_e + HENDL;
					float* sinio = spsinline_o + HENDL;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = (HENDL << 1);
					//even line
					__m256 cos0, cos1, sin0, sin1;
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 msin, mcos;
					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]) - cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					else
					{
						dste[I] += alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]) - cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]) - cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]) - cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]) - cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					else
					{
						dsto[I] += alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]) - cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]) - cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]) - cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
				}
#endif
				sfpy_cos += widths;
				sfpy_sin += widths;
				ppye_cos += 2 * width;
				ppye_sin += 2 * width;
				ppyo_cos += 2 * width;
				ppyo_sin += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(linebuffer);
		_mm_free(W);
	}

	template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseFourierTableLevel>
	void LocalMultiScaleFilterFourier::buildLaplacianFourierPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidCos, vector<Mat>& FourierPyramidSin)
	{
		const int rs = radius >> 1;
		const int D = 2 * radius + 1;
		const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidCos.size() != level + 1)FourierPyramidCos.resize(level + 1);
		if (FourierPyramidSin.size() != level + 1)FourierPyramidSin.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidCos[0].create(imSize, CV_32F);
		FourierPyramidSin[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidCos[l].create(pySize, CV_32F);
			FourierPyramidSin[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidCos[level].create(pySize, CV_32F);
			FourierPyramidSin[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* linebuffer = (float*)_mm_malloc(sizeof(float) * linesize * 4, AVX_ALIGN);;
		float* spcosline_e = linebuffer + 0 * linesize;
		float* spsinline_e = linebuffer + 1 * linesize;
		float* spcosline_o = linebuffer + 2 * linesize;
		float* spsinline_o = linebuffer + 3 * linesize;

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 meven_ratio = _mm256_set1_ps(evenratio);
		const __m256 modd__ratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = sigma_range * sigma_range * omega[k] * alpha[k] * boost;
		__m256 malpha_k = _mm256_set1_ps(alphak);
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);
		const float base = float(2.0 * sqrt(CV_2PI) * omega[k] / T);
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			//l=0
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* splatCos = (__m256*)FourierPyramidCos[0].ptr<float>();
				__m256* splatSin = (__m256*)FourierPyramidSin[0].ptr<float>();
				const int SIZEREMAP = width * (D - 1) / 8;
				const int SIZEREMAP32 = SIZEREMAP / 4;
				const int SIZEREMAP8 = (SIZEREMAP * 8 - SIZEREMAP32 * 32) / 8;
				if (isUseFourierTable0)
				{
#ifdef USE_GATHER8U
					const __m64* sptr = (__m64*)(src8u.ptr<uchar>());
					const float* stable = &sinTable[FourierTableSize * k];
					const float* ctable = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZEREMAP32; ++i)
					{
						__m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					for (int i = 0; i < SIZEREMAP8; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
#else
					const __m256* gptr = (__m256*)GaussianPyramid[0].ptr<float>();
					const float* stable = &sinTable[FourierTableSize * k];
					const float* ctable = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZEREMAP32; ++i)
					{
						__m256i idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					for (int i = 0; i < SIZEREMAP8; ++i)
					{
						const __m256i idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
#endif
				}
				else
				{
					__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
					for (int i = 0; i < SIZEREMAP32; i++)
					{
						__m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
					for (int i = 0; i < SIZEREMAP8; i++)
					{
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_cos = FourierPyramidCos[0].ptr<float>();
			float* sfpy_sin = FourierPyramidSin[0].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[1].ptr<float>(rs, rs);
			float* dfpyn_sin = FourierPyramidSin[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH32 = get_simd_floor(width, 32);
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND32 = get_simd_floor(hend, 32);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL32 = get_simd_floor(hendl, 32);
			const int HENDL = get_simd_floor(hendl, 8);

			const int SIZEREMAP = 2 * width / 8;
			const int SIZEREMAP32 = SIZEREMAP / 4;
			const int SIZEREMAP8 = (SIZEREMAP * 8 - SIZEREMAP32 * 32) / 8;

			const __m128i maskhend = get_storemask1(hend, 8);
			const __m256i maskhendl = get_simd_residualmask_epi32(hendl);
			__m256i maskhendll, maskhendlr;
			get_storemask2(hendl, maskhendll, maskhendlr, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* splatCos = (__m256*)(sfpy_cos + (D - 1) * width);
					__m256* splatSin = (__m256*)(sfpy_sin + (D - 1) * width);
					if (isUseFourierTable0)
					{
#ifdef USE_GATHER8U
						const __m64* sptr = (__m64*)(src8u.ptr<uchar>(j + D - 1));
						const float* stable = &sinTable[FourierTableSize * k];
						const float* ctable = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZEREMAP32; ++i)
						{
							__m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
						for (int i = 0; i < SIZEREMAP8; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
#else
						__m256* gptr = (__m256*)GaussianPyramid[0].ptr<float>(j + D - 1);
						const float* stable = &sinTable[FourierTableSize * k];
						const float* ctable = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZEREMAP32; i++)
						{
							__m256i idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
						for (int i = 0; i < SIZEREMAP8; i++)
						{
							const __m256i idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
#endif
					}
					else
					{
						__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
						for (int i = 0; i < SIZEREMAP32; i++)
						{
							__m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
						for (int i = 0; i < SIZEREMAP8; ++i)
						{
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
					}
				}

				//v filter
				__m256* spc = (__m256*)spcosline_e;
				__m256* sps = (__m256*)spsinline_e;
				for (int i = 0; i < WIDTH32; i += 32)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc));
					__m256 sumc1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 8));
					__m256 sumc2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 16));
					__m256 sumc3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 24));
					__m256 sums0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss));
					__m256 sums1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 8));
					__m256 sums2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 16));
					__m256 sums3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 24));
					ss += width;
					sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc0 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc0);
						sumc1 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 8), sumc1);
						sumc2 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 16), sumc2);
						sumc3 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 24), sumc3);
						sums0 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums0);
						sums1 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 8), sums1);
						sums2 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 16), sums2);
						sums3 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 24), sums3);
						sc += width;
						ss += width;
					}
					*spc++ = sumc0;
					*spc++ = sumc1;
					*spc++ = sumc2;
					*spc++ = sumc3;
					*sps++ = sums0;
					*sps++ = sums1;
					*sps++ = sums2;
					*sps++ = sums3;
				}
				for (int i = WIDTH32; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					*spc++ = sumc;
					*sps++ = sums;
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spcosline_e[i] = sumc;
					spsinline_e[i] = sums;
				}
				sfpy_cos += 2 * width;
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND32; i += 32)
				{
					float* cosi0 = spcosline_e + i;
					float* sini0 = spsinline_e + i;
					float* cosi1 = spcosline_e + i + 8;
					float* sini1 = spsinline_e + i + 8;
					float* cosi2 = spcosline_e + i + 16;
					float* sini2 = spsinline_e + i + 16;
					float* cosi3 = spcosline_e + i + 24;
					float* sini3 = spsinline_e + i + 24;
					__m256 sum0 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi0++), _mm256_loadu_ps(sini0++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum1 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi1++), _mm256_loadu_ps(sini1++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum2 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi2++), _mm256_loadu_ps(sini2++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum3 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi3++), _mm256_loadu_ps(sini3++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum0 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi0++), _mm256_loadu_ps(sini0++), _MM_SHUFFLE(2, 0, 2, 0)), sum0);
						sum1 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi1++), _mm256_loadu_ps(sini1++), _MM_SHUFFLE(2, 0, 2, 0)), sum1);
						sum2 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi2++), _mm256_loadu_ps(sini2++), _MM_SHUFFLE(2, 0, 2, 0)), sum2);
						sum3 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi3++), _mm256_loadu_ps(sini3++), _MM_SHUFFLE(2, 0, 2, 0)), sum3);
					}
					sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum0));

					sum1 = _mm256_permute4x64_ps(sum1, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 8) >> 1), _mm256_castps256_ps128(sum1));
					_mm_storeu_ps(dfpyn_sin + ((i + 8) >> 1), _mm256_castps256hi_ps128(sum1));

					sum2 = _mm256_permute4x64_ps(sum2, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 16) >> 1), _mm256_castps256_ps128(sum2));
					_mm_storeu_ps(dfpyn_sin + ((i + 16) >> 1), _mm256_castps256hi_ps128(sum2));

					sum3 = _mm256_permute4x64_ps(sum3, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 24) >> 1), _mm256_castps256_ps128(sum3));
					_mm_storeu_ps(dfpyn_sin + ((i + 24) >> 1), _mm256_castps256hi_ps128(sum3));
				}
				for (int i = HEND32; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;
					float* sini = spsinline_e + i;
#if 1
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum));
#else
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sumc = _mm256_shuffle_ps(sumc, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sumc));
#endif
				}
#ifdef MASKSTORE
				//last 
				{
					float* cosi = spcosline_e + HEND;
					float* sini = spsinline_e + HEND;
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_maskstore_ps(dfpyn_cos + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum));
					_mm_maskstore_ps(dfpyn_sin + (HEND >> 1), maskhend, _mm256_castps256hi_ps128(sum));
				}
#else 
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
					dfpyn_sin[i >> 1] = sums;
				}
#endif
				dfpyn_cos += widths;
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region Laplacian0
			float* stable = nullptr;
			float* ctable = nullptr;
			if constexpr (isUseFourierTable0)
			{
				stable = &sinTable[FourierTableSize * k];
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[1].ptr<float>(0, rs);
			sfpy_sin = FourierPyramidSin[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				__m256* spce = (__m256*)(spcosline_e + rs);
				__m256* spco = (__m256*)(spcosline_o + rs);
				__m256* spse = (__m256*)(spsinline_e + rs);
				__m256* spso = (__m256*)(spsinline_o + rs);
				for (int i = 0; i < HENDL32; i += 32)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					__m256 sumce0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc));
					__m256 sumse0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss));
					__m256 sumco0 = _mm256_setzero_ps();
					__m256 sumso0 = _mm256_setzero_ps();
					__m256 sumce1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 8));
					__m256 sumse1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 8));
					__m256 sumco1 = _mm256_setzero_ps();
					__m256 sumso1 = _mm256_setzero_ps();
					__m256 sumce2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 16));
					__m256 sumse2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 16));
					__m256 sumco2 = _mm256_setzero_ps();
					__m256 sumso2 = _mm256_setzero_ps();
					__m256 sumce3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 24));
					__m256 sumse3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 24));
					__m256 sumco3 = _mm256_setzero_ps();
					__m256 sumso3 = _mm256_setzero_ps();
					ss += widths;
					sc += widths;
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						__m256 msc = _mm256_loadu_ps(sc);
						sumce0 = _mm256_fmadd_ps(W[m], msc, sumce0);
						sumco0 = _mm256_fmadd_ps(W[m - 1], msc, sumco0);
						msc = _mm256_loadu_ps(sc + 8);
						sumce1 = _mm256_fmadd_ps(W[m], msc, sumce1);
						sumco1 = _mm256_fmadd_ps(W[m - 1], msc, sumco1);
						msc = _mm256_loadu_ps(sc + 16);
						sumce2 = _mm256_fmadd_ps(W[m], msc, sumce2);
						sumco2 = _mm256_fmadd_ps(W[m - 1], msc, sumco2);
						msc = _mm256_loadu_ps(sc + 24);
						sumce3 = _mm256_fmadd_ps(W[m], msc, sumce3);
						sumco3 = _mm256_fmadd_ps(W[m - 1], msc, sumco3);
						sc += widths;
						//sin
						__m256 mss = _mm256_loadu_ps(ss);
						sumse0 = _mm256_fmadd_ps(W[m], mss, sumse0);
						sumso0 = _mm256_fmadd_ps(W[m - 1], mss, sumso0);
						mss = _mm256_loadu_ps(ss + 8);
						sumse1 = _mm256_fmadd_ps(W[m], mss, sumse1);
						sumso1 = _mm256_fmadd_ps(W[m - 1], mss, sumso1);
						mss = _mm256_loadu_ps(ss + 16);
						sumse2 = _mm256_fmadd_ps(W[m], mss, sumse2);
						sumso2 = _mm256_fmadd_ps(W[m - 1], mss, sumso2);
						mss = _mm256_loadu_ps(ss + 24);
						sumse3 = _mm256_fmadd_ps(W[m], mss, sumse3);
						sumso3 = _mm256_fmadd_ps(W[m - 1], mss, sumso3);
						ss += widths;
					}
					*spce++ = _mm256_mul_ps(meven_ratio, sumce0);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce1);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce2);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce3);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco0);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco1);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco2);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco3);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse0);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse1);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse2);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse3);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso0);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso1);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso2);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso3);
				}
				for (int i = HENDL32; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						//sin
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					*spce++ = _mm256_mul_ps(meven_ratio, sumce);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso);
				}
#ifdef MASKSTORE
				{
					float* sc = sfpy_cos + HENDL;
					float* ss = sfpy_sin + HENDL;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						//sin
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_maskstore_ps(spcosline_e + HENDL + rs, maskhendl, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_maskstore_ps(spcosline_o + HENDL + rs, maskhendl, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_maskstore_ps(spsinline_e + HENDL + rs, maskhendl, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_maskstore_ps(spsinline_o + HENDL + rs, maskhendl, _mm256_mul_ps(modd__ratio, sumso));
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumco = 0.f;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumse += GaussWeight[m] * *ss;
						sumco += GaussWeight[m - 1] * *sc;
						sumso += GaussWeight[m - 1] * *ss;
						sc += widths;
						ss += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}
#endif

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 msin, mcos;
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#ifdef MASKSTORE
				//last 
				{
					float* cosie = spcosline_e + HENDL;
					float* cosio = spcosline_o + HENDL;
					float* sinie = spsinline_e + HENDL;
					float* sinio = spsinline_o + HENDL;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = (HENDL << 1);
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 msin, mcos;
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * (sin(os) * sumcee - cos(os) * sumsee);
					}
					else
					{
						dste[I] += alphak * evenratio * (sin(os) * sumcee - cos(os) * sumsee);
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * (sin(os) * sumcoe - cos(os) * sumsoe);
					}
					else
					{
						dste[I + 1] += alphak * oddratio * (sin(os) * sumcoe - cos(os) * sumsoe);
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * (sin(os) * sumceo - cos(os) * sumseo);
					}
					else
					{
						dsto[I] += alphak * evenratio * (sin(os) * sumceo - cos(os) * sumseo);
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * (sin(os) * sumcoo - cos(os) * sumsoo);
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * (sin(os) * sumcoo - cos(os) * sumsoo);
					}
				}
#endif
				sfpy_cos += widths;
				sfpy_sin += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

#pragma region GaussianLevel
			float* sfpy_cos = FourierPyramidCos[l].ptr<float>();
			float* sfpy_sin = FourierPyramidSin[l].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[l + 1].ptr<float>(rs, rs);
			float* dfpyn_sin = FourierPyramidSin[l + 1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			const __m128i maskhend = get_storemask1(hend, 8);
			const __m256i maskhendl = get_simd_residualmask_epi32(hendl);
			__m256i maskhendll, maskhendlr;
			get_storemask2(hendl, maskhendll, maskhendlr, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spcosline_e[i] = sumc;
					spsinline_e[i] = sums;
				}
				sfpy_cos += 2 * width;
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;
					float* sini = spsinline_e + i;
#if 1
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum));
#else
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sumc = _mm256_shuffle_ps(sumc, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sumc));
#endif
				}
#ifdef MASKSTORE
				{
					float* cosi = spcosline_e + HEND;
					float* sini = spsinline_e + HEND;
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_maskstore_ps(dfpyn_cos + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum));
					_mm_maskstore_ps(dfpyn_sin + (HEND >> 1), maskhend, _mm256_castps256hi_ps128(sum));
				}
#else
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
					dfpyn_sin[i >> 1] = sums;
				}
#endif
				dfpyn_cos += widths;
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region LaplacianLevel
			float* stable = nullptr;
			float* ctable = nullptr;
			if constexpr (isUseFourierTableLevel)
			{
				stable = &sinTable[FourierTableSize * k];
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[l + 1].ptr<float>(0, rs);
			sfpy_sin = FourierPyramidSin[l + 1].ptr<float>(0, rs);
			float* ppye_cos = FourierPyramidCos[l].ptr<float>(radius, radius);
			float* ppye_sin = FourierPyramidSin[l].ptr<float>(radius, radius);
			float* ppyo_cos = FourierPyramidCos[l].ptr<float>(radius + 1, radius);
			float* ppyo_sin = FourierPyramidSin[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(modd__ratio, sumso));
				}
#ifdef MASKSTORE
				{
					float* sc = sfpy_cos + HENDL;
					float* ss = sfpy_sin + HENDL;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_maskstore_ps(spcosline_e + rs + HENDL, maskhendl, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_maskstore_ps(spsinline_e + rs + HENDL, maskhendl, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_maskstore_ps(spcosline_o + rs + HENDL, maskhendl, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_maskstore_ps(spsinline_o + rs + HENDL, maskhendl, _mm256_mul_ps(modd__ratio, sumso));
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumco = 0.f;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumse += GaussWeight[m] * *ss;
						sumco += GaussWeight[m - 1] * *sc;
						sumso += GaussWeight[m - 1] * *ss;
						sc += widths;
						ss += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spsinline_e[i + rs] = sumse * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}
#endif

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}
					const int I = i << 1;
					//even line
					__m256 cos0, cos1, sin0, sin1;
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 msin, mcos;
					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}

				}
#ifdef MASKSTORE
				//last
				{
					float* cosie = spcosline_e + HENDL;
					float* cosio = spcosline_o + HENDL;
					float* sinie = spsinline_e + HENDL;
					float* sinio = spsinline_o + HENDL;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = (HENDL << 1);
					//even line
					__m256 cos0, cos1, sin0, sin1;
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 msin, mcos;
					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]) - cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					else
					{
						dste[I] += alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]) - cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]) - cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]) - cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]) - cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					else
					{
						dsto[I] += alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]) - cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]) - cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]) - cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
				}
#endif
				sfpy_cos += widths;
				sfpy_sin += widths;
				ppye_cos += 2 * width;
				ppye_sin += 2 * width;
				ppyo_cos += 2 * width;
				ppyo_sin += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(linebuffer);
		_mm_free(W);
	}




	template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseFourierTableLevel, int D, int D2>
	void LocalMultiScaleFilterFourier::buildLaplacianSinPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidSin)
	{
		const int rs = radius >> 1;
		//const int D = 2 * radius + 1;
		//const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidSin.size() != level + 1)FourierPyramidSin.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidSin[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidSin[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidSin[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* spsinline_e = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		float* spsinline_o = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = -sigma_range * sigma_range * omega[k] * alpha[k] * boost;//-alphak
		__m256 malpha_k = _mm256_set1_ps(alphak);//-alphak
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);//-alphak
		const float base = -float(2.0 * sqrt(CV_2PI) * omega[k] / T);//-base
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
				__m256* splatSin = (__m256*)FourierPyramidSin[0].ptr<float>();
				const int SIZE = width * (D - 1) / 8;

				if (isUseFourierTable0)
				{
					const __m64* sptr = (__m64*)src8u.ptr<uchar>();
					const float* sinptr = &sinTable[FourierTableSize * k];
					for (int i = 0; i < SIZE; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatSin++) = _mm256_i32gather_ps(sinptr, idx, sizeof(float));
					}
				}
				else
				{
					for (int i = 0; i < SIZE; ++i)
					{
						//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_sin = FourierPyramidSin[0].ptr<float>();
			float* dfpyn_sin = FourierPyramidSin[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
					__m256* splatSin = (__m256*)(sfpy_sin + (D - 1) * width);
					const int SIZE = 2 * width / 8;

					if (isUseFourierTable0)
					{
						const __m64* guidePtr = (__m64*)src8u.ptr<uchar>(j + D - 1);
						const float* sxiPtr = &sinTable[FourierTableSize * k];
						for (int i = 0; i < SIZE; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
							*(splatSin++) = _mm256_i32gather_ps(sxiPtr, idx, sizeof(float));
						}
					}
					else
					{
						for (int i = 0; i < SIZE; ++i)
						{
							//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
					}
				}
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* ss = sfpy_sin + i;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* ss = sfpy_sin + i;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spsinline_e[i] = sums;
				}
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* sini = spsinline_e + i;

					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sums = _mm256_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sums = _mm256_permute4x64_ps(sums, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256_ps128(sums));
				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_sin[i >> 1] = sums;
				}
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region Laplacian0
			float* ctable = nullptr;
			if constexpr (isUseFourierTable0)
			{
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_sin = FourierPyramidSin[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* ss = sfpy_sin + i;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(mevenratio, sumse));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(moddratio, sumso));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* ss = sfpy_sin + i;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumse += GaussWeight[m] * *ss;
						sumso += GaussWeight[m - 1] * *ss;
						ss += widths;
					}
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					__m256 s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 mcos;
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * cos(os) * sumsee;
					}
					else
					{
						dste[I] += alphak * evenratio * cos(os) * sumsee;
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * cos(os) * sumsoe;
					}
					else
					{
						dste[I + 1] += alphak * oddratio * cos(os) * sumsoe;
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * cos(os) * sumseo;
					}
					else
					{
						dsto[I] += alphak * evenratio * cos(os) * sumseo;
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * cos(os) * sumsoo;
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * cos(os) * sumsoo;
					}
				}
				sfpy_sin += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

#pragma region GaussianLevel
			float* sfpy_sin = FourierPyramidSin[l].ptr<float>();
			float* dfpyn_sin = FourierPyramidSin[l + 1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* ss = sfpy_sin + i;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* ss = sfpy_sin + i;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spsinline_e[i] = sums;
				}
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* sini = spsinline_e + i;

					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sums = _mm256_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sums = _mm256_permute4x64_ps(sums, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256_ps128(sums));

				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_sin[i >> 1] = sums;
				}
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region LaplacianLevel

			float* ctable = nullptr;
			if constexpr (isUseFourierTableLevel)
			{
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_sin = FourierPyramidSin[l + 1].ptr<float>(0, rs);
			float* ppye_sin = FourierPyramidSin[l].ptr<float>(radius, radius);
			float* ppyo_sin = FourierPyramidSin[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* ss = sfpy_sin + i;

					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(mevenratio, sumse));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(moddratio, sumso));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* ss = sfpy_sin + i;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumse += GaussWeight[m] * *ss;
						sumso += GaussWeight[m - 1] * *ss;
						ss += widths;
					}
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					//even line
					__m256 temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 mcos;
					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						mcos = _mm256_cos_ps(ms);
					}
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						mcos = _mm256_cos_ps(ms);
					}
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					else
					{
						dste[I] += alphak * (cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					else
					{
						dsto[I] += alphak * (cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
				}
				sfpy_sin += widths;
				ppye_sin += 2 * width;
				ppyo_sin += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(spsinline_o);
		_mm_free(spsinline_e);
		_mm_free(W);
	}

	template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseFourierTableLevel>
	void LocalMultiScaleFilterFourier::buildLaplacianSinPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidSin)
	{
		const int rs = radius >> 1;
		const int D = 2 * radius + 1;
		const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidSin.size() != level + 1)FourierPyramidSin.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidSin[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidSin[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidSin[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* spsinline_e = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		float* spsinline_o = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = -sigma_range * sigma_range * omega[k] * alpha[k] * boost;//-alphak
		__m256 malpha_k = _mm256_set1_ps(alphak);//-alphak
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);//-alphak
		const float base = -float(2.0 * sqrt(CV_2PI) * omega[k] / T);//-base
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
				__m256* splatSin = (__m256*)FourierPyramidSin[0].ptr<float>();
				const int SIZE = width * (D - 1) / 8;

				if (isUseFourierTable0)
				{
					const __m64* guidePtr = (__m64*)src8u.ptr<uchar>();
					const float* sxiPtr = &sinTable[FourierTableSize * k];
					for (int i = 0; i < SIZE; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
						*(splatSin++) = _mm256_i32gather_ps(sxiPtr, idx, sizeof(float));
					}
				}
				else
				{
					for (int i = 0; i < SIZE; ++i)
					{
						//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_sin = FourierPyramidSin[0].ptr<float>();
			float* dfpyn_sin = FourierPyramidSin[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
					__m256* splatSin = (__m256*)(sfpy_sin + (D - 1) * width);
					const int SIZE = 2 * width / 8;

					if (isUseFourierTable0)
					{
						const __m64* guidePtr = (__m64*)src8u.ptr<uchar>(j + D - 1);
						const float* sxiPtr = &sinTable[FourierTableSize * k];
						for (int i = 0; i < SIZE; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
							*(splatSin++) = _mm256_i32gather_ps(sxiPtr, idx, sizeof(float));
						}
					}
					else
					{
						for (int i = 0; i < SIZE; ++i)
						{
							//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
					}
				}
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* ss = sfpy_sin + i;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* ss = sfpy_sin + i;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spsinline_e[i] = sums;
				}
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* sini = spsinline_e + i;

					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sums = _mm256_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sums = _mm256_permute4x64_ps(sums, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256_ps128(sums));
				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_sin[i >> 1] = sums;
				}
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region Laplacian0
			float* ctable = nullptr;
			if constexpr (isUseFourierTable0)
			{
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_sin = FourierPyramidSin[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* ss = sfpy_sin + i;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(mevenratio, sumse));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(moddratio, sumso));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* ss = sfpy_sin + i;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumse += GaussWeight[m] * *ss;
						sumso += GaussWeight[m - 1] * *ss;
						ss += widths;
					}
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					__m256 s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 mcos;
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * cos(os) * sumsee;
					}
					else
					{
						dste[I] += alphak * evenratio * cos(os) * sumsee;
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * cos(os) * sumsoe;
					}
					else
					{
						dste[I + 1] += alphak * oddratio * cos(os) * sumsoe;
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * cos(os) * sumseo;
					}
					else
					{
						dsto[I] += alphak * evenratio * cos(os) * sumseo;
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * cos(os) * sumsoo;
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * cos(os) * sumsoo;
					}
				}
				sfpy_sin += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

#pragma region GaussianLevel
			float* sfpy_sin = FourierPyramidSin[l].ptr<float>();
			float* dfpyn_sin = FourierPyramidSin[l + 1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* ss = sfpy_sin + i;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* ss = sfpy_sin + i;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spsinline_e[i] = sums;
				}
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* sini = spsinline_e + i;

					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sums = _mm256_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sums = _mm256_permute4x64_ps(sums, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256_ps128(sums));

				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_sin[i >> 1] = sums;
				}
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region LaplacianLevel

			float* ctable = nullptr;
			if constexpr (isUseFourierTableLevel)
			{
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_sin = FourierPyramidSin[l + 1].ptr<float>(0, rs);
			float* ppye_sin = FourierPyramidSin[l].ptr<float>(radius, radius);
			float* ppyo_sin = FourierPyramidSin[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* ss = sfpy_sin + i;

					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(mevenratio, sumse));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(moddratio, sumso));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* ss = sfpy_sin + i;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumse += GaussWeight[m] * *ss;
						sumso += GaussWeight[m - 1] * *ss;
						ss += widths;
					}
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					//even line
					__m256 temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 mcos;
					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						mcos = _mm256_cos_ps(ms);
					}
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						mcos = _mm256_cos_ps(ms);
					}
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					else
					{
						dste[I] += alphak * (cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					else
					{
						dsto[I] += alphak * (cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
				}
				sfpy_sin += widths;
				ppye_sin += 2 * width;
				ppyo_sin += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(spsinline_o);
		_mm_free(spsinline_e);
		_mm_free(W);
	}


	template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseFourierTableLevel>
	void LocalMultiScaleFilterFourier::buildLaplacianCosPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidCos)
	{
		const int rs = radius >> 1;
		const int D = 2 * radius + 1;
		const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidCos.size() != level + 1)FourierPyramidCos.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidCos[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidCos[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidCos[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* spcosline_e = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		float* spcosline_o = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = sigma_range * sigma_range * omega[k] * alpha[k] * boost;
		__m256 malpha_k = _mm256_set1_ps(alphak);
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);
		const float base = float(2.0 * sqrt(CV_2PI) * omega[k] / T);
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
				__m256* splatCos = (__m256*)FourierPyramidCos[0].ptr<float>();
				const int SIZE = width * (D - 1) / 8;

				if (isUseFourierTable0)
				{
					const __m64* guidePtr = (__m64*)src8u.ptr<uchar>();
					const float* cxiPtr = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZE; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
						*(splatCos++) = _mm256_i32gather_ps(cxiPtr, idx, sizeof(float));
					}
				}
				else
				{
					for (int i = 0; i < SIZE; ++i)
					{
						//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_cos = FourierPyramidCos[0].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
					__m256* splatCos = (__m256*)(sfpy_cos + (D - 1) * width);
					const int SIZE = 2 * width / 8;

					if (isUseFourierTable0)
					{
						const __m64* guidePtr = (__m64*)src8u.ptr<uchar>(j + D - 1);
						const float* cxiPtr = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZE; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
							*(splatCos++) = _mm256_i32gather_ps(cxiPtr, idx, sizeof(float));
						}
					}
					else
					{
						for (int i = 0; i < SIZE; ++i)
						{
							//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
						}
					}
				}
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
					}
					spcosline_e[i] = sumc;
				}
				sfpy_cos += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;

					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
					}
					sumc = _mm256_shuffle_ps(sumc, sumc, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
				}
				dfpyn_cos += widths;
			}
#pragma endregion

#pragma region Laplacian0
			float* stable = nullptr;
			if constexpr (isUseFourierTable0)
			{
				stable = &sinTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumco = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(mevenratio, sumce));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(moddratio, sumco));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumco = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumco += GaussWeight[m - 1] * *sc;
						sc += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
				}

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 msin;
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_sin_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_sin_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_sin_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (isUseFourierTable0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_sin_ps(ms);
					}
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * sin(os) * sumcee;
					}
					else
					{
						dste[I] += alphak * evenratio * sin(os) * sumcee;
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * sin(os) * sumcoe;
					}
					else
					{
						dste[I + 1] += alphak * oddratio * sin(os) * sumcoe;
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * sin(os) * sumceo;
					}
					else
					{
						dsto[I] += alphak * evenratio * sin(os) * sumceo;
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * sin(os) * sumcoo;
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * sin(os) * sumcoo;
					}
				}
				sfpy_cos += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

#pragma region GaussianLevel
			float* sfpy_cos = FourierPyramidCos[l].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[l + 1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
					}
					spcosline_e[i] = sumc;
				}
				sfpy_cos += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;

					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
					}
					sumc = _mm256_shuffle_ps(sumc, sumc, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));

				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
				}
				dfpyn_cos += widths;
			}
#pragma endregion

#pragma region LaplacianLevel
			float* stable = nullptr;
			if constexpr (isUseFourierTableLevel)
			{
				stable = &sinTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[l + 1].ptr<float>(0, rs);
			float* ppye_cos = FourierPyramidCos[l].ptr<float>(radius, radius);
			float* ppyo_cos = FourierPyramidCos[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumco = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(mevenratio, sumce));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(moddratio, sumco));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumco = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumco += GaussWeight[m - 1] * *sc;
						sc += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
				}

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
					}

					const int I = i << 1;
					//even line
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 msin;
					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						msin = _mm256_sin_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_sin_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_sin_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (isUseFourierTableLevel)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_sin_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]));
					}
					else
					{
						dste[I] += alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]));
					}
					else
					{
						dsto[I] += alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]));
					}
				}
				sfpy_cos += widths;
				ppye_cos += 2 * width;
				ppyo_cos += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(spcosline_o);
		_mm_free(spcosline_e);
		_mm_free(W);
	}

	template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseFourierTableLevel, int D, int D2>
	void LocalMultiScaleFilterFourier::buildLaplacianCosPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidCos)
	{
		const int rs = radius >> 1;
		//const int D = 2 * radius + 1;
		//const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidCos.size() != level + 1)FourierPyramidCos.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		//if(destPyramid[0].size()!=imSize)
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidCos[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidCos[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidCos[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* spcosline_e = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		float* spcosline_o = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = sigma_range * sigma_range * omega[k] * alpha[k] * boost;
		__m256 malpha_k = _mm256_set1_ps(alphak);
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);
		const float base = float(2.0 * sqrt(CV_2PI) * omega[k] / T);
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
				__m256* splatCos = (__m256*)FourierPyramidCos[0].ptr<float>();
				const int SIZE = width * (D - 1) / 8;

				if (isUseFourierTable0)
				{
					const __m64* guidePtr = (__m64*)src8u.ptr<uchar>();
					const float* cxiPtr = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZE; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
						*(splatCos++) = _mm256_i32gather_ps(cxiPtr, idx, sizeof(float));
					}
				}
				else
				{
					for (int i = 0; i < SIZE; ++i)
					{
						//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_cos = FourierPyramidCos[0].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
					__m256* splatCos = (__m256*)(sfpy_cos + (D - 1) * width);
					const int SIZE = 2 * width / 8;

					if (isUseFourierTable0)
					{
						const __m64* guidePtr = (__m64*)src8u.ptr<uchar>(j + D - 1);
						const float* cxiPtr = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZE; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
							*(splatCos++) = _mm256_i32gather_ps(cxiPtr, idx, sizeof(float));
						}
					}
					else
					{
						for (int i = 0; i < SIZE; ++i)
						{
							//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
						}
					}
				}
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
					}
					spcosline_e[i] = sumc;
				}
				sfpy_cos += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;

					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
					}
					sumc = _mm256_shuffle_ps(sumc, sumc, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
				}
				dfpyn_cos += widths;
			}
#pragma endregion

#pragma region Laplacian0
			sfpy_cos = FourierPyramidCos[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumco = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(mevenratio, sumce));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(moddratio, sumco));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumco = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumco += GaussWeight[m - 1] * *sc;
						sc += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
				}

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
					__m256 msin = _mm256_sin_ps(ms);
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dste + I)));
					}

					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
					msin = _mm256_sin_ps(ms);
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
					msin = _mm256_sin_ps(ms);
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dsto + I)));
					}
					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
					msin = _mm256_sin_ps(ms);
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * sin(os) * sumcee;
					}
					else
					{
						dste[I] += alphak * evenratio * sin(os) * sumcee;
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * sin(os) * sumcoe;
					}
					else
					{
						dste[I + 1] += alphak * oddratio * sin(os) * sumcoe;
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * sin(os) * sumceo;
					}
					else
					{
						dsto[I] += alphak * evenratio * sin(os) * sumceo;
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * sin(os) * sumcoo;
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * sin(os) * sumcoo;
					}
				}
				sfpy_cos += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

#pragma region GaussianLevel
			float* sfpy_cos = FourierPyramidCos[l].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[l + 1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
					}
					spcosline_e[i] = sumc;
				}
				sfpy_cos += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;

					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
					}
					sumc = _mm256_shuffle_ps(sumc, sumc, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));

				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
				}
				dfpyn_cos += widths;
			}
#pragma endregion

#pragma region LaplacianLevel
			sfpy_cos = FourierPyramidCos[l + 1].ptr<float>(0, rs);
			float* ppye_cos = FourierPyramidCos[l].ptr<float>(radius, radius);
			float* ppyo_cos = FourierPyramidCos[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumco = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(mevenratio, sumce));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(moddratio, sumco));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumco = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumco += GaussWeight[m - 1] * *sc;
						sc += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
				}

				//h filter
				if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
					}

					const int I = i << 1;
					//even line
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
					__m256 msin = _mm256_sin_ps(ms);
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dste + I)));
					}

					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
					msin = _mm256_sin_ps(ms);
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
					msin = _mm256_sin_ps(ms);
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dsto + I)));
					}

					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
					msin = _mm256_sin_ps(ms);
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]));
					}
					else
					{
						dste[I] += alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]));
					}
					else
					{
						dsto[I] += alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]));
					}
				}
				sfpy_cos += widths;
				ppye_cos += 2 * width;
				ppyo_cos += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(spcosline_o);
		_mm_free(spcosline_e);
		_mm_free(W);
	}


	void LocalMultiScaleFilterFourier::sumPyramid(const vector<vector<Mat>>& srcPyramids, vector<Mat>& destPyramid, const int numberPyramids, const int level, vector<bool>& used)
	{
		vector<vector<int>> h(level);
		for (int l = 0; l < level; l++)
		{
			h[l].resize(threadMax + 1);
			h[l][0] = 0;
			const int basestep = destPyramid[l].rows / threadMax;
			int rem = destPyramid[l].rows % threadMax;

			for (int t = 0; t < threadMax; t++)
			{
				h[l][t + 1] = h[l][t] + basestep + ((rem > 0) ? 1 : 0);
				rem--;
				//print_debug4(l, t, h[l][t + 1], destPyramid[l].rows);
			}
		}

#pragma omp parallel for schedule (dynamic)
		for (int t = 0; t < threadMax; t++)
		{
			for (int l = 0; l < level; l++)
			{
				const int hs = h[l][t];
				const int he = h[l][t + 1];
				const int w = destPyramid[l].cols;
				const int size = w * (he - hs);
				const int SIZE32 = get_simd_floor(size, 32);
				const int SIZE8 = get_simd_floor(size, 8);
				float* d = destPyramid[l].ptr<float>(hs);
				for (int k = numberPyramids - 1; k >= 0; --k)
				{
					if (used[k])
					{
						const float* s = srcPyramids[k][l].ptr<float>(hs);
						for (int i = 0; i < SIZE32; i += 32)
						{
							_mm256_storeu_ps(d + i, _mm256_add_ps(_mm256_loadu_ps(d + i), _mm256_loadu_ps(s + i)));
							_mm256_storeu_ps(d + i + 8, _mm256_add_ps(_mm256_loadu_ps(d + i + 8), _mm256_loadu_ps(s + i + 8)));
							_mm256_storeu_ps(d + i + 16, _mm256_add_ps(_mm256_loadu_ps(d + i + 16), _mm256_loadu_ps(s + i + 16)));
							_mm256_storeu_ps(d + i + 24, _mm256_add_ps(_mm256_loadu_ps(d + i + 24), _mm256_loadu_ps(s + i + 24)));
						}
						for (int i = SIZE32; i < SIZE8; i += 8)
						{
							_mm256_storeu_ps(d + i, _mm256_add_ps(_mm256_loadu_ps(d + i), _mm256_loadu_ps(s + i)));
						}
						for (int i = SIZE8; i < size; i++)
						{
							d[i] += s[i];
						}
					}
				}
			}
		}
	}


	void LocalMultiScaleFilterFourier::pyramidParallel(const Mat& src, Mat& dest)
	{
		layerSize.resize(level + 1);
		allocImageBuffer(order, level);

		const int gfRadius = getGaussianRadius(sigma_space);
		const int lowr = 2 * gfRadius + gfRadius;
		const int r_pad0 = lowr * (int)pow(2, level - 1);
		if (pyramidComputeMethod == IgnoreBoundary)
		{
			if (src.depth() == CV_8U)
			{
				copyMakeBorder(src, src8u, r_pad0, r_pad0, r_pad0, r_pad0, borderType);
				src8u.convertTo(ImageStack[0], CV_32F);
			}
			else
			{
				cv::copyMakeBorder(src, ImageStack[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
				if (isUseFourierTable0) ImageStack[0].convertTo(src8u, CV_8U);
			}

			if (adaptiveMethod)
			{
				adaptiveBoostBorder.resize(level);
				adaptiveSigmaBorder.resize(level);

				bool isEachBorder = false;
				if (isEachBorder)
				{
					for (int l = 0; l < level; l++)
					{
						int rr = (r_pad0 >> l);
						cv::copyMakeBorder(adaptiveBoostMap[l], adaptiveBoostBorder[l], rr, rr, rr, rr, borderType);
						cv::copyMakeBorder(adaptiveSigmaMap[l], adaptiveSigmaBorder[l], rr, rr, rr, rr, borderType);
					}
				}
				else
				{
					cv::copyMakeBorder(adaptiveBoostMap[0], adaptiveBoostBorder[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
					cv::copyMakeBorder(adaptiveSigmaMap[0], adaptiveSigmaBorder[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
					//print_debug2(srcf.size(), adaptiveBoostMap[0].size());
					//print_debug2(border.size(), adaptiveBoostBorder[0].size());
					for (int l = 0; l < level - 1; l++)
					{
						resize(adaptiveBoostBorder[l], adaptiveBoostBorder[l + 1], Size(), 0.5, 0.5, INTER_NEAREST);
						resize(adaptiveSigmaBorder[l], adaptiveSigmaBorder[l + 1], Size(), 0.5, 0.5, INTER_NEAREST);
					}
				}
			}
		}
		else
		{
			if (src.depth() == CV_8U)
			{
				src.convertTo(ImageStack[0], CV_32F);
				src8u = src;
			}
			else
			{
				src.copyTo(ImageStack[0]);
				if (isUseFourierTable0) src.convertTo(src8u, CV_8U);
			}
		}

		for (int l = 0; l < level + 1; l++)
		{
			layerSize[l] = ImageStack[l].size();
		}

		//compute alpha, T, and table: 0.065ms
		{
			//cp::Timer t("initRangeFourier");
			initRangeFourier(order, sigma_range, boost);
		}

		//Build Gaussian Pyramid for Input Image
		buildGaussianPyramid(ImageStack[0], ImageStack, level, sigma_space);

		//Build Outoput Laplacian Pyramid
		if (computeScheduleFourier == MergeFourier) //merge cos and sin
		{
			vector<bool> init(threadMax);
			for (int t = 0; t < threadMax; t++)init[t] = true;

#pragma omp parallel for schedule (dynamic)
			for (int k = 0; k < order + 1; k++)
			{
				const int tidx = omp_get_thread_num();
				if (k == order)
				{
					//DC pyramid
					buildLaplacianPyramid(ImageStack, LaplacianPyramid, level, sigma_space);
				}
				else
				{
					if (init[tidx])
					{
#pragma omp critical
						init[tidx] = false;
						if (isUseFourierTable0)
						{
							if (isUseFourierTableLevel)
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
							}
						}
						else
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<true, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<true, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<true, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
							}
						}
					}
					else
					{
						if (isUseFourierTable0)
						{
							if (isUseFourierTableLevel)
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								}
							}
						}
						else
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidCos[tidx], FourierPyramidSin[tidx]);
							}
						}
					}
				}
			}

			for (int i = 0; i < threadMax; i++)init[i] = init[i] ? false : true;
			sumPyramid(destEachOrder, LaplacianPyramid, threadMax, level, init);
		}
		else if (computeScheduleFourier == SplitFourier) //split cos and sin
		{
			vector<bool> init(threadMax);
			for (int t = 0; t < threadMax; t++)init[t] = true;

			const int NC = 2 * order;
#pragma omp parallel for schedule (dynamic)
			for (int nc = 0; nc < NC + 1; nc++)
			{
				const int tidx = omp_get_thread_num();
				if (nc == NC)
				{
					//DC pyramid
					buildLaplacianPyramid(ImageStack, LaplacianPyramid, level, sigma_space);
				}
				else
				{
					const int k = nc / 2;

					if (init[tidx])
					{
#pragma omp critical
						init[tidx] = false;
						if (nc % 2 == 0)
						{
							if (isUseFourierTable0)
							{
								if (isUseFourierTableLevel)
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
								}
								else
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<true, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<true, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<true, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
							}
						}
						else
						{
							if (isUseFourierTable0)
							{
								if (isUseFourierTableLevel)
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
								}
								else
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<true, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<true, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<true, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
							}
						}
					}
					else
					{
						if (nc % 2 == 0)
						{
							if (isUseFourierTable0)
							{
								if (isUseFourierTableLevel)
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
								}
								else
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
							}
						}
						else
						{
							if (isUseFourierTable0)
							{
								if (isUseFourierTableLevel)
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
								}
								else
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									}
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierPyramidSin[tidx]);
								}
							}
						}
					}
				}
			}

			for (int i = 0; i < threadMax; i++)init[i] = init[i] ? false : true;
			//for (int i = 0; i < threadMax; i++) { if (!init[i])cout << i << "not used" << endl; }
			sumPyramid(destEachOrder, LaplacianPyramid, threadMax, level, init);
			//sumPyramid(destEachOrder, LaplacianPyramid, 1, level);//for non parallel
		}
		LaplacianPyramid[level] = ImageStack[level];

		collapseLaplacianPyramid(LaplacianPyramid, LaplacianPyramid[0]);
		LaplacianPyramid[0](Rect(r_pad0, r_pad0, src.cols, src.rows)).copyTo(dest);

		if (isPlot)
		{
			kernelPlot(GAUSS, order, 255, boost, sigma_range, Salpha, Sbeta, sigma_range, 0, 255, 255, T, sinTable, cosTable, alpha, beta, windowType, "GFP f(x)");
			isPlotted = true;
		}
		else
		{
			if (isPlotted)
			{
				cv::destroyWindow("GFP f(x)");
				isPlotted = false;
			}
		}
	}

	void LocalMultiScaleFilterFourier::pyramidSerial(const Mat& src, Mat& dest)
	{
		layerSize.resize(level + 1);
		allocImageBuffer(order, level);

		const int lowr = 3 * radius;
		const int r_pad0 = lowr * (int)pow(2, level - 1);

		if (src.depth() == CV_8U)
		{
			src.convertTo(ImageStack[0], CV_32F);
			src8u = src;
		}
		else
		{
			ImageStack[0] = src;
#ifdef USE_GATHER8U
			if (isUseFourierTable0) src.convertTo(src8u, CV_8U);
#endif
		}

		if (adaptiveMethod)
		{
			adaptiveBoostBorder.resize(level);
			adaptiveSigmaBorder.resize(level);
			for (int l = 0; l < level; l++)
			{
				adaptiveBoostBorder[l] = adaptiveBoostMap[l];
				adaptiveSigmaBorder[l] = adaptiveSigmaMap[l];
			}
		}

		for (int l = 0; l < level + 1; l++)
		{
			layerSize[l] = ImageStack[l].size();
		}

		{
			//cp::Timer t("initRangeFourier");
			//compute alpha, T, and table: 0.065ms
			initRangeFourier(order, sigma_range, boost);
		}

		//Build Gaussian Pyramid for Input Image
		buildGaussianLaplacianPyramid(ImageStack[0], ImageStack, LaplacianPyramid, level, sigma_space);

		//Build Outoput Laplacian Pyramid
		if (isUseFourierTable0)
		{
			if (isUseFourierTableLevel)
			{
				if (computeScheduleFourier == MergeFourier) //merge cos and sin use table
				{
					for (int k = 0; k < order; k++)
					{
						if (radius == 2)
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
						}
						else if (radius == 4)
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
						}
						else
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
						}
					}
				}
				else if (computeScheduleFourier == SplitFourier) //split cos and sin use table
				{
					const int NC = 2 * order;
					for (int nc = 0; nc < NC; nc++)
					{
						const int k = nc / 2;

						if (nc % 2 == 0)
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
						}
						else
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
						}
					}
				}
			}
			else
			{
				if (computeScheduleFourier == MergeFourier) //merge cos and sin use table
				{
					for (int k = 0; k < order; k++)
					{
						if (radius == 2)
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
						}
						else if (radius == 4)
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
						}
						else
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
						}
					}
				}
				else if (computeScheduleFourier == SplitFourier) //split cos and sin use table
				{
					const int NC = 2 * order;
					for (int nc = 0; nc < NC; nc++)
					{
						const int k = nc / 2;

						if (nc % 2 == 0)
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
						}
						else
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							}
						}
					}
				}
			}
		}
		else
		{
			if (computeScheduleFourier == MergeFourier) //merge cos and sin
			{
				for (int k = 0; k < order; k++)
				{
					if (radius == 2)
					{
						if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
						else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
					}
					else if (radius == 4)
					{
						if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
						else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
					}
					else
					{
						if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
						else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidCos[0], FourierPyramidSin[0]);
					}
				}
			}
			else if (computeScheduleFourier == SplitFourier) //split cos and sin
			{
				const int NC = 2 * order;
				for (int nc = 0; nc < NC; nc++)
				{
					const int k = nc / 2;

					if (nc % 2 == 0)
					{
						if (radius == 2)
						{
							if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
						}
						else if (radius == 4)
						{
							if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
						}
						else
						{
							if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
						}
					}
					else
					{
						if (radius == 2)
						{
							if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
						}
						else if (radius == 4)
						{
							if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
						}
						else
						{
							if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
							else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, LaplacianPyramid, k, level, FourierPyramidSin[0]);
						}
					}
				}
			}
		}

		LaplacianPyramid[level] = ImageStack[level];
		collapseLaplacianPyramid(LaplacianPyramid, dest);
	}

	void LocalMultiScaleFilterFourier::pyramid(const Mat& src, Mat& dest)
	{
		rangeDescope(src);

		if (isParallel)	pyramidParallel(src, dest);
		else pyramidSerial(src, dest);
	}

	LocalMultiScaleFilterFourier::~LocalMultiScaleFilterFourier()
	{
		_mm_free(sinTable);
		_mm_free(cosTable);
	}

#pragma region DoG
	void LocalMultiScaleFilterFourier::make_sin_cos(cv::Mat src, cv::Mat& dest_sin, cv::Mat& dest_cos, int k)
	{
		dest_sin.create(src.size(), CV_32F);
		dest_cos.create(src.size(), CV_32F);
		float* s = src.ptr<float>();
		float* ds = dest_sin.ptr<float>();
		float* dc = dest_cos.ptr<float>();
		const float omega_k = float(CV_2PI * (k + 1) / 463.9);
		for (int i = 0; i < dest_sin.size().area(); i++)
		{
			*ds = sin((*s) * omega_k);
			*dc = cos((*s) * omega_k);
			ds++;
			dc++;
			s++;
		}
	}

	void LocalMultiScaleFilterFourier::splattingBlurring(const cv::Mat& src, float sigma_space, int l, int level, int k, std::vector<std::vector<cv::Mat>>& splatBuffer, bool islast)
	{
		cv::Mat guide;

		src.convertTo(guide, CV_8U);

		int simdsize = sizeof(__m256) / sizeof(float);
		splatBuffer[0][k].create(src.size(), CV_32F);
		splatBuffer[1][k].create(src.size(), CV_32F);

		const __m64* guidePtr = (__m64*)guide.ptr<uchar>();
		uchar* test = guide.ptr<uchar>();
		__m256* IntermPtr_l = (__m256*)splatBuffer[0][k].ptr<float>();
		__m256* IntermPtr_r = (__m256*)splatBuffer[1][k].ptr<float>();
		__m256 wsum = _mm256_set1_ps(w_sum);
		const float* sxiPtr = &sinTable[256 * k];
		const float* cxiPtr = &cosTable[256 * k];
		__m256 mxi_sin, mxi_cos;

		__m256i temp = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr);

		const uchar* guidePtr_ = guide.ptr<uchar>();
		float* IntermPtr_l_ = splatBuffer[0][k].ptr<float>();
		float* IntermPtr_r_ = splatBuffer[1][k].ptr<float>();

		//normal implementation
		/*for (int i = 0; i < guide.size().area(); i++)
		{
			*IntermPtr_l_ = cxiPtr[*guidePtr_];
			*IntermPtr_r_ = sxiPtr[*guidePtr_];
			IntermPtr_l_++;
			IntermPtr_r_++;
			guidePtr_++;
		}*/

		if (l == 0)
		{
			//simd implementation
			for (int i = 0; i < guide.size().area() / simdsize; i++)
			{
				mxi_sin = _mm256_i32gather_ps(sxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtr), sizeof(float));
				mxi_cos = _mm256_i32gather_ps(cxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)(guidePtr)), sizeof(float));
				if (islast)
				{
					*(IntermPtr_l) = mxi_cos;
					*(IntermPtr_r) = mxi_sin;
					//*(IntermPtr_l) = mxi_sin;
					//*(IntermPtr_r) = mxi_cos;
				}
				else
				{
					*(IntermPtr_l) = mxi_cos;
					*(IntermPtr_r) = mxi_sin;
					//*(IntermPtr_l) = mxi_sin;
					//*(IntermPtr_r) = mxi_cos;
				}
				IntermPtr_l++;
				IntermPtr_r++;
				guidePtr++;
			}

			buildGaussianStack(splatBuffer[0][k], cos_pyramid[k], sigma_space, level);
			buildGaussianStack(splatBuffer[1][k], sin_pyramid[k], sigma_space, level);
			splatBuffer[0][k] = cos_pyramid[k][l + 1].clone();
			splatBuffer[1][k] = sin_pyramid[k][l + 1].clone();
			/*Mat tmp(src.size(), CV_32F);
			GaussianBlur(Interm[0], tmp, cv::Size(2 * r + 1, 2 * r + 1), (level)*sigma_space);
			Interm[0] = tmp.clone();
			GaussianBlur(Interm[1], tmp, cv::Size(2 * r + 1, 2 * r + 1), (level)*sigma_space);
			Interm[1] = tmp.clone();*/
		}
		else
		{
			splatBuffer[0][k] = cos_pyramid[k][l].clone();
			splatBuffer[1][k] = sin_pyramid[k][l].clone();
			splatBuffer[2][k] = cos_pyramid[k][l + 1].clone();
			splatBuffer[3][k] = sin_pyramid[k][l + 1].clone();
		}
	}

	void LocalMultiScaleFilterFourier::productSummingTrig_last(cv::Mat& src, cv::Mat& dest, float sigma_range, int k, std::vector<cv::Mat>& splatBuffer, int level)
	{
		cv::Mat guide;
		src.convertTo(guide, CV_8U);
		int simdsize = sizeof(__m256) / sizeof(float);
		int r = (int)ceil(3.f * 1.f);
		const int width = guide.size().width;
		const int height = guide.size().height;
		const int simdWidth = width / simdsize;
		//float omega;
		const __m64* guidePtr = (__m64*)guide.ptr<uchar>();
		__m256* IntermPtr_l = (__m256*)splatBuffer[0].ptr<float>();
		__m256* IntermPtr_r = (__m256*)splatBuffer[1].ptr<float>();
		__m256* destPtr = (__m256*)dest.ptr<float>();
		__m256 mxi_sin, mxi_cos;
		const float* sxiPtr = &sinTable[256 * k];
		const float* cxiPtr = &cosTable[256 * k];
		__m256 malpha = _mm256_set1_ps(alpha[k]);
		for (int i = 0; i < src.size().area() / simdsize; i++)
		{
			mxi_sin = _mm256_i32gather_ps(sxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtr), sizeof(float));
			mxi_cos = _mm256_i32gather_ps(cxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtr), sizeof(float));
			*(IntermPtr_l) = _mm256_mul_ps(mxi_sin, _mm256_mul_ps(malpha, *(IntermPtr_l)));
			*(IntermPtr_r) = _mm256_mul_ps(mxi_cos, _mm256_mul_ps(malpha, *(IntermPtr_r)));
			*(destPtr) = _mm256_add_ps(*(destPtr), _mm256_sub_ps(*(IntermPtr_l), *(IntermPtr_r)));
			guidePtr++;
			destPtr++;
			IntermPtr_l++;
			IntermPtr_r++;
		}
	}

	void LocalMultiScaleFilterFourier::productSummingTrig(cv::Mat& srcn, cv::Mat& dest, cv::Mat& src8u, float sigma_range, int k, std::vector<std::vector<cv::Mat>>& splatBuffer, int l)
	{
		cv::Mat guiden, guiden_1;
		src8u.convertTo(guiden, CV_8U);
		int simdsize = sizeof(__m256) / sizeof(float);
		int r = (int)ceil(3.f * 1.f);

		//float omega;
		const __m64* guidePtr = (__m64*)guiden.ptr<uchar>();//sigma_n
		//const __m64* guidePtrn_1 = (__m64*)guiden_1.ptr<uchar>();//sigma_n-1
		__m256* IntermPtr_l = (__m256*)splatBuffer[0][k].ptr<float>();
		__m256* IntermPtr_r = (__m256*)splatBuffer[1][k].ptr<float>();
		__m256* IntermPtr_l2 = (__m256*)splatBuffer[2][k].ptr<float>();
		__m256* IntermPtr_r2 = (__m256*)splatBuffer[3][k].ptr<float>();
		__m256* destPtr = (__m256*)dest.ptr<float>();
		//__m256 mxi_sin_l, mxi_sin_r, mxi_cos_l, mxi_cos_r;
		__m256 	mxi_sin, mxi_cos;
		const float* sxiPtr = &sinTable[256 * k];
		const float* cxiPtr = &cosTable[256 * k];
		__m256 malpha = _mm256_set1_ps(alpha[k]);
		if (l == 0)
		{
			for (int i = 0; i < srcn.size().area() / simdsize; i++)
			{
				//mxi_sin_l = _mm256_i32gather_ps(sxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtrn_1), sizeof(float));
				//mxi_sin_r = _mm256_i32gather_ps(sxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtrn), sizeof(float));
				//mxi_cos_l = _mm256_i32gather_ps(cxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtrn_1), sizeof(float));
				//mxi_cos_r = _mm256_i32gather_ps(cxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtrn), sizeof(float));

				//*(IntermPtr_l) = _mm256_mul_ps(_mm256_sub_ps(mxi_sin_l, mxi_sin_r), _mm256_mul_ps(malpha, *(IntermPtr_l)));
				//*(IntermPtr_r) = _mm256_mul_ps(_mm256_sub_ps(mxi_cos_l, mxi_cos_r), _mm256_mul_ps(malpha, *(IntermPtr_r)));
				//*(destPtr) = _mm256_add_ps(*(destPtr), _mm256_sub_ps(*(IntermPtr_l), *(IntermPtr_r)));

				mxi_sin = _mm256_i32gather_ps(sxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtr), sizeof(float));
				mxi_cos = _mm256_i32gather_ps(cxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtr), sizeof(float));
				*(IntermPtr_l) = _mm256_mul_ps(mxi_sin, *(IntermPtr_l));
				*(IntermPtr_r) = _mm256_mul_ps(mxi_cos, *(IntermPtr_r));
				*(destPtr) = _mm256_fmadd_ps(malpha, _mm256_sub_ps(*(IntermPtr_l), *(IntermPtr_r)), *(destPtr));



				//guidePtrn++;
				//guidePtrn_1++;
				guidePtr++;
				destPtr++;
				IntermPtr_l++;
				IntermPtr_r++;
			}
		}
		else
		{
			cv::Mat guide;
			srcn.convertTo(guide, CV_8U);
			const __m64* guidePtr = (__m64*)guide.ptr<uchar>();
			cv::Mat dest_sin, dest_cos;
			make_sin_cos(srcn, dest_sin, dest_cos, k);
			__m256* sin_ptr = (__m256*)dest_sin.ptr<float>();
			__m256* cos_ptr = (__m256*)dest_cos.ptr<float>();
			for (int i = 0; i < srcn.size().area() / simdsize; i++)
			{
				/*	mxi_sin = _mm256_i32gather_ps(sxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtr), sizeof(float));
					mxi_cos = _mm256_i32gather_ps(cxiPtr, _mm256_cvtepu8_epi32(*(__m128i*)guidePtr), sizeof(float));
					*(IntermPtr_l) = _mm256_mul_ps(mxi_sin, *(IntermPtr_l));
					*(IntermPtr_l2) = _mm256_mul_ps(mxi_sin, *(IntermPtr_l2));
					*(IntermPtr_r) = _mm256_mul_ps(mxi_cos, *(IntermPtr_r));
					*(IntermPtr_r2) = _mm256_mul_ps(mxi_cos, *(IntermPtr_r2));*/

				*(IntermPtr_l) = _mm256_mul_ps(*sin_ptr, *(IntermPtr_l));
				*(IntermPtr_l2) = _mm256_mul_ps(*(sin_ptr++), *(IntermPtr_l2));
				*(IntermPtr_r) = _mm256_mul_ps(*cos_ptr, *(IntermPtr_r));
				*(IntermPtr_r2) = _mm256_mul_ps(*(cos_ptr++), *(IntermPtr_r2));

				*(destPtr) = _mm256_fmadd_ps(malpha, _mm256_sub_ps(_mm256_sub_ps(*(IntermPtr_l2), *(IntermPtr_r2)), _mm256_sub_ps(*(IntermPtr_l), *(IntermPtr_r))), *(destPtr));
				guidePtr++;
				destPtr++;
				IntermPtr_l++;
				IntermPtr_r++;
				IntermPtr_l2++;
				IntermPtr_r2++;
			}
		}
	}

	void LocalMultiScaleFilterFourier::dog(const Mat& src, Mat& dest)
	{
		initRangeTable(sigma_range, boost);
		int numIntermImages = 2 * order;
		sinTable = (float*)_mm_malloc(sizeof(float) * 256 * order, 32);
		cosTable = (float*)_mm_malloc(sizeof(float) * 256 * order, 32);
		alpha.resize(order + 1);
		cv::Mat srcf;

		{
			int r = 2;
			const double coeff_s = -1.0 / (2.0 * sigma_space * sigma_space);
			w_sum = 0.f;
			for (int j = -r; j <= r; j++)
			{
				for (int i = -r; i <= r; i++)
				{
					double dis = double(i * i + j * j);
					float v = (float)exp(dis * coeff_s);
					w_sum += v;
				}
			}
		}

		if (src.depth() == CV_8U)
		{
			src.convertTo(srcf, CV_32F);
			src.convertTo(dest, CV_32F);
		}
		else
		{
			srcf = src.clone();
			dest = src.clone();
		}

		initRangeFourier(order, sigma_range, boost);

		std::vector<cv::Mat> GaussianStack;
		buildGaussianStack(srcf, GaussianStack, sigma_space, level);
		//buildGaussianStack_(srcf, GaussianStack, w_sum, sigma_space, level);

		//generate radual signal
		const int r = (int)ceil(sigma_space * level * 3);
		cv::Mat rasidual;
		cv::GaussianBlur(srcf, rasidual, cv::Size(2 * r + 1, 2 * r + 1), sigma_space * level);

		std::vector<std::vector<cv::Mat>> splatBuffer(4);
		std::vector<cv::Mat> DoG_Lap(level + 1);
		splatBuffer[0].resize(order);
		splatBuffer[1].resize(order);
		splatBuffer[2].resize(order);
		splatBuffer[3].resize(order);
		std::vector<cv::Mat> Interm_temp(2);
		Interm_temp[0] = cv::Mat::zeros(GaussianStack[0].size(), CV_32F);
		Interm_temp[1] = cv::Mat::zeros(GaussianStack[0].size(), CV_32F);
		sin_pyramid.resize(order);
		cos_pyramid.resize(order);

		for (int l = 0; l < level; l++)
		{
			cv::subtract(GaussianStack[l], GaussianStack[l + 1], DoG_Lap[l]);
		}

		destEachOrder.resize(order);
		cv::Mat temp = cv::Mat::zeros(src.size(), CV_32F);
		for (int k = 0; k < order; k++)
		{
			//for (int l = 1; l <= level; l++)
			for (int l = 0; l < level; ++l)
			{
#if 0
				if (l != level)
				{
					//cout << "init" << endl;
					splattingBlurring(srcf, sigma_space, l, level, k, Interm, false);
					productSummingTrig(GaussianStack[l], GaussianStack[l - 1], temp, sigma_range, k, Interm_temp, l);
				}
				if (l == level)
				{
					splattingBlurring(srcf, sigma_space, l, level, k, Interm, true);
					productSummingTrig_last(srcf, temp, sigma_range, k, Interm_temp, l);
				}
#else if
				destEachOrder[0][l] = cv::Mat::zeros(GaussianStack[l].size(), CV_32F);

				splattingBlurring(srcf, sigma_space, l, level, k, splatBuffer, false);
				productSummingTrig(GaussianStack[l], destEachOrder[0][l], srcf, sigma_range, k, splatBuffer, l);
				cv::add(DoG_Lap[l], destEachOrder[k], DoG_Lap[l]);
#endif
			}
		}
		DoG_Lap[level] = GaussianStack[level].clone();
		collapseDoGStack(DoG_Lap, dest);
		//dest = src.clone();
		//simd_add_src(temp, dest);
	}
#pragma endregion

	void LocalMultiScaleFilterFourier::filter(const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);

		this->order = order;

		this->sigma_space = sigma_space;
		this->level = max(level, 1);
		this->sigma_range = sigma_range;
		this->boost = boost;

		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);
		freeSpaceWeight();
	}

	void LocalMultiScaleFilterFourier::setPeriodMethod(Period scaleSpaceMethod)
	{
		periodMethod = scaleSpaceMethod;
	}

	void LocalMultiScaleFilterFourier::setComputeScheduleMethod(int schedule, bool useTable0, bool useTableLevel)
	{
		computeScheduleFourier = schedule;
		isUseFourierTable0 = useTable0;
		isUseFourierTableLevel = useTableLevel;
	}

	string LocalMultiScaleFilterFourier::getComputeScheduleName()
	{
		string ret = "";
		if (computeScheduleFourier == MergeFourier) ret += "MergeFourier_";
		if (computeScheduleFourier == SplitFourier) ret += "SplitFourier_";

		if (isUseFourierTable0) ret += "Gather0_";
		else  ret += "Compute0_";
		if (isUseFourierTableLevel) ret += "GatherL";
		else  ret += "ComputeL";

		return ret;
	}

	void LocalMultiScaleFilterFourier::setIsParallel(const bool flag)
	{
		this->isParallel = flag;
	}

	void  LocalMultiScaleFilterFourier::setIsPlot(const bool flag)
	{
		this->isPlot = flag;
	}

	std::string LocalMultiScaleFilterFourier::getPeriodName()
	{
		string ret = "";
		if (periodMethod == Period::GAUSS_DIFF)ret = "GAUSS_DIFF";
		if (periodMethod == Period::OPTIMIZE)ret = "OPTIMIZE";
		if (periodMethod == Period::PRE_SET)ret = "PRE_SET";

		return ret;
	}
#pragma endregion



#pragma region TileLocalMultiScaleFilterInterpolation
	TileLocalMultiScaleFilterInterpolation::TileLocalMultiScaleFilterInterpolation()
	{
		msf = new LocalMultiScaleFilterInterpolation[threadMax];
		for (int i = 0; i < threadMax; i++)
			msf[i].setIsParallel(false);
	}

	TileLocalMultiScaleFilterInterpolation::~TileLocalMultiScaleFilterInterpolation()
	{
		delete[] msf;
	}

	void TileLocalMultiScaleFilterInterpolation::setComputeScheduleMethod(bool useTable)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setComputeScheduleMethod(useTable);
	}

	string TileLocalMultiScaleFilterInterpolation::getComputeScheduleName()
	{
		return msf[0].getComputeScheduleName();
	}

	void TileLocalMultiScaleFilterInterpolation::setAdaptive(const bool flag, const cv::Size div, cv::Mat& adaptiveSigmaMap, cv::Mat& adaptiveBoostMap)
	{
		if (flag)
		{
			vector<Mat> g{ adaptiveSigmaMap, adaptiveBoostMap };
			initGuide(div, g);
		}
		else
		{
			unsetUseGuide();
		}
	}

	void TileLocalMultiScaleFilterInterpolation::setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod scaleSpaceMethod)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setRangeDescopeMethod(scaleSpaceMethod);
	}

	void TileLocalMultiScaleFilterInterpolation::setCubicAlpha(const float alpha)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setCubicAlpha(alpha);
	}

	void TileLocalMultiScaleFilterInterpolation::process(const cv::Mat& src, cv::Mat& dst, const int threadIndex, const int imageIndex)
	{
		if (isUseGuide)
		{
			msf[threadIndex].setAdaptive(true, guideTile[0][imageIndex], guideTile[1][imageIndex], level);
		}
		else
		{
			Mat a;
			msf[threadIndex].setAdaptive(false, a, a, 0);
		}

		msf[threadIndex].filter(src, dst, order, sigma_range, sigma_space, boost, level, scaleSpaceMethod, interpolation);
	}

	void TileLocalMultiScaleFilterInterpolation::filter(const Size div, const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const MultiScaleFilter::ScaleSpace scaleSpaceMethod, int interpolation)
	{
		this->order = order;
		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->boost = boost;
		this->level = level;

		this->scaleSpaceMethod = scaleSpaceMethod;
		this->interpolation = interpolation;

		const int lowr = 3 * msf[0].getGaussianRadius(sigma_space);
		const int r_pad0 = lowr * (int)pow(2, level - 1);
		invoker(div, src, dest, r_pad0);
	}
#pragma endregion

#pragma region TileLocalMultiScaleFilterFourier
	TileLocalMultiScaleFilterFourier::TileLocalMultiScaleFilterFourier()
	{
		msf = new LocalMultiScaleFilterFourier[threadMax];
		for (int i = 0; i < threadMax; i++)
			msf[i].setIsParallel(false);
	}

	TileLocalMultiScaleFilterFourier::~TileLocalMultiScaleFilterFourier()
	{
		delete[] msf;
	}


	void TileLocalMultiScaleFilterFourier::setComputeScheduleMethod(int schedule, bool useTable0, bool useTableLevel)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setComputeScheduleMethod(schedule, useTable0, useTableLevel);
	}

	string TileLocalMultiScaleFilterFourier::getComputeScheduleName()
	{
		return msf[0].getComputeScheduleName();
	}

	void TileLocalMultiScaleFilterFourier::setAdaptive(const bool flag, const Size div, cv::Mat& adaptiveSigmaMap, cv::Mat& adaptiveBoostMap)
	{
		if (flag)
		{
			vector<Mat> g{ adaptiveSigmaMap, adaptiveBoostMap };
			initGuide(div, g);
		}
		else
		{
			unsetUseGuide();
		}
	}

	void TileLocalMultiScaleFilterFourier::setPeriodMethod(const int method)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setPeriodMethod((LocalMultiScaleFilterFourier::Period)method);

	}

	void TileLocalMultiScaleFilterFourier::setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod scaleSpaceMethod)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setRangeDescopeMethod(scaleSpaceMethod);
	}

	void TileLocalMultiScaleFilterFourier::process(const cv::Mat& src, cv::Mat& dst, const int threadIndex, const int imageIndex)
	{
		if (isUseGuide)
		{
			msf[threadIndex].setAdaptive(true, guideTile[0][imageIndex], guideTile[1][imageIndex], level);
		}
		else
		{
			Mat a;
			msf[threadIndex].setAdaptive(false, a, a, 0);
		}

		msf[threadIndex].filter(src, dst, order, sigma_range, sigma_space, boost, level, scaleSpaceMethod);
	}

	void TileLocalMultiScaleFilterFourier::filter(const Size div, const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const MultiScaleFilter::ScaleSpace scaleSpaceMethod)
	{
		this->order = order;
		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->boost = boost;
		this->level = level;

		this->scaleSpaceMethod = scaleSpaceMethod;
		const int lowr = 3 * msf[0].getGaussianRadius(sigma_space);
		const int r_pad0 = lowr * (int)pow(2, level - 1);
		invoker(div, src, dest, r_pad0);
	}
#pragma endregion

#pragma endregion
}