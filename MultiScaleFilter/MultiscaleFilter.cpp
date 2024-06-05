#include "multiscalefilter/MultiScaleFilter.hpp"


//#define AMD_OPTIMIZATION
#define USE_GATHER8U
#define MASKSTORE
//#define MASKSTORELASTLINEAR

#ifdef AMD_OPTIMIZATION
inline __m256 _mm256_i32gather_ps(float* s, __m256i index, int val)
{
	return _mm256_setr_ps(s[((int*)&index)[0]], s[((int*)&index)[1]], s[((int*)&index)[2]], s[((int*)&index)[3]], s[((int*)&index)[4]], s[((int*)&index)[5]], s[((int*)&index)[6]], s[((int*)&index)[7]]);
}
#endif 


//#define USE_SLIDING_DCT
#ifdef USE_SLIDING_DCT
#pragma comment(lib, "ConstantTimeBF.lib")
//#include <E:/Github/Sliding-DCT-Gaussian-Filtering/ConstantTimeBF/include/SpatialFilter.hpp>
#include <C:/Users/ckv14073/Desktop/lab\Project_folder/Sliding-DCT-Gaussian-Filtering/ConstantTimeBF/include/SpatialFilter.hpp>
#endif

using namespace cv;
using namespace std;

namespace cp
{
#pragma region MultiScaleFilter
	inline float MultiScaleFilter::getGaussianRangeWeight(const float v, const float sigma_range, const float boost)
	{
		//int n = 2;const float ret = (float)detail_param * exp(pow(abs(v), n) / (-n * pow(sigma_range, n)));
		return float(boost * exp(v * v / (-2.0 * sigma_range * sigma_range)));
	}
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
			if (sigma_range > 0.f)
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
				const float* s = src.ptr<float>();
				float* d = dest.ptr<float>();
				const int size = src.size().area();
				const int SIZE32 = get_simd_floor(size, 32);
				const int SIZE8 = get_simd_floor(size, 8);
				const __m256 mdetail = _mm256_set1_ps(boost);

				for (int i = 0; i < SIZE32; i += 32)
				{
					_mm256_storeu_ps(d + i + 0, _mm256_mul_ps(mdetail, _mm256_loadu_ps(s + i + 0)));
					_mm256_storeu_ps(d + i + 8, _mm256_mul_ps(mdetail, _mm256_loadu_ps(s + i + 8)));
					_mm256_storeu_ps(d + i + 16, _mm256_mul_ps(mdetail, _mm256_loadu_ps(s + i + 16)));
					_mm256_storeu_ps(d + i + 24, _mm256_mul_ps(mdetail, _mm256_loadu_ps(s + i + 24)));
				}
				for (int i = SIZE32; i < SIZE8; i += 8)
				{
					_mm256_storeu_ps(d + i, _mm256_mul_ps(mdetail, _mm256_loadu_ps(s + i)));
				}
				for (int i = SIZE8; i < size; i++)
				{
					d[i] = s[i] * boost;
				}
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
			//cout << "rangeDescopeMethod == RangeDescopeMethod::MINMAX" << endl;
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

	//D is filtering diameter
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

		const int step = src.cols;//src.cols
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
				//k=0
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
					sume = _mm256_fmadd_ps(GW[k + 0], ms, sume);
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
					sume += GaussWeight[k + 0] * *si;
					sumo += GaussWeight[k - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}

			// h filter
			float* deptr = dest.ptr<float>(j, radius);
			float* doptr = dest.ptr<float>(j + 1, radius);
			const float* daeptr = addsubsrc.ptr<float>(j + 0, radius);
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
		_mm_free(GW);
	}

#pragma endregion

#pragma region pyramid building
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
			destPyramid[0] = src.clone();
			for (int i = 0; i < level; i++)
			{
				Mat temp;
				pyrUp(destPyramid[i + 1], temp, destPyramid[i].size(), borderType);
				subtract(destPyramid[i], temp, destPyramid[i]);
			}
		}
	}

	void MultiScaleFilter::buildContrastPyramid(const Mat& src, vector<Mat>& destPyramid, const int level, const float sigma)
	{
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);

		buildPyramid(src, destPyramid, level);
		destPyramid[0] = destPyramid[0].clone();
		for (int l = 0; l < level; l++)
		{
			Mat tmp;
			pyrUp(destPyramid[l + 1], tmp);
			destPyramid[l] = destPyramid[l] / tmp - 1;
		}
	}

	//D: //diameter for downsampling, d2 for upsampling
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

	void MultiScaleFilter::collapseLaplacianPyramid(vector<Mat>& LaplacianPyramid, Mat& dest, const int depth)
	{
		const int level = (int)LaplacianPyramid.size();

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			for (int i = level - 1; i > 1; i--)
			{
				GaussUpAddIgnoreBoundary<true>(LaplacianPyramid[i], LaplacianPyramid[i - 1], LaplacianPyramid[i - 1]);
			}
			if (depth == CV_32F) GaussUpAddIgnoreBoundary<true>(LaplacianPyramid[1], LaplacianPyramid[0], dest);
			else
			{
				GaussUpAddIgnoreBoundary<true>(LaplacianPyramid[1], LaplacianPyramid[0], LaplacianPyramid[0]);
				LaplacianPyramid[0].convertTo(dest, depth);
			}
		}
		else if (pyramidComputeMethod == Fast)
		{
			for (int i = level - 1; i > 1; i--)
			{
				GaussUpAdd<true>(LaplacianPyramid[i], LaplacianPyramid[i - 1], LaplacianPyramid[i - 1]);
			}
			if (depth == CV_32F) GaussUpAdd<true>(LaplacianPyramid[1], LaplacianPyramid[0], dest);
			else
			{
				GaussUpAdd<true>(LaplacianPyramid[1], LaplacianPyramid[0], LaplacianPyramid[0]);
				LaplacianPyramid[0].convertTo(dest, depth);
			}
		}
		else if (pyramidComputeMethod == Full)
		{
			for (int i = level - 1; i > 1; i--)
			{
				GaussUpAddFull<true>(LaplacianPyramid[i], LaplacianPyramid[i - 1], LaplacianPyramid[i - 1], sigma_space, borderType);
			}
			if (depth == CV_32F) GaussUpAddFull<true>(LaplacianPyramid[1], LaplacianPyramid[0], dest, sigma_space, borderType);
			else
			{
				GaussUpAddFull<true>(LaplacianPyramid[1], LaplacianPyramid[0], LaplacianPyramid[0], sigma_space, borderType);
				LaplacianPyramid[0].convertTo(dest, depth);
			}
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
			if (depth == CV_32F) cv::add(ret, LaplacianPyramid[0], dest);
			else
			{
				cv::add(ret, LaplacianPyramid[0], LaplacianPyramid[0]);
				LaplacianPyramid[0].convertTo(dest, depth);
			}
		}
	}

	void MultiScaleFilter::collapseContrastPyramid(vector<Mat>& contrastPyramid, Mat& dest, const int depth)
	{
		const int level = (int)contrastPyramid.size() - 1;
		Mat ret;
		pyrUp(contrastPyramid[level], ret);
		for (int l = level - 1; l >= 0; l--)
		{
			ret = (contrastPyramid[l] + 1).mul(ret);
			if (l != 0) pyrUp(ret, ret);
			else
			{
				if (depth == CV_32F) ret.copyTo(dest);
				else ret.convertTo(dest, depth);
			}
		}
	}
#pragma endregion

#pragma region DoG building
	void MultiScaleFilter::buildGaussianStack(const Mat& src, vector<Mat>& GaussianStack, const float sigma_s, const int level)
	{
		if (GaussianStack.size() != level + 1) GaussianStack.resize(level + 1);

		if (src.depth() == CV_32F) src.copyTo(GaussianStack[0]);
		else src.convertTo(GaussianStack[0], CV_32F);
		//#pragma omp parallel for
		for (int l = 1; l <= level; l++)
		{
			const float sigma_l = (float)getPyramidSigma(sigma_s, l);
			const int r = (int)ceil(sigma_l * nsigma);
			const Size ksize(2 * r + 1, 2 * r + 1);
#ifdef USE_SLIDING_DCT
			gf::SpatialFilterSlidingDCT5_AVX_32F sf(gf::DCT_COEFFICIENTS::FULL_SEARCH_NOOPT);
			sf.filter(GaussianStack[0], GaussianStack[i], sigma_l, 2);
#else
			GaussianBlur(GaussianStack[0], GaussianStack[l], ksize, sigma_l, sigma_l, borderType);
#endif
		}
	}

	void MultiScaleFilter::buildDoGStack(const Mat& src, vector<Mat>& ImageStack, const float sigma_s, const int level)
	{
		buildGaussianStack(src, ImageStack, sigma_s, level);
		for (int l = 0; l < level; l++)
		{
			if (isDoGPyramidApprox)
			{
				Mat temp;
				const double ss = pow(2.0, l) * sigma_s;
				const int rr = int(ceil(ss * nsigma));
				GaussianBlur(ImageStack[l + 1], temp, Size(2 * rr + 1, 2 * rr + 1), ss);//DoG is not separable	
				subtract(ImageStack[l], temp, ImageStack[l]);
			}
			else
			{
				if (l == 0) subtract(src, ImageStack[l + 1], ImageStack[l]);
				else  subtract(ImageStack[l], ImageStack[l + 1], ImageStack[l]);
			}
		}
	}

	void MultiScaleFilter::buildCoGStack(const Mat& src, vector<Mat>& ImageStack, const float sigma_s, const int level)
	{
		buildGaussianStack(src, ImageStack, sigma_s, level);
		for (int l = 0; l < level; l++)
		{
			const int size = src.size().area();
			const int simdsize = get_simd_floor(size, 8);
			const __m256 mone = _mm256_set1_ps(1.f);
			const __m256 meps = _mm256_set1_ps(FLT_EPSILON);
			if (l == 0)
			{
				const float* s = src.ptr<float>();
				const float* d1 = ImageStack[l + 1].ptr<float>();
				float* d0 = ImageStack[l].ptr<float>();
				for (int i = 0; i < simdsize; i += 8)
				{
					_mm256_storeu_ps(d0 + i, _mm256_sub_ps(_mm256_div_ps(_mm256_loadu_ps(s + i), _mm256_add_ps(meps, _mm256_loadu_ps(d1 + i))), mone));
				}
				for (int i = simdsize; i < size; i++)
				{
					d0[i] = s[i] / (d1[i] + FLT_EPSILON) - 1.f;
				}
			}
			else
			{
				const float* d1 = ImageStack[l + 1].ptr<float>();
				float* d0 = ImageStack[l + 0].ptr<float>();
				for (int i = 0; i < simdsize; i += 8)
				{
					_mm256_storeu_ps(d0 + i, _mm256_sub_ps(_mm256_div_ps(_mm256_loadu_ps(d0 + i), _mm256_add_ps(meps, _mm256_loadu_ps(d1 + i))), mone));
				}
				for (int i = simdsize; i < size; i++)
				{
					d0[i] = d0[i] / (d1[i] + FLT_EPSILON) - 1.f;
				}
			}
		}
	}

	void MultiScaleFilter::buildDoGSeparableStack(const Mat& src, vector<Mat>& ImageStack, const float sigma_s, const int level)
	{
		if (ImageStack.size() != level + 1) ImageStack.resize(level + 1);
		//for (int l = 0; l <= level; l++) ImageStack[l].create(src.size(), CV_32F);

		constexpr float nsigma = 3.f;//3.f;

		//#pragma omp parallel for
		for (int l = 0; l < level; l++)
		{
			const float sigma_c = (float)getPyramidSigma(sigma_s, l + 0);
			const float sigma_l = (float)getPyramidSigma(sigma_s, l + 1);
			const int r = (int)ceil(sigma_l * nsigma);
			const Size ksize(2 * r + 1, 2 * r + 1);
			Mat gc = getGaussianKernel(2 * r + 1, sigma_c);
			Mat gl = getGaussianKernel(2 * r + 1, sigma_l);
			Mat kernel = gc - gl;
			cv::sepFilter2D(src, ImageStack[l], CV_32F, kernel, kernel);
		}
		ImageStack[level] = src.clone();
	}

	void MultiScaleFilter::collapseDoGStack(vector<Mat>& ImageStack, Mat& dest, const int depth)
	{
		const int level = (int)ImageStack.size();

		if (isDoGPyramidApprox)
		{
			Mat tmp;
			double sigma = pow(2, level - 2);
			int r = int(ceil(3.f * sigma));
			GaussianBlur(ImageStack[level - 1], tmp, Size(2 * r + 1, 2 * r + 1), sigma);
			Mat ret = tmp.clone();
			for (int i = level - 2; i != 0; i--)
			{
				Mat tmp;
				double sigma = pow(2, i - 1);
				int r = int(ceil(3.f * sigma));
				GaussianBlur(ImageStack[i], tmp, Size(2 * r + 1, 2 * r + 1), sigma);
				cv::add(ret, tmp, ret);
			}
			if (depth == CV_32F) cv::add(ret, ImageStack[0], dest);
			else
			{
				cv::add(ret, ImageStack[0], ret);
				ret.convertTo(dest, depth);
			}
		}
		else
		{
			/*DoGStack[0].copyTo(dest);
			for (int i = 1; i < level; i++)
			{
				dest += DoGStack[i];
			}
			*/
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
			if (depth != CV_32F) dest.convertTo(dest, depth);
		}

	}

	void MultiScaleFilter::collapseCoGStack(vector<Mat>& ImageStack, Mat& dest, const int depth)
	{
		const int level = (int)ImageStack.size() - 1;

		dest.create(ImageStack[0].size(), CV_32F);
		AutoBuffer<float*> ptr(level + 1);
		for (int l = 0; l < level + 1; l++)
		{
			ptr[l] = ImageStack[l].ptr<float>();
		}
		float* dptr = dest.ptr<float>();
		const int size = ImageStack[0].rows * ImageStack[0].cols;
		const int simd_end = get_simd_floor(size, 32);
		const __m256 mone = _mm256_set1_ps(1.f);
#pragma omp parallel for
		for (int i = 0; i < simd_end; i += 32)
		{
			__m256 sum0 = _mm256_loadu_ps(ptr[level] + i);
			__m256 sum1 = _mm256_loadu_ps(ptr[level] + i + 8);
			__m256 sum2 = _mm256_loadu_ps(ptr[level] + i + 16);
			__m256 sum3 = _mm256_loadu_ps(ptr[level] + i + 24);

			for (int l = level - 1; l >= 0; l--)
			{
				const float* idx = ptr[l] + i;
				sum0 = _mm256_mul_ps(sum0, _mm256_add_ps(mone, _mm256_loadu_ps(idx + 0)));
				sum1 = _mm256_mul_ps(sum1, _mm256_add_ps(mone, _mm256_loadu_ps(idx + 8)));
				sum2 = _mm256_mul_ps(sum2, _mm256_add_ps(mone, _mm256_loadu_ps(idx + 16)));
				sum3 = _mm256_mul_ps(sum3, _mm256_add_ps(mone, _mm256_loadu_ps(idx + 24)));
			}
			_mm256_storeu_ps(dptr + i + 0, sum0);
			_mm256_storeu_ps(dptr + i + 8, sum1);
			_mm256_storeu_ps(dptr + i + 16, sum2);
			_mm256_storeu_ps(dptr + i + 24, sum3);
		}
		//for (int i = 0; i < size; i++)
		for (int i = simd_end; i < size; i++)
		{
			dptr[i] = ptr[level][i];
			for (int l = level - 1; l >= 0; l--)
			{
				dptr[i] *= (ptr[l][i] + 1.f);
			}
		}
		if (depth != CV_32F) dest.convertTo(dest, depth);
	}

#pragma endregion

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
		if (scalespaceMethod == Pyramid) pyramid(src, dest);
		if (scalespaceMethod == DoG) dog(src, dest);
		if (scalespaceMethod == ContrastPyramid) contrastpyramid(src, dest);
		if (scalespaceMethod == CoG) cog(src, dest);
		if (scalespaceMethod == DoGSep) dogsep(src, dest);
	}

	void MultiScaleFilter::showImageStack(string wname)
	{
		Mat show;
		hconcat(ImageStack, show);
		imshowScale(wname, show);
	}

	void MultiScaleFilter::showPyramid(string wname, vector<Mat>& pyramid, float scale, bool isShowLevel)
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
		vconcat(pyramid[1] * scale, expandv, expandv);
		hconcat(pyramid[0] * scale, expandv, show);
		int vindex = pyramid[1].rows;
		for (int l = 2; l < size; l++)
		{
			Mat roi = show(Rect(pyramid[0].cols, vindex, pyramid[l].cols, pyramid[l].rows));
			if (l != size - 1) Mat(pyramid[l] * scale).copyTo(roi);
			else pyramid[l].copyTo(roi);
			vindex += pyramid[l].rows;
		}
		Mat show8u;
		show.convertTo(show8u, CV_8U);
		if (show8u.channels() == 1) cvtColor(show8u, show8u, COLOR_GRAY2BGR);
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
			cout << "do not initialized: drawRemap" << endl;
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
	void MultiScaleFilter::setAdaptive(const bool adaptive_method, cv::Mat& adaptiveSigmaMap, cv::Mat& adaptiveBoostMap, const int level)
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

	void MultiScaleFilter::setDoGPyramidApprox(bool flag)
	{
		this->isDoGPyramidApprox = flag;
	}

	void MultiScaleFilter::setIsCompute(bool flag)
	{
		this->isCompute = flag;
	}

	int MultiScaleFilter::getGaussianRadius(const float sigma)
	{
		int ret = 0;
		if (pyramidComputeMethod == OpenCV)
		{
			ret = 2;
		}
		else
		{
			ret = int(ceil(nsigma * sigma));
			//ret= get_simd_ceil(int(ceil(nsigma * sigma)), 2);//currently 2 multiple is required for pyramid
		}
		return ret;
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

#pragma region LocalMultiScaleFilter
#if 1
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

		//ComputePyramidSize cps((int)pow(2, level + 1), radius, level + 1);
		//imshow("py", cps.vizPyramid());
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

			/*int bl, br, is, bs;
			bool eo;
			cps.get(l, bl, br, is, bs, eo);
			print_debug4(r_pad_gfpyl[l], r_pad_localpyl[l], block_sizel[l], patch_sizel[l]);
			print_debug5(bl, br, is, bs, eo);*/
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
			const int height = heightl[l];//src.rows / (int)pow(2, l)
			const int width = widthl[l];// src.colss / (int)pow(2, l)
			const int r_pad_gfpy = r_pad_gfpyl[l];
			const int r_pad_localpy = r_pad_localpyl[l];
			const int amp = ampl[l]; //pow(2, l)
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
					//print_debug(radius);
#pragma omp parallel for schedule (dynamic)
					for (int j = 0; j < height; j += block_size)
					{
						vector<Mat> llp;
						Mat rm(patch_size, patch_size, CV_32F);
						cv::AutoBuffer<float> linebuff(patch_size);
						for (int i = 0; i < width; i += block_size)
						{
							//generating pyramid from 0 to l+1
							const int x = i * amp + r_pad0 - r_pad_localpy;
							const int y = j * amp + r_pad0 - r_pad_localpy;
							//print_debug5(x, y, patch_size, ImageStack[0].cols, ImageStack[0].rows);
							//if (ImageStack[0].cols - (x + patch_size) <= 0) cout << "ng x:" << ImageStack[0].cols - (x + patch_size) << endl;
							//if (ImageStack[0].rows - (y + patch_size) <= 0)cout << "ng y:" << ImageStack[0].rows - (y + patch_size) << endl;
							//ImageStack[0].size() := Size(src.cols+r_pad0, src.rows + r_pad0)
							const Mat rect = ImageStack[0](Rect(x, y, patch_size, patch_size)).clone();
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
									else if (radius == 3)
									{
										buildLaplacianPyramid<7, 4, 8>(rm, llp, l + 1, sigma_space, linebuff);
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
					}
				}
			}
		}
		if (isDebug) showPyramid("Laplacian Pyramid Paris2011", LaplacianPyramid);
		collapseLaplacianPyramid(LaplacianPyramid, dest, src.depth());//override srcf for saving memory	
		dest(Rect(r, r, src_.cols, src_.rows)).copyTo(dest_);
	}
#endif
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
		const int rmax = (int)ceil(sigma_lmax * nsigma);
		const Size ksizemax(2 * rmax + 1, 2 * rmax + 1);

		const int r_pad = (int)pow(2, level + 1);//2^(level+1)
		vector<Mat>  DetailStack(level + 1);

		//(1) build Gaussian stack
		buildGaussianStack(srcf, ImageStack, sigma_space, level);
		for (int i = 0; i <= level; i++)
		{
			DetailStack[i].create(ImageStack[0].size(), CV_32F);
		}
		Mat im;
		copyMakeBorder(srcf, im, rmax, rmax, rmax, rmax, borderType);

		//(2) build DoG stack (0 to level-1)
		for (int l = 0; l < level; l++)
		{
			const float sigma_lc = (float)getPyramidSigma(sigma_space, l + 0);
			const float sigma_lp = (float)getPyramidSigma(sigma_space, l + 1);
			const int r1 = (int)ceil(sigma_lc * nsigma);
			const int r2 = (int)ceil(sigma_lp * nsigma);
			const Size ksize1(2 * r1 + 1, 2 * r1 + 1);
			const Size ksize2(2 * r2 + 1, 2 * r2 + 1);
			AutoBuffer<float> weightDoG(ksize2.area());
			AutoBuffer<int> indexDoG(ksize2.area());
			setDoGKernel(weightDoG, indexDoG, im.cols, ksize1, ksize2, sigma_lc, sigma_lp);

#pragma omp parallel for schedule (dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				for (int i = 0; i < src.cols; i++)
				{
					const float g = ImageStack[l].at<float>(j, i);
					DetailStack[l].at<float>(j, i) = getRemapDoGConv(im, g, j + rmax, i + rmax, ksize2.area(), indexDoG, weightDoG, isCompute);
				}
			}
		}
		//(2) the last level is a copy of the last level of Gaussian stack
		ImageStack[level].copyTo(DetailStack[level]);

		//(3) collapseDoG
		collapseDoGStack(DetailStack, dest, src.depth());
	}

	void LocalMultiScaleFilter::filter(const Mat& src, Mat& dest, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);

		this->sigma_space = sigma_space;
		this->level = max(level, 1);
		this->boost = boost;
		this->sigma_range = sigma_range;
		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);
		freeSpaceWeight();
	}

	void LocalMultiScaleFilter::setDoGKernel(float* weight, int* index, const int index_step, Size ksize1, Size ksize2, const float sigma1, const float sigma2)
	{
		CV_Assert(sigma2 > sigma1);

		const int r1 = ksize1.width / 2;
		const int r2 = ksize2.width / 2;
		int count = 0;
		if (sigma1 <= FLT_EPSILON)
		{
			double sum2 = 0.0;
			const float coeff2 = float(1.0 / (-2.0 * sigma2 * sigma2));
			for (int j = -r2; j <= r2; j++)
			{
				for (int i = -r2; i <= r2; i++)
				{
					const float dist = float(j * j + i * i);
					const float v2 = exp(dist * coeff2);
					weight[count] = v2;
					index[count] = j * index_step + i;
					count++;
					sum2 += v2;
				}
			}
			sum2 = 1.0 / sum2;
			for (int i = 0; i < ksize2.area(); i++)
			{
				weight[i] = float(0.0 - weight[i] * sum2);
			}
			weight[ksize2.area() / 2] = 1.f + weight[ksize2.area() / 2];
		}
		else
		{
			AutoBuffer<float> weight2(ksize2.area());
			double sum1 = 0.0;
			double sum2 = 0.0;
			const float coeff1 = float(1.0 / (-2.0 * sigma1 * sigma1));
			const float coeff2 = float(1.0 / (-2.0 * sigma2 * sigma2));
			for (int j = -r2; j <= r2; j++)
			{
				for (int i = -r2; i <= r2; i++)
				{
					float dist = float(j * j + i * i);
					float v1 = (abs(i) <= r1 && abs(j) <= r1) ? exp(dist * coeff1) : 0.f;
					float v2 = exp(dist * coeff2);
					weight[count] = v1;
					weight2[count] = v2;
					index[count] = j * index_step + i;
					count++;
					sum1 += (double)v1;
					sum2 += (double)v2;
				}
			}
			sum1 = 1.0 / sum1;
			sum2 = 1.0 / sum2;
			for (int i = 0; i < ksize2.area(); i++)
			{
				weight[i] = float(weight[i] * sum1 - weight2[i] * sum2);
			}
		}
	}

	float LocalMultiScaleFilter::getRemapDoGConv(const Mat& src, const float g, const int y, const int x, const int kernelSize, int* index, float* weight, bool isCompute)
	{
		const int simd_ksize = get_simd_floor(kernelSize, 8);
		const float* sptr = src.ptr<float>(y, x);

		const __m256 mg = _mm256_set1_ps(g);
		float sum = 0.f;
		__m256 msum = _mm256_setzero_ps();
		if (isCompute)
		{
			const float coeff = 1.f / (-2.f * sigma_range * sigma_range);
			const __m256 mcoeff = _mm256_set1_ps(coeff);
			const __m256 mboost = _mm256_set1_ps(boost);
			for (int i = 0; i < simd_ksize; i += 8)
			{
				const __m256i idx = _mm256_load_si256((const __m256i*)index);
				const __m256 ms = _mm256_i32gather_ps(sptr, idx, sizeof(float));
				const __m256 mx = _mm256_sub_ps(ms, mg);
				const __m256 md = _mm256_fmadd_ps(_mm256_mul_ps(mboost, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(mx, mx), mcoeff))), mx, ms);
				msum = _mm256_fmadd_ps(_mm256_load_ps(weight), md, msum);
				weight += 8;
				index += 8;
			}
			sum = _mm256_reduceadd_ps(msum);
			//for (int i = 0; i < kernelSize; i++)
			for (int i = simd_ksize; i < kernelSize; i++)
			{
				const float s = sptr[*index];
				const float x = (s - g);
				const float d = fma(boost * exp(x * x * coeff), x, s);
				sum += *weight * d;
				weight++;
				index++;
			}
		}
		else
		{
			const float* rptr = &rangeTable[0];
			for (int i = 0; i < simd_ksize; i += 8)
			{
				const __m256i idx = _mm256_load_si256((const __m256i*)index);
				const __m256 ms = _mm256_i32gather_ps(sptr, idx, sizeof(float));
				const __m256 mx = _mm256_sub_ps(ms, mg);
				const __m256 md = _mm256_fmadd_ps(_mm256_i32gather_ps(rptr, _mm256_cvtps_epi32(_mm256_abs_ps(mx)), sizeof(float)), mx, ms);
				msum = _mm256_fmadd_ps(_mm256_load_ps(weight), md, msum);
				weight += 8;
				index += 8;
			}

			sum = _mm256_reduceadd_ps(msum);
			//for (int i = 0; i < size; i++)
			for (int i = simd_ksize; i < kernelSize; i++)
			{
				const float s = sptr[*index];
				const float x = (s - g);
				const float d = s + x * rangeTable[saturate_cast<uchar>(abs(x))];
				sum += *weight * d;
				weight++;
				index++;
			}
		}
		return sum;
	}
#pragma endregion
#pragma endregion
}