#include "stdafx.h"

using namespace cv;
using namespace std;

namespace cp
{
	//cv::BORDER_CONSTANT
	//inline int GaussianFilterSeparableFIR::border_s(const int val) { return (val < 0) ? -1 : val; };
	//inline int GaussianFilterSeparableFIR::border_e(const int val, const int maxval) { return (val > maxval) ? -1 : val; };

	//cv::BORDER_REPLICATE
	//inline int GaussianFilterSeparableFIR::border_s(const int val) { return cv::max(0, val); }
	//inline int GaussianFilterSeparableFIR::border_e(const int val, const int maxval) { return cv::min(maxval, val); }

	//cv::BORDER_REFLECT
#define BORDER_REFLECT_FILTER 1
	inline int GaussianFilterSeparableFIR::border_s(const int val) { return (val >= 0) ? val : -val - 1; }
	inline int GaussianFilterSeparableFIR::border_e(const int val, const int maxval) { return (val <= maxval) ? val : 2 * maxval - val + 1; }

	inline int border_s(const int val) { return (val >= 0) ? val : -val - 1; }
	inline int border_e(const int val, const int maxval) { return (val <= maxval) ? val : 2 * maxval - val + 1; }

	//cv::BORDER_REFLECT101
	//inline int GaussianFilterSeparableFIR::border_s(const int val) { return cv::abs(val); }
	//inline int GaussianFilterSeparableFIR::border_e(const int val, const int maxval) { return maxval- cv::abs(maxval-val); }

	//#define BORDER_CONSTANT 1
	inline int get_simd_ceil(int val, int simdwidth)
	{
		int v = (val % simdwidth == 0) ? val : (val / simdwidth + 1) * simdwidth;
		return v;
	}

	inline int get_simd_floor(int val, int simdwidth)
	{
		return (val / simdwidth) * simdwidth;
	}

	inline int get_simd_floor_end(int start, int end, int simdwidth)
	{
		return get_simd_floor(end - start, simdwidth) + start;
	}

	inline int get_simd_ceil_end(int start, int end, int simdwidth)
	{
		return get_simd_ceil(end - start, simdwidth) + start;
	}

	//modified
	void myCopyMakeBorder32F(Mat& src, Mat& dest, int top, int bottom, int left, int right, int type)
	{
		Size size = Size(src.cols + left + right, src.rows + top + bottom);
		if (dest.size() != size)
		{
			//cout << "create" << endl;
			dest.create(size, src.type());
		}
		/*if (left == 0 && right == 0)
		{
			pv(src, dest, top, bottom, type);
			return;
		}*/

		const int LEFT = get_simd_floor(left, 8);
		const int RIGHT = get_simd_floor(right, 8);
		/*
		rが大きいと，例外処理が発生する
		for (int n = 0; n < max_core; n++)
		{
			const int strip = dest.rows / max_core;
			if (n == 0)
			{
				const int start = 0;
				const int end = (n + 1)*strip;
				for (int j = start; j < end; j++)
				{
					float* s = src.ptr<float>(j);
					float* d = dest.ptr<float>(j + top);
					for (int i = 0; i < left; i++)
					{
						d[i] = s[left - i - 1];
					}
					memcpy(d + left, s, sizeof(float)*src.cols);
					for (int i = 0; i < right; i++)
					{
						d[left + src.cols + i] = s[src.cols - 1 - i];
					}
				}

				for (int j = 0; j < top; j++)
				{
					float* s = dest.ptr<float>(2 * top - j - 1);
					float* d = dest.ptr<float>(j);
					memcpy(d, s, sizeof(float)*(src.cols + left + right));
				}
			}
			else if (n == max_core - 1)
			{
				const int start = n*strip;
				const int end = src.rows;
				for (int j = start; j < end; j++)
				{
					float* s = src.ptr<float>(j);
					float* d = dest.ptr<float>(j + top);
					for (int i = 0; i < left; i++)
					{
						d[i] = s[left - i - 1];
					}
					memcpy(d + left, s, sizeof(float)*src.cols);
					for (int i = 0; i < right; i++)
					{
						d[left + src.cols + i] = s[src.cols - 1 - i];
					}
				}

				for (int j = 0; j < bottom; j++)
				{
					float* s = dest.ptr<float>(src.rows + top - 1 - j);
					float* d = dest.ptr<float>(src.rows + top + j);
					memcpy(d, s, sizeof(float)*(src.cols + left + right));
				}
			}
			else
			{
				const int start = n*strip;
				const int end = (n + 1)*strip;
				for (int j = start; j < end; j++)
				{
					float* s = src.ptr<float>(j);
					float* d = dest.ptr<float>(j + top);
					for (int i = 0; i < left; i++)
					{
						d[i] = s[left - i - 1];
					}
					memcpy(d + left, s, sizeof(float)*src.cols);
					for (int i = 0; i < right; i++)
					{
						d[left + src.cols + i] = s[src.cols - 1 - i];
					}
				}
			}
		}*/

		if (left % 8 == 0 && right % 8 == 0)
		{
			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = dest.ptr<float>(j + top);

				for (int i = 0; i < LEFT; i += 8)
				{
					__m256 a = _mm256_load_ps(s + LEFT - i - 8);
					a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
					a = _mm256_permute2f128_ps(a, a, 1);
					_mm256_store_ps(d + i, a);
				}
				memcpy(d + LEFT, s, sizeof(float) * src.cols);

				for (int i = 0; i < RIGHT; i += 8)
				{
					__m256 a = _mm256_load_ps(s + src.cols - 8 - i);
					a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
					a = _mm256_permute2f128_ps(a, a, 1);
					_mm256_store_ps(d + src.cols + i + LEFT, a);
				}
			}

			for (int j = 0; j < top; j++)
			{
				float* s = dest.ptr<float>(2 * top - j - 1);
				float* d = dest.ptr<float>(j);
				memcpy(d, s, sizeof(float) * (src.cols + left + right));
			}
			for (int j = 0; j < bottom; j++)
			{
				float* s = dest.ptr<float>(src.rows + top - 1 - j);
				float* d = dest.ptr<float>(src.rows + top + j);
				memcpy(d, s, sizeof(float) * (src.cols + left + right));
			}
		}
		else
		{
			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = dest.ptr<float>(j + top);

				for (int i = 0; i < left; i++)
				{
					d[i] = s[left - i - 1];
				}

				memcpy(d + left, s, sizeof(float) * src.cols);

				for (int i = 0; i < right; i++)
				{
					d[left + src.cols + i] = s[src.cols - 1 - i];
				}
			}

			for (int j = 0; j < top; j++)
			{
				float* s = dest.ptr<float>(2 * top - j - 1);
				float* d = dest.ptr<float>(j);
				memcpy(d, s, sizeof(float) * (src.cols + left + right));
			}
			for (int j = 0; j < bottom; j++)
			{
				float* s = dest.ptr<float>(src.rows + top - 1 - j);
				float* d = dest.ptr<float>(src.rows + top + j);
				memcpy(d, s, sizeof(float) * (src.cols + left + right));
			}
		}
	}

	void myCopyMakeBorder64F(Mat& src, Mat& dest, int top, int bottom, int left, int right, int type)
	{
		Size size = Size(src.cols + left + right, src.rows + top + bottom);
		if (dest.size() != size)
		{
			//cout << "create" << endl;
			dest.create(size, src.type());
		}
		/*if (left == 0 && right == 0)
		{
			pv(src, dest, top, bottom, type);
			return;
		}*/

		const int LEFT = get_simd_floor(left, 4);
		const int RIGHT = get_simd_floor(right, 4);
		/*
		rが大きいと，例外処理が発生する
		for (int n = 0; n < max_core; n++)
		{
			const int strip = dest.rows / max_core;
			if (n == 0)
			{
				const int start = 0;
				const int end = (n + 1)*strip;
				for (int j = start; j < end; j++)
				{
					float* s = src.ptr<float>(j);
					float* d = dest.ptr<float>(j + top);
					for (int i = 0; i < left; i++)
					{
						d[i] = s[left - i - 1];
					}
					memcpy(d + left, s, sizeof(float)*src.cols);
					for (int i = 0; i < right; i++)
					{
						d[left + src.cols + i] = s[src.cols - 1 - i];
					}
				}

				for (int j = 0; j < top; j++)
				{
					float* s = dest.ptr<float>(2 * top - j - 1);
					float* d = dest.ptr<float>(j);
					memcpy(d, s, sizeof(float)*(src.cols + left + right));
				}
			}
			else if (n == max_core - 1)
			{
				const int start = n*strip;
				const int end = src.rows;
				for (int j = start; j < end; j++)
				{
					float* s = src.ptr<float>(j);
					float* d = dest.ptr<float>(j + top);
					for (int i = 0; i < left; i++)
					{
						d[i] = s[left - i - 1];
					}
					memcpy(d + left, s, sizeof(float)*src.cols);
					for (int i = 0; i < right; i++)
					{
						d[left + src.cols + i] = s[src.cols - 1 - i];
					}
				}

				for (int j = 0; j < bottom; j++)
				{
					float* s = dest.ptr<float>(src.rows + top - 1 - j);
					float* d = dest.ptr<float>(src.rows + top + j);
					memcpy(d, s, sizeof(float)*(src.cols + left + right));
				}
			}
			else
			{
				const int start = n*strip;
				const int end = (n + 1)*strip;
				for (int j = start; j < end; j++)
				{
					float* s = src.ptr<float>(j);
					float* d = dest.ptr<float>(j + top);
					for (int i = 0; i < left; i++)
					{
						d[i] = s[left - i - 1];
					}
					memcpy(d + left, s, sizeof(float)*src.cols);
					for (int i = 0; i < right; i++)
					{
						d[left + src.cols + i] = s[src.cols - 1 - i];
					}
				}
			}
		}*/

		if (left % 4 == 0 && right % 4 == 0)
		{
			for (int j = 0; j < src.rows; j++)
			{
				double* s = src.ptr<double>(j);
				double* d = dest.ptr<double>(j + top);

				for (int i = 0; i < LEFT; i += 4)
				{
					__m256d a = _mm256_load_pd(s + LEFT - i - 4);
					a = _mm256_shuffle_pd(a, a, 0b0101);
					a = _mm256_permute2f128_pd(a, a, 1);
					_mm256_store_pd(d + i, a);
				}
				memcpy(d + LEFT, s, sizeof(float) * src.cols);

				for (int i = 0; i < RIGHT; i += 4)
				{
					__m256d a = _mm256_load_pd(s + src.cols - 4 - i);
					a = _mm256_shuffle_pd(a, a, 0b0101);
					a = _mm256_permute2f128_pd(a, a, 1);
					_mm256_store_pd(d + src.cols + i + LEFT, a);
				}
			}

			for (int j = 0; j < top; j++)
			{
				double* s = dest.ptr<double>(2 * top - j - 1);
				double* d = dest.ptr<double>(j);
				memcpy(d, s, sizeof(double) * (src.cols + left + right));
			}
			for (int j = 0; j < bottom; j++)
			{
				double* s = dest.ptr<double>(src.rows + top - 1 - j);
				double* d = dest.ptr<double>(src.rows + top + j);
				memcpy(d, s, sizeof(double) * (src.cols + left + right));
			}
		}
		else
		{
			for (int j = 0; j < src.rows; j++)
			{
				double* s = src.ptr<double>(j);
				double* d = dest.ptr<double>(j + top);

				for (int i = 0; i < left; i++)
				{
					d[i] = s[left - i - 1];
				}

				memcpy(d + left, s, sizeof(double) * src.cols);

				for (int i = 0; i < right; i++)
				{
					d[left + src.cols + i] = s[src.cols - 1 - i];
				}
			}

			for (int j = 0; j < top; j++)
			{
				double* s = dest.ptr<double>(2 * top - j - 1);
				double* d = dest.ptr<double>(j);
				memcpy(d, s, sizeof(double) * (src.cols + left + right));
			}
			for (int j = 0; j < bottom; j++)
			{
				double* s = dest.ptr<double>(src.rows + top - 1 - j);
				double* d = dest.ptr<double>(src.rows + top + j);
				memcpy(d, s, sizeof(double) * (src.cols + left + right));
			}
		}
	}

	void myCopyMakeBorder(Mat& src, Mat& dest, int top, int bottom, int left, int right, int type)
	{
		if (src.depth() == CV_32F)myCopyMakeBorder32F(src, dest, top, bottom, left, right, type);
		if (src.depth() == CV_64F)myCopyMakeBorder64F(src, dest, top, bottom, left, right, type);
	}

	inline void copyMakeBorderLineWithoutBodyCopy(float* s, float* d, const int srcwidth, int left, int right, int type)
	{
		d += left;
		for (int i = 0; i < left; i += 8)
		{
			__m256 a = _mm256_load_ps(s + i);
			a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
			a = _mm256_permute2f128_ps(a, a, 1);
			_mm256_store_ps(d - i - 8, a);
		}
		s += srcwidth;
		d += srcwidth;
		for (int i = 0; i < right; i += 8)
		{
			__m256 a = _mm256_load_ps(s - 8 - i);
			a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
			a = _mm256_permute2f128_ps(a, a, 1);
			_mm256_store_ps(d + i, a);
		}
	}

	void copyMakeBorderLine(float* s, float* d, const int srcwidth, int left, int right, int type)
	{
		for (int i = 0; i < left; i += 8)
		{
			__m256 a = _mm256_load_ps(s + i);
			a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
			a = _mm256_permute2f128_ps(a, a, 1);
			_mm256_store_ps(d - i - 8 + left, a);
		}
		memcpy(d + left, s, sizeof(float) * srcwidth);
		for (int i = 0; i < right; i += 8)
		{
			__m256 a = _mm256_load_ps(s + srcwidth - 8 - i);
			a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
			a = _mm256_permute2f128_ps(a, a, 1);
			_mm256_store_ps(d + srcwidth + i + left, a);
		}
	}

	void copyMakeBorderVerticalLine(float* s, float* d, const int colindex, const int height, const int widthStep, int top, int bottom, int type)
	{
		//for gather vertical copy
		const __m256i access_patternV = _mm256_setr_epi32
		(
			widthStep * 0,
			widthStep * 1,
			widthStep * 2,
			widthStep * 3,
			widthStep * 4,
			widthStep * 5,
			widthStep * 6,
			widthStep * 7
		);
		{
			float* si = s + colindex;
			float* di = d + top;
			const int sstep = 8 * widthStep;
			for (int j = 0; j < height; j += 8)
			{
				__m256 a = _mm256_i32gather_ps(si, access_patternV, sizeof(float));
				_mm256_store_ps(di, a);
				si += sstep;
				di += 8;
			}
		}
		//for (int j = 0; j < height; j++)
		//	d[j+top] = s[widthStep*j+colindex];//faster than gather operation

		for (int i = 0; i < top; i += 8)
		{
			__m256 a = _mm256_load_ps(d + i + top);
			a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
			a = _mm256_permute2f128_ps(a, a, 1);
			_mm256_store_ps(d - i - 8 + top, a);
		}

		for (int i = 0; i < bottom; i += 8)
		{
			__m256 a = _mm256_load_ps(d + height - 8 - i + top);
			a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
			a = _mm256_permute2f128_ps(a, a, 1);
			_mm256_store_ps(d + height + i + top, a);
		}
	}

	void verticalLineCopy(Mat& src, Mat& dest, const int cols_index)
	{
		/*float* s = src.ptr<float>(0) + cols_index;
		float* d = dest.ptr<float>(0);
		for (int i = 0; i < src.rows; i++)
		{
			*d = *s;
			s += src.cols;
			d++;
		}*/

		const __m256i access_pattern = _mm256_setr_epi32
		(
			src.cols * 0,
			src.cols * 1,
			src.cols * 2,
			src.cols * 3,
			src.cols * 4,
			src.cols * 5,
			src.cols * 6,
			src.cols * 7
		);
		float* s = src.ptr<float>(0) + cols_index;
		float* d = dest.ptr<float>(0);
		const int hstep = src.cols * 8;
		const int simdend = get_simd_floor(src.rows, 8);
		for (int i = 0; i < simdend; i += 8)
		{
			__m256 a = _mm256_i32gather_ps(s, access_pattern, sizeof(float));
			_mm256_store_ps(d, a);
			s += hstep;
			d += 8;
		}
		for (int i = simdend; i < src.rows; i++)
		{
			*d = *s;
			s += src.cols;
			d++;
		}
	}



	void GaussianFilterSeparableFIR::createTileIndex(const int tileSizeX, const int tileSizeY)
	{
		for (int j = 0; j < tileDiv.height; j++)
		{
			for (int i = 0; i < tileDiv.width; i++)
			{
				tileIndex[j * tileDiv.width + i] = (Size(i * tileSizeX, j * tileSizeY));
			}
		}
	}

	void GaussianFilterSeparableFIR::setTileDiv(const int tileDivX, const int tileDivY)
	{
		tileDiv = Size(tileDivX, tileDivY);
		bufferSubImage.resize(tileDiv.area());
		tileIndex.resize(tileDiv.area());
		numTiles = tileDiv.area();
		const int max_core = 1;//max_core or 1
		numTilesPerThread = numTiles / max_core;
	}


	GaussianFilterSeparableFIR::GaussianFilterSeparableFIR(cv::Size imgSize, double sigma, int trunc, int depth)
		: SpatialFilterBase(imgSize, depth)
	{
		this->gf_order = trunc;
		this->sigma = sigma;
		radius = (int)ceil(trunc * sigma);
		d = 2 * radius + 1;

		useParallelBorder = true;
		//useParallelBorder = false;
		constVal = 0;

		const int max_core = 1;
		num_threads = max_core;
		setTileDiv(1, max_core);
		bufferLineCols.resize(max_core);
		bufferLineRows.resize(max_core);
		bufferTile.resize(max_core);
		bufferTile2.resize(max_core);
		bufferTileLine.resize(max_core);
	}

	GaussianFilterSeparableFIR::GaussianFilterSeparableFIR(const int schedule, const int depth)
	{
		this->schedule = schedule;
		this->gf_order = 3;
		this->depth = depth;
		this->radius = (int)ceil(3 * sigma);
		this->d = 2 * radius + 1;

		useParallelBorder = true;
		//useParallelBorder = false;
		constVal = 0;

		const int max_core = 1;
		num_threads = max_core;
		setTileDiv(1, max_core);
		bufferLineCols.resize(max_core);
		bufferLineRows.resize(max_core);
		bufferTile.resize(max_core);
		bufferTile2.resize(max_core);
		bufferTileLine.resize(max_core);
	}

	GaussianFilterSeparableFIR::~GaussianFilterSeparableFIR()
	{
		_mm_free(gauss32F);
		_mm_free(gauss64F);
	}

	void GaussianFilterSeparableFIR::createGaussianTable32F(const int r, const float sigma)
	{
		const int ksize = 2 * r + 1;
		_mm_free(gauss32F);
		gauss32F = (float*)_mm_malloc(sizeof(float) * ksize, 32);
		const float gfrac = -1.f / (2.f * sigma * sigma);
		double gsum = 0.0;
		for (int j = -r, index = 0; j <= r; j++)
		{
			float v = exp((j * j) * gfrac);
			gsum += (double)v;
			gauss32F[index] = v;
			index++;
		}

		const int rend = int(9.0 * sigma);
		double eout = 0.0;
		for (int i = r + 1; i <= rend; i++)
		{
			const double v = exp(i * i * gfrac);
			eout += v;
		}
		const double alpha = 1.0;
		const double ialpha = 1.0 - alpha;
		gauss32F[2 * r] += (eout * alpha);
		gauss32F[0] += (eout * alpha);
		gsum += 2.0 * eout;

		for (int j = -r, index = 0; j <= r; j++)
		{
			//gauss[index] = max(FLT_EPSILON, gauss[index]/gsum);
			gauss32F[index] = (float)(gauss32F[index] / gsum);
			index++;
		}
	}

	void GaussianFilterSeparableFIR::createGaussianTable64F(const int r, const double sigma)
	{
		const int ksize = 2 * r + 1;
		_mm_free(gauss64F);
		gauss64F = (double*)_mm_malloc(sizeof(double) * ksize, 32);
		const double gfrac = -1.0 / (2.0 * sigma * sigma);
		double gsum = 0.0;
		for (int j = -r, index = 0; j <= r; j++)
		{
			double v = exp((j * j) * gfrac);
			gsum += v;
			gauss64F[index] = v;
			index++;
		}

		for (int j = -r, index = 0; j <= r; j++)
		{
			//gauss[index] = max(FLT_EPSILON, gauss[index]/gsum);
			gauss64F[index] /= gsum;
			index++;
		}
	}

	void GaussianFilterSeparableFIR::filter(const Mat& src_, Mat& dest, const int r, const float sigma, int method, int border, int vectorization, bool useAllocBuffer)
	{
		Mat src = (Mat)src_;
		if (src.depth() == CV_32F) createGaussianTable32F(r, sigma);
		else if (src.depth() == CV_64F) createGaussianTable64F(r, sigma);

		dest.create(src.size(), src.depth());

		switch (method)
		{
		case FIR2D_Border:
			filter2DFIR(src, dest, r, sigma, border);
			break;

		case FIR2D2_Border:
			filter2DFIR2(src, dest, r, sigma, border);
			break;

		case HV_Line:
			filterHVLine(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_LineBH:
			filterHVLineBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_LineBVP:
			filterHVLineBVP(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_LineBHBVP:
			filterHVLineHBVPB(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_Image:
			filterHVImage(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_ImageBH:
			filterHVImageBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_ImageBHD:
			filterHVImageBHD(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_ImageBV:
			filterHVImageBV(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_ImageBHBV:
			filterHVImageBHBV(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_ImageBHDBV:
			filterHVImageBHDBV(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_ImageBVP:
			filterHVImageBVP(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_ImageBHBVP:
			filterHVImageBHBVP(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_ImageBHDBVP:
			filterHVImageBHDBVP(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_ImageBTr:
			filterHVImageTrB(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_ImageBHBTr:
			filterHVImageBHBTr(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_Line:
			filterVHLine(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_LineBVP:
			filterVHLineBVP(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_LineBH:
			filterVHLineBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_LineBVPBH:
			filterVHLineBVPBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_Image:
			filterVHImage(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_ImageBV:
			filterVHImageBV(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_ImageBH:
			filterVHImageBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_ImageBVBH:
			filterVHImageBVBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_ImageBVP:
			filterVHImageBVP(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_ImageBVPBH:
			filterVHImageBVPBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HVI_Line:
			filterHVILine(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HVI_LineB:
			filterHVILineB(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HVI_Image:
			filterHVIImage(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHI_Line:
			filterVHILine(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHI_LineBH:
			filterVHILineB(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHIO_Line:
			filterVHILineBufferOverRun(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHI_Image:
			filterVHIImage(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHI_ImageBH:
			filterVHIImageBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHI_ImageBVBH:
			filterVHIImageBVBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_T_Image:
			filterHVTileImage(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_T_ImageBH:
			filterHVTileImageBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_T_ImageTr:
			filterHVTileImageTr(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_T_ImageBHTr:
			filterHVTileImageBHTr(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_T_ImageT2:
			filterHVTileImageT2(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_T_Image:
			filterVHTileImage(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_T_ImageBH:
			filterVHTileImageBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_T_ImageBV:
			filterVHTileImageBV(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHI_T_LineBH:
			filterVHITileLineBH(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHI_T_ImageBV:
			filterVHITileImageBV(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HVI_T_Line:
			filterHVITileLine(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_Border:
			filterHVBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HVN_Border:
			filterHVNonRasterBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_BorderD:
			filterHVDelayedBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_BorderDVP:
			filterHVDelayedVPBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_Border:
			filterVHBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_BorderD:
			filterVHDelayedBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HVI_Border:
			filterHVIBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHI_Border:
			filterVHIBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHI_BorderB:
			filterVHIBlockBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_T_Border:
			filterHVTileBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VH_T_Border:
			filterVHTileBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case VHI_T_Border:
			filterVHITileBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HVI_T_Border:
			filterHVITileBorder(src, dest, r, sigma, border, vectorization, useAllocBuffer);
			break;

		case HV_T_Sub:
			filterTileSubImage(src, dest, r, sigma, border, vectorization, useAllocBuffer, HV_T_Sub);
			break;

		case HV_T_SubD:
			filterTileSubImage(src, dest, r, sigma, border, vectorization, useAllocBuffer, HV_T_SubD);
			break;

		case HV_T_SubDVP:
			filterTileSubImage(src, dest, r, sigma, border, vectorization, useAllocBuffer, HV_T_SubDVP);
			break;

		case VH_T_Sub:
			filterTileSubImage(src, dest, r, sigma, border, vectorization, useAllocBuffer, VH_T_Sub);
			break;

		case VH_T_SubD:
			filterTileSubImage(src, dest, r, sigma, border, vectorization, useAllocBuffer, VH_T_SubD);
			break;

		case VH_T_SubVP:
			filterTileSubImage(src, dest, r, sigma, border, vectorization, useAllocBuffer, VH_T_SubVP);
			break;

		case HVI_T_Sub:
			filterTileSubImage(src, dest, r, sigma, border, vectorization, useAllocBuffer, HVI_T_Sub);
			break;

		case VHI_T_Sub:
			filterTileSubImage(src, dest, r, sigma, border, vectorization, useAllocBuffer, VHI_T_Sub);
			break;

		default:
			break;
		}
	}

	void GaussianFilterSeparableFIR::filter(const Mat& src, Mat& dest, const double sigma, const int order, const int border)
	{
		this->sigma = sigma;
		this->gf_order = order;
		this->radius = (int)ceil(order * sigma);
		this->d = 2 * radius + 1;

		filter(src, dest, radius, (float)sigma, schedule, border, vectorization, true);
	}

	void GaussianFilterSeparableFIR::body(const cv::Mat& src, cv::Mat& dst, int borderType)
	{
		//VHI_LineBH
		//HV_BorderD
		//HV_BorderDVP
		//HV_Image
		//int method = GaussianFilterSeparableFIR::HV_BorderD;
		//int method = GaussianFilterSeparableFIR::HV_BorderD;
		int method = GaussianFilterSeparableFIR::VHI_LineBH;
		Mat a = src;
		filter(a, dst, radius, float(sigma), method, borderType, VECTOR_AVX, true);
	}


	void GaussianFilterSeparableFIR::filter2DFIR2(Mat& src, Mat& dest, const int r, const float sigma, int border)
	{
		if (dest.empty())dest.create(src.size(), src.type());

		copyMakeBorder(src, bufferImageBorder, r, r, r, r, border);

		const int kwidth = 2 * r + 1;
		const int ksize = (kwidth) * (kwidth);

		float* kernel = (float*)_mm_malloc(sizeof(float) * ksize, 32);
		int* pos = (int*)_mm_malloc(sizeof(int) * ksize, 32);

		//Mat kernel(Size(2 * r + 1, 2 * r + 1), CV_32F);
		const float coeff = -1.f / (2.f * sigma * sigma);
		float sv = 0.f;

		int index = 0;
		for (int j = 0; j < kwidth; j++)
		{
			float* k = kernel;
			for (int i = 0; i < kwidth; i++)
			{
				pos[index] = bufferImageBorder.cols * j + i;
				float d = (float)(i - r) * (i - r) + (float)(j - r) * (j - r);
				const float v = cv::exp(d * coeff);
				k[index] = v;
				sv += v;
				index++;
			}
		}
		sv = float(1.0 / sv);
		float* k = kernel;

		for (int i = 0; i < ksize; i++)k[i] *= sv;

		float* ker = kernel;

		for (int j = 0; j < src.rows; j++)
		{
			for (int i = 0; i < src.cols; i += 32)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				float* si = s + i;
				__m256 mv1 = _mm256_setzero_ps();
				__m256 mv2 = _mm256_setzero_ps();
				__m256 mv3 = _mm256_setzero_ps();
				__m256 mv4 = _mm256_setzero_ps();
				for (int k = 0; k < ksize; k++)
				{
					__m256 mw = _mm256_set1_ps(ker[k]);
					__m256 mr1 = _mm256_loadu_ps(si + 0 + pos[k]);
					__m256 mr2 = _mm256_loadu_ps(si + 8 + pos[k]);
					__m256 mr3 = _mm256_loadu_ps(si + 16 + pos[k]);
					__m256 mr4 = _mm256_loadu_ps(si + 24 + pos[k]);
					mv1 = _mm256_fmadd_ps(mw, mr1, mv1);
					mv2 = _mm256_fmadd_ps(mw, mr2, mv2);
					mv3 = _mm256_fmadd_ps(mw, mr3, mv3);
					mv4 = _mm256_fmadd_ps(mw, mr4, mv4);
				}
				_mm256_store_ps(d + i + 0, mv1);
				_mm256_store_ps(d + i + 8, mv2);
				_mm256_store_ps(d + i + 16, mv3);
				_mm256_store_ps(d + i + 24, mv4);
			}
		}

		/*
		for (int j = 0; j < src.rows; j++)
		{
			float* s = sim.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 0; i < src.cols; i += 32)
			{
				float* si = s + i;
				__m256 mv1 = _mm256_setzero_ps();
				__m256 mv2 = _mm256_setzero_ps();
				__m256 mv3 = _mm256_setzero_ps();
				__m256 mv4 = _mm256_setzero_ps();
				for (int k = 0; k < ksize; k++)
				{
					__m256 mw = _mm256_set1_ps(ker[k]);
					__m256 mr1 = _mm256_loadu_ps(si + 0 + pos[k]);
					__m256 mr2 = _mm256_loadu_ps(si + 8 + pos[k]);
					__m256 mr3 = _mm256_loadu_ps(si + 16 + pos[k]);
					__m256 mr4 = _mm256_loadu_ps(si + 24 + pos[k]);
					mv1 = _mm256_fmadd_ps(mw, mr1, mv1);
					mv2 = _mm256_fmadd_ps(mw, mr2, mv2);
					mv3 = _mm256_fmadd_ps(mw, mr3, mv3);
					mv4 = _mm256_fmadd_ps(mw, mr4, mv4);
				}
				_mm256_store_ps(d + i + 0, mv1);
				_mm256_store_ps(d + i + 8, mv2);
				_mm256_store_ps(d + i + 16, mv3);
				_mm256_store_ps(d + i + 24, mv4);
			}
		}
		*/
		_mm_free(kernel);
		_mm_free(pos);
	}

	void GaussianFilterSeparableFIR::filter2DFIR(Mat& src, Mat& dest, const int r, const float sigma, int border)
	{
		Mat kernel(Size(2 * r + 1, 2 * r + 1), CV_64F);
		const double coeff = -1.0 / (2.0 * sigma * sigma);
		double sv = 0.0;

		for (int j = 0; j < kernel.rows; j++)
		{
			double* k = kernel.ptr<double>(j);
			for (int i = 0; i < kernel.cols; i++)
			{
				double d = (double)(i - r) * (i - r) + (j - r) * (j - r);
				const double v = exp(d * coeff);
				k[i] = v;
				sv += v;
			}
		}
		sv = 1.0 / sv;
		double* k = kernel.ptr<double>(0);

		for (int i = 0; i < kernel.size().area(); i++)k[i] *= sv;

		kernel.convertTo(kernel, CV_32F);

		cv::filter2D(src, dest, CV_32F, kernel, Point(r, r), 0.0, border);
	}

	//HV filtering
	void GaussianFilterSeparableFIR::filterHVLine(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int wstep = src.cols;

		if (opt == VECTOR_WITHOUT)
		{
			// h filter
			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = dest.ptr<float>(j);

				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						idx = border_e(idx, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}

			// v filter	
			for (int i = 0; i < src.cols; i++)
			{
				const int tidx = 0;
				bufferLineRows[tidx].create(get_simd_ceil(src.rows, 8), 1, CV_32F);
				for (int j = 0; j < src.rows; j++) bufferLineRows[tidx].at<float>(j) = dest.at<float>(j, i);
				float* b = bufferLineRows[tidx].ptr<float>(0);

				for (int j = 0; j < r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(j + k - r);
						v += (idx >= 0) ? gauss32F[k] * si[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int j = r; j < src.rows - r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = (j + k - r);
						v += gauss32F[k] * si[idx];
					}
					d[i] = v;
				}
				for (int j = src.rows - r; j < src.rows; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(j + k - r, vmax);
						v += (idx >= 0) ? gauss32F[k] * si[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;

			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}
#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int max_core = 1;
			const int R = get_simd_ceil(r, 8);
			// h filter
			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = src.ptr<float>(start);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					for (int i = 0; i < r; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - i; k++)
							{
								int idx = i + k;
								int maskIdx = max(0, k + i - r + 8);
								__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							int idx = i;
							for (int k = 0; k < r - i; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
						}
						float* si = s;
						for (int k = r - i; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = R; i < src.cols - R; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i - r;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = src.cols - R; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const int e = src.cols - (i + 8);
						float* si = s + i - r;
						for (int k = 0; k < r + 1 + e; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						int idx = 0;
						for (int k = r + 1 + e; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(s, end_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += src.cols;
					d += dest.cols;
				}
			}

			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.rows - 7 + i, vmax),
					border_e(src.rows - 6 + i, vmax),
					border_e(src.rows - 5 + i, vmax),
					border_e(src.rows - 4 + i, vmax),
					border_e(src.rows - 3 + i, vmax),
					border_e(src.rows - 2 + i, vmax),
					border_e(src.rows - 1 + i, vmax),
					border_e(src.rows - 0 + i, vmax)
				);
			}

			//v filter
			const int wstep0 = 0 * wstep;
			const int wstep1 = 1 * wstep;
			const int wstep2 = 2 * wstep;
			const int wstep3 = 3 * wstep;
			const int wstep4 = 4 * wstep;
			const int wstep5 = 5 * wstep;
			const int wstep6 = 6 * wstep;
			const int wstep7 = 7 * wstep;
			const int wstep8 = 8 * wstep;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int simdwidth = get_simd_ceil(src.rows, 8);
				if (!useAllocBuffer)bufferLineRows[tidx].release();
				if (bufferLineRows[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);

				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;
				for (int i = start; i < end; i++)
				{
					//verticalLineCopy(dest, bufferLineRows[tidx], i);
					for (int j = 0; j < src.rows; j++) b[j] = dest.at<float>(j, i);//faster than gather operation

					float* d = dest.ptr<float>(0) + i;

					for (int j = 0; j < r; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - j; k++)
							{
								int idx = j + k;
								int maskIdx = max(0, k + j - r + 8);
								__m256 ms = _mm256_mask_i32gather_ps(mVal, si, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							int idx = j;
							for (int k = 0; k < r - j; k++)
							{
								__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
						}
						float* bi = b;
						for (int k = r - j; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}

						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
					for (int j = R; j < src.rows - R; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = b + j - r;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
					for (int j = src.rows - R; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const int e = src.rows - (j + 8);
						float* bi = b + j - r;
						for (int k = 0; k < r + 1 + e; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						int idx = 0;
						for (int k = r + 1 + e; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVLineBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int wstep = src.cols;
		if (opt == VECTOR_WITHOUT)
		{
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = dest.ptr<float>(j);

				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						idx = border_e(idx, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter	

			for (int i = 0; i < src.cols; i++)
			{
				const int tidx = 0;
				bufferLineRows[tidx].create(get_simd_ceil(src.rows, 8), 1, CV_32F);
				for (int j = 0; j < src.rows; j++) bufferLineRows[tidx].at<float>(j) = dest.at<float>(j, i);
				float* b = bufferLineRows[tidx].ptr<float>(0);

				for (int j = 0; j < r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(j + k - r);
						v += (idx >= 0) ? gauss32F[k] * si[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int j = r; j < src.rows - r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = (j + k - r);
						v += gauss32F[k] * si[idx];
					}
					d[i] = v;
				}
				for (int j = src.rows - r; j < src.rows; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(j + k - r, vmax);
						v += (idx >= 0) ? gauss32F[k] * si[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;

			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}
#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif

			const int max_core = 1;
			const int R = get_simd_ceil(r, 8);
			// h filter

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				const int simdwidth = src.cols + 2 * R;
				if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);
				float* bptr = b + R - r;

				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = src.ptr<float>(start);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					copyMakeBorderLine(s, b, src.cols, R, R, 0);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += src.cols;
					d += dest.cols;
				}
			}

			//v filter
			const int wstep0 = 0 * wstep;
			const int wstep1 = 1 * wstep;
			const int wstep2 = 2 * wstep;
			const int wstep3 = 3 * wstep;
			const int wstep4 = 4 * wstep;
			const int wstep5 = 5 * wstep;
			const int wstep6 = 6 * wstep;
			const int wstep7 = 7 * wstep;
			const int wstep8 = 8 * wstep;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int simdwidth = get_simd_ceil(src.rows, 8);
				if (!useAllocBuffer)bufferLineRows[tidx].release();
				if (bufferLineRows[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);

				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;
				for (int i = start; i < end; i++)
				{
					//verticalLineCopy(dest, bufferLineRows[tidx], i);
					for (int j = 0; j < src.rows; j++) b[j] = dest.at<float>(j, i);//faster than gather operation

					float* d = dest.ptr<float>(0) + i;

					for (int j = 0; j < r; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - j; k++)
							{
								int idx = j + k;
								int maskIdx = max(0, k + j - r + 8);
								__m256 ms = _mm256_mask_i32gather_ps(mVal, si, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							int idx = j;
							for (int k = 0; k < r - j; k++)
							{
								__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
						}
						float* bi = b;
						for (int k = r - j; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
					for (int j = R; j < src.rows - R; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = b + j - r;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
					for (int j = src.rows - R; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const int e = src.rows - (j + 8);
						float* bi = b + j - r;
						for (int k = 0; k < r + 1 + e; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						int idx = 0;
						for (int k = r + 1 + e; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVLineBVP(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int wstep = src.cols;

		if (opt == VECTOR_WITHOUT)
		{
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = dest.ptr<float>(j);

				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						idx = border_e(idx, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter	

			for (int i = 0; i < src.cols; i++)
			{
				const int tidx = 0;
				bufferLineRows[tidx].create(get_simd_ceil(src.rows, 8), 1, CV_32F);
				for (int j = 0; j < src.rows; j++) bufferLineRows[tidx].at<float>(j) = dest.at<float>(j, i);
				float* b = bufferLineRows[tidx].ptr<float>(0);

				for (int j = 0; j < r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(j + k - r);
						v += (idx >= 0) ? gauss32F[k] * si[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int j = r; j < src.rows - r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = (j + k - r);
						v += gauss32F[k] * si[idx];
					}
					d[i] = v;
				}
				for (int j = src.rows - r; j < src.rows; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(j + k - r, vmax);
						v += (idx >= 0) ? gauss32F[k] * si[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;

			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}
#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int max_core = 1;
			const int R = get_simd_ceil(r, 8);
			// h filter

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				//const int tidx = 1;
				for (int j = start; j < end; j++)
				{
					float* s = src.ptr<float>(j);
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < r; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - i; k++)
							{
								int idx = i + k;
								int maskIdx = max(0, k + i - r + 8);
								__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							int idx = i;
							for (int k = 0; k < r - i; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
						}
						float* si = s;
						for (int k = r - i; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = R; i < src.cols - R; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i - r;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = src.cols - R; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const int e = src.cols - (i + 8);
						float* si = s + i - r;
						for (int k = 0; k < r + 1 + e; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						int idx = 0;
						for (int k = r + 1 + e; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(s, end_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}
			}

			//v filter
			const int wstep0 = 0 * wstep;
			const int wstep1 = 1 * wstep;
			const int wstep2 = 2 * wstep;
			const int wstep3 = 3 * wstep;
			const int wstep4 = 4 * wstep;
			const int wstep5 = 5 * wstep;
			const int wstep6 = 6 * wstep;
			const int wstep7 = 7 * wstep;
			const int wstep8 = 8 * wstep;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int simdwidth = src.rows + 2 * R;
				if (!useAllocBuffer)bufferLineRows[tidx].release();
				if (bufferLineRows[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);
				float* bptr = b + R - r;
				float* dptr = dest.ptr<float>(0);

				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;

				for (int i = start; i < end; i++)
				{
					copyMakeBorderVerticalLine(dptr, b, i, dest.rows, dest.cols, R, R, 0);
					float* d = dptr + i;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bj = bptr + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bj);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bj++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVLineHBVPB(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int wstep = src.cols;

		if (opt == VECTOR_WITHOUT)
		{
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = dest.ptr<float>(j);

				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						idx = border_e(idx, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter	

			for (int i = 0; i < src.cols; i++)
			{
				const int tidx = 0;
				bufferLineRows[tidx].create(get_simd_ceil(src.rows, 8), 1, CV_32F);
				for (int j = 0; j < src.rows; j++) bufferLineRows[tidx].at<float>(j) = dest.at<float>(j, i);
				float* b = bufferLineRows[tidx].ptr<float>(0);

				for (int j = 0; j < r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(j + k - r);
						v += (idx >= 0) ? gauss32F[k] * si[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int j = r; j < src.rows - r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = (j + k - r);
						v += gauss32F[k] * si[idx];
					}
					d[i] = v;
				}
				for (int j = src.rows - r; j < src.rows; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					float* si = b;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(j + k - r, vmax);
						v += (idx >= 0) ? gauss32F[k] * si[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (VECTOR_AVX)
		{
			const int max_core = 1;
			const int R = get_simd_ceil(r, 8);
			// h filter

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int tidx = 1;
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				const int simdwidth = src.cols + 2 * R;
				if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);
				float* bptr = b + R - r;
				for (int j = start; j < end; j++)
				{
					float* s = src.ptr<float>(j);
					copyMakeBorderLine(s, b, src.cols, R, R, 0);
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}
			}

			//v filter
			const int wstep0 = 0 * wstep;
			const int wstep1 = 1 * wstep;
			const int wstep2 = 2 * wstep;
			const int wstep3 = 3 * wstep;
			const int wstep4 = 4 * wstep;
			const int wstep5 = 5 * wstep;
			const int wstep6 = 6 * wstep;
			const int wstep7 = 7 * wstep;
			const int wstep8 = 8 * wstep;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer) bufferLineRows[tidx].release();
				const int simdwidth = src.rows + 2 * R;
				if (bufferLineRows[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);
				float* bptr = b + R - r;
				float* dptr = dest.ptr<float>(0);

				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;

				for (int i = start; i < end; i++)
				{
					copyMakeBorderVerticalLine(dptr, b, i, dest.rows, dest.cols, R, R, 0);
					float* d = dptr + i;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
				}
			}
		}
	}

	//modified
	void GaussianFilterSeparableFIR::filterHVImage(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)buffer.release();
		if (buffer.size() != src.size()) buffer.create(src.size(), src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int wstep = src.cols;
			const int R = get_simd_ceil(r, 8);
			//h filter
			{
				const int start = 0;
				const int end = src.rows;
				float* s = src.ptr<float>(start);
				float* d = buffer.ptr<float>(start);

				for (int j = start; j < end; j++)
				{
					for (int i = 0; i < R; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - i; k++)
							{
								int idx = i + k;
								int maskIdx = max(0, k + i - r + 8);
								__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							int idx = i;
							for (int k = 0; k < r - i; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
						}
						float* si = s;
						for (int k = r - i; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}

					for (int i = R; i < src.cols - R; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i - r;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = src.cols - R; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const int e = src.cols - (i + 8);
						float* si = s + i - r;
						for (int k = 0; k < r + 1 + e; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						int idx = 0;
						for (int k = r + 1 + e; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(s, end_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += wstep;
					d += wstep;
				}
			}

			//v filter
			{
				//const int tidx = 0;
				const int start = 0;
				const int end = src.rows;

				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;

				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < Y0; j++)
				{
					for (int i = 0; i < src.cols; i += 32)
					{
						__m256 mv = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = j;
							for (int k = 0; k < r + 1 + e; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 ms1 = _mm256_load_ps(si + 8 + idx);
								__m256 ms2 = _mm256_load_ps(si + 16 + idx);
								__m256 ms3 = _mm256_load_ps(si + 24 + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
								mv2 = _mm256_fmadd_ps(ms2, mg, mv2);
								mv3 = _mm256_fmadd_ps(ms3, mg, mv3);
							}
							si = si + (j + 1 + e) * wstep;
							for (int k = r + 1 + e; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(si);
								__m256 ms1 = _mm256_load_ps(si + 8);
								__m256 ms2 = _mm256_load_ps(si + 16);
								__m256 ms3 = _mm256_load_ps(si + 24);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
								mv2 = _mm256_fmadd_ps(ms2, mg, mv2);
								mv3 = _mm256_fmadd_ps(ms3, mg, mv3);
								si += wstep;
							}
						}
						_mm256_store_ps(d + i, mv);
						_mm256_store_ps(d + i + 8, mv1);
						_mm256_store_ps(d + i + 16, mv2);
						_mm256_store_ps(d + i + 24, mv3);
					}
					d += dest.cols;
				}
				d = dest.ptr<float>(start + Y0);
				for (int j = start + Y0; j < end - Y1; j++)
				{
					for (int i = 0; i < src.cols; i += 32)
					{
						__m256 mv = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						float* si = s + i + (j - r) * wstep;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 ms1 = _mm256_load_ps(si + 8);
							__m256 ms2 = _mm256_load_ps(si + 16);
							__m256 ms3 = _mm256_load_ps(si + 24);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
							mv2 = _mm256_fmadd_ps(ms2, mg, mv2);
							mv3 = _mm256_fmadd_ps(ms3, mg, mv3);
							si += wstep;
						}
						_mm256_store_ps(d + i, mv);
						_mm256_store_ps(d + i + 8, mv1);
						_mm256_store_ps(d + i + 16, mv2);
						_mm256_store_ps(d + i + 24, mv3);
					}
					d += dest.cols;
				}
				d = dest.ptr<float>(end - Y1);
				for (int j = end - Y1; j < end; j++)
				{
					for (int i = 0; i < src.cols; i += 32)
					{
						__m256 mv = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = src.rows - j - 1;
							float* sii = si + (j - r) * wstep;
							for (int k = 0; k < r + 1 + e; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 ms1 = _mm256_load_ps(sii + 8);
								__m256 ms2 = _mm256_load_ps(sii + 16);
								__m256 ms3 = _mm256_load_ps(sii + 24);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
								mv2 = _mm256_fmadd_ps(ms2, mg, mv2);
								mv3 = _mm256_fmadd_ps(ms3, mg, mv3);
								sii += wstep;
							}
							for (int k = r + 1 + e; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 ms1 = _mm256_load_ps(si + 8 + idx);
								__m256 ms2 = _mm256_load_ps(si + 16 + idx);
								__m256 ms3 = _mm256_load_ps(si + 24 + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
								mv2 = _mm256_fmadd_ps(ms2, mg, mv2);
								mv3 = _mm256_fmadd_ps(ms3, mg, mv3);
							}
						}
						_mm256_store_ps(d + i, mv);
						_mm256_store_ps(d + i + 8, mv1);
						_mm256_store_ps(d + i + 16, mv2);
						_mm256_store_ps(d + i + 24, mv3);
					}
					d += dest.cols;
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVImageBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)buffer.release();
		Size asize = Size(src.cols + 8, src.rows);
		if (buffer.size() != asize) buffer.create(asize, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			//h filter
			const int R = get_simd_ceil(r, 8);
			const int wstep = src.cols;
			const int bstep = buffer.cols;
			const int max_core = 1;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (bufferLineCols[tidx].size() != Size(src.cols + 2 * R, 1))bufferLineCols[tidx].create(Size(src.cols + 2 * R, 1), CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);
				float* bptr = b + R - r;

				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = src.ptr<float>(start);
				float* d = buffer.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					copyMakeBorderLine(s, b, wstep, R, R, border);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += wstep;
					d += bstep;
				}
			}
			//v filter

			for (int n = 0; n < max_core; n++)
			{
				//const int tidx = 0;
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;

				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;

				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < Y0; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = j;
							for (int k = 0; k < r + 1 + e; k++)
							{
								int idx = border_s(j + k - r) * bstep;
								__m256 ms = _mm256_loadu_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							si = si + (j + 1 + e) * bstep;
							for (int k = r + 1 + e; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += bstep;
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
				d = dest.ptr<float>(start + Y0);
				for (int j = start + Y0; j < end - Y1; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i + (j - r) * bstep;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += bstep;
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
				d = dest.ptr<float>(end - Y1);
				for (int j = end - Y1; j < end; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = src.rows - j - 1;
							float* sii = si + (j - r) * bstep;
							for (int k = 0; k < r + 1 + e; k++)
							{
								__m256 ms = _mm256_loadu_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += bstep;
							}
							for (int k = r + 1 + e; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * bstep;
								__m256 ms = _mm256_loadu_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVImageBHD(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;

		if (!useAllocBuffer)bufferImageBorder.release();
		const int R = get_simd_ceil(r, 8);
		Size asize = Size(src.cols + 2 * R, src.rows);
		if (bufferImageBorder.size() != asize) bufferImageBorder.create(asize, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter

			const int wstep = bufferImageBorder.cols;
			const int max_core = 1;


			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = src.ptr<float>(start);
				float* d = bufferImageBorder.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					float* bptr = d + R - r;
					copyMakeBorderLine(s, d, src.cols, R, R, border);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += src.cols;
					d += wstep;
				}
			}

			//v filter

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;

				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;

				float* s = bufferImageBorder.ptr<float>(0);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < Y0; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = j;
							for (int k = 0; k < r + 1 + e; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_loadu_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							si = si + (j + 1 + e) * wstep;
							for (int k = r + 1 + e; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
				d = dest.ptr<float>(start + Y0);
				for (int j = start + Y0; j < end - Y1; j++)
				{
					float* sptr = s + (j - r) * wstep;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = sptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += wstep;
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
				d = dest.ptr<float>(end - Y1);
				for (int j = end - Y1; j < end; j++)
				{
					float* sptr = s + (j - r) * wstep;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = src.rows - j - 1;
							float* sii = sptr + i;
							for (int k = 0; k < r + 1 + e; k++)
							{
								__m256 ms = _mm256_loadu_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							for (int k = r + 1 + e; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms = _mm256_loadu_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVImageBV(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)buffer.release();
		const int R = get_simd_ceil(r, 8);
		Size asize = Size(src.cols, src.rows + 2 * r);
		if (bufferImageBorder.size() != asize) bufferImageBorder.create(asize, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int wstep = src.cols;
			//h filter
			const int max_core = 1;

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = src.ptr<float>(start);
				float* d = bufferImageBorder.ptr<float>(start + r);
				for (int j = start; j < end; j++)
				{
					for (int i = 0; i < R; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - i; k++)
							{
								int idx = i + k;
								int maskIdx = max(0, k + i - r + 8);
								__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							int idx = i;
							for (int k = 0; k < r - i; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
						}
						float* si = s;
						for (int k = r - i; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}

					for (int i = R; i < src.cols - R; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i - r;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = src.cols - R; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const int e = src.cols - (i + 8);
						float* si = s + i - r;
						for (int k = 0; k < r + 1 + e; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						int idx = 0;
						for (int k = r + 1 + e; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(s, end_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += wstep;
					d += bufferImageBorder.cols;
				}
			}
			//border
			/*
	 schedule(dynamic)
			for (int j = 0; j < r; j++)
			{
				float* s = bufferImageBorder.ptr<float>(src.rows + r - 1 - j);
				float* d = bufferImageBorder.ptr<float>(src.rows + r + j);
				memcpy(d, s, sizeof(float)*(bufferImageBorder.cols));

				s = bufferImageBorder.ptr<float>(2 * r - j - 1);
				d = bufferImageBorder.ptr<float>(j);
				memcpy(d, s, sizeof(float)*(bufferImageBorder.cols));
			}*/

			for (int j = 0; j < r; j++)
			{
				float* s = bufferImageBorder.ptr<float>(src.rows + r - 1 - j);
				float* d = bufferImageBorder.ptr<float>(src.rows + r + j);
				memcpy(d, s, sizeof(float) * (bufferImageBorder.cols));
			}
			for (int j = 0; j < r; j++)
			{
				float* s = bufferImageBorder.ptr<float>(2 * r - j - 1);
				float* d = bufferImageBorder.ptr<float>(j);
				memcpy(d, s, sizeof(float) * (bufferImageBorder.cols));
			}

			//v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i + (j)*wstep;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVImageBHDBV(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)bufferImageBorder.release();
		const int R = get_simd_ceil(r, 8);
		Size asize = Size(src.cols + 2 * R, src.rows + 2 * r);
		if (bufferImageBorder.size() != asize) bufferImageBorder.create(asize, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter
			const int wstep = src.cols;
			const int max_core = 1;

			for (int n = 0; n < max_core; n++)
			{
				//const int tidx = 0;
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = src.ptr<float>(start);
				float* d = bufferImageBorder.ptr<float>(start + r);
				for (int j = start; j < end; j++)
				{
					float* bptr = d + R - r;
					copyMakeBorderLine(s, d, wstep, R, R, border);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += wstep;
					d += bufferImageBorder.cols;
				}
			}
			//border
			/* schedule(dynamic)
			for (int j = 0; j < r; j++)
			{
			float* s = bufferImageBorder.ptr<float>(src.rows + r - 1 - j);
			float* d = bufferImageBorder.ptr<float>(src.rows + r + j);
			memcpy(d, s, sizeof(float)*(bufferImageBorder.cols));

			s = bufferImageBorder.ptr<float>(2 * r - j - 1);
			d = bufferImageBorder.ptr<float>(j);
			memcpy(d, s, sizeof(float)*(bufferImageBorder.cols));
			}*/

			for (int j = 0; j < r; j++)
			{
				float* s = bufferImageBorder.ptr<float>(src.rows + r - 1 - j);
				float* d = bufferImageBorder.ptr<float>(src.rows + r + j);
				memcpy(d, s, sizeof(float) * (bufferImageBorder.cols));
			}
			for (int j = 0; j < r; j++)
			{
				float* s = bufferImageBorder.ptr<float>(2 * r - j - 1);
				float* d = bufferImageBorder.ptr<float>(j);
				memcpy(d, s, sizeof(float) * (bufferImageBorder.cols));
			}

			//v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += bufferImageBorder.cols;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVImageBHBV(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)bufferImageBorder.release();
		const int R = get_simd_ceil(r, 8);
		Size asize = Size(src.cols, src.rows + 2 * r);
		if (bufferImageBorder.size() != asize) bufferImageBorder.create(asize, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int wstep = src.cols;
			const int max_core = 1;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (bufferLineCols[tidx].size() != Size(src.cols + 2 * R, 1))bufferLineCols[tidx].create(Size(src.cols + 2 * R, 1), CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);
				float* bptr = b + R - r;

				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = src.ptr<float>(start);
				float* d = bufferImageBorder.ptr<float>(start + r);
				for (int j = start; j < end; j++)
				{
					copyMakeBorderLine(s, b, wstep, R, R, border);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += wstep;
					d += bufferImageBorder.cols;
				}
			}
			//border
			/*
			 schedule(dynamic)
			for (int j = 0; j < r; j++)
			{
			float* s = bufferImageBorder.ptr<float>(src.rows + r - 1 - j);
			float* d = bufferImageBorder.ptr<float>(src.rows + r + j);
			memcpy(d, s, sizeof(float)*(bufferImageBorder.cols));

			s = bufferImageBorder.ptr<float>(2 * r - j - 1);
			d = bufferImageBorder.ptr<float>(j);
			memcpy(d, s, sizeof(float)*(bufferImageBorder.cols));
			}*/

			for (int j = 0; j < r; j++)
			{
				float* s = bufferImageBorder.ptr<float>(src.rows + r - 1 - j);
				float* d = bufferImageBorder.ptr<float>(src.rows + r + j);
				memcpy(d, s, sizeof(float) * (bufferImageBorder.cols));
			}
			for (int j = 0; j < r; j++)
			{
				float* s = bufferImageBorder.ptr<float>(2 * r - j - 1);
				float* d = bufferImageBorder.ptr<float>(j);
				memcpy(d, s, sizeof(float) * (bufferImageBorder.cols));
			}

			//v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i + (j)*wstep;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVImageBVP(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)buffer.release();
		if (buffer.size() != src.size()) buffer.create(src.size(), src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int wstep = src.cols;
			const int R = get_simd_ceil(r, 8);
			//h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < R; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
					if (border == BORDER_CONSTANT)
					{
						for (int k = 0; k < r - i; k++)
						{
							int idx = i + k;
							int maskIdx = max(0, k + i - r + 8);
							__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
						}
					}
					else
#endif
					{
						int idx = i;
						for (int k = 0; k < r - i; k++)
						{
							__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
					}
					float* si = s;
					for (int k = r - i; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}

				for (int i = R; i < src.cols - R; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i - r;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				for (int i = src.cols - R; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					const int e = src.cols - (i + 8);
					float* si = s + i - r;
					for (int k = 0; k < r + 1 + e; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					int idx = 0;
					for (int k = r + 1 + e; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(s, end_access_pattern[idx], sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						idx++;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
			//v filter
			const int max_core = 1;
			const int wstep0 = 0 * wstep;
			const int wstep1 = 1 * wstep;
			const int wstep2 = 2 * wstep;
			const int wstep3 = 3 * wstep;
			const int wstep4 = 4 * wstep;
			const int wstep5 = 5 * wstep;
			const int wstep6 = 6 * wstep;
			const int wstep7 = 7 * wstep;
			const int wstep8 = 8 * wstep;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;
				const int simdwidth = src.rows + 2 * R;
				if (!useAllocBuffer) bufferLineRows[tidx].release();
				if (bufferLineRows[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);
				float* bptr = b + R - r;
				float* bimptr = buffer.ptr<float>(0);
				for (int i = start; i < end; i++)
				{
					copyMakeBorderVerticalLine(bimptr, b, i, dest.rows, dest.cols, R, R, 0);
					float* d = dest.ptr<float>(0) + i;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVImageBHBVP(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)buffer.release();
		if (buffer.size() != src.size()) buffer.create(src.size(), src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int max_core = 1;
			const int wstep = src.cols;
			const int R = get_simd_ceil(r, 8);

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (bufferLineCols[tidx].size() != Size(src.cols + 2 * R, 1))bufferLineCols[tidx].create(Size(src.cols + 2 * R, 1), CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);
				float* bptr = b + R - r;

				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = src.ptr<float>(start);
				float* d = buffer.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					copyMakeBorderLine(s, b, wstep, R, R, border);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += wstep;
					d += wstep;
				}
			}
			//v filter
			const int wstep0 = 0 * wstep;
			const int wstep1 = 1 * wstep;
			const int wstep2 = 2 * wstep;
			const int wstep3 = 3 * wstep;
			const int wstep4 = 4 * wstep;
			const int wstep5 = 5 * wstep;
			const int wstep6 = 6 * wstep;
			const int wstep7 = 7 * wstep;
			const int wstep8 = 8 * wstep;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;
				const int simdwidth = src.rows + 2 * R;
				if (!useAllocBuffer) bufferLineRows[tidx].release();
				if (bufferLineRows[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);
				float* bptr = b + R - r;
				float* bimptr = buffer.ptr<float>(0);
				for (int i = start; i < end; i++)
				{
					copyMakeBorderVerticalLine(bimptr, b, i, dest.rows, dest.cols, R, R, 0);
					float* d = dest.ptr<float>(0) + i;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVImageBHDBVP(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int R = get_simd_ceil(r, 8);
		Size asize = Size(src.cols + 2 * R, src.rows);
		if (bufferImageBorder.size() != asize) bufferImageBorder.create(asize, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter
			const int wstep = src.cols;
			const int max_core = 1;

			for (int n = 0; n < max_core; n++)
			{
				//const int tidx = 0;
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = src.ptr<float>(start);
				float* d = bufferImageBorder.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					float* bptr = d + R - r;
					copyMakeBorderLine(s, d, wstep, R, R, border);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += wstep;
					d += bufferImageBorder.cols;
				}
			}
			//v filter
			const int wstep0 = 0 * wstep;
			const int wstep1 = 1 * wstep;
			const int wstep2 = 2 * wstep;
			const int wstep3 = 3 * wstep;
			const int wstep4 = 4 * wstep;
			const int wstep5 = 5 * wstep;
			const int wstep6 = 6 * wstep;
			const int wstep7 = 7 * wstep;
			const int wstep8 = 8 * wstep;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;
				const int simdwidth = src.rows + 2 * R;
				if (!useAllocBuffer) bufferLineRows[tidx].release();
				if (bufferLineRows[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);
				float* bptr = b + R - r;
				float* bimptr = bufferImageBorder.ptr<float>(0);
				for (int i = start; i < end; i++)
				{
					copyMakeBorderVerticalLine(bimptr, b, i, dest.rows, bufferImageBorder.cols, R, R, 0);
					float* d = dest.ptr<float>(0) + i;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVImageTrB(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)bufferImageBorder.release();
		const int R = get_simd_ceil(r, 8);
		Size size(src.rows + 2 * R, src.cols);
		if (bufferImageBorder.size() != size) bufferImageBorder.create(size, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif

			const int bstep = bufferImageBorder.cols;
			const int bstep0 = bstep * 0;
			const int bstep1 = bstep * 1;
			const int bstep2 = bstep * 2;
			const int bstep3 = bstep * 3;
			const int bstep4 = bstep * 4;
			const int bstep5 = bstep * 5;
			const int bstep6 = bstep * 6;
			const int bstep7 = bstep * 7;
			const int bstep8 = bstep * 8;
			float* dptr = bufferImageBorder.ptr<float>(0);
			//h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = dptr + j + R;
				for (int i = 0; i < R; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
					if (border == BORDER_CONSTANT)
					{
						for (int k = 0; k < r - i; k++)
						{
							int idx = i + k;
							int maskIdx = max(0, k + i - r + 8);
							__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
						}
					}
					else
#endif
					{
						int idx = i;
						for (int k = 0; k < r - i; k++)
						{
							__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
					}
					float* si = s;
					for (int k = r - i; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					d[bstep0] = ((float*)&mv)[0];
					d[bstep1] = ((float*)&mv)[1];
					d[bstep2] = ((float*)&mv)[2];
					d[bstep3] = ((float*)&mv)[3];
					d[bstep4] = ((float*)&mv)[4];
					d[bstep5] = ((float*)&mv)[5];
					d[bstep6] = ((float*)&mv)[6];
					d[bstep7] = ((float*)&mv)[7];
					d += bstep8;
				}
				for (int i = R; i < src.cols - R; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i - r;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					d[bstep0] = ((float*)&mv)[0];
					d[bstep1] = ((float*)&mv)[1];
					d[bstep2] = ((float*)&mv)[2];
					d[bstep3] = ((float*)&mv)[3];
					d[bstep4] = ((float*)&mv)[4];
					d[bstep5] = ((float*)&mv)[5];
					d[bstep6] = ((float*)&mv)[6];
					d[bstep7] = ((float*)&mv)[7];
					d += bstep8;
				}
				for (int i = src.cols - R; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					const int e = src.cols - (i + 8);
					float* si = s + i - r;
					for (int k = 0; k < r + 1 + e; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					int idx = 0;
					for (int k = r + 1 + e; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(s, end_access_pattern[idx], sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						idx++;
					}
					d[bstep0] = ((float*)&mv)[0];
					d[bstep1] = ((float*)&mv)[1];
					d[bstep2] = ((float*)&mv)[2];
					d[bstep3] = ((float*)&mv)[3];
					d[bstep4] = ((float*)&mv)[4];
					d[bstep5] = ((float*)&mv)[5];
					d[bstep6] = ((float*)&mv)[6];
					d[bstep7] = ((float*)&mv)[7];
					d += bstep8;
				}
			}
			//v filter
			const int dstep = dest.cols;
			const int dstep0 = dstep * 0;
			const int dstep1 = dstep * 1;
			const int dstep2 = dstep * 2;
			const int dstep3 = dstep * 3;
			const int dstep4 = dstep * 4;
			const int dstep5 = dstep * 5;
			const int dstep6 = dstep * 6;
			const int dstep7 = dstep * 7;
			const int dstep8 = dstep * 8;
			const int max_core = 1;
			dptr = dest.ptr<float>(0);

			for (int n = 0; n < max_core; n++)
			{
				//const int tidx = 0;
				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;

				float* s = bufferImageBorder.ptr<float>(start) + R - r;
				for (int i = start; i < end; i++)
				{
					copyMakeBorderLineWithoutBodyCopy(s + r, s - R + r, src.rows, R, R, 0);
					float* d = dptr + i;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						d[dstep0] = ((float*)&mv)[0];
						d[dstep1] = ((float*)&mv)[1];
						d[dstep2] = ((float*)&mv)[2];
						d[dstep3] = ((float*)&mv)[3];
						d[dstep4] = ((float*)&mv)[4];
						d[dstep5] = ((float*)&mv)[5];
						d[dstep6] = ((float*)&mv)[6];
						d[dstep7] = ((float*)&mv)[7];
						d += dstep8;
					}
					s += bstep;
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVImageBHBTr(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)buffer.release();
		const int R = get_simd_ceil(r, 8);
		Size size(src.rows + 2 * R, src.cols);
		if (buffer.size() != size) buffer.create(size, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = src.ptr<float>(j);
				float* d = buffer.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				if (j < r)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int dstep = dest.cols;
			const int dstep0 = dstep * 0;
			const int dstep1 = dstep * 1;
			const int dstep2 = dstep * 2;
			const int dstep3 = dstep * 3;
			const int dstep4 = dstep * 4;
			const int dstep5 = dstep * 5;
			const int dstep6 = dstep * 6;
			const int dstep7 = dstep * 7;
			const int dstep8 = dstep * 8;

			const int bstep = buffer.cols;
			const int bstep0 = bstep * 0;
			const int bstep1 = bstep * 1;
			const int bstep2 = bstep * 2;
			const int bstep3 = bstep * 3;
			const int bstep4 = bstep * 4;
			const int bstep5 = bstep * 5;
			const int bstep6 = bstep * 6;
			const int bstep7 = bstep * 7;
			const int bstep8 = bstep * 8;

			//const int wstep = src.cols;
			//h filter
			const int max_core = 1;

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int tidx = 0;
				if (bufferLineCols[tidx].size() != Size(src.cols + 2 * R, 1))bufferLineCols[tidx].create(Size(src.cols + 2 * R, 1), CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);
				float* bptr = b + R - r;
				float* dptr = buffer.ptr<float>(0) + R;
				for (int j = start; j < end; j++)
				{
					float* s = src.ptr<float>(j);
					copyMakeBorderLine(s, b, src.cols, R, R, 0);
					float* d = dptr + j;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[bstep0] = ((float*)&mv)[0];
						d[bstep1] = ((float*)&mv)[1];
						d[bstep2] = ((float*)&mv)[2];
						d[bstep3] = ((float*)&mv)[3];
						d[bstep4] = ((float*)&mv)[4];
						d[bstep5] = ((float*)&mv)[5];
						d[bstep6] = ((float*)&mv)[6];
						d[bstep7] = ((float*)&mv)[7];
						d += bstep8;
					}
				}
			}
			//v filter	

			for (int n = 0; n < max_core; n++)
			{
				//const int tidx = 0;
				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;

				float* s = buffer.ptr<float>(start) + R - r;
				float* dptr = dest.ptr<float>(0);
				for (int i = start; i < end; i++)
				{
					copyMakeBorderLineWithoutBodyCopy(s + r, s - R + r, src.rows, R, R, 0);
					float* d = dptr + i;
					for (int j = 0; j < src.rows; j += 8)
					{
						float* si = s + j;
						__m256 mv = _mm256_setzero_ps();
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						d[dstep0] = ((float*)&mv)[0];
						d[dstep1] = ((float*)&mv)[1];
						d[dstep2] = ((float*)&mv)[2];
						d[dstep3] = ((float*)&mv)[3];
						d[dstep4] = ((float*)&mv)[4];
						d[dstep5] = ((float*)&mv)[5];
						d[dstep6] = ((float*)&mv)[6];
						d[dstep7] = ((float*)&mv)[7];
						d += dstep8;
					}
					s += bstep;
				}
			}
		}
	}

	//VH filtering
	void GaussianFilterSeparableFIR::filterVHLine(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				if (j < r)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}

			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(Size(dest.cols, 1), CV_32F);
				memcpy(buff.ptr<float>(0), dest.ptr<float>(j), sizeof(float) * dest.cols);
				float* s = buff.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int wstep = src.cols;
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int max_core = 1;
			//vfilter
			float* s = src.ptr<float>(0);

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				int start = n * strip;
				int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;
				float* d = dest.ptr<float>(start);
				for (int j = start; j < Y0; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = j + r;
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + (j + e + 1 - r) * wstep;
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
				for (int j = start + Y0; j < end - Y1; j++)
				{
					float* sptr = s + (j - r) * wstep;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = sptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += wstep;
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
				for (int j = end - Y1; j < end; j++)
				{
					float* sptr = s + (j - r) * wstep;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = -(src.rows - j) + r;
							float* sii = sptr + i;
							for (int k = 0; k < e + 1; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							for (int k = e + 1; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
			}

			//hfilter
			const int R = get_simd_ceil(r, 8);

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int simdwidth = get_simd_ceil(src.cols, 8);
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					memcpy(b, d, sizeof(float) * dest.cols);
					for (int i = 0; i < r; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - i; k++)
							{
								int idx = i + k;
								int maskIdx = max(0, k + i - r + 8);
								__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							int idx = i;
							for (int k = 0; k < r - i; k++)
							{
								__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
						}
						float* si = b;
						for (int k = r - i; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = R; i < src.cols - R; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = b + i - r;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = src.cols - R; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						int e = src.cols - (i + 8);
						float* si = b + i - r;
						for (int k = 0; k < r + 1 + e; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						int idx = 0;
						for (int k = r + 1 + e; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	//modified
	void GaussianFilterSeparableFIR::filterVHLineBVP(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				if (j < r)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}

			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(Size(dest.cols, 1), CV_32F);
				memcpy(buff.ptr<float>(0), dest.ptr<float>(j), sizeof(float) * dest.cols);
				float* s = buff.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//vfilter
			float* s = src.ptr<float>(0);
			const int R = get_simd_ceil(r, 8);
			const int wstep = src.cols;
			const int wstep0 = wstep * 0;
			const int wstep1 = wstep * 1;
			const int wstep2 = wstep * 2;
			const int wstep3 = wstep * 3;
			const int wstep4 = wstep * 4;
			const int wstep5 = wstep * 5;
			const int wstep6 = wstep * 6;
			const int wstep7 = wstep * 7;
			const int wstep8 = wstep * 8;

			const int tidx = 0;
			int start = 0;
			int end = src.cols;
			int simdwidth = src.rows + 2 * R;
			if (!useAllocBuffer)bufferLineCols[tidx].release();
			if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
			float* b = bufferLineRows[tidx].ptr<float>(0);

			//const int Y0 = (start < r) ? r - start : 0;
			//const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;
			for (int i = start; i < end; i++)
			{
				copyMakeBorderVerticalLine(s, b, i, src.rows, src.cols, R, R, 0);
				float* d = dest.ptr<float>(0) + i;
				float* bptr = b + R - r;
				for (int j = 0; j < src.rows; j += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* bi = bptr + j;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(bi);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						bi++;
					}
					d[wstep0] = ((float*)&mv)[0];
					d[wstep1] = ((float*)&mv)[1];
					d[wstep2] = ((float*)&mv)[2];
					d[wstep3] = ((float*)&mv)[3];
					d[wstep4] = ((float*)&mv)[4];
					d[wstep5] = ((float*)&mv)[5];
					d[wstep6] = ((float*)&mv)[6];
					d[wstep7] = ((float*)&mv)[7];
					d += wstep8;
				}
			}

			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			//hfilter
			start = 0;
			end = src.rows;
			simdwidth = get_simd_ceil(src.cols, 8);
			if (!useAllocBuffer)bufferLineCols[tidx].release();
			if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
			b = bufferLineCols[tidx].ptr<float>(0);
			float* d = dest.ptr<float>(start);
			for (int j = start; j < end; j++)
			{
				memcpy(b, d, sizeof(float) * dest.cols);
				for (int i = 0; i < r; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
					if (border == BORDER_CONSTANT)
					{
						for (int k = 0; k < r - i; k++)
						{
							int idx = i + k;
							int maskIdx = max(0, k + i - r + 8);
							__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
						}
					}
					else
#endif
					{
						int idx = i;
						for (int k = 0; k < r - i; k++)
						{
							__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
					}
					float* si = b;
					for (int k = r - i; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				for (int i = R; i < src.cols - R; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = b + i - r;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				for (int i = src.cols - R; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					int e = src.cols - (i + 8);
					float* si = b + i - r;
					for (int k = 0; k < r + 1 + e; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					int idx = 0;
					for (int k = r + 1 + e; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						idx++;
					}
					_mm256_store_ps(d + i, mv);
				}
				d += dest.cols;
			}

			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterVHLineBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				if (j < r)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}

			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(Size(dest.cols, 1), CV_32F);
				memcpy(buff.ptr<float>(0), dest.ptr<float>(j), sizeof(float) * dest.cols);
				float* s = buff.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int max_core = 1;
			const int wstep = src.cols;
			//vfilter
			float* s = src.ptr<float>(0);

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;

				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;
				float* d = dest.ptr<float>(start);
				for (int j = start; j < Y0; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = j + r;
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + (j + e + 1 - r) * wstep;
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
				for (int j = start + Y0; j < end - Y1; j++)
				{
					float* sptr = s + (j - r) * wstep;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = sptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += wstep;
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
				for (int j = end - Y1; j < end; j++)
				{
					float* sptr = s + (j - r) * wstep;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = -(src.rows - j) + r;
							float* sii = sptr + i;
							for (int k = 0; k < e + 1; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							for (int k = e + 1; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
			}
			//hfilter
			const int R = get_simd_ceil(r, 8);

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int simdwidth = src.cols + 2 * R;
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
				float* buffptr = bufferLineCols[tidx].ptr<float>(0);
				float* b = buffptr + R - r;

				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					copyMakeBorderLine(d, buffptr, dest.cols, R, R, border);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = b + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					d += dest.cols;
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterVHLineBVPBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				if (j < r)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}

			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(Size(dest.cols, 1), CV_32F);
				memcpy(buff.ptr<float>(0), dest.ptr<float>(j), sizeof(float) * dest.cols);
				float* s = buff.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int max_core = 1;
			//vfilter
			float* s = src.ptr<float>(0);
			const int R = get_simd_ceil(r, 8);
			const int wstep = src.cols;
			const int wstep0 = wstep * 0;
			const int wstep1 = wstep * 1;
			const int wstep2 = wstep * 2;
			const int wstep3 = wstep * 3;
			const int wstep4 = wstep * 4;
			const int wstep5 = wstep * 5;
			const int wstep6 = wstep * 6;
			const int wstep7 = wstep * 7;
			const int wstep8 = wstep * 8;


			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;

				if (!useAllocBuffer)bufferLineCols[tidx].release();
				const int simdwidth = src.rows + 2 * R;
				if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);

				//const int Y0 = (start < r) ? r - start : 0;
				//const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;
				for (int i = start; i < end; i++)
				{
					copyMakeBorderVerticalLine(s, b, i, src.rows, src.cols, R, R, 0);
					float* d = dest.ptr<float>(0) + i;
					float* bptr = b + R - r;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[wstep0] = ((float*)&mv)[0];
						d[wstep1] = ((float*)&mv)[1];
						d[wstep2] = ((float*)&mv)[2];
						d[wstep3] = ((float*)&mv)[3];
						d[wstep4] = ((float*)&mv)[4];
						d[wstep5] = ((float*)&mv)[5];
						d[wstep6] = ((float*)&mv)[6];
						d[wstep7] = ((float*)&mv)[7];
						d += wstep8;
					}
				}
			}

			//hfilter

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int simdwidth = src.cols + 2 * R;
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
				float* buffptr = bufferLineCols[tidx].ptr<float>(0);
				float* s = buffptr + R - r;
				for (int j = start; j < end; j++)
				{
					copyMakeBorderLine(dest.ptr<float>(j), buffptr, dest.cols, R, R, border);
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHImage(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)buffer.release();
		if (buffer.size() != src.size())buffer.create(src.size(), src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				if (j < r)
				{
					float* d = buffer.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					float* d = buffer.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					float* d = buffer.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}
			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			float* s = src.ptr<float>(0);
			const int max_core = 1;
			const int wstep = src.cols;
			//vfilter

			for (int n = 0; n < max_core; n++)
			{
				//const int tidx = 0;
				const int strip = src.rows / max_core;
				int start = n * strip;
				int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;

				for (int j = start; j < Y0; j++)
				{
					float* d = buffer.ptr<float>(j);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = j + r;
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_loadu_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + (j + e + 1 - r) * wstep;
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
						}
						_mm256_store_ps(d + i, mv);
					}
				}

				for (int j = start + Y0; j < end - Y1; j++)
				{
					float* d = buffer.ptr<float>(j);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						//float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							float* si = s + i + (j - r) * wstep;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
						}
						_mm256_store_ps(d + i, mv);
					}
				}

				for (int j = end - Y1; j < end; j++)
				{
					float* d = buffer.ptr<float>(j);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = -(src.rows - j) + r;
							float* sii = si + (j - r) * wstep;
							for (int k = 0; k < e + 1; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							for (int k = e + 1; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(d + i, mv);
					}
				}
			}

			//h filter
			const int R = get_simd_ceil(r, 8);

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < r; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
					if (border == BORDER_CONSTANT)
					{
						for (int k = 0; k < r - i; k++)
						{
							int idx = i + k;
							int maskIdx = max(0, k + i - r + 8);
							__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
						}
					}
					else
#endif
					{
						int idx = i;
						for (int k = 0; k < r - i; k++)
						{
							__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
					}
					float* si = s;
					for (int k = r - i; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				for (int i = R; i < src.cols - R; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i - r;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				for (int i = src.cols - R; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					int e = src.cols - (i + 8);
					float* si = s + i - r;
					for (int k = 0; k < r + 1 + e; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					int idx = 0;
					for (int k = r + 1 + e; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(s, end_access_pattern[idx], sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						idx++;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterVHImageBV(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int R = get_simd_ceil(r, 8);
		if (!useAllocBuffer)
		{
			buffer.release();
			bufferImageBorder.release();
		}
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, 0, 0, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, 0, 0, border);
		if (buffer.size() != src.size())buffer.create(src.size(), CV_32F);

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			const int max_core = 1;
			//v filter

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;

				for (int j = start; j < Y0; j++)
				{
					float* d = bufferImageBorder.ptr<float>(j) + R;
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
				for (int j = start + Y0; j < end - Y1; j++)
				{
					//float* s = src.ptr<float>(j);
					float* d = bufferImageBorder.ptr<float>(j) + R;
					for (int i = 0; i < src.cols; i++)
					{
						float* si = s + i + (j - r) * wstep;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							//int idx = j + k - r;
							v += gauss32F[k] * si[0];
							si += wstep;
						}
						d[i] = v;
					}
				}
				for (int j = end - Y1; j < end; j++)
				{
					float* d = bufferImageBorder.ptr<float>(j) + R;
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}

			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < R; i += 8)
				{
					__m256 a = _mm256_load_ps(s + i + R);
					a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
					a = _mm256_permute2f128_ps(a, a, 1);
					_mm256_store_ps(s - i - 8 + R, a);

					a = _mm256_load_ps(s + src.cols - 8 - i + R);
					a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
					a = _mm256_permute2f128_ps(a, a, 1);
					_mm256_store_ps(s + src.cols + i + R, a);
				}
				s = s + R - r;
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
					//d[i] = s[i+r];
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int wstep = bufferImageBorder.cols;
			const int max_core = 1;
			//vfilter

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				int start = n * strip;
				int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = bufferImageBorder.ptr<float>(start);
				float* d = buffer.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += wstep;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += bufferImageBorder.cols;
					d += buffer.cols;
				}
			}
			//h filter	
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif

			const int R = get_simd_ceil(r, 8);

			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < r; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
					if (border == BORDER_CONSTANT)
					{
						for (int k = 0; k < r - i; k++)
						{
							int idx = i + k;
							int maskIdx = max(0, k + i - r + 8);
							__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
						}
					}
					else
#endif
					{
						int idx = i;
						for (int k = 0; k < r - i; k++)
						{
							__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
					}
					float* si = s;
					for (int k = r - i; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				for (int i = R; i < src.cols - R; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i - r;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				for (int i = src.cols - R; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					int e = src.cols - (i + 8);
					float* si = s + i - r;
					for (int k = 0; k < r + 1 + e; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					int idx = 0;
					for (int k = r + 1 + e; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(s, end_access_pattern[idx], sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						idx++;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterVHImageBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int R = get_simd_ceil(r, 8);
		Size asize = Size(src.cols + 2 * R, src.rows);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (bufferImageBorder.size() != asize) bufferImageBorder.create(asize, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			const int max_core = 1;
			//v filter

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;

				for (int j = start; j < Y0; j++)
				{
					float* d = bufferImageBorder.ptr<float>(j) + R;
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
				for (int j = start + Y0; j < end - Y1; j++)
				{
					//float* s = src.ptr<float>(j);
					float* d = bufferImageBorder.ptr<float>(j) + R;
					for (int i = 0; i < src.cols; i++)
					{
						float* si = s + i + (j - r) * wstep;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							//int idx = j + k - r;
							//here!!!!!
							//idx++;
							//idx--;
							//v += gauss[k] * src.at<float>(idx+j, i);
							v += gauss32F[k] * si[0];
							si += wstep;
						}
						d[i] = v;
					}
				}
				for (int j = end - Y1; j < end; j++)
				{
					float* d = bufferImageBorder.ptr<float>(j) + R;
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}

			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < R; i += 8)
				{
					__m256 a = _mm256_load_ps(s + i + R);
					a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
					a = _mm256_permute2f128_ps(a, a, 1);
					_mm256_store_ps(s - i - 8 + R, a);

					a = _mm256_load_ps(s + src.cols - 8 - i + R);
					a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
					a = _mm256_permute2f128_ps(a, a, 1);
					_mm256_store_ps(s + src.cols + i + R, a);
				}
				s = s + R - r;
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
					//d[i] = s[i+r];
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			const int max_core = 1;
			//vfilter

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				int start = n * strip;
				int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;
				float* d = bufferImageBorder.ptr<float>(start) + R;
				for (int j = start; j < Y0; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = j + r;
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_loadu_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + (j + e + 1 - r) * wstep;
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += bufferImageBorder.cols;
				}
				for (int j = start + Y0; j < end - Y1; j++)
				{
					float* sptr = s + (j - r) * wstep;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						//float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							float* si = sptr + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += bufferImageBorder.cols;
				}
				for (int j = end - Y1; j < end; j++)
				{
					float* sptr = s + (j - r) * wstep;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = -(src.rows - j) + r;
							float* sii = sptr + i;
							for (int k = 0; k < e + 1; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							for (int k = e + 1; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(d + i, mv);
					}
					d += bufferImageBorder.cols;
				}
			}
			//h filter	

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = bufferImageBorder.ptr<float>(start);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					//border
					for (int i = 0; i < R; i += 8)
					{
						__m256 a = _mm256_load_ps(s + i + R);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(s - i - 8 + R, a);

						a = _mm256_load_ps(s + src.cols - 8 - i + R);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(s + src.cols + i + R, a);
					}
					float* sptr = s + R - r;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = sptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += bufferImageBorder.cols;
					d += dest.cols;
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterVHImageBVBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		//const int hmax = src.cols - 1;
		const int R = get_simd_ceil(r, 8);
		if (!useAllocBuffer)
		{
			bufferImageBorder.release();
			bufferImageBorder2.release();
		}
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, 0, 0, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, 0, 0, border);
		Size asize = Size(src.cols + 2 * R, src.rows);
		if (bufferImageBorder2.size() != asize) bufferImageBorder2.create(asize, CV_32F);


		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			const int max_core = 1;
			//v filter

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;

				for (int j = start; j < Y0; j++)
				{
					float* d = bufferImageBorder.ptr<float>(j) + R;
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
				for (int j = start + Y0; j < end - Y1; j++)
				{
					//float* s = src.ptr<float>(j);
					float* d = bufferImageBorder.ptr<float>(j) + R;
					for (int i = 0; i < src.cols; i++)
					{
						float* si = s + i + (j - r) * wstep;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							//int idx = j + k - r;
							//here!!!!!
							//idx++;
							//idx--;
							//v += gauss[k] * src.at<float>(idx+j, i);
							v += gauss32F[k] * si[0];
							si += wstep;
						}
						d[i] = v;
					}
				}
				for (int j = end - Y1; j < end; j++)
				{
					float* d = bufferImageBorder.ptr<float>(j) + R;
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}

			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < R; i += 8)
				{
					__m256 a = _mm256_load_ps(s + i + R);
					a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
					a = _mm256_permute2f128_ps(a, a, 1);
					_mm256_store_ps(s - i - 8 + R, a);

					a = _mm256_load_ps(s + src.cols - 8 - i + R);
					a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
					a = _mm256_permute2f128_ps(a, a, 1);
					_mm256_store_ps(s + src.cols + i + R, a);
				}
				s = s + R - r;
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
					//d[i] = s[i+r];
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int wstep = bufferImageBorder.cols;
			const int max_core = 1;
			//vfilter

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				int start = n * strip;
				int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = bufferImageBorder.ptr<float>(start);
				float* d = bufferImageBorder2.ptr<float>(start) + R;
				for (int j = start; j < end; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += wstep;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += bufferImageBorder.cols;
					d += bufferImageBorder2.cols;
				}
			}
			//h filter	

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = bufferImageBorder2.ptr<float>(start);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					//border
					for (int i = 0; i < R; i += 8)
					{
						__m256 a = _mm256_load_ps(s + i + R);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(s - i - 8 + R, a);

						a = _mm256_load_ps(s + src.cols - 8 - i + R);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(s + src.cols + i + R, a);
					}
					float* sptr = s + R - r;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = sptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += bufferImageBorder2.cols;
					d += dest.cols;
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHImageBVP(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;

		if (!useAllocBuffer)buffer.release();
		if (buffer.size() != src.size()) buffer.create(src.size(), src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				if (j < r)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}

			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(Size(dest.cols, 1), CV_32F);
				memcpy(buff.ptr<float>(0), dest.ptr<float>(j), sizeof(float) * dest.cols);
				float* s = buff.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int R = get_simd_ceil(r, 8);
			const int max_core = 1;
			//vfilter
			float* s = src.ptr<float>(0);
			const int bstep = bufferImageBorder.cols;
			const int bstep0 = bstep * 0;
			const int bstep1 = bstep * 1;
			const int bstep2 = bstep * 2;
			const int bstep3 = bstep * 3;
			const int bstep4 = bstep * 4;
			const int bstep5 = bstep * 5;
			const int bstep6 = bstep * 6;
			const int bstep7 = bstep * 7;
			const int bstep8 = bstep * 8;


			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				const int simdwidth = src.rows + 2 * R;
				if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);

				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;
				//const int Y0 = (start < r) ? r - start : 0;
				//const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;
				float* bptr = b + R - r;
				float* dptr = buffer.ptr<float>(0);
				for (int i = start; i < end; i++)
				{
					copyMakeBorderVerticalLine(s, b, i, src.rows, src.cols, R, R, 0);
					float* d = dptr + i;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[bstep0] = ((float*)&mv)[0];
						d[bstep1] = ((float*)&mv)[1];
						d[bstep2] = ((float*)&mv)[2];
						d[bstep3] = ((float*)&mv)[3];
						d[bstep4] = ((float*)&mv)[4];
						d[bstep5] = ((float*)&mv)[5];
						d[bstep6] = ((float*)&mv)[6];
						d[bstep7] = ((float*)&mv)[7];
						d += bstep8;
					}
				}
			}

			//h filter	
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif


			for (int j = 0; j < src.rows; j++)
			{
				float* s = buffer.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < r; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
					if (border == BORDER_CONSTANT)
					{
						for (int k = 0; k < r - i; k++)
						{
							int idx = i + k;
							int maskIdx = max(0, k + i - r + 8);
							__m256 ms = _mm256_mask_i32gather_ps(mVal, s, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
						}
					}
					else
#endif
					{
						int idx = i;
						for (int k = 0; k < r - i; k++)
						{
							__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
					}
					float* si = s;
					for (int k = r - i; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				for (int i = R; i < src.cols - R; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i - r;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				for (int i = src.cols - R; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					int e = src.cols - (i + 8);
					float* si = s + i - r;
					for (int k = 0; k < r + 1 + e; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					int idx = 0;
					for (int k = r + 1 + e; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(s, end_access_pattern[idx], sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						idx++;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterVHImageBVPBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int R = get_simd_ceil(r, 8);
		Size asize = Size(src.cols + 2 * R, src.rows);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (bufferImageBorder.size() != asize) bufferImageBorder.create(asize, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			// v filter

			for (int j = 0; j < src.rows; j++)
			{
				if (j < r)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else if (j > src.rows - r - 1)
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * s[i + idx * wstep] : gauss32F[k] * constVal;
						}
						d[i] = v;
					}
				}
				else
				{
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * s[i + idx * wstep];
						}
						d[i] = v;
					}
				}
			}

			// h filter

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(Size(dest.cols, 1), CV_32F);
				memcpy(buff.ptr<float>(0), dest.ptr<float>(j), sizeof(float) * dest.cols);
				float* s = buff.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(i + k - r);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int i = r; i < src.cols - r; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = i + k - r;
						v += gauss32F[k] * s[idx];
					}
					d[i] = v;
				}
				for (int i = src.cols - r; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(i + k - r, hmax);
						v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int max_core = 1;
			//vfilter
			float* s = src.ptr<float>(0);
			const int R = get_simd_ceil(r, 8);
			const int bstep = bufferImageBorder.cols;
			const int bstep0 = bstep * 0;
			const int bstep1 = bstep * 1;
			const int bstep2 = bstep * 2;
			const int bstep3 = bstep * 3;
			const int bstep4 = bstep * 4;
			const int bstep5 = bstep * 5;
			const int bstep6 = bstep * 6;
			const int bstep7 = bstep * 7;
			const int bstep8 = bstep * 8;


			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				const int simdwidth = src.rows + 2 * R;
				if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineRows[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);

				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;
				//const int Y0 = (start < r) ? r - start : 0;
				//const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;
				float* bptr = b + R - r;
				float* dptr = bufferImageBorder.ptr<float>(0) + R;
				for (int i = start; i < end; i++)
				{
					copyMakeBorderVerticalLine(s, b, i, src.rows, src.cols, R, R, 0);
					float* d = dptr + i;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[bstep0] = ((float*)&mv)[0];
						d[bstep1] = ((float*)&mv)[1];
						d[bstep2] = ((float*)&mv)[2];
						d[bstep3] = ((float*)&mv)[3];
						d[bstep4] = ((float*)&mv)[4];
						d[bstep5] = ((float*)&mv)[5];
						d[bstep6] = ((float*)&mv)[6];
						d[bstep7] = ((float*)&mv)[7];
						d += bstep8;
					}
				}
			}

			//hfilter

			for (int n = 0; n < max_core; n++)
			{
				//const int tidx = 0;
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				//const int simdwidth = src.cols + 2 * R;

				float* buffptr = bufferImageBorder.ptr<float>(start);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					copyMakeBorderLineWithoutBodyCopy(buffptr + R, buffptr, dest.cols, R, R, border);
					float* s = buffptr + R - r;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(d + i, mv);
					}
					buffptr += bufferImageBorder.cols;
					d += dest.cols;
				}
			}
		}
	}

	//HVI filtering
	void GaussianFilterSeparableFIR::filterHVILine(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int wstep = src.cols;
		const int hmax = src.cols - 1;
		const int vmax = src.rows - 1;

		if (opt == VECTOR_WITHOUT)
		{

			for (int i = 0; i < src.cols; i++)
			{
				Mat buff(src.rows, 1, CV_32F);
				float* b = buff.ptr<float>(0);

				//h filter
				if (i < r)
				{
					for (int j = 0; j < src.rows; j++)
					{
						float* s = src.ptr<float>(j);
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = border_s(i + k - r);
							v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
						}
						b[j] = v;
					}
				}
				else if (i > src.cols - 1 - r)
				{
					for (int j = 0; j < src.rows; j++)
					{
						float* s = src.ptr<float>(j);
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = border_e(i + k - r, hmax);
							v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
						}
						b[j] = v;
					}
				}
				else
				{
					for (int j = 0; j < src.rows; j++)
					{
						float* s = src.ptr<float>(j);
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = i + k - r;
							v += gauss32F[k] * s[idx];
						}
						b[j] = v;
					}
				}

				// v filter
				for (int j = 0; j < r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(j + k - r);
						v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int j = r; j < src.rows - r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = j + k - r;
						v += gauss32F[k] * b[idx];
					}
					d[i] = v;
				}
				for (int j = src.rows - r; j < src.rows; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(j + k - r, vmax);
						v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			__m256i midx = _mm256_setr_epi32(0 * wstep, 1 * wstep, 2 * wstep, 3 * wstep, 4 * wstep, 5 * wstep, 6 * wstep, 7 * wstep);
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.rows - 7 + i, vmax),
					border_e(src.rows - 6 + i, vmax),
					border_e(src.rows - 5 + i, vmax),
					border_e(src.rows - 4 + i, vmax),
					border_e(src.rows - 3 + i, vmax),
					border_e(src.rows - 2 + i, vmax),
					border_e(src.rows - 1 + i, vmax),
					border_e(src.rows - 0 + i, vmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			//h filter
			const int max_core = 1;
			const int wstep0 = 0 * wstep;
			const int wstep1 = 1 * wstep;
			const int wstep2 = 2 * wstep;
			const int wstep3 = 3 * wstep;
			const int wstep4 = 4 * wstep;
			const int wstep5 = 5 * wstep;
			const int wstep6 = 6 * wstep;
			const int wstep7 = 7 * wstep;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;
				const int simdsize = get_simd_ceil(src.rows, 8);
				if (!useAllocBuffer)bufferLineRows[tidx].release();
				if (bufferLineRows[tidx].size().area() != simdsize)bufferLineRows[tidx].create(simdsize, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);
				for (int i = start; i < end; i++)
				{
					//h filter
					if (i < r)
					{
						for (int j = 0; j < src.rows; j += 8)
						{
							float* s = src.ptr<float>(j);
							__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
							if (border == BORDER_CONSTANT)
							{
								for (int k = 0; k < r - i; k++)
								{
									int idx = border_s(i + k - r);
									__m256 ms;
									if (idx >= 0) ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
									else ms = mVal;
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
								for (int k = r - i; k < ksize; k++)
								{
									int idx = border_s(i + k - r);
									__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
							}
							else
#endif
							{
								int idx_ = i - r;
								for (int k = 0; k < ksize; k++)
								{
									int idx = border_s(idx_);
									__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
									idx_++;
								}
							}
							_mm256_store_ps(b + j, mv);
						}
					}
					else if (i > src.cols - 1 - r)
					{
						for (int j = 0; j < src.rows; j += 8)
						{
							float* s = src.ptr<float>(j);
							__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
							if (border == BORDER_CONSTANT)
							{
								for (int k = 0; k < src.cols - 7 + r - i; k++)
								{
									int idx = i + k - r;
									__m256 ms = ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}

								for (int k = src.cols - 7 + r - i; k < ksize; k++)
								{
									int idx = border_e(i + k - r, hmax);
									__m256 ms;
									if (idx >= 0) ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
									else ms = mVal;
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
							}
							else
#endif
							{
								for (int k = 0; k < ksize; k++)
								{
									int idx = border_e(i + k - r, hmax);
									__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
							}
							_mm256_store_ps(b + j, mv);
						}
					}
					else //mid
					{
						for (int j = 0; j < src.rows; j += 8)
						{
							float* s = src.ptr<float>(j);
							__m256 mv = _mm256_setzero_ps();
							int idx = i - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							_mm256_store_ps(b + j, mv);
						}
					}
					//v filter
					const int R = get_simd_ceil(r, 8);
					const int simdend = get_simd_floor_end(R, src.rows - r, 8);
					for (int j = 0; j < r; j += 8)
					{
						float* d = dest.ptr<float>(j);
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - j; k++)
							{
								int idx = j + k;
								int maskIdx = max(0, k + j - r + 8);
								__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							int idx = j;
							for (int k = 0; k < r - j; k++)
							{
								__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
						}
						float* bi = b;
						for (int k = r - j; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[i + 0 * wstep] = ((float*)&mv)[0];
						d[i + 1 * wstep] = ((float*)&mv)[1];
						d[i + 2 * wstep] = ((float*)&mv)[2];
						d[i + 3 * wstep] = ((float*)&mv)[3];
						d[i + 4 * wstep] = ((float*)&mv)[4];
						d[i + 5 * wstep] = ((float*)&mv)[5];
						d[i + 6 * wstep] = ((float*)&mv)[6];
						d[i + 7 * wstep] = ((float*)&mv)[7];
					}
					for (int j = R; j < simdend; j += 8)
					{
						float* d = dest.ptr<float>(j);
						__m256 mv = _mm256_setzero_ps();
						float* bi = b + j - r;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[i + wstep0] = ((float*)&mv)[0];
						d[i + wstep1] = ((float*)&mv)[1];
						d[i + wstep2] = ((float*)&mv)[2];
						d[i + wstep3] = ((float*)&mv)[3];
						d[i + wstep4] = ((float*)&mv)[4];
						d[i + wstep5] = ((float*)&mv)[5];
						d[i + wstep6] = ((float*)&mv)[6];
						d[i + wstep7] = ((float*)&mv)[7];
					}
					for (int j = simdend; j < src.rows; j += 8)
					{
						float* d = dest.ptr<float>(j);
						__m256 mv = _mm256_setzero_ps();
						int e = src.rows - (j + 8);
						float* bi = b + j - r;
						for (int k = 0; k < r + 1 + e; k++)
						{
							__m256 ms = _mm256_load_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						int idx = 0;
						for (int k = r + 1 + e; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
						d[i + wstep0] = ((float*)&mv)[0];
						d[i + wstep1] = ((float*)&mv)[1];
						d[i + wstep2] = ((float*)&mv)[2];
						d[i + wstep3] = ((float*)&mv)[3];
						d[i + wstep4] = ((float*)&mv)[4];
						d[i + wstep5] = ((float*)&mv)[5];
						d[i + wstep6] = ((float*)&mv)[6];
						d[i + wstep7] = ((float*)&mv)[7];
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVILineB(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int wstep = src.cols;
		const int hmax = src.cols - 1;
		const int vmax = src.rows - 1;

		if (opt == VECTOR_WITHOUT)
		{

			for (int i = 0; i < src.cols; i++)
			{
				Mat buff(src.rows, 1, CV_32F);
				float* b = buff.ptr<float>(0);

				//h filter
				if (i < r)
				{
					for (int j = 0; j < src.rows; j++)
					{
						float* s = src.ptr<float>(j);
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = border_s(i + k - r);
							v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
						}
						b[j] = v;
					}
				}
				else if (i > src.cols - 1 - r)
				{
					for (int j = 0; j < src.rows; j++)
					{
						float* s = src.ptr<float>(j);
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = border_e(i + k - r, hmax);
							v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
						}
						b[j] = v;
					}
				}
				else
				{
					for (int j = 0; j < src.rows; j++)
					{
						float* s = src.ptr<float>(j);
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = i + k - r;
							v += gauss32F[k] * s[idx];
						}
						b[j] = v;
					}
				}

				// v filter
				for (int j = 0; j < r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(j + k - r);
						v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int j = r; j < src.rows - r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = j + k - r;
						v += gauss32F[k] * b[idx];
					}
					d[i] = v;
				}
				for (int j = src.rows - r; j < src.rows; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(j + k - r, vmax);
						v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			__m256i midx = _mm256_setr_epi32(0 * wstep, 1 * wstep, 2 * wstep, 3 * wstep, 4 * wstep, 5 * wstep, 6 * wstep, 7 * wstep);
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.rows - 7 + i, vmax),
					border_e(src.rows - 6 + i, vmax),
					border_e(src.rows - 5 + i, vmax),
					border_e(src.rows - 4 + i, vmax),
					border_e(src.rows - 3 + i, vmax),
					border_e(src.rows - 2 + i, vmax),
					border_e(src.rows - 1 + i, vmax),
					border_e(src.rows - 0 + i, vmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			//h filter
			const int max_core = 1;
			const int wstep0 = 0 * wstep;
			const int wstep1 = 1 * wstep;
			const int wstep2 = 2 * wstep;
			const int wstep3 = 3 * wstep;
			const int wstep4 = 4 * wstep;
			const int wstep5 = 5 * wstep;
			const int wstep6 = 6 * wstep;
			const int wstep7 = 7 * wstep;
			const int vstep = wstep * 8;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;
				const int R = get_simd_ceil(r, 8);
				const int simdsize = get_simd_ceil(src.rows + 2 * R, 8);
				if (!useAllocBuffer)bufferLineRows[tidx].release();
				if (bufferLineRows[tidx].size().area() != simdsize)bufferLineRows[tidx].create(simdsize, 1, CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0) + R;
				for (int i = start; i < end; i++)
				{
					//h filter
					if (i < r)
					{
						float* s = src.ptr<float>(0);
						for (int j = 0; j < src.rows; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
							if (border == BORDER_CONSTANT)
							{
								for (int k = 0; k < r - i; k++)
								{
									int idx = border_s(i + k - r);
									__m256 ms;
									if (idx >= 0) ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
									else ms = mVal;
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
								for (int k = r - i; k < ksize; k++)
								{
									int idx = border_s(i + k - r);
									__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
							}
							else
#endif
							{
								int idx_ = i - r;
								for (int k = 0; k < ksize; k++)
								{
									int idx = border_s(idx_);
									__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));//vload
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
									idx_++;
								}
							}
							_mm256_store_ps(b + j, mv);
							s += vstep;
						}
					}
					else if (i > src.cols - 1 - r)
					{
						float* s = src.ptr<float>(0);
						for (int j = 0; j < src.rows; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
							if (border == BORDER_CONSTANT)
							{
								for (int k = 0; k < src.cols - 7 + r - i; k++)
								{
									int idx = i + k - r;
									__m256 ms = ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}

								for (int k = src.cols - 7 + r - i; k < ksize; k++)
								{
									int idx = border_e(i + k - r, hmax);
									__m256 ms;
									if (idx >= 0) ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
									else ms = mVal;
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
							}
							else
#endif
							{
								for (int k = 0; k < ksize; k++)
								{
									int idx = border_e(i + k - r, hmax);
									__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));//vload
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
							}
							_mm256_store_ps(b + j, mv);
							s += vstep;
						}
					}
					else //mid
					{
						float* s = src.ptr<float>(0);
						for (int j = 0; j < src.rows; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							int idx = i - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));//vload
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							_mm256_store_ps(b + j, mv);
							s += vstep;
						}
					}
					//border
					for (int i = 0; i < R; i += 8)
					{
						__m256 a = _mm256_load_ps(b + i);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(b - i - 8, a);

						a = _mm256_load_ps(b + src.rows - 8 - i);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(b + src.rows + i, a);
					}

					//v filter					
					float* d = dest.ptr<float>(0);
					float* br = b - r;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bj = br + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bj);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bj++;
						}
						d[i + wstep0] = ((float*)&mv)[0];
						d[i + wstep1] = ((float*)&mv)[1];
						d[i + wstep2] = ((float*)&mv)[2];
						d[i + wstep3] = ((float*)&mv)[3];
						d[i + wstep4] = ((float*)&mv)[4];
						d[i + wstep5] = ((float*)&mv)[5];
						d[i + wstep6] = ((float*)&mv)[6];
						d[i + wstep7] = ((float*)&mv)[7];
						d += vstep;
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVIImage(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int wstep = src.cols;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)buffer.release();
		if (buffer.size() != src.size())buffer.create(Size(src.rows + 8, src.cols), src.type());

		if (opt == VECTOR_WITHOUT)
		{

			for (int i = 0; i < src.cols; i++)
			{
				float* b = buffer.ptr<float>(0) + i;
				//h filter
				if (i < r)
				{
					for (int j = 0; j < src.rows; j++)
					{
						float* s = src.ptr<float>(j);
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = border_s(i + k - r);
							v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
						}
						b[wstep * j] = v;
					}
				}
				else if (i > src.cols - 1 - r)
				{
					for (int j = 0; j < src.rows; j++)
					{
						float* s = src.ptr<float>(j);
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = border_e(i + k - r, hmax);
							v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
						}
						b[wstep * j] = v;
					}
				}
				else
				{
					for (int j = 0; j < src.rows; j++)
					{
						float* s = src.ptr<float>(j);
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = i + k - r;
							v += gauss32F[k] * s[idx];
						}
						b[wstep * j] = v;
					}
				}

				// v filter
				for (int j = 0; j < r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_s(j + k - r);
						v += (idx >= 0) ? gauss32F[k] * b[idx * wstep] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
				for (int j = r; j < src.rows - r; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = j + k - r;
						v += gauss32F[k] * b[idx * wstep];
					}
					d[i] = v;
				}
				for (int j = src.rows - r; j < src.rows; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						int idx = border_e(j + k - r, vmax);
						v += (idx >= 0) ? gauss32F[k] * b[idx * wstep] : gauss32F[k] * constVal;
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			__m256i midx = _mm256_setr_epi32(0 * wstep, 1 * wstep, 2 * wstep, 3 * wstep, 4 * wstep, 5 * wstep, 6 * wstep, 7 * wstep);

			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.rows - 7 + i, vmax),
					border_e(src.rows - 6 + i, vmax),
					border_e(src.rows - 5 + i, vmax),
					border_e(src.rows - 4 + i, vmax),
					border_e(src.rows - 3 + i, vmax),
					border_e(src.rows - 2 + i, vmax),
					border_e(src.rows - 1 + i, vmax),
					border_e(src.rows - 0 + i, vmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif

			for (int i = 0; i < src.cols; i++)
			{
				float* b = buffer.ptr<float>(i);
				if (i < r)
				{
					for (int j = 0; j < src.rows; j += 8)
					{
						float* s = src.ptr<float>(j);
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - i; k++)
							{
								int idx = border_s(i + k - r);
								__m256 ms;
								if (idx >= 0) ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							for (int k = r - i; k < ksize; k++)
							{
								int idx = border_s(i + k - r);
								__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}

						}
						else
#endif
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(i + k - r);
								__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(b + j, mv);
					}
				}
				else if (i > src.cols - 1 - r)
				{
					for (int j = 0; j < src.rows; j += 8)
					{
						float* s = src.ptr<float>(j);
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < src.cols - 7 + r - i; k++)
							{
								int idx = i + k - r;
								__m256 ms = ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}

							for (int k = src.cols - 7 + r - i; k < ksize; k++)
							{
								int idx = border_e(i + k - r, hmax);
								__m256 ms;
								if (idx >= 0) ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(i + k - r, hmax);
								__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(b + j, mv);
					}
				}
				else
				{
					for (int j = 0; j < src.rows; j += 8)
					{
						float* s = src.ptr<float>(j);
						__m256 mv = _mm256_setzero_ps();
						for (int k = 0; k < ksize; k++)
						{
							int idx = i + k - r;
							__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);

						}
						_mm256_store_ps(b + j, mv);
					}
				}
				//v filter
				const int R = get_simd_ceil(r, 8);
				const int simdend = get_simd_floor_end(R, src.rows - r, 8);
				const int wstep0 = 0 * wstep;
				const int wstep1 = 1 * wstep;
				const int wstep2 = 2 * wstep;
				const int wstep3 = 3 * wstep;
				const int wstep4 = 4 * wstep;
				const int wstep5 = 5 * wstep;
				const int wstep6 = 6 * wstep;
				const int wstep7 = 7 * wstep;
				for (int j = 0; j < r; j += 8)
				{
					float* d = dest.ptr<float>(j);
					__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
					if (border == BORDER_CONSTANT)
					{
						for (int k = 0; k < r - j; k++)
						{
							int idx = j + k;
							int maskIdx = max(0, k + j - r + 8);
							__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
						}
					}
					else
#endif
					{
						int idx = j;
						for (int k = 0; k < r - j; k++)
						{
							__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
					}
					float* bi = b;
					for (int k = r - j; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(bi);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						bi++;
					}
					d[i + wstep0] = ((float*)&mv)[0];
					d[i + wstep1] = ((float*)&mv)[1];
					d[i + wstep2] = ((float*)&mv)[2];
					d[i + wstep3] = ((float*)&mv)[3];
					d[i + wstep4] = ((float*)&mv)[4];
					d[i + wstep5] = ((float*)&mv)[5];
					d[i + wstep6] = ((float*)&mv)[6];
					d[i + wstep7] = ((float*)&mv)[7];
				}

				for (int j = R; j < simdend; j += 8)
				{
					float* d = dest.ptr<float>(j);
					__m256 mv = _mm256_setzero_ps();
					float* bi = b + j - r;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(bi);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						bi++;
					}
					d[i + wstep0] = ((float*)&mv)[0];
					d[i + wstep1] = ((float*)&mv)[1];
					d[i + wstep2] = ((float*)&mv)[2];
					d[i + wstep3] = ((float*)&mv)[3];
					d[i + wstep4] = ((float*)&mv)[4];
					d[i + wstep5] = ((float*)&mv)[5];
					d[i + wstep6] = ((float*)&mv)[6];
					d[i + wstep7] = ((float*)&mv)[7];
				}
				for (int j = simdend; j < src.rows; j += 8)
				{
					float* d = dest.ptr<float>(j);
					__m256 mv = _mm256_setzero_ps();
					int e = src.rows - (j + 8);
					float* bi = b + j - r;
					for (int k = 0; k < r + 1 + e; k++)
					{
						__m256 ms = _mm256_loadu_ps(bi);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						bi++;
					}
					int idx = 0;
					for (int k = r + 1 + e; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						idx++;
					}
					d[i + wstep0] = ((float*)&mv)[0];
					d[i + wstep1] = ((float*)&mv)[1];
					d[i + wstep2] = ((float*)&mv)[2];
					d[i + wstep3] = ((float*)&mv)[3];
					d[i + wstep4] = ((float*)&mv)[4];
					d[i + wstep5] = ((float*)&mv)[5];
					d[i + wstep6] = ((float*)&mv)[6];
					d[i + wstep7] = ((float*)&mv)[7];
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	//VHI filtering
	void GaussianFilterSeparableFIR::filterVHILine(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const float* s = src.ptr<float>(0);

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(Size(dest.cols, 1), CV_32F);
				float* b = buff.ptr<float>(0);

				if (j < r)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}
					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else if (j > src.rows - r - 1)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						const float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * si[idx * wstep];
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}
#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int wstep = src.cols;
			const int max_core = 1;
			float* s = src.ptr<float>(0);
			const int R = get_simd_ceil(r, 8);
			const int simdend = get_simd_floor_end(R, src.cols - r, 8);

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int simdwidth = get_simd_ceil(src.cols, 8);
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);

				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;
				for (int j = start; j < Y0; j++)
				{
					//v filter
					for (int i = 0; i < src.cols; i += 8)
					{
						float* si = s + i;
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = j + k - r;
								idx = border_s(idx) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = j + r;
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_loadu_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + wstep * (j + e + 1 - r);
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
						}
						_mm256_store_ps(b + i, mv);
					}
					//h filter
					{
						float* d = dest.ptr<float>(j);

						for (int i = 0; i < R; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
							if (border == BORDER_CONSTANT)
							{
								for (int k = 0; k < r - i; k++)
								{
									int idx = i + k;
									int maskIdx = max(0, k + i - r + 8);
									__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
							}
							else
#endif
							{
								int idx = i;
								for (int k = 0; k < r - i; k++)
								{
									__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
									idx++;
								}
							}
							float* bi = b;
							for (int k = r - i; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}

						for (int i = R; i < simdend; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* bi = b + i - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
						for (int i = simdend; i < src.cols; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							int e = src.cols - (i + 8);
							float* bi = b + i - r;
							for (int k = 0; k < r + 1 + e; k++)
							{
								__m256 ms = _mm256_load_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							int idx = 0;
							for (int k = r + 1 + e; k < ksize; k++)
							{
								__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							_mm256_store_ps(d + i, mv);
						}
					}
				}
				for (int j = start + Y0; j < end - Y1; j++)
				{
					//v filter
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const float* si = s + i + (j - r) * wstep;

						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += wstep;
						}
						_mm256_store_ps(b + i, mv);
					}

					//h filter
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < R; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - i; k++)
							{
								int idx = i + k;
								int maskIdx = max(0, k + i - r + 8);
								__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							int idx = i;
							for (int k = 0; k < r - i; k++)
							{
								__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
						}
						float* bi = b;
						for (int k = r - i; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = R; i < simdend; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = b + i - r;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = simdend; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const int e = src.cols - (i + 8);
						float* bi = b + i - r;
						for (int k = 0; k < r + 1 + e; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						int idx = 0;
						for (int k = r + 1 + e; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}
				for (int j = end - Y1; j < end; j++)
				{
					//v filter
					for (int i = 0; i < src.cols; i += 8)
					{
						float* si = s + i;
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = j + k - r;
								idx = border_e(idx, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = -(src.rows - j) + r;
							float* sii = si + (j - r) * wstep;
							for (int k = 0; k < e + 1; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							for (int k = e + 1; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(b + i, mv);
					}
					//h filter
					float* d = dest.ptr<float>(j);
					for (int i = 0; i < R; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < r - i; k++)
							{
								int idx = i + k;
								int maskIdx = max(0, k + i - r + 8);
								__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							int idx = i;
							for (int k = 0; k < r - i; k++)
							{
								__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
						}
						float* bi = b;
						for (int k = r - i; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = R; i < simdend; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = b + i - r;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					for (int i = simdend; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						int e = src.cols - (i + 8);
						float* bi = b + i - r;
						for (int k = 0; k < r + 1 + e; k++)
						{
							__m256 ms = _mm256_load_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						int idx = 0;
						for (int k = r + 1 + e; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							idx++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

#if 0
	void GaussianFilterSeparableFIR::filterVHILineB(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		myCopyMakeBorder(src, buffer, 0, 0, 0, 8, border);
		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const float* s = src.ptr<float>(0);

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(Size(dest.cols, 1), CV_32F);
				float* b = buff.ptr<float>(0);

				if (j < r)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss[k] * si[idx * wstep] : gauss[k] * constVal;
						}
						b[i] = v;
					}
					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss[k] * b[idx] : gauss[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss[k] * b[idx] : gauss[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else if (j > src.rows - r - 1)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss[k] * si[idx * wstep] : gauss[k] * constVal;
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss[k] * b[idx] : gauss[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss[k] * b[idx] : gauss[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						const float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss[k] * si[idx * wstep];
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss[k] * b[idx] : gauss[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss[k] * b[idx] : gauss[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int sstep = src.cols;
			const int bstep = buffer.cols;
			const int max_core = 1;
			float* s = buffer.ptr<float>(0);
			const int R = get_simd_ceil(r, 8);
			const int simdwidth = get_simd_ceil(src.cols, 8) * 2 * R;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);

				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;
				for (int j = start; j < Y0; j++)
				{
					//v filter
					const int e = j + r;
					const int siistep = bstep * (j + e + 1 - r);
					for (int i = 0; i < src.cols; i += 8)
					{
						float* si = s + i;
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = j + k - r;
								idx = border_s(idx) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * bstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + siistep;
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += bstep;
							}
						}
						_mm256_store_ps(b + i + R, mv);
					}
					//h filter
					{
#ifdef  BORDER_REFLECT_FILTER
						float* d = b + R;
						for (int i = 0; i < R; i += 8)
						{
							__m256 a = _mm256_load_ps(d + i);
							a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
							a = _mm256_permute2f128_ps(a, a, 1);
							_mm256_store_ps(d - i - 8, a);

							a = _mm256_load_ps(d + src.cols - 8 - i);
							a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
							a = _mm256_permute2f128_ps(a, a, 1);
							_mm256_store_ps(d + src.cols + i, a);
						}
#endif
						d = dest.ptr<float>(j);
						float* bptr = b + R - r;
						for (int i = 0; i < src.cols; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* bi = bptr + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
					}
				}
				for (int j = start + Y0; j < end - Y1; j++)
				{
					//v filter
					float* sptr = s + (j - r) * bstep;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const float* si = sptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += bstep;
						}
						_mm256_store_ps(b + i + R, mv);
					}

					//h filter
#ifdef  BORDER_REFLECT_FILTER
					float* d = b + R;
					for (int i = 0; i < R; i += 8)
					{
						__m256 a = _mm256_load_ps(d + i);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(d - i - 8, a);

						a = _mm256_load_ps(d + src.cols - 8 - i);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(d + src.cols + i, a);
					}
#endif
					d = dest.ptr<float>(j);
					float* bptr = b + R - r;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}
				for (int j = end - Y1; j < end; j++)
				{
					//v filter
					const int siistep = (j - r) * bstep;
					const int e = -(src.rows - j) + r;
					for (int i = 0; i < src.cols; i += 8)
					{
						float* si = s + i;
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = j + k - r;
								idx = border_e(idx, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							float* sii = si + siistep;
							for (int k = 0; k < e + 1; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += bstep;
							}
							for (int k = e + 1; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * bstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(b + i + R, mv);
					}
					//h filter
#ifdef  BORDER_REFLECT_FILTER
					float* d = b + R;
					for (int i = 0; i < R; i += 8)
					{
						__m256 a = _mm256_load_ps(d + i);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(d - i - 8, a);

						a = _mm256_load_ps(d + src.cols - 8 - i);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(d + src.cols + i, a);
					}
#endif
					d = dest.ptr<float>(j);
					float* bptr = b + R - r;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}
			}
		}
	}
#endif

	//modified
	void GaussianFilterSeparableFIR::filterVHILineB(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data == dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const float* s = src.ptr<float>(0);

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(Size(dest.cols, 1), CV_32F);
				float* b = buff.ptr<float>(0);

				if (j < r)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}
					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else if (j > src.rows - r - 1)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						const float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * si[idx * wstep];
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int wstep = src.cols;

			const float* s = src.ptr<float>();
			const int R = get_simd_ceil(r, 8);
			const int simdwidth = get_simd_ceil(src.cols, 16) * 2 * R;
			{
				AutoBuffer<float> bufferLineCols(simdwidth);
				//const int tidx = 0;
				//if (!useAllocBuffer)bufferLineCols[tidx].release();
				//bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
				float* linebuffer = &bufferLineCols[0];

				const int start = 0;
				const int end = src.rows;
				const int Y0 = (start < r) ? r - start : 0;
				const int Y1 = (end > src.rows - r) ? r + end - src.rows : 0;

				for (int j = start; j < Y0; j++)
				{
					//v filter
					const int e = j + r;
					const int siistep = wstep * (j + e + 1 - r);
					for (int i = 0; i < src.cols; i += 16)
					{
						const float* si = s + i;
						__m256 mv = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = j + k - r;
								idx = border_s(idx) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 ms1 = _mm256_load_ps(si + 8 + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
							}
							const float* sii = si + siistep;
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 ms1 = _mm256_load_ps(sii + 8);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
								sii += wstep;
							}
						}
						_mm256_store_ps(linebuffer + i + R, mv);
						_mm256_store_ps(linebuffer + i + 8 + R, mv1);
					}
					//h filter
					{
#ifdef  BORDER_REFLECT_FILTER
						float* d = linebuffer + R;
						for (int i = 0; i < R; i += 8)
						{
							__m256 a = _mm256_load_ps(d + i);
							a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
							a = _mm256_permute2f128_ps(a, a, 1);
							_mm256_store_ps(d - i - 8, a);

							a = _mm256_load_ps(d + src.cols - 8 - i);
							a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
							a = _mm256_permute2f128_ps(a, a, 1);
							_mm256_store_ps(d + src.cols + i, a);
						}
#endif		
						d = dest.ptr<float>(j);
						float* bptr = linebuffer + R - r;
						for (int i = 0; i < src.cols; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* bi = bptr + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
					}
				}

				for (int j = start + Y0; j < end - Y1; j++)
				{
					//v filter
					const float* sptr = s + (j - r) * wstep;
					for (int i = 0; i < src.cols; i += 16)
					{
						__m256 mv = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						const float* si = sptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 ms1 = _mm256_load_ps(si + 8);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
							si += wstep;
						}
						_mm256_store_ps(linebuffer + i + 0 + R, mv);
						_mm256_store_ps(linebuffer + i + 8 + R, mv1);
					}

					//h filter
#ifdef  BORDER_REFLECT_FILTER
					float* d = linebuffer + R;
					for (int i = 0; i < R; i += 8)
					{
						__m256 a = _mm256_load_ps(d + i);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(d - i - 8, a);

						a = _mm256_load_ps(d + src.cols - 8 - i);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(d + src.cols + i, a);
					}
#endif	
					d = dest.ptr<float>(j);
					float* bptr = linebuffer + R - r;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}

				for (int j = end - Y1; j < end; j++)
				{
					//v filter
					const int siistep = (j - r) * wstep;
					const int e = -(src.rows - j) + r;
					for (int i = 0; i < src.cols; i += 8)
					{
						const float* si = s + i;
						__m256 mv = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = j + k - r;
								idx = border_e(idx, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const float* sii = si + siistep;
							for (int k = 0; k < e + 1; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 ms1 = _mm256_load_ps(sii + 8);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
								sii += wstep;
							}
							for (int k = e + 1; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 ms1 = _mm256_load_ps(si + 8 + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
							}
						}
						_mm256_store_ps(linebuffer + i + R, mv);
						_mm256_store_ps(linebuffer + i + R + 8, mv1);
					}
					//h filter
#ifdef  BORDER_REFLECT_FILTER
					float* d = linebuffer + R;
					for (int i = 0; i < R; i += 8)
					{
						__m256 a = _mm256_load_ps(d + i);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(d - i - 8, a);

						a = _mm256_load_ps(d + src.cols - 8 - i);
						a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
						a = _mm256_permute2f128_ps(a, a, 1);
						_mm256_store_ps(d + src.cols + i, a);
					}
#endif
					d = dest.ptr<float>(j);
					float* bptr = linebuffer + R - r;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}

			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHIImage(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)buffer.release();
		if (buffer.size() != src.size())buffer.create(src.size(), src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const float* s = src.ptr<float>(0);

			for (int j = 0; j < src.rows; j++)
			{
				float* b = buffer.ptr<float>(j);

				if (j < r)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}
					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else if (j > src.rows - r - 1)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						const float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * si[idx * wstep];
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}
#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);
			const int R = get_simd_ceil(r, 8);
			const int simdend = get_simd_floor_end(R, src.cols - r, 8);

			for (int j = 0; j < src.rows; j++)
			{
				float* b = buffer.ptr<float>(j);
				if (j < r)
				{
					//v filter
					for (int i = 0; i < src.cols; i += 8)
					{
						float* si = s + i;
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = j + r;
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_loadu_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + wstep * (j + e + 1 - r);
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
						}
						_mm256_store_ps(b + i, mv);
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);

						for (int i = 0; i < r; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
							if (border == BORDER_CONSTANT)
							{
								for (int k = 0; k < r - i; k++)
								{
									int idx = i + k;
									int maskIdx = max(0, k + i - r + 8);
									__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
							}
							else
#endif
							{
								int idx = i;
								for (int k = 0; k < r - i; k++)
								{
									__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
									idx++;
								}
							}
							float* bi = b;
							for (int k = r - i; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
						for (int i = R; i < simdend; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* bi = b + i - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
						for (int i = simdend; i < src.cols; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							int e = src.cols - (i + 8);
							float* bi = b + i - r;
							for (int k = 0; k < r + 1 + e; k++)
							{
								__m256 ms = _mm256_load_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							int idx = 0;
							for (int k = r + 1 + e; k < ksize; k++)
							{
								__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							_mm256_store_ps(d + i, mv);
						}
					}
				}
				else if (j > src.rows - r - 1)
				{
					//v filter
					for (int i = 0; i < src.cols; i += 8)
					{
						float* si = s + i;
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = -(src.rows - j) + r;
							float* sii = si + (j - r) * wstep;
							for (int k = 0; k < e + 1; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							for (int k = e + 1; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(b + i, mv);
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
							if (border == BORDER_CONSTANT)
							{
								for (int k = 0; k < r - i; k++)
								{
									int idx = i + k;
									int maskIdx = max(0, k + i - r + 8);
									__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
							}
							else
#endif
							{
								int idx = i;
								for (int k = 0; k < r - i; k++)
								{
									__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
									idx++;
								}
							}
							float* bi = b;
							for (int k = r - i; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
						for (int i = R; i < simdend; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* bi = b + i - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
						for (int i = simdend; i < src.cols; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							int e = src.cols - (i + 8);
							float* bi = b + i - r;
							for (int k = 0; k < r + 1 + e; k++)
							{
								__m256 ms = _mm256_load_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							int idx = 0;
							for (int k = r + 1 + e; k < ksize; k++)
							{
								__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							_mm256_store_ps(d + i, mv);
						}
					}
				}
				else
				{
					//v filter
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const float* si = s + i;

						for (int k = 0; k < ksize; k++)
						{
							int idx = (j + k - r) * wstep;
							__m256 ms = _mm256_loadu_ps(si + idx);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
						}
						_mm256_store_ps(b + i, mv);
					}
					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
							if (border == BORDER_CONSTANT)
							{
								for (int k = 0; k < r - i; k++)
								{
									int idx = i + k;
									int maskIdx = max(0, k + i - r + 8);
									__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
							}
							else
#endif
							{
								int idx = i;
								for (int k = 0; k < r - i; k++)
								{
									__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
									idx++;
								}
							}
							float* bi = b;
							for (int k = r - i; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
						for (int i = R; i < simdend; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* bi = b + i - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
						for (int i = simdend; i < src.cols; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							const int e = src.cols - (i + 8);
							float* bi = b + i - r;
							for (int k = 0; k < r + 1 + e; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							int idx = 0;
							for (int k = r + 1 + e; k < ksize; k++)
							{
								__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							_mm256_store_ps(d + i, mv);
						}
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterVHIImageBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)bufferImageBorder.release();
		const int R = get_simd_ceil(r, 8);
		Size asize = Size(src.cols + 2 * R, src.rows);
		if (bufferImageBorder.size() != asize)bufferImageBorder.create(asize, src.type());

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const float* s = src.ptr<float>(0);

			for (int j = 0; j < src.rows; j++)
			{
				float* b = buffer.ptr<float>(j);

				if (j < r)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}
					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else if (j > src.rows - r - 1)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						const float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * si[idx * wstep];
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}
#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int wstep = src.cols;
			float* s = src.ptr<float>(0);

			for (int j = 0; j < src.rows; j++)
			{
				float* b = bufferImageBorder.ptr<float>(j) + R;
				if (j < r)
				{
					//v filter
					for (int i = 0; i < src.cols; i += 8)
					{
						float* si = s + i;
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = j + r;
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_loadu_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + wstep * (j + e + 1 - r);
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
						}
						_mm256_store_ps(b + i, mv);
					}

					//h filter
					float* d = dest.ptr<float>(j);
					float* bptr = b - r;
					copyMakeBorderLineWithoutBodyCopy(b, b - R, src.cols, R, R, border);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}
				else if (j > src.rows - r - 1)
				{
					//v filter
					for (int i = 0; i < src.cols; i += 8)
					{
						float* si = s + i;
						__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
						if (border == BORDER_CONSTANT)
						{
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms;
								if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
								else ms = mVal;
								__m256 mg = _mm256_set1_ps(gauss[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						else
#endif
						{
							const int e = -(src.rows - j) + r;
							float* sii = si + (j - r) * wstep;
							for (int k = 0; k < e + 1; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							for (int k = e + 1; k < ksize; k++)
							{
								int idx = border_e(j + k - r, vmax) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
						}
						_mm256_store_ps(b + i, mv);
					}
					//h filter
					float* d = dest.ptr<float>(j);
					float* bptr = b - r;
					copyMakeBorderLineWithoutBodyCopy(b, b - R, src.cols, R, R, border);
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}
				else
				{
					//v filter
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const float* si = s + i;

						for (int k = 0; k < ksize; k++)
						{
							int idx = (j + k - r) * wstep;
							__m256 ms = _mm256_loadu_ps(si + idx);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
						}
						_mm256_store_ps(b + i, mv);
					}
					//h filter
					float* d = dest.ptr<float>(j);
					copyMakeBorderLineWithoutBodyCopy(b, b - R, src.cols, R, R, border);
					float* bptr = b - r;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterVHIImageBVBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		if (!useAllocBuffer)bufferImageBorder.release();

		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, 0, 16, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, 0, 16, border);

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const float* s = src.ptr<float>(0);

			for (int j = 0; j < src.rows; j++)
			{
				float* b = buffer.ptr<float>(j);

				if (j < r)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}
					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else if (j > src.rows - r - 1)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						const float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * si[idx * wstep];
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int R = get_simd_ceil(r, 8);
			//const int sstep = src.cols;
			const int bstep = bufferImageBorder.cols;
			const int max_core = 1;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer) bufferLineCols[tidx].release();
				if (bufferLineCols[tidx].size() != Size(src.cols + 2 * R, 1)) bufferLineCols[tidx].create(src.cols + 2 * R, 1, CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0) + R;

				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				float* s = bufferImageBorder.ptr<float>(start);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					//v filter
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						const float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += bstep;
						}
						_mm256_store_ps(b + i, mv);
					}
					//h filter
					copyMakeBorderLineWithoutBodyCopy(b, b - R, src.cols, R, R, border);
					float* bptr = b - r;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = bptr + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += bufferImageBorder.cols;
					d += dest.cols;
				}
			}
		}
	}

	//HV Tile Filtering
	void GaussianFilterSeparableFIR::filterHVTileImage(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		//const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int max_core = 1;

		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (opt == VECTOR_WITHOUT)
		{
			const int tstep = TILE_X;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(TILE_X, TILE_Y + 2 * r)) bufferTile[tidx].create(Size(TILE_X, TILE_Y + 2 * r), CV_32F);
				float* blurx = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					const int sbottom = r - bottom;

					//h filter
					for (int j = -stop; j < TILE_Y + sbottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + tstep * (j + r);

						for (int i = 0; i < LEFT; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(i + k - r);
								v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}

						for (int i = LEFT; i < TILE_X - RIGHT; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * s[idx];
							}
							d[i] = v;
						}

						for (int i = TILE_X - RIGHT; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(i + tilex + k - r, hmax);
								v += (idx >= 0) ? gauss32F[k] * s[idx - tilex] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}

					if (tiley == 0)
					{
						//upper border
#ifdef  BORDER_REFLECT_FILTER
						for (int i = 0; i < top; i++)
						{
							const int v = min(r + i, TILE_Y + 2 * r - 1);
							//const int v = r + i;
							memcpy(blurx + (r - 1 - i) * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					if (tiley == src.rows - TILE_Y)
					{
#ifdef  BORDER_REFLECT_FILTER
						//downer border
						for (int i = TILE_Y + r; i < TILE_Y + 2 * r; i++)
						{
							const int v = max(2 * (TILE_Y + r) - i - 1, 0);
							memcpy(blurx + i * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					// v filter
					for (int j = 0; j < TILE_Y; j++)
					{
						float* s = blurx + j * tstep;
						float* d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * si[0];
								si += tstep;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				const int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif

			const int tstep = TILE_X;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(TILE_X, TILE_Y + 2 * r)) bufferTile[tidx].create(Size(TILE_X, TILE_Y + 2 * r), CV_32F);
				float* blurx = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					const int sbottom = r - bottom;
					//h filter
					for (int j = -stop; j < TILE_Y + sbottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + tstep * (j + r);

						for (int i = 0; i < LEFT; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							int idx = i;
							for (int k = 0; k < r - i; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							float* si = s;
							for (int k = r - i; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							_mm256_store_ps(d + i, mv);
						}

						for (int i = LEFT; i < TILE_X - RIGHT; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							_mm256_store_ps(d + i, mv);
						}

						for (int i = TILE_X - RIGHT; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();

							const int e = src.cols - (i + tilex + 8);
							float* si = s + i - r;
							for (int k = 0; k < r + 1 + e; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}

							int idx = 0;
							for (int k = r + 1 + e; k < ksize; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s - tilex, end_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							_mm256_store_ps(d + i, mv);
						}
					}

					if (tiley == 0)
					{
						//upper border
#ifdef  BORDER_REFLECT_FILTER
						for (int i = 0; i < top; i++)
						{
							const int v = min(r + i, TILE_Y + 2 * r - 1);
							//const int v = r + i;
							memcpy(blurx + (r - 1 - i) * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					if (tiley == src.rows - TILE_Y)
					{
#ifdef  BORDER_REFLECT_FILTER
						//downer border
						for (int i = TILE_Y + r; i < TILE_Y + 2 * r; i++)
						{
							const int v = max(2 * (TILE_Y + r) - i - 1, 0);
							memcpy(blurx + i * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}

					// v filter
					for (int j = 0; j < TILE_Y; j++)
					{
						float* s = blurx + j * tstep;
						float* d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += tstep;
							}
							_mm256_store_ps(d + i, mv);
						}
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVTileImageBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		//const int vmax = src.rows - 1;
		//const int hmax = src.cols - 1;
		const int max_core = 1;

		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (!useAllocBuffer)bufferImageBorder.release();
		const int R = get_simd_ceil(r, 8);
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, 0, 0, R, R, border);
		else copyMakeBorder(src, bufferImageBorder, 0, 0, R, R, border);

		if (opt == VECTOR_WITHOUT)
		{
			;
		}
		else if (opt == VECTOR_AVX)
		{
			const int tstep = TILE_X;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(TILE_X, TILE_Y + 2 * r)) bufferTile[tidx].create(Size(TILE_X, TILE_Y + 2 * r), CV_32F);
				float* blurx = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;

					const int top = max(0, r - tiley);
					const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					const int sbottom = r - bottom;
					//h filter
					float* s = bufferImageBorder.ptr<float>(tiley - stop) + tilex + R - r;
					float* d = blurx + tstep * (-stop + r);
					for (int j = -stop; j < TILE_Y + sbottom; j++)
					{
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							_mm256_store_ps(d + i, mv);
						}
						s += bufferImageBorder.cols;
						d += tstep;
					}

					if (tiley == 0)
					{
						//upper border
#ifdef  BORDER_REFLECT_FILTER
						for (int i = 0; i < top; i++)
						{
							const int v = min(r + i, TILE_Y + 2 * r - 1);
							//const int v = r + i;
							memcpy(blurx + (r - 1 - i) * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					if (tiley == src.rows - TILE_Y)
					{
#ifdef  BORDER_REFLECT_FILTER
						//downer border
						for (int i = TILE_Y + r; i < TILE_Y + 2 * r; i++)
						{
							const int v = max(2 * (TILE_Y + r) - i - 1, 0);
							memcpy(blurx + i * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					// v filter
					s = blurx;
					d = dest.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						for (int i = 0; i < TILE_X; i += 16)
						{
							__m256 mv = _mm256_setzero_ps();
							__m256 mv1 = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(si);
								__m256 ms1 = _mm256_load_ps(si + 8);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								mv1 = _mm256_fmadd_ps(ms1, mg, mv1);
								si += tstep;
							}
							_mm256_store_ps(d + i, mv);
							_mm256_store_ps(d + i + 8, mv1);
						}
						s += tstep;
						d += dest.cols;
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVTileImageTr(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		//const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int max_core = 1;

		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (opt == VECTOR_WITHOUT)
		{
			const int tstep = TILE_X;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(TILE_X, TILE_Y + 2 * r)) bufferTile[tidx].create(Size(TILE_X, TILE_Y + 2 * r), CV_32F);
				float* blurx = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					const int sbottom = r - bottom;

					//h filter
					for (int j = -stop; j < TILE_Y + sbottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + tstep * (j + r);

						for (int i = 0; i < LEFT; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(i + k - r);
								v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}

						for (int i = LEFT; i < TILE_X - RIGHT; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * s[idx];
							}
							d[i] = v;
						}

						for (int i = TILE_X - RIGHT; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(i + tilex + k - r, hmax);
								v += (idx >= 0) ? gauss32F[k] * s[idx - tilex] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}

					if (tiley == 0)
					{
						//upper border
#ifdef  BORDER_REFLECT_FILTER
						for (int i = 0; i < top; i++)
						{
							const int v = min(r + i, TILE_Y + 2 * r - 1);
							//const int v = r + i;
							memcpy(blurx + (r - 1 - i) * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					if (tiley == src.rows - TILE_Y)
					{
#ifdef  BORDER_REFLECT_FILTER
						//downer border
						for (int i = TILE_Y + r; i < TILE_Y + 2 * r; i++)
						{
							const int v = max(2 * (TILE_Y + r) - i - 1, 0);
							memcpy(blurx + i * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					// v filter
					for (int j = 0; j < TILE_Y; j++)
					{
						float* s = blurx + j * tstep;
						float* d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * si[0];
								si += tstep;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				const int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif

			const int R = get_simd_ceil(r, 8);
			const int tstep = TILE_Y + 2 * R;
			const int tstep0 = tstep * 0;
			const int tstep1 = tstep * 1;
			const int tstep2 = tstep * 2;
			const int tstep3 = tstep * 3;
			const int tstep4 = tstep * 4;
			const int tstep5 = tstep * 5;
			const int tstep6 = tstep * 6;
			const int tstep7 = tstep * 7;
			const int tstep8 = tstep * 8;

			//const int wstep = src.cols;
			const int wstep0 = src.cols * 0;
			const int wstep1 = src.cols * 1;
			const int wstep2 = src.cols * 2;
			const int wstep3 = src.cols * 3;
			const int wstep4 = src.cols * 4;
			const int wstep5 = src.cols * 5;
			const int wstep6 = src.cols * 6;
			const int wstep7 = src.cols * 7;
			const int wstep8 = src.cols * 8;


			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;

				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(TILE_Y + 2 * R, TILE_X)) bufferTile[tidx].create(Size(TILE_Y + 2 * R, TILE_X), CV_32F);
				float* blurx = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, R - tiley);
					const int stop = R - top;
					const int bottom = max(0, R + (tiley + TILE_Y - src.rows));
					const int sbottom = R - bottom;
					//h filter
					for (int j = -stop; j < TILE_Y + sbottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + R + j;
						for (int i = 0; i < LEFT; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							int idx = i;
							for (int k = 0; k < r - i; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							float* si = s;
							for (int k = r - i; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							d[tstep0] = ((float*)&mv)[0];
							d[tstep1] = ((float*)&mv)[1];
							d[tstep2] = ((float*)&mv)[2];
							d[tstep3] = ((float*)&mv)[3];
							d[tstep4] = ((float*)&mv)[4];
							d[tstep5] = ((float*)&mv)[5];
							d[tstep6] = ((float*)&mv)[6];
							d[tstep7] = ((float*)&mv)[7];
							d += tstep8;
						}

						for (int i = LEFT; i < TILE_X - RIGHT; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							d[tstep0] = ((float*)&mv)[0];
							d[tstep1] = ((float*)&mv)[1];
							d[tstep2] = ((float*)&mv)[2];
							d[tstep3] = ((float*)&mv)[3];
							d[tstep4] = ((float*)&mv)[4];
							d[tstep5] = ((float*)&mv)[5];
							d[tstep6] = ((float*)&mv)[6];
							d[tstep7] = ((float*)&mv)[7];
							d += tstep8;
						}

						for (int i = TILE_X - RIGHT; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							const int e = src.cols - (i + tilex + 8);
							float* si = s + i - r;
							for (int k = 0; k < r + 1 + e; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}

							int idx = 0;
							for (int k = r + 1 + e; k < ksize; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s - tilex, end_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							d[tstep0] = ((float*)&mv)[0];
							d[tstep1] = ((float*)&mv)[1];
							d[tstep2] = ((float*)&mv)[2];
							d[tstep3] = ((float*)&mv)[3];
							d[tstep4] = ((float*)&mv)[4];
							d[tstep5] = ((float*)&mv)[5];
							d[tstep6] = ((float*)&mv)[6];
							d[tstep7] = ((float*)&mv)[7];
							d += tstep8;
						}
					}

					// v filter
					float* dptr = dest.ptr<float>(tiley) + tilex;
					for (int i = 0; i < TILE_X; i++)
					{
						copyMakeBorderLineWithoutBodyCopy(blurx + tstep * i + top, blurx + tstep * i, TILE_Y + top + bottom, top, bottom, 0);
						float* d = dptr + i;
						float* bptr = blurx + tstep * i + R - r;
						for (int j = 0; j < TILE_Y; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* bj = bptr + j;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bj);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bj++;
							}
							d[wstep0] = ((float*)&mv)[0];
							d[wstep1] = ((float*)&mv)[1];
							d[wstep2] = ((float*)&mv)[2];
							d[wstep3] = ((float*)&mv)[3];
							d[wstep4] = ((float*)&mv)[4];
							d[wstep5] = ((float*)&mv)[5];
							d[wstep6] = ((float*)&mv)[6];
							d[wstep7] = ((float*)&mv)[7];
							d += wstep8;
						}
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVTileImageBHTr(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		//const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int max_core = 1;

		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (!useAllocBuffer)bufferImageBorder.release();
		const int R = get_simd_ceil(r, 8);
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, 0, 0, R, R, border);
		else copyMakeBorder(src, bufferImageBorder, 0, 0, R, R, border);

		if (opt == VECTOR_WITHOUT)
		{
			const int tstep = TILE_X;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(TILE_X, TILE_Y + 2 * r)) bufferTile[tidx].create(Size(TILE_X, TILE_Y + 2 * r), CV_32F);
				float* blurx = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					const int sbottom = r - bottom;

					//h filter
					for (int j = -stop; j < TILE_Y + sbottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + tstep * (j + r);

						for (int i = 0; i < LEFT; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(i + k - r);
								v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}

						for (int i = LEFT; i < TILE_X - RIGHT; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * s[idx];
							}
							d[i] = v;
						}

						for (int i = TILE_X - RIGHT; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(i + tilex + k - r, hmax);
								v += (idx >= 0) ? gauss32F[k] * s[idx - tilex] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}

					if (tiley == 0)
					{
						//upper border
#ifdef  BORDER_REFLECT_FILTER
						for (int i = 0; i < top; i++)
						{
							const int v = min(r + i, TILE_Y + 2 * r - 1);
							//const int v = r + i;
							memcpy(blurx + (r - 1 - i) * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					if (tiley == src.rows - TILE_Y)
					{
#ifdef  BORDER_REFLECT_FILTER
						//downer border
						for (int i = TILE_Y + r; i < TILE_Y + 2 * r; i++)
						{
							const int v = max(2 * (TILE_Y + r) - i - 1, 0);
							memcpy(blurx + i * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					// v filter
					for (int j = 0; j < TILE_Y; j++)
					{
						float* s = blurx + j * tstep;
						float* d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * si[0];
								si += tstep;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int R = get_simd_ceil(r, 8);
			const int tstep = TILE_Y + 2 * R;
			const int tstep0 = tstep * 0;
			const int tstep1 = tstep * 1;
			const int tstep2 = tstep * 2;
			const int tstep3 = tstep * 3;
			const int tstep4 = tstep * 4;
			const int tstep5 = tstep * 5;
			const int tstep6 = tstep * 6;
			const int tstep7 = tstep * 7;
			const int tstep8 = tstep * 8;

			//const int wstep = src.cols;
			const int wstep0 = src.cols * 0;
			const int wstep1 = src.cols * 1;
			const int wstep2 = src.cols * 2;
			const int wstep3 = src.cols * 3;
			const int wstep4 = src.cols * 4;
			const int wstep5 = src.cols * 5;
			const int wstep6 = src.cols * 6;
			const int wstep7 = src.cols * 7;
			const int wstep8 = src.cols * 8;


			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;

				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(TILE_Y + 2 * R, TILE_X)) bufferTile[tidx].create(Size(TILE_Y + 2 * R, TILE_X), CV_32F);
				float* blurx = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;

					const int top = max(0, R - tiley);
					const int stop = R - top;
					const int bottom = max(0, R + (tiley + TILE_Y - src.rows));
					const int sbottom = R - bottom;
					//h filter
					float* s = bufferImageBorder.ptr<float>(tiley - stop) + tilex + R - r;
					for (int j = -stop; j < TILE_Y + sbottom; j++)
					{
						float* d = blurx + R + j;
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							d[tstep0] = ((float*)&mv)[0];
							d[tstep1] = ((float*)&mv)[1];
							d[tstep2] = ((float*)&mv)[2];
							d[tstep3] = ((float*)&mv)[3];
							d[tstep4] = ((float*)&mv)[4];
							d[tstep5] = ((float*)&mv)[5];
							d[tstep6] = ((float*)&mv)[6];
							d[tstep7] = ((float*)&mv)[7];
							d += tstep8;
						}
						s += bufferImageBorder.cols;
					}

					// v filter
					float* dptr = dest.ptr<float>(tiley) + tilex;
					for (int i = 0; i < TILE_X; i++)
					{
						copyMakeBorderLineWithoutBodyCopy(blurx + tstep * i + top, blurx + tstep * i, TILE_Y + top + bottom, top, bottom, 0);
						float* d = dptr + i;
						float* bptr = blurx + tstep * i + R - r;
						for (int j = 0; j < TILE_Y; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* bj = bptr + j;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bj);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bj++;
							}
							d[wstep0] = ((float*)&mv)[0];
							d[wstep1] = ((float*)&mv)[1];
							d[wstep2] = ((float*)&mv)[2];
							d[wstep3] = ((float*)&mv)[3];
							d[wstep4] = ((float*)&mv)[4];
							d[wstep5] = ((float*)&mv)[5];
							d[wstep6] = ((float*)&mv)[6];
							d[wstep7] = ((float*)&mv)[7];
							d += wstep8;
						}
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVTileImageT2(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		//const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int max_core = 1;

		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (opt == VECTOR_WITHOUT)
		{
			const int tstep = TILE_X;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(TILE_X, TILE_Y + 2 * r)) bufferTile[tidx].create(Size(TILE_X, TILE_Y + 2 * r), CV_32F);
				float* blurx = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					const int sbottom = r - bottom;

					//h filter
					for (int j = -stop; j < TILE_Y + sbottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + tstep * (j + r);

						for (int i = 0; i < LEFT; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(i + k - r);
								v += (idx >= 0) ? gauss32F[k] * s[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}

						for (int i = LEFT; i < TILE_X - RIGHT; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * s[idx];
							}
							d[i] = v;
						}

						for (int i = TILE_X - RIGHT; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(i + tilex + k - r, hmax);
								v += (idx >= 0) ? gauss32F[k] * s[idx - tilex] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}

					if (tiley == 0)
					{
						//upper border
#ifdef  BORDER_REFLECT_FILTER
						for (int i = 0; i < top; i++)
						{
							const int v = min(r + i, TILE_Y + 2 * r - 1);
							//const int v = r + i;
							memcpy(blurx + (r - 1 - i) * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					if (tiley == src.rows - TILE_Y)
					{
#ifdef  BORDER_REFLECT_FILTER
						//downer border
						for (int i = TILE_Y + r; i < TILE_Y + 2 * r; i++)
						{
							const int v = max(2 * (TILE_Y + r) - i - 1, 0);
							memcpy(blurx + i * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					// v filter
					for (int j = 0; j < TILE_Y; j++)
					{
						float* s = blurx + j * tstep;
						float* d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * si[0];
								si += tstep;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				const int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif

			const int tstep = TILE_X;
			const int wstep = src.cols;
			const int wstep0 = src.cols * 0;
			const int wstep1 = src.cols * 1;
			const int wstep2 = src.cols * 2;
			const int wstep3 = src.cols * 3;
			const int wstep4 = src.cols * 4;
			const int wstep5 = src.cols * 5;
			const int wstep6 = src.cols * 6;
			const int wstep7 = src.cols * 7;


			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;

				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(TILE_X, TILE_Y + 2 * r)) bufferTile[tidx].create(Size(TILE_X, TILE_Y + 2 * r), CV_32F);
				const int R = get_simd_ceil(r, 8);
				if (bufferTileLine[tidx].size() != Size(TILE_Y + 2 * R, 1)) bufferTileLine[tidx].create(Size(TILE_Y + 2 * R, 1), CV_32F);
				float* blurx = bufferTile[tidx].ptr<float>(0);
				float* blury = bufferTileLine[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					const int sbottom = r - bottom;
					//h filter
					for (int j = -stop; j < TILE_Y + sbottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + tstep * (j + r);

						for (int i = 0; i < LEFT; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							int idx = i;
							for (int k = 0; k < r - i; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s, start_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							float* si = s;
							for (int k = r - i; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							_mm256_store_ps(d + i, mv);
						}

						for (int i = LEFT; i < TILE_X - RIGHT; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							_mm256_store_ps(d + i, mv);
						}

						for (int i = TILE_X - RIGHT; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();

							const int e = src.cols - (i + tilex + 8);
							float* si = s + i - r;
							for (int k = 0; k < r + 1 + e; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}

							int idx = 0;
							for (int k = r + 1 + e; k < ksize; k++)
							{
								__m256 ms = _mm256_i32gather_ps(s - tilex, end_access_pattern[idx], sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								idx++;
							}
							_mm256_store_ps(d + i, mv);
						}
					}

					if (tiley == 0)
					{
						//upper border
#ifdef  BORDER_REFLECT_FILTER
						for (int i = 0; i < top; i++)
						{
							const int v = min(r + i, TILE_Y + 2 * r - 1);
							//const int v = r + i;
							memcpy(blurx + (r - 1 - i) * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}
					if (tiley == src.rows - TILE_Y)
					{
#ifdef  BORDER_REFLECT_FILTER
						//downer border
						for (int i = TILE_Y + r; i < TILE_Y + 2 * r; i++)
						{
							const int v = max(2 * (TILE_Y + r) - i - 1, 0);
							memcpy(blurx + i * tstep, blurx + v * tstep, sizeof(float) * tstep);
						}
#endif
					}

					// v filter
					float* dptr = dest.ptr<float>(tiley) + tilex;
					for (int i = 0; i < TILE_X; i++)
					{
						verticalLineCopy(bufferTile[tidx], bufferTileLine[tidx], i);
						float* d = dptr + i;
						for (int j = 0; j < TILE_Y; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = blury + j;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							d[wstep0] = ((float*)&mv)[0];
							d[wstep1] = ((float*)&mv)[1];
							d[wstep2] = ((float*)&mv)[2];
							d[wstep3] = ((float*)&mv)[3];
							d[wstep4] = ((float*)&mv)[4];
							d[wstep5] = ((float*)&mv)[5];
							d[wstep6] = ((float*)&mv)[6];
							d[wstep7] = ((float*)&mv)[7];
							d += wstep * 8;
						}
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterVHTileImage(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		//const int hmax = src.cols - 1;
		const int max_core = 1;

		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const int R = get_simd_ceil(r, 8);
			const int tstep = 2 * R + TILE_X;
			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(tstep, TILE_Y))bufferTile[tidx].create(Size(tstep, TILE_Y), CV_32F);
				float* blurxPtr = bufferTile[tidx].ptr<float>(0);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					//const int LEFT = get_simd_ceil(left, 8);
					//const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					//const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					//const int sbottom = r - bottom;

					//v filter
					for (int j = 0; j < top; j++)
					{
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + tstep * j + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r);
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}
					}
					for (int j = top; j < TILE_Y - bottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + tstep * j + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = k - r;
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}
					}
					for (int j = TILE_Y - bottom; j < TILE_Y; j++)
					{
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + tstep * j + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(tiley + j + k - r, vmax);
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}
					}

					if (tilex == 0)
					{
#ifdef  BORDER_REFLECT_FILTER
						for (int j = 0; j < TILE_Y; j++)
						{
							float* d = blurx + tstep * j + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
						}
#endif
					}
					else if (tilex == src.cols - TILE_X)
					{
#ifdef  BORDER_REFLECT_FILTER
						for (int j = 0; j < TILE_Y; j++)
						{
							float* d = blurx + tstep * j + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_X + i] = d[TILE_X - 1 - i];
							}
						}
#endif
					}
					// h filter
					for (int j = 0; j < TILE_Y; j++)
					{
						float* s = blurx + tstep * j;
						float* d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[i + k];
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int R = get_simd_ceil(r, 8);
			const int wstep = src.cols;
			const int tstep = TILE_X + 2 * R;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(tstep, TILE_Y))bufferTile[tidx].create(Size(tstep, TILE_Y), CV_32F);
				float* blurxPtr = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					//const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					//const int sbottom = r - bottom;

					const int STX = -R + LEFT;
					const int EDX = TILE_X + R - RIGHT;
					//v filter
					float* d = blurx + R;
					for (int j = 0; j < top; j++)
					{
						float* s = src.ptr<float>(0) + tilex;

						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;

							const int e = j + r;
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_load_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + (j + e + 1 - r) * wstep;
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							_mm256_store_ps(d + i, mv);
						}
						d += tstep;
					}
					float* s = src.ptr<float>(tiley + top - r) + tilex;
					d = blurx + tstep * top + R;
					for (int j = top; j < TILE_Y - bottom; j++)
					{
						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
							_mm256_store_ps(d + i, mv);
						}
						s += wstep;
						d += tstep;
					}
					d = blurx + tstep * (TILE_Y - bottom) + R;
					for (int j = TILE_Y - bottom; j < TILE_Y; j++)
					{
						float* s = src.ptr<float>(0) + tilex;

						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i + (j + tiley - r) * wstep;
							for (int k = 0; k < r + 1; k++)
							{
								__m256 ms = _mm256_load_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
							si = s + i;
							for (int k = r + 1; k < ksize; k++)
							{
								int idx = border_e(j + tiley + k - r, vmax);
								__m256 ms = _mm256_load_ps(si + wstep * idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							_mm256_store_ps(d + i, mv);
						}
						d += tstep;
					}

					if (tilex == 0)
					{
#ifdef  BORDER_REFLECT_FILTER
						float* d = blurx + R;
						for (int j = 0; j < TILE_Y; j++)
						{
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d - i - 8, a);
							}
							d += tstep;
						}
#endif
					}
					else if (tilex == src.cols - TILE_X)
					{
#ifdef  BORDER_REFLECT_FILTER
						float* d = blurx + R;
						for (int j = 0; j < TILE_Y; j++)
						{
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + TILE_X - 8 - i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d + TILE_X + i, a);
							}
							d += tstep;
						}
#endif
					}
					// h filter
					s = blurx + R - r;
					d = dest.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							_mm256_store_ps(d + i, mv);
						}
						s += tstep;
						d += dest.cols;
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHTileImageBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		//const int hmax = src.cols - 1;
		const int max_core = 1;

		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (!useAllocBuffer)bufferImageBorder.release();
		const int R = get_simd_ceil(r, 8);
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, 0, 0, R, R, border);
		else copyMakeBorder(src, bufferImageBorder, 0, 0, R, R, border);

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const int R = get_simd_ceil(r, 8);
			const int tstep = 2 * R + TILE_X;
			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(tstep, TILE_Y))bufferTile[tidx].create(Size(tstep, TILE_Y), CV_32F);
				float* blurxPtr = bufferTile[tidx].ptr<float>(0);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					//const int LEFT = get_simd_ceil(left, 8);
					//const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					//const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					//const int sbottom = r - bottom;

					//v filter
					for (int j = 0; j < top; j++)
					{
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + tstep * j + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r);
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}
					}
					for (int j = top; j < TILE_Y - bottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + tstep * j + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = k - r;
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}
					}
					for (int j = TILE_Y - bottom; j < TILE_Y; j++)
					{
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + tstep * j + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(tiley + j + k - r, vmax);
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}
					}

					if (tilex == 0)
					{
#ifdef  BORDER_REFLECT_FILTER
						for (int j = 0; j < TILE_Y; j++)
						{
							float* d = blurx + tstep * j + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
						}
#endif
					}
					else if (tilex == src.cols - TILE_X)
					{
#ifdef  BORDER_REFLECT_FILTER
						for (int j = 0; j < TILE_Y; j++)
						{
							float* d = blurx + tstep * j + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_X + i] = d[TILE_X - 1 - i];
							}
						}
#endif
					}
					// h filter
					for (int j = 0; j < TILE_Y; j++)
					{
						float* s = blurx + tstep * j;
						float* d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[i + k];
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int R = get_simd_ceil(r, 8);
			const int wstep = bufferImageBorder.cols;
			const int tstep = TILE_X + 2 * R;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(tstep, TILE_Y))bufferTile[tidx].create(Size(tstep, TILE_Y), CV_32F);
				float* blurxPtr = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int top = max(0, r - tiley);
					//const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					//const int sbottom = r - bottom;

					const int STX = -R;
					const int EDX = TILE_X + R;
					//v filter
					float* d = blurx + R;
					for (int j = 0; j < top; j++)
					{
						float* s = bufferImageBorder.ptr<float>(0) + tilex + R;

						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;

							const int e = j + r;
							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								__m256 ms = _mm256_loadu_ps(si + idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + (j + e + 1 - r) * wstep;
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							_mm256_store_ps(d + i, mv);
						}
						d += tstep;
					}
					float* s = bufferImageBorder.ptr<float>(tiley + top - r) + tilex + R;
					d = blurx + tstep * top + R;
					for (int j = top; j < TILE_Y - bottom; j++)
					{
						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
							_mm256_store_ps(d + i, mv);
						}
						s += wstep;
						d += tstep;
					}
					d = blurx + tstep * (TILE_Y - bottom) + R;
					for (int j = TILE_Y - bottom; j < TILE_Y; j++)
					{
						float* s = bufferImageBorder.ptr<float>(0) + tilex + R;
						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i + (j + tiley - r) * wstep;
							for (int k = 0; k < r + 1; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
							si = s + i;
							for (int k = r + 1; k < ksize; k++)
							{
								int idx = border_e(j + tiley + k - r, vmax);
								__m256 ms = _mm256_loadu_ps(si + wstep * idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							_mm256_store_ps(d + i, mv);
						}
						d += tstep;
					}
					// h filter
					s = blurx + R - r;
					d = dest.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							_mm256_store_ps(d + i, mv);
						}
						s += tstep;
						d += dest.cols;
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHTileImageBV(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		//const int hmax = src.cols - 1;
		const int max_core = 1;

		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (!useAllocBuffer)bufferImageBorder.release();
		const int R = get_simd_ceil(r, 8);
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, R, R, 0, 16, border);
		else copyMakeBorder(src, bufferImageBorder, R, R, 0, 0, border);

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const int R = get_simd_ceil(r, 8);
			const int tstep = 2 * R + TILE_X;
			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(tstep, TILE_Y))bufferTile[tidx].create(Size(tstep, TILE_Y), CV_32F);
				float* blurxPtr = bufferTile[tidx].ptr<float>(0);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					//const int LEFT = get_simd_ceil(left, 8);
					//const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					//const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					//const int sbottom = r - bottom;

					//v filter
					for (int j = 0; j < top; j++)
					{
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + tstep * j + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r);
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}
					}
					for (int j = top; j < TILE_Y - bottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + tstep * j + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = k - r;
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}
					}
					for (int j = TILE_Y - bottom; j < TILE_Y; j++)
					{
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + tstep * j + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(tiley + j + k - r, vmax);
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}
					}

					if (tilex == 0)
					{
#ifdef  BORDER_REFLECT_FILTER
						for (int j = 0; j < TILE_Y; j++)
						{
							float* d = blurx + tstep * j + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
						}
#endif
					}
					else if (tilex == src.cols - TILE_X)
					{
#ifdef  BORDER_REFLECT_FILTER
						for (int j = 0; j < TILE_Y; j++)
						{
							float* d = blurx + tstep * j + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_X + i] = d[TILE_X - 1 - i];
							}
						}
#endif
					}
					// h filter
					for (int j = 0; j < TILE_Y; j++)
					{
						float* s = blurx + tstep * j;
						float* d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[i + k];
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int R = get_simd_ceil(r, 8);
			const int wstep = bufferImageBorder.cols;
			const int tstep = TILE_X + 2 * R;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(tstep, TILE_Y))bufferTile[tidx].create(Size(tstep, TILE_Y), CV_32F);
				float* blurxPtr = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int STX = -R + LEFT;
					const int EDX = TILE_X + R - RIGHT;
					//v filter
					float* d = blurx + R;
					float* s = bufferImageBorder.ptr<float>(tiley) + tilex;
					d = blurx + R;
					for (int j = 0; j < TILE_Y; j++)
					{
						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
							_mm256_store_ps(d + i, mv);
						}
						s += wstep;
						d += tstep;
					}

					if (tilex == 0)
					{
#ifdef  BORDER_REFLECT_FILTER
						float* d = blurx + R;
						for (int j = 0; j < TILE_Y; j++)
						{
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d - i - 8, a);
							}
							d += tstep;
						}
#endif
					}
					else if (tilex == src.cols - TILE_X)
					{
#ifdef  BORDER_REFLECT_FILTER
						float* d = blurx + R;
						for (int j = 0; j < TILE_Y; j++)
						{
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + TILE_X - 8 - i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d + TILE_X + i, a);
							}
							d += tstep;
						}
#endif
					}
					// h filter
					s = blurx + R - r;
					d = dest.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							_mm256_store_ps(d + i, mv);
						}
						s += tstep;
						d += dest.cols;
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHITileLineBH(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		//const int hmax = src.cols - 1;
		const int max_core = 1;
		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const int R = get_simd_ceil(r, 8);
			const int tstep = 2 * R + TILE_X;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTileLine[tidx].release();
				if (bufferTileLine[tidx].size() != Size(tstep, 1))bufferTileLine[tidx].create(Size(tstep, 1), CV_32F);
				float* blurxPtr = bufferTileLine[tidx].ptr<float>(0);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					//const int LEFT = get_simd_ceil(left, 8);
					//const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					//const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					//const int sbottom = r - bottom;

					for (int j = 0; j < top; j++)
					{
						//v filter
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								v += gauss32F[k] * s[idx + i];
							}
							d[i] = v;
						}

						if (tilex == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
#endif
						}
						else if (tilex == src.cols - TILE_X)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_X + i] = d[TILE_X - 1 - i];
							}
#endif
						}
						// h filter
						s = blurx;
						d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[i + k];
							}
							d[i] = v;
						}
					}

					for (int j = top; j < TILE_Y - bottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = k - r;
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}

						if (tilex == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
#endif
						}
						else if (tilex == src.cols - TILE_X)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_X + i] = d[TILE_X - 1 - i];
							}
#endif
						}
						// h filter
						s = blurx;
						d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[i + k];
							}
							d[i] = v;
						}
					}
					for (int j = TILE_Y - bottom; j < TILE_Y; j++)
					{
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(tiley + j + k - r, vmax);
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}

						if (tilex == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
#endif
						}
						else if (tilex == src.cols - TILE_X)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_X + i] = d[TILE_X - 1 - i];
							}
#endif
						}
						// h filter
						s = blurx;
						d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[i + k];
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int wstep = src.cols;
			const int R = get_simd_ceil(r, 8);
			const int tstep = 2 * R + TILE_X;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTileLine[tidx].release();
				if (bufferTileLine[tidx].size() != Size(tstep, 1))bufferTileLine[tidx].create(Size(tstep, 1), CV_32F);
				float* blurxPtr = bufferTileLine[tidx].ptr<float>(0);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					//const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					//const int sbottom = r - bottom;

					const int STX = -R + LEFT;
					const int EDX = TILE_X + R - RIGHT;
					for (int j = 0; j < top; j++)
					{
						//v filter
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + R;
						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							const int e = j + r;

							for (int k = 0; k < e + 1; k++)
							{
								int idx = border_s(j + k - r);
								__m256 ms = _mm256_load_ps(si + wstep * idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							float* sii = si + (j + e + 1 - r) * wstep;
							//si = si + j*wstep;
							for (int k = e + 1; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(sii);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								sii += wstep;
							}
							_mm256_store_ps(d + i, mv);
						}

						if (tilex == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d - i - 8, a);
							}
#endif
						}
						else if (tilex == src.cols - TILE_X)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + TILE_X - 8 - i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d + TILE_X + i, a);
							}
#endif
						}
						// h filter
						s = blurx + R - r;
						d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							_mm256_store_ps(d + i, mv);
						}
					}

					float* s = src.ptr<float>(tiley + top - r) + tilex;
					for (int j = top; j < TILE_Y - bottom; j++)
					{
						float* d = blurx + R;
						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
							_mm256_store_ps(d + i, mv);
						}

						if (tilex == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d - i - 8, a);
							}
#endif
						}
						else if (tilex == src.cols - TILE_X)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + TILE_X - 8 - i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d + TILE_X + i, a);
							}
#endif
						}
						// h filter
						float* b = blurx + R - r;
						d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* bi = b + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
						s += src.cols;
					}
					for (int j = TILE_Y - bottom; j < TILE_Y; j++)
					{
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + R;
						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i + (j + tiley - r) * wstep;
							for (int k = 0; k < r + 1; k++)
							{
								__m256 ms = _mm256_load_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
							si = s + i;
							for (int k = r + 1; k < ksize; k++)
							{
								int idx = border_e(j + tiley + k - r, vmax);
								__m256 ms = _mm256_load_ps(si + wstep * idx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							_mm256_store_ps(d + i, mv);
						}

						if (tilex == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d - i - 8, a);
							}
#endif
						}
						else if (tilex == src.cols - TILE_X)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + TILE_X - 8 - i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d + TILE_X + i, a);
							}
#endif
						}
						// h filter
						s = blurx + R - r;
						d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si++;
							}
							_mm256_store_ps(d + i, mv);
						}
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHITileImageBV(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		//const int hmax = src.cols - 1;
		const int max_core = 1;
		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (!useAllocBuffer)bufferImageBorder.release();
		//const int R = get_simd_ceil(r, 8);
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, 0, 0, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, 0, 0, border);

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const int R = get_simd_ceil(r, 8);
			const int tstep = 2 * R + TILE_X;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTileLine[tidx].release();
				if (bufferTileLine[tidx].size() != Size(tstep, 1))bufferTileLine[tidx].create(Size(tstep, 1), CV_32F);
				float* blurxPtr = bufferTileLine[tidx].ptr<float>(0);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					//const int LEFT = get_simd_ceil(left, 8);
					//const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					//const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					//const int sbottom = r - bottom;

					for (int j = 0; j < top; j++)
					{
						//v filter
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(j + k - r) * wstep;
								v += gauss32F[k] * s[idx + i];
							}
							d[i] = v;
						}

						if (tilex == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
#endif
						}
						else if (tilex == src.cols - TILE_X)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_X + i] = d[TILE_X - 1 - i];
							}
#endif
						}
						// h filter
						s = blurx;
						d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[i + k];
							}
							d[i] = v;
						}
					}

					for (int j = top; j < TILE_Y - bottom; j++)
					{
						float* s = src.ptr<float>(tiley + j) + tilex;
						float* d = blurx + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = k - r;
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}

						if (tilex == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
#endif
						}
						else if (tilex == src.cols - TILE_X)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_X + i] = d[TILE_X - 1 - i];
							}
#endif
						}
						// h filter
						s = blurx;
						d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[i + k];
							}
							d[i] = v;
						}
					}
					for (int j = TILE_Y - bottom; j < TILE_Y; j++)
					{
						float* s = src.ptr<float>(0) + tilex;
						float* d = blurx + r;
						for (int i = -r + left; i < TILE_X + r - right; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(tiley + j + k - r, vmax);
								v += gauss32F[k] * s[idx * wstep + i];
							}
							d[i] = v;
						}

						if (tilex == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
#endif
						}
						else if (tilex == src.cols - TILE_X)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_X + i] = d[TILE_X - 1 - i];
							}
#endif
						}
						// h filter
						s = blurx;
						d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[i + k];
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int wstep = bufferImageBorder.cols;
			const int R = get_simd_ceil(r, 8);
			const int tstep = 2 * R + TILE_X;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTileLine[tidx].release();
				if (bufferTileLine[tidx].size() != Size(tstep, 1))bufferTileLine[tidx].create(Size(tstep, 1), CV_32F);
				float* blurxPtr = bufferTileLine[tidx].ptr<float>(0);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int LEFT = get_simd_ceil(left, 8);
					const int RIGHT = get_simd_ceil(right, 8);

					const int STX = -R + LEFT;
					const int EDX = TILE_X + R - RIGHT;

					float* s = bufferImageBorder.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						float* d = blurx + R;
						for (int i = STX; i < EDX; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* si = s + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(si);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								si += wstep;
							}
							_mm256_store_ps(d + i, mv);
						}

						if (tilex == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d - i - 8, a);
							}
#endif
						}
						else if (tilex == src.cols - TILE_X)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + TILE_X - 8 - i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d + TILE_X + i, a);
							}
#endif
						}

						// h filter
						float* b = blurx + R - r;
						d = dest.ptr<float>(j + tiley) + tilex;
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* bi = b + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(bi);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								bi++;
							}
							_mm256_store_ps(d + i, mv);
						}
						s += wstep;
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVITileLine(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		//const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;
		const int max_core = 1;
		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);
		CV_Assert(r > TILE_X || TILE_Y > r);

		if (opt == VECTOR_WITHOUT)
		{
			//const int wstep = src.cols;
			const int tstep = 2 * r + TILE_Y;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTileLine[tidx].release();
				if (bufferTileLine[tidx].size() != Size(tstep, 1))bufferTileLine[tidx].create(Size(tstep, 1), CV_32F);
				float* blurxPtr = bufferTileLine[tidx].ptr<float>(0);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* blurx = blurxPtr;

					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					//const int LEFT = get_simd_ceil(left, 8);
					//const int RIGHT = get_simd_ceil(right, 8);

					const int top = max(0, r - tiley);
					//const int stop = r - top;
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					//const int sbottom = r - bottom;

					for (int i = 0; i < left; i++)
					{
						//h filter
						float* d = blurx;
						for (int j = -r + top; j < TILE_Y + r - bottom; j++)
						{
							float* s = src.ptr<float>(j + tiley) + tilex;
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(i + k - r);
								v += gauss32F[k] * s[idx];
							}
							d[j + r] = v;
						}

						if (tiley == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
#endif
						}
						else if (tiley == src.rows - TILE_Y)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_Y + i] = d[TILE_Y - 1 - i];
							}
#endif
						}
						// v filter
						float* s = blurx;
						for (int j = 0; j < TILE_Y; j++)
						{
							d = dest.ptr<float>(j + tiley) + tilex;
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[j + k];
							}
							d[i] = v;
						}
					}

					for (int i = left; i < TILE_X - right; i++)
					{
						//h filter
						float* d = blurx;
						for (int j = -r + top; j < TILE_Y + r - bottom; j++)
						{
							float* s = src.ptr<float>(tiley + j) + tilex;
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i - r + k;
								v += gauss32F[k] * s[idx];
							}
							d[j + r] = v;
						}

						if (tiley == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
#endif
						}
						else if (tiley == src.rows - TILE_Y)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_Y + i] = d[TILE_Y - 1 - i];
							}
#endif
						}
						// v filter
						float* s = blurx;
						for (int j = 0; j < TILE_Y; j++)
						{
							d = dest.ptr<float>(j + tiley) + tilex;
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[j + k];
							}
							d[i] = v;
						}
					}
					for (int i = TILE_X - right; i < TILE_X; i++)
					{
						//h filter
						float* d = blurx;
						for (int j = -r + top; j < TILE_Y + r - bottom; j++)
						{
							float* s = src.ptr<float>(tiley + j) + tilex;
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(tilex + i + k - r, hmax) - tilex;
								v += gauss32F[k] * s[idx];
							}
							d[j + r] = v;
						}

						if (tiley == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[-i - 1] = d[i];
							}
#endif
						}
						else if (tiley == src.rows - TILE_Y)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + r;
							for (int i = 0; i < r; i++)
							{
								d[TILE_Y + i] = d[TILE_Y - 1 - i];
							}
#endif
						}
						// v filter
						float* s = blurx;
						for (int j = 0; j < TILE_Y; j++)
						{
							d = dest.ptr<float>(j + tiley) + tilex;
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * s[j + k];
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}

#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int wstep = src.cols;
			const int R = get_simd_ceil(r, 8);
			const int tstep = 2 * R + TILE_Y;
			const int wstep0 = 0 * wstep;
			const int wstep1 = 1 * wstep;
			const int wstep2 = 2 * wstep;
			const int wstep3 = 3 * wstep;
			const int wstep4 = 4 * wstep;
			const int wstep5 = 5 * wstep;
			const int wstep6 = 6 * wstep;
			const int wstep7 = 7 * wstep;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTileLine[tidx].release();
				if (bufferTileLine[tidx].size() != Size(tstep, 1))bufferTileLine[tidx].create(Size(tstep, 1), CV_32F);
				float* blurxPtr = bufferTileLine[tidx].ptr<float>(0);
				const __m256i midx = _mm256_setr_epi32(0 * wstep, 1 * wstep, 2 * wstep, 3 * wstep, 4 * wstep, 5 * wstep, 6 * wstep, 7 * wstep);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					const int left = max(0, r - tilex);
					const int right = max(0, r + tilex + TILE_X - src.cols);
					const int top = max(0, r - tiley);
					const int bottom = max(0, r + (tiley + TILE_Y - src.rows));
					const int TOP = get_simd_ceil(top, 8);
					const int BOTTOM = get_simd_ceil(bottom, 8);

					for (int i = 0; i < left; i++)
					{
						float* blurx = blurxPtr;
						//h filter
						float* d = blurxPtr;
						for (int j = -R + TOP; j < TILE_Y + R - BOTTOM; j += 8)
						{
							float* s = src.ptr<float>(j + tiley) + tilex;
							__m256 mv = _mm256_setzero_ps();
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_s(i + k - r);
								__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							_mm256_store_ps(d + j + R, mv);
						}

						if (tiley == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d - i - 8, a);
							}
#endif
						}
						else if (tiley == src.rows - TILE_Y)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + TILE_Y - 8 - i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d + TILE_Y + i, a);
							}
#endif
						}

						// v filter
						float* dii = dest.ptr<float>(tiley) + tilex + i;
						for (int j = 0; j < TILE_Y; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							blurx = (float*)blurxPtr + j + R - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(blurx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								blurx++;
							}
							dii[wstep0] = ((float*)&mv)[0];
							dii[wstep1] = ((float*)&mv)[1];
							dii[wstep2] = ((float*)&mv)[2];
							dii[wstep3] = ((float*)&mv)[3];
							dii[wstep4] = ((float*)&mv)[4];
							dii[wstep5] = ((float*)&mv)[5];
							dii[wstep6] = ((float*)&mv)[6];
							dii[wstep7] = ((float*)&mv)[7];
							dii += dest.cols * 8;
						}
					}

					for (int i = left; i < TILE_X - right; i++)
					{
						//h filter
						float* blurx = blurxPtr;
						float* d = blurx;
						for (int j = -R + TOP; j < TILE_Y + R - BOTTOM; j += 8)
						{
							float* s = src.ptr<float>(tiley + j) + tilex;
							__m256 mv = _mm256_setzero_ps();
							for (int k = 0; k < ksize; k++)
							{
								int idx = i - r + k;
								__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							_mm256_store_ps(d + j + R, mv);
						}

						if (tiley == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d - i - 8, a);
							}
#endif
						}
						else if (tiley == src.rows - TILE_Y)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + TILE_Y - 8 - i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d + TILE_Y + i, a);
							}
#endif
						}

						// v filter
						float* dii = dest.ptr<float>(tiley) + tilex + i;
						for (int j = 0; j < TILE_Y; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							blurx = (float*)blurxPtr + j + R - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(blurx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								blurx++;
							}
							dii[wstep0] = ((float*)&mv)[0];
							dii[wstep1] = ((float*)&mv)[1];
							dii[wstep2] = ((float*)&mv)[2];
							dii[wstep3] = ((float*)&mv)[3];
							dii[wstep4] = ((float*)&mv)[4];
							dii[wstep5] = ((float*)&mv)[5];
							dii[wstep6] = ((float*)&mv)[6];
							dii[wstep7] = ((float*)&mv)[7];
							dii += dest.cols * 8;
						}
					}
					for (int i = TILE_X - right; i < TILE_X; i++)
					{
						float* blurx = blurxPtr;
						//h filter
						float* d = blurx;
						for (int j = -R + TOP; j < TILE_Y + R - BOTTOM; j += 8)
						{
							float* s = src.ptr<float>(tiley + j) + tilex;
							__m256 mv = _mm256_setzero_ps();
							for (int k = 0; k < ksize; k++)
							{
								int idx = border_e(tilex + i + k - r, hmax) - tilex;
								__m256 ms = _mm256_i32gather_ps(s + idx, midx, sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
							}
							_mm256_store_ps(d + j + R, mv);
						}

						if (tiley == 0)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d - i - 8, a);
							}
#endif
						}
						else if (tiley == src.rows - TILE_Y)
						{
#ifdef  BORDER_REFLECT_FILTER
							float* d = blurx + R;
							for (int i = 0; i < R; i += 8)
							{
								__m256 a = _mm256_load_ps(d + TILE_Y - 8 - i);
								a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
								a = _mm256_permute2f128_ps(a, a, 1);
								_mm256_store_ps(d + TILE_Y + i, a);
							}
#endif
						}
						// v filter
						float* dii = dest.ptr<float>(tiley) + tilex + i;
						for (int j = 0; j < TILE_Y; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							blurx = (float*)blurxPtr + j + R - r;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(blurx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								blurx++;
							}
							dii[wstep0] = ((float*)&mv)[0];
							dii[wstep1] = ((float*)&mv)[1];
							dii[wstep2] = ((float*)&mv)[2];
							dii[wstep3] = ((float*)&mv)[3];
							dii[wstep4] = ((float*)&mv)[4];
							dii[wstep5] = ((float*)&mv)[5];
							dii[wstep6] = ((float*)&mv)[6];
							dii[wstep7] = ((float*)&mv)[7];
							dii += dest.cols * 8;
						}
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterVHILineBufferOverRun(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.data != dest.data);
		const int ksize = 2 * r + 1;
		const int vmax = src.rows - 1;
		const int hmax = src.cols - 1;

		if (opt == VECTOR_WITHOUT)
		{
			const int wstep = src.cols;
			const float* s = src.ptr<float>(0);

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(Size(dest.cols, 1), CV_32F);
				float* b = buff.ptr<float>(0);

				if (j < r)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_s(idx);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}
					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else if (j > src.rows - r - 1)
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						const float* si = s + i;
						float v = 0.f;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							idx = border_e(idx, vmax);
							v += (idx >= 0) ? gauss32F[k] * si[idx * wstep] : gauss32F[k] * constVal;
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
				else
				{
					//v filter
					for (int i = 0; i < src.cols; i++)
					{
						float v = 0.f;
						const float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							int idx = j + k - r;
							v += gauss32F[k] * si[idx * wstep];
						}
						b[i] = v;
					}

					//h filter
					{
						float* d = dest.ptr<float>(j);
						for (int i = 0; i < r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_s(idx);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
						for (int i = r; i < src.cols - r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								v += gauss32F[k] * b[idx];
							}
							d[i] = v;
						}
						for (int i = src.cols - r; i < src.cols; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								int idx = i + k - r;
								idx = border_e(idx, hmax);
								v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
							}
							d[i] = v;
						}
					}
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int wstep = src.cols;
			// access pattern for image boundary
			__m256i* access_pattern = (__m256i*)_mm_malloc(sizeof(__m256i) * 2 * r, 32);
			__m256i* start_access_pattern = access_pattern;
			__m256i* end_access_pattern = access_pattern + r;
			for (int i = 0; i < r; i++)
			{
				int idx = i - r;
				start_access_pattern[i] = _mm256_setr_epi32
				(
					border_s(idx + 0),
					border_s(idx + 1),
					border_s(idx + 2),
					border_s(idx + 3),
					border_s(idx + 4),
					border_s(idx + 5),
					border_s(idx + 6),
					border_s(idx + 7)
				);
			}
			for (int i = 0; i < r; i++)
			{
				end_access_pattern[i] = _mm256_setr_epi32
				(
					border_e(src.cols - 7 + i, hmax),
					border_e(src.cols - 6 + i, hmax),
					border_e(src.cols - 5 + i, hmax),
					border_e(src.cols - 4 + i, hmax),
					border_e(src.cols - 3 + i, hmax),
					border_e(src.cols - 2 + i, hmax),
					border_e(src.cols - 1 + i, hmax),
					border_e(src.cols - 0 + i, hmax)
				);
			}
#ifdef BORDER_CONSTANT
			__m256* mMask_s = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_s[0] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);
			mMask_s[1] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, -1);
			mMask_s[2] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, -1, -1);
			mMask_s[3] = _mm256_setr_ps(0, 0, 0, 0, 0, -1, -1, -1);
			mMask_s[4] = _mm256_setr_ps(0, 0, 0, 0, -1, -1, -1, -1);
			mMask_s[5] = _mm256_setr_ps(0, 0, 0, -1, -1, -1, -1, -1);
			mMask_s[6] = _mm256_setr_ps(0, 0, -1, -1, -1, -1, -1, -1);
			mMask_s[7] = _mm256_setr_ps(0, -1, -1, -1, -1, -1, -1, -1);

			__m256* mMask_e = (__m256*)_mm_malloc(sizeof(__m256) * 8, 32);
			mMask_e[0] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, -1, 0);
			mMask_e[1] = _mm256_setr_ps(-1, -1, -1, -1, -1, -1, 0, 0);
			mMask_e[2] = _mm256_setr_ps(-1, -1, -1, -1, -1, 0, 0, 0);
			mMask_e[3] = _mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);
			mMask_e[4] = _mm256_setr_ps(-1, -1, -1, 0, 0, 0, 0, 0);
			mMask_e[5] = _mm256_setr_ps(-1, -1, 0, 0, 0, 0, 0, 0);
			mMask_e[6] = _mm256_setr_ps(-1, 0, 0, 0, 0, 0, 0, 0);
			mMask_e[7] = _mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 0);

			__m256 mVal = _mm256_set1_ps((float)constVal);
#endif
			const int max_core = 1;
			float* s = src.ptr<float>(0);
			const int R = get_simd_ceil(r, 8);
			const int simdend = get_simd_floor_end(R, src.cols - r, 8);
			if (r >= 8)//for overrun. OK
			{

				for (int n = 0; n < max_core; n++)
				{
					const int tidx = 0;
					const int strip = src.rows / max_core;
					const int start = n * strip;
					const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
					const int simdwidth = get_simd_ceil(src.cols, 8);
					if (!useAllocBuffer) bufferLineCols[tidx].release();
					if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
					float* b = bufferLineCols[tidx].ptr<float>(0);
					for (int j = start; j < end; j++)
					{
						if (j < r)
						{
							//v filter
							for (int i = 0; i < src.cols; i += 8)
							{
								float* si = s + i;
								__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
								if (border == BORDER_CONSTANT)
								{
									for (int k = 0; k < ksize; k++)
									{
										int idx = j + k - r;
										idx = border_s(idx) * wstep;
										__m256 ms;
										if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
										else ms = mVal;
										__m256 mg = _mm256_set1_ps(gauss[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
								}
								else
#endif
								{
									const int e = j + r;
									for (int k = 0; k < e + 1; k++)
									{
										int idx = border_s(j + k - r) * wstep;
										__m256 ms = _mm256_loadu_ps(si + idx);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
									float* sii = si + wstep * (j + e + 1 - r);
									for (int k = e + 1; k < ksize; k++)
									{
										__m256 ms = _mm256_loadu_ps(sii);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										sii += wstep;
									}
								}
								_mm256_store_ps(b + i, mv);
							}

							//h filter
							{
								float* d = dest.ptr<float>(j);

								for (int i = 0; i < r; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
									if (border == BORDER_CONSTANT)
									{
										for (int k = 0; k < r - i; k++)
										{
											int idx = i + k;
											int maskIdx = max(0, k + i - r + 8);
											__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									else
#endif
									{
										int idx = i;
										for (int k = 0; k < r - i; k++)
										{
											__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss32F[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
											idx++;
										}
									}
									float* bi = b;
									for (int k = r - i; k < ksize; k++)
									{
										__m256 ms = _mm256_loadu_ps(bi);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										bi++;
									}
									_mm256_store_ps(d + i, mv);
								}
								for (int i = r; i < src.cols - r; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
									float* bi = b + i - r;
									for (int k = 0; k < ksize; k++)
									{
										__m256 ms = _mm256_loadu_ps(bi);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										bi++;
									}
									_mm256_storeu_ps(d + i, mv);
								}
								int v = r / 8;
								for (int vv = 0; vv < v; vv++)//over run
								{
									int i = src.cols - r + 8 * vv;
									__m256 mv = _mm256_setzero_ps();
									float* bi = b + i - r;
									for (int k = 0; k < src.cols - 7 + r - i; k++)
									{
										__m256 ms = _mm256_load_ps(bi);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										bi++;
									}
#ifdef BORDER_CONSTANT
									if (border == BORDER_CONSTANT)
									{
										for (int k = src.cols - 7 + r - i; k < ksize; k++)
										{
											int idx = k - src.cols + 7 - r + i;
											int maskIdx = min(k - (src.cols - 7 + r - i), 7);
											__m256 ms = _mm256_mask_i32gather_ps(mVal, b, end_access_pattern[idx], mMask_e[maskIdx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									else
#endif
									{
										int idx = 0;
										for (int k = src.cols - 7 + r - i; k < ksize; k++)
										{
											__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss32F[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
											idx++;
										}
									}
									_mm256_storeu_ps(d + i, mv);
								}
								for (int i = src.cols - r + 8 * v; i < src.cols; i++)
								{
									float v = 0.f;
									for (int k = 0; k < ksize; k++)
									{
										int idx = i + k - r;
										idx = border_e(idx, hmax);
										v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
									}
									d[i] = v;
								}
							}
						}
						else if (j > src.rows - r - 1)
						{
							//v filter
							for (int i = 0; i < src.cols; i += 8)
							{
								float* si = s + i;
								__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
								if (border == BORDER_CONSTANT)
								{
									for (int k = 0; k < ksize; k++)
									{
										int idx = j + k - r;
										idx = border_e(idx, vmax) * wstep;
										__m256 ms;
										if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
										else ms = mVal;
										__m256 mg = _mm256_set1_ps(gauss[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
								}
								else
#endif
								{
									const int e = -(src.rows - j) + r;
									float* sii = si + (j - r) * wstep;
									for (int k = 0; k < e + 1; k++)
									{
										__m256 ms = _mm256_load_ps(sii);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										sii += wstep;
									}
									for (int k = e + 1; k < ksize; k++)
									{
										int idx = border_e(j + k - r, vmax) * wstep;
										__m256 ms = _mm256_load_ps(si + idx);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
								}
								_mm256_store_ps(b + i, mv);
							}

							//h filter
							{
								float* d = dest.ptr<float>(j);
								for (int i = 0; i < r; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
									if (border == BORDER_CONSTANT)
									{
										for (int k = 0; k < r - i; k++)
										{
											int idx = i + k;
											int maskIdx = max(0, k + i - r + 8);
											__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									else
#endif
									{
										int idx = i;
										for (int k = 0; k < r - i; k++)
										{
											__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss32F[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
											idx++;
										}
									}
									float* bi = b;
									for (int k = r - i; k < ksize; k++)
									{
										__m256 ms = _mm256_load_ps(bi);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										bi++;
									}
									_mm256_storeu_ps(d + i, mv);
								}
								for (int i = r; i < src.cols - r; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
									float* bi = b + i - r;
									for (int k = 0; k < ksize; k++)
									{
										__m256 ms = _mm256_load_ps(bi);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										bi++;
									}
									_mm256_storeu_ps(d + i, mv);
								}
								int v = r / 8;
								//for (int i = src.cols - r; i < src.cols-v; i += 8)
								for (int vv = 0; vv < v; vv++)//orverrun
								{
									int i = src.cols - r + 8 * vv;
									__m256 mv = _mm256_setzero_ps();
									float* bi = b + i - r;
									for (int k = 0; k < src.cols - 7 + r - i; k++)
									{
										__m256 ms = _mm256_load_ps(bi);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										bi++;
									}
#ifdef BORDER_CONSTANT
									if (border == BORDER_CONSTANT)
									{
										for (int k = src.cols - 7 + r - i; k < ksize; k++)
										{
											int idx = k - src.cols + 7 - r + i;
											int maskIdx = min(k - (src.cols - 7 + r - i), 7);
											__m256 ms = _mm256_mask_i32gather_ps(mVal, b, end_access_pattern[idx], mMask_e[maskIdx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									else
#endif
									{
										int idx = 0;
										for (int k = src.cols - 7 + r - i; k < ksize; k++)
										{
											__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss32F[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
											idx++;
										}
									}
									_mm256_storeu_ps(d + i, mv);
								}
								for (int i = src.cols - r + 8 * v; i < src.cols; i++)
								{
									float v = 0.f;
									for (int k = 0; k < ksize; k++)
									{
										int idx = i + k - r;
										idx = border_e(idx, hmax);
										v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
									}
									d[i] = v;
								}
							}
						}
						else
						{
							//v filter
							for (int i = 0; i < src.cols; i += 8)
							{
								__m256 mv = _mm256_setzero_ps();
								float* si = s + i + (j - r) * wstep;

								for (int k = 0; k < ksize; k++)
								{
									__m256 ms = _mm256_loadu_ps(si);
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
									si += wstep;
								}
								_mm256_store_ps(b + i, mv);
							}

							//h filter
							{
								float* d = dest.ptr<float>(j);
								for (int i = 0; i < r; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
									if (border == BORDER_CONSTANT)
									{
										for (int k = 0; k < r - i; k++)
										{
											int idx = i + k;
											int maskIdx = max(0, k + i - r + 8);
											__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									else
#endif
									{
										int idx = i;
										for (int k = 0; k < r - i; k++)
										{
											__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss32F[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
											idx++;
										}
									}
									float* bi = b;
									for (int k = r - i; k < ksize; k++)
									{
										__m256 ms = _mm256_load_ps(bi);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										bi++;
									}
									_mm256_storeu_ps(d + i, mv);
								}

								for (int i = R; i < src.cols - r; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
									float* bi = b + i - r;
									for (int k = 0; k < ksize; k++)
									{
										__m256 ms = _mm256_load_ps(bi);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										bi++;
									}
									_mm256_storeu_ps(d + i, mv);
								}
								int v = r / 8;
								//for (int i = src.cols - r; i < src.cols-v; i += 8)
								for (int vv = 0; vv < v; vv++)//orverrun
								{
									int i = src.cols - r + 8 * vv;
									__m256 mv = _mm256_setzero_ps();
									float* bi = b + i - r;
									for (int k = 0; k < src.cols - 7 + r - i; k++)
									{
										__m256 ms = _mm256_load_ps(bi);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										bi++;
									}
#ifdef BORDER_CONSTANT
									if (border == BORDER_CONSTANT)
									{
										for (int k = src.cols - 7 + r - i; k < ksize; k++)
										{
											int idx = k - src.cols + 7 - r + i;
											int maskIdx = min(k - (src.cols - 7 + r - i), 7);
											__m256 ms = _mm256_mask_i32gather_ps(mVal, b, end_access_pattern[idx], mMask_e[maskIdx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									else
#endif
									{
										for (int k = src.cols - 7 + r - i; k < ksize; k++)
										{
											int idx = k - src.cols + 7 - r + i;

											__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss32F[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									_mm256_storeu_ps(d + i, mv);
								}

								for (int i = src.cols - r + 8 * v; i < src.cols; i++)
								{
									float v = 0.f;
									for (int k = 0; k < ksize; k++)
									{
										int idx = i + k - r;
										idx = border_e(idx, hmax);
										v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
									}
									d[i] = v;
								}
							}
						}
					}
				}
			}
			else
			{

				for (int n = 0; n < max_core; n++)
				{
					const int tidx = 0;
					const int strip = src.rows / max_core;
					int start = n * strip;
					int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
					const int simdwidth = get_simd_ceil(src.cols, 8);
					if (!useAllocBuffer) bufferLineCols[tidx].release();
					if (bufferLineCols[tidx].size() != Size(simdwidth, 1)) bufferLineCols[tidx].create(simdwidth, 1, CV_32F);
					float* b = bufferLineCols[tidx].ptr<float>(0);
					for (int j = start; j < end; j++)
					{
						if (j < r)
						{
							//v filter
							for (int i = 0; i < src.cols; i += 8)
							{
								float* si = s + i;
								__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
								if (border == BORDER_CONSTANT)
								{
									for (int k = 0; k < ksize; k++)
									{
										int idx = j + k - r;
										idx = border_s(idx) * wstep;
										__m256 ms;
										if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
										else ms = mVal;
										__m256 mg = _mm256_set1_ps(gauss[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
								}
								else
#endif
								{
									const int e = j + r;
									for (int k = 0; k < e + 1; k++)
									{
										int idx = border_s(j + k - r) * wstep;
										__m256 ms = _mm256_loadu_ps(si + idx);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
									float* sii = si + wstep * (j + e + 1 - r);
									for (int k = e + 1; k < ksize; k++)
									{
										__m256 ms = _mm256_loadu_ps(sii);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
										sii += wstep;
									}
								}
								_mm256_store_ps(b + i, mv);
							}

							//h filter
							{
								float* d = dest.ptr<float>(j);

								for (int i = 0; i < r; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
									if (border == BORDER_CONSTANT)
									{
										for (int k = 0; k < r - i; k++)
										{
											int idx = i + k;
											int maskIdx = max(0, k + i - r + 8);
											__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									else
#endif
									{
										for (int k = 0; k < r - i; k++)
										{
											int idx = i + k;
											__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss32F[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									for (int k = r - i; k < ksize; k++)
									{
										int idx = i + k - r;
										__m256 ms = _mm256_load_ps(b + idx);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
									_mm256_storeu_ps(d + i, mv);
								}

								for (int i = R; i < simdend; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
									for (int k = 0; k < ksize; k++)
									{
										int idx = i + k - r;
										__m256 ms = _mm256_loadu_ps(b + idx);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
									_mm256_storeu_ps(d + i, mv);
								}
								//i=simdend i+=8
								__m256 mv = _mm256_setzero_ps();
								for (int k = 0; k < r + 1; k++)
								{
									int idx = simdend + k - r;
									__m256 ms = _mm256_load_ps(b + idx);
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
								for (int k = r + 1; k < ksize; k++)
								{
									int idx = k - r - 1;
									__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
								_mm256_storeu_ps(d + simdend, mv);

								for (int i = simdend + 8; i < src.cols; i++)
								{
									float v = 0.f;
									for (int k = 0; k < ksize; k++)
									{
										int idx = i + k - r;
										idx = border_e(idx, hmax);
										v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
									}
									d[i] = v;
								}
							}
						}
						else if (j > src.rows - r - 1)
						{
							//v filter
							for (int i = 0; i < src.cols; i += 8)
							{
								const float* si = s + i;
								__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
								if (border == BORDER_CONSTANT)
								{
									for (int k = 0; k < ksize; k++)
									{
										int idx = j + k - r;
										idx = border_s(idx) * wstep;
										__m256 ms;
										if (idx >= 0) ms = _mm256_loadu_ps(si + idx);
										else ms = mVal;
										__m256 mg = _mm256_set1_ps(gauss[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
								}
								else
#endif
								{
									for (int k = 0; k < ksize; k++)
									{
										int idx = j + k - r;
										idx = border_e(idx, vmax) * wstep;
										__m256 ms = _mm256_loadu_ps(si + idx);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
								}
								_mm256_store_ps(b + i, mv);
							}

							//h filter
							{
								float* d = dest.ptr<float>(j);
								for (int i = 0; i < r; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
									if (border == BORDER_CONSTANT)
									{
										for (int k = 0; k < r - i; k++)
										{
											int idx = i + k;
											int maskIdx = max(0, k + i - r + 8);
											__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									else
#endif
									{
										for (int k = 0; k < r - i; k++)
										{
											int idx = i + k;
											__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss32F[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									for (int k = r - i; k < ksize; k++)
									{
										int idx = i + k - r;
										__m256 ms = _mm256_load_ps(b + idx);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
									_mm256_storeu_ps(d + i, mv);
								}
								for (int i = R; i < simdend; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
									for (int k = 0; k < ksize; k++)
									{
										int idx = i + k - r;
										__m256 ms = _mm256_load_ps(b + idx);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
									_mm256_storeu_ps(d + i, mv);
								}
								//i = simdend; i+=8
								__m256 mv = _mm256_setzero_ps();
								for (int k = 0; k < r + 1; k++)
								{
									int idx = simdend + k - r;
									__m256 ms = _mm256_load_ps(b + idx);
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
								for (int k = r + 1; k < ksize; k++)
								{
									int idx = k - r - 1;
									__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
								_mm256_storeu_ps(d + simdend, mv);
								for (int i = simdend + 8; i < src.cols; i++)
								{
									float v = 0.f;
									for (int k = 0; k < ksize; k++)
									{
										int idx = i + k - r;
										idx = border_e(idx, hmax);
										v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
									}
									d[i] = v;
								}
							}
						}
						else
						{
							//v filter
							for (int i = 0; i < src.cols; i += 8)
							{
								__m256 mv = _mm256_setzero_ps();
								const float* si = s + i;

								for (int k = 0; k < ksize; k++)
								{
									int idx = (j + k - r) * wstep;
									__m256 ms = _mm256_loadu_ps(si + idx);
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
								_mm256_store_ps(b + i, mv);
							}

							//h filter
							{
								float* d = dest.ptr<float>(j);

								for (int i = 0; i < r; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
#ifdef BORDER_CONSTANT
									if (border == BORDER_CONSTANT)
									{
										for (int k = 0; k < r - i; k++)
										{
											int idx = i + k;
											int maskIdx = max(0, k + i - r + 8);
											__m256 ms = _mm256_mask_i32gather_ps(mVal, b, start_access_pattern[idx], mMask_s[maskIdx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									else
#endif
									{
										for (int k = 0; k < r - i; k++)
										{
											int idx = i + k;
											__m256 ms = _mm256_i32gather_ps(b, start_access_pattern[idx], sizeof(float));
											__m256 mg = _mm256_set1_ps(gauss32F[k]);
											mv = _mm256_fmadd_ps(ms, mg, mv);
										}
									}
									for (int k = r - i; k < ksize; k++)
									{
										int idx = i + k - r;
										__m256 ms = _mm256_load_ps(b + idx);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
									_mm256_storeu_ps(d + i, mv);
								}
								for (int i = R; i < simdend; i += 8)
								{
									__m256 mv = _mm256_setzero_ps();
									for (int k = 0; k < ksize; k++)
									{
										int idx = i + k - r;
										__m256 ms = _mm256_load_ps(b + idx);
										__m256 mg = _mm256_set1_ps(gauss32F[k]);
										mv = _mm256_fmadd_ps(ms, mg, mv);
									}
									_mm256_storeu_ps(d + i, mv);
								}
								//i = simdend;i+=8;
								__m256 mv = _mm256_setzero_ps();
								for (int k = 0; k < r + 1; k++)
								{
									int idx = simdend + k - r;
									__m256 ms = _mm256_load_ps(b + idx);
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
								for (int k = r + 1; k < ksize; k++)
								{
									int idx = k - r - 1;
									__m256 ms = _mm256_i32gather_ps(b, end_access_pattern[idx], sizeof(float));
									__m256 mg = _mm256_set1_ps(gauss32F[k]);
									mv = _mm256_fmadd_ps(ms, mg, mv);
								}
								_mm256_storeu_ps(d + simdend, mv);
								for (int i = simdend + 8; i < src.cols; i++)
								{
									float v = 0.f;
									for (int k = 0; k < ksize; k++)
									{
										int idx = i + k - r;
										idx = border_e(idx, hmax);
										v += (idx >= 0) ? gauss32F[k] * b[idx] : gauss32F[k] * constVal;
									}
									d[i] = v;
								}
							}
						}
					}
				}
			}
			_mm_free(access_pattern);
#ifdef BORDER_CONSTANT
			_mm_free(mMask_s);
			_mm_free(mMask_e);
#endif
		}
	}

	void GaussianFilterSeparableFIR::filterHVBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer)
		{
			bufferImageBorder.release();
			bufferImageBorder2.release();
		}
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		if (bufferImageBorder2.size() != bufferImageBorder.size()) bufferImageBorder2.create(bufferImageBorder.size(), CV_32F);
		const int wstep = bufferImageBorder.cols;

		if (opt == VECTOR_WITHOUT)
		{

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j) + r;
				float* d = bufferImageBorder2.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + (k - r)];
					}
					d[i] = v;
				}
			}

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder2.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k * wstep];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j) + r;
				float* d = bufferImageBorder2.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i - r;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
			//v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder2.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVNonRasterBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer)
		{
			bufferImageBorder.release();
			bufferImageBorder2.release();
		}
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r + rem, r, r, border);
		else copyMakeBorder(src, bufferImageBorder, r, r + rem, r, r, border);
		if (bufferImageBorder2.size() != bufferImageBorder.size()) bufferImageBorder2.create((bufferImageBorder.size()), CV_32F);
		const int wstep = bufferImageBorder.cols;

		if (opt == VECTOR_WITHOUT)
		{

			for (int j = 0; j < src.cols; j++)
			{
				float* s = bufferImageBorder.ptr<float>(0) + j;
				float* d = bufferImageBorder2.ptr<float>(0) + j;
				for (int i = 0; i < bufferImageBorder.rows; i++)
				{
					float v = 0.f;
					float* si = s + i * wstep;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * si[k];
					}
					d[i * wstep] = v;
				}
			}

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder2.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k * wstep];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const __m256i midx = _mm256_setr_epi32(0, wstep, 2 * wstep, 3 * wstep, 4 * wstep, 5 * wstep, 6 * wstep, 7 * wstep);
			//h filter

			for (int j = 0; j < src.cols; j++)
			{
				float* s = bufferImageBorder.ptr<float>(0) + j;
				float* d = bufferImageBorder2.ptr<float>(0) + j;
				for (int i = 0; i < bufferImageBorder.rows; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i * wstep;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(si + k, midx, sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
					}
					d[(i + 0) * wstep] = ((float*)&mv)[0];
					d[(i + 1) * wstep] = ((float*)&mv)[1];
					d[(i + 2) * wstep] = ((float*)&mv)[2];
					d[(i + 3) * wstep] = ((float*)&mv)[3];
					d[(i + 4) * wstep] = ((float*)&mv)[4];
					d[(i + 5) * wstep] = ((float*)&mv)[5];
					d[(i + 6) * wstep] = ((float*)&mv)[6];
					d[(i + 7) * wstep] = ((float*)&mv)[7];
				}
			}
			//v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder2.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(s + i + k * wstep);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
					}
					_mm256_storeu_ps(d + i, mv);
				}
			}
		}
	}


	void GaussianFilterSeparableFIR::filterHVDelayedBorderLJ(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		const int wstep = bufferImageBorder.cols;

		float* vv = (float*)_mm_malloc(ksize * sizeof(float), 32);
		_mm_free(vv);
		if (opt == VECTOR_WITHOUT)
		{

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);

				for (int k = 0; k < ksize; k++)
				{
					//float v = 0.f;

					for (int i = 0; i < src.cols; i++)
					{
						//v += gauss[k] * s[i + k];
						vv[k] += gauss32F[k] * s[i];
					}
					//vv[0]
				}
				//s[i] = v;
			}

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k * wstep];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(s + i, mv);
				}
			}
			//v filter
			const int max_core = 1;

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;

				float* s = bufferImageBorder.ptr<float>(start);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += wstep;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += wstep;
					d += dest.cols;
				}
			}
			/*

			for (int j = 0; j < src.rows; j++)
			{
			float* s = bufferImageBorder.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 0; i < src.cols; i += 8)
			{
			__m256 mv = _mm256_setzero_ps();
			float* si = s + i;
			for (int k = 0; k < ksize; k++)
			{
			__m256 ms = _mm256_load_ps(si);
			__m256 mg = _mm256_set1_ps(gauss[k]);
			mv = _mm256_fmadd_ps(ms, mg, mv);
			si += wstep;
			}
			_mm256_store_ps(d + i, mv);
			}
			}
			*/
		}
	}

	//modefied
	void GaussianFilterSeparableFIR::filterHVDelayedBorder32F(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.depth() == CV_32F);

		const int ksize = 2 * r + 1;
		int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		const int wstep = bufferImageBorder.cols;

		if (opt == VECTOR_WITHOUT)
		{

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k];
					}
					s[i] = v;
				}
			}

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k * wstep];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter
			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(s + i, mv);
				}
			}
			//v filter
			{
				const int start = 0;
				const int end = src.rows;

				float* s = bufferImageBorder.ptr<float>(start);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += wstep;
						}
						_mm256_store_ps(d + i, mv);
					}
					s += wstep;
					d += dest.cols;
				}
			}
			/*

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
			*/
		}
	}

	void GaussianFilterSeparableFIR::filterHVDelayedBorderSort32F(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.depth() == CV_32F);

		const int ksize = 2 * r + 1;
		int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		const int wstep = bufferImageBorder.cols;

		if (opt == VECTOR_WITHOUT)
		{

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = gauss32F[r] * s[i + r];
					for (int k = 0; k < r; k++)
					{
						v += gauss32F[k] * s[i + k];
						v += gauss32F[k] * s[i + ksize - 1 - k];
					}
					s[i] = v;
				}
			}

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = gauss32F[r] * s[i + r * wstep];
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k * wstep];
						v += gauss32F[k] * s[i + (ksize - 1 - k) * wstep];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter
			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					float* si = s + i;
					__m256 ms;
					__m256 mg;
					__m256 mv = _mm256_setzero_ps();
					for (int k = 0; k < r; k++)
					{
						ms = _mm256_loadu_ps(si + k);
						mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);

						ms = _mm256_loadu_ps(si + ksize - 1 - k);
						mv = _mm256_fmadd_ps(ms, mg, mv);
					}
					ms = _mm256_loadu_ps(si + r);
					mg = _mm256_set1_ps(gauss32F[r]);
					mv = _mm256_fmadd_ps(ms, mg, mv);
					_mm256_store_ps(s + i, mv);
				}
			}
			//v filter
			{
				const int start = 0;
				const int end = src.rows;

				float* s = bufferImageBorder.ptr<float>(start);
				float* d = dest.ptr<float>(start);
				for (int j = start; j < end; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						float* si = s + i;
						__m256 ms;
						__m256 mg;
						__m256 mv = _mm256_setzero_ps();
						for (int k = 0; k < ksize; k++)
						{
							ms = _mm256_load_ps(si + wstep * k);
							mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);

							ms = _mm256_load_ps(si + wstep * (ksize - 1 - k));
							mv = _mm256_fmadd_ps(ms, mg, mv);
						}
						ms = _mm256_loadu_ps(si + wstep * r);
						mg = _mm256_set1_ps(gauss32F[r]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						_mm256_store_ps(d + i, mv);
					}
					s += wstep;
					d += dest.cols;
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVDelayedBorder64F(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.depth() == CV_64F);

		const int ksize = 2 * r + 1;
		int rem = get_simd_ceil(src.cols + 2 * r, 4) - (src.cols + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		const int wstep = bufferImageBorder.cols;

		if (opt == VECTOR_WITHOUT)
		{
			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				double* s = bufferImageBorder.ptr<double>(j);
				for (int i = 0; i < src.cols; i++)
				{
					double v = 0.0;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss64F[k] * s[i + k];
					}
					s[i] = v;
				}
			}

			for (int j = 0; j < src.rows; j++)
			{
				double* s = bufferImageBorder.ptr<double>(j);
				double* d = dest.ptr<double>(j);
				for (int i = 0; i < src.cols; i++)
				{
					double v = 0.0;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss64F[k] * s[i + k * wstep];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter
			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				double* s = bufferImageBorder.ptr<double>(j);
				for (int i = 0; i < src.cols; i += 4)
				{
					__m256d mv = _mm256_setzero_pd();
					double* si = s + i;

					for (int k = 0; k < ksize; k++)
					{
						__m256d ms = _mm256_loadu_pd(si);
						__m256d mg = _mm256_set1_pd(gauss64F[k]);
						mv = _mm256_fmadd_pd(ms, mg, mv);
						si++;
					}

					_mm256_store_pd(s + i, mv);
				}
			}
			//v filter
			{
				const int start = 0;
				const int end = src.rows;

				double* s = bufferImageBorder.ptr<double>(start);
				double* d = dest.ptr<double>(start);
				for (int j = start; j < end; j++)
				{
					for (int i = 0; i < src.cols; i += 4)
					{
						__m256d mv = _mm256_setzero_pd();
						double* si = s + i;

						for (int k = 0; k < ksize; k++)
						{
							__m256d ms = _mm256_load_pd(si);
							__m256d mg = _mm256_set1_pd(gauss64F[k]);
							mv = _mm256_fmadd_pd(ms, mg, mv);
							si += wstep;
						}

						_mm256_store_pd(d + i, mv);
					}
					s += wstep;
					d += dest.cols;
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVDelayedBorderSort64F(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		CV_Assert(src.depth() == CV_64F);
		const int ksize = 2 * r + 1;
		int rem = get_simd_ceil(src.cols + 2 * r, 4) - (src.cols + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		const int wstep = bufferImageBorder.cols;

		if (opt == VECTOR_WITHOUT)
		{
			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				double* s = bufferImageBorder.ptr<double>(j);
				for (int i = 0; i < src.cols; i++)
				{
					double v = 0.0;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss64F[k] * s[i + k];
					}
					s[i] = v;
				}
			}

			for (int j = 0; j < src.rows; j++)
			{
				double* s = bufferImageBorder.ptr<double>(j);
				double* d = dest.ptr<double>(j);
				for (int i = 0; i < src.cols; i++)
				{
					double v = 0.0;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss64F[k] * s[i + k * wstep];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter
			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				double* s = bufferImageBorder.ptr<double>(j);
				for (int i = 0; i < src.cols; i += 4)
				{
					double* si = s + i;
					__m256d ms;
					__m256d mg;
					__m256d mv = _mm256_setzero_pd();
					for (int k = 0; k < r; k++)
					{
						ms = _mm256_loadu_pd(si + k);
						mg = _mm256_set1_pd(gauss64F[k]);
						mv = _mm256_fmadd_pd(ms, mg, mv);

						ms = _mm256_loadu_pd(si + ksize - 1 - k);
						mv = _mm256_fmadd_pd(ms, mg, mv);
					}
					ms = _mm256_loadu_pd(si + r);
					mg = _mm256_set1_pd(gauss64F[r]);
					mv = _mm256_fmadd_pd(mg, ms, mv);

					_mm256_store_pd(s + i, mv);
				}
			}
			//v filter
			{
				const int start = 0;
				const int end = src.rows;

				double* s = bufferImageBorder.ptr<double>(start);
				double* d = dest.ptr<double>(start);
				for (int j = start; j < end; j++)
				{
					for (int i = 0; i < src.cols; i += 4)
					{
						double* si = s + i;
						__m256d ms;
						__m256d mg;
						__m256d mv = _mm256_setzero_pd();
						for (int k = 0; k < r; k++)
						{
							ms = _mm256_load_pd(si + wstep * k);
							mg = _mm256_set1_pd(gauss64F[k]);
							mv = _mm256_fmadd_pd(ms, mg, mv);

							ms = _mm256_load_pd(si + wstep * (ksize - k - 1));
							mv = _mm256_fmadd_pd(ms, mg, mv);
						}
						ms = _mm256_load_pd(si + wstep * r);
						mg = _mm256_set1_pd(gauss64F[r]);
						mv = _mm256_fmadd_pd(mg, ms, mv);
						_mm256_store_pd(d + i, mv);
					}
					s += wstep;
					d += dest.cols;
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVDelayedBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const bool isSort = false;
		if (isSort)
		{
			if (src.depth() == CV_32F)filterHVDelayedBorderSort32F(src, dest, r, sigma, border, opt, useAllocBuffer);
			if (src.depth() == CV_64F)filterHVDelayedBorderSort64F(src, dest, r, sigma, border, opt, useAllocBuffer);
		}
		else
		{
			if (src.depth() == CV_32F)filterHVDelayedBorder32F(src, dest, r, sigma, border, opt, useAllocBuffer);
			if (src.depth() == CV_64F)filterHVDelayedBorder64F(src, dest, r, sigma, border, opt, useAllocBuffer);
		}
	}


	void GaussianFilterSeparableFIR::filterHVDelayedVPBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		const int wstep = bufferImageBorder.cols;

		if (opt == VECTOR_WITHOUT)
		{

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k];
					}
					s[i] = v;
				}
			}

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k * wstep];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(s + i, mv);
				}
			}
			//v filter
			const int dstep = dest.cols;
			const int dstep0 = dstep * 0;
			const int dstep1 = dstep * 1;
			const int dstep2 = dstep * 2;
			const int dstep3 = dstep * 3;
			const int dstep4 = dstep * 4;
			const int dstep5 = dstep * 5;
			const int dstep6 = dstep * 6;
			const int dstep7 = dstep * 7;
			const int dstep8 = dstep * 8;
			const int max_core = 1;

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferLineRows[tidx].release();
				if (bufferLineRows[tidx].size() != Size(bufferImageBorder.rows, 1))bufferLineRows[tidx].create(Size(bufferImageBorder.rows, 1), CV_32F);
				float* b = bufferLineRows[tidx].ptr<float>(0);
				float* dptr = dest.ptr<float>(0);
				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;

				for (int i = start; i < end; i++)
				{
					verticalLineCopy(bufferImageBorder, bufferLineRows[tidx], i);
					float* d = dptr + i;
					for (int j = 0; j < src.rows; j += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bj = b + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_loadu_ps(bj);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bj++;
						}
						d[dstep0] = ((float*)&mv)[0];
						d[dstep1] = ((float*)&mv)[1];
						d[dstep2] = ((float*)&mv)[2];
						d[dstep3] = ((float*)&mv)[3];
						d[dstep4] = ((float*)&mv)[4];
						d[dstep5] = ((float*)&mv)[5];
						d[dstep6] = ((float*)&mv)[6];
						d[dstep7] = ((float*)&mv)[7];
						d += dstep8;
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		if (!useAllocBuffer)
		{
			bufferImageBorder.release();
			bufferImageBorder2.release();
		}
		const int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		Size asize = Size(bufferImageBorder.cols, src.rows);
		if (bufferImageBorder2.size() != asize)bufferImageBorder2.create(asize, bufferImageBorder.type());

		const int wstep = bufferImageBorder.cols;
		if (opt == VECTOR_WITHOUT)
		{

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j + r);
				float* d = bufferImageBorder2.ptr<float>(j + r);

				for (int i = 0; i < bufferImageBorder.cols; i++)
				{
					float* si = s + i;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * si[(k - r) * wstep];
					}
					d[i] = v;
				}
			}

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder2.ptr<float>(j + r);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float* si = s + i;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * si[k];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//v filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = bufferImageBorder2.ptr<float>(j);
				for (int i = 0; i < bufferImageBorder.cols; i += 8)
				{
					float* si = s + i;
					__m256 mv = _mm256_setzero_ps();
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					_mm256_store_ps(d + i, mv);
				}
			}

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder2.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					float* si = s + i;
					__m256 mv = _mm256_setzero_ps();
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHDelayedBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		if (!useAllocBuffer)bufferImageBorder.release();
		const int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		const int wstep = bufferImageBorder.cols;
		if (opt == VECTOR_WITHOUT)
		{
			//v filter
			for (int j = 0; j < bufferImageBorder.cols; j++)
			{
				float* s = bufferImageBorder.ptr<float>(0) + j;
				for (int i = 0; i < src.rows; i++)
				{
					float* si = s + i * wstep;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * si[k * wstep];
					}
					s[i * wstep] = v;
				}
			}
			//h filter

			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float* si = s + i;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * si[k];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//v filter
			const __m256i midx = _mm256_setr_epi32(0, wstep, 2 * wstep, 3 * wstep, 4 * wstep, 5 * wstep, 6 * wstep, 7 * wstep);
			const int wstep0 = wstep * 0;
			const int wstep1 = wstep * 1;
			const int wstep2 = wstep * 2;
			const int wstep3 = wstep * 3;
			const int wstep4 = wstep * 4;
			const int wstep5 = wstep * 5;
			const int wstep6 = wstep * 6;
			const int wstep7 = wstep * 7;
			const int wstep8 = wstep * 8;

			for (int i = 0; i < bufferImageBorder.cols; i++)
			{
				float* s = bufferImageBorder.ptr<float>(0) + i;
				for (int j = 0; j < src.rows; j += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(si, midx, sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					s[wstep0] = ((float*)&mv)[0];
					s[wstep1] = ((float*)&mv)[1];
					s[wstep2] = ((float*)&mv)[2];
					s[wstep3] = ((float*)&mv)[3];
					s[wstep4] = ((float*)&mv)[4];
					s[wstep5] = ((float*)&mv)[5];
					s[wstep6] = ((float*)&mv)[6];
					s[wstep7] = ((float*)&mv)[7];
					s += wstep8;
				}
			}

			// h filter
			for (int j = 0; j < src.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVIBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int rem = get_simd_ceil(src.rows + 2 * r, 8) - (src.rows + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r + rem, r, r, border);
		else copyMakeBorder(src, bufferImageBorder, r, r + rem, r, r, border);

		const int wstep = bufferImageBorder.cols;
		const int wstep2 = src.cols;
		if (opt == VECTOR_WITHOUT)
		{

			for (int i = 0; i < src.cols; i++)
			{
				Mat buff(bufferImageBorder.rows, 1, CV_32F);
				float* b = buff.ptr<float>(0);

				for (int j = 0; j < bufferImageBorder.rows; j++)
				{
					float* s = bufferImageBorder.ptr<float>(j) + i;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[k];
					}
					b[j] = v;
				}

				for (int j = 0; j < src.rows; j++)
				{
					float* d = dest.ptr<float>(j);
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * b[k + j];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const __m256i midx = _mm256_setr_epi32(0 * wstep, 1 * wstep, 2 * wstep, 3 * wstep, 4 * wstep, 5 * wstep, 6 * wstep, 7 * wstep);
			const int max_core = 1;
			const int wstep20 = 0 * wstep2;
			const int wstep21 = 1 * wstep2;
			const int wstep22 = 2 * wstep2;
			const int wstep23 = 3 * wstep2;
			const int wstep24 = 4 * wstep2;
			const int wstep25 = 5 * wstep2;
			const int wstep26 = 6 * wstep2;
			const int wstep27 = 7 * wstep2;

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.cols / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.cols : (n + 1) * strip;

				const int tidx = 0;
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				if (bufferLineCols[tidx].size() != Size(bufferImageBorder.rows, 1))bufferLineCols[tidx].create(Size(bufferImageBorder.rows, 1), CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);
				for (int i = start; i < end; i++)
				{
					for (int j = 0; j < bufferImageBorder.rows; j += 8)
					{
						float* si = bufferImageBorder.ptr<float>(j) + i;
						__m256 mv = _mm256_setzero_ps();
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_i32gather_ps(si, midx, sizeof(float));
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si++;
						}
						_mm256_store_ps(b + j, mv);
					}
					for (int j = 0; j < src.rows; j += 8)
					{
						float* d = dest.ptr<float>(j);
						__m256 mv = _mm256_setzero_ps();
						float* bi = b + j;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						d[i + wstep20] = ((float*)&mv)[0];
						d[i + wstep21] = ((float*)&mv)[1];
						d[i + wstep22] = ((float*)&mv)[2];
						d[i + wstep23] = ((float*)&mv)[3];
						d[i + wstep24] = ((float*)&mv)[4];
						d[i + wstep25] = ((float*)&mv)[5];
						d[i + wstep26] = ((float*)&mv)[6];
						d[i + wstep27] = ((float*)&mv)[7];
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHIBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer) bufferImageBorder.release();
		//if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		//else 
			copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		const int wstep = bufferImageBorder.cols;
		if (opt == VECTOR_WITHOUT)
		{
			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(bufferImageBorder.cols, 1, CV_32F);
				float* s = bufferImageBorder.ptr<float>(j);
				float* b = buff.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < bufferImageBorder.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k * wstep];
					}
					b[i] = v;
				}
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * b[i + k];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int max_core = 1;
			for (int n = 0; n < max_core; n++)
			{
				//const int tidx = 0;
				const int strip = src.rows / max_core;
				const int start = n * strip;
				const int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				//if (!useAllocBuffer)bufferLineCols[tidx].release();
				//if (bufferLineCols[tidx].size() != Size(bufferImageBorder.cols, 1)) bufferLineCols[tidx].create(bufferImageBorder.cols, 1, CV_32F);
				//float* b = bufferLineCols[tidx].ptr<float>(0);
				AutoBuffer<float> bufferLineCols(bufferImageBorder.cols);
				float* b = &bufferLineCols[0];

				for (int j = start; j < end; j++)
				{
					float* s = bufferImageBorder.ptr<float>(j);
					float* d = dest.ptr<float>(j);
#if 1 //unroll 1
					for (int i = 0; i < bufferImageBorder.cols; i += 8)
					{
						float* si = s + i;
						__m256 mv = _mm256_mul_ps(_mm256_set1_ps(gauss32F[0]), _mm256_load_ps(si));
						si += wstep;
						for (int k = 1; k < ksize; k++)
						{
							const __m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(mg, _mm256_load_ps(si), mv);
							si += wstep;
						}
						_mm256_store_ps(b + i, mv);
					}
					for (int i = 0; i < src.cols; i += 8)
					{
						float* bi = b + i;
						__m256 mv = _mm256_mul_ps(_mm256_set1_ps(gauss32F[0]), _mm256_load_ps(bi));
						bi++;
						for (int k = 1; k < ksize; k++)
						{
							const __m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(mg, _mm256_loadu_ps(bi), mv);
							bi++;
						}
						_mm256_store_ps(d + i, mv);
					}
#else
					for (int i = 0; i < bufferImageBorder.cols; i += 16)
					{
						float* si = s + i;
						__m256 mv1 = _mm256_mul_ps(_mm256_set1_ps(gauss32F[0]), _mm256_load_ps(si + 0));
						__m256 mv2 = _mm256_mul_ps(_mm256_set1_ps(gauss32F[0]), _mm256_load_ps(si + 8));
						si += wstep;
						for (int k = 1; k < ksize; k++)
						{
							const __m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv1 = _mm256_fmadd_ps(mg, _mm256_load_ps(si + 0), mv1);
							mv2 = _mm256_fmadd_ps(mg, _mm256_load_ps(si + 8), mv2);
							si += wstep;
						}
						_mm256_store_ps(b + i + 0, mv1);
						_mm256_store_ps(b + i + 8, mv2);
					}
					for (int i = 0; i < src.cols; i += 16)
					{
						float* bi = b + i;
						__m256 mv1 = _mm256_mul_ps(_mm256_set1_ps(gauss32F[0]), _mm256_load_ps(bi + 0));
						__m256 mv2 = _mm256_mul_ps(_mm256_set1_ps(gauss32F[0]), _mm256_load_ps(bi + 8));
						bi++;
						for (int k = 1; k < ksize; k++)
						{
							const __m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv1 = _mm256_fmadd_ps(mg, _mm256_loadu_ps(bi + 0), mv1);
							mv2 = _mm256_fmadd_ps(mg, _mm256_loadu_ps(bi + 8), mv2);
							bi++;
						}
						_mm256_store_ps(d + i + 0, mv1);
						_mm256_store_ps(d + i + 8, mv2);
					}
#endif
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHIBlockBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		const int wstep = bufferImageBorder.cols;

		if (opt == VECTOR_WITHOUT)
		{

			for (int j = 0; j < src.rows; j++)
			{
				Mat buff(bufferImageBorder.cols, 1, CV_32F);
				float* s = bufferImageBorder.ptr<float>(j);
				float* b = buff.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < bufferImageBorder.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k * wstep];
					}
					b[i] = v;
				}
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * b[i + k];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int max_core = 1;

			for (int n = 0; n < max_core; n++)
			{
				const int strip = src.rows / max_core;
				int start = n * strip;
				int end = (n == max_core - 1) ? src.rows : (n + 1) * strip;
				int tidx = 0;
				if (!useAllocBuffer)bufferLineCols[tidx].release();
				if (bufferLineCols[tidx].size() != Size(bufferImageBorder.cols, 1)) bufferLineCols[tidx].create(bufferImageBorder.cols, 1, CV_32F);
				float* b = bufferLineCols[tidx].ptr<float>(0);

				for (int j = start; j < end; j++)
				{
					float* s = bufferImageBorder.ptr<float>(j);
					float* d = dest.ptr<float>(j);

					int startx = 0;
					int endx = src.cols / 2;
					//vfilter
					for (int i = 0; i < endx + 2 * r; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += wstep;
						}
						_mm256_store_ps(b + i, mv);
					}
					//hfilter
					for (int i = 0; i < endx; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = b + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_storeu_ps(d + i, mv);
					}
					startx = endx;
					endx = src.cols;
					//vfilter
					for (int i = startx + 2 * r; i < endx + 2 * r; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* si = s + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(si);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							si += wstep;
						}
						_mm256_storeu_ps(b + i, mv);
					}
					//hfilter
					for (int i = startx; i < endx; i += 8)
					{
						__m256 mv = _mm256_setzero_ps();
						float* bi = b + i;
						for (int k = 0; k < ksize; k++)
						{
							__m256 ms = _mm256_load_ps(bi);
							__m256 mg = _mm256_set1_ps(gauss32F[k]);
							mv = _mm256_fmadd_ps(ms, mg, mv);
							bi++;
						}
						_mm256_storeu_ps(d + i, mv);
					}
				}

				/*
				for (int tilex = 0; tilex < src.cols; tilex += tileX)
				{
				int start = tilex;
				int end = tilex + tileX;
				//vfilter
				for (int i = 0; i < end + 2 * r; i += 8)
				{
				__m256 mv = _mm256_setzero_ps();
				float * si = s + i;
				for (int k = 0; k < ksize; k++)
				{
				__m256 ms = _mm256_load_ps(si);
				__m256 mg = _mm256_set1_ps(gauss[k]);
				mv = _mm256_fmadd_ps(ms, mg, mv);
				si += wstep;
				}
				_mm256_store_ps(b + i, mv);
				}
				//hfilter
				for (int i = 0; i < end; i += 8)
				{
				__m256 mv = _mm256_setzero_ps();
				for (int k = 0; k < ksize; k++)
				{
				__m256 ms = _mm256_load_ps(b + i + k);
				__m256 mg = _mm256_set1_ps(gauss[k]);
				mv = _mm256_fmadd_ps(ms, mg, mv);
				}
				_mm256_storeu_ps(d + i, mv);
				}
				}
				*/
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVTileBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		//const int wstep = bufferImageBorder.cols;
		const int max_core = 1;
		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);

		if (opt == VECTOR_WITHOUT)
		{

			for (int tiley = 0; tiley < src.rows; tiley += TILE_Y)
			{
				float* blurx = (float*)_mm_malloc(sizeof(float) * (TILE_X * (TILE_Y + 2 * r)), 32);
				for (int tilex = 0; tilex < src.cols; tilex += TILE_X)
				{
					for (int j = -r, idx = 0; j < TILE_Y + r; j++)
					{
						float* in = bufferImageBorder.ptr<float>(tiley + j + r) + tilex;
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * in[i + k];
							}
							blurx[idx++] = v;
						}
					}
					float* d = dest.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * blurx[i + (k + j) * TILE_X];
							}
							d[i] = v;
						}
						d += dest.cols;
					}
				}
				_mm_free(blurx);
			}
		}
		else if (opt == VECTOR_AVX)
		{

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(TILE_X, TILE_Y + 2 * r))bufferTile[tidx].create(Size(TILE_X, TILE_Y + 2 * r), CV_32F);
				float* blurxPtr = bufferTile[tidx].ptr<float>(0);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					//h filter
					float* blurx = (float*)blurxPtr;
					for (int j = -r; j < TILE_Y + r; j++)
					{
						float* in = bufferImageBorder.ptr<float>(tiley + j + r) + tilex;
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* s = in;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(s);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								s++;
							}
							_mm256_store_ps(blurx, mv);
							blurx += 8;
							in += 8;
						}
					}
					//v filter
					float* d = dest.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						const int JTY = j * TILE_X;
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							blurx = (float*)blurxPtr + JTY + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(blurx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								blurx += TILE_X;
							}
							_mm256_store_ps(d + i, mv);
						}
						d += dest.cols;
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHTileBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		const int wstep = bufferImageBorder.cols;
		const int max_core = 1;
		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);

		if (opt == VECTOR_WITHOUT)
		{

			for (int tiley = 0; tiley < src.rows; tiley += TILE_Y)
			{
				float* blurx = (float*)_mm_malloc(sizeof(float) * ((TILE_X + 2 * r) * (TILE_Y)), 32);
				for (int tilex = 0; tilex < src.cols; tilex += TILE_X)
				{
					for (int j = 0, idx = 0; j < TILE_Y; j++)
					{
						float* in = bufferImageBorder.ptr<float>(tiley + j) + tilex + r;
						for (int i = -r; i < TILE_X + r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * in[i + k * wstep];
							}
							blurx[idx++] = v;
						}
					}
					float* d = dest.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * blurx[j * (TILE_X + 2 * r) + i + k];
							}
							d[i] = v;
						}
						d += dest.cols;
					}
				}
				_mm_free(blurx);
			}
		}
		else if (opt == VECTOR_AVX)
		{

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;

				int simdtilex = get_simd_ceil(TILE_X + 2 * r, 8);
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(simdtilex, TILE_Y)) bufferTile[tidx].create(Size(simdtilex, TILE_Y), CV_32F);
				float* blurxPtr = bufferTile[tidx].ptr<float>(0);
				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;

					float* blurx = (float*)blurxPtr;
					for (int j = 0; j < TILE_Y; j++)
					{
						float* in = bufferImageBorder.ptr<float>(tiley + j) + tilex;
						for (int i = -r; i < TILE_X + r; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* inV = in;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(inV);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								inV += wstep;
							}
							_mm256_store_ps(blurx, mv);
							blurx += 8;
							in += 8;
						}
					}

					float* d = dest.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						const int JTY = j * simdtilex;
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							blurx = (float*)blurxPtr + JTY + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(blurx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								blurx++;
							}
							_mm256_storeu_ps(d + i, mv);
						}
						d += dest.cols;
					}

				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHITileBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int rem = get_simd_ceil(src.cols + 2 * r, 8) - (src.cols + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);
		else copyMakeBorder(src, bufferImageBorder, r, r, r, r + rem, border);

		const int wstep = bufferImageBorder.cols;
		const int max_core = 1;
		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);

		if (opt == VECTOR_WITHOUT)
		{

			for (int tiley = 0; tiley < src.rows; tiley += TILE_Y)
			{
				float* blury = (float*)_mm_malloc(sizeof(float) * ((TILE_X + 2 * r)), 32);
				for (int tilex = 0; tilex < src.cols; tilex += TILE_X)
				{
					float* d = dest.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						float* in = bufferImageBorder.ptr<float>(tiley + j) + tilex + r;
						for (int i = -r, idx = 0; i < TILE_X + r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * in[i + k * wstep];
							}
							blury[idx++] = v;
						}
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * blury[i + k];
							}
							d[i] = v;
						}
						d += dest.cols;
					}
				}
				_mm_free(blury);
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int simdend = get_simd_ceil_end(-r, TILE_X + r, 8);

			for (int n = 0; n < max_core; n++)
			{
				const int tidx = 0;
				const int simdtilex = get_simd_ceil(TILE_X + 2 * r, 8);
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(simdtilex, 1)) bufferTile[tidx].create(Size(simdtilex, 1), CV_32F);
				float* blurPtr = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* d = dest.ptr<float>(tiley) + tilex;

					for (int j = 0; j < TILE_Y; j++)
					{
						float* blury = (float*)blurPtr;
						float* in = bufferImageBorder.ptr<float>(tiley + j) + tilex;
						for (int i = -r; i < simdend; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* inV = in;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_load_ps(inV);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								inV += wstep;
							}
							_mm256_store_ps(blury, mv);
							blury += 8;
							in += 8;
						}
						for (int i = 0; i < TILE_X; i += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							blury = (float*)blurPtr + i;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(blury);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								blury++;
							}
							_mm256_store_ps(d + i, mv);
						}
						d += dest.cols;
					}
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVITileBorder(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int rem = get_simd_ceil(src.rows + 2 * r, 8) - (src.rows + 2 * r);
		if (!useAllocBuffer)bufferImageBorder.release();
		if (useParallelBorder)	myCopyMakeBorder(src, bufferImageBorder, r, r + rem, r, r, border);
		else copyMakeBorder(src, bufferImageBorder, r, r + rem, r, r, border);

		const int wstep = bufferImageBorder.cols;
		const int max_core = 1;
		const int TILE_X = src.cols / tileDiv.width;
		const int TILE_Y = src.rows / tileDiv.height;
		createTileIndex(TILE_X, TILE_Y);

		if (opt == VECTOR_WITHOUT)
		{

			for (int tiley = 0; tiley < src.rows; tiley += TILE_Y)
			{
				float* blury = (float*)_mm_malloc(sizeof(float) * ((TILE_X + 2 * r)), 32);
				for (int tilex = 0; tilex < src.cols; tilex += TILE_X)
				{
					float* d = dest.ptr<float>(tiley) + tilex;
					for (int j = 0; j < TILE_Y; j++)
					{
						float* in = bufferImageBorder.ptr<float>(tiley + j) + tilex + r;
						for (int i = -r, idx = 0; i < TILE_X + r; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * in[i + k * wstep];
							}
							blury[idx++] = v;
						}
						for (int i = 0; i < TILE_X; i++)
						{
							float v = 0.f;
							for (int k = 0; k < ksize; k++)
							{
								v += gauss32F[k] * blury[i + k];
							}
							d[i] = v;
						}
						d += dest.cols;
					}
				}
				_mm_free(blury);
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const int simdend = get_simd_ceil_end(-r, TILE_Y + r, 8);
			const int dstep0 = dest.cols * 0;
			const int dstep1 = dest.cols * 1;
			const int dstep2 = dest.cols * 2;
			const int dstep3 = dest.cols * 3;
			const int dstep4 = dest.cols * 4;
			const int dstep5 = dest.cols * 5;
			const int dstep6 = dest.cols * 6;
			const int dstep7 = dest.cols * 7;

			for (int n = 0; n < max_core; n++)
			{
				const __m256i midx = _mm256_setr_epi32(0 * wstep, 1 * wstep, 2 * wstep, 3 * wstep, 4 * wstep, 5 * wstep, 6 * wstep, 7 * wstep);
				const int tidx = 0;
				const int simdtilex = get_simd_ceil(TILE_Y + 2 * r, 8);
				if (!useAllocBuffer)bufferTile[tidx].release();
				if (bufferTile[tidx].size() != Size(simdtilex, 1)) bufferTile[tidx].create(Size(simdtilex, 1), CV_32F);
				float* blurPtr = bufferTile[tidx].ptr<float>(0);

				for (int t = 0; t < numTilesPerThread; t++)
				{
					int tilex = tileIndex[n * numTilesPerThread + t].x;
					int tiley = tileIndex[n * numTilesPerThread + t].y;
					float* d = dest.ptr<float>(tiley) + tilex;
					for (int i = 0; i < TILE_X; i++)
					{
						//h filter
						float* di = d + i;
						float* blurx = (float*)blurPtr;
						float* in = bufferImageBorder.ptr<float>(tiley) + tilex + i;
						for (int j = -r; j < simdend; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							float* inV = in;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_i32gather_ps(inV, midx, sizeof(float));
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								inV++;
							}
							_mm256_store_ps(blurx, mv);
							blurx += 8;
							in += bufferImageBorder.cols * 8;
						}
						//v filter
						float* dii = di;
						for (int j = 0; j < TILE_Y; j += 8)
						{
							__m256 mv = _mm256_setzero_ps();
							blurx = (float*)blurPtr + j;
							for (int k = 0; k < ksize; k++)
							{
								__m256 ms = _mm256_loadu_ps(blurx);
								__m256 mg = _mm256_set1_ps(gauss32F[k]);
								mv = _mm256_fmadd_ps(ms, mg, mv);
								blurx++;
							}
							dii[dstep0] = ((float*)&mv)[0];
							dii[dstep1] = ((float*)&mv)[1];
							dii[dstep2] = ((float*)&mv)[2];
							dii[dstep3] = ((float*)&mv)[3];
							dii[dstep4] = ((float*)&mv)[4];
							dii[dstep5] = ((float*)&mv)[5];
							dii[dstep6] = ((float*)&mv)[6];
							dii[dstep7] = ((float*)&mv)[7];
							dii += dest.cols * 8;
						}
					}
				}
			}
		}
	}

	void createSubImage(Mat& src, Mat& dest, Size div_size, Point idx, const int r, int borderType = 0)
	{
		const int tilex = src.cols / div_size.width;
		const int tiley = src.rows / div_size.height;
		const int dest_tilex = get_simd_ceil(tilex + 2 * r, 8);
		const int dest_tiley = tiley + 2 * r;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_32F);

		const int top_tilex = tilex * idx.x;

		const int left = max(0, r - top_tilex);
		const int sleft = r - left;
		const int right = max(0, dest_tilex + top_tilex - src.cols - r);
		const int sright = -right + dest_tilex - tilex;
		const int copysizex = dest_tilex - left - right;

		const int top = max(0, r - tiley * idx.y);
		const int stop = r - top;
		const int bottom = max(0, r + (tiley * (idx.y + 1) - src.rows));
		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_floor(right, 8);
		float* s = src.ptr<float>(tiley * idx.y - stop) + top_tilex;
		float* d = dest.ptr<float>(top);

		const int LOAD_OFFSET1 = left - 8 - top_tilex;
		const int LOAD_OFFSET2 = src.cols - 8 - top_tilex;
		const int LOAD_OFFSET3 = src.cols - 1 - top_tilex;
		const int STORE_OFFSET = tilex + sright;

		for (int j = 0; j < copysizey; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				__m256 a = _mm256_load_ps(s + LOAD_OFFSET1 - i);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_store_ps(d + i, a);
			}

			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));

			for (int i = 0; i < RIGHT; i += 8)
			{
				__m256 a = _mm256_load_ps(s + LOAD_OFFSET2 - i);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i];
			}

			s += src.cols;
			d += dest.cols;
		}
		for (int j = 0; j < top; j++)
		{
			float* s = dest.ptr<float>(max(0, 2 * r - j - 1));
			float* d = dest.ptr<float>(j);
			memcpy(d, s, sizeof(float) * (dest_tilex));
		}
		for (int j = 0; j < bottom; j++)
		{
			float* s = dest.ptr<float>(max(0, tiley + r - 1 - j));
			float* d = dest.ptr<float>(tiley + r + j);
			memcpy(d, s, sizeof(float) * dest_tilex);
		}
	}

	void GaussianFilterSeparableFIR::filterCopyBorderSingle(Mat& srcBorder, Size tileSize, Mat& dest, Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int destTopY = tileIndex.y * tileSize.height;
		const int destTopX = tileIndex.x * tileSize.width;
		for (int j = 0; j < tileSize.height; j++)
		{
			float* s = srcBorder.ptr<float>(j + r) + r;
			float* d = dest.ptr<float>(j + destTopY) + destTopX;
			for (int i = 0; i < tileSize.width; i += 8)
			{
				float* si = s + i;
				__m256 ms = _mm256_load_ps(si);
				_mm256_store_ps(d + i, ms);
				//__m256 mv = _mm256_setzero_ps();
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVBorderSingle(Mat& srcBorder, Size tileSize, Mat& dest, Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int wstep = srcBorder.cols;
		const int destTopY = tileIndex.y * tileSize.height;
		const int destTopX = tileIndex.x * tileSize.width;

		const int tidx = 0;
		if (!useAllocBuffer)
		{
			bufferTile2[tidx].release();
		}
		if (bufferTile2[tidx].size() != srcBorder.size()) bufferTile2[tidx].create(srcBorder.size(), srcBorder.type());

		if (opt == VECTOR_WITHOUT)
		{
			/*

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
			float* s = bufferImageBorder.ptr<float>(j) + r;
			float* d = bufferImageBorder2.ptr<float>(j);
			for (int i = 0; i < src.cols; i++)
			{
			float v = 0.f;
			for (int k = 0; k < ksize; k++)
			{
			v += gauss[k] * s[i + (k - r)];
			}
			d[i] = v;
			}
			}

			for (int j = 0; j < src.rows; j++)
			{
			float* s = bufferImageBorder2.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 0; i < src.cols; i++)
			{
			float v = 0.f;
			for (int k = 0; k < ksize; k++)
			{
			v += gauss[k] * s[i + k*wstep];
			}
			d[i] = v;
			}
			}
			*/
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter
			float* s = srcBorder.ptr<float>(0);
			float* d = bufferTile2[tidx].ptr<float>(0);
			for (int j = 0; j < srcBorder.rows; j++)
			{
				for (int i = 0; i < tileSize.width; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				s += wstep;
				d += wstep;
			}
			//v filter
			s = bufferTile2[tidx].ptr<float>(0);
			d = dest.ptr<float>(destTopY) + destTopX;
			for (int j = 0; j < tileSize.height; j++)
			{
				for (int i = 0; i < tileSize.width; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					_mm256_store_ps(d + i, mv);
				}
				s += wstep;
				d += dest.cols;
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVDelayedBorderSingle(Mat& srcBorder, Size tileSize, Mat& dest, Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int wstep = srcBorder.cols;
		const int destTopY = tileIndex.y * tileSize.height;
		const int destTopX = tileIndex.x * tileSize.width;
		if (opt == VECTOR_WITHOUT)
		{
			/*

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				for (int i = 0; i < srcBorder.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss[k] * s[i + k];
					}
					s[i] = v;
				}
			}

			for (int j = 0; j < tileSize.height; j++)
			{
				float* s = bufferImageBorder.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < tileSize.width; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss[k] * s[i + k*wstep];
					}
					d[i] = v;
				}
			}
			*/
		}
		else if (opt == VECTOR_AVX)
		{
			//h filter
			float* s = srcBorder.ptr<float>(0);
			for (int j = 0; j < srcBorder.rows; j++)
			{
				for (int i = 0; i < tileSize.width; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(s + i, mv);
				}
				s += srcBorder.cols;
			}
			//v filter
			s = srcBorder.ptr<float>(0);
			float* d = dest.ptr<float>(destTopY) + destTopX;
			for (int j = 0; j < tileSize.height; j++)
			{
				for (int i = 0; i < tileSize.width; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					_mm256_store_ps(d + i, mv);
				}
				s += srcBorder.cols;
				d += dest.cols;
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVDelayedVPBorderSingle(Mat& srcBorder, Size tileSize, Mat& dest, Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		//const int wstep = srcBorder.cols;
		const int destTopY = tileIndex.y * tileSize.height;
		const int destTopX = tileIndex.x * tileSize.width;
		if (opt == VECTOR_WITHOUT)
		{
			/*

			for (int j = 0; j < bufferImageBorder.rows; j++)
			{
			float* s = bufferImageBorder.ptr<float>(j);
			for (int i = 0; i < srcBorder.cols; i++)
			{
			float v = 0.f;
			for (int k = 0; k < ksize; k++)
			{
			v += gauss[k] * s[i + k];
			}
			s[i] = v;
			}
			}

			for (int j = 0; j < tileSize.height; j++)
			{
			float* s = bufferImageBorder.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 0; i < tileSize.width; i++)
			{
			float v = 0.f;
			for (int k = 0; k < ksize; k++)
			{
			v += gauss[k] * s[i + k*wstep];
			}
			d[i] = v;
			}
			}
			*/
		}
		else if (opt == VECTOR_AVX)
		{
			const int dstep = dest.cols;
			const int dstep0 = dstep * 0;
			const int dstep1 = dstep * 1;
			const int dstep2 = dstep * 2;
			const int dstep3 = dstep * 3;
			const int dstep4 = dstep * 4;
			const int dstep5 = dstep * 5;
			const int dstep6 = dstep * 6;
			const int dstep7 = dstep * 7;
			const int dstep8 = dstep * 8;
			//h filter
			float* s = srcBorder.ptr<float>(0);
			for (int j = 0; j < srcBorder.rows; j++)
			{
				for (int i = 0; i < tileSize.width; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(s + i, mv);
				}
				s += srcBorder.cols;
			}
			//v filter
			Mat buff(Size(srcBorder.rows, 1), CV_32F);
			float* dptr = dest.ptr<float>(destTopY) + destTopX;
			float* b = buff.ptr<float>(0);
			for (int i = 0; i < tileSize.width; i++)
			{
				verticalLineCopy(srcBorder, buff, i);
				float* d = dptr + i;
				for (int j = 0; j < tileSize.height; j += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* bi = b + j;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(bi);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						bi++;
					}
					d[dstep0] = ((float*)&mv)[0];
					d[dstep1] = ((float*)&mv)[1];
					d[dstep2] = ((float*)&mv)[2];
					d[dstep3] = ((float*)&mv)[3];
					d[dstep4] = ((float*)&mv)[4];
					d[dstep5] = ((float*)&mv)[5];
					d[dstep6] = ((float*)&mv)[6];
					d[dstep7] = ((float*)&mv)[7];
					d += dstep8;
				}
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHBorderSingle(Mat& srcBorder, Size tileSize, Mat& dest, Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int destTopY = tileIndex.y * tileSize.height;
		const int destTopX = tileIndex.x * tileSize.width;
		const int wstep = srcBorder.cols;

		const int tidx = 0;
		if (!useAllocBuffer)
		{
			bufferTile2[tidx].release();
		}
		if (bufferTile2[tidx].size() != srcBorder.size()) bufferTile2[tidx].create(srcBorder.size(), srcBorder.type());

		if (opt == VECTOR_WITHOUT)
		{
			/*

			for (int j = 0; j < tileSize.height; j++)
			{
			float* s = bufferImageBorder.ptr<float>(j + r);
			float* d = bufferImageBorder2.ptr<float>(j + r);

			for (int i = 0; i < bufferImageBorder.cols; i++)
			{
			float* si = s + i;
			float v = 0.f;
			for (int k = 0; k < ksize; k++)
			{
			v += gauss[k] * si[(k - r)*wstep];
			}
			d[i] = v;
			}
			}

			for (int j = 0; j < src.rows; j++)
			{
			float* s = bufferImageBorder2.ptr<float>(j + r);
			float* d = dest.ptr<float>(j);
			for (int i = 0; i < src.cols; i++)
			{
			float* si = s + i;
			float v = 0.f;
			for (int k = 0; k < ksize; k++)
			{
			v += gauss[k] * si[k];
			}
			d[i] = v;
			}
			}
			*/
		}
		else if (opt == VECTOR_AVX)
		{
			//v filter
			float* s = srcBorder.ptr<float>(0);
			float* d = bufferTile2[tidx].ptr<float>(r);
			for (int j = 0; j < tileSize.height; j++)
			{
				for (int i = 0; i < srcBorder.cols; i += 8)
				{
					float* si = s + i;
					__m256 mv = _mm256_setzero_ps();
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					_mm256_store_ps(d + i, mv);
				}
				s += wstep;
				d += wstep;
			}
			//h filter
			s = bufferTile2[tidx].ptr<float>(r);
			d = dest.ptr<float>(destTopY) + destTopX;
			for (int j = 0; j < tileSize.height; j++)
			{
				for (int i = 0; i < tileSize.width; i += 8)
				{
					float* si = s + i;
					__m256 mv = _mm256_setzero_ps();
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				s += wstep;
				d += dest.cols;
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHDelayedBorderSingle(Mat& srcBorder, Size tileSize, Mat& dest, Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int wstep = srcBorder.cols;
		const int destTopY = tileIndex.y * tileSize.height;
		const int destTopX = tileIndex.x * tileSize.width;
		if (opt == VECTOR_WITHOUT)
		{
			//v filter
			for (int j = 0; j < srcBorder.cols; j++)
			{
				float* s = srcBorder.ptr<float>(0) + j;
				for (int i = 0; i < tileSize.height; i++)
				{
					float* si = s + i * wstep;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * si[k * wstep];
					}
					s[i * wstep] = v;
				}
			}
			//h filter
			for (int j = 0; j < tileSize.height; j++)
			{
				float* s = srcBorder.ptr<float>(j);
				float* d = dest.ptr<float>(destTopY + j) + destTopX;
				for (int i = 0; i < tileSize.width; i++)
				{
					float* si = s + i;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * si[k];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			__m256i midx = _mm256_setr_epi32(0, wstep, 2 * wstep, 3 * wstep, 4 * wstep, 5 * wstep, 6 * wstep, 7 * wstep);
			//v filter
			float* s = srcBorder.ptr<float>(0);
			for (int j = 0; j < srcBorder.cols; j++)
			{
				for (int i = 0; i < tileSize.height; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i * wstep;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(si, midx, sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					s[(i + 0) * wstep] = ((float*)&mv)[0];
					s[(i + 1) * wstep] = ((float*)&mv)[1];
					s[(i + 2) * wstep] = ((float*)&mv)[2];
					s[(i + 3) * wstep] = ((float*)&mv)[3];
					s[(i + 4) * wstep] = ((float*)&mv)[4];
					s[(i + 5) * wstep] = ((float*)&mv)[5];
					s[(i + 6) * wstep] = ((float*)&mv)[6];
					s[(i + 7) * wstep] = ((float*)&mv)[7];
				}
				s++;
			}
			// h filter
			s = srcBorder.ptr<float>(0);
			float* d = dest.ptr<float>(destTopY) + destTopX;
			for (int j = 0; j < tileSize.height; j++)
			{
				for (int i = 0; i < tileSize.width; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				s += wstep;
				d += dest.cols;
			}
		}
	}

	void GaussianFilterSeparableFIR::filterVHVPBorderSingle(Mat& srcBorder, Size tileSize, Mat& dest, Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int wstep = srcBorder.cols;
		const int destTopY = tileIndex.y * tileSize.height;
		const int destTopX = tileIndex.x * tileSize.width;
		if (opt == VECTOR_WITHOUT)
		{
			//v filter
			for (int j = 0; j < srcBorder.cols; j++)
			{
				float* s = srcBorder.ptr<float>(0) + j;
				for (int i = 0; i < tileSize.height; i++)
				{
					float* si = s + i * wstep;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * si[k * wstep];
					}
					s[i * wstep] = v;
				}
			}
			//h filter
			for (int j = 0; j < tileSize.height; j++)
			{
				float* s = srcBorder.ptr<float>(j);
				float* d = dest.ptr<float>(destTopY + j) + destTopX;
				for (int i = 0; i < tileSize.width; i++)
				{
					float* si = s + i;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * si[k];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			//v filter
			Mat buff(Size(srcBorder.rows, 1), CV_32F);
			float* b = buff.ptr<float>(0);
			const int wstep0 = srcBorder.cols * 0;
			const int wstep1 = srcBorder.cols * 1;
			const int wstep2 = srcBorder.cols * 2;
			const int wstep3 = srcBorder.cols * 3;
			const int wstep4 = srcBorder.cols * 4;
			const int wstep5 = srcBorder.cols * 5;
			const int wstep6 = srcBorder.cols * 6;
			const int wstep7 = srcBorder.cols * 7;
			const int wstep8 = srcBorder.cols * 8;
			float* sptr = srcBorder.ptr<float>(0);
			for (int i = 0; i < srcBorder.cols; i++)
			{
				verticalLineCopy(srcBorder, buff, i);
				float* s = sptr + i;
				for (int j = 0; j < tileSize.height; j += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* bj = b + j;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(bj);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						bj++;
					}
					s[wstep0] = ((float*)&mv)[0];
					s[wstep1] = ((float*)&mv)[1];
					s[wstep2] = ((float*)&mv)[2];
					s[wstep3] = ((float*)&mv)[3];
					s[wstep4] = ((float*)&mv)[4];
					s[wstep5] = ((float*)&mv)[5];
					s[wstep6] = ((float*)&mv)[6];
					s[wstep7] = ((float*)&mv)[7];
					s += wstep8;
				}
			}
			// h filter
			float* s = srcBorder.ptr<float>(0);
			float* d = dest.ptr<float>(destTopY) + destTopX;
			for (int j = 0; j < tileSize.height; j++)
			{
				for (int i = 0; i < tileSize.width; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(d + i, mv);
				}
				s += wstep;
				d += dest.cols;
			}
		}
	}

	void GaussianFilterSeparableFIR::filterHVIBorderSingle(Mat& srcBorder, Size tileSize, Mat& dest, Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int wstep = srcBorder.cols;
		const int destTopY = tileIndex.y * tileSize.height;
		const int destTopX = tileIndex.x * tileSize.width;

		if (opt == VECTOR_WITHOUT)
		{
			Mat buff(srcBorder.rows, 1, CV_32F);
			//float* s = srcBorder.ptr<float>(0);
			//float* b = buff.ptr<float>(0);
			for (int i = 0; i < tileSize.width; i++)
			{
				float* b = buff.ptr<float>(0);

				for (int j = 0; j < srcBorder.rows; j++)
				{
					float* s = srcBorder.ptr<float>(j) + i;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[k];
					}
					b[j] = v;
				}

				for (int j = 0; j < tileSize.height; j++)
				{
					float* d = dest.ptr<float>(j + destTopY) + destTopX;
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * b[k + j];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			const __m256i midx = _mm256_setr_epi32(0 * wstep, 1 * wstep, 2 * wstep, 3 * wstep, 4 * wstep, 5 * wstep, 6 * wstep, 7 * wstep);
			const int dstep0 = 0 * dest.cols;
			const int dstep1 = 1 * dest.cols;
			const int dstep2 = 2 * dest.cols;
			const int dstep3 = 3 * dest.cols;
			const int dstep4 = 4 * dest.cols;
			const int dstep5 = 5 * dest.cols;
			const int dstep6 = 6 * dest.cols;
			const int dstep7 = 7 * dest.cols;
			Mat buff(srcBorder.rows, 1, CV_32F);
			//float* s = srcBorder.ptr<float>(0);
			float* b = buff.ptr<float>(0);

			const int H0 = get_simd_floor(srcBorder.rows, 8);
			for (int i = 0; i < tileSize.width; i++)
			{
				//h filter		
				float* s = srcBorder.ptr<float>(0);
				for (int j = 0; j < H0; j += 8)
				{
					float* si = s + i;
					__m256 mv = _mm256_setzero_ps();
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_i32gather_ps(si, midx, sizeof(float));
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si++;
					}
					_mm256_store_ps(b + j, mv);
					s += wstep * 8;
				}
				for (int j = H0; j < srcBorder.rows; j++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[k + i];
					}
					b[j] = v;
					s += wstep;
				}

				//v filter	
				float* d = dest.ptr<float>(destTopY) + destTopX;
				for (int j = 0; j < tileSize.height; j += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* bi = b + j;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(bi);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						bi++;
					}
					d[i + dstep0] = ((float*)&mv)[0];
					d[i + dstep1] = ((float*)&mv)[1];
					d[i + dstep2] = ((float*)&mv)[2];
					d[i + dstep3] = ((float*)&mv)[3];
					d[i + dstep4] = ((float*)&mv)[4];
					d[i + dstep5] = ((float*)&mv)[5];
					d[i + dstep6] = ((float*)&mv)[6];
					d[i + dstep7] = ((float*)&mv)[7];
					d += dest.cols * 8;
				}
			}

		}
	}

	void GaussianFilterSeparableFIR::filterVHIBorderSingle(Mat& srcBorder, Size tileSize, Mat& dest, Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer)
	{
		const int ksize = 2 * r + 1;
		const int destTopY = tileIndex.y * tileSize.height;
		const int destTopX = tileIndex.x * tileSize.width;
		const int wstep = srcBorder.cols;
		if (opt == VECTOR_WITHOUT)
		{

			for (int j = 0; j < tileSize.height; j++)
			{
				Mat buff(srcBorder.cols, 1, CV_32F);
				float* s = srcBorder.ptr<float>(j);
				float* b = buff.ptr<float>(0);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < srcBorder.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * s[i + k * wstep];
					}
					b[i] = v;
				}
				for (int i = 0; i < tileSize.width; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss32F[k] * b[i + k];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
			Mat buff(srcBorder.cols, 1, CV_32F);
			float* b = buff.ptr<float>(0);
			float* s = srcBorder.ptr<float>(0);
			float* d = dest.ptr<float>(destTopY) + destTopX;
			for (int j = 0; j < tileSize.height; j++)
			{
				for (int i = 0; i < srcBorder.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* si = s + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_load_ps(si);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						si += wstep;
					}
					_mm256_store_ps(b + i, mv);
				}
				for (int i = 0; i < tileSize.width; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					float* bi = b + i;
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(bi);
						__m256 mg = _mm256_set1_ps(gauss32F[k]);
						mv = _mm256_fmadd_ps(ms, mg, mv);
						bi++;
					}
					_mm256_store_ps(d + i, mv);
				}
				s += srcBorder.cols;
				d += dest.cols;
			}
		}
	}

	void GaussianFilterSeparableFIR::filterTileSubImageInplace(Mat& srcdest, const int r, const float sigma, int border, int opt, bool useAllocBuffer, int method)
	{
		const int tilex = srcdest.cols / tileDiv.width;
		const int tiley = srcdest.rows / tileDiv.height;

		vector<Point> tileIndex;
		for (int j = 0; j < tileDiv.height; j++)
		{
			for (int i = 0; i < tileDiv.width; i++)
			{
				tileIndex.push_back(Point(i, j));
			}
		}


		for (int n = 0; n < tileIndex.size(); n++)
		{
			if (!useAllocBuffer)bufferSubImage[n].release();
			createSubImage(srcdest, bufferSubImage[n], tileDiv, tileIndex[n], r);
		}


		for (int n = 0; n < tileIndex.size(); n++)
		{
			const int tidx = 0;
			if (!useAllocBuffer)bufferTile[tidx].release();

			if (method == HV_T_Sub)
			{
				filterHVBorderSingle(bufferSubImage[n], Size(tilex, tiley), srcdest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == HV_T_SubD)
			{
				filterHVDelayedBorderSingle(bufferSubImage[tidx], Size(tilex, tiley), srcdest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == HV_T_SubDVP)
			{
				filterHVDelayedVPBorderSingle(bufferSubImage[tidx], Size(tilex, tiley), srcdest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == VH_T_Sub)
			{
				filterVHBorderSingle(bufferSubImage[tidx], Size(tilex, tiley), srcdest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == VH_T_SubD)
			{
				filterVHDelayedBorderSingle(bufferSubImage[tidx], Size(tilex, tiley), srcdest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == VH_T_SubVP)
			{
				filterVHVPBorderSingle(bufferSubImage[tidx], Size(tilex, tiley), srcdest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == VHI_T_Sub)
			{
				filterVHIBorderSingle(bufferSubImage[tidx], Size(tilex, tiley), srcdest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == HVI_T_Sub)
			{
				filterHVIBorderSingle(bufferSubImage[tidx], Size(tilex, tiley), srcdest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else
			{
				filterCopyBorderSingle(bufferTile[tidx], Size(tilex, tiley), srcdest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
		}
	}

	void GaussianFilterSeparableFIR::filterTileSubImage(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer, int method)
	{
		if (src.data == dest.data)
		{
			filterTileSubImageInplace(src, r, sigma, border, opt, useAllocBuffer, method);
			return;
		}
		const int tilex = src.cols / tileDiv.width;
		const int tiley = src.rows / tileDiv.height;

		vector<Point> tileIndex;
		for (int j = 0; j < tileDiv.height; j++)
		{
			for (int i = 0; i < tileDiv.width; i++)
			{
				tileIndex.push_back(Point(i, j));
			}
		}


		for (int n = 0; n < tileIndex.size(); n++)
		{
			const int tidx = 0;
			if (!useAllocBuffer)bufferTile[tidx].release();
			createSubImage(src, bufferTile[tidx], tileDiv, tileIndex[n], r);

			if (method == HV_T_Sub)
			{
				filterHVBorderSingle(bufferTile[tidx], Size(tilex, tiley), dest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == HV_T_SubD)
			{
				filterHVDelayedBorderSingle(bufferTile[tidx], Size(tilex, tiley), dest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == HV_T_SubDVP)
			{
				filterHVDelayedVPBorderSingle(bufferTile[tidx], Size(tilex, tiley), dest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == VH_T_Sub)
			{
				filterVHBorderSingle(bufferTile[tidx], Size(tilex, tiley), dest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == VH_T_SubD)
			{
				filterVHDelayedBorderSingle(bufferTile[tidx], Size(tilex, tiley), dest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == VH_T_SubVP)
			{
				filterVHVPBorderSingle(bufferTile[tidx], Size(tilex, tiley), dest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == VHI_T_Sub)
			{
				filterVHIBorderSingle(bufferTile[tidx], Size(tilex, tiley), dest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else if (method == HVI_T_Sub)
			{
				filterHVIBorderSingle(bufferTile[tidx], Size(tilex, tiley), dest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
			else
			{
				filterCopyBorderSingle(bufferTile[tidx], Size(tilex, tiley), dest, tileIndex[n], r, sigma, border, opt, useAllocBuffer);
			}
		}
	}

}