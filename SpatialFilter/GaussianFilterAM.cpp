#include "stdafx.h"
#include "spatialfilter/SpatialFilter.hpp"

using namespace std;
using namespace cv;

namespace cp
{
#pragma region GaussianFilterAM_Naive

	template<typename Type>
	GaussianFilterAM_Naive<Type>::GaussianFilterAM_Naive(cv::Size imgSize, Type sigma, int order)
		: SpatialFilterBase(imgSize, cp::typeToCVDepth<Type>()), order(order), sigma(sigma), tol(Type(1.0e-6))
	{
		const double q = sigma * (1.0 + (0.3165 * order + 0.5695) / ((order + 0.7818) * (order + 0.7818)));
		const double lambda = (q * q) / (2.0 * order);
		const double dnu = (1.0 + 2.0 * lambda - sqrt(1.0 + 4.0 * lambda)) / (2.0 * lambda);
		nu = (Type)dnu;
		r_init = (int)ceil(log((1.0 - nu) * tol) / log(nu));
		scale = (Type)(pow(dnu / lambda, order));

		h = new Type[r_init];

		h[0] = (Type)1.0;
		for (int i = 1; i < r_init; ++i)
		{
			h[i] = nu * h[i - 1];
		}
	}

	template<typename Type>
	GaussianFilterAM_Naive<Type>::~GaussianFilterAM_Naive()
	{
		delete[] h;
	}

	template<typename Type>
	void GaussianFilterAM_Naive<Type>::horizontalbody(cv::Mat& img)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		Type* imgPtr;
		Type accum;

		//forward direction
		imgPtr = img.ptr<Type>();
		for (int y = 0; y < height; ++y, imgPtr += width)
		{
			//boundary processing
			accum = imgPtr[0];
			for (int m = 1; m < r_init; ++m)
			{
				accum += h[m] * imgPtr[ref_lborder(-m, borderType)];
			}
			imgPtr[0] = accum;

			//IIR filtering
			for (int x = 1; x < width; ++x)
			{
				imgPtr[x] += nu * imgPtr[x - 1];
			}
		}

		//reverse direction
		imgPtr = img.ptr<Type>();
		for (int y = 0; y < height; ++y, imgPtr += width)
		{
			//boundary processing
			imgPtr[width - 1] /= ((Type)1.0 - nu);

			//IIR filtering
			for (int x = width - 2; 0 <= x; --x)
			{
				imgPtr[x] += nu * imgPtr[x + 1];
			}
		}
	};

	template<typename Type>
	void  GaussianFilterAM_Naive<Type>::verticalbody(Mat& img)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		Type* imgPtr;
		Type* prePtr;
		Type accum;

		//forward direction
		imgPtr = img.ptr<Type>();
		for (int x = 0; x < width; ++x)
		{
			//boundary processing
			accum = imgPtr[x];
			for (int m = 1; m < r_init; ++m)
			{
				accum += h[m] * *(imgPtr + ref_tborder(-m, width, borderType) + x);
			}
			imgPtr[x] = accum;
		}

		//IIR filtering
		prePtr = imgPtr;
		imgPtr += width;
		for (int y = 1; y < height; ++y, prePtr = imgPtr, imgPtr += width)
		{
			for (int x = 0; x < width; ++x)
			{
				imgPtr[x] += nu * prePtr[x];
			}
		}

		//reverse direction
		//boundary processing
		imgPtr = img.ptr<Type>(height - 1);
		for (int x = 0; x < width; ++x)
		{
			imgPtr[x] /= ((Type)1.0 - nu);
		}

		//IIR filtering
		prePtr = imgPtr;
		imgPtr -= width;
		for (int y = height - 2; 0 <= y; --y, prePtr = imgPtr, imgPtr -= width)
		{
			for (int x = 0; x < width; ++x)
			{
				imgPtr[x] += nu * prePtr[x];
			}
		}
	};

	template<class Type>
	void GaussianFilterAM_Naive<Type>::body(const cv::Mat& src, cv::Mat& dst, const int border)
	{
		if (src.depth() == depth)
			multiply(src, scale, dst);
		else
			src.convertTo(dst, depth, scale);

		for (int k = 0; k < order; ++k)
		{
			horizontalbody(dst);
		}

		dst = scale * dst;

		for (int k = 0; k < order; ++k)
		{
			verticalbody(dst);
		}
	}

	template class GaussianFilterAM_Naive<float>;
	template class GaussianFilterAM_Naive<double>;

#pragma endregion

#pragma region GaussianFilterAM_AVX_32F

	void GaussianFilterAM_AVX_32F::allocBuffer()
	{
		const double q = sigma * (1.0 + (0.3165 * gf_order + 0.5695) / ((gf_order + 0.7818) * (gf_order + 0.7818)));
		const double lambda = (q * q) / (2.0 * gf_order);
		const double dnu = ((1.0 + 2.0 * lambda - sqrt(1.0 + 4.0 * lambda)) / (2.0 * lambda));
		this->nu = (float)dnu;
		this->r_init = (int)ceil(log((1.0 - dnu) * tol) / log(dnu));
		this->scale = (float)(pow(dnu / lambda, gf_order));

		this->norm = float(1.0 - (double)nu);

		delete[] h;
		this->h = new float[r_init];
		h[0] = 1.f;
		for (int i = 1; i < r_init; ++i)
		{
			h[i] = nu * h[i - 1];
		}
	}

	GaussianFilterAM_AVX_32F::GaussianFilterAM_AVX_32F(cv::Size imgSize, float sigma, int order)
		: SpatialFilterBase(imgSize, CV_32F)
	{
		this->gf_order = clipOrder(order, SpatialFilterAlgorithm::IIR_AM);
		this->sigma = sigma;

		allocBuffer();
	}

	GaussianFilterAM_AVX_32F::GaussianFilterAM_AVX_32F(const int dest_depth)
	{ 
		this->dest_depth = dest_depth;
		this->depth = CV_32F; 
	}

	GaussianFilterAM_AVX_32F::~GaussianFilterAM_AVX_32F()
	{
		delete[] h;
	}

	//gather_vload->store transpose unroll 8
	void GaussianFilterAM_AVX_32F::horizontalFilterVLoadGatherTransposeStore(cv::Mat& img)
	{
		const int width = imgSize.width;
		const __m256i mm_offset = _mm256_set_epi32(7 * width, 6 * width, 5 * width, 4 * width, 3 * width, 2 * width, width, 0);
		__m256 patch[8];
		__m256 patch_t[8];

		const int height = imgSize.height;

		float* img_ptr;
		float* dst;
		__m256 input;
		int refx;

		//forward direction
		for (int y = 0; y < height; y += 8)
		{
			//boundary processing
			img_ptr = img.ptr<float>(y);
			dst = img_ptr;
			patch[0] = _mm256_i32gather_ps(img_ptr, mm_offset, sizeof(float));
			for (int m = 1; m < r_init; ++m)
			{
				refx = ref_lborder(-m, borderType);
				input = _mm256_i32gather_ps(img_ptr + refx, mm_offset, sizeof(float));
				//patch[0] = _mm256_add_ps(patch[0], _mm256_mul_ps(_mm256_set1_ps(h[m]), input));
				patch[0] = _mm256_fmadd_ps(_mm256_set1_ps(h[m]), input, patch[0]);
			}
			++img_ptr;
			for (int i = 1; i < 8; ++i)
			{
				input = _mm256_i32gather_ps(img_ptr, mm_offset, sizeof(float));
				//patch[i] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[i - 1]));
				patch[i] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[i - 1], input);
				++img_ptr;
			}
			_mm256_transpose8_ps(patch, patch_t);
			_mm256_storepatch_ps(dst, patch_t, width);
			dst += 8;

			//IIR filtering
			for (int x = 8; x < width; x += 8)
			{
				input = _mm256_i32gather_ps(img_ptr, mm_offset, sizeof(float));
				//patch[0] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[7]));
				patch[0] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[7], input);
				++img_ptr;

				for (int i = 1; i < 8; ++i)
				{
					input = _mm256_i32gather_ps(img_ptr, mm_offset, sizeof(float));
					//patch[i] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[i - 1]));
					patch[i] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[i - 1], input);
					++img_ptr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst += 8;
			}
		}

		//reverse direction
		for (int y = 0; y < height; y += 8)
		{
			//boundary processing
			img_ptr = img.ptr<float>(y) + width - 1;
			dst = img_ptr - 7;
			input = _mm256_i32gather_ps(img_ptr, mm_offset, sizeof(float));
			patch[7] = _mm256_div_ps(input, _mm256_set1_ps(norm));
			--img_ptr;

			for (int i = 6; i >= 0; --i)
			{
				input = _mm256_i32gather_ps(img_ptr, mm_offset, sizeof(float));
				//patch[i] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[i + 1]));
				patch[i] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[i + 1], input);
				--img_ptr;
			}

			_mm256_transpose8_ps(patch, patch_t);
			_mm256_storepatch_ps(dst, patch_t, width);
			dst -= 8;

			//IIR filtering
			for (int x = width - 16; 0 <= x; x -= 8)
			{
				input = _mm256_i32gather_ps(img_ptr, mm_offset, sizeof(float));
				//patch[7] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[0]));
				patch[7] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[0], input);
				--img_ptr;

				for (int i = 6; i >= 0; --i)
				{
					input = _mm256_i32gather_ps(img_ptr, mm_offset, sizeof(float));
					//patch[i] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[i + 1]));
					patch[i] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[i + 1], input);
					--img_ptr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst -= 8;
			}
		}
	}

	//set_vload->store transpose unroll 8
	void GaussianFilterAM_AVX_32F::horizontalFilterVLoadSetTransposeStore(cv::Mat& img)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		float* img_ptr;
		float* dst;
		__m256 input;
		__m256 patch[8];
		__m256 patch_t[8];
		int refx;

		//forward direction
		for (int y = 0; y < height; y += 8)
		{
			//boundary processing
			img_ptr = img.ptr<float>(y);
			dst = img_ptr;
			patch[0] = _mm256_set_ps(img_ptr[7 * width], img_ptr[6 * width], img_ptr[5 * width], img_ptr[4 * width], img_ptr[3 * width], img_ptr[2 * width], img_ptr[width], img_ptr[0]);

			for (int m = 1; m < r_init; ++m)
			{
				refx = ref_lborder(-m, borderType);
				input = _mm256_set_ps(img_ptr[7 * width + refx], img_ptr[6 * width + refx], img_ptr[5 * width + refx], img_ptr[4 * width + refx], img_ptr[3 * width + refx], img_ptr[2 * width + refx], img_ptr[width + refx], img_ptr[refx]);
				//patch[0] = _mm256_add_ps(patch[0], _mm256_mul_ps(_mm256_set1_ps(h[m]), input));
				patch[0] = _mm256_fmadd_ps(_mm256_set1_ps(h[m]), input, patch[0]);
			}
			++img_ptr;
			for (int i = 1; i < 8; ++i)
			{
				input = _mm256_set_ps(img_ptr[7 * width], img_ptr[6 * width], img_ptr[5 * width], img_ptr[4 * width], img_ptr[3 * width], img_ptr[2 * width], img_ptr[width], img_ptr[0]);
				//patch[i] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[i - 1]));
				patch[i] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[i - 1], input);
				++img_ptr;
			}
			_mm256_transpose8_ps(patch, patch_t);
			_mm256_storepatch_ps(dst, patch_t, width);
			dst += 8;

			//IIR filtering
			for (int x = 8; x < width; x += 8)
			{
				input = _mm256_set_ps(img_ptr[7 * width], img_ptr[6 * width], img_ptr[5 * width], img_ptr[4 * width], img_ptr[3 * width], img_ptr[2 * width], img_ptr[width], img_ptr[0]);
				//patch[0] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[7]));
				patch[0] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[7], input);
				++img_ptr;

				for (int i = 1; i < 8; ++i)
				{
					input = _mm256_set_ps(img_ptr[7 * width], img_ptr[6 * width], img_ptr[5 * width], img_ptr[4 * width], img_ptr[3 * width], img_ptr[2 * width], img_ptr[width], img_ptr[0]);
					//patch[i] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[i - 1]));
					patch[i] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[i - 1], input);
					++img_ptr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst += 8;
			}
		}

		//reverse direction
		for (int y = 0; y < height; y += 8)
		{
			//boundary processing
			img_ptr = img.ptr<float>(y) + width - 1;
			dst = img_ptr - 7;
			input = _mm256_set_ps(img_ptr[7 * width], img_ptr[6 * width], img_ptr[5 * width], img_ptr[4 * width], img_ptr[3 * width], img_ptr[2 * width], img_ptr[width], img_ptr[0]);
			patch[7] = _mm256_div_ps(input, _mm256_set1_ps(norm));
			--img_ptr;

			for (int i = 6; i >= 0; --i)
			{
				input = _mm256_set_ps(img_ptr[7 * width], img_ptr[6 * width], img_ptr[5 * width], img_ptr[4 * width], img_ptr[3 * width], img_ptr[2 * width], img_ptr[width], img_ptr[0]);
				//patch[i] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[i + 1]));
				patch[i] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[i + 1], input);
				--img_ptr;
			}

			_mm256_transpose8_ps(patch, patch_t);
			_mm256_storepatch_ps(dst, patch_t, width);
			dst -= 8;

			//IIR filtering
			for (int x = width - 16; 0 <= x; x -= 8)
			{
				input = _mm256_set_ps(img_ptr[7 * width], img_ptr[6 * width], img_ptr[5 * width], img_ptr[4 * width], img_ptr[3 * width], img_ptr[2 * width], img_ptr[width], img_ptr[0]);
				//patch[7] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[0]));
				patch[7] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[0], input);
				--img_ptr;

				for (int i = 6; i >= 0; --i)
				{
					input = _mm256_set_ps(img_ptr[7 * width], img_ptr[6 * width], img_ptr[5 * width], img_ptr[4 * width], img_ptr[3 * width], img_ptr[2 * width], img_ptr[width], img_ptr[0]);
					//patch[i] = _mm256_add_ps(input, _mm256_mul_ps(_mm256_set1_ps(nu), patch[i + 1]));
					patch[i] = _mm256_fmadd_ps(_mm256_set1_ps(nu), patch[i + 1], input);
					--img_ptr;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst -= 8;
			}
		}
	}

	void  GaussianFilterAM_AVX_32F::verticalFilter(Mat& img)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		float* imgPtr;
		__m256 accum;

		//forward processing
		imgPtr = img.ptr<float>();
		for (int x = 0; x < width; x += 8)
		{
			accum = _mm256_setzero_ps();
			for (int m = 0; m < r_init; ++m)
			{
				accum = _mm256_add_ps(accum, _mm256_mul_ps(_mm256_set1_ps(h[m]), *(__m256*)(imgPtr + x + ref_tborder(-m, width, borderType))));
			}
			_mm256_store_ps(imgPtr + x, accum);
		}

		for (int y = 1; y < height; ++y)
		{
			imgPtr = img.ptr<float>(y);
			for (int x = 0; x < width; x += 8)
				*(__m256*)(imgPtr + x) = _mm256_add_ps(*(__m256*)(imgPtr + x), _mm256_mul_ps(_mm256_set1_ps(nu), *(__m256*)(imgPtr + x - width)));
		}

		//backward processing
		imgPtr = img.ptr<float>(height - 1);
		for (int x = 0; x < width; x += 8)
			*(__m256*)(imgPtr + x) = _mm256_div_ps(*(__m256*)(imgPtr + x), _mm256_set1_ps(norm));

		for (int y = height - 2; y >= 0; --y)
		{
			imgPtr = img.ptr<float>(y);
			for (int x = 0; x < width; x += 8)
				*(__m256*)(imgPtr + x) = _mm256_add_ps(*(__m256*)(imgPtr + x), _mm256_mul_ps(_mm256_set1_ps(nu), *(__m256*)(imgPtr + x + width)));
		}
	}

	void GaussianFilterAM_AVX_32F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		this->borderType = borderType;
		CV_Assert(src.cols % 8 == 0);
		CV_Assert(src.rows % 8 == 0);
		CV_Assert(src.depth()==CV_8U|| src.depth() == CV_32F);

		if (dest_depth == CV_32F)
		{
			if (src.depth() == CV_32F)
				multiply(src, scale, dst);
			else
				src.convertTo(dst, CV_32F, scale);

			for (int k = 0; k < gf_order; ++k)
			{
				//horizontalFilterVLoadSetTransposeStore(dst);
				horizontalFilterVLoadGatherTransposeStore(dst);
			}

			multiply(dst, scale, dst);

			for (int k = 0; k < gf_order; ++k)
			{
				verticalFilter(dst);
			}
		}
		else
		{
			inter.create(src.size(), CV_32F);

			if (src.depth() == CV_32F)
				multiply(src, scale, inter);
			else
				src.convertTo(inter, CV_32F, scale);

			for (int k = 0; k < gf_order; ++k)
			{
				//horizontalFilterVLoadSetTransposeStore(inter);
				horizontalFilterVLoadGatherTransposeStore(inter);
			}

			multiply(inter, scale, inter);

			for (int k = 0; k < gf_order; ++k)
			{
				verticalFilter(inter);
			}
			inter.convertTo(dst, dest_depth);
		}
	}

	void GaussianFilterAM_AVX_32F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		int corder = clipOrder(order, SpatialFilterAlgorithm::IIR_AM);
		if (this->sigma != sigma || this->gf_order != corder || imgSize != src.size())
		{
			this->sigma = sigma;
			this->gf_order = corder;
			this->imgSize = src.size();
			allocBuffer();
		}

		body(src, dst, borderType);
	}

#pragma endregion

#pragma region GaussianFilterAM_AVX_64F

	void GaussianFilterAM_AVX_64F::allocBuffer()
	{
		const double q = sigma * (1.0 + (0.3165 * gf_order + 0.5695) / ((gf_order + 0.7818) * (gf_order + 0.7818)));
		const double lambda = (q * q) / (2.0 * gf_order);
		nu = ((1.0 + 2.0 * lambda - sqrt(1.0 + 4.0 * lambda)) / (2.0 * lambda));
		r_init = (int)ceil(log((1.0 - nu) * tol) / log(nu));
		scale = pow(nu / lambda, gf_order);

		__m256d mm_nu = _mm256_broadcast_sd(&nu);
		this->mm_h = (__m256d*)_mm_malloc(r_init * sizeof(__m256d), 32);

		this->mm_h[0] = _mm256_set1_pd(1.0);
		for (int i = 1; i < r_init; ++i)
		{
			this->mm_h[i] = _mm256_mul_pd(mm_nu, mm_h[i - 1]);
		}
	}

	GaussianFilterAM_AVX_64F::GaussianFilterAM_AVX_64F(cv::Size imgSize, double sigma, int order)
		: SpatialFilterBase(imgSize, CV_64F)
	{
		this->gf_order = clipOrder(order, SpatialFilterAlgorithm::IIR_AM);
		this->sigma = sigma;

		allocBuffer();
	}

	GaussianFilterAM_AVX_64F::GaussianFilterAM_AVX_64F(const int dest_depth)
	{ 
		this->dest_depth = dest_depth;
		this->depth = CV_64F;
	}

	GaussianFilterAM_AVX_64F::~GaussianFilterAM_AVX_64F()
	{
		_mm_free(mm_h);
	}

	//gather_vload->store transpose unroll 4
	void GaussianFilterAM_AVX_64F::horizontalFilterVloadGatherTranposeStore(cv::Mat& img)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		double* img_ptr;
		__m256d cur;
		__m256d patch[4];
		__m256d patch_t[4];
		__m128i mm_offset = _mm_set_epi32(3 * width, 2 * width, width, 0);

		__m256d mm_nu= _mm256_broadcast_sd(&nu);
		__m256d mm_norm= _mm256_set1_pd((1.0 - nu));
		
		//forward direction
		for (int y = 0; y < height; y += 4)
		{
			//boundary processing
			img_ptr = img.ptr<double>(y);
			patch[0] = _mm256_setzero_pd();

			for (int m = 0; m < r_init; ++m)
			{
				cur = _mm256_i32gather_pd(img_ptr + ref_lborder(-m, borderType), mm_offset, sizeof(double));
				//patch[0] = _mm256_add_pd(patch[0], _mm256_mul_pd(mm_h[m], cur));
				patch[0] = _mm256_fmadd_pd(mm_h[m], cur, patch[0]);
			}
			for (int i = 1; i < 4; ++i)
			{
				cur = _mm256_i32gather_pd(img_ptr + i, mm_offset, sizeof(double));
				//patch[i] = _mm256_add_pd(cur, _mm256_mul_pd(mm_nu, patch[i - 1]));
				patch[i] = _mm256_fmadd_pd(mm_nu, patch[i - 1], cur);
			}
			_mm256_transpose4_pd(patch, patch_t);
			_mm256_storeupatch_pd(img_ptr, patch_t, width);

			//IIR filtering
			for (int x = 4; x < width; x += 4)
			{
				cur = _mm256_i32gather_pd(img_ptr + x, mm_offset, sizeof(double));
				//patch[0] = _mm256_add_pd(cur, _mm256_mul_pd(mm_nu, patch[3]));
				patch[0] = _mm256_fmadd_pd(mm_nu, patch[3], cur);

				for (int i = 1; i < 4; ++i)
				{
					cur = _mm256_i32gather_pd(img_ptr + x + i, mm_offset, sizeof(double));
					//patch[i] = _mm256_add_pd(cur, _mm256_mul_pd(mm_nu, patch[i - 1]));
					patch[i] = _mm256_fmadd_pd(mm_nu, patch[i - 1], cur);
				}

				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(img_ptr + x, patch_t, width);
			}
		}

		//reverse direction
		for (int y = 0; y < height; y += 4)
		{
			//boundary processing
			img_ptr = img.ptr<double>(y);
			patch[3] = _mm256_i32gather_pd(img_ptr + width - 1, mm_offset, sizeof(double));
			patch[3] = _mm256_div_pd(patch[3], mm_norm);

			for (int i = 2; i >= 0; --i)
			{
				cur = _mm256_i32gather_pd(img_ptr + width - 4 + i, mm_offset, sizeof(double));
				//patch[i] = _mm256_add_pd(cur, _mm256_mul_pd(mm_nu, patch[i + 1]));
				patch[i] = _mm256_fmadd_pd(mm_nu, patch[i + 1], cur);
			}

			_mm256_transpose4_pd(patch, patch_t);
			_mm256_storeupatch_pd(img_ptr + width - 4, patch_t, width);

			//IIR filtering
			for (int x = width - 8; 0 <= x; x -= 4)
			{
				cur = _mm256_i32gather_pd(img_ptr + x + 3, mm_offset, sizeof(double));
				//patch[3] = _mm256_add_pd(cur, _mm256_mul_pd(mm_nu, patch[0]));
				patch[3] = _mm256_fmadd_pd(mm_nu, patch[0], cur);

				for (int i = 2; i >= 0; --i)
				{
					cur = _mm256_i32gather_pd(img_ptr + x + i, mm_offset, sizeof(double));
					//patch[i] = _mm256_add_pd(cur, _mm256_mul_pd(mm_nu, patch[i + 1]));
					patch[i] = _mm256_fmadd_pd(mm_nu, patch[i + 1], cur);
				}

				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(img_ptr + x, patch_t, width);
			}
		}
	}

	void  GaussianFilterAM_AVX_64F::verticalFilter(Mat& img)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		//forward direction
		double* imgPtr = img.ptr<double>();
		__m256d mm_nu = _mm256_broadcast_sd(&nu);
		__m256d mm_norm = _mm256_set1_pd((1.0 - nu));
		for (int x = 0; x < width; x += 4)
		{
			//boundary processing
			__m256d accum = _mm256_setzero_pd();
			for (int m = 0; m < r_init; ++m)
			{
				//accum = _mm256_add_pd(accum, _mm256_mul_pd(mm_h[m], *(__m256d*)(imgPtr + x + ref_tborder(-m, width, borderType))));
				accum = _mm256_fmadd_pd(mm_h[m], *(__m256d*)(imgPtr + x + ref_tborder(-m, width, borderType)), accum);
			}

			_mm256_store_pd(imgPtr + x, accum);
		}

		//IIR filtering
		for (int y = 1; y < height; ++y)
		{
			imgPtr = img.ptr<double>(y);
			for (int x = 0; x < width; x += 4)
			{
				//*(__m256d*)(imgPtr + x) = _mm256_add_pd(*(__m256d*)(imgPtr + x), _mm256_mul_pd(mm_nu, *(__m256d*)(imgPtr + x - width)));
				*(__m256d*)(imgPtr + x) = _mm256_fmadd_pd(mm_nu, *(__m256d*)(imgPtr + x - width), *(__m256d*)(imgPtr + x));
			}
		}

		//reverse direction
		imgPtr = img.ptr<double>(height - 1);
		__m256d mdiv = _mm256_div_pd(_mm256_set1_pd(1.0), mm_norm);
		//boundary processing
		for (int x = 0; x < width; x += 4)
		{
			*(__m256d*)(imgPtr + x) = _mm256_mul_pd(*(__m256d*)(imgPtr + x), mdiv);
		}

		//IIR filtering
		for (int y = height - 2; y >= 0; --y)
		{
			imgPtr = img.ptr<double>(y);
			for (int x = 0; x < width; x += 4)
			{
				//*(__m256d*)(imgPtr + x) = _mm256_add_pd(*(__m256d*)(imgPtr + x), _mm256_mul_pd(mm_nu, *(__m256d*)(imgPtr + x + width)));
				*(__m256d*)(imgPtr + x) = _mm256_fmadd_pd(mm_nu, *(__m256d*)(imgPtr + x + width), *(__m256d*)(imgPtr + x));
			}
		}
	}

	void GaussianFilterAM_AVX_64F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		this->borderType = borderType;
		CV_Assert(src.cols % 4 == 0);
		CV_Assert(src.rows % 4 == 0);
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F || src.depth() == CV_64F);

		if (dest_depth == CV_64F)
		{
			if (src.depth() == CV_64F)
				multiply(src, scale, dst);
			else
				src.convertTo(dst, CV_64F, scale);

			for (int k = 0; k < gf_order; ++k)
			{
				horizontalFilterVloadGatherTranposeStore(dst);
			}

			multiply(dst, scale, dst);

			for (int k = 0; k < gf_order; ++k)
			{
				verticalFilter(dst);
			}
		}
		else
		{
			inter.create(src.size(), CV_64F);

			if (src.depth() == CV_64F)
				multiply(src, scale, inter);
			else
				src.convertTo(inter, CV_64F, scale);

			for (int k = 0; k < gf_order; ++k)
			{
				horizontalFilterVloadGatherTranposeStore(inter);
			}

			multiply(inter, scale, inter);

			for (int k = 0; k < gf_order; ++k)
			{
				verticalFilter(inter);
			}
			inter.convertTo(dst, dest_depth);
		}
	}

	void GaussianFilterAM_AVX_64F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		int corder = clipOrder(order, SpatialFilterAlgorithm::IIR_AM);
		if (this->sigma != sigma || this->gf_order != corder || imgSize != src.size())
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