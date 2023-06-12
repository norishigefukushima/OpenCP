#include "stdafx.h"

using namespace std;
using namespace cv;

namespace cp
{
#pragma region DCT_AVX_32F

	void SpatialFilterFullDCT_AVX_32F::allocLUT()
	{
		if (lutsizex != imgSize.width || lutsizey != imgSize.height)
		{
			_mm_free(LUTx);
			_mm_free(LUTy);
			LUTx = (float*)_mm_malloc(imgSize.width * sizeof(float), AVX_ALIGN);
			LUTy = (float*)_mm_malloc(imgSize.height * sizeof(float), AVX_ALIGN);
			lutsizex = imgSize.width;
			lutsizey = imgSize.height;
		}
	}

	void SpatialFilterFullDCT_AVX_32F::setGaussianKernel(const bool isForcedUpdate)
	{
		bool isDenormal = true;

		if (lutsigma != sigma|| isForcedUpdate)
		{
			lutsigma = (float)sigma;

			if (imgSize.width == imgSize.height)
			{
				const double temp = double(sigma * CV_PI / (imgSize.width));
				const double a = -temp * temp / 2.0;
				for (int i = 0; i < imgSize.width; i++)
				{
					float arg = isDenormal ? max(float(a * i * i), denormalclip) : float(a * i * i);
					float v = exp(arg);
					LUTx[i] = v;
					LUTy[i] = v;
				}
			}
			else
			{
				const double tx = double(sigma * CV_PI / (imgSize.width));
				const double ax = -tx * tx / 2.0;
				const double ty = double(sigma * CV_PI / (imgSize.height));
				const double ay = -ty * ty / 2.0;
				for (int i = 0; i < imgSize.width; i++)
				{
					float arg = isDenormal ? max(float(ax * i * i), denormalclip) : float(ax * i * i);//clip denormal number
					float v = exp(arg);
					LUTx[i] = v;
				}
				for (int i = 0; i < imgSize.height; i++)
				{
					float arg = isDenormal ? max(float(ay * i * i), denormalclip) : float(ay * i * i);//clip denormal number
					float v = exp(arg);
					LUTy[i] = v;
				}
			}
		}
	}

	SpatialFilterFullDCT_AVX_32F::SpatialFilterFullDCT_AVX_32F(cv::Size imgSize, float sigma, int order)
		: SpatialFilterBase(imgSize, CV_32F)
	{
		this->algorithm = SpatialFilterAlgorithm::DCTFULL_OPENCV;
		this->gf_order = order;
		this->sigma = sigma;
		allocLUT();
		setGaussianKernel(true);
	}

	SpatialFilterFullDCT_AVX_32F::SpatialFilterFullDCT_AVX_32F(const int dest_depth)
	{ 
		this->algorithm = SpatialFilterAlgorithm::DCTFULL_OPENCV;
		this->dest_depth = dest_depth;
		this->depth = CV_32F; 
	}

	SpatialFilterFullDCT_AVX_32F::~SpatialFilterFullDCT_AVX_32F()
	{
		_mm_free(LUTx);
		_mm_free(LUTy);
	}

	void SpatialFilterFullDCT_AVX_32F::body(const cv::Mat& _src, cv::Mat& dst, int borderType)
	{
		CV_Assert(imgSize.width % 8 == 0);

		if (_src.depth() == depth)
			inter = _src;
		else
			_src.convertTo(inter, depth);

		cv::dct(inter, frec);

		if (inter.cols % 32 == 0)
		{
			for (int j = 0; j < frec.rows; j++)
			{
				float* f = frec.ptr<float>(j);
				const __m256 b = _mm256_set1_ps(LUTy[j]);
				for (int i = 0; i < frec.cols; i += 32)
				{
					__m256 a = _mm256_load_ps(LUTx + i);
					__m256 c = _mm256_load_ps(f + i);
					_mm256_store_ps(f + i, _mm256_mul_ps(b, _mm256_mul_ps(c, a)));
					a = _mm256_load_ps(LUTx + i + 8);
					c = _mm256_load_ps(f + i + 8);
					_mm256_store_ps(f + i + 8, _mm256_mul_ps(b, _mm256_mul_ps(c, a)));
					a = _mm256_load_ps(LUTx + i + 16);
					c = _mm256_load_ps(f + i + 16);
					_mm256_store_ps(f + i + 16, _mm256_mul_ps(b, _mm256_mul_ps(c, a)));
					a = _mm256_load_ps(LUTx + i + 24);
					c = _mm256_load_ps(f + i + 24);
					_mm256_store_ps(f + i + 24, _mm256_mul_ps(b, _mm256_mul_ps(c, a)));
				}
				//for (int i = 0; i <frec.cols; i ++)
				//{
				//float v = LUTx[i] * LUTy[j];
				//f[i] *= v;
				//}
			}
		}
		else
		{
			for (int j = 0; j < frec.rows; j++)
			{
				float* f = frec.ptr<float>(j);
				const __m256 b = _mm256_set1_ps(LUTy[j]);
				for (int i = 0; i < frec.cols; i += 8)
				{
					__m256 a = _mm256_load_ps(LUTx + i);
					__m256 c = _mm256_load_ps(f + i);
					_mm256_store_ps(f + i, _mm256_mul_ps(b, _mm256_mul_ps(c, a)));
				}
			}
		}

		if (dest_depth == depth)
		{
			idct(frec, dst);
		}
		else
		{
			idct(frec, inter);
			inter.convertTo(dst, dest_depth);
		}
	}

	void SpatialFilterFullDCT_AVX_32F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		if (this->sigma != sigma || this->gf_order != order || imgSize != src.size())
		{
			this->sigma = sigma;
			this->gf_order = order;
			this->imgSize = src.size();
			allocLUT();
		}
		setGaussianKernel(true);
		if (borderType != cv::BORDER_REFLECT)
		{
			cout << "cv::dct/cv::idct is DCT-II: Only support borderType == cv::BORDER_REFLECT" << endl;
		}
		body(src, dst, borderType);
	}

#pragma endregion

#pragma region DCT_AVX_64F

	void SpatialFilterFullDCT_AVX_64F::allocLUT()
	{
		CV_Assert(imgSize.width % 4 == 0);

		bool isDenormal = false;

		if (lutsizex != imgSize.width || lutsizey != imgSize.height)
		{
			_mm_free(LUTx);
			_mm_free(LUTy);
			LUTx = (double*)_mm_malloc(imgSize.width * sizeof(double), AVX_ALIGN);
			LUTy = (double*)_mm_malloc(imgSize.height * sizeof(double), AVX_ALIGN);
			lutsizex = imgSize.width;
			lutsizey = imgSize.height;
		}

		if (lutsigma != sigma)
		{
			lutsigma = sigma;

			if (imgSize.width == imgSize.height)
			{
				const double temp = sigma * CV_PI / (imgSize.width);
				const double a = -temp * temp / 2.0;

				for (int i = 0; i < imgSize.width; i++)
				{
					double arg = isDenormal ? max(a * i * i, denormalclip) : a * i * i;
					double v = exp(arg);
					LUTx[i] = v;
					LUTy[i] = v;
				}
			}
			else
			{
				const double tx = sigma * CV_PI / (imgSize.width);
				const double ax = -tx * tx / 2.0;
				const double ty = sigma * CV_PI / (imgSize.height);
				const double ay = -ty * ty / 2.0;

				for (int i = 0; i < imgSize.width; i++)
				{
					double arg = isDenormal ? max(ax * i * i, denormalclip) : ax * i * i;
					double v = exp(arg);
					LUTx[i] = v;
				}
				for (int i = 0; i < imgSize.height; i++)
				{
					double arg = isDenormal ? max(ay * i * i, denormalclip) : ay * i * i;
					double v = exp(arg);
					LUTy[i] = v;
				}
			}
		}
	}

	SpatialFilterFullDCT_AVX_64F::SpatialFilterFullDCT_AVX_64F(cv::Size imgSize, double sigma, int order)
		: SpatialFilterBase(imgSize, CV_64F)
	{
		this->gf_order = order;
		this->sigma = sigma;
		allocLUT();
	}

	SpatialFilterFullDCT_AVX_64F::SpatialFilterFullDCT_AVX_64F(const int dest_depth)
	{
		this->dest_depth = dest_depth;
		this->depth = CV_64F;
	}

	SpatialFilterFullDCT_AVX_64F::~SpatialFilterFullDCT_AVX_64F()
	{
		_mm_free(LUTx);
		_mm_free(LUTy);
	}

	void SpatialFilterFullDCT_AVX_64F::body(const cv::Mat& _src, cv::Mat& dst, const int borderType)
	{
		if (_src.depth() == depth)
			inter = _src;
		else
			_src.convertTo(inter, depth);

		cv::dct(inter, frec);

		if (inter.cols % 16 == 0)
		{
			for (int j = 0; j < frec.rows; j++)
			{
				double* f = frec.ptr<double>(j);
				const __m256d b = _mm256_set1_pd(LUTy[j]);
				for (int i = 0; i < frec.cols; i += 16)
				{
					__m256d a = _mm256_load_pd(LUTx + i);
					__m256d c = _mm256_load_pd(f + i);
					_mm256_store_pd(f + i, _mm256_mul_pd(b, _mm256_mul_pd(c, a)));

					a = _mm256_load_pd(LUTx + i + 4);
					c = _mm256_load_pd(f + i + 4);
					_mm256_store_pd(f + i + 4, _mm256_mul_pd(b, _mm256_mul_pd(c, a)));

					a = _mm256_load_pd(LUTx + i + 8);
					c = _mm256_load_pd(f + i + 8);
					_mm256_store_pd(f + i + 8, _mm256_mul_pd(b, _mm256_mul_pd(c, a)));

					a = _mm256_load_pd(LUTx + i + 12);
					c = _mm256_load_pd(f + i + 12);
					_mm256_store_pd(f + i + 12, _mm256_mul_pd(b, _mm256_mul_pd(c, a)));
				}
				//for (int i = 0; i <frec.cols; i ++)
				//{
				//float v = LUTx[i] * LUTy[j];
				//f[i] *= v;
				//}
			}
		}
		else
		{
			for (int j = 0; j < frec.rows; j++)
			{
				double* f = frec.ptr<double>(j);
				const __m256d b = _mm256_set1_pd(LUTy[j]);
				for (int i = 0; i < frec.cols; i += 4)
				{
					__m256d a = _mm256_load_pd(LUTx + i);
					__m256d c = _mm256_load_pd(f + i);
					_mm256_store_pd(f + i, _mm256_mul_pd(b, _mm256_mul_pd(c, a)));
				}
			}
		}

		if (dest_depth == depth)
		{
			idct(frec, dst);
		}
		else
		{
			idct(frec, inter);
			inter.convertTo(dst, dest_depth);
		}
	}

	void SpatialFilterFullDCT_AVX_64F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		if (this->sigma != sigma || this->gf_order != order || this->imgSize != src.size())
		{
			this->sigma = sigma;
			this->gf_order = order;
			this->imgSize = src.size();
			allocLUT();
		}

		if (borderType != cv::BORDER_REFLECT)
		{
			cout << "cv::dct/cv::idct is DCT-II: Only support borderType == cv::BORDER_REFLECT" << endl;
		}
		body(src, dst, borderType);
	}

#pragma endregion
}
