#include "dualExponentialSmoothing.hpp"
#include <intrin.h>

using namespace std;
using namespace cv;

namespace cp
{
	float sigma2LaplacianSmootihngAlpha(const float sigma, float p)
	{
		return 1.f - exp(-p / (1.f * sigma));
	}

	void LaplacianSmoothingFIR2DFilter(Mat& src, Mat& dest, float sigma)
	{
		int r = (int)ceil(5.0 * sigma);
		Mat kernel = Mat::zeros(2 * r + 1, 2 * r + 1, CV_32F);
		float total = 0.f;
		for (int j = 0; j < kernel.rows; j++)
		{
			for (int i = 0; i < kernel.cols; i++)
			{
				float p = abs(sqrtf(float((i - r) * (i - r) + (j - r) * (j - r))));
				float v = expf(-p / sigma);
				kernel.at<float>(j, i) = v;
				total += v;
			}
		}
		for (int j = 0; j < kernel.rows; j++)
		{
			for (int i = 0; i < kernel.cols; i++)
			{
				kernel.at<float>(j, i) /= total;
			}
		}
		filter2D(src, dest, CV_32F, kernel);
	}

	void LaplacianSmoothingFIRFilter(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt)
	{
		if (dest.empty())dest.create(src.size(), src.type());
		const int ksize = (2 * r + 1);
		Mat im;
		copyMakeBorder(src, im, r, r, r, r, border);

		float* gauss = (float*)_mm_malloc(sizeof(float) * ksize, 32);
		const float gfrac = -1.f / (sigma);
		float gsum = 0.f;
		for (int j = -r, index = 0; j <= r; j++)
		{
			float v = exp(abs(j) * gfrac);
			gsum += v;
			gauss[index] = v;
			index++;
		}
		for (int j = -r, index = 0; j <= r; j++)
		{
			//gauss[index] = max(FLT_EPSILON, gauss[index]/gsum);
			gauss[index] /= gsum;
			index++;
		}

		const int wstep = im.cols;
		if (opt == VECTOR_WITHOUT)
		{
#pragma omp parallel for
			for (int j = 0; j < im.rows; j++)
			{
				float* s = im.ptr<float>(j);
				float* d = im.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss[k] * s[i + k];
					}
					d[i] = v;
				}
			}
#pragma omp parallel for
			for (int j = 0; j < src.rows; j++)
			{
				float* s = im.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float v = 0.f;
					for (int k = 0; k < ksize; k++)
					{
						v += gauss[k] * s[i + k * wstep];
					}
					d[i] = v;
				}
			}
		}
		else if (opt == VECTOR_AVX)
		{
#pragma omp parallel for
			for (int j = 0; j < im.rows; j++)
			{
				float* s = im.ptr<float>(j);
				float* d = im.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(s + i + k);
						__m256 mg = _mm256_set1_ps(gauss[k]);
						mv = _mm256_add_ps(mv, _mm256_mul_ps(ms, mg));
					}
					_mm256_storeu_ps(d + i, mv);
				}
			}

#pragma omp parallel for
			for (int j = 0; j < src.rows; j++)
			{
				float* s = im.ptr<float>(j);
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mv = _mm256_setzero_ps();
					for (int k = 0; k < ksize; k++)
					{
						__m256 ms = _mm256_loadu_ps(s + i + k * wstep);
						__m256 mg = _mm256_set1_ps(gauss[k]);
						mv = _mm256_add_ps(mv, _mm256_mul_ps(ms, mg));
					}
					_mm256_storeu_ps(d + i, mv);
				}
			}
		}
		_mm_free(gauss);
	}

	void LaplacianSmoothingIIRFilterBase(Mat& src, Mat& dest, const double sigma_)
	{
		if (dest.empty())dest.create(src.size(), src.type());
		Mat tmp(src.size(), src.type());
		Mat tmp2(src.rows, 1, src.type());
		if (src.depth() == CV_32F)
		{
			const float sigma = (float)sigma_;
			const float is = 1.f - sigma;
			for (int j = 0; j < src.rows; j++)
			{
				float* im = src.ptr<float>(j);
				float* dt = dest.ptr<float>(j);
				float* tp = tmp.ptr<float>(j);
				dt[0] = im[0];
				for (int i = 1; i < src.cols; i++)
				{
					dt[i] = sigma * im[i] + is * dt[i - 1];
				}

				tp[src.cols - 1] = im[src.cols - 1];
				dt[src.cols - 1] = (im[src.cols - 1] + dt[src.cols - 1]) * 0.5f;
				for (int i = src.cols - 2; i >= 0; i--)
				{
					tp[i] = sigma * im[i] + is * tp[i + 1];
					dt[i] = (dt[i] + tp[i]) * 0.5f;
				}
			}
			for (int i = 0; i < src.cols; i++)
			{
				float* im = dest.ptr<float>(0) + i;
				float* dt = dest.ptr<float>(0) + i;
				float* tp = tmp.ptr<float>(0) + i;
				float* tp2 = tmp2.ptr<float>(0);
				tp[0] = im[0];
				for (int j = 1; j < src.rows; j++)
				{
					tp[j * src.cols] = sigma * im[j * src.cols] + is * tp[(j - 1) * src.cols];
				}
				tp2[src.rows - 1] = im[src.cols * (src.rows - 1)];
				dt[src.cols * (src.rows - 1)] = (im[src.cols * (src.rows - 1)] + tp[src.cols * (src.rows - 1)]) * 0.5f;
				for (int j = src.rows - 2; j >= 0; j--)
				{
					tp2[j] = sigma * im[j * src.cols] + is * tp2[j + 1];
					dt[j * src.cols] = (tp[j * src.cols] + tp2[j]) * 0.5f;
				}
			}
		}
		else if (src.depth() == CV_64F)
		{
			const double sigma = sigma_;
			const double is = 1.f - sigma;

			for (int j = 0; j < src.rows; j++)
			{
				double* im = src.ptr<double>(j);
				double* dt = dest.ptr<double>(j);
				double* tp = tmp.ptr<double>(j);
				dt[0] = im[0];
				for (int i = 1; i < src.cols; i++)
				{
					dt[i] = sigma * im[i] + is * dt[i - 1];
				}

				tp[src.cols - 1] = im[src.cols - 1];
				dt[src.cols - 1] = (im[src.cols - 1] + dt[src.cols - 1]) * 0.5f;
				for (int i = src.cols - 2; i >= 0; i--)
				{
					tp[i] = sigma * im[i] + is * tp[i + 1];
					dt[i] = (dt[i] + tp[i]) * 0.5f;
				}
			}
			for (int i = 0; i < src.cols; i++)
			{
				double* im = dest.ptr<double>(0) + i;
				double* dt = dest.ptr<double>(0) + i;
				double* tp = tmp.ptr<double>(0) + i;
				double* tp2 = tmp2.ptr<double>(0);
				tp[0] = im[0];
				for (int j = 1; j < src.rows; j++)
				{
					tp[j * src.cols] = sigma * im[j * src.cols] + is * tp[(j - 1) * src.cols];
				}
				tp2[src.rows - 1] = im[src.cols * (src.rows - 1)];
				dt[src.cols * (src.rows - 1)] = (im[src.cols * (src.rows - 1)] + tp[src.cols * (src.rows - 1)]) * 0.5;
				for (int j = src.rows - 2; j >= 0; j--)
				{
					tp2[j] = sigma * im[j * src.cols] + is * tp2[j + 1];
					dt[j * src.cols] = (tp[j * src.cols] + tp2[j]) * 0.5f;
				}
			}
		}
	}

	void LaplacianSmoothingIIRFilterAVX(Mat& src, Mat& dest, const float sigma)
	{
		if (dest.empty())dest.create(src.size(), src.type());
		Mat buff(max(src.cols * 8, src.rows * 8), 1, src.type());

		const __m256i gidx = _mm256_set_epi32(0, src.cols, 2 * src.cols, 3 * src.cols, 4 * src.cols, 5 * src.cols, 6 * src.cols, 7 * src.cols);
		for (int j = 0; j < src.rows; j += 8)
		{
			const __m256 ms = _mm256_set1_ps(sigma);
			const __m256 mis = _mm256_set1_ps(1.f - sigma);
			const __m256 half = _mm256_set1_ps(0.5f);
			float* im = src.ptr<float>(j);
			float* dt = dest.ptr<float>(j);
			float* b = buff.ptr<float>(0);

			const int idx0 = 0;
			const int idx1 = 1 * src.cols;
			const int idx2 = 2 * src.cols;
			const int idx3 = 3 * src.cols;
			const int idx4 = 4 * src.cols;
			const int idx5 = 5 * src.cols;
			const int idx6 = 6 * src.cols;
			const int idx7 = 7 * src.cols;

			__m256 pv = _mm256_i32gather_ps(im, gidx, 4);
			_mm256_store_ps(b, pv);
			dt[idx0] = im[idx0];
			dt[idx1] = im[idx1];
			dt[idx2] = im[idx2];
			dt[idx3] = im[idx3];
			dt[idx4] = im[idx4];
			dt[idx5] = im[idx5];
			dt[idx6] = im[idx6];
			dt[idx7] = im[idx7];
			for (int i = 1; i < src.cols; i++)
			{
				pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i, gidx, 4), _mm256_mul_ps(mis, pv));
				_mm256_store_ps(b + 8 * i, pv);
			}
			__m256 t = pv;
			pv = _mm256_i32gather_ps(im + src.cols - 1, gidx, 4);
			t = _mm256_mul_ps(half, _mm256_add_ps(t, pv));

			dt[idx0 + src.cols - 1] = ((float*)&t)[7];
			dt[idx1 + src.cols - 1] = ((float*)&t)[6];
			dt[idx2 + src.cols - 1] = ((float*)&t)[5];
			dt[idx3 + src.cols - 1] = ((float*)&t)[4];
			dt[idx4 + src.cols - 1] = ((float*)&t)[3];
			dt[idx5 + src.cols - 1] = ((float*)&t)[2];
			dt[idx6 + src.cols - 1] = ((float*)&t)[1];
			dt[idx7 + src.cols - 1] = ((float*)&t)[0];
			for (int i = src.cols - 2; i >= 0; i--)
			{
				pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i, gidx, 4), _mm256_mul_ps(mis, pv));
				__m256 v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * i), pv));
				dt[i + idx0] = ((float*)&v)[7];
				dt[i + idx1] = ((float*)&v)[6];
				dt[i + idx2] = ((float*)&v)[5];
				dt[i + idx3] = ((float*)&v)[4];
				dt[i + idx4] = ((float*)&v)[3];
				dt[i + idx5] = ((float*)&v)[2];
				dt[i + idx6] = ((float*)&v)[1];
				dt[i + idx7] = ((float*)&v)[0];
			}
		}

		for (int i = 0; i < src.cols; i += 8)
		{
			const __m256 ms = _mm256_set1_ps(sigma);
			const __m256 mis = _mm256_set1_ps(1.f - sigma);
			const __m256 half = _mm256_set1_ps(0.5f);
			float* im = dest.ptr<float>(0) + i;
			float* dt = dest.ptr<float>(0) + i;
			float* b = buff.ptr<float>(0);

			__m256 pv = _mm256_load_ps(im);
			_mm256_storeu_ps(b, pv);
			for (int j = 1; j < src.rows; j++)
			{
				pv = _mm256_fmadd_ps(ms, _mm256_loadu_ps(im + j * src.cols), _mm256_mul_ps(mis, pv));
				_mm256_store_ps(b + j * 8, pv);
			}

			__m256 t = pv;
			pv = _mm256_loadu_ps(im + src.cols * (src.rows - 1));
			t = _mm256_mul_ps(half, _mm256_add_ps(t, pv));
			_mm256_storeu_ps(dt + (src.rows - 1) * src.cols, t);

			for (int j = src.rows - 2; j >= 0; j--)
			{
				pv = _mm256_fmadd_ps(ms, _mm256_loadu_ps(im + j * src.cols), _mm256_mul_ps(mis, pv));
				_mm256_storeu_ps(dt + j * src.cols, _mm256_mul_ps(half, _mm256_add_ps(_mm256_loadu_ps(b + j * 8), pv)));
			}
		}
	}

	void LaplacianSmoothingIIRFilter(Mat& src, Mat& dest, const double sigma, int opt)
	{
		if (opt == VECTOR_WITHOUT) LaplacianSmoothingIIRFilterBase(src, dest, sigma);
		else if (opt == VECTOR_AVX)LaplacianSmoothingIIRFilterAVX(src, dest, float(sigma));
	}
}