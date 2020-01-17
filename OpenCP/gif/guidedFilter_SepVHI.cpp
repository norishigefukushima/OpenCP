#include "guidedFilter_SepVHI.h"
#include <iostream>
using namespace std;
using namespace cv;

#include <arithmetic.hpp>
#include <inlineSIMDFunctions.hpp>
using namespace cp;

void Ip2ab_Guide1_sep_VHI_AVX(cv::Mat& I, cv::Mat& p, const int r, float eps, cv::Mat& a, cv::Mat& b)
{
	const int width = I.cols;
	const int height = I.rows;
	cv::Size size = cv::Size(width, height);
	a.create(size, CV_32F);
	b.create(size, CV_32F);

	const int d = 2 * r + 1;
	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const __m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat temp(Size(width + 2 * R, 4), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp__I = temp.ptr<float>(0, R);
		float* tp__p = temp.ptr<float>(1, R);
		float* tp_II = temp.ptr<float>(2, R);
		float* tp_Ip = temp.ptr<float>(3, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* Iptr = I.ptr<float>(i - r, j);
				float* pptr = p.ptr<float>(i - r, j);

				__m256 mSum_I = _mm256_load_ps(Iptr);
				__m256 mSum_II = _mm256_mul_ps(mSum_I, mSum_I);
				__m256 mSum_p = _mm256_load_ps(pptr);
				__m256 mSum_Ip = _mm256_mul_ps(mSum_I, mSum_p);

				Iptr += width;
				pptr += width;
				for (int k = 1; k < d; k++)
				{
					__m256 mi = _mm256_load_ps(Iptr);
					mSum_I = _mm256_add_ps(mSum_I, mi);
					mSum_II = _mm256_fmadd_ps(mi, mi, mSum_II);

					__m256 mp = _mm256_load_ps(pptr);
					mSum_p = _mm256_add_ps(mSum_p, mp);
					mSum_Ip = _mm256_fmadd_ps(mp, mi, mSum_Ip);

					Iptr += width;
					pptr += width;
				}

				_mm256_store_ps(tp__I, mSum_I);
				_mm256_store_ps(tp__p, mSum_p);
				_mm256_store_ps(tp_II, mSum_II);
				_mm256_store_ps(tp_Ip, mSum_Ip);

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp__I = temp.ptr<float>(0, roffset);
			tp__p = temp.ptr<float>(1, roffset);
			tp_II = temp.ptr<float>(2, roffset);
			tp_Ip = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I = _mm256_loadu_ps(tp__I);
				__m256 mSum_p = _mm256_loadu_ps(tp__p);
				__m256 mSum_II = _mm256_loadu_ps(tp_II);
				__m256 mSum_Ip = _mm256_loadu_ps(tp_Ip);

				for (int k = 1; k < d; k++)
				{
					mSum_I = _mm256_add_ps(mSum_I, _mm256_loadu_ps(tp__I + k));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp__p + k));
					mSum_II = _mm256_add_ps(mSum_II, _mm256_loadu_ps(tp_II + k));
					mSum_Ip = _mm256_add_ps(mSum_Ip, _mm256_loadu_ps(tp_Ip + k));
				}

				const __m256 m_I = _mm256_mul_ps(mSum_I, mDiv);
				const __m256 m_p = _mm256_mul_ps(mSum_p, mDiv);
				const __m256 mII = _mm256_mul_ps(mSum_II, mDiv);
				const __m256 mIp = _mm256_mul_ps(mSum_Ip, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				__m256 mvar = _mm256_fnmadd_ps(m_I, m_I, mII);
				mvar = _mm256_add_ps(mvar, meps);
				__m256 mcov = _mm256_fnmadd_ps(m_I, m_p, mIp);

				__m256 ma = _mm256_div_ps(mcov, mvar);
				_mm256_store_ps(aptr, ma);
				aptr += 8;

				_mm256_store_ps(bptr, _mm256_fnmadd_ps(ma, m_I, m_p));
				bptr += 8;

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				int v = max(0, min(height - 1, i - r));
				float* Iptr = I.ptr<float>(v, j);
				float* pptr = p.ptr<float>(v, j);

				__m256 mSum_I = _mm256_load_ps(Iptr);
				__m256 mSum_II = _mm256_mul_ps(mSum_I, mSum_I);
				__m256 mSum_p = _mm256_load_ps(pptr);
				__m256 mSum_Ip = _mm256_mul_ps(mSum_I, mSum_p);

				for (int k = 1; k < d; k++)
				{
					int v = max(0, min(height - 1, i - r + k));
					float* Iptr = I.ptr<float>(v, j);
					float* pptr = p.ptr<float>(v, j);

					const __m256 mb0 = _mm256_load_ps(Iptr);
					mSum_I = _mm256_add_ps(mSum_I, mb0);
					mSum_II = _mm256_fmadd_ps(mb0, mb0, mSum_II);

					__m256 mpl = _mm256_load_ps(pptr);
					mSum_p = _mm256_add_ps(mSum_p, mpl);
					mSum_Ip = _mm256_fmadd_ps(mpl, mb0, mSum_Ip);
				}

				_mm256_store_ps(tp__I, mSum_I);
				_mm256_store_ps(tp__p, mSum_p);
				_mm256_store_ps(tp_II, mSum_II);
				_mm256_store_ps(tp_Ip, mSum_Ip);

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp__I = temp.ptr<float>(0, roffset);
			tp__p = temp.ptr<float>(1, roffset);
			tp_II = temp.ptr<float>(2, roffset);
			tp_Ip = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I = _mm256_loadu_ps(tp__I);
				__m256 mSum_p = _mm256_loadu_ps(tp__p);
				__m256 mSum_II = _mm256_loadu_ps(tp_II);
				__m256 mSum_Ip = _mm256_loadu_ps(tp_Ip);

				for (int k = 1; k < d; k++)
				{
					mSum_I = _mm256_add_ps(mSum_I, _mm256_loadu_ps(tp__I + k));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp__p + k));
					mSum_II = _mm256_add_ps(mSum_II, _mm256_loadu_ps(tp_II + k));
					mSum_Ip = _mm256_add_ps(mSum_Ip, _mm256_loadu_ps(tp_Ip + k));
				}

				const __m256 m_I = _mm256_mul_ps(mSum_I, mDiv);
				const __m256 m_p = _mm256_mul_ps(mSum_p, mDiv);
				const __m256 mII = _mm256_mul_ps(mSum_II, mDiv);
				const __m256 mIp = _mm256_mul_ps(mSum_Ip, mDiv);

				__m256 mvar = _mm256_fnmadd_ps(m_I, m_I, mII);
				mvar = _mm256_add_ps(mvar, _mm256_set1_ps(eps));
				__m256 mcov = _mm256_fnmadd_ps(m_I, m_p, mIp);

				__m256 ma = _mm256_div_ps(mcov, mvar);
				_mm256_store_ps(aptr, ma);
				aptr += 8;

				_mm256_store_ps(bptr, _mm256_fnmadd_ps(ma, m_I, m_p));
				bptr += 8;

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}
		}
	}
}

void Ip2ab_Guide1_sep_VHI_AVX_omp(cv::Mat& I, cv::Mat& p, const int r, float eps, cv::Mat& a, cv::Mat& b)
{
	const int width = I.cols;
	const int height = I.rows;
	cv::Size size = cv::Size(width, height);
	a.create(size, CV_32F);
	b.create(size, CV_32F);

	const int d = 2 * r + 1;
	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const __m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat buff(Size(width + 2 * R, 4 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 4 * omp_get_thread_num(), width + 2 * R, 4));

		float* tp__I = temp.ptr<float>(0, R);
		float* tp__p = temp.ptr<float>(1, R);
		float* tp_II = temp.ptr<float>(2, R);
		float* tp_Ip = temp.ptr<float>(3, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* Iptr = I.ptr<float>(i - r, j);
				float* pptr = p.ptr<float>(i - r, j);

				__m256 mSum_I = _mm256_load_ps(Iptr);
				__m256 mSum_II = _mm256_mul_ps(mSum_I, mSum_I);
				__m256 mSum_p = _mm256_load_ps(pptr);
				__m256 mSum_Ip = _mm256_mul_ps(mSum_I, mSum_p);

				Iptr += width;
				pptr += width;
				for (int k = 1; k < d; k++)
				{
					__m256 mi = _mm256_load_ps(Iptr);
					mSum_I = _mm256_add_ps(mSum_I, mi);
					mSum_II = _mm256_fmadd_ps(mi, mi, mSum_II);

					__m256 mp = _mm256_load_ps(pptr);
					mSum_p = _mm256_add_ps(mSum_p, mp);
					mSum_Ip = _mm256_fmadd_ps(mp, mi, mSum_Ip);

					Iptr += width;
					pptr += width;
				}

				_mm256_store_ps(tp__I, mSum_I);
				_mm256_store_ps(tp__p, mSum_p);
				_mm256_store_ps(tp_II, mSum_II);
				_mm256_store_ps(tp_Ip, mSum_Ip);

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp__I = temp.ptr<float>(0, roffset);
			tp__p = temp.ptr<float>(1, roffset);
			tp_II = temp.ptr<float>(2, roffset);
			tp_Ip = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I = _mm256_loadu_ps(tp__I);
				__m256 mSum_p = _mm256_loadu_ps(tp__p);
				__m256 mSum_II = _mm256_loadu_ps(tp_II);
				__m256 mSum_Ip = _mm256_loadu_ps(tp_Ip);

				for (int k = 1; k < d; k++)
				{
					mSum_I = _mm256_add_ps(mSum_I, _mm256_loadu_ps(tp__I + k));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp__p + k));
					mSum_II = _mm256_add_ps(mSum_II, _mm256_loadu_ps(tp_II + k));
					mSum_Ip = _mm256_add_ps(mSum_Ip, _mm256_loadu_ps(tp_Ip + k));
				}

				const __m256 m_I = _mm256_mul_ps(mSum_I, mDiv);
				const __m256 m_p = _mm256_mul_ps(mSum_p, mDiv);
				const __m256 mII = _mm256_mul_ps(mSum_II, mDiv);
				const __m256 mIp = _mm256_mul_ps(mSum_Ip, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				__m256 mvar = _mm256_fnmadd_ps(m_I, m_I, mII);
				mvar = _mm256_add_ps(mvar, meps);
				__m256 mcov = _mm256_fnmadd_ps(m_I, m_p, mIp);

				__m256 ma = _mm256_div_ps(mcov, mvar);
				_mm256_store_ps(aptr, ma);
				aptr += 8;

				_mm256_store_ps(bptr, _mm256_fnmadd_ps(ma, m_I, m_p));
				bptr += 8;

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				int v = max(0, min(height - 1, i - r));
				float* Iptr = I.ptr<float>(v, j);
				float* pptr = p.ptr<float>(v, j);

				__m256 mSum_I = _mm256_load_ps(Iptr);
				__m256 mSum_II = _mm256_mul_ps(mSum_I, mSum_I);
				__m256 mSum_p = _mm256_load_ps(pptr);
				__m256 mSum_Ip = _mm256_mul_ps(mSum_I, mSum_p);

				for (int k = 1; k < d; k++)
				{
					int v = max(0, min(height - 1, i - r + k));
					float* Iptr = I.ptr<float>(v, j);
					float* pptr = p.ptr<float>(v, j);

					const __m256 mb0 = _mm256_load_ps(Iptr);
					mSum_I = _mm256_add_ps(mSum_I, mb0);
					mSum_II = _mm256_fmadd_ps(mb0, mb0, mSum_II);

					__m256 mpl = _mm256_load_ps(pptr);
					mSum_p = _mm256_add_ps(mSum_p, mpl);
					mSum_Ip = _mm256_fmadd_ps(mpl, mb0, mSum_Ip);
				}

				_mm256_store_ps(tp__I, mSum_I);
				_mm256_store_ps(tp__p, mSum_p);
				_mm256_store_ps(tp_II, mSum_II);
				_mm256_store_ps(tp_Ip, mSum_Ip);

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp__I = temp.ptr<float>(0, roffset);
			tp__p = temp.ptr<float>(1, roffset);
			tp_II = temp.ptr<float>(2, roffset);
			tp_Ip = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I = _mm256_loadu_ps(tp__I);
				__m256 mSum_p = _mm256_loadu_ps(tp__p);
				__m256 mSum_II = _mm256_loadu_ps(tp_II);
				__m256 mSum_Ip = _mm256_loadu_ps(tp_Ip);

				for (int k = 1; k < d; k++)
				{
					mSum_I = _mm256_add_ps(mSum_I, _mm256_loadu_ps(tp__I + k));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp__p + k));
					mSum_II = _mm256_add_ps(mSum_II, _mm256_loadu_ps(tp_II + k));
					mSum_Ip = _mm256_add_ps(mSum_Ip, _mm256_loadu_ps(tp_Ip + k));
				}

				const __m256 m_I = _mm256_mul_ps(mSum_I, mDiv);
				const __m256 m_p = _mm256_mul_ps(mSum_p, mDiv);
				const __m256 mII = _mm256_mul_ps(mSum_II, mDiv);
				const __m256 mIp = _mm256_mul_ps(mSum_Ip, mDiv);

				__m256 mvar = _mm256_fnmadd_ps(m_I, m_I, mII);
				mvar = _mm256_add_ps(mvar, _mm256_set1_ps(eps));
				__m256 mcov = _mm256_fnmadd_ps(m_I, m_p, mIp);

				__m256 ma = _mm256_div_ps(mcov, mvar);
				_mm256_store_ps(aptr, ma);
				aptr += 8;

				_mm256_store_ps(bptr, _mm256_fnmadd_ps(ma, m_I, m_p));
				bptr += 8;

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}
		}
	}
}

void Ip2ab_Guide1_sep_VHI_CenterAvoid_AVX_omp(cv::Mat& I, cv::Mat& p, const int r, float eps, cv::Mat& a, cv::Mat& b)
{
	const int width = I.cols;
	const int height = I.rows;
	cv::Size size = cv::Size(width, height);
	a.create(size, CV_32F);
	b.create(size, CV_32F);

	const int d = 2 * r + 1;
	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const __m256 mDiv = _mm256_set1_ps(1.f / (4 * r*r));

	Mat buff(Size(width + 2 * R, 4 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 4 * omp_get_thread_num(), width + 2 * R, 4));

		float* tp__I = temp.ptr<float>(0, R);
		float* tp__p = temp.ptr<float>(1, R);
		float* tp_II = temp.ptr<float>(2, R);
		float* tp_Ip = temp.ptr<float>(3, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* Iptr = I.ptr<float>(i - r, j);
				float* pptr = p.ptr<float>(i - r, j);

				__m256 mSum_I = _mm256_load_ps(Iptr);
				__m256 mSum_II = _mm256_mul_ps(mSum_I, mSum_I);
				__m256 mSum_p = _mm256_load_ps(pptr);
				__m256 mSum_Ip = _mm256_mul_ps(mSum_I, mSum_p);

				Iptr += width;
				pptr += width;
				for (int k = 1; k < d; k++)
				{
					if (k == r)
					{
						Iptr += width;
						pptr += width;
						continue;
					}
					__m256 mi = _mm256_load_ps(Iptr);
					mSum_I = _mm256_add_ps(mSum_I, mi);
					mSum_II = _mm256_fmadd_ps(mi, mi, mSum_II);

					__m256 mp = _mm256_load_ps(pptr);
					mSum_p = _mm256_add_ps(mSum_p, mp);
					mSum_Ip = _mm256_fmadd_ps(mp, mi, mSum_Ip);

					Iptr += width;
					pptr += width;
				}

				_mm256_store_ps(tp__I, mSum_I);
				_mm256_store_ps(tp__p, mSum_p);
				_mm256_store_ps(tp_II, mSum_II);
				_mm256_store_ps(tp_Ip, mSum_Ip);

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp__I = temp.ptr<float>(0, roffset);
			tp__p = temp.ptr<float>(1, roffset);
			tp_II = temp.ptr<float>(2, roffset);
			tp_Ip = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I = _mm256_loadu_ps(tp__I);
				__m256 mSum_p = _mm256_loadu_ps(tp__p);
				__m256 mSum_II = _mm256_loadu_ps(tp_II);
				__m256 mSum_Ip = _mm256_loadu_ps(tp_Ip);

				for (int k = 1; k < d; k++)
				{
					if (k == r)
					{
						continue;
					}
					mSum_I = _mm256_add_ps(mSum_I, _mm256_loadu_ps(tp__I + k));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp__p + k));
					mSum_II = _mm256_add_ps(mSum_II, _mm256_loadu_ps(tp_II + k));
					mSum_Ip = _mm256_add_ps(mSum_Ip, _mm256_loadu_ps(tp_Ip + k));
				}

				const __m256 m_I = _mm256_mul_ps(mSum_I, mDiv);
				const __m256 m_p = _mm256_mul_ps(mSum_p, mDiv);
				const __m256 mII = _mm256_mul_ps(mSum_II, mDiv);
				const __m256 mIp = _mm256_mul_ps(mSum_Ip, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				__m256 mvar = _mm256_fnmadd_ps(m_I, m_I, mII);
				mvar = _mm256_add_ps(mvar, meps);
				__m256 mcov = _mm256_fnmadd_ps(m_I, m_p, mIp);

				__m256 ma = _mm256_div_ps(mcov, mvar);
				_mm256_store_ps(aptr, ma);
				aptr += 8;

				_mm256_store_ps(bptr, _mm256_fnmadd_ps(ma, m_I, m_p));
				bptr += 8;

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				int v = max(0, min(height - 1, i - r));
				float* Iptr = I.ptr<float>(v, j);
				float* pptr = p.ptr<float>(v, j);

				__m256 mSum_I = _mm256_load_ps(Iptr);
				__m256 mSum_II = _mm256_mul_ps(mSum_I, mSum_I);
				__m256 mSum_p = _mm256_load_ps(pptr);
				__m256 mSum_Ip = _mm256_mul_ps(mSum_I, mSum_p);

				for (int k = 1; k < d; k++)
				{
					int v = max(0, min(height - 1, i - r + k));
					float* Iptr = I.ptr<float>(v, j);
					float* pptr = p.ptr<float>(v, j);

					const __m256 mb0 = _mm256_load_ps(Iptr);
					mSum_I = _mm256_add_ps(mSum_I, mb0);
					mSum_II = _mm256_fmadd_ps(mb0, mb0, mSum_II);

					__m256 mpl = _mm256_load_ps(pptr);
					mSum_p = _mm256_add_ps(mSum_p, mpl);
					mSum_Ip = _mm256_fmadd_ps(mpl, mb0, mSum_Ip);
				}

				_mm256_store_ps(tp__I, mSum_I);
				_mm256_store_ps(tp__p, mSum_p);
				_mm256_store_ps(tp_II, mSum_II);
				_mm256_store_ps(tp_Ip, mSum_Ip);

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp__I = temp.ptr<float>(0, roffset);
			tp__p = temp.ptr<float>(1, roffset);
			tp_II = temp.ptr<float>(2, roffset);
			tp_Ip = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I = _mm256_loadu_ps(tp__I);
				__m256 mSum_p = _mm256_loadu_ps(tp__p);
				__m256 mSum_II = _mm256_loadu_ps(tp_II);
				__m256 mSum_Ip = _mm256_loadu_ps(tp_Ip);

				for (int k = 1; k < d; k++)
				{
					mSum_I = _mm256_add_ps(mSum_I, _mm256_loadu_ps(tp__I + k));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp__p + k));
					mSum_II = _mm256_add_ps(mSum_II, _mm256_loadu_ps(tp_II + k));
					mSum_Ip = _mm256_add_ps(mSum_Ip, _mm256_loadu_ps(tp_Ip + k));
				}

				const __m256 m_I = _mm256_mul_ps(mSum_I, mDiv);
				const __m256 m_p = _mm256_mul_ps(mSum_p, mDiv);
				const __m256 mII = _mm256_mul_ps(mSum_II, mDiv);
				const __m256 mIp = _mm256_mul_ps(mSum_Ip, mDiv);

				__m256 mvar = _mm256_fnmadd_ps(m_I, m_I, mII);
				mvar = _mm256_add_ps(mvar, _mm256_set1_ps(eps));
				__m256 mcov = _mm256_fnmadd_ps(m_I, m_p, mIp);

				__m256 ma = _mm256_div_ps(mcov, mvar);
				_mm256_store_ps(aptr, ma);
				aptr += 8;

				_mm256_store_ps(bptr, _mm256_fnmadd_ps(ma, m_I, m_p));
				bptr += 8;

				tp__I += 8;
				tp__p += 8;
				tp_II += 8;
				tp_Ip += 8;
			}
		}
	}
}


void ab2q_Guide1_sep_VHI_AVX(cv::Mat& a, cv::Mat& b, cv::Mat& guide, const int r, cv::Mat& dest)
{
	const int width = a.cols;
	const int height = a.rows;
	cv::Size size = cv::Size(width, height);

	dest.create(size, CV_32F);

	const int d = 2 * r + 1;
	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const __m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat temp(Size(width + 2 * R, 2), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp_a = temp.ptr<float>(0, R);
		float* tp_b = temp.ptr<float>(1, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* aptr = a.ptr<float>(i - r, j);
				float* bptr = b.ptr<float>(i - r, j);

				__m256 mSum_a = _mm256_load_ps(aptr);
				__m256 mSum_b = _mm256_load_ps(bptr);

				aptr += width;
				bptr += width;
				for (int k = 1; k < d; k++)
				{
					mSum_a = _mm256_add_ps(mSum_a, _mm256_load_ps(aptr));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(bptr));
					aptr += width;
					bptr += width;
				}

				_mm256_store_ps(tp_a, mSum_a);
				_mm256_store_ps(tp_b, mSum_b);
				tp_a += 8;
				tp_b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* gptr = guide.ptr<float>(i);
			float* dptr = dest.ptr<float>(i);
			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp_a = temp.ptr<float>(0, roffset);
			tp_b = temp.ptr<float>(1, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a = _mm256_loadu_ps(tp_a);
				__m256 mSum_b = _mm256_loadu_ps(tp_b);

				for (int k = 1; k < d; k++)
				{
					mSum_a = _mm256_add_ps(mSum_a, _mm256_loadu_ps(tp_a + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}

				const __m256 m_a = _mm256_mul_ps(mSum_a, mDiv);
				const __m256 m_b = _mm256_mul_ps(mSum_b, mDiv);

				_mm256_store_ps(dptr, _mm256_fmadd_ps(m_a, _mm256_load_ps(gptr), m_b));
				aptr += 8;
				bptr += 8;
				gptr += 8;
				dptr += 8;

				tp_a += 8;
				tp_b += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				int v = max(0, min(height - 1, i - r));

				float* aptr = a.ptr<float>(v, j);
				float* bptr = b.ptr<float>(v, j);

				__m256 mSum_a = _mm256_load_ps(aptr);
				__m256 mSum_b = _mm256_load_ps(bptr);

				for (int k = 1; k < d; k++)
				{
					int v = max(0, min(height - 1, i - r + k));

					float* aptr = a.ptr<float>(v, j);
					float* bptr = b.ptr<float>(v, j);
					mSum_a = _mm256_add_ps(mSum_a, _mm256_load_ps(aptr));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(bptr));
				}

				_mm256_store_ps(tp_a, mSum_a);
				_mm256_store_ps(tp_b, mSum_b);
				tp_a += 8;
				tp_b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* gptr = guide.ptr<float>(i);
			float* dptr = dest.ptr<float>(i);
			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp_a = temp.ptr<float>(0, roffset);
			tp_b = temp.ptr<float>(1, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a = _mm256_loadu_ps(tp_a);
				__m256 mSum_b = _mm256_loadu_ps(tp_b);

				for (int k = 1; k < d; k++)
				{
					mSum_a = _mm256_add_ps(mSum_a, _mm256_loadu_ps(tp_a + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}

				const __m256 m_a = _mm256_mul_ps(mSum_a, mDiv);
				const __m256 m_b = _mm256_mul_ps(mSum_b, mDiv);

				_mm256_store_ps(dptr, _mm256_fmadd_ps(m_a, _mm256_load_ps(gptr), m_b));
				aptr += 8;
				bptr += 8;
				gptr += 8;
				dptr += 8;

				tp_a += 8;
				tp_b += 8;
			}
		}
	}
}

void ab2q_Guide1_sep_VHI_AVX_omp(cv::Mat& a, cv::Mat& b, cv::Mat& guide, const int r, cv::Mat& dest)
{
	const int width = a.cols;
	const int height = a.rows;
	cv::Size size = cv::Size(width, height);

	dest.create(size, CV_32F);

	const int d = 2 * r + 1;
	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const __m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat buff(Size(width + 2 * R, 2 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 2 * omp_get_thread_num(), width + 2 * R, 2));

		float* tp_a = temp.ptr<float>(0, R);
		float* tp_b = temp.ptr<float>(1, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* aptr = a.ptr<float>(i - r, j);
				float* bptr = b.ptr<float>(i - r, j);

				__m256 mSum_a = _mm256_load_ps(aptr);
				__m256 mSum_b = _mm256_load_ps(bptr);

				aptr += width;
				bptr += width;
				for (int k = 1; k < d; k++)
				{
					mSum_a = _mm256_add_ps(mSum_a, _mm256_load_ps(aptr));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(bptr));
					aptr += width;
					bptr += width;
				}

				_mm256_store_ps(tp_a, mSum_a);
				_mm256_store_ps(tp_b, mSum_b);
				tp_a += 8;
				tp_b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* gptr = guide.ptr<float>(i);
			float* dptr = dest.ptr<float>(i);
			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp_a = temp.ptr<float>(0, roffset);
			tp_b = temp.ptr<float>(1, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a = _mm256_loadu_ps(tp_a);
				__m256 mSum_b = _mm256_loadu_ps(tp_b);

				for (int k = 1; k < d; k++)
				{
					mSum_a = _mm256_add_ps(mSum_a, _mm256_loadu_ps(tp_a + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}

				const __m256 m_a = _mm256_mul_ps(mSum_a, mDiv);
				const __m256 m_b = _mm256_mul_ps(mSum_b, mDiv);

				_mm256_store_ps(dptr, _mm256_fmadd_ps(m_a, _mm256_load_ps(gptr), m_b));
				aptr += 8;
				bptr += 8;
				gptr += 8;
				dptr += 8;

				tp_a += 8;
				tp_b += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				int v = max(0, min(height - 1, i - r));

				float* aptr = a.ptr<float>(v, j);
				float* bptr = b.ptr<float>(v, j);

				__m256 mSum_a = _mm256_load_ps(aptr);
				__m256 mSum_b = _mm256_load_ps(bptr);

				for (int k = 1; k < d; k++)
				{
					int v = max(0, min(height - 1, i - r + k));

					float* aptr = a.ptr<float>(v, j);
					float* bptr = b.ptr<float>(v, j);
					mSum_a = _mm256_add_ps(mSum_a, _mm256_load_ps(aptr));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(bptr));
				}

				_mm256_store_ps(tp_a, mSum_a);
				_mm256_store_ps(tp_b, mSum_b);
				tp_a += 8;
				tp_b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* gptr = guide.ptr<float>(i);
			float* dptr = dest.ptr<float>(i);
			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp_a = temp.ptr<float>(0, roffset);
			tp_b = temp.ptr<float>(1, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a = _mm256_loadu_ps(tp_a);
				__m256 mSum_b = _mm256_loadu_ps(tp_b);

				for (int k = 1; k < d; k++)
				{
					mSum_a = _mm256_add_ps(mSum_a, _mm256_loadu_ps(tp_a + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}

				const __m256 m_a = _mm256_mul_ps(mSum_a, mDiv);
				const __m256 m_b = _mm256_mul_ps(mSum_b, mDiv);

				_mm256_store_ps(dptr, _mm256_fmadd_ps(m_a, _mm256_load_ps(gptr), m_b));
				aptr += 8;
				bptr += 8;
				gptr += 8;
				dptr += 8;

				tp_a += 8;
				tp_b += 8;
			}
		}
	}
}

void ab2q_Guide1_sep_VHI_CenterAboid_AVX_omp(cv::Mat& a, cv::Mat& b, cv::Mat& guide, const int r, cv::Mat& dest)
{
	const int width = a.cols;
	const int height = a.rows;
	cv::Size size = cv::Size(width, height);

	dest.create(size, CV_32F);

	const int d = 2 * r + 1;
	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const __m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat buff(Size(width + 2 * R, 2 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 2 * omp_get_thread_num(), width + 2 * R, 2));

		float* tp_a = temp.ptr<float>(0, R);
		float* tp_b = temp.ptr<float>(1, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* aptr = a.ptr<float>(i - r, j);
				float* bptr = b.ptr<float>(i - r, j);

				__m256 mSum_a = _mm256_load_ps(aptr);
				__m256 mSum_b = _mm256_load_ps(bptr);

				aptr += width;
				bptr += width;
				for (int k = 1; k < d; k++)
				{
					if (k == r)
					{
						aptr += width;
						bptr += width;
						continue;
					}
					mSum_a = _mm256_add_ps(mSum_a, _mm256_load_ps(aptr));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(bptr));

					aptr += width;
					bptr += width;
				}

				_mm256_store_ps(tp_a, mSum_a);
				_mm256_store_ps(tp_b, mSum_b);
				tp_a += 8;
				tp_b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* gptr = guide.ptr<float>(i);
			float* dptr = dest.ptr<float>(i);
			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp_a = temp.ptr<float>(0, roffset);
			tp_b = temp.ptr<float>(1, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a = _mm256_loadu_ps(tp_a);
				__m256 mSum_b = _mm256_loadu_ps(tp_b);

				for (int k = 1; k < d; k++)
				{
					if (k == r)continue;
					mSum_a = _mm256_add_ps(mSum_a, _mm256_loadu_ps(tp_a + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}

				const __m256 m_a = _mm256_mul_ps(mSum_a, mDiv);
				const __m256 m_b = _mm256_mul_ps(mSum_b, mDiv);

				_mm256_store_ps(dptr, _mm256_fmadd_ps(m_a, _mm256_load_ps(gptr), m_b));
				aptr += 8;
				bptr += 8;
				gptr += 8;
				dptr += 8;

				tp_a += 8;
				tp_b += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				int v = max(0, min(height - 1, i - r));

				float* aptr = a.ptr<float>(v, j);
				float* bptr = b.ptr<float>(v, j);

				__m256 mSum_a = _mm256_load_ps(aptr);
				__m256 mSum_b = _mm256_load_ps(bptr);

				for (int k = 1; k < d; k++)
				{
					int v = max(0, min(height - 1, i - r + k));

					float* aptr = a.ptr<float>(v, j);
					float* bptr = b.ptr<float>(v, j);
					mSum_a = _mm256_add_ps(mSum_a, _mm256_load_ps(aptr));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(bptr));
				}

				_mm256_store_ps(tp_a, mSum_a);
				_mm256_store_ps(tp_b, mSum_b);
				tp_a += 8;
				tp_b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* gptr = guide.ptr<float>(i);
			float* dptr = dest.ptr<float>(i);
			float* bptr = b.ptr<float>(i);
			float* aptr = a.ptr<float>(i);

			tp_a = temp.ptr<float>(0, roffset);
			tp_b = temp.ptr<float>(1, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a = _mm256_loadu_ps(tp_a);
				__m256 mSum_b = _mm256_loadu_ps(tp_b);

				for (int k = 1; k < d; k++)
				{
					mSum_a = _mm256_add_ps(mSum_a, _mm256_loadu_ps(tp_a + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}

				const __m256 m_a = _mm256_mul_ps(mSum_a, mDiv);
				const __m256 m_b = _mm256_mul_ps(mSum_b, mDiv);

				_mm256_store_ps(dptr, _mm256_fmadd_ps(m_a, _mm256_load_ps(gptr), m_b));
				aptr += 8;
				bptr += 8;
				gptr += 8;
				dptr += 8;

				tp_a += 8;
				tp_b += 8;
			}
		}
	}
}


void Ip2ab_Guide3_sep_VHI_Unroll2_AVX(Mat& I_b, Mat& I_g, Mat& I_r, Mat& p, const int r, float eps,
	Mat& a_b, Mat& a_g, Mat& a_r, Mat& b)
{
	const int width = I_b.cols;
	const int height = I_b.rows;

	Size size = Size(width, height);
	a_b.create(size, CV_32F);
	a_g.create(size, CV_32F);
	a_r.create(size, CV_32F);
	b.create(size, CV_32F);

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	__m256 mDiv = _mm256_set1_ps(1.f / ((2 * r + 1)*(2 * r + 1)));

	Mat temp(Size(width + 2 * R, 13), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp_I_b = temp.ptr<float>(0, R);
		float* tp_I_g = temp.ptr<float>(1, R);
		float* tp_I_r = temp.ptr<float>(2, R);

		float* tp_I_bb = temp.ptr<float>(3, R);
		float* tp_I_bg = temp.ptr<float>(4, R);
		float* tp_I_br = temp.ptr<float>(5, R);
		float* tp_I_gg = temp.ptr<float>(6, R);
		float* tp_I_gr = temp.ptr<float>(7, R);
		float* tp_I_rr = temp.ptr<float>(8, R);

		float* tp_p = temp.ptr<float>(9, R);
		float* tp_Ip_b = temp.ptr<float>(10, R);
		float* tp_Ip_g = temp.ptr<float>(11, R);
		float* tp_Ip_r = temp.ptr<float>(12, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* I_bptr = I_b.ptr<float>(i - r, j);
				float* I_gptr = I_g.ptr<float>(i - r, j);
				float* I_rptr = I_r.ptr<float>(i - r, j);
				float* pptr = p.ptr<float>(i - r, j);

				__m256 mSum_Ib = _mm256_load_ps(I_bptr);
				__m256 mSum_Ig = _mm256_load_ps(I_gptr);
				__m256 mSum_Ir = _mm256_load_ps(I_rptr);

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p = _mm256_load_ps(pptr);

				__m256 mSum_Ipb = _mm256_mul_ps(mSum_Ib, mSum_p);
				__m256 mSum_Ipg = _mm256_mul_ps(mSum_Ig, mSum_p);
				__m256 mSum_Ipr = _mm256_mul_ps(mSum_Ir, mSum_p);

				I_bptr += width;
				I_gptr += width;
				I_rptr += width;
				pptr += width;
				const int step = 2 * width;
				for (int k = 1; k <= r; k++)
				{
					__m256 mb0 = _mm256_load_ps(I_bptr);
					__m256 mb1 = _mm256_load_ps(I_bptr + width);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb1);

					__m256 mg0 = _mm256_load_ps(I_gptr);
					__m256 mg1 = _mm256_load_ps(I_gptr + width);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg1);

					__m256 mr0 = _mm256_load_ps(I_rptr);
					__m256 mr1 = _mm256_load_ps(I_rptr + width);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr1);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibb = _mm256_fmadd_ps(mb1, mb1, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibg = _mm256_fmadd_ps(mb1, mg1, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Ibr = _mm256_fmadd_ps(mb1, mr1, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igg = _mm256_fmadd_ps(mg1, mg1, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Igr = _mm256_fmadd_ps(mg1, mr1, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);
					mSum_Irr = _mm256_fmadd_ps(mr1, mr1, mSum_Irr);

					__m256 mp0 = _mm256_load_ps(pptr);
					__m256 mp1 = _mm256_load_ps(pptr + width);
					mSum_p = _mm256_add_ps(mSum_p, mp0);
					mSum_p = _mm256_add_ps(mSum_p, mp1);

					mSum_Ipb = _mm256_fmadd_ps(mp0, mb0, mSum_Ipb);
					mSum_Ipb = _mm256_fmadd_ps(mp1, mb1, mSum_Ipb);

					mSum_Ipg = _mm256_fmadd_ps(mp0, mg0, mSum_Ipg);
					mSum_Ipg = _mm256_fmadd_ps(mp1, mg1, mSum_Ipg);

					mSum_Ipr = _mm256_fmadd_ps(mp0, mr0, mSum_Ipr);
					mSum_Ipr = _mm256_fmadd_ps(mp1, mr1, mSum_Ipr);

					I_bptr += step;
					I_gptr += step;
					I_rptr += step;
					pptr += step;
				}

				_mm256_store_ps(tp_I_b, mSum_Ib);
				_mm256_store_ps(tp_I_g, mSum_Ig);
				_mm256_store_ps(tp_I_r, mSum_Ir);

				_mm256_store_ps(tp_I_bb, mSum_Ibb);
				_mm256_store_ps(tp_I_bg, mSum_Ibg);
				_mm256_store_ps(tp_I_br, mSum_Ibr);
				_mm256_store_ps(tp_I_gg, mSum_Igg);
				_mm256_store_ps(tp_I_gr, mSum_Igr);
				_mm256_store_ps(tp_I_rr, mSum_Irr);

				_mm256_store_ps(tp_p, mSum_p);

				_mm256_store_ps(tp_Ip_b, mSum_Ipb);
				_mm256_store_ps(tp_Ip_g, mSum_Ipg);
				_mm256_store_ps(tp_Ip_r, mSum_Ipr);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b_p = b.ptr<float>(i);
			float* ab = a_b.ptr<float>(i);
			float* ag = a_g.ptr<float>(i);
			float* ar = a_r.ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp_p = temp.ptr<float>(9, roffset);
			tp_Ip_b = temp.ptr<float>(10, roffset);
			tp_Ip_g = temp.ptr<float>(11, roffset);
			tp_Ip_r = temp.ptr<float>(12, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I_b = _mm256_loadu_ps(tp_I_b);
				__m256 mSum_I_g = _mm256_loadu_ps(tp_I_g);
				__m256 mSum_I_r = _mm256_loadu_ps(tp_I_r);

				__m256 mSum_I_bb = _mm256_loadu_ps(tp_I_bb);
				__m256 mSum_I_bg = _mm256_loadu_ps(tp_I_bg);
				__m256 mSum_I_br = _mm256_loadu_ps(tp_I_br);
				__m256 mSum_I_gg = _mm256_loadu_ps(tp_I_gg);
				__m256 mSum_I_gr = _mm256_loadu_ps(tp_I_gr);
				__m256 mSum_I_rr = _mm256_loadu_ps(tp_I_rr);

				__m256 mSum_p = _mm256_loadu_ps(tp_p);

				__m256 mSum_Ip_b = _mm256_loadu_ps(tp_Ip_b);
				__m256 mSum_Ip_g = _mm256_loadu_ps(tp_Ip_g);
				__m256 mSum_Ip_r = _mm256_loadu_ps(tp_Ip_r);

				for (int k = 1; k <= r; k++)
				{
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b + k));
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b + k + r));

					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g + k));
					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g + k + r));

					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r + k));
					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r + k + r));


					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb + k));
					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb + k + r));

					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg + k));
					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg + k + r));

					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br + k));
					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br + k + r));

					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg + k));
					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg + k + r));

					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr + k));
					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr + k + r));

					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr + k));
					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr + k + r));

					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p + k));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p + k + r));

					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b + k));
					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b + k + r));

					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g + k));
					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g + k + r));

					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r + k));
					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r + k + r));
				}

				__m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				__m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				__m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				__m256 meps = _mm256_set1_ps(eps);

				__m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				__m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				__m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				__m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				__m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				__m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				__m256 mp = _mm256_mul_ps(mSum_p, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp, _mm256_mul_ps(mSum_Ip_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp, _mm256_mul_ps(mSum_Ip_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp, _mm256_mul_ps(mSum_Ip_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp);
				_mm256_store_ps(ab, mTmp);
				ab += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);
				_mm256_store_ps(ag, mTmp);
				ag += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);
				_mm256_store_ps(ar, mTmp);
				ar += 8;

				_mm256_store_ps(b_p, mB);

				b_p += 8;

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_Ib = _mm256_load_ps(I_b.ptr<float>(i, j));
				__m256 mSum_Ig = _mm256_load_ps(I_g.ptr<float>(i, j));
				__m256 mSum_Ir = _mm256_load_ps(I_r.ptr<float>(i, j));

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p = _mm256_load_ps(p.ptr<float>(i, j));

				__m256 mSum_Ipb = _mm256_mul_ps(mSum_Ib, mSum_p);
				__m256 mSum_Ipg = _mm256_mul_ps(mSum_Ig, mSum_p);
				__m256 mSum_Ipr = _mm256_mul_ps(mSum_Ir, mSum_p);

				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = I_b.ptr<float>(vl, j);
					float* sp2 = I_b.ptr<float>(vh, j);
					__m256 mb0 = _mm256_load_ps(sp1);
					__m256 mb1 = _mm256_load_ps(sp2);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb1);

					sp1 = I_g.ptr<float>(vl, j);
					sp2 = I_g.ptr<float>(vh, j);
					__m256 mg0 = _mm256_load_ps(sp1);
					__m256 mg1 = _mm256_load_ps(sp2);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg1);

					sp1 = I_r.ptr<float>(vl, j);
					sp2 = I_r.ptr<float>(vh, j);
					__m256 mr0 = _mm256_load_ps(sp1);
					__m256 mr1 = _mm256_load_ps(sp2);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr1);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibb = _mm256_fmadd_ps(mb1, mb1, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibg = _mm256_fmadd_ps(mb1, mg1, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Ibr = _mm256_fmadd_ps(mb1, mr1, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igg = _mm256_fmadd_ps(mg1, mg1, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Igr = _mm256_fmadd_ps(mg1, mr1, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);
					mSum_Irr = _mm256_fmadd_ps(mr1, mr1, mSum_Irr);

					sp1 = p.ptr<float>(vl, j);
					sp2 = p.ptr<float>(vh, j);
					__m256 mp0 = _mm256_load_ps(sp1);
					__m256 mp1 = _mm256_load_ps(sp2);
					mSum_p = _mm256_add_ps(mSum_p, mp0);
					mSum_p = _mm256_add_ps(mSum_p, mp1);

					mSum_Ipb = _mm256_fmadd_ps(mp0, mb0, mSum_Ipb);
					mSum_Ipb = _mm256_fmadd_ps(mp1, mb1, mSum_Ipb);

					mSum_Ipg = _mm256_fmadd_ps(mp0, mg0, mSum_Ipg);
					mSum_Ipg = _mm256_fmadd_ps(mp1, mg1, mSum_Ipg);

					mSum_Ipr = _mm256_fmadd_ps(mp0, mr0, mSum_Ipr);
					mSum_Ipr = _mm256_fmadd_ps(mp1, mr1, mSum_Ipr);
				}

				_mm256_store_ps(tp_I_b, mSum_Ib);
				_mm256_store_ps(tp_I_g, mSum_Ig);
				_mm256_store_ps(tp_I_r, mSum_Ir);

				_mm256_store_ps(tp_I_bb, mSum_Ibb);
				_mm256_store_ps(tp_I_bg, mSum_Ibg);
				_mm256_store_ps(tp_I_br, mSum_Ibr);
				_mm256_store_ps(tp_I_gg, mSum_Igg);
				_mm256_store_ps(tp_I_gr, mSum_Igr);
				_mm256_store_ps(tp_I_rr, mSum_Irr);

				_mm256_store_ps(tp_p, mSum_p);

				_mm256_store_ps(tp_Ip_b, mSum_Ipb);
				_mm256_store_ps(tp_Ip_g, mSum_Ipg);
				_mm256_store_ps(tp_Ip_r, mSum_Ipr);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b_p = b.ptr<float>(i);
			float* ab = a_b.ptr<float>(i);
			float* ag = a_g.ptr<float>(i);
			float* ar = a_r.ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp_p = temp.ptr<float>(9, roffset);
			tp_Ip_b = temp.ptr<float>(10, roffset);
			tp_Ip_g = temp.ptr<float>(11, roffset);
			tp_Ip_r = temp.ptr<float>(12, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I_b = _mm256_loadu_ps(tp_I_b + r);
				__m256 mSum_I_g = _mm256_loadu_ps(tp_I_g + r);
				__m256 mSum_I_r = _mm256_loadu_ps(tp_I_r + r);

				__m256 mSum_I_bb = _mm256_loadu_ps(tp_I_bb + r);
				__m256 mSum_I_bg = _mm256_loadu_ps(tp_I_bg + r);
				__m256 mSum_I_br = _mm256_loadu_ps(tp_I_br + r);
				__m256 mSum_I_gg = _mm256_loadu_ps(tp_I_gg + r);
				__m256 mSum_I_gr = _mm256_loadu_ps(tp_I_gr + r);
				__m256 mSum_I_rr = _mm256_loadu_ps(tp_I_rr + r);

				__m256 mSum_p = _mm256_loadu_ps(tp_p + r);

				__m256 mSum_Ip_b = _mm256_loadu_ps(tp_Ip_b + r);
				__m256 mSum_Ip_g = _mm256_loadu_ps(tp_Ip_g + r);
				__m256 mSum_Ip_r = _mm256_loadu_ps(tp_Ip_r + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b - k + r));
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b + k + r));

					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g - k + r));
					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g + k + r));

					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r - k + r));
					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r + k + r));


					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb - k + r));
					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb + k + r));

					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg - k + r));
					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg + k + r));

					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br - k + r));
					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br + k + r));

					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg - k + r));
					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg + k + r));

					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr - k + r));
					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr + k + r));

					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr - k + r));
					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr + k + r));

					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p - k + r));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p + k + r));

					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b - k + r));
					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b + k + r));

					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g - k + r));
					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g + k + r));

					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r - k + r));
					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r + k + r));
				}

				__m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				__m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				__m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				__m256 meps = _mm256_set1_ps(eps);

				__m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				__m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				__m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				__m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				__m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				__m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				__m256 mp = _mm256_mul_ps(mSum_p, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp, _mm256_mul_ps(mSum_Ip_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp, _mm256_mul_ps(mSum_Ip_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp, _mm256_mul_ps(mSum_Ip_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp);
				_mm256_store_ps(ab, mTmp);
				ab += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);
				_mm256_store_ps(ag, mTmp);
				ag += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);
				_mm256_store_ps(ar, mTmp);
				ar += 8;

				_mm256_store_ps(b_p, mB);

				b_p += 8;

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}
		}
	}
}

void Ip2ab_Guide3_sep_VHI_Unroll2_AVX_omp(Mat& I_b, Mat& I_g, Mat& I_r, Mat& p, const int r, float eps,
	Mat& a_b, Mat& a_g, Mat& a_r, Mat& b)
{
	const int width = I_b.cols;
	const int height = I_b.rows;

	Size size = Size(width, height);
	a_b.create(size, CV_32F);
	a_g.create(size, CV_32F);
	a_r.create(size, CV_32F);
	b.create(size, CV_32F);

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	__m256 mDiv = _mm256_set1_ps(1.f / ((2 * r + 1)*(2 * r + 1)));

	Mat buff(Size(width + 2 * R, 13 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 13 * omp_get_thread_num(), width + 2 * R, 13));

		float* tp_I_b = temp.ptr<float>(0, R);
		float* tp_I_g = temp.ptr<float>(1, R);
		float* tp_I_r = temp.ptr<float>(2, R);

		float* tp_I_bb = temp.ptr<float>(3, R);
		float* tp_I_bg = temp.ptr<float>(4, R);
		float* tp_I_br = temp.ptr<float>(5, R);
		float* tp_I_gg = temp.ptr<float>(6, R);
		float* tp_I_gr = temp.ptr<float>(7, R);
		float* tp_I_rr = temp.ptr<float>(8, R);

		float* tp_p = temp.ptr<float>(9, R);
		float* tp_Ip_b = temp.ptr<float>(10, R);
		float* tp_Ip_g = temp.ptr<float>(11, R);
		float* tp_Ip_r = temp.ptr<float>(12, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* I_bptr = I_b.ptr<float>(i - r, j);
				float* I_gptr = I_g.ptr<float>(i - r, j);
				float* I_rptr = I_r.ptr<float>(i - r, j);
				float* pptr = p.ptr<float>(i - r, j);

				__m256 mSum_Ib = _mm256_load_ps(I_bptr);
				__m256 mSum_Ig = _mm256_load_ps(I_gptr);
				__m256 mSum_Ir = _mm256_load_ps(I_rptr);

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p = _mm256_load_ps(pptr);

				__m256 mSum_Ipb = _mm256_mul_ps(mSum_Ib, mSum_p);
				__m256 mSum_Ipg = _mm256_mul_ps(mSum_Ig, mSum_p);
				__m256 mSum_Ipr = _mm256_mul_ps(mSum_Ir, mSum_p);

				I_bptr += width;
				I_gptr += width;
				I_rptr += width;
				pptr += width;
				const int step = 2 * width;
				for (int k = 1; k <= r; k++)
				{
					__m256 mb0 = _mm256_load_ps(I_bptr);
					__m256 mb1 = _mm256_load_ps(I_bptr + width);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb1);

					__m256 mg0 = _mm256_load_ps(I_gptr);
					__m256 mg1 = _mm256_load_ps(I_gptr + width);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg1);

					__m256 mr0 = _mm256_load_ps(I_rptr);
					__m256 mr1 = _mm256_load_ps(I_rptr + width);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr1);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibb = _mm256_fmadd_ps(mb1, mb1, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibg = _mm256_fmadd_ps(mb1, mg1, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Ibr = _mm256_fmadd_ps(mb1, mr1, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igg = _mm256_fmadd_ps(mg1, mg1, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Igr = _mm256_fmadd_ps(mg1, mr1, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);
					mSum_Irr = _mm256_fmadd_ps(mr1, mr1, mSum_Irr);

					__m256 mp0 = _mm256_load_ps(pptr);
					__m256 mp1 = _mm256_load_ps(pptr + width);
					mSum_p = _mm256_add_ps(mSum_p, mp0);
					mSum_p = _mm256_add_ps(mSum_p, mp1);

					mSum_Ipb = _mm256_fmadd_ps(mp0, mb0, mSum_Ipb);
					mSum_Ipb = _mm256_fmadd_ps(mp1, mb1, mSum_Ipb);

					mSum_Ipg = _mm256_fmadd_ps(mp0, mg0, mSum_Ipg);
					mSum_Ipg = _mm256_fmadd_ps(mp1, mg1, mSum_Ipg);

					mSum_Ipr = _mm256_fmadd_ps(mp0, mr0, mSum_Ipr);
					mSum_Ipr = _mm256_fmadd_ps(mp1, mr1, mSum_Ipr);

					I_bptr += step;
					I_gptr += step;
					I_rptr += step;
					pptr += step;
				}

				_mm256_store_ps(tp_I_b, mSum_Ib);
				_mm256_store_ps(tp_I_g, mSum_Ig);
				_mm256_store_ps(tp_I_r, mSum_Ir);

				_mm256_store_ps(tp_I_bb, mSum_Ibb);
				_mm256_store_ps(tp_I_bg, mSum_Ibg);
				_mm256_store_ps(tp_I_br, mSum_Ibr);
				_mm256_store_ps(tp_I_gg, mSum_Igg);
				_mm256_store_ps(tp_I_gr, mSum_Igr);
				_mm256_store_ps(tp_I_rr, mSum_Irr);

				_mm256_store_ps(tp_p, mSum_p);

				_mm256_store_ps(tp_Ip_b, mSum_Ipb);
				_mm256_store_ps(tp_Ip_g, mSum_Ipg);
				_mm256_store_ps(tp_Ip_r, mSum_Ipr);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b_p = b.ptr<float>(i);
			float* ab = a_b.ptr<float>(i);
			float* ag = a_g.ptr<float>(i);
			float* ar = a_r.ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp_p = temp.ptr<float>(9, roffset);
			tp_Ip_b = temp.ptr<float>(10, roffset);
			tp_Ip_g = temp.ptr<float>(11, roffset);
			tp_Ip_r = temp.ptr<float>(12, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I_b = _mm256_loadu_ps(tp_I_b);
				__m256 mSum_I_g = _mm256_loadu_ps(tp_I_g);
				__m256 mSum_I_r = _mm256_loadu_ps(tp_I_r);

				__m256 mSum_I_bb = _mm256_loadu_ps(tp_I_bb);
				__m256 mSum_I_bg = _mm256_loadu_ps(tp_I_bg);
				__m256 mSum_I_br = _mm256_loadu_ps(tp_I_br);
				__m256 mSum_I_gg = _mm256_loadu_ps(tp_I_gg);
				__m256 mSum_I_gr = _mm256_loadu_ps(tp_I_gr);
				__m256 mSum_I_rr = _mm256_loadu_ps(tp_I_rr);

				__m256 mSum_p = _mm256_loadu_ps(tp_p);

				__m256 mSum_Ip_b = _mm256_loadu_ps(tp_Ip_b);
				__m256 mSum_Ip_g = _mm256_loadu_ps(tp_Ip_g);
				__m256 mSum_Ip_r = _mm256_loadu_ps(tp_Ip_r);

				for (int k = 1; k <= r; k++)
				{
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b + k));
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b + k + r));

					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g + k));
					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g + k + r));

					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r + k));
					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r + k + r));


					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb + k));
					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb + k + r));

					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg + k));
					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg + k + r));

					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br + k));
					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br + k + r));

					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg + k));
					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg + k + r));

					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr + k));
					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr + k + r));

					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr + k));
					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr + k + r));

					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p + k));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p + k + r));

					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b + k));
					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b + k + r));

					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g + k));
					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g + k + r));

					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r + k));
					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r + k + r));
				}

				__m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				__m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				__m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				__m256 meps = _mm256_set1_ps(eps);

				__m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				__m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				__m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				__m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				__m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				__m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				__m256 mp = _mm256_mul_ps(mSum_p, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp, _mm256_mul_ps(mSum_Ip_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp, _mm256_mul_ps(mSum_Ip_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp, _mm256_mul_ps(mSum_Ip_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp);
				_mm256_store_ps(ab, mTmp);
				ab += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);
				_mm256_store_ps(ag, mTmp);
				ag += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);
				_mm256_store_ps(ar, mTmp);
				ar += 8;

				_mm256_store_ps(b_p, mB);

				b_p += 8;

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_Ib = _mm256_load_ps(I_b.ptr<float>(i, j));
				__m256 mSum_Ig = _mm256_load_ps(I_g.ptr<float>(i, j));
				__m256 mSum_Ir = _mm256_load_ps(I_r.ptr<float>(i, j));

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p = _mm256_load_ps(p.ptr<float>(i, j));

				__m256 mSum_Ipb = _mm256_mul_ps(mSum_Ib, mSum_p);
				__m256 mSum_Ipg = _mm256_mul_ps(mSum_Ig, mSum_p);
				__m256 mSum_Ipr = _mm256_mul_ps(mSum_Ir, mSum_p);

				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = I_b.ptr<float>(vl, j);
					float* sp2 = I_b.ptr<float>(vh, j);
					__m256 mb0 = _mm256_load_ps(sp1);
					__m256 mb1 = _mm256_load_ps(sp2);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb1);

					sp1 = I_g.ptr<float>(vl, j);
					sp2 = I_g.ptr<float>(vh, j);
					__m256 mg0 = _mm256_load_ps(sp1);
					__m256 mg1 = _mm256_load_ps(sp2);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg1);

					sp1 = I_r.ptr<float>(vl, j);
					sp2 = I_r.ptr<float>(vh, j);
					__m256 mr0 = _mm256_load_ps(sp1);
					__m256 mr1 = _mm256_load_ps(sp2);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr1);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibb = _mm256_fmadd_ps(mb1, mb1, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibg = _mm256_fmadd_ps(mb1, mg1, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Ibr = _mm256_fmadd_ps(mb1, mr1, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igg = _mm256_fmadd_ps(mg1, mg1, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Igr = _mm256_fmadd_ps(mg1, mr1, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);
					mSum_Irr = _mm256_fmadd_ps(mr1, mr1, mSum_Irr);

					sp1 = p.ptr<float>(vl, j);
					sp2 = p.ptr<float>(vh, j);
					__m256 mp0 = _mm256_load_ps(sp1);
					__m256 mp1 = _mm256_load_ps(sp2);
					mSum_p = _mm256_add_ps(mSum_p, mp0);
					mSum_p = _mm256_add_ps(mSum_p, mp1);

					mSum_Ipb = _mm256_fmadd_ps(mp0, mb0, mSum_Ipb);
					mSum_Ipb = _mm256_fmadd_ps(mp1, mb1, mSum_Ipb);

					mSum_Ipg = _mm256_fmadd_ps(mp0, mg0, mSum_Ipg);
					mSum_Ipg = _mm256_fmadd_ps(mp1, mg1, mSum_Ipg);

					mSum_Ipr = _mm256_fmadd_ps(mp0, mr0, mSum_Ipr);
					mSum_Ipr = _mm256_fmadd_ps(mp1, mr1, mSum_Ipr);
				}

				_mm256_store_ps(tp_I_b, mSum_Ib);
				_mm256_store_ps(tp_I_g, mSum_Ig);
				_mm256_store_ps(tp_I_r, mSum_Ir);

				_mm256_store_ps(tp_I_bb, mSum_Ibb);
				_mm256_store_ps(tp_I_bg, mSum_Ibg);
				_mm256_store_ps(tp_I_br, mSum_Ibr);
				_mm256_store_ps(tp_I_gg, mSum_Igg);
				_mm256_store_ps(tp_I_gr, mSum_Igr);
				_mm256_store_ps(tp_I_rr, mSum_Irr);

				_mm256_store_ps(tp_p, mSum_p);

				_mm256_store_ps(tp_Ip_b, mSum_Ipb);
				_mm256_store_ps(tp_Ip_g, mSum_Ipg);
				_mm256_store_ps(tp_Ip_r, mSum_Ipr);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b_p = b.ptr<float>(i);
			float* ab = a_b.ptr<float>(i);
			float* ag = a_g.ptr<float>(i);
			float* ar = a_r.ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp_p = temp.ptr<float>(9, roffset);
			tp_Ip_b = temp.ptr<float>(10, roffset);
			tp_Ip_g = temp.ptr<float>(11, roffset);
			tp_Ip_r = temp.ptr<float>(12, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I_b = _mm256_loadu_ps(tp_I_b + r);
				__m256 mSum_I_g = _mm256_loadu_ps(tp_I_g + r);
				__m256 mSum_I_r = _mm256_loadu_ps(tp_I_r + r);

				__m256 mSum_I_bb = _mm256_loadu_ps(tp_I_bb + r);
				__m256 mSum_I_bg = _mm256_loadu_ps(tp_I_bg + r);
				__m256 mSum_I_br = _mm256_loadu_ps(tp_I_br + r);
				__m256 mSum_I_gg = _mm256_loadu_ps(tp_I_gg + r);
				__m256 mSum_I_gr = _mm256_loadu_ps(tp_I_gr + r);
				__m256 mSum_I_rr = _mm256_loadu_ps(tp_I_rr + r);

				__m256 mSum_p = _mm256_loadu_ps(tp_p + r);

				__m256 mSum_Ip_b = _mm256_loadu_ps(tp_Ip_b + r);
				__m256 mSum_Ip_g = _mm256_loadu_ps(tp_Ip_g + r);
				__m256 mSum_Ip_r = _mm256_loadu_ps(tp_Ip_r + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b - k + r));
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b + k + r));

					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g - k + r));
					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g + k + r));

					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r - k + r));
					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r + k + r));


					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb - k + r));
					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb + k + r));

					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg - k + r));
					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg + k + r));

					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br - k + r));
					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br + k + r));

					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg - k + r));
					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg + k + r));

					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr - k + r));
					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr + k + r));

					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr - k + r));
					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr + k + r));

					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p - k + r));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p + k + r));

					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b - k + r));
					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b + k + r));

					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g - k + r));
					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g + k + r));

					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r - k + r));
					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r + k + r));
				}

				__m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				__m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				__m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				__m256 meps = _mm256_set1_ps(eps);

				__m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				__m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				__m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				__m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				__m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				__m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				__m256 mp = _mm256_mul_ps(mSum_p, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp, _mm256_mul_ps(mSum_Ip_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp, _mm256_mul_ps(mSum_Ip_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp, _mm256_mul_ps(mSum_Ip_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp);
				_mm256_store_ps(ab, mTmp);
				ab += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);
				_mm256_store_ps(ag, mTmp);
				ag += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);
				_mm256_store_ps(ar, mTmp);
				ar += 8;

				_mm256_store_ps(b_p, mB);

				b_p += 8;

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}
		}
	}
}


void Ip2ab_Guide3_sep_VHI_AVX(Mat& I_b, Mat& I_g, Mat& I_r, Mat& p, const int r, float eps,
	Mat& a_b, Mat& a_g, Mat& a_r, Mat& b)
{
	const int width = I_b.cols;
	const int height = I_b.rows;

	Size size = Size(width, height);
	a_b.create(size, CV_32F);
	a_g.create(size, CV_32F);
	a_r.create(size, CV_32F);
	b.create(size, CV_32F);

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat temp(Size(width + 2 * R, 13), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp_I_b = temp.ptr<float>(0, R);
		float* tp_I_g = temp.ptr<float>(1, R);
		float* tp_I_r = temp.ptr<float>(2, R);

		float* tp_I_bb = temp.ptr<float>(3, R);
		float* tp_I_bg = temp.ptr<float>(4, R);
		float* tp_I_br = temp.ptr<float>(5, R);
		float* tp_I_gg = temp.ptr<float>(6, R);
		float* tp_I_gr = temp.ptr<float>(7, R);
		float* tp_I_rr = temp.ptr<float>(8, R);

		float* tp_p = temp.ptr<float>(9, R);
		float* tp_Ip_b = temp.ptr<float>(10, R);
		float* tp_Ip_g = temp.ptr<float>(11, R);
		float* tp_Ip_r = temp.ptr<float>(12, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* I_bptr = I_b.ptr<float>(i - r, j);
				float* I_gptr = I_g.ptr<float>(i - r, j);
				float* I_rptr = I_r.ptr<float>(i - r, j);
				float* pptr = p.ptr<float>(i - r, j);

				__m256 mSum_Ib = _mm256_load_ps(I_bptr);
				__m256 mSum_Ig = _mm256_load_ps(I_gptr);
				__m256 mSum_Ir = _mm256_load_ps(I_rptr);

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p = _mm256_load_ps(pptr);

				__m256 mSum_Ipb = _mm256_mul_ps(mSum_Ib, mSum_p);
				__m256 mSum_Ipg = _mm256_mul_ps(mSum_Ig, mSum_p);
				__m256 mSum_Ipr = _mm256_mul_ps(mSum_Ir, mSum_p);

				I_bptr += width;
				I_gptr += width;
				I_rptr += width;
				pptr += width;

				for (int k = 1; k < d; k++)
				{
					__m256 mb0 = _mm256_load_ps(I_bptr);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);

					__m256 mg0 = _mm256_load_ps(I_gptr);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);

					__m256 mr0 = _mm256_load_ps(I_rptr);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);

					__m256 mp0 = _mm256_load_ps(pptr);
					mSum_p = _mm256_add_ps(mSum_p, mp0);

					mSum_Ipb = _mm256_fmadd_ps(mp0, mb0, mSum_Ipb);
					mSum_Ipg = _mm256_fmadd_ps(mp0, mg0, mSum_Ipg);
					mSum_Ipr = _mm256_fmadd_ps(mp0, mr0, mSum_Ipr);

					I_bptr += width;
					I_gptr += width;
					I_rptr += width;
					pptr += width;
				}

				_mm256_store_ps(tp_I_b, mSum_Ib);
				_mm256_store_ps(tp_I_g, mSum_Ig);
				_mm256_store_ps(tp_I_r, mSum_Ir);

				_mm256_store_ps(tp_I_bb, mSum_Ibb);
				_mm256_store_ps(tp_I_bg, mSum_Ibg);
				_mm256_store_ps(tp_I_br, mSum_Ibr);
				_mm256_store_ps(tp_I_gg, mSum_Igg);
				_mm256_store_ps(tp_I_gr, mSum_Igr);
				_mm256_store_ps(tp_I_rr, mSum_Irr);

				_mm256_store_ps(tp_p, mSum_p);

				_mm256_store_ps(tp_Ip_b, mSum_Ipb);
				_mm256_store_ps(tp_Ip_g, mSum_Ipg);
				_mm256_store_ps(tp_Ip_r, mSum_Ipr);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b_p = b.ptr<float>(i);
			float* ab = a_b.ptr<float>(i);
			float* ag = a_g.ptr<float>(i);
			float* ar = a_r.ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp_p = temp.ptr<float>(9, roffset);
			tp_Ip_b = temp.ptr<float>(10, roffset);
			tp_Ip_g = temp.ptr<float>(11, roffset);
			tp_Ip_r = temp.ptr<float>(12, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I_b = _mm256_loadu_ps(tp_I_b);
				__m256 mSum_I_g = _mm256_loadu_ps(tp_I_g);
				__m256 mSum_I_r = _mm256_loadu_ps(tp_I_r);

				__m256 mSum_I_bb = _mm256_loadu_ps(tp_I_bb);
				__m256 mSum_I_bg = _mm256_loadu_ps(tp_I_bg);
				__m256 mSum_I_br = _mm256_loadu_ps(tp_I_br);
				__m256 mSum_I_gg = _mm256_loadu_ps(tp_I_gg);
				__m256 mSum_I_gr = _mm256_loadu_ps(tp_I_gr);
				__m256 mSum_I_rr = _mm256_loadu_ps(tp_I_rr);

				__m256 mSum_p = _mm256_loadu_ps(tp_p);

				__m256 mSum_Ip_b = _mm256_loadu_ps(tp_Ip_b);
				__m256 mSum_Ip_g = _mm256_loadu_ps(tp_Ip_g);
				__m256 mSum_Ip_r = _mm256_loadu_ps(tp_Ip_r);

				for (int k = 1; k < d; k++)
				{
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b + k));
					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g + k));
					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r + k));

					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb + k));
					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg + k));
					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br + k));
					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg + k));
					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr + k));
					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr + k));

					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p + k));

					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b + k));
					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g + k));
					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r + k));
				}

				__m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				__m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				__m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				__m256 meps = _mm256_set1_ps(eps);

				__m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				__m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				__m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				__m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				__m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				__m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				__m256 mp = _mm256_mul_ps(mSum_p, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp, _mm256_mul_ps(mSum_Ip_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp, _mm256_mul_ps(mSum_Ip_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp, _mm256_mul_ps(mSum_Ip_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp);
				_mm256_store_ps(ab, mTmp);
				ab += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);
				_mm256_store_ps(ag, mTmp);
				ag += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);
				_mm256_store_ps(ar, mTmp);
				ar += 8;

				_mm256_store_ps(b_p, mB);

				b_p += 8;

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_Ib = _mm256_load_ps(I_b.ptr<float>(i, j));
				__m256 mSum_Ig = _mm256_load_ps(I_g.ptr<float>(i, j));
				__m256 mSum_Ir = _mm256_load_ps(I_r.ptr<float>(i, j));

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p = _mm256_load_ps(p.ptr<float>(i, j));

				__m256 mSum_Ipb = _mm256_mul_ps(mSum_Ib, mSum_p);
				__m256 mSum_Ipg = _mm256_mul_ps(mSum_Ig, mSum_p);
				__m256 mSum_Ipr = _mm256_mul_ps(mSum_Ir, mSum_p);

				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = I_b.ptr<float>(vl, j);
					float* sp2 = I_b.ptr<float>(vh, j);
					__m256 mb0 = _mm256_load_ps(sp1);
					__m256 mb1 = _mm256_load_ps(sp2);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb1);

					sp1 = I_g.ptr<float>(vl, j);
					sp2 = I_g.ptr<float>(vh, j);
					__m256 mg0 = _mm256_load_ps(sp1);
					__m256 mg1 = _mm256_load_ps(sp2);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg1);

					sp1 = I_r.ptr<float>(vl, j);
					sp2 = I_r.ptr<float>(vh, j);
					__m256 mr0 = _mm256_load_ps(sp1);
					__m256 mr1 = _mm256_load_ps(sp2);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr1);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibb = _mm256_fmadd_ps(mb1, mb1, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibg = _mm256_fmadd_ps(mb1, mg1, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Ibr = _mm256_fmadd_ps(mb1, mr1, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igg = _mm256_fmadd_ps(mg1, mg1, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Igr = _mm256_fmadd_ps(mg1, mr1, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);
					mSum_Irr = _mm256_fmadd_ps(mr1, mr1, mSum_Irr);

					sp1 = p.ptr<float>(vl, j);
					sp2 = p.ptr<float>(vh, j);
					__m256 mp0 = _mm256_load_ps(sp1);
					__m256 mp1 = _mm256_load_ps(sp2);
					mSum_p = _mm256_add_ps(mSum_p, mp0);
					mSum_p = _mm256_add_ps(mSum_p, mp1);

					mSum_Ipb = _mm256_fmadd_ps(mp0, mb0, mSum_Ipb);
					mSum_Ipb = _mm256_fmadd_ps(mp1, mb1, mSum_Ipb);

					mSum_Ipg = _mm256_fmadd_ps(mp0, mg0, mSum_Ipg);
					mSum_Ipg = _mm256_fmadd_ps(mp1, mg1, mSum_Ipg);

					mSum_Ipr = _mm256_fmadd_ps(mp0, mr0, mSum_Ipr);
					mSum_Ipr = _mm256_fmadd_ps(mp1, mr1, mSum_Ipr);
				}

				_mm256_store_ps(tp_I_b, mSum_Ib);
				_mm256_store_ps(tp_I_g, mSum_Ig);
				_mm256_store_ps(tp_I_r, mSum_Ir);

				_mm256_store_ps(tp_I_bb, mSum_Ibb);
				_mm256_store_ps(tp_I_bg, mSum_Ibg);
				_mm256_store_ps(tp_I_br, mSum_Ibr);
				_mm256_store_ps(tp_I_gg, mSum_Igg);
				_mm256_store_ps(tp_I_gr, mSum_Igr);
				_mm256_store_ps(tp_I_rr, mSum_Irr);

				_mm256_store_ps(tp_p, mSum_p);

				_mm256_store_ps(tp_Ip_b, mSum_Ipb);
				_mm256_store_ps(tp_Ip_g, mSum_Ipg);
				_mm256_store_ps(tp_Ip_r, mSum_Ipr);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b_p = b.ptr<float>(i);
			float* ab = a_b.ptr<float>(i);
			float* ag = a_g.ptr<float>(i);
			float* ar = a_r.ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp_p = temp.ptr<float>(9, roffset);
			tp_Ip_b = temp.ptr<float>(10, roffset);
			tp_Ip_g = temp.ptr<float>(11, roffset);
			tp_Ip_r = temp.ptr<float>(12, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I_b = _mm256_loadu_ps(tp_I_b + r);
				__m256 mSum_I_g = _mm256_loadu_ps(tp_I_g + r);
				__m256 mSum_I_r = _mm256_loadu_ps(tp_I_r + r);

				__m256 mSum_I_bb = _mm256_loadu_ps(tp_I_bb + r);
				__m256 mSum_I_bg = _mm256_loadu_ps(tp_I_bg + r);
				__m256 mSum_I_br = _mm256_loadu_ps(tp_I_br + r);
				__m256 mSum_I_gg = _mm256_loadu_ps(tp_I_gg + r);
				__m256 mSum_I_gr = _mm256_loadu_ps(tp_I_gr + r);
				__m256 mSum_I_rr = _mm256_loadu_ps(tp_I_rr + r);

				__m256 mSum_p = _mm256_loadu_ps(tp_p + r);

				__m256 mSum_Ip_b = _mm256_loadu_ps(tp_Ip_b + r);
				__m256 mSum_Ip_g = _mm256_loadu_ps(tp_Ip_g + r);
				__m256 mSum_Ip_r = _mm256_loadu_ps(tp_Ip_r + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b - k + r));
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b + k + r));

					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g - k + r));
					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g + k + r));

					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r - k + r));
					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r + k + r));


					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb - k + r));
					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb + k + r));

					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg - k + r));
					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg + k + r));

					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br - k + r));
					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br + k + r));

					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg - k + r));
					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg + k + r));

					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr - k + r));
					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr + k + r));

					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr - k + r));
					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr + k + r));

					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p - k + r));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp_p + k + r));

					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b - k + r));
					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b + k + r));

					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g - k + r));
					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g + k + r));

					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r - k + r));
					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r + k + r));
				}

				__m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				__m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				__m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				__m256 meps = _mm256_set1_ps(eps);

				__m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				__m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				__m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				__m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				__m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				__m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				__m256 mp = _mm256_mul_ps(mSum_p, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp, _mm256_mul_ps(mSum_Ip_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp, _mm256_mul_ps(mSum_Ip_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp, _mm256_mul_ps(mSum_Ip_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp);
				_mm256_store_ps(ab, mTmp);
				ab += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);
				_mm256_store_ps(ag, mTmp);
				ag += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);
				_mm256_store_ps(ar, mTmp);
				ar += 8;

				_mm256_store_ps(b_p, mB);

				b_p += 8;

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}
		}
	}
}

void Ip2ab_Guide3_sep_VHI_AVX_omp(Mat& I_b, Mat& I_g, Mat& I_r, Mat& p, const int r, float eps,
	Mat& a_b, Mat& a_g, Mat& a_r, Mat& b)
{
	const int width = I_b.cols;
	const int height = I_b.rows;

	Size size = Size(width, height);
	a_b.create(size, CV_32F);
	a_g.create(size, CV_32F);
	a_r.create(size, CV_32F);
	b.create(size, CV_32F);

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat buff(Size(width + 2 * R, 13 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 13 * omp_get_thread_num(), width + 2 * R, 13));

		float* tp_I_b = temp.ptr<float>(0, R);
		float* tp_I_g = temp.ptr<float>(1, R);
		float* tp_I_r = temp.ptr<float>(2, R);

		float* tp_I_bb = temp.ptr<float>(3, R);
		float* tp_I_bg = temp.ptr<float>(4, R);
		float* tp_I_br = temp.ptr<float>(5, R);
		float* tp_I_gg = temp.ptr<float>(6, R);
		float* tp_I_gr = temp.ptr<float>(7, R);
		float* tp_I_rr = temp.ptr<float>(8, R);

		float* tp____p = temp.ptr<float>(9, R);
		float* tp_Ip_b = temp.ptr<float>(10, R);
		float* tp_Ip_g = temp.ptr<float>(11, R);
		float* tp_Ip_r = temp.ptr<float>(12, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* I_bptr = I_b.ptr<float>(i - r, j);
				float* I_gptr = I_g.ptr<float>(i - r, j);
				float* I_rptr = I_r.ptr<float>(i - r, j);
				float* pptr = p.ptr<float>(i - r, j);

				__m256 mSum_Ib = _mm256_load_ps(I_bptr);
				__m256 mSum_Ig = _mm256_load_ps(I_gptr);
				__m256 mSum_Ir = _mm256_load_ps(I_rptr);

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p = _mm256_load_ps(pptr);

				__m256 mSum_Ipb = _mm256_mul_ps(mSum_Ib, mSum_p);
				__m256 mSum_Ipg = _mm256_mul_ps(mSum_Ig, mSum_p);
				__m256 mSum_Ipr = _mm256_mul_ps(mSum_Ir, mSum_p);

				I_bptr += width;
				I_gptr += width;
				I_rptr += width;
				pptr += width;

				for (int k = 1; k < d; k++)
				{
					__m256 mb0 = _mm256_load_ps(I_bptr);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);

					__m256 mg0 = _mm256_load_ps(I_gptr);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);

					__m256 mr0 = _mm256_load_ps(I_rptr);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);

					__m256 mp0 = _mm256_load_ps(pptr);
					mSum_p = _mm256_add_ps(mSum_p, mp0);

					mSum_Ipb = _mm256_fmadd_ps(mp0, mb0, mSum_Ipb);
					mSum_Ipg = _mm256_fmadd_ps(mp0, mg0, mSum_Ipg);
					mSum_Ipr = _mm256_fmadd_ps(mp0, mr0, mSum_Ipr);

					I_bptr += width;
					I_gptr += width;
					I_rptr += width;
					pptr += width;
				}

				_mm256_store_ps(tp_I_b, mSum_Ib);
				_mm256_store_ps(tp_I_g, mSum_Ig);
				_mm256_store_ps(tp_I_r, mSum_Ir);

				_mm256_store_ps(tp_I_bb, mSum_Ibb);
				_mm256_store_ps(tp_I_bg, mSum_Ibg);
				_mm256_store_ps(tp_I_br, mSum_Ibr);
				_mm256_store_ps(tp_I_gg, mSum_Igg);
				_mm256_store_ps(tp_I_gr, mSum_Igr);
				_mm256_store_ps(tp_I_rr, mSum_Irr);

				_mm256_store_ps(tp____p, mSum_p);

				_mm256_store_ps(tp_Ip_b, mSum_Ipb);
				_mm256_store_ps(tp_Ip_g, mSum_Ipg);
				_mm256_store_ps(tp_Ip_r, mSum_Ipr);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp____p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b_p = b.ptr<float>(i);
			float* ab = a_b.ptr<float>(i);
			float* ag = a_g.ptr<float>(i);
			float* ar = a_r.ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp____p = temp.ptr<float>(9, roffset);
			tp_Ip_b = temp.ptr<float>(10, roffset);
			tp_Ip_g = temp.ptr<float>(11, roffset);
			tp_Ip_r = temp.ptr<float>(12, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I_b = _mm256_loadu_ps(tp_I_b);
				__m256 mSum_I_g = _mm256_loadu_ps(tp_I_g);
				__m256 mSum_I_r = _mm256_loadu_ps(tp_I_r);

				__m256 mSum_I_bb = _mm256_loadu_ps(tp_I_bb);
				__m256 mSum_I_bg = _mm256_loadu_ps(tp_I_bg);
				__m256 mSum_I_br = _mm256_loadu_ps(tp_I_br);
				__m256 mSum_I_gg = _mm256_loadu_ps(tp_I_gg);
				__m256 mSum_I_gr = _mm256_loadu_ps(tp_I_gr);
				__m256 mSum_I_rr = _mm256_loadu_ps(tp_I_rr);

				__m256 mSum_p = _mm256_loadu_ps(tp____p);

				__m256 mSum_Ip_b = _mm256_loadu_ps(tp_Ip_b);
				__m256 mSum_Ip_g = _mm256_loadu_ps(tp_Ip_g);
				__m256 mSum_Ip_r = _mm256_loadu_ps(tp_Ip_r);

				for (int k = 1; k < d; k++)
				{
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b + k));
					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g + k));
					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r + k));

					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb + k));
					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg + k));
					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br + k));
					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg + k));
					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr + k));
					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr + k));

					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp____p + k));

					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b + k));
					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g + k));
					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r + k));
				}

				__m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				__m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				__m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				__m256 meps = _mm256_set1_ps(eps);

				__m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				__m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				__m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				__m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				__m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				__m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				__m256 mp = _mm256_mul_ps(mSum_p, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp, _mm256_mul_ps(mSum_Ip_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp, _mm256_mul_ps(mSum_Ip_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp, _mm256_mul_ps(mSum_Ip_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp);
				_mm256_store_ps(ab, mTmp);
				ab += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);
				_mm256_store_ps(ag, mTmp);
				ag += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);
				_mm256_store_ps(ar, mTmp);
				ar += 8;

				_mm256_store_ps(b_p, mB);

				b_p += 8;

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp____p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_Ib = _mm256_load_ps(I_b.ptr<float>(i, j));
				__m256 mSum_Ig = _mm256_load_ps(I_g.ptr<float>(i, j));
				__m256 mSum_Ir = _mm256_load_ps(I_r.ptr<float>(i, j));

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p = _mm256_load_ps(p.ptr<float>(i, j));

				__m256 mSum_Ipb = _mm256_mul_ps(mSum_Ib, mSum_p);
				__m256 mSum_Ipg = _mm256_mul_ps(mSum_Ig, mSum_p);
				__m256 mSum_Ipr = _mm256_mul_ps(mSum_Ir, mSum_p);

				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = I_b.ptr<float>(vl, j);
					float* sp2 = I_b.ptr<float>(vh, j);
					__m256 mb0 = _mm256_load_ps(sp1);
					__m256 mb1 = _mm256_load_ps(sp2);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb1);

					sp1 = I_g.ptr<float>(vl, j);
					sp2 = I_g.ptr<float>(vh, j);
					__m256 mg0 = _mm256_load_ps(sp1);
					__m256 mg1 = _mm256_load_ps(sp2);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg1);

					sp1 = I_r.ptr<float>(vl, j);
					sp2 = I_r.ptr<float>(vh, j);
					__m256 mr0 = _mm256_load_ps(sp1);
					__m256 mr1 = _mm256_load_ps(sp2);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr1);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibb = _mm256_fmadd_ps(mb1, mb1, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibg = _mm256_fmadd_ps(mb1, mg1, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Ibr = _mm256_fmadd_ps(mb1, mr1, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igg = _mm256_fmadd_ps(mg1, mg1, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Igr = _mm256_fmadd_ps(mg1, mr1, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);
					mSum_Irr = _mm256_fmadd_ps(mr1, mr1, mSum_Irr);

					sp1 = p.ptr<float>(vl, j);
					sp2 = p.ptr<float>(vh, j);
					__m256 mp0 = _mm256_load_ps(sp1);
					__m256 mp1 = _mm256_load_ps(sp2);
					mSum_p = _mm256_add_ps(mSum_p, mp0);
					mSum_p = _mm256_add_ps(mSum_p, mp1);

					mSum_Ipb = _mm256_fmadd_ps(mp0, mb0, mSum_Ipb);
					mSum_Ipb = _mm256_fmadd_ps(mp1, mb1, mSum_Ipb);

					mSum_Ipg = _mm256_fmadd_ps(mp0, mg0, mSum_Ipg);
					mSum_Ipg = _mm256_fmadd_ps(mp1, mg1, mSum_Ipg);

					mSum_Ipr = _mm256_fmadd_ps(mp0, mr0, mSum_Ipr);
					mSum_Ipr = _mm256_fmadd_ps(mp1, mr1, mSum_Ipr);
				}

				_mm256_store_ps(tp_I_b, mSum_Ib);
				_mm256_store_ps(tp_I_g, mSum_Ig);
				_mm256_store_ps(tp_I_r, mSum_Ir);

				_mm256_store_ps(tp_I_bb, mSum_Ibb);
				_mm256_store_ps(tp_I_bg, mSum_Ibg);
				_mm256_store_ps(tp_I_br, mSum_Ibr);
				_mm256_store_ps(tp_I_gg, mSum_Igg);
				_mm256_store_ps(tp_I_gr, mSum_Igr);
				_mm256_store_ps(tp_I_rr, mSum_Irr);

				_mm256_store_ps(tp____p, mSum_p);

				_mm256_store_ps(tp_Ip_b, mSum_Ipb);
				_mm256_store_ps(tp_Ip_g, mSum_Ipg);
				_mm256_store_ps(tp_Ip_r, mSum_Ipr);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp____p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b_p = b.ptr<float>(i);
			float* ab = a_b.ptr<float>(i);
			float* ag = a_g.ptr<float>(i);
			float* ar = a_r.ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp____p = temp.ptr<float>(9, roffset);
			tp_Ip_b = temp.ptr<float>(10, roffset);
			tp_Ip_g = temp.ptr<float>(11, roffset);
			tp_Ip_r = temp.ptr<float>(12, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_I_b = _mm256_loadu_ps(tp_I_b + r);
				__m256 mSum_I_g = _mm256_loadu_ps(tp_I_g + r);
				__m256 mSum_I_r = _mm256_loadu_ps(tp_I_r + r);

				__m256 mSum_I_bb = _mm256_loadu_ps(tp_I_bb + r);
				__m256 mSum_I_bg = _mm256_loadu_ps(tp_I_bg + r);
				__m256 mSum_I_br = _mm256_loadu_ps(tp_I_br + r);
				__m256 mSum_I_gg = _mm256_loadu_ps(tp_I_gg + r);
				__m256 mSum_I_gr = _mm256_loadu_ps(tp_I_gr + r);
				__m256 mSum_I_rr = _mm256_loadu_ps(tp_I_rr + r);

				__m256 mSum_p = _mm256_loadu_ps(tp____p + r);

				__m256 mSum_Ip_b = _mm256_loadu_ps(tp_Ip_b + r);
				__m256 mSum_Ip_g = _mm256_loadu_ps(tp_Ip_g + r);
				__m256 mSum_Ip_r = _mm256_loadu_ps(tp_Ip_r + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b - k + r));
					mSum_I_b = _mm256_add_ps(mSum_I_b, _mm256_loadu_ps(tp_I_b + k + r));

					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g - k + r));
					mSum_I_g = _mm256_add_ps(mSum_I_g, _mm256_loadu_ps(tp_I_g + k + r));

					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r - k + r));
					mSum_I_r = _mm256_add_ps(mSum_I_r, _mm256_loadu_ps(tp_I_r + k + r));


					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb - k + r));
					mSum_I_bb = _mm256_add_ps(mSum_I_bb, _mm256_loadu_ps(tp_I_bb + k + r));

					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg - k + r));
					mSum_I_bg = _mm256_add_ps(mSum_I_bg, _mm256_loadu_ps(tp_I_bg + k + r));

					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br - k + r));
					mSum_I_br = _mm256_add_ps(mSum_I_br, _mm256_loadu_ps(tp_I_br + k + r));

					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg - k + r));
					mSum_I_gg = _mm256_add_ps(mSum_I_gg, _mm256_loadu_ps(tp_I_gg + k + r));

					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr - k + r));
					mSum_I_gr = _mm256_add_ps(mSum_I_gr, _mm256_loadu_ps(tp_I_gr + k + r));

					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr - k + r));
					mSum_I_rr = _mm256_add_ps(mSum_I_rr, _mm256_loadu_ps(tp_I_rr + k + r));

					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp____p - k + r));
					mSum_p = _mm256_add_ps(mSum_p, _mm256_loadu_ps(tp____p + k + r));

					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b - k + r));
					mSum_Ip_b = _mm256_add_ps(mSum_Ip_b, _mm256_loadu_ps(tp_Ip_b + k + r));

					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g - k + r));
					mSum_Ip_g = _mm256_add_ps(mSum_Ip_g, _mm256_loadu_ps(tp_Ip_g + k + r));

					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r - k + r));
					mSum_Ip_r = _mm256_add_ps(mSum_Ip_r, _mm256_loadu_ps(tp_Ip_r + k + r));
				}

				__m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				__m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				__m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				__m256 meps = _mm256_set1_ps(eps);

				__m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				__m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				__m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				__m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				__m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				__m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				__m256 mp = _mm256_mul_ps(mSum_p, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp, _mm256_mul_ps(mSum_Ip_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp, _mm256_mul_ps(mSum_Ip_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp, _mm256_mul_ps(mSum_Ip_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp);
				_mm256_store_ps(ab, mTmp);
				ab += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);
				_mm256_store_ps(ag, mTmp);
				ag += 8;

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);
				_mm256_store_ps(ar, mTmp);
				ar += 8;

				_mm256_store_ps(b_p, mB);

				b_p += 8;

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp____p += 8;
				tp_Ip_b += 8;
				tp_Ip_g += 8;
				tp_Ip_r += 8;
			}
		}
	}
}


void ab2q_Guide3_sep_VHI_Unroll2_AVX(Mat& a_b, Mat& a_g, Mat& a_r, Mat& b, Mat& guide_b, Mat& guide_g, Mat& guide_r, const int r, Mat& dest)
{
	const int width = a_b.cols;
	const int height = a_b.rows;

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	__m256 mDiv = _mm256_set1_ps(1.f / ((2 * r + 1)*(2 * r + 1)));

	Mat temp(Size(width + 2 * R, 4), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp_a_b = temp.ptr<float>(0, R);
		float* tp_a_g = temp.ptr<float>(1, R);
		float* tp_a_r = temp.ptr<float>(2, R);
		float* tp___b = temp.ptr<float>(3, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* sp1_a_b = a_b.ptr<float>(i - r, j);
				float* sp1_a_g = a_g.ptr<float>(i - r, j);
				float* sp1_a_r = a_r.ptr<float>(i - r, j);
				float* sp1_b = b.ptr<float>(i - r, j);

				__m256 mSum_ab = _mm256_load_ps(sp1_a_b + r * width);
				__m256 mSum_ag = _mm256_load_ps(sp1_a_g + r * width);
				__m256 mSum_ar = _mm256_load_ps(sp1_a_r + r * width);
				__m256 mSum_b = _mm256_load_ps(sp1_b + r * width);
				const int step = (r + 1) * width;
				for (int k = 1; k <= r; k++)
				{
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b));
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b + step));

					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g + step));

					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1_a_r));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1_a_r + step));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1_b));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1_b + step));

					sp1_a_b += width;
					sp1_a_g += width;
					sp1_a_r += width;
					sp1_b += width;
				}

				_mm256_store_ps(tp_a_b, mSum_ab);
				_mm256_store_ps(tp_a_g, mSum_ag);
				_mm256_store_ps(tp_a_r, mSum_ar);
				_mm256_store_ps(tp___b, mSum_b);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = guide_b.ptr<float>(i);
			float* dp_a_g = guide_g.ptr<float>(i);
			float* dp_a_r = guide_r.ptr<float>(i);
			float* dp_b = dest.ptr<float>(i);

			tp_a_b = temp.ptr<float>(0, roffset);
			tp_a_g = temp.ptr<float>(1, roffset);
			tp_a_r = temp.ptr<float>(2, roffset);
			tp___b = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
#if 0
				__m256 mSum_a_b = _mm256_setzero_ps();
				__m256 mSum_a_g = _mm256_setzero_ps();
				__m256 mSum_a_r = _mm256_setzero_ps();
				__m256 mSum_b = _mm256_setzero_ps();
				for (int k = 0; k <= 2 * r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}
#else				
				__m256 mSum_a_b = _mm256_loadu_ps(tp_a_b + r);
				__m256 mSum_a_g = _mm256_loadu_ps(tp_a_g + r);
				__m256 mSum_a_r = _mm256_loadu_ps(tp_a_r + r);
				__m256 mSum_b = _mm256_loadu_ps(tp___b + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b - k + r));
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k + r));

					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g - k + r));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k + r));

					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r - k + r));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k + r));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b - k + r));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b + k + r));
				}
#endif
				__m256 v = _mm256_mul_ps(mSum_b, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_b), _mm256_mul_ps(mSum_a_b, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_g), _mm256_mul_ps(mSum_a_g, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_r), _mm256_mul_ps(mSum_a_r, mDiv), v);
				_mm256_store_ps(dp_b, v);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_ab = _mm256_load_ps(a_b.ptr<float>(i, j));
				__m256 mSum_ag = _mm256_load_ps(a_g.ptr<float>(i, j));
				__m256 mSum_ar = _mm256_load_ps(a_r.ptr<float>(i, j));
				__m256 mSum_b = _mm256_load_ps(b.ptr<float>(i, j));
				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = a_b.ptr<float>(vl, j);
					float* sp2 = a_b.ptr<float>(vh, j);
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1));
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp2));

					sp1 = a_g.ptr<float>(vl, j);
					sp2 = a_g.ptr<float>(vh, j);
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp2));

					sp1 = a_r.ptr<float>(vl, j);
					sp2 = a_r.ptr<float>(vh, j);
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp2));

					sp1 = b.ptr<float>(vl, j);
					sp2 = b.ptr<float>(vh, j);
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp2));
				}

				_mm256_store_ps(tp_a_b, mSum_ab);
				_mm256_store_ps(tp_a_g, mSum_ag);
				_mm256_store_ps(tp_a_r, mSum_ar);
				_mm256_store_ps(tp___b, mSum_b);
				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = guide_b.ptr<float>(i);
			float* dp_a_g = guide_g.ptr<float>(i);
			float* dp_a_r = guide_r.ptr<float>(i);
			float* dp_b = dest.ptr<float>(i);

			tp_a_b = temp.ptr<float>(0, roffset);
			tp_a_g = temp.ptr<float>(1, roffset);
			tp_a_r = temp.ptr<float>(2, roffset);
			tp___b = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				/*
				__m256 mSum_a_b = _mm256_setzero_ps();
				__m256 mSum_a_g = _mm256_setzero_ps();
				__m256 mSum_a_r = _mm256_setzero_ps();
				__m256 mSum_b = _mm256_setzero_ps();
				for (int k = 0; k <= 2 * r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}
				*/

				__m256 mSum_a_b = _mm256_loadu_ps(tp_a_b + r);
				__m256 mSum_a_g = _mm256_loadu_ps(tp_a_g + r);
				__m256 mSum_a_r = _mm256_loadu_ps(tp_a_r + r);
				__m256 mSum_b = _mm256_loadu_ps(tp___b + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b - k + r));
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k + r));

					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g - k + r));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k + r));

					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r - k + r));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k + r));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b - k + r));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_b), _mm256_mul_ps(mSum_a_b, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_g), _mm256_mul_ps(mSum_a_g, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_r), _mm256_mul_ps(mSum_a_r, mDiv), v);
				_mm256_store_ps(dp_b, v);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
	}
}

void ab2q_guide3_sep_VHI_Unroll2_AVX_omp(Mat& a_b, Mat& a_g, Mat& a_r, Mat& b, Mat& guide_b, Mat& guide_g, Mat& guide_r, const int r, Mat& dest)
{
	const int width = a_b.cols;
	const int height = a_b.rows;

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat buff(Size(width + 2 * R, 4 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 4 * omp_get_thread_num(), width + 2 * R, 4));

		float* tp_a_b = temp.ptr<float>(0, R);
		float* tp_a_g = temp.ptr<float>(1, R);
		float* tp_a_r = temp.ptr<float>(2, R);
		float* tp___b = temp.ptr<float>(3, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* sp1_a_b = a_b.ptr<float>(i - r, j);
				float* sp1_a_g = a_g.ptr<float>(i - r, j);
				float* sp1_a_r = a_r.ptr<float>(i - r, j);
				float* sp1_b = b.ptr<float>(i - r, j);

				__m256 mSum_ab = _mm256_load_ps(sp1_a_b);
				__m256 mSum_ag = _mm256_load_ps(sp1_a_g);
				__m256 mSum_ar = _mm256_load_ps(sp1_a_r);
				__m256 mSum_b = _mm256_load_ps(sp1_b);

				sp1_a_b += width;
				sp1_a_g += width;
				sp1_a_r += width;
				sp1_b += width;

				const int step = 2 * width;
				for (int k = 0; k < r; k++)
				{
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b));
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b + width));

					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g + width));

					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1_a_r));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1_a_r + width));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1_b));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1_b + width));

					sp1_a_b += step;
					sp1_a_g += step;
					sp1_a_r += step;
					sp1_b += step;
				}

				_mm256_store_ps(tp_a_b, mSum_ab);
				_mm256_store_ps(tp_a_g, mSum_ag);
				_mm256_store_ps(tp_a_r, mSum_ar);
				_mm256_store_ps(tp___b, mSum_b);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = guide_b.ptr<float>(i);
			float* dp_a_g = guide_g.ptr<float>(i);
			float* dp_a_r = guide_r.ptr<float>(i);
			float* dp_b = dest.ptr<float>(i);

			tp_a_b = temp.ptr<float>(0, roffset);
			tp_a_g = temp.ptr<float>(1, roffset);
			tp_a_r = temp.ptr<float>(2, roffset);
			tp___b = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b = _mm256_loadu_ps(tp_a_b);
				__m256 mSum_a_g = _mm256_loadu_ps(tp_a_g);
				__m256 mSum_a_r = _mm256_loadu_ps(tp_a_r);
				__m256 mSum_b = _mm256_loadu_ps(tp___b);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k));
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k + r));

					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k + r));

					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k + r));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_b), _mm256_mul_ps(mSum_a_b, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_g), _mm256_mul_ps(mSum_a_g, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_r), _mm256_mul_ps(mSum_a_r, mDiv), v);
				_mm256_store_ps(dp_b, v);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_ab = _mm256_load_ps(a_b.ptr<float>(i, j));
				__m256 mSum_ag = _mm256_load_ps(a_g.ptr<float>(i, j));
				__m256 mSum_ar = _mm256_load_ps(a_r.ptr<float>(i, j));
				__m256 mSum_b = _mm256_load_ps(b.ptr<float>(i, j));
				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = a_b.ptr<float>(vl, j);
					float* sp2 = a_b.ptr<float>(vh, j);
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1));
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp2));

					sp1 = a_g.ptr<float>(vl, j);
					sp2 = a_g.ptr<float>(vh, j);
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp2));

					sp1 = a_r.ptr<float>(vl, j);
					sp2 = a_r.ptr<float>(vh, j);
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp2));

					sp1 = b.ptr<float>(vl, j);
					sp2 = b.ptr<float>(vh, j);
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp2));
				}

				_mm256_store_ps(tp_a_b, mSum_ab);
				_mm256_store_ps(tp_a_g, mSum_ag);
				_mm256_store_ps(tp_a_r, mSum_ar);
				_mm256_store_ps(tp___b, mSum_b);
				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = guide_b.ptr<float>(i);
			float* dp_a_g = guide_g.ptr<float>(i);
			float* dp_a_r = guide_r.ptr<float>(i);
			float* dp_b = dest.ptr<float>(i);

			tp_a_b = temp.ptr<float>(0, roffset);
			tp_a_g = temp.ptr<float>(1, roffset);
			tp_a_r = temp.ptr<float>(2, roffset);
			tp___b = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				/*
				__m256 mSum_a_b = _mm256_setzero_ps();
				__m256 mSum_a_g = _mm256_setzero_ps();
				__m256 mSum_a_r = _mm256_setzero_ps();
				__m256 mSum_b = _mm256_setzero_ps();
				for (int k = 0; k <= 2 * r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}
				*/

				__m256 mSum_a_b = _mm256_loadu_ps(tp_a_b + r);
				__m256 mSum_a_g = _mm256_loadu_ps(tp_a_g + r);
				__m256 mSum_a_r = _mm256_loadu_ps(tp_a_r + r);
				__m256 mSum_b = _mm256_loadu_ps(tp___b + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b - k + r));
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k + r));

					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g - k + r));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k + r));

					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r - k + r));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k + r));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b - k + r));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_b), _mm256_mul_ps(mSum_a_b, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_g), _mm256_mul_ps(mSum_a_g, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_r), _mm256_mul_ps(mSum_a_r, mDiv), v);
				_mm256_store_ps(dp_b, v);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
	}
}


void ab2q_Guide3_sep_VHI_AVX(Mat& a_b, Mat& a_g, Mat& a_r, Mat& b, Mat& guide_b, Mat& guide_g, Mat& guide_r, const int r, Mat& dest)
{
	const int width = a_b.cols;
	const int height = a_b.rows;

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat temp(Size(width + 2 * R, 4), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp_a_b = temp.ptr<float>(0, R);
		float* tp_a_g = temp.ptr<float>(1, R);
		float* tp_a_r = temp.ptr<float>(2, R);
		float* tp___b = temp.ptr<float>(3, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* sp1_a_b = a_b.ptr<float>(i - r, j);
				float* sp1_a_g = a_g.ptr<float>(i - r, j);
				float* sp1_a_r = a_r.ptr<float>(i - r, j);
				float* sp1_b = b.ptr<float>(i - r, j);

				__m256 mSum_ab = _mm256_load_ps(sp1_a_b);
				__m256 mSum_ag = _mm256_load_ps(sp1_a_g);
				__m256 mSum_ar = _mm256_load_ps(sp1_a_r);
				__m256 mSum_b = _mm256_load_ps(sp1_b);

				sp1_a_b += width;
				sp1_a_g += width;
				sp1_a_r += width;
				sp1_b += width;

				for (int k = 1; k < d; k++)
				{
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1_a_r));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1_b));
					sp1_a_b += width;
					sp1_a_g += width;
					sp1_a_r += width;
					sp1_b += width;
				}

				_mm256_store_ps(tp_a_b, mSum_ab);
				_mm256_store_ps(tp_a_g, mSum_ag);
				_mm256_store_ps(tp_a_r, mSum_ar);
				_mm256_store_ps(tp___b, mSum_b);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = guide_b.ptr<float>(i);
			float* dp_a_g = guide_g.ptr<float>(i);
			float* dp_a_r = guide_r.ptr<float>(i);
			float* dp_b = dest.ptr<float>(i);

			tp_a_b = temp.ptr<float>(0, roffset);
			tp_a_g = temp.ptr<float>(1, roffset);
			tp_a_r = temp.ptr<float>(2, roffset);
			tp___b = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b = _mm256_loadu_ps(tp_a_b);
				__m256 mSum_a_g = _mm256_loadu_ps(tp_a_g);
				__m256 mSum_a_r = _mm256_loadu_ps(tp_a_r);
				__m256 mSum_b = _mm256_loadu_ps(tp___b);
				for (int k = 1; k < d; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b + k));
				}

				__m256 v = _mm256_mul_ps(mSum_b, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_b), _mm256_mul_ps(mSum_a_b, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_g), _mm256_mul_ps(mSum_a_g, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_r), _mm256_mul_ps(mSum_a_r, mDiv), v);
				_mm256_store_ps(dp_b, v);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_ab = _mm256_load_ps(a_b.ptr<float>(i, j));
				__m256 mSum_ag = _mm256_load_ps(a_g.ptr<float>(i, j));
				__m256 mSum_ar = _mm256_load_ps(a_r.ptr<float>(i, j));
				__m256 mSum_b = _mm256_load_ps(b.ptr<float>(i, j));
				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = a_b.ptr<float>(vl, j);
					float* sp2 = a_b.ptr<float>(vh, j);
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1));
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp2));

					sp1 = a_g.ptr<float>(vl, j);
					sp2 = a_g.ptr<float>(vh, j);
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp2));

					sp1 = a_r.ptr<float>(vl, j);
					sp2 = a_r.ptr<float>(vh, j);
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp2));

					sp1 = b.ptr<float>(vl, j);
					sp2 = b.ptr<float>(vh, j);
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp2));
				}

				_mm256_store_ps(tp_a_b, mSum_ab);
				_mm256_store_ps(tp_a_g, mSum_ag);
				_mm256_store_ps(tp_a_r, mSum_ar);
				_mm256_store_ps(tp___b, mSum_b);
				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = guide_b.ptr<float>(i);
			float* dp_a_g = guide_g.ptr<float>(i);
			float* dp_a_r = guide_r.ptr<float>(i);
			float* dp_b = dest.ptr<float>(i);

			tp_a_b = temp.ptr<float>(0, roffset);
			tp_a_g = temp.ptr<float>(1, roffset);
			tp_a_r = temp.ptr<float>(2, roffset);
			tp___b = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b = _mm256_loadu_ps(tp_a_b + r);
				__m256 mSum_a_g = _mm256_loadu_ps(tp_a_g + r);
				__m256 mSum_a_r = _mm256_loadu_ps(tp_a_r + r);
				__m256 mSum_b = _mm256_loadu_ps(tp___b + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b - k + r));
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k + r));

					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g - k + r));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k + r));

					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r - k + r));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k + r));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b - k + r));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_b), _mm256_mul_ps(mSum_a_b, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_g), _mm256_mul_ps(mSum_a_g, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_r), _mm256_mul_ps(mSum_a_r, mDiv), v);
				_mm256_store_ps(dp_b, v);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
	}
}

void ab2q_Guide3_sep_VHI_AVX_omp(Mat& a_b, Mat& a_g, Mat& a_r, Mat& b, Mat& guide_b, Mat& guide_g, Mat& guide_r, const int r, Mat& dest)
{
	const int width = a_b.cols;
	const int height = a_b.rows;

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat buff(Size(width + 2 * R, 4 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 4 * omp_get_thread_num(), width + 2 * R, 4));

		float* tp_a_b = temp.ptr<float>(0, R);
		float* tp_a_g = temp.ptr<float>(1, R);
		float* tp_a_r = temp.ptr<float>(2, R);
		float* tp___b = temp.ptr<float>(3, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* sp1_a_b = a_b.ptr<float>(i - r, j);
				float* sp1_a_g = a_g.ptr<float>(i - r, j);
				float* sp1_a_r = a_r.ptr<float>(i - r, j);
				float* sp1_b = b.ptr<float>(i - r, j);

				__m256 mSum_ab = _mm256_load_ps(sp1_a_b);
				__m256 mSum_ag = _mm256_load_ps(sp1_a_g);
				__m256 mSum_ar = _mm256_load_ps(sp1_a_r);
				__m256 mSum_b = _mm256_load_ps(sp1_b);

				sp1_a_b += width;
				sp1_a_g += width;
				sp1_a_r += width;
				sp1_b += width;

				for (int k = 1; k < d; k++)
				{
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1_a_r));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1_b));
					sp1_a_b += width;
					sp1_a_g += width;
					sp1_a_r += width;
					sp1_b += width;
				}

				_mm256_store_ps(tp_a_b, mSum_ab);
				_mm256_store_ps(tp_a_g, mSum_ag);
				_mm256_store_ps(tp_a_r, mSum_ar);
				_mm256_store_ps(tp___b, mSum_b);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = guide_b.ptr<float>(i);
			float* dp_a_g = guide_g.ptr<float>(i);
			float* dp_a_r = guide_r.ptr<float>(i);
			float* dp_b = dest.ptr<float>(i);

			tp_a_b = temp.ptr<float>(0, roffset);
			tp_a_g = temp.ptr<float>(1, roffset);
			tp_a_r = temp.ptr<float>(2, roffset);
			tp___b = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b = _mm256_loadu_ps(tp_a_b);
				__m256 mSum_a_g = _mm256_loadu_ps(tp_a_g);
				__m256 mSum_a_r = _mm256_loadu_ps(tp_a_r);
				__m256 mSum_b = _mm256_loadu_ps(tp___b);
				for (int k = 1; k < d; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b + k));
				}

				__m256 v = _mm256_mul_ps(mSum_b, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_b), _mm256_mul_ps(mSum_a_b, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_g), _mm256_mul_ps(mSum_a_g, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_r), _mm256_mul_ps(mSum_a_r, mDiv), v);
				_mm256_store_ps(dp_b, v);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_ab = _mm256_load_ps(a_b.ptr<float>(i, j));
				__m256 mSum_ag = _mm256_load_ps(a_g.ptr<float>(i, j));
				__m256 mSum_ar = _mm256_load_ps(a_r.ptr<float>(i, j));
				__m256 mSum_b = _mm256_load_ps(b.ptr<float>(i, j));
				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = a_b.ptr<float>(vl, j);
					float* sp2 = a_b.ptr<float>(vh, j);
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1));
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp2));

					sp1 = a_g.ptr<float>(vl, j);
					sp2 = a_g.ptr<float>(vh, j);
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp2));

					sp1 = a_r.ptr<float>(vl, j);
					sp2 = a_r.ptr<float>(vh, j);
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp2));

					sp1 = b.ptr<float>(vl, j);
					sp2 = b.ptr<float>(vh, j);
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp2));
				}

				_mm256_store_ps(tp_a_b, mSum_ab);
				_mm256_store_ps(tp_a_g, mSum_ag);
				_mm256_store_ps(tp_a_r, mSum_ar);
				_mm256_store_ps(tp___b, mSum_b);
				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = guide_b.ptr<float>(i);
			float* dp_a_g = guide_g.ptr<float>(i);
			float* dp_a_r = guide_r.ptr<float>(i);
			float* dp_b = dest.ptr<float>(i);

			tp_a_b = temp.ptr<float>(0, roffset);
			tp_a_g = temp.ptr<float>(1, roffset);
			tp_a_r = temp.ptr<float>(2, roffset);
			tp___b = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b = _mm256_loadu_ps(tp_a_b + r);
				__m256 mSum_a_g = _mm256_loadu_ps(tp_a_g + r);
				__m256 mSum_a_r = _mm256_loadu_ps(tp_a_r + r);
				__m256 mSum_b = _mm256_loadu_ps(tp___b + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b - k + r));
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k + r));

					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g - k + r));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k + r));

					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r - k + r));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k + r));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b - k + r));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp___b + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_b), _mm256_mul_ps(mSum_a_b, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_g), _mm256_mul_ps(mSum_a_g, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_r), _mm256_mul_ps(mSum_a_r, mDiv), v);
				_mm256_store_ps(dp_b, v);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp___b += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
	}
}


//for upsampling
void blurSeparableVHI_omp(const Mat& src0, const Mat& src1, const int r, Mat& dest0, Mat& dest1)
{
	const int width = src0.cols;
	const int height = src0.rows;

	dest0.create(src0.size(), CV_32F);
	dest1.create(src0.size(), CV_32F);

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / ((2 * r + 1)*(2 * r + 1)));

	Mat buff(Size(width + 2 * R, 2 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 2 * omp_get_thread_num(), width + 2 * R, 2));

		float* tp0 = temp.ptr<float>(0, R);
		float* tp1 = temp.ptr<float>(1, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				const float* s0 = src0.ptr<float>(i - r, j);
				const float* s1 = src1.ptr<float>(i - r, j);

				__m256 ms0 = _mm256_load_ps(s0);
				__m256 ms1 = _mm256_load_ps(s1);

				s0 += width;
				s1 += width;

				for (int k = 1; k < d; k++)
				{
					ms0 = _mm256_add_ps(ms0, _mm256_load_ps(s0));
					ms1 = _mm256_add_ps(ms1, _mm256_load_ps(s1));

					s0 += width;
					s1 += width;
				}

				_mm256_store_ps(tp0, ms0);
				_mm256_store_ps(tp1, ms1);

				tp0 += 8;
				tp1 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* d0 = dest0.ptr<float>(i);
			float* d1 = dest1.ptr<float>(i);

			tp0 = temp.ptr<float>(0, roffset);
			tp1 = temp.ptr<float>(1, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 ms0 = _mm256_loadu_ps(tp0);
				__m256 ms1 = _mm256_loadu_ps(tp1);

				for (int k = 1; k < d; k++)
				{
					ms0 = _mm256_add_ps(ms0, _mm256_loadu_ps(tp0 + k));
					ms1 = _mm256_add_ps(ms1, _mm256_loadu_ps(tp1 + k));
				}

				_mm256_store_ps(d0, _mm256_mul_ps(ms0, mDiv));
				_mm256_store_ps(d1, _mm256_mul_ps(ms1, mDiv));

				tp0 += 8;
				tp1 += 8;
				d0 += 8;
				d1 += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				const int v = max(0, min(height - 1, i - r));

				const float* sp1_a_b = src0.ptr<float>(v, j);
				const float* sp1_a_g = src1.ptr<float>(v, j);

				__m256 mSum_ab = _mm256_load_ps(sp1_a_b);
				__m256 mSum_ag = _mm256_load_ps(sp1_a_g);

				for (int k = 1; k < d; k++)
				{
					const int v = max(0, min(height - 1, i - r + k));

					const float* sp1_a_b = src0.ptr<float>(v, j);
					const float* sp1_a_g = src1.ptr<float>(v, j);

					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g));
				}

				_mm256_store_ps(tp0, mSum_ab);
				_mm256_store_ps(tp1, mSum_ag);

				tp0 += 8;
				tp1 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = dest0.ptr<float>(i);
			float* dp_a_g = dest1.ptr<float>(i);

			tp0 = temp.ptr<float>(0, roffset);
			tp1 = temp.ptr<float>(1, roffset);
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b = _mm256_loadu_ps(tp0);
				__m256 mSum_a_g = _mm256_loadu_ps(tp1);

				for (int k = 1; k < d; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp0 + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp1 + k));
				}

				_mm256_store_ps(dp_a_b, _mm256_mul_ps(mSum_a_b, mDiv));
				_mm256_store_ps(dp_a_g, _mm256_mul_ps(mSum_a_g, mDiv));

				tp0 += 8;
				tp1 += 8;
				dp_a_b += 8;
				dp_a_g += 8;
			}
		}
	}
}

void blurSeparableVHI(const Mat& src0, const Mat& src1, const int r, Mat& dest0, Mat& dest1)
{
	const int width = src0.cols;
	const int height = src0.rows;

	dest0.create(src0.size(), CV_32F);
	dest1.create(src0.size(), CV_32F);

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / ((2 * r + 1)*(2 * r + 1)));

	Mat temp(Size(width + 2 * R, 2), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp0 = temp.ptr<float>(0, R);
		float* tp1 = temp.ptr<float>(1, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				const float* s0 = src0.ptr<float>(i - r, j);
				const float* s1 = src1.ptr<float>(i - r, j);

				__m256 ms0 = _mm256_load_ps(s0);
				__m256 ms1 = _mm256_load_ps(s1);

				s0 += width;
				s1 += width;

				for (int k = 1; k < d; k++)
				{
					ms0 = _mm256_add_ps(ms0, _mm256_load_ps(s0));
					ms1 = _mm256_add_ps(ms1, _mm256_load_ps(s1));

					s0 += width;
					s1 += width;
				}

				_mm256_store_ps(tp0, ms0);
				_mm256_store_ps(tp1, ms1);

				tp0 += 8;
				tp1 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* d0 = dest0.ptr<float>(i);
			float* d1 = dest1.ptr<float>(i);

			tp0 = temp.ptr<float>(0, roffset);
			tp1 = temp.ptr<float>(1, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 ms0 = _mm256_loadu_ps(tp0);
				__m256 ms1 = _mm256_loadu_ps(tp1);

				for (int k = 1; k < d; k++)
				{
					ms0 = _mm256_add_ps(ms0, _mm256_loadu_ps(tp0 + k));
					ms1 = _mm256_add_ps(ms1, _mm256_loadu_ps(tp1 + k));
				}

				_mm256_store_ps(d0, _mm256_mul_ps(ms0, mDiv));
				_mm256_store_ps(d1, _mm256_mul_ps(ms1, mDiv));

				tp0 += 8;
				tp1 += 8;
				d0 += 8;
				d1 += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				const int v = max(0, min(height - 1, i - r));

				const float* sp1_a_b = src0.ptr<float>(v, j);
				const float* sp1_a_g = src1.ptr<float>(v, j);

				__m256 mSum_ab = _mm256_load_ps(sp1_a_b);
				__m256 mSum_ag = _mm256_load_ps(sp1_a_g);

				for (int k = 1; k < d; k++)
				{
					const int v = max(0, min(height - 1, i - r + k));

					const float* sp1_a_b = src0.ptr<float>(v, j);
					const float* sp1_a_g = src1.ptr<float>(v, j);

					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g));
				}

				_mm256_store_ps(tp0, mSum_ab);
				_mm256_store_ps(tp1, mSum_ag);

				tp0 += 8;
				tp1 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = dest0.ptr<float>(i);
			float* dp_a_g = dest1.ptr<float>(i);

			tp0 = temp.ptr<float>(0, roffset);
			tp1 = temp.ptr<float>(1, roffset);
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b = _mm256_loadu_ps(tp0);
				__m256 mSum_a_g = _mm256_loadu_ps(tp1);

				for (int k = 1; k < d; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp0 + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp1 + k));
				}

				_mm256_store_ps(dp_a_b, _mm256_mul_ps(mSum_a_b, mDiv));
				_mm256_store_ps(dp_a_g, _mm256_mul_ps(mSum_a_g, mDiv));

				tp0 += 8;
				tp1 += 8;
				dp_a_b += 8;
				dp_a_g += 8;
			}
		}
	}
}

void blurSeparableVHI_omp(const Mat& src0, const Mat& src1, const Mat& src2, const Mat& src3, const int r,
	Mat& dest0, Mat& dest1, Mat& dest2, Mat& dest3)
{
	const int width = src0.cols;
	const int height = src0.rows;

	dest0.create(src0.size(), CV_32F);
	dest1.create(src0.size(), CV_32F);
	dest2.create(src0.size(), CV_32F);
	dest3.create(src0.size(), CV_32F);

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / ((2 * r + 1)*(2 * r + 1)));

	Mat buff(Size(width + 2 * R, 4 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 4 * omp_get_thread_num(), width + 2 * R, 4));

		float* tp0 = temp.ptr<float>(0, R);
		float* tp1 = temp.ptr<float>(1, R);
		float* tp2 = temp.ptr<float>(2, R);
		float* tp3 = temp.ptr<float>(3, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				const float* s0 = src0.ptr<float>(i - r, j);
				const float* s1 = src1.ptr<float>(i - r, j);
				const float* s2 = src2.ptr<float>(i - r, j);
				const float* s3 = src3.ptr<float>(i - r, j);

				__m256 ms0 = _mm256_load_ps(s0);
				__m256 ms1 = _mm256_load_ps(s1);
				__m256 ms2 = _mm256_load_ps(s2);
				__m256 ms3 = _mm256_load_ps(s3);

				s0 += width;
				s1 += width;
				s2 += width;
				s3 += width;

				for (int k = 1; k < d; k++)
				{
					ms0 = _mm256_add_ps(ms0, _mm256_load_ps(s0));
					ms1 = _mm256_add_ps(ms1, _mm256_load_ps(s1));
					ms2 = _mm256_add_ps(ms2, _mm256_load_ps(s2));
					ms3 = _mm256_add_ps(ms3, _mm256_load_ps(s3));

					s0 += width;
					s1 += width;
					s2 += width;
					s3 += width;
				}

				_mm256_store_ps(tp0, ms0);
				_mm256_store_ps(tp1, ms1);
				_mm256_store_ps(tp2, ms2);
				_mm256_store_ps(tp3, ms3);

				tp0 += 8;
				tp1 += 8;
				tp2 += 8;
				tp3 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* d0 = dest0.ptr<float>(i);
			float* d1 = dest1.ptr<float>(i);
			float* d2 = dest2.ptr<float>(i);
			float* d3 = dest3.ptr<float>(i);

			tp0 = temp.ptr<float>(0, roffset);
			tp1 = temp.ptr<float>(1, roffset);
			tp2 = temp.ptr<float>(2, roffset);
			tp3 = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 ms0 = _mm256_loadu_ps(tp0);
				__m256 ms1 = _mm256_loadu_ps(tp1);
				__m256 ms2 = _mm256_loadu_ps(tp2);
				__m256 ms3 = _mm256_loadu_ps(tp3);

				for (int k = 1; k < d; k++)
				{
					ms0 = _mm256_add_ps(ms0, _mm256_loadu_ps(tp0 + k));
					ms1 = _mm256_add_ps(ms1, _mm256_loadu_ps(tp1 + k));
					ms2 = _mm256_add_ps(ms2, _mm256_loadu_ps(tp2 + k));
					ms3 = _mm256_add_ps(ms3, _mm256_loadu_ps(tp3 + k));
				}

				_mm256_store_ps(d0, _mm256_mul_ps(ms0, mDiv));
				_mm256_store_ps(d1, _mm256_mul_ps(ms1, mDiv));
				_mm256_store_ps(d2, _mm256_mul_ps(ms2, mDiv));
				_mm256_store_ps(d3, _mm256_mul_ps(ms3, mDiv));

				tp0 += 8;
				tp1 += 8;
				tp2 += 8;
				tp3 += 8;
				d0 += 8;
				d1 += 8;
				d2 += 8;
				d3 += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				const int v = max(0, min(height - 1, i - r));

				const float* sp1_a_b = src0.ptr<float>(v, j);
				const float* sp1_a_g = src1.ptr<float>(v, j);
				const float* sp1_a_r = src2.ptr<float>(v, j);
				const float* sp1___b = src3.ptr<float>(v, j);

				__m256 mSum_ab = _mm256_load_ps(sp1_a_b);
				__m256 mSum_ag = _mm256_load_ps(sp1_a_g);
				__m256 mSum_ar = _mm256_load_ps(sp1_a_r);
				__m256 mSum__b = _mm256_load_ps(sp1___b);

				for (int k = 1; k < d; k++)
				{
					const int v = max(0, min(height - 1, i - r + k));

					const float* sp1_a_b = src0.ptr<float>(v, j);
					const float* sp1_a_g = src1.ptr<float>(v, j);
					const float* sp1_a_r = src2.ptr<float>(v, j);
					const float* sp1___b = src3.ptr<float>(v, j);

					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1_a_r));
					mSum__b = _mm256_add_ps(mSum__b, _mm256_load_ps(sp1___b));
				}

				_mm256_store_ps(tp0, mSum_ab);
				_mm256_store_ps(tp1, mSum_ag);
				_mm256_store_ps(tp2, mSum_ar);
				_mm256_store_ps(tp3, mSum__b);

				tp0 += 8;
				tp1 += 8;
				tp2 += 8;
				tp3 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = dest0.ptr<float>(i);
			float* dp_a_g = dest1.ptr<float>(i);
			float* dp_a_r = dest2.ptr<float>(i);
			float* dp_b = dest3.ptr<float>(i);

			tp0 = temp.ptr<float>(0, roffset);
			tp1 = temp.ptr<float>(1, roffset);
			tp2 = temp.ptr<float>(2, roffset);
			tp3 = temp.ptr<float>(3, roffset);
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b = _mm256_loadu_ps(tp0);
				__m256 mSum_a_g = _mm256_loadu_ps(tp1);
				__m256 mSum_a_r = _mm256_loadu_ps(tp2);
				__m256 mSum___b = _mm256_loadu_ps(tp3);

				for (int k = 1; k < d; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp0 + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp1 + k));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp2 + k));
					mSum___b = _mm256_add_ps(mSum___b, _mm256_loadu_ps(tp3 + k));
				}

				_mm256_store_ps(dp_a_b, _mm256_mul_ps(mSum_a_b, mDiv));
				_mm256_store_ps(dp_a_g, _mm256_mul_ps(mSum_a_g, mDiv));
				_mm256_store_ps(dp_a_r, _mm256_mul_ps(mSum_a_r, mDiv));
				_mm256_store_ps(dp_b, _mm256_mul_ps(mSum___b, mDiv));

				tp0 += 8;
				tp1 += 8;
				tp2 += 8;
				tp3 += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
	}
}

void blurSeparableVHI(const Mat& src0, const Mat& src1, const Mat& src2, const Mat& src3, const int r,
	Mat& dest0, Mat& dest1, Mat& dest2, Mat& dest3)
{
	const int width = src0.cols;
	const int height = src0.rows;

	dest0.create(src0.size(), CV_32F);
	dest1.create(src0.size(), CV_32F);
	dest2.create(src0.size(), CV_32F);
	dest3.create(src0.size(), CV_32F);

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / ((2 * r + 1)*(2 * r + 1)));

	Mat temp(Size(width + 2 * R, 4), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp0 = temp.ptr<float>(0, R);
		float* tp1 = temp.ptr<float>(1, R);
		float* tp2 = temp.ptr<float>(2, R);
		float* tp3 = temp.ptr<float>(3, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				const float* s0 = src0.ptr<float>(i - r, j);
				const float* s1 = src1.ptr<float>(i - r, j);
				const float* s2 = src2.ptr<float>(i - r, j);
				const float* s3 = src3.ptr<float>(i - r, j);

				__m256 ms0 = _mm256_load_ps(s0);
				__m256 ms1 = _mm256_load_ps(s1);
				__m256 ms2 = _mm256_load_ps(s2);
				__m256 ms3 = _mm256_load_ps(s3);

				s0 += width;
				s1 += width;
				s2 += width;
				s3 += width;

				for (int k = 1; k < d; k++)
				{
					ms0 = _mm256_add_ps(ms0, _mm256_load_ps(s0));
					ms1 = _mm256_add_ps(ms1, _mm256_load_ps(s1));
					ms2 = _mm256_add_ps(ms2, _mm256_load_ps(s2));
					ms3 = _mm256_add_ps(ms3, _mm256_load_ps(s3));

					s0 += width;
					s1 += width;
					s2 += width;
					s3 += width;
				}

				_mm256_store_ps(tp0, ms0);
				_mm256_store_ps(tp1, ms1);
				_mm256_store_ps(tp2, ms2);
				_mm256_store_ps(tp3, ms3);

				tp0 += 8;
				tp1 += 8;
				tp2 += 8;
				tp3 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* d0 = dest0.ptr<float>(i);
			float* d1 = dest1.ptr<float>(i);
			float* d2 = dest2.ptr<float>(i);
			float* d3 = dest3.ptr<float>(i);

			tp0 = temp.ptr<float>(0, roffset);
			tp1 = temp.ptr<float>(1, roffset);
			tp2 = temp.ptr<float>(2, roffset);
			tp3 = temp.ptr<float>(3, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 ms0 = _mm256_loadu_ps(tp0);
				__m256 ms1 = _mm256_loadu_ps(tp1);
				__m256 ms2 = _mm256_loadu_ps(tp2);
				__m256 ms3 = _mm256_loadu_ps(tp3);

				for (int k = 1; k < d; k++)
				{
					ms0 = _mm256_add_ps(ms0, _mm256_loadu_ps(tp0 + k));
					ms1 = _mm256_add_ps(ms1, _mm256_loadu_ps(tp1 + k));
					ms2 = _mm256_add_ps(ms2, _mm256_loadu_ps(tp2 + k));
					ms3 = _mm256_add_ps(ms3, _mm256_loadu_ps(tp3 + k));
				}

				_mm256_store_ps(d0, _mm256_mul_ps(ms0, mDiv));
				_mm256_store_ps(d1, _mm256_mul_ps(ms1, mDiv));
				_mm256_store_ps(d2, _mm256_mul_ps(ms2, mDiv));
				_mm256_store_ps(d3, _mm256_mul_ps(ms3, mDiv));

				tp0 += 8;
				tp1 += 8;
				tp2 += 8;
				tp3 += 8;
				d0 += 8;
				d1 += 8;
				d2 += 8;
				d3 += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				const int v = max(0, min(height - 1, i - r));

				const float* sp1_a_b = src0.ptr<float>(v, j);
				const float* sp1_a_g = src1.ptr<float>(v, j);
				const float* sp1_a_r = src2.ptr<float>(v, j);
				const float* sp1___b = src3.ptr<float>(v, j);

				__m256 mSum_ab = _mm256_load_ps(sp1_a_b);
				__m256 mSum_ag = _mm256_load_ps(sp1_a_g);
				__m256 mSum_ar = _mm256_load_ps(sp1_a_r);
				__m256 mSum__b = _mm256_load_ps(sp1___b);

				for (int k = 1; k < d; k++)
				{
					const int v = max(0, min(height - 1, i - r + k));

					const float* sp1_a_b = src0.ptr<float>(v, j);
					const float* sp1_a_g = src1.ptr<float>(v, j);
					const float* sp1_a_r = src2.ptr<float>(v, j);
					const float* sp1___b = src3.ptr<float>(v, j);

					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1_a_r));
					mSum__b = _mm256_add_ps(mSum__b, _mm256_load_ps(sp1___b));
				}

				_mm256_store_ps(tp0, mSum_ab);
				_mm256_store_ps(tp1, mSum_ag);
				_mm256_store_ps(tp2, mSum_ar);
				_mm256_store_ps(tp3, mSum__b);

				tp0 += 8;
				tp1 += 8;
				tp2 += 8;
				tp3 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* dp_a_b = dest0.ptr<float>(i);
			float* dp_a_g = dest1.ptr<float>(i);
			float* dp_a_r = dest2.ptr<float>(i);
			float* dp_b = dest3.ptr<float>(i);

			tp0 = temp.ptr<float>(0, roffset);
			tp1 = temp.ptr<float>(1, roffset);
			tp2 = temp.ptr<float>(2, roffset);
			tp3 = temp.ptr<float>(3, roffset);
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b = _mm256_loadu_ps(tp0);
				__m256 mSum_a_g = _mm256_loadu_ps(tp1);
				__m256 mSum_a_r = _mm256_loadu_ps(tp2);
				__m256 mSum___b = _mm256_loadu_ps(tp3);

				for (int k = 1; k < d; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp0 + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp1 + k));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp2 + k));
					mSum___b = _mm256_add_ps(mSum___b, _mm256_loadu_ps(tp3 + k));
				}

				_mm256_store_ps(dp_a_b, _mm256_mul_ps(mSum_a_b, mDiv));
				_mm256_store_ps(dp_a_g, _mm256_mul_ps(mSum_a_g, mDiv));
				_mm256_store_ps(dp_a_r, _mm256_mul_ps(mSum_a_r, mDiv));
				_mm256_store_ps(dp_b, _mm256_mul_ps(mSum___b, mDiv));

				tp0 += 8;
				tp1 += 8;
				tp2 += 8;
				tp3 += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
	}
}

void ab_upNN_2q_sep_VHI_omp(Mat& a_b, Mat& a_g, Mat& a_r, Mat& b, Mat& guide_b, Mat& guide_g, Mat& guide_r, const int r, Mat& dest)
{
	const int scale = guide_b.cols / a_b.cols;
	const int width = a_b.cols;
	const int height = a_b.rows;

	__m256 mDiv = _mm256_set1_ps(1.f / ((2 * r + 1)*(2 * r + 1)));

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp(Size(width + 2 * r, 4), CV_32FC1);

		float* tp_a_b = temp.ptr<float>(0, r);
		float* tp_a_g = temp.ptr<float>(1, r);
		float* tp_a_r = temp.ptr<float>(2, r);
		float* tp_b = temp.ptr<float>(3, r);
		/*if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* sp1_a_b = a_b.ptr<float>(i - r, j);
				float* sp1_a_g = a_g.ptr<float>(i - r, j);
				float* sp1_a_r = a_r.ptr<float>(i - r, j);
				float* sp1_b = b.ptr<float>(i - r, j);

				__m256 mSum_ab = _mm256_load_ps(sp1_a_b + r * width);
				__m256 mSum_ag = _mm256_load_ps(sp1_a_g + r * width);
				__m256 mSum_ar = _mm256_load_ps(sp1_a_r + r * width);
				__m256 mSum_b = _mm256_load_ps(sp1_b + r * width);
				const int step = (r + 1) * width;
				for (int k = 1; k <= r; k++)
				{
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b));
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1_a_b + step));

					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1_a_g + step));

					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1_a_r));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1_a_r + step));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1_b));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1_b + step));

					sp1_a_b += width;
					sp1_a_g += width;
					sp1_a_r += width;
					sp1_b += width;
				}

				_mm256_storeu_ps(tp_a_b, mSum_ab);
				_mm256_storeu_ps(tp_a_g, mSum_ag);
				_mm256_storeu_ps(tp_a_r, mSum_ar);
				_mm256_storeu_ps(tp_b, mSum_b);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp_b += 8;
			}

			for (int c = 0; c < 4; c++)
			{
				float vs = temp.at<float>(c, r);
				float ve = temp.at<float>(c, r + width - 1);
				for (int k = 1; k <= r; k++)
				{
					temp.at<float>(c, k - 1) = vs;
					temp.at<float>(c, 2 * r + width - k) = ve;
				}
			}

			float* dp_a_b = guide_b.ptr<float>(i);
			float* dp_a_g = guide_g.ptr<float>(i);
			float* dp_a_r = guide_r.ptr<float>(i);
			float* dp_b = dest.ptr<float>(i);

			tp_a_b = temp.ptr<float>(0);
			tp_a_g = temp.ptr<float>(1);
			tp_a_r = temp.ptr<float>(2);
			tp_b = temp.ptr<float>(3);
			for (int j = 0; j < width; j += 8)
			{
#if 0
				__m256 mSum_a_b = _mm256_setzero_ps();
				__m256 mSum_a_g = _mm256_setzero_ps();
				__m256 mSum_a_r = _mm256_setzero_ps();
				__m256 mSum_b = _mm256_setzero_ps();
				for (int k = 0; k <= 2 * r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}
#else
				__m256 mSum_a_b = _mm256_loadu_ps(tp_a_b + r);
				__m256 mSum_a_g = _mm256_loadu_ps(tp_a_g + r);
				__m256 mSum_a_r = _mm256_loadu_ps(tp_a_r + r);
				__m256 mSum_b = _mm256_loadu_ps(tp_b + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b - k + r));
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k + r));

					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g - k + r));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k + r));

					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r - k + r));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k + r));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b - k + r));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k + r));
				}
#endif
				__m256 v = _mm256_mul_ps(mSum_b, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_b), _mm256_mul_ps(mSum_a_b, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_g), _mm256_mul_ps(mSum_a_g, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(dp_a_r), _mm256_mul_ps(mSum_a_r, mDiv), v);
				_mm256_store_ps(dp_b, v);

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp_b += 8;
				dp_a_b += 8;
				dp_a_g += 8;
				dp_a_r += 8;
				dp_b += 8;
			}
		}
		else*/
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_ab = _mm256_load_ps(a_b.ptr<float>(i, j));
				__m256 mSum_ag = _mm256_load_ps(a_g.ptr<float>(i, j));
				__m256 mSum_ar = _mm256_load_ps(a_r.ptr<float>(i, j));
				__m256 mSum_b = _mm256_load_ps(b.ptr<float>(i, j));
				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = a_b.ptr<float>(vl, j);
					float* sp2 = a_b.ptr<float>(vh, j);
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp1));
					mSum_ab = _mm256_add_ps(mSum_ab, _mm256_load_ps(sp2));

					sp1 = a_g.ptr<float>(vl, j);
					sp2 = a_g.ptr<float>(vh, j);
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp1));
					mSum_ag = _mm256_add_ps(mSum_ag, _mm256_load_ps(sp2));

					sp1 = a_r.ptr<float>(vl, j);
					sp2 = a_r.ptr<float>(vh, j);
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp1));
					mSum_ar = _mm256_add_ps(mSum_ar, _mm256_load_ps(sp2));

					sp1 = b.ptr<float>(vl, j);
					sp2 = b.ptr<float>(vh, j);
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp1));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_load_ps(sp2));
				}

				_mm256_storeu_ps(tp_a_b, mSum_ab);
				_mm256_storeu_ps(tp_a_g, mSum_ag);
				_mm256_storeu_ps(tp_a_r, mSum_ar);
				_mm256_storeu_ps(tp_b, mSum_b);
				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp_b += 8;
			}

			for (int c = 0; c < 4; c++)
			{
				float vs = temp.at<float>(c, r);
				float ve = temp.at<float>(c, r + width - 1);
				for (int k = 1; k <= r; k++)
				{
					temp.at<float>(c, k - 1) = vs;
					temp.at<float>(c, 2 * r + width - k) = ve;
				}
			}

			float* gb = guide_b.ptr<float>(i*scale);
			float* gg = guide_g.ptr<float>(i*scale);
			float* gr = guide_r.ptr<float>(i*scale);
			float* d = dest.ptr<float>(i*scale);

			tp_a_b = temp.ptr<float>(0);
			tp_a_g = temp.ptr<float>(1);
			tp_a_r = temp.ptr<float>(2);
			tp_b = temp.ptr<float>(3);
			for (int j = 0; j < width; j += 8)
			{
				/*
				__m256 mSum_a_b = _mm256_setzero_ps();
				__m256 mSum_a_g = _mm256_setzero_ps();
				__m256 mSum_a_r = _mm256_setzero_ps();
				__m256 mSum_b = _mm256_setzero_ps();
				for (int k = 0; k <= 2 * r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k));
				}
				*/

				__m256 mSum_a_b = _mm256_loadu_ps(tp_a_b + r);
				__m256 mSum_a_g = _mm256_loadu_ps(tp_a_g + r);
				__m256 mSum_a_r = _mm256_loadu_ps(tp_a_r + r);
				__m256 mSum_b = _mm256_loadu_ps(tp_b + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b - k + r));
					mSum_a_b = _mm256_add_ps(mSum_a_b, _mm256_loadu_ps(tp_a_b + k + r));

					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g - k + r));
					mSum_a_g = _mm256_add_ps(mSum_a_g, _mm256_loadu_ps(tp_a_g + k + r));

					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r - k + r));
					mSum_a_r = _mm256_add_ps(mSum_a_r, _mm256_loadu_ps(tp_a_r + k + r));

					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b - k + r));
					mSum_b = _mm256_add_ps(mSum_b, _mm256_loadu_ps(tp_b + k + r));
				}

				mSum_a_b = _mm256_mul_ps(mSum_a_b, mDiv);
				mSum_a_g = _mm256_mul_ps(mSum_a_g, mDiv);
				mSum_a_r = _mm256_mul_ps(mSum_a_r, mDiv);
				mSum_b = _mm256_mul_ps(mSum_b, mDiv);

				if (scale == 2)
				{
					for (int x = 0; x < 8; x += 4)
					{
						const int m = 2 * x;
						__m256 mmb = _mm256_setr_ps(mSum_b.m256_f32[x], mSum_b.m256_f32[x], mSum_b.m256_f32[x + 1], mSum_b.m256_f32[x + 1], mSum_b.m256_f32[x + 2], mSum_b.m256_f32[x + 2], mSum_b.m256_f32[x + 3], mSum_b.m256_f32[x + 3]);
						__m256 ms = _mm256_setr_ps(mSum_a_b.m256_f32[x], mSum_a_b.m256_f32[x], mSum_a_b.m256_f32[x + 1], mSum_a_b.m256_f32[x + 1], mSum_a_b.m256_f32[x + 2], mSum_a_b.m256_f32[x + 2], mSum_a_b.m256_f32[x + 3], mSum_a_b.m256_f32[x + 3]);
						__m256 v0 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m), mmb);
						__m256 v1 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + dest.cols), mmb);

						ms = _mm256_setr_ps(mSum_a_g.m256_f32[x], mSum_a_g.m256_f32[x], mSum_a_g.m256_f32[x + 1], mSum_a_g.m256_f32[x + 1], mSum_a_g.m256_f32[x + 2], mSum_a_g.m256_f32[x + 2], mSum_a_g.m256_f32[x + 3], mSum_a_g.m256_f32[x + 3]);
						v0 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gg + m), v0);
						v1 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gg + m + dest.cols), v1);

						ms = _mm256_setr_ps(mSum_a_r.m256_f32[x], mSum_a_r.m256_f32[x], mSum_a_r.m256_f32[x + 1], mSum_a_r.m256_f32[x + 1], mSum_a_r.m256_f32[x + 2], mSum_a_r.m256_f32[x + 2], mSum_a_r.m256_f32[x + 3], mSum_a_r.m256_f32[x + 3]);
						v0 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gr + m), v0);
						v1 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gr + m + dest.cols), v1);
						_mm256_store_ps(d + m, v0);
						_mm256_store_ps(d + m + dest.cols, v1);
					}
				}
				else if (scale == 4)
				{
					for (int i = 0; i < 8; i += 2)
					{
						const int m = 4 * i;
						__m256 mmb = _mm256_setr_ps(mSum_b.m256_f32[i], mSum_b.m256_f32[i], mSum_b.m256_f32[i], mSum_b.m256_f32[i], mSum_b.m256_f32[i + 1], mSum_b.m256_f32[i + 1], mSum_b.m256_f32[i + 1], mSum_b.m256_f32[i + 1]);
						__m256 ms = _mm256_setr_ps(mSum_a_b.m256_f32[i], mSum_a_b.m256_f32[i], mSum_a_b.m256_f32[i], mSum_a_b.m256_f32[i], mSum_a_b.m256_f32[i + 1], mSum_a_b.m256_f32[i + 1], mSum_a_b.m256_f32[i + 1], mSum_a_b.m256_f32[i + 1]);
						__m256 v0 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m), mmb);
						__m256 v1 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + dest.cols), mmb);
						__m256 v2 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + 2 * dest.cols), mmb);
						__m256 v3 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + 3 * dest.cols), mmb);

						ms = _mm256_setr_ps(mSum_a_g.m256_f32[i], mSum_a_g.m256_f32[i], mSum_a_g.m256_f32[i], mSum_a_g.m256_f32[i], mSum_a_g.m256_f32[i + 1], mSum_a_g.m256_f32[i + 1], mSum_a_g.m256_f32[i + 1], mSum_a_g.m256_f32[i + 1]);
						v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m), v0);
						v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + dest.cols), v1);
						v2 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 2 * dest.cols), v2);
						v3 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 3 * dest.cols), v3);

						ms = _mm256_setr_ps(mSum_a_r.m256_f32[i], mSum_a_r.m256_f32[i], mSum_a_r.m256_f32[i], mSum_a_r.m256_f32[i], mSum_a_r.m256_f32[i + 1], mSum_a_r.m256_f32[i + 1], mSum_a_r.m256_f32[i + 1], mSum_a_r.m256_f32[i + 1]);
						v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m), v0);
						v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + dest.cols), v1);
						v2 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 2 * dest.cols), v2);
						v3 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 3 * dest.cols), v3);
						_mm256_store_ps(d + m, v0);
						_mm256_store_ps(d + m + dest.cols, v1);
						_mm256_store_ps(d + m + 2 * dest.cols, v2);
						_mm256_store_ps(d + m + 3 * dest.cols, v3);
					}
				}
				else if (scale == 8)
				{
					for (int i = 0; i < 8; i++)
					{
						const int m = 8 * i;
						__m256 mmb = _mm256_set1_ps(mSum_b.m256_f32[i]);
						__m256 ms = _mm256_set1_ps(mSum_a_b.m256_f32[i]);
						__m256 v0 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m), mmb);
						__m256 v1 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + dest.cols), mmb);
						__m256 v2 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + 2 * dest.cols), mmb);
						__m256 v3 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + 3 * dest.cols), mmb);
						__m256 v4 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + 4 * dest.cols), mmb);
						__m256 v5 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + 5 * dest.cols), mmb);
						__m256 v6 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + 6 * dest.cols), mmb);
						__m256 v7 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(gb + m + 7 * dest.cols), mmb);

						ms = _mm256_set1_ps(mSum_a_g.m256_f32[i]);
						v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m), v0);
						v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + dest.cols), v1);
						v2 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 2 * dest.cols), v2);
						v3 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 3 * dest.cols), v3);
						v4 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 4 * dest.cols), v4);
						v5 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 5 * dest.cols), v5);
						v6 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 6 * dest.cols), v6);
						v7 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 7 * dest.cols), v7);

						ms = _mm256_set1_ps(mSum_a_r.m256_f32[i]);
						v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m), v0);
						v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + dest.cols), v1);
						v2 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 2 * dest.cols), v2);
						v3 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 3 * dest.cols), v3);
						v4 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 4 * dest.cols), v4);
						v5 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 5 * dest.cols), v5);
						v6 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 6 * dest.cols), v6);
						v7 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 7 * dest.cols), v7);

						_mm256_store_ps(d + m, v0);
						_mm256_store_ps(d + m + dest.cols, v1);
						_mm256_store_ps(d + m + 2 * dest.cols, v2);
						_mm256_store_ps(d + m + 3 * dest.cols, v3);
						_mm256_store_ps(d + m + 4 * dest.cols, v4);
						_mm256_store_ps(d + m + 5 * dest.cols, v5);
						_mm256_store_ps(d + m + 6 * dest.cols, v6);
						_mm256_store_ps(d + m + 7 * dest.cols, v7);
					}
				}

				tp_a_b += 8;
				tp_a_g += 8;
				tp_a_r += 8;
				tp_b += 8;
				gb += 8 * scale;
				gg += 8 * scale;
				gr += 8 * scale;
				d += 8 * scale;
			}
		}
	}
}



void guidedFilter_SepVHI::filter()
{
	if (src.channels() == 1)
	{
		if (guide.channels() == 1)
		{
			filter_Guide1(src, guide, dest);
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);

			filter_Guide3(src, vguide, dest);
		}
	}
	else if (src.channels() == 3)
	{
		split(src, vsrc);

		const int depth = src.depth();
		vdest[0].create(src.size(), depth);
		vdest[1].create(src.size(), depth);
		vdest[2].create(src.size(), depth);

		if (guide.channels() == 1)
		{
			filter_Guide1(vsrc[0], guide, vdest[0]);
			filter_Guide1(vsrc[1], guide, vdest[1]);
			filter_Guide1(vsrc[2], guide, vdest[2]);
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);

			filter_Guide3(vsrc[0], vguide, vdest[0]);
			filter_Guide3(vsrc[1], vguide, vdest[1]);
			filter_Guide3(vsrc[2], vguide, vdest[2]);
		}

		merge(vdest, dest);
	}
}

void guidedFilter_SepVHI::filterVector()
{
	if (src.channels() == 1)
	{
		if (guide.channels() == 1)
		{
			filter_Guide1(vsrc[0], vguide[0], vdest[0]);
		}
		else if (guide.channels() == 3)
		{
			filter_Guide3(vsrc[0], vguide, vdest[0]);
		}
	}
	else if (src.channels() == 3)
	{
		const int depth = src.depth();

		if (guide.channels() == 1)
		{
			filter_Guide1(vsrc[0], vguide[0], vdest[0]);
			filter_Guide1(vsrc[1], vguide[0], vdest[1]);
			filter_Guide1(vsrc[2], vguide[0], vdest[2]);
		}
		else if (guide.channels() == 3)
		{
			filter_Guide3(vsrc[0], vguide, vdest[0]);
			filter_Guide3(vsrc[1], vguide, vdest[1]);
			filter_Guide3(vsrc[2], vguide, vdest[2]);
		}
	}
}

void guidedFilter_SepVHI::upsample()
{
	if (guide.size() != src.size())
	{
		src_low = src;
	}

	if (src.channels() == 1)
	{
		if (guide.channels() == 1)
		{
			upsample_Guide1(src_low, guide, guide_low, dest);
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);
			split(guide_low, vguide_low);

			upsample_Guide3(src_low, vguide, vguide_low, dest);
		}
	}
	else if (src.channels() == 3)
	{
		split(src_low, vsrc_low);

		const int depth = src.depth();
		vdest[0].create(guide.size(), depth);
		vdest[1].create(guide.size(), depth);
		vdest[2].create(guide.size(), depth);

		if (guide.channels() == 1)
		{
			upsample_Guide1(vsrc_low[0], guide, guide_low, vdest[0]);
			upsample_Guide1(vsrc_low[1], guide, guide_low, vdest[1]);
			upsample_Guide1(vsrc_low[2], guide, guide_low, vdest[2]);
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);
			split(guide_low, vguide_low);

			upsample_Guide3(vsrc_low[0], vguide, vguide_low, vdest[0]);
			upsample_Guide3(vsrc_low[1], vguide, vguide_low, vdest[1]);
			upsample_Guide3(vsrc_low[2], vguide, vguide_low, vdest[2]);
		}

		merge(vdest, dest);
	}
}

void guidedFilter_SepVHI::filter_Guide1(cv::Mat& input, cv::Mat& guide, cv::Mat& output)
{
	//ip2ab
	if (parallelType == NAIVE)
	{
		Ip2ab_Guide1_sep_VHI_AVX(guide, input, r, eps, a, b);
	}
	else
	{
		Ip2ab_Guide1_sep_VHI_CenterAvoid_AVX_omp(guide, input, r, eps, a, b);
		//Ip2ab_Guide1_sep_VHI_AVX_omp(guide, input, r, eps, a, b);
	}

	//ab2q
	if (parallelType == NAIVE)
	{
		ab2q_Guide1_sep_VHI_AVX(a, b, guide, r, output);
	}
	else
	{
		//ab2q_Guide1_sep_VHI_AVX_omp(a, b, guide, r, output);
		ab2q_Guide1_sep_VHI_CenterAboid_AVX_omp(a, b, guide, r, output);
	}
}

void guidedFilter_SepVHI::upsample_Guide1(cv::Mat& input_low, cv::Mat& guide, cv::Mat& guide_low, cv::Mat& output)
{
	//ip2ab
	if (parallelType == NAIVE)
	{
		Ip2ab_Guide1_sep_VHI_AVX(guide_low, input_low, r, eps, a, b);
	}
	else
	{
		Ip2ab_Guide1_sep_VHI_AVX_omp(guide_low, input_low, r, eps, a, b);
	}

	//ab2q
	blurSeparableVHI_omp(a, b, r, mean_a_b, mean_b);

	Mat temp_high = a_high_b;
	resize(mean_a_b, output, guide.size(), 0, 0, upsample_method);
	resize(mean_b, temp_high, guide.size(), 0, 0, upsample_method);

	fmadd(output, guide, temp_high, output);
}

void guidedFilter_SepVHI::filter_Guide3(cv::Mat& input, std::vector<cv::Mat>& guide, cv::Mat& output)
{
	//Ip2ab
	if (parallelType == NAIVE)
	{
		Ip2ab_Guide3_sep_VHI_AVX(guide[0], guide[1], guide[2], input, r, eps, a_b, a_g, a_r, b);
	}
	else
	{
		Ip2ab_Guide3_sep_VHI_AVX_omp(guide[0], guide[1], guide[2], input, r, eps, a_b, a_g, a_r, b);
	}

	//ab2q
	if (parallelType == NAIVE)
	{
		ab2q_Guide3_sep_VHI_AVX(a_b, a_g, a_r, b, guide[0], guide[1], guide[2], r, output);
	}
	else
	{
		ab2q_Guide3_sep_VHI_AVX_omp(a_b, a_g, a_r, b, guide[0], guide[1], guide[2], r, output);
	}
}

void ab2q_fmadd(Mat& a_b, Mat& a_g, Mat& a_r, Mat& g_b, Mat& g_g, Mat& g_r, Mat& b, Mat& dest)
{
	const int size = a_b.size().area();
	const int simdsize = size / 8;
	float* ab = a_b.ptr<float>();
	float* ag = a_g.ptr<float>();
	float* ar = a_r.ptr<float>();
	float* gb = g_b.ptr<float>();
	float* gg = g_g.ptr<float>();
	float* gr = g_r.ptr<float>();
	float* bp = b.ptr<float>();
	float* d = dest.ptr<float>();

	for (int i = 0; i < simdsize; i++)
	{
		__m256 v = _mm256_fmadd_ps(_mm256_load_ps(ab), _mm256_load_ps(gb), _mm256_load_ps(bp));
		v = _mm256_fmadd_ps(_mm256_load_ps(ag), _mm256_load_ps(gg), v);
		v = _mm256_fmadd_ps(_mm256_load_ps(ar), _mm256_load_ps(gr), v);
		_mm256_store_ps(d, v);
		ab += 8;
		ag += 8;
		ar += 8;
		gb += 8;
		gg += 8;
		gr += 8;
		bp += 8;
		d += 8;
	}
}

void ab2q_fmadd_omp(Mat& a_b, Mat& a_g, Mat& a_r, Mat& g_b, Mat& g_g, Mat& g_r, Mat& b, Mat& dest)
{
	const int size = a_b.cols;
	const int simdwidth = size / 8;

#pragma omp parallel for
	for (int j = 0; j < a_b.rows; j++)
	{
		float* ab = a_b.ptr<float>(j);
		float* ag = a_g.ptr<float>(j);
		float* ar = a_r.ptr<float>(j);
		float* gb = g_b.ptr<float>(j);
		float* gg = g_g.ptr<float>(j);
		float* gr = g_r.ptr<float>(j);
		float* bp = b.ptr<float>(j);
		float* d = dest.ptr<float>(j);

		for (int i = 0; i < simdwidth; i++)
		{
			__m256 v = _mm256_fmadd_ps(_mm256_load_ps(ab), _mm256_load_ps(gb), _mm256_load_ps(bp));
			v = _mm256_fmadd_ps(_mm256_load_ps(ag), _mm256_load_ps(gg), v);
			v = _mm256_fmadd_ps(_mm256_load_ps(ar), _mm256_load_ps(gr), v);
			_mm256_store_ps(d, v);
			ab += 8;
			ag += 8;
			ar += 8;
			gb += 8;
			gg += 8;
			gr += 8;
			bp += 8;
			d += 8;
		}
	}
}

void upsampleNN_omp(Mat& a_b, Mat& a_g, Mat& a_r, Mat& g_b, Mat& g_g, Mat& g_r, Mat& meanb, Mat& dest)
{
	const int scale = g_b.cols / a_b.cols;
	const int width = a_b.cols;
	const int height = a_b.rows;
	//__m256 a = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);		
	//printf("%f %f %f %f %f %f %f %f\n", a.m256_f32[0], a.m256_f32[1], a.m256_f32[2], a.m256_f32[3], a.m256_f32[4], a.m256_f32[5], a.m256_f32[6], a.m256_f32[7]);

	if (scale == 2)
	{
#pragma omp parallel for
		for (int j = 0; j < height; j++)
		{
			const int n = 2 * j;
			float* mb = meanb.ptr<float>(j);
			float* b = a_b.ptr<float>(j);
			float* g = a_g.ptr<float>(j);
			float* r = a_r.ptr<float>(j);
			float* gb = g_b.ptr<float>(n);
			float* gg = g_g.ptr<float>(n);
			float* gr = g_r.ptr<float>(n);
			float* d = dest.ptr<float>(n);
			for (int i = 0; i < width; i += 4)
			{
				const int m = 2 * i;
				__m256 mmb = _mm256_setr_ps(mb[i], mb[i], mb[i + 1], mb[i + 1], mb[i + 2], mb[i + 2], mb[i + 3], mb[i + 3]);
				__m256 ms = _mm256_setr_ps(b[i], b[i], b[i + 1], b[i + 1], b[i + 2], b[i + 2], b[i + 3], b[i + 3]);
				__m256 v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m), mmb);
				__m256 v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + dest.cols), mmb);

				ms = _mm256_setr_ps(g[i], g[i], g[i + 1], g[i + 1], g[i + 2], g[i + 2], g[i + 3], g[i + 3]);
				v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m), v0);
				v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + dest.cols), v1);

				ms = _mm256_setr_ps(r[i], r[i], r[i + 1], r[i + 1], r[i + 2], r[i + 2], r[i + 3], r[i + 3]);
				v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m), v0);
				v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + dest.cols), v1);
				_mm256_store_ps(d + m, v0);
				_mm256_store_ps(d + m + dest.cols, v1);
			}
		}
	}
	else if (scale == 4)
	{
#pragma omp parallel for
		for (int j = 0; j < height; j++)
		{
			const int n = 4 * j;
			float* mb = meanb.ptr<float>(j);
			float* b = a_b.ptr<float>(j);
			float* g = a_g.ptr<float>(j);
			float* r = a_r.ptr<float>(j);
			float* gb = g_b.ptr<float>(n);
			float* gg = g_g.ptr<float>(n);
			float* gr = g_r.ptr<float>(n);
			float* d = dest.ptr<float>(n);
			for (int i = 0; i < width; i += 2)
			{
				const int m = 4 * i;
				__m256 mmb = _mm256_setr_ps(mb[i], mb[i], mb[i], mb[i], mb[i + 1], mb[i + 1], mb[i + 1], mb[i + 1]);
				__m256 ms = _mm256_setr_ps(b[i], b[i], b[i], b[i], b[i + 1], b[i + 1], b[i + 1], b[i + 1]);
				__m256 v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m), mmb);
				__m256 v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + dest.cols), mmb);
				__m256 v2 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + 2 * dest.cols), mmb);
				__m256 v3 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + 3 * dest.cols), mmb);

				ms = _mm256_setr_ps(g[i + 0], g[i + 0], g[i + 0], g[i + 0], g[i + 1], g[i + 1], g[i + 1], g[i + 1]);
				v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m), v0);
				v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + dest.cols), v1);
				v2 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 2 * dest.cols), v2);
				v3 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 3 * dest.cols), v3);

				ms = _mm256_setr_ps(r[i], r[i], r[i + 0], r[i + 0], r[i + 1], r[i + 1], r[i + 1], r[i + 1]);
				v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m), v0);
				v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + dest.cols), v1);
				v2 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 2 * dest.cols), v2);
				v3 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 3 * dest.cols), v3);
				_mm256_store_ps(d + m, v0);
				_mm256_store_ps(d + m + dest.cols, v1);
				_mm256_store_ps(d + m + 2 * dest.cols, v2);
				_mm256_store_ps(d + m + 3 * dest.cols, v3);
			}
		}
	}
	else if (scale == 8)
	{
#pragma omp parallel for
		for (int j = 0; j < height; j++)
		{
			const int n = 8 * j;
			float* mb = meanb.ptr<float>(j);
			float* b = a_b.ptr<float>(j);
			float* g = a_g.ptr<float>(j);
			float* r = a_r.ptr<float>(j);
			float* gb = g_b.ptr<float>(n);
			float* gg = g_g.ptr<float>(n);
			float* gr = g_r.ptr<float>(n);
			float* d = dest.ptr<float>(n);
			for (int i = 0; i < width; i++)
			{
				const int m = 8 * i;
				__m256 mmb = _mm256_set1_ps(mb[i]);
				__m256 ms = _mm256_set1_ps(b[i]);

				__m256 v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m), mmb);
				__m256 v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + dest.cols), mmb);
				__m256 v2 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + 2 * dest.cols), mmb);
				__m256 v3 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + 3 * dest.cols), mmb);
				__m256 v4 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + 4 * dest.cols), mmb);
				__m256 v5 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + 5 * dest.cols), mmb);
				__m256 v6 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + 6 * dest.cols), mmb);
				__m256 v7 = _mm256_fmadd_ps(ms, _mm256_load_ps(gb + m + 7 * dest.cols), mmb);

				ms = _mm256_set1_ps(g[i]);
				v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m), v0);
				v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + dest.cols), v1);
				v2 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 2 * dest.cols), v2);
				v3 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 3 * dest.cols), v3);
				v4 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 4 * dest.cols), v4);
				v5 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 5 * dest.cols), v5);
				v6 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 6 * dest.cols), v6);
				v7 = _mm256_fmadd_ps(ms, _mm256_load_ps(gg + m + 7 * dest.cols), v7);

				ms = _mm256_set1_ps(r[i]);
				v0 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m), v0);
				v1 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + dest.cols), v1);
				v2 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 2 * dest.cols), v2);
				v3 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 3 * dest.cols), v3);
				v4 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 4 * dest.cols), v4);
				v5 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 5 * dest.cols), v5);
				v6 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 6 * dest.cols), v6);
				v7 = _mm256_fmadd_ps(ms, _mm256_load_ps(gr + m + 7 * dest.cols), v7);

				_mm256_store_ps(d + 8 * i, v0);
				_mm256_store_ps(d + 8 * i + 1 * dest.cols, v1);
				_mm256_store_ps(d + 8 * i + 2 * dest.cols, v2);
				_mm256_store_ps(d + 8 * i + 3 * dest.cols, v3);
				_mm256_store_ps(d + 8 * i + 4 * dest.cols, v4);
				_mm256_store_ps(d + 8 * i + 5 * dest.cols, v5);
				_mm256_store_ps(d + 8 * i + 6 * dest.cols, v6);
				_mm256_store_ps(d + 8 * i + 7 * dest.cols, v7);
			}
		}
	}
	else
	{
		for (int j = 0; j < height; j++)
		{
			int n = j * scale;
			float* s = a_b.ptr<float>(j);

			for (int i = 0, m = 0; i < width; i++, m += scale)
			{
				const float ltd = s[i];
				for (int l = 0; l < width; l++)
				{
					float* d = dest.ptr<float>(n + l);
					for (int k = 0; k < scale; k++)
					{
						d[m + k] = ltd;
					}
				}
			}
		}
	}
}

void guidedFilter_SepVHI::upsample_Guide3(cv::Mat& input_low, std::vector<cv::Mat>& guide, std::vector<cv::Mat>& guide_low, cv::Mat& output)
{
	//Ip2ab
	if (parallelType == NAIVE)
	{
		Ip2ab_Guide3_sep_VHI_AVX(guide_low[0], guide_low[1], guide_low[2], input_low, r, eps, a_b, a_g, a_r, b);
	}
	else
	{
		Ip2ab_Guide3_sep_VHI_AVX_omp(guide_low[0], guide_low[1], guide_low[2], input_low, r, eps, a_b, a_g, a_r, b);
	}

	//ab2q

	if (upsample_method == INTER_NEAREST)
	{
		//blur_ab_sep_VHI_omp(a_b, a_g, a_r, b, r, mean_a_b, mean_a_g, mean_a_r, mean_b);
	//upsampleNN(mean_a_b, mean_a_g, mean_a_r, guide[0], guide[1], guide[2], mean_b, output);

		ab_upNN_2q_sep_VHI_omp(a_b, a_g, a_r, b, guide[0], guide[1], guide[2], r, output);
	}
	else
	{
		//blurSeparableVHI(a_b, a_g, a_r, b, r, mean_a_b, mean_a_g, mean_a_r, mean_b);
		blurSeparableVHI_omp(a_b, a_g, a_r, b, r, mean_a_b, mean_a_g, mean_a_r, mean_b);

		resize(mean_b, output, guide[0].size(), 0, 0, upsample_method);
		resize(mean_a_b, a_high_b, guide[0].size(), 0, 0, upsample_method);
		resize(mean_a_g, a_high_g, guide[0].size(), 0, 0, upsample_method);
		resize(mean_a_r, a_high_r, guide[0].size(), 0, 0, upsample_method);

		ab2q_fmadd_omp(a_high_b, a_high_g, a_high_r, guide[0], guide[1], guide[2], output, output);
	}
}