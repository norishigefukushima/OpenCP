#include "guidedFilter_SepVHI.h"
#include "guidedFilter_SepVHI_Share.h"

#include <iostream>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;

#include <arithmetic.hpp>
#include <inlineSIMDFunctions.hpp>
using namespace cp;

void Ip2ab_Guide1Src3_sep_VHIShare_AVX(cv::Mat& I, std::vector<cv::Mat>& p, const int r, float eps, vector<cv::Mat>& a, vector<cv::Mat>& b)
{
	const int width = I.cols;
	const int height = I.rows;
	cv::Size size = cv::Size(width, height);
	a[0].create(size, CV_32F);
	b[0].create(size, CV_32F);
	a[1].create(size, CV_32F);
	b[1].create(size, CV_32F);
	a[2].create(size, CV_32F);
	b[2].create(size, CV_32F);

	const int d = 2 * r + 1;
	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const __m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat temp(Size(width + 2 * R, 8 * omp_get_max_threads()), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp___I = temp.ptr<float>(0, R);
		float* tp__II = temp.ptr<float>(1, R);
		float* tp__p0 = temp.ptr<float>(2, R);
		float* tp_Ip0 = temp.ptr<float>(3, R);
		float* tp__p1 = temp.ptr<float>(4, R);
		float* tp_Ip1 = temp.ptr<float>(5, R);
		float* tp__p2 = temp.ptr<float>(6, R);
		float* tp_Ip2 = temp.ptr<float>(7, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* Iptr = I.ptr<float>(i - r, j);
				float* p0ptr = p[0].ptr<float>(i - r, j);
				float* p1ptr = p[1].ptr<float>(i - r, j);
				float* p2ptr = p[2].ptr<float>(i - r, j);

				__m256 mSum_I = _mm256_load_ps(Iptr);
				__m256 mSum_II = _mm256_mul_ps(mSum_I, mSum_I);
				__m256 mSum_p0 = _mm256_load_ps(p0ptr);
				__m256 mSum_Ip0 = _mm256_mul_ps(mSum_I, mSum_p0);
				__m256 mSum_p1 = _mm256_load_ps(p1ptr);
				__m256 mSum_Ip1 = _mm256_mul_ps(mSum_I, mSum_p1);
				__m256 mSum_p2 = _mm256_load_ps(p2ptr);
				__m256 mSum_Ip2 = _mm256_mul_ps(mSum_I, mSum_p2);

				Iptr += width;
				p0ptr += width;
				p1ptr += width;
				p2ptr += width;
				for (int k = 1; k < d; k++)
				{
					__m256 mi = _mm256_load_ps(Iptr);
					mSum_I = _mm256_add_ps(mSum_I, mi);
					mSum_II = _mm256_fmadd_ps(mi, mi, mSum_II);

					__m256 mp = _mm256_load_ps(p0ptr);
					mSum_p0 = _mm256_add_ps(mSum_p0, mp);
					mSum_Ip0 = _mm256_fmadd_ps(mp, mi, mSum_Ip0);

					mp = _mm256_load_ps(p1ptr);
					mSum_p1 = _mm256_add_ps(mSum_p1, mp);
					mSum_Ip1 = _mm256_fmadd_ps(mp, mi, mSum_Ip1);

					mp = _mm256_load_ps(p2ptr);
					mSum_p2 = _mm256_add_ps(mSum_p2, mp);
					mSum_Ip2 = _mm256_fmadd_ps(mp, mi, mSum_Ip2);

					Iptr += width;
					p0ptr += width;
					p1ptr += width;
					p2ptr += width;
				}

				_mm256_store_ps(tp___I, mSum_I);
				_mm256_store_ps(tp__II, mSum_II);
				_mm256_store_ps(tp__p0, mSum_p0);
				_mm256_store_ps(tp_Ip0, mSum_Ip0);
				_mm256_store_ps(tp__p1, mSum_p1);
				_mm256_store_ps(tp_Ip1, mSum_Ip1);
				_mm256_store_ps(tp__p2, mSum_p2);
				_mm256_store_ps(tp_Ip2, mSum_Ip2);

				tp___I += 8;
				tp__II += 8;
				tp__p0 += 8;
				tp_Ip0 += 8;
				tp__p1 += 8;
				tp_Ip1 += 8;
				tp__p2 += 8;
				tp_Ip2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b0ptr = b[0].ptr<float>(i);
			float* a0ptr = a[0].ptr<float>(i);
			float* b1ptr = b[1].ptr<float>(i);
			float* a1ptr = a[1].ptr<float>(i);
			float* b2ptr = b[2].ptr<float>(i);
			float* a2ptr = a[2].ptr<float>(i);

			tp___I = temp.ptr<float>(0, roffset);
			tp__II = temp.ptr<float>(1, roffset);
			tp__p0 = temp.ptr<float>(2, roffset);
			tp_Ip0 = temp.ptr<float>(3, roffset);
			tp__p1 = temp.ptr<float>(4, roffset);
			tp_Ip1 = temp.ptr<float>(5, roffset);
			tp__p2 = temp.ptr<float>(6, roffset);
			tp_Ip2 = temp.ptr<float>(7, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum__I = _mm256_loadu_ps(tp___I);
				__m256 mSum_II = _mm256_loadu_ps(tp__II);
				__m256 mSum_p0 = _mm256_loadu_ps(tp__p0);
				__m256 mSumIp0 = _mm256_loadu_ps(tp_Ip0);
				__m256 mSum_p1 = _mm256_loadu_ps(tp__p1);
				__m256 mSumIp1 = _mm256_loadu_ps(tp_Ip1);
				__m256 mSum_p2 = _mm256_loadu_ps(tp__p2);
				__m256 mSumIp2 = _mm256_loadu_ps(tp_Ip2);

				for (int k = 1; k < d; k++)
				{
					mSum__I = _mm256_add_ps(mSum__I, _mm256_loadu_ps(tp___I + k));
					mSum_II = _mm256_add_ps(mSum_II, _mm256_loadu_ps(tp__II + k));
					mSum_p0 = _mm256_add_ps(mSum_p0, _mm256_loadu_ps(tp__p0 + k));
					mSumIp0 = _mm256_add_ps(mSumIp0, _mm256_loadu_ps(tp_Ip0 + k));
					mSum_p1 = _mm256_add_ps(mSum_p1, _mm256_loadu_ps(tp__p1 + k));
					mSumIp1 = _mm256_add_ps(mSumIp1, _mm256_loadu_ps(tp_Ip1 + k));
					mSum_p2 = _mm256_add_ps(mSum_p2, _mm256_loadu_ps(tp__p2 + k));
					mSumIp2 = _mm256_add_ps(mSumIp2, _mm256_loadu_ps(tp_Ip2 + k));
				}

				const __m256 m__I = _mm256_mul_ps(mSum__I, mDiv);
				const __m256 m_II = _mm256_mul_ps(mSum_II, mDiv);
				const __m256 m_p0 = _mm256_mul_ps(mSum_p0, mDiv);
				const __m256 mIp0 = _mm256_mul_ps(mSumIp0, mDiv);
				const __m256 m_p1 = _mm256_mul_ps(mSum_p1, mDiv);
				const __m256 mIp1 = _mm256_mul_ps(mSumIp1, mDiv);
				const __m256 m_p2 = _mm256_mul_ps(mSum_p2, mDiv);
				const __m256 mIp2 = _mm256_mul_ps(mSumIp2, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				__m256 mvar = _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_add_ps(_mm256_fnmadd_ps(m__I, m__I, m_II), meps));


				__m256 ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p0, mIp0), mvar);
				_mm256_store_ps(a0ptr, ma);
				a0ptr += 8;
				_mm256_store_ps(b0ptr, _mm256_fnmadd_ps(ma, m__I, m_p0));
				b0ptr += 8;

				ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p1, mIp1), mvar);
				_mm256_store_ps(a1ptr, ma);
				a1ptr += 8;
				_mm256_store_ps(b1ptr, _mm256_fnmadd_ps(ma, m__I, m_p1));
				b1ptr += 8;

				ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p2, mIp2), mvar);
				_mm256_store_ps(a2ptr, ma);
				a2ptr += 8;
				_mm256_store_ps(b2ptr, _mm256_fnmadd_ps(ma, m__I, m_p2));
				b2ptr += 8;

				tp___I += 8;
				tp__II += 8;
				tp__p0 += 8;
				tp_Ip0 += 8;
				tp__p1 += 8;
				tp_Ip1 += 8;
				tp__p2 += 8;
				tp_Ip2 += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				const int v = max(0, min(height - 1, i - r));

				float* Iptr = I.ptr<float>(v, j);
				float* p0ptr = p[0].ptr<float>(v, j);
				float* p1ptr = p[1].ptr<float>(v, j);
				float* p2ptr = p[2].ptr<float>(v, j);

				__m256 mSum_I = _mm256_load_ps(Iptr);
				__m256 mSum_II = _mm256_mul_ps(mSum_I, mSum_I);
				__m256 mSum_p0 = _mm256_load_ps(p0ptr);
				__m256 mSum_Ip0 = _mm256_mul_ps(mSum_I, mSum_p0);
				__m256 mSum_p1 = _mm256_load_ps(p1ptr);
				__m256 mSum_Ip1 = _mm256_mul_ps(mSum_I, mSum_p1);
				__m256 mSum_p2 = _mm256_load_ps(p2ptr);
				__m256 mSum_Ip2 = _mm256_mul_ps(mSum_I, mSum_p2);

				for (int k = 1; k < d; k++)
				{
					const int v = max(0, min(height - 1, i - r + k));

					float* Iptr = I.ptr<float>(v, j);
					float* p0ptr = p[0].ptr<float>(v, j);
					float* p1ptr = p[1].ptr<float>(v, j);
					float* p2ptr = p[2].ptr<float>(v, j);

					__m256 mi = _mm256_load_ps(Iptr);
					mSum_I = _mm256_add_ps(mSum_I, mi);
					mSum_II = _mm256_fmadd_ps(mi, mi, mSum_II);

					__m256 mp = _mm256_load_ps(p0ptr);
					mSum_p0 = _mm256_add_ps(mSum_p0, mp);
					mSum_Ip0 = _mm256_fmadd_ps(mp, mi, mSum_Ip0);

					mp = _mm256_load_ps(p1ptr);
					mSum_p1 = _mm256_add_ps(mSum_p1, mp);
					mSum_Ip1 = _mm256_fmadd_ps(mp, mi, mSum_Ip1);

					mp = _mm256_load_ps(p2ptr);
					mSum_p2 = _mm256_add_ps(mSum_p2, mp);
					mSum_Ip2 = _mm256_fmadd_ps(mp, mi, mSum_Ip2);
				}

				_mm256_store_ps(tp___I, mSum_I);
				_mm256_store_ps(tp__II, mSum_II);
				_mm256_store_ps(tp__p0, mSum_p0);
				_mm256_store_ps(tp_Ip0, mSum_Ip0);
				_mm256_store_ps(tp__p1, mSum_p1);
				_mm256_store_ps(tp_Ip1, mSum_Ip1);
				_mm256_store_ps(tp__p2, mSum_p2);
				_mm256_store_ps(tp_Ip2, mSum_Ip2);

				tp___I += 8;
				tp__II += 8;
				tp__p0 += 8;
				tp_Ip0 += 8;
				tp__p1 += 8;
				tp_Ip1 += 8;
				tp__p2 += 8;
				tp_Ip2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b0ptr = b[0].ptr<float>(i);
			float* a0ptr = a[0].ptr<float>(i);
			float* b1ptr = b[1].ptr<float>(i);
			float* a1ptr = a[1].ptr<float>(i);
			float* b2ptr = b[2].ptr<float>(i);
			float* a2ptr = a[2].ptr<float>(i);

			tp___I = temp.ptr<float>(0, roffset);
			tp__II = temp.ptr<float>(1, roffset);
			tp__p0 = temp.ptr<float>(2, roffset);
			tp_Ip0 = temp.ptr<float>(3, roffset);
			tp__p1 = temp.ptr<float>(4, roffset);
			tp_Ip1 = temp.ptr<float>(5, roffset);
			tp__p2 = temp.ptr<float>(6, roffset);
			tp_Ip2 = temp.ptr<float>(7, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum__I = _mm256_loadu_ps(tp___I);
				__m256 mSum_II = _mm256_loadu_ps(tp__II);
				__m256 mSum_p0 = _mm256_loadu_ps(tp__p0);
				__m256 mSumIp0 = _mm256_loadu_ps(tp_Ip0);
				__m256 mSum_p1 = _mm256_loadu_ps(tp__p1);
				__m256 mSumIp1 = _mm256_loadu_ps(tp_Ip1);
				__m256 mSum_p2 = _mm256_loadu_ps(tp__p2);
				__m256 mSumIp2 = _mm256_loadu_ps(tp_Ip2);

				for (int k = 1; k < d; k++)
				{
					mSum__I = _mm256_add_ps(mSum__I, _mm256_loadu_ps(tp___I + k));
					mSum_II = _mm256_add_ps(mSum_II, _mm256_loadu_ps(tp__II + k));
					mSum_p0 = _mm256_add_ps(mSum_p0, _mm256_loadu_ps(tp__p0 + k));
					mSumIp0 = _mm256_add_ps(mSumIp0, _mm256_loadu_ps(tp_Ip0 + k));
					mSum_p1 = _mm256_add_ps(mSum_p1, _mm256_loadu_ps(tp__p1 + k));
					mSumIp1 = _mm256_add_ps(mSumIp1, _mm256_loadu_ps(tp_Ip1 + k));
					mSum_p2 = _mm256_add_ps(mSum_p2, _mm256_loadu_ps(tp__p2 + k));
					mSumIp2 = _mm256_add_ps(mSumIp2, _mm256_loadu_ps(tp_Ip2 + k));
				}

				const __m256 m__I = _mm256_mul_ps(mSum__I, mDiv);
				const __m256 m_II = _mm256_mul_ps(mSum_II, mDiv);
				const __m256 m_p0 = _mm256_mul_ps(mSum_p0, mDiv);
				const __m256 mIp0 = _mm256_mul_ps(mSumIp0, mDiv);
				const __m256 m_p1 = _mm256_mul_ps(mSum_p1, mDiv);
				const __m256 mIp1 = _mm256_mul_ps(mSumIp1, mDiv);
				const __m256 m_p2 = _mm256_mul_ps(mSum_p2, mDiv);
				const __m256 mIp2 = _mm256_mul_ps(mSumIp2, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				__m256 mvar = _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_add_ps(_mm256_fnmadd_ps(m__I, m__I, m_II), meps));


				__m256 ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p0, mIp0), mvar);
				_mm256_store_ps(a0ptr, ma);
				a0ptr += 8;
				_mm256_store_ps(b0ptr, _mm256_fnmadd_ps(ma, m__I, m_p0));
				b0ptr += 8;

				ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p1, mIp1), mvar);
				_mm256_store_ps(a1ptr, ma);
				a1ptr += 8;
				_mm256_store_ps(b1ptr, _mm256_fnmadd_ps(ma, m__I, m_p1));
				b1ptr += 8;

				ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p2, mIp2), mvar);
				_mm256_store_ps(a2ptr, ma);
				a2ptr += 8;
				_mm256_store_ps(b2ptr, _mm256_fnmadd_ps(ma, m__I, m_p2));
				b2ptr += 8;

				tp___I += 8;
				tp__II += 8;
				tp__p0 += 8;
				tp_Ip0 += 8;
				tp__p1 += 8;
				tp_Ip1 += 8;
				tp__p2 += 8;
				tp_Ip2 += 8;
			}
		}
	}
}

void Ip2ab_Guide1Src3_sep_VHIShare_AVX_omp(cv::Mat& I, std::vector<cv::Mat>& p, const int r, float eps, vector<cv::Mat>& a, vector<cv::Mat>& b)
{
	const int width = I.cols;
	const int height = I.rows;
	cv::Size size = cv::Size(width, height);
	a[0].create(size, CV_32F);
	b[0].create(size, CV_32F);
	a[1].create(size, CV_32F);
	b[1].create(size, CV_32F);
	a[2].create(size, CV_32F);
	b[2].create(size, CV_32F);

	const int d = 2 * r + 1;
	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const __m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat buff(Size(width + 2 * R, 8 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 8 * omp_get_thread_num(), width + 2 * R, 8));

		float* tp___I = temp.ptr<float>(0, R);
		float* tp__II = temp.ptr<float>(1, R);
		float* tp__p0 = temp.ptr<float>(2, R);
		float* tp_Ip0 = temp.ptr<float>(3, R);
		float* tp__p1 = temp.ptr<float>(4, R);
		float* tp_Ip1 = temp.ptr<float>(5, R);
		float* tp__p2 = temp.ptr<float>(6, R);
		float* tp_Ip2 = temp.ptr<float>(7, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* Iptr = I.ptr<float>(i - r, j);
				float* p0ptr = p[0].ptr<float>(i - r, j);
				float* p1ptr = p[1].ptr<float>(i - r, j);
				float* p2ptr = p[2].ptr<float>(i - r, j);

				__m256 mSum_I = _mm256_load_ps(Iptr);
				__m256 mSum_II = _mm256_mul_ps(mSum_I, mSum_I);
				__m256 mSum_p0 = _mm256_load_ps(p0ptr);
				__m256 mSum_Ip0 = _mm256_mul_ps(mSum_I, mSum_p0);
				__m256 mSum_p1 = _mm256_load_ps(p1ptr);
				__m256 mSum_Ip1 = _mm256_mul_ps(mSum_I, mSum_p1);
				__m256 mSum_p2 = _mm256_load_ps(p2ptr);
				__m256 mSum_Ip2 = _mm256_mul_ps(mSum_I, mSum_p2);

				Iptr += width;
				p0ptr += width;
				p1ptr += width;
				p2ptr += width;

				for (int k = 1; k < d; k++)
				{
					__m256 mi = _mm256_load_ps(Iptr);
					mSum_I = _mm256_add_ps(mSum_I, mi);
					mSum_II = _mm256_fmadd_ps(mi, mi, mSum_II);

					__m256 mp = _mm256_load_ps(p0ptr);
					mSum_p0 = _mm256_add_ps(mSum_p0, mp);
					mSum_Ip0 = _mm256_fmadd_ps(mp, mi, mSum_Ip0);

					mp = _mm256_load_ps(p1ptr);
					mSum_p1 = _mm256_add_ps(mSum_p1, mp);
					mSum_Ip1 = _mm256_fmadd_ps(mp, mi, mSum_Ip1);

					mp = _mm256_load_ps(p2ptr);
					mSum_p2 = _mm256_add_ps(mSum_p2, mp);
					mSum_Ip2 = _mm256_fmadd_ps(mp, mi, mSum_Ip2);

					Iptr += width;
					p0ptr += width;
					p1ptr += width;
					p2ptr += width;
				}

				_mm256_store_ps(tp___I, mSum_I);
				_mm256_store_ps(tp__II, mSum_II);
				_mm256_store_ps(tp__p0, mSum_p0);
				_mm256_store_ps(tp_Ip0, mSum_Ip0);
				_mm256_store_ps(tp__p1, mSum_p1);
				_mm256_store_ps(tp_Ip1, mSum_Ip1);
				_mm256_store_ps(tp__p2, mSum_p2);
				_mm256_store_ps(tp_Ip2, mSum_Ip2);

				tp___I += 8;
				tp__II += 8;
				tp__p0 += 8;
				tp_Ip0 += 8;
				tp__p1 += 8;
				tp_Ip1 += 8;
				tp__p2 += 8;
				tp_Ip2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b0ptr = b[0].ptr<float>(i);
			float* a0ptr = a[0].ptr<float>(i);
			float* b1ptr = b[1].ptr<float>(i);
			float* a1ptr = a[1].ptr<float>(i);
			float* b2ptr = b[2].ptr<float>(i);
			float* a2ptr = a[2].ptr<float>(i);

			tp___I = temp.ptr<float>(0, roffset);
			tp__II = temp.ptr<float>(1, roffset);
			tp__p0 = temp.ptr<float>(2, roffset);
			tp_Ip0 = temp.ptr<float>(3, roffset);
			tp__p1 = temp.ptr<float>(4, roffset);
			tp_Ip1 = temp.ptr<float>(5, roffset);
			tp__p2 = temp.ptr<float>(6, roffset);
			tp_Ip2 = temp.ptr<float>(7, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum__I = _mm256_loadu_ps(tp___I);
				__m256 mSum_II = _mm256_loadu_ps(tp__II);
				__m256 mSum_p0 = _mm256_loadu_ps(tp__p0);
				__m256 mSumIp0 = _mm256_loadu_ps(tp_Ip0);
				__m256 mSum_p1 = _mm256_loadu_ps(tp__p1);
				__m256 mSumIp1 = _mm256_loadu_ps(tp_Ip1);
				__m256 mSum_p2 = _mm256_loadu_ps(tp__p2);
				__m256 mSumIp2 = _mm256_loadu_ps(tp_Ip2);

				for (int k = 1; k < d; k++)
				{
					mSum__I = _mm256_add_ps(mSum__I, _mm256_loadu_ps(tp___I + k));
					mSum_II = _mm256_add_ps(mSum_II, _mm256_loadu_ps(tp__II + k));
					mSum_p0 = _mm256_add_ps(mSum_p0, _mm256_loadu_ps(tp__p0 + k));
					mSumIp0 = _mm256_add_ps(mSumIp0, _mm256_loadu_ps(tp_Ip0 + k));
					mSum_p1 = _mm256_add_ps(mSum_p1, _mm256_loadu_ps(tp__p1 + k));
					mSumIp1 = _mm256_add_ps(mSumIp1, _mm256_loadu_ps(tp_Ip1 + k));
					mSum_p2 = _mm256_add_ps(mSum_p2, _mm256_loadu_ps(tp__p2 + k));
					mSumIp2 = _mm256_add_ps(mSumIp2, _mm256_loadu_ps(tp_Ip2 + k));
				}

				const __m256 m__I = _mm256_mul_ps(mSum__I, mDiv);
				const __m256 m_II = _mm256_mul_ps(mSum_II, mDiv);
				const __m256 m_p0 = _mm256_mul_ps(mSum_p0, mDiv);
				const __m256 mIp0 = _mm256_mul_ps(mSumIp0, mDiv);
				const __m256 m_p1 = _mm256_mul_ps(mSum_p1, mDiv);
				const __m256 mIp1 = _mm256_mul_ps(mSumIp1, mDiv);
				const __m256 m_p2 = _mm256_mul_ps(mSum_p2, mDiv);
				const __m256 mIp2 = _mm256_mul_ps(mSumIp2, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				__m256 mvar = _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_add_ps(_mm256_fnmadd_ps(m__I, m__I, m_II), meps));


				__m256 ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p0, mIp0), mvar);
				_mm256_store_ps(a0ptr, ma);
				a0ptr += 8;
				_mm256_store_ps(b0ptr, _mm256_fnmadd_ps(ma, m__I, m_p0));
				b0ptr += 8;

				ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p1, mIp1), mvar);
				_mm256_store_ps(a1ptr, ma);
				a1ptr += 8;
				_mm256_store_ps(b1ptr, _mm256_fnmadd_ps(ma, m__I, m_p1));
				b1ptr += 8;

				ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p2, mIp2), mvar);
				_mm256_store_ps(a2ptr, ma);
				a2ptr += 8;
				_mm256_store_ps(b2ptr, _mm256_fnmadd_ps(ma, m__I, m_p2));
				b2ptr += 8;

				tp___I += 8;
				tp__II += 8;
				tp__p0 += 8;
				tp_Ip0 += 8;
				tp__p1 += 8;
				tp_Ip1 += 8;
				tp__p2 += 8;
				tp_Ip2 += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				const int v = max(0, min(height - 1, i - r));

				float* Iptr = I.ptr<float>(v, j);
				float* p0ptr = p[0].ptr<float>(v, j);
				float* p1ptr = p[1].ptr<float>(v, j);
				float* p2ptr = p[2].ptr<float>(v, j);

				__m256 mSum_I = _mm256_load_ps(Iptr);
				__m256 mSum_II = _mm256_mul_ps(mSum_I, mSum_I);
				__m256 mSum_p0 = _mm256_load_ps(p0ptr);
				__m256 mSum_Ip0 = _mm256_mul_ps(mSum_I, mSum_p0);
				__m256 mSum_p1 = _mm256_load_ps(p1ptr);
				__m256 mSum_Ip1 = _mm256_mul_ps(mSum_I, mSum_p1);
				__m256 mSum_p2 = _mm256_load_ps(p2ptr);
				__m256 mSum_Ip2 = _mm256_mul_ps(mSum_I, mSum_p2);

				for (int k = 1; k < d; k++)
				{
					const int v = max(0, min(height - 1, i - r + k));

					float* Iptr = I.ptr<float>(v, j);
					float* p0ptr = p[0].ptr<float>(v, j);
					float* p1ptr = p[1].ptr<float>(v, j);
					float* p2ptr = p[2].ptr<float>(v, j);

					__m256 mi = _mm256_load_ps(Iptr);
					mSum_I = _mm256_add_ps(mSum_I, mi);
					mSum_II = _mm256_fmadd_ps(mi, mi, mSum_II);

					__m256 mp = _mm256_load_ps(p0ptr);
					mSum_p0 = _mm256_add_ps(mSum_p0, mp);
					mSum_Ip0 = _mm256_fmadd_ps(mp, mi, mSum_Ip0);

					mp = _mm256_load_ps(p1ptr);
					mSum_p1 = _mm256_add_ps(mSum_p1, mp);
					mSum_Ip1 = _mm256_fmadd_ps(mp, mi, mSum_Ip1);

					mp = _mm256_load_ps(p2ptr);
					mSum_p2 = _mm256_add_ps(mSum_p2, mp);
					mSum_Ip2 = _mm256_fmadd_ps(mp, mi, mSum_Ip2);
				}

				_mm256_store_ps(tp___I, mSum_I);
				_mm256_store_ps(tp__II, mSum_II);
				_mm256_store_ps(tp__p0, mSum_p0);
				_mm256_store_ps(tp_Ip0, mSum_Ip0);
				_mm256_store_ps(tp__p1, mSum_p1);
				_mm256_store_ps(tp_Ip1, mSum_Ip1);
				_mm256_store_ps(tp__p2, mSum_p2);
				_mm256_store_ps(tp_Ip2, mSum_Ip2);

				tp___I += 8;
				tp__II += 8;
				tp__p0 += 8;
				tp_Ip0 += 8;
				tp__p1 += 8;
				tp_Ip1 += 8;
				tp__p2 += 8;
				tp_Ip2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b0ptr = b[0].ptr<float>(i);
			float* a0ptr = a[0].ptr<float>(i);
			float* b1ptr = b[1].ptr<float>(i);
			float* a1ptr = a[1].ptr<float>(i);
			float* b2ptr = b[2].ptr<float>(i);
			float* a2ptr = a[2].ptr<float>(i);

			tp___I = temp.ptr<float>(0, roffset);
			tp__II = temp.ptr<float>(1, roffset);
			tp__p0 = temp.ptr<float>(2, roffset);
			tp_Ip0 = temp.ptr<float>(3, roffset);
			tp__p1 = temp.ptr<float>(4, roffset);
			tp_Ip1 = temp.ptr<float>(5, roffset);
			tp__p2 = temp.ptr<float>(6, roffset);
			tp_Ip2 = temp.ptr<float>(7, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum__I = _mm256_loadu_ps(tp___I);
				__m256 mSum_II = _mm256_loadu_ps(tp__II);
				__m256 mSum_p0 = _mm256_loadu_ps(tp__p0);
				__m256 mSumIp0 = _mm256_loadu_ps(tp_Ip0);
				__m256 mSum_p1 = _mm256_loadu_ps(tp__p1);
				__m256 mSumIp1 = _mm256_loadu_ps(tp_Ip1);
				__m256 mSum_p2 = _mm256_loadu_ps(tp__p2);
				__m256 mSumIp2 = _mm256_loadu_ps(tp_Ip2);

				for (int k = 1; k < d; k++)
				{
					mSum__I = _mm256_add_ps(mSum__I, _mm256_loadu_ps(tp___I + k));
					mSum_II = _mm256_add_ps(mSum_II, _mm256_loadu_ps(tp__II + k));
					mSum_p0 = _mm256_add_ps(mSum_p0, _mm256_loadu_ps(tp__p0 + k));
					mSumIp0 = _mm256_add_ps(mSumIp0, _mm256_loadu_ps(tp_Ip0 + k));
					mSum_p1 = _mm256_add_ps(mSum_p1, _mm256_loadu_ps(tp__p1 + k));
					mSumIp1 = _mm256_add_ps(mSumIp1, _mm256_loadu_ps(tp_Ip1 + k));
					mSum_p2 = _mm256_add_ps(mSum_p2, _mm256_loadu_ps(tp__p2 + k));
					mSumIp2 = _mm256_add_ps(mSumIp2, _mm256_loadu_ps(tp_Ip2 + k));
				}

				const __m256 m__I = _mm256_mul_ps(mSum__I, mDiv);
				const __m256 m_II = _mm256_mul_ps(mSum_II, mDiv);
				const __m256 m_p0 = _mm256_mul_ps(mSum_p0, mDiv);
				const __m256 mIp0 = _mm256_mul_ps(mSumIp0, mDiv);
				const __m256 m_p1 = _mm256_mul_ps(mSum_p1, mDiv);
				const __m256 mIp1 = _mm256_mul_ps(mSumIp1, mDiv);
				const __m256 m_p2 = _mm256_mul_ps(mSum_p2, mDiv);
				const __m256 mIp2 = _mm256_mul_ps(mSumIp2, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				__m256 mvar = _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_add_ps(_mm256_fnmadd_ps(m__I, m__I, m_II), meps));


				__m256 ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p0, mIp0), mvar);
				_mm256_store_ps(a0ptr, ma);
				a0ptr += 8;
				_mm256_store_ps(b0ptr, _mm256_fnmadd_ps(ma, m__I, m_p0));
				b0ptr += 8;

				ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p1, mIp1), mvar);
				_mm256_store_ps(a1ptr, ma);
				a1ptr += 8;
				_mm256_store_ps(b1ptr, _mm256_fnmadd_ps(ma, m__I, m_p1));
				b1ptr += 8;

				ma = _mm256_mul_ps(_mm256_fnmadd_ps(m__I, m_p2, mIp2), mvar);
				_mm256_store_ps(a2ptr, ma);
				a2ptr += 8;
				_mm256_store_ps(b2ptr, _mm256_fnmadd_ps(ma, m__I, m_p2));
				b2ptr += 8;

				tp___I += 8;
				tp__II += 8;
				tp__p0 += 8;
				tp_Ip0 += 8;
				tp__p1 += 8;
				tp_Ip1 += 8;
				tp__p2 += 8;
				tp_Ip2 += 8;
			}
		}
	}
}


void Ip2ab_Guide3Src3_sep_VHIShare_AVX(std::vector<cv::Mat>& I, std::vector<cv::Mat>& p, const int r, float eps,
	std::vector<cv::Mat>& ab_p, std::vector<cv::Mat>& ag_p, std::vector<cv::Mat>& ar_p, std::vector<cv::Mat>& b_p)
{
	const int width = I[0].cols;
	const int height = I[0].rows;
	cv::Size size = cv::Size(width, height);

	ab_p[0].create(size, CV_32F);
	ab_p[1].create(size, CV_32F);
	ab_p[2].create(size, CV_32F);
	ag_p[0].create(size, CV_32F);
	ag_p[1].create(size, CV_32F);
	ag_p[2].create(size, CV_32F);
	ar_p[0].create(size, CV_32F);
	ar_p[1].create(size, CV_32F);
	ar_p[2].create(size, CV_32F);
	b_p[0].create(size, CV_32F);
	b_p[1].create(size, CV_32F);
	b_p[2].create(size, CV_32F);

	const int d = 2 * r + 1;
	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const __m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat temp(Size(width + 2 * R, 21), CV_32FC1);

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

		float* tp_p___0 = temp.ptr<float>(9, R);
		float* tp_Ip0_b = temp.ptr<float>(10, R);
		float* tp_Ip0_g = temp.ptr<float>(11, R);
		float* tp_Ip0_r = temp.ptr<float>(12, R);

		float* tp_p___1 = temp.ptr<float>(13, R);
		float* tp_Ip1_b = temp.ptr<float>(14, R);
		float* tp_Ip1_g = temp.ptr<float>(15, R);
		float* tp_Ip1_r = temp.ptr<float>(16, R);

		float* tp_p___2 = temp.ptr<float>(17, R);
		float* tp_Ip2_b = temp.ptr<float>(18, R);
		float* tp_Ip2_g = temp.ptr<float>(19, R);
		float* tp_Ip2_r = temp.ptr<float>(20, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* I0ptr = I[0].ptr<float>(i - r, j);
				float* I1ptr = I[1].ptr<float>(i - r, j);
				float* I2ptr = I[2].ptr<float>(i - r, j);
				float* p0ptr = I[0].ptr<float>(i - r, j);
				float* p1ptr = I[1].ptr<float>(i - r, j);
				float* p2ptr = I[2].ptr<float>(i - r, j);

				__m256 mSum_Ib = _mm256_load_ps(I0ptr);
				__m256 mSum_Ig = _mm256_load_ps(I1ptr);
				__m256 mSum_Ir = _mm256_load_ps(I2ptr);

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p0 = _mm256_load_ps(p0ptr);
				__m256 mSum_p1 = _mm256_load_ps(p1ptr);
				__m256 mSum_p2 = _mm256_load_ps(p2ptr);

				__m256 mSum_Ip0b = _mm256_mul_ps(mSum_Ib, mSum_p0);
				__m256 mSum_Ip0g = _mm256_mul_ps(mSum_Ig, mSum_p0);
				__m256 mSum_Ip0r = _mm256_mul_ps(mSum_Ir, mSum_p0);

				__m256 mSum_Ip1b = _mm256_mul_ps(mSum_Ib, mSum_p1);
				__m256 mSum_Ip1g = _mm256_mul_ps(mSum_Ig, mSum_p1);
				__m256 mSum_Ip1r = _mm256_mul_ps(mSum_Ir, mSum_p1);

				__m256 mSum_Ip2b = _mm256_mul_ps(mSum_Ib, mSum_p2);
				__m256 mSum_Ip2g = _mm256_mul_ps(mSum_Ig, mSum_p2);
				__m256 mSum_Ip2r = _mm256_mul_ps(mSum_Ir, mSum_p2);

				I0ptr += width;
				I1ptr += width;
				I2ptr += width;
				p0ptr += width;
				p1ptr += width;
				p2ptr += width;

				for (int k = 1; k < d; k++)
				{
					const __m256 mb0 = _mm256_load_ps(I0ptr);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);

					const __m256 mg0 = _mm256_load_ps(I1ptr);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);

					const __m256 mr0 = _mm256_load_ps(I2ptr);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);

					__m256 mpl = _mm256_load_ps(p0ptr);
					mSum_p0 = _mm256_add_ps(mSum_p0, mpl);
					mSum_Ip0b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip0b);
					mSum_Ip0g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip0g);
					mSum_Ip0r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip0r);

					mpl = _mm256_load_ps(p1ptr);
					mSum_p1 = _mm256_add_ps(mSum_p1, mpl);
					mSum_Ip1b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip1b);
					mSum_Ip1g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip1g);
					mSum_Ip1r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip1r);

					mpl = _mm256_load_ps(p2ptr);
					mSum_p2 = _mm256_add_ps(mSum_p2, mpl);
					mSum_Ip2b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip2b);
					mSum_Ip2g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip2g);
					mSum_Ip2r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip2r);

					I0ptr += width;
					I1ptr += width;
					I2ptr += width;
					p0ptr += width;
					p1ptr += width;
					p2ptr += width;
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

				_mm256_store_ps(tp_p___0, mSum_p0);
				_mm256_store_ps(tp_Ip0_b, mSum_Ip0b);
				_mm256_store_ps(tp_Ip0_g, mSum_Ip0g);
				_mm256_store_ps(tp_Ip0_r, mSum_Ip0r);

				_mm256_store_ps(tp_p___1, mSum_p1);
				_mm256_store_ps(tp_Ip1_b, mSum_Ip1b);
				_mm256_store_ps(tp_Ip1_g, mSum_Ip1g);
				_mm256_store_ps(tp_Ip1_r, mSum_Ip1r);

				_mm256_store_ps(tp_p___2, mSum_p2);
				_mm256_store_ps(tp_Ip2_b, mSum_Ip2b);
				_mm256_store_ps(tp_Ip2_g, mSum_Ip2g);
				_mm256_store_ps(tp_Ip2_r, mSum_Ip2r);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p___0 += 8;
				tp_Ip0_b += 8;
				tp_Ip0_g += 8;
				tp_Ip0_r += 8;

				tp_p___1 += 8;
				tp_Ip1_b += 8;
				tp_Ip1_g += 8;
				tp_Ip1_r += 8;

				tp_p___2 += 8;
				tp_Ip2_b += 8;
				tp_Ip2_g += 8;
				tp_Ip2_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b__p0 = b_p[0].ptr<float>(i);
			float* ab_p0 = ab_p[0].ptr<float>(i);
			float* ag_p0 = ag_p[0].ptr<float>(i);
			float* ar_p0 = ar_p[0].ptr<float>(i);

			float* b__p1 = b_p[1].ptr<float>(i);
			float* ab_p1 = ab_p[1].ptr<float>(i);
			float* ag_p1 = ag_p[1].ptr<float>(i);
			float* ar_p1 = ar_p[1].ptr<float>(i);

			float* b__p2 = b_p[2].ptr<float>(i);
			float* ab_p2 = ab_p[2].ptr<float>(i);
			float* ag_p2 = ag_p[2].ptr<float>(i);
			float* ar_p2 = ar_p[2].ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp_p___0 = temp.ptr<float>(9, roffset);
			tp_Ip0_b = temp.ptr<float>(10, roffset);
			tp_Ip0_g = temp.ptr<float>(11, roffset);
			tp_Ip0_r = temp.ptr<float>(12, roffset);

			tp_p___1 = temp.ptr<float>(13, roffset);
			tp_Ip1_b = temp.ptr<float>(14, roffset);
			tp_Ip1_g = temp.ptr<float>(15, roffset);
			tp_Ip1_r = temp.ptr<float>(16, roffset);

			tp_p___2 = temp.ptr<float>(17, roffset);
			tp_Ip2_b = temp.ptr<float>(18, roffset);
			tp_Ip2_g = temp.ptr<float>(19, roffset);
			tp_Ip2_r = temp.ptr<float>(20, roffset);

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

				__m256 mSum_p0 = _mm256_loadu_ps(tp_p___0);
				__m256 mSum_p1 = _mm256_loadu_ps(tp_p___1);
				__m256 mSum_p2 = _mm256_loadu_ps(tp_p___2);

				__m256 mSum_Ip0_b = _mm256_loadu_ps(tp_Ip0_b);
				__m256 mSum_Ip0_g = _mm256_loadu_ps(tp_Ip0_g);
				__m256 mSum_Ip0_r = _mm256_loadu_ps(tp_Ip0_r);

				__m256 mSum_Ip1_b = _mm256_loadu_ps(tp_Ip1_b);
				__m256 mSum_Ip1_g = _mm256_loadu_ps(tp_Ip1_g);
				__m256 mSum_Ip1_r = _mm256_loadu_ps(tp_Ip1_r);

				__m256 mSum_Ip2_b = _mm256_loadu_ps(tp_Ip2_b);
				__m256 mSum_Ip2_g = _mm256_loadu_ps(tp_Ip2_g);
				__m256 mSum_Ip2_r = _mm256_loadu_ps(tp_Ip2_r);

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

					mSum_p0 = _mm256_add_ps(mSum_p0, _mm256_loadu_ps(tp_p___0 + k));
					mSum_p1 = _mm256_add_ps(mSum_p1, _mm256_loadu_ps(tp_p___1 + k));
					mSum_p2 = _mm256_add_ps(mSum_p2, _mm256_loadu_ps(tp_p___2 + k));

					mSum_Ip0_b = _mm256_add_ps(mSum_Ip0_b, _mm256_loadu_ps(tp_Ip0_b + k));
					mSum_Ip0_g = _mm256_add_ps(mSum_Ip0_g, _mm256_loadu_ps(tp_Ip0_g + k));
					mSum_Ip0_r = _mm256_add_ps(mSum_Ip0_r, _mm256_loadu_ps(tp_Ip0_r + k));

					mSum_Ip1_b = _mm256_add_ps(mSum_Ip1_b, _mm256_loadu_ps(tp_Ip1_b + k));
					mSum_Ip1_g = _mm256_add_ps(mSum_Ip1_g, _mm256_loadu_ps(tp_Ip1_g + k));
					mSum_Ip1_r = _mm256_add_ps(mSum_Ip1_r, _mm256_loadu_ps(tp_Ip1_r + k));

					mSum_Ip2_b = _mm256_add_ps(mSum_Ip2_b, _mm256_loadu_ps(tp_Ip2_b + k));
					mSum_Ip2_g = _mm256_add_ps(mSum_Ip2_g, _mm256_loadu_ps(tp_Ip2_g + k));
					mSum_Ip2_r = _mm256_add_ps(mSum_Ip2_r, _mm256_loadu_ps(tp_Ip2_r + k));
				}

				const __m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				const __m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				const __m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				const __m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				const __m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				const __m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				const __m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				const __m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				const __m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				const __m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				const __m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				const __m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				const __m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				const __m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				const __m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				//p0
				__m256 mp0 = _mm256_mul_ps(mSum_p0, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp0, _mm256_mul_ps(mSum_Ip0_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp0, _mm256_mul_ps(mSum_Ip0_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp0, _mm256_mul_ps(mSum_Ip0_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p0, mTmp);
				ab_p0 += 8;
				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp0);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p0, mTmp);
				ag_p0 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p0, mTmp);
				ar_p0 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p0, mB);
				b__p0 += 8;

				//p1
				__m256 mp1 = _mm256_mul_ps(mSum_p1, mDiv);

				mCovB = _mm256_fnmadd_ps(mb, mp1, _mm256_mul_ps(mSum_Ip1_b, mDiv));
				mCovG = _mm256_fnmadd_ps(mg, mp1, _mm256_mul_ps(mSum_Ip1_g, mDiv));
				mCovR = _mm256_fnmadd_ps(mr, mp1, _mm256_mul_ps(mSum_Ip1_r, mDiv));

				mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p1, mTmp);
				ab_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mb, mp1);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p1, mTmp);
				ag_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p1, mTmp);
				ar_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p1, mB);
				b__p1 += 8;


				//p2
				__m256 mp2 = _mm256_mul_ps(mSum_p2, mDiv);

				mCovB = _mm256_fnmadd_ps(mb, mp2, _mm256_mul_ps(mSum_Ip2_b, mDiv));
				mCovG = _mm256_fnmadd_ps(mg, mp2, _mm256_mul_ps(mSum_Ip2_g, mDiv));
				mCovR = _mm256_fnmadd_ps(mr, mp2, _mm256_mul_ps(mSum_Ip2_r, mDiv));

				mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p2, mTmp);
				ab_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mb, mp2);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p2, mTmp);
				ag_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p2, mTmp);
				ar_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p2, mB);
				b__p2 += 8;


				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p___0 += 8;
				tp_Ip0_b += 8;
				tp_Ip0_g += 8;
				tp_Ip0_r += 8;

				tp_p___1 += 8;
				tp_Ip1_b += 8;
				tp_Ip1_g += 8;
				tp_Ip1_r += 8;

				tp_p___2 += 8;
				tp_Ip2_b += 8;
				tp_Ip2_g += 8;
				tp_Ip2_r += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				int v = max(0, min(i - r, height - 1));

				__m256 mSum_Ib = _mm256_load_ps(I[0].ptr<float>(v, j));
				__m256 mSum_Ig = _mm256_load_ps(I[1].ptr<float>(v, j));
				__m256 mSum_Ir = _mm256_load_ps(I[2].ptr<float>(v, j));

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p0 = _mm256_load_ps(p[0].ptr<float>(v, j));
				__m256 mSum_p1 = _mm256_load_ps(p[1].ptr<float>(v, j));
				__m256 mSum_p2 = _mm256_load_ps(p[2].ptr<float>(v, j));

				__m256 mSum_Ip0b = _mm256_mul_ps(mSum_Ib, mSum_p0);
				__m256 mSum_Ip0g = _mm256_mul_ps(mSum_Ig, mSum_p0);
				__m256 mSum_Ip0r = _mm256_mul_ps(mSum_Ir, mSum_p0);

				__m256 mSum_Ip1b = _mm256_mul_ps(mSum_Ib, mSum_p1);
				__m256 mSum_Ip1g = _mm256_mul_ps(mSum_Ig, mSum_p1);
				__m256 mSum_Ip1r = _mm256_mul_ps(mSum_Ir, mSum_p1);

				__m256 mSum_Ip2b = _mm256_mul_ps(mSum_Ib, mSum_p2);
				__m256 mSum_Ip2g = _mm256_mul_ps(mSum_Ig, mSum_p2);
				__m256 mSum_Ip2r = _mm256_mul_ps(mSum_Ir, mSum_p2);

				for (int k = 1; k < d; k++)
				{
					int v = max(0, min(i + k - r, height - 1));

					float* sp = I[0].ptr<float>(v, j);
					const __m256 mb0 = _mm256_load_ps(sp);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);

					sp = I[1].ptr<float>(v, j);
					const __m256 mg0 = _mm256_load_ps(sp);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);

					sp = I[2].ptr<float>(v, j);
					const __m256 mr0 = _mm256_load_ps(sp);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);

					sp = p[0].ptr<float>(v, j);
					__m256 mpl = _mm256_load_ps(sp);
					mSum_p0 = _mm256_add_ps(mSum_p0, mpl);
					mSum_Ip0b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip0b);
					mSum_Ip0g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip0g);
					mSum_Ip0r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip0r);

					sp = p[1].ptr<float>(v, j);
					mpl = _mm256_load_ps(sp);
					mSum_p1 = _mm256_add_ps(mSum_p1, mpl);
					mSum_Ip1b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip1b);
					mSum_Ip1g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip1g);
					mSum_Ip1r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip1r);

					sp = p[2].ptr<float>(v, j);
					mpl = _mm256_load_ps(sp);
					mSum_p2 = _mm256_add_ps(mSum_p2, mpl);
					mSum_Ip2b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip2b);
					mSum_Ip2g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip2g);
					mSum_Ip2r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip2r);
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

				_mm256_store_ps(tp_p___0, mSum_p0);
				_mm256_store_ps(tp_Ip0_b, mSum_Ip0b);
				_mm256_store_ps(tp_Ip0_g, mSum_Ip0g);
				_mm256_store_ps(tp_Ip0_r, mSum_Ip0r);

				_mm256_store_ps(tp_p___1, mSum_p1);
				_mm256_store_ps(tp_Ip1_b, mSum_Ip1b);
				_mm256_store_ps(tp_Ip1_g, mSum_Ip1g);
				_mm256_store_ps(tp_Ip1_r, mSum_Ip1r);

				_mm256_store_ps(tp_p___2, mSum_p2);
				_mm256_store_ps(tp_Ip2_b, mSum_Ip2b);
				_mm256_store_ps(tp_Ip2_g, mSum_Ip2g);
				_mm256_store_ps(tp_Ip2_r, mSum_Ip2r);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p___0 += 8;
				tp_Ip0_b += 8;
				tp_Ip0_g += 8;
				tp_Ip0_r += 8;

				tp_p___1 += 8;
				tp_Ip1_b += 8;
				tp_Ip1_g += 8;
				tp_Ip1_r += 8;

				tp_p___2 += 8;
				tp_Ip2_b += 8;
				tp_Ip2_g += 8;
				tp_Ip2_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b__p0 = b_p[0].ptr<float>(i);
			float* ab_p0 = ab_p[0].ptr<float>(i);
			float* ag_p0 = ag_p[0].ptr<float>(i);
			float* ar_p0 = ar_p[0].ptr<float>(i);

			float* b__p1 = b_p[1].ptr<float>(i);
			float* ab_p1 = ab_p[1].ptr<float>(i);
			float* ag_p1 = ag_p[1].ptr<float>(i);
			float* ar_p1 = ar_p[1].ptr<float>(i);

			float* b__p2 = b_p[2].ptr<float>(i);
			float* ab_p2 = ab_p[2].ptr<float>(i);
			float* ag_p2 = ag_p[2].ptr<float>(i);
			float* ar_p2 = ar_p[2].ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp_p___0 = temp.ptr<float>(9, roffset);
			tp_Ip0_b = temp.ptr<float>(10, roffset);
			tp_Ip0_g = temp.ptr<float>(11, roffset);
			tp_Ip0_r = temp.ptr<float>(12, roffset);

			tp_p___1 = temp.ptr<float>(13, roffset);
			tp_Ip1_b = temp.ptr<float>(14, roffset);
			tp_Ip1_g = temp.ptr<float>(15, roffset);
			tp_Ip1_r = temp.ptr<float>(16, roffset);

			tp_p___2 = temp.ptr<float>(17, roffset);
			tp_Ip2_b = temp.ptr<float>(18, roffset);
			tp_Ip2_g = temp.ptr<float>(19, roffset);
			tp_Ip2_r = temp.ptr<float>(20, roffset);

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

				__m256 mSum_I_b = _mm256_loadu_ps(tp_I_b);
				__m256 mSum_I_g = _mm256_loadu_ps(tp_I_g);
				__m256 mSum_I_r = _mm256_loadu_ps(tp_I_r);

				__m256 mSum_I_bb = _mm256_loadu_ps(tp_I_bb);
				__m256 mSum_I_bg = _mm256_loadu_ps(tp_I_bg);
				__m256 mSum_I_br = _mm256_loadu_ps(tp_I_br);
				__m256 mSum_I_gg = _mm256_loadu_ps(tp_I_gg);
				__m256 mSum_I_gr = _mm256_loadu_ps(tp_I_gr);
				__m256 mSum_I_rr = _mm256_loadu_ps(tp_I_rr);

				__m256 mSum_p0 = _mm256_loadu_ps(tp_p___0);
				__m256 mSum_p1 = _mm256_loadu_ps(tp_p___1);
				__m256 mSum_p2 = _mm256_loadu_ps(tp_p___2);

				__m256 mSum_Ip0_b = _mm256_loadu_ps(tp_Ip0_b);
				__m256 mSum_Ip0_g = _mm256_loadu_ps(tp_Ip0_g);
				__m256 mSum_Ip0_r = _mm256_loadu_ps(tp_Ip0_r);

				__m256 mSum_Ip1_b = _mm256_loadu_ps(tp_Ip1_b);
				__m256 mSum_Ip1_g = _mm256_loadu_ps(tp_Ip1_g);
				__m256 mSum_Ip1_r = _mm256_loadu_ps(tp_Ip1_r);

				__m256 mSum_Ip2_b = _mm256_loadu_ps(tp_Ip2_b);
				__m256 mSum_Ip2_g = _mm256_loadu_ps(tp_Ip2_g);
				__m256 mSum_Ip2_r = _mm256_loadu_ps(tp_Ip2_r);

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

					mSum_p0 = _mm256_add_ps(mSum_p0, _mm256_loadu_ps(tp_p___0 + k));
					mSum_p1 = _mm256_add_ps(mSum_p1, _mm256_loadu_ps(tp_p___1 + k));
					mSum_p2 = _mm256_add_ps(mSum_p2, _mm256_loadu_ps(tp_p___2 + k));

					mSum_Ip0_b = _mm256_add_ps(mSum_Ip0_b, _mm256_loadu_ps(tp_Ip0_b + k));
					mSum_Ip0_g = _mm256_add_ps(mSum_Ip0_g, _mm256_loadu_ps(tp_Ip0_g + k));
					mSum_Ip0_r = _mm256_add_ps(mSum_Ip0_r, _mm256_loadu_ps(tp_Ip0_r + k));

					mSum_Ip1_b = _mm256_add_ps(mSum_Ip1_b, _mm256_loadu_ps(tp_Ip1_b + k));
					mSum_Ip1_g = _mm256_add_ps(mSum_Ip1_g, _mm256_loadu_ps(tp_Ip1_g + k));
					mSum_Ip1_r = _mm256_add_ps(mSum_Ip1_r, _mm256_loadu_ps(tp_Ip1_r + k));

					mSum_Ip2_b = _mm256_add_ps(mSum_Ip2_b, _mm256_loadu_ps(tp_Ip2_b + k));
					mSum_Ip2_g = _mm256_add_ps(mSum_Ip2_g, _mm256_loadu_ps(tp_Ip2_g + k));
					mSum_Ip2_r = _mm256_add_ps(mSum_Ip2_r, _mm256_loadu_ps(tp_Ip2_r + k));
				}

				const __m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				const __m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				const __m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				const __m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				const __m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				const __m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				const __m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				const __m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				const __m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				const __m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				const __m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				const __m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				const __m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				const __m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				const __m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				//p0
				__m256 mp0 = _mm256_mul_ps(mSum_p0, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp0, _mm256_mul_ps(mSum_Ip0_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp0, _mm256_mul_ps(mSum_Ip0_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp0, _mm256_mul_ps(mSum_Ip0_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p0, mTmp);
				ab_p0 += 8;
				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp0);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p0, mTmp);
				ag_p0 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p0, mTmp);
				ar_p0 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p0, mB);
				b__p0 += 8;

				//p1
				__m256 mp1 = _mm256_mul_ps(mSum_p1, mDiv);

				mCovB = _mm256_fnmadd_ps(mb, mp1, _mm256_mul_ps(mSum_Ip1_b, mDiv));
				mCovG = _mm256_fnmadd_ps(mg, mp1, _mm256_mul_ps(mSum_Ip1_g, mDiv));
				mCovR = _mm256_fnmadd_ps(mr, mp1, _mm256_mul_ps(mSum_Ip1_r, mDiv));

				mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p1, mTmp);
				ab_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mb, mp1);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p1, mTmp);
				ag_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p1, mTmp);
				ar_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p1, mB);
				b__p1 += 8;


				//p2
				__m256 mp2 = _mm256_mul_ps(mSum_p2, mDiv);

				mCovB = _mm256_fnmadd_ps(mb, mp2, _mm256_mul_ps(mSum_Ip2_b, mDiv));
				mCovG = _mm256_fnmadd_ps(mg, mp2, _mm256_mul_ps(mSum_Ip2_g, mDiv));
				mCovR = _mm256_fnmadd_ps(mr, mp2, _mm256_mul_ps(mSum_Ip2_r, mDiv));

				mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p2, mTmp);
				ab_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mb, mp2);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p2, mTmp);
				ag_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p2, mTmp);
				ar_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p2, mB);
				b__p2 += 8;


				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp_p___0 += 8;
				tp_Ip0_b += 8;
				tp_Ip0_g += 8;
				tp_Ip0_r += 8;

				tp_p___1 += 8;
				tp_Ip1_b += 8;
				tp_Ip1_g += 8;
				tp_Ip1_r += 8;

				tp_p___2 += 8;
				tp_Ip2_b += 8;
				tp_Ip2_g += 8;
				tp_Ip2_r += 8;
			}
		}
	}
}

void Ip2ab_Guide3Src3_sep_VHIShare_AVX_omp(std::vector<cv::Mat>& I, std::vector<cv::Mat>& p, const int r, float eps,
	std::vector<cv::Mat>& a_b, std::vector<cv::Mat>& a_g, std::vector<cv::Mat>& a_r, std::vector<cv::Mat>& b)
{
	const int width = I[0].cols;
	const int height = I[0].rows;

	cv::Size size = cv::Size(width, height);
	a_b[0].create(size, CV_32F);
	a_b[1].create(size, CV_32F);
	a_b[2].create(size, CV_32F);
	a_g[0].create(size, CV_32F);
	a_g[1].create(size, CV_32F);
	a_g[2].create(size, CV_32F);
	a_r[0].create(size, CV_32F);
	a_r[1].create(size, CV_32F);
	a_r[2].create(size, CV_32F);
	b[0].create(size, CV_32F);
	b[1].create(size, CV_32F);
	b[2].create(size, CV_32F);

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	const __m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat buff(Size(width + 2 * R, 21 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 21 * omp_get_thread_num(), width + 2 * R, 21));

		float* tp_I_b = temp.ptr<float>(0, R);
		float* tp_I_g = temp.ptr<float>(1, R);
		float* tp_I_r = temp.ptr<float>(2, R);

		float* tp_I_bb = temp.ptr<float>(3, R);
		float* tp_I_bg = temp.ptr<float>(4, R);
		float* tp_I_br = temp.ptr<float>(5, R);
		float* tp_I_gg = temp.ptr<float>(6, R);
		float* tp_I_gr = temp.ptr<float>(7, R);
		float* tp_I_rr = temp.ptr<float>(8, R);

		float* tp____p0 = temp.ptr<float>(9, R);
		float* tp_Ip0_b = temp.ptr<float>(10, R);
		float* tp_Ip0_g = temp.ptr<float>(11, R);
		float* tp_Ip0_r = temp.ptr<float>(12, R);

		float* tp____p1 = temp.ptr<float>(13, R);
		float* tp_Ip1_b = temp.ptr<float>(14, R);
		float* tp_Ip1_g = temp.ptr<float>(15, R);
		float* tp_Ip1_r = temp.ptr<float>(16, R);

		float* tp____p2 = temp.ptr<float>(17, R);
		float* tp_Ip2_b = temp.ptr<float>(18, R);
		float* tp_Ip2_g = temp.ptr<float>(19, R);
		float* tp_Ip2_r = temp.ptr<float>(20, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* I0ptr = I[0].ptr<float>(i - r, j);
				float* I1ptr = I[1].ptr<float>(i - r, j);
				float* I2ptr = I[2].ptr<float>(i - r, j);
				float* p0ptr = p[0].ptr<float>(i - r, j);
				float* p1ptr = p[1].ptr<float>(i - r, j);
				float* p2ptr = p[2].ptr<float>(i - r, j);

				__m256 mSum_Ib = _mm256_load_ps(I0ptr);
				__m256 mSum_Ig = _mm256_load_ps(I1ptr);
				__m256 mSum_Ir = _mm256_load_ps(I2ptr);

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p0 = _mm256_load_ps(p0ptr);
				__m256 mSum_p1 = _mm256_load_ps(p1ptr);
				__m256 mSum_p2 = _mm256_load_ps(p2ptr);

				__m256 mSum_Ip0b = _mm256_mul_ps(mSum_Ib, mSum_p0);
				__m256 mSum_Ip0g = _mm256_mul_ps(mSum_Ig, mSum_p0);
				__m256 mSum_Ip0r = _mm256_mul_ps(mSum_Ir, mSum_p0);

				__m256 mSum_Ip1b = _mm256_mul_ps(mSum_Ib, mSum_p1);
				__m256 mSum_Ip1g = _mm256_mul_ps(mSum_Ig, mSum_p1);
				__m256 mSum_Ip1r = _mm256_mul_ps(mSum_Ir, mSum_p1);

				__m256 mSum_Ip2b = _mm256_mul_ps(mSum_Ib, mSum_p2);
				__m256 mSum_Ip2g = _mm256_mul_ps(mSum_Ig, mSum_p2);
				__m256 mSum_Ip2r = _mm256_mul_ps(mSum_Ir, mSum_p2);

				I0ptr += width;
				I1ptr += width;
				I2ptr += width;
				p0ptr += width;
				p1ptr += width;
				p2ptr += width;

				for (int k = 1; k < d; k++)
				{
					const __m256 mb0 = _mm256_load_ps(I0ptr);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);

					const __m256 mg0 = _mm256_load_ps(I1ptr);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);

					const __m256 mr0 = _mm256_load_ps(I2ptr);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);

					__m256 mpl = _mm256_load_ps(p0ptr);
					mSum_p0 = _mm256_add_ps(mSum_p0, mpl);
					mSum_Ip0b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip0b);
					mSum_Ip0g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip0g);
					mSum_Ip0r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip0r);

					mpl = _mm256_load_ps(p1ptr);
					mSum_p1 = _mm256_add_ps(mSum_p1, mpl);
					mSum_Ip1b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip1b);
					mSum_Ip1g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip1g);
					mSum_Ip1r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip1r);

					mpl = _mm256_load_ps(p2ptr);
					mSum_p2 = _mm256_add_ps(mSum_p2, mpl);
					mSum_Ip2b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip2b);
					mSum_Ip2g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip2g);
					mSum_Ip2r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip2r);

					I0ptr += width;
					I1ptr += width;
					I2ptr += width;
					p0ptr += width;
					p1ptr += width;
					p2ptr += width;
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

				_mm256_store_ps(tp____p0, mSum_p0);
				_mm256_store_ps(tp_Ip0_b, mSum_Ip0b);
				_mm256_store_ps(tp_Ip0_g, mSum_Ip0g);
				_mm256_store_ps(tp_Ip0_r, mSum_Ip0r);

				_mm256_store_ps(tp____p1, mSum_p1);
				_mm256_store_ps(tp_Ip1_b, mSum_Ip1b);
				_mm256_store_ps(tp_Ip1_g, mSum_Ip1g);
				_mm256_store_ps(tp_Ip1_r, mSum_Ip1r);

				_mm256_store_ps(tp____p2, mSum_p2);
				_mm256_store_ps(tp_Ip2_b, mSum_Ip2b);
				_mm256_store_ps(tp_Ip2_g, mSum_Ip2g);
				_mm256_store_ps(tp_Ip2_r, mSum_Ip2r);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp____p0 += 8;
				tp_Ip0_b += 8;
				tp_Ip0_g += 8;
				tp_Ip0_r += 8;

				tp____p1 += 8;
				tp_Ip1_b += 8;
				tp_Ip1_g += 8;
				tp_Ip1_r += 8;

				tp____p2 += 8;
				tp_Ip2_b += 8;
				tp_Ip2_g += 8;
				tp_Ip2_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b__p0 = b[0].ptr<float>(i);
			float* ab_p0 = a_b[0].ptr<float>(i);
			float* ag_p0 = a_g[0].ptr<float>(i);
			float* ar_p0 = a_r[0].ptr<float>(i);

			float* b__p1 = b[1].ptr<float>(i);
			float* ab_p1 = a_b[1].ptr<float>(i);
			float* ag_p1 = a_g[1].ptr<float>(i);
			float* ar_p1 = a_r[1].ptr<float>(i);

			float* b__p2 = b[2].ptr<float>(i);
			float* ab_p2 = a_b[2].ptr<float>(i);
			float* ag_p2 = a_g[2].ptr<float>(i);
			float* ar_p2 = a_r[2].ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp____p0 = temp.ptr<float>(9, roffset);
			tp_Ip0_b = temp.ptr<float>(10, roffset);
			tp_Ip0_g = temp.ptr<float>(11, roffset);
			tp_Ip0_r = temp.ptr<float>(12, roffset);

			tp____p1 = temp.ptr<float>(13, roffset);
			tp_Ip1_b = temp.ptr<float>(14, roffset);
			tp_Ip1_g = temp.ptr<float>(15, roffset);
			tp_Ip1_r = temp.ptr<float>(16, roffset);

			tp____p2 = temp.ptr<float>(17, roffset);
			tp_Ip2_b = temp.ptr<float>(18, roffset);
			tp_Ip2_g = temp.ptr<float>(19, roffset);
			tp_Ip2_r = temp.ptr<float>(20, roffset);

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

				__m256 mSum_p0 = _mm256_loadu_ps(tp____p0);
				__m256 mSum_p1 = _mm256_loadu_ps(tp____p1);
				__m256 mSum_p2 = _mm256_loadu_ps(tp____p2);

				__m256 mSum_Ip0_b = _mm256_loadu_ps(tp_Ip0_b);
				__m256 mSum_Ip0_g = _mm256_loadu_ps(tp_Ip0_g);
				__m256 mSum_Ip0_r = _mm256_loadu_ps(tp_Ip0_r);

				__m256 mSum_Ip1_b = _mm256_loadu_ps(tp_Ip1_b);
				__m256 mSum_Ip1_g = _mm256_loadu_ps(tp_Ip1_g);
				__m256 mSum_Ip1_r = _mm256_loadu_ps(tp_Ip1_r);

				__m256 mSum_Ip2_b = _mm256_loadu_ps(tp_Ip2_b);
				__m256 mSum_Ip2_g = _mm256_loadu_ps(tp_Ip2_g);
				__m256 mSum_Ip2_r = _mm256_loadu_ps(tp_Ip2_r);

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

					mSum_p0 = _mm256_add_ps(mSum_p0, _mm256_loadu_ps(tp____p0 + k));
					mSum_p1 = _mm256_add_ps(mSum_p1, _mm256_loadu_ps(tp____p1 + k));
					mSum_p2 = _mm256_add_ps(mSum_p2, _mm256_loadu_ps(tp____p2 + k));

					mSum_Ip0_b = _mm256_add_ps(mSum_Ip0_b, _mm256_loadu_ps(tp_Ip0_b + k));
					mSum_Ip0_g = _mm256_add_ps(mSum_Ip0_g, _mm256_loadu_ps(tp_Ip0_g + k));
					mSum_Ip0_r = _mm256_add_ps(mSum_Ip0_r, _mm256_loadu_ps(tp_Ip0_r + k));

					mSum_Ip1_b = _mm256_add_ps(mSum_Ip1_b, _mm256_loadu_ps(tp_Ip1_b + k));
					mSum_Ip1_g = _mm256_add_ps(mSum_Ip1_g, _mm256_loadu_ps(tp_Ip1_g + k));
					mSum_Ip1_r = _mm256_add_ps(mSum_Ip1_r, _mm256_loadu_ps(tp_Ip1_r + k));

					mSum_Ip2_b = _mm256_add_ps(mSum_Ip2_b, _mm256_loadu_ps(tp_Ip2_b + k));
					mSum_Ip2_g = _mm256_add_ps(mSum_Ip2_g, _mm256_loadu_ps(tp_Ip2_g + k));
					mSum_Ip2_r = _mm256_add_ps(mSum_Ip2_r, _mm256_loadu_ps(tp_Ip2_r + k));
				}

				const __m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				const __m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				const __m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				const __m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				const __m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				const __m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				const __m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				const __m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				const __m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				const __m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				const __m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				const __m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				const __m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				const __m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				const __m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				//p0
				__m256 mp0 = _mm256_mul_ps(mSum_p0, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp0, _mm256_mul_ps(mSum_Ip0_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp0, _mm256_mul_ps(mSum_Ip0_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp0, _mm256_mul_ps(mSum_Ip0_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p0, mTmp);
				ab_p0 += 8;
				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp0);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p0, mTmp);
				ag_p0 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p0, mTmp);
				ar_p0 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p0, mB);
				b__p0 += 8;

				//p1
				__m256 mp1 = _mm256_mul_ps(mSum_p1, mDiv);

				mCovB = _mm256_fnmadd_ps(mb, mp1, _mm256_mul_ps(mSum_Ip1_b, mDiv));
				mCovG = _mm256_fnmadd_ps(mg, mp1, _mm256_mul_ps(mSum_Ip1_g, mDiv));
				mCovR = _mm256_fnmadd_ps(mr, mp1, _mm256_mul_ps(mSum_Ip1_r, mDiv));

				mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p1, mTmp);
				ab_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mb, mp1);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p1, mTmp);
				ag_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p1, mTmp);
				ar_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p1, mB);
				b__p1 += 8;


				//p2
				__m256 mp2 = _mm256_mul_ps(mSum_p2, mDiv);

				mCovB = _mm256_fnmadd_ps(mb, mp2, _mm256_mul_ps(mSum_Ip2_b, mDiv));
				mCovG = _mm256_fnmadd_ps(mg, mp2, _mm256_mul_ps(mSum_Ip2_g, mDiv));
				mCovR = _mm256_fnmadd_ps(mr, mp2, _mm256_mul_ps(mSum_Ip2_r, mDiv));

				mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p2, mTmp);
				ab_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mb, mp2);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p2, mTmp);
				ag_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p2, mTmp);
				ar_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p2, mB);
				b__p2 += 8;


				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp____p0 += 8;
				tp_Ip0_b += 8;
				tp_Ip0_g += 8;
				tp_Ip0_r += 8;

				tp____p1 += 8;
				tp_Ip1_b += 8;
				tp_Ip1_g += 8;
				tp_Ip1_r += 8;

				tp____p2 += 8;
				tp_Ip2_b += 8;
				tp_Ip2_g += 8;
				tp_Ip2_r += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				int v = max(0, min(i - r, height - 1));

				__m256 mSum_Ib = _mm256_load_ps(I[0].ptr<float>(v, j));
				__m256 mSum_Ig = _mm256_load_ps(I[1].ptr<float>(v, j));
				__m256 mSum_Ir = _mm256_load_ps(I[2].ptr<float>(v, j));

				__m256 mSum_Ibb = _mm256_mul_ps(mSum_Ib, mSum_Ib);
				__m256 mSum_Ibg = _mm256_mul_ps(mSum_Ib, mSum_Ig);
				__m256 mSum_Ibr = _mm256_mul_ps(mSum_Ib, mSum_Ir);
				__m256 mSum_Igg = _mm256_mul_ps(mSum_Ig, mSum_Ig);
				__m256 mSum_Igr = _mm256_mul_ps(mSum_Ig, mSum_Ir);
				__m256 mSum_Irr = _mm256_mul_ps(mSum_Ir, mSum_Ir);

				__m256 mSum_p0 = _mm256_load_ps(p[0].ptr<float>(v, j));
				__m256 mSum_p1 = _mm256_load_ps(p[1].ptr<float>(v, j));
				__m256 mSum_p2 = _mm256_load_ps(p[2].ptr<float>(v, j));

				__m256 mSum_Ip0b = _mm256_mul_ps(mSum_Ib, mSum_p0);
				__m256 mSum_Ip0g = _mm256_mul_ps(mSum_Ig, mSum_p0);
				__m256 mSum_Ip0r = _mm256_mul_ps(mSum_Ir, mSum_p0);

				__m256 mSum_Ip1b = _mm256_mul_ps(mSum_Ib, mSum_p1);
				__m256 mSum_Ip1g = _mm256_mul_ps(mSum_Ig, mSum_p1);
				__m256 mSum_Ip1r = _mm256_mul_ps(mSum_Ir, mSum_p1);

				__m256 mSum_Ip2b = _mm256_mul_ps(mSum_Ib, mSum_p2);
				__m256 mSum_Ip2g = _mm256_mul_ps(mSum_Ig, mSum_p2);
				__m256 mSum_Ip2r = _mm256_mul_ps(mSum_Ir, mSum_p2);

				for (int k = 1; k < d; k++)
				{
					int v = max(0, min(i + k - r, height - 1));

					float* sp = I[0].ptr<float>(v, j);
					const __m256 mb0 = _mm256_load_ps(sp);
					mSum_Ib = _mm256_add_ps(mSum_Ib, mb0);

					sp = I[1].ptr<float>(v, j);
					const __m256 mg0 = _mm256_load_ps(sp);
					mSum_Ig = _mm256_add_ps(mSum_Ig, mg0);

					sp = I[2].ptr<float>(v, j);
					const __m256 mr0 = _mm256_load_ps(sp);
					mSum_Ir = _mm256_add_ps(mSum_Ir, mr0);

					mSum_Ibb = _mm256_fmadd_ps(mb0, mb0, mSum_Ibb);
					mSum_Ibg = _mm256_fmadd_ps(mb0, mg0, mSum_Ibg);
					mSum_Ibr = _mm256_fmadd_ps(mb0, mr0, mSum_Ibr);
					mSum_Igg = _mm256_fmadd_ps(mg0, mg0, mSum_Igg);
					mSum_Igr = _mm256_fmadd_ps(mg0, mr0, mSum_Igr);
					mSum_Irr = _mm256_fmadd_ps(mr0, mr0, mSum_Irr);

					sp = p[0].ptr<float>(v, j);
					__m256 mpl = _mm256_load_ps(sp);
					mSum_p0 = _mm256_add_ps(mSum_p0, mpl);
					mSum_Ip0b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip0b);
					mSum_Ip0g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip0g);
					mSum_Ip0r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip0r);

					sp = p[1].ptr<float>(v, j);
					mpl = _mm256_load_ps(sp);
					mSum_p1 = _mm256_add_ps(mSum_p1, mpl);
					mSum_Ip1b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip1b);
					mSum_Ip1g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip1g);
					mSum_Ip1r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip1r);

					sp = p[2].ptr<float>(v, j);
					mpl = _mm256_load_ps(sp);
					mSum_p2 = _mm256_add_ps(mSum_p2, mpl);
					mSum_Ip2b = _mm256_fmadd_ps(mpl, mb0, mSum_Ip2b);
					mSum_Ip2g = _mm256_fmadd_ps(mpl, mg0, mSum_Ip2g);
					mSum_Ip2r = _mm256_fmadd_ps(mpl, mr0, mSum_Ip2r);
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

				_mm256_store_ps(tp____p0, mSum_p0);
				_mm256_store_ps(tp_Ip0_b, mSum_Ip0b);
				_mm256_store_ps(tp_Ip0_g, mSum_Ip0g);
				_mm256_store_ps(tp_Ip0_r, mSum_Ip0r);

				_mm256_store_ps(tp____p1, mSum_p1);
				_mm256_store_ps(tp_Ip1_b, mSum_Ip1b);
				_mm256_store_ps(tp_Ip1_g, mSum_Ip1g);
				_mm256_store_ps(tp_Ip1_r, mSum_Ip1r);

				_mm256_store_ps(tp____p2, mSum_p2);
				_mm256_store_ps(tp_Ip2_b, mSum_Ip2b);
				_mm256_store_ps(tp_Ip2_g, mSum_Ip2g);
				_mm256_store_ps(tp_Ip2_r, mSum_Ip2r);

				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp____p0 += 8;
				tp_Ip0_b += 8;
				tp_Ip0_g += 8;
				tp_Ip0_r += 8;

				tp____p1 += 8;
				tp_Ip1_b += 8;
				tp_Ip1_g += 8;
				tp_Ip1_r += 8;

				tp____p2 += 8;
				tp_Ip2_b += 8;
				tp_Ip2_g += 8;
				tp_Ip2_r += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* b__p0 = b[0].ptr<float>(i);
			float* ab_p0 = a_b[0].ptr<float>(i);
			float* ag_p0 = a_g[0].ptr<float>(i);
			float* ar_p0 = a_r[0].ptr<float>(i);

			float* b__p1 = b[1].ptr<float>(i);
			float* ab_p1 = a_b[1].ptr<float>(i);
			float* ag_p1 = a_g[1].ptr<float>(i);
			float* ar_p1 = a_r[1].ptr<float>(i);

			float* b__p2 = b[2].ptr<float>(i);
			float* ab_p2 = a_b[2].ptr<float>(i);
			float* ag_p2 = a_g[2].ptr<float>(i);
			float* ar_p2 = a_r[2].ptr<float>(i);

			tp_I_b = temp.ptr<float>(0, roffset);
			tp_I_g = temp.ptr<float>(1, roffset);
			tp_I_r = temp.ptr<float>(2, roffset);

			tp_I_bb = temp.ptr<float>(3, roffset);
			tp_I_bg = temp.ptr<float>(4, roffset);
			tp_I_br = temp.ptr<float>(5, roffset);
			tp_I_gg = temp.ptr<float>(6, roffset);
			tp_I_gr = temp.ptr<float>(7, roffset);
			tp_I_rr = temp.ptr<float>(8, roffset);

			tp____p0 = temp.ptr<float>(9, roffset);
			tp_Ip0_b = temp.ptr<float>(10, roffset);
			tp_Ip0_g = temp.ptr<float>(11, roffset);
			tp_Ip0_r = temp.ptr<float>(12, roffset);

			tp____p1 = temp.ptr<float>(13, roffset);
			tp_Ip1_b = temp.ptr<float>(14, roffset);
			tp_Ip1_g = temp.ptr<float>(15, roffset);
			tp_Ip1_r = temp.ptr<float>(16, roffset);

			tp____p2 = temp.ptr<float>(17, roffset);
			tp_Ip2_b = temp.ptr<float>(18, roffset);
			tp_Ip2_g = temp.ptr<float>(19, roffset);
			tp_Ip2_r = temp.ptr<float>(20, roffset);

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

				__m256 mSum_I_b = _mm256_loadu_ps(tp_I_b);
				__m256 mSum_I_g = _mm256_loadu_ps(tp_I_g);
				__m256 mSum_I_r = _mm256_loadu_ps(tp_I_r);

				__m256 mSum_I_bb = _mm256_loadu_ps(tp_I_bb);
				__m256 mSum_I_bg = _mm256_loadu_ps(tp_I_bg);
				__m256 mSum_I_br = _mm256_loadu_ps(tp_I_br);
				__m256 mSum_I_gg = _mm256_loadu_ps(tp_I_gg);
				__m256 mSum_I_gr = _mm256_loadu_ps(tp_I_gr);
				__m256 mSum_I_rr = _mm256_loadu_ps(tp_I_rr);

				__m256 mSum_p0 = _mm256_loadu_ps(tp____p0);
				__m256 mSum_p1 = _mm256_loadu_ps(tp____p1);
				__m256 mSum_p2 = _mm256_loadu_ps(tp____p2);

				__m256 mSum_Ip0_b = _mm256_loadu_ps(tp_Ip0_b);
				__m256 mSum_Ip0_g = _mm256_loadu_ps(tp_Ip0_g);
				__m256 mSum_Ip0_r = _mm256_loadu_ps(tp_Ip0_r);

				__m256 mSum_Ip1_b = _mm256_loadu_ps(tp_Ip1_b);
				__m256 mSum_Ip1_g = _mm256_loadu_ps(tp_Ip1_g);
				__m256 mSum_Ip1_r = _mm256_loadu_ps(tp_Ip1_r);

				__m256 mSum_Ip2_b = _mm256_loadu_ps(tp_Ip2_b);
				__m256 mSum_Ip2_g = _mm256_loadu_ps(tp_Ip2_g);
				__m256 mSum_Ip2_r = _mm256_loadu_ps(tp_Ip2_r);

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

					mSum_p0 = _mm256_add_ps(mSum_p0, _mm256_loadu_ps(tp____p0 + k));
					mSum_p1 = _mm256_add_ps(mSum_p1, _mm256_loadu_ps(tp____p1 + k));
					mSum_p2 = _mm256_add_ps(mSum_p2, _mm256_loadu_ps(tp____p2 + k));

					mSum_Ip0_b = _mm256_add_ps(mSum_Ip0_b, _mm256_loadu_ps(tp_Ip0_b + k));
					mSum_Ip0_g = _mm256_add_ps(mSum_Ip0_g, _mm256_loadu_ps(tp_Ip0_g + k));
					mSum_Ip0_r = _mm256_add_ps(mSum_Ip0_r, _mm256_loadu_ps(tp_Ip0_r + k));

					mSum_Ip1_b = _mm256_add_ps(mSum_Ip1_b, _mm256_loadu_ps(tp_Ip1_b + k));
					mSum_Ip1_g = _mm256_add_ps(mSum_Ip1_g, _mm256_loadu_ps(tp_Ip1_g + k));
					mSum_Ip1_r = _mm256_add_ps(mSum_Ip1_r, _mm256_loadu_ps(tp_Ip1_r + k));

					mSum_Ip2_b = _mm256_add_ps(mSum_Ip2_b, _mm256_loadu_ps(tp_Ip2_b + k));
					mSum_Ip2_g = _mm256_add_ps(mSum_Ip2_g, _mm256_loadu_ps(tp_Ip2_g + k));
					mSum_Ip2_r = _mm256_add_ps(mSum_Ip2_r, _mm256_loadu_ps(tp_Ip2_r + k));
				}

				const __m256 mb = _mm256_mul_ps(mSum_I_b, mDiv);
				const __m256 mg = _mm256_mul_ps(mSum_I_g, mDiv);
				const __m256 mr = _mm256_mul_ps(mSum_I_r, mDiv);

				const __m256 meps = _mm256_set1_ps(eps);
				const __m256 mBB = _mm256_fnmadd_ps(mb, mb, _mm256_fmadd_ps(mSum_I_bb, mDiv, meps));
				const __m256 mBG = _mm256_fnmadd_ps(mb, mg, _mm256_mul_ps(mSum_I_bg, mDiv));
				const __m256 mBR = _mm256_fnmadd_ps(mb, mr, _mm256_mul_ps(mSum_I_br, mDiv));
				const __m256 mGG = _mm256_fnmadd_ps(mg, mg, _mm256_fmadd_ps(mSum_I_gg, mDiv, meps));
				const __m256 mGR = _mm256_fnmadd_ps(mg, mr, _mm256_mul_ps(mSum_I_gr, mDiv));
				const __m256 mRR = _mm256_fnmadd_ps(mr, mr, _mm256_fmadd_ps(mSum_I_rr, mDiv, meps));

				__m256 mDet = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
				mDet = _mm256_add_ps(mDet, mDet);
				mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
				mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
				mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
				mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				const __m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				const __m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				const __m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				const __m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				const __m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				const __m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				//p0
				__m256 mp0 = _mm256_mul_ps(mSum_p0, mDiv);

				__m256 mCovB = _mm256_fnmadd_ps(mb, mp0, _mm256_mul_ps(mSum_Ip0_b, mDiv));
				__m256 mCovG = _mm256_fnmadd_ps(mg, mp0, _mm256_mul_ps(mSum_Ip0_g, mDiv));
				__m256 mCovR = _mm256_fnmadd_ps(mr, mp0, _mm256_mul_ps(mSum_Ip0_r, mDiv));

				__m256 mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p0, mTmp);
				ab_p0 += 8;
				__m256 mB = _mm256_fnmadd_ps(mTmp, mb, mp0);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p0, mTmp);
				ag_p0 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p0, mTmp);
				ar_p0 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p0, mB);
				b__p0 += 8;

				//p1
				__m256 mp1 = _mm256_mul_ps(mSum_p1, mDiv);

				mCovB = _mm256_fnmadd_ps(mb, mp1, _mm256_mul_ps(mSum_Ip1_b, mDiv));
				mCovG = _mm256_fnmadd_ps(mg, mp1, _mm256_mul_ps(mSum_Ip1_g, mDiv));
				mCovR = _mm256_fnmadd_ps(mr, mp1, _mm256_mul_ps(mSum_Ip1_r, mDiv));

				mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p1, mTmp);
				ab_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mb, mp1);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p1, mTmp);
				ag_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p1, mTmp);
				ar_p1 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p1, mB);
				b__p1 += 8;


				//p2
				__m256 mp2 = _mm256_mul_ps(mSum_p2, mDiv);

				mCovB = _mm256_fnmadd_ps(mb, mp2, _mm256_mul_ps(mSum_Ip2_b, mDiv));
				mCovG = _mm256_fnmadd_ps(mg, mp2, _mm256_mul_ps(mSum_Ip2_g, mDiv));
				mCovR = _mm256_fnmadd_ps(mr, mp2, _mm256_mul_ps(mSum_Ip2_r, mDiv));

				mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);

				_mm256_store_ps(ab_p2, mTmp);
				ab_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mb, mp2);

				mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ag_p2, mTmp);
				ag_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mg, mB);

				mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
				mTmp = _mm256_mul_ps(mTmp, mDet);
				_mm256_store_ps(ar_p2, mTmp);
				ar_p2 += 8;
				mB = _mm256_fnmadd_ps(mTmp, mr, mB);

				_mm256_store_ps(b__p2, mB);
				b__p2 += 8;


				tp_I_b += 8;
				tp_I_g += 8;
				tp_I_r += 8;

				tp_I_bb += 8;
				tp_I_bg += 8;
				tp_I_br += 8;
				tp_I_gg += 8;
				tp_I_gr += 8;
				tp_I_rr += 8;

				tp____p0 += 8;
				tp_Ip0_b += 8;
				tp_Ip0_g += 8;
				tp_Ip0_r += 8;

				tp____p1 += 8;
				tp_Ip1_b += 8;
				tp_Ip1_g += 8;
				tp_Ip1_r += 8;

				tp____p2 += 8;
				tp_Ip2_b += 8;
				tp_Ip2_g += 8;
				tp_Ip2_r += 8;
			}
		}
	}
}


void ab2q_Guide3Src3_sep_VHIShare_AVX(std::vector<cv::Mat>& a_b, std::vector<Mat>& a_g, std::vector<cv::Mat>& a_r, std::vector<cv::Mat>& b, std::vector<cv::Mat>& guide, const int r, std::vector<cv::Mat>& dest)
{
	const int width = a_b[0].cols;
	const int height = a_b[0].rows;

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat temp(Size(width + 2 * R, 12), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp_a_b0 = temp.ptr<float>(0, R);
		float* tp_a_g0 = temp.ptr<float>(1, R);
		float* tp_a_r0 = temp.ptr<float>(2, R);
		float* tp_b__0 = temp.ptr<float>(3, R);
		float* tp_a_b1 = temp.ptr<float>(4, R);
		float* tp_a_g1 = temp.ptr<float>(5, R);
		float* tp_a_r1 = temp.ptr<float>(6, R);
		float* tp_b__1 = temp.ptr<float>(7, R);
		float* tp_a_b2 = temp.ptr<float>(8, R);
		float* tp_a_g2 = temp.ptr<float>(9, R);
		float* tp_a_r2 = temp.ptr<float>(10, R);
		float* tp_b__2 = temp.ptr<float>(11, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* a_b0ptr = a_b[0].ptr<float>(i - r, j);
				float* a_g0ptr = a_g[0].ptr<float>(i - r, j);
				float* a_r0ptr = a_r[0].ptr<float>(i - r, j);
				float* b0ptr = b[0].ptr<float>(i - r, j);
				float* a_b1ptr = a_b[1].ptr<float>(i - r, j);
				float* a_g1ptr = a_g[1].ptr<float>(i - r, j);
				float* a_r1ptr = a_r[1].ptr<float>(i - r, j);
				float* b1ptr = b[1].ptr<float>(i - r, j);
				float* a_b2ptr = a_b[2].ptr<float>(i - r, j);
				float* a_g2ptr = a_g[2].ptr<float>(i - r, j);
				float* a_r2ptr = a_r[2].ptr<float>(i - r, j);
				float* b2ptr = b[2].ptr<float>(i - r, j);

				__m256 mSum_ab0 = _mm256_load_ps(a_b0ptr);
				__m256 mSum_ag0 = _mm256_load_ps(a_g0ptr);
				__m256 mSum_ar0 = _mm256_load_ps(a_r0ptr);
				__m256 mSum_b0 = _mm256_load_ps(b0ptr);

				__m256 mSum_ab1 = _mm256_load_ps(a_b1ptr);
				__m256 mSum_ag1 = _mm256_load_ps(a_g1ptr);
				__m256 mSum_ar1 = _mm256_load_ps(a_r1ptr);
				__m256 mSum_b1 = _mm256_load_ps(b1ptr);

				__m256 mSum_ab2 = _mm256_load_ps(a_b2ptr);
				__m256 mSum_ag2 = _mm256_load_ps(a_g2ptr);
				__m256 mSum_ar2 = _mm256_load_ps(a_r2ptr);
				__m256 mSum_b2 = _mm256_load_ps(b2ptr);

				a_b0ptr += width;
				a_g0ptr += width;
				a_r0ptr += width;
				b0ptr += width;
				a_b1ptr += width;
				a_g1ptr += width;
				a_r1ptr += width;
				b1ptr += width;
				a_b2ptr += width;
				a_g2ptr += width;
				a_r2ptr += width;
				b2ptr += width;

				for (int k = 0; k < d - 1; k++)
				{
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(a_b0ptr));
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(a_g0ptr));
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(a_r0ptr));
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(b0ptr));

					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(a_b1ptr));
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(a_g1ptr));
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(a_r1ptr));
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(b1ptr));

					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(a_b2ptr));
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(a_g2ptr));
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(a_r2ptr));
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(b2ptr));

					a_b0ptr += width;
					a_g0ptr += width;
					a_r0ptr += width;
					b0ptr += width;
					a_b1ptr += width;
					a_g1ptr += width;
					a_r1ptr += width;
					b1ptr += width;
					a_b2ptr += width;
					a_g2ptr += width;
					a_r2ptr += width;
					b2ptr += width;
				}

				_mm256_store_ps(tp_a_b0, mSum_ab0);
				_mm256_store_ps(tp_a_g0, mSum_ag0);
				_mm256_store_ps(tp_a_r0, mSum_ar0);
				_mm256_store_ps(tp_b__0, mSum_b0);

				_mm256_store_ps(tp_a_b1, mSum_ab1);
				_mm256_store_ps(tp_a_g1, mSum_ag1);
				_mm256_store_ps(tp_a_r1, mSum_ar1);
				_mm256_store_ps(tp_b__1, mSum_b1);

				_mm256_store_ps(tp_a_b2, mSum_ab2);
				_mm256_store_ps(tp_a_g2, mSum_ag2);
				_mm256_store_ps(tp_a_r2, mSum_ar2);
				_mm256_store_ps(tp_b__2, mSum_b2);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;

				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;

				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* guideptr_0 = guide[0].ptr<float>(i);
			float* guideptr_1 = guide[1].ptr<float>(i);
			float* guideptr_2 = guide[2].ptr<float>(i);
			float* dptr0 = dest[0].ptr<float>(i);
			float* dptr1 = dest[1].ptr<float>(i);
			float* dptr2 = dest[2].ptr<float>(i);

			tp_a_b0 = temp.ptr<float>(0, roffset);
			tp_a_g0 = temp.ptr<float>(1, roffset);
			tp_a_r0 = temp.ptr<float>(2, roffset);
			tp_b__0 = temp.ptr<float>(3, roffset);
			tp_a_b1 = temp.ptr<float>(4, roffset);
			tp_a_g1 = temp.ptr<float>(5, roffset);
			tp_a_r1 = temp.ptr<float>(6, roffset);
			tp_b__1 = temp.ptr<float>(7, roffset);
			tp_a_b2 = temp.ptr<float>(8, roffset);
			tp_a_g2 = temp.ptr<float>(9, roffset);
			tp_a_r2 = temp.ptr<float>(10, roffset);
			tp_b__2 = temp.ptr<float>(11, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b0 = _mm256_loadu_ps(tp_a_b0);
				__m256 mSum_a_g0 = _mm256_loadu_ps(tp_a_g0);
				__m256 mSum_a_r0 = _mm256_loadu_ps(tp_a_r0);
				__m256 mSum_b__0 = _mm256_loadu_ps(tp_b__0);

				__m256 mSum_a_b1 = _mm256_loadu_ps(tp_a_b1);
				__m256 mSum_a_g1 = _mm256_loadu_ps(tp_a_g1);
				__m256 mSum_a_r1 = _mm256_loadu_ps(tp_a_r1);
				__m256 mSum_b__1 = _mm256_loadu_ps(tp_b__1);

				__m256 mSum_a_b2 = _mm256_loadu_ps(tp_a_b2);
				__m256 mSum_a_g2 = _mm256_loadu_ps(tp_a_g2);
				__m256 mSum_a_r2 = _mm256_loadu_ps(tp_a_r2);
				__m256 mSum_b__2 = _mm256_loadu_ps(tp_b__2);

				for (int k = 1; k < d; k++)
				{
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 + k));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 + k));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 + k));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 + k));

					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 + k));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 + k));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 + k));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 + k));

					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 + k));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 + k));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 + k));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 + k));
				}

				__m256 v = _mm256_mul_ps(mSum_b__0, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r0, mDiv), v);
				_mm256_store_ps(dptr0, v);

				v = _mm256_mul_ps(mSum_b__1, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r1, mDiv), v);
				_mm256_store_ps(dptr1, v);

				v = _mm256_mul_ps(mSum_b__2, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r2, mDiv), v);
				_mm256_store_ps(dptr2, v);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;
				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;
				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
				guideptr_0 += 8;
				guideptr_1 += 8;
				guideptr_2 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_ab0 = _mm256_load_ps(a_b[0].ptr<float>(i, j));
				__m256 mSum_ag0 = _mm256_load_ps(a_g[0].ptr<float>(i, j));
				__m256 mSum_ar0 = _mm256_load_ps(a_r[0].ptr<float>(i, j));
				__m256 mSum_b0 = _mm256_load_ps(b[0].ptr<float>(i, j));

				__m256 mSum_ab1 = _mm256_load_ps(a_b[1].ptr<float>(i, j));
				__m256 mSum_ag1 = _mm256_load_ps(a_g[1].ptr<float>(i, j));
				__m256 mSum_ar1 = _mm256_load_ps(a_r[1].ptr<float>(i, j));
				__m256 mSum_b1 = _mm256_load_ps(b[1].ptr<float>(i, j));

				__m256 mSum_ab2 = _mm256_load_ps(a_b[2].ptr<float>(i, j));
				__m256 mSum_ag2 = _mm256_load_ps(a_g[2].ptr<float>(i, j));
				__m256 mSum_ar2 = _mm256_load_ps(a_r[2].ptr<float>(i, j));
				__m256 mSum_b2 = _mm256_load_ps(b[2].ptr<float>(i, j));
				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = a_b[0].ptr<float>(vl, j);
					float* sp2 = a_b[0].ptr<float>(vh, j);
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(sp1));
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(sp2));

					sp1 = a_g[0].ptr<float>(vl, j);
					sp2 = a_g[0].ptr<float>(vh, j);
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(sp1));
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(sp2));

					sp1 = a_r[0].ptr<float>(vl, j);
					sp2 = a_r[0].ptr<float>(vh, j);
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(sp1));
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(sp2));

					sp1 = b[0].ptr<float>(vl, j);
					sp2 = b[0].ptr<float>(vh, j);
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(sp1));
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(sp2));

					sp1 = a_b[1].ptr<float>(vl, j);
					sp2 = a_b[1].ptr<float>(vh, j);
					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(sp1));
					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(sp2));

					sp1 = a_g[1].ptr<float>(vl, j);
					sp2 = a_g[1].ptr<float>(vh, j);
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(sp1));
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(sp2));

					sp1 = a_r[1].ptr<float>(vl, j);
					sp2 = a_r[1].ptr<float>(vh, j);
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(sp1));
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(sp2));

					sp1 = b[1].ptr<float>(vl, j);
					sp2 = b[1].ptr<float>(vh, j);
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(sp1));
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(sp2));

					sp1 = a_b[2].ptr<float>(vl, j);
					sp2 = a_b[2].ptr<float>(vh, j);
					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(sp1));
					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(sp2));

					sp1 = a_g[2].ptr<float>(vl, j);
					sp2 = a_g[2].ptr<float>(vh, j);
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(sp1));
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(sp2));

					sp1 = a_r[2].ptr<float>(vl, j);
					sp2 = a_r[2].ptr<float>(vh, j);
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(sp1));
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(sp2));

					sp1 = b[2].ptr<float>(vl, j);
					sp2 = b[2].ptr<float>(vh, j);
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(sp1));
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(sp2));
				}

				_mm256_store_ps(tp_a_b0, mSum_ab0);
				_mm256_store_ps(tp_a_g0, mSum_ag0);
				_mm256_store_ps(tp_a_r0, mSum_ar0);
				_mm256_store_ps(tp_b__0, mSum_b0);

				_mm256_store_ps(tp_a_b1, mSum_ab1);
				_mm256_store_ps(tp_a_g1, mSum_ag1);
				_mm256_store_ps(tp_a_r1, mSum_ar1);
				_mm256_store_ps(tp_b__1, mSum_b1);

				_mm256_store_ps(tp_a_b2, mSum_ab2);
				_mm256_store_ps(tp_a_g2, mSum_ag2);
				_mm256_store_ps(tp_a_r2, mSum_ar2);
				_mm256_store_ps(tp_b__2, mSum_b2);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;

				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;

				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* guideptr_0 = guide[0].ptr<float>(i);
			float* guideptr_1 = guide[1].ptr<float>(i);
			float* guideptr_2 = guide[2].ptr<float>(i);
			float* dptr0 = dest[0].ptr<float>(i);
			float* dptr1 = dest[1].ptr<float>(i);
			float* dptr2 = dest[2].ptr<float>(i);

			tp_a_b0 = temp.ptr<float>(0, roffset);
			tp_a_g0 = temp.ptr<float>(1, roffset);
			tp_a_r0 = temp.ptr<float>(2, roffset);
			tp_b__0 = temp.ptr<float>(3, roffset);
			tp_a_b1 = temp.ptr<float>(4, roffset);
			tp_a_g1 = temp.ptr<float>(5, roffset);
			tp_a_r1 = temp.ptr<float>(6, roffset);
			tp_b__1 = temp.ptr<float>(7, roffset);
			tp_a_b2 = temp.ptr<float>(8, roffset);
			tp_a_g2 = temp.ptr<float>(9, roffset);
			tp_a_r2 = temp.ptr<float>(10, roffset);
			tp_b__2 = temp.ptr<float>(11, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b0 = _mm256_loadu_ps(tp_a_b0 + r);
				__m256 mSum_a_g0 = _mm256_loadu_ps(tp_a_g0 + r);
				__m256 mSum_a_r0 = _mm256_loadu_ps(tp_a_r0 + r);
				__m256 mSum_b__0 = _mm256_loadu_ps(tp_b__0 + r);

				__m256 mSum_a_b1 = _mm256_loadu_ps(tp_a_b1 + r);
				__m256 mSum_a_g1 = _mm256_loadu_ps(tp_a_g1 + r);
				__m256 mSum_a_r1 = _mm256_loadu_ps(tp_a_r1 + r);
				__m256 mSum_b__1 = _mm256_loadu_ps(tp_b__1 + r);

				__m256 mSum_a_b2 = _mm256_loadu_ps(tp_a_b2 + r);
				__m256 mSum_a_g2 = _mm256_loadu_ps(tp_a_g2 + r);
				__m256 mSum_a_r2 = _mm256_loadu_ps(tp_a_r2 + r);
				__m256 mSum_b__2 = _mm256_loadu_ps(tp_b__2 + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 - k + r));
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 + k + r));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 - k + r));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 + k + r));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 - k + r));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 + k + r));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 - k + r));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 + k + r));

					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 - k + r));
					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 + k + r));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 - k + r));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 + k + r));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 - k + r));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 + k + r));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 - k + r));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 + k + r));

					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 - k + r));
					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 + k + r));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 - k + r));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 + k + r));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 - k + r));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 + k + r));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 - k + r));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b__0, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r0, mDiv), v);
				_mm256_store_ps(dptr0, v);

				v = _mm256_mul_ps(mSum_b__1, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r1, mDiv), v);
				_mm256_store_ps(dptr1, v);

				v = _mm256_mul_ps(mSum_b__2, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r2, mDiv), v);
				_mm256_store_ps(dptr2, v);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;
				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;
				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
				guideptr_0 += 8;
				guideptr_1 += 8;
				guideptr_2 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
	}
}

void ab2q_Guide3Src3_sep_VHIShare_AVX_omp(std::vector<cv::Mat>& a_b, std::vector<Mat>& a_g, std::vector<cv::Mat>& a_r, std::vector<cv::Mat>& b, std::vector<cv::Mat>& guide, const int r, std::vector<cv::Mat>& dest)
{
	const int width = a_b[0].cols;
	const int height = a_b[0].rows;

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	const int d = 2 * r + 1;
	__m256 mDiv = _mm256_set1_ps(1.f / (d*d));

	Mat buff(Size(width + 2 * R, 12 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 12 * omp_get_thread_num(), width + 2 * R, 12));

		float* tp_a_b0 = temp.ptr<float>(0, R);
		float* tp_a_g0 = temp.ptr<float>(1, R);
		float* tp_a_r0 = temp.ptr<float>(2, R);
		float* tp_b__0 = temp.ptr<float>(3, R);
		float* tp_a_b1 = temp.ptr<float>(4, R);
		float* tp_a_g1 = temp.ptr<float>(5, R);
		float* tp_a_r1 = temp.ptr<float>(6, R);
		float* tp_b__1 = temp.ptr<float>(7, R);
		float* tp_a_b2 = temp.ptr<float>(8, R);
		float* tp_a_g2 = temp.ptr<float>(9, R);
		float* tp_a_r2 = temp.ptr<float>(10, R);
		float* tp_b__2 = temp.ptr<float>(11, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* a_b0ptr = a_b[0].ptr<float>(i - r, j);
				float* a_g0ptr = a_g[0].ptr<float>(i - r, j);
				float* a_r0ptr = a_r[0].ptr<float>(i - r, j);
				float* b0ptr = b[0].ptr<float>(i - r, j);
				float* a_b1ptr = a_b[1].ptr<float>(i - r, j);
				float* a_g1ptr = a_g[1].ptr<float>(i - r, j);
				float* a_r1ptr = a_r[1].ptr<float>(i - r, j);
				float* b1ptr = b[1].ptr<float>(i - r, j);
				float* a_b2ptr = a_b[2].ptr<float>(i - r, j);
				float* a_g2ptr = a_g[2].ptr<float>(i - r, j);
				float* a_r2ptr = a_r[2].ptr<float>(i - r, j);
				float* b2ptr = b[2].ptr<float>(i - r, j);

				__m256 mSum_ab0 = _mm256_load_ps(a_b0ptr);
				__m256 mSum_ag0 = _mm256_load_ps(a_g0ptr);
				__m256 mSum_ar0 = _mm256_load_ps(a_r0ptr);
				__m256 mSum_b0 = _mm256_load_ps(b0ptr);

				__m256 mSum_ab1 = _mm256_load_ps(a_b1ptr);
				__m256 mSum_ag1 = _mm256_load_ps(a_g1ptr);
				__m256 mSum_ar1 = _mm256_load_ps(a_r1ptr);
				__m256 mSum_b1 = _mm256_load_ps(b1ptr);

				__m256 mSum_ab2 = _mm256_load_ps(a_b2ptr);
				__m256 mSum_ag2 = _mm256_load_ps(a_g2ptr);
				__m256 mSum_ar2 = _mm256_load_ps(a_r2ptr);
				__m256 mSum_b2 = _mm256_load_ps(b2ptr);

				a_b0ptr += width;
				a_g0ptr += width;
				a_r0ptr += width;
				b0ptr += width;
				a_b1ptr += width;
				a_g1ptr += width;
				a_r1ptr += width;
				b1ptr += width;
				a_b2ptr += width;
				a_g2ptr += width;
				a_r2ptr += width;
				b2ptr += width;

				for (int k = 0; k < d - 1; k++)
				{
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(a_b0ptr));
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(a_g0ptr));
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(a_r0ptr));
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(b0ptr));

					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(a_b1ptr));
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(a_g1ptr));
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(a_r1ptr));
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(b1ptr));

					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(a_b2ptr));
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(a_g2ptr));
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(a_r2ptr));
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(b2ptr));

					a_b0ptr += width;
					a_g0ptr += width;
					a_r0ptr += width;
					b0ptr += width;
					a_b1ptr += width;
					a_g1ptr += width;
					a_r1ptr += width;
					b1ptr += width;
					a_b2ptr += width;
					a_g2ptr += width;
					a_r2ptr += width;
					b2ptr += width;
				}

				_mm256_store_ps(tp_a_b0, mSum_ab0);
				_mm256_store_ps(tp_a_g0, mSum_ag0);
				_mm256_store_ps(tp_a_r0, mSum_ar0);
				_mm256_store_ps(tp_b__0, mSum_b0);

				_mm256_store_ps(tp_a_b1, mSum_ab1);
				_mm256_store_ps(tp_a_g1, mSum_ag1);
				_mm256_store_ps(tp_a_r1, mSum_ar1);
				_mm256_store_ps(tp_b__1, mSum_b1);

				_mm256_store_ps(tp_a_b2, mSum_ab2);
				_mm256_store_ps(tp_a_g2, mSum_ag2);
				_mm256_store_ps(tp_a_r2, mSum_ar2);
				_mm256_store_ps(tp_b__2, mSum_b2);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;

				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;

				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* guideptr_0 = guide[0].ptr<float>(i);
			float* guideptr_1 = guide[1].ptr<float>(i);
			float* guideptr_2 = guide[2].ptr<float>(i);
			float* dptr0 = dest[0].ptr<float>(i);
			float* dptr1 = dest[1].ptr<float>(i);
			float* dptr2 = dest[2].ptr<float>(i);

			tp_a_b0 = temp.ptr<float>(0, roffset);
			tp_a_g0 = temp.ptr<float>(1, roffset);
			tp_a_r0 = temp.ptr<float>(2, roffset);
			tp_b__0 = temp.ptr<float>(3, roffset);
			tp_a_b1 = temp.ptr<float>(4, roffset);
			tp_a_g1 = temp.ptr<float>(5, roffset);
			tp_a_r1 = temp.ptr<float>(6, roffset);
			tp_b__1 = temp.ptr<float>(7, roffset);
			tp_a_b2 = temp.ptr<float>(8, roffset);
			tp_a_g2 = temp.ptr<float>(9, roffset);
			tp_a_r2 = temp.ptr<float>(10, roffset);
			tp_b__2 = temp.ptr<float>(11, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b0 = _mm256_loadu_ps(tp_a_b0);
				__m256 mSum_a_g0 = _mm256_loadu_ps(tp_a_g0);
				__m256 mSum_a_r0 = _mm256_loadu_ps(tp_a_r0);
				__m256 mSum_b__0 = _mm256_loadu_ps(tp_b__0);

				__m256 mSum_a_b1 = _mm256_loadu_ps(tp_a_b1);
				__m256 mSum_a_g1 = _mm256_loadu_ps(tp_a_g1);
				__m256 mSum_a_r1 = _mm256_loadu_ps(tp_a_r1);
				__m256 mSum_b__1 = _mm256_loadu_ps(tp_b__1);

				__m256 mSum_a_b2 = _mm256_loadu_ps(tp_a_b2);
				__m256 mSum_a_g2 = _mm256_loadu_ps(tp_a_g2);
				__m256 mSum_a_r2 = _mm256_loadu_ps(tp_a_r2);
				__m256 mSum_b__2 = _mm256_loadu_ps(tp_b__2);

				for (int k = 1; k < d; k++)
				{
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 + k));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 + k));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 + k));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 + k));

					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 + k));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 + k));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 + k));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 + k));

					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 + k));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 + k));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 + k));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 + k));
				}

				__m256 v = _mm256_mul_ps(mSum_b__0, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r0, mDiv), v);
				_mm256_store_ps(dptr0, v);

				v = _mm256_mul_ps(mSum_b__1, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r1, mDiv), v);
				_mm256_store_ps(dptr1, v);

				v = _mm256_mul_ps(mSum_b__2, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r2, mDiv), v);
				_mm256_store_ps(dptr2, v);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;
				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;
				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
				guideptr_0 += 8;
				guideptr_1 += 8;
				guideptr_2 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_ab0 = _mm256_load_ps(a_b[0].ptr<float>(i, j));
				__m256 mSum_ag0 = _mm256_load_ps(a_g[0].ptr<float>(i, j));
				__m256 mSum_ar0 = _mm256_load_ps(a_r[0].ptr<float>(i, j));
				__m256 mSum_b0 = _mm256_load_ps(b[0].ptr<float>(i, j));

				__m256 mSum_ab1 = _mm256_load_ps(a_b[1].ptr<float>(i, j));
				__m256 mSum_ag1 = _mm256_load_ps(a_g[1].ptr<float>(i, j));
				__m256 mSum_ar1 = _mm256_load_ps(a_r[1].ptr<float>(i, j));
				__m256 mSum_b1 = _mm256_load_ps(b[1].ptr<float>(i, j));

				__m256 mSum_ab2 = _mm256_load_ps(a_b[2].ptr<float>(i, j));
				__m256 mSum_ag2 = _mm256_load_ps(a_g[2].ptr<float>(i, j));
				__m256 mSum_ar2 = _mm256_load_ps(a_r[2].ptr<float>(i, j));
				__m256 mSum_b2 = _mm256_load_ps(b[2].ptr<float>(i, j));
				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = a_b[0].ptr<float>(vl, j);
					float* sp2 = a_b[0].ptr<float>(vh, j);
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(sp1));
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(sp2));

					sp1 = a_g[0].ptr<float>(vl, j);
					sp2 = a_g[0].ptr<float>(vh, j);
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(sp1));
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(sp2));

					sp1 = a_r[0].ptr<float>(vl, j);
					sp2 = a_r[0].ptr<float>(vh, j);
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(sp1));
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(sp2));

					sp1 = b[0].ptr<float>(vl, j);
					sp2 = b[0].ptr<float>(vh, j);
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(sp1));
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(sp2));

					sp1 = a_b[1].ptr<float>(vl, j);
					sp2 = a_b[1].ptr<float>(vh, j);
					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(sp1));
					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(sp2));

					sp1 = a_g[1].ptr<float>(vl, j);
					sp2 = a_g[1].ptr<float>(vh, j);
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(sp1));
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(sp2));

					sp1 = a_r[1].ptr<float>(vl, j);
					sp2 = a_r[1].ptr<float>(vh, j);
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(sp1));
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(sp2));

					sp1 = b[1].ptr<float>(vl, j);
					sp2 = b[1].ptr<float>(vh, j);
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(sp1));
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(sp2));

					sp1 = a_b[2].ptr<float>(vl, j);
					sp2 = a_b[2].ptr<float>(vh, j);
					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(sp1));
					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(sp2));

					sp1 = a_g[2].ptr<float>(vl, j);
					sp2 = a_g[2].ptr<float>(vh, j);
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(sp1));
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(sp2));

					sp1 = a_r[2].ptr<float>(vl, j);
					sp2 = a_r[2].ptr<float>(vh, j);
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(sp1));
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(sp2));

					sp1 = b[2].ptr<float>(vl, j);
					sp2 = b[2].ptr<float>(vh, j);
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(sp1));
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(sp2));
				}

				_mm256_store_ps(tp_a_b0, mSum_ab0);
				_mm256_store_ps(tp_a_g0, mSum_ag0);
				_mm256_store_ps(tp_a_r0, mSum_ar0);
				_mm256_store_ps(tp_b__0, mSum_b0);

				_mm256_store_ps(tp_a_b1, mSum_ab1);
				_mm256_store_ps(tp_a_g1, mSum_ag1);
				_mm256_store_ps(tp_a_r1, mSum_ar1);
				_mm256_store_ps(tp_b__1, mSum_b1);

				_mm256_store_ps(tp_a_b2, mSum_ab2);
				_mm256_store_ps(tp_a_g2, mSum_ag2);
				_mm256_store_ps(tp_a_r2, mSum_ar2);
				_mm256_store_ps(tp_b__2, mSum_b2);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;

				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;

				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* guideptr_0 = guide[0].ptr<float>(i);
			float* guideptr_1 = guide[1].ptr<float>(i);
			float* guideptr_2 = guide[2].ptr<float>(i);
			float* dptr0 = dest[0].ptr<float>(i);
			float* dptr1 = dest[1].ptr<float>(i);
			float* dptr2 = dest[2].ptr<float>(i);

			tp_a_b0 = temp.ptr<float>(0, roffset);
			tp_a_g0 = temp.ptr<float>(1, roffset);
			tp_a_r0 = temp.ptr<float>(2, roffset);
			tp_b__0 = temp.ptr<float>(3, roffset);
			tp_a_b1 = temp.ptr<float>(4, roffset);
			tp_a_g1 = temp.ptr<float>(5, roffset);
			tp_a_r1 = temp.ptr<float>(6, roffset);
			tp_b__1 = temp.ptr<float>(7, roffset);
			tp_a_b2 = temp.ptr<float>(8, roffset);
			tp_a_g2 = temp.ptr<float>(9, roffset);
			tp_a_r2 = temp.ptr<float>(10, roffset);
			tp_b__2 = temp.ptr<float>(11, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b0 = _mm256_loadu_ps(tp_a_b0 + r);
				__m256 mSum_a_g0 = _mm256_loadu_ps(tp_a_g0 + r);
				__m256 mSum_a_r0 = _mm256_loadu_ps(tp_a_r0 + r);
				__m256 mSum_b__0 = _mm256_loadu_ps(tp_b__0 + r);

				__m256 mSum_a_b1 = _mm256_loadu_ps(tp_a_b1 + r);
				__m256 mSum_a_g1 = _mm256_loadu_ps(tp_a_g1 + r);
				__m256 mSum_a_r1 = _mm256_loadu_ps(tp_a_r1 + r);
				__m256 mSum_b__1 = _mm256_loadu_ps(tp_b__1 + r);

				__m256 mSum_a_b2 = _mm256_loadu_ps(tp_a_b2 + r);
				__m256 mSum_a_g2 = _mm256_loadu_ps(tp_a_g2 + r);
				__m256 mSum_a_r2 = _mm256_loadu_ps(tp_a_r2 + r);
				__m256 mSum_b__2 = _mm256_loadu_ps(tp_b__2 + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 - k + r));
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 + k + r));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 - k + r));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 + k + r));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 - k + r));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 + k + r));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 - k + r));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 + k + r));

					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 - k + r));
					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 + k + r));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 - k + r));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 + k + r));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 - k + r));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 + k + r));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 - k + r));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 + k + r));

					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 - k + r));
					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 + k + r));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 - k + r));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 + k + r));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 - k + r));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 + k + r));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 - k + r));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b__0, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r0, mDiv), v);
				_mm256_store_ps(dptr0, v);

				v = _mm256_mul_ps(mSum_b__1, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r1, mDiv), v);
				_mm256_store_ps(dptr1, v);

				v = _mm256_mul_ps(mSum_b__2, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r2, mDiv), v);
				_mm256_store_ps(dptr2, v);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;
				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;
				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
				guideptr_0 += 8;
				guideptr_1 += 8;
				guideptr_2 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
	}
}


void ab2q_Guide3Src3_sep_VHIShare_Unroll2_AVX(std::vector<cv::Mat>& a_b, std::vector<Mat>& a_g, std::vector<cv::Mat>& a_r, std::vector<cv::Mat>& b, std::vector<cv::Mat>& guide, const int r, std::vector<cv::Mat>& dest)
{
	const int width = a_b[0].cols;
	const int height = a_b[0].rows;

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	__m256 mDiv = _mm256_set1_ps(1.f / ((2 * r + 1)*(2 * r + 1)));

	Mat temp(Size(width + 2 * R, 12), CV_32FC1);

	for (int i = 0; i < height; i++)
	{
		float* tp_a_b0 = temp.ptr<float>(0, R);
		float* tp_a_g0 = temp.ptr<float>(1, R);
		float* tp_a_r0 = temp.ptr<float>(2, R);
		float* tp_b__0 = temp.ptr<float>(3, R);
		float* tp_a_b1 = temp.ptr<float>(4, R);
		float* tp_a_g1 = temp.ptr<float>(5, R);
		float* tp_a_r1 = temp.ptr<float>(6, R);
		float* tp_b__1 = temp.ptr<float>(7, R);
		float* tp_a_b2 = temp.ptr<float>(8, R);
		float* tp_a_g2 = temp.ptr<float>(9, R);
		float* tp_a_r2 = temp.ptr<float>(10, R);
		float* tp_b__2 = temp.ptr<float>(11, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* a_b0ptr = a_b[0].ptr<float>(i - r, j);
				float* a_g0ptr = a_g[0].ptr<float>(i - r, j);
				float* a_r0ptr = a_r[0].ptr<float>(i - r, j);
				float* b0ptr = b[0].ptr<float>(i - r, j);
				float* a_b1ptr = a_b[1].ptr<float>(i - r, j);
				float* a_g1ptr = a_g[1].ptr<float>(i - r, j);
				float* a_r1ptr = a_r[1].ptr<float>(i - r, j);
				float* b1ptr = b[1].ptr<float>(i - r, j);
				float* a_b2ptr = a_b[2].ptr<float>(i - r, j);
				float* a_g2ptr = a_g[2].ptr<float>(i - r, j);
				float* a_r2ptr = a_r[2].ptr<float>(i - r, j);
				float* b2ptr = b[2].ptr<float>(i - r, j);

				__m256 mSum_ab0 = _mm256_load_ps(a_b0ptr);
				__m256 mSum_ag0 = _mm256_load_ps(a_g0ptr);
				__m256 mSum_ar0 = _mm256_load_ps(a_r0ptr);
				__m256 mSum_b0 = _mm256_load_ps(b0ptr);

				__m256 mSum_ab1 = _mm256_load_ps(a_b1ptr);
				__m256 mSum_ag1 = _mm256_load_ps(a_g1ptr);
				__m256 mSum_ar1 = _mm256_load_ps(a_r1ptr);
				__m256 mSum_b1 = _mm256_load_ps(b1ptr);

				__m256 mSum_ab2 = _mm256_load_ps(a_b2ptr);
				__m256 mSum_ag2 = _mm256_load_ps(a_g2ptr);
				__m256 mSum_ar2 = _mm256_load_ps(a_r2ptr);
				__m256 mSum_b2 = _mm256_load_ps(b2ptr);

				a_b0ptr += width;
				a_g0ptr += width;
				a_r0ptr += width;
				b0ptr += width;
				a_b1ptr += width;
				a_g1ptr += width;
				a_r1ptr += width;
				b1ptr += width;
				a_b2ptr += width;
				a_g2ptr += width;
				a_r2ptr += width;
				b2ptr += width;

				const int step = 2 * width;
				for (int k = 0; k < r; k++)
				{
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(a_b0ptr));
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(a_b0ptr + width));
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(a_g0ptr));
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(a_g0ptr + width));
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(a_r0ptr));
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(a_r0ptr + width));
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(b0ptr));
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(b0ptr + width));

					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(a_b1ptr));
					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(a_b1ptr + width));
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(a_g1ptr));
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(a_g1ptr + width));
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(a_r1ptr));
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(a_r1ptr + width));
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(b1ptr));
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(b1ptr + width));

					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(a_b2ptr));
					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(a_b2ptr + width));
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(a_g2ptr));
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(a_g2ptr + width));
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(a_r2ptr));
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(a_r2ptr + width));
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(b2ptr));
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(b2ptr + width));

					a_b0ptr += step;
					a_g0ptr += step;
					a_r0ptr += step;
					b0ptr += step;
					a_b1ptr += step;
					a_g1ptr += step;
					a_r1ptr += step;
					b1ptr += step;
					a_b2ptr += step;
					a_g2ptr += step;
					a_r2ptr += step;
					b2ptr += step;
				}

				_mm256_store_ps(tp_a_b0, mSum_ab0);
				_mm256_store_ps(tp_a_g0, mSum_ag0);
				_mm256_store_ps(tp_a_r0, mSum_ar0);
				_mm256_store_ps(tp_b__0, mSum_b0);

				_mm256_store_ps(tp_a_b1, mSum_ab1);
				_mm256_store_ps(tp_a_g1, mSum_ag1);
				_mm256_store_ps(tp_a_r1, mSum_ar1);
				_mm256_store_ps(tp_b__1, mSum_b1);

				_mm256_store_ps(tp_a_b2, mSum_ab2);
				_mm256_store_ps(tp_a_g2, mSum_ag2);
				_mm256_store_ps(tp_a_r2, mSum_ar2);
				_mm256_store_ps(tp_b__2, mSum_b2);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;

				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;

				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* guideptr_0 = guide[0].ptr<float>(i);
			float* guideptr_1 = guide[1].ptr<float>(i);
			float* guideptr_2 = guide[2].ptr<float>(i);
			float* dptr0 = dest[0].ptr<float>(i);
			float* dptr1 = dest[1].ptr<float>(i);
			float* dptr2 = dest[2].ptr<float>(i);

			tp_a_b0 = temp.ptr<float>(0, roffset);
			tp_a_g0 = temp.ptr<float>(1, roffset);
			tp_a_r0 = temp.ptr<float>(2, roffset);
			tp_b__0 = temp.ptr<float>(3, roffset);
			tp_a_b1 = temp.ptr<float>(4, roffset);
			tp_a_g1 = temp.ptr<float>(5, roffset);
			tp_a_r1 = temp.ptr<float>(6, roffset);
			tp_b__1 = temp.ptr<float>(7, roffset);
			tp_a_b2 = temp.ptr<float>(8, roffset);
			tp_a_g2 = temp.ptr<float>(9, roffset);
			tp_a_r2 = temp.ptr<float>(10, roffset);
			tp_b__2 = temp.ptr<float>(11, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b0 = _mm256_loadu_ps(tp_a_b0);
				__m256 mSum_a_g0 = _mm256_loadu_ps(tp_a_g0);
				__m256 mSum_a_r0 = _mm256_loadu_ps(tp_a_r0);
				__m256 mSum_b__0 = _mm256_loadu_ps(tp_b__0);

				__m256 mSum_a_b1 = _mm256_loadu_ps(tp_a_b1);
				__m256 mSum_a_g1 = _mm256_loadu_ps(tp_a_g1);
				__m256 mSum_a_r1 = _mm256_loadu_ps(tp_a_r1);
				__m256 mSum_b__1 = _mm256_loadu_ps(tp_b__1);

				__m256 mSum_a_b2 = _mm256_loadu_ps(tp_a_b2);
				__m256 mSum_a_g2 = _mm256_loadu_ps(tp_a_g2);
				__m256 mSum_a_r2 = _mm256_loadu_ps(tp_a_r2);
				__m256 mSum_b__2 = _mm256_loadu_ps(tp_b__2);

				for (int k = 1; k <= r; k++)
				{
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 + k));
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 + k + r));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 + k));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 + k + r));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 + k));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 + k + r));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 + k));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 + k + r));

					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 + k));
					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 + k + r));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 + k));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 + k + r));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 + k));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 + k + r));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 + k));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 + k + r));

					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 + k));
					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 + k + r));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 + k));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 + k + r));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 + k));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 + k + r));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 + k));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b__0, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r0, mDiv), v);
				_mm256_store_ps(dptr0, v);

				v = _mm256_mul_ps(mSum_b__1, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r1, mDiv), v);
				_mm256_store_ps(dptr1, v);

				v = _mm256_mul_ps(mSum_b__2, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r2, mDiv), v);
				_mm256_store_ps(dptr2, v);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;
				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;
				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
				guideptr_0 += 8;
				guideptr_1 += 8;
				guideptr_2 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_ab0 = _mm256_load_ps(a_b[0].ptr<float>(i, j));
				__m256 mSum_ag0 = _mm256_load_ps(a_g[0].ptr<float>(i, j));
				__m256 mSum_ar0 = _mm256_load_ps(a_r[0].ptr<float>(i, j));
				__m256 mSum_b0 = _mm256_load_ps(b[0].ptr<float>(i, j));

				__m256 mSum_ab1 = _mm256_load_ps(a_b[1].ptr<float>(i, j));
				__m256 mSum_ag1 = _mm256_load_ps(a_g[1].ptr<float>(i, j));
				__m256 mSum_ar1 = _mm256_load_ps(a_r[1].ptr<float>(i, j));
				__m256 mSum_b1 = _mm256_load_ps(b[1].ptr<float>(i, j));

				__m256 mSum_ab2 = _mm256_load_ps(a_b[2].ptr<float>(i, j));
				__m256 mSum_ag2 = _mm256_load_ps(a_g[2].ptr<float>(i, j));
				__m256 mSum_ar2 = _mm256_load_ps(a_r[2].ptr<float>(i, j));
				__m256 mSum_b2 = _mm256_load_ps(b[2].ptr<float>(i, j));
				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = a_b[0].ptr<float>(vl, j);
					float* sp2 = a_b[0].ptr<float>(vh, j);
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(sp1));
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(sp2));

					sp1 = a_g[0].ptr<float>(vl, j);
					sp2 = a_g[0].ptr<float>(vh, j);
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(sp1));
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(sp2));

					sp1 = a_r[0].ptr<float>(vl, j);
					sp2 = a_r[0].ptr<float>(vh, j);
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(sp1));
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(sp2));

					sp1 = b[0].ptr<float>(vl, j);
					sp2 = b[0].ptr<float>(vh, j);
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(sp1));
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(sp2));

					sp1 = a_b[1].ptr<float>(vl, j);
					sp2 = a_b[1].ptr<float>(vh, j);
					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(sp1));
					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(sp2));

					sp1 = a_g[1].ptr<float>(vl, j);
					sp2 = a_g[1].ptr<float>(vh, j);
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(sp1));
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(sp2));

					sp1 = a_r[1].ptr<float>(vl, j);
					sp2 = a_r[1].ptr<float>(vh, j);
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(sp1));
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(sp2));

					sp1 = b[1].ptr<float>(vl, j);
					sp2 = b[1].ptr<float>(vh, j);
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(sp1));
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(sp2));

					sp1 = a_b[2].ptr<float>(vl, j);
					sp2 = a_b[2].ptr<float>(vh, j);
					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(sp1));
					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(sp2));

					sp1 = a_g[2].ptr<float>(vl, j);
					sp2 = a_g[2].ptr<float>(vh, j);
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(sp1));
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(sp2));

					sp1 = a_r[2].ptr<float>(vl, j);
					sp2 = a_r[2].ptr<float>(vh, j);
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(sp1));
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(sp2));

					sp1 = b[2].ptr<float>(vl, j);
					sp2 = b[2].ptr<float>(vh, j);
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(sp1));
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(sp2));
				}

				_mm256_store_ps(tp_a_b0, mSum_ab0);
				_mm256_store_ps(tp_a_g0, mSum_ag0);
				_mm256_store_ps(tp_a_r0, mSum_ar0);
				_mm256_store_ps(tp_b__0, mSum_b0);

				_mm256_store_ps(tp_a_b1, mSum_ab1);
				_mm256_store_ps(tp_a_g1, mSum_ag1);
				_mm256_store_ps(tp_a_r1, mSum_ar1);
				_mm256_store_ps(tp_b__1, mSum_b1);

				_mm256_store_ps(tp_a_b2, mSum_ab2);
				_mm256_store_ps(tp_a_g2, mSum_ag2);
				_mm256_store_ps(tp_a_r2, mSum_ar2);
				_mm256_store_ps(tp_b__2, mSum_b2);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;

				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;

				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* guideptr_0 = guide[0].ptr<float>(i);
			float* guideptr_1 = guide[1].ptr<float>(i);
			float* guideptr_2 = guide[2].ptr<float>(i);
			float* dptr0 = dest[0].ptr<float>(i);
			float* dptr1 = dest[1].ptr<float>(i);
			float* dptr2 = dest[2].ptr<float>(i);

			tp_a_b0 = temp.ptr<float>(0, roffset);
			tp_a_g0 = temp.ptr<float>(1, roffset);
			tp_a_r0 = temp.ptr<float>(2, roffset);
			tp_b__0 = temp.ptr<float>(3, roffset);
			tp_a_b1 = temp.ptr<float>(4, roffset);
			tp_a_g1 = temp.ptr<float>(5, roffset);
			tp_a_r1 = temp.ptr<float>(6, roffset);
			tp_b__1 = temp.ptr<float>(7, roffset);
			tp_a_b2 = temp.ptr<float>(8, roffset);
			tp_a_g2 = temp.ptr<float>(9, roffset);
			tp_a_r2 = temp.ptr<float>(10, roffset);
			tp_b__2 = temp.ptr<float>(11, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b0 = _mm256_loadu_ps(tp_a_b0 + r);
				__m256 mSum_a_g0 = _mm256_loadu_ps(tp_a_g0 + r);
				__m256 mSum_a_r0 = _mm256_loadu_ps(tp_a_r0 + r);
				__m256 mSum_b__0 = _mm256_loadu_ps(tp_b__0 + r);

				__m256 mSum_a_b1 = _mm256_loadu_ps(tp_a_b1 + r);
				__m256 mSum_a_g1 = _mm256_loadu_ps(tp_a_g1 + r);
				__m256 mSum_a_r1 = _mm256_loadu_ps(tp_a_r1 + r);
				__m256 mSum_b__1 = _mm256_loadu_ps(tp_b__1 + r);

				__m256 mSum_a_b2 = _mm256_loadu_ps(tp_a_b2 + r);
				__m256 mSum_a_g2 = _mm256_loadu_ps(tp_a_g2 + r);
				__m256 mSum_a_r2 = _mm256_loadu_ps(tp_a_r2 + r);
				__m256 mSum_b__2 = _mm256_loadu_ps(tp_b__2 + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 - k + r));
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 + k + r));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 - k + r));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 + k + r));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 - k + r));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 + k + r));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 - k + r));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 + k + r));

					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 - k + r));
					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 + k + r));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 - k + r));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 + k + r));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 - k + r));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 + k + r));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 - k + r));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 + k + r));

					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 - k + r));
					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 + k + r));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 - k + r));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 + k + r));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 - k + r));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 + k + r));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 - k + r));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b__0, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r0, mDiv), v);
				_mm256_store_ps(dptr0, v);

				v = _mm256_mul_ps(mSum_b__1, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r1, mDiv), v);
				_mm256_store_ps(dptr1, v);

				v = _mm256_mul_ps(mSum_b__2, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r2, mDiv), v);
				_mm256_store_ps(dptr2, v);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;
				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;
				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
				guideptr_0 += 8;
				guideptr_1 += 8;
				guideptr_2 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
	}
}

void ab2q_Guide3Src3_sep_VHIShare_Unroll2_AVX_omp(std::vector<cv::Mat>& a_b, std::vector<Mat>& a_g, std::vector<cv::Mat>& a_r, std::vector<cv::Mat>& b, std::vector<cv::Mat>& guide, const int r, std::vector<cv::Mat>& dest)
{
	const int width = a_b[0].cols;
	const int height = a_b[0].rows;

	const int R = get_simd_ceil(r, 8);
	const int roffset = R - r;//R-r
	__m256 mDiv = _mm256_set1_ps(1.f / ((2 * r + 1)*(2 * r + 1)));

	Mat buff(Size(width + 2 * R, 12 * omp_get_max_threads()), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		Mat temp = buff(Rect(0, 12 * omp_get_thread_num(), width + 2 * R, 12));

		float* tp_a_b0 = temp.ptr<float>(0, R);
		float* tp_a_g0 = temp.ptr<float>(1, R);
		float* tp_a_r0 = temp.ptr<float>(2, R);
		float* tp_b__0 = temp.ptr<float>(3, R);
		float* tp_a_b1 = temp.ptr<float>(4, R);
		float* tp_a_g1 = temp.ptr<float>(5, R);
		float* tp_a_r1 = temp.ptr<float>(6, R);
		float* tp_b__1 = temp.ptr<float>(7, R);
		float* tp_a_b2 = temp.ptr<float>(8, R);
		float* tp_a_g2 = temp.ptr<float>(9, R);
		float* tp_a_r2 = temp.ptr<float>(10, R);
		float* tp_b__2 = temp.ptr<float>(11, R);

		if (r <= i && i <= height - 1 - r)
		{
			for (int j = 0; j < width; j += 8)
			{
				float* a_b0ptr = a_b[0].ptr<float>(i - r, j);
				float* a_g0ptr = a_g[0].ptr<float>(i - r, j);
				float* a_r0ptr = a_r[0].ptr<float>(i - r, j);
				float* b0ptr = b[0].ptr<float>(i - r, j);
				float* a_b1ptr = a_b[1].ptr<float>(i - r, j);
				float* a_g1ptr = a_g[1].ptr<float>(i - r, j);
				float* a_r1ptr = a_r[1].ptr<float>(i - r, j);
				float* b1ptr = b[1].ptr<float>(i - r, j);
				float* a_b2ptr = a_b[2].ptr<float>(i - r, j);
				float* a_g2ptr = a_g[2].ptr<float>(i - r, j);
				float* a_r2ptr = a_r[2].ptr<float>(i - r, j);
				float* b2ptr = b[2].ptr<float>(i - r, j);

				__m256 mSum_ab0 = _mm256_load_ps(a_b0ptr);
				__m256 mSum_ag0 = _mm256_load_ps(a_g0ptr);
				__m256 mSum_ar0 = _mm256_load_ps(a_r0ptr);
				__m256 mSum_b0 = _mm256_load_ps(b0ptr);

				__m256 mSum_ab1 = _mm256_load_ps(a_b1ptr);
				__m256 mSum_ag1 = _mm256_load_ps(a_g1ptr);
				__m256 mSum_ar1 = _mm256_load_ps(a_r1ptr);
				__m256 mSum_b1 = _mm256_load_ps(b1ptr);

				__m256 mSum_ab2 = _mm256_load_ps(a_b2ptr);
				__m256 mSum_ag2 = _mm256_load_ps(a_g2ptr);
				__m256 mSum_ar2 = _mm256_load_ps(a_r2ptr);
				__m256 mSum_b2 = _mm256_load_ps(b2ptr);

				a_b0ptr += width;
				a_g0ptr += width;
				a_r0ptr += width;
				b0ptr += width;
				a_b1ptr += width;
				a_g1ptr += width;
				a_r1ptr += width;
				b1ptr += width;
				a_b2ptr += width;
				a_g2ptr += width;
				a_r2ptr += width;
				b2ptr += width;

				const int step = 2 * width;
				for (int k = 0; k < r; k++)
				{
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(a_b0ptr));
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(a_b0ptr + width));
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(a_g0ptr));
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(a_g0ptr + width));
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(a_r0ptr));
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(a_r0ptr + width));
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(b0ptr));
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(b0ptr + width));

					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(a_b1ptr));
					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(a_b1ptr + width));
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(a_g1ptr));
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(a_g1ptr + width));
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(a_r1ptr));
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(a_r1ptr + width));
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(b1ptr));
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(b1ptr + width));

					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(a_b2ptr));
					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(a_b2ptr + width));
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(a_g2ptr));
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(a_g2ptr + width));
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(a_r2ptr));
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(a_r2ptr + width));
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(b2ptr));
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(b2ptr + width));

					a_b0ptr += step;
					a_g0ptr += step;
					a_r0ptr += step;
					b0ptr += step;
					a_b1ptr += step;
					a_g1ptr += step;
					a_r1ptr += step;
					b1ptr += step;
					a_b2ptr += step;
					a_g2ptr += step;
					a_r2ptr += step;
					b2ptr += step;
				}

				_mm256_store_ps(tp_a_b0, mSum_ab0);
				_mm256_store_ps(tp_a_g0, mSum_ag0);
				_mm256_store_ps(tp_a_r0, mSum_ar0);
				_mm256_store_ps(tp_b__0, mSum_b0);

				_mm256_store_ps(tp_a_b1, mSum_ab1);
				_mm256_store_ps(tp_a_g1, mSum_ag1);
				_mm256_store_ps(tp_a_r1, mSum_ar1);
				_mm256_store_ps(tp_b__1, mSum_b1);

				_mm256_store_ps(tp_a_b2, mSum_ab2);
				_mm256_store_ps(tp_a_g2, mSum_ag2);
				_mm256_store_ps(tp_a_r2, mSum_ar2);
				_mm256_store_ps(tp_b__2, mSum_b2);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;

				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;

				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* guideptr_0 = guide[0].ptr<float>(i);
			float* guideptr_1 = guide[1].ptr<float>(i);
			float* guideptr_2 = guide[2].ptr<float>(i);
			float* dptr0 = dest[0].ptr<float>(i);
			float* dptr1 = dest[1].ptr<float>(i);
			float* dptr2 = dest[2].ptr<float>(i);

			tp_a_b0 = temp.ptr<float>(0, roffset);
			tp_a_g0 = temp.ptr<float>(1, roffset);
			tp_a_r0 = temp.ptr<float>(2, roffset);
			tp_b__0 = temp.ptr<float>(3, roffset);
			tp_a_b1 = temp.ptr<float>(4, roffset);
			tp_a_g1 = temp.ptr<float>(5, roffset);
			tp_a_r1 = temp.ptr<float>(6, roffset);
			tp_b__1 = temp.ptr<float>(7, roffset);
			tp_a_b2 = temp.ptr<float>(8, roffset);
			tp_a_g2 = temp.ptr<float>(9, roffset);
			tp_a_r2 = temp.ptr<float>(10, roffset);
			tp_b__2 = temp.ptr<float>(11, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b0 = _mm256_loadu_ps(tp_a_b0);
				__m256 mSum_a_g0 = _mm256_loadu_ps(tp_a_g0);
				__m256 mSum_a_r0 = _mm256_loadu_ps(tp_a_r0);
				__m256 mSum_b__0 = _mm256_loadu_ps(tp_b__0);

				__m256 mSum_a_b1 = _mm256_loadu_ps(tp_a_b1);
				__m256 mSum_a_g1 = _mm256_loadu_ps(tp_a_g1);
				__m256 mSum_a_r1 = _mm256_loadu_ps(tp_a_r1);
				__m256 mSum_b__1 = _mm256_loadu_ps(tp_b__1);

				__m256 mSum_a_b2 = _mm256_loadu_ps(tp_a_b2);
				__m256 mSum_a_g2 = _mm256_loadu_ps(tp_a_g2);
				__m256 mSum_a_r2 = _mm256_loadu_ps(tp_a_r2);
				__m256 mSum_b__2 = _mm256_loadu_ps(tp_b__2);

				for (int k = 1; k <= r; k++)
				{
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 + k));
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 + k + r));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 + k));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 + k + r));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 + k));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 + k + r));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 + k));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 + k + r));

					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 + k));
					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 + k + r));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 + k));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 + k + r));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 + k));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 + k + r));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 + k));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 + k + r));

					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 + k));
					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 + k + r));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 + k));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 + k + r));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 + k));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 + k + r));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 + k));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b__0, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r0, mDiv), v);
				_mm256_store_ps(dptr0, v);

				v = _mm256_mul_ps(mSum_b__1, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r1, mDiv), v);
				_mm256_store_ps(dptr1, v);

				v = _mm256_mul_ps(mSum_b__2, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r2, mDiv), v);
				_mm256_store_ps(dptr2, v);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;
				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;
				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
				guideptr_0 += 8;
				guideptr_1 += 8;
				guideptr_2 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
		else
		{
			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_ab0 = _mm256_load_ps(a_b[0].ptr<float>(i, j));
				__m256 mSum_ag0 = _mm256_load_ps(a_g[0].ptr<float>(i, j));
				__m256 mSum_ar0 = _mm256_load_ps(a_r[0].ptr<float>(i, j));
				__m256 mSum_b0 = _mm256_load_ps(b[0].ptr<float>(i, j));

				__m256 mSum_ab1 = _mm256_load_ps(a_b[1].ptr<float>(i, j));
				__m256 mSum_ag1 = _mm256_load_ps(a_g[1].ptr<float>(i, j));
				__m256 mSum_ar1 = _mm256_load_ps(a_r[1].ptr<float>(i, j));
				__m256 mSum_b1 = _mm256_load_ps(b[1].ptr<float>(i, j));

				__m256 mSum_ab2 = _mm256_load_ps(a_b[2].ptr<float>(i, j));
				__m256 mSum_ag2 = _mm256_load_ps(a_g[2].ptr<float>(i, j));
				__m256 mSum_ar2 = _mm256_load_ps(a_r[2].ptr<float>(i, j));
				__m256 mSum_b2 = _mm256_load_ps(b[2].ptr<float>(i, j));
				for (int k = 1; k <= r; k++)
				{
					int vl = max(i - k, 0);
					int vh = min(i + k, height - 1);

					float* sp1 = a_b[0].ptr<float>(vl, j);
					float* sp2 = a_b[0].ptr<float>(vh, j);
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(sp1));
					mSum_ab0 = _mm256_add_ps(mSum_ab0, _mm256_load_ps(sp2));

					sp1 = a_g[0].ptr<float>(vl, j);
					sp2 = a_g[0].ptr<float>(vh, j);
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(sp1));
					mSum_ag0 = _mm256_add_ps(mSum_ag0, _mm256_load_ps(sp2));

					sp1 = a_r[0].ptr<float>(vl, j);
					sp2 = a_r[0].ptr<float>(vh, j);
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(sp1));
					mSum_ar0 = _mm256_add_ps(mSum_ar0, _mm256_load_ps(sp2));

					sp1 = b[0].ptr<float>(vl, j);
					sp2 = b[0].ptr<float>(vh, j);
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(sp1));
					mSum_b0 = _mm256_add_ps(mSum_b0, _mm256_load_ps(sp2));

					sp1 = a_b[1].ptr<float>(vl, j);
					sp2 = a_b[1].ptr<float>(vh, j);
					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(sp1));
					mSum_ab1 = _mm256_add_ps(mSum_ab1, _mm256_load_ps(sp2));

					sp1 = a_g[1].ptr<float>(vl, j);
					sp2 = a_g[1].ptr<float>(vh, j);
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(sp1));
					mSum_ag1 = _mm256_add_ps(mSum_ag1, _mm256_load_ps(sp2));

					sp1 = a_r[1].ptr<float>(vl, j);
					sp2 = a_r[1].ptr<float>(vh, j);
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(sp1));
					mSum_ar1 = _mm256_add_ps(mSum_ar1, _mm256_load_ps(sp2));

					sp1 = b[1].ptr<float>(vl, j);
					sp2 = b[1].ptr<float>(vh, j);
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(sp1));
					mSum_b1 = _mm256_add_ps(mSum_b1, _mm256_load_ps(sp2));

					sp1 = a_b[2].ptr<float>(vl, j);
					sp2 = a_b[2].ptr<float>(vh, j);
					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(sp1));
					mSum_ab2 = _mm256_add_ps(mSum_ab2, _mm256_load_ps(sp2));

					sp1 = a_g[2].ptr<float>(vl, j);
					sp2 = a_g[2].ptr<float>(vh, j);
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(sp1));
					mSum_ag2 = _mm256_add_ps(mSum_ag2, _mm256_load_ps(sp2));

					sp1 = a_r[2].ptr<float>(vl, j);
					sp2 = a_r[2].ptr<float>(vh, j);
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(sp1));
					mSum_ar2 = _mm256_add_ps(mSum_ar2, _mm256_load_ps(sp2));

					sp1 = b[2].ptr<float>(vl, j);
					sp2 = b[2].ptr<float>(vh, j);
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(sp1));
					mSum_b2 = _mm256_add_ps(mSum_b2, _mm256_load_ps(sp2));
				}

				_mm256_store_ps(tp_a_b0, mSum_ab0);
				_mm256_store_ps(tp_a_g0, mSum_ag0);
				_mm256_store_ps(tp_a_r0, mSum_ar0);
				_mm256_store_ps(tp_b__0, mSum_b0);

				_mm256_store_ps(tp_a_b1, mSum_ab1);
				_mm256_store_ps(tp_a_g1, mSum_ag1);
				_mm256_store_ps(tp_a_r1, mSum_ar1);
				_mm256_store_ps(tp_b__1, mSum_b1);

				_mm256_store_ps(tp_a_b2, mSum_ab2);
				_mm256_store_ps(tp_a_g2, mSum_ag2);
				_mm256_store_ps(tp_a_r2, mSum_ar2);
				_mm256_store_ps(tp_b__2, mSum_b2);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;

				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;

				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
			}

			copyMakeBorderReplicateForLineBuffers(temp, R);

			float* guideptr_0 = guide[0].ptr<float>(i);
			float* guideptr_1 = guide[1].ptr<float>(i);
			float* guideptr_2 = guide[2].ptr<float>(i);
			float* dptr0 = dest[0].ptr<float>(i);
			float* dptr1 = dest[1].ptr<float>(i);
			float* dptr2 = dest[2].ptr<float>(i);

			tp_a_b0 = temp.ptr<float>(0, roffset);
			tp_a_g0 = temp.ptr<float>(1, roffset);
			tp_a_r0 = temp.ptr<float>(2, roffset);
			tp_b__0 = temp.ptr<float>(3, roffset);
			tp_a_b1 = temp.ptr<float>(4, roffset);
			tp_a_g1 = temp.ptr<float>(5, roffset);
			tp_a_r1 = temp.ptr<float>(6, roffset);
			tp_b__1 = temp.ptr<float>(7, roffset);
			tp_a_b2 = temp.ptr<float>(8, roffset);
			tp_a_g2 = temp.ptr<float>(9, roffset);
			tp_a_r2 = temp.ptr<float>(10, roffset);
			tp_b__2 = temp.ptr<float>(11, roffset);

			for (int j = 0; j < width; j += 8)
			{
				__m256 mSum_a_b0 = _mm256_loadu_ps(tp_a_b0 + r);
				__m256 mSum_a_g0 = _mm256_loadu_ps(tp_a_g0 + r);
				__m256 mSum_a_r0 = _mm256_loadu_ps(tp_a_r0 + r);
				__m256 mSum_b__0 = _mm256_loadu_ps(tp_b__0 + r);

				__m256 mSum_a_b1 = _mm256_loadu_ps(tp_a_b1 + r);
				__m256 mSum_a_g1 = _mm256_loadu_ps(tp_a_g1 + r);
				__m256 mSum_a_r1 = _mm256_loadu_ps(tp_a_r1 + r);
				__m256 mSum_b__1 = _mm256_loadu_ps(tp_b__1 + r);

				__m256 mSum_a_b2 = _mm256_loadu_ps(tp_a_b2 + r);
				__m256 mSum_a_g2 = _mm256_loadu_ps(tp_a_g2 + r);
				__m256 mSum_a_r2 = _mm256_loadu_ps(tp_a_r2 + r);
				__m256 mSum_b__2 = _mm256_loadu_ps(tp_b__2 + r);
				for (int k = 1; k <= r; k++)
				{
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 - k + r));
					mSum_a_b0 = _mm256_add_ps(mSum_a_b0, _mm256_loadu_ps(tp_a_b0 + k + r));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 - k + r));
					mSum_a_g0 = _mm256_add_ps(mSum_a_g0, _mm256_loadu_ps(tp_a_g0 + k + r));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 - k + r));
					mSum_a_r0 = _mm256_add_ps(mSum_a_r0, _mm256_loadu_ps(tp_a_r0 + k + r));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 - k + r));
					mSum_b__0 = _mm256_add_ps(mSum_b__0, _mm256_loadu_ps(tp_b__0 + k + r));

					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 - k + r));
					mSum_a_b1 = _mm256_add_ps(mSum_a_b1, _mm256_loadu_ps(tp_a_b1 + k + r));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 - k + r));
					mSum_a_g1 = _mm256_add_ps(mSum_a_g1, _mm256_loadu_ps(tp_a_g1 + k + r));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 - k + r));
					mSum_a_r1 = _mm256_add_ps(mSum_a_r1, _mm256_loadu_ps(tp_a_r1 + k + r));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 - k + r));
					mSum_b__1 = _mm256_add_ps(mSum_b__1, _mm256_loadu_ps(tp_b__1 + k + r));

					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 - k + r));
					mSum_a_b2 = _mm256_add_ps(mSum_a_b2, _mm256_loadu_ps(tp_a_b2 + k + r));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 - k + r));
					mSum_a_g2 = _mm256_add_ps(mSum_a_g2, _mm256_loadu_ps(tp_a_g2 + k + r));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 - k + r));
					mSum_a_r2 = _mm256_add_ps(mSum_a_r2, _mm256_loadu_ps(tp_a_r2 + k + r));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 - k + r));
					mSum_b__2 = _mm256_add_ps(mSum_b__2, _mm256_loadu_ps(tp_b__2 + k + r));
				}

				__m256 v = _mm256_mul_ps(mSum_b__0, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g0, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r0, mDiv), v);
				_mm256_store_ps(dptr0, v);

				v = _mm256_mul_ps(mSum_b__1, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g1, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r1, mDiv), v);
				_mm256_store_ps(dptr1, v);

				v = _mm256_mul_ps(mSum_b__2, mDiv);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_0), _mm256_mul_ps(mSum_a_b2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_1), _mm256_mul_ps(mSum_a_g2, mDiv), v);
				v = _mm256_fmadd_ps(_mm256_load_ps(guideptr_2), _mm256_mul_ps(mSum_a_r2, mDiv), v);
				_mm256_store_ps(dptr2, v);

				tp_a_b0 += 8;
				tp_a_g0 += 8;
				tp_a_r0 += 8;
				tp_b__0 += 8;
				tp_a_b1 += 8;
				tp_a_g1 += 8;
				tp_a_r1 += 8;
				tp_b__1 += 8;
				tp_a_b2 += 8;
				tp_a_g2 += 8;
				tp_a_r2 += 8;
				tp_b__2 += 8;
				guideptr_0 += 8;
				guideptr_1 += 8;
				guideptr_2 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
	}
}


void guidedFilter_SepVHI_Share::filter()
{
	if (src.channels() == 1)
	{
		if (guide.channels() == 1)
		{
			if (parallelType == NAIVE)
			{
				Ip2ab_Guide1_sep_VHIShare_AVX(guide, src, r, eps, ab_p[0], b_p[0]);
				ab2q_Guide1_sep_VHIShare_AVX(ab_p[0], b_p[0], guide, r, dest);
			}
			else
			{
				Ip2ab_Guide1_sep_VHIShare_AVX_omp(guide, src, r, eps, ab_p[0], b_p[0]);
				ab2q_Guide1_sep_VHIShare_AVX_omp(ab_p[0], b_p[0], guide, r, dest);
			}
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);

			if (parallelType == NAIVE)
			{
				Ip2ab_Guide3_sep_VHI_AVX(vguide[0], vguide[1], vguide[2], src, r, eps, ab_p[0], ag_p[0], ar_p[0], b_p[0]);
				ab2q_Guide3_sep_VHI_AVX(ab_p[0], ag_p[0], ar_p[0], b_p[0], vguide[0], vguide[1], vguide[2], r, dest);
			}
			else
			{
				Ip2ab_Guide3_sep_VHI_AVX_omp(vguide[0], vguide[1], vguide[2], src, r, eps, ab_p[0], ag_p[0], ar_p[0], b_p[0]);
				ab2q_Guide3_sep_VHI_AVX_omp(ab_p[0], ag_p[0], ar_p[0], b_p[0], vguide[0], vguide[1], vguide[2], r, dest);
			}
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
			if (parallelType == NAIVE)
			{
				Ip2ab_Guide1Src3_sep_VHIShare_AVX(guide, vsrc, r, eps, ab_p, b_p);

				ab2q_Guide1_sep_VHIShare_AVX(ab_p[0], b_p[0], guide, r, vdest[0]);
				ab2q_Guide1_sep_VHIShare_AVX(ab_p[1], b_p[1], guide, r, vdest[1]);
				ab2q_Guide1_sep_VHIShare_AVX(ab_p[2], b_p[2], guide, r, vdest[2]);
			}
			else
			{
				Ip2ab_Guide1Src3_sep_VHIShare_AVX_omp(guide, vsrc, r, eps, ab_p, b_p);

				ab2q_Guide1_sep_VHIShare_AVX_omp(ab_p[0], b_p[0], guide, r, vdest[0]);
				ab2q_Guide1_sep_VHIShare_AVX_omp(ab_p[1], b_p[1], guide, r, vdest[1]);
				ab2q_Guide1_sep_VHIShare_AVX_omp(ab_p[2], b_p[2], guide, r, vdest[2]);
			}
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);

			if (parallelType == NAIVE)
			{
				Ip2ab_Guide3Src3_sep_VHIShare_AVX(vguide, vsrc, r, eps, ab_p, ag_p, ar_p, b_p);

				ab2q_Guide3_sep_VHI_AVX(ab_p[0], ag_p[0], ar_p[0], b_p[0], vguide[0], vguide[1], vguide[2], r, vdest[0]);
				ab2q_Guide3_sep_VHI_AVX(ab_p[1], ag_p[1], ar_p[1], b_p[1], vguide[0], vguide[1], vguide[2], r, vdest[1]);
				ab2q_Guide3_sep_VHI_AVX(ab_p[2], ag_p[2], ar_p[2], b_p[2], vguide[0], vguide[1], vguide[2], r, vdest[2]);
			}
			else
			{
				Ip2ab_Guide3Src3_sep_VHIShare_AVX_omp(vguide, vsrc, r, eps, ab_p, ag_p, ar_p, b_p);
				///Ip2ab_Guide3Src3_sep_VHIShare_AVX_omp(vsrc, vguide, r, eps, ab_p, ag_p, ar_p, b_p);

				ab2q_Guide3_sep_VHI_AVX_omp(ab_p[0], ag_p[0], ar_p[0], b_p[0], vguide[0], vguide[1], vguide[2], r, vdest[0]);
				ab2q_Guide3_sep_VHI_AVX_omp(ab_p[1], ag_p[1], ar_p[1], b_p[1], vguide[0], vguide[1], vguide[2], r, vdest[1]);
				ab2q_Guide3_sep_VHI_AVX_omp(ab_p[2], ag_p[2], ar_p[2], b_p[2], vguide[0], vguide[1], vguide[2], r, vdest[2]);

				//ab2q_Guide3_sep_VHI_AVX_omp(ab_p[0], ab_p[1], ab_p[2], b_p[0], vguide[0], vguide[1], vguide[2], r, vdest[0]);
				//ab2q_Guide3_sep_VHI_AVX_omp(ag_p[0], ag_p[1], ag_p[2], b_p[1], vguide[0], vguide[1], vguide[2], r, vdest[1]);
				//ab2q_Guide3_sep_VHI_AVX_omp(ar_p[0], ar_p[1], ar_p[2], b_p[2], vguide[0], vguide[1], vguide[2], r, vdest[2]);
			}
		}

		merge(vdest, dest);
	}
}

void guidedFilter_SepVHI_Share::filterVector()
{
	if (src.channels() == 1)
	{
		if (guide.channels() == 1)
		{
			if (parallelType == NAIVE)
			{
				Ip2ab_Guide1_sep_VHIShare_AVX(vguide[0], vsrc[0], r, eps, ab_p[0], b_p[0]);
				ab2q_Guide1_sep_VHIShare_AVX(ab_p[0], b_p[0], vguide[0], r, vdest[0]);
			}
			else
			{
				Ip2ab_Guide1_sep_VHIShare_AVX_omp(vguide[0], vsrc[0], r, eps, ab_p[0], b_p[0]);
				ab2q_Guide1_sep_VHIShare_AVX_omp(ab_p[0], b_p[0], vguide[0], r, vdest[0]);
			}
		}
		else if (guide.channels() == 3)
		{
			if (parallelType == NAIVE)
			{
				Ip2ab_Guide3_sep_VHI_AVX(vguide[0], vguide[1], vguide[2], vsrc[0], r, eps, ab_p[0], ag_p[0], ar_p[0], b_p[0]);
				ab2q_Guide3_sep_VHI_AVX(ab_p[0], ag_p[0], ar_p[0], b_p[0], vguide[0], vguide[1], vguide[2], r, vdest[0]);
			}
			else
			{
				Ip2ab_Guide3_sep_VHI_AVX_omp(vguide[0], vguide[1], vguide[2], vsrc[0], r, eps, ab_p[0], ag_p[0], ar_p[0], b_p[0]);
				ab2q_Guide3_sep_VHI_AVX_omp(ab_p[0], ag_p[0], ar_p[0], b_p[0], vguide[0], vguide[1], vguide[2], r, vdest[0]);
			}
		}
	}
	else if (src.channels() == 3)
	{
		if (guide.channels() == 1)
		{
			if (parallelType == NAIVE)
			{
				Ip2ab_Guide1Src3_sep_VHIShare_AVX(vguide[0], vsrc, r, eps, ab_p, b_p);

				ab2q_Guide1_sep_VHIShare_AVX(ab_p[0], b_p[0], vguide[0], r, vdest[0]);
				ab2q_Guide1_sep_VHIShare_AVX(ab_p[1], b_p[1], vguide[0], r, vdest[1]);
				ab2q_Guide1_sep_VHIShare_AVX(ab_p[2], b_p[2], vguide[0], r, vdest[2]);
			}
			else
			{
				Ip2ab_Guide1Src3_sep_VHIShare_AVX_omp(vguide[0], vsrc, r, eps, ab_p, b_p);

				ab2q_Guide1_sep_VHIShare_AVX_omp(ab_p[0], b_p[0], vguide[0], r, vdest[0]);
				ab2q_Guide1_sep_VHIShare_AVX_omp(ab_p[1], b_p[1], vguide[0], r, vdest[1]);
				ab2q_Guide1_sep_VHIShare_AVX_omp(ab_p[2], b_p[2], vguide[0], r, vdest[2]);
			}
		}
		else if (guide.channels() == 3)
		{
			if (parallelType == NAIVE)
			{
				Ip2ab_Guide3Src3_sep_VHIShare_AVX(vguide, vsrc, r, eps, ab_p, ag_p, ar_p, b_p);

				ab2q_Guide3_sep_VHI_AVX(ab_p[0], ag_p[0], ar_p[0], b_p[0], vguide[0], vguide[1], vguide[2], r, vdest[0]);
				ab2q_Guide3_sep_VHI_AVX(ab_p[1], ag_p[1], ar_p[1], b_p[1], vguide[0], vguide[1], vguide[2], r, vdest[1]);
				ab2q_Guide3_sep_VHI_AVX(ab_p[2], ag_p[2], ar_p[2], b_p[2], vguide[0], vguide[1], vguide[2], r, vdest[2]);
			}
			else
			{
				Ip2ab_Guide3Src3_sep_VHIShare_AVX_omp(vguide, vsrc, r, eps, ab_p, ag_p, ar_p, b_p);

				ab2q_Guide3_sep_VHI_AVX_omp(ab_p[0], ag_p[0], ar_p[0], b_p[0], vguide[0], vguide[1], vguide[2], r, vdest[0]);
				ab2q_Guide3_sep_VHI_AVX_omp(ab_p[1], ag_p[1], ar_p[1], b_p[1], vguide[0], vguide[1], vguide[2], r, vdest[1]);
				ab2q_Guide3_sep_VHI_AVX_omp(ab_p[2], ag_p[2], ar_p[2], b_p[2], vguide[0], vguide[1], vguide[2], r, vdest[2]);
			}
		}
	}
}

void guidedFilter_SepVHI_Share::upsample()
{
	if (guide.size() != src.size())
	{
		src_low = src;
	}

	if (src.channels() == 1)
	{
		if (guide.channels() == 1)
		{
			if (parallelType == NAIVE)
			{
				Ip2ab_Guide1_sep_VHIShare_AVX(guide_low, src_low, r, eps, ab_p[0], b_p[0]);

				blurSeparableVHI(ab_p[0], b_p[0], r, mean_a_b, mean_b);

				Mat temp_high = a_high_b;
				resize(mean_a_b, dest, guide.size(), 0, 0, upsample_method);
				resize(mean_b, temp_high, guide.size(), 0, 0, upsample_method);

				fmadd(dest, guide, temp_high, dest);
			}
			else
			{
				Ip2ab_Guide1_sep_VHIShare_AVX_omp(guide_low, src_low, r, eps, ab_p[0], b_p[0]);

				blurSeparableVHI_omp(ab_p[0], b_p[0], r, mean_a_b, mean_b);

				Mat temp_high = a_high_b;
				resize(mean_a_b, dest, guide.size(), 0, 0, upsample_method);
				resize(mean_b, temp_high, guide.size(), 0, 0, upsample_method);

				fmadd(dest, guide, temp_high, dest);
			}
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);
			split(guide_low, vguide_low);

			if (parallelType == NAIVE)
			{
				Ip2ab_Guide3_sep_VHI_AVX(vguide_low[0], vguide_low[1], vguide_low[2], src_low, r, eps, ab_p[0], ag_p[0], ar_p[0], b_p[0]);

				Size s = vguide[0].size();
				blurSeparableVHI(ab_p[0], ag_p[0], ar_p[0], b_p[0], r, mean_a_b, mean_a_g, mean_a_r, mean_b);
				resize(mean_b, dest, s, 0, 0, upsample_method);
				resize(mean_a_b, a_high_b, s, 0, 0, upsample_method);
				resize(mean_a_g, a_high_g, s, 0, 0, upsample_method);
				resize(mean_a_r, a_high_r, s, 0, 0, upsample_method);
				ab2q_fmadd(a_high_b, a_high_g, a_high_r, vguide[0], vguide[1], vguide[2], dest, dest);
			}
			else
			{
				Ip2ab_Guide3_sep_VHI_AVX_omp(vguide_low[0], vguide_low[1], vguide_low[2], src_low, r, eps, ab_p[0], ag_p[0], ar_p[0], b_p[0]);

				Size s = vguide[0].size();
				blurSeparableVHI_omp(ab_p[0], ag_p[0], ar_p[0], b_p[0], r, mean_a_b, mean_a_g, mean_a_r, mean_b);
				resize(mean_b, dest, s, 0, 0, upsample_method);
				resize(mean_a_b, a_high_b, s, 0, 0, upsample_method);
				resize(mean_a_g, a_high_g, s, 0, 0, upsample_method);
				resize(mean_a_r, a_high_r, s, 0, 0, upsample_method);
				ab2q_fmadd_omp(a_high_b, a_high_g, a_high_r, vguide[0], vguide[1], vguide[2], dest, dest);
			}
		}
	}
	else if (src.channels() == 3)
	{
		split(src_low, vsrc_low);

		const int depth = src.depth();
		vdest[0].create(src.size(), depth);
		vdest[1].create(src.size(), depth);
		vdest[2].create(src.size(), depth);

		if (guide.channels() == 1)
		{
			if (parallelType == NAIVE)
			{
				Ip2ab_Guide1Src3_sep_VHIShare_AVX(guide_low, vsrc_low, r, eps, ab_p, b_p);

				Size s = guide.size();
				blurSeparableVHI(ab_p[0], b_p[0], r, mean_a_b, mean_b);
				resize(mean_a_b, vdest[0], s, 0, 0, upsample_method);
				resize(mean_b, a_high_b, s, 0, 0, upsample_method);
				fmadd(vdest[0], guide, a_high_b, vdest[0]);

				blurSeparableVHI(ab_p[1], b_p[1], r, mean_a_b, mean_b);
				resize(mean_a_b, vdest[1], s, 0, 0, upsample_method);
				resize(mean_b, a_high_b, s, 0, 0, upsample_method);
				fmadd(vdest[1], guide, a_high_b, vdest[1]);

				blurSeparableVHI(ab_p[2], b_p[2], r, mean_a_b, mean_b);
				resize(mean_a_b, vdest[2], s, 0, 0, upsample_method);
				resize(mean_b, a_high_b, s, 0, 0, upsample_method);
				fmadd(vdest[2], guide, a_high_b, vdest[2]);
			}
			else
			{
				Ip2ab_Guide1Src3_sep_VHIShare_AVX_omp(guide_low, vsrc_low, r, eps, ab_p, b_p);

				Size s = guide.size();
				blurSeparableVHI_omp(ab_p[0], b_p[0], r, mean_a_b, mean_b);
				resize(mean_a_b, vdest[0], s, 0, 0, upsample_method);
				resize(mean_b, a_high_b, s, 0, 0, upsample_method);
				fmadd(vdest[0], guide, a_high_b, vdest[0]);

				blurSeparableVHI_omp(ab_p[1], b_p[1], r, mean_a_b, mean_b);
				resize(mean_a_b, vdest[1], s, 0, 0, upsample_method);
				resize(mean_b, a_high_b, s, 0, 0, upsample_method);
				fmadd(vdest[1], guide, a_high_b, vdest[1]);

				blurSeparableVHI_omp(ab_p[2], b_p[2], r, mean_a_b, mean_b);
				resize(mean_a_b, vdest[2], s, 0, 0, upsample_method);
				resize(mean_b, a_high_b, s, 0, 0, upsample_method);
				fmadd(vdest[2], guide, a_high_b, vdest[2]);
			}
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);
			split(guide_low, vguide_low);

			if (parallelType == NAIVE)
			{
				Ip2ab_Guide3Src3_sep_VHIShare_AVX(vguide_low, vsrc_low, r, eps, ab_p, ag_p, ar_p, b_p);

				Size s = vguide[0].size();
				blurSeparableVHI(ab_p[0], ag_p[0], ar_p[0], b_p[0], r, mean_a_b, mean_a_g, mean_a_r, mean_b);
				resize(mean_b, vdest[0], s, 0, 0, upsample_method);
				resize(mean_a_b, a_high_b, s, 0, 0, upsample_method);
				resize(mean_a_g, a_high_g, s, 0, 0, upsample_method);
				resize(mean_a_r, a_high_r, s, 0, 0, upsample_method);
				ab2q_fmadd(a_high_b, a_high_g, a_high_r, vguide[0], vguide[1], vguide[2], vdest[0], vdest[0]);

				blurSeparableVHI(ab_p[1], ag_p[1], ar_p[1], b_p[1], r, mean_a_b, mean_a_g, mean_a_r, mean_b);
				resize(mean_b, vdest[1], s, 0, 0, upsample_method);
				resize(mean_a_b, a_high_b, s, 0, 0, upsample_method);
				resize(mean_a_g, a_high_g, s, 0, 0, upsample_method);
				resize(mean_a_r, a_high_r, s, 0, 0, upsample_method);
				ab2q_fmadd(a_high_b, a_high_g, a_high_r, vguide[0], vguide[1], vguide[2], vdest[1], vdest[1]);

				blurSeparableVHI(ab_p[2], ag_p[2], ar_p[2], b_p[2], r, mean_a_b, mean_a_g, mean_a_r, mean_b);
				resize(mean_b, vdest[2], s, 0, 0, upsample_method);
				resize(mean_a_b, a_high_b, s, 0, 0, upsample_method);
				resize(mean_a_g, a_high_g, s, 0, 0, upsample_method);
				resize(mean_a_r, a_high_r, s, 0, 0, upsample_method);
				ab2q_fmadd(a_high_b, a_high_g, a_high_r, vguide[0], vguide[1], vguide[2], vdest[2], vdest[2]);
			}
			else
			{
				Ip2ab_Guide3Src3_sep_VHIShare_AVX_omp(vguide_low, vsrc_low, r, eps, ab_p, ag_p, ar_p, b_p);
				//Ip2ab_Guide3Src3_sep_VHIShare_AVX_omp(vsrc_low, vguide_low, r, eps, ab_p, ag_p, ar_p, b_p);

				Size s = vguide[0].size();
				blurSeparableVHI_omp(ab_p[0], ag_p[0], ar_p[0], b_p[0], r, mean_a_b, mean_a_g, mean_a_r, mean_b);
				resize(mean_b, vdest[0], s, 0, 0, upsample_method);
				resize(mean_a_b, a_high_b, s, 0, 0, upsample_method);
				resize(mean_a_g, a_high_g, s, 0, 0, upsample_method);
				resize(mean_a_r, a_high_r, s, 0, 0, upsample_method);
				ab2q_fmadd_omp(a_high_b, a_high_g, a_high_r, vguide[0], vguide[1], vguide[2], vdest[0], vdest[0]);

				blurSeparableVHI_omp(ab_p[1], ag_p[1], ar_p[1], b_p[1], r, mean_a_b, mean_a_g, mean_a_r, mean_b);
				resize(mean_b, vdest[1], s, 0, 0, upsample_method);
				resize(mean_a_b, a_high_b, s, 0, 0, upsample_method);
				resize(mean_a_g, a_high_g, s, 0, 0, upsample_method);
				resize(mean_a_r, a_high_r, s, 0, 0, upsample_method);
				ab2q_fmadd_omp(a_high_b, a_high_g, a_high_r, vguide[0], vguide[1], vguide[2], vdest[1], vdest[1]);

				blurSeparableVHI_omp(ab_p[2], ag_p[2], ar_p[2], b_p[2], r, mean_a_b, mean_a_g, mean_a_r, mean_b);
				resize(mean_b, vdest[2], s, 0, 0, upsample_method);
				resize(mean_a_b, a_high_b, s, 0, 0, upsample_method);
				resize(mean_a_g, a_high_g, s, 0, 0, upsample_method);
				resize(mean_a_r, a_high_r, s, 0, 0, upsample_method);
				ab2q_fmadd_omp(a_high_b, a_high_g, a_high_r, vguide[0], vguide[1], vguide[2], vdest[2], vdest[2]);
			}
		}

		merge(vdest, dest);
	}
}