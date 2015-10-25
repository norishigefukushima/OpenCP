#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void disp16S2depth32F(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), CV_32F);
		if (dest.type() != CV_32F)dest = Mat::zeros(src.size(), CV_32F);

#if CV_SSE4_1
		const int ssesize = src.size().area() / 8;
		const int remsize = src.size().area() - 8 * ssesize;
		short* s = src.ptr<short>(0);
		float*  d = dest.ptr<float>(0);
		const __m128 maf = _mm_set1_ps(a*focal_baseline);
		if (b == 0.f)
		{
			for (int i = 0; i < ssesize; i++)
			{
				__m128i r0 = _mm_loadl_epi64((const __m128i*)(s));
				__m128i r1 = _mm_loadl_epi64((const __m128i*)(s + 4));

				__m128 v1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
				__m128 v2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));


				v1 = _mm_div_ps(maf, v1);
				v2 = _mm_div_ps(maf, v2);


				_mm_stream_ps(d, v1);
				_mm_stream_ps(d + 4, v2);

				s += 8;
				d += 8;
			}
		}
		else
		{
			const __m128 mb = _mm_set1_ps(b);
			for (int i = 0; i < ssesize; i++)
			{
				__m128i r0 = _mm_loadl_epi64((const __m128i*)(s));
				__m128i r1 = _mm_loadl_epi64((const __m128i*)(s + 4));

				__m128 v1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
				__m128 v2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));


				v1 = _mm_add_ps(_mm_div_ps(maf, v1), mb);
				v2 = _mm_add_ps(_mm_div_ps(maf, v2), mb);

				_mm_stream_ps(d, v1);
				_mm_stream_ps(d + 4, v2);

				s += 8;
				d += 8;
			}
		}
		for (int i = 0; i < remsize; i++)
		{
			*d = a*focal_baseline / *s + b;
			s++;
			d++;
		}

#else
		Mat temp;
		divide(a*focal_baseline, src, temp);
		add(temp, b, temp);
		temp.convertTo(dest, CV_8U);
#endif
	}

	void disp16S2depth16U(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), CV_16U);
		if (dest.type() != CV_16U)dest = Mat::zeros(src.size(), CV_16U);


#if CV_SSE4_1
		const int ssesize = src.size().area() / 16;
		const int remsize = src.size().area() - 16 * ssesize;
		short* s = src.ptr<short>(0);
		ushort*  d = dest.ptr<ushort>(0);
		const __m128 maf = _mm_set1_ps(a*focal_baseline);
		if (b == 0.f)
		{
			for (int i = 0; i < ssesize; i++)
			{
				__m128i r0 = _mm_loadl_epi64((const __m128i*)(s));
				__m128i r1 = _mm_loadl_epi64((const __m128i*)(s + 4));

				__m128 v1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
				__m128 v2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

				r0 = _mm_loadl_epi64((const __m128i*)(s + 8));
				r1 = _mm_loadl_epi64((const __m128i*)(s + 12));
				__m128 v3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
				__m128 v4 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

				v1 = _mm_div_ps(maf, v1);
				v2 = _mm_div_ps(maf, v2);
				v3 = _mm_div_ps(maf, v3);
				v4 = _mm_div_ps(maf, v4);

				_mm_stream_si128((__m128i*)(d), _mm_packs_epi32(_mm_cvtps_epi32(v1), _mm_cvtps_epi32(v2)));
				_mm_stream_si128((__m128i*)(d + 8), _mm_packs_epi32(_mm_cvtps_epi32(v3), _mm_cvtps_epi32(v4)));

				s += 16;
				d += 16;
			}
		}
		else
		{
			const __m128 mb = _mm_set1_ps(b);
			for (int i = 0; i < ssesize; i++)
			{
				__m128i r0 = _mm_loadl_epi64((const __m128i*)(s));
				__m128i r1 = _mm_loadl_epi64((const __m128i*)(s + 4));

				__m128 v1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
				__m128 v2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

				r0 = _mm_loadl_epi64((const __m128i*)(s + 8));
				r1 = _mm_loadl_epi64((const __m128i*)(s + 12));
				__m128 v3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
				__m128 v4 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

				v1 = _mm_add_ps(_mm_div_ps(maf, v1), mb);
				v2 = _mm_add_ps(_mm_div_ps(maf, v2), mb);
				v3 = _mm_add_ps(_mm_div_ps(maf, v3), mb);
				v4 = _mm_add_ps(_mm_div_ps(maf, v4), mb);

				_mm_stream_si128((__m128i*)(d), _mm_packs_epi32(_mm_cvtps_epi32(v1), _mm_cvtps_epi32(v2)));
				_mm_stream_si128((__m128i*)(d + 8), _mm_packs_epi32(_mm_cvtps_epi32(v3), _mm_cvtps_epi32(v4)));

				s += 16;
				d += 16;
			}
		}
		for (int i = 0; i < remsize; i++)
		{
			*d = (ushort)cvRound(a*focal_baseline / *s + b);
			s++;
			d++;
		}

#else
		Mat temp;
		divide(a*focal_baseline, src, temp);
		add(temp, b, temp);
		temp.convertTo(dest, CV_8U);
#endif
	}


	void depth32F2disp8U(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), CV_8U);
		if (dest.type() != CV_8U)dest = Mat::zeros(src.size(), CV_8U);

#if CV_SSE4_1
		const int ssesize = src.size().area() / 16;
		const int remsize = src.size().area() - 16 * ssesize;
		float* s = src.ptr<float>(0);
		uchar*  d = dest.ptr<uchar>(0);
		const __m128 maf = _mm_set1_ps(a*focal_baseline);
		if (b == 0.f)
		{
			for (int i = 0; i < ssesize; i++)
			{
				__m128 v1 = _mm_load_ps(s);
				__m128 v2 = _mm_load_ps(s + 4);
				__m128 v3 = _mm_load_ps(s + 8);
				__m128 v4 = _mm_load_ps(s + 12);

				v1 = _mm_div_ps(maf, v1);
				v2 = _mm_div_ps(maf, v2);
				v3 = _mm_div_ps(maf, v3);
				v4 = _mm_div_ps(maf, v4);

				_mm_stream_si128((__m128i*)(d), _mm_packus_epi16(
					_mm_packs_epi32(_mm_cvtps_epi32(v1), _mm_cvtps_epi32(v2)),
					_mm_packs_epi32(_mm_cvtps_epi32(v3), _mm_cvtps_epi32(v4))
					));
				s += 16;
				d += 16;
			}
		}
		else
		{
			const __m128 mb = _mm_set1_ps(b);
			for (int i = 0; i < ssesize; i++)
			{

				__m128 v1 = _mm_load_ps(s);
				__m128 v2 = _mm_load_ps(s + 4);
				__m128 v3 = _mm_load_ps(s + 8);
				__m128 v4 = _mm_load_ps(s + 12);

				v1 = _mm_add_ps(_mm_div_ps(maf, v1), mb);
				v2 = _mm_add_ps(_mm_div_ps(maf, v2), mb);
				v3 = _mm_add_ps(_mm_div_ps(maf, v3), mb);
				v4 = _mm_add_ps(_mm_div_ps(maf, v4), mb);

				_mm_stream_si128((__m128i*)(d), _mm_packus_epi16(
					_mm_packs_epi32(_mm_cvtps_epi32(v1), _mm_cvtps_epi32(v2)),
					_mm_packs_epi32(_mm_cvtps_epi32(v3), _mm_cvtps_epi32(v4))
					));
				s += 16;
				d += 16;
			}
		}
		for (int i = 0; i < remsize; i++)
		{
			*d = (uchar)cvRound(a*focal_baseline / *s + b);
			s++;
			d++;
		}

#else
		Mat temp;
		divide(a*focal_baseline, src, temp);
		add(temp, b, temp);
		temp.convertTo(dest, CV_8U);
#endif
	}

	void depth16U2disp8U(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), CV_8U);
		if (dest.type() != CV_8U)dest = Mat::zeros(src.size(), CV_8U);

#if CV_SSE4_1
		const int ssesize = src.size().area() / 16;
		const int remsize = src.size().area() - 16 * ssesize;
		ushort* s = src.ptr<ushort>(0);
		uchar*  d = dest.ptr<uchar>(0);
		const __m128 maf = _mm_set1_ps(a*focal_baseline);
		if (b == 0.f)
		{
			for (int i = 0; i < ssesize; i++)
			{
				__m128i r0 = _mm_loadl_epi64((const __m128i*)(s));
				__m128i r1 = _mm_loadl_epi64((const __m128i*)(s + 4));

				__m128 v1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
				__m128 v2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

				r0 = _mm_loadl_epi64((const __m128i*)(s + 8));
				r1 = _mm_loadl_epi64((const __m128i*)(s + 12));
				__m128 v3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
				__m128 v4 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

				v1 = _mm_div_ps(maf, v1);
				v2 = _mm_div_ps(maf, v2);
				v3 = _mm_div_ps(maf, v3);
				v4 = _mm_div_ps(maf, v4);

				_mm_stream_si128((__m128i*)(d), _mm_packus_epi16(
					_mm_packs_epi32(_mm_cvtps_epi32(v1), _mm_cvtps_epi32(v2)),
					_mm_packs_epi32(_mm_cvtps_epi32(v3), _mm_cvtps_epi32(v4))
					));
				s += 16;
				d += 16;
			}
		}
		else
		{
			const __m128 mb = _mm_set1_ps(b);
			for (int i = 0; i < ssesize; i++)
			{
				__m128i r0 = _mm_loadl_epi64((const __m128i*)(s));
				__m128i r1 = _mm_loadl_epi64((const __m128i*)(s + 4));

				__m128 v1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
				__m128 v2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

				r0 = _mm_loadl_epi64((const __m128i*)(s + 8));
				r1 = _mm_loadl_epi64((const __m128i*)(s + 12));
				__m128 v3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
				__m128 v4 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

				v1 = _mm_add_ps(_mm_div_ps(maf, v1), mb);
				v2 = _mm_add_ps(_mm_div_ps(maf, v2), mb);
				v3 = _mm_add_ps(_mm_div_ps(maf, v3), mb);
				v4 = _mm_add_ps(_mm_div_ps(maf, v4), mb);

				_mm_stream_si128((__m128i*)(d), _mm_packus_epi16(
					_mm_packs_epi32(_mm_cvtps_epi32(v1), _mm_cvtps_epi32(v2)),
					_mm_packs_epi32(_mm_cvtps_epi32(v3), _mm_cvtps_epi32(v4))
					));
				s += 16;
				d += 16;
			}
		}
		for (int i = 0; i < remsize; i++)
		{
			*d = (uchar)cvRound(a*focal_baseline / *s + b);
			s++;
			d++;
		}

#else
		Mat temp;
		divide(a*focal_baseline, src, temp);
		add(temp, b, temp);
		temp.convertTo(dest, CV_8U);
#endif
	}

	void disp8U2depth32F(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), CV_32F);
		if (dest.type() != CV_32F)dest = Mat::zeros(src.size(), CV_32F);

#if CV_SSE4_1
		const int ssesize = src.size().area() / 16;
		const int remsize = src.size().area() - 16 * ssesize;

		uchar* s = src.ptr<uchar>(0);
		float*  d = dest.ptr<float>(0);

		const __m128 maf = _mm_set1_ps(a*focal_baseline);
		const __m128i zeros = _mm_setzero_si128();
		if (b == 0.f)
		{
			for (int i = 0; i < ssesize; i++)
			{
				__m128i r0 = _mm_load_si128((const __m128i*)(s));

				__m128i r1 = _mm_unpackhi_epi8(r0, zeros);
				r0 = _mm_unpacklo_epi8(r0, zeros);

				__m128i r2 = _mm_unpacklo_epi16(r0, zeros);
				__m128 v1 = _mm_cvtepi32_ps(r2);
				r2 = _mm_unpackhi_epi16(r0, zeros);
				__m128 v2 = _mm_cvtepi32_ps(r2);

				r2 = _mm_unpacklo_epi16(r1, zeros);
				__m128 v3 = _mm_cvtepi32_ps(r2);
				r2 = _mm_unpackhi_epi16(r1, zeros);
				__m128 v4 = _mm_cvtepi32_ps(r2);

				v1 = _mm_div_ps(maf, v1);
				v2 = _mm_div_ps(maf, v2);
				v3 = _mm_div_ps(maf, v3);
				v4 = _mm_div_ps(maf, v4);

				_mm_stream_ps((d), v1);
				_mm_stream_ps((d + 4), v2);
				_mm_stream_ps((d + 8), v3);
				_mm_stream_ps((d + 12), v4);

				s += 16;
				d += 16;
			}
		}
		else
		{
			cout << "not support" << endl;
			/*
		 const __m128 mb = _mm_set1_ps(b);
		 for(int i=0;i<ssesize;i++)
		 {
		 __m128i r0 = _mm_loadl_epi64((const __m128i*)(s));
		 __m128i r1 = _mm_loadl_epi64((const __m128i*)(s + 8));

		 __m128i r2 = _mm_unpacklo_epi8(r0,zeros);
		 __m128 v1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r2,r2), 16));
		 r2 = _mm_unpackhi_epi8(r0,zeros);
		 __m128 v2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r2,r2), 16));

		 r2 = _mm_unpacklo_epi8(r1,zeros);
		 __m128 v3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r2,r2), 16));
		 r2 = _mm_unpackhi_epi8(r1,zeros);
		 __m128 v4 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r2,r2), 16));

		 v1 = _mm_add_ps(_mm_div_ps(maf,v1),mb);
		 v2 = _mm_add_ps(_mm_div_ps(maf,v2),mb);
		 v3 = _mm_add_ps(_mm_div_ps(maf,v3),mb);
		 v4 = _mm_add_ps(_mm_div_ps(maf,v4),mb);

		 _mm_stream_ps((d),v1);
		 _mm_stream_ps((d+4),v2);
		 _mm_stream_ps((d+8),v3);
		 _mm_stream_ps((d+12),v4);

		 s+=16;
		 d+=16;
		 }*/
		}

		for (int i = 0; i < remsize; i++)
		{
			*d = a*focal_baseline / *s + b;
			s++;
			d++;
		}
#else
		src.convertTo(dest, CV_32F);
		divide(a*focal_baseline, dest, dest);
		add(dest, b, dest);
#endif
	}
}