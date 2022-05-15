#include "statistic.hpp"
#include "inlineSIMDFunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	void drawMinMaxPoints(InputArray src_, OutputArray dest, const uchar minv, const uchar maxv, Scalar minColor, Scalar maxColor, const int circle_r)
	{
		Mat src = src_.getMat();

		CV_Assert(src.channels() == 1);

		Mat s8u;
		src.convertTo(s8u, CV_8U);
		cvtColor(s8u, dest, COLOR_GRAY2BGR);
		Mat dst = dest.getMat();
		for (int j = 0; j < src.rows; j++)
		{
			const float* s = src.ptr<const float>(j);
			for (int i = 0; i < src.cols; i++)
			{
				if (s[i] == minv)
					circle(dst, Point(i, j), 3, minColor);
				if (s[i] == maxv)
					circle(dst, Point(i, j), 3, maxColor);
			}
		}
	}

	void calcMinMax(InputArray src_, uchar& minv, uchar& maxv)
	{
		Mat src = src_.getMat();
		const uchar* s = src.ptr<uchar>(0);
		const int size = src.size().area();
		const int simdsize = get_simd_floor(size, 128);

		__m256i minvec = _mm256_set1_epi8(0xFF);
		__m256i maxvec = _mm256_setzero_si256();
		for (int i = 0; i < simdsize; i += 128)
		{
			__m256i v1 = _mm256_load_si256((__m256i*)(s + i+0));
			__m256i v2 = _mm256_load_si256((__m256i*)(s + i+32));
			__m256i v3 = _mm256_load_si256((__m256i*)(s + i+64));
			__m256i v4 = _mm256_load_si256((__m256i*)(s + i+96));
			maxvec = _mm256_max_epu8(maxvec, v1);
			minvec = _mm256_min_epu8(minvec, v1);
			maxvec = _mm256_max_epu8(maxvec, v2);
			minvec = _mm256_min_epu8(minvec, v2);
			maxvec = _mm256_max_epu8(maxvec, v3);
			minvec = _mm256_min_epu8(minvec, v3);
			maxvec = _mm256_max_epu8(maxvec, v4);
			minvec = _mm256_min_epu8(minvec, v4);
		}

		uchar lminv = 255;
		uchar lmaxv = 0;
		for (int i = 0; i < 32; i++)
		{
			lmaxv = max(lmaxv, maxvec.m256i_u8[i]);
			lminv = min(lminv, minvec.m256i_u8[i]);
		}
		for (int i = simdsize; i < size; i++)
		{
			lmaxv = max(lmaxv, s[i]);
			lminv = min(lminv, s[i]);
		}

		minv = lminv;
		maxv = lmaxv;
	}
}