#include "arithmetic.hpp"
#include "fmath.hpp"
using namespace std;
using namespace cv;
using namespace fmath;

namespace cp
{
// if you do not use fmath.hpp
/*
// Fast SSE pow for range [0, 1]
// Adapted from C. Schlick with one more iteration each for exp(x) and ln(x)
// 8 muls, 5 adds, 1 rcp
inline __m128 _mm_pow01_ps(__m128 x, __m128 y)
{
static const __m128 fourOne = _mm_set1_ps(1.0f);
static const __m128 fourHalf = _mm_set1_ps(0.5f);

__m128 a = _mm_sub_ps(fourOne, y);
__m128 b = _mm_sub_ps(x, fourOne);
__m128 aSq = _mm_mul_ps(a, a);
__m128 bSq = _mm_mul_ps(b, b);
__m128 c = _mm_mul_ps(fourHalf, bSq);
__m128 d = _mm_sub_ps(b, c);
__m128 dSq = _mm_mul_ps(d, d);
__m128 e = _mm_mul_ps(aSq, dSq);
__m128 f = _mm_mul_ps(a, d);
__m128 g = _mm_mul_ps(fourHalf, e);
__m128 h = _mm_add_ps(fourOne, f);
__m128 i = _mm_add_ps(h, g);
__m128 iRcp = _mm_rcp_ps(i);
//__m128 iRcp = _mm_rcp_22bit_ps(i);
__m128 result = _mm_mul_ps(x, iRcp);

return result;
}
#define _mm_pow_ps _mm_pow01_ps
*/

inline __m128 _mm_pow_ps(__m128 a, __m128 b)
{
	return exp_ps(_mm_mul_ps(b, log_ps(a)));
}


void pow_fmath(const float a, const Mat& src, Mat & dest)
{
	if (dest.empty())dest.create(src.size(), CV_32F);

	int width = src.cols;
	int height = src.rows;

	int size = src.size().area();
	int i = 0;

	const float* s = src.ptr<float>(0);
	float* d = dest.ptr<float>(0);
	const __m128 ma = _mm_set1_ps(a);
	for (i = 0; i <= size - 4; i += 4)
	{
		_mm_store_ps(d + i, _mm_pow_ps(ma, _mm_load_ps(s + i)));
	}
	for (; i < size; i++)
	{
		d[i] = cv::pow(a, s[i]);
	}
}

void pow_fmath(const Mat& src, const float a, Mat & dest)
{
	if (dest.empty())dest.create(src.size(), CV_32F);

	int width = src.cols;
	int height = src.rows;

	int size = src.size().area();
	int i = 0;

	const float* s = src.ptr<float>(0);
	float* d = dest.ptr<float>(0);
	const __m128 ma = _mm_set1_ps(a);
	for (i = 0; i <= size - 4; i += 4)
	{
		_mm_store_ps(d + i, _mm_pow_ps(_mm_load_ps(s + i), ma));
	}
	for (; i < size; i++)
	{
		d[i] = cv::pow(s[i], a);
	}
}

void pow_fmath(const Mat& src1, const Mat& src2, Mat & dest)
{
	if (dest.empty())dest.create(src1.size(), CV_32F);

	int width = src1.cols;
	int height = src1.rows;

	int size = src1.size().area();
	int i = 0;

	const float* s1 = src1.ptr<float>(0);
	const float* s2 = src2.ptr<float>(0);
	float* d = dest.ptr<float>(0);

	for (i = 0; i <= size - 4; i += 4)
	{
		_mm_store_ps(d + i, _mm_pow_ps(_mm_load_ps(s1 + i), _mm_load_ps(s2 + i)));
	}
	for (; i < size; i++)
	{
		d[i] = cv::pow(s1[i], s2[i]);
	}
}

void compareRange(InputArray src, OutputArray destMask, const double validMin, const double validMax)
{
	Mat gray;
	if (src.channels() == 1) gray = src.getMat();
	else cvtColor(src, gray, COLOR_BGR2GRAY);

	Mat mask1;
	Mat mask2;
	compare(gray, validMin, mask1, cv::CMP_GE);
	compare(gray, validMax, mask2, cv::CMP_LE);
	bitwise_and(mask1, mask2, destMask);
}

void setTypeMaxValue(InputOutputArray src)
{
	Mat s = src.getMat();
	if (s.depth() == CV_8U)s.setTo(UCHAR_MAX);
	else if (s.depth() == CV_16U)s.setTo(USHRT_MAX);
	else if (s.depth() == CV_16S)s.setTo(SHRT_MAX);
	else if (s.depth() == CV_32S)s.setTo(INT_MAX);
	else if (s.depth() == CV_32F)s.setTo(FLT_MAX);
	else if (s.depth() == CV_64F)s.setTo(DBL_MAX);
}

void setTypeMinValue(InputOutputArray src)
{
	Mat s = src.getMat();
	if (s.depth() == CV_8U)s.setTo(0);
	else if (s.depth() == CV_16U)s.setTo(0);
	else if (s.depth() == CV_16S)s.setTo(SHRT_MIN);
	else if (s.depth() == CV_32S)s.setTo(INT_MIN);
	else if (s.depth() == CV_32F)s.setTo(FLT_MIN);
	else if (s.depth() == CV_64F)s.setTo(DBL_MIN);
}

}
