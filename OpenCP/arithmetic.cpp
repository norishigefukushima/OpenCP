#include "opencp.hpp"
#include <opencv2/core/internal.hpp>

#include "fmath.hpp"
using namespace fmath;

inline __m128 _mm_pow_ps(__m128 a, __m128 b)
{
	return exp_ps(_mm_mul_ps(b,log_ps(a)));
}

void pow_fmath(const float a , const Mat& src, Mat & dest)
{
	if(dest.empty())dest.create(src.size(),CV_32F);

	int width = src.cols;
	int height = src.rows;

	int size  =src.size().area();
	int i=0;

	const float* s = src.ptr<float>(0);
	float* d = dest.ptr<float>(0);
	const __m128 ma = _mm_set1_ps(a);
	for(i=0;i<=size-4;i+=4)
	{
		_mm_store_ps(d+i, _mm_pow_ps(ma, _mm_load_ps(s+i)));
	}
	for(;i<size;i++)
	{
		d[i] = cv::pow(a, s[i]); 
	}
}

void pow_fmath(const Mat& src,const float a , Mat & dest)
{
	if(dest.empty())dest.create(src.size(),CV_32F);

	int width = src.cols;
	int height = src.rows;

	int size  =src.size().area();
	int i=0;

	const float* s = src.ptr<float>(0);
	float* d = dest.ptr<float>(0);
	const __m128 ma = _mm_set1_ps(a);
	for(i=0;i<=size-4;i+=4)
	{
		_mm_store_ps(d+i, _mm_pow_ps(_mm_load_ps(s+i), ma));
	}
	for(;i<size;i++)
	{
		d[i] = cv::pow(s[i],a); 
	}
}

void pow_fmath(const Mat& src1, const Mat& src2, Mat & dest)
{
	if(dest.empty())dest.create(src1.size(),CV_32F);

	int width = src1.cols;
	int height = src1.rows;

	int size  =src1.size().area();
	int i=0;

	const float* s1 = src1.ptr<float>(0);
	const float* s2 = src2.ptr<float>(0);
	float* d = dest.ptr<float>(0);
	
	for(i=0;i<=size-4;i+=4)
	{
		_mm_store_ps(d+i, _mm_pow_ps(_mm_load_ps(s1+i), _mm_load_ps(s2+i)));
	}
	for(;i<size;i++)
	{
		d[i] = cv::pow(s1[i], s2[i]); 
	}
}
