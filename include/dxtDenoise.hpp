#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT DenoiseDXTShrinkage
	{
	private:
		enum
		{
			DenoiseDCT = 0,
			DenoiseDHT = 1,
			DenoiseDWT = 2//not supported
		};

		int basis;
		cv::Size patch_size;
		cv::Size size;
		cv::Mat buff;
		cv::Mat sum;

		cv::Mat im;

		int channel;
		void body(float *src0, float* dest0, float *src1, float* dest1, float *src2, float* dest2, float Th);

		void body(float *src, float* dest, float Th);
		void body(float *src, float* dest, float* wmap, float Th);

		void bodyTest(float *src, float* dest, float Th);

		void body(float *src, float* dest, float Th, int dr);

		void div(float* inplace0, float* inplace1, float* inplace2, float* w0, float* w1, float* w2, const int size1);
		void div(float* inplace0, float* inplace1, float* inplace2, const int patch_area, const int size1);

		void div(float* inplace0, float* w0, const int size1);
		void div(float* inplace0, const int patch_area, const int size1);

		void decorrelateColorForward(float* src, float* dest, int width, int height);
		void decorrelateColorInvert(float* src, float* dest, int width, int height);

	public:
		bool isSSE;
		void cvtColorOrder32F_BGR2BBBBGGGGRRRR(const cv::Mat& src, cv::Mat& dest);
		void cvtColorOrder32F_BBBBGGGGRRRR2BGR(const cv::Mat& src, cv::Mat& dest);

		void init(cv::Size size_, int color_, cv::Size patch_size_);
		DenoiseDXTShrinkage(cv::Size size, int color, cv::Size patch_size_ = cv::Size(8, 8));
		DenoiseDXTShrinkage();
		void operator()(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), int transform_basis = 0);

		void shearable(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), int transform_basis = 0, int direct = 0);
		void weighted(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), int transform_basis = 0);

		void test(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8));
	};
}