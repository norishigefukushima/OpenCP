#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void pow_fmath(const float a, const cv::Mat& src, cv::Mat& dest);
	CP_EXPORT void pow_fmath(const cv::Mat& src, const float a, cv::Mat& dest);
	CP_EXPORT void pow_fmath(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dest);
	CP_EXPORT void compareRange(cv::InputArray src, cv::OutputArray destMask, const double validMin, const double validMax);
	CP_EXPORT void setTypeMaxValue(cv::InputOutputArray src);
	CP_EXPORT void setTypeMinValue(cv::InputOutputArray src);

	//a*x+b
	CP_EXPORT void fmadd(cv::Mat& a, cv::Mat& x, cv::Mat& b, cv::Mat& dest);
	//a*x-b
	CP_EXPORT void fmsub(cv::Mat& a, cv::Mat& x, cv::Mat& b, cv::Mat& dest);
	//-a*x+b
	CP_EXPORT void fnmadd(cv::Mat& a, cv::Mat& x, cv::Mat& b, cv::Mat& dest);
	//-a*x-b
	CP_EXPORT void fnmsub(cv::Mat& a, cv::Mat& x, cv::Mat& b, cv::Mat& dest);

	//dest=src>>shift, lostbit=src-dest<<shift
	CP_EXPORT void bitshiftRight(cv::InputArray src, cv::OutputArray dest, cv::OutputArray lostbit, const int shift);
	//src>>shift
	CP_EXPORT void bitshiftRight(cv::InputArray src, cv::OutputArray dest, const int shift);

	CP_EXPORT double average(const cv::Mat& src, const int left = 0, const int right = 0, const int top = 0, const int bottom = 0, const bool isNormalize = true);
	CP_EXPORT void average_variance(const cv::Mat& src, double& ave, double& var, const int left = 0, const int right = 0, const int top = 0, const int bottom = 0, const bool isNormalize = true);

	CP_EXPORT void clip(cv::InputArray src, cv::OutputArray dst, const double minval=0.0, const double maxval=255.0);

	//sqrt(max(src, 0))
	CP_EXPORT void sqrtZeroClip(cv::InputArray src, cv::OutputArray dest);
	//sign(src)*pow(abs(src),v)
	CP_EXPORT void powsign(cv::InputArray src, const float v, cv::OutputArray dest);
	//pow(max(src,0),v)
	CP_EXPORT void powZeroClip(cv::InputArray src, const float v, cv::OutputArray dest);
}
