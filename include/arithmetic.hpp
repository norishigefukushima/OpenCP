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
	CP_EXPORT void fmadd(const cv::Mat& a, const cv::Mat& x, const cv::Mat& b, cv::Mat& dest);
	//a*x-b
	CP_EXPORT void fmsub(const cv::Mat& a, const cv::Mat& x, const cv::Mat& b, cv::Mat& dest);
	//-a*x+b
	CP_EXPORT void fnmadd(const cv::Mat& a, const cv::Mat& x, const cv::Mat& b, cv::Mat& dest);
	//-a*x-b
	CP_EXPORT void fnmsub(const cv::Mat& a, const cv::Mat& x, const cv::Mat& b, cv::Mat& dest);

	//dest=src>>shift, lostbit=src-dest<<shift
	CP_EXPORT void bitshiftRight(cv::InputArray src, cv::OutputArray dest, cv::OutputArray lostbit, const int shift);
	//src>>shift (if shift==0, there is no processing)
	CP_EXPORT void bitshiftRight(cv::InputArray src, cv::OutputArray dest, const int shift);

	CP_EXPORT void clip(cv::InputArray src, cv::OutputArray dst, const double minval = 0.0, const double maxval = 255.0);

	//dest=(src1-src2)^2
	CP_EXPORT void squareDiff(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dest);

	//sqrt(max(src, 0))
	CP_EXPORT void sqrtZeroClip(cv::InputArray src, cv::OutputArray dest);
	//sign(src)*pow(abs(src),v)
	CP_EXPORT void powsign(cv::InputArray src, const float v, cv::OutputArray dest);
	//pow(max(src,0),v)
	CP_EXPORT void powZeroClip(cv::InputArray src, const float v, cv::OutputArray dest);
}
