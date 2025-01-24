#pragma once

#include "common.hpp"
#define USE_OPENCP_AVIF 1
namespace cp
{
	CP_EXPORT void addNoise(cv::InputArray src, cv::OutputArray dest, const double sigma, const double solt_papper_ratio = 0.0, const uint64 seed = 0);

	//return bpp
	CP_EXPORT double addJPEGNoise(cv::InputArray src, cv::OutputArray dest, const int quality);
	//return bpp
	CP_EXPORT double addJPEG2000Noise(cv::InputArray src, cv::OutputArray dest, const int quality);
	//return bpp, method: fast(0)-slow(6) default(4), colorSpace yuv(0) yuv_sharp(1) rgb(2)
	CP_EXPORT double addWebPNoise(cv::InputArray src, cv::OutputArray dest, const int quality, const int method = 4, const int colorSpace = 0);
#ifdef USE_OPENCP_AVIF
	//return bpp, method: fast(9)-slow(0) default(6)
	CP_EXPORT double addAVIFNoise(cv::InputArray src, cv::OutputArray dest, const int quality, const int method = 6);
#endif
}