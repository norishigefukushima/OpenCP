#pragma once

#include "common.hpp"

namespace cp
{
	//ImwriteFlags::IMWRITE_WEBP_QUALITY=64
#define IMWRITE_WEBP_METHOD 65 //0(fast)-6(slower, better), default 4
#define IMWRITE_WEBP_COLORSPACE 66 //0: YUV, 1: YUV_SHARP, 2: RGB
#define IMWRITE_WEBP_LOOPCOUNT 67 //0: infinit
#define IMWRITE_WEBP_TIMEMSPERFRAME 68//time(ms) per frame default(33)

	//return writting data size
	CP_EXPORT int imwriteAnimationWebp(std::string name, std::vector<cv::Mat>& src, const std::vector<int>& parameters = std::vector<int>());
}
