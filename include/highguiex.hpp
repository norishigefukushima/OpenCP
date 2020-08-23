#pragma once

#include "common.hpp"

namespace cp
{

	CP_EXPORT cv::Size getTextSizeQt(std::string message, std::string font, const int fontSize);
	CP_EXPORT cv::Mat getTextImageQt(std::string message, std::string font, const int fontSize, cv::Scalar text_color = cv::Scalar::all(0), cv::Scalar background_color = cv::Scalar(255, 255, 255, 0), bool isItalic = false);
}
