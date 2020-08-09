#pragma once
#include "common.hpp"

namespace cp
{

	CP_EXPORT void splitCopyMakeBorder(cv::InputArray src, cv::OutputArrayOfArrays dest, const int top, const int bottom, const int left, const int right, const int borderType, const cv::Scalar& color= cv::Scalar());
	CP_EXPORT void copyMakeBorderReplicate(cv::InputArray src, cv::OutputArray dest, const int top, const int bottom, const int left, const int right);	

}
