#pragma once
#include "common.hpp"

namespace cp
{

	//The splitCopyMakeBorder function is semi-optimal. The function can be more optimized.
	CP_EXPORT void splitCopyMakeBorder(cv::InputArray src, cv::OutputArrayOfArrays dest, const int top, const int bottom, const int left, const int right, const int borderType, const cv::Scalar& color= cv::Scalar());
	CP_EXPORT void copyMakeBorderReplicate(cv::InputArray src, cv::OutputArray dest, const int top, const int bottom, const int left, const int right);	

	inline int border_replicate(int p, int max_val)
	{
		return std::min(std::max(0, p), max_val);
	}

	inline int border_min_replicate(int p, int max_val)
	{
		return std::max(0, p);
	}

	inline int border_max_replicate(int p, int max_val)
	{
		return std::min(p, max_val);
	}

	inline int border_reflect101(int p, int max_val)
	{
		if (p < 0)return -p;
		if (p > max_val) return 2 * max_val - p;
		return p;
	}

	inline int border_min_reflect101(int p)
	{
		return (p < 0) ? -p : p;
	}

	inline int border_max_reflect101(int p, int max_val)
	{
		return (p > max_val) ? 2 * max_val - p : p;
	}

	inline int border_reflect(int p, int max_val)
	{
		if (p < 0) return -(p + 1);
		if (p > max_val) return 2 * max_val - (p - 1);
		return p;
	}

	inline int border_min_reflect(int p)
	{
		return (p < 0) ? -(p + 1) : p;
	}

	inline int border_max_reflect(int p, int max_val)
	{
		return (p > max_val) ? 2 * max_val - (p - 1) : p;
	}

}
