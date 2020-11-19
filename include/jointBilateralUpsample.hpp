#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void nnUpsample(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void linearUpsample(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void cubicUpsample(cv::InputArray src, cv::OutputArray dest, const double a = -1.0);
	CP_EXPORT void setUpsampleMask(cv::InputArray src, cv::OutputArray dst);

	enum class JBUSchedule
	{
		CLASS,
		ALLOC_BORDER_OMP,
		COMPUTE_BORDER_OMP,
		COMPUTE_BORDER_NODOWNSAMPLE_OMP,

		SIZE
	};
	CP_EXPORT void jointBilateralUpsampe(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int r, const double sigma_r, const double sigma_s, const JBUSchedule schedule = JBUSchedule::CLASS);

	class CP_EXPORT JointBilateralUpsample
	{
		cv::Mat src_b;
		cv::Mat guide_low_b;
		cv::Mat guide_low;
		cv::Mat weightmap;
	public:
		void upsample(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int r, const double sigma_r, const double sigma_s);
		void upsample64F(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int r, const double sigma_r, const double sigma_s);
	};
}