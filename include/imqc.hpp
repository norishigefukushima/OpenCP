#pragma once

#include "common.hpp"

namespace cp
{
	enum
	{
		IQM_PSNR = 0,
		IQM_MSE, //MSE
		IQM_MSAD, //mean SAD
		IQM_DELTA,
		IQM_SSIM,
		IQM_SSIM_FAST,//only cropping processing area 
		IQM_SSIM_MODIFY,//only changed alpha beta gamma parameter of SSIM
		IQM_SSIM_FASTMODIFY,
		IQM_CWSSIM,
		IQM_CWSSIM_FAST,
		IQM_MSSSIM,
		IQM_MSSSIM_FAST
	};
	CP_EXPORT double calcImageQualityMetric(cv::InputArray src, cv::InputArray target, const int metric = IQM_PSNR, const int boundingBox = 0);

}