#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void blurRemoveMinMax(const cv::Mat& src, cv::Mat& dest, const int r);
	//MORPH_RECT=0, MORPH_CROSS=1, MORPH_ELLIPSE
	CP_EXPORT void maxFilter(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, int shape = cv::MORPH_RECT);
	CP_EXPORT void maxFilter(cv::InputArray src, cv::OutputArray dest, int radius);
	//MORPH_RECT=0, MORPH_CROSS=1, MORPH_ELLIPSE
	CP_EXPORT void minFilter(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, int shape = cv::MORPH_RECT);
	CP_EXPORT void minFilter(cv::InputArray src, cv::OutputArray dest, int radius);

	
	/// <summary>
	/// computing local variance in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="kernelSize">kernelShape (rectangle kernel)</param>
	CP_EXPORT void varianceFilter(cv::InputArray src, cv::OutputArray dest, const cv::Size kernelSize);

	/// <summary>
	/// computing local variance in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="radius">radius (square kernel)</param>
	CP_EXPORT void varianceFilter(cv::InputArray src, cv::OutputArray dest, const int radius);

	/// <summary>
	/// computing standard deviation in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="kernelSize">kernelShape (rectangle kernel)</param>
	CP_EXPORT void stdFilter(cv::InputArray src, cv::OutputArray dest, const cv::Size kernelSize);

	/// <summary>
	/// computing standard deviation in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="radius">radius (square kernel)</param>
	CP_EXPORT void stdFilter(cv::InputArray src, cv::OutputArray dest, const int radius);


	/// <summary>
	/// computing local mean and variance in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="mean"> mean output</param>
	/// <param name="variance"> variance output</param>
	/// <param name="kernelSize">kernelShape (rectangle kernel)</param>
	CP_EXPORT void meanVarianceFilter(cv::InputArray src, cv::OutputArray mean, cv::OutputArray variance, const cv::Size kernelSize);

	/// <summary>
	/// computing local mean and variance in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="mean"> mean output</param>
	/// <param name="variance"> variance output</param>
	/// <param name="kernelSize">kernelShape (square kernel)</param>
	CP_EXPORT void meanVarianceFilter(cv::InputArray src, cv::OutputArray mean, cv::OutputArray variance, const int radius);

	/// <summary>
	/// computing local mean and standard deviation in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="mean"> mean output</param>
	/// <param name="std"> standard deviation output</param>
	/// <param name="kernelSize">kernelShape (rectangle kernel)</param>
	CP_EXPORT void meanStdFilter(cv::InputArray src, cv::OutputArray mean, cv::OutputArray std, const cv::Size kernelSize);

	/// <summary>
	/// computing local mean and standard deviation in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="mean"> mean output</param>
	/// <param name="std"> standard deviation output</param>
	/// <param name="kernelSize">kernelShape (square kernel)</param>
	/// 
	CP_EXPORT void meanStdFilter(cv::InputArray src, cv::OutputArray mean, cv::OutputArray std, const int radius);

	/// <summary>
	/// computing local Gaussian weighted variance in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="kernelSize">kernelShape (rectangle kernel)</param>
	/// <param name="sigma">Gaussian sigma</param>
	CP_EXPORT void varianceFilterGaussian(cv::InputArray src, cv::OutputArray dest, const cv::Size kernelSize, const double sigma);

	/// <summary>
	/// computing local Gaussian weighted variance in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="radius">radius (square kernel)</param>
	/// <param name="sigma">Gaussian sigma</param>
	CP_EXPORT void varianceFilterGaussian(cv::InputArray src, cv::OutputArray dest, const int radius, const double sigma);

	/// /// <summary>
	/// computing local Gaussian weighted standard deviation in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="kernelSize">kernelShape (rectangle kernel)</param>
	/// <param name="sigma">Gaussian sigma</param>
	CP_EXPORT void stdFilterGaussian(cv::InputArray src, cv::OutputArray dest, const cv::Size kernelSize, const double sigma);

	/// <summary>
	/// computing local Gaussian weighted standard deviation in the kernel.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="radius">radius (square kernel)</param>
	/// <param name="sigma">Gaussian sigma</param>
	CP_EXPORT void stdFilterGaussian(cv::InputArray src, cv::OutputArray dest, const int radius, const double sigma);
}