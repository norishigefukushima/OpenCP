#pragma once

#include "common.hpp"

namespace cp
{
	enum class HDGFSchedule
	{
		COMPUTE,
		LUT_SQRT
	};
	CP_EXPORT void highDimensionalGaussianFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border = cv::BORDER_DEFAULT, HDGFSchedule schedule = HDGFSchedule::COMPUTE);
	CP_EXPORT void highDimensionalGaussianFilter(cv::InputArray src, cv::InputArray guide, cv::InputArray center,cv::OutputArray dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border = cv::BORDER_DEFAULT, HDGFSchedule schedule = HDGFSchedule::COMPUTE);

	class CP_EXPORT TileHDGF
	{
	private:
		const int thread_max;
		cv::Size div;
		cv::Size divImageSize;
		cv::Size tileSize;
		int boundaryLength = 0;
		std::vector<cv::Mat> split_src, split_dst, subImageOutput;
		std::vector<std::vector<cv::Mat>> subImageInput;
		std::vector<std::vector<cv::Mat>> subImageGuide;
		std::vector<std::vector<cv::Mat>> subImageBuff;

		std::vector<cv::Mat> srcSplit;
		std::vector<cv::Mat> guideSplit;
		int channels = 3;
		int guide_channels = 3;

	public:
		TileHDGF(cv::Size div);
		~TileHDGF();

		void nlmFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, const int patch_r, const int reduced_dim, const int pca_method, double truncateBoundary = 3.0, const int borderType = cv::BORDER_DEFAULT);
		void pcaFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, const int reduced_dim, double truncateBoundary = 3.0);
		void cvtgrayFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, const int method, double truncateBoundary = 3.0);
		void filter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, double truncateBoundary = 3.0);

		cv::Size getTileSize();
		void getTileInfo();
	};
}