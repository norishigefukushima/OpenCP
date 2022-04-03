#pragma once

#include "common.hpp"

namespace cp
{
	//R. Bridson "Fast Poisson Disk Sampling in Arbitrary Dimensions," Proc. ACM SIGGRAPH sketches, 2007
	class CP_EXPORT PoissonDiskSampling
	{
	private:
		int grid_width;
		int grid_height;
		float cell_size_inv;
		float min_d;
		float min_d2;
		const int k = 30;
		cv::Mat background_grid_pt;
		std::vector<cv::Point> pointAroundCandidate;

		inline void set(const cv::Point pt);
		inline cv::Point imageToGrid(cv::Point pt);
		inline float getDistance(const cv::Point pt1, const cv::Point pt2);
		inline cv::Point generateRandomPointAround(const cv::Point pt, cv::RNG& rng);

		virtual bool isAvailable(const cv::Point pt);
		virtual cv::Point initializeStart(cv::RNG& rng);

	protected:
		cv::Size imageSize;
		inline bool inImage(const cv::Point pt);
		inline bool inNeibourhood(const cv::Point pt);

	public:
		PoissonDiskSampling(const float min_d, const cv::Size imageSize);
		virtual int generate(cv::Mat& mask, cv::RNG& rng, const cv::Point start = cv::Point(-1, -1), const int max_sample = -1);		
	};
}