#include "windowFunction.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
namespace cp
{
	cv::Mat createWindowFunction(const int r, const double sigma, int depth, const int window_type, const bool isSeparable)
	{
		CV_Assert(depth == CV_32F || depth == CV_64F);

		const int D = 2 * r + 1;
		Mat ret;
		if (isSeparable)
		{
			Mat sep(D, 1, depth);

			if (depth == CV_32F)
			{
				for (int i = -r; i <= r; i++)
				{
					sep.at<float>(i + r) = (float)getRangeKernelFunction(i, sigma, window_type);
				}
			}
			else if (depth == CV_32F)
			{
				for (int i = -r; i <= r; i++)
				{
					sep.at<double>(i + r) = getRangeKernelFunction(i, sigma, window_type);
				}
			}
			ret = sep * sep.t();
		}
		else
		{
			ret.create(D, D, depth);
			if (depth == CV_32F)
			{
				for (int j = -r; j <= r; j++)
				{
					for (int i = -r; i <= r; i++)
					{
						double d = sqrt(i * i + j * j);
						ret.at<float>(j + r, i + r) = (float)getRangeKernelFunction(d, sigma, window_type);
					}
				}
			}
			else if (depth == CV_32F)
			{
				for (int j = -r; j <= r; j++)
				{
					for (int i = -r; i <= r; i++)
					{
						double d = sqrt(i * i + j * j);
						ret.at<double>(j + r, i + r) = getRangeKernelFunction(d, sigma, window_type);
					}
				}
			}
		}

		return ret;
	}
}
