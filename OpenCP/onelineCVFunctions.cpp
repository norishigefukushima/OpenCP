#include "onelineCVFunctions.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	cv::Mat convert(cv::Mat& src, const int depth, const double alpha, const double beta)
	{
		cv::Mat ret;
		src.convertTo(ret, depth, alpha, beta);
		return ret;
	}

	cv::Mat cenvertCentering(cv::InputArray src, int depth, double a, double b)
	{
		Mat ret;
		src.getMat().convertTo(ret, depth, a, -a * b + b);
		return ret;
	}

	Mat border(Mat& src, const int r, const int borderType)
	{
		Mat ret;
		copyMakeBorder(src, ret, r, r, r, r, borderType);
		return ret;
	}

	Mat border(Mat& src, const int top, const int bottom, const int left, const int right, const int borderType)
	{
		Mat ret;
		copyMakeBorder(src, ret, top, bottom, left, right, borderType);
		return ret;
	}
}