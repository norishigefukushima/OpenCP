#include "onelineCVFunctions.hpp"
#include "inlineSIMDFunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{	
	inline cv::Mat convert(cv::InputArray& src, const int depth, const double alpha, const double beta)
	{
		cv::Mat ret;
		src.getMat().convertTo(ret, depth, alpha, beta);
		return ret;
	}

	cv::Mat cenvertCentering(cv::InputArray src, int depth, double a, double b)
	{
		cv::Mat ret;
		src.getMat().convertTo(ret, depth, a, -a * b + b);
		return ret;
	}

	cv::Mat convertGray(cv::InputArray& src, const int depth, const double alpha, const double beta)
	{
		cv::Mat ret;
		cv::Mat s = src.getMat();
		if(s.channels()==1)s.convertTo(ret, depth, alpha, beta);
		else
		{
			cv::cvtColor(s, ret, COLOR_BGR2GRAY);
			ret.convertTo(ret, depth, alpha, beta);
		}
		
		return ret;
	}

	cv::Mat border(cv::Mat& src, const int r, const int borderType)
	{
		cv::Mat ret;
		cv::copyMakeBorder(src, ret, r, r, r, r, borderType);
		return ret;
	}

	cv::Mat border(cv::Mat& src, const int top, const int bottom, const int left, const int right, const int borderType)
	{
		cv::Mat ret;
		cv::copyMakeBorder(src, ret, top, bottom, left, right, borderType);
		return ret;
	}

	void printMinMax(cv::InputArray src)
	{
		double minv, maxv;
		minMaxLoc(src, &minv, &maxv);
		std::cout << minv << "," << maxv << endl;
	}
}