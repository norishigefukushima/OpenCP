#include "count.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	int countDenormalizedNumber(InputArray src_)
	{
		Mat src = src_.getMat();
		CV_Assert(src.depth() == CV_32F || src.depth() == CV_64F);

		int ret = 0;
		const int size = src.size().area()*src.channels();
		if (src.depth() == CV_32F)
		{
			float* s = src.ptr<float>();
			for (int i = 0; i < src.size().area(); i++)
			{
				if (fpclassify(s[i]) == FP_SUBNORMAL)
				{
					ret++;
				}
			}
		}
		else if (src.depth() == CV_64F)
		{
			double* s = src.ptr<double>();
			for (int i = 0; i < src.size().area(); i++)
			{
				if (fpclassify(s[i]) == FP_SUBNORMAL)
				{
					ret++;
				}
			}
		}

		return ret;
	}

	double countDenormalizedNumberRatio(InputArray src_)
	{
		Mat src = src_.getMat();
		return countDenormalizedNumber(src) / (double)src.size().area();
	}

	int countNaN(InputArray src_)
	{
		Mat src = src_.getMat();
		CV_Assert(src.depth() == CV_32F || src.depth() == CV_64F);

		int ret = 0;
		const int size = src.size().area()*src.channels();
		if (src.depth() == CV_32F)
		{
			float* s = src.ptr<float>();
			for (int i = 0; i < src.size().area(); i++)
			{
				ret += cvIsNaN(s[i]);
			}
		}
		else if (src.depth() == CV_64F)
		{
			double* s = src.ptr<double>();
			for (int i = 0; i < src.size().area(); i++)
			{
				ret += cvIsNaN(s[i]);
			}
		}

		return ret;
	}

	int countInf(InputArray src_)
	{
		Mat src = src_.getMat();
		CV_Assert(src.depth() == CV_32F || src.depth() == CV_64F);

		int ret = 0;
		const int size = src.size().area()*src.channels();
		if (src.depth() == CV_32F)
		{
			float* s = src.ptr<float>();
			for (int i = 0; i < size; i++)
			{
				ret += cvIsInf(s[i]);
			}
		}
		else if (src.depth() == CV_64F)
		{
			double* s = src.ptr<double>();
			for (int i = 0; i < size; i++)
			{
				ret += cvIsInf(s[i]);
			}
		}
		return ret;
	}
}