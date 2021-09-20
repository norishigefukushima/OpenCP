#include "bitconvertDD.hpp"
#include <intrin.h>
using namespace cv;

namespace cp
{
	void convertDDTo(const doubledouble* src, Mat& dest)
	{
		CV_Assert(!dest.empty());
		convertDDTo(src, dest.size(), dest, dest.depth());
	}

	void convertDDTo(const doubledouble* src, const Size size, Mat& dest, const int depth)
	{
		Mat s(size.height, size.width, CV_64FC2, (void*)src);
		std::vector<Mat> ss;
		split(s, ss);
		ss[1].convertTo(dest, depth);
	}

	void convertToDD(const Mat& src, doubledouble* dest)
	{
		CV_Assert(dest != nullptr);

		if (src.depth() == CV_8U)
		{
			for (int i = 0; i < src.size().area(); i++)
			{
				dest[i] = { (double)src.at<uchar>(i),0.0 };
			}
		}
		else if (src.depth() == CV_16S)
		{
			for (int i = 0; i < src.size().area(); i++)
			{
				dest[i] = { (double)src.at<short>(i),0.0 };
			}
		}
		else if (src.depth() == CV_16U)
		{
			for (int i = 0; i < src.size().area(); i++)
			{
				dest[i] = { (double)src.at<ushort>(i),0.0 };
			}
		}
		else if (src.depth() == CV_32S)
		{
			for (int i = 0; i < src.size().area(); i++)
			{
				dest[i] = { (double)src.at<int>(i),0.0 };
			}
		}
		else if (src.depth() == CV_32F)
		{
			for (int i = 0; i < src.size().area(); i++)
			{
				dest[i] = { (double)src.at<float>(i),0.0 };
			}
		}
		else if (src.depth() == CV_64F)
		{
			for (int i = 0; i < src.size().area(); i++)
			{
				dest[i] = { src.at<double>(i),0.0 };
			}
		}
	}
}
