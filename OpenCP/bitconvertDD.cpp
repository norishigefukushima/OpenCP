#include "bitconvertDD.hpp"
#include <intrin.h>
using namespace cv;

namespace cp
{
	void cvtDDtoMat(const doubledouble* src, Mat& dest)
	{
		CV_Assert(!dest.empty());
		cvtDDtoMat(src, dest.size(), dest, dest.depth());
	}

	void cvtDDtoMat(const doubledouble* src, const Size size, Mat& dest, const int depth)
	{
		dest.create(size, depth);

		if (dest.depth() == CV_8U)
		{
			for (int i = 0; i < dest.size().area(); i++)
			{
				dest.at<uchar>(i) = saturate_cast<uchar>(src[i].lo);
			}
		}
		else if (depth == CV_16U)
		{
			for (int i = 0; i < dest.size().area(); i++)
			{
				dest.at<ushort>(i) = saturate_cast<ushort>(src[i].lo);
			}
		}
		else if (depth == CV_16S)
		{
			for (int i = 0; i < dest.size().area(); i++)
			{
				dest.at<short>(i) = saturate_cast<short>(src[i].lo);
			}
		}
		else if (depth == CV_32S)
		{
			for (int i = 0; i < dest.size().area(); i++)
			{
				dest.at<int>(i) = saturate_cast<int>(src[i].lo);
			}
		}
		else if (depth == CV_32F)
		{
			for (int i = 0; i < dest.size().area(); i++)
			{
				dest.at<float>(i) = saturate_cast<float>(src[i].lo);
			}
		}
		else if (depth == CV_64F)
		{
			for (int i = 0; i < dest.size().area(); i++)
			{
				dest.at<double>(i) = src[i].lo;
			}
		}
		else
		{
			printf("Do not support this type in DD2Mat.\n");
		}
	}

	void cvtMattoDD(const Mat& src, doubledouble* dest)
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
