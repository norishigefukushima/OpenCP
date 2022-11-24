#include "circleFilter.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void circleFilter(InputArray src, OutputArray dest, const int r, const int borderType)
	{
		Mat_<float> kernel(2 * r + 1, 2 * r + 1);
		int count = 0;
		for (int j = -r; j <= r; j++)
		{
			for (int i = -r; i <=r; i++)
			{
				if (i * i + j * j <= r * r)count++;
			}
		}
		for (int j = 0; j < kernel.rows; j++)
		{
			for (int i = 0; i < kernel.cols; i++)
			{
				int x = i - r;
				int y = j - r;
				if (x * x + y * y <= r * r)
				{
					kernel(j, i) = 1.f / count;
				}
				else
				{
					kernel(j, i) = 0.f;
				}
			}
		}
		
		Mat srcf; src.getMat().convertTo(srcf, CV_32F);
		Mat destf;
		cv::filter2D(srcf, destf, CV_32F, kernel, Point(-1,-1), 0.0, borderType);
		destf.convertTo(dest, CV_8U);
	}
}