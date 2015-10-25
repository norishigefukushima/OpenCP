#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void stereoAnaglyph(Mat& lim, Mat& rim, Mat& dest, int shift)
	{
		Mat g1, g2, swap;

		cvtColor(rim, swap, CV_BGR2GRAY);
		warpShift(swap, g1, -shift, 0);

		cvtColor(lim, swap, CV_BGR2GRAY);
		warpShift(swap, g2, shift, 0);

		vector<Mat> v(3);
		v[0] = g1;
		v[1] = g1.clone();
		v[2] = g2;
		merge(v, dest);
	}

	void stereoInterlace(Mat& lim, Mat& rim, Mat& dest, int d, int left_right_swap)
	{
		warpShift(lim, dest, -d, 0);

		Mat swap;

		warpShift(rim, swap, d, 0);

		//0 right left right left
		//1 left right left right
		int flg = left_right_swap % 2;

		int channel = lim.channels();
		for (int j = 0; j < lim.rows; j++)
		{
			uchar* l = dest.ptr(j);
			uchar* r = swap.ptr(j);
			if (j % 2 == flg)
			{
				memcpy(l, r, sizeof(lim.cols*channel));
			}
		}
	}

}