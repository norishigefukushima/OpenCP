#include "maskoperation.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	Mat createBoxMask(const Size size, const int boundx, const int boundy)
	{
		Mat ret = Mat::zeros(size, CV_8U);
		ret(Rect(boundx, boundy, size.width - 2 * boundx, size.height - 2 * boundy)).setTo(255);
		return ret;
	}

	void setBoxMask(Mat& mask, const int boundx, const int boundy)
	{
		Size size = mask.size();
		mask.setTo(0);
		mask(Rect(boundx, boundy, size.width - 2 * boundx, size.height - 2 * boundy)).setTo(255);
	}

	void addBoxMask(Mat& mask, const int boundx, const int boundy)
	{
		Size size = mask.size();
		Mat m = createBoxMask(size, boundx, boundy);
		mask.setTo(0, ~m);
	}
}