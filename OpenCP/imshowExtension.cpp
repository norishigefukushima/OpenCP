#include "imshowExtension.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void imshowNormalize(string wname, InputArray src)
	{
		Mat show;
		normalize(src.getMat(), show, 255, 0, NORM_MINMAX, CV_8U);
		imshow(wname, show);
	}

	void imshowScale(string name, InputArray src, const double alpha, const double beta)
	{
		Mat show;
		src.getMat().convertTo(show, CV_8U, alpha, beta);
		imshow(name, show);
	}

	void imshowCountDown(string wname, InputArray src, const int waitTime, Scalar color, const int pointSize, std::string fontName)
	{
		Mat s;
		src.copyTo(s);
		addText(s, "3", Point(s.cols / 2, s.rows / 2), fontName, pointSize, color);
		imshow(wname, s);
		waitKey(waitTime);

		src.copyTo(s);
		addText(s, "2", Point(s.cols / 2, s.rows / 2), fontName, pointSize, color);
		imshow(wname, s);
		waitKey(waitTime);

		src.copyTo(s);
		addText(s, "1", Point(s.cols / 2, s.rows / 2), fontName, pointSize, color);
		imshow(wname, s);
		waitKey(waitTime);
	}
}