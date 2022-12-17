#include "threshold.hpp"
#include "blend.hpp"
using namespace std;
using namespace cv;
namespace cp
{
	std::string getThresholdName(int flag)
	{
		string ret = "";
		switch (flag)
		{
		case THRESH_BINARY:		ret = "THRESH_BINARY"; break;
		case THRESH_BINARY_INV:	ret = "THRESH_BINARY_INV"; break;
		case THRESH_TRUNC:		ret = "THRESH_TRUNC"; break;
		case THRESH_TOZERO:		ret = "THRESH_TOZERO"; break;
		case THRESH_TOZERO_INV:	ret = "THRESH_TOZERO_INV"; break;
		case THRESH_BINARY | THRESH_OTSU:			ret = "THRESH_BINARY|THRESH_OTSU"; break;
		case THRESH_BINARY_INV | THRESH_OTSU:		ret = "THRESH_BINARY_INV|THRESH_OTSU"; break;
		case THRESH_TRUNC | THRESH_OTSU:			ret = "THRESH_TRUNC|THRESH_OTSU"; break;
		case THRESH_TOZERO | THRESH_OTSU:			ret = "THRESH_TOZERO|THRESH_OTSU"; break;
		case THRESH_TOZERO_INV | THRESH_OTSU:		ret = "THRESH_TOZERO_INV|THRESH_OTSU"; break;
		case THRESH_BINARY | THRESH_TRIANGLE:		ret = "THRESH_BINARY|THRESH_TRIANGLE"; break;
		case THRESH_BINARY_INV | THRESH_TRIANGLE:	ret = "THRESH_BINARY_INV|THRESH_TRIANGLE"; break;
		case THRESH_TRUNC | THRESH_TRIANGLE:		ret = "THRESH_TRUNC|THRESH_TRIANGLE"; break;
		case THRESH_TOZERO | THRESH_TRIANGLE:		ret = "THRESH_TOZERO|THRESH_TRIANGLE"; break;
		case THRESH_TOZERO_INV | THRESH_TRIANGLE:	ret = "THRESH_TOZERO_INV|THRESH_TRIANGLE"; break;
		default:
			break;
		}
		return ret;
	}

	void guiThreshold(cv::InputArray src_, std::string wname)
	{
		namedWindow(wname);
		Mat src = src_.getMat();
		static int thresh = 0;
		createTrackbar("threshold", wname, &thresh, 255);
		static int sw = 0;
		createTrackbar("sw", wname, &sw, 4);
		static int flg = 0;
		createTrackbar("flag", wname, &flg, 2);
		static int alpha = 0;
		createTrackbar("alpha", wname, &alpha, 100);
		int key = 0;
		Mat show;
		string tname;
		bool isName = true;
		while (key != 'q')
		{
			const int flag = (flg == 1) ? THRESH_OTSU : (flg == 2) ? THRESH_TRIANGLE : 0;
			tname = getThresholdName(sw | flag);
			threshold(src, show, thresh, 255, sw | flag);
			cp::alphaBlend(show, src, 1.0 - alpha * 0.01, show);
			if (isName) putText(show, tname, Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, COLOR_GRAY120);
			imshow(wname, show);
			key = waitKey(1);
			if (key == 't')isName = isName ? false : true;
		}
		destroyWindow(wname);
	}
}