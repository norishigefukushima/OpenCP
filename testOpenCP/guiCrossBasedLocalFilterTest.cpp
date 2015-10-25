#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void guiCrossBasedLocalFilter(Mat& src_)
{
	Mat src;
	if (src_.channels() == 3)cvtColor(src_, src, COLOR_BGR2GRAY);
	else src = src_;
	
	//src_.copyTo(src);
	Mat dest;

	string wname = "cross based local filter";
	namedWindow(wname);

	int a = 0; createTrackbar("a", wname, &a, 100);
	int r = 1; createTrackbar("r", wname, &r, 10);
	int thresh = 1; createTrackbar("threshold", wname, &thresh, 100);

	int key = 0;
	Mat show;
	while (key != 'q')
	{
		CrossBasedLocalFilter clf(src_, r, thresh);
		{
			CalcTime t;
			clf(src, dest);
		}

		alphaBlend(src, dest, a / 100.0, show);

		imshow(wname, show);
		key = waitKey(1);
	}
	destroyWindow(wname);
}