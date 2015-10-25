#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void guiCoherenceEnhancingShockFilter(Mat& src, Mat& dest)
{
	Mat bl;

	int p1 = 1;
	int p2 = 1;
	int iter = 1;
	int alpha = 50;

	string wname = "shock filter";
	namedWindow(wname);
	int sigma = 5;
	int r = 2;
	createTrackbar("input blur sigma", wname, &sigma, 20);
	createTrackbar("input blur r", wname, &r, 20);

	createTrackbar("sobel_sigma", wname, &p1, 20);
	createTrackbar("eigen_sigma", wname, &p2, 20);
	createTrackbar("iter", wname, &iter, 50);
	createTrackbar("alpha", wname, &alpha, 100);
	int key = 0;

	Mat src2; src.copyTo(src2);
	Mat show;
	while (key != 'q')
	{
		if (r == 0)src.copyTo(bl);
		else GaussianBlur(src2, bl, Size(2 * r + 1, 2 * r + 1), sigma);


		{
			//			CalcTime t(wname);
			
			coherenceEnhancingShockFilter(bl, dest, 2 * p1 + 1, 2 * p2 + 1, alpha / 100.0, iter);
			showMatInfo(dest);
		}
		dest.convertTo(show, CV_8U);
		imshow(wname, show);
		key = waitKey(1);
	}
}
