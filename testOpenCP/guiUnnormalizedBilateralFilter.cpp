#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void guiUnnormalizedBilateralFilter(Mat& src)
{
	string wname = "UnnormalizedBilateralFilter";
	namedWindow(wname);
	int sr = 30; createTrackbar("sr", wname, &sr, 255);
	int ss = 3; createTrackbar("ss", wname, &ss, 10);
	int isEnhance = 0; createTrackbar("enhance", wname, &isEnhance, 1);

	int key = 0;
	Mat dest;
	cp::ConsoleImage ci;
	cp::Timer t("", TIME_MSEC);
	cp::UpdateCheck uc(ss);
	while (key != 'q')
	{
		if (uc.isUpdate(ss))t.clearStat();
		t.start();
		cp::unnormalizedBilateralFilter(src, dest, (int)ceil(ss * 3.f), float(sr), float(ss), isEnhance == 1);
		t.getpushLapTime();
		imshow(wname, dest);
		ci("Time %f (%d)", t.getLapTimeMedian(), t.getStatSize());
		ci.show();
		key = waitKey(1);
	}
}