#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void testHistgram(Mat& src)
{
	Mat gray, hist;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	drawHistogramImageGray(gray, hist, COLOR_GRAY200, COLOR_RED);
	imshow("hist", hist);
	waitKey();

	drawAccumulateHistogramImageGray(gray, hist, COLOR_GRAY150, COLOR_RED);
	imshow("hist", hist);
	waitKey();

	drawHistogramImage(src, hist, COLOR_ORANGE);
	imshow("hist", hist);
	waitKey();

	drawAccumulateHistogramImage(src, hist, COLOR_ORANGE);
	imshow("hist", hist);
	waitKey();
}