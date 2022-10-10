#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testWindowFunction()
{
	string wname = "window function";
	namedWindow(wname);
	int windowType = GAUSSIAN_WINDOW; createTrackbar("type", wname, &windowType, WINDOW_TYPE_SIZE - 1);
	int isSeparable = 0; createTrackbar("separable", wname, &isSeparable, 1);

	int key = 0;
	while (key != 'q')
	{
		Mat show = cp::createWindowFunction(100, 100, CV_32F, windowType, isSeparable == 1);
		imshow(wname, show);
		key = waitKey(1);
	}
}