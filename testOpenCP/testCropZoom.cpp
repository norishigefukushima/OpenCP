#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testCropZoom()
{
	Mat im1 = imread("img/stereo/Dolls/view1.png");
	Mat im2 = imread("img/stereo/Dolls/view5.png");

	cout << "call guiCropZoom" << endl;
	guiCropZoom(im1);
	cout << "check zoom factor and rectangle at same position" << endl;
	Rect roi;
	int factor;
	guiCropZoom(im2, roi, factor);

	Mat crop1, crop2, srcmark1, srcmark2;
	cropZoomWithSrcMarkAndBoundingBox(im1, crop1, srcmark1, roi, factor);
	cropZoomWithSrcMarkAndBoundingBox(im2, crop2, srcmark2, roi, factor);
	imshow("crop0", crop1);
	imshow("crop1", crop2);
	imshow("srcmark", srcmark1);
	imshow("srcmark", srcmark2);
	waitKey(0);
	/*imwrite("crop1.png", crop1);
	imwrite("crop2.png", crop2);
	imwrite("srcmark1.png", srcmark2);
	imwrite("srcmark2.png", srcmark2);*/
}
