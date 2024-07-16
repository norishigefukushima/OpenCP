#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

static void guiInpaintPreviewMouse(int event, int x, int y, int flags, void* param)
{
	Rect* ret = (Rect*)param;
	if (flags == EVENT_FLAG_LBUTTON)
	{
		ret->x = max(0, min(ret->width - 1, x));
		ret->y = max(0, min(ret->height - 1, y));
	}
}

void guiInpaint(Mat& src, string wname)
{
	cout << wname << endl;
	namedWindow(wname);
	int alpha = 0; createTrackbar("alpha_", wname, &alpha, 100);
	int r = 3;	createTrackbar("inpaint_r", wname, &r, 20);
	Mat dst;
	int key = 0;
	Mat mask = Mat::zeros(src.size(), CV_8U);
	mask.setTo(255);

	Rect pt = Rect(src.cols / 2, src.rows / 2, src.cols, src.rows);
	setMouseCallback(wname, (MouseCallback)guiInpaintPreviewMouse, (void*)&pt);
	src.copyTo(dst);
	Mat show;
	while (key != 'q')
	{
		cout << pt << endl;
		if (key == 'i')
		{
			xphoto::inpaint(src, mask, dst, 0);
		}
		cp::alphaBlend(dst, mask, alpha * 0.01, show);
		imshow(wname, show);
		circle(mask, Point(pt.x, pt.y), r, Scalar::all(0), cv::FILLED);
		//mask.at<uchar>() = 0;

		key = waitKey(1);
	}
}