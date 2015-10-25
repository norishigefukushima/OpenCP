#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	static void onMouse(int events, int x, int y, int flags, void *param)
	{
		Point* pt = (Point*)param;
		//if(events==CV_EVENT_LBUTTONDOWN)
		if (flags & CV_EVENT_FLAG_LBUTTON)
		{
			pt->x = x;
			pt->y = y;
		}
	}

	void guiZoom(InputArray src, OutputArray dst)
	{
		const int width = src.size().width;
		const int height = src.size().height;

		string wname = "Zoom";
		namedWindow(wname);
		int th = 1;
		//Point mpt = Point(width/2, height/2);
		Point mpt = Point(410, 230);
		int d = 40;
		int z = 7;

		createTrackbar("x", wname, &mpt.x, width - 1);
		createTrackbar("y", wname, &mpt.y, height - 1);
		createTrackbar("d", wname, &d, min(width, height) - 1);
		createTrackbar("zoom", wname, &z, 20);
		int key = 0;

		setMouseCallback(wname, onMouse, &mpt);

		Mat dest;
		Mat input = src.getMat();

		int mode = 0;
		while (key != 'q')
		{
			input.copyTo(dest);
			z = max(z, 1);

			Rect rct = Rect(mpt.x, mpt.y, d, d);
			Scalar color = Scalar(0, 0, 255);
			int thick = 2;

			rectangle(dest, rct, color, thick);

			Mat crop;
			dest(rct).copyTo(crop);
			Mat res;
			resize(crop, res, Size(z*d, z*d), 0, 0, INTER_NEAREST);

			if (res.cols <= dest.cols && res.rows <= dest.rows)
			{
				if (mode == 0) res.copyTo(dest(Rect(0, 0, res.size().width, res.size().height)));
				else if (mode == 1) res.copyTo(dest(Rect(dest.cols - 1 - res.size().width, 0, res.size().width, res.size().height)));
				else if (mode == 2) res.copyTo(dest(Rect(dest.cols - 1 - res.size().width, dest.rows - 1 - res.size().height, res.size().width, res.size().height)));
				else if (mode == 3) res.copyTo(dest(Rect(0, dest.rows - 1 - res.size().height, res.size().width, res.size().height)));
			}

			imshow(wname, dest);
			key = waitKey(1);

			setTrackbarPos("x", wname, mpt.x);
			setTrackbarPos("y", wname, mpt.y);
			if (key == 'r')
			{
				mode++;
				if (mode > 4) mode = 0;
			}
			if (key == 's')
			{
				imwrite("out.png", dest);
			}
		}

		dest.copyTo(dst);
		destroyWindow(wname);
	}
}