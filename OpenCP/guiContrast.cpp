#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void guiContrast(InputArray src_)
	{
		string window_name = "contrast";
		Mat src = src_.getMat();
		namedWindow(window_name);
		int a = 10;
		int b = 0;
		cv::createTrackbar("a/10", window_name, &a, 1024);
		cv::createTrackbar("b", window_name, &b, 256);
		int key = 0;
		cv::Mat show;
		while (key != 'q')
		{
			show = a / 10.0*src + b;
			imshow(window_name, show);
			key = waitKey(33);

			if (key == 'l')
			{
				a--;
				setTrackbarPos("a/10", window_name, a);
			}
			if (key == 'j')
			{
				a++;
				setTrackbarPos("a/10", window_name, a);
			}
			if (key == 'i')
			{
				b++;
				setTrackbarPos("b", window_name, b);
			}
			if (key == 'k')
			{
				b--;
				setTrackbarPos("b", window_name, b);
			}
		}
		destroyWindow(window_name);
	}
}