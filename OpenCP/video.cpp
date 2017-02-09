#include "video.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void guiVideoShow(std::string wname)
	{
		VideoCapture cap(0);
		if (!cap.isOpened())
		{
			cout << "video open error" << endl;
			return;
		}

		namedWindow(wname, 1);
		for (;;)
		{
			Mat frame;
			cap >> frame;
			imshow(wname, frame);
			if (waitKey(30) >= 0) break;
		}
	}

}