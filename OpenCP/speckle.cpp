#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void guiFilterSpeckle(cv::InputArray src_)
	{
		Mat src = src_.getMat();
		string wname = "filterSpeckle";
		namedWindow(wname);
		int ss = 20; createTrackbar("ss", wname, &ss, 1000);
		int md = 2; createTrackbar("md", wname, &md, 1000);
		int key = 0;
		while (key != 'q')
		{
			Mat target;
			src.copyTo(target);

			filterSpeckles(target, 0, ss, md);
			Mat mask;
			compare(src, target, mask, CMP_NE);

			Mat show;
			cvtColor(src, show, CV_GRAY2BGR);
			show.setTo(Scalar(0, 255, 0), mask);
			imshow(wname, show);
			key = waitKey(1);
		}
	}
}