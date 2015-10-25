#include "opencp.hpp"

using namespace cv;
using namespace std;

namespace cp
{

	void showMatInfo(InputArray src_, string name)
	{
		Mat src = src_.getMat();

		cout << name << ":" << endl;
		if (src.empty())
		{
			cout << "empty" << endl;
			return;
		}
		cout << "size    " << src.size() << endl;
		cout << "channel " << src.channels() << endl;
		if (src.depth() == CV_8U)cout << "8U" << endl;
		else if (src.depth() == CV_16S)cout << "16S" << endl;
		else if (src.depth() == CV_16U)cout << "16U" << endl;
		else if (src.depth() == CV_32S)cout << "32S" << endl;
		else if (src.depth() == CV_32F)cout << "32F" << endl;
		else if (src.depth() == CV_64F)cout << "64F" << endl;

		if (src.channels() == 1)
		{
			Scalar v = mean(src);
			cout << "mean  : " << v.val[0] << endl;
			double minv, maxv;
			minMaxLoc(src, &minv, &maxv);
			cout << "minmax: " << minv << "," << maxv << endl;
		}
		else if (src.channels() == 3)
		{
			Scalar v = mean(src);
			cout << "mean  : " << v.val[0] << "," << v.val[1] << "," << v.val[2] << endl;

			vector<Mat> vv;
			split(src, vv);
			double minv, maxv;
			minMaxLoc(vv[0], &minv, &maxv);
			cout << "minmax0: " << minv << "," << maxv << endl;
			minMaxLoc(vv[1], &minv, &maxv);
			cout << "minmax1: " << minv << "," << maxv << endl;
			minMaxLoc(vv[2], &minv, &maxv);
			cout << "minmax2: " << minv << "," << maxv << endl;
		}
	}
}