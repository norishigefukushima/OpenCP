#include "matinfo.hpp"

using namespace cv;
using namespace std;

namespace cp
{
	static void showMatInfo_internal(cv::Mat src, string name, const bool isStatInfo)
	{
		cout << "name    : " << name << endl;
		if (src.empty())
		{
			cout << "empty" << endl;
			return;
		}

		cout << "size    : " << src.size() << endl;
		cout << "channel : " << src.channels() << endl;
		cout << "depth   : ";
		if (src.depth() == CV_8U)cout << "8U" << endl;
		else if (src.depth() == CV_8S)cout << "8S" << endl;
		else if (src.depth() == CV_16S)cout << "16S" << endl;
		else if (src.depth() == CV_16U)cout << "16U" << endl;
		else if (src.depth() == CV_32S)cout << "32S" << endl;
		else if (src.depth() == CV_32F)cout << "32F" << endl;
		else if (src.depth() == CV_64F)cout << "64F" << endl;
		else if (src.depth() == CV_16F)cout << "16F" << endl;
		
		if (src.isContinuous()) cout << "continue: true"<<endl;
		else  cout << "continue: false" << endl;
		if (src.isSubmatrix()) cout << "ROI     : true" << endl;
		else  cout << "ROI     : false" << endl;

		if (src.channels() < 5)
		{
			Scalar v, stdv;
			cv::meanStdDev(src, v, stdv);
			
			if (v.val[0] == 0 && v.val[1] == 0 && v.val[2] == 0 && v.val[3] == 0)
			{
				cout << "zero set" << endl;
			}
			if (isStatInfo)
			{
				if (src.channels() == 1)
				{
					cout << "mean  : " << v.val[0] << endl;
					cout << "std   : " << stdv.val[0] << endl;
					double minv, maxv;
					minMaxLoc(src, &minv, &maxv);
					cout << "minmax: " << minv << "," << maxv << endl;
				}
				else if (src.channels() == 3)
				{
					cout << "mean  : " << v.val[0] << "," << v.val[1] << "," << v.val[2] << endl;
					cout << "std   : " << stdv.val[0] << "," << stdv.val[1] << "," << stdv.val[2] << endl;

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
		cout << endl;
	}

	void showMatInfo(InputArray src_, string name, const bool isStatInfo)
	{
		if (src_.isMatVector())
		{
			cout << "type    : vector<Mat>" << endl;
			vector<Mat> v;
			src_.getMatVector(v);
			for (int i = 0; i < v.size(); i++)
			{
				showMatInfo_internal(v[i], name + to_string(i), isStatInfo);;
			}

		}
		else if (src_.isUMat())
		{
			cout << "type    : UMat" << endl;
		}
		else if (src_.isUMatVector())
		{
			cout << "type    : vector<UMat>" << endl;
		}
		else if (src_.isGpuMat())
		{
			cout << "type    : GpuMat" << endl;
		}
		else if (src_.isGpuMatVector())
		{
			cout << "type    : vector<GpuMat>" << endl;
		}
		else
		{
			cout << "type    : Mat" << endl;
			Mat s = src_.getMat();
			showMatInfo_internal(s, name, isStatInfo);
		}
	}
}