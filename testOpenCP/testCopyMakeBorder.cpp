#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void copyMakeBorderTest(Mat& src)
{
	const int r = 3;
	const int borderType = BORDER_REPLICATE;

	CV_Assert(src.channels() == 3);
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	Mat src32fc1 = convert(gray, CV_32F);
	Mat src32fc3 = convert(src, CV_32F);

	const int iteration = 10000;
	{
		Mat dstcv;
		Mat dstcp;
		cout << "gray" << endl;
		{
			Timer t("cv");
			for (int i = 0; i < iteration; i++)
				cv::copyMakeBorder(src32fc1, dstcv, r, r, r, r, borderType);
		}
		{
			Timer t("cp");
			for (int i = 0; i < iteration; i++)
				cp::copyMakeBorderReplicate(src32fc1, dstcp, r, r, r, r);
		}
		cout << getPSNR(dstcv, dstcp) << "dB" << endl;
	}

	{
		Mat dstcv;
		Mat dstcp;
		cout << "color" << endl;
		{
			Timer t("cv");
			for (int i = 0; i < iteration; i++)
				cv::copyMakeBorder(src32fc3, dstcv, r, r, r, r, borderType);
		}
		{
			Timer t("cp");
			for (int i = 0; i < iteration; i++)
				cp::copyMakeBorderReplicate(src32fc3, dstcp, r, r, r, r);
		}
		cout << getPSNR(dstcv, dstcp) << "dB" << endl;
		guiAlphaBlend(dstcv, dstcp);
	}

}