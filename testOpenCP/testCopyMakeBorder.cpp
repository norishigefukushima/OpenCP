#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void copyMakeBorderTest(Mat& src)
{
	//cv::setNumThreads(1);
	//cv::ipp::setUseIPP(true);
	//cv::setUseOptimized(false);

	const int r = 3;
	const int iteration = 10000;
	print_debug2(r, iteration);
	const int borderType = BORDER_REPLICATE;

	CV_Assert(src.channels() == 3);
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	Mat src32fc1 = convert(gray, CV_32F);
	Mat src32fc3 = convert(src, CV_32F);

	bool isGrayTest = true;
	bool isColorTest = false;

	if(isGrayTest)
	{
		Mat dstcv;
		Mat dstcp;
		cout << "gray" << endl;
		{
			Timer t("cv 32f");
			for (int i = 0; i < iteration; i++)
				cv::copyMakeBorder(src32fc1, dstcv, r, r, r, r, borderType);
		}
		{
			Timer t("cp 32f");
			for (int i = 0; i < iteration; i++)
				cp::copyMakeBorderReplicate(src32fc1, dstcp, r, r, r, r);
		}
		{
			Timer t("cv 32f");
			for (int i = 0; i < iteration; i++)
				cv::copyMakeBorder(src32fc1, dstcv, r, r, r, r, borderType);
		}
		{
			Timer t("cp 32f");
			for (int i = 0; i < iteration; i++)
				cp::copyMakeBorderReplicate(src32fc1, dstcp, r, r, r, r);
		}
		cout << getPSNR(dstcv, dstcp) << "dB" << endl;

		guiAlphaBlend(dstcv, dstcp);
	}

	if (isColorTest)
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