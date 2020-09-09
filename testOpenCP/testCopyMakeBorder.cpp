#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void copyMakeBorderTest(Mat& src)
{
	const int r = 30;
	//const int iteration = 1;
	const int iteration = 1000;
	//const int iteration = 10000;

	print_debug2(r, iteration);
	const int borderType = BORDER_REPLICATE;
	//const int borderType = BORDER_DEFAULT;
	//const int borderType = BORDER_REFLECT;

	CV_Assert(src.channels() == 3);
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	Mat src32fc1 = convert(gray, CV_32F);
	Mat src32fc3 = convert(src, CV_32F);

	bool isGrayTest = false;
	bool isColorTest = false;
	bool isSplitTest = true;

	if (isGrayTest)
	{
		Mat dstcv8u;
		Mat dstcp8u;
		Mat dstcv32f;
		Mat dstcp32f;
		cout << "gray" << endl;
		{
			Timer t("cv 8u");
			for (int i = 0; i < iteration; i++)
				cv::copyMakeBorder(gray, dstcv8u, r, r, r, r, borderType);
		}
		{
			Timer t("cp 8u");
			for (int i = 0; i < iteration; i++)
				cp::copyMakeBorderReplicate(gray, dstcp8u, r, r, r, r);
		}
		cout << getPSNR(dstcv8u, dstcp8u) << "dB" << endl;
		//guiAlphaBlend(dstcv8u, dstcp8u);
		{
			Timer t("cv 32f");
			for (int i = 0; i < iteration; i++)
				cv::copyMakeBorder(src32fc1, dstcv32f, r, r, r, r, borderType);
		}
		{
			Timer t("cp 32f");
			for (int i = 0; i < iteration; i++)
				cp::copyMakeBorderReplicate(src32fc1, dstcp32f, r, r, r, r);
		}
		cout << getPSNR(dstcv32f, dstcp32f) << "dB" << endl;
		//guiAlphaBlend(dstcv32f, dstcp32f);
	}

	if (isColorTest)
	{
		Mat dstcv8u;
		Mat dstcp8u;
		Mat dstcv32f;
		Mat dstcp32f;
		cout << "color" << endl;
		{
			Timer t("cv 8u");
			for (int i = 0; i < iteration; i++)
				cv::copyMakeBorder(src, dstcv8u, r, r, r, r, borderType);
		}
		{
			Timer t("cp 8u");
			for (int i = 0; i < iteration; i++)
				cp::copyMakeBorderReplicate(src, dstcp8u, r, r, r, r);
		}
		cout << getPSNR(dstcv8u, dstcp8u) << "dB" << endl;
		//guiAlphaBlend(dstcv8u, dstcp8u);
		{
			Timer t("cv 32f");
			for (int i = 0; i < iteration; i++)
				cv::copyMakeBorder(src32fc3, dstcv32f, r, r, r, r, borderType);
		}
		{
			Timer t("cp 32f");
			for (int i = 0; i < iteration; i++)
				cp::copyMakeBorderReplicate(src32fc3, dstcp32f, r, r, r, r);
		}
		cout << getPSNR(dstcv32f, dstcp32f) << "dB" << endl;
		guiAlphaBlend(dstcv32f, dstcp32f);
	}

	if (isSplitTest)
	{
		cout << "split" << endl;

		vector<Mat> dstcv32f(3);
		vector<Mat> dstcv8u(3);
		vector<Mat> dstcp32f;
		vector<Mat> dstsplit32f;
		vector<Mat> dstcp8u;
		vector<Mat> dstsplit8u;
		{
			Timer t("cv 8u");
			for (int i = 0; i < iteration; i++)
			{
				split(src, dstsplit8u);
				cv::copyMakeBorder(dstsplit8u[0], dstcv8u[0], r, r, r, r, borderType);
				cv::copyMakeBorder(dstsplit8u[1], dstcv8u[1], r, r, r, r, borderType);
				cv::copyMakeBorder(dstsplit8u[2], dstcv8u[2], r, r, r, r, borderType);
			}
		}
		{
			Timer t("cp 8u");
			for (int i = 0; i < iteration; i++)
			{
				cp::splitCopyMakeBorder(src, dstcp8u, r, r, r, r, borderType);
			}
		}
		cout << getPSNR(dstcv8u[0], dstcp8u[0]) << "dB" << endl;
		cout << getPSNR(dstcv8u[1], dstcp8u[1]) << "dB" << endl;
		cout << getPSNR(dstcv8u[2], dstcp8u[2]) << "dB" << endl;
		//guiAlphaBlend(dstcv8u[0], dstcp8u[0]);

		{
			Timer t("cv 32f");
			for (int i = 0; i < iteration; i++)
			{
				split(src32fc3, dstsplit32f);
				cv::copyMakeBorder(dstsplit32f[0], dstcv32f[0], r, r, r, r, borderType);
				cv::copyMakeBorder(dstsplit32f[1], dstcv32f[1], r, r, r, r, borderType);
				cv::copyMakeBorder(dstsplit32f[2], dstcv32f[2], r, r, r, r, borderType);
			}
		}
		{
			Timer t("cp 32f");
			for (int i = 0; i < iteration; i++)
			{
				cp::splitCopyMakeBorder(src32fc3, dstcp32f, r, r, r, r, borderType);
			}
		}
		cout << getPSNR(dstcv32f[0], dstcp32f[0]) << "dB" << endl;
		cout << getPSNR(dstcv32f[1], dstcp32f[1]) << "dB" << endl;
		cout << getPSNR(dstcv32f[2], dstcp32f[2]) << "dB" << endl;
		
		//guiAlphaBlend(dstcv32f[0], dstcp32f[0]);
	}
}