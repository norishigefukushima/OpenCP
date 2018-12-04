#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void guiLaplacianSmoothingIIRFilterTest(Mat& src, String wname = "Laplacian")
{
	namedWindow(wname);
	int sigma = 10;
	createTrackbar("sigma", wname, &sigma, 100);
	int n = 9;
	createTrackbar("sigman", wname, &n, 20);
	int p = 100;
	createTrackbar("p", wname, &p, 200);
	Mat dest = src.clone();
	Mat dest2 = src.clone();
	Mat srcf;

	int key = 0;
	int64 start;
	int64 end;
	Stat st1;
	Stat st2;

	ConsoleImage ci;
	const int iteration = 1;
	Mat destf;
	while (key != 'q')
	{
		src.convertTo(srcf, CV_32F);

		int r = cvRound(sigma*3.f);
		int d = 2 * r + 1;

		for (int i = 0; i < iteration; i++)
		{
			start = cv::getTickCount();
			//LaplacianSmoothingIIRFilter(srcf, destf, sigma2LaplacianSmootihngAlpha(sigma, p / 100.0), cp::VECTOR_WITHOUT);
			LaplacianSmoothingFIRFilter(srcf, destf, sigma*n, sigma, BORDER_REPLICATE, cp::VECTOR_AVX);
			end = cv::getTickCount();
			st1.push_back((end - start) * 1000 / cv::getTickFrequency());
		}
		destf.convertTo(dest, CV_8U);
		//imshow("base", dest);
		ci(format("AVX: %f [ms]", st1.getMedian()));

		for (int i = 0; i < iteration; i++)
		{
			start = cv::getTickCount();
			LaplacianSmoothingIIRFilter(srcf, destf, sigma2LaplacianSmootihngAlpha(sigma, p / 100.0));
			/*
			Mat srcd; src.convertTo(srcd, CV_64F);
			Mat destd;
			//LaplacianSmoothingIIRFilterDouble(srcd, destd, sigma2Laplacianalpha(sigma,p/100.0));
			destd.convertTo(destf, CV_32F);
			*/
			end = cv::getTickCount();
			st2.push_back((end - start) * 1000 / cv::getTickFrequency());
		}
		destf.convertTo(dest2, CV_8U);

		//imshow("avx", dest2);
		imshow("diff", abs(dest2 - dest) * 30);
		ci(format("AVX: %f [ms]", st2.getMedian()));
		ci(format("PSNR: %f [dB]", PSNR(dest, dest2)));
		if (key == 't')
		{
			guiAlphaBlend(dest, dest2);
		}
		ci.show();
		imshow(wname, dest2);
		key = waitKey(1);
		if (key == 'r')
		{
			st1.clear();
			st2.clear();
		}
	}
}