#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void guiColorCorrectionTest(Mat& src, Mat& ref)
{
	Mat cmat;
	findColorMatrixAvgStdDev(src, ref, cmat, 100, 200);
	Mat dest;
	cout << cmat << endl;
	cvtColorMatrix(src, dest, cmat);
	guiAlphaBlend(src, dest);
}

void ocvcvtPCA(const Mat& src, Mat& dst, const int dest_channels)
{
	Mat x = src.reshape(1, src.size().area());
	//PCA pca(x, cv::Mat(), cv::PCA::DATA_AS_ROW, dest_channels);
	//dst = pca.project(x).reshape(dest_channels, src.rows);
	Mat cov, mean;
	cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
	Mat eval, evec;
	eigen(cov, eval, evec);
	Mat transmat;
	evec(Rect(0, 0, evec.cols, dest_channels)).convertTo(transmat, CV_32F);
	cv::transform(src, dst, transmat);
}

void guiCvtColorPCATest()
{
	Mat src;
	Mat srcf;
	vector<Mat> vsrcf;
#pragma region setup
	const int channels = 6;
	if (channels == 2)
	{
		Mat flash = imread("img/flash/cave-flash.png", 0);
		Mat noflash = imread("img/flash/cave-noflash.png", 0);
		vsrcf.resize(2);
		flash.convertTo(vsrcf[0], CV_32F);
		noflash.convertTo(vsrcf[1], CV_32F);
		merge(vsrcf, srcf);
		src = convert(srcf, CV_8U);
	}
	else if (channels == 3)
	{
		Mat flash = imread("img/flash/cave-flash.png");
		flash.copyTo(src);
		srcf = convert(flash, CV_32F);
		split(srcf, vsrcf);
	}
	else if (channels == 4)
	{
		Mat flash = imread("img/flash/cave-flash.png", 0);
		Mat noflash = imread("img/flash/cave-noflash.png");
		vector<Mat> v2; split(noflash, v2);
		vsrcf.resize(4);
		flash.convertTo(vsrcf[0], CV_32F);
		v2[0].convertTo(vsrcf[1], CV_32F);
		v2[1].convertTo(vsrcf[2], CV_32F);
		v2[2].convertTo(vsrcf[3], CV_32F);
		merge(vsrcf, srcf);
		src = convert(srcf, CV_8U);
	}
	else if (channels == 5)
	{
		Mat flash = imread("img/flash/cave-flash.png");
		Mat noflash = imread("img/flash/cave-noflash.png");
		vector<Mat> v1; split(flash, v1);
		vector<Mat> v2; split(noflash, v2);
		vsrcf.resize(5);
		v1[0].convertTo(vsrcf[0], CV_32F);
		v1[1].convertTo(vsrcf[1], CV_32F);
		v1[2].convertTo(vsrcf[2], CV_32F);
		v2[0].convertTo(vsrcf[3], CV_32F);
		v2[1].convertTo(vsrcf[4], CV_32F);
		merge(vsrcf, srcf);
		src = convert(srcf, CV_8U);
	}
	else if (channels == 6)
	{
		Mat flash = imread("img/flash/cave-flash.png");
		Mat noflash = imread("img/flash/cave-noflash.png");
		vector<Mat> v1; split(flash, v1);
		vector<Mat> v2; split(noflash, v2);
		vsrcf.resize(6);
		v1[0].convertTo(vsrcf[0], CV_32F);
		v1[1].convertTo(vsrcf[1], CV_32F);
		v1[2].convertTo(vsrcf[2], CV_32F);
		v2[0].convertTo(vsrcf[3], CV_32F);
		v2[1].convertTo(vsrcf[4], CV_32F);
		v2[2].convertTo(vsrcf[5], CV_32F);
		merge(vsrcf, srcf);
		src = convert(srcf, CV_8U);
	}
	#pragma endregion


	string wname = "cvtColorPCATest";
	namedWindow(wname);
	int method = 0; createTrackbar("method", wname, &method, 2);
	int ch = 1; createTrackbar("num ch", wname, &ch, src.channels());
	int sch = 1; createTrackbar("show ch", wname, &sch, src.channels());
	setTrackbarMin("num ch", wname, 1);
	setTrackbarMin("show ch", wname, 1);

	Mat dest;
	std::vector<Mat> dst;
	std::vector<Mat> show(src.channels());

	int key = 0;
	cp::ConsoleImage ci;
	cp::Timer t1("", TIME_MSEC);
	cp::Timer t2("", TIME_MSEC);
	cp::Timer t3("", TIME_MSEC);
	cp::UpdateCheck uc(ch);

	double psnr = 0.0;
	while (key != 'q')
	{
		bool isClear = false;
		const int cindex = min(sch, ch) - 1;

		ci("loop %d", t1.getStatSize());

		t1.start();
		cvtColorPCA(srcf, dest, ch);
		t1.getpushLapTime();
		ci("normal Time %f ms", t1.getLapTimeMedian());
		if (method == 0)
		{
			split(dest, dst);
			dst[cindex].copyTo(show[cindex]);
		}

		t2.start();
		cvtColorPCA(vsrcf, dst, ch);
		t2.getpushLapTime();
		ci("split Time %f ms", t2.getLapTimeMedian());
		if (method == 1)
		{
			dst[cindex].copyTo(show[cindex]);
		}

		psnr = cvtColorPCAErrorPSNR(vsrcf, ch);
		

		t3.start();
		ocvcvtPCA(srcf, dest, ch);
		t3.getpushLapTime();
		ci("ocv Time %f ms", t3.getLapTimeMedian());
		if (method == 2)
		{
			split(dest, dst);
			dst[cindex].copyTo(show[cindex]);
		}

		key = waitKey(1);
		imshowNormalize(wname, show[cindex]);
		ci("psnr %5.2f", psnr);
		ci.show();

		if (key == 'r')
		{
			isClear = true;
		}
		if (isClear || uc.isUpdate(ch))
		{
			t1.clearStat();
			t2.clearStat();
		}
	}
	destroyWindow(wname);
}