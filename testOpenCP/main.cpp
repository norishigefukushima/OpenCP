#include <opencp.hpp>
#include "test.hpp"
using namespace std;

struct MouseLocalDiffHistogramParameter
{
	cv::Rect pt;
	std::string wname;
	MouseLocalDiffHistogramParameter(int x, int y, int width, int height, std::string name)
	{
		pt = cv::Rect(x, y, width, height);
		wname = name;
	}
};

void guiLocalDiffHistogramOnMouse(int event, int x, int y, int flags, void* param)
{
	MouseLocalDiffHistogramParameter* retp = (MouseLocalDiffHistogramParameter*)param;

	if (flags == EVENT_FLAG_LBUTTON)
	{
		retp->pt.x = max(0, min(retp->pt.width - 1, x));
		retp->pt.y = max(0, min(retp->pt.height - 1, y));

		setTrackbarPos("x", retp->wname, x);
		setTrackbarPos("y", retp->wname, y);
	}
}

void guiLocalDiffHistogram(Mat& src, bool isWait = true, string wname = "gui");
void guiLocalDiffHistogram(Mat& src, bool isWait, string wname)
{
	Mat img;
	if (src.channels() == 3)
	{
		cvtColor(src, img, COLOR_BGR2GRAY);
	}
	else
	{
		img = src;
	}


	namedWindow(wname);

	static MouseLocalDiffHistogramParameter param(src.cols / 2, src.rows / 2, src.cols, src.rows, wname);

	setMouseCallback(wname, (MouseCallback)guiLocalDiffHistogramOnMouse, (void*)&param);
	createTrackbar("x", wname, &param.pt.x, src.cols - 1);
	createTrackbar("y", wname, &param.pt.y, src.rows - 1);
	int r = 0;  createTrackbar("r", wname, &r, 50);
	int histmax = 100; createTrackbar("histmax", wname, &histmax, 255);
	int key = 0;
	Mat show;

	int hist[512];

	Plot pt;
	Mat hist_img(Size(512, 512), CV_8UC3);
	while (key != 'q')
	{

		hist_img.setTo(0);
		img.copyTo(show);
		Mat crop;
		Point pt = Point(param.pt.x, param.pt.y);

		const int d = 2 * r + 1;
		cropZoom(img, crop, pt, d);
		rectangle(show, Rect(pt.x - r, pt.y - r, d, d), COLOR_WHITE);
		for (int i = 0; i < 512; i++)hist[i] = 0;

		int b = crop.at<uchar>(r, r);

		int minv = 255;
		int maxv = -255;
		for (int j = 0; j < d; j++)
		{
			for (int i = 0; i < d; i++)
			{
				int val = (int)crop.at<uchar>(j, i) - b + 255;
				hist[val]++;

				maxv = max(maxv, val - 255);
				minv = min(minv, val - 255);
			}
		}

		line(hist_img, Point(255, hist_img.rows - 1), Point(255, 0), COLOR_GRAY50);
		line(hist_img, Point(minv + 255, hist_img.rows - 1), Point(minv + 255, 0), COLOR_GRAY40);
		line(hist_img, Point(maxv + 255, hist_img.rows - 1), Point(maxv + 255, 0), COLOR_GRAY40);
		for (int i = 0; i < 512; i++)
		{
			int v = (hist_img.rows - 1)*(histmax - min(histmax, hist[i])) / (double)histmax;
			line(hist_img, Point(i, hist_img.rows - 1), Point(i, v), COLOR_WHITE);
		}

		displayOverlay(wname, format("%d %d", minv, maxv));
		imshow("hist", hist_img);
		imshow(wname, show);
		key = waitKey(1);

		if (!isWait)break;
	}

	if (!isWait)destroyWindow(wname);
}

void filter2DTest(Mat& src)
{
	Mat dst;
	Mat kernel = cv::getGaussianKernel(11, 3, CV_32F);
	filter2D(src, dst, -1, kernel*kernel.t());

	imshowNormalize("a", dst);
	waitKey();
}

int main(int argc, char** argv)
{
#pragma region setup
	//cout << cv::getBuildInformation() << endl;


	//Mat img = imread("img/lenna.png");
	Mat img = imread("img/Kodak/kodim07.png");
	//filter2DTest(img); return 0;
#pragma endregion

#pragma region core
	testTiling(img); return 0;
	//copyMakeBorderTest(img); return 0;
	//testConcat(); return 0;
	//testsimd(); return 0;

	//testHistogram(); return 0;
	//testPlot(); return 0;
	//testPlot2D(); return 0;
	//guiGuidedImageFilterTest();
	//guiHazeRemoveTest();

	//testCropZoom(); return 0;
	//testAddNoise(img); return 0;
	//testLocalPSNR(img); return 0;
	//testPSNR(img); return 0;
	//resize(img, a, Size(513, 513));
	testSplitMerge(img); return 0;
	//testHistgram(img);
	//testRGBHistogram();
	//testRGBHistogram2();
	//testTimer(img);
	//testAlphaBlend(left, right);
	//testAlphaBlendMask(left, right);
	//guiDissolveSlide(left, dmap);
	//guiLocalDiffHistogram(img);
	//guiContrast(img);
	//guiContrast(guiCropZoom(img));
	
#pragma endregion
	//Mat img = imread("img/cameraman.png",0);
	//Mat img = imread("img/barbara.png", 0);

	guiWeightedHistogramFilterTest();
	//guiWeightedHistogramFilterTest(img,img);
	
	//test stereo
#pragma region stereo
	//Mat right = imread("left.png");
	//Mat left = imread("right.png");
	Mat left = imread("img/stereo/Dolls/view1.png");
	//resize(left, left, Size(), 1, 0.25);
	Mat right = imread("img/stereo/Dolls/view5.png");
	//resize(right, right, Size(), 1, 0.25);
	//Mat dmap = imread("img/stereo/Dolls/disp1.png", 0);
	Mat leftg, rightg;
	cvtColor(left, leftg, COLOR_BGR2GRAY);
	cvtColor(right, rightg, COLOR_BGR2GRAY);
	//guiShift(left, right, 300);
	//StereoBMEx sbm(0, get_simd_ceil(120, 16), get_simd_ceil(100, 16), 5);
	StereoBase sbm(5, get_simd_ceil(16, 16), get_simd_ceil(100, 16));
	cp::StereoEval eval;
	Mat disp;
	sbm.gui(left, right, disp, eval);
	//sbm.check(leftg, rightg, disp);
	//guiStereoBMTest(leftg, rightg, get_simd_ceil(326, 16), get_simd_ceil(100, 16)); return 0;
	//guiStereoSGBMTest(left, right, get_simd_ceil(326, 16), get_simd_ceil(100, 16)); return 0;
#pragma endregion
	

	//guiUpsampleTest(img);return 0;
	//guiDomainTransformFilterTest(img);
	//guiMedianFilterTest(img);
	//VisualizeDenormalKernel vdk;
	//vdk.run(img);
	//return 0;
	//VizKernel vk;
	//vk.run(img, 2);

	
	//guiShift(left,right); return 0;
	//
	//iirGuidedFilterTest2(img); return 0;
	//iirGuidedFilterTest1(dmap, left); return 0;
	//iirGuidedFilterTest(); return 0;
	//iirGuidedFilterTest(left); return 0;
	//fitPlaneTest(); return 0;
	//guiWeightMapTest(); return 0;


	//guiGeightedJointBilateralFilterTest();
	//Mat haze = imread("img/haze/haze2.jpg"); guiHazeRemoveTest(haze);
	guiDenoiseTest(img);
	//Mat ff3 = imread("img/pixelart/ff3.png");

	Mat src = imread("img/lenna.png");

	//Mat src = imread("img/Kodak/kodim07.png",0);
	guiIterativeBackProjectionTest(src);
	//Mat src = imread("img/Kodak/kodim15.png",0);

	//Mat src = imread("img/cave-flash.png");
	//Mat src = imread("img/feathering/toy.png");
	//Mat src = imread("Clipboard01.png");

	//timeGuidedFilterTest(src);
	//Mat src = imread("img/flower.png");
	//Mat src = imread("img/teddy_disp1.png");
	//Mat src_ = imread("img/stereo/Art/view1.png",0);
	//	Mat src;
	//	copyMakeBorder(src_,src,0,1,0,1,BORDER_REPLICATE);

	//Mat src = imread("img/lenna.png", 0);

	//Mat src = imread("img/stereo/Dolls/view1.png");
	//guiDenoiseTest(src);
	//guiBilateralFilterTest(src);
	Mat ref = imread("img/stereo/Dolls/view6.png");
	//guiColorCorrectionTest(src, ref); return 0;
	//Mat src = imread("img/flower.png");
	//guiAnalysisImage(src);
	Mat dst = src.clone();
	//paralleldenoise(src, dst, 5);
	//Mat disp = imread("img/stereo/Dolls/disp1.png", 0);
	//	Mat src;
	Mat dest;

	//guiCrossBasedLocalFilter(src); return 0;
	

	//eraseBoundary(src,10);
	//	resize(src,mega,Size(1024,1024));
	//resize(src,mega,Size(640,480));

	//guiDualBilateralFilterTest(src,disp);
	//guiGausianFilterTest(src); return 0;

	//guiCoherenceEnhancingShockFilter(src, dest);

	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	//guiDisparityPlaneFitSLICTest(src, ref, disp); return 0;
	//getPSNRRealtimeO1BilateralFilterKodak();
	//guiRealtimeO1BilateralFilterTest(src); return 0;

	Mat flashImg = imread("img/flash/cave-flash.png");
	Mat noflashImg = imread("img/flash/cave-noflash.png");
	Mat noflashImgGray; cvtColor(noflashImg, noflashImgGray, COLOR_BGR2GRAY);
	Mat flashImgGray; cvtColor(flashImg, flashImgGray, COLOR_BGR2GRAY);
	Mat fmega, nmega;
	resize(flashImgGray, fmega, Size(1024, 1024));
	resize(noflashImg, nmega, Size(1024, 1024));

	//guiEdgePresevingFilterOpenCV(src);
	//guiSLICTest(src);
	

	//guiJointRealtimeO1BilateralFilterTest(noflashImgGray, flashImgGray); return 0;
	//guiJointRealtimeO1BilateralFilterTest(noflashImg, flashImgGray); return 0;
	//guiJointRealtimeO1BilateralFilterTest(noflashImgGray, flashImg); return 0;
	//guiJointRealtimeO1BilateralFilterTest(noflashImg, flashImg); return 0;

	//guiWeightedHistogramFilterTest(noflashImgGray, flashImg); return 0;
	//guiRealtimeO1BilateralFilterTest(noflashImgGray); return 0;
	//guiRealtimeO1BilateralFilterTest(src); return 0;
	//guiDMFTest(nmega, nmega, fmega); return 0;
	//guiGausianFilterTest(src); return 0;


	//guiAlphaBlend(ff3,ff3);
	//guiJointNearestFilterTest(ff3);
	//guiViewSynthesis();

	//guiSeparableBilateralFilterTest(src);
	//guiBilateralFilterSPTest(mega);
	//guiRecursiveBilateralFilterTest(mega);
	//fftTest(src);

	Mat feather = imread("img/feathering/toy-mask.png");
	//Mat guide = imread("img/feathering/toy.png");
	//timeBirateralTest(mega);

	Mat flash = imread("img/cave-flash.png");
	Mat noflash = imread("img/cave-noflash.png");
	Mat disparity = imread("img/teddy_disp1.png", 0);
	//guiJointBirateralFilterTest(noflash,flash);
	//guiBinalyWeightedRangeFilterTest(disparity);
	//guiCodingDistortionRemoveTest(disparity);
	//guiJointBinalyWeightedRangeFilterTest(noflash,flash);

	//guiNonLocalMeansTest(src);

	//application 
	//guiDetailEnhancement(src);
	//	guiDomainTransformFilterTest(mega);
	return 0;
}