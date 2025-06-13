#include <opencp.hpp>
#include "test.hpp"

using namespace std;
using namespace cv;
using namespace cp;
void benchmark()
{
	benchStreamSet();
}

void testCore(Mat& img)
{
	testCSV(); return;
	//testIsSame(); return 0;
	testCorrelationCoefficient(); return;
	//webPAnimationTest(); return 0;
	//guiPixelizationTest();
	//testStreamConvert8U(); return 0;
	testKMeans(img); return;
	//testTiling(img); return 0;
	//copyMakeBorderTest(img); return 0;
	//testSplitMerge(img); return 0;
	//consoleImageTest(); return 0;
	//testConcat(); return 0;
	//testsimd(); return 0;

	//testHistogram(); return 0;
	//testPlot(); return 0;
	testPlot2D(); return;

	//guiHazeRemoveTest();

	//testCropZoom(); return 0;
	//testAddNoise(img); return 0;
	//testLocalPSNR(img); return 0;
	//testPSNR(img); return 0;
	//resize(img, a, Size(513, 513));
	//testHistgram(img);
	//testRGBHistogram();
	//testRGBHistogram2();
	//testTimer(img);
	//testMatInfo(); return 0;
	//testStat(); return;
	//testDestinationTimePrediction(img); return 0;
	//testAlphaBlend(left, right);
	//testAlphaBlendMask(left, right);
	//guiDissolveSlide(left, dmap);
	//guiLocalDiffHistogram(img);
	//guiContrast(img);
	//guiContrast(guiCropZoom(img));
	//testVideoSubtitle();
	//guiWindowFunction();
}

void testImgproc(Mat& img)
{
	//guiCvtColorPCATest(); return 0;
}

void testFilter(Mat& img)
{
	//rangeBlurFilterRef(aa, t0, 5, 3);
	//rangeBlurFilter(aa, t1, 5, 3);
	//guiAlphaBlend(convert(t0,CV_8U), convert(t1,CV_8U));
	//filter2DTest(img); return 0;
	//testUnnormalizedBilateralFilter(); return 0;
	//detailTest(); return 0;
	//testGuidedImageFilter(Mat(), Mat()); return 0;
	testGuidedBilateralFilter(img); return;
	//highDimentionalGaussianFilterTest(img); return;
	//highDimentionalGaussianFilterHSITest(); return 0;
	//guiDenoiseTest(img);
	//testWeightedHistogramFilterDisparity(); return 0;
	//testWeightedHistogramFilter();return 0;
	//guiUpsampleTest(img); return 0;
	guiDomainTransformFilterTest(img);
	//guiMedianFilterTest(img);
	//VisualizeDenormalKernel vdk;
	//vdk.run(img);
	//return 0;
	//VizKernel vk;
	//vk.run(img, 2);

	//iirGuidedFilterTest2(img); return 0;
	//iirGuidedFilterTest1(dmap, left); return 0;
	//iirGuidedFilterTest(); return 0;
	//iirGuidedFilterTest(left); return 0;
	//fitPlaneTest(); return 0;
	//guiWeightMapTest(); return 0;

	//guiGeightedJointBilateralFilterTest();
	//guiHazeRemoveTest();
	//Mat ff3 = imread("img/pixelart/ff3.png");

	Mat src = imread("img/lenna.png");

	//Mat src = imread("img/Kodak/kodim07.png",0);
	//guiIterativeBackProjectionTest(src);
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
	guiBilateralFilterTest(src);
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

	//Mat feather = imread("img/feathering/toy-mask.png");
	//Mat guide = imread("img/feathering/toy.png");
	//timeBirateralTest(mega);
	//Mat disparity = imread("img/teddy_disp1.png", 0);
	//guiJointBirateralFilterTest(noflash,flash);
	//guiBinalyWeightedRangeFilterTest(disparity);
	//guiCodingDistortionRemoveTest(disparity);
	//guiJointBinalyWeightedRangeFilterTest(noflash,flash);

	//guiNonLocalMeansTest(src);

	//application 
	//guiDetailEnhancement(src);
	//guiDomainTransformFilterTest(mega);
}

void testStereo()
{
	//testStereoBase(); return 0;
	//testCVStereoBM(); return 0;
	//testCVStereoSGBM(); return 0;
}

class GridImage
{
	cv::Size gridSize;
	int gridPixel;
	int lineWidth = 1;
	int cellSize = 0;
	cv::Scalar backgroundColor = COLOR_GRAY200;
	cv::Scalar gridColor = COLOR_BLACK;

	void setGridImage()
	{
		for (int j = 0; j < gridSize.height; j++)
		{
			for (int i = 0; i < gridSize.width; i++)
			{
				rectangle(image, Rect(i * cellSize, j * cellSize, cellSize, cellSize), gridColor, lineWidth);
			}
		}
	}
public:
	cv::Mat image;

	void overlayKernelColor(int i, int j, int diameter, Scalar color, double ratio)
	{
		const int x = i * cellSize;
		const int y = j * cellSize;
		const int w = cellSize * diameter;
		if (x<0 || y<0 || x + w>image.cols || y + w>image.rows)
		{
			cout << "out of range" << endl;
			return;
		}

		Mat buff = image.clone();
		rectangle(buff, Rect(x, y, w, w), color, cv::FILLED);
		alphaBlend(buff, image, ratio, image);
	}

	void overlayKernelCenterColor(int i, int j, int radius, Scalar color, double ratio)
	{
		const int x = (i - radius) * cellSize;
		const int y = (j - radius) * cellSize;
		const int w = cellSize * (2 * radius + 1);
		if (x<0 || y<0 || x + w>image.cols || y + w>image.rows)
		{
			cout << "out of range" << endl;
			return;
		}

		Mat buff = image.clone();
		rectangle(buff, Rect(x, y, w, w), color, cv::FILLED);
		alphaBlend(buff, image, ratio, image);
	}

	void setGridColor(int i, int j, Scalar color, bool isFill)
	{
		if (isFill) rectangle(image, Rect(i * cellSize, j * cellSize, cellSize, cellSize), color, cv::FILLED);
		else rectangle(image, Rect(i * cellSize, j * cellSize, cellSize, cellSize), color, lineWidth);
	}

	GridImage(Size gridsize, int gridpixel) : gridSize(gridsize), gridPixel(gridpixel)
	{
		cellSize = gridPixel + 2 * lineWidth;
		image.create(Size(gridSize.width * cellSize, gridSize.height * cellSize), CV_8UC4);
		image.setTo(backgroundColor);
	}
	void show(string wname)
	{
		setGridImage();
		imshow(wname, image);
	}
};
void testGridImage()
{
	int gs = 24;
	int key = 0;
	string wname = "grid";
	namedWindow(wname);
	int r = 10; createTrackbar("r", wname, &r, 10);
	int x = 0; createTrackbar("x", wname, &x, 6);
	int y = 0; createTrackbar("y", wname, &y, 6);
	while (key != 'q')
	{
		GridImage gi(Size(gs, gs), 32);
		for (int j = 0; j < gs; j++)
		{
			for (int i = 0; i < gs; i++)
			{
				if (i % 2 == 0 && j % 2 == 0) gi.setGridColor(i, j, COLOR_ORANGE, true);
			}
		}
		gi.overlayKernelColor(gs / 2 - x, gs / 2 - y, r, Scalar(100, 200, 100), 0.3);
		gi.show(wname);
		key = waitKey(1);
	}
}

void imageIOSpeedTest()
{
	const int iter = 10;
	cp::ConsoleImage ci;
	vector<Mat> img(24);
	for (int i = 1; i < 25; i++)
	{
		img[i - 1] = imread(format("img/Kodak/kodim%02d.png", i));
	}

	int key = 0;
	//string dir = "./";
	string dir = "Z:/";
	while (key != 'q')
	{
		{
			cp::Timer t("", TIME_MSEC, false);
			for (int i = 0; i < img.size(); i++)
			{
				imwrite(dir + format("%d.bmp", i), img[i]);
				Mat a = imread(dir + format("%d.bmp", i));
				imwrite(dir + format("%d.bmp", i), a);
			}
			ci(format("bmp     %f", t.getTime()));
		}
		{
			cp::Timer t("", TIME_MSEC, false);
			for (int i = 0; i < img.size(); i++)
			{
				imwrite(dir + format("%d.ppm", i), img[i]);
				Mat a = imread(dir + format("%d.ppm", i));
				imwrite(dir + format("%d.ppm", i), a);
			}
			ci(format("ppm     %f", t.getTime()));
		}
		{
			vector<int> params;
			params.push_back(IMWRITE_PNG_COMPRESSION);
			params.push_back(0);
			//params.push_back(IMWRITE_PNG_STRATEGY);
			//params.push_back(IMWRITE_PNG_STRATEGY_FIXED);
			cp::Timer t("", TIME_MSEC, false);
			for (int i = 0; i < img.size(); i++)
			{
				imwrite(dir + format("%d.png", i), img[i], params);
				Mat a = imread(dir + format("%d.png", i));
				imwrite(dir + format("%d.png", i), a, params);
			}
			ci(format("png rle %f", t.getTime()));
		}
		{
			vector<int> params;
			params.push_back(IMWRITE_PNG_COMPRESSION);
			params.push_back(0);
			params.push_back(IMWRITE_PNG_STRATEGY);
			params.push_back(IMWRITE_PNG_STRATEGY_RLE);
			cp::Timer t("", TIME_MSEC, false);
			for (int i = 0; i < img.size(); i++)
			{
				imwrite(dir + format("%d.png", i), img[i], params);
				Mat a = imread(dir + format("%d.png", i));
				imwrite(dir + format("%d.png", i), a, params);
			}
			ci(format("png rle %f", t.getTime()));
		}
		{
			vector<int> params;
			params.push_back(IMWRITE_PNG_COMPRESSION);
			params.push_back(0);
			params.push_back(IMWRITE_PNG_STRATEGY);
			params.push_back(IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY);
			cp::Timer t("", TIME_MSEC, false);
			for (int i = 0; i < img.size(); i++)
			{
				imwrite(dir + format("%d.png", i), img[i], params);
				Mat a = imread(dir + format("%d.png", i));
				imwrite(dir + format("%d.png", i), a, params);
			}
			ci(format("png huf %f", t.getTime()));
		}
		if (false)
		{
			vector<int> params;
			params.push_back(IMWRITE_PNG_COMPRESSION);
			params.push_back(9);
			params.push_back(IMWRITE_PNG_STRATEGY);
			params.push_back(IMWRITE_PNG_STRATEGY_FILTERED);
			cp::Timer t("", TIME_MSEC, false);
			for (int i = 0; i < img.size(); i++)
			{
				imwrite(dir + format("%d.png", i), img[i], params);
				Mat a = imread(dir + format("%d.png", i));
				imwrite(dir + format("%d.png", i), a, params);
			}
			ci(format("png max %f", t.getTime()));
		}
		ci.show();
		key = waitKey(1);
	}
}

void DCCITest()
{
	int num = 1;
	Mat a = imread(format("calib/%02d.png", num), 0);
	Mat b, c, d;
	cp::DCCI(a, b);

	resize(a, c, b.size(), 0, 0, INTER_LINEAR);
	//resize(a, d, b.size(), 0, 0, INTER_CUBIC);
	resize(a, d, b.size(), 0, 0, INTER_NEAREST);
	print_debug(a.size());
	print_debug(b.size());
	cp::guiAlphaBlend(b, c);
}
int main(int argc, char** argv)
{
	DCCITest(); return 0;
	//cv::ipp::setUseIPP(false);
	//cv::setUseOptimized(false);
	const bool isShowInfo = false;
	if (isShowInfo)
	{
		cout << getInformation() << endl;
		cout << cv::getBuildInformation() << endl;
		cout << getOpenCLInformation() << endl;
#if USE_OPENCV_CUDA
		cv::cuda::printCudaDeviceInfo(0);
#endif
	}

	imageIOSpeedTest(); return 0;
	//benchmark(); return 0;

#pragma region setup
	//Mat img = imread("img/lenna.png");
	Mat img = imread("img/Kodak/kodim07.png");
	//Mat img = imread("img/Kodak/kodim07.png",0);
	//Mat img = imread("img/cameraman.png",0);
	//Mat img = imread("img/barbara.png", 0);
#pragma endregion

	testCore(img); return 0;
	//testImgproc(img);
	//testStereo();
	//testFilter(img);
	//testSpatialFilter(gra);	
	//testVizPyramid(); return 0;
	testMultiScaleFilter(); return 0;
	return 0;
}