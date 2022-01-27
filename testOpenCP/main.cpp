#include <opencp.hpp>
#include "test.hpp"
using namespace std;
using namespace cv;
using namespace cp;

void testIsSame()
{
	Mat a(256, 256, CV_8U);
	Mat b(256, 256, CV_8U);
	randu(a, 0, 255);
	randu(b, 0, 255);
	isSame(a, b);
	isSame(a, a);
	isSame(a, b, 10);
	isSame(a, a, 10);
}

string getInformation()
{
	string ret = "version: " + cv::getVersionString() + "\n";
	ret += "==============\n";
	if (cv::useOptimized()) ret += "cv::useOptimized: true\n";
	else ret += "cv::useOptimized: false\n";
	if (cv::ipp::useIPP()) ret += "cv::ipp::useIPP: true\n";
	else ret += "cv::ipp::useIPP: true\n";
	ret += cv::ipp::getIppVersion() + "\n";
	ret += format("cv::getNumberOfCPUs = %d\n", cv::getNumberOfCPUs());
	ret += format("cv::getNumThreads = %d\n", cv::getNumThreads());
	ret += getCPUFeaturesLine() + "\n";
	ret += "==============\n";

	return ret;
}

template<typename T>
void detailBoostLinear_(Mat& src, Mat& smooth, Mat& dest, const double boost)
{
	T* imptr = src.ptr<T>();
	T* smptr = smooth.ptr<T>();
	T* dptr = dest.ptr<T>();
	const int size = src.size().area();
	for (int i = 0; i < size; i++)
	{
		dptr[i] = saturate_cast<T>(smptr[i] + boost * (imptr[i] - smptr[i]));
	}
}

template<typename T>
void detailBoostGauss_(Mat& src, Mat& smooth, Mat& dest, const double boost, const double sigma)
{
	T* imptr = src.ptr<T>();
	T* smptr = smooth.ptr<T>();
	T* dptr = dest.ptr<T>();
	const int size = src.size().area();
	for (int i = 0; i < size; i++)
	{
		const double sub = imptr[i] - smptr[i];
		dptr[i] = saturate_cast<T>(smptr[i] + boost*exp(sub*sub/(-2.0*sigma*sigma)) * (sub));
	}
}

inline double contrastLinear_(double val, double a, double c)
{
	return a * (val - c) + c;
}

inline double contrastGauss_(double val, double a, double c, double coeff)
{
	double sub = (val - c);
	return val - a * exp(coeff * sub*sub) * sub;
}

inline double contrastGamma_(double val, double gamma)
{
	return pow(val / 255.0, gamma) * 255.0;
}


template<typename T>
void baseContrastLinear_(Mat& src, Mat& smooth, Mat& dest, const double a, const double c, const int method)
{
	T* imptr = src.ptr<T>();
	T* smptr = smooth.ptr<T>();
	T* dptr = dest.ptr<T>();
	const int size = src.size().area();
	for (int i = 0; i < size; i++)
	{
		dptr[i] = saturate_cast<T>(contrastLinear_(smptr[i], a, c) + (imptr[i] - smptr[i]));
	}
}

template<typename T>
void baseContrastGauss_(Mat& src, Mat& smooth, Mat& dest, const double a, const double c, const double sigma, const int method)
{
	T* imptr = src.ptr<T>();
	T* smptr = smooth.ptr<T>();
	T* dptr = dest.ptr<T>();
	const int size = src.size().area();
	const double coeff = -1.0 / (2.0 * sigma * sigma);
	for (int i = 0; i < size; i++)
	{
		dptr[i] = saturate_cast<T>(contrastGauss_(smptr[i], a, c, coeff) + (imptr[i] - smptr[i]));
		//dptr[i] = saturate_cast<T>(contrastLinear_(smptr[i], a, c) + (imptr[i] - smptr[i]));
	}
}

template<typename T>
void baseContrastGamma_(Mat& src, Mat& smooth, Mat& dest, const double gamma)
{
	T* imptr = src.ptr<T>();
	T* smptr = smooth.ptr<T>();
	T* dptr = dest.ptr<T>();
	const int size = src.size().area();
	for (int i = 0; i < size; i++)
	{
		dptr[i] = saturate_cast<T>(contrastGamma_(smptr[i], gamma) + (imptr[i] - smptr[i]));
		//dptr[i] = saturate_cast<T>(contrastLinear_(smptr[i], a, c) + (imptr[i] - smptr[i]));
	}
}

void baseContrast(cv::InputArray src, cv::InputArray smooth, cv::OutputArray dest, const double a, const double c, const double sigma, const int method)
{
	dest.create(src.size(), src.type());
	Plot pt;
	pt.setXRange(0, 256);
	pt.setYRange(0, 256);
	const double coeff = -1.0 / (2.0 * sigma * sigma);
	for (int i = 0; i < 256; i++)
	{
		pt.push_back(i, contrastLinear_(i, a, c), 0);
		pt.push_back(i, contrastGauss_(i, a, c, coeff), 1);
		pt.push_back(i, contrastGamma_(i, a), 2);

	}
	pt.plot("contrast", false);

	Mat im = src.getMat();
	Mat sm = smooth.getMat();
	Mat dst = dest.getMat();
	if (method == 0)
	{
		if (im.depth() == CV_8U)  baseContrastGamma_<uchar>(im, sm, dst, a);
		if (im.depth() == CV_32F) baseContrastGamma_<float>(im, sm, dst, a);
		if (im.depth() == CV_64F) baseContrastGamma_<double>(im, sm, dst, a);
	}
	else if (method == 1)
	{
		if (im.depth() == CV_8U)  baseContrastLinear_<uchar>(im, sm, dst, a, c, method);
		if (im.depth() == CV_32F) baseContrastLinear_<float>(im, sm, dst, a, c, method);
		if (im.depth() == CV_64F) baseContrastLinear_<double>(im, sm, dst, a, c, method);
	}
	else if(method==2)
	{
		if (im.depth() == CV_8U)  baseContrastGauss_<uchar>(im, sm, dst, a, c, sigma, method);
		if (im.depth() == CV_32F) baseContrastGauss_<float>(im, sm, dst, a, c, sigma, method);
		if (im.depth() == CV_64F) baseContrastGauss_<double>(im, sm, dst, a, c, sigma, method);
	}
	else if (method == 3)
	{
		if (im.depth() == CV_8U)  baseContrastGamma_<uchar>(im, sm, dst, a);
		if (im.depth() == CV_32F) baseContrastGamma_<float>(im, sm, dst, a);
		if (im.depth() == CV_64F) baseContrastGamma_<double>(im, sm, dst, a);
	}
}

void detailBoost(cv::InputArray src, cv::InputArray smooth, cv::OutputArray dest, const double boost, const double sigma, const int method)
{
	dest.create(src.size(), src.type());

	Mat im = src.getMat();
	Mat sm = smooth.getMat();
	Mat dst = dest.getMat();
	if (method == 0)
	{
		if (im.depth() == CV_8U)  detailBoostGauss_<uchar>(im, sm, dst, boost, sigma);
		if (im.depth() == CV_32F) detailBoostGauss_<float>(im, sm, dst, boost, sigma);
		if (im.depth() == CV_64F) detailBoostGauss_<double>(im, sm, dst, boost, sigma);
	}
	else if (method == 1)
	{
		if (im.depth() == CV_8U) detailBoostLinear_<uchar>(im, sm, dst, boost  );
		if (im.depth() == CV_32F) detailBoostLinear_<float>(im, sm, dst, boost );
		if (im.depth() == CV_64F) detailBoostLinear_<double>(im, sm, dst, boost);
	}
}

void detailTest()
{
	string wname = "enhance";
	namedWindow(wname);
	int boost = 100; createTrackbar("boost", wname, &boost, 300);
	int center = 128; createTrackbar("center", wname, &center, 255);
	int sigma = 30; createTrackbar("sigma", wname, &sigma, 300);
	int sr = 30; createTrackbar("sr", wname, &sr, 300);
	int ss = 5; createTrackbar("ss", wname, &ss, 20);
	int method = 0; createTrackbar("method", wname, &method, 2);
	//Mat src = imread("img/flower.png", 0);
	Mat src = imread("img/lenna.png", 0);
	//addNoise(src, src, 5);
	Mat smooth, smooth2;
	
	int key = 0;
	Mat show, show2;
	cp::ConsoleImage ci;
	while (key != 'q')
	{
		const int d = 2 * (ss * 3) + 1;
		cv::bilateralFilter(src, smooth, d, sr, ss);
		edgePreservingFilter(src, smooth2, 1, ss, sr / 255.0);
		//cp::highDimensionalGaussianFilterPermutohedralLattice(src, smooth2, sr, ss);
		
		detailBoost(src, smooth, show, boost*0.01, sigma, method);
		detailBoost(src, smooth2, show2, boost*0.01,sigma, method);
		//baseContrast(src, smooth, show, boost * 0.01, center, sigma, 0);
		//baseContrast(src, smooth2, show2, boost * 0.01, center, sigma, 0);

		ci("PSNR smooth  %f", getPSNR(smooth, smooth2));
		ci("PSNR enhance %f", getPSNR(show, show2));
		ci.show();
		imshow(wname, show);
		imshow(wname+"2", show2);

		key = waitKey(1);
		if (key == 'c')guiAlphaBlend(show, show2);
	}
}

int main(int argc, char** argv)
{	
	/*__m256i a = _mm256_set_step_epi32(0);
	__m256i b = _mm256_set_step_epi32(8);
	__m256i c = _mm256_set_step_epi32(16);
	__m256i d = _mm256_set_step_epi32(24);
	print_uchar(_mm256_packus_epi16(_mm256_packus_epi32(a, b), _mm256_packus_epi32(c, d)));
	return 0;*/
	/*
	cout << getInformation() << endl; return 0;
	cout << cv::getBuildInformation() << endl;
	cv::ipp::setUseIPP(false);
	cv::setUseOptimized(false);
	*/
	//testMultiScaleFilter(); return 0;
	//testIsSame(); return 0;


	//detailTest(); return 0;
#pragma region setup
	//Mat img = imread("img/lenna.png");
	Mat img = imread("img/Kodak/kodim07.png");
	Mat imgg; cvtColor(img, imgg, COLOR_BGR2GRAY);
	//Mat img = imread("img/cameraman.png",0);
	//Mat img = imread("img/barbara.png", 0);
	//filter2DTest(img); return 0;
	
#pragma endregion


#pragma region core
	//guiPixelizationTest();
	//testStreamConvert8U(); return 0;
	//testKMeans(img); return 0;
	//testTiling(img); return 0;
	//copyMakeBorderTest(img); return 0;
	//testSplitMerge(img); return 0;
	//consoleImageTest(); return 0;
	//testConcat(); return 0;
	//testsimd(); return 0;

	//testHistogram(); return 0;
	//testPlot(); return 0;
	//testPlot2D(); return 0;

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
	//testStat(); return 0;
	//testDestinationTimePrediction(img); return 0;
	//testAlphaBlend(left, right);
	//testAlphaBlendMask(left, right);
	//guiDissolveSlide(left, dmap);
	//guiLocalDiffHistogram(img);
	//guiContrast(img);
	//guiContrast(guiCropZoom(img));
	//testVideoSubtitle();
#pragma endregion

#pragma region imgproc
	//guiCvtColorPCATest(); return 0;
#pragma endregion

#pragma region stereo
	//testStereoBase(); return 0;
	//testCVStereoBM(); return 0;
	//testCVStereoSGBM(); return 0;
#pragma endregion

#pragma region filter
	//testGuidedImageFilter(Mat(), Mat()); return 0;
	//highDimentionalGaussianFilterTest(imgg); return 0;
	//highDimentionalGaussianFilterTest(img); return 0;
	guiDenoiseTest(img);
	//testWeightedHistogramFilterDisparity(); return 0;
	//testWeightedHistogramFilter();return 0;
#pragma endregion 

	//guiUpsampleTest(img); return 0;
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
	return 0;
}