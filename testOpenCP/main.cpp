#include <opencp.hpp>
#include "test.hpp"

using namespace std;


void vectortest(InputArray point)
{
	vector<Mat> a; 
	point.getMatVector(a);
	
	showMatInfo(a[0]);
	cout << a[0] << endl;
	cout << a[1] << endl;


//cout << point.kind() << endl;;
//cout << _InputArray::STD_VECTOR_MAT << endl;
//cout << _InputArray::STD_VECTOR_VECTOR << endl;
//cv:
}

void vectorvectortest(InputArray point)
{
	vector<Mat> a;
	point.getMatVector(a);
	
	//showMatInfo(a[0]);
	showMatInfo(a[0]);
	cout << a[0] << endl;
	cout << a[1] << endl;
}

void InputArrayOutputArrayTest()
{
	vector<Point3f> a;
	a.push_back(Point3f(1, 2, 3));
	a.push_back(Point3f(4, 5, 6));
	a.push_back(Point3f(1, 2, 3));
	a.push_back(Point3f(4, 5, 6));

	//vectortest(a);

	vector<Point3f> b;
	b.push_back(Point3f(7, 8, 9));
	b.push_back(Point3f(10, 11, 12));
	b.push_back(Point3f(7, 8, 9));
	b.push_back(Point3f(10, 11, 12));

	vector<vector<Point3f>> ab;
	ab.push_back(a);
	ab.push_back(b);

	vectorvectortest(ab);
}

int main(int argc, char** argv)
{
	//InputArrayOutputArrayTest(); return 0;
	//fitPlaneTest(); return 0;
	//guiWeightMapTest(); return 0;
	//guiStereo(); return 0;
	//guiPlotTest(); return 0;
	//zoom(argc, argv);return 0;

	//guiGeightedJointBilateralFilterTest();
	//Mat haze = imread("img/haze/haze2.jpg"); guiHazeRemoveTest(haze);
	//Mat fuji = imread("img/fuji.png"); guiDenoiseTest(fuji);
	//Mat ff3 = imread("img/pixelart/ff3.png");

	//Mat src = imread("img/lenna.png");
	//Mat src = imread("img/cave-flash.png");
	//Mat src = imread("img/feathering/toy.png");
	//Mat src = imread("Clipboard01.png");

	//timeGuidedFilterTest(src);
	//Mat src = imread("img/flower.png");
	//Mat src = imread("img/teddy_disp1.png");
	//Mat src_ = imread("img/stereo/Art/view1.png",0);
	//	Mat src;
	//	copyMakeBorder(src_,src,0,1,0,1,BORDER_REPLICATE);

	Mat src = imread("img/stereo/Dolls/view1.png");
	Mat ref = imread("img/stereo/Dolls/view6.png");
	//guiColorCorrectionTest(src, ref); return 0;
	//Mat src = imread("img/flower.png");
	//guiAnalysisImage(src);
	Mat dst = src.clone();
	//paralleldenoise(src, dst, 5);
	Mat disp = imread("img/stereo/Dolls/disp1.png",0 );
	//	Mat src;
	Mat dest;

	//guiDisparityPlaneFitSLICTest(src, ref, disp); return 0;

	//guiCrossBasedLocalFilter(src); return 0;
	//guiHistgramTest(src);
	//Mat src = imread("img/kodim22.png");
	//Mat src = imread("img/teddy_view1.png");

	//eraseBoundary(src,10);
	Mat mega;
	resize(src, mega, Size(1024, 1024));
	//	resize(src,mega,Size(1024,1024));
	//resize(src,mega,Size(640,480));

	//guiDualBilateralFilterTest(src,disp);
	//guiGausianFilterTest(src); return 0;

	//guiCoherenceEnhancingShockFilter(src, dest);
	
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY);

	guiRealtimeO1BilateralFilterTest(src); return 0;

	Mat flashImg = imread("img/flash/cave-flash.png");
	Mat noflashImg = imread("img/flash/cave-noflash.png");
	Mat noflashImgGray; cvtColor(noflashImg, noflashImgGray, COLOR_BGR2GRAY);
	Mat flashImgGray; cvtColor(flashImg, flashImgGray, COLOR_BGR2GRAY);
	Mat fmega, nmega;
	resize(flashImgGray, fmega, Size(1024, 1024));
	resize(noflashImg, nmega, Size(1024, 1024));
	
	guiSLICTest(src);
	//guiEdgePresevingFilterOpenCV(src);

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
	
	//guiBilateralFilterTest(src);
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

	//guiIterativeBackProjectionTest(src);

	//application 
	//guiDetailEnhancement(src);
	//guiGuidedFilterTest(mega);
	//	guiDomainTransformFilterTest(mega);
	//guiDenoiseTest(src);
	return 0;
}