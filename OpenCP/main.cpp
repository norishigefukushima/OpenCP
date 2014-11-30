#include "opencp.hpp"
#include "test.hpp"

int main(int argc, char** argv)
{
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
		

	Mat src = imread("img/stereo/Art/view1.png");
	Mat disp = imread("img/stereo/Art/disp1.png");
//	Mat src;
	Mat dest;

	//Mat src = imread("img/kodim22.png");
	//Mat src = imread("img/teddy_view1.png");
	
	//eraseBoundary(src,10);
	Mat mega;
	resize(src,mega,Size(1024,1024));
//	resize(src,mega,Size(1024,1024));
	//resize(src,mega,Size(640,480));

	//guiDualBilateralFilterTest(src,disp);
	guiGausianFilterTest(src);
	//guiRealtimeO1BilateralFilterTest(src);

	//guiAlphaBlend(ff3,ff3);
	//guiJointNearestFilterTest(ff3);
	//guiViewSynthesis();
	//guiSLICTest(src);
	//guiBilateralFilterTest(src);
	guiSeparableBilateralFilterTest(src);
	//guiBilateralFilterSPTest(mega);
	//guiRecursiveBilateralFilterTest(mega);
	//fftTest(src);
	
	Mat feather = imread("img/feathering/toy-mask.png");
	//Mat guide = imread("img/feathering/toy.png");
	//timeBirateralTest(mega);

	Mat flash = imread("img/cave-flash.png");
	Mat noflash = imread("img/cave-noflash.png");
	Mat disparity = imread("img/teddy_disp1.png",0);
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