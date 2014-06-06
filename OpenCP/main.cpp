#include "opencp.hpp"
#include "test.hpp"


void eraseBoundary(Mat& src, int step)
{
	Mat temp = src(Rect(step,step,src.cols-2*step,src.rows-2*step));
	Mat a;temp.copyTo(a);
	Mat b;
	copyMakeBorder(a,b,step,step,step,step,BORDER_REPLICATE);
	//guiAlphaBlend(b,src);
	b.copyTo(src);

	
}
int main(int argc, char** argv)
{
	Mat src = imread("img/lenna.png");
	//Mat src = imread("Clipboard01.png");
	
	timeGuidedFilterTest(src);
	//Mat src = imread("img/flower.png");

	
	//Mat src = imread("img/kodim22.png");
	//Mat src = imread("img/teddy_view1.png");
	
	eraseBoundary(src,10);
	Mat mega;
	resize(src,mega,Size(1024,1024));

	//guiSLICTest(src);
	//guiBilateralFilterTest(mega);
	//guiRecursiveBilateralFilterTest(mega);

	
	Mat feather = imread("img/feathering/toy-mask.png");
	//Mat guide = imread("img/feathering/toy.png");
	//timeBirateralTest(mega);

	Mat flash = imread("img/cave-flash.png");
	Mat noflash = imread("img/cave-noflash.png");
	Mat disparity = imread("img/teddy_disp1.png",0);
	//guiJointBirateralFilterTest(noflash,flash);
	//guiBinalyWeightedRangeFilterTest(disparity);
	//guiJointBinalyWeightedRangeFilterTest(noflash,flash);

	//guiNonLocalMeansTest(src);

	//guiIterativeBackProjectionTest(src);

	//application 
	//guiDetailEnhancement(src);
	guiGuidedFilterTest(mega);
	guiDenoiseTest(src);
	return 0;
}