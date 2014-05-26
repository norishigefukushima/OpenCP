#include "opencp.hpp"
#include "test.hpp"


int main(int argc, char** argv)
{
	Mat src = imread("img/lenna.png");
	Mat mega;
	

	//guiSLICTest(src);
	//guiBirateralFilterTest(src);

	resize(src,mega,Size(1024,1024));
	//timeBirateralTest(mega);

	Mat flash = imread("img/cave-flash.png");
	Mat noflash = imread("img/cave-noflash.png");
	Mat disparity = imread("img/teddy_disp1.png",0);
	//guiJointBirateralFilterTest(noflash,flash);
	//guiBinalyWeightedRangeFilterTest(disparity);
	//guiJointBinalyWeightedRangeFilterTest(noflash,flash);

	guiNonLocalMeansTest(src);
	return 0;
}