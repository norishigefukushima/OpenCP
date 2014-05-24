#include "opencp.hpp"
#include "test.hpp"



int main(int argc, char** argv)
{
	Mat src = imread("lenna.png");
	Mat mega;
	resize(src,mega,Size(1024,1024));

	//guiSLICTest(src);
	//guiBirateralFilterTest(src);
	//timeBirateralTest(mega);

	Mat flash = imread("img/cave-flash.png");
	Mat noflash = imread("img/cave-noflash.png");

	guiJointBirateralFilterTest(noflash,flash);
	return 0;
}