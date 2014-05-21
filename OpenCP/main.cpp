#include "opencp.hpp"
#include "test.hpp"


int main(int argc, char** argv)
{
	Mat src = imread("lenna.png");
	Mat mega;
	resize(src,mega,Size(1024,1024));
	//guiSLICTest(src);
	//guiBirateralFilterTest(src);
	timeBirateralTest(mega);
	return 0;
}