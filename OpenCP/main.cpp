#include "opencp.hpp"
#include "test.hpp"

int main(int argc, char** argv)
{
	Mat src = imread("lenna.png");
	//guiSLICTest(src);
	guiBirateralFilterTest(src);

	return 0;
}