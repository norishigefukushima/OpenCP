#include "opencp.hpp"

int main(int argc, char** argv)
{
	Mat src = imread("lenna.png");

	
	Mat filtered;
	Mat temp;
	src.copyTo(temp);
	for(int i=0;i<15;i++)
	{
		bilateralFilter(temp,filtered,21,35,5,BORDER_REPLICATE);
		filtered.copyTo(temp);
	}

	Mat dest;
	patchBlendImage(src,filtered, dest);
	imshow("show",dest);
	waitKey();

	return 0;
}