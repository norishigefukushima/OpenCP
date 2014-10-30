#include "opencp.hpp"


void maxFilter(const Mat& src, Mat& dest, Size kernelSize, int shape)
{	
	Mat element = getStructuringElement(shape, kernelSize);
	dilate(src,dest,element);
}

void minFilter(const Mat& src, Mat& dest, Size kernelSize, int shape)
{
	Mat element = getStructuringElement(shape, kernelSize);
	erode(src,dest,element);
}

void minFilter(const Mat& src, Mat& dest, int radius)
{
	minFilter(src,dest, Size(2*radius+1,2*radius+1));
}

void maxFilter(const Mat& src, Mat& dest, int radius)
{
	maxFilter(src,dest, Size(2*radius+1,2*radius+1));
}