#include "opencp.hpp"

double YPSNR(const Mat& src1, const Mat& src2)
{
	Mat g1,g2;
	cvtColor(src1,g1,COLOR_BGR2GRAY);
	cvtColor(src2,g2,COLOR_BGR2GRAY);
	return PSNR(g1,g2);
}

double calcBadPixel(const Mat& src, const Mat& ref, int threshold)
{
	Mat g1,g2;
	if(src.channels()==3)
	{
		cvtColor(src,g1,CV_BGR2GRAY);
		cvtColor(ref,g2,CV_BGR2GRAY);
	}
	else
	{
		g1=src;
		g2=ref;
	}
	Mat temp;
	absdiff(g1,g2,temp);
	Mat mask;
	compare(temp,threshold,mask,CMP_GE);
	return 100.0*countNonZero(mask)/src.size().area();
}