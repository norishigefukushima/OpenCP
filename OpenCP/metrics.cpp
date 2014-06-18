#include "opencp.hpp"

double YPSNR(const Mat& src1, const Mat& src2)
{
	Mat g1,g2;
	cvtColor(src1,g1,COLOR_BGR2GRAY);
	cvtColor(src2,g2,COLOR_BGR2GRAY);
	return PSNR(g1,g2);
}