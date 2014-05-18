#include "opencp.hpp"


void setTriangleMask_8u(Mat& src)
{
	float aspect = (float)src.cols/src.rows;
	src.setTo(0);
	for(int j=0;j<src.rows;j++)
	{
		uchar* s = src.ptr<uchar>(j);
		memset(s,1,sizeof(uchar)*src.cols-j*aspect);
	}
}

void patchBlendImage(Mat& src1, Mat& src2, Mat& dest, Scalar linecolor, int linewidth, int direction)
{
	CV_Assert(src1.size()==src2.size());

	Mat mask = Mat::zeros(src1.size(),CV_8U);
	setTriangleMask_8u(mask);

	if(direction==0)
	{
		src2.copyTo(dest);
		src1.copyTo(dest,mask);
		line(dest,Point(src1.cols-1,0),Point(0,src1.rows-1),linecolor,linewidth);
	}
	if(direction==1)
	{
		src2.copyTo(dest);
		src1.copyTo(dest,mask);
		line(dest,Point(src1.cols-1,0),Point(0,src1.rows-1),linecolor,linewidth);
	}
}