#include "opencp.hpp"

void eraseBoundary(const Mat& src, Mat& dest, int step, int border)
{
	Mat temp = src(Rect(step,step,src.cols-2*step,src.rows-2*step));
	Mat a;temp.copyTo(a);
	Mat b;
	copyMakeBorder(a,dest,step,step,step,step,border);
}

template <class T>
void setTriangleMask(Mat& src)
{
	float aspect = (float)src.cols/src.rows;
	src.setTo(0);
	for(int j=0;j<src.rows;j++)
	{
		T* s = src.ptr<T>(j);
		int v = (int)(j*aspect);
		memset(s,1,(sizeof(T)*src.cols-v));
	}
}

void imshowScale(string name, Mat& dest, double alpha, double beta)
{
	Mat show;
	dest.convertTo(show,CV_8U,alpha, beta);
	imshow(name,show);
}

void patchBlendImage(Mat& src1, Mat& src2, Mat& dest, Scalar linecolor, int linewidth, int direction)
{
	Mat s1,s2;
	if(src1.channels()==src2.channels())
	{
		s1=src1;
		s2=src2;
	}
	else
	{
		if(src1.channels()==1)
		{
			cvtColor(src1,s1,COLOR_GRAY2BGR);
			s2=src2;
		}
		else if(src2.channels()==1)
		{
			s1=src1;
			cvtColor(src2,s2,COLOR_GRAY2BGR);
		}
	}
		
	CV_Assert(src1.size()==src2.size());

	Mat mask = Mat::zeros(src1.size(),CV_8U);
	setTriangleMask<uchar>(mask);

	if(direction==0)
	{
		s2.copyTo(dest);
		s1.copyTo(dest,mask);
		line(dest,Point(src1.cols-1,0),Point(0,src1.rows-1),linecolor,linewidth);
	}
	if(direction==1)
	{
		s2.copyTo(dest);
		s1.copyTo(dest,mask);
		line(dest,Point(src1.cols-1,0),Point(0,src1.rows-1),linecolor,linewidth);
	}
	if(direction==2)
	{
		s2.copyTo(dest);
		flip(mask,mask,0);
		s1.copyTo(dest,mask);
		line(dest,Point(0,0),Point(src1.cols-1,src1.rows-1),linecolor,linewidth);
	}
	if(direction==3)
	{
		s2.copyTo(dest);
		s1.copyTo(dest,mask);
		flip(mask,mask,0);
		line(dest,Point(0,0),Point(src1.cols-1,src1.rows-1),linecolor,linewidth);
	}
}