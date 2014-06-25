#include "opencp.hpp"
#include <opencv2/core/internal.hpp>

void warpShiftSubpix(InputArray src_, OutputArray dest_, double shiftx, double shifty, const int inter_method)
{
	Mat src = src_.getMat();
	dest_.create( src.size(), src.type() );
	Mat dest = dest_.getMat();

	Mat aff = Mat::zeros(2,3,CV_64F);

	aff.at<double>(0,0)=1.0;
	aff.at<double>(0,1)=0.0;
	aff.at<double>(0,2)=shiftx;

	aff.at<double>(1,0)=0.0;
	aff.at<double>(1,1)=1.0;
	aff.at<double>(1,2)=shifty;

	warpAffine(src,dest,aff,src.size(), inter_method,0);
}

void warpShift_(Mat& src, Mat& dst, int shiftx, int shifty)
{
	Mat dest =Mat::zeros(src.size(),src.type());
	
	int width = src.cols;
	int height = src.rows;
	if(shiftx>=0 &&shifty>=0)
	{
		Mat d = dest(Rect( shiftx, shifty, width-shiftx, height-shifty ));
		Mat(src(Rect(0, 0, width-shiftx, height-shifty))).copyTo(d);
	}
	else if(shiftx>=0 &&shifty<0)
	{
		Mat d = dest(Rect( shiftx, 0, width-shiftx, height+shifty));
		Mat(src(Rect( 0, -shifty, width-shiftx, height+shifty))).copyTo(d);
	}
	else if(shiftx<0 &&shifty<0)
	{
		Mat d = dest(Rect( 0, 0, width+shiftx, height+shifty));
		Mat(src(Rect( -shiftx, -shifty, width+shiftx, height+shifty))).copyTo(d);
	}
	else if(shiftx<0 &&shifty>=0)
	{
		Mat d = dest(Rect( 0, shifty, width+shiftx, height-shifty));
		Mat(src(Rect( -shiftx, 0, width+shiftx, height-shifty))).copyTo(d);
	}
	dest.copyTo(dst);
}


void warpShift_(Mat& src, Mat& dest, int shiftx, int shifty, int borderType)
{
	if(dest.empty())dest=Mat::zeros(src.size(),src.type());
	else dest.setTo(0);

	int width = src.cols;
	int height = src.rows;
	if(shiftx>=0 &&shifty>=0)
	{
		Mat im; copyMakeBorder(src,im,shifty,0,shiftx,0, borderType);
		Mat(im(Rect(0, 0, width, height))).copyTo(dest);
	}
	else if(shiftx>=0 &&shifty<0)
	{
		Mat im; copyMakeBorder(src,im,0,-shifty,shiftx,0, borderType);
		Mat(im(Rect(0, -shifty, width, height))).copyTo(dest);
	}
	else if(shiftx<0 &&shifty<0)
	{
		Mat im; copyMakeBorder(src,im,0,-shifty,0, -shiftx, borderType);
		Mat(im(Rect(-shiftx, -shifty, width, height))).copyTo(dest);
	}
	else if(shiftx<0 &&shifty>=0)
	{
		Mat im; copyMakeBorder(src,im,shifty, 0, 0, -shiftx, borderType);
		Mat(im(Rect(-shiftx, 0, width, height))).copyTo(dest);
	}
}


void warpShift(InputArray src_, OutputArray dest_, int shiftx, int shifty, int borderType)
{
	Mat src = src_.getMat();
	dest_.create( src.size(), src.type() );
	Mat dest = dest_.getMat();

	if(borderType<0)
		warpShift_(src,dest,shiftx,shifty);
	else
		warpShift_(src,dest,shiftx,shifty,borderType);
}