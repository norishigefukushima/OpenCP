#include "opencp.hpp"

void iterativeBackProjectionDeblurGaussian(const Mat& src, Mat& dest, const Size ksize, const double sigma, const double lambda, const int iteration)
{
	Mat srcf;
	Mat destf;
	Mat subf;
	src.convertTo(srcf,CV_32FC3);
	src.convertTo(destf,CV_32FC3);
	Mat bdest;

	double maxe = DBL_MAX;

	int i;
	for(i=0;i<iteration;i++)
	{
		GaussianBlur(destf,bdest,ksize,sigma);
		subtract(srcf,bdest,subf);
		double e = norm(subf);

		GaussianBlur(subf,subf,ksize,sigma);

		destf += lambda*subf;

		//printf("%f\n",e);
		if(i!=0)
		{
			if(maxe>e)
			{
				maxe=e;
			}
			else break;
		}
		//if(isWrite)
		//{
		//	destf.convertTo(dest,CV_8UC3);
		//	imwrite(format("B%03d.png",i),dest);
		//}
	}
	printf("%d\n",i);
	destf.convertTo(dest,CV_8UC3);
}

void iterativeBackProjectionDeblurBilateral(const Mat& src, Mat& dest, const Size ksize, const double sigma_color, const double sigma_space, const double lambda, const int iteration)
{
	Mat srcf;
	Mat destf;
	Mat subf;
	src.convertTo(srcf,CV_32FC3);
	src.convertTo(destf,CV_32FC3);
	Mat bdest;

	double maxe = DBL_MAX;

	int i;
	for(i=0;i<iteration;i++)
	{
		GaussianBlur(destf,bdest,ksize,sigma_space);

		subtract(srcf,bdest,subf);
		
		//normarize from 0 to 255 for joint birateral filter (range 0 to 255)
		double minv, maxv;
		minMaxLoc(subf,&minv,&maxv);
		subtract(subf,Scalar(minv,minv,minv),subf);
		multiply(subf,Scalar(2,2,2),subf);
		
		jointBilateralFilter(subf,destf,subf,ksize,sigma_color,sigma_space);

		multiply(subf,Scalar(0.5,0.5,0.5),subf);
		add(subf,Scalar(minv,minv,minv),subf);
		
		double e = norm(subf);
		
		//imshow("a",subf);
		multiply(subf,Scalar(lambda,lambda,lambda),subf);
		
		add(destf,subf,destf);
		//destf += ((float)lambda)*subf;

		//printf("%f\n",e);
	/*	if(i!=0)
		{
			if(maxe>e)
			{
				maxe=e;
			}
			else break;
		}*/
		//if(isWrite)
		//{
		//	destf.convertTo(dest,CV_8UC3);
		//	imwrite(format("B%03d.png",i),dest);
		//}
	}
	destf.convertTo(dest,CV_8UC3);
}