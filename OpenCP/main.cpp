#include "opencp.hpp"
#include "test.hpp"


void depthSubsumpleForgroundBiasedHalf(Mat& src, Mat& dest, int th, int invalid=0)
{
	dest.create(Size(src.cols/2,src.rows/2), src.type());

	for(int j=0; j<dest.rows; j++)
	{
		for(int i=0; i<dest.cols; i++)
		{
			int v=0;
			int sum=0;
			int count=0;
			
			int vmax = max(max(src.at<uchar>(2*j+0,2*i+0),src.at<uchar>(2*j+0,2*i+1)),max(src.at<uchar>(2*j+1,2*i+0),src.at<uchar>(2*j+1,2*i+1)));

			v = src.at<uchar>(2*j+0,2*i+0);
			if(v!=invalid && abs(v-vmax)<th)
			{
				sum+=v;
				count++;
			}
			v = src.at<uchar>(2*j+0,2*i+1);
			if(v!=invalid  && abs(v-vmax)<th)
			{
				sum+=v;
				count++;
			}
			v = src.at<uchar>(2*j+1,2*i+0);
			if(v!=invalid && abs(v-vmax)<th)
			{
				sum+=v;
				count++;
			}
			v = src.at<uchar>(2*j+1,2*i+1);
			if(v!=invalid && abs(v-vmax)<th)
			{
				sum+=v;
				count++;
			}
			if(count!=0) dest.at<uchar>(j,i)=(uchar)(sum/(double)count+0.5);
			else dest.at<uchar>(j,i)=invalid;
		}
	}
}

void depthSubsumpleHalf(Mat& src, Mat& dest, int invalid=0)
{
	dest.create(Size(src.cols/2,src.rows/2), src.type());

	for(int j=0; j<dest.rows; j++)
	{
		for(int i=0; i<dest.cols; i++)
		{
			int v=0;
			int sum=0;
			int count=0;
			v = src.at<uchar>(2*j+0,2*i+0);
			if(v!=invalid)
			{
				sum+=v;
				count++;
			}
			v = src.at<uchar>(2*j+0,2*i+1);
			if(v!=invalid)
			{
				sum+=v;
				count++;
			}
			v = src.at<uchar>(2*j+1,2*i+0);
			if(v!=invalid)
			{
				sum+=v;
				count++;
			}
			v = src.at<uchar>(2*j+1,2*i+1);
			if(v!=invalid)
			{
				sum+=v;
				count++;
			}
			if(count!=0) dest.at<uchar>(j,i)=(uchar)(sum/(double)count+0.5);
			else dest.at<uchar>(j,i)=invalid;
		}
	}
}

int main(int argc, char** argv)
{	
	//Mat ff3 = imread("img/pixelart/ff3.png");
	
	//Mat src = imread("img/lenna.png");
	//Mat src = imread("img/cave-flash.png");
	//Mat src = imread("img/feathering/toy.png");
	//Mat src = imread("Clipboard01.png");
	
	//timeGuidedFilterTest(src);
	//Mat src = imread("img/flower.png");
	//Mat src = imread("img/teddy_disp1.png");
	Mat src_ = imread("img/stereo/Art/view1.png",0);
	Mat src;
	copyMakeBorder(src_,src,0,1,0,1,BORDER_REPLICATE);
		
	Mat dest;

	//Mat src = imread("img/kodim22.png");
	//Mat src = imread("img/teddy_view1.png");
	
	//eraseBoundary(src,10);
	Mat mega;
	resize(src,mega,Size(1024,1024));
//	resize(src,mega,Size(1024,1024));
	//resize(src,mega,Size(640,480));

	//guiGauusianFilterTest(src);
	guiRealtimeO1BilateralFilterTest(src);

	//guiAlphaBlend(ff3,ff3);
	//guiJointNearestFilterTest(ff3);
	//guiViewSynthesis();
	//guiSLICTest(src);
	//guiBilateralFilterTest(src);
	//guiBilateralFilterSPTest(mega);
	//guiRecursiveBilateralFilterTest(mega);
	//fftTest(src);
	
	Mat feather = imread("img/feathering/toy-mask.png");
	//Mat guide = imread("img/feathering/toy.png");
	//timeBirateralTest(mega);

	Mat flash = imread("img/cave-flash.png");
	Mat noflash = imread("img/cave-noflash.png");
	Mat disparity = imread("img/teddy_disp1.png",0);
	//guiJointBirateralFilterTest(noflash,flash);
	//guiBinalyWeightedRangeFilterTest(disparity);
	//guiJointBinalyWeightedRangeFilterTest(noflash,flash);

	//guiNonLocalMeansTest(src);

	//guiIterativeBackProjectionTest(src);

	//application 
	//guiDetailEnhancement(src);
	//guiGuidedFilterTest(mega);
//	guiDomainTransformFilterTest(mega);
	//guiDenoiseTest(src);
	return 0;
}