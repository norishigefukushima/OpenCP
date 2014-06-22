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

void generateData()
{

	/*int i=2;
	Mat im = imread(format("img/teddyF/disp%d.pgm",i),0);
	Mat dest;
	//resize(im,dest,Size(im.cols/2,im.rows/2),0,0,INTER_AREA);
	//depthSubsumpleHalf(im,dest,0);
	depthSubsumpleForgroundBiasedHalf(im,dest,8,0);
	imwrite(format("img/teddyF/disp%d.png",i-1),dest);*/

	for(int i = 1;i<9;i++)
	{
	Mat im = imread(format("img/teddyF/im%d.ppm",i));
	Mat dest;
	resize(im,dest,Size(im.cols/2,im.rows/2),0,0,INTER_LINEAR);
	imwrite(format("img/teddyF/view%d.png",i-1),dest);
	}
}
void test()
{
	Mat a = Mat::zeros(Size(15000,10000),CV_8UC3);
	
	
	{
		CalcTime t;
		Mat dest;
		GaussianBlur(a,dest,Size(11,11),11);
	}
}
int main(int argc, char** argv)
{
	//test();

	generateData();
	//Mat ff3 = imread("img/pixelart/ff3.png");
	//guiAlphaBlend(ff3,ff3);
	//guiJointNearestFilterTest(ff3);
	guiViewSynthesis();

	Mat src = imread("img/lenna.png");
	//Mat src = imread("Clipboard01.png");
	
	//timeGuidedFilterTest(src);
	//Mat src = imread("img/flower.png");

	
	//Mat src = imread("img/kodim22.png");
	//Mat src = imread("img/teddy_view1.png");
	
	//eraseBoundary(src,10);
	Mat mega;
	//resize(src,mega,Size(1024,1024));
	//resize(src,mega,Size(1024,1024));
	resize(src,mega,Size(640,480));

	//guiSLICTest(src);
	//guiBilateralFilterTest(mega);
	//guiRecursiveBilateralFilterTest(mega);

	
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
	guiDomainTransformFilterTest(mega);
	//guiDenoiseTest(src);
	return 0;
}