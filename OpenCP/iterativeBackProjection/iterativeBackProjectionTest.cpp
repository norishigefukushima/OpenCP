#include "../opencp.hpp"
#include <fstream>
using namespace std;

void guiIterativeBackProjectionTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "Iterative Back Projection";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 1);

	int d_sigma = 20; createTrackbar("d_sigma",wname,&d_sigma,2000);

	int r_sigma = 20; createTrackbar("r_sigma",wname,&r_sigma,2000);
	int r = 4; createTrackbar("r",wname,&r,100);
	int iter = 5;createTrackbar("iteration",wname,&iter,100);
	int l = 10; createTrackbar("lambda",wname,&l,10);
	
	int cs = 80; createTrackbar("cs",wname,&cs,2550);
	//int noise_s = 100; createTrackbar("noise",wname,&noise_s,2550);
	int key = 0;
	Mat show;

	Mat srcf; src.convertTo(srcf,CV_32F);
	Mat blurred;
	while(key!='q')
	{
		double color_sigma = cs/10.0;
		int d = 2*r+1;
		double lambda = l/10.0;

		GaussianBlur(src, blurred,Size(d,d),d_sigma/10.0);

		if(sw==0)
		{
			CalcTime t("ibp");
			iterativeBackProjectionDeblurGaussian(blurred, dest, Size(d,d), r_sigma/10.0, lambda, iter);
		}
		if(sw==1)
		{
			CalcTime t("ibp bilateral");
			iterativeBackProjectionDeblurBirateral(blurred, dest, Size(d,d), color_sigma, r_sigma/10.0,lambda, iter);
		}

//		cout<<"before:"<<PSNR(src,noise)<<endl;
	//	cout<<"filter:"<<PSNR(src,dest)<<endl<<endl;;

		patchBlendImage(blurred,dest,dest,Scalar(255,255,255));
		alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}