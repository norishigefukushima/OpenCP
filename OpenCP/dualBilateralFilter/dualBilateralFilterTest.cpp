#include "../opencp.hpp"
#include <fstream>
using namespace std;

void accDualBilateralFilterTest(Mat& src1, Mat& src2)
{
	CV_Assert(src1.channels()==3 || src2.channels()==3);

	Mat srcf1,srcf2;
	src1.convertTo(srcf1,CV_32F);
	src2.convertTo(srcf2,CV_32F);

	Mat g1,g2,gf1, gf2;
	cvtColor(src1,g1,CV_BGR2GRAY);
	cvtColor(src2,g2,CV_BGR2GRAY);
	g1.convertTo(gf1,CV_32F);
	g2.convertTo(gf2,CV_32F);

	Mat ref_gg;
	Mat ref_gc;
	Mat ref_cg;
	Mat ref_cc;
	Mat dest,destf;
	Size kernelSize = Size(11,11);
	float sc1 = 31.3f;
	float sc2 = 20.8f;
	float ss = 5.1f;
	dualBilateralFilter(g1  ,   g2, ref_gg, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	dualBilateralFilter(src1,   g2, ref_cg, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	dualBilateralFilter(g1  , src2, ref_gc, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	dualBilateralFilter(src1, src2, ref_cc, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);

	dualBilateralFilter(g1, g2, dest, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	cout<<"SSE gg"<<": "<<PSNR(ref_gg,dest)<<endl;
	dest.release();

	dualBilateralFilter(src1, g2, dest, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	cout<<"SSE cg"<<": "<<PSNR(ref_cg,dest)<<endl;
	dest.release();

	dualBilateralFilter(g1, src2, dest, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	cout<<"SSE gc"<<": "<<PSNR(ref_gc,dest)<<endl;
	dest.release();

	dualBilateralFilter(src1, src2, dest, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	cout<<"SSE cc"<<": "<<PSNR(ref_cc, dest)<<endl;
	dest.release();

	dualBilateralFilter(gf1, gf2, destf, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	destf.convertTo(dest,CV_8U);
	cout<<"slow gg f"<<": "<<PSNR(ref_gg, dest)<<endl;
	dest.release();destf.release();

	dualBilateralFilter(gf1, srcf2, destf, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	destf.convertTo(dest,CV_8U);
	cout<<"slow gc f"<<": "<<PSNR(ref_gc, dest)<<endl;
	dest.release();destf.release();

	dualBilateralFilter(srcf1, gf2, destf, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	destf.convertTo(dest,CV_8U);
	cout<<"slow cg f"<<": "<<PSNR(ref_cg, dest)<<endl;
	dest.release();destf.release();

	dualBilateralFilter(srcf1, srcf2, destf, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	destf.convertTo(dest,CV_8U);
	cout<<"slow cc f"<<": "<<PSNR(ref_cc, dest)<<endl;
	dest.release();destf.release();

	dualBilateralFilter(gf1, gf2, destf, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	destf.convertTo(dest,CV_8U);
	cout<<"SSE gg f"<<": "<<PSNR(ref_gg, dest)<<endl;
	dest.release();destf.release();

	dualBilateralFilter(gf1, srcf2, destf, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	destf.convertTo(dest,CV_8U);
	cout<<"SSE gc f"<<": "<<PSNR(ref_gc, dest)<<endl;
	dest.release();destf.release();

	dualBilateralFilter(srcf1, gf2, destf, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	destf.convertTo(dest,CV_8U);
	cout<<"SSE cg f"<<": "<<PSNR(ref_cg, dest)<<endl;
	dest.release();destf.release();

	dualBilateralFilter(srcf1, srcf2, destf, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	destf.convertTo(dest,CV_8U);
	cout<<"SSE cc f"<<": "<<PSNR(ref_cc, dest)<<endl;
	dest.release();destf.release();
	
	
	jointDualBilateralFilter(g1, g1, g2, dest, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	cout<<"slow joint ggg"<<": "<<PSNR(ref_gg, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(g1, g1, src2, dest, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	cout<<"slow joint ggc"<<": "<<PSNR(ref_gc, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(g1, src2, g1, dest, kernelSize, sc2,sc1, ss, FILTER_SLOWEST);
	cout<<"slow joint gcg"<<": "<<PSNR(ref_gc, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(src1, src1, g2, dest, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	cout<<"slow joint ccg"<<": "<<PSNR(ref_cg, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(src1, g2, src1, dest, kernelSize, sc2,sc1, ss, FILTER_SLOWEST);
	cout<<"slow joint cgc"<<": "<<PSNR(ref_cg, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(src1, src1, src2, dest, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	cout<<"slow joint ccc"<<": "<<PSNR(ref_cc, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(gf1, gf1, gf2, destf, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	destf.convertTo(dest,CV_8U);
	cout<<"slow joint ggg f"<<": "<<PSNR(ref_gg, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(gf1, gf1, srcf2, destf, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	destf.convertTo(dest,CV_8U);
	cout<<"slow joint ggc f"<<": "<<PSNR(ref_gc, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(gf1, srcf2, gf1, destf, kernelSize, sc2,sc1, ss, FILTER_SLOWEST);
	destf.convertTo(dest,CV_8U);
	cout<<"slow joint gcg f"<<": "<<PSNR(ref_gc, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(srcf1, srcf1, gf2, destf, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	destf.convertTo(dest,CV_8U);
	cout<<"slow joint ccg f"<<": "<<PSNR(ref_cg, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(srcf1, gf2, srcf1, destf, kernelSize, sc2, sc1, ss, FILTER_SLOWEST);
	destf.convertTo(dest,CV_8U);
	cout<<"slow joint cgc f"<<": "<<PSNR(ref_cg, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(srcf1, srcf1, srcf2, destf, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	destf.convertTo(dest,CV_8U);
	cout<<"slow joint ccc f"<<": "<<PSNR(ref_cc, dest)<<endl;
	dest.release();destf.release();
	
	
	jointDualBilateralFilter(g1, g1, g2, dest, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	cout<<"SSE joint ggg"<<": "<<PSNR(ref_gg, dest)<<endl;
	dest.release();destf.release();

	jointDualBilateralFilter(g1, g1, src2, dest, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	cout<<"SSE joint ggc"<<": "<<PSNR(ref_gc, dest)<<endl;
	dest.release();destf.release();
	
	jointDualBilateralFilter(g1, src2, g1, dest, kernelSize, sc2,sc1, ss, FILTER_CIRCLE);
	cout<<"SSE joint gcg"<<": "<<PSNR(ref_gc, dest)<<endl;
	dest.release();destf.release();

	Mat ref_gcc;
	jointDualBilateralFilter(g1, src1, src2, ref_gcc, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);

	jointDualBilateralFilter(g1, src1, src2, dest, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	cout<<"SSE joint gcc"<<": "<<PSNR(ref_gcc, dest)<<endl;
	dest.release();destf.release();
	
	Mat ref_cgg;
	jointDualBilateralFilter(src1, g1, g2, ref_cgg, kernelSize, sc1,sc2, ss, FILTER_SLOWEST);
	jointDualBilateralFilter(src1, g1, g2, dest, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	cout<<"SSE joint cgg"<<": "<<PSNR(ref_cgg, dest)<<endl;
	dest.release();destf.release();
	
	jointDualBilateralFilter(src1, g2, src1, dest, kernelSize, sc2,sc1, ss, FILTER_CIRCLE);
	cout<<"SSE joint cgc"<<": "<<PSNR(ref_cg, dest)<<endl;
	dest.release();destf.release();
	
	jointDualBilateralFilter(src1, src1, src2, dest, kernelSize, sc1,sc2, ss, FILTER_CIRCLE);
	cout<<"SSE joint ccc"<<": "<<PSNR(ref_cc, dest)<<endl;
	//dest.release();destf.release();
	
	//guiAlphaBlend(src1, ref_cc);

	guiAlphaBlend(ref_cc, dest);
	guiAbsDiffCompareGE(ref_cc, dest);
	dest.release();
}


void guiSeparableDualBilateralFilterTest(Mat& src1, Mat& src2)
{
	Mat dest,dest2;

	string wname = "bilateral filter";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 0; createTrackbar("switch",wname,&sw, 4);
	int r = 5; createTrackbar("r",wname,&r,200);
	int space = 300; createTrackbar("space",wname,&space,2000);
	
	int color = 500; createTrackbar("color",wname,&color,2550);
	int color2 = 500; createTrackbar("color2",wname,&color2,2550);
	int a1 = 100; createTrackbar("a1",wname,&a1,200);
	int a2 = 100; createTrackbar("a2",wname,&a2,200);
	int key = 0;
	Mat show;
	Mat ref;
	ConsoleImage ci;
	{
		float sigma_color = color/10.f;
		float sigma_color2 = color2/10.f;
		float sigma_space = space/10.f;
		int d = 2*r+1;
		dualBilateralFilter(src1, src2, ref, Size(d,d), sigma_color, sigma_color2, sigma_space, FILTER_CIRCLE);
	}
	while(key!='q')
	{
		float sigma_color = color/10.f;
		float sigma_color2 = color2/10.f;
		float sigma_space = space/10.f;
		int d = 2*r+1;

		if(key=='r')
		{
			dualBilateralFilter(src1, src2, ref, Size(d,d), sigma_color, sigma_color2, sigma_space, FILTER_CIRCLE);
		}
		if(sw==0)
		{
			CalcTime t("dual bilateral filter SP");
			dualBilateralFilter(src1, src2, dest, Size(d,d), sigma_color, sigma_color2, sigma_space,FILTER_SEPARABLE);
		}
		else if(sw==1)
		{
			CalcTime t("dual bilateral filter DualKernel");
			separableDualBilateralFilter(src1, src2, dest, Size(d,d), sigma_color, sigma_color2, sigma_space,a1/100.0,a2/100.0, FILTER_CIRCLE);
		}
		else if(sw==2)
		{
			CalcTime t("dual bilateral filter DualKernel");
			separableDualBilateralFilter(src1, src2, dest, Size(d,d), sigma_color, sigma_color2, sigma_space,a1/100.0,a2/100.0, FILTER_SEPARABLE);
		}
		ci("%f dB", PSNR(dest, ref));
		ci.flush();
		alphaBlend(src1, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}
void guiDualBilateralFilterTest(Mat& src1, Mat& src2)
{
	//accDualBilateralFilterTest(src1, src2);
	guiSeparableDualBilateralFilterTest(src1,src2);
	Mat dest,dest2;

	string wname = "bilateral filter";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 0; createTrackbar("switch",wname,&sw, 4);
	int r = 5; createTrackbar("r",wname,&r,200);
	int space = 300; createTrackbar("space",wname,&space,2000);
	
	int color = 500; createTrackbar("color",wname,&color,2550);
	int color2 = 500; createTrackbar("color2",wname,&color2,2550);
	int rate = 100; createTrackbar("rate",wname,&rate,100);
	int key = 0;
	Mat show;
	
	while(key!='q')
	{
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color/10.f;
		float sigma_color2 = color2/10.f;
		float sigma_space = space/10.f;
		int d = 2*r+1;

		
		if(sw==0)
		{
			CalcTime t("birateral filter: opencv");
			dualBilateralFilter(src1, src2, dest, Size(d,d), sigma_color, sigma_color2, sigma_space,FILTER_SLOWEST);
			//bilateralFilter(src, dest, d, sigma_color, sigma_space);
			//recursiveBilateralFilter(src, dest, sigma_color, sigma_space,0);
		}
		else if(sw==1)
		{
			CalcTime t("birateral filter: opencv");
			dualBilateralFilter(src1, src2, dest, Size(d,d), sigma_color, sigma_color2, sigma_space,FILTER_CIRCLE);
			//bilateralFilter(src, dest, d, sigma_color, sigma_space);
			//recursiveBilateralFilter(src, dest, sigma_color, sigma_space,0);
		}
		//else if(sw==1)
		//{
		//	CalcTime t("birateral filter: fastest opencp implimentation");
		//	bilateralFilterSP_test3_8u(src, dest, Size(d,d), sigma_color,sigma_color*rate/100.0, sigma_space,BORDER_REPLICATE);
		//	//bilateralFilterSP2_8u(src, dest, Size(d,d), sigma_color, sigma_space,BORDER_REPLICATE);
		//	//recursiveBilateralFilter(src, dest, sigma_color, sigma_space,1);

		//	//bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_CIRCLE);
		//}
		//else if(sw==2)
		//{
		//	CalcTime t("birateral filter: fastest opencp implimentation with rectangle kernel");
		//	//rbf(src, dest, sigma_color, sigma_space);
		//	bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
		//	//recursiveBilateralFilter(src, dest2, sigma_color, sigma_space,0);
		//	//recursiveBilateralFilter(src, dest, sigma_color, sigma_space,1);
		//	//cout<<norm(dest,dest2)<<endl;
		//	//bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
		//	
		//}
		//else if(sw==3)
		//{
		//	CalcTime t("birateral filter: fastest: sepalable approximation of opencp");
		//	bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_SEPARABLE);

		//}
		//else if(sw==4)
		//{
		//	CalcTime t("birateral filter: slowest: non-parallel and inefficient implimentation");
		//	bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, FILTER_SLOWEST);
		//}

		alphaBlend(src1, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}