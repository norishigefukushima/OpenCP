#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;
/*
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <iostream>
*/
using namespace std;

#ifdef _OPENMP
#include <omp.h>
#endif

void fillOcclusionHV(Mat& src, int invalid = 0)
{
	Mat dst = src.clone();
	fillOcclusion(dst, invalid);
	Mat dst2;
	transpose(src, dst2);
	fillOcclusion(dst2, invalid);
	transpose(dst2, src);
	min(src, dst, src);
}


double getsn(StereoViewSynthesis& svs, Mat& matimL, Mat& matimR, Mat& max_disp_l, Mat& max_disp_r, Mat& ref)
{
	Mat dest, destdisp;
	svs(matimL, matimR, max_disp_l, max_disp_r, dest, destdisp, 0.5, 0, 2);
	return getPSNR(dest, ref);
}
void average(const Mat& src1, const Mat& src2, Mat& dest)
{
	Mat temp;
	add(src1, src2, temp, noArray(), CV_32F);
	temp.convertTo(dest, src1.type(), 0.5);
}

Mat clipmergeHorizon(const Mat& src, int index, int max_index)
{

	uchar* ptr = NULL;
	int num = max_index;

	int vp = 0;
	int vc = 0;
	for (int i = 0; i < num; i++)
	{
		vc = (int)(src.rows*(i + 1) / (double)num);
		if (i == index)
		{
			ptr = (uchar*)src.ptr<uchar>(vp);
			break;
		}
		vp = vc;
	}

	Mat ret(Size(src.cols, vc - vp), src.type(), ptr);
	return ret;
}

class StereoViewSynthesisInvoker : public cv::ParallelLoopBody
{
	vector<Mat> imgL;
	vector<Mat> imgR;
	vector<Mat> disparityL;
	vector<Mat> disparityR;

	Mat* imgDest;
	Mat* dispDest;
	int splitnum;
public:
	~StereoViewSynthesisInvoker()
	{
		//mergeHorizon(destdisparity,*dispDest);
		//mergeHorizon(destim,*imgDest);
	}

	StereoViewSynthesisInvoker(Mat& imL, Mat& imR, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, int splitnum_) :
		splitnum(splitnum_), dispDest(&destdisp), imgDest(&dest)
	{
		splitHorizon(imL, imgL, splitnum);
		splitHorizon(dispL, disparityL, splitnum);

		splitHorizon(imR, imgR, splitnum);
		splitHorizon(dispR, disparityR, splitnum);

		imgDest->create(imL.size(), imL.type());
		dispDest->create(dispL.size(), dispL.type());
	}
	virtual void operator() (const Range& range) const
	{
		StereoViewSynthesis svs(StereoViewSynthesis::PRESET_SLOWEST);
		for (int i = range.start; i != range.end; i++)
		{
			Mat a = clipmergeHorizon(*imgDest, i, splitnum);
			Mat b = clipmergeHorizon(*dispDest, i, splitnum);
			/*
			average(imgL[i], imgR[i], a);
			average(disparityL[i], disparityR[i], b);
			*/
			svs(imgL[i], imgR[i], disparityL[i], disparityR[i], a, b, 0.5, 0, 2);
		}
	}
};

void guiViewSynthesis()
{
	vector<string> sequence(50);
	int s_index = 0;
	sequence[s_index++] = "teddyH";
	sequence[s_index++] = "conesH";
	sequence[s_index++] = "Aloe";
	sequence[s_index++] = "Art";
	sequence[s_index++] = "Baby1";
	sequence[s_index++] = "Baby2";
	sequence[s_index++] = "Baby3";
	sequence[s_index++] = "Books";
	sequence[s_index++] = "Bowling1";
	sequence[s_index++] = "Bowling2";
	sequence[s_index++] = "Cloth1";
	sequence[s_index++] = "Cloth2";
	sequence[s_index++] = "Cloth3";
	sequence[s_index++] = "Cloth4";
	sequence[s_index++] = "Dolls";
	sequence[s_index++] = "Flowerpots";
	sequence[s_index++] = "Lampshade1";
	sequence[s_index++] = "Lampshade2";
	sequence[s_index++] = "Midd1";
	sequence[s_index++] = "Midd2";
	sequence[s_index++] = "Reindeer";
	sequence[s_index++] = "Laundry";
	sequence[s_index++] = "Moebius";
	sequence[s_index++] = "Monopoly";
	sequence[s_index++] = "Plastic";
	sequence[s_index++] = "Rocks1";
	sequence[s_index++] = "Rocks2";
	sequence[s_index++] = "Wood1";
	sequence[s_index++] = "Wood2";
	int s_index_max = s_index;
	s_index = 2;

	Mat matdiL = imread("img/stereo/" + sequence[s_index] + "/disp1.png", 0);
	Mat matdiR = imread("img/stereo/" + sequence[s_index] + "/disp5.png", 0);
	Mat matimL = imread("img/stereo/" + sequence[s_index] + "/view1.png");
	Mat matimR = imread("img/stereo/" + sequence[s_index] + "/view5.png");

	Mat ref = imread("img/stereo/" + sequence[s_index] + "/view3.png");

	fillOcclusion(matdiL);
	fillOcclusion(matdiR);

	Mat rdL;
	Mat rdR;
	Mat FVdest;

	dispRefinement drL;
	dispRefinement drR;

	mattingMethod mmL;
	mattingMethod mmR;

	namedWindow("FV_parameter");
	//int alphaPos = 491;createTrackbar("alphaPos","FV_parameter",&alphaPos,1000);
	int alphaPos = 500; createTrackbar("alphaPos", "FV_parameter", &alphaPos, 1000);
	int dispAmp = 2; createTrackbar("dispAmp", "FV_parameter", &dispAmp, 10);
	createTrackbar("index", "FV_parameter", &s_index, s_index_max - 1);

	namedWindow("Refine_parameter");
	int r_R = 4; createTrackbar("r", "Refine_parameter", &r_R, 20);
	int th_R = 10; createTrackbar("th", "Refine_parameter", &th_R, 30);
	int r_g_R = 3; createTrackbar("r_g", "Refine_parameter", &r_g_R, 20);
	int eps_g_R = 2; createTrackbar("eps_g", "Refine_parameter", &eps_g_R, 2550);
	int iter_g_R = 2; createTrackbar("iter_g", "Refine_parameter", &iter_g_R, 5);
	int iter_ex_R = 3; createTrackbar("iter_ex", "Refine_parameter", &iter_ex_R, 5);
	int iter_R = 3; createTrackbar("iter", "Refine_parameter", &iter_R, 5);
	int th_r_R = 50; createTrackbar("th_r", "Refine_parameter", &th_r_R, 100);
	int r_flip_R = 1; createTrackbar("r_flip", "Refine_parameter", &r_flip_R, 10);
	int th_FB_R = 85; createTrackbar("th_FB", "Refine_parameter", &th_FB_R, 100);

	namedWindow("Matting_parameter");
	int r_M = 3; createTrackbar("r", "Matting_parameter", &r_M, 20);
	int th_M = 10; createTrackbar("th", "Matting_parameter", &th_M, 30);
	int r_g_M = 3; createTrackbar("r_g", "Matting_parameter", &r_g_M, 20);
	int eps_g_M = 2; createTrackbar("eps_g", "Matting_parameter", &eps_g_M, 2550);
	int iter_g_M = 2; createTrackbar("iter_g", "Matting_parameter", &iter_g_M, 10);
	int iter_M = 6; createTrackbar("iter", "Matting_parameter", &iter_M, 10);
	int r_Wgauss_M = 1; createTrackbar("r_Wgauss", "Matting_parameter", &r_Wgauss_M, 10);
	int th_FB_M = 85; createTrackbar("th_FB", "Matting_parameter", &th_FB_M, 100);

	StereoViewSynthesis svsM(StereoViewSynthesis::PRESET_SLOWEST);
	StereoViewSynthesis svs(StereoViewSynthesis::PRESET_SLOWEST);
	StereoViewSynthesis svsL(StereoViewSynthesis::PRESET_SLOWEST);
	StereoViewSynthesis svsR(StereoViewSynthesis::PRESET_SLOWEST);

	svs.postFilterMethod = StereoViewSynthesis::POST_FILL;
	svsL.postFilterMethod = StereoViewSynthesis::POST_FILL;
	svsR.postFilterMethod = StereoViewSynthesis::POST_FILL;
	int key = 0;

	ConsoleImage ci;

	Mat dest, destdisp, max_disp_l, max_disp_r;

	//svs.depthfiltermode = StereoViewSynthesis::DEPTH_FILTER_MEDIAN;

	string wname = "view";
	namedWindow(wname);

	int alpha = 0; createTrackbar("alpha", wname, &alpha, 100);
	createTrackbar("index", wname, &s_index, s_index_max - 1);
	int dilation_rad = 1; createTrackbar("dilation r", wname, &dilation_rad, 10);
	int blur_rad = 1; createTrackbar("blur r", wname, &blur_rad, 10);
	int isOcc = 2; createTrackbar("is Occ", wname, &isOcc, 2);
	int isOccD = 1; createTrackbar("is OccD", wname, &isOccD, 3);
	int zth = 32; createTrackbar("z thresh", wname, &zth, 100);
	int grate = 100; createTrackbar("grate", wname, &grate, 100);
	int ljump = 0; createTrackbar("ljump", wname, &ljump, 1000);
	int ncore = 1; createTrackbar("ncore", wname, &ncore, 32);
	int blend = 0; createTrackbar("blend", wname, &blend, 1);
	int inter = 2; createTrackbar("inter", wname, &inter, 2);

	int color = 300; createTrackbar("color", wname, &color, 2000);
	int space = 300; createTrackbar("space", wname, &space, 2000);
	int iter = 0; createTrackbar("iter", wname, &iter, 30);


	while (key != 'q')
	{
		Mat dest, destdisp, max_disp_l, max_disp_r;
		StereoViewSynthesis svs(StereoViewSynthesis::PRESET_SLOWEST);

		//cout<<"img/stereo/"+sequence[s_index]<<endl;
		Mat matdiL = imread("img/stereo/" + sequence[s_index] + "/disp1.png", 0);
		Mat matdiR = imread("img/stereo/" + sequence[s_index] + "/disp5.png", 0);
		Mat matimL = imread("img/stereo/" + sequence[s_index] + "/view1.png");
		Mat matimR = imread("img/stereo/" + sequence[s_index] + "/view5.png");

		Mat ref = imread("img/stereo/" + sequence[s_index] + "/view3.png");

		fillOcclusion(matdiL);
		if ((matdiL.size().area() - countNonZero(matdiL) != 0))fillOcclusionHV(matdiL);
		fillOcclusion(matdiR);
		if ((matdiL.size().area() - countNonZero(matdiR) != 0))fillOcclusionHV(matdiR);

		Mat temp;
		for (int i = 0; i < iter; i++)
		{

			//weightedModeFilter(matdiR,matimR, matdiR,3,8,space/10.0,color/10.0,2,2);
			//weightedModeFilter(matdiL,matimL, matdiL,3,8,space/10.0,color/10.0,2,2);

			//jointBilateralFilter(matdiR,matimR, temp,Size(7,7),color/10.0,space/10.0);
			//jointNearestFilterBase(temp,matdiR,Size(3,3),matdiR);

			//jointBilateralFilter(matdiL,matimL, temp,Size(7,7),color/10.0,space/10.0);
			//jointNearestFilterBase(temp,matdiL,Size(3,3),matdiL);
		}
		Mat show;
		alphaBlend(matimL, matdiL, alpha / 100.0, show);
		imshow("disp", show);


		svs.blendMethod = blend;
		svs.large_jump = ljump;

		svs.boundaryGaussianRatio = (double)grate / 100.0;
		svs.blend_z_thresh = zth;

		svs.occBlurSize = Size(2 * blur_rad + 1, 2 * blur_rad + 1);
		svs.inpaintMethod = FILL_OCCLUSION_HV;

		if (isOcc == 0) svs.postFilterMethod = StereoViewSynthesis::POST_NONE;
		if (isOcc == 1) svs.postFilterMethod = StereoViewSynthesis::POST_FILL;
		if (isOcc == 2) svs.postFilterMethod = StereoViewSynthesis::POST_GAUSSIAN_FILL;
		if (isOccD == 0)svs.inpaintMethod = FILL_OCCLUSION_LINE;
		if (isOccD == 1)svs.inpaintMethod = FILL_OCCLUSION_HV;
		if (isOccD == 2)svs.inpaintMethod = FILL_OCCLUSION_INPAINT_NS;
		if (isOccD == 3)svs.inpaintMethod = FILL_OCCLUSION_INPAINT_TELEA;
		//
		svs.warpInterpolationMethod = inter;
		{
			Timer t("time", 0, false);
			maxFilter(matdiL, max_disp_l, dilation_rad);
			maxFilter(matdiR, max_disp_r, dilation_rad);
			svs(matimL, matimR, max_disp_l, max_disp_r, dest, destdisp, alphaPos*0.001, 0, dispAmp);

			//StereoViewSynthesisInvoker body(matimL,matimR,max_disp_l,max_disp_r, dest, destdisp,ncore);
			//parallel_for_(Range(0, ncore), body);
			ci("%f ms", t.getTime());
		}

		/*Mat dest16;
		{
			Mat a = Mat::zeros(matimL.size(),CV_16S);
			Mat b = Mat::zeros(matimL.size(),CV_16S);
			DepthMapSubpixelRefinment dsr;


			dsr.naive(matimL,matimR,matdiL,matdiR,2, a,b);
			//Mat a2;
			//a.convertTo(a2,CV_8U,1.0/8);
			//guiAlphaBlend(matdiL,a2);
			CalcTime t("time",0,false);
			maxFilter(a, a,dilation_rad);
			maxFilter(b, b,dilation_rad);

			svs(matimL,matimR,a,b, dest16, destdisp,alphaPos*0.001,0,16);

			//StereoViewSynthesisInvoker body(matimL,matimR,max_disp_l,max_disp_r, dest, destdisp,ncore);
			//parallel_for_(Range(0, ncore), body);
			ci("%f ms",t.getTime());
		}*/

		if (key == 'c')
		{
			Mat gdest;
			GaussianBlur(dest, gdest, Size(5, 5), svs.boundarySigma);
			guiCompareDiff(dest, gdest, ref);
		}
		//svs.check(matimL,matimR,max_disp_l,max_disp_r, dest, destdisp,alphaPos*0.001,0,dispAmp,ref);

		if (key == 'p')
		{
			Stat st;
			for (int i = 0; i < s_index_max; i++)
			{
				Mat matdiL = imread("img/stereo/" + sequence[i] + "/disp1.png", 0);
				Mat matdiR = imread("img/stereo/" + sequence[i] + "/disp5.png", 0);
				Mat matimL = imread("img/stereo/" + sequence[i] + "/view1.png");
				Mat matimR = imread("img/stereo/" + sequence[i] + "/view5.png");
				Mat ref = imread("img/stereo/" + sequence[i] + "/view3.png");
				fillOcclusion(matdiL);
				fillOcclusion(matdiR);
				maxFilter(matdiL, max_disp_l, dilation_rad);
				maxFilter(matdiR, max_disp_r, dilation_rad);
				double sn = getsn(svs, matimL, matimR, max_disp_l, max_disp_r, ref);
				st.push_back(sn);
				//cout<<sequence[i]<<endl;
			}
			cout << "ave" << st.getMean() << endl;
		}



		//	alphaBlend(destdisp,dest,alpha/100.0,dest);
		alphaBlend(ref, dest, alpha / 100.0, dest);

		imshow(wname, dest);
		key = waitKey(1);
		if (key == 'f') alpha = (alpha == 0) ? 100 : 0;
		if (key == 'b') guiAlphaBlend(dest, destdisp);
		if (key == 'a') guiAlphaBlend(dest, ref);

		if (key == 'd') guiAbsDiffCompareGE(dest, ref);
		ci("%f dB", getPSNR(dest, ref));
		//ci("%f dB",YPSNR(dest16,ref));
		//if(key=='k')guiCompareDiff(dest,dest16,ref);

		ci.show();
	}

	key = 0;

	while (key != 'q')
	{
		Mat matdiL = imread("img/stereo/" + sequence[s_index] + "/disp1.png", 0);
		Mat matdiR = imread("img/stereo/" + sequence[s_index] + "/disp5.png", 0);
		Mat matimL = imread("img/stereo/" + sequence[s_index] + "/view1.png");
		Mat matimR = imread("img/stereo/" + sequence[s_index] + "/view5.png");

		Mat ref = imread("img/stereo/" + sequence[s_index] + "/view3.png");

		fillOcclusion(matdiL);
		fillOcclusion(matdiR);

#pragma omp parallel 
		{
#pragma omp sections
			{
#pragma omp section
				{
					drL.r = r_R;
					drL.th = th_R;
					drL.r_g = r_g_R;
					drL.eps_g = eps_g_R;
					drL.iter_g = iter_g_R;
					drL.iter_ex = iter_ex_R;
					drL.iter = iter_R;
					drL.th_r = th_r_R;
					drL.r_flip = r_flip_R;
					drL.th_FB = th_FB_R;
					drL(matdiL, matimL, rdL);
				}
#pragma omp section
				{
					drR.r = r_R;
					drR.th = th_R;
					drR.r_g = r_g_R;
					drR.eps_g = eps_g_R;
					drR.iter_g = iter_g_R;
					drR.iter_ex = iter_ex_R;
					drR.iter = iter_R;
					drR.th_r = th_r_R;
					drR.r_flip = r_flip_R;
					drR.th_FB = th_FB_R;
					drR(matdiR, matimR, rdR);
				}
			}
		}

		vector<Mat> srcMat(30);
		vector<Mat> destMat(30);
		Mat destL, destR;
		Mat aL, aR;

		matimL.copyTo(srcMat[0]);
		matimR.copyTo(srcMat[2]);

		rdL.copyTo(destMat[0]);
		rdR.copyTo(destMat[1]);
		Mat alphaL, alphaR;
		Mat fL, fR;
		Mat bL, bR;
		Mat base;
		Mat basic;
		Mat dilateSynth;

#pragma omp parallel
#pragma omp sections
		{
#pragma omp section
			{
				mmL.r = r_M;
				mmL.th = th_M;
				mmL.r_g = r_g_M;
				mmL.eps_g = eps_g_M * 255;
				mmL.iter_g = iter_g_M;
				mmL.iter = iter_M;
				mmL.r_Wgauss = r_Wgauss_M;
				mmL.th_FB = th_FB_M;
				mmL(matimL, rdL, alphaL, fL, bL);
				cvtColor(alphaL, aL, COLOR_GRAY2BGR);
			}
#pragma omp section
			{
				mmR.r = r_M;
				mmR.th = th_M;
				mmR.r_g = r_g_M;
				mmR.eps_g = eps_g_M * 255;
				mmR.iter_g = iter_g_M;
				mmR.iter = iter_M;
				mmR.r_Wgauss = r_Wgauss_M;
				mmR.th_FB = th_FB_M;
				mmR(matimR, rdR, alphaR, fR, bR);
				cvtColor(alphaR, aR, COLOR_GRAY2BGR);
			}
		}


		Mat max_disp_l, max_disp_r;
		int dilation_rad = 1;


		maxFilter(matdiL, max_disp_l, dilation_rad);
		maxFilter(matdiR, max_disp_r, dilation_rad);
		svsM(matimL, matimR, max_disp_l, max_disp_r, dilateSynth, destMat[9], alphaPos*0.001, 0, dispAmp);
		svsM(matimL, matimR, rdL, rdR, basic, destMat[9], alphaPos*0.001, 0, dispAmp);

		imshow("disp", rdL);
		svs(bL, bR, rdL, rdR, base, destMat[9], alphaPos*0.001, 0, dispAmp);

		//#pragma omp parallel 
		{
#pragma omp sections
			{
#pragma omp section
				{
					Mat wfL;
					Mat waL;
					Mat dL;
					Mat amapL;
					Mat aML;
					maxFilter(rdL, dL, Size(3, 3));
					cvtColor(aL, aL, COLOR_BGR2GRAY);
					maxFilter(aL, aL, Size(1, 1));
					cvtColor(aL, aML, COLOR_GRAY2BGR);
					svsL.viewsynthSingleAlphaMap(fL, dL, wfL, destMat[11], alphaPos*0.001, 0, dispAmp, dL.type());
					svsL.viewsynthSingleAlphaMap(aML, dL, waL, destMat[13], alphaPos*0.001, 0, dispAmp, dL.type());
					cvtColor(waL, amapL, COLOR_BGR2GRAY);
					alphaBlend(wfL, base, amapL, destL);
				}
#pragma omp section
				{
					Mat wfR;
					Mat waR;
					Mat dR;
					Mat amapR;
					Mat aMR;
					cvtColor(aR, aR, COLOR_BGR2GRAY);
					maxFilter(rdR, dR, Size(3, 3));
					maxFilter(aR, aR, Size(1, 1));
					cvtColor(aR, aMR, COLOR_GRAY2BGR);
					svsR.viewsynthSingleAlphaMap(fR, dR, wfR, destMat[15], -alphaPos * 0.001, 0, dispAmp, dR.depth());
					svsR.viewsynthSingleAlphaMap(aMR, dR, waR, destMat[17], -alphaPos * 0.001, 0, dispAmp, dR.depth());
					cvtColor(waR, amapR, COLOR_BGR2GRAY);
					alphaBlend(wfR, base, amapR, destR);
					imshow("fore", wfR);
					imshow("alpha", waR);
					//guiAlphaBlend(waR,wfR);
				}
			}
		}


		if (key == 'b')guiAlphaBlend(base, destL);
		alphaBlend(destL, destR, alphaPos*0.001, FVdest);

		imshow("FVdest", FVdest);
		key = waitKey(1);
		//guiAlphaBlend(FVdest,ref);
		//cout<<YPSNR(FVdest,ref)<<endl;

		int thresh = 10;
		ci("basic %f", getPSNR(basic, ref));
		ci("basic %f", getBadPixel(basic, ref, thresh));
		ci("max.  %f", getPSNR(dilateSynth, ref));
		ci("max.  %f", getBadPixel(dilateSynth, ref, thresh));

		//ci("Base. %f",YPSNR(base,ref));

		ci("Matt. %f", getPSNR(FVdest, ref));
		ci("Matt. %f", getBadPixel(FVdest, ref, thresh));

		if (key == 'o')guiAlphaBlend(FVdest, dilateSynth);
		ci.show();
	}

	//imwrite("FVdest.png",FVdest);
}