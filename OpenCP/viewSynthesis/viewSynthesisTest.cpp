#include "../opencp.hpp"
#include <fstream>
using namespace std;

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

void fillOcclusionHV(Mat& src, int invalid=0)
{
	Mat dst = src.clone();
	fillOcclusion(dst,invalid);
	Mat dst2;
	transpose(src,dst2);
	fillOcclusion(dst2,invalid);
	transpose(dst2,src);
	min(src,dst,src);
}


double getsn(StereoViewSynthesis& svs, Mat& matimL, Mat& matimR, Mat& max_disp_l, Mat& max_disp_r, Mat& ref)
{
	Mat dest,destdisp;
	svs(matimL,matimR,max_disp_l,max_disp_r, dest, destdisp,0.5,0, 2);
	return YPSNR(dest,ref);
}

void guiViewSynthesis()
{
	vector<string> sequence(50);
	int s_index=0;
	sequence[s_index++]="teddyH";
	sequence[s_index++]="conesH";
	sequence[s_index++]="Aloe";
	sequence[s_index++]="Art";
	sequence[s_index++]="Baby1";
	sequence[s_index++]="Baby2";
	sequence[s_index++]="Baby3";
	sequence[s_index++]="Books";
	sequence[s_index++]="Bowling1";
	sequence[s_index++]="Bowling2";
	sequence[s_index++]="Cloth1";
	sequence[s_index++]="Cloth2";
	sequence[s_index++]="Cloth3";
	sequence[s_index++]="Cloth4";
	sequence[s_index++]="Dolls";
	sequence[s_index++]="Flowerpots";
	sequence[s_index++]="Lampshade1";
	sequence[s_index++]="Lampshade2";
	sequence[s_index++]="Midd1";
	sequence[s_index++]="Midd2";
	sequence[s_index++]="Reindeer";
	sequence[s_index++]="Laundry";
	sequence[s_index++]="Moebius";
	sequence[s_index++]="Monopoly";
	sequence[s_index++]="Plastic";
	sequence[s_index++]="Rocks1";
	sequence[s_index++]="Rocks2";
	sequence[s_index++]="Wood1";
	sequence[s_index++]="Wood2";
	//sequence[s_index++]="Venus";
	int s_index_max=s_index;
	s_index=2;

	Mat matdiL = imread("img/stereo/"+sequence[s_index]+"/disp1.png",0);
	Mat matdiR = imread("img/stereo/"+sequence[s_index]+"/disp5.png",0);
	Mat matimL= imread("img/stereo/"+sequence[s_index]+"/view1.png");
	Mat matimR= imread("img/stereo/"+sequence[s_index]+"/view5.png");

	Mat ref= imread("img/stereo/"+sequence[s_index]+"/view3.png");

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
	int alphaPos = 500;createTrackbar("alphaPos","FV_parameter",&alphaPos,1000);
	int dispAmp = 2;cvCreateTrackbar("dispAmp","FV_parameter",&dispAmp,10);

	namedWindow("Refine_parameter");
	int r_R = 4;createTrackbar("r","Refine_parameter",&r_R,20);
	int th_R = 10;createTrackbar("th","Refine_parameter",&th_R,30);
	int r_g_R = 3;createTrackbar("r_g","Refine_parameter",&r_g_R,20);
	int eps_g_R = 2;createTrackbar("eps_g","Refine_parameter",&eps_g_R,255);
	int iter_g_R = 2;createTrackbar("iter_g","Refine_parameter",&iter_g_R,5);
	int iter_ex_R = 3;createTrackbar("iter_ex","Refine_parameter",&iter_ex_R,5);
	int iter_R = 3;createTrackbar("iter","Refine_parameter",&iter_R,5);
	int th_r_R = 50;createTrackbar("th_r","Refine_parameter",&th_r_R,100);
	int r_flip_R = 1;createTrackbar("r_flip","Refine_parameter",&r_flip_R,10);
	int th_FB_R = 85;createTrackbar("th_FB","Refine_parameter",&th_FB_R,100);

	namedWindow("Matting_parameter");
	int r_M = 3;createTrackbar("r","Matting_parameter",&r_M,20);
	int th_M = 10;createTrackbar("th","Matting_parameter",&th_M,30);
	int r_g_M = 3;createTrackbar("r_g","Matting_parameter",&r_g_M,20);
	int eps_g_M = 2;createTrackbar("eps_g","Matting_parameter",&eps_g_M,255);
	int iter_g_M = 2;createTrackbar("iter_g","Matting_parameter",&iter_g_M,10);
	int iter_M = 6;createTrackbar("iter","Matting_parameter",&iter_M,10);
	int r_Wgauss_M = 1;createTrackbar("r_Wgauss","Matting_parameter",&r_Wgauss_M,10);
	int th_FB_M = 85;createTrackbar("th_FB","Matting_parameter",&th_FB_M,100);

	StereoViewSynthesis svs(StereoViewSynthesis::PRESET_SLOWEST);
	StereoViewSynthesis svsL;
	StereoViewSynthesis svsR;

	int key = 0;

	Mat dest,destdisp, max_disp_l, max_disp_r;

	//svs.depthfiltermode = StereoViewSynthesis::DEPTH_FILTER_MEDIAN;

	string wname = "view";
	namedWindow(wname);

	int alpha = 0; createTrackbar("alpha", wname, &alpha,100);
	createTrackbar("index", wname, &s_index,s_index_max-1);
	int dilation_rad = 1; createTrackbar("dilation r", wname, &dilation_rad,10);
	int blur_rad = 1; createTrackbar("blur r", wname, &blur_rad,10);
	int isOcc = 2; createTrackbar("is Occ", wname, &isOcc,2);
	int isOccD = 1; createTrackbar("is OccD", wname, &isOccD,1);
	int zth = 0; createTrackbar("z thresh", wname, &zth,100);
	int grate = 100; createTrackbar("grate", wname, &grate,100);
	
	ConsoleImage ci;
	while(key!='q')
	{
		Mat dest,destdisp, max_disp_l, max_disp_r;
		StereoViewSynthesis svs(StereoViewSynthesis::PRESET_SLOWEST);
		//cout<<"img/stereo/"+sequence[s_index]<<endl;
		Mat matdiL = imread("img/stereo/"+sequence[s_index]+"/disp1.png",0);
		Mat matdiR = imread("img/stereo/"+sequence[s_index]+"/disp5.png",0);
		Mat matimL= imread("img/stereo/"+sequence[s_index]+"/view1.png");
		Mat matimR= imread("img/stereo/"+sequence[s_index]+"/view5.png");

		Mat ref= imread("img/stereo/"+sequence[s_index]+"/view3.png");

		fillOcclusion(matdiL);
		fillOcclusion(matdiR);


		svs.boundaryGaussianRatio = (double)grate/100.0;
		svs.blend_z_thresh=zth;

		svs.occBlurSize = Size(2*blur_rad+1,2*blur_rad+1);
		svs.inpaintMethod = StereoViewSynthesis::FILL_OCCLUSION_HV;
		
		if(isOcc==0) svs.postFilterMethod=StereoViewSynthesis::POST_NONE;
		if(isOcc==1) svs.postFilterMethod=StereoViewSynthesis::POST_FILL;
		if(isOcc==2) svs.postFilterMethod=StereoViewSynthesis::POST_GAUSSIAN_FILL;
		if(isOccD==0)svs.inpaintMethod=StereoViewSynthesis::FILL_OCCLUSION_LINE;
		//
		svs.warpInterpolationMethod = INTER_CUBIC;
		{
			CalcTime t("time",0,false);
			maxFilter(matdiL, max_disp_l,dilation_rad);
			maxFilter(matdiR, max_disp_r,dilation_rad);
			svs(matimL,matimR,max_disp_l,max_disp_r, dest, destdisp,alphaPos*0.001,0,dispAmp);
			ci("%f ms",t.getTime());
		}

		if(key=='c')
		{
			Mat gdest;
			GaussianBlur(dest,gdest,Size(5,5),svs.boundarySigma);
			guiCompareDiff(dest,gdest,ref);
		}
		//svs.check(matimL,matimR,max_disp_l,max_disp_r, dest, destdisp,alphaPos*0.001,0,dispAmp,ref);

		if(key == 'p')
		{
			Stat st;
			for(int i=0;i<s_index_max;i++)
			{
				Mat matdiL = imread("img/stereo/"+sequence[i]+"/disp1.png",0);
				Mat matdiR = imread("img/stereo/"+sequence[i]+"/disp5.png",0);
				Mat matimL= imread("img/stereo/"+sequence[i]+"/view1.png");
				Mat matimR= imread("img/stereo/"+sequence[i]+"/view5.png");
				Mat ref= imread("img/stereo/"+sequence[i]+"/view3.png");
				fillOcclusion(matdiL);
				fillOcclusion(matdiR);
				maxFilter(matdiL, max_disp_l,dilation_rad);
				maxFilter(matdiR, max_disp_r,dilation_rad);
				double sn = 		getsn(svs, matimL, matimR, max_disp_l, max_disp_r, ref);
				st.push_back(sn);
				//cout<<sequence[i]<<endl;
			}
			cout<<"ave"<<st.getMean()<<endl;


		}

		alphaBlend(destdisp,dest,alpha/100.0,dest);
		imshow(wname, dest);
		key = waitKey(1);
		if(key=='f') alpha = (alpha == 0) ? 100 : 0;
		if(key=='b') guiAlphaBlend(dest,destdisp);
		if(key=='a') guiAlphaBlend(dest,ref);
		
		if(key=='d') guiAbsDiffCompareGE(dest,ref);
		ci("%f dB",YPSNR(dest,ref));
		ci.flush();
	}

	
	while(key!='q')
	{
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
					drL(matdiL,matimL,rdL);
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
					drR(matdiR,matimR,rdR);
				}
			}
		}
		
		vector<Mat> srcMat(30);
		vector<Mat> destMat(30);
		Mat destL,destR;
		Mat aL,aR;

		matimL.copyTo(srcMat[0]);
		matimR.copyTo(srcMat[2]);

		rdL.copyTo(destMat[0]);
		rdR.copyTo(destMat[1]);
		Mat alphaL,alphaR;
		Mat fL,fR;
		Mat bL,bR;
		Mat base;

#pragma omp parallel
#pragma omp sections
		{
#pragma omp section
			{
				mmL.r = r_M;
				mmL.th = th_M;
				mmL.r_g = r_g_M;
				mmL.eps_g = eps_g_M;
				mmL.iter_g = iter_g_M;
				mmL.iter = iter_M;
				mmL.r_Wgauss = r_Wgauss_M;
				mmL.th_FB = th_FB_M;
				mmL(matimL,rdL,alphaL,fL,bL);
				cvtColor(alphaL,aL,CV_GRAY2BGR);
			}
#pragma omp section
			{
				mmR.r = r_M;
				mmR.th = th_M;
				mmR.r_g = r_g_M;
				mmR.eps_g = eps_g_M;
				mmR.iter_g = iter_g_M;
				mmR.iter = iter_M;
				mmR.r_Wgauss = r_Wgauss_M;
				mmR.th_FB = th_FB_M;
				mmR(matimR,rdR,alphaR,fR,bR);
				cvtColor(alphaR,aR,CV_GRAY2BGR);
			}
		}

		svs(bL,bR,rdL,rdR,base,destMat[9],alphaPos*0.001,0,dispAmp);

#pragma omp parallel 
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
					maxFilter(rdL,dL,Size(3,3));
					cvtColor(aL,aL,CV_BGR2GRAY);
					maxFilter(aL,aL,Size(1,1));
					cvtColor(aL,aML,CV_GRAY2BGR);
					svsL.viewsynthSingleAlphaMap(fL,dL,wfL,destMat[11],alphaPos*0.001,0,dispAmp,dL.type());
					svsL.viewsynthSingleAlphaMap(aML,dL,waL,destMat[13],alphaPos*0.001,0,dispAmp,dL.type());
					cvtColor(waL,amapL,CV_BGR2GRAY);
					alphaBlend(wfL,base,amapL,destL);

				}
#pragma omp section
				{
					Mat wfR;
					Mat waR;
					Mat dR;
					Mat amapR;
					Mat aMR;
					cvtColor(aR,aR,CV_BGR2GRAY);
					maxFilter(rdR,dR,Size(3,3));
					maxFilter(aR,aR,Size(1,1));
					cvtColor(aR,aMR,CV_GRAY2BGR);
					svsR.viewsynthSingleAlphaMap(fR,dR,wfR,destMat[15],-alphaPos*0.001,0,dispAmp,dR.depth());
					svsR.viewsynthSingleAlphaMap(aMR,dR,waR,destMat[17],-alphaPos*0.001,0,dispAmp,dR.depth());
					cvtColor(waR,amapR,CV_BGR2GRAY);
					alphaBlend(wfR,base,amapR,destR);
				}
			}
		}
		
		alphaBlend(destL,destR,alphaPos*0.001,FVdest);


		imshow("FVdest",FVdest);
		waitKey(1);
		guiAlphaBlend(FVdest,ref);


		cout<<YPSNR(FVdest,ref)<<endl;
		cout<<PSNR(FVdest,ref)<<endl;
		
	}

	//imwrite("FVdest.png",FVdest);
}