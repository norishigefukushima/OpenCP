#include "StereoEx.hpp"
#include "viewsynthesis.hpp"
#include "depthfilter.hpp"
#include "binalyWeightedRangeFilter.hpp"
#include "minmaxfilter.hpp"
#include "noise.hpp"
#include "stat.hpp"
#include "timer.hpp"
#include "alphaBlend.hpp"
#include "inlineSIMDfunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	//////////////////////////////////////////////////////////////////////////////////////////
	//StereoBMEx::
	StereoBMEx::StereoBMEx(int preset, const int ndisparities, const int disparity_min, const int SADWindowSize_)
	{
		prefilterAlpha = 1.0;
		SADWindowSize = SADWindowSize_;
		numberOfDisparities = ndisparities;

		bm = StereoBM::create(ndisparities, SADWindowSize);
		bm->setMinDisparity(disparity_min);

		preFilterType = bm->getPreFilterType();
		preFilterSize = bm->getPreFilterSize();
		preFilterCap = bm->getPreFilterCap();
		SADWindowSize = bm->getBlockSize();
		minDisparity = bm->getMinDisparity();
		numberOfDisparities = bm->getNumDisparities();
		textureThreshold = bm->getTextureThreshold();
		uniquenessRatio = bm->getUniquenessRatio();
		speckleWindowSize = bm->getSpeckleWindowSize();
		speckleRange = bm->getSpeckleRange();
		//trySmallerWindows = bm->getSmallerBlockSize();
		disp12MaxDiff = bm->getDisp12MaxDiff();

		medianKernel = 0;
		isOcclusion = 0;
	}
	void StereoBMEx::showPostFilter()
	{
		std::cout << "textureThreshold: " << textureThreshold << "uniquenessRatio: " << uniquenessRatio << "disp12MaxDiff:" << disp12MaxDiff << std::endl;
		std::cout << "Speckle Window: " << speckleWindowSize << "Speckle Range: " << speckleRange << std::endl;
	}
	void StereoBMEx::showPreFilter()
	{
		std::cout << "Type: " << preFilterType << " Size: " << preFilterSize << " Cap: " << preFilterCap << std::endl;
	}
	void StereoBMEx::setPreFilter(int preFilterType_, int preFilterSize_, int preFilterCap_)
	{
		preFilterType = std::max(std::min(preFilterType_, (int)StereoBM::PREFILTER_XSOBEL), (int)StereoBM::PREFILTER_NORMALIZED_RESPONSE);
		preFilterSize = max(min(preFilterSize_, 21), 5);
		preFilterCap = std::min(std::max(preFilterCap_, 1), 63);
	}
	void StereoBMEx::parameterUpdate()
	{
		bm->setPreFilterCap(preFilterCap);
		bm->setPreFilterSize(preFilterSize);
		
		bm->setPreFilterType(preFilterType);//PREFILTER_NORMALIZED_RESPONSE=0 or PREFILTER_XSOBEL=1
		bm->setBlockSize(max(SADWindowSize, 5));
		bm->setMinDisparity(minDisparity);
		bm->setNumDisparities(get_simd_ceil(numberOfDisparities, 16));
		bm->setTextureThreshold(textureThreshold);
		bm->setUniquenessRatio(uniquenessRatio);
		bm->setSpeckleWindowSize(speckleWindowSize);
		bm->setSpeckleRange(speckleRange);
		//bm->setSmallerBlockSize(trySmallerWindows);
		bm->setDisp12MaxDiff(disp12MaxDiff);
	}

	void StereoBMEx::prefilter(Mat& sl, Mat& sr)
	{
		if (prefilterAlpha < 1.0)
		{
			Mat sobel;
			Mat temp1;
			cv::Sobel(sl, sobel, CV_16S, 1, 0, 1, (1.0 - prefilterAlpha)*127.0 / 256.0);
			//cv::Sobel(sl,sobel,CV_16S,1,0,3);
			sl.convertTo(temp1, CV_16S);
			cv::add(temp1, sobel, sl, Mat(), CV_8U);

			cv::Sobel(sr, sobel, CV_16S, 1, 0, 1, (1.0 - prefilterAlpha)*127.0 / 256.0);
			sr.convertTo(temp1, CV_16S);
			cv::add(temp1, sobel, sr, Mat(), CV_8U);
		}
		imshow("R", sr);
		imshow("L", sl);
	}

	void StereoBMEx::operator()(Mat& leftim, Mat& rightim, Mat& dispL, Mat& dispR, int bd)
	{
		parameterUpdate();
		int bs = SADWindowSize >> 1;
		Mat sl, sr;
		Mat slb, srb;
		Mat disp2;
		if (leftim.channels() == 3)
			cvtColor(leftim, sl, COLOR_BGR2GRAY);
		else
			sl = leftim;
		if (rightim.channels() == 3)
			cvtColor(rightim, sr, COLOR_BGR2GRAY);
		else
			sr = rightim;

		//prefilter(sl,sr);
		cv::copyMakeBorder(sl, slb, bs, bs, bd, bd, cv::BORDER_REPLICATE);
		cv::copyMakeBorder(sr, srb, bs, bs, bd, bd, cv::BORDER_REPLICATE);

		bm->compute(slb, srb, disp2);
		Mat(disp2(cv::Rect(bd, bs, leftim.cols, leftim.rows))).copyTo(dispL);
		cv::threshold(dispL, dispL, (minDisparity << 4) - 1, 0, cv::THRESH_TOZERO);

		cv::flip(slb, slb, -1);
		cv::flip(srb, srb, -1);
		disp2.setTo(0);
		bm->compute(srb, slb, disp2);
		Mat(disp2(cv::Rect(bd, bs, leftim.cols, leftim.rows))).copyTo(dispR);
		cv::flip(dispR, dispR, -1);
		cv::threshold(dispR, dispR, (minDisparity << 4) - 1, 0, cv::THRESH_TOZERO);

		LRCheckDisparity(dispL, dispR, 0, lr_thresh, 0, 16);


		if (isOcclusion)
		{
			fillOcclusion(dispL);
			Mat a = dispL.t();
			fillOcclusion(a);
			Mat(a.t()).copyTo(dispL);

			fillOcclusion(dispR);
			a = dispR.t();
			fillOcclusion(a);
			Mat(a.t()).copyTo(dispR);
		}

		if (medianKernel > 1)
		{
			medianBlur(dispL, dispL, 2 * (medianKernel >> 1) + 1);
			medianBlur(dispR, dispR, 2 * (medianKernel >> 1) + 1);
		}
		/*if(isStreakingRemove)
		{
		removeStreakingNoise(dispL,dispL);
		removeStreakingNoise(dispR,dispR);
		}*/
	}

	void StereoBMEx::operator()(Mat& leftim, Mat& rightim, Mat& disp, int bd)
	{
		int bs = SADWindowSize >> 1;
		Mat sl, sr;
		Mat slb, srb;
		Mat disp2;

		if (leftim.channels() == 3)
			cvtColor(leftim, sl, COLOR_BGR2GRAY);
		else
			sl = leftim;

		if (rightim.channels() == 3)
			cvtColor(rightim, sr, COLOR_BGR2GRAY);
		else
			sr = rightim;

		//prefilter(sl,sr);

		cv::copyMakeBorder(sl, slb, bs, bs, bd, bd, cv::BORDER_REPLICATE);
		cv::copyMakeBorder(sr, srb, bs, bs, bd, bd, cv::BORDER_REPLICATE);

		parameterUpdate();

		bm->compute(slb, srb, disp2);
		Mat(disp2(cv::Rect(bd, bs, leftim.cols, leftim.rows))).copyTo(disp);
		cv::threshold(disp, disp, (minDisparity << 4) - 1, 0, cv::THRESH_TOZERO);

		if (isOcclusion == 1)
		{
			fillOcclusion(disp);

			Mat a = disp.t();
			fillOcclusion(a);
			Mat(a.t()).copyTo(disp);
		}
		if (isOcclusion == 2)
		{
			Mat disp2 = disp.clone();
			fillOcclusion(disp);

			Mat a = disp2.t();
			fillOcclusion(a);
			Mat(a.t()).copyTo(disp2);
			min(disp, disp2, disp);
		}
		if (medianKernel > 1)
		{
			medianBlur(disp, disp, 2 * (medianKernel >> 1) + 1);
		}
	}

	void StereoBMEx::check(Mat& leftim, Mat& rightim, Mat& disp)
	{
		StereoEval eval;
		check(leftim, rightim, disp, eval);
	}

	void StereoBMEx::check(Mat& leftim, Mat& rightim, Mat& disp, StereoEval& eval)
	{
		string wname = "StereoBMEx";
		namedWindow(wname);

		int preFilterType = StereoBM::PREFILTER_NORMALIZED_RESPONSE; // =CV_STEREO_BM_NORMALIZED_RESPONSE now

		int preA = 100;
		createTrackbar("pre A", wname, &preA, 100);

		preFilterType = 1;
		createTrackbar("pre type", wname, &preFilterType, 1);
		int preFilterSize = 4; // averaging window size: ~5x5..21x21
		createTrackbar("pre size", wname, &preFilterSize, 10);
		int preFilterCap = 80; // the output of pre-filtering is clipped by [-preFilterCap,preFilterCap]
		createTrackbar("pre cap", wname, &preFilterCap, 255);

		// correspondence using Sum of Absolute Difference (SAD)
		int SADWindowSize = 11; // ~5x5..21x21
		createTrackbar("window", wname, &SADWindowSize, 30);
		// post-filtering
		int textureThreshold = 10;  // the disparity is only computed for pixels
		// with textured enough neighborhood
		createTrackbar("texture thresh", wname, &textureThreshold, 10000);
		int uniquenessRatio = 10;   // accept the computed disparity d* only if
		// SAD(d) >= SAD(d*)*(1 + uniquenessRatio/100.)
		// for any d != d*+/-1 within the search range.
		createTrackbar("post uniq", wname, &uniquenessRatio, 100);
		int speckleWindowSize = 0; // disparity variation window
		createTrackbar("speckle window", wname, &speckleWindowSize, 100);
		int speckleRange = 100; // acceptable range of variation in window
		createTrackbar("speckle range", wname, &speckleRange, 100);

		createTrackbar("mindisp", wname, &this->minDisparity, leftim.cols - 1);
		createTrackbar("numdisp", wname, &this->numberOfDisparities, leftim.cols - 1);
		
		int trySmallerWindows; // if 1, the results may be more accurate,
		// at the expense of slower processing 
		createTrackbar("small window", wname, &trySmallerWindows, 1);
		int disp12MaxDiff = 0; createTrackbar("disp12", wname, &disp12MaxDiff, 100);

		int bd = this->minDisparity;
		createTrackbar("border", wname, &bd, 300);
		createTrackbar("occlusion", wname, &isOcclusion, 2);
		createTrackbar("post med", wname, &medianKernel, 15);

		int alpha = 100; createTrackbar("alpha", wname, &alpha, 100);
		int isColor = 0; createTrackbar("color", wname, &isColor, 2);
		int noise = 0; createTrackbar("noise", wname, &noise, 1000);

		Mat dshow;
		Mat show;
		int key = 0;
		while (key != 'q')
		{
			this->prefilterAlpha = preA / 100.0;
			this->setPreFilter(preFilterType, 2 * preFilterSize + 1, preFilterCap);
			this->SADWindowSize = 2 * SADWindowSize + 1;
			this->uniquenessRatio = uniquenessRatio;
			this->textureThreshold = textureThreshold;
			this->speckleWindowSize = speckleWindowSize;
			this->speckleRange = speckleRange;
			this->disp12MaxDiff = disp12MaxDiff;
			this->trySmallerWindows = trySmallerWindows;

			//Mat lim, rim;
			//addNoise(leftim,lim,noise/10.0);
			//cout<<calcPSNR(leftim,lim)<<endl;;
			//addNoise(rightim,rim,noise/10.0);

			{
				cp::Timer t("BM");
				operator()(leftim, rightim, disp, bd);
				//operator()(lim,rim,disp,bd);
			}

			/*if(eval.isInit)
			{
			Mat dd;
			disp.convertTo(dd,CV_8U,eval.amp/16.0);
			eval(dd);

			imshow("eval",eval.all_th);
			}*/

			cvtDisparityColor(disp, dshow, minDisparity, numberOfDisparities, isColor);

			alphaBlend(leftim, dshow, 1.0 - (alpha / 100.0), show);
			imshow(wname, show);
			key = waitKey(1);
		}
		destroyWindow(wname);
	}


	StereoSGBMEx::StereoSGBMEx(int minDisparity_, int numDisparities_, Size SADWindowSize_,
		int P1_, int P2_, int disp12MaxDiff_,
		int preFilterCap_, int uniquenessRatio_,
		int speckleWindowSize_, int speckleRange_,
		bool fullDP_)
	{
		minDisparity = minDisparity_;
		numberOfDisparities = numDisparities_;
		SADWindowSize = SADWindowSize_;
		P1 = P1_;
		P2 = P2_;
		disp12MaxDiff = disp12MaxDiff_;
		preFilterCap = preFilterCap_;
		uniquenessRatio = uniquenessRatio_;
		speckleWindowSize = speckleWindowSize_;
		speckleRange = speckleRange_;
		fullDP = fullDP_;

		sgbm.minDisparity = minDisparity_;
		sgbm.numberOfDisparities = numDisparities_;
		sgbm.SADWindowSize = SADWindowSize_;
		sgbm.P1 = P1_;
		sgbm.P2 = P2_;
		sgbm.disp12MaxDiff = disp12MaxDiff_;
		sgbm.preFilterCap = preFilterCap_;
		sgbm.uniquenessRatio = uniquenessRatio_;
		sgbm.speckleWindowSize = speckleWindowSize_;
		sgbm.speckleRange = speckleRange_;
		sgbm.fullDP = fullDP_;

		costAlpha = 1.0;
		ad_max = 31;
		subpixel_r = 4;
		subpixel_th = 32;
		cross_check_threshold = 0;
		isStreakingRemove = 0;
	}
	void StereoSGBMEx::operator()(Mat& leftim, Mat& rightim, Mat& dispL, Mat& dispR, int bd, int lr_thresh)
	{
		const int bordertype = cv::BORDER_REPLICATE;
		int bs = SADWindowSize.width >> 1;
		Mat slb, srb;
		Mat disp2;
		/*Mat temp = Mat::zeros(leftim.size(),CV_8U);
		prefilterXSobel(leftim,temp,31);
		cv::copyMakeBorder(temp,slb,bs,bs,bd,bd,bordertype);
		prefilterXSobel(rightim,temp,31);
		cv::copyMakeBorder(temp,srb,bs,bs,bd,bd,bordertype);*/
		cv::copyMakeBorder(leftim, slb, bs, bs, bd, bd, bordertype);
		cv::copyMakeBorder(rightim, srb, bs, bs, bd, bd, bordertype);

		sgbm.minDisparity = minDisparity;
		sgbm.numberOfDisparities = numberOfDisparities;
		sgbm.SADWindowSize = SADWindowSize;
		sgbm.P1 = P1;
		sgbm.P2 = P2;
		sgbm.disp12MaxDiff = disp12MaxDiff;
		sgbm.preFilterCap = preFilterCap;
		sgbm.uniquenessRatio = uniquenessRatio;
		sgbm.speckleWindowSize = speckleWindowSize;
		sgbm.speckleRange = speckleRange;
		sgbm.fullDP = fullDP;

		sgbm.ad_max = ad_max;
		sgbm.costAlpha = costAlpha;
		sgbm.subpixel_r = subpixel_r;
		sgbm.subpixel_th = subpixel_th;

		sgbm(slb, srb, disp2);
		Mat(disp2(cv::Rect(bd, bs, leftim.cols, leftim.rows))).copyTo(dispL);

		cv::threshold(dispL, dispL, (minDisparity << 4) - 1, 0, cv::THRESH_TOZERO);


		cv::flip(slb, slb, -1);
		cv::flip(srb, srb, -1);
		disp2.setTo(0);
		sgbm(srb, slb, disp2);
		Mat(disp2(cv::Rect(bd, bs, leftim.cols, leftim.rows))).copyTo(dispR);
		cv::flip(dispR, dispR, -1);
		cv::threshold(dispR, dispR, (minDisparity << 4) - 1, 0, cv::THRESH_TOZERO);

		//Mat ll,rr;dispL.convertTo(ll,CV_8U,0.25);dispR.convertTo(rr,CV_8U,0.25);guiAlphaBlend(ll,rr);
		//fillOcclusion(dispL);
		//fillOcclusion(dispR);

		LRCheckDisparity(dispL, dispR, 0, lr_thresh, 0, 16);

		if (isOcclusion)
		{
			fillOcclusion(dispL);
			Mat a = dispL.t();
			fillOcclusion(a);
			Mat(a.t()).copyTo(dispL);

			fillOcclusion(dispR);
			a = dispR.t();
			fillOcclusion(a);
			Mat(a.t()).copyTo(dispR);
		}
		if (isStreakingRemove)
		{
			removeStreakingNoise(dispL, dispL, 16);
			removeStreakingNoise(dispR, dispR, 16);
		}

	}
	void StereoSGBMEx::operator()(Mat& leftim, Mat& rightim, Mat& disp, int bd)
	{
		int bs = SADWindowSize.width >> 1;
		Mat slb, srb;
		Mat disp2;

		/*Mat temp = Mat::zeros(leftim.size(),CV_8U);
		prefilterXSobel(leftim,temp,31);
		cv::copyMakeBorder(temp,slb,bs,bs,bd,bd,BORDER_REPLICATE);
		prefilterXSobel(rightim,temp,31);
		cv::copyMakeBorder(temp,srb,bs,bs,bd,bd,BORDER_REPLICATE);*/

		/*Mat temp = Mat::zeros(leftim.size(),CV_8U);
		guidedFilter(leftim,temp,4,0.2);
		addWeighted(leftim,1.0,temp,-1.0,127,temp);
		cv::copyMakeBorder(temp,slb,bs,bs,bd,bd,BORDER_REPLICATE);
		guidedFilter(rightim,temp,4,0.2);
		addWeighted(rightim,1.0,temp,-1.0,127,temp);
		cv::copyMakeBorder(temp,srb,bs,bs,bd,bd,BORDER_REPLICATE);*/

		cv::copyMakeBorder(leftim, slb, bs, bs, bd, bd, cv::BORDER_REPLICATE);
		cv::copyMakeBorder(rightim, srb, bs, bs, bd, bd, cv::BORDER_REPLICATE);

		sgbm.minDisparity = minDisparity;
		sgbm.numberOfDisparities = numberOfDisparities;
		sgbm.SADWindowSize = SADWindowSize;
		sgbm.P1 = P1;
		sgbm.P2 = P2;
		sgbm.disp12MaxDiff = disp12MaxDiff;
		sgbm.preFilterCap = preFilterCap;
		sgbm.uniquenessRatio = uniquenessRatio;
		sgbm.speckleWindowSize = speckleWindowSize;
		sgbm.speckleRange = speckleRange;
		sgbm.fullDP = fullDP;

		sgbm.ad_max = ad_max;
		sgbm.costAlpha = costAlpha;

		//sgbm(slb,srb,disp2);
		sgbm(slb, srb, disp2);
		Mat(disp2(cv::Rect(bd, bs, leftim.cols, leftim.rows))).copyTo(disp);
		cv::threshold(disp, disp, (minDisparity << 4) - 1, 0, cv::THRESH_TOZERO);

		if (medianKernel > 1)
		{
			//cout<<"here"<<endl;
			//medianBlur(disp,disp,2*(medianKernel>>1)+1);
		}
		if (subpixel_r != 0)
			binalyWeightedRangeFilter(disp, disp, Size(subpixel_r * 2 + 1, subpixel_r * 2 + 1), (float)subpixel_th);
		if (isOcclusion)
		{
			fillOcclusion(disp);
			Mat a = disp.t();
			fillOcclusion(a);
			Mat(a.t()).copyTo(disp);
		}
		if (isStreakingRemove)
		{
			removeStreakingNoise(disp, disp, 16);
		}
	}

	void StereoSGBMEx::test(Mat& leftim, Mat& rightim, Mat& disp, int bd, Point& pt, Mat& gt)
	{
		int bs = SADWindowSize.width >> 1;
		Mat slb, srb;
		Mat disp2;

		/*Mat temp = Mat::zeros(leftim.size(),CV_8U);
		prefilterXSobel(leftim,temp,31);
		cv::copyMakeBorder(temp,slb,bs,bs,bd,bd,BORDER_REPLICATE);
		prefilterXSobel(rightim,temp,31);
		cv::copyMakeBorder(temp,srb,bs,bs,bd,bd,BORDER_REPLICATE);*/

		/*Mat temp = Mat::zeros(leftim.size(),CV_8U);
		guidedFilter(leftim,temp,4,0.2);
		addWeighted(leftim,1.0,temp,-1.0,127,temp);
		cv::copyMakeBorder(temp,slb,bs,bs,bd,bd,BORDER_REPLICATE);
		guidedFilter(rightim,temp,4,0.2);
		addWeighted(rightim,1.0,temp,-1.0,127,temp);
		cv::copyMakeBorder(temp,srb,bs,bs,bd,bd,BORDER_REPLICATE);*/

		cv::copyMakeBorder(leftim, slb, bs, bs, bd, bd, cv::BORDER_REPLICATE);
		cv::copyMakeBorder(rightim, srb, bs, bs, bd, bd, cv::BORDER_REPLICATE);

		sgbm.minDisparity = minDisparity;
		sgbm.numberOfDisparities = numberOfDisparities;
		sgbm.SADWindowSize = SADWindowSize;
		sgbm.P1 = P1;
		sgbm.P2 = P2;
		sgbm.disp12MaxDiff = disp12MaxDiff;
		sgbm.preFilterCap = preFilterCap;
		sgbm.uniquenessRatio = uniquenessRatio;
		sgbm.speckleWindowSize = speckleWindowSize;
		sgbm.speckleRange = speckleRange;
		sgbm.fullDP = fullDP;

		sgbm.ad_max = ad_max;
		sgbm.costAlpha = costAlpha;

		//sgbm(slb,srb,disp2);
		Point pt2 = Point(pt.x + bd, pt.y + bs);
		sgbm.test(slb, srb, disp2, pt2, gt);
		Mat(disp2(cv::Rect(bd, bs, leftim.cols, leftim.rows))).copyTo(disp);
		cv::threshold(disp, disp, (minDisparity << 4) - 1, 0, cv::THRESH_TOZERO);

		if (medianKernel > 1)
		{
			//cout<<"here"<<endl;
			//medianBlur(disp,disp,2*(medianKernel>>1)+1);
		}
		if (subpixel_r != 0)
			binalyWeightedRangeFilter(disp, disp, Size(subpixel_r * 2 + 1, subpixel_r * 2 + 1), (float)subpixel_th);
		if (isOcclusion)
		{
			fillOcclusion(disp);
			Mat a = disp.t();
			fillOcclusion(a);
			Mat(a.t()).copyTo(disp);
		}
		if (isStreakingRemove)
		{
			removeStreakingNoise(disp, disp, 16);
		}
	}

	void StereoSGBMEx::check(Mat& leftim, Mat& rightim, Mat& dispL, Mat& dispR, InputArray ref_)
	{
		Mat ref = ref_.getMat();
		string wname = "StereoSGBMEx_LR";
		namedWindow(wname);

		P1 = 20;
		createTrackbar("P1", wname, &P1, 255);
		P2 = 160;
		createTrackbar("P2", wname, &P2, 255);
		preFilterCap = 25;
		createTrackbar("pre cap", wname, &preFilterCap, 255);

		int WW = 1;
		int WH = 1;
		createTrackbar("windowW", wname, &WW, 10);
		createTrackbar("windowH", wname, &WH, 10);
		uniquenessRatio = 25;
		createTrackbar("post uniq", wname, &uniquenessRatio, 100);
		speckleWindowSize = 100;
		createTrackbar("speckle window", wname, &speckleWindowSize, 100);
		speckleRange = 26;
		createTrackbar("speckle range", wname, &speckleRange, 100);
		createTrackbar("disp12", wname, &disp12MaxDiff, 100);

		int bd = 20;
		createTrackbar("border", wname, &bd, 100);

		cross_check_threshold = 16;
		createTrackbar("cross check", wname, &cross_check_threshold, 50);
		isOcclusion = 1;
		createTrackbar("occlusion", wname, &isOcclusion, 1);
		createTrackbar("isStreak", wname, &isStreakingRemove, 1);
		createTrackbar("post med", wname, &medianKernel, 15);

		int isfullDP = 1;
		createTrackbar("full", wname, &isfullDP, 1);

		int alpha = 0;
		createTrackbar("alpha", wname, &alpha, 100);
		int isColor = 0;
		createTrackbar("color", wname, &isColor, 2);

		int k = 1;
		createTrackbar("mx", wname, &k, 5);
		int dcomp = 1;
		createTrackbar("dcomp", wname, &dcomp, 128);

		int lj = 1;
		createTrackbar("lj", wname, &lj, 128);

		Mat dshow;
		Mat show;
		int key = 0;

		StereoViewSynthesis svs;

		bool isSub = true;
		while (key != 'q')
		{
			SADWindowSize = Size(2 * WW + 1, 2 * WH + 1);

			svs.large_jump = lj;
			fullDP = (isfullDP != 0) ? true : false;
			{
				Timer t("SGBM");
				this->operator()(leftim, rightim, dispL, dispR, bd, cross_check_threshold);
			}

			if (!ref.empty())
			{
				Mat synth, synthd;
				maxFilter(dispL, dispL, Size(2 * k + 1, 2 * k + 1));
				maxFilter(dispR, dispR, Size(2 * k + 1, 2 * k + 1));
				minFilter(dispL, dispL, Size(2 * k + 1, 2 * k + 1));
				minFilter(dispR, dispR, Size(2 * k + 1, 2 * k + 1));

				maxFilter(dispR, dispR, Size(2 * k + 1, 2 * k + 1));

				binalyWeightedRangeFilter(dispL, dispL, Size(9, 9), 32);
				binalyWeightedRangeFilter(dispR, dispR, Size(9, 9), 32);
				binalyWeightedRangeFilter(dispL, dispL, Size(9, 9), 16);
				binalyWeightedRangeFilter(dispR, dispR, Size(9, 9), 16);
				dcomp = max(dcomp, 1);
				if (isSub)
				{
					Mat dl, dr;


					dispL.convertTo(dispL, CV_16S, 1.0 / (double)dcomp);
					dispL.convertTo(dispL, CV_16S, (double)dcomp);
					dispR.convertTo(dispR, CV_16S, 1.0 / dcomp);
					dispR.convertTo(dispR, CV_16S, dcomp);
					Timer t("synth sub");
					//

					/*dispL.convertTo(dl,CV_16S,1.0/16);
					dispR.convertTo(dr,CV_16S,1.0/16);
					dl.convertTo(dispL,CV_16S,16);
					dr.convertTo(dispR,CV_16S,16);*/

					svs(leftim, rightim, dispL, dispR, synth, synthd, 0.5, 0, 16);

					//svs.check(leftim,rightim,dispL,dispR,synth,synthd,0.5,0,16,ref);
					//svs(leftim,dispL,synth,synthd,0.5,0,16);
					//svs(leftim,dispL,synth,synthd,1.0,0,16);
					//svs(leftim,dispL,synth,synthd,1.0,0,16);imshowAnalysisCompare("ana",synth,rightim);
					//svs(rightim,dispR,synth,synthd,-1.0,0,16);imshowAnalysisCompare("ana",synth,leftim);
					//svs(rightim,dispR,synth,synthd,-0.5,0,16);
				}
				else
				{

					Mat dl, dr;
					/*	dispL.convertTo(dl,CV_8U,1.0/16);
					dispR.convertTo(dr,CV_8U,1.0/16);
					//jointBilateralModeFilter(dl,dl,7,10.0,255.0,leftim);
					//jointBilateralModeFilter(dr,dr,7,10.0,255.0,leftim);
					dl.convertTo(dl,CV_8U,1.0/dcomp);
					dl.convertTo(dl,CV_8U,dcomp);
					dr.convertTo(dr,CV_8U,1.0/dcomp);
					dr.convertTo(dr,CV_8U,dcomp);*/

					dispL.convertTo(dl, CV_16S, 1.0 / 16);
					dispR.convertTo(dr, CV_16S, 1.0 / 16);
					//dl.convertTo(dispL,CV_16S,16);
					dr.convertTo(dispR, CV_16S, 16);
					Timer t("synth");
					svs(leftim, rightim, dl, dr, synth, synthd, 0.5, 0, 1);

					//svs(leftim,dl,synth,synthd,0.5,0,1.0);
					//svs(leftim,dispL,synth,synthd,1.0,0,16);
					//svs(rightim,dr,synth,synthd,-0.5,0,1.0);
					//svs(leftim,dispL,synth,synthd,1.0,0,16);imshowAnalysisCompare("ana",synth,rightim);
					//svs(rightim,dispR,synth,synthd,-1.0,0,16);imshowAnalysisCompare("ana",synth,leftim);
				}
				//imshow("synth",synth);
				//imshowAnalysisCompare("comp", synth, ref);
				//Mat mk;Mat gg;cvtColor(synth,gg,COLOR_BGR2GRAY); cv::compare(gg,0,mk,cv::CMP_NE);cout<<calcPSNR(synth,rightim,0,82,mk)<<endl;
				//	Mat mk;Mat gg;cvtColor(synth,gg,COLOR_BGR2GRAY); cv::compare(gg,0,mk,cv::CMP_NE);cout<<calcPSNR(synth,ref,0,82,mk)<<endl;
				imshowDisparity("disp", dispL, 2);
				//cout<<calcPSNRBB(synth,ref,10,10)<<endl;
				//cout<<calcPSNRBB(synth,rightim,10,10)<<endl;
			}

			cvtDisparityColor(dispL, dshow, minDisparity, numberOfDisparities, isColor);

			alphaBlend(leftim, dshow, alpha / 100.0, show);
			imshow(wname, show);
			//imshowAnalysisCompare(wname,dshow,leftim);
			key = waitKey(1);
			if (key == 's')
			{
				isSub = isSub ? false : true;
			}
			if (key == 'b')
			{
				//svs.gui(leftim,dispL,0,16);
			}
			if (key == 'v')
			{
				//svs.gui(leftim,rightim,dispL,dispR,0,16);
			}

		}

		destroyWindow(wname);
	}

	void StereoSGBMEx::check(Mat& leftim, Mat& rightim, Mat& disp, StereoEval& eval)
	{
		string wname = "StereoSGBMEx";
		namedWindow(wname);

		createTrackbar("P1", wname, &P1, 255);
		createTrackbar("P2", wname, &P2, 255);
		createTrackbar("pre cap", wname, &preFilterCap, 255);
		int WW = 1;
		int WH = 1;
		createTrackbar("windowW", wname, &WW, 10);
		createTrackbar("windowH", wname, &WH, 10);

		createTrackbar("post uniq", wname, &uniquenessRatio, 100);
		createTrackbar("speckle window", wname, &speckleWindowSize, 100);
		createTrackbar("speckle range", wname, &speckleRange, 100);
		createTrackbar("disp12", wname, &disp12MaxDiff, 100);

		int bd = 0;
		createTrackbar("border", wname, &bd, 150);
		int occr = 4;
		createTrackbar("occlusion", wname, &isOcclusion, 1);
		createTrackbar("occlusionR", wname, &occr, 15);
		createTrackbar("post med", wname, &medianKernel, 9);

		int isfullDP = 1;
		createTrackbar("full", wname, &isfullDP, 1);

		int alpha = 100;
		createTrackbar("alpha", wname, &alpha, 100);
		int isColor = 0;
		createTrackbar("color", wname, &isColor, 2);

		int noise = 0;
		createTrackbar("noise", wname, &noise, 100);

		int alphaCost = 0;
		createTrackbar("alphaC", wname, &alphaCost, 100);

		ad_max = 31;
		createTrackbar("ad max", wname, &ad_max, 255);


		createTrackbar("sb r", wname, &subpixel_r, 25);
		createTrackbar("sb thresh", wname, &subpixel_th, 255);
		int sbr = 2;
		int sbth = 64;
		createTrackbar("psb r", wname, &sbr, 25);
		createTrackbar("psb thresh", wname, &sbth, 255);

		int crr = 1;
		createTrackbar("crr", wname, &crr, 20);
		int creth = 1;
		createTrackbar("cr th", wname, &creth, 255);
		isStreakingRemove = true;
		int br = 3;
		int pr = 2;
		int sc = 50;
		int ss = 50;
		createTrackbar("br", wname, &br, 20);
		createTrackbar("cs", wname, &sc, 1000);
		createTrackbar("ss", wname, &ss, 255);
		createTrackbar("pr", wname, &pr, 20);
		Mat dshow;
		Mat show;

		Stat st;
		int key = 0;
		while (key != 'q')
		{
			SADWindowSize = Size(2 * WW + 1, 2 * WH + 1);
			fullDP = (isfullDP != 0) ? true : false;
			costAlpha = alphaCost / 100.0;

			Mat lim, rim;
			addNoise(leftim, lim, noise / 10.0);

			imshow("noise", lim);
			cout << "psnr:" << PSNR(leftim, lim) << endl;
			addNoise(rightim, rim, noise / 10.0);

			{
				Timer t("SGBM");
				this->operator()(lim, rim, disp, bd);

				Mat pre;
				Mat mask;
				compare(disp, 0, mask, CMP_EQ);

				/*
				Mat weight = Mat::ones(disp.size(),CV_32F);
				weight.setTo(0.000001,mask);
				Mat disps;
				weightedBoxFilter(disp, weight, disps,Size(2*br+1,2*br+1));


				//boxFilter(disp,  disp,-1,Size(2*br+1,2*br+1));
				//jointBilateralFilter(disp,lim,pre,Size(2*br+1,2*br+1),sc/10.0,ss/10.0);
				//jointNearestFilterBF(pre,disp,Size(2*pr+1,2*pr+1),disp);

				fillOcclusion(disps);
				transpose(disps,pre);
				fillOcclusion(pre);
				transpose(pre,disps);
				disps.copyTo(disp,mask);
				*/
				//correctDisparityBoundary(disp, lim, crr,creth,disp);
				//boxSubpixelRefinement(disp,disp,sbr*2+1,sbth);

				/*Mat dd;
				disp.convertTo(dd,CV_8U,eval.amp/16.0);
				Mat temp;
				jointBilateralFilter(dd,lim,temp,Size(2*br+1,2*br+1),sc/10.0,ss/10.0);
				possibleFilter(temp,dd,Size(2*pr+1,2*pr+1),dd);
				dd.convertTo(disp,CV_16S,4);*/
				/*
				Mat d2;
				guidedFilter(disp,lim,d2,2,0.01);

				*/
			}

			/*
			if(eval.isInit)
			{
			Mat dd;
			disp.convertTo(dd,CV_8U,eval.amp/16.0);
			eval(dd);
			st.push_back(eval.nonocc);
			st.show();

			imshow("eval",eval.nonocc_th);
			}*/

			if (isColor == 0)
				cvtDisparityColor(disp, dshow, minDisparity, numberOfDisparities, isColor);
			else
				disp.convertTo(dshow, CV_8U, eval.amp / 16.0);

			alphaBlend(leftim, dshow, 1.0 - (alpha / 100.0), show);

			imshow("show", show);
			key = waitKey(1);
			if (key == 'k')isStreakingRemove = isStreakingRemove ? false : true;


			if (key == 'r')st.clear();
		}
		destroyWindow(wname);
	}

	//////////////////////////////////////////////////////////////////////////////////////////
	//StereoGCx::
	/*
	StereoGCEx::StereoGCEx(int numDisparities_, int maxIteration)
	{
	maxIters = maxIteration;
	numDisparities = numDisparities_;
	gc = cvCreateStereoGCState(numDisparities, maxIters);
	}
	StereoGCEx::~StereoGCEx()
	{
	cvReleaseStereoGCState(&gc);
	}
	void StereoGCEx::operator()(Mat& leftim, Mat& rightim, Mat& dispL, int bd)
	{
	Mat temp;
	operator()(leftim, rightim, dispL, temp, bd);
	}
	void StereoGCEx::operator()(Mat& leftim, Mat& rightim, Mat& dispL, Mat& dispR, int bd)
	{
	Mat l, r;
	if (leftim.channels() == 3)cvtColor(leftim, l, COLOR_BGR2GRAY);
	else l = leftim;
	if (rightim.channels() == 3)cvtColor(rightim, r, COLOR_BGR2GRAY);
	else r = rightim;

	Mat lb, rb;
	cv::copyMakeBorder(l, lb, 0, 0, bd, bd, cv::BORDER_REPLICATE);
	cv::copyMakeBorder(r, rb, 0, 0, bd, bd, cv::BORDER_REPLICATE);
	//V. Kolmogorov and R. Zabih. Computing visual correspondence with occlusions using graph cuts. ICCV 2001.?
	//Graph Cutの設定

	gc->interactionRadius;
	gc->K;
	gc->lambda;
	gc->lambda1;
	gc->lambda2;
	gc->occlusionCost;
	gc->minDisparity = minDisparity;

	IplImage lim = lb;//MatからIplImageへの変換
	IplImage rim = rb;
	Ptr<IplImage> ipldisp = cvCreateImage(cvGetSize(&lim), IPL_DEPTH_16S, 1);//スマートポインタ．こうするとIplImageでも自動解放
	Ptr<IplImage> ipldisp_r = cvCreateImage(cvGetSize(&lim), IPL_DEPTH_16S, 1);//右奥行き画像,この実装では左右の奥行き画像の整合性と取りながら最適化するため，左右両方とも奥行き画像が計算される

	cvFindStereoCorrespondenceGC(&lim, &rim, ipldisp, ipldisp_r, gc);

	Mat disptemp;
	Mat(ipldisp).convertTo(disptemp, CV_16S, -16);//マイナスに注意
	Mat(disptemp(cv::Rect(bd, 0, leftim.cols, leftim.rows))).copyTo(dispL);

	Mat(ipldisp_r).convertTo(disptemp, CV_16S, 16);//マイナスに注意
	Mat(disptemp(cv::Rect(bd, 0, leftim.cols, leftim.rows))).copyTo(dispR);

	cv::threshold(dispL, dispL, (minDisparity << 4) - 1, 0, cv::THRESH_TOZERO);
	cv::threshold(dispR, dispR, (minDisparity << 4) - 1, 0, cv::THRESH_TOZERO);
	if (isOcclusion)
	{
	fillOcclusion(dispL);
	Mat a = dispL.t();
	fillOcclusion(a);
	Mat(a.t()).copyTo(dispL);

	fillOcclusion(dispR);
	a = dispR.t();
	fillOcclusion(a);
	Mat(a.t()).copyTo(dispR);
	}
	if (medianKernel>1)
	{
	//medianBlur(disp,disp,2*(medianKernel>>1)+1);
	}
	}
	void StereoGCEx::check(Mat& leftim, Mat& rightim, Mat& disp)
	{
	Mat temp;
	check(leftim, rightim, disp, temp);
	}
	void StereoGCEx::check(Mat& leftim, Mat& rightim, Mat& dispL, Mat& dispR)
	{
	string wname = "StereoGCEx";
	namedWindow(wname);

	int preFilterType = StereoBM::PREFILTER_NORMALIZED_RESPONSE; // =CV_STEREO_BM_NORMALIZED_RESPONSE now

	int kf = K;
	int l = lambda;
	int l1 = lambda1;
	int l2 = lambda2;

	createTrackbar("interactionRadius", wname, &interactionRadius, 255);
	createTrackbar("K", wname, &kf, 255);
	createTrackbar("lambda", wname, &l, 255);
	createTrackbar("lambda2", wname, &l1, 255);
	createTrackbar("lambda3", wname, &l2, 255);
	createTrackbar("occcost", wname, &occlusionCost, 255);


	int bd = 0;
	createTrackbar("border", wname, &bd, 100);
	createTrackbar("occlusion", wname, &isOcclusion, 1);
	createTrackbar("post med", wname, &medianKernel, 15);

	int alpha = 100;
	createTrackbar("alpha", wname, &alpha, 100);
	int isColor = 0;
	createTrackbar("color", wname, &isColor, 2);

	Mat dshow;
	Mat show;
	int key = 0;
	while (key != 'q')
	{
	K = kf;
	l = lambda = l;
	l1 = lambda1 = l1;
	l2 = lambda2 = l2;
	{
	CalcTime t("GC");
	this->operator()(leftim, rightim, dispL, dispR, bd);
	}
	cvtDisparityColor(dispL, dshow, minDisparity, numDisparities, isColor);
	alphaBlend(leftim, dshow, alpha / 100.0, show);
	imshow(wname, show);
	key = waitKey(1);
	}
	destroyWindow(wname);
	}
	*/
}