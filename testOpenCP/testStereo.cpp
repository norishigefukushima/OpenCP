#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void guiDisparityPlaneFitSLICTest(Mat& leftim, Mat& rightim, Mat& GT)
{
	string wname = "disparitySLIC";
	namedWindow(wname);
	int mindisparity = 40;
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(mindisparity, 112, 3);
	Mat disparity;
	//guiShift(leftim, rightim);
	sgbm->compute(leftim, rightim, disparity);

	//GT.convertTo(disparity, CV_16S, 8);

	Mat disp = disparity.clone();

	//fillOcclusion(disparity, 0, cp::FILL_DISPARITY);
	fillOcclusion(disparity, 16 * (mindisparity - 1), cp::FILL_DISPARITY);
	/*{
		Mat temp1, temp2;

		disparity.convertTo(temp1, CV_8U, 0.125);
		disp.convertTo(temp2, CV_8U, 0.125);
		guiAlphaBlend(temp1, temp2);
	}*/

	Mat dispshowRef;
	disparity.convertTo(dispshowRef, CV_8U, 0.125);

	int a = 0; createTrackbar("a", wname, &a, 100);
	int S = 16; createTrackbar("S", wname, &S, 200);
	int m = 30; createTrackbar("m", wname, &m, 800);
	int mrs = 10; createTrackbar("ratio of min region size", wname, &mrs, 100);
	int iter = 20; createTrackbar("iteration", wname, &iter, 1000);
	int nransac = 30; createTrackbar("samples", wname, &nransac, 1000);
	int transac = 3; createTrackbar("threshold", wname, &transac, 1000);
	int key = 0;

	Mat refine;
	Mat dispshow;
	Mat show;
	Mat seg;
	Mat slicout;

	bool isStop = false;
	Mat buffer;
	namedWindow("SLIC"); moveWindow("SLIC", 0, 0);
	while (key != 'q')
	{
		{
			Timer t;
			disparityFitPlane(disparity, leftim, refine, S, (float)m, (float)(mrs / 100.0f), iter, nransac, (float)transac);
		}

		//binalyWeightedRangeFilter(disparity, disparity, Size(7, 7), 16);
		if (isStop) buffer.copyTo(refine);
		else refine.copyTo(buffer);

		SLIC(leftim, seg, S, (float)m, mrs / 100.0f, iter);

		refine.convertTo(dispshow, CV_8U, 0.125);
		imshow(wname, dispshow);

		drawSLIC(dispshow, seg, slicout, false, true, Scalar(255, 255, 0));
		imshow("SLIC", slicout);

		alphaBlend(dispshowRef, dispshow, a / 100.0, show);
		imshow(wname, show);

		key = waitKey(1);


		Mat save; cvtColor(show, save, COLOR_GRAY2BGR); static int count = 0; imwrite(format("GIF/out%d.png", count++), save);
		if (key == 'p')
		{
			isStop = (isStop) ? false : true;
		}
		if (key == 'f')
		{
			a = (a != 0) ? 0 : 100;
			setTrackbarPos("a", wname, a);
		}
	}

}

void testCVStereoBM()
{
	Mat disp;
	Mat leftim = imread("img/stereo/Dolls/view1.png", 0);
	Mat rightim = imread("img/stereo/Dolls/view5.png", 0);
	Mat dmap_ = imread("img/stereo/Dolls/disp1.png", 0);
	//guiShift(left, right, 300);
	const int disp_min = get_simd_ceil(32, 16);
	const int disp_max = get_simd_ceil(100, 16);
	cp::StereoEval eval(dmap_, 2, disp_max);

	String wname = "Stereo BM Test";
	namedWindow(wname);
	int blockRadius = 4; createTrackbar("bs", wname, &blockRadius, 30);
	int prefilterCap = 31; createTrackbar("pre cap", wname, &prefilterCap, 63);
	int textureThreshold = 10; createTrackbar("Tx thresh", wname, &textureThreshold, 255);
	int uniquenessRatio = 15; createTrackbar("uniqueness", wname, &uniquenessRatio, 100);
	int lrThreshold = 1; createTrackbar("LR thresh", wname, &lrThreshold, 20);
	int spWindow = 100; createTrackbar("speckle win", wname, &spWindow, 2550);
	int spRange = 32; createTrackbar("speck range", wname, &spRange, 255);
	int minDisp = disp_min; createTrackbar("min disp", wname, &minDisp, 255);
	//int numDisp = numDisparities / 16; createTrackbar("num disp", wname, &numDisp, numDisparities / 16 * 2);

	Ptr<StereoBM> bm = StereoBM::create(disp_max - disp_min, 2 * blockRadius + 1);

	Mat dispshow;
	int key = 0;
	ConsoleImage ci;

	while (key != 'q')
	{
		int SADWindowSize = 2 * blockRadius + 1;
		bm->setPreFilterCap(max(1, prefilterCap));
		bm->setBlockSize(max(SADWindowSize, 5));
		bm->setTextureThreshold(textureThreshold);
		bm->setUniquenessRatio(uniquenessRatio);
		bm->setDisp12MaxDiff(lrThreshold);
		bm->setSpeckleWindowSize(spWindow);
		bm->setSpeckleRange(spRange);
		bm->setMinDisparity(minDisp);
		//bm->setNumDisparities(numDisp * 16);

		bm->compute(leftim, rightim, disp);
		const int invalid = (minDisp - 1) * 16;
		fillOcclusion(disp, invalid);
		//normalize(disp, dispshow, minDisp, numDisp * 16, NORM_MINMAX, CV_8U);
		//disp.convertTo(dispshow, CV_8U, 255.0/(16.0*numDisparities));
		ci("th 0.5:" + eval(disp, 0.5, 16, false));
		ci("th 1.0:" + eval(disp, 1.0, 16, false));
		ci("th 2.0:" + eval(disp, 2.0, 16, false));
		imshowScale(wname, disp, 2.0 / 16);
		ci.show();
		key = waitKey(1);
	}
}

void testCVStereoSGBM()
{
	Mat disp;
	Mat leftim = imread("img/stereo/Dolls/view1.png");
	Mat rightim = imread("img/stereo/Dolls/view5.png");
	Mat dmap_ = imread("img/stereo/Dolls/disp1.png",0);
	//guiShift(left, right, 300);
	const int disp_min = get_simd_ceil(32, 16);
	const int disp_max = get_simd_ceil(100, 16);
	int numDisparities = disp_max - disp_min;
	cp::StereoEval eval(dmap_, 2, disp_max);

	String wname = "Stereo SGBM Test";
	namedWindow(wname);
	int blockRadius = 1; createTrackbar("bs", wname, &blockRadius, 30);

	int P1 = 8; createTrackbar("P1", wname, &P1, 255);
	int P2 = 32; createTrackbar("P2", wname, &P2, 255);
	int prefilterCap = 31; createTrackbar("pre cap", wname, &prefilterCap, 63);
	//int textureThreshold = 10; createTrackbar("Tx thresh", wname, &textureThreshold, 255);
	int uniquenessRatio = 15; createTrackbar("uniqueness", wname, &uniquenessRatio, 100);
	int lrThreshold = 1; createTrackbar("LR thresh", wname, &lrThreshold, 20);
	int spWindow = 100; createTrackbar("speckle win", wname, &spWindow, 2550);
	int spRange = 32; createTrackbar("speck range", wname, &spRange, 255);

	Ptr<StereoSGBM> bm = StereoSGBM::create(disp_min, numDisparities, 2 * blockRadius + 1);

	Mat lim; copyMakeBorder(leftim, lim, 0, 0, numDisparities, 0, BORDER_REPLICATE);
	Mat rim; copyMakeBorder(rightim, rim, 0, 0, numDisparities, 0, BORDER_REPLICATE);
	Mat dispshowROI, dispshow;
	int key = 0;
	ConsoleImage ci;

	while (key != 'q')
	{
		int SADWindowSize = 2 * blockRadius + 1;
		bm->setPreFilterCap(max(1, prefilterCap));
		bm->setP1(P1);
		bm->setP2(P2);
		bm->setBlockSize(SADWindowSize);
		//bm->setTextureThreshold(textureThreshold);
		bm->setUniquenessRatio(uniquenessRatio);
		bm->setDisp12MaxDiff(lrThreshold);
		bm->setSpeckleWindowSize(spWindow);
		bm->setSpeckleRange(spRange);
		/*
		bm->setROI1(roi1);
		bm->setROI2(roi2);
		bm->setMinDisparity(0);
		bm->setNumDisparities(numberOfDisparities);
		*/
		bm->compute(lim, rim, disp);

		const int invalid = (disp_min - 1) * 16;
		fillOcclusion(disp, invalid);
		Mat(disp(Rect(numDisparities, 0, leftim.cols, leftim.rows))).copyTo(dispshow);
		//guiAlphaBlend(dispshow, dmap_);
		
		imshowScale(wname, dispshow, 2.0 / 16);
		
		ci("th 0.5:" + eval(dispshow, 0.5, 16, false));
		ci("th 1.0:" + eval(dispshow, 1.0, 16, false));
		ci("th 2.0:" + eval(dispshow, 2.0, 16, false));
		ci.show();
		key = waitKey(1);
	}
}

void testStereoBase()
{
	Mat disp;
	Mat left_ = imread("img/stereo/Dolls/view1.png");
	Mat right = imread("img/stereo/Dolls/view5.png");
	Mat dmap_ = imread("img/stereo/Dolls/disp1.png", 0);
	//guiShift(left, right, 300);
	const int disp_min = get_simd_ceil(32, 16);
	const int disp_max = get_simd_ceil(100, 16);
	cp::StereoEval eval(dmap_, 2, disp_max);

	//resize(left, left, Size(), 1, 0.25);
	//resize(right, right, Size(), 1, 0.25);

	StereoBase sbm(5, disp_min, disp_max);
	sbm.gui(left_, right, disp, eval);
	//sbm.check(leftg, rightg, disp);
}