#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void normal(Mat& src, Mat& dest)
{
	double minv, maxv;
	minMaxLoc(src, &minv, &maxv);
	src.convertTo(dest, src.depth(), 1.0 / (maxv - minv), -minv / (maxv - minv));
}

void guiWeightMapTest()
{
	string sequence = "teddy"; int minDisparity = 10; int maxDisparity = 64; int ampDisparity = 4; int rangeDisparity = maxDisparity - minDisparity;
	//string sequence = "Art"; int minDisparity = 0; int maxDisparity = 128; int ampDisparity = 2;
	//string sequence = "Baby1"; int minDisparity = 0; int maxDisparity = 128; int ampDisparity = 2;

	Mat leftim = imread("img/stereo/" + sequence + "/view1.png");
	Mat rightim = imread("img/stereo/" + sequence + "/view5.png");
	Mat leftdisp = imread("img/stereo/" + sequence + "/disp1.png", 0);
	Mat rightdisp = imread("img/stereo/" + sequence + "/disp5.png", 0);

	Mat maskall;
	Mat masknonocc;

	createDisparityALLMask(leftdisp, maskall);
	createDisparityNonOcclusionMask(leftdisp, ampDisparity, 1, masknonocc);
	StereoEval steval(leftdisp, masknonocc, maskall, maskall, ampDisparity);
	//guiShift(leftim, rightim);

	Mat disp;
	
	//ConsoleImage ci;
	string wname = "stereo";
	namedWindow(wname);
	int a = 0; createTrackbar("alpha", wname, &a, 100);
	int wr = 7; createTrackbar("wr", wname, &wr, 20);
	int wsd = 10; createTrackbar("wsd", wname, &wsd, 2550);
	int wsc = 500; createTrackbar("wsc", wname, &wsc, 2550);
	int wss = 500; createTrackbar("wss", wname, &wss, 500);
	
	int iter = 0; createTrackbar("iter", wname, &iter, 10);


	int key = 0;
	Mat show;
	Mat weightMap;
	while (key != 'q')
		//for (int i = 0; i < ;i++)
	{
		Size weightKernelSize = Size(2 * wr + 1, 2 * wr + 1);

		double sigmaWeightDepth = wsd/10.0;
		double sigmaWeightColor = wsc/10.0;
		double sigmaWeightSpace = wss/10.0;
		
		dualBilateralWeightMap(leftdisp, leftim, weightMap, weightKernelSize, sigmaWeightDepth, sigmaWeightColor, sigmaWeightSpace);
		
		//bilateralWeightMap(disp, weightMap, weightKernelSize, sigmaWeightDepth, sigmaWeightSpace);


		//normalize(weightMap, weightMap, 1.0, 0.0, NORM_MINMAX);
		normal(weightMap, weightMap);
		showMatInfo(weightMap);
		imshow("wmap2", weightMap);

		alphaBlend(leftim, leftdisp, a / 100.0, show);
		imshow(wname, show);
		key = waitKey(1);
	}
	/*ibm.doMedianFeedback(leftim, rightim, disp, 0, 0);
	steval(disp);*/

	guiAlphaBlend(leftim, disp);
}