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
	fillOcclusion(disparity, 16*(mindisparity-1), cp::FILL_DISPARITY);
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
			CalcTime t;
			dispalityFitPlane(disparity, leftim, refine, S, (float)m, mrs / 100.0f, iter, nransac, transac);
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

		
		Mat save; cvtColor(show, save, CV_GRAY2BGR); static int count = 0; imwrite(format("GIF/out%d.png", count++), save);
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

void guiStereo()
{
	;
}