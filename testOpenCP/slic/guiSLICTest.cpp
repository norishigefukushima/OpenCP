#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void SLICTestSegment2Vector(Mat& src)
{
	Mat seg;
	SLIC(src, seg, 16, 30, 0.1, 20);

	vector<vector<Point>> pts;
	SLICSegment2Vector(seg, pts);
	Mat segout;
	SLICVector2Segment(pts, src.size(), segout);

	if (MSE(seg, segout) == 0.0)cout << "segmentation seg2vec test:OK\n";
}

void guiSLICTest(Mat& src)
{
	
	Mat dest, dest2;

	string wname = "SLIC";
	namedWindow(wname);

	int a = 50; createTrackbar("a", wname, &a, 100);
	int S = 16; createTrackbar("S", wname, &S, 200);
	int m = 30; createTrackbar("m", wname, &m, 800);

	int mrs = 10; createTrackbar("ratio of min region size", wname, &mrs, 100);
	int iter = 20; createTrackbar("iteration", wname, &iter, 1000);
	int key = 0;
	Mat seg;
	Mat lab;
	while (key != 'q')
	{
		Mat show;
		{
			CalcTime t("slic all");
			cvtColor(src, lab, COLOR_BGR2Lab);
			SLIC(lab, seg, S, (float)m, mrs / 100.0f, iter);
		}
		drawSLIC(src, seg, dest, true, Scalar(255, 255, 0));

		Mat dest2;
		SLIC(lab, seg, S * 2, (float)m, mrs / 100.f, iter);
		drawSLIC(src, seg, dest2, true, Scalar(255, 255, 0));

		patchBlendImage(dest, dest2, show, Scalar(255, 255, 255));

		alphaBlend(src, show, a / 100.0, show);
		imshow(wname, show);
		key = waitKey(1);

		if (key == 't')
		{
			SLICTestSegment2Vector(src);
		}
	}
}