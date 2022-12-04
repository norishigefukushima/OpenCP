#include <opencp.hpp>
#include <opencv2/ximgproc.hpp>
using namespace std;
using namespace cv;
using namespace cp;

struct MouseDTFParameter
{
	cv::Rect pt;
	std::string wname;
	MouseDTFParameter(int x, int y, int width, int height, std::string name)
	{
		pt = cv::Rect(x, y, width, height);
		wname = name;
	}
};

void guiMouseDTFOnMouse(int event, int x, int y, int flags, void* param)
{
	MouseDTFParameter* retp = (MouseDTFParameter*)param;

	if (flags == EVENT_FLAG_LBUTTON)
	{
		retp->pt.x = max(0, min(retp->pt.width - 1, x));
		retp->pt.y = max(0, min(retp->pt.height - 1, y));

		setTrackbarPos("x", retp->wname, x);
		setTrackbarPos("y", retp->wname, y);
	}
}


void guiDomainTransformFilterTest(Mat& src_)
{
	Mat src = cp::convert(src_, CV_32F);
	string wname = "domain transform  filter";
	namedWindow(wname);
	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 1; createTrackbar("switch", wname, &sw, 2);
	
	int space = 500; createTrackbar("space", wname, &space, 2000);
	int color = 700; createTrackbar("color", wname, &color, 2550);
	int iter = 3; createTrackbar("iter", wname, &iter, 10);
	static MouseDTFParameter param(src.cols / 2, src.rows / 2, src.cols, src.rows, wname);
	setMouseCallback(wname, (MouseCallback)guiMouseDTFOnMouse, (void*)&param);
	createTrackbar("x", wname, &param.pt.x, src.cols - 1);
	createTrackbar("y", wname, &param.pt.y, src.rows - 1);

	Mat dest;
	int key = 0;
	Mat show;

	while (key != 'q')
	{
		Mat aa = Mat::zeros(src.size(), CV_32F);
		aa.at<float>(param.pt.y, param.pt.x) = 1.0;
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color / 10.f;
		float sigma_space = space / 10.f;

		if (sw == 0)
		{
			Timer t("epf: domain transform in namespace cv");
			edgePreservingFilter(src, dest, RECURS_FILTER, sigma_space, sigma_color / 255.f);
		}
		else if (sw == 1)
		{
			Timer t("domain transform in namespace xphoto");
			ximgproc::dtFilter(src, aa, dest, sigma_space, sigma_color, 1, iter);//NC, RF, IC
			normalize(dest, dest, 255, 0, NORM_MINMAX);
		}
		else if (sw == 2)
		{
			Timer t("domain transform in opencp");
			//ximgproc::jointBilateralFilter(src, aa, dest, (sigma_space*3)*2+1, sigma_color, sigma_space);//NC, RF, IC
			cp::jointBilateralFilter(aa, src, dest, (int)ceil(sigma_space * 3) * 2 + 1, sigma_color, sigma_space);
			normalize(dest, dest, 255, 0, NORM_MINMAX);
			//domainTransformFilter(src, dest, sigma_color, sigma_space, iter, 1, DTF_RF, DTF_BGRA_SSE_PARALLEL);
		}
		
		if (key == 'f')
		{
			a = (a == 0) ? 100 : 0;
			setTrackbarPos("a", wname, a);
		}
		Mat cc; dest.convertTo(cc, CV_8U);
		applyColorMap(cc, show, 2);
		alphaBlend(src_, show, a / 100.0, show);
		cp::imshowScale(wname, show);
		key = waitKey(1);
	}
}