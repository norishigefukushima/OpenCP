#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testWebPAnimation()
{
	//Webp test
	vector<Mat> a(80);

	for (int i = 0; i < 80; i++)
	{
		Mat img = imread("img/flower.png");
		add(img, Scalar::all(i), a[i]);
		//addNoise(a[i], a[i], 5);
		putText(a[i], format("%d", i), Point(a[i].cols / 2, a[i].rows / 2), cv::FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 1);
		//imwrite(format("data/o%03d.png", i), a[i]);
		//imshow("a", a[i]);
		//waitKey(1);
	}

	int size = 0;
	vector<int> parameters;
	parameters.push_back(IMWRITE_WEBP_COLORSPACE);
	parameters.push_back(0);
	parameters.push_back(IMWRITE_WEBP_METHOD);
	parameters.push_back(6);
	parameters.push_back(IMWRITE_WEBP_QUALITY);
	parameters.push_back(10);
	parameters.push_back(IMWRITE_WEBP_TIMEMSPERFRAME);
	parameters.push_back(67);//15 fps
	{
		Timer t("write YUV");
		size = cp::imwriteAnimationWebp("webp/yuv.webp", a, parameters);
	}
	cout << size / 1024.0 << " Kbyte" << endl;

	parameters[1] = 1;
	{
		Timer t("write YUV Sharp");
		size = cp::imwriteAnimationWebp("webp/rgb.webp", a, parameters);
	}
	cout << size / 1024.0 << " Kbyte" << endl;

	parameters[1] = 2;
	{
		Timer t("write RGB");
		size = cp::imwriteAnimationWebp("webp/rgb.webp", a, parameters);
	}
	cout << size / 1024.0 << " Kbyte" << endl;
}