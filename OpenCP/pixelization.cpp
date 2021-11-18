#include "pixelization.hpp"
#include "inlineSIMDfunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	void pixelization(cv::InputArray src, cv::OutputArray dest, const cv::Size pixelSize, const cv::Scalar color, const int thichness)
	{
		Size srcsize = src.size();
		Mat res, bb, bbdst;
		const int w = get_simd_ceil(srcsize.width, pixelSize.width);
		const int h = get_simd_ceil(srcsize.height, pixelSize.height);
		copyMakeBorder(src, bb, 0, h - srcsize.height, 0, w - srcsize.width, cv::BORDER_REPLICATE);
		Size ressize = Size(bb.cols / pixelSize.width, bb.rows / pixelSize.height);

		resize(bb, res, ressize, 0.0, 0.0, INTER_AREA);
		resize(res, bbdst, bb.size(), 0.0, 0.0, INTER_NEAREST);

		if (thichness > 0)
		{
			for (int i = 0; i < ressize.width; i++)
			{
				const int x = pixelSize.width * i;
				if (x < srcsize.width) line(bbdst, Point(x, 0), Point(x, srcsize.height), color, thichness);
			}
			for (int i = 0; i < ressize.height; i++)
			{
				const int y = pixelSize.height * i;
				if (y < srcsize.height) line(bbdst, Point(0, y), Point(srcsize.width, y), color, thichness);
			}
		}
		bbdst(Rect(0, 0, srcsize.width, srcsize.height)).copyTo(dest);
	}

	void guiPixelization(std::string wname, cv::Mat& src)
	{
		namedWindow(wname);
		static int pixel_guiPixelization = 16; createTrackbar("pixel", wname, &pixel_guiPixelization, 50); setTrackbarMin("pixel", wname, 1);
		static int thickness_guiPixelization = 1; createTrackbar("thickness", wname, &thickness_guiPixelization, 10);
		static int gray_guiPixelization = 100; createTrackbar("gray", wname, &gray_guiPixelization, 255);
		Mat dest;
		int key = 0;
		while (key != 'q')
		{
			pixelization(src, dest, Size(pixel_guiPixelization, pixel_guiPixelization), Scalar(gray_guiPixelization, gray_guiPixelization, gray_guiPixelization), thickness_guiPixelization);
			imshow(wname, dest);
			key = waitKey(1);
		}
	}
}