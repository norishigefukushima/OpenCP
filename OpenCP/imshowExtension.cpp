#include "imshowExtension.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void imshowNormalize(string wname, InputArray src, const int norm_type)
	{
		Mat show;
		normalize(src, show, 255, 0, norm_type, CV_8U);
		imshow(wname, show);
	}

	void imshowScale(string name, InputArray src, const double alpha, const double beta)
	{
		Mat show;
		src.getMat().convertTo(show, CV_8U, alpha, beta);
		imshow(name, show);
	}

	void imshowResize(std::string name, cv::InputArray src, const cv::Size dsize, const double fx, const double fy, const int interpolation, const bool isCast8U)
	{
		Mat show;
		if (src.depth() != CV_8U && isCast8U)
		{
			Mat temp;
			src.getMat().convertTo(temp, CV_8U);
			resize(temp, show, dsize, fx, fy, interpolation);
		}
		else
		{
			resize(src, show, dsize, fx, fy, interpolation);
		}
		imshow(name, show);
	}

	void imshowCountDown(string wname, InputArray src, const int waitTime, Scalar color, const int pointSize, std::string fontName)
	{
		Mat s;
		src.copyTo(s);
		addText(s, "3", Point(s.cols / 2, s.rows / 2), fontName, pointSize, color);
		imshow(wname, s);
		waitKey(waitTime);

		src.copyTo(s);
		addText(s, "2", Point(s.cols / 2, s.rows / 2), fontName, pointSize, color);
		imshow(wname, s);
		waitKey(waitTime);

		src.copyTo(s);
		addText(s, "1", Point(s.cols / 2, s.rows / 2), fontName, pointSize, color);
		imshow(wname, s);
		waitKey(waitTime);
	}

	StackImage::StackImage(std::string window_name)
	{
		wname = window_name;
	}

	void StackImage::setWindowName(std::string window_name)
	{
		wname = window_name;
	}

	void StackImage::overwrite(cv::Mat& src)
	{
		if (stack.size() == 0)
		{
			push(src);
			return;
		}

		src.copyTo(stack[stack_max - 1]);
		if (stack_max > 1)
		{
			namedWindow(wname);
			createTrackbar("num", wname, &num_stack, stack_max);
			setTrackbarMax("num", wname, stack_max - 1);
			setTrackbarPos("num", wname, stack_max - 1);
		}
	}

	void StackImage::push(cv::Mat& src)
	{
		stack.push_back(src.clone());
		stack_max = (int)stack.size();

		if (stack_max > 0)
		{
			namedWindow(wname);
			createTrackbar("num", wname, &num_stack, stack_max);
			setTrackbarMax("num", wname, stack_max);
			setTrackbarPos("num", wname, stack_max);
		}
	}

	void StackImage::show(cv::Mat& src)
	{
		if (stack_max == 0) imshow(wname, src);
		else if (stack_max == num_stack) imshow(wname, src);
		else  imshow(wname, stack[num_stack]);
	}

	void StackImage::show()
	{
		if (stack_max > 0) imshow(wname, stack[min(num_stack, stack_max - 1)]);
	}
}