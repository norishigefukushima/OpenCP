#include "highguiex.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	Rect getNonZeroMaxRectangle(InputArray src)
	{
		Mat s = src.getMat();
		int t = 0, b = 0, l = 0, r = 0;

		for (int i = 0; i < s.rows; i++)
		{
			if (countNonZero(s.row(i)) != 0)
			{
				t = i;
				break;
			}
		}
		for (int i = s.rows - 1; i >= 0; i--)
		{
			if (countNonZero(s.row(i)) != 0)
			{
				b = i;
				break;
			}
		}
		for (int i = 0; i < s.cols; i++)
		{
			if (countNonZero(s.col(i)) != 0)
			{
				l = i;
				break;
			}
		}
		for (int i = s.cols - 1; i >= 0; i--)
		{
			if (countNonZero(s.col(i)) != 0)
			{
				r = i;
				break;
			}
		}
		if (r == l)return Rect(0, 0, 0, 0);
		return Rect(l, t, r - l + 1, b - t + 1);
	}

	cv::Mat getTextImageQt(string message, string font, const int fontSize, Scalar text_color, Scalar background_color, bool isItalic)
	{
		int count = message.size() + 1;
		Mat image = Mat::zeros(2 * (fontSize + 1), (fontSize + 1) * count, CV_8UC3);
		if (isItalic)
		{
			cv::addText(image, message, Point(fontSize, fontSize), font, fontSize, Scalar(255, 255, 255, 0), QT_FONT_NORMAL, QT_STYLE_ITALIC);
		}
		else
		{
			cv::addText(image, message, Point(fontSize, fontSize), font, fontSize, Scalar(255, 255, 255, 0), QT_FONT_NORMAL, QT_STYLE_NORMAL);
		}
		Mat bw; cvtColor(image, bw, COLOR_BGR2GRAY);
		Rect r = getNonZeroMaxRectangle(bw);

		Mat ret = Mat::zeros(2 * (fontSize + 1), (fontSize + 1) * count, CV_8UC3);
		ret.setTo(background_color);
		if (isItalic)
		{
			cv::addText(ret, message, Point(fontSize, fontSize), font, fontSize, text_color, QT_FONT_NORMAL, QT_STYLE_ITALIC);
		}
		else
		{
			cv::addText(ret, message, Point(fontSize, fontSize), font, fontSize, text_color);
		}
		return ret(r);
	}

	Size getTextSizeQt(string message, string font, const int fontSize)
	{
		Mat image = getTextImageQt(message, font, fontSize, Scalar(255, 255, 255, 0), Scalar::all(0));
		return Size(image.cols, image.rows);
	}
}