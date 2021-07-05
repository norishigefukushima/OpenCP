#include "VideoSubtitle.hpp"


using namespace std;
using namespace cv;

namespace cp
{
	int VideoSubtitle::getAlpha()
	{
		double time = tscript.getTime();
		int ret = int((time - time_dissolve_start) / (time_dissolve) * 255.0);
		ret = max(0, min(255, ret));

		return ret;
	}

	cv::Rect VideoSubtitle::getRectText(std::vector<std::string>& text, std::vector<int>& fontSize)
	{
		int vtextsize = text.size();
		int h = fontSize[0];

		for (int i = 0; i < vtextsize; i++)
		{
			h += fontSize[i] + vspace;
		}

		int w = (text[0].size() + 1) * fontSize[0];
		int argmax = 0;
		for (int i = 1; i < vtextsize; i++)
		{
			const int size = (text[i].size() + 1) * fontSize[i];
			if (size > w)
			{
				w = size;
				argmax = i;
			}
		}

		const int textsize = text.size();
		Mat temp = Mat::zeros(h, w, CV_8UC3);
		addVText(temp, text, Point(0, 0), fontSize, Scalar(255, 255, 255, 0));
		Rect roi_ = boundingRect(temp.reshape(1, temp.rows));
		Rect roi = Rect(roi_.x / 3, roi_.y, roi_.width / 3, roi_.height);

		return roi;
	}

	void VideoSubtitle::addVText(Mat& image, std::vector<std::string>& text, Point point, std::vector<int>& fontSize, Scalar color)
	{
		for (int i = 0; i < text.size(); i++)
		{
			int v = 0;
			for (int n = 0; n <= i; n++)
			{
				v += fontSize[n] + vspace;
			}
			addText(image, text[i], Point(point.x, point.y + v), font, fontSize[i], color);
		}
	}

	//public

	VideoSubtitle::VideoSubtitle()
	{
		tscript.init("", cp::TIME_MSEC, false);
	}

	void VideoSubtitle::restart()
	{
		tscript.start();
	}

	void VideoSubtitle::setDisolveTime(const double start, const double end)
	{
		if (start > end)
		{
			time_dissolve_start = end;
			time_dissolve_end = start;
		}
		else
		{
			time_dissolve_start = start;
			time_dissolve_end = end;
		}
		time_dissolve = time_dissolve_end - time_dissolve_start;
		restart();
	}

	void VideoSubtitle::setFontType(std::string font)
	{
		this->font = font;
	}

	void VideoSubtitle::setVSpace(const int vspace)
	{
		this->vspace = vspace;
	}

	void VideoSubtitle::setTitle(const cv::Size size, std::string text, int fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor, POSITION pos)
	{
		vector<string> vtext = { text };
		vector<int> vfontSize = { fontSize };
		setTitle(size, vtext, vfontSize, textcolor, backgroundcolor, pos);
	}

	void VideoSubtitle::setTitle(const cv::Size size, std::vector<std::string>& text, std::vector<int>& fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor, POSITION pos)
	{
		this->text.resize(0);
		this->fontSize.resize(0);
		copy(text.begin(), text.end(), back_inserter(this->text));
		copy(fontSize.begin(), fontSize.end(), back_inserter(this->fontSize));

		title.create(size, CV_8UC3);
		title.setTo(backgroundcolor);
		textROI = getRectText(text, fontSize);

		if (pos == POSITION::TOP) textPoint = Point(title.cols / 2 - textROI.width / 2, vspace);
		if (pos == POSITION::CENTER) textPoint = Point(title.cols / 2 - textROI.width / 2, title.rows / 2- textROI.height / 2);
		if (pos == POSITION::BOTTOM) textPoint = Point(title.cols / 2 - textROI.width / 2, title.rows-textROI.height-vspace);

		addVText(title, text, textPoint, fontSize, textcolor);
	}

	void VideoSubtitle::showTitleDissolve(std::string wname, const cv::Mat& image)
	{
		int text_alpha = getAlpha();
		if (text_alpha != 255)addWeighted(image, text_alpha / 255.0, title, 1.0 - text_alpha / 255.0, 0.0, show);
		else image.copyTo(show);

		imshow(wname, show);
	}

	void VideoSubtitle::showScriptDissolve(std::string wname, const cv::Mat& image, const cv::Scalar textColor)
	{
		int text_alpha = getAlpha();
		image.copyTo(show);
		if (text_alpha != 255) addVText(show, text, textPoint, fontSize, Scalar(textColor.val[0], textColor.val[1], textColor.val[2], text_alpha));

		imshow(wname, show);
	}

	void VideoSubtitle::showTitle(std::string wname, const cv::Size size, std::vector<std::string>& text, std::vector<int>& fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor)
	{
		setTitle(size, text, fontSize, textcolor, backgroundcolor);
		imshow(wname, title);
	}

	void VideoSubtitle::showTitle(std::string wname, const cv::Size size, std::string text, const int fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor)
	{
		vector<string> vtext = { text };
		vector<int> vfontSize = { fontSize };
		showTitle(wname, size, vtext, vfontSize, textcolor, backgroundcolor);
	}
}