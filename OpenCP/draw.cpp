#include "draw.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void triangle(InputOutputArray src_, Point pt, int length, Scalar& color, int thickness)
	{
		Mat src = src_.getMat();
		int npt[] = { 3, 0 };
		cv::Point pt1[1][3];
		const int h = cvRound(1.7320508*0.5*length);
		pt1[0][0] = Point(pt.x, pt.y - h / 2);;
		pt1[0][1] = Point(pt.x + length / 2, pt.y + h / 2);
		pt1[0][2] = Point(pt.x - length / 2, pt.y + h / 2);

		const cv::Point *ppt[1] = { pt1[0] };

		if (thickness == FILLED)
		{
			fillPoly(src, ppt, npt, 1, color, 1);
		}
		else
		{
			polylines(src, ppt, npt, 1, true, color, thickness);
		}
		src.copyTo(src_);
	}

	void triangleinv(InputOutputArray src_, Point pt, int length, Scalar& color, int thickness)
	{
		Mat src = src_.getMat();
		int npt[] = { 3, 0 };
		cv::Point pt1[1][3];
		const int h = cvRound(1.7320508*0.5*length);
		pt1[0][0] = Point(pt.x, pt.y + h / 2);;
		pt1[0][1] = Point(pt.x + length / 2, pt.y - h / 2);
		pt1[0][2] = Point(pt.x - length / 2, pt.y - h / 2);

		const cv::Point *ppt[1] = { pt1[0] };

		if (thickness == FILLED)
		{
			fillPoly(src, ppt, npt, 1, color, 1);
		}
		else
		{
			polylines(src, ppt, npt, 1, true, color, thickness);
		}
		src.copyTo(src_);
	}

	void drawPlus(InputOutputArray src, Point crossCenter, int length, Scalar& color, int thickness, int line_type, int shift)
	{
		Mat dest = src.getMat();
		if (crossCenter.x == 0 && crossCenter.y == 0)
		{
			crossCenter.x = dest.cols / 2;
			crossCenter.y = dest.rows / 2;
		}

		int hl = length / 2;
		line(dest, Point(crossCenter.x - hl, crossCenter.y), Point(crossCenter.x + hl, crossCenter.y), color, thickness, line_type, shift);
		line(dest, Point(crossCenter.x, crossCenter.y - hl), Point(crossCenter.x, crossCenter.y + hl), color, thickness, line_type, shift);

		dest.copyTo(src);
	}

	void drawTimes(InputOutputArray src, Point crossCenter, int length, Scalar& color, int thickness, int line_type, int shift)
	{
		Mat dest = src.getMat();
		if (crossCenter.x == 0 && crossCenter.y == 0)
		{
			crossCenter.x = dest.cols / 2;
			crossCenter.y = dest.rows / 2;
		}
		int hl = cvRound((double)length / 2.0 / sqrt(2.0));
		line(dest, Point(crossCenter.x - hl, crossCenter.y - hl), Point(crossCenter.x + hl, crossCenter.y + hl), color, thickness, line_type, shift);
		line(dest, Point(crossCenter.x + hl, crossCenter.y - hl), Point(crossCenter.x - hl, crossCenter.y + hl), color, thickness, line_type, shift);

		dest.copyTo(src);
	}

	void drawGrid(InputOutputArray src, Point crossCenter, Scalar& color, int thickness, int line_type, int shift)
	{
		Mat dest = src.getMat();
		if (crossCenter.x == 0 && crossCenter.y == 0)
		{
			crossCenter.x = dest.cols / 2;
			crossCenter.y = dest.rows / 2;
		}

		line(dest, Point(0, crossCenter.y), Point(dest.cols, crossCenter.y), color, thickness, line_type, shift);
		line(dest, Point(crossCenter.x, 0), Point(crossCenter.x, dest.rows), color, thickness, line_type, shift);

		dest.copyTo(src);
	}

	void drawAsterisk(InputOutputArray src, Point crossCenter, int length, Scalar& color, int thickness, int line_type, int shift)
	{
		Mat dest = src.getMat();
		if (crossCenter.x == 0 && crossCenter.y == 0)
		{
			crossCenter.x = dest.cols / 2;
			crossCenter.y = dest.rows / 2;
		}

		int hl = cvRound((double)length / 2.0 / sqrt(2.0));
		line(dest, Point(crossCenter.x - hl, crossCenter.y - hl), Point(crossCenter.x + hl, crossCenter.y + hl), color, thickness, line_type, shift);
		line(dest, Point(crossCenter.x + hl, crossCenter.y - hl), Point(crossCenter.x - hl, crossCenter.y + hl), color, thickness, line_type, shift);

		hl = length / 2;
		line(dest, Point(crossCenter.x - hl, crossCenter.y), Point(crossCenter.x + hl, crossCenter.y), color, thickness, line_type, shift);
		line(dest, Point(crossCenter.x, crossCenter.y - hl), Point(crossCenter.x, crossCenter.y + hl), color, thickness, line_type, shift);

		dest.copyTo(src);
	}

	void eraseBoundary(const Mat& src, Mat& dest, int step, int border)
	{
		Mat temp = src(Rect(step, step, src.cols - 2 * step, src.rows - 2 * step));
		Mat a; temp.copyTo(a);
		Mat b;
		copyMakeBorder(a, dest, step, step, step, step, border);
	}

	template <class T>
	void setTriangleMask(Mat& src)
	{
		float aspect = (float)src.cols / src.rows;
		src.setTo(0);
		for (int j = 0; j < src.rows; j++)
		{
			T* s = src.ptr<T>(j);
			int v = (int)(j*aspect);
			memset(s, 1, (sizeof(T)*src.cols - v));
		}
	}

	void imshowNormalize(string wname, InputArray src)
	{
		//CV_Assert(src.depth() != CV_8U);
		//CV_Assert(src.channels() == 1);
		Mat show;
		normalize(src.getMat(), show, 255, 0, NORM_MINMAX, CV_8U);
		imshow(wname, show);
	}


	void imshowScale(string name, InputArray src, const double alpha, const double beta)
	{
		Mat show;
		src.getMat().convertTo(show, CV_8U, alpha, beta);
		imshow(name, show);
	}

	void patchBlendImage(Mat& src1, Mat& src2, Mat& dest, Scalar linecolor, int linewidth, int direction)
	{
		Mat s1, s2;
		if (src1.channels() == src2.channels())
		{
			s1 = src1;
			s2 = src2;
		}
		else
		{
			if (src1.channels() == 1)
			{
				cvtColor(src1, s1, COLOR_GRAY2BGR);
				s2 = src2;
			}
			else if (src2.channels() == 1)
			{
				s1 = src1;
				cvtColor(src2, s2, COLOR_GRAY2BGR);
			}
		}

		CV_Assert(src1.size() == src2.size());

		Mat mask = Mat::zeros(src1.size(), CV_8U);
		setTriangleMask<uchar>(mask);

		if (direction == 0)
		{
			s2.copyTo(dest);
			s1.copyTo(dest, mask);
			line(dest, Point(src1.cols - 1, 0), Point(0, src1.rows - 1), linecolor, linewidth);
		}
		if (direction == 1)
		{
			s2.copyTo(dest);
			s1.copyTo(dest, mask);
			line(dest, Point(src1.cols - 1, 0), Point(0, src1.rows - 1), linecolor, linewidth);
		}
		if (direction == 2)
		{
			s2.copyTo(dest);
			flip(mask, mask, 0);
			s1.copyTo(dest, mask);
			line(dest, Point(0, 0), Point(src1.cols - 1, src1.rows - 1), linecolor, linewidth);
		}
		if (direction == 3)
		{
			s2.copyTo(dest);
			s1.copyTo(dest, mask);
			flip(mask, mask, 0);
			line(dest, Point(0, 0), Point(src1.cols - 1, src1.rows - 1), linecolor, linewidth);
		}
	}
}