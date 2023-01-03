#include "draw.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void triangle(InputOutputArray src_, Point pt, int length, Scalar color, int thickness)
	{
		Mat src = src_.getMat();
		int npt[] = { 3, 0 };
		cv::Point pt1[1][3];
		const int h = cvRound(1.7320508 * 0.5 * length);
		pt1[0][0] = Point(pt.x, pt.y - h / 2);;
		pt1[0][1] = Point(pt.x + length / 2, pt.y + h / 2);
		pt1[0][2] = Point(pt.x - length / 2, pt.y + h / 2);

		const cv::Point* ppt[1] = { pt1[0] };

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

	void triangleinv(InputOutputArray src_, Point pt, int length, Scalar color, int thickness)
	{
		Mat src = src_.getMat();
		int npt[] = { 3, 0 };
		cv::Point pt1[1][3];
		const int h = cvRound(1.7320508 * 0.5 * length);
		pt1[0][0] = Point(pt.x, pt.y + h / 2);;
		pt1[0][1] = Point(pt.x + length / 2, pt.y - h / 2);
		pt1[0][2] = Point(pt.x - length / 2, pt.y - h / 2);

		const cv::Point* ppt[1] = { pt1[0] };

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

	void diamond(InputOutputArray src_, Point pt, int length, Scalar color, int thickness)
	{
		Mat src = src_.getMat();
		int npt[] = { 4, 0 };
		cv::Point pt1[1][4];

		pt1[0][0] = Point(pt.x, int(pt.y + length * 0.5));
		pt1[0][1] = Point(int(pt.x + length * 0.5), pt.y);
		pt1[0][2] = Point(pt.x, int(pt.y - length * 0.5));
		pt1[0][3] = Point(int(pt.x - length * 0.5), pt.y);

		const cv::Point* ppt[1] = { pt1[0] };

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

	void pentagon(InputOutputArray src_, Point pt, int length, Scalar color, int thickness)
	{
		Mat src = src_.getMat();
		int npt[] = { 5, 0 };
		cv::Point pt1[1][5];
		for (int i = 0; i < 5; i++)
		{
			const int xx = int(cos(CV_2PI * i / 5.0) * 0.5 * length);
			const int yy = int(sin(CV_2PI * i / 5.0) * 0.5 * length);
			pt1[0][i] = Point(pt.x + yy, pt.y + xx);
		}

		const cv::Point* ppt[1] = { pt1[0] };

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

	void drawPlus(InputOutputArray src, Point crossCenter, int length, Scalar color, int thickness, int line_type, int shift)
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

	void drawTimes(InputOutputArray src, Point crossCenter, int length, Scalar color, int thickness, int line_type, int shift)
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

	void drawGrid(InputOutputArray src, Point crossCenter, Scalar color, int thickness, int line_type, int shift)
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

	void drawGridMulti(InputOutputArray src, Size division, Scalar color, int thickness, int line_type, int shift)
	{
		const int w = src.size().width;
		const int h = src.size().height;
		for (int j = 0; j < division.height - 1; j++)
		{
			int py = (j + 1) * (w / division.height);
			line(src, Point(py, 0), Point(py, h), color, thickness, line_type, shift);
		}

		for (int i = 0; i < division.width - 1; i++)
		{
			int px = (i + 1) * (h / division.height);
			line(src, Point(0, px), Point(w, px), color, thickness, line_type, shift);
		}
	}

	void drawAsterisk(InputOutputArray src, Point crossCenter, int length, Scalar color, int thickness, int line_type, int shift)
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

	void polylines(InputOutputArray src, vector<Point2f>& points, const bool isclosed, Scalar color, const int thickness, const int lineType, const int shift)
	{
		vector<Point> pt;
		for (int i = 0; i < points.size(); i++)pt.push_back(Point(points[i]));
		cv::polylines(src, pt, isclosed, color, thickness, lineType, shift);
	}

	void eraseBoundary(const Mat& src, Mat& dest, const int step, const int borderType)
	{
		Mat temp = src(Rect(step, step, src.cols - 2 * step, src.rows - 2 * step));
		Mat a; temp.copyTo(a);
		Mat b;
		copyMakeBorder(a, dest, step, step, step, step, borderType);
	}

}