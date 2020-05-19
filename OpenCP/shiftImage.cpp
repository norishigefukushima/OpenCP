#include "shiftImage.hpp"
#include "blend.hpp"
#include "matinfo.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void warpShiftSubpix(InputArray src_, OutputArray dest_, double shiftx, double shifty, const int inter_method)
	{
		Mat src = src_.getMat();
		dest_.create(src.size(), src.type());
		Mat dest = dest_.getMat();

		Mat aff = Mat::zeros(2, 3, CV_64F);

		aff.at<double>(0, 0) = 1.0;
		aff.at<double>(0, 1) = 0.0;
		aff.at<double>(0, 2) = shiftx;

		aff.at<double>(1, 0) = 0.0;
		aff.at<double>(1, 1) = 1.0;
		aff.at<double>(1, 2) = shifty;

		warpAffine(src, dest, aff, src.size(), inter_method, 0);
	}

	template <typename T>
	void warpShiftH_(InputArray src_, OutputArray dest_, const int shiftH)
	{
		if (dest_.empty())dest_.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat dest = dest_.getMat();

		int c = src.channels();
		int shiftPixel = shiftH*c;
		const int step = src.cols*c;
		T* s = src.ptr<T>(0);
		T* d = dest.ptr<T>(0);

		if (shiftH >= 0)
		{
			if (c == 1)
			{
				for (int j = 0; j < src.rows; j++)
				{
					const T v = s[0];
					for (int n = 0; n < shiftPixel; n++) d[n] = v;
					memcpy(d + shiftPixel, s, sizeof(T)*(step - shiftPixel));
					s += step; d += step;
				}
			}
			else if (c == 3)
			{
				for (int j = 0; j < src.rows; j++)
				{
					T* v = &s[0];
					for (int i = 0; i < shiftH; i++) memcpy(d + 3 * i, v, sizeof(T) * 3);
					memcpy(d + shiftPixel, s, sizeof(T)*(step - shiftPixel));
					s += step; d += step;
				}
			}
		}
		else
		{
			if (c == 1)
			{
				for (int j = 0; j < src.rows; j++)
				{
					const T v = s[step - c];
					memcpy(d, s - shiftPixel, sizeof(T)*(step + shiftPixel));
					for (int n = 0; n < -shiftPixel; n++) d[step + shiftPixel + n] = v;
					s += step; d += step;
				}
			}
			else if (c == 3)
			{
				for (int j = 0; j < src.rows; j++)
				{
					T* v = &s[step - c];
					memcpy(d, s - shiftPixel, sizeof(T)*(step + shiftPixel));
					for (int i = -shiftH; i>0; i--)
						memcpy(d + step - c * i, v, sizeof(T) * 3);
					s += step; d += step;
				}
			}
		}
	}

	void warpShiftH_8u(InputArray src_, OutputArray dest_, const int shiftH)
	{
		if (dest_.empty())dest_.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat dest = dest_.getMat();

		int c = src.channels();
		int shiftPixel = shiftH*c;
		const int step = src.cols*c;
		uchar* s = src.ptr<uchar>(0);
		uchar* d = dest.ptr<uchar>(0);

		if (shiftH >= 0)
		{
			if (c == 1)
			{
				for (int j = 0; j < src.rows; j++)
				{
					const uchar v = s[0];
					memset(d, v, shiftPixel);
					memcpy(d + shiftPixel, s, (step - shiftPixel));
					s += step; d += step;
				}
			}
			else if (c == 3)
			{
				for (int j = 0; j < src.rows; j++)
				{
					uchar* v = &s[0];
					for (int i = 0; i < shiftH; i++) memcpy(d + 3 * i, v, 3);
					memcpy(d + shiftPixel, s, (step - shiftPixel));
					s += step; d += step;
				}
			}
		}
		else
		{
			if (c == 1)
			{
				for (int j = 0; j < src.rows; j++)
				{
					const uchar v = s[step - c];
					memcpy(d, s - shiftPixel, (step + shiftPixel));
					memset(d + step + shiftPixel, v, -1 * shiftPixel);
					s += step; d += step;
				}
			}
			else if (c == 3)
			{
				for (int j = 0; j < src.rows; j++)
				{
					uchar* v = &s[step - c];
					memcpy(d, s - shiftPixel, (step + shiftPixel));
					for (int i = -shiftH; i>0; i--)
						memcpy(d + step - c * i, v, 3);
					s += step; d += step;
				}
			}
		}
	}

	void warpShiftH(InputArray src, OutputArray dest, const int shiftH)
	{
		if (src.depth() == CV_8U) warpShiftH_8u(src, dest, shiftH);
		else if (src.depth() == CV_16U) warpShiftH_<ushort>(src, dest, shiftH);
		else if (src.depth() == CV_16S) warpShiftH_<short>(src, dest, shiftH);
		else if (src.depth() == CV_32S) warpShiftH_<int>(src, dest, shiftH);
		else if (src.depth() == CV_32F) warpShiftH_<float>(src, dest, shiftH);
		else if (src.depth() == CV_64F) warpShiftH_<double>(src, dest, shiftH);
	}

	void warpShift_(Mat& src, Mat& dst, int shiftx, int shifty)
	{
		Mat dest = Mat::zeros(src.size(), src.type());

		int width = src.cols;
		int height = src.rows;
		if (shiftx >= 0 && shifty >= 0)
		{
			Mat d = dest(Rect(shiftx, shifty, width - shiftx, height - shifty));
			Mat(src(Rect(0, 0, width - shiftx, height - shifty))).copyTo(d);
		}
		else if (shiftx >= 0 && shifty < 0)
		{
			Mat d = dest(Rect(shiftx, 0, width - shiftx, height + shifty));
			Mat(src(Rect(0, -shifty, width - shiftx, height + shifty))).copyTo(d);
		}
		else if (shiftx < 0 && shifty < 0)
		{
			Mat d = dest(Rect(0, 0, width + shiftx, height + shifty));
			Mat(src(Rect(-shiftx, -shifty, width + shiftx, height + shifty))).copyTo(d);
		}
		else if (shiftx < 0 && shifty >= 0)
		{
			Mat d = dest(Rect(0, shifty, width + shiftx, height - shifty));
			Mat(src(Rect(-shiftx, 0, width + shiftx, height - shifty))).copyTo(d);
		}
		dest.copyTo(dst);
	}


	void warpShift_(Mat& src, Mat& dest, int shiftx, int shifty, int borderType)
	{
		int width = src.cols;
		int height = src.rows;
		if (shiftx >= 0 && shifty >= 0)
		{
			Mat im; copyMakeBorder(src, im, shifty, 0, shiftx, 0, borderType);
			Mat(im(Rect(0, 0, width, height))).copyTo(dest);
		}
		else if (shiftx >= 0 && shifty < 0)
		{
			Mat im; copyMakeBorder(src, im, 0, -shifty, shiftx, 0, borderType);
			Mat(im(Rect(0, -shifty, width, height))).copyTo(dest);
		}
		else if (shiftx < 0 && shifty < 0)
		{
			Mat im; copyMakeBorder(src, im, 0, -shifty, 0, -shiftx, borderType);
			Mat(im(Rect(-shiftx, -shifty, width, height))).copyTo(dest);
		}
		else if (shiftx < 0 && shifty >= 0)
		{
			Mat im; copyMakeBorder(src, im, shifty, 0, 0, -shiftx, borderType);
			Mat(im(Rect(-shiftx, 0, width, height))).copyTo(dest);
		}
	}

	void warpShift(InputArray src_, OutputArray dest_, int shiftx, int shifty, int borderType)
	{
		Mat src = src_.getMat();
		if(dest_.empty() || dest_.size()!=src_.size() || dest_.type()!=src_.type())dest_.create(src.size(), src.type());
		Mat dest = dest_.getMat();

		if (borderType < 0)
			warpShift_(src, dest, shiftx, shifty);
		else
			warpShift_(src, dest, shiftx, shifty, borderType);
	}

	

	Mat guiShift(cv::InputArray fiximg, cv::InputArray moveimg, const int max_move, std::string window_name)
	{
		Mat dest = Mat::zeros(fiximg.size(), fiximg.type());
		namedWindow(window_name);
		int a = 50;
		createTrackbar("alpha", window_name, &a, 100);
		int x = max_move;
		int y = max_move;
		createTrackbar("x-max_move", window_name, &x, 2 * max_move);
		createTrackbar("y-max_move", window_name, &y, 2 * max_move);

		int key = 0;
		Mat show;
		
		while (key != 'q')
		{
			warpShift(moveimg, dest, x - max_move, y - max_move, BORDER_REPLICATE);
			alphaBlend(fiximg, dest, 1.0 - a / 100.0, show);
			imshow(window_name, show);
			key = waitKey(1);

			if (key == 'f')
			{
				a = (a == 0) ? a = 100 : 0;
				setTrackbarPos("alpha", window_name, a);
			}
			if (key == 'l')
			{
				x--;
				setTrackbarPos("x", window_name, x);
			}
			if (key == 'j')
			{
				x++;
				setTrackbarPos("x", window_name, x);
			}
			if (key == 'i')
			{
				y++;
				setTrackbarPos("y", window_name, y);
			}
			if (key == 'k')
			{
				y--;
				setTrackbarPos("y", window_name, y);
			}
		}
		destroyWindow(window_name);
		return dest;
	}

	void guiShift(cv::InputArray centerimg, cv::InputArray leftimg, cv::InputArray rightimg, int max_move, std::string window_name)
	{
		namedWindow(window_name);
		int a = 50;
		createTrackbar("alpha", window_name, &a, 100);
		int x = max_move;
		int y = max_move;
		createTrackbar("x-max_move", window_name, &x, 2 * max_move);
		createTrackbar("y-max_move", window_name, &y, 2 * max_move);

		int key = 0;
		Mat show;
		Mat show2;
		while (key != 'q')
		{
			warpShiftSubpix(leftimg, show, (x - max_move)*0.1, (y - max_move)*0.1);
			warpShiftSubpix(rightimg, show2, -(x - max_move)*0.1, -(y - max_move)*0.1);

			showMatInfo(show);
			showMatInfo(show2);
			alphaBlend(show, show2, 0.5, show);

			alphaBlend(show, centerimg, 1.0 - a / 100.0, show);
			imshow(window_name, show);
			key = waitKey(1);

			if (key == 'f')
			{
				a = (a > 0) ? 0 : 100;
				setTrackbarPos("alpha", window_name, a);
			}

			if (key == 'l')
			{
				x--;
				setTrackbarPos("x", window_name, x);
			}
			if (key == 'j')
			{
				x++;
				setTrackbarPos("x", window_name, x);
			}
			if (key == 'i')
			{
				y++;
				setTrackbarPos("y", window_name, y);
			}
			if (key == 'k')
			{
				y--;
				setTrackbarPos("y", window_name, y);
			}
		}
		destroyWindow(window_name);
	}
}