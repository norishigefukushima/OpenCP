#include "crop.hpp"
#include "draw.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	void cropCenter(InputArray src, OutputArray crop, const int window_size)
	{
		Mat s = src.getMat();
		cropZoom(src, crop, Point(s.cols / 2, s.rows / 2), window_size);
	}

	void cropZoom(InputArray src, OutputArray crop_zoom, const Rect roi, const int zoom_factor)
	{
		const int w = roi.width;
		const int h = roi.height;
		Mat bb;
		copyMakeBorder(src, bb, h, h, w, w, BORDER_CONSTANT, 0);
		Mat cropimage;
		bb(Rect(roi.x + w, roi.y + h, roi.width, roi.height)).copyTo(cropimage);

		resize(cropimage, crop_zoom, Size(zoom_factor*w, zoom_factor*h), 0, 0, INTER_NEAREST);
	}

	void cropZoom(InputArray src, OutputArray crop_zoom, const Point center, const int window_size, const int zoom_factor)
	{
		Rect roi = Rect(center.x - window_size / 2, center.y - window_size / 2, window_size, window_size);
		cropZoom(src, crop_zoom, roi, zoom_factor);
	}

	void cropZoomWithBoundingBox(InputArray src, OutputArray crop_zoom, const Rect roi, const int zoom_factor, const Scalar color, const int thickness)
	{
		const int w = roi.width;
		const int h = roi.height;
		Mat bb;
		copyMakeBorder(src, bb, h, h, w, w, BORDER_CONSTANT, 0);
		Mat cropimage;
		bb(Rect(roi.x + w, roi.y + h, roi.width, roi.height)).copyTo(cropimage);
		rectangle(cropimage, Rect(0, 0, w, h), color, thickness);

		resize(cropimage, crop_zoom, Size(zoom_factor*w, zoom_factor*h), 0, 0, INTER_NEAREST);
	}

	void cropZoomWithBoundingBox(InputArray src, OutputArray crop_zoom, const Point center, const int window_size, const int zoom_factor, const Scalar color, const int thickness)
	{
		Rect roi = Rect(center.x - window_size / 2, center.y - window_size / 2, window_size, window_size);
		cropZoomWithBoundingBox(src, crop_zoom, roi, zoom_factor, color, thickness);
	}

	void cropZoomWithSrcMarkAndBoundingBox(InputArray src, OutputArray crop_zoom, OutputArray src_mark, const Rect roi, const int zoom_factor, const Scalar color, const int thickness)
	{
		Mat temp;
		src.copyTo(temp);
		rectangle(temp, roi, color, thickness);
		temp.copyTo(src_mark);

		cropZoomWithBoundingBox(src, crop_zoom, roi, zoom_factor, color, thickness);
	}

	void cropZoomWithSrcMarkAndBoundingBox(InputArray src, OutputArray crop_zoom, OutputArray src_mark, const Point center, const int window_size, const int zoom_factor, const Scalar color, const int thickness)
	{
		Rect roi = Rect(center.x - window_size / 2, center.y - window_size / 2, window_size, window_size);
		cropZoomWithSrcMarkAndBoundingBox(src, crop_zoom, src_mark, roi, zoom_factor, color, thickness);
	}

	struct MouseParameterGUICropZoom
	{
		cv::Rect pt;
		std::string wname;
	};

	void onMouseGUICropZoom(int events, int x, int y, int flags, void *param)
	{
		MouseParameterGUICropZoom* retp = (MouseParameterGUICropZoom*)param;
		//if(events==CV_EVENT_LBUTTONDOWN)
		if (flags & EVENT_FLAG_LBUTTON)
		{
			retp->pt.x = max(0, min(retp->pt.width - 1, x));
			retp->pt.y = max(0, min(retp->pt.height - 1, y));

			setTrackbarPos("zoom_x", retp->wname, x);
			setTrackbarPos("zoom_y", retp->wname, y);
		}
	}

	cv::Mat guiCropZoom(InputArray src, Rect& dest_roi, int& dest_zoom_factor, const Scalar color, const int thickness, const bool isWait, const string wname)
	{
		const int zoom_factor_max = 32;

		const int width = src.size().width;
		const int height = src.size().height;

		static MouseParameterGUICropZoom param
		{
			Rect(width / 2, height / 2, width, height),
			wname
		};

		namedWindow(wname);
		setMouseCallback(wname, onMouseGUICropZoom, (void*)&param);

		static int zoom_show_option = 0; createTrackbar("zoom_show_op", wname, &zoom_show_option, 1);
		static int zoom_position = 0; createTrackbar("zoom_position", wname, &zoom_position, 4);
		createTrackbar("zoom_x", wname, &param.pt.x, width - 1);
		createTrackbar("zoom_y", wname, &param.pt.y, height - 1);
		static int zoom_count = 0;
		static int zoom_window = 40; createTrackbar("zoom_window", wname, &zoom_window, min(width, height) - 1);
		static int zoom_factor = 8; createTrackbar("zoom_factor", wname, &zoom_factor, zoom_factor_max);
		static int thick = thickness; createTrackbar("thickness", wname, &thick, 10);
		Mat show;
		Mat crop_resize;
		Mat input = src.getMat();

		int key = 0;
		displayOverlay(wname, "s: save, p: change position, ?: help", 5000);

		bool isScale = false;
		if (input.depth() == CV_32F)
		{
			double minv, maxv;
			cv::minMaxLoc(input, &minv, &maxv);
			if (maxv > 1)isScale = true;
		}
		while (key != 'q')
		{
			if(isScale) input.convertTo(show, CV_8U, 255);
			else input.copyTo(show);
				
			

			zoom_factor = max(zoom_factor, 1);

			if (zoom_show_option == 0)
			{
				cropZoomWithSrcMarkAndBoundingBox(input, crop_resize, show, Point(param.pt.x, param.pt.y), zoom_window, zoom_factor, color, thick);
			}
			else if (zoom_show_option == 1)
			{
				cropZoom(input, crop_resize, Point(param.pt.x, param.pt.y), zoom_window, zoom_factor);
			}

			imshow(wname + "_image", crop_resize);

			if (crop_resize.cols < width&&crop_resize.rows < height)
			{
				if (zoom_position == 1) crop_resize.copyTo(show(Rect(0, 0, crop_resize.size().width, crop_resize.size().height)));
				else if (zoom_position == 2) crop_resize.copyTo(show(Rect(show.cols - 1 - crop_resize.size().width, 0, crop_resize.size().width, crop_resize.size().height)));
				else if (zoom_position == 3) crop_resize.copyTo(show(Rect(show.cols - 1 - crop_resize.size().width, show.rows - 1 - crop_resize.size().height, crop_resize.size().width, crop_resize.size().height)));
				else if (zoom_position == 4) crop_resize.copyTo(show(Rect(0, show.rows - 1 - crop_resize.size().height, crop_resize.size().width, crop_resize.size().height)));
			}

			imshow(wname, show);
			if (!isWait) break;
			key = waitKey(1);
			if (key == 'i')
			{
				param.pt.y = max(param.pt.y - 1, 0);
				setTrackbarPos("zoom_y", wname, param.pt.x);
			}
			if (key == 'j')
			{
				param.pt.x = max(param.pt.x - 1, 0);
				setTrackbarPos("zoom_x", wname, param.pt.x);
			}
			if (key == 'k')
			{
				param.pt.x = min(param.pt.y + 1, height - 1);
				setTrackbarPos("zoom_y", wname, param.pt.y);
			}
			if (key == 'l')
			{
				param.pt.x = min(param.pt.x + 1, width - 1);
				setTrackbarPos("zoom_x", wname, param.pt.x);
			}
			if (key == 'z')
			{
				zoom_factor = min(zoom_factor + 1, zoom_factor_max);
				setTrackbarPos("zoom_factor", wname, zoom_factor);
			}
			if (key == 'x')
			{
				zoom_factor = max(zoom_factor - 1, 1);
				setTrackbarPos("zoom_factor", wname, zoom_factor);
			}
			if (key == 'p')
			{
				zoom_position++;
				zoom_position = (zoom_position > 4) ? 0 : zoom_position;
				setTrackbarPos("zoom_position", wname, zoom_position);
			}
			if (key == 's')
			{
				imwrite(wname + to_string(zoom_count) + "_mark.png", show);
				imwrite(wname + to_string(zoom_count) + ".png", crop_resize);
				zoom_count++;
			}
			if (key == '?')
			{
				cout << "i,j,k,l: move x, y, like vim" << endl;
				cout << "z: zoom factor++" << endl;
				cout << "x: zoom factor--" << endl;
				cout << "p: change position of zoomed image" << endl;
				cout << "s: save image" << endl;
				cout << "q: quit" << endl;
			}
		}

		if (isWait) destroyWindow(wname);
		return crop_resize;
	}

	cv::Mat guiCropZoom(InputArray src, const Scalar color, const int thickness, const bool isWait, const string wname)
	{
		Rect roi;
		int zf;
		return guiCropZoom(src, roi, zf, color, thickness, isWait, wname);
	}
}