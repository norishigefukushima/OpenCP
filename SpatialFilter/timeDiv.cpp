#include "stdafx.h"

using namespace std;
using namespace cv;

void TileDivision::update_pt()
{
	for (int i = 0; i < div.width; i++) pt[i].y = 0;
	pt[0].x = 0;
	for (int i = 1; i < div.width; i++)
		pt[i].x = pt[i - 1].x + tileSize[i - 1].width;

	for (int j = 1; j < div.height; j++)
	{
		pt[div.width * j].x = 0;
		pt[div.width * j].y = pt[div.width * (j - 1)].y + tileSize[div.width * (j - 1)].height;
		for (int i = 1; i < div.width; i++)
		{
			const int ptidx = div.width * (j - 1) + i;
			const int tidx = div.width * j + i;
			pt[tidx].x = pt[tidx - 1].x + tileSize[tidx - 1].width;
			pt[tidx].y = pt[ptidx].y + tileSize[ptidx].height;
		}
	}
}

//div.width * y + x;
cv::Rect TileDivision::getROI(const int x, int y)
{
	const int tindex = div.width * y + x;
	return getROI(tindex);
}

cv::Rect TileDivision::getROI(const int index)
{
	return cv::Rect(pt[index].x, pt[index].y, tileSize[index].width, tileSize[index].height);
}

TileDivision::TileDivision(cv::Size imgSize, cv::Size div) : div(div), imgSize(imgSize)
{
	this->div.width = std::max(div.width, 1);
	this->div.height = std::max(div.height, 1);

	tileSize.resize(this->div.area());
	pt.resize(this->div.area());
}

bool TileDivision::compute(const int width_step_, const int height_step_)
{
	width_step = width_step_;
	height_step = height_step_;

	const int xmulti = width_step * div.width;
	const int ymulti = width_step * div.height;
	const int x_base_width = get_simd_floor(imgSize.width, xmulti);
	const int x_base_height = get_simd_floor(imgSize.height, ymulti);
	const int x_rem = imgSize.width - x_base_width;
	const int y_rem = imgSize.height - x_base_height;

	const int x_base_tilewidth = x_base_width / div.width;
	const int y_base_tileheight = x_base_height / div.height;

	const int num_base_padtile_x = x_rem / width_step;
	const int num_base_padtile_y = y_rem / height_step;

	const int num_last_padtile_x = x_rem - num_base_padtile_x * width_step;
	const int num_last_padtile_y = y_rem - num_base_padtile_y * height_step;

	for (int i = 0; i < div.area(); i++)
	{
		tileSize[i].width = x_base_tilewidth;
	}
	if (num_last_padtile_x == 0)
	{
		for (int j = 0; j < div.height; j++)
		{
			for (int i = 0; i < num_base_padtile_x; i++)
			{
				const int tidx = div.width * j + (div.width - 1 - i);
				tileSize[tidx].width = x_base_tilewidth + width_step;
			}
		}
	}
	else
	{
		for (int j = 0; j < div.height; j++)
		{
			for (int i = 0; i < num_base_padtile_x; i++)
			{
				const int tidx = div.width * j + (div.width - 2 - i);
				tileSize[tidx].width = x_base_tilewidth + width_step;
			}
			tileSize[div.width * j + (div.width - 1)].width = x_base_tilewidth + num_last_padtile_x;
		}
	}

	for (int i = 0; i < div.area(); i++)
	{
		tileSize[i].height = y_base_tileheight;
	}
	if (num_last_padtile_y == 0)
	{
		for (int i = 0; i < div.width; i++)
		{
			for (int j = 0; j < num_base_padtile_y; j++)
			{
				const int tidx = div.width * (div.height - 1 - j) + i;
				tileSize[tidx].height = y_base_tileheight + height_step;
			}
		}
	}
	else
	{
		for (int i = 0; i < div.width; i++)
		{
			for (int j = 0; j < num_base_padtile_y; j++)
			{
				const int tidx = div.width * (div.height - 2 - j) + i;
				tileSize[tidx].height = y_base_tileheight + height_step;
			}
			tileSize[div.width * (div.height - 1) + i].height = y_base_tileheight + num_last_padtile_y;
		}
	}

	update_pt();
	threadnum.resize(div.area());
	#pragma omp parallel for schedule (static)
	for (int i = 0; i < div.area(); i++)
	{
		threadnum[i] = omp_get_thread_num();
	}
	
	return (num_last_padtile_x == 0 && num_last_padtile_y == 0);
}

void TileDivision::draw(cv::Mat& src, cv::Mat& dst)
{
	if (src.data == dst.data)
	{
		dst = src;
	}
	else
	{
		dst.create(src.size(), CV_8UC3);
		switch (src.type())
		{
		case CV_8UC3: dst = src.clone(); break;
		case CV_8UC1: cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR); break;

		default:
		{
			if (src.channels() == 3)
			{
				dst = cp::convert(src, CV_8U); break;
			}
			else
			{
				cv::Mat temp = cp::convert(src, CV_8U);
				cv::cvtColor(temp, dst, cv::COLOR_GRAY2BGR); break;
			}
			break;
		}
		}
	}

	const int threadMax = omp_get_max_threads();
	
	for (int i = 0; i < div.area(); i++)
	{	
		const int threadNumber = threadnum[i];
		//const int threadNumber = 0;
		cv::Rect roi = getROI(i);
		if (tileSize[i].width % width_step == 0 && tileSize[i].height % height_step == 0)
		{
			rectangle(dst, roi, COLOR_GREEN);		
		}
		else
		{
			rectangle(dst, roi, COLOR_YELLOW);
		}
		if (pt[i].x + tileSize[i].width > imgSize.width || pt[i].y + tileSize[i].height > imgSize.height)
		{
			rectangle(dst, roi, COLOR_RED);
		}
		const Scalar txtcolor = COLOR_WHITE;
		//const Scalar txtcolor = COLOR_GREEN;
		addText(dst, std::to_string(tileSize[i].width) + "x" + std::to_string(tileSize[i].height), pt[i] + cv::Point(10, 20), "Consolas", 12, txtcolor);
		addText(dst, cv::format("#T %d/%d", threadNumber, threadMax - 1), pt[i] + cv::Point(10, 40), "Consolas", 12, txtcolor);
		
		//cout << roi << endl;
	}
}

void TileDivision::show(std::string wname)
{
	cv::Mat show = cv::Mat::zeros(imgSize, CV_8UC3);
	draw(show, show);
	imshow(wname, show);
}
