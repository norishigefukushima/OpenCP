#include "concat.hpp"
#include "debugcp.hpp"
using namespace std;
using namespace cv;

namespace cp
{

	void concat(cv::InputArrayOfArrays src, cv::OutputArray dest, const int tile_cols, const int tile_rows)
	{
		CV_Assert(!src.empty());

		vector<Mat> v;
		src.getMatVector(v);
		if (v.size() == 1)
		{
			v[0].copyTo(dest);
			return;
		}

		for (int i = 0; i < (int)v.size(); i++)
		{
			CV_Assert(v[0].cols == v[i].cols);
			CV_Assert(v[0].rows == v[i].rows);
		}

		if (tile_rows == 1)
		{
			hconcat(src, dest);
		}
		else if (tile_cols == 1)
		{
			vconcat(src, dest);
		}
		else
		{
			CV_Assert(v.size() <= tile_cols * tile_rows);
			vector<Mat> vc(tile_rows);
			vector<Mat> hc(tile_cols);
			for (int j = 0; j < tile_rows; j++)
			{
				for (int i = 0; i < tile_cols; i++)
				{
					const int index = j * tile_cols + i;
					if (index < v.size()) hc[i] = v[index];
					else hc[i] = Mat::zeros(v[0].size(), v[0].type());
				}
				hconcat(hc, vc[j]);
			}
			vconcat(vc, dest);
		}
	}

	void concatMerge(cv::InputArrayOfArrays src, cv::OutputArray dest, const int tile_cols, const int tile_rows)
	{
		concat(src, dest, tile_cols, tile_rows);
	}

	void concatSplit(cv::InputArray src, cv::OutputArrayOfArrays dest, const cv::Size image_size)
	{
		const int tile_cols = src.size().width / image_size.width;
		const int tile_rows = src.size().height / image_size.height;
		const int size = tile_cols * tile_rows;

		dest.create(size, 1, src.type());

		for (int i = 0; i < size; i++)
		{
			Mat temp;
			concatExtract(src, image_size, temp, i);
			dest.getMatRef(i) = temp;
		}
	}

	void concatSplit(cv::InputArray src, cv::OutputArrayOfArrays dest, const int tile_cols, const int tile_rows)
	{
		const cv::Size image_size(src.size().width / tile_cols, src.size().height / tile_rows);
		concatSplit(src, dest, image_size);
	}

	void concatExtract(cv::InputArray src, cv::OutputArray dest, const int tile_col_index, const int tile_row_index)
	{
		CV_Assert(!dest.empty());
		concatExtract(src, dest.size(), dest, tile_col_index, tile_row_index);
	}

	void concatExtract(cv::InputArray src, cv::Size size, cv::OutputArray dest, const int tile_col_index, const int tile_row_index)
	{
		Mat s = src.getMat();

		const int twidth = s.cols / size.width;
		const int theight = s.rows / size.height;

		CV_Assert(s.size().area() == Size(size.width * twidth, size.height * theight).area());
		int tile_index = twidth * tile_row_index + tile_col_index;

		int twi = tile_col_index;
		int thi = tile_row_index;
		if (twidth <= tile_col_index && tile_row_index == 0)
		{
			twi = tile_col_index % twidth;
			thi = tile_col_index / twidth;
		}
		s(Rect(size.width * twi, size.height * thi, size.width, size.height)).copyTo(dest);
	}

}