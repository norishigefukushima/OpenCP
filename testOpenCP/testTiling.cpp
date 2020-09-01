#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void createSubImageCV(const Mat& src, Mat& dest, const Size div, const Point index, const int top, const int bottom, const int left, const int right, const int borderType)
{
	const int tile_width = src.cols / div.width;
	const int tile_height = src.rows / div.height;

	Mat im; copyMakeBorder(src, im, top, bottom, left, right, borderType);
	dest.create(Size(tile_width + left + right, tile_height + left + right), src.type());
	Rect roi = Rect(tile_width * index.x, tile_height * index.y, dest.cols, dest.rows);
	im(roi).copyTo(dest);
}

void createSubImageCVAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType, const int align_x, const int align_y, const int left_multiple = 1, const int top_multiple = 1)
{
	const int tilex = src.cols / div_size.width;
	const int tiley = src.rows / div_size.height;

	const int L = get_simd_ceil(r, left_multiple);
	const int T = get_simd_ceil(r, top_multiple);

	const int align_width = get_simd_ceil(tilex + L + r, align_x);
	const int padx = align_width - (tilex + L + r);
	const int align_height = get_simd_ceil(tiley + T + r, align_y);
	const int pady = align_height - (tiley + T + r);
	const int R = r + padx;
	const int B = r + pady;

	createSubImageCV(src, dest, div_size, idx, T, B, L, R, borderType);
}

void testTiling(Mat& src)
{
	Mat srcf = convert(src, CV_32F);
	//Mat srcf = convert(src, CV_64F);

	vector<Mat> v;
	vector<Mat> sv(3);
	vector<Mat> sv2(3);
	split(srcf, v);
	Mat destf = Mat::zeros(src.size(), srcf.type());
	Mat destf_ref = Mat::zeros(src.size(), srcf.type());
	Mat sub_ref, sub, sub_s;
	const Size div = Size(4, 4);

	const int r = 31;
	//int borderType = BORDER_REFLECT101;
	int borderType = BORDER_REFLECT;
	//int borderType = BORDER_REPLICATE;

	for (int j = 0; j < div.height; j++)
	{
		for (int i = 0; i < div.width; i++)
		{
			const Point index = Point(i, j);
			cout << index << endl;;

			//non optilized;
			createSubImageCVAlign(srcf, sub_ref, div, index, r, borderType, 8, 8);

			//split then ...
			for (int c = 0; c < 3; c++)
			{
				sv[c] = Mat::zeros(sv[c].size(), sv[c].type());
			}
			for (int c = 0; c < 3; c++)
				createSubImageAlign(v[c], sv[c], div, index, r, borderType, 8, 8);
			merge(sv, sub);
			cout << "sub  :" << getPSNR(sub_ref, sub, 0, 0) << endl;
			setSubImageAlign(sub, destf_ref, div, index, r);

			//merge split
			for (int c = 0; c < 3; c++)
			{
				sv[c] = Mat::zeros(sv[c].size(), sv[c].type());
			}
			cropSplitTileAlign(srcf, sv2, div, index, r, borderType, 8, 8);
			merge(sv2, sub_s);
			cout << "sub s:" << getPSNR(sub_ref, sub_s, 0, 0) << endl;
			setSubImageAlign(sub_s, destf, div, index, r, 8, 8);

#if 1
			if (i == 0 && j == 0)
			{
				//guiAlphaBlend(sub_s, sub_ref);
				guiDiff(sub_s, sub_ref);
				imshowScale("tile2", destf);
				imshowScale("sub", sub_s);
			}
#endif
		}
	}


	//guiAlphaBlend(destf_ref, destf);
}