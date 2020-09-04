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

void testTilingTime(Mat& src, const int r = 32, const Size div = Size(4, 4), const Point index = Point(0, 0), const int borderType = BORDER_DEFAULT)
{
	const int iteration = 10000;
	vector<Mat> vsrc;
	split(src, vsrc);
	vector<Mat> vtile(3);
	vector<Mat> vtiles(3);
	Mat tile_refcv, tile, tile_s;
	{
		Timer t("cv x100    ", TIME_MSEC);
		for (int i = 0; i < iteration / 100; i++)
		{
			createSubImageCVAlign(src, tile_refcv, div, index, r, borderType, 8, 8);
			//split(tile_refcv, vtile);
		}
	}
	{
		Timer t("outer split", TIME_MSEC);
		for (int i = 0; i < iteration; i++)
		{
			for (int c = 0; c < 3; c++)
				cropTileAlign(vsrc[c], vtile[c], div, index, r, borderType, 8, 8);
		}
	}
	{
		Timer t("inner split", TIME_MSEC);
		for (int i = 0; i < iteration; i++)
		{
			cropSplitTileAlign(src, vtiles, div, index, r, borderType, 8, 8);
		}
	}
}

void testTilingAccuracy(Mat& src, bool isPrint = false, const int r = 32, const Size div = Size(4, 4), const int borderType = BORDER_DEFAULT)
{
	vector<Mat> vsrc;
	vector<Mat> sv(3);
	vector<Mat> sv2(3);
	split(src, vsrc);

	Mat dest = src.clone();
	Mat dest_tile = Mat::zeros(src.size(), src.type());
	Mat dest_tile_s = Mat::zeros(src.size(), src.type());
	
	Mat tile_refcv, tile, tile_s;
	double psnr = 0.0;
	bool isOK = true;
	for (int j = 0; j < div.height; j++)
	{
		for (int i = 0; i < div.width; i++)
		{
			const Point index = Point(i, j);
			if (isPrint)cout << index << endl;

			//non optilized;
			createSubImageCVAlign(src, tile_refcv, div, index, r, borderType, 8, 8);

			//split then ...
			for (int c = 0; c < 3; c++)
			{
				sv[c] = Mat::zeros(sv[c].size(), src.depth());
			}
			for (int c = 0; c < 3; c++)
				cropTileAlign(vsrc[c], sv[c], div, index, r, borderType, 8, 8);
			merge(sv, tile);
			psnr = getPSNR(tile_refcv, tile, 0, 0);
			if (psnr != 0)isOK = false;
			if (isPrint)cout << "sub  :" << psnr << endl;
			pasteTileAlign(tile, dest_tile, div, index, r);

#if 0
			if (i == 3 && j == 0)
			{
				//guiAlphaBlend(tile_s, tile_refcv);
				guiDiff(tile, tile_refcv);
				imshowScale("tile2", destf_tile_s);
				imshowScale("sub", tile_s);
			}
#endif

			//merge split
			for (int c = 0; c < 3; c++)
			{
				sv2[c] = Mat::zeros(sv[c].size(), src.depth());
			}
			cropSplitTileAlign(src, sv2, div, index, r, borderType, 8, 8);
			merge(sv2, tile_s);
			psnr = getPSNR(tile_refcv, tile_s, 0, 0);
			if (psnr != 0)isOK = false;
			if (isPrint)cout << "sub s:" << psnr << endl;
			pasteTileAlign(tile_s, dest_tile_s, div, index, r);

#if 0
			if (i == 3 && j == 0)
			{
				//guiAlphaBlend(tile_s, tile_refcv);
				guiDiff(tile_s, tile_refcv);
				imshowScale("tile2", destf_tile_s);
				imshowScale("sub", tile_s);
			}
#endif
		}
	}

	if (isOK)cout << "crop OK: ";
	else cout << "crop NG: ";
	double psnr1 = getPSNR(dest_tile, dest, 0, 0);
	double psnr2 = getPSNR(dest_tile_s, dest, 0, 0);
	if (psnr1 != 0)isOK = false;
	if (psnr2 != 0)isOK = false;
	if (isOK)cout << "set OK: " << endl;
	else
	{
		cout << "set NG: " << endl;
		guiAlphaBlend(dest_tile_s, dest);
	}
	if (isPrint)
	{
		cout << "merge sub   :" << psnr1 << endl;
		cout << "merge sub s :" << psnr2 << endl;
	}
}

void testTiling(Mat& src)
{
	int r = 31;
	Size div = Size(4, 4);
	Mat src32f = convert(src, CV_32F);
	Mat src64f = convert(src, CV_64F);

	const bool is8U = true;
	const bool is32F = true;
	const bool is64F = true;
	const bool isTimer = false;
	const bool isAccur = true;
	if (is8U)
	{
		cout << "CV_8U: " << r << ", tile" << div << ", BORDER_REPLICATE" << endl;
		if (isAccur)testTilingAccuracy(src, false, r, div, cv::BORDER_REPLICATE);
		if (isTimer)testTilingTime(src, r, div, Point(0, 0), cv::BORDER_REPLICATE);

		cout << "CV_8U: " << r << ", tile" << div << ", BORDER_REFLECT101" << endl;
		if (isAccur)testTilingAccuracy(src, false, r, div, cv::BORDER_REFLECT101);
		if (isTimer)testTilingTime(src, r, div, Point(0, 0), cv::BORDER_REFLECT101);

		cout << "CV_8U: " << r << ", tile" << div << ", BORDER_REFLECT" << endl;
		if (isAccur)testTilingAccuracy(src, false, r, div, cv::BORDER_REFLECT);
		if (isTimer)testTilingTime(src, r, div, Point(0, 0), cv::BORDER_REFLECT);

		cout << endl;
	}
	if (is32F)
	{
		cout << "CV_32F: " << r << ", tile" << div << ", BORDER_REPLICATE" << endl;
		if (isAccur)testTilingAccuracy(src32f, false, r, div, cv::BORDER_REPLICATE);
		if (isTimer)testTilingTime(src32f, r, div, Point(0, 0), cv::BORDER_REPLICATE);

		cout << "CV_32F: " << r << ", tile" << div << ", BORDER_REFLECT101" << endl;
		if (isAccur)testTilingAccuracy(src32f, false, r, div, cv::BORDER_REFLECT101);
		if (isTimer)testTilingTime(src32f, r, div, Point(0, 0), cv::BORDER_REFLECT101);

		cout << "CV_32F: " << r << ", tile" << div << ", BORDER_REFLECT" << endl;
		if (isAccur)testTilingAccuracy(src32f, false, r, div, cv::BORDER_REFLECT);
		if (isTimer)testTilingTime(src32f, r, div, Point(0, 0), cv::BORDER_REFLECT);

		cout << endl;
	}
	if (is64F)
	{
		cout << "CV_64F: " << r << ", tile" << div << ", BORDER_REPLICATE" << endl;
		if (isAccur)testTilingAccuracy(src64f, false, r, div, cv::BORDER_REPLICATE);
		if (isTimer)testTilingTime(src64f, r, div, Point(0, 0), cv::BORDER_REPLICATE);

		cout << "CV_64F: " << r << ", tile" << div << ", BORDER_REFLECT101" << endl;
		if (isAccur)testTilingAccuracy(src64f, false, r, div, cv::BORDER_REFLECT101);
		if (isTimer)testTilingTime(src64f, r, div, Point(0, 0), cv::BORDER_REFLECT101);

		cout << "CV_64F: " << r << ", tile" << div << ", BORDER_REFLECT" << endl;
		if (isAccur)testTilingAccuracy(src64f, false, r, div, cv::BORDER_REFLECT);
		if (isTimer)testTilingTime(src64f, r, div, Point(0, 0), cv::BORDER_REFLECT);

		cout << endl;
	}
}