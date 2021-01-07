#include "parallel_type.hpp"
#include "guidedFilter.hpp"

#include "guidedFilter_Merge_OnePass.h"
#include "guidedFilter_Merge_forTiling.h"
#include "guidedFilter_Merge.h"

using namespace std;
using namespace cv;

#include<tiling.hpp>
using namespace cp;

#define EXCLUDE_SPLIT_TIME
#define INCLUDE_SPLIT_TIME

GuidedImageFilterTiling::GuidedImageFilterTiling()
{
	gf.resize(OMP_THREADS_MAX);

	buffer.resize(OMP_THREADS_MAX);
	sub_src.resize(OMP_THREADS_MAX);
	sub_guide.resize(OMP_THREADS_MAX);
	sub_guideColor.resize(OMP_THREADS_MAX);
	for (int i = 0; i < OMP_THREADS_MAX; i++)
	{
		buffer[i].resize(3);
		sub_src[i].resize(3);
		sub_guide[i].resize(3);
	}
}

GuidedImageFilterTiling::GuidedImageFilterTiling(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, cv::Size _div)
	: src(_src), guide(_guide), dest(_dest), r(_r), eps(_eps), div(_div)
{
	src_sub_vec.resize(div.area());
	guide_sub_vec.resize(div.area());
	src_sub_temp.resize(3);
	guide_sub_temp.resize(3);
	dest_sub_temp.resize(3);
	guide_sub_temp[0] = Mat::zeros(src.size(), CV_32FC1);
	guide_sub_temp[1] = Mat::zeros(src.size(), CV_32FC1);
	guide_sub_temp[2] = Mat::zeros(src.size(), CV_32FC1);

#ifdef EXCLUDE_SPLIT_TIME
	if (src.channels() == 3)
	{
		split(src, vSrc);
		split(dest, vDest);
	}
	if (guide.channels() == 3) split(guide, vGuide);
#endif

	gf.resize(OMP_THREADS_MAX);
	sub_guideColor.resize(OMP_THREADS_MAX);


	buffer.resize(OMP_THREADS_MAX);
	sub_src.resize(OMP_THREADS_MAX);
	sub_guide.resize(OMP_THREADS_MAX);

	for (int i = 0; i < OMP_THREADS_MAX; i++)
	{
		buffer[i].resize(3);
		sub_src[i].resize(3);
		sub_guide[i].resize(3);
	}
}

void GuidedImageFilterTiling::filter_SSAT()
{
	if (src.channels() == 1 && guide.channels() == 1)
	{
		if (div.area() == 1)
		{
			guidedImageFilter_Merge_Base(src, guide, dest, r, eps, OMP).filter();
		}
		else
		{
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;
				cropTile(src, src_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(guide, guide_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				Mat temp(src_sub_vec[sub_index].size(), CV_32FC1);
				tiling::guidedFilter_Merge_nonVec(src_sub_vec[sub_index], guide_sub_vec[sub_index], temp, r, eps, parallelType).filter();

				pasteTile(temp, dest, div, idx, 2 * r);
			}
		}
	}
	else if (src.channels() == 1 && guide.channels() == 3)
	{
		if (div.area() == 1)
		{
			guidedImageFilter_Merge_Base(src, guide, dest, r, eps, OMP).filter();
		}
		else
		{
#ifdef INCLUDE_SPLIT_TIME
			split(guide, vGuide);
#endif
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;
				cropTile(src, src_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				vector<Mat> guide_sub_temp(3);
				cropTile(vGuide[0], guide_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[1], guide_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[2], guide_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);
				merge(guide_sub_temp, guide_sub_vec[sub_index]);

				Mat temp(src_sub_vec[sub_index].size(), CV_32FC1);
				tiling::guidedFilter_Merge_nonVec(src_sub_vec[sub_index], guide_sub_vec[sub_index], temp, r, eps, parallelType).filter();

				pasteTile(temp, dest, div, idx, 2 * r);
			}
		}
	}
	else if (src.channels() == 3 && guide.channels() == 1)
	{
		if (div.area() == 1)
		{
			guidedImageFilter_Merge_Base(src, guide, dest, r, eps, OMP).filter();
		}
#if 1
		else
		{
#ifdef INCLUDE_SPLIT_TIME
			split(src, vSrc);
			split(dest, vDest);
#endif
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;


				vector<Mat> src_sub_temp(3);
				cropTile(vSrc[0], src_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[1], src_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[2], src_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);

				cropTile(guide, guide_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				vector<Mat> dest_sub_temp(3);
				dest_sub_temp[0] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[1] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[2] = Mat(src_sub_temp[0].size(), CV_32FC1);

				tiling::guidedFilter_Merge_nonVec(src_sub_temp[0], guide_sub_vec[sub_index], dest_sub_temp[0], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_nonVec(src_sub_temp[1], guide_sub_vec[sub_index], dest_sub_temp[1], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_nonVec(src_sub_temp[2], guide_sub_vec[sub_index], dest_sub_temp[2], r, eps, parallelType).filter();

				pasteTile(dest_sub_temp[0], vDest[0], div, idx, 2 * r);
				pasteTile(dest_sub_temp[1], vDest[1], div, idx, 2 * r);
				pasteTile(dest_sub_temp[2], vDest[2], div, idx, 2 * r);
			}
			merge(vDest, dest);
		}
#else
		else
		{

#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				tiling::guidedFilter_Merge_nonVec(src_sub_b[n], guide_sub_vec[n], dest_sub_b[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_nonVec(src_sub_g[n], guide_sub_vec[n], dest_sub_g[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_nonVec(src_sub_r[n], guide_sub_vec[n], dest_sub_r[n], r, eps, parallelType).filter();

			}

		}


#endif
	}
	else if (src.channels() == 3 && guide.channels() == 3)
	{
		if (div.area() == 1)
		{
			guidedImageFilter_Merge_Base(src, guide, dest, r, eps, OMP).filter();
		}
		else
		{
#if 1
#ifdef INCLUDE_SPLIT_TIME
			split(src, vSrc);
			split(guide, vGuide);
			split(dest, vDest);
#endif
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;

				vector<Mat> src_sub_temp(3);
				cropTile(vSrc[0], src_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[1], src_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[2], src_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);

				vector<Mat> guide_sub_temp(3);
				cropTile(vGuide[0], guide_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[1], guide_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[2], guide_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);
				merge(guide_sub_temp, guide_sub_vec[sub_index]);

				vector<Mat> dest_sub_temp(3);
				dest_sub_temp[0] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[1] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[2] = Mat(src_sub_temp[0].size(), CV_32FC1);

				tiling::guidedFilter_Merge_nonVec(src_sub_temp[0], guide_sub_vec[sub_index], dest_sub_temp[0], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_nonVec(src_sub_temp[1], guide_sub_vec[sub_index], dest_sub_temp[1], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_nonVec(src_sub_temp[2], guide_sub_vec[sub_index], dest_sub_temp[2], r, eps, parallelType).filter();


				pasteTile(dest_sub_temp[0], vDest[0], div, idx, 2 * r);
				pasteTile(dest_sub_temp[1], vDest[1], div, idx, 2 * r);
				pasteTile(dest_sub_temp[2], vDest[2], div, idx, 2 * r);
			}
			merge(vDest, dest);
#else
			vector<Mat> vSrc(3);
			split(src, vSrc);

			vector<Mat> src_sub_b;
			vector<Mat> src_sub_g;
			vector<Mat> src_sub_r;
			splitSubImage(vSrc[0], src_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vSrc[1], src_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vSrc[2], src_sub_r, div, 2 * r, BORDER_REPLICATE);

			vector<Mat> vDest(3);
			split(dest, vDest);
			vector<Mat> dest_sub_b;
			vector<Mat> dest_sub_g;
			vector<Mat> dest_sub_r;
			splitSubImage(vDest[0], dest_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vDest[1], dest_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vDest[2], dest_sub_r, div, 2 * r, BORDER_REPLICATE);

			vector<Mat> vGuide;
			split(guide, vGuide);
			vector<Mat> guide_sub_b;
			vector<Mat> guide_sub_g;
			vector<Mat> guide_sub_r;
			splitSubImage(vGuide[0], guide_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vGuide[1], guide_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vGuide[2], guide_sub_r, div, 2 * r, BORDER_REPLICATE);
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				Mat temp(guide_sub_b[0].size(), CV_32FC3);
				vector<Mat> guide_sub(3);
				guide_sub[0] = guide_sub_b[n];
				guide_sub[1] = guide_sub_g[n];
				guide_sub[2] = guide_sub_r[n];
				merge(guide_sub, temp);
				tiling::guidedFilter_Merge_nonVec(src_sub_b[n], temp, dest_sub_b[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_nonVec(src_sub_g[n], temp, dest_sub_g[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_nonVec(src_sub_r[n], temp, dest_sub_r[n], r, eps, parallelType).filter();

			}

			mergeSubImage(dest_sub_b, vDest[0], div, 2 * r);
			mergeSubImage(dest_sub_g, vDest[1], div, 2 * r);
			mergeSubImage(dest_sub_r, vDest[2], div, 2 * r);
			merge(vDest, dest);
#endif
		}
	}

}

void GuidedImageFilterTiling::filter_OPSAT()
{
	if (src.channels() == 1 && guide.channels() == 1)
	{
		if (div.area() == 1)
		{
			guidedFilter_Merge_OnePass(src, guide, dest, r, eps, OMP).filter();
		}
		else
		{
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;
				cropTile(src, src_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(guide, guide_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				Mat temp(src_sub_vec[sub_index].size(), CV_32FC1);
				guidedFilter_Merge_OnePass(src_sub_vec[sub_index], guide_sub_vec[sub_index], temp, r, eps, parallelType).filter();

				pasteTile(temp, dest, div, idx, 2 * r);
			}
		}
	}
	else if (src.channels() == 1 && guide.channels() == 3)
	{
		if (div.area() == 1)
		{
			guidedFilter_Merge_OnePass(src, guide, dest, r, eps, OMP).filter();
		}
		else
		{
#ifdef INCLUDE_SPLIT_TIME
			split(guide, vGuide);
#endif
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;
				cropTileAlign(src, src_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				vector<Mat> guide_sub_temp(3);
				cropTile(vGuide[0], guide_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[1], guide_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[2], guide_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);
				merge(guide_sub_temp, guide_sub_vec[sub_index]);

				Mat temp(src_sub_vec[sub_index].size(), CV_32FC1);
				guidedFilter_Merge_OnePass(src_sub_vec[sub_index], guide_sub_vec[sub_index], temp, r, eps, parallelType).filter();

				pasteTile(temp, dest, div, idx, 2 * r);
			}
		}
	}
	else if (src.channels() == 3 && guide.channels() == 1)
	{
		if (div.area() == 1)
		{
			guidedFilter_Merge_OnePass(src, guide, dest, r, eps, OMP).filter();
		}
		else
		{
#if 1
#ifdef INCLUDE_SPLIT_TIME
			split(src, vSrc);
			split(dest, vDest);
#endif

#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;

				vector<Mat> src_sub_temp(3);
				cropTile(vSrc[0], src_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[1], src_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[2], src_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);

				vector<Mat> dest_sub_temp(3);
				dest_sub_temp[0] = Mat::zeros(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[1] = Mat::zeros(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[2] = Mat::zeros(src_sub_temp[0].size(), CV_32FC1);

				cropTile(guide, guide_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				//Mat temp(src_sub_vec[sub_index].size(), src.type());
				guidedFilter_Merge_OnePass(src_sub_temp[0], guide_sub_vec[sub_index], dest_sub_temp[0], r, eps, parallelType).filter();
				guidedFilter_Merge_OnePass(src_sub_temp[1], guide_sub_vec[sub_index], dest_sub_temp[1], r, eps, parallelType).filter();
				guidedFilter_Merge_OnePass(src_sub_temp[2], guide_sub_vec[sub_index], dest_sub_temp[2], r, eps, parallelType).filter();

				//merge(dest_sub_temp, temp);
				pasteTile(dest_sub_temp[0], vDest[0], div, idx, 2 * r);
				pasteTile(dest_sub_temp[1], vDest[1], div, idx, 2 * r);
				pasteTile(dest_sub_temp[2], vDest[2], div, idx, 2 * r);
			}
			merge(vDest, dest);
#else
			vector<Mat> vSrc(3);
			split(src, vSrc);

			vector<Mat> src_sub_b;
			vector<Mat> src_sub_g;
			vector<Mat> src_sub_r;
			splitSubImage(vSrc[0], src_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vSrc[1], src_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vSrc[2], src_sub_r, div, 2 * r, BORDER_REPLICATE);

			splitSubImage(guide, guide_sub_vec, div, 2 * r, BORDER_REPLICATE);

			vector<Mat> vDest(3);
			split(dest, vDest);
			vector<Mat> dest_sub_b;
			vector<Mat> dest_sub_g;
			vector<Mat> dest_sub_r;
			splitSubImage(vDest[0], dest_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vDest[1], dest_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vDest[2], dest_sub_r, div, 2 * r, BORDER_REPLICATE);


#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				guidedFilter_Merge_OnePass(src_sub_b[n], guide_sub_vec[n], dest_sub_b[n], r, eps, parallelType).filter();
				guidedFilter_Merge_OnePass(src_sub_g[n], guide_sub_vec[n], dest_sub_g[n], r, eps, parallelType).filter();
				guidedFilter_Merge_OnePass(src_sub_r[n], guide_sub_vec[n], dest_sub_r[n], r, eps, parallelType).filter();

			}
			mergeSubImage(dest_sub_b, vDest[0], div, 2 * r);
			mergeSubImage(dest_sub_g, vDest[1], div, 2 * r);
			mergeSubImage(dest_sub_r, vDest[2], div, 2 * r);
			merge(vDest, dest);
#endif
		}
	}
	else if (src.channels() == 3 && guide.channels() == 3)
	{
		if (div.area() == 1)
		{
			guidedFilter_Merge_OnePass(src, guide, dest, r, eps, OMP).filter();
		}
		else
		{
#if 1
#ifdef INCLUDE_SPLIT_TIME
			split(src, vSrc);
			split(guide, vGuide);
			split(dest, vDest);
#endif
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;

				vector<Mat> src_sub_temp(3);
				cropTile(vSrc[0], src_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[1], src_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[2], src_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);

				vector<Mat> guide_sub_temp(3);
				cropTile(vGuide[0], guide_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[1], guide_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[2], guide_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);
				merge(guide_sub_temp, guide_sub_vec[sub_index]);

				vector<Mat> dest_sub_temp(3);
				dest_sub_temp[0] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[1] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[2] = Mat(src_sub_temp[0].size(), CV_32FC1);

				guidedFilter_Merge_OnePass(src_sub_temp[0], guide_sub_vec[sub_index], dest_sub_temp[0], r, eps, parallelType).filter();
				guidedFilter_Merge_OnePass(src_sub_temp[1], guide_sub_vec[sub_index], dest_sub_temp[1], r, eps, parallelType).filter();
				guidedFilter_Merge_OnePass(src_sub_temp[2], guide_sub_vec[sub_index], dest_sub_temp[2], r, eps, parallelType).filter();

				//Mat temp(src_sub_temp[0].size(), src.type());
				//merge(dest_sub_temp, temp);
				pasteTile(dest_sub_temp[0], vDest[0], div, idx, 2 * r);
				pasteTile(dest_sub_temp[1], vDest[1], div, idx, 2 * r);
				pasteTile(dest_sub_temp[2], vDest[2], div, idx, 2 * r);
			}
			merge(vDest, dest);
#else
			vector<Mat> vSrc(3);
			split(src, vSrc);

			vector<Mat> src_sub_b;
			vector<Mat> src_sub_g;
			vector<Mat> src_sub_r;
			splitSubImage(vSrc[0], src_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vSrc[1], src_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vSrc[2], src_sub_r, div, 2 * r, BORDER_REPLICATE);

			vector<Mat> vDest(3);
			split(dest, vDest);
			vector<Mat> dest_sub_b;
			vector<Mat> dest_sub_g;
			vector<Mat> dest_sub_r;
			splitSubImage(vDest[0], dest_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vDest[1], dest_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vDest[2], dest_sub_r, div, 2 * r, BORDER_REPLICATE);

			vector<Mat> vGuide;
			split(guide, vGuide);
			vector<Mat> guide_sub_b;
			vector<Mat> guide_sub_g;
			vector<Mat> guide_sub_r;
			splitSubImage(vGuide[0], guide_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vGuide[1], guide_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vGuide[2], guide_sub_r, div, 2 * r, BORDER_REPLICATE);

#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				Mat temp(guide_sub_b[0].size(), CV_32FC3);
				vector<Mat> guide_sub(3);
				guide_sub[0] = guide_sub_b[n];
				guide_sub[1] = guide_sub_g[n];
				guide_sub[2] = guide_sub_r[n];
				merge(guide_sub, temp);
				guidedFilter_Merge_OnePass(src_sub_b[n], temp, dest_sub_b[n], r, eps, parallelType).filter();
				guidedFilter_Merge_OnePass(src_sub_g[n], temp, dest_sub_g[n], r, eps, parallelType).filter();
				guidedFilter_Merge_OnePass(src_sub_r[n], temp, dest_sub_r[n], r, eps, parallelType).filter();

			}

			mergeSubImage(dest_sub_b, vDest[0], div, 2 * r);
			mergeSubImage(dest_sub_g, vDest[1], div, 2 * r);
			mergeSubImage(dest_sub_r, vDest[2], div, 2 * r);
			merge(vDest, dest);
#endif
		}
	}
}

void GuidedImageFilterTiling::filter_SSAT_AVX()
{
	if (src.channels() == 1 && guide.channels() == 1)
	{
		if (div.area() == 1)
		{
			guidedFilter_Merge_AVX(src, guide, dest, r, eps, OMP).filter();
		}
		else
		{
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;
				cropTileAlign(src, src_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);
				cropTileAlign(guide, guide_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				Mat temp(src_sub_vec[sub_index].size(), CV_32FC1);
				tiling::guidedFilter_Merge_AVX(src_sub_vec[sub_index], guide_sub_vec[sub_index], temp, r, eps, parallelType).filter();

				pasteTile(temp, dest, div, idx, 2 * r);
			}
		}
	}
	else if (src.channels() == 1 && guide.channels() == 3)
	{
		if (div.area() == 1)
		{
			guidedFilter_Merge_AVX(src, guide, dest, r, eps, OMP).filter();
		}
		else
		{
#ifdef INCLUDE_SPLIT_TIME
			split(guide, vGuide);
#endif
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;
				cropTileAlign(src, src_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				vector<Mat> guide_sub_temp(3);
				cropTileAlign(vGuide[0], guide_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTileAlign(vGuide[1], guide_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTileAlign(vGuide[2], guide_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);
				merge(guide_sub_temp, guide_sub_vec[sub_index]);

				Mat temp(src_sub_vec[sub_index].size(), CV_32FC1);
				tiling::guidedFilter_Merge_AVX(src_sub_vec[sub_index], guide_sub_vec[sub_index], temp, r, eps, parallelType).filter();

				pasteTile(temp, dest, div, idx, 2 * r);
			}
		}
	}
	else if (src.channels() == 3 && guide.channels() == 1)
	{
		if (div.area() == 1)
		{
			guidedFilter_Merge_AVX(src, guide, dest, r, eps, OMP).filter();
		}
#if 1
		else
		{
#ifdef INCLUDE_SPLIT_TIME
			split(src, vSrc);
			split(dest, vDest);
#endif

#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;

				vector<Mat> src_sub_temp(3);
				cropTile(vSrc[0], src_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[1], src_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[2], src_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);
				//merge(src_sub_temp, src_sub_vec[sub_index]);
				vector<Mat> dest_sub_temp(3);
				dest_sub_temp[0] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[1] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[2] = Mat(src_sub_temp[0].size(), CV_32FC1);

				cropTile(guide, guide_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				//Mat temp(src_sub_vec[sub_index].size(), src.type());
				tiling::guidedFilter_Merge_AVX(src_sub_temp[0], guide_sub_vec[sub_index], dest_sub_temp[0], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_temp[1], guide_sub_vec[sub_index], dest_sub_temp[1], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_temp[2], guide_sub_vec[sub_index], dest_sub_temp[2], r, eps, parallelType).filter();

				//merge(dest_sub_temp, temp);
				pasteTile(dest_sub_temp[0], vDest[0], div, idx, 2 * r);
				pasteTile(dest_sub_temp[1], vDest[1], div, idx, 2 * r);
				pasteTile(dest_sub_temp[2], vDest[2], div, idx, 2 * r);


			}
			merge(vDest, dest);
		}
#else
		else
		{

#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				tiling::guidedFilter_Merge_AVX(src_sub_b[n], guide_sub_vec[n], dest_sub_b[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_g[n], guide_sub_vec[n], dest_sub_g[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_r[n], guide_sub_vec[n], dest_sub_r[n], r, eps, parallelType).filter();

			}

		}


#endif
	}

	else if (src.channels() == 3 && guide.channels() == 3)
	{
		if (div.area() == 1)
		{
			guidedFilter_Merge_AVX(src, guide, dest, r, eps, OMP).filter();
		}
		else
		{
#if 1
#ifdef INCLUDE_SPLIT_TIME
			split(src, vSrc);
			split(guide, vGuide);
			split(dest, vDest);
#endif
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;

				vector<Mat> src_sub_temp(3);
				cropTile(vSrc[0], src_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[1], src_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[2], src_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);

				vector<Mat> guide_sub_temp(3);
				cropTile(vGuide[0], guide_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[1], guide_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[2], guide_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);
				merge(guide_sub_temp, guide_sub_vec[sub_index]);

				vector<Mat> dest_sub_temp(3);
				dest_sub_temp[0] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[1] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[2] = Mat(src_sub_temp[0].size(), CV_32FC1);

				tiling::guidedFilter_Merge_AVX(src_sub_temp[0], guide_sub_vec[sub_index], dest_sub_temp[0], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_temp[1], guide_sub_vec[sub_index], dest_sub_temp[1], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_temp[2], guide_sub_vec[sub_index], dest_sub_temp[2], r, eps, parallelType).filter();

				//Mat temp(src_sub_temp[0].size(), src.type());
				//merge(dest_sub_temp, temp);
				pasteTile(dest_sub_temp[0], vDest[0], div, idx, 2 * r);
				pasteTile(dest_sub_temp[1], vDest[1], div, idx, 2 * r);
				pasteTile(dest_sub_temp[2], vDest[2], div, idx, 2 * r);
			}
			merge(vDest, dest);
#else
			vector<Mat> vSrc(3);
			split(src, vSrc);

			vector<Mat> src_sub_b;
			vector<Mat> src_sub_g;
			vector<Mat> src_sub_r;
			splitSubImage(vSrc[0], src_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vSrc[1], src_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vSrc[2], src_sub_r, div, 2 * r, BORDER_REPLICATE);

			vector<Mat> vDest(3);
			split(dest, vDest);
			vector<Mat> dest_sub_b;
			vector<Mat> dest_sub_g;
			vector<Mat> dest_sub_r;
			splitSubImage(vDest[0], dest_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vDest[1], dest_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vDest[2], dest_sub_r, div, 2 * r, BORDER_REPLICATE);

			vector<Mat> vGuide;
			split(guide, vGuide);
			vector<Mat> guide_sub_b;
			vector<Mat> guide_sub_g;
			vector<Mat> guide_sub_r;
			splitSubImage(vGuide[0], guide_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vGuide[1], guide_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vGuide[2], guide_sub_r, div, 2 * r, BORDER_REPLICATE);
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				Mat temp(guide_sub_b[0].size(), CV_32FC3);
				vector<Mat> guide_sub(3);
				guide_sub[0] = guide_sub_b[n];
				guide_sub[1] = guide_sub_g[n];
				guide_sub[2] = guide_sub_r[n];
				merge(guide_sub, temp);
				tiling::guidedFilter_Merge_AVX(src_sub_b[n], temp, dest_sub_b[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_g[n], temp, dest_sub_g[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_r[n], temp, dest_sub_r[n], r, eps, parallelType).filter();

			}

			mergeSubImage(dest_sub_b, vDest[0], div, 2 * r);
			mergeSubImage(dest_sub_g, vDest[1], div, 2 * r);
			mergeSubImage(dest_sub_r, vDest[2], div, 2 * r);
			merge(vDest, dest);
#endif
		}
	}
}

void GuidedImageFilterTiling::filter(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, const int _r, const float _eps, const cv::Size _div, const GuidedTypes guidedType)
{
	src = _src;
	guide = _guide;
	if (src.type() != _dest.type())
	{
		_dest.create(src.size(), src.type());
	}
	else if (src.size() != _dest.size())
	{
		_dest.create(src.size(), src.type());
	}

	dest = _dest;
	r = _r;
	eps = _eps;
	div = _div;
	filter(guidedType);
}

void GuidedImageFilterTiling::filter(GuidedTypes guidedType)
{
	if (src.channels() == 1 && guide.channels() == 1)
	{
#pragma omp parallel for
		for (int n = 0; n < div.area(); n++)
		{
			const int tidx = omp_get_thread_num();

			const int j = n / div.width;
			const int i = n % div.width;
			Point idx = Point(i, j);

			cropTileAlign(src, sub_src[tidx][0], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(guide, sub_guide[tidx][0], div, idx, 2 * r, BORDER_REPLICATE);

			gf[tidx].filter(sub_src[tidx][0], sub_guide[tidx][0], sub_src[tidx][0], r, eps, guidedType, ParallelTypes::NAIVE);

			pasteTile(sub_src[tidx][0], dest, div, idx, 2 * r);
		}
	}
	else if (src.channels() == 1 && guide.channels() == 3)
	{
#ifdef INCLUDE_SPLIT_TIME
		cv::split(guide, vGuide);
#endif
#pragma omp parallel for
		for (int n = 0; n < div.area(); n++)
		{
			const int tidx = omp_get_thread_num();

			const int j = n / div.width;
			const int i = n % div.width;
			const Point idx = Point(i, j);

			cropTileAlign(src, sub_src[tidx][0], div, idx, 2 * r, BORDER_REPLICATE);

			cropTileAlign(vGuide[0], sub_guide[tidx][0], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(vGuide[1], sub_guide[tidx][1], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(vGuide[2], sub_guide[tidx][2], div, idx, 2 * r, BORDER_REPLICATE);

			//cv::merge(sub_guide[tidx], sub_guideColor[tidx]);
			//gf[tidx].filter(sub_src[tidx][0], sub_guideColor[tidx], sub_src[tidx][0], r, eps, guidedType, ParallelTypes::NAIVE);

			gf[tidx].filter(sub_src[tidx][0], sub_guide[tidx], sub_src[tidx][0], r, eps, guidedType, ParallelTypes::NAIVE);

			pasteTile(sub_src[tidx][0], dest, div, idx, 2 * r);
		}
	}
	else if (src.channels() == 3 && guide.channels() == 1)
	{
#ifdef INCLUDE_SPLIT_TIME
		split(src, vSrc);
		split(dest, vDest);
#endif
#pragma omp parallel for
		for (int n = 0; n < div.area(); n++)
		{
			const int tidx = omp_get_thread_num();

			const int j = n / div.width;
			const int i = n % div.width;
			const Point idx = Point(i, j);

			cropTileAlign(vSrc[0], sub_src[tidx][0], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(vSrc[1], sub_src[tidx][1], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(vSrc[2], sub_src[tidx][2], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(guide, sub_guide[tidx][0], div, idx, 2 * r, BORDER_REPLICATE);

			//merge(sub_src[tidx], sub_guideColor[tidx]);

			gf[tidx].filter(sub_src[tidx], sub_guide[tidx][0], sub_src[tidx], r, eps, guidedType, ParallelTypes::NAIVE);

			pasteTile(sub_src[tidx][0], vDest[0], div, idx, 2 * r);
			pasteTile(sub_src[tidx][1], vDest[1], div, idx, 2 * r);
			pasteTile(sub_src[tidx][2], vDest[2], div, idx, 2 * r);
		}
		merge(vDest, dest);
	}
	else if (src.channels() == 3 && guide.channels() == 3)
	{
#ifdef INCLUDE_SPLIT_TIME
		split(src, vSrc);
		split(dest, vDest);
		cv::split(guide, vGuide);
#endif
#pragma omp parallel for
		for (int n = 0; n < div.area(); n++)
		{
			const int tidx = omp_get_thread_num();

			const int j = n / div.width;
			const int i = n % div.width;
			const Point idx = Point(i, j);

			cropTileAlign(vSrc[0], sub_src[tidx][0], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(vSrc[1], sub_src[tidx][1], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(vSrc[2], sub_src[tidx][2], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(vGuide[0], sub_guide[tidx][0], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(vGuide[1], sub_guide[tidx][1], div, idx, 2 * r, BORDER_REPLICATE);
			cropTileAlign(vGuide[2], sub_guide[tidx][2], div, idx, 2 * r, BORDER_REPLICATE);
			
			//merge(sub_src[tidx], sub_guideColor[tidx]);

			gf[tidx].filter(sub_src[tidx], sub_guide[tidx], sub_src[tidx], r, eps, guidedType, ParallelTypes::NAIVE);

			pasteTile(sub_src[tidx][0], vDest[0], div, idx, 2 * r);
			pasteTile(sub_src[tidx][1], vDest[1], div, idx, 2 * r);
			pasteTile(sub_src[tidx][2], vDest[2], div, idx, 2 * r);
		}
		merge(vDest, dest);
	}
}

void GuidedImageFilterTiling::filter_func(GuidedTypes guidedType)
{
	if (div.area() == 1)
	{
		
		guidedImageFilter(src, guide, dest, r, eps, guidedType, BoxFilterMethod::OPENCV, ParallelTypes::NAIVE);
	}
	else
	{
		if (src.channels() == 1 && guide.channels() == 1)
		{
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;
				cropTileAlign(src, src_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);
				cropTileAlign(guide, guide_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				Mat temp(src_sub_vec[sub_index].size(), CV_32FC1);
				guidedImageFilter(src_sub_vec[sub_index], guide_sub_vec[sub_index], temp, r, eps, guidedType, BoxFilterMethod::OPENCV, ParallelTypes::NAIVE);

				pasteTile(temp, dest, div, idx, 2 * r);
			}
		}
		else if (src.channels() == 1 && guide.channels() == 3)
		{
#ifdef INCLUDE_SPLIT_TIME
			split(guide, vGuide);
#endif
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;
				cropTileAlign(src, src_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				vector<Mat> guide_sub_temp(3);
				cropTileAlign(vGuide[0], guide_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTileAlign(vGuide[1], guide_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTileAlign(vGuide[2], guide_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);
				merge(guide_sub_temp, guide_sub_vec[sub_index]);

				Mat temp(src_sub_vec[sub_index].size(), CV_32FC1);
				guidedImageFilter(src_sub_vec[sub_index], guide_sub_vec[sub_index], temp, r, eps, guidedType, BoxFilterMethod::OPENCV, ParallelTypes::NAIVE);

				pasteTile(temp, dest, div, idx, 2 * r);
			}
		}
		else if (src.channels() == 3 && guide.channels() == 1)
		{

#if 1
#ifdef INCLUDE_SPLIT_TIME
			split(src, vSrc);
			split(dest, vDest);
#endif

#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;

				vector<Mat> src_sub_temp(3);
				cropTile(vSrc[0], src_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[1], src_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[2], src_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);
				//merge(src_sub_temp, src_sub_vec[sub_index]);
				vector<Mat> dest_sub_temp(3);
				dest_sub_temp[0] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[1] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[2] = Mat(src_sub_temp[0].size(), CV_32FC1);

				cropTile(guide, guide_sub_vec[sub_index], div, idx, 2 * r, BORDER_REPLICATE);

				//Mat temp(src_sub_vec[sub_index].size(), src.type());
				tiling::guidedFilter_Merge_AVX(src_sub_temp[0], guide_sub_vec[sub_index], dest_sub_temp[0], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_temp[1], guide_sub_vec[sub_index], dest_sub_temp[1], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_temp[2], guide_sub_vec[sub_index], dest_sub_temp[2], r, eps, parallelType).filter();

				//merge(dest_sub_temp, temp);
				pasteTile(dest_sub_temp[0], vDest[0], div, idx, 2 * r);
				pasteTile(dest_sub_temp[1], vDest[1], div, idx, 2 * r);
				pasteTile(dest_sub_temp[2], vDest[2], div, idx, 2 * r);
			}
			merge(vDest, dest);
#else
		else
		{

#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				tiling::guidedFilter_Merge_AVX(src_sub_b[n], guide_sub_vec[n], dest_sub_b[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_g[n], guide_sub_vec[n], dest_sub_g[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_r[n], guide_sub_vec[n], dest_sub_r[n], r, eps, parallelType).filter();

			}

		}
#endif
		}

		else if (src.channels() == 3 && guide.channels() == 3)
		{
#if 1
#ifdef INCLUDE_SPLIT_TIME
			split(src, vSrc);
			split(guide, vGuide);
			split(dest, vDest);
#endif
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				const int j = n / div.width;
				const int i = n % div.width;
				Point idx = Point(i, j);
				int sub_index = div.width*j + i;

				vector<Mat> src_sub_temp(3);
				cropTile(vSrc[0], src_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[1], src_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vSrc[2], src_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);

				vector<Mat> guide_sub_temp(3);
				cropTile(vGuide[0], guide_sub_temp[0], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[1], guide_sub_temp[1], div, idx, 2 * r, BORDER_REPLICATE);
				cropTile(vGuide[2], guide_sub_temp[2], div, idx, 2 * r, BORDER_REPLICATE);
				merge(guide_sub_temp, guide_sub_vec[sub_index]);

				vector<Mat> dest_sub_temp(3);
				dest_sub_temp[0] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[1] = Mat(src_sub_temp[0].size(), CV_32FC1);
				dest_sub_temp[2] = Mat(src_sub_temp[0].size(), CV_32FC1);

				tiling::guidedFilter_Merge_AVX(src_sub_temp[0], guide_sub_vec[sub_index], dest_sub_temp[0], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_temp[1], guide_sub_vec[sub_index], dest_sub_temp[1], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_temp[2], guide_sub_vec[sub_index], dest_sub_temp[2], r, eps, parallelType).filter();

				//Mat temp(src_sub_temp[0].size(), src.type());
				//merge(dest_sub_temp, temp);
				pasteTile(dest_sub_temp[0], vDest[0], div, idx, 2 * r);
				pasteTile(dest_sub_temp[1], vDest[1], div, idx, 2 * r);
				pasteTile(dest_sub_temp[2], vDest[2], div, idx, 2 * r);
			}
			merge(vDest, dest);
#else
			vector<Mat> vSrc(3);
			split(src, vSrc);

			vector<Mat> src_sub_b;
			vector<Mat> src_sub_g;
			vector<Mat> src_sub_r;
			splitSubImage(vSrc[0], src_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vSrc[1], src_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vSrc[2], src_sub_r, div, 2 * r, BORDER_REPLICATE);

			vector<Mat> vDest(3);
			split(dest, vDest);
			vector<Mat> dest_sub_b;
			vector<Mat> dest_sub_g;
			vector<Mat> dest_sub_r;
			splitSubImage(vDest[0], dest_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vDest[1], dest_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vDest[2], dest_sub_r, div, 2 * r, BORDER_REPLICATE);

			vector<Mat> vGuide;
			split(guide, vGuide);
			vector<Mat> guide_sub_b;
			vector<Mat> guide_sub_g;
			vector<Mat> guide_sub_r;
			splitSubImage(vGuide[0], guide_sub_b, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vGuide[1], guide_sub_g, div, 2 * r, BORDER_REPLICATE);
			splitSubImage(vGuide[2], guide_sub_r, div, 2 * r, BORDER_REPLICATE);
#pragma omp parallel for
			for (int n = 0; n < div.area(); n++)
			{
				Mat temp(guide_sub_b[0].size(), CV_32FC3);
				vector<Mat> guide_sub(3);
				guide_sub[0] = guide_sub_b[n];
				guide_sub[1] = guide_sub_g[n];
				guide_sub[2] = guide_sub_r[n];
				merge(guide_sub, temp);
				tiling::guidedFilter_Merge_AVX(src_sub_b[n], temp, dest_sub_b[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_g[n], temp, dest_sub_g[n], r, eps, parallelType).filter();
				tiling::guidedFilter_Merge_AVX(src_sub_r[n], temp, dest_sub_r[n], r, eps, parallelType).filter();

			}

			mergeSubImage(dest_sub_b, vDest[0], div, 2 * r);
			mergeSubImage(dest_sub_g, vDest[1], div, 2 * r);
			mergeSubImage(dest_sub_r, vDest[2], div, 2 * r);
			merge(vDest, dest);
#endif
		}
	}
}



guidedFilter_tiling_noMakeBorder::guidedFilter_tiling_noMakeBorder(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, cv::Size _div)
	: src(_src), guide(_guide), dest(_dest), r(_r), eps(_eps), div(_div)
{
	init();
}

void guidedFilter_tiling_noMakeBorder::init()
{
	padRow = r + r;
	padCol = r + r;

	CV_Assert(src.cols % div.width == 0 && src.rows % div.height == 0);
	divCol = src.cols / div.width;
	divRow = src.rows / div.height;

	int divNum = div.area();
	divSrc.resize(divNum);
	divGuide.resize(divNum);
	divDest.resize(divNum);

	if (div.width == 1)
	{
		divDest[0].create(Size(divCol, divRow + padRow), src.type());
		int i = 1;
		for (; i < divNum - 1; i++)
		{
			divDest[i].create(Size(divCol, divRow + padRow + padRow), src.type());
		}
		divDest[i].create(Size(divCol, divRow + padRow), src.type());
	}
	else
	{
		divDest[0].create(Size(divCol + padCol, divRow + padRow), src.type());
		int i = 1;
		for (; i < div.width - 1; i++)
		{
			divDest[i].create(Size(divCol + padCol + padCol, divRow + padRow), src.type());
		}
		divDest[i].create(Size(divCol + padCol, divRow + padRow), src.type());
		i++;
		//上から２番目から下から２番目までの行のタイル
		for (; i < (div.width*(div.height - 1)); i++)
		{
			if (i%div.width == 0 || i % div.width == div.width - 1)
			{
				divDest[i].create(Size(divCol + padCol, divRow + padRow + padRow), src.type());
			}
			else
			{
				divDest[i].create(Size(divCol + padCol + padCol, divRow + padRow + padRow), src.type());
			}
		}
		//一番下の行のタイル
		divDest[i].create(Size(divCol + padCol, divRow + padRow), src.type());
		i++;
		for (; i < div.height*div.width - 1; i++)
		{
			divDest[i].create(Size(divCol + padCol + padCol, divRow + padRow), src.type());
		}
		divDest[i].create(Size(divCol + padCol, divRow + padRow), src.type());

	}
}

void guidedFilter_tiling_noMakeBorder::splitImage()
{
	vector<Rect> roi(divSrc.size());

	if (div.width == 1)
	{
		roi[0] = Rect(0, 0, divCol, divRow + padRow);
		int i = 1;
		for (; i < div.area() - 1; i++)
		{
			roi[i] = Rect(0, divRow*i - padRow, divCol, divRow + padRow + padRow);
		}
		roi[i] = Rect(0, divRow*i - padRow, divCol, divRow + padRow);
	}
	else
	{
		roi[0] = Rect(0, 0, divCol + padCol, divRow + padRow);
		int i = 1;
		for (; i < div.width - 1; i++)
		{
			roi[i] = Rect(divCol*i - padCol, 0, divCol + padCol + padCol, divRow + padRow);
		}
		roi[i] = Rect(divCol*i - padCol, 0, divCol + padCol, divRow + padRow);
		i++;
		for (; i < (div.width*(div.height - 1)); i++)
		{
			if (i%div.width == 0)
			{
				roi[i] = Rect(0, divRow*(i / div.width) - padRow, divCol + padCol, divRow + padRow + padRow);
			}
			else if (i%div.width == div.width - 1)
			{
				roi[i] = Rect(divCol*(i%div.width) - padCol, divRow*(i / div.width) - padRow, divCol + padCol, divRow + padRow + padRow);
			}
			else
			{
				roi[i] = Rect(divCol*(i%div.width) - padCol, divRow*(i / div.width) - padRow, divCol + padCol + padCol, divRow + padRow + padRow);
			}
		}
		roi[i] = Rect(0, divRow*(div.height - 1) - padRow, divCol + padCol, divRow + padRow);
		i++;
		for (; i < div.height*div.width - 1; i++)
		{
			roi[i] = Rect(divCol*(i%div.width) - padCol, divRow*(div.height - 1) - padRow, divCol + padCol + padCol, divRow + padRow);
		}
		roi[i] = Rect(divCol*(div.width - 1) - padCol, divRow*(div.height - 1) - padRow, divCol + padCol, divRow + padRow);
	}

#pragma omp parallel for
	for (int i = 0; i < divSrc.size(); i++)
	{
		src(roi[i]).copyTo(divSrc[i]);
		guide(roi[i]).copyTo(divGuide[i]);
	}

	//for (int i = 0; i < divSrc.size(); i++)
	//{
	//	Mat temp;
	//	divSrc[i].convertTo(temp, CV_8U);
	//	imshow("rect", temp);
	//	waitKey(100);
	//}

}

void guidedFilter_tiling_noMakeBorder::mergeImage()
{
	if (div.width == 1)
	{
		divDest[0](Rect(0, 0, divCol, divRow)).copyTo(dest(Rect(0, 0, divCol, divRow)));
		int i = 1;
		for (; i < div.area(); i++)
		{
			divDest[i](Rect(0, padRow, divCol, divRow)).copyTo(dest(Rect(0, divRow*i, divCol, divRow)));
		}
	}
	else
	{
		divDest[0](Rect(0, 0, divCol, divRow)).copyTo(dest(Rect(0, 0, divCol, divRow)));
		int i = 1;
		for (; i < div.width; i++)
		{
			divDest[i](Rect(padCol, 0, divCol, divRow)).copyTo(dest(Rect(divCol*i, 0, divCol, divRow)));
		}
		for (; i < div.area(); i++)
		{
			if (i%div.width == 0)
			{
				divDest[i](Rect(0, padRow, divCol, divRow)).copyTo(dest(Rect(0, divRow*(i / div.width), divCol, divRow)));
			}
			else
			{
				divDest[i](Rect(padCol, padRow, divCol, divRow)).copyTo(dest(Rect(divCol*(i%div.width), divRow*(i / div.width), divCol, divRow)));
			}
		}
	}
}

void guidedFilter_tiling_noMakeBorder::filter()
{
	splitImage();
	for (int i = 0; i < divSrc.size(); i++)
		guidedFilter_Merge_OnePass(divSrc[i], divGuide[i], divDest[i], r, eps, OMP).filter();
	//guidedFilter_Merge_nonVec(divSrc[i], divGuide[i], divDest[i], r, eps, NAIVE).filter();


	mergeImage();
}