#include "guidedFilter_Div.h"
#include "guidedFilter_Merge_OnePass.h"

using namespace std;
using namespace cv;
using namespace cp;

/*
 *	--- divType ---
 *	0			1			2			3			4		5
 *	┌───┐　┌─┬─┐　┌┬┬┬┐　┌─┬─┐	2*6		2*18
 *	│　　　│　│　│　│　│││││　├─┼─┤
 *	├───┤　├─┼─┤　├┼┼┼┤　├─┼─┤
 *	│　　　│　│　│　│　│││││　├─┼─┤
 *	└───┘　└─┴─┘　└┴┴┴┘　└─┴─┘
 */

const int tate[] = { 0,0,0,4,6,18,36 };



guidedFilter_Div::guidedFilter_Div(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType, int _divType)
	: src(_src), guide(_guide), dest(_dest), r(_r), eps(_eps), parallelType(_parallelType), divType(_divType)
{
	init();
}

void guidedFilter_Div::init()
{
	padRow = r + r;
	padCol = r + r;

	if (divType == 0)
	{
		divRow = src.rows / 2;
		divCol = src.cols;

		divSrc.resize(2);
		divGuide.resize(2);
		divDest.resize(2);

		for (int i = 0; i < 2; i++)
		{
			divDest[i].create(Size(divCol, divRow + padRow), src.type());
		}
	}
	else if (divType == 1)
	{
		divRow = src.rows / 2;
		divCol = src.cols / 2;

		divSrc.resize(4);
		divGuide.resize(4);
		divDest.resize(4);

		for (int i = 0; i < 4; i++)
		{
			divDest[i].create(Size(divCol + padCol, divRow + padRow), src.type());
		}
	}
	else if (divType == 2)
	{
		divRow = src.rows / 2;
		divCol = src.cols / 4;

		divSrc.resize(8);
		divGuide.resize(8);
		divDest.resize(8);

		divDest[0].create(Size(divCol + padCol, divRow + padRow), src.type());
		divDest[1].create(Size(divCol + padCol + padCol, divRow + padRow), src.type());
		divDest[2].create(Size(divCol + padCol + padCol, divRow + padRow), src.type());
		divDest[3].create(Size(divCol + padCol, divRow + padRow), src.type());
		divDest[4].create(Size(divCol + padCol, divRow + padRow), src.type());
		divDest[5].create(Size(divCol + padCol + padCol, divRow + padRow), src.type());
		divDest[6].create(Size(divCol + padCol + padCol, divRow + padRow), src.type());
		divDest[7].create(Size(divCol + padCol, divRow + padRow), src.type());
	}
	else if (divType >= 3)
	{
		divRow = src.rows / tate[divType];
		divCol = src.cols / 2;

		divSrc.resize(tate[divType] * 2);
		divGuide.resize(tate[divType] * 2);
		divDest.resize(tate[divType] * 2);

		divDest[0].create(Size(divCol + padCol, divRow + padRow), src.type());
		divDest[1].create(Size(divCol + padCol, divRow + padRow), src.type());
#pragma omp parallel for
		for (int i = 2; i < divSrc.size() - 2; i++)
		{
			divDest[i].create(Size(divCol + padCol, divRow + padRow + padRow), src.type());
		}
		divDest[divSrc.size() - 2].create(Size(divCol + padCol, src.rows - (divRow * (tate[divType] - 1)) + padRow), src.type());
		divDest[divSrc.size() - 1].create(Size(divCol + padCol, src.rows - (divRow * (tate[divType] - 1)) + padRow), src.type());
	}
}

void guidedFilter_Div::splitImage()
{
	vector<Rect> roi(divSrc.size());
	if (divType == 0)
	{
		roi[0] = Rect(0, 0, divCol, divRow + padRow);
		roi[1] = Rect(0, divRow - padRow, divCol, divRow + padRow);
	}
	else if (divType == 1)
	{
		roi[0] = Rect(0, 0, divCol + padCol, divRow + padRow);
		roi[1] = Rect(divCol - padCol, 0, divCol + padCol, divRow + padRow);
		roi[2] = Rect(0, divRow - padRow, divCol + padCol, divRow + padRow);
		roi[3] = Rect(divCol - padCol, divRow - padRow, divCol + padCol, divRow + padRow);
	}
	else if (divType == 2)
	{
		roi[0] = Rect(0, 0, divCol + padCol, divRow + padRow);
		roi[1] = Rect(divCol - padCol, 0, divCol + padCol + padCol, divRow + padRow);
		roi[2] = Rect(divCol * 2 - padCol, 0, divCol + padCol + padCol, divRow + padRow);
		roi[3] = Rect(divCol * 3 - padCol, 0, divCol + padCol, divRow + padRow);
		roi[4] = Rect(0, divRow - padRow, divCol + padCol, divRow + padRow);
		roi[5] = Rect(divCol - padCol, divRow - padRow, divCol + padCol + padCol, divRow + padRow);
		roi[6] = Rect(divCol * 2 - padCol, divRow - padRow, divCol + padCol + padCol, divRow + padRow);
		roi[7] = Rect(divCol * 3 - padCol, divRow - padRow, divCol + padCol, divRow + padRow);
	}
	else if (divType >= 3)
	{
		roi[0] = Rect(0, 0, divCol + padCol, divRow + padRow);
		roi[1] = Rect(divCol - padCol, 0, divCol + padCol, divRow + padRow);
#pragma omp parallel for
		for (int i = 1; i < tate[divType] - 1; i++)
		{
			roi[i * 2] = Rect(0, divRow * i - padRow, divCol + padCol, divRow + padRow + padRow);
			roi[i * 2 + 1] = Rect(divCol - padCol, divRow * i - padRow, divCol + padCol, divRow + padRow + padRow);
		}
		roi[divSrc.size() - 2] = Rect(0, divRow * (tate[divType] - 1) - padRow, divCol + padCol, src.rows - (divRow * (tate[divType] - 1)) + padRow);
		roi[divSrc.size() - 1] = Rect(divCol - padCol, divRow * (tate[divType] - 1) - padRow, divCol + padCol, src.rows - (divRow * (tate[divType] - 1)) + padRow);
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
	//	waitKey(1000);
	//}
}

void guidedFilter_Div::mergeImage()
{
	if (divType == 0)
	{
		divDest[0](Rect(0, 0, divCol, divRow)).copyTo(dest(Rect(0, 0, divCol, divRow)));
		divDest[1](Rect(0, padRow, divCol, divRow)).copyTo(dest(Rect(0, divRow, divCol, divRow)));
	}
	else if (divType == 1)
	{
		divDest[0](Rect(0, 0, divCol, divRow)).copyTo(dest(Rect(0, 0, divCol, divRow)));
		divDest[1](Rect(padCol, 0, divCol, divRow)).copyTo(dest(Rect(divCol, 0, divCol, divRow)));
		divDest[2](Rect(0, padRow, divCol, divRow)).copyTo(dest(Rect(0, divRow, divCol, divRow)));
		divDest[3](Rect(padCol, padRow, divCol, divRow)).copyTo(dest(Rect(divCol, divRow, divCol, divRow)));
	}
	else if (divType == 2)
	{
		divDest[0](Rect(0, 0, divCol, divRow)).copyTo(dest(Rect(0, 0, divCol, divRow)));
		divDest[1](Rect(padCol, 0, divCol, divRow)).copyTo(dest(Rect(divCol, 0, divCol, divRow)));
		divDest[2](Rect(padCol, 0, divCol, divRow)).copyTo(dest(Rect(divCol * 2, 0, divCol, divRow)));
		divDest[3](Rect(padCol, 0, divCol, divRow)).copyTo(dest(Rect(divCol * 3, 0, divCol, divRow)));
		divDest[4](Rect(0, padRow, divCol, divRow)).copyTo(dest(Rect(0, divRow, divCol, divRow)));
		divDest[5](Rect(padCol, padRow, divCol, divRow)).copyTo(dest(Rect(divCol, divRow, divCol, divRow)));
		divDest[6](Rect(padCol, padRow, divCol, divRow)).copyTo(dest(Rect(divCol * 2, divRow, divCol, divRow)));
		divDest[7](Rect(padCol, padRow, divCol, divRow)).copyTo(dest(Rect(divCol * 3, divRow, divCol, divRow)));
	}
	else if (divType >= 3)
	{
		divDest[0](Rect(0, 0, divCol, divRow)).copyTo(dest(Rect(0, 0, divCol, divRow)));
		divDest[1](Rect(padCol, 0, divCol, divRow)).copyTo(dest(Rect(divCol, 0, divCol, divRow)));
#pragma omp parallel for
		for (int i = 1; i < tate[divType] - 1; i++)
		{
			divDest[i * 2](Rect(0, padRow, divCol, divRow)).copyTo(dest(Rect(0, divRow * i, divCol, divRow)));
			divDest[i * 2 + 1](Rect(padCol, padRow, divCol, divRow)).copyTo(dest(Rect(divCol, divRow * i, divCol, divRow)));
		}
		divDest[divDest.size() - 2](Rect(0, padRow, divCol, divDest[divDest.size() - 2].rows - padRow)).copyTo(dest(Rect(0, divRow * (tate[divType] - 1), divCol, divDest[divDest.size() - 2].rows - padRow)));
		divDest[divDest.size() - 1](Rect(padCol, padRow, divCol, divDest[divDest.size() - 1].rows - padRow)).copyTo(dest(Rect(divCol, divRow * (tate[divType] - 1), divCol, divDest[divDest.size() - 1].rows - padRow)));
	}
}

void guidedFilter_Div::filter()
{
	splitImage();

	if (parallelType == NAIVE)
	{
		for (int i = 0; i < divSrc.size(); i++)
			guidedFilter_Merge_OnePass(divSrc[i], divGuide[i], divDest[i], r, eps, NAIVE).filter();
	}
	else if (parallelType == OMP)
	{
		// 空間並列
		int cn = src.channels();
#pragma omp parallel for
		for (int i = 0; i < divSrc.size(); i++)
			guidedFilter_Merge_OnePass(divSrc[i], divGuide[i], divDest[i], r, eps, NAIVE).filter();
	}
	else if (parallelType == PARALLEL_FOR_)
	{
		// チャネル・空間並列
		if (src.channels() == 1)
		{
			if (guide.channels() == 1)
			{
#pragma omp parallel for
				for (int i = 0; i < divSrc.size(); i++)
					guidedFilter_Merge_OnePass(divSrc[i], divGuide[i], divDest[i], r, eps, NAIVE).filter_Guide1(0);
			}
			else if (guide.channels() == 3)
			{
#pragma omp parallel for
				for (int i = 0; i < divSrc.size(); i++)
					guidedFilter_Merge_OnePass(divSrc[i], divGuide[i], divDest[i], r, eps, NAIVE).filter_Guide3(0);
			}
		}
		else if (src.channels() == 3)
		{
			if (guide.channels() == 1)
			{
#pragma omp parallel for
				for (int i = 0; i < divSrc.size() * 3; i++)
					guidedFilter_Merge_OnePass(divSrc[i / 3], divGuide[i / 3], divDest[i / 3], r, eps, NAIVE).filter_Guide1(i % 3);
			}
			else if (guide.channels() == 3)
			{
#pragma omp parallel for
				for (int i = 0; i < divSrc.size() * 3; i++)
					guidedFilter_Merge_OnePass(divSrc[i / 3], divGuide[i / 3], divDest[i / 3], r, eps, NAIVE).filter_Guide3(i % 3);
			}
		}
	}

	mergeImage();
}