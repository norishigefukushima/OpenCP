#pragma once

#include "boxFilter.hpp"

//one pass box filtering AoS

class boxFilter_OPSAT_AoS
{
protected:
	cv::Mat src;
	cv::Mat dest;
	int r;
	int parallelType;

	float div;
	int row;
	int col;
	int cn;
	int loop;

	virtual void filter_impl(int cnNum);
public:
	boxFilter_OPSAT_AoS(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
	{
		div = 1.f / ((2 * r + 1)*(2 * r + 1));
		row = src.rows;
		col = src.cols;
		cn = src.channels();

		init();
	}
	virtual void init()
	{
		loop = cn;		
	}
	void filter()
	{
		if (parallelType == ParallelTypes::NAIVE)
		{
			for (int i = 1; i <= loop; i++)
			{
				filter_impl(i - 1);
			}
		}
		else if (parallelType == ParallelTypes::OMP)
		{
#pragma omp parallel for
			for (int i = 1; i <= loop; i++)
			{
				filter_impl(i - 1);
			}
		}
		else if (parallelType == PARALLEL_FOR_)
		{
#pragma omp parallel sections
			{
#pragma omp section
				{
					for (int i = 0; i < loop / 8; i++)
						filter_impl(i);
				}
#pragma omp section
				{
					for (int i = loop / 8; i < loop / 4; i++)
						filter_impl(i);
				}
#pragma omp section
				{
					for (int i = loop / 4; i < loop / 8 * 3; i++)
						filter_impl(i);
				}
#pragma omp section
				{
					for (int i = loop / 8 * 3; i < loop / 2; i++)
						filter_impl(i);
				}
#pragma omp section
				{
					for (int i = loop / 2; i < loop / 8 * 5; i++)
						filter_impl(i);
				}
#pragma omp section
				{
					for (int i = loop / 8 * 5; i < loop / 4 * 3; i++)
						filter_impl(i);
				}
#pragma omp section
				{
					for (int i = loop / 4 * 3; i < loop / 8 * 7; i++)
						filter_impl(i);
				}
#pragma omp section
				{
					for (int i = loop / 8 * 7; i < loop; i++)
						filter_impl(i);
				}
			}
		}
	}
};


class boxFilter_OPSAT_AoS_SSE : public boxFilter_OPSAT_AoS
{
private:
	__m128 mDiv;
	__m128 mBorder;

	void filter_impl(int cnNum) override;
public:
	boxFilter_OPSAT_AoS_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_OPSAT_AoS(_src, _dest, _r, _parallelType)
	{
		init();
	}
	void init() override
	{
		loop = cn / 4;

		mDiv = _mm_set1_ps(div);
		mBorder = _mm_set1_ps(static_cast<float>(r + 1));
	}
};

class boxFilter_OPSAT_AoS_AVX : public boxFilter_OPSAT_AoS
{
private:
	__m256 mDiv;
	__m256 mBorder;

	void filter_impl(int cnNum) override;
public:
	boxFilter_OPSAT_AoS_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_OPSAT_AoS(_src, _dest, _r, _parallelType)
	{
		init();
	}
	void init() override
	{
		loop = cn / 8;

		mDiv = _mm256_set1_ps(div);
		mBorder = _mm256_set1_ps(static_cast<float>(r + 1));
	}
};




// 3channel loop unroll
class boxFilter_OPSAT_BGR
{
private:
	cv::Mat src;
	cv::Mat temp;
	cv::Mat dest;
	int r;
	int parallelType;

	float div;
	int row;
	int col;
	int cn;

	__m128 mBorder;
	__m128 mDiv;

	void filter_impl();
public:
	boxFilter_OPSAT_BGR(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
	{
		div = 1.f / ((2 * r + 1)*(2 * r + 1));
		row = src.rows;
		col = src.cols;
		cn = src.channels();

		mBorder = _mm_set1_ps(static_cast<float>(r + 1));
		mDiv = _mm_set1_ps(div);

		temp.create(src.rows, src.cols + 1, CV_32FC3);
	}

	void filter()
	{
		if (parallelType == ParallelTypes::NAIVE)
		{
			filter_impl();
		}
	}
};



class boxFilter_OPSAT_BGRA
{
private:
	cv::Mat src;
	cv::Mat srcBGRA;
	cv::Mat destBGRA;
	cv::Mat dest;
	int r;
	int parallelType;

	float div;
	int row;
	int col;
	int cn;

	__m128 mBorder;
	__m128 mDiv;

	void filter_impl();
public:
	boxFilter_OPSAT_BGRA(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
	{
		div = 1.f / ((2 * r + 1)*(2 * r + 1));
		row = src.rows;
		col = src.cols;
		cn = src.channels();

		mBorder = _mm_set1_ps(static_cast<float>(r + 1));
		mDiv = _mm_set1_ps(div);

		srcBGRA.create(src.size(), CV_32FC4);
		destBGRA.create(src.size(), CV_32FC4);
	}

	void filter()
	{
		if (parallelType == ParallelTypes::NAIVE)
		{
			filter_impl();
		}
	}
};
