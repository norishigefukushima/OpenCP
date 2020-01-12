#pragma once

#include "boxFilter_Base.h"

//one pass box filtering SoA

class boxFilter_OPSAT_SoA
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

	std::vector<cv::Mat> vSrc;
	std::vector<cv::Mat> vDest;

	virtual void filter_impl(cv::Mat& input, cv::Mat& output);
public:
	boxFilter_OPSAT_SoA(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
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

		vSrc.resize(loop);
		vDest.resize(loop);
		for (int i = 0; i < loop; i++)
		{
			vSrc[i].create(src.size(), CV_32FC1);
			vDest[i].create(src.size(), CV_32FC1);
		}
	}

	virtual void AoS2SoA();
	virtual void SoA2AoS();

	void filter()
	{
		AoS2SoA();
		if (parallelType == ParallelTypes::NAIVE)
		{
			for (int i = 0; i < loop; i++)
				filter_impl(vSrc[i], vDest[i]);
		}
		else if (parallelType == ParallelTypes::OMP)
		{
#pragma omp parallel for
			for (int i = 0; i < loop; i++)
				filter_impl(vSrc[i], vDest[i]);
		}
		SoA2AoS();
	}

	void filterOnly()
	{
		if (parallelType == ParallelTypes::NAIVE)
		{
			for (int i = 0; i < loop; i++)
				filter_impl(vSrc[i], vDest[i]);
		}
		else if (parallelType == ParallelTypes::OMP)
		{
#pragma omp parallel for
			for (int i = 0; i < loop; i++)
				filter_impl(vSrc[i], vDest[i]);
		}
	}
};

class boxFilter_OPSAT_SoA_SSE : public boxFilter_OPSAT_SoA
{
private:
	__m128 mDiv;
	__m128 mBorder;

	void filter_impl(cv::Mat& input, cv::Mat& output) override;
public:
	boxFilter_OPSAT_SoA_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_OPSAT_SoA(_src, _dest, _r, _parallelType)
	{
		init();
	}
	void init() override
	{
		mDiv = _mm_set1_ps(div);
		mBorder = _mm_set1_ps(static_cast<float>(r + 1));
		loop = cn >> 2;

		vSrc.resize(loop);
		vDest.resize(loop);
		for (int i = 0; i < loop; i++)
		{
			vSrc[i].create(src.size(), CV_32FC4);
			vDest[i].create(src.size(), CV_32FC4);
		}
	}
	void AoS2SoA() override;
	void SoA2AoS() override;
};

class boxFilter_OPSAT_SoA_AVX : public boxFilter_OPSAT_SoA
{
private:
	__m256 mDiv;
	__m256 mBorder;

	void filter_impl(cv::Mat& input, cv::Mat& output) override;
public:
	boxFilter_OPSAT_SoA_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_OPSAT_SoA(_src, _dest, _r, _parallelType)
	{
		init();
	}
	void init() override
	{
		mDiv = _mm256_set1_ps(div);
		mBorder = _mm256_set1_ps(static_cast<float>(r + 1));
		loop = cn >> 3;

		vSrc.resize(loop);
		vDest.resize(loop);
		for (int i = 0; i < loop; i++)
		{
			vSrc[i].create(src.size(), CV_32FC(8));
			vDest[i].create(src.size(), CV_32FC(8));
		}
	}
	void AoS2SoA() override;
	void SoA2AoS() override;
};
