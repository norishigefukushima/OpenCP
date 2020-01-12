#pragma once

#include "boxFilter_Base.h"

/*
 * SSAT SoA(SAoS) Space Parallel
 */
class boxFilter_SSAT_SoA_Space_nonVec
{
protected:
	cv::Mat src;
	cv::Mat dest;
	int r;
	int parallelType;

	int cn;
	int loop;

	std::vector<cv::Mat> vSrc;
	std::vector<cv::Mat> vDest;
	std::vector<cv::Mat> vTemp;

	virtual void filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp);

public:
	boxFilter_SSAT_SoA_Space_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void filter();
	void filterOnly();
	virtual void AoS2SoA();
	virtual void SoA2AoS();

	virtual void init();
};

class boxFilter_SSAT_SoA_Space_SSE : public boxFilter_SSAT_SoA_Space_nonVec
{
private:
	void filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp) override;
public:
	boxFilter_SSAT_SoA_Space_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void AoS2SoA() override;
	void SoA2AoS() override;
	void init() override;
};

class boxFilter_SSAT_SoA_Space_AVX : public boxFilter_SSAT_SoA_Space_nonVec
{
private:
	void filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp) override;
public:
	boxFilter_SSAT_SoA_Space_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void AoS2SoA() override;
	void SoA2AoS() override;
	void init() override;
};



/*
 * SSAT SoA(SAoS) Channel Parallel
 */
class boxFilter_SSAT_SoA_Channel_nonVec
{
protected:
	cv::Mat src;
	cv::Mat dest;
	int r;
	int parallelType;

	int cn;
	int loop;

	std::vector<cv::Mat> vSrc;
	std::vector<cv::Mat> vDest;
	std::vector<cv::Mat> vTemp;

	virtual void filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp);

public:
	boxFilter_SSAT_SoA_Channel_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void filter();
	void filterOnly();
	virtual void AoS2SoA();
	virtual void SoA2AoS();

	virtual void init();
};

class boxFilter_SSAT_SoA_Channel_SSE : public boxFilter_SSAT_SoA_Channel_nonVec
{
private:
	void filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp) override;
public:
	boxFilter_SSAT_SoA_Channel_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void AoS2SoA() override;
	void SoA2AoS() override;
	void init() override;
};

class boxFilter_SSAT_SoA_Channel_AVX : public boxFilter_SSAT_SoA_Channel_nonVec
{
private:
	void filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp) override;
public:
	boxFilter_SSAT_SoA_Channel_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void AoS2SoA() override;
	void SoA2AoS() override;
	void init() override;
};
