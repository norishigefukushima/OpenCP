#include "boxFilter_SSAT_SoA.h"
#include "boxFilter_SSAT_HV.h"

using namespace std;
using namespace cv;

boxFilter_SSAT_SoA_Space_nonVec::boxFilter_SSAT_SoA_Space_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	cn = src.channels();

	init();
}

void boxFilter_SSAT_SoA_Space_nonVec::init()
{
	loop = cn;

	vSrc.resize(loop);
	vDest.resize(loop);
	vTemp.resize(loop);
	for (int i = 0; i < loop; i++)
	{
		vSrc[i].create(src.size(), CV_32FC1);
		vDest[i].create(src.size(), CV_32FC1);
		vTemp[i].create(src.size(), CV_32FC1);
	}
}

void boxFilter_SSAT_SoA_Space_nonVec::filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp)
{
	RowSumFilter(input, temp, r, parallelType).filter();
	ColumnSumFilter_nonVec(temp, output, r, parallelType).filter();
}

void boxFilter_SSAT_SoA_Space_nonVec::filter()
{
	AoS2SoA();
	for (int i = 0; i < loop; i++)
		filter_impl(vSrc[i], vDest[i], vTemp[i]);
	SoA2AoS();
}

void boxFilter_SSAT_SoA_Space_nonVec::filterOnly()
{
	for (int i = 0; i < loop; i++)
		filter_impl(vSrc[i], vDest[i], vTemp[i]);
}

void boxFilter_SSAT_SoA_Space_nonVec::AoS2SoA()
{
#pragma omp parallel for
	for (int i = 0; i < src.rows; i++)
	{
		float* sp = src.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* dp = vSrc[k].ptr<float>(i) + j;

				*dp = *sp;
				sp++;
			}
		}
	}
}

void boxFilter_SSAT_SoA_Space_nonVec::SoA2AoS()
{
#pragma omp parallel for
	for (int i = 0; i < vDest[0].rows; i++)
	{
		float* dp = dest.ptr<float>(i);

		for (int j = 0; j < vDest[0].cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* sp = vDest[k].ptr<float>(i) + j;

				*dp = *sp;
				dp++;
			}
		}
	}
}



boxFilter_SSAT_SoA_Space_SSE::boxFilter_SSAT_SoA_Space_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_SSAT_SoA_Space_nonVec(_src, _dest, _r, _parallelType)
{
	init();
}

void boxFilter_SSAT_SoA_Space_SSE::init()
{
	//loop = cn;

	//vSrc.resize(loop);
	//vDest.resize(loop);
	//vTemp.resize(loop);
	//for (int i = 0; i < loop; i++)
	//{
	//	vSrc[i].create(src.size(), CV_32FC1);
	//	vDest[i].create(src.size(), CV_32FC1);
	//	vTemp[i].create(src.size(), CV_32FC1);
	//}

	loop = cn >> 2;

	vSrc.resize(loop);
	vDest.resize(loop);
	vTemp.resize(loop);
	for (int i = 0; i < loop; i++)
	{
		vSrc[i].create(src.size(), CV_32FC4);
		vDest[i].create(src.size(), CV_32FC4);
		vTemp[i].create(src.size(), CV_32FC4);
	}
}

void boxFilter_SSAT_SoA_Space_SSE::filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp)
{
	RowSumFilter_SSE(input, temp, r, parallelType).filter();
	ColumnSumFilter_SSE(temp, output, r, parallelType).filter();
}

void boxFilter_SSAT_SoA_Space_SSE::AoS2SoA()
{
	//split(src, vSrc);

#pragma omp parallel for
	for (int i = 0; i < src.rows; i++)
	{
		float* sp = src.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* dp = vSrc[k].ptr<float>(i) + j * 4;

				_mm_stream_ps(dp, _mm_load_ps(sp));
				sp += 4;
			}
		}
	}
}

void boxFilter_SSAT_SoA_Space_SSE::SoA2AoS()
{
	//merge(vDest, dest);

#pragma omp parallel for
	for (int i = 0; i < vDest[0].rows; i++)
	{
		float* dp = dest.ptr<float>(i);

		for (int j = 0; j < vDest[0].cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* sp = vDest[k].ptr<float>(i) + j * 4;

				_mm_stream_ps(dp, _mm_load_ps(sp));
				dp += 4;
			}
		}
	}
}



boxFilter_SSAT_SoA_Space_AVX::boxFilter_SSAT_SoA_Space_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_SSAT_SoA_Space_nonVec(_src, _dest, _r, _parallelType)
{
	init();
}

void boxFilter_SSAT_SoA_Space_AVX::init()
{
	//loop = cn;

	//vSrc.resize(loop);
	//vDest.resize(loop);
	//vTemp.resize(loop);
	//for (int i = 0; i < loop; i++)
	//{
	//	vSrc[i].create(src.size(), CV_32FC1);
	//	vDest[i].create(src.size(), CV_32FC1);
	//	vTemp[i].create(src.size(), CV_32FC1);
	//}

	loop = cn >> 3;

	vSrc.resize(loop);
	vDest.resize(loop);
	vTemp.resize(loop);
	for (int i = 0; i < loop; i++)
	{
		vSrc[i].create(src.size(), CV_32FC(8));
		vDest[i].create(src.size(), CV_32FC(8));
		vTemp[i].create(src.size(), CV_32FC(8));
	}
}

void boxFilter_SSAT_SoA_Space_AVX::filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp)
{
	RowSumFilter_AVX(input, temp, r, parallelType).filter();
	ColumnSumFilter_AVX(temp, output, r, parallelType).filter();
}

void boxFilter_SSAT_SoA_Space_AVX::AoS2SoA()
{
	//split(src, vSrc);

#pragma omp parallel for
	for (int i = 0; i < src.rows; i++)
	{
		float* sp = src.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* dp = vSrc[k].ptr<float>(i) + j * 8;

				_mm256_stream_ps(dp, _mm256_load_ps(sp));
				sp += 8;
			}
		}
	}
}

void boxFilter_SSAT_SoA_Space_AVX::SoA2AoS()
{
	//merge(vDest, dest);

#pragma omp parallel for
	for (int i = 0; i < vDest[0].rows; i++)
	{
		float* dp = dest.ptr<float>(i);

		for (int j = 0; j < vDest[0].cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* sp = vDest[k].ptr<float>(i) + j * 8;

				_mm256_stream_ps(dp, _mm256_load_ps(sp));
				dp += 8;
			}
		}
	}
}



boxFilter_SSAT_SoA_Channel_nonVec::boxFilter_SSAT_SoA_Channel_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	cn = src.channels();

	init();
}

void boxFilter_SSAT_SoA_Channel_nonVec::init()
{
	loop = cn;

	vSrc.resize(loop);
	vDest.resize(loop);
	vTemp.resize(loop);
	for (int i = 0; i < loop; i++)
	{
		vSrc[i].create(src.size(), CV_32FC1);
		vDest[i].create(src.size(), CV_32FC1);
		vTemp[i].create(src.size(), CV_32FC1);
	}
}

void boxFilter_SSAT_SoA_Channel_nonVec::filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp)
{
	RowSumFilter(input, temp, r, NAIVE).filter();
	ColumnSumFilter_nonVec(temp, output, r, NAIVE).filter();
}

void boxFilter_SSAT_SoA_Channel_nonVec::filter()
{
	AoS2SoA();
	if (parallelType == NAIVE)
	{
		for (int i = 0; i < loop; i++)
			filter_impl(vSrc[i], vDest[i], vTemp[i]);
	}
	else if (parallelType == OMP)
	{
#pragma omp parallel for
		for (int i = 0; i < loop; i++)
			filter_impl(vSrc[i], vDest[i], vTemp[i]);
	}
	SoA2AoS();
}

void boxFilter_SSAT_SoA_Channel_nonVec::filterOnly()
{
	if (parallelType == NAIVE)
	{
		for (int i = 0; i < loop; i++)
			filter_impl(vSrc[i], vDest[i], vTemp[i]);
	}
	else if (parallelType == OMP)
	{
#pragma omp parallel for
		for (int i = 0; i < loop; i++)
			filter_impl(vSrc[i], vDest[i], vTemp[i]);
	}
}

void boxFilter_SSAT_SoA_Channel_nonVec::AoS2SoA()
{
#pragma omp parallel for
	for (int i = 0; i < src.rows; i++)
	{
		float* sp = src.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* dp = vSrc[k].ptr<float>(i) + j;

				*dp = *sp;
				sp++;
			}
		}
	}
}

void boxFilter_SSAT_SoA_Channel_nonVec::SoA2AoS()
{
#pragma omp parallel for
	for (int i = 0; i < vDest[0].rows; i++)
	{
		float* dp = dest.ptr<float>(i);

		for (int j = 0; j < vDest[0].cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* sp = vDest[k].ptr<float>(i) + j;

				*dp = *sp;
				dp++;
			}
		}
	}
}



boxFilter_SSAT_SoA_Channel_SSE::boxFilter_SSAT_SoA_Channel_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_SSAT_SoA_Channel_nonVec(_src, _dest, _r, _parallelType)
{
	init();
}

void boxFilter_SSAT_SoA_Channel_SSE::init()
{
	//loop = cn;

	//vSrc.resize(loop);
	//vDest.resize(loop);
	//vTemp.resize(loop);
	//for (int i = 0; i < loop; i++)
	//{
	//	vSrc[i].create(src.size(), CV_32FC1);
	//	vDest[i].create(src.size(), CV_32FC1);
	//	vTemp[i].create(src.size(), CV_32FC1);
	//}

	loop = cn >> 2;

	vSrc.resize(loop);
	vDest.resize(loop);
	vTemp.resize(loop);
	for (int i = 0; i < loop; i++)
	{
		vSrc[i].create(src.size(), CV_32FC4);
		vDest[i].create(src.size(), CV_32FC4);
		vTemp[i].create(src.size(), CV_32FC4);
	}
}

void boxFilter_SSAT_SoA_Channel_SSE::filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp)
{
	RowSumFilter_SSE(input, temp, r, NAIVE).filter();
	ColumnSumFilter_SSE(temp, output, r, NAIVE).filter();
}

void boxFilter_SSAT_SoA_Channel_SSE::AoS2SoA()
{
	//split(src, vSrc);

#pragma omp parallel for
	for (int i = 0; i < src.rows; i++)
	{
		float* sp = src.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* dp = vSrc[k].ptr<float>(i) + j * 4;

				_mm_stream_ps(dp, _mm_load_ps(sp));
				sp += 4;
			}
		}
	}
}

void boxFilter_SSAT_SoA_Channel_SSE::SoA2AoS()
{
	//merge(vDest, dest);

#pragma omp parallel for
	for (int i = 0; i < vDest[0].rows; i++)
	{
		float* dp = dest.ptr<float>(i);

		for (int j = 0; j < vDest[0].cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* sp = vDest[k].ptr<float>(i) + j * 4;

				_mm_stream_ps(dp, _mm_load_ps(sp));
				dp += 4;
			}
		}
	}
}



boxFilter_SSAT_SoA_Channel_AVX::boxFilter_SSAT_SoA_Channel_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_SSAT_SoA_Channel_nonVec(_src, _dest, _r, _parallelType)
{
	init();
}

void boxFilter_SSAT_SoA_Channel_AVX::init()
{
	//loop = cn;

	//vSrc.resize(loop);
	//vDest.resize(loop);
	//vTemp.resize(loop);
	//for (int i = 0; i < loop; i++)
	//{
	//	vSrc[i].create(src.size(), CV_32FC1);
	//	vDest[i].create(src.size(), CV_32FC1);
	//	vTemp[i].create(src.size(), CV_32FC1);
	//}

	loop = cn >> 3;

	vSrc.resize(loop);
	vDest.resize(loop);
	vTemp.resize(loop);
	for (int i = 0; i < loop; i++)
	{
		vSrc[i].create(src.size(), CV_32FC(8));
		vDest[i].create(src.size(), CV_32FC(8));
		vTemp[i].create(src.size(), CV_32FC(8));
	}
}

void boxFilter_SSAT_SoA_Channel_AVX::filter_impl(cv::Mat& input, cv::Mat& output, cv::Mat& temp)
{
	RowSumFilter_AVX(input, temp, r, NAIVE).filter();
	ColumnSumFilter_AVX(temp, output, r, NAIVE).filter();
}

void boxFilter_SSAT_SoA_Channel_AVX::AoS2SoA()
{
	//split(src, vSrc);

#pragma omp parallel for
	for (int i = 0; i < src.rows; i++)
	{
		float* sp = src.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* dp = vSrc[k].ptr<float>(i) + j * 8;

				_mm256_stream_ps(dp, _mm256_load_ps(sp));
				sp += 8;
			}
		}
	}
}

void boxFilter_SSAT_SoA_Channel_AVX::SoA2AoS()
{
	//merge(vDest, dest);

#pragma omp parallel for
	for (int i = 0; i < vDest[0].rows; i++)
	{
		float* dp = dest.ptr<float>(i);

		for (int j = 0; j < vDest[0].cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* sp = vDest[k].ptr<float>(i) + j * 8;

				_mm256_stream_ps(dp, _mm256_load_ps(sp));
				dp += 8;
			}
		}
	}
}
