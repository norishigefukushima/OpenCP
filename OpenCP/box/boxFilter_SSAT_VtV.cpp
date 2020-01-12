#include "boxFilter_SSAT_VtV.h"
#include "boxFilter_SSAT_HV.h"
#include "boxFilter_SSAT_VH.h"

using namespace std;
using namespace cv;

void trans(cv::Mat& input, cv::Mat& output)
{
	const int in_col = input.cols;
	const int out_col = output.cols;
#pragma omp parallel for
	for (int i = 0; i < input.rows; i += 8)
	{
		for (int j = 0; j < input.cols; j += 8)
		{
			float* sp = input.ptr<float>(i) + j;
			float* dp = output.ptr<float>(j) + i;

			__m256 mRef0 = _mm256_load_ps(sp);
			__m256 mRef1 = _mm256_load_ps(sp + in_col);
			__m256 mRef2 = _mm256_load_ps(sp + in_col * 2);
			__m256 mRef3 = _mm256_load_ps(sp + in_col * 3);
			__m256 mRef4 = _mm256_load_ps(sp + in_col * 4);
			__m256 mRef5 = _mm256_load_ps(sp + in_col * 5);
			__m256 mRef6 = _mm256_load_ps(sp + in_col * 6);
			__m256 mRef7 = _mm256_load_ps(sp + in_col * 7);

			__m256 mTmp0 = _mm256_unpacklo_ps(mRef0, mRef2);
			__m256 mTmp1 = _mm256_unpacklo_ps(mRef1, mRef3);
			__m256 mTmp2 = _mm256_unpacklo_ps(mRef4, mRef6);
			__m256 mTmp3 = _mm256_unpacklo_ps(mRef5, mRef7);
			__m256 mTmp4 = _mm256_unpackhi_ps(mRef0, mRef2);
			__m256 mTmp5 = _mm256_unpackhi_ps(mRef1, mRef3);
			__m256 mTmp6 = _mm256_unpackhi_ps(mRef4, mRef6);
			__m256 mTmp7 = _mm256_unpackhi_ps(mRef5, mRef7);

			mRef0 = _mm256_permute2f128_ps(mTmp0, mTmp2, 0x20);
			mRef1 = _mm256_permute2f128_ps(mTmp1, mTmp3, 0x20);
			mRef2 = _mm256_permute2f128_ps(mTmp4, mTmp6, 0x20);
			mRef3 = _mm256_permute2f128_ps(mTmp5, mTmp7, 0x20);
			mRef4 = _mm256_permute2f128_ps(mTmp0, mTmp2, 0x31);
			mRef5 = _mm256_permute2f128_ps(mTmp1, mTmp3, 0x31);
			mRef6 = _mm256_permute2f128_ps(mTmp4, mTmp6, 0x31);
			mRef7 = _mm256_permute2f128_ps(mTmp5, mTmp7, 0x31);

			__m256 mDst0 = _mm256_unpacklo_ps(mRef0, mRef1);
			__m256 mDst1 = _mm256_unpackhi_ps(mRef0, mRef1);
			__m256 mDst2 = _mm256_unpacklo_ps(mRef2, mRef3);
			__m256 mDst3 = _mm256_unpackhi_ps(mRef2, mRef3);
			__m256 mDst4 = _mm256_unpacklo_ps(mRef4, mRef5);
			__m256 mDst5 = _mm256_unpackhi_ps(mRef4, mRef5);
			__m256 mDst6 = _mm256_unpacklo_ps(mRef6, mRef7);
			__m256 mDst7 = _mm256_unpackhi_ps(mRef6, mRef7);

			_mm256_stream_ps(dp, mDst0);
			_mm256_stream_ps(dp + out_col, mDst1);
			_mm256_stream_ps(dp + out_col * 2, mDst2);
			_mm256_stream_ps(dp + out_col * 3, mDst3);
			_mm256_stream_ps(dp + out_col * 4, mDst4);
			_mm256_stream_ps(dp + out_col * 5, mDst5);
			_mm256_stream_ps(dp + out_col * 6, mDst6);
			_mm256_stream_ps(dp + out_col * 7, mDst7);
		}
	}
}



boxFilter_SSAT_VtV_nonVec::boxFilter_SSAT_VtV_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	temp.create(Size(src.cols, src.rows), src.type());
	temp_t.create(Size(src.rows, src.cols), src.type());
	dest_t.create(Size(src.rows, src.cols), src.type());
}

void boxFilter_SSAT_VtV_nonVec::filter()
{
	ColumnSumFilter_VH_nonVec(src, temp, r, parallelType).filter();
	temp_t = temp.t();
	ColumnSumFilter_nonVec(temp_t, dest_t, r, parallelType).filter();
	dest = dest_t.t();
}



void boxFilter_SSAT_VtV_SSE::filter()
{
	ColumnSumFilter_VH_SSE(src, temp, r, parallelType).filter();
	temp_t = temp.t();
	ColumnSumFilter_SSE(temp_t, dest_t, r, parallelType).filter();
	dest = dest_t.t();
}



void boxFilter_SSAT_VtV_AVX::filter()
{
	ColumnSumFilter_VH_AVX(src, temp, r, parallelType).filter();
	//temp_t = temp.t();
	trans(temp, temp_t);
	ColumnSumFilter_AVX(temp_t, dest_t, r, parallelType).filter();
	//dest = dest_t.t();
	trans(dest_t, dest);
}
