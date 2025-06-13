#include "DCCI.hpp"
#include "inlineSIMDFunctions.hpp"
#include "inlineMathFunctions.hpp"
#include "upsample.hpp"
#include <omp.h>
using namespace std;
using namespace cv;

namespace cp
{
	inline void ringbuffering2(float*& a, float*& b)
	{
		std::swap(a, b);
	}

	inline void ringbuffering3(float*& a, float*& b, float*& c)
	{
		float* t = a;
		a = b;
		b = c;
		c = t;
	}

	inline void ringbuffering4(float*& a, float*& b, float*& c, float*& d)
	{
		float* t = a;
		a = b;
		b = c;
		c = d;
		d = t;
	}

	inline void ringbuffering5(float*& a, float*& b, float*& c, float*& d, float*& e)
	{
		float* t = a;
		a = b;
		b = c;
		c = d;
		d = e;
		e = t;
	}


#define __CALC_FULL_GRAD__
#define __USE_STREAM_INSTRUCTION__//destへのstoreをstreamに（color）
#define __USE_L2NORM__
	//#define __INTERPOLATE_BORDER__
#define __LINEAR_INTERPOLATION_WITH_OPENMP__

#define _mm256_loadx_ps _mm256_loadu_ps
//#define _mm256_loadx_ps _mm256_lddqu_ps
#define _mm256_storea_ps _mm256_store_ps
//#define _mm256_storea_ps _mm256_stream_ps

	using namespace std;
	using namespace cv;

	//#define _USE_GLOBAL_BUFFER_
#ifdef _USE_GLOBAL_BUFFER_
	AutoBuffer<float*> buffer_global(128);
	AutoBuffer<int> bfsize_global(128);
#endif

	//#define __USE_GLOBAL_BUFFER__
	int bufferGlobalLFSize[128];
	float* bufferGlobalLFPtr[128];

	void DCCI32FC1_SIMD_LoopFusionOld(const Mat& src_, Mat& dest, const float threshold, int ompNumThreads)
	{
		CV_Assert(src_.type() == CV_32FC1);

		const Mat src = src_.data == dest.data ? src_.clone() : src_;
		const Size ssize = src.size();
		dest = Mat::zeros(ssize.height << 1, ssize.width << 1, CV_32FC1);
		const Size dsize = dest.size();

		/* borderの補間は今回関係ないので無視
		#ifdef __INTERPOLATE_BORDER__
			{
				//線形補間
				const __m256 mDiv_2 = _mm256_set1_ps(0.5f);//線形補間用
				__m256i rightEnd = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 6);//右端のgather用
		#pragma omp sections //若干速い
				{
		#pragma omp section
					{//top:dest6行分

						const float* sp = src.ptr<float>(0);
						float* dp = dest.ptr<float>(0);
						for (int x = 0; x < ssize.width - 8; x += 8)
						{
							__m256 mSrc0_0 = _mm256_load_ps(sp);
							__m256 mSrc0_1 = _mm256_loadu_ps(sp + 1);
							__m256 mSrcA = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc0_0, mSrc0_1));

							__m256 mTmpHi = _mm256_unpackhi_ps(mSrc0_0, mSrcA);
							__m256 mTmpLo = _mm256_unpacklo_ps(mSrc0_0, mSrcA);
							__m256 mSrcHi_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp, mSrcLo_A);
							_mm256_store_ps(dp + 8, mSrcHi_A);

							__m256 mSrc1_0 = _mm256_load_ps(sp + ssize.width);
							__m256 mSrc1_1 = _mm256_loadu_ps(sp + ssize.width + 1);
							__m256 mSrcB = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc1_0, mSrc1_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc1_0, mSrcB);
							mTmpLo = _mm256_unpacklo_ps(mSrc1_0, mSrcB);
							__m256 mSrcHi_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_A, mSrcLo_B)));
							_mm256_store_ps(dp + dsize.width + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_A, mSrcHi_B)));

							_mm256_store_ps(dp + (dsize.width << 1), mSrcLo_B);
							_mm256_store_ps(dp + (dsize.width << 1) + 8, mSrcHi_B);

							__m256 mSrc2_0 = _mm256_load_ps(sp + (ssize.width << 1));
							__m256 mSrc2_1 = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
							__m256 mSrcC = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc2_0, mSrc2_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc2_0, mSrcC);
							mTmpLo = _mm256_unpacklo_ps(mSrc2_0, mSrcC);
							__m256 mSrcHi_C = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_C = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width + (dsize.width << 1), _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_B, mSrcLo_C)));
							_mm256_store_ps(dp + dsize.width + (dsize.width << 1) + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_B, mSrcHi_C)));

							_mm256_store_ps(dp + (dsize.width << 2), mSrcLo_C);
							_mm256_store_ps(dp + (dsize.width << 2) + 8, mSrcHi_C);

							__m256 mSrc3_0 = _mm256_load_ps(sp + ssize.width + (ssize.width << 1));
							__m256 mSrc3_1 = _mm256_loadu_ps(sp + ssize.width + (ssize.width << 1) + 1);
							__m256 mSrcD = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc3_0, mSrc3_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc3_0, mSrcD);
							mTmpLo = _mm256_unpacklo_ps(mSrc3_0, mSrcD);
							__m256 mSrcHi_D = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_D = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width + (dsize.width << 2), _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_C, mSrcLo_D)));
							_mm256_store_ps(dp + dsize.width + (dsize.width << 2) + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_C, mSrcHi_D)));//5
							sp += 8;
							dp += 16;
						}
					}//top

		#pragma omp section
					{//left
						const float* sp = src.ptr<float>(3);
						float* dp = dest.ptr<float>(6);
						for (int y = 3; y < ssize.height - 1; ++y)
						{
							__m256 mSrc0_0 = _mm256_load_ps(sp);
							__m256 mSrc0_1 = _mm256_loadu_ps(sp + 1);
							__m256 mSrcA = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc0_0, mSrc0_1));

							__m256 mTmpHi = _mm256_unpackhi_ps(mSrc0_0, mSrcA);
							__m256 mTmpLo = _mm256_unpacklo_ps(mSrc0_0, mSrcA);
							__m256 mSrcHi_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp, mSrcLo_A);
							_mm256_store_ps(dp + 8, mSrcHi_A);

							__m256 mSrc1_0 = _mm256_load_ps(sp + ssize.width);
							__m256 mSrc1_1 = _mm256_loadu_ps(sp + ssize.width + 1);
							__m256 mSrcB = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc1_0, mSrc1_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc1_0, mSrcB);
							mTmpLo = _mm256_unpacklo_ps(mSrc1_0, mSrcB);
							__m256 mSrcHi_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_A, mSrcLo_B)));
							_mm256_store_ps(dp + dsize.width + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_A, mSrcHi_B)));//1

							sp += ssize.width;
							dp += (dsize.width << 1);
						}
					}
		#pragma omp section
					{
						const float* sp = src.ptr<float>(0) + ssize.width - 8;
						float* dp = dest.ptr<float>(0) + dsize.width - 16;
						for (int y = 0; y < ssize.height - 1; ++y)
						{//right:

							__m256 mSrc0_0 = _mm256_load_ps(sp);
							__m256 mSrc0_1 = _mm256_i32gather_ps(sp + 1, rightEnd, sizeof(float));
							__m256 mSrcA = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc0_0, mSrc0_1));

							__m256 mTmpHi = _mm256_unpackhi_ps(mSrc0_0, mSrcA);
							__m256 mTmpLo = _mm256_unpacklo_ps(mSrc0_0, mSrcA);
							__m256 mSrcHi_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp, mSrcLo_A);
							_mm256_store_ps(dp + 8, mSrcHi_A);

							__m256 mSrc1_0 = _mm256_load_ps(sp + ssize.width);
							__m256 mSrc1_1 = _mm256_i32gather_ps(sp + ssize.width + 1, rightEnd, sizeof(float));
							__m256 mSrcB = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc1_0, mSrc1_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc1_0, mSrcB);
							mTmpLo = _mm256_unpacklo_ps(mSrc1_0, mSrcB);
							__m256 mSrcHi_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_A, mSrcLo_B)));
							_mm256_store_ps(dp + dsize.width + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_A, mSrcHi_B)));//1

							sp += ssize.width;
							dp += (dsize.width << 1);

						}
						{//right:一番下の列

							__m256 mSrc0_0 = _mm256_load_ps(sp);
							__m256 mSrc0_1 = _mm256_i32gather_ps(sp + 1, rightEnd, sizeof(float));
							__m256 mSrcA = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc0_0, mSrc0_1));

							__m256 mTmpHi = _mm256_unpackhi_ps(mSrc0_0, mSrcA);
							__m256 mTmpLo = _mm256_unpacklo_ps(mSrc0_0, mSrcA);
							__m256 mSrcHi_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp, mSrcLo_A);
							_mm256_store_ps(dp + 8, mSrcHi_A);

							_mm256_store_ps(dp + dsize.width, mSrcLo_A);
							_mm256_store_ps(dp + dsize.width + 8, mSrcHi_A);

						}
					}

		#pragma omp section
					{
						const float* sp = src.ptr<float>(ssize.height - 3);
						float* dp = dest.ptr<float>(dsize.height - 6);
						//bottom:dest2行分->6行分
						for (int x = 0; x < ssize.width - 8; x += 8)
						{
							__m256 mSrc0_0 = _mm256_load_ps(sp);
							__m256 mSrc0_1 = _mm256_loadu_ps(sp + 1);
							__m256 mSrcA = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc0_0, mSrc0_1));

							__m256 mTmpHi = _mm256_unpackhi_ps(mSrc0_0, mSrcA);
							__m256 mTmpLo = _mm256_unpacklo_ps(mSrc0_0, mSrcA);
							__m256 mSrcHi_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp, mSrcLo_A);
							_mm256_store_ps(dp + 8, mSrcHi_A);

							__m256 mSrc1_0 = _mm256_load_ps(sp + ssize.width);
							__m256 mSrc1_1 = _mm256_loadu_ps(sp + ssize.width + 1);
							__m256 mSrcB = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc1_0, mSrc1_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc1_0, mSrcB);
							mTmpLo = _mm256_unpacklo_ps(mSrc1_0, mSrcB);
							__m256 mSrcHi_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_A, mSrcLo_B)));
							_mm256_store_ps(dp + dsize.width + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_A, mSrcHi_B)));

							_mm256_store_ps(dp + (dsize.width << 1), mSrcLo_B);
							_mm256_store_ps(dp + (dsize.width << 1) + 8, mSrcHi_B);

							__m256 mSrc2_0 = _mm256_load_ps(sp + (ssize.width << 1));
							__m256 mSrc2_1 = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
							__m256 mSrcC = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc2_0, mSrc2_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc2_0, mSrcC);
							mTmpLo = _mm256_unpacklo_ps(mSrc2_0, mSrcC);
							__m256 mSrcHi_C = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_C = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width + (dsize.width << 1), _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_B, mSrcLo_C)));
							_mm256_store_ps(dp + dsize.width + (dsize.width << 1) + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_B, mSrcHi_C)));

							_mm256_store_ps(dp + (dsize.width << 2), mSrcLo_C);
							_mm256_store_ps(dp + (dsize.width << 2) + 8, mSrcHi_C);

							_mm256_store_ps(dp + dsize.width + (dsize.width << 2), mSrcLo_C);
							_mm256_store_ps(dp + dsize.width + (dsize.width << 2) + 8, mSrcHi_C);

							sp += 8;
							dp += 16;
						}
					}
				}
			}//線形補間
		#endif
		*/
		//------------------ DCCI ----------------------------------------------------------------------

		const int threadsNum = ompNumThreads;
#pragma omp parallel for num_threads(threadsNum) schedule(dynamic)
		for (int i = 1; i <= threadsNum; ++i)//threads
		{
			const int tempSize = ssize.width + 3;
			const __m256 signMask = _mm256_set1_ps(-0.0f); // 0x80000000
			const __m256 mThreshold = _mm256_set1_ps(threshold);
			const __m256 mOnef = _mm256_set1_ps(1.f);
			const __m256 cciCoeff_1 = _mm256_set1_ps(-1.f / 16);
			const __m256 cciCoeff_9 = _mm256_set1_ps(9.f / 16);

			const int buffer_width = ssize.width + 8;
			float* const buffer = (float*)_mm_malloc(sizeof(float) * buffer_width * 4, 32);

			std::array<float*, 4>buf =
			{
				buffer,
				buffer + buffer_width,
				buffer + buffer_width * 2,
				buffer + buffer_width * 3
			};

			float* tp = nullptr;

			int threadsWidth = (ssize.height / threadsNum);

			int start = (i == 1 ? 0 : threadsWidth * (i - 1) - 3);
			int end = start + 3;
			//バッファーの初期設定：tempを本来一段階目で補間される画素値で埋める．
			for (int y = start; y < end; ++y)
			{

				const float* sp = src.ptr<float>(y) + 4;

				{
					//一列目
					float grad45 = 0, grad135 = 0;

					sp += (ssize.width - 2);
					__m256 mTmp0 = _mm256_setzero_ps(), mTmp1 = _mm256_setzero_ps(), mTmp2 = _mm256_setzero_ps();
					//  0   4   8   C
					//  1   5   9   D
					//  2   6   A   E
					//  3   7   B   F

					//UpRight G1
					__m256 mK4 = _mm256_loadu_ps(sp - ssize.width);
					__m256 mK1 = _mm256_loadu_ps(sp - 1);
					__m256 gradUpRight = _mm256_setzero_ps();
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1)));

					__m256 mK8 = _mm256_loadu_ps(sp - ssize.width + 1);
					__m256 mK5 = _mm256_loadu_ps(sp);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));

					__m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);
					__m256 mK9 = _mm256_loadu_ps(sp + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));

					__m256 mK2 = _mm256_loadu_ps(sp + ssize.width - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));

					__m256 mK6 = _mm256_loadu_ps(sp + ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));

					__m256 mKD = _mm256_loadu_ps(sp + 2);
					__m256 mKA = _mm256_loadu_ps(sp + ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));

					__m256 mK3 = _mm256_loadu_ps(sp + (ssize.width << 1) - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));

					__m256 mK7 = _mm256_loadu_ps(sp + (ssize.width << 1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));

					__m256 mKB = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
					__m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

					//DownRight G2
					__m256 mK0 = _mm256_loadu_ps(sp - ssize.width - 1);
					__m256 gradDownRight = _mm256_setzero_ps();
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));

					__m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));


					/* --- 2(b) --- */
					mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					mTmp1 = _mm256_add_ps(mOnef, gradDownRight);

					//if (1+G1) / (1+G2) > T then 135
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					__m256 pxUpRight = _mm256_add_ps(mK3, mKC);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK6, mK9), pxUpRight);

					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					__m256 pxDownRight = _mm256_add_ps(mK0, mKF);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK5, mKA), pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp2 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp2 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp2);
					__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp2);
					mTmp0 = _mm256_add_ps(weight1, weight2);

					//0で最初の引数をとる
					__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_store_ps(buf[0], mDst);
					sp += 3;

					mTmp0 = _mm256_setzero_ps(), mTmp1 = _mm256_setzero_ps(), mTmp2 = _mm256_setzero_ps();

					//  C G K O
					//  D H L P
					//  E I M Q
					//  F J N R	

					//UpRight G1
					__m256 mKG = _mm256_loadu_ps(sp - ssize.width);
					gradUpRight = _mm256_setzero_ps();
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD)));

					__m256 mKH = _mm256_loadu_ps(sp);
					__m256 mKK = _mm256_loadu_ps(sp - ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

					__m256 mKO = _mm256_loadu_ps(sp - ssize.width + 2);
					__m256 mKL = _mm256_loadu_ps(sp + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

					__m256 mKI = _mm256_loadu_ps(sp + ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

					__m256 mKP = _mm256_loadu_ps(sp + 2);
					__m256 mKM = _mm256_loadu_ps(sp + ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));

					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

					__m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

					__m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
					__m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

					//DownRight G2
					gradDownRight = _mm256_setzero_ps();
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

					__m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));

					/* --- 2(b) --- */
					mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
					//if (1+G1) / (1+G2) > T then 135
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					pxUpRight = _mm256_add_ps(mKF, mKO);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKI, mKL), pxUpRight);

					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					pxDownRight = _mm256_add_ps(mKC, mKR);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKH, mKM), pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);
					mTmp0 = _mm256_add_ps(weight1, weight2);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_storeu_ps(buf[0] + 3, mDst);
					sp -= (ssize.width - 7);
					buf[0] += 8;
					buf[1] += 8;
					buf[2] += 8;
				}//x


				for (int x = 0; x < ssize.width - 16; x += 8)//バッファー埋め一列目以降；右下の●だけ求める処理
				{
					float grad45 = 0, grad135 = 0;

					//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
					sp += (ssize.width - 2);
					__m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);

					//  C G K O
					//  D H L P
					//  E I M Q
					//  F J N R	
					//UpRight G1
					__m256 mKD = _mm256_loadu_ps(sp + 2);
					__m256 mKG = _mm256_loadu_ps(sp - ssize.width + 3);
					__m256 gradUpRight = _mm256_setzero_ps();
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD)));

					__m256 mKH = _mm256_loadu_ps(sp + 3);
					__m256 mKK = _mm256_loadu_ps(sp - ssize.width + 4);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

					__m256 mKO = _mm256_loadu_ps(sp - ssize.width + 5);
					__m256 mKL = _mm256_loadu_ps(sp + 4);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

					__m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

					__m256 mKI = _mm256_loadu_ps(sp + ssize.width + 3);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

					__m256 mKM = _mm256_loadu_ps(sp + ssize.width + 4);
					__m256 mKP = _mm256_loadu_ps(sp + 5);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));

					__m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

					__m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1) + 3);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

					__m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 4);
					__m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 5);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

					//DownRight G2
					__m256 gradDownRight = _mm256_setzero_ps();
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

					__m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 5);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));

					sp += 3;

					/* --- 2(b) --- */
					__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);

					//if (1+G1) / (1+G2) > T then 135
					__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					__m256 pxUpRight = _mm256_add_ps(mKF, mKO);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKI, mKL), pxUpRight);

					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					__m256 pxDownRight = _mm256_add_ps(mKC, mKR);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKH, mKM), pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1 + w2p2) / (w1 + w2)
					mTmp2 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp2 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp2);
					__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp2);

					//0で最初の引数をとる
					__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_storeu_ps(buf[0] + 3, mDst);

					sp -= (ssize.width - 7);

					buf[0] += 8;
					buf[1] += 8;
					buf[2] += 8;
				}//x
				tp = buf[0];
				buf[0] = buf[2];
				buf[2] = buf[1];
				buf[1] = tp;

				buf[0] -= tempSize - 11;
				buf[1] -= tempSize - 11;
				buf[2] -= tempSize - 11;
			}//y

			tp = buf[0];
			buf[0] = buf[3];
			buf[3] = tp;

			//DCCIメイン処理
			start = end;
			end = (i == threadsNum ? i * threadsWidth + ssize.height % threadsNum - 3 : i * threadsWidth);
			for (int y = start; y < end; ++y)
			{
				const float* sp = src.ptr<float>(y) + 4;
				float* dp = dest.ptr<float>((y) << 1) + 8;//align(8)
				float grad45 = 0, grad135 = 0;

				{
					//一列目の処理．左下，右下の●を求める必要がある
					//（奇数，奇数）の補間
					dp += dsize.width + 1;

					//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
					sp += (ssize.width - 2);
					dp += ((dsize.width << 1) - 4);
					//  1   5   9   D
					//  0   4   8   C
					//  2   6   A   E
					//  3   7   B   F

					//UpRight G1
					__m256 mK1 = _mm256_loadu_ps(sp - 1);
					__m256 mK4 = _mm256_loadu_ps(sp - ssize.width);
					__m256 gradUpRight = _mm256_setzero_ps();
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1)));

					__m256 mK5 = _mm256_loadu_ps(sp);
					__m256 mK8 = _mm256_loadu_ps(sp - ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));

					__m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);
					__m256 mK9 = _mm256_loadu_ps(sp + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));

					__m256 mK2 = _mm256_loadu_ps(sp + ssize.width - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));

					__m256 mK6 = _mm256_loadu_ps(sp + ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));

					__m256 mKA = _mm256_loadu_ps(sp + ssize.width + 1);
					__m256 mKD = _mm256_loadu_ps(sp + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));

					__m256 mK3 = _mm256_loadu_ps(sp + (ssize.width << 1) - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));

					__m256 mK7 = _mm256_loadu_ps(sp + (ssize.width << 1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));

					__m256 mKB = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
					__m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

					//DownRight G2
					__m256 mK0 = _mm256_loadu_ps(sp - ssize.width - 1);
					__m256 gradDownRight = _mm256_setzero_ps();
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));
					__m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));

					/* --- 2(b) --- */
					__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
					//if (1+G1) / (1+G2) > T then 135
					__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					__m256 pxUpRight = _mm256_add_ps(mK3, mKC);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mK6, mK9), cciCoeff_9, pxUpRight);

					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					__m256 pxDownRight = _mm256_add_ps(mK0, mKF);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mK5, mKA), cciCoeff_9, pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
					__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);


					//0で最初の引数をとる
					__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_store_ps(buf[0], mDst);

					sp += 3;
					dp += 6;

					pxUpRight = _mm256_setzero_ps(), pxDownRight = _mm256_setzero_ps(), pxSmooth = _mm256_setzero_ps();
					gradUpRight = _mm256_setzero_ps(), gradDownRight = _mm256_setzero_ps();
					mTmp0 = _mm256_setzero_ps(), mTmp1 = _mm256_setzero_ps(), mTmp2 = _mm256_setzero_ps();

					//  C G K O
					//  D H L P
					//  E I M Q
					//  F J N R	

					//UpRight G1
					__m256 mKG = _mm256_loadu_ps(sp - ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD)));

					__m256 mKH = _mm256_loadu_ps(sp);
					__m256 mKK = _mm256_loadu_ps(sp - ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

					__m256 mKO = _mm256_loadu_ps(sp - ssize.width + 2);
					__m256 mKL = _mm256_loadu_ps(sp + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

					__m256 mKI = _mm256_loadu_ps(sp + ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

					__m256 mKP = _mm256_loadu_ps(sp + 2);
					__m256 mKM = _mm256_loadu_ps(sp + ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));

					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

					__m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

					__m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
					__m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

					//DownRight G2
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

					__m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));

					/* --- 2(b) --- */
					mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
					//if (1+G1) / (1+G2) > T then 135
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					//mTmp2 = _mm256_div_ps(mTmp0, mTmp1);
					maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					//mTmp2 = _mm256_div_ps(mTmp1, mTmp0);
					maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					pxUpRight = _mm256_add_ps(mKF, mKO);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mKI, mKL), cciCoeff_9, pxUpRight);

					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					pxDownRight = _mm256_add_ps(mKC, mKR);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mKH, mKM), cciCoeff_9, pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_storeu_ps(buf[0] + 3, mDst);

					sp -= (ssize.width + 1);
					dp -= ((dsize.width << 1) + 2);

					/*(o,e)を補間-------------------------------------
					#:OddEven		@:EvenOdd
					S		|		X
					X X X X X	|	X S X T X
					8 X C @ G	|	X X X X X
					x X x # x X x	| X X C @ G X X
					9 X D X H	|	X # x X X
					X t X t X	|	X D X H X
					E		|		t		*/

					//horizontal
					__m256 mKd1 = _mm256_loadu_ps(buf[2] + 1);
					__m256 mKd2 = _mm256_loadu_ps(buf[2] + 2);
					__m256 gradHorizontal = _mm256_setzero_ps();
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKC)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));

					__m256 mKx1 = _mm256_loadu_ps(buf[1] + 1);
					__m256 mKx2 = _mm256_loadu_ps(buf[1] + 2);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKD)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

					__m256 mKt1 = _mm256_loadu_ps(buf[0] + 1);
					__m256 mKt2 = _mm256_loadu_ps(buf[0] + 2);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKt1, mKt2)));

					//Vertical
					__m256 gradVertical = _mm256_setzero_ps();
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK9)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKt1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKt2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));

					//Horizontal方向にエッジがある場合，Horizontal方向に補間
					__m256 pxHorizontal = _mm256_setzero_ps();
					__m256 mKx0 = _mm256_load_ps(buf[1] + 0);
					__m256 mKx3 = _mm256_loadu_ps(buf[1] + 3);
					pxHorizontal = _mm256_add_ps(mKx0, mKx3);
					pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
					pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKx1, mKx2), cciCoeff_9, pxHorizontal);

					//Vertical方向にエッジがある場合，Vertical方向に補間
					__m256 pxVertical = _mm256_setzero_ps();
					__m256 mKS = _mm256_loadu_ps(sp - ssize.width);
					pxVertical = _mm256_add_ps(mKS, mKE);
					pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
					pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKD), cciCoeff_9, pxVertical);

					//weight = 1 / (1+G^5)
					//weight1はgradHorizontalを使う
					weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradVerticalを使う
					weight2 = _mm256_mul_ps(gradVertical, gradVertical);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
					mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);
					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
					mTmp1 = _mm256_add_ps(mOnef, gradVertical);
					//if (1+G1) / (1+G2) > T then 135
					//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					//cmpの結果を論理演算に使うとバグる
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
					mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

					//mk7:1234 5678
					//dst:abcd efgh-->a1b2c3d4...=
					__m256 mTmpHi = _mm256_unpackhi_ps(mDst, mKx2);//a 1 b 2 / e 5 f 6
					__m256 mTmpLo = _mm256_unpacklo_ps(mDst, mKx2);//c 3 d 4 / g 7 h 8
					__m256 mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);//1 0 2 0 / 5 0 6 0
					__m256 mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);//3 0 4 0 / 7 0 8 0

					_mm256_store_ps(dp - 1, mSrcLo);
					_mm256_store_ps(dp + 7, mSrcHi);

					//horizontal
					gradHorizontal = _mm256_setzero_ps();
					__m256 mKT = _mm256_loadu_ps(sp - ssize.width + 1);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKT)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2)));
					__m256 mKd3 = _mm256_loadu_ps(buf[2] + 3);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKd3)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKx3)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

					//Vertical
					gradVertical = _mm256_setzero_ps();
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKC)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKT, mKG)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd3, mKx3)));

					//Horizontal方向にエッジがある場合，Horizontal方向に補間
					pxHorizontal = _mm256_add_ps(mK8, mKK);
					pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
					pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKG), cciCoeff_9, pxHorizontal);

					//Vertical方向にエッジがある場合，Vertical方向に補間
					__m256 mKd0 = _mm256_loadu_ps(buf[3] + 2);//1 0 2 0 / 3 0 4 0
					pxVertical = _mm256_add_ps(mKd0, mKt2);
					pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
					pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKd2, mKx2), cciCoeff_9, pxVertical);

					//weight = 1 / (1+G^5)
					//weight1はgradHorizontalを使う
					weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradVerticalを使う
					weight2 = _mm256_mul_ps(gradVertical, gradVertical);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1 + w2p2) / (w1 + w2)
					mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
					mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);
					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
					mTmp1 = _mm256_add_ps(mOnef, gradVertical);

					//if (1+G1) / (1+G2) > T then 135
					//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					//cmpの結果を論理演算に使うとバグる
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
					mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

					mTmpHi = _mm256_unpackhi_ps(mKC, mDst);//a 1 b 2 / e 5 f 6
					mTmpLo = _mm256_unpacklo_ps(mKC, mDst);//c 3 d 4 / g 7 h 8
					mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);//1 0 2 0 / 5 0 6 0
					mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);//3 0 4 0 / 7 0 8 0

					_mm256_store_ps(dp - 1 - dsize.width, mSrcLo);
					_mm256_store_ps(dp + 7 - dsize.width, mSrcHi);

					sp += 8;
					dp -= (dsize.width - 15);
					buf[0] += 8;
					buf[1] += 8;
					buf[2] += 8;
					buf[3] += 8;
				}//ここまでがmainの一列目の処理

				for (int x = 0; x < ssize.width - 16; x += 8)//2列目以降．●は右下だけでいい
				{
					float grad45 = 0, grad135 = 0;

					//（奇数，奇数）の補間
					//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
					sp += (ssize.width + 1);
					dp += (dsize.width + (dsize.width << 1) + 3);
					__m256 mKG = _mm256_loadu_ps(sp - ssize.width);
					//  0   4   8   C
					//  1   5   9   D
					//  2   6   A   E
					//  3   7   B   F

					//  C G K O
					//  D H L P
					//  E I M Q
					//  F J N R

					//UpRight G1
					__m256 mKD = _mm256_loadu_ps(sp - 1);
					__m256 gradUpRight = _mm256_setzero_ps();
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD)));

					__m256 mKH = _mm256_loadu_ps(sp);
					__m256 mKK = _mm256_loadu_ps(sp - ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

					__m256 mKO = _mm256_loadu_ps(sp - ssize.width + 2);
					__m256 mKL = _mm256_loadu_ps(sp + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));
					__m256 gradDownRight = _mm256_setzero_ps();
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));//

					__m256 mKE = _mm256_loadu_ps(sp + ssize.width - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

					__m256 mKI = _mm256_loadu_ps(sp + ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));//

					__m256 mKP = _mm256_loadu_ps(sp + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));//
					__m256 mKM = _mm256_loadu_ps(sp + ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));//

					__m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

					__m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));//

					__m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));//
					__m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));//

					//DownRight G2
					__m256 mKC = _mm256_loadu_ps(sp - ssize.width - 1);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));


					__m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));

					/* --- 2(b) --- */
					__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
					//if (1+G1) / (1+G2) > T then 135
					__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					__m256 pxUpRight = _mm256_add_ps(mKF, mKO);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mKI, mKL), cciCoeff_9, pxUpRight);


					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					__m256 pxDownRight = _mm256_add_ps(mKC, mKR);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mKH, mKM), cciCoeff_9, pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
					__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					//0で最初の引数をとる
					__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_storeu_ps(buf[0] + 3, mDst);

					sp -= (ssize.width + 1);
					dp -= ((dsize.width << 1) + 2);

					//(o,e)を補間-------------------------------------
					//		#:OddEven		@:EvenOdd
					//			S		|		X		
					//		X X X X X	|	X S X T X	
					//		8 X C @ G	|	X X X X X	
					//	  x X x # x X x	| X X C @ G X X
					//		9 X D X H	|	X # x X X	
					//		X t X t X	|	X D X H X	
					//			E		|		t	

					//horizontal
					__m256 gradHorizontal = _mm256_setzero_ps();
					__m256 mKd1 = _mm256_loadu_ps(buf[2] + 1);
					__m256 mKd2 = _mm256_loadu_ps(buf[2] + 2);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2)));

					__m256 mK8 = _mm256_loadu_ps(sp - 1);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKC)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));

					__m256 mKx1 = _mm256_loadu_ps(buf[1] + 1);
					__m256 mKx2 = _mm256_loadu_ps(buf[1] + 2);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
					__m256 mK9 = _mm256_loadu_ps(sp + ssize.width - 1);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKD)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

					__m256 mKt1 = _mm256_loadu_ps(buf[0] + 1);
					__m256 mKt2 = _mm256_loadu_ps(buf[0] + 2);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKt1, mKt2)));

					//Vertical
					__m256 gradVertical = _mm256_setzero_ps();
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK9)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKt1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKt2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));

					//Horizontal方向にエッジがある場合，Horizontal方向に補間
					__m256 pxHorizontal = _mm256_setzero_ps();
					__m256 mKx0 = _mm256_load_ps(buf[1]);
					__m256 mKx3 = _mm256_loadu_ps(buf[1] + 3);
					pxHorizontal = _mm256_add_ps(mKx0, mKx3);
					pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
					pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKx1, mKx2), cciCoeff_9, pxHorizontal);

					//Vertical方向にエッジがある場合，Vertical方向に補間
					__m256 pxVertical = _mm256_setzero_ps();
					__m256 mKS = _mm256_loadu_ps(sp - ssize.width);
					pxVertical = _mm256_add_ps(mKS, mKE);
					pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
					pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKD), cciCoeff_9, pxVertical);

					//weight = 1 / (1+G^5)
					//weight1はgradHorizontalを使う
					weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradVerticalを使う
					weight2 = _mm256_mul_ps(gradVertical, gradVertical);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
					mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);

					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
					mTmp1 = _mm256_add_ps(mOnef, gradVertical);

					//if (1+G1) / (1+G2) > T then 135
					//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					//cmpの結果を論理演算に使うとバグる
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
					mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

					//mk7:1234 5678
					//dst:abcd efgh-->a1b2c3d4...=
					__m256 mTmpHi = _mm256_unpackhi_ps(mDst, mKx2);//a 1 b 2 / e 5 f 6
					__m256 mTmpLo = _mm256_unpacklo_ps(mDst, mKx2);//c 3 d 4 / g 7 h 8
					__m256 mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);//1 0 2 0 / 5 0 6 0
					__m256 mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);//3 0 4 0 / 7 0 8 0

					_mm256_store_ps(dp - 1, mSrcLo);
					_mm256_store_ps(dp + 7, mSrcHi);

					//horizontal
					gradHorizontal = _mm256_setzero_ps();
					__m256 mKT = _mm256_loadu_ps(sp - ssize.width + 1);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKT)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2)));

					__m256 mKd3 = _mm256_loadu_ps(buf[2] + 3);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKd3)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKx3)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

					//Vertical
					gradVertical = _mm256_setzero_ps();
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKC)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKT, mKG)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd3, mKx3)));

					//Horizontal方向にエッジがある場合，Horizontal方向に補間
					pxHorizontal = _mm256_add_ps(mK8, mKK);
					pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
					pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKG), cciCoeff_9, pxHorizontal);


					//Vertical方向にエッジがある場合，Vertical方向に補間
					__m256 mKd0 = _mm256_loadu_ps(buf[3] + 2);//1 0 2 0 / 3 0 4 0
					pxVertical = _mm256_add_ps(mKd0, mKt2);
					pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
					pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKd2, mKx2), cciCoeff_9, pxVertical);

					//weight = 1 / (1+G^5)
					//weight1はgradHorizontalを使う
					weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradVerticalを使う
					weight2 = _mm256_mul_ps(gradVertical, gradVertical);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1 + w2p2) / (w1 + w2)
					mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
					mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);
					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
					mTmp1 = _mm256_add_ps(mOnef, gradVertical);

					//if (1+G1) / (1+G2) > T then 135
					//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					//cmpの結果を論理演算に使うとバグる
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
					mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

					mTmpHi = _mm256_unpackhi_ps(mKC, mDst);
					mTmpLo = _mm256_unpacklo_ps(mKC, mDst);
					mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
					mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

					_mm256_stream_ps(dp - 1 - dsize.width, mSrcLo);
					_mm256_stream_ps(dp - dsize.width + 7, mSrcHi);//koko ha store yori hayai kamo
					//_mm256_storeu_ps(dp - 1 - dsize.width, mSrcLo);
					//_mm256_storeu_ps(dp - dsize.width + 7, mSrcHi);//koko ha store yori hayai kamo

					sp += 8;
					dp -= (dsize.width - 15);
					buf[0] += 8;
					buf[1] += 8;
					buf[2] += 8;
					buf[3] += 8;
				}//x メインループの2列目以降
				tp = buf[0];
				buf[0] = buf[3];
				buf[3] = buf[2];
				buf[2] = buf[1];
				buf[1] = tp;

				buf[0] -= tempSize - 11;
				buf[1] -= tempSize - 11;
				buf[2] -= tempSize - 11;
				buf[3] -= tempSize - 11;
			}//y
			_mm_free(buffer);
		}//parallel
	}

	void DCCI32FC1_SIMD_LoopFusion(const Mat& src_, Mat& dest, const float threshold, int ompNumThreads)
	{
		CV_Assert(src_.type() == CV_32FC1);

		const Mat src = src_.data == dest.data ? src_.clone() : src_;
		const Size ssize = src.size();
		//dest = Mat::zeros(ssize.height << 1, ssize.width << 1, CV_32FC1);//これがだめ
		if (dest.size() != src.size() * 2) dest.create(ssize.height << 1, ssize.width << 1, CV_32FC1);
		const Size dsize = dest.size();

		/* borderの補間は今回関係ないので無視
		#ifdef __INTERPOLATE_BORDER__
			{
				//線形補間
				const __m256 mDiv_2 = _mm256_set1_ps(0.5f);//線形補間用
				__m256i rightEnd = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 6);//右端のgather用
		#pragma omp sections //若干速い
				{
		#pragma omp section
					{//top:dest6行分

						const float* sp = src.ptr<float>(0);
						float* dp = dest.ptr<float>(0);
						for (int x = 0; x < ssize.width - 8; x += 8)
						{
							__m256 mSrc0_0 = _mm256_load_ps(sp);
							__m256 mSrc0_1 = _mm256_loadu_ps(sp + 1);
							__m256 mSrcA = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc0_0, mSrc0_1));

							__m256 mTmpHi = _mm256_unpackhi_ps(mSrc0_0, mSrcA);
							__m256 mTmpLo = _mm256_unpacklo_ps(mSrc0_0, mSrcA);
							__m256 mSrcHi_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp, mSrcLo_A);
							_mm256_store_ps(dp + 8, mSrcHi_A);

							__m256 mSrc1_0 = _mm256_load_ps(sp + ssize.width);
							__m256 mSrc1_1 = _mm256_loadu_ps(sp + ssize.width + 1);
							__m256 mSrcB = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc1_0, mSrc1_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc1_0, mSrcB);
							mTmpLo = _mm256_unpacklo_ps(mSrc1_0, mSrcB);
							__m256 mSrcHi_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_A, mSrcLo_B)));
							_mm256_store_ps(dp + dsize.width + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_A, mSrcHi_B)));

							_mm256_store_ps(dp + (dsize.width << 1), mSrcLo_B);
							_mm256_store_ps(dp + (dsize.width << 1) + 8, mSrcHi_B);

							__m256 mSrc2_0 = _mm256_load_ps(sp + (ssize.width << 1));
							__m256 mSrc2_1 = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
							__m256 mSrcC = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc2_0, mSrc2_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc2_0, mSrcC);
							mTmpLo = _mm256_unpacklo_ps(mSrc2_0, mSrcC);
							__m256 mSrcHi_C = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_C = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width + (dsize.width << 1), _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_B, mSrcLo_C)));
							_mm256_store_ps(dp + dsize.width + (dsize.width << 1) + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_B, mSrcHi_C)));

							_mm256_store_ps(dp + (dsize.width << 2), mSrcLo_C);
							_mm256_store_ps(dp + (dsize.width << 2) + 8, mSrcHi_C);

							__m256 mSrc3_0 = _mm256_load_ps(sp + ssize.width + (ssize.width << 1));
							__m256 mSrc3_1 = _mm256_loadu_ps(sp + ssize.width + (ssize.width << 1) + 1);
							__m256 mSrcD = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc3_0, mSrc3_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc3_0, mSrcD);
							mTmpLo = _mm256_unpacklo_ps(mSrc3_0, mSrcD);
							__m256 mSrcHi_D = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_D = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width + (dsize.width << 2), _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_C, mSrcLo_D)));
							_mm256_store_ps(dp + dsize.width + (dsize.width << 2) + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_C, mSrcHi_D)));//5
							sp += 8;
							dp += 16;
						}
					}//top

		#pragma omp section
					{//left
						const float* sp = src.ptr<float>(3);
						float* dp = dest.ptr<float>(6);
						for (int y = 3; y < ssize.height - 1; ++y)
						{
							__m256 mSrc0_0 = _mm256_load_ps(sp);
							__m256 mSrc0_1 = _mm256_loadu_ps(sp + 1);
							__m256 mSrcA = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc0_0, mSrc0_1));

							__m256 mTmpHi = _mm256_unpackhi_ps(mSrc0_0, mSrcA);
							__m256 mTmpLo = _mm256_unpacklo_ps(mSrc0_0, mSrcA);
							__m256 mSrcHi_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp, mSrcLo_A);
							_mm256_store_ps(dp + 8, mSrcHi_A);

							__m256 mSrc1_0 = _mm256_load_ps(sp + ssize.width);
							__m256 mSrc1_1 = _mm256_loadu_ps(sp + ssize.width + 1);
							__m256 mSrcB = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc1_0, mSrc1_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc1_0, mSrcB);
							mTmpLo = _mm256_unpacklo_ps(mSrc1_0, mSrcB);
							__m256 mSrcHi_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_A, mSrcLo_B)));
							_mm256_store_ps(dp + dsize.width + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_A, mSrcHi_B)));//1

							sp += ssize.width;
							dp += (dsize.width << 1);
						}
					}
		#pragma omp section
					{
						const float* sp = src.ptr<float>(0) + ssize.width - 8;
						float* dp = dest.ptr<float>(0) + dsize.width - 16;
						for (int y = 0; y < ssize.height - 1; ++y)
						{//right:

							__m256 mSrc0_0 = _mm256_load_ps(sp);
							__m256 mSrc0_1 = _mm256_i32gather_ps(sp + 1, rightEnd, sizeof(float));
							__m256 mSrcA = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc0_0, mSrc0_1));

							__m256 mTmpHi = _mm256_unpackhi_ps(mSrc0_0, mSrcA);
							__m256 mTmpLo = _mm256_unpacklo_ps(mSrc0_0, mSrcA);
							__m256 mSrcHi_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp, mSrcLo_A);
							_mm256_store_ps(dp + 8, mSrcHi_A);

							__m256 mSrc1_0 = _mm256_load_ps(sp + ssize.width);
							__m256 mSrc1_1 = _mm256_i32gather_ps(sp + ssize.width + 1, rightEnd, sizeof(float));
							__m256 mSrcB = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc1_0, mSrc1_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc1_0, mSrcB);
							mTmpLo = _mm256_unpacklo_ps(mSrc1_0, mSrcB);
							__m256 mSrcHi_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_A, mSrcLo_B)));
							_mm256_store_ps(dp + dsize.width + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_A, mSrcHi_B)));//1

							sp += ssize.width;
							dp += (dsize.width << 1);

						}
						{//right:一番下の列

							__m256 mSrc0_0 = _mm256_load_ps(sp);
							__m256 mSrc0_1 = _mm256_i32gather_ps(sp + 1, rightEnd, sizeof(float));
							__m256 mSrcA = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc0_0, mSrc0_1));

							__m256 mTmpHi = _mm256_unpackhi_ps(mSrc0_0, mSrcA);
							__m256 mTmpLo = _mm256_unpacklo_ps(mSrc0_0, mSrcA);
							__m256 mSrcHi_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp, mSrcLo_A);
							_mm256_store_ps(dp + 8, mSrcHi_A);

							_mm256_store_ps(dp + dsize.width, mSrcLo_A);
							_mm256_store_ps(dp + dsize.width + 8, mSrcHi_A);

						}
					}

		#pragma omp section
					{
						const float* sp = src.ptr<float>(ssize.height - 3);
						float* dp = dest.ptr<float>(dsize.height - 6);
						//bottom:dest2行分->6行分
						for (int x = 0; x < ssize.width - 8; x += 8)
						{
							__m256 mSrc0_0 = _mm256_load_ps(sp);
							__m256 mSrc0_1 = _mm256_loadu_ps(sp + 1);
							__m256 mSrcA = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc0_0, mSrc0_1));

							__m256 mTmpHi = _mm256_unpackhi_ps(mSrc0_0, mSrcA);
							__m256 mTmpLo = _mm256_unpacklo_ps(mSrc0_0, mSrcA);
							__m256 mSrcHi_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_A = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp, mSrcLo_A);
							_mm256_store_ps(dp + 8, mSrcHi_A);

							__m256 mSrc1_0 = _mm256_load_ps(sp + ssize.width);
							__m256 mSrc1_1 = _mm256_loadu_ps(sp + ssize.width + 1);
							__m256 mSrcB = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc1_0, mSrc1_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc1_0, mSrcB);
							mTmpLo = _mm256_unpacklo_ps(mSrc1_0, mSrcB);
							__m256 mSrcHi_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_B = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_A, mSrcLo_B)));
							_mm256_store_ps(dp + dsize.width + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_A, mSrcHi_B)));

							_mm256_store_ps(dp + (dsize.width << 1), mSrcLo_B);
							_mm256_store_ps(dp + (dsize.width << 1) + 8, mSrcHi_B);

							__m256 mSrc2_0 = _mm256_load_ps(sp + (ssize.width << 1));
							__m256 mSrc2_1 = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
							__m256 mSrcC = _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrc2_0, mSrc2_1));

							mTmpHi = _mm256_unpackhi_ps(mSrc2_0, mSrcC);
							mTmpLo = _mm256_unpacklo_ps(mSrc2_0, mSrcC);
							__m256 mSrcHi_C = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
							__m256 mSrcLo_C = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

							_mm256_store_ps(dp + dsize.width + (dsize.width << 1), _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcLo_B, mSrcLo_C)));
							_mm256_store_ps(dp + dsize.width + (dsize.width << 1) + 8, _mm256_mul_ps(mDiv_2, _mm256_add_ps(mSrcHi_B, mSrcHi_C)));

							_mm256_store_ps(dp + (dsize.width << 2), mSrcLo_C);
							_mm256_store_ps(dp + (dsize.width << 2) + 8, mSrcHi_C);

							_mm256_store_ps(dp + dsize.width + (dsize.width << 2), mSrcLo_C);
							_mm256_store_ps(dp + dsize.width + (dsize.width << 2) + 8, mSrcHi_C);

							sp += 8;
							dp += 16;
						}
					}
				}
			}//線形補間
		#endif
		*/
		//------------------ DCCI ----------------------------------------------------------------------

		const int threadsNum = ompNumThreads;
#pragma omp parallel for num_threads(threadsNum) schedule(dynamic)
		for (int i = 1; i <= threadsNum; ++i)//threads
		{
			const int tempSize = ssize.width + 3;
			const __m256 signMask = _mm256_set1_ps(-0.0f); // 0x80000000
			const __m256 mThreshold = _mm256_set1_ps(threshold);
			const __m256 mOnef = _mm256_set1_ps(1.f);
			const __m256 cciCoeff_1 = _mm256_set1_ps(-1.f / 16);
			const __m256 cciCoeff_9 = _mm256_set1_ps(9.f / 16);

			const int buffer_width = ssize.width + 8;

#ifdef _USE_GLOBAL_BUFFER_
			const int thidx = omp_get_thread_num();
			if (bfsize_global[thidx] != buffer_width * 4)
			{
				bfsize_global[thidx] = buffer_width * 4;
				//_mm_free(buffer_global[thidx]);
				buffer_global[thidx] = (float*)_mm_malloc(sizeof(float) * buffer_width * 4, 32);
			}
			float* buffer = buffer_global[thidx];
#else
			float* const buffer = (float*)_mm_malloc(sizeof(float) * buffer_width * 4, 32);
			//cv::AutoBuffer <float> buffer(buffer_width * 4);
#endif
			std::array<float*, 4>buf =
			{
				buffer,
				buffer + buffer_width,
				buffer + buffer_width * 2,
				buffer + buffer_width * 3
			};

			float* tp = nullptr;

			int threadsWidth = (ssize.height / threadsNum);
			int start = (i == 1 ? 0 : threadsWidth * (i - 1) - 3);
			int end = start + 3;
			//バッファーの初期設定：tempを本来一段階目で補間される画素値で埋める．
			for (int y = start; y < end; ++y)
			{
				const float* sp = src.ptr<float>(y, 4);

				{
					//一列目
					float grad45 = 0.f, grad135 = 0.f;

					sp += (ssize.width - 2);
					//  0   4   8   C
					//  1   5   9   D
					//  2   6   A   E
					//  3   7   B   F

					//UpRight G1
					const __m256 mK4 = _mm256_loadu_ps(sp - ssize.width);
					const __m256 mK1 = _mm256_loadu_ps(sp - 1);
					__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1));

					const __m256 mK8 = _mm256_loadu_ps(sp - ssize.width + 1);
					const __m256 mK5 = _mm256_loadu_ps(sp);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));

					const __m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);
					const __m256 mK9 = _mm256_loadu_ps(sp + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));

					const __m256 mK2 = _mm256_loadu_ps(sp + ssize.width - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));

					const __m256 mK6 = _mm256_loadu_ps(sp + ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));

					const __m256 mKD = _mm256_loadu_ps(sp + 2);
					const __m256 mKA = _mm256_loadu_ps(sp + ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));

					const __m256 mK3 = _mm256_loadu_ps(sp + (ssize.width << 1) - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));

					const __m256 mK7 = _mm256_loadu_ps(sp + (ssize.width << 1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));

					const __m256 mKB = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
					const __m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

					//DownRight G2
					const __m256 mK0 = _mm256_loadu_ps(sp - ssize.width - 1);
					__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));

					const __m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));


					/* --- 2(b) --- */
					__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);

					//if (1+G1) / (1+G2) > T then 135
					__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					__m256 pxUpRight = _mm256_add_ps(mK3, mKC);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK6, mK9), pxUpRight);

					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					__m256 pxDownRight = _mm256_add_ps(mK0, mKF);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK5, mKA), pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp2 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp2 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp2);
					__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp2);
					mTmp0 = _mm256_add_ps(weight1, weight2);

					//0で最初の引数をとる
					__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_store_ps(buf[0], mDst);
					sp += 3;


					//  C G K O
					//  D H L P
					//  E I M Q
					//  F J N R	

					//UpRight G1
					const __m256 mKG = _mm256_loadu_ps(sp - ssize.width);
					gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

					const __m256 mKH = _mm256_loadu_ps(sp);
					const __m256 mKK = _mm256_loadu_ps(sp - ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

					const __m256 mKO = _mm256_loadu_ps(sp - ssize.width + 2);
					const __m256 mKL = _mm256_loadu_ps(sp + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

					const __m256 mKI = _mm256_loadu_ps(sp + ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

					const __m256 mKP = _mm256_loadu_ps(sp + 2);
					const __m256 mKM = _mm256_loadu_ps(sp + ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));

					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

					const __m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

					const __m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
					const __m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

					//DownRight G2
					gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

					const __m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));

					/* --- 2(b) --- */
					mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
					//if (1+G1) / (1+G2) > T then 135
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					pxUpRight = _mm256_add_ps(mKF, mKO);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKI, mKL), pxUpRight);

					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					pxDownRight = _mm256_add_ps(mKC, mKR);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKH, mKM), pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);
					mTmp0 = _mm256_add_ps(weight1, weight2);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_storeu_ps(buf[0] + 3, mDst);
					sp -= (ssize.width - 7);
					buf[0] += 8;
					buf[1] += 8;
					buf[2] += 8;
				}//x

				for (int x = 0; x < ssize.width - 16; x += 8)//バッファー埋め一列目以降；右下の●だけ求める処理
				{
					//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
					sp += (ssize.width - 2);
					const __m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);

					//  C G K O
					//  D H L P
					//  E I M Q
					//  F J N R	
					//UpRight G1
					const __m256 mKD = _mm256_loadu_ps(sp + 2);
					const __m256 mKG = _mm256_loadu_ps(sp - ssize.width + 3);
					__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

					const __m256 mKH = _mm256_loadu_ps(sp + 3);
					const __m256 mKK = _mm256_loadu_ps(sp - ssize.width + 4);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

					const __m256 mKO = _mm256_loadu_ps(sp - ssize.width + 5);
					const __m256 mKL = _mm256_loadu_ps(sp + 4);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

					const __m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

					const __m256 mKI = _mm256_loadu_ps(sp + ssize.width + 3);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

					const __m256 mKM = _mm256_loadu_ps(sp + ssize.width + 4);
					const __m256 mKP = _mm256_loadu_ps(sp + 5);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));

					const __m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

					const __m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1) + 3);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

					const __m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 4);
					const __m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 5);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

					//DownRight G2

					__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

					const __m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 5);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));

					sp += 3;

					/* --- 2(b) --- */
					__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);

					//if (1+G1) / (1+G2) > T then 135
					__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					__m256 pxUpRight = _mm256_add_ps(mKF, mKO);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKI, mKL), pxUpRight);

					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					__m256 pxDownRight = _mm256_add_ps(mKC, mKR);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKH, mKM), pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1 + w2p2) / (w1 + w2)
					mTmp2 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp2 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp2);
					__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp2);

					//0で最初の引数をとる
					__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_storeu_ps(buf[0] + 3, mDst);

					sp -= (ssize.width - 7);

					buf[0] += 8;
					buf[1] += 8;
					buf[2] += 8;
				}//x
				ringbuffering3(buf[0], buf[2], buf[1]);

				buf[0] -= tempSize - 11;
				buf[1] -= tempSize - 11;
				buf[2] -= tempSize - 11;
			}//y
			ringbuffering2(buf[0], buf[3]);

			//DCCIメイン処理
			start = end;
			end = (i == threadsNum ? i * threadsWidth + ssize.height % threadsNum - 3 : i * threadsWidth);
			for (int y = start; y < end; ++y)
			{
				const float* sp = src.ptr<float>(y, 4);
				float* dp = dest.ptr<float>(y * 2, 8);//align(8)

				{
					//一列目の処理．左下，右下の●を求める必要がある
					//（奇数，奇数）の補間
					dp += dsize.width + 1;

					//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
					sp += (ssize.width - 2);
					dp += ((dsize.width << 1) - 4);
					//  1   5   9   D
					//  0   4   8   C
					//  2   6   A   E
					//  3   7   B   F

					//UpRight G1
#if 0 //interleave
					const __m256 mK1 = _mm256_loadu_ps(sp - 1);
					const __m256 mK4 = _mm256_loadu_ps(sp - ssize.width);
					__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1));

					const __m256 mK5 = _mm256_loadu_ps(sp);
					const __m256 mK8 = _mm256_loadu_ps(sp - ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));

					const __m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);
					const __m256 mK9 = _mm256_loadu_ps(sp + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));

					const __m256 mK2 = _mm256_loadu_ps(sp + ssize.width - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));

					const __m256 mK6 = _mm256_loadu_ps(sp + ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));

					const __m256 mKA = _mm256_loadu_ps(sp + ssize.width + 1);
					const __m256 mKD = _mm256_loadu_ps(sp + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));

					const __m256 mK3 = _mm256_loadu_ps(sp + (ssize.width << 1) - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));

					const __m256 mK7 = _mm256_loadu_ps(sp + (ssize.width << 1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));

					const __m256 mKB = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
					const __m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

					//DownRight G2
					const __m256 mK0 = _mm256_loadu_ps(sp - ssize.width - 1);
					__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));
					const __m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));
#else
					const __m256 mK0 = _mm256_loadx_ps(sp - ssize.width - 1);
					const __m256 mK4 = _mm256_loadx_ps(sp - ssize.width);
					const __m256 mK8 = _mm256_loadx_ps(sp - ssize.width + 1);
					const __m256 mKC = _mm256_loadx_ps(sp - ssize.width + 2);
					const __m256 mK1 = _mm256_loadx_ps(sp - 1);
					const __m256 mK5 = _mm256_loadx_ps(sp);
					const __m256 mK9 = _mm256_loadx_ps(sp + 1);
					const __m256 mKD = _mm256_loadx_ps(sp + 2);
					const __m256 mK3 = _mm256_loadx_ps(sp + (ssize.width << 1) - 1);
					const __m256 mK7 = _mm256_loadx_ps(sp + (ssize.width << 1));
					const __m256 mKB = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
					const __m256 mKF = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);
					const __m256 mK2 = _mm256_loadx_ps(sp + ssize.width - 1);
					const __m256 mK6 = _mm256_loadx_ps(sp + ssize.width);
					const __m256 mKA = _mm256_loadx_ps(sp + ssize.width + 1);
					const __m256 mKE = _mm256_loadx_ps(sp + ssize.width + 2);

					__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

					//DownRight G2
					__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));
#endif
					/* --- 2(b) --- */
					__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
					//if (1+G1) / (1+G2) > T then 135
					__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					__m256 pxUpRight = _mm256_add_ps(mK3, mKC);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mK6, mK9), cciCoeff_9, pxUpRight);

					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					__m256 pxDownRight = _mm256_add_ps(mK0, mKF);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mK5, mKA), cciCoeff_9, pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
					__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);


					//0で最初の引数をとる
					__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_store_ps(buf[0], mDst);

					sp += 3;
					dp += 6;

					//  C G K O
					//  D H L P
					//  E I M Q
					//  F J N R	

#if 0 //interleave
				//UpRight G1
					const __m256 mKG = _mm256_loadu_ps(sp - ssize.width);
					gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

					const __m256 mKH = _mm256_loadu_ps(sp);
					const __m256 mKK = _mm256_loadu_ps(sp - ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

					const __m256 mKO = _mm256_loadu_ps(sp - ssize.width + 2);
					const __m256 mKL = _mm256_loadu_ps(sp + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

					const __m256 mKI = _mm256_loadu_ps(sp + ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

					const __m256 mKP = _mm256_loadu_ps(sp + 2);
					const __m256 mKM = _mm256_loadu_ps(sp + ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));

					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

					const __m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

					const __m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
					const __m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

					//DownRight G2
					gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

					const __m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
#else
				//UpRight G1
					const __m256 mKG = _mm256_loadx_ps(sp - ssize.width);
					const __m256 mKK = _mm256_loadx_ps(sp - ssize.width + 1);
					const __m256 mKO = _mm256_loadx_ps(sp - ssize.width + 2);
					const __m256 mKH = _mm256_loadx_ps(sp);
					const __m256 mKL = _mm256_loadx_ps(sp + 1);
					const __m256 mKP = _mm256_loadx_ps(sp + 2);
					const __m256 mKJ = _mm256_loadx_ps(sp + (ssize.width << 1));
					const __m256 mKN = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
					const __m256 mKR = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);
					const __m256 mKI = _mm256_loadx_ps(sp + ssize.width);
					const __m256 mKM = _mm256_loadx_ps(sp + ssize.width + 1);
					const __m256 mKQ = _mm256_loadx_ps(sp + ssize.width + 2);

					gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

					//DownRight G2
					gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
#endif
					/* --- 2(b) --- */
					mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
					//if (1+G1) / (1+G2) > T then 135
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					//mTmp2 = _mm256_div_ps(mTmp0, mTmp1);
					maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					//mTmp2 = _mm256_div_ps(mTmp1, mTmp0);
					maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					pxUpRight = _mm256_add_ps(mKF, mKO);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mKI, mKL), cciCoeff_9, pxUpRight);

					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					pxDownRight = _mm256_add_ps(mKC, mKR);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mKH, mKM), cciCoeff_9, pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_storeu_ps(buf[0] + 3, mDst);

					sp -= (ssize.width + 1);
					dp -= ((dsize.width << 1) + 2);

					/*(o,e)を補間-------------------------------------
					#:OddEven		@:EvenOdd
					S		|		X
					X X X X X	|	X S X T X
					8 X C @ G	|	X X X X X
					x X x # x X x	| X X C @ G X X
					9 X D X H	|	X # x X X
					X t X t X	|	X D X H X
					E		|		t		*/

					//horizontal
					const __m256 mKd1 = _mm256_loadu_ps(buf[2] + 1);
					const __m256 mKd2 = _mm256_loadu_ps(buf[2] + 2);

					__m256 gradHorizontal = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKC)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));

					const __m256 mKx1 = _mm256_loadu_ps(buf[1] + 1);
					const __m256 mKx2 = _mm256_loadu_ps(buf[1] + 2);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKD)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

					const __m256 mKt1 = _mm256_loadu_ps(buf[0] + 1);
					const __m256 mKt2 = _mm256_loadu_ps(buf[0] + 2);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKt1, mKt2)));

					//Vertical

					__m256 gradVertical = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK9));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKt1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKt2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));

					//Horizontal方向にエッジがある場合，Horizontal方向に補間
					const __m256 mKx0 = _mm256_load_ps(buf[1] + 0);
					const __m256 mKx3 = _mm256_loadu_ps(buf[1] + 3);
					__m256 pxHorizontal = _mm256_add_ps(mKx0, mKx3);
					pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
					pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKx1, mKx2), cciCoeff_9, pxHorizontal);

					//Vertical方向にエッジがある場合，Vertical方向に補間
					const __m256 mKS = _mm256_loadu_ps(sp - ssize.width);
					__m256 pxVertical = _mm256_add_ps(mKS, mKE);
					pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
					pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKD), cciCoeff_9, pxVertical);

					//weight = 1 / (1+G^5)
					//weight1はgradHorizontalを使う
					weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradVerticalを使う
					weight2 = _mm256_mul_ps(gradVertical, gradVertical);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
					mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);
					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
					mTmp1 = _mm256_add_ps(mOnef, gradVertical);
					//if (1+G1) / (1+G2) > T then 135
					//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					//cmpの結果を論理演算に使うとバグる
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
					mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

					//mk7:1234 5678
					//dst:abcd efgh-->a1b2c3d4...=
					__m256 mTmpHi = _mm256_unpackhi_ps(mDst, mKx2);//a 1 b 2 / e 5 f 6
					__m256 mTmpLo = _mm256_unpacklo_ps(mDst, mKx2);//c 3 d 4 / g 7 h 8
					__m256 mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);//1 0 2 0 / 5 0 6 0
					__m256 mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);//3 0 4 0 / 7 0 8 0

					_mm256_store_ps(dp - 1, mSrcLo);
					_mm256_store_ps(dp + 7, mSrcHi);

					//horizontal
					gradHorizontal = _mm256_setzero_ps();
					__m256 mKT = _mm256_loadu_ps(sp - ssize.width + 1);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKT)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2)));
					__m256 mKd3 = _mm256_loadu_ps(buf[2] + 3);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKd3)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKx3)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

					//Vertical
					gradVertical = _mm256_setzero_ps();
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKC)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKT, mKG)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd3, mKx3)));

					//Horizontal方向にエッジがある場合，Horizontal方向に補間
					pxHorizontal = _mm256_add_ps(mK8, mKK);
					pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
					pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKG), cciCoeff_9, pxHorizontal);

					//Vertical方向にエッジがある場合，Vertical方向に補間
					__m256 mKd0 = _mm256_loadu_ps(buf[3] + 2);//1 0 2 0 / 3 0 4 0
					pxVertical = _mm256_add_ps(mKd0, mKt2);
					pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
					pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKd2, mKx2), cciCoeff_9, pxVertical);

					//weight = 1 / (1+G^5)
					//weight1はgradHorizontalを使う
					weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradVerticalを使う
					weight2 = _mm256_mul_ps(gradVertical, gradVertical);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1 + w2p2) / (w1 + w2)
					mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
					mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);
					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
					mTmp1 = _mm256_add_ps(mOnef, gradVertical);

					//if (1+G1) / (1+G2) > T then 135
					//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					//cmpの結果を論理演算に使うとバグる
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
					mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

					mTmpHi = _mm256_unpackhi_ps(mKC, mDst);//a 1 b 2 / e 5 f 6
					mTmpLo = _mm256_unpacklo_ps(mKC, mDst);//c 3 d 4 / g 7 h 8
					mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);//1 0 2 0 / 5 0 6 0
					mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);//3 0 4 0 / 7 0 8 0

					_mm256_store_ps(dp - 1 - dsize.width, mSrcLo);
					_mm256_store_ps(dp + 7 - dsize.width, mSrcHi);

					sp += 8;
					dp -= (dsize.width - 15);
					buf[0] += 8;
					buf[1] += 8;
					buf[2] += 8;
					buf[3] += 8;
				}//ここまでがmainの一列目の処理

				for (int x = 0; x < ssize.width - 16; x += 8)//2列目以降．●は右下だけでいい
				{
					float grad45 = 0.f, grad135 = 0.f;

					//（奇数，奇数）の補間
					//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
					sp += (ssize.width + 1);
					dp += (dsize.width + (dsize.width << 1) + 3);
					//  0   4   8   C
					//  1   5   9   D
					//  2   6   A   E
					//  3   7   B   F

					//  C G K O
					//  D H L P
					//  E I M Q
					//  F J N R

#if 0 //interleave
				//UpRight G1
					const __m256 mKG = _mm256_loadx_ps(sp - ssize.width);
					const __m256 mKD = _mm256_loadx_ps(sp - 1);

					__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

					const __m256 mKH = _mm256_loadx_ps(sp);
					const __m256 mKK = _mm256_loadx_ps(sp - ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

					const __m256 mKO = _mm256_loadx_ps(sp - ssize.width + 2);
					const __m256 mKL = _mm256_loadx_ps(sp + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

					__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL));//

					const __m256 mKE = _mm256_loadx_ps(sp + ssize.width - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

					const __m256 mKI = _mm256_loadx_ps(sp + ssize.width);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));//

					const __m256 mKP = _mm256_loadx_ps(sp + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));//
					const __m256 mKM = _mm256_loadx_ps(sp + ssize.width + 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));//

					const __m256 mKF = _mm256_loadx_ps(sp + (ssize.width << 1) - 1);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

					const __m256 mKJ = _mm256_loadx_ps(sp + (ssize.width << 1));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));//

					const __m256 mKN = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));//
					const __m256 mKQ = _mm256_loadx_ps(sp + ssize.width + 2);
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));//

					//DownRight G2
					const __m256 mKC = _mm256_loadx_ps(sp - ssize.width - 1);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));


					const __m256 mKR = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
#else
				//UpRight G1
					const __m256 mKC = _mm256_loadx_ps(sp - ssize.width - 1);
					const __m256 mKG = _mm256_loadx_ps(sp - ssize.width);
					const __m256 mKK = _mm256_loadx_ps(sp - ssize.width + 1);
					const __m256 mKO = _mm256_loadx_ps(sp - ssize.width + 2);
					const __m256 mKD = _mm256_loadx_ps(sp - 1);
					const __m256 mKH = _mm256_loadx_ps(sp);
					const __m256 mKL = _mm256_loadx_ps(sp + 1);
					const __m256 mKP = _mm256_loadx_ps(sp + 2);
					const __m256 mKE = _mm256_loadx_ps(sp + ssize.width - 1);
					const __m256 mKI = _mm256_loadx_ps(sp + ssize.width);
					const __m256 mKM = _mm256_loadx_ps(sp + ssize.width + 1);
					const __m256 mKQ = _mm256_loadx_ps(sp + ssize.width + 2);
					const __m256 mKF = _mm256_loadx_ps(sp + (ssize.width << 1) - 1);
					const __m256 mKJ = _mm256_loadx_ps(sp + (ssize.width << 1));
					const __m256 mKN = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
					const __m256 mKR = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);

					__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));
					gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

					//DownRight G2
					__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL));//
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));//
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));//
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));//
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));//
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));//
					gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));//
#endif
					/* --- 2(b) --- */
					__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
					__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
					//if (1+G1) / (1+G2) > T then 135
					__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					/* --- 2(c) --- */
					//UpRight方向にエッジがある場合，UpRight方向に補間 p1
					__m256 pxUpRight = _mm256_add_ps(mKF, mKO);
					pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
					pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mKI, mKL), cciCoeff_9, pxUpRight);


					//DownRight方向にエッジがある場合，DownRight方向に補間 p2
					__m256 pxDownRight = _mm256_add_ps(mKC, mKR);
					pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
					pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mKH, mKM), cciCoeff_9, pxDownRight);

					//weight = 1 / (1+G^5)
					//weight1はgradUpRightを使う
					__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradDownRightを使う
					__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
					mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
					__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					//0で最初の引数をとる
					__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
					mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

					_mm256_storeu_ps(buf[0] + 3, mDst);

					sp -= (ssize.width + 1);
					dp -= ((dsize.width << 1) + 2);

					//(o,e)を補間-------------------------------------
					//		#:OddEven		@:EvenOdd
					//			S		|		X		
					//		X X X X X	|	X S X T X	
					//		8 X C @ G	|	X X X X X	
					//	  x X x # x X x	| X X C @ G X X
					//		9 X D X H	|	X # x X X	
					//		X t X t X	|	X D X H X	
					//			E		|		t	


					//horizontal
					const __m256 mKd1 = _mm256_loadx_ps(buf[2] + 1);
					const __m256 mKd2 = _mm256_loadx_ps(buf[2] + 2);
					__m256 gradHorizontal = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2));

					const __m256 mK8 = _mm256_loadx_ps(sp - 1);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKC)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));

					const __m256 mKx1 = _mm256_loadx_ps(buf[1] + 1);
					const __m256 mKx2 = _mm256_loadx_ps(buf[1] + 2);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
					const __m256 mK9 = _mm256_loadx_ps(sp + ssize.width - 1);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKD)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

					const __m256 mKt1 = _mm256_loadx_ps(buf[0] + 1);
					const __m256 mKt2 = _mm256_loadx_ps(buf[0] + 2);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKt1, mKt2)));

					//Vertical
					__m256 gradVertical = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK9));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKt1)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKt2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));

					//Horizontal方向にエッジがある場合，Horizontal方向に補間
					const __m256 mKx0 = _mm256_load_ps(buf[1]);
					const __m256 mKx3 = _mm256_loadx_ps(buf[1] + 3);
					__m256 pxHorizontal = _mm256_add_ps(mKx0, mKx3);
					pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
					pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKx1, mKx2), cciCoeff_9, pxHorizontal);

					//Vertical方向にエッジがある場合，Vertical方向に補間
					const __m256 mKS = _mm256_loadx_ps(sp - ssize.width);
					__m256 pxVertical = _mm256_add_ps(mKS, mKE);
					pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
					pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKD), cciCoeff_9, pxVertical);

					//weight = 1 / (1+G^5)
					//weight1はgradHorizontalを使う
					weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradVerticalを使う
					weight2 = _mm256_mul_ps(gradVertical, gradVertical);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1+w2p2) / (w1+w2)
					mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
					mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);

					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
					mTmp1 = _mm256_add_ps(mOnef, gradVertical);

					//if (1+G1) / (1+G2) > T then 135
					//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					__m256 maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					//cmpの結果を論理演算に使うとバグる
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					__m256 maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
					mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

					//mk7:1234 5678
					//dst:abcd efgh-->a1b2c3d4...=
					__m256 mTmpHi = _mm256_unpackhi_ps(mDst, mKx2);//a 1 b 2 / e 5 f 6
					__m256 mTmpLo = _mm256_unpacklo_ps(mDst, mKx2);//c 3 d 4 / g 7 h 8
					__m256 mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);//1 0 2 0 / 5 0 6 0
					__m256 mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);//3 0 4 0 / 7 0 8 0

					_mm256_storeu_ps(dp - 1, mSrcLo);
					_mm256_storeu_ps(dp + 7, mSrcHi);

					//horizontal
					const __m256 mKT = _mm256_loadx_ps(sp - ssize.width + 1);
					gradHorizontal = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKT));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2)));

					const __m256 mKd3 = _mm256_loadx_ps(buf[2] + 3);
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKd3)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKx3)));
					gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

					//Vertical
					gradVertical = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKC)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKT, mKG)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));
					gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd3, mKx3)));

					//Horizontal方向にエッジがある場合，Horizontal方向に補間
					pxHorizontal = _mm256_add_ps(mK8, mKK);
					pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
					pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKG), cciCoeff_9, pxHorizontal);


					//Vertical方向にエッジがある場合，Vertical方向に補間
					const __m256 mKd0 = _mm256_loadx_ps(buf[3] + 2);//1 0 2 0 / 3 0 4 0
					pxVertical = _mm256_add_ps(mKd0, mKt2);
					pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
					pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKd2, mKx2), cciCoeff_9, pxVertical);

					//weight = 1 / (1+G^5)
					//weight1はgradHorizontalを使う
					weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
					weight1 = _mm256_mul_ps(weight1, weight1);
					weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
					weight1 = _mm256_rcp_ps(weight1);

					//weight2はgradVerticalを使う
					weight2 = _mm256_mul_ps(gradVertical, gradVertical);
					weight2 = _mm256_mul_ps(weight2, weight2);
					weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
					weight2 = _mm256_rcp_ps(weight2);

					//p = (w1p1 + w2p2) / (w1 + w2)
					mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
					mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);
					pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
					pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

					mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
					mTmp1 = _mm256_add_ps(mOnef, gradVertical);

					//if (1+G1) / (1+G2) > T then 135
					//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
					mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
					maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//if (1+G2) / (1+G1) > T then 45
					//cmpの結果を論理演算に使うとバグる
					mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
					maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

					//0で最初の引数をとる
					mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
					mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

					mTmpHi = _mm256_unpackhi_ps(mKC, mDst);
					mTmpLo = _mm256_unpacklo_ps(mKC, mDst);
					mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
					mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

					_mm256_storeu_ps(dp - 1 - dsize.width, mSrcLo);
					_mm256_storeu_ps(dp - dsize.width + 7, mSrcHi);//koko ha store yori hayai kamo
					//_mm256_storeu_ps(dp - 1 - dsize.width, mSrcLo);
					//_mm256_storeu_ps(dp - dsize.width + 7, mSrcHi);//koko ha store yori hayai kamo

					sp += 8;
					dp -= (dsize.width - 15);
					buf[0] += 8;
					buf[1] += 8;
					buf[2] += 8;
					buf[3] += 8;
				}//x メインループの2列目以降

				buf[0] -= tempSize - 11; buf[1] -= tempSize - 11; buf[2] -= tempSize - 11; buf[3] -= tempSize - 11;
				ringbuffering4(buf[0], buf[3], buf[2], buf[1]);
			}//y

#ifndef _USE_GLOBAL_BUFFER_
			_mm_free(buffer);
#endif
		}//parallel
	}

	class DCCI32FC1_SIMD_LoopFusionInvorkerOld : public ParallelLoopBody
	{
		Mat src;
		Mat& dest;
		const float threshold;
		Size ssize;
		Size dsize;
		const int division;
	public:
		DCCI32FC1_SIMD_LoopFusionInvorkerOld(const Mat& src_, Mat& dest, const float threshold, const int division) : src(src_), dest(dest), threshold(threshold), division(division)
		{
			src = src_.data == dest.data ? src_.clone() : src_;
			ssize = src.size();
			if (dest.size() != src.size() * 2)dest.create(ssize.height << 1, ssize.width << 1, CV_32FC1);
			dsize = dest.size();
		}

		void operator()(const Range& range) const override
		{
			const __m256 signMask = _mm256_set1_ps(-0.0f); // 0x80000000
			const __m256 mThreshold = _mm256_set1_ps(threshold);
			const __m256 mOnef = _mm256_set1_ps(1.f);
			const __m256 cciCoeff_1 = _mm256_set1_ps(-1.f / 16);
			const __m256 cciCoeff_9 = _mm256_set1_ps(9.f / 16);
			const int tempSize = ssize.width + 3;
			const int buffer_width = ssize.width + 8;
			float* const buffer = (float*)_mm_malloc(sizeof(float) * buffer_width * 4, 32);
			std::array<float*, 4>buf =
			{
				buffer,
				buffer + buffer_width,
				buffer + buffer_width * 2,
				buffer + buffer_width * 3
			};
			const int threadsWidth = (ssize.height / division);

			for (int i = range.start + 1; i <= range.end; i++)//threads
			{
				float* tp = nullptr;

				int start = (i == 1 ? 0 : threadsWidth * (i - 1) - 3);
				int end = start + 3;
				//バッファーの初期設定：tempを本来一段階目で補間される画素値で埋める．
				for (int y = start; y < end; ++y)
				{
					const float* sp = src.ptr<float>(y) + 4;

					{
						//一列目

						sp += (ssize.width - 2);
						//  0   4   8   C
						//  1   5   9   D
						//  2   6   A   E
						//  3   7   B   F

						//UpRight G1
						const __m256 mK4 = _mm256_loadu_ps(sp - ssize.width);
						const __m256 mK1 = _mm256_loadu_ps(sp - 1);
						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1));

						const __m256 mK8 = _mm256_loadu_ps(sp - ssize.width + 1);
						const __m256 mK5 = _mm256_loadu_ps(sp);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));

						const __m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);
						const __m256 mK9 = _mm256_loadu_ps(sp + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));

						const __m256 mK2 = _mm256_loadu_ps(sp + ssize.width - 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));

						const __m256 mK6 = _mm256_loadu_ps(sp + ssize.width);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));

						const __m256 mKD = _mm256_loadu_ps(sp + 2);
						const __m256 mKA = _mm256_loadu_ps(sp + ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));

						const __m256 mK3 = _mm256_loadu_ps(sp + (ssize.width << 1) - 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));

						const __m256 mK7 = _mm256_loadu_ps(sp + (ssize.width << 1));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));

						const __m256 mKB = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

						//DownRight G2
						const __m256 mK0 = _mm256_loadu_ps(sp - ssize.width - 1);
						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));

						const __m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));


						/* --- 2(b) --- */
						__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
						__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);

						//if (1+G1) / (1+G2) > T then 135
						__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						/* --- 2(c) --- */
						//UpRight方向にエッジがある場合，UpRight方向に補間 p1
						__m256 pxUpRight = _mm256_add_ps(mK3, mKC);
						pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
						pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK6, mK9), pxUpRight);

						//DownRight方向にエッジがある場合，DownRight方向に補間 p2
						__m256 pxDownRight = _mm256_add_ps(mK0, mKF);
						pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
						pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK5, mKA), pxDownRight);

						//weight = 1 / (1+G^5)
						//weight1はgradUpRightを使う
						__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradDownRightを使う
						__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1+w2p2) / (w1+w2)
						mTmp2 = _mm256_mul_ps(weight1, pxUpRight);
						mTmp2 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp2);
						__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp2);
						mTmp0 = _mm256_add_ps(weight1, weight2);

						//0で最初の引数をとる
						__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
						mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

						_mm256_store_ps(buf[0], mDst);
						sp += 3;


						//  C G K O
						//  D H L P
						//  E I M Q
						//  F J N R	

						//UpRight G1
						const __m256 mKG = _mm256_loadu_ps(sp - ssize.width);
						gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

						const __m256 mKH = _mm256_loadu_ps(sp);
						const __m256 mKK = _mm256_loadu_ps(sp - ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

						const __m256 mKO = _mm256_loadu_ps(sp - ssize.width + 2);
						const __m256 mKL = _mm256_loadu_ps(sp + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

						const __m256 mKI = _mm256_loadu_ps(sp + ssize.width);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

						const __m256 mKP = _mm256_loadu_ps(sp + 2);
						const __m256 mKM = _mm256_loadu_ps(sp + ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));

						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

						const __m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

						const __m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

						//DownRight G2
						gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

						const __m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));

						/* --- 2(b) --- */
						mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
						mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
						//if (1+G1) / (1+G2) > T then 135
						mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						/* --- 2(c) --- */
						//UpRight方向にエッジがある場合，UpRight方向に補間 p1
						pxUpRight = _mm256_add_ps(mKF, mKO);
						pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
						pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKI, mKL), pxUpRight);

						//DownRight方向にエッジがある場合，DownRight方向に補間 p2
						pxDownRight = _mm256_add_ps(mKC, mKR);
						pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
						pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKH, mKM), pxDownRight);

						//weight = 1 / (1+G^5)
						//weight1はgradUpRightを使う
						weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradDownRightを使う
						weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1+w2p2) / (w1+w2)
						mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
						mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
						pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);
						mTmp0 = _mm256_add_ps(weight1, weight2);

						//0で最初の引数をとる
						mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
						mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

						_mm256_storeu_ps(buf[0] + 3, mDst);
						sp -= (ssize.width - 7);
						buf[0] += 8;
						buf[1] += 8;
						buf[2] += 8;
					}//x


					for (int x = 0; x < ssize.width - 16; x += 8)//バッファー埋め一列目以降；右下の●だけ求める処理
					{
						float grad45 = 0.f, grad135 = 0.f;

						//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
						sp += (ssize.width - 2);
						const __m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);

						//  C G K O
						//  D H L P
						//  E I M Q
						//  F J N R	
						//UpRight G1
						const __m256 mKD = _mm256_loadu_ps(sp + 2);
						const __m256 mKG = _mm256_loadu_ps(sp - ssize.width + 3);
						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

						const __m256 mKH = _mm256_loadu_ps(sp + 3);
						const __m256 mKK = _mm256_loadu_ps(sp - ssize.width + 4);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

						const __m256 mKO = _mm256_loadu_ps(sp - ssize.width + 5);
						const __m256 mKL = _mm256_loadu_ps(sp + 4);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

						const __m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

						const __m256 mKI = _mm256_loadu_ps(sp + ssize.width + 3);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

						const __m256 mKM = _mm256_loadu_ps(sp + ssize.width + 4);
						const __m256 mKP = _mm256_loadu_ps(sp + 5);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));

						const __m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

						const __m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1) + 3);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

						const __m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 4);
						const __m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 5);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

						//DownRight G2

						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

						const __m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 5);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));

						sp += 3;

						/* --- 2(b) --- */
						__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
						__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);

						//if (1+G1) / (1+G2) > T then 135
						__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						/* --- 2(c) --- */
						//UpRight方向にエッジがある場合，UpRight方向に補間 p1
						__m256 pxUpRight = _mm256_add_ps(mKF, mKO);
						pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
						pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKI, mKL), pxUpRight);

						//DownRight方向にエッジがある場合，DownRight方向に補間 p2
						__m256 pxDownRight = _mm256_add_ps(mKC, mKR);
						pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
						pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKH, mKM), pxDownRight);

						//weight = 1 / (1+G^5)
						//weight1はgradUpRightを使う
						__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradDownRightを使う
						__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1 + w2p2) / (w1 + w2)
						mTmp2 = _mm256_mul_ps(weight1, pxUpRight);
						mTmp2 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp2);
						__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp2);

						//0で最初の引数をとる
						__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
						mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

						_mm256_storeu_ps(buf[0] + 3, mDst);

						sp -= (ssize.width - 7);

						buf[0] += 8;
						buf[1] += 8;
						buf[2] += 8;
					}//x
					tp = buf[0];
					buf[0] = buf[2];
					buf[2] = buf[1];
					buf[1] = tp;

					buf[0] -= tempSize - 11;
					buf[1] -= tempSize - 11;
					buf[2] -= tempSize - 11;
				}//y

				tp = buf[0];
				buf[0] = buf[3];
				buf[3] = tp;

				//DCCIメイン処理
				start = end;
				end = (i == division ? i * threadsWidth + ssize.height % division - 3 : i * threadsWidth);
				for (int y = start; y < end; ++y)
				{
					const float* sp = src.ptr<float>(y) + 4;
					float* dp = (float*)dest.ptr<float>((y) << 1) + 8;//align(8)
					{
						//一列目の処理．左下，右下の●を求める必要がある
						//（奇数，奇数）の補間
						dp += dsize.width + 1;

						//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
						sp += (ssize.width - 2);
						dp += ((dsize.width << 1) - 4);
						//  1   5   9   D
						//  0   4   8   C
						//  2   6   A   E
						//  3   7   B   F

						//UpRight G1
#if 0 //interleave
						const __m256 mK1 = _mm256_loadu_ps(sp - 1);
						const __m256 mK4 = _mm256_loadu_ps(sp - ssize.width);
						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1));

						const __m256 mK5 = _mm256_loadu_ps(sp);
						const __m256 mK8 = _mm256_loadu_ps(sp - ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));

						const __m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);
						const __m256 mK9 = _mm256_loadu_ps(sp + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));

						const __m256 mK2 = _mm256_loadu_ps(sp + ssize.width - 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));

						const __m256 mK6 = _mm256_loadu_ps(sp + ssize.width);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));

						const __m256 mKA = _mm256_loadu_ps(sp + ssize.width + 1);
						const __m256 mKD = _mm256_loadu_ps(sp + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));

						const __m256 mK3 = _mm256_loadu_ps(sp + (ssize.width << 1) - 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));

						const __m256 mK7 = _mm256_loadu_ps(sp + (ssize.width << 1));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));

						const __m256 mKB = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

						//DownRight G2
						const __m256 mK0 = _mm256_loadu_ps(sp - ssize.width - 1);
						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));
						const __m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));
#else
						const __m256 mK0 = _mm256_loadx_ps(sp - ssize.width - 1);
						const __m256 mK4 = _mm256_loadx_ps(sp - ssize.width);
						const __m256 mK8 = _mm256_loadx_ps(sp - ssize.width + 1);
						const __m256 mKC = _mm256_loadx_ps(sp - ssize.width + 2);
						const __m256 mK1 = _mm256_loadx_ps(sp - 1);
						const __m256 mK5 = _mm256_loadx_ps(sp);
						const __m256 mK9 = _mm256_loadx_ps(sp + 1);
						const __m256 mKD = _mm256_loadx_ps(sp + 2);
						const __m256 mK3 = _mm256_loadx_ps(sp + (ssize.width << 1) - 1);
						const __m256 mK7 = _mm256_loadx_ps(sp + (ssize.width << 1));
						const __m256 mKB = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKF = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);
						const __m256 mK2 = _mm256_loadx_ps(sp + ssize.width - 1);
						const __m256 mK6 = _mm256_loadx_ps(sp + ssize.width);
						const __m256 mKA = _mm256_loadx_ps(sp + ssize.width + 1);
						const __m256 mKE = _mm256_loadx_ps(sp + ssize.width + 2);

						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

						//DownRight G2
						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));
#endif
						/* --- 2(b) --- */
						__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
						__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
						//if (1+G1) / (1+G2) > T then 135
						__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						/* --- 2(c) --- */
						//UpRight方向にエッジがある場合，UpRight方向に補間 p1
						__m256 pxUpRight = _mm256_add_ps(mK3, mKC);
						pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
						pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mK6, mK9), cciCoeff_9, pxUpRight);

						//DownRight方向にエッジがある場合，DownRight方向に補間 p2
						__m256 pxDownRight = _mm256_add_ps(mK0, mKF);
						pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
						pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mK5, mKA), cciCoeff_9, pxDownRight);

						//weight = 1 / (1+G^5)
						//weight1はgradUpRightを使う
						__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradDownRightを使う
						__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1+w2p2) / (w1+w2)
						mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
						mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
						__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);


						//0で最初の引数をとる
						__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
						mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

						_mm256_store_ps(buf[0], mDst);

						sp += 3;
						dp += 6;

						//  C G K O
						//  D H L P
						//  E I M Q
						//  F J N R	

#if 0 //interleave
				//UpRight G1
						const __m256 mKG = _mm256_loadu_ps(sp - ssize.width);
						gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

						const __m256 mKH = _mm256_loadu_ps(sp);
						const __m256 mKK = _mm256_loadu_ps(sp - ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

						const __m256 mKO = _mm256_loadu_ps(sp - ssize.width + 2);
						const __m256 mKL = _mm256_loadu_ps(sp + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

						const __m256 mKI = _mm256_loadu_ps(sp + ssize.width);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

						const __m256 mKP = _mm256_loadu_ps(sp + 2);
						const __m256 mKM = _mm256_loadu_ps(sp + ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));

						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

						const __m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

						const __m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

						//DownRight G2
						gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

						const __m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
#else
				//UpRight G1
						const __m256 mKG = _mm256_loadx_ps(sp - ssize.width);
						const __m256 mKK = _mm256_loadx_ps(sp - ssize.width + 1);
						const __m256 mKO = _mm256_loadx_ps(sp - ssize.width + 2);
						const __m256 mKH = _mm256_loadx_ps(sp);
						const __m256 mKL = _mm256_loadx_ps(sp + 1);
						const __m256 mKP = _mm256_loadx_ps(sp + 2);
						const __m256 mKJ = _mm256_loadx_ps(sp + (ssize.width << 1));
						const __m256 mKN = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKR = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);
						const __m256 mKI = _mm256_loadx_ps(sp + ssize.width);
						const __m256 mKM = _mm256_loadx_ps(sp + ssize.width + 1);
						const __m256 mKQ = _mm256_loadx_ps(sp + ssize.width + 2);

						gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

						//DownRight G2
						gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
#endif
						/* --- 2(b) --- */
						mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
						mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
						//if (1+G1) / (1+G2) > T then 135
						mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						//mTmp2 = _mm256_div_ps(mTmp0, mTmp1);
						maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						//mTmp2 = _mm256_div_ps(mTmp1, mTmp0);
						maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						/* --- 2(c) --- */
						//UpRight方向にエッジがある場合，UpRight方向に補間 p1
						pxUpRight = _mm256_add_ps(mKF, mKO);
						pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
						pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mKI, mKL), cciCoeff_9, pxUpRight);

						//DownRight方向にエッジがある場合，DownRight方向に補間 p2
						pxDownRight = _mm256_add_ps(mKC, mKR);
						pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
						pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mKH, mKM), cciCoeff_9, pxDownRight);

						//weight = 1 / (1+G^5)
						//weight1はgradUpRightを使う
						weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradDownRightを使う
						weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1+w2p2) / (w1+w2)
						mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
						mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
						pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

						//0で最初の引数をとる
						mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
						mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

						_mm256_storeu_ps(buf[0] + 3, mDst);

						sp -= (ssize.width + 1);
						dp -= ((dsize.width << 1) + 2);

						/*(o,e)を補間-------------------------------------
						#:OddEven		@:EvenOdd
						S		|		X
						X X X X X	|	X S X T X
						8 X C @ G	|	X X X X X
						x X x # x X x	| X X C @ G X X
						9 X D X H	|	X # x X X
						X t X t X	|	X D X H X
						E		|		t		*/

						//horizontal
						const __m256 mKd1 = _mm256_loadu_ps(buf[2] + 1);
						const __m256 mKd2 = _mm256_loadu_ps(buf[2] + 2);

						__m256 gradHorizontal = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKC)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));

						const __m256 mKx1 = _mm256_loadu_ps(buf[1] + 1);
						const __m256 mKx2 = _mm256_loadu_ps(buf[1] + 2);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKD)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

						const __m256 mKt1 = _mm256_loadu_ps(buf[0] + 1);
						const __m256 mKt2 = _mm256_loadu_ps(buf[0] + 2);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKt1, mKt2)));

						//Vertical

						__m256 gradVertical = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK9));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKt1)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKt2)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));

						//Horizontal方向にエッジがある場合，Horizontal方向に補間
						const __m256 mKx0 = _mm256_load_ps(buf[1] + 0);
						const __m256 mKx3 = _mm256_loadu_ps(buf[1] + 3);
						__m256 pxHorizontal = _mm256_add_ps(mKx0, mKx3);
						pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
						pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKx1, mKx2), cciCoeff_9, pxHorizontal);

						//Vertical方向にエッジがある場合，Vertical方向に補間
						const __m256 mKS = _mm256_loadu_ps(sp - ssize.width);
						__m256 pxVertical = _mm256_add_ps(mKS, mKE);
						pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
						pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKD), cciCoeff_9, pxVertical);

						//weight = 1 / (1+G^5)
						//weight1はgradHorizontalを使う
						weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradVerticalを使う
						weight2 = _mm256_mul_ps(gradVertical, gradVertical);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1+w2p2) / (w1+w2)
						mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
						mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);
						pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

						mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
						mTmp1 = _mm256_add_ps(mOnef, gradVertical);
						//if (1+G1) / (1+G2) > T then 135
						//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
						mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						__m256 maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						//cmpの結果を論理演算に使うとバグる
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						__m256 maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//0で最初の引数をとる
						mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
						mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

						//mk7:1234 5678
						//dst:abcd efgh-->a1b2c3d4...=
						__m256 mTmpHi = _mm256_unpackhi_ps(mDst, mKx2);//a 1 b 2 / e 5 f 6
						__m256 mTmpLo = _mm256_unpacklo_ps(mDst, mKx2);//c 3 d 4 / g 7 h 8
						__m256 mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);//1 0 2 0 / 5 0 6 0
						__m256 mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);//3 0 4 0 / 7 0 8 0

						_mm256_store_ps(dp - 1, mSrcLo);
						_mm256_store_ps(dp + 7, mSrcHi);

						//horizontal
						gradHorizontal = _mm256_setzero_ps();
						__m256 mKT = _mm256_loadu_ps(sp - ssize.width + 1);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKT)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2)));
						__m256 mKd3 = _mm256_loadu_ps(buf[2] + 3);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKd3)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKx3)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

						//Vertical
						gradVertical = _mm256_setzero_ps();
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKC)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKT, mKG)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd3, mKx3)));

						//Horizontal方向にエッジがある場合，Horizontal方向に補間
						pxHorizontal = _mm256_add_ps(mK8, mKK);
						pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
						pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKG), cciCoeff_9, pxHorizontal);

						//Vertical方向にエッジがある場合，Vertical方向に補間
						__m256 mKd0 = _mm256_loadu_ps(buf[3] + 2);//1 0 2 0 / 3 0 4 0
						pxVertical = _mm256_add_ps(mKd0, mKt2);
						pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
						pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKd2, mKx2), cciCoeff_9, pxVertical);

						//weight = 1 / (1+G^5)
						//weight1はgradHorizontalを使う
						weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradVerticalを使う
						weight2 = _mm256_mul_ps(gradVertical, gradVertical);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1 + w2p2) / (w1 + w2)
						mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
						mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);
						pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

						mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
						mTmp1 = _mm256_add_ps(mOnef, gradVertical);

						//if (1+G1) / (1+G2) > T then 135
						//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
						mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						//cmpの結果を論理演算に使うとバグる
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//0で最初の引数をとる
						mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
						mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

						mTmpHi = _mm256_unpackhi_ps(mKC, mDst);//a 1 b 2 / e 5 f 6
						mTmpLo = _mm256_unpacklo_ps(mKC, mDst);//c 3 d 4 / g 7 h 8
						mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);//1 0 2 0 / 5 0 6 0
						mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);//3 0 4 0 / 7 0 8 0

						_mm256_store_ps(dp - 1 - dsize.width, mSrcLo);
						_mm256_store_ps(dp + 7 - dsize.width, mSrcHi);

						sp += 8;
						dp -= (dsize.width - 15);
						buf[0] += 8;
						buf[1] += 8;
						buf[2] += 8;
						buf[3] += 8;
					}//ここまでがmainの一列目の処理

					for (int x = 0; x < ssize.width - 16; x += 8)//2列目以降．●は右下だけでいい
					{
						float grad45 = 0.f, grad135 = 0.f;

						//（奇数，奇数）の補間
						//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
						sp += (ssize.width + 1);
						dp += (dsize.width + (dsize.width << 1) + 3);
						//  0   4   8   C
						//  1   5   9   D
						//  2   6   A   E
						//  3   7   B   F

						//  C G K O
						//  D H L P
						//  E I M Q
						//  F J N R

#if 0 //interleave
				//UpRight G1
						const __m256 mKG = _mm256_loadx_ps(sp - ssize.width);
						const __m256 mKD = _mm256_loadx_ps(sp - 1);

						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

						const __m256 mKH = _mm256_loadx_ps(sp);
						const __m256 mKK = _mm256_loadx_ps(sp - ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

						const __m256 mKO = _mm256_loadx_ps(sp - ssize.width + 2);
						const __m256 mKL = _mm256_loadx_ps(sp + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL));//

						const __m256 mKE = _mm256_loadx_ps(sp + ssize.width - 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

						const __m256 mKI = _mm256_loadx_ps(sp + ssize.width);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));//

						const __m256 mKP = _mm256_loadx_ps(sp + 2);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));//
						const __m256 mKM = _mm256_loadx_ps(sp + ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));//

						const __m256 mKF = _mm256_loadx_ps(sp + (ssize.width << 1) - 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

						const __m256 mKJ = _mm256_loadx_ps(sp + (ssize.width << 1));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));//

						const __m256 mKN = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));//
						const __m256 mKQ = _mm256_loadx_ps(sp + ssize.width + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));//

						//DownRight G2
						const __m256 mKC = _mm256_loadx_ps(sp - ssize.width - 1);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));


						const __m256 mKR = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
#else
				//UpRight G1
						const __m256 mKC = _mm256_loadx_ps(sp - ssize.width - 1);
						const __m256 mKG = _mm256_loadx_ps(sp - ssize.width);
						const __m256 mKK = _mm256_loadx_ps(sp - ssize.width + 1);
						const __m256 mKO = _mm256_loadx_ps(sp - ssize.width + 2);
						const __m256 mKD = _mm256_loadx_ps(sp - 1);
						const __m256 mKH = _mm256_loadx_ps(sp);
						const __m256 mKL = _mm256_loadx_ps(sp + 1);
						const __m256 mKP = _mm256_loadx_ps(sp + 2);
						const __m256 mKE = _mm256_loadx_ps(sp + ssize.width - 1);
						const __m256 mKI = _mm256_loadx_ps(sp + ssize.width);
						const __m256 mKM = _mm256_loadx_ps(sp + ssize.width + 1);
						const __m256 mKQ = _mm256_loadx_ps(sp + ssize.width + 2);
						const __m256 mKF = _mm256_loadx_ps(sp + (ssize.width << 1) - 1);
						const __m256 mKJ = _mm256_loadx_ps(sp + (ssize.width << 1));
						const __m256 mKN = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKR = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);

						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

						//DownRight G2
						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));//
#endif
						/* --- 2(b) --- */
						__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
						__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
						//if (1+G1) / (1+G2) > T then 135
						__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						/* --- 2(c) --- */
						//UpRight方向にエッジがある場合，UpRight方向に補間 p1
						__m256 pxUpRight = _mm256_add_ps(mKF, mKO);
						pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
						pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mKI, mKL), cciCoeff_9, pxUpRight);


						//DownRight方向にエッジがある場合，DownRight方向に補間 p2
						__m256 pxDownRight = _mm256_add_ps(mKC, mKR);
						pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
						pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mKH, mKM), cciCoeff_9, pxDownRight);

						//weight = 1 / (1+G^5)
						//weight1はgradUpRightを使う
						__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradDownRightを使う
						__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1+w2p2) / (w1+w2)
						mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
						mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
						__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

						//0で最初の引数をとる
						__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
						mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

						_mm256_storeu_ps(buf[0] + 3, mDst);

						sp -= (ssize.width + 1);
						dp -= ((dsize.width << 1) + 2);

						//(o,e)を補間-------------------------------------
						//		#:OddEven		@:EvenOdd
						//			S		|		X		
						//		X X X X X	|	X S X T X	
						//		8 X C @ G	|	X X X X X	
						//	  x X x # x X x	| X X C @ G X X
						//		9 X D X H	|	X # x X X	
						//		X t X t X	|	X D X H X	
						//			E		|		t	


						//horizontal
						const __m256 mKd1 = _mm256_loadx_ps(buf[2] + 1);
						const __m256 mKd2 = _mm256_loadx_ps(buf[2] + 2);
						__m256 gradHorizontal = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2));

						const __m256 mK8 = _mm256_loadx_ps(sp - 1);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKC)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));

						const __m256 mKx1 = _mm256_loadx_ps(buf[1] + 1);
						const __m256 mKx2 = _mm256_loadx_ps(buf[1] + 2);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
						const __m256 mK9 = _mm256_loadx_ps(sp + ssize.width - 1);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKD)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

						const __m256 mKt1 = _mm256_loadx_ps(buf[0] + 1);
						const __m256 mKt2 = _mm256_loadx_ps(buf[0] + 2);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKt1, mKt2)));

						//Vertical
						__m256 gradVertical = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK9));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKt1)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKt2)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));

						//Horizontal方向にエッジがある場合，Horizontal方向に補間
						const __m256 mKx0 = _mm256_load_ps(buf[1]);
						const __m256 mKx3 = _mm256_loadx_ps(buf[1] + 3);
						__m256 pxHorizontal = _mm256_add_ps(mKx0, mKx3);
						pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
						pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKx1, mKx2), cciCoeff_9, pxHorizontal);

						//Vertical方向にエッジがある場合，Vertical方向に補間
						const __m256 mKS = _mm256_loadx_ps(sp - ssize.width);
						__m256 pxVertical = _mm256_add_ps(mKS, mKE);
						pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
						pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKD), cciCoeff_9, pxVertical);

						//weight = 1 / (1+G^5)
						//weight1はgradHorizontalを使う
						weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradVerticalを使う
						weight2 = _mm256_mul_ps(gradVertical, gradVertical);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1+w2p2) / (w1+w2)
						mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
						mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);

						pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

						mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
						mTmp1 = _mm256_add_ps(mOnef, gradVertical);

						//if (1+G1) / (1+G2) > T then 135
						//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
						mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						__m256 maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						//cmpの結果を論理演算に使うとバグる
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						__m256 maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//0で最初の引数をとる
						mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
						mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

						//mk7:1234 5678
						//dst:abcd efgh-->a1b2c3d4...=
						__m256 mTmpHi = _mm256_unpackhi_ps(mDst, mKx2);//a 1 b 2 / e 5 f 6
						__m256 mTmpLo = _mm256_unpacklo_ps(mDst, mKx2);//c 3 d 4 / g 7 h 8
						__m256 mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);//1 0 2 0 / 5 0 6 0
						__m256 mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);//3 0 4 0 / 7 0 8 0

						_mm256_storeu_ps(dp - 1, mSrcLo);
						_mm256_storeu_ps(dp + 7, mSrcHi);

						//horizontal
						const __m256 mKT = _mm256_loadx_ps(sp - ssize.width + 1);
						gradHorizontal = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKT));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2)));

						const __m256 mKd3 = _mm256_loadx_ps(buf[2] + 3);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKd3)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKx3)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

						//Vertical
						gradVertical = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKC)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKT, mKG)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd3, mKx3)));

						//Horizontal方向にエッジがある場合，Horizontal方向に補間
						pxHorizontal = _mm256_add_ps(mK8, mKK);
						pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
						pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKG), cciCoeff_9, pxHorizontal);


						//Vertical方向にエッジがある場合，Vertical方向に補間
						const __m256 mKd0 = _mm256_loadx_ps(buf[3] + 2);//1 0 2 0 / 3 0 4 0
						pxVertical = _mm256_add_ps(mKd0, mKt2);
						pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
						pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKd2, mKx2), cciCoeff_9, pxVertical);

						//weight = 1 / (1+G^5)
						//weight1はgradHorizontalを使う
						weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradVerticalを使う
						weight2 = _mm256_mul_ps(gradVertical, gradVertical);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1 + w2p2) / (w1 + w2)
						mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
						mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);
						pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

						mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
						mTmp1 = _mm256_add_ps(mOnef, gradVertical);

						//if (1+G1) / (1+G2) > T then 135
						//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
						mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						//cmpの結果を論理演算に使うとバグる
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//0で最初の引数をとる
						mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
						mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

						mTmpHi = _mm256_unpackhi_ps(mKC, mDst);
						mTmpLo = _mm256_unpacklo_ps(mKC, mDst);
						mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
						mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

						_mm256_storeu_ps(dp - 1 - dsize.width, mSrcLo);
						_mm256_storeu_ps(dp - dsize.width + 7, mSrcHi);//koko ha store yori hayai kamo
						//_mm256_storeu_ps(dp - 1 - dsize.width, mSrcLo);
						//_mm256_storeu_ps(dp - dsize.width + 7, mSrcHi);//koko ha store yori hayai kamo

						sp += 8;
						dp -= (dsize.width - 15);
						buf[0] += 8;
						buf[1] += 8;
						buf[2] += 8;
						buf[3] += 8;
					}//x メインループの2列目以降
					tp = buf[0];
					buf[0] = buf[3];
					buf[3] = buf[2];
					buf[2] = buf[1];
					buf[1] = tp;

					buf[0] -= tempSize - 11;
					buf[1] -= tempSize - 11;
					buf[2] -= tempSize - 11;
					buf[3] -= tempSize - 11;
				}//y
			}//parallel
			_mm_free(buffer);
		}
	};

	void DCCI32FC1_SIMD_LoopFusionCVParallelForOld(const Mat& src_, Mat& dest, const float threshold, int ompNumThreads)
	{
		CV_Assert(src_.type() == CV_32FC1);

		ompNumThreads = 8;
		int loopdiv = ompNumThreads * 4;
		int looploopdiv = ompNumThreads * 2;
		cv::parallel_for_(Range{ 0, loopdiv }, DCCI32FC1_SIMD_LoopFusionInvorkerOld{ src_, dest, threshold, loopdiv }, looploopdiv);
	}

	void DCCI32FC1_SIMD_LoopFusionCVParallelForNoSMT(const Mat& src_, Mat& dest, const float threshold, int division)
	{
		CV_Assert(src_.type() == CV_32FC1);

		// paralle_for_(Range{0, div}, ParalleLoopBody, nstrips)
		// と
		//#pragma omp parallel for num_threads(nstrips)
		//	for (int i = 0; i < div; i++)
		//	{
		//		ParallelLoopBody::operator()の処理
		//	}
		// が等価なら，以下の記述で，「スレッド数をコア数と同じ値に制限した状態で，分割数をスレッド数以上に設定する」ことができているはず

		int ompNumThreads = omp_get_max_threads() / 2;
		int div = division;
		cv::parallel_for_(Range{ 0, div }, DCCI32FC1_SIMD_LoopFusionInvorkerOld{ src_, dest, threshold, div }, ompNumThreads);
	}



	class DCCI32FC1_SIMD_LoopFusionInvorker : public ParallelLoopBody
	{
		Mat src;
		Mat& dest;
		const float threshold;
		Size ssize;
		Size dsize;
		const cv::Size tileDiv;
	public:
		DCCI32FC1_SIMD_LoopFusionInvorker(const Mat& src_, Mat& dest, const float threshold, const cv::Size tileDiv) : src(src_), dest(dest), threshold(threshold), tileDiv(tileDiv)
		{
			ssize = src.size();
			dsize = dest.size();
		}

		void operator()(const Range& range) const override
		{
#pragma region init
			const __m256 signMask = _mm256_set1_ps(-0.0f); // 0x80000000
			const __m256 mThreshold = _mm256_set1_ps(threshold);
			const __m256 mOnef = _mm256_set1_ps(1.f);
			const __m256 cciCoeff_1 = _mm256_set1_ps(-1.f / 16.f);
			const __m256 cciCoeff_9 = _mm256_set1_ps(9.f / 16.f);

			const int buffer_width = ssize.width + 8;//ssize.width + 8
			const int bsize32 = get_simd_ceil(buffer_width, 8);

			//alloc large buffer
#ifndef __USE_GLOBAL_BUFFER__
			float* const buffer = (float*)_mm_malloc(sizeof(float) * bsize32 * 4, 32);
			//AutoBuffer<float> buffer(bw32 * 4);
#else	
			const int tidx = cv::getThreadNum();
			if (bufferGlobalLFSize[tidx] != bsize32 * 4)
			{
				_mm_free(bufferGlobalLFPtr[tidx]);
				bufferGlobalLFPtr[tidx] = (float*)_mm_malloc(sizeof(float) * (bsize32 * 4), 32);
				bufferGlobalLFSize[tidx] = bsize32 * 4;
			}
			float* buffer = bufferGlobalLFPtr[tidx];
			//initialize each buffer top
			for (int i = 0; i < 4; i++)
			{
				_mm256_store_ps(buffer + bsize32 * i, _mm256_setzero_ps());
			}
#endif
			const int divx_step = ssize.width / tileDiv.width;
			const int divy_step = ssize.height / tileDiv.height;
			const int ddivx_step = dsize.width / tileDiv.width;
#pragma endregion

			for (int t = range.start; t < range.end; t++)
			{
				const int tileIndexX = t % tileDiv.width;
				const int tileIndexY = t / tileDiv.width;

				const int start_x = max(0, (tileIndexX + 0) * divx_step - 8);
				const int end_x = min(ssize.width, (tileIndexX + 1) * divx_step);
				const int bstep = end_x - start_x;
				const int init_y = 3;
				const int vpixelstep2 = ssize.width * 2;
#pragma region init_buffer
				std::array<float*, 4> buf =
				{
					buffer + bsize32 * 0 + start_x,
					buffer + bsize32 * 1 + start_x,
					buffer + bsize32 * 2 + start_x,
					buffer + bsize32 * 3 + start_x
				};
#pragma endregion

#pragma region redundant_bufferfilling_first_3_lines
				int start_y = max(0, (tileIndexY + 0) * divy_step - init_y);
				int end_y = start_y + init_y;
				for (int y = start_y; y < end_y; y++)
				{
					const float* sp = src.ptr<float>(y, start_x);

					for (int x = start_x; x < end_x; x += 8)//バッファー埋め一列目以降；右下の●だけ求める処理
					{
						//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
						sp += (ssize.width - 2);
						const __m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);

						//  C G K O
						//  D H L P
						//  E I M Q
						//  F J N R	

						//UpRight G1
						const __m256 mKD = _mm256_loadu_ps(sp + 2);
						const __m256 mKG = _mm256_loadu_ps(sp - ssize.width + 3);
						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

						const __m256 mKH = _mm256_loadu_ps(sp + 3);
						const __m256 mKK = _mm256_loadu_ps(sp - ssize.width + 4);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

						const __m256 mKO = _mm256_loadu_ps(sp - ssize.width + 5);
						const __m256 mKL = _mm256_loadu_ps(sp + 4);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

						const __m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

						const __m256 mKI = _mm256_loadu_ps(sp + ssize.width + 3);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

						const __m256 mKM = _mm256_loadu_ps(sp + ssize.width + 4);
						const __m256 mKP = _mm256_loadu_ps(sp + 5);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));


						const __m256 mKF = _mm256_loadu_ps(sp + vpixelstep2 + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

						const __m256 mKJ = _mm256_loadu_ps(sp + vpixelstep2 + 3);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

						const __m256 mKN = _mm256_loadu_ps(sp + vpixelstep2 + 4);
						const __m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 5);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

						//DownRight G2
						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

						const __m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 5);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));

						sp += 3;

						/* --- 2(b) --- */
						__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
						__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);

						//if (1+G1) / (1+G2) > T then 135
						__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						/* --- 2(c) --- */
						//UpRight方向にエッジがある場合，UpRight方向に補間 p1
						__m256 pxUpRight = _mm256_add_ps(mKF, mKO);
						pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
						pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKI, mKL), pxUpRight);

						//DownRight方向にエッジがある場合，DownRight方向に補間 p2
						__m256 pxDownRight = _mm256_add_ps(mKC, mKR);
						pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
						pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mKH, mKM), pxDownRight);

						//weight = 1 / (1+G^5)
						//weight1はgradUpRightを使う
						__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradDownRightを使う
						__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1 + w2p2) / (w1 + w2)
						mTmp2 = _mm256_mul_ps(weight1, pxUpRight);
						mTmp2 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp2);
						__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp2);

						//0で最初の引数をとる
						__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
						mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

						_mm256_storeu_ps(buf[0] + 3, mDst);

						sp -= (ssize.width - 7);

						buf[0] += 8;
						buf[1] += 8;
						buf[2] += 8;
					}//x-pre-processing
					buf[0] -= bstep; buf[1] -= bstep; buf[2] -= bstep;
					ringbuffering3(buf[0], buf[2], buf[1]);
				}//y-pre-processing
				ringbuffering2(buf[0], buf[3]);
#pragma endregion

#pragma region main_processing
				start_y = end_y;
				end_y = min(ssize.height - 3, (tileIndexY + 1) * divy_step);
				for (int y = start_y; y < end_y; y++)
				{
					const float* sp = src.ptr<float>(y, start_x);
					float* dp = (float*)dest.ptr<float>(y * 2, max((tileIndexX + 0) * ddivx_step, 0));

					if (start_x != 0)
					{
						//一列目の処理．左下，右下の●を求める必要がある
						//（奇数，奇数）の補間

						//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
						sp += (ssize.width - 2);
						//  1   5   9   D
						//  0   4   8   C
						//  2   6   A   E
						//  3   7   B   F

						//UpRight G1
#if 0 //interleave
						const __m256 mK1 = _mm256_loadu_ps(sp - 1);
						const __m256 mK4 = _mm256_loadu_ps(sp - ssize.width);
						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1));

						const __m256 mK5 = _mm256_loadu_ps(sp);
						const __m256 mK8 = _mm256_loadu_ps(sp - ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));

						const __m256 mKC = _mm256_loadu_ps(sp - ssize.width + 2);
						const __m256 mK9 = _mm256_loadu_ps(sp + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));

						const __m256 mK2 = _mm256_loadu_ps(sp + ssize.width - 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));

						const __m256 mK6 = _mm256_loadu_ps(sp + ssize.width);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));

						const __m256 mKA = _mm256_loadu_ps(sp + ssize.width + 1);
						const __m256 mKD = _mm256_loadu_ps(sp + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));

						const __m256 mK3 = _mm256_loadu_ps(sp + (ssize.width << 1) - 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));

						const __m256 mK7 = _mm256_loadu_ps(sp + (ssize.width << 1));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));

						const __m256 mKB = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKE = _mm256_loadu_ps(sp + ssize.width + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

						//DownRight G2
						const __m256 mK0 = _mm256_loadu_ps(sp - ssize.width - 1);
						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));
						const __m256 mKF = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));
#else
						const __m256 mK0 = _mm256_loadx_ps(sp - ssize.width - 1);
						const __m256 mK4 = _mm256_loadx_ps(sp - ssize.width);
						const __m256 mK8 = _mm256_loadx_ps(sp - ssize.width + 1);
						const __m256 mKC = _mm256_loadx_ps(sp - ssize.width + 2);
						const __m256 mK1 = _mm256_loadx_ps(sp - 1);
						const __m256 mK5 = _mm256_loadx_ps(sp);
						const __m256 mK9 = _mm256_loadx_ps(sp + 1);
						const __m256 mKD = _mm256_loadx_ps(sp + 2);
						const __m256 mK3 = _mm256_loadx_ps(sp + (ssize.width << 1) - 1);
						const __m256 mK7 = _mm256_loadx_ps(sp + (ssize.width << 1));
						const __m256 mKB = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKF = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);
						const __m256 mK2 = _mm256_loadx_ps(sp + ssize.width - 1);
						const __m256 mK6 = _mm256_loadx_ps(sp + ssize.width);
						const __m256 mKA = _mm256_loadx_ps(sp + ssize.width + 1);
						const __m256 mKE = _mm256_loadx_ps(sp + ssize.width + 2);

						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

						//DownRight G2
						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));
#endif

						/* --- 2(b) --- */
						__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
						__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
						//if (1+G1) / (1+G2) > T then 135
						__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						/* --- 2(c) --- */
						//UpRight方向にエッジがある場合，UpRight方向に補間 p1
						__m256 pxUpRight = _mm256_add_ps(mK3, mKC);
						pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
						pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mK6, mK9), cciCoeff_9, pxUpRight);

						//DownRight方向にエッジがある場合，DownRight方向に補間 p2
						__m256 pxDownRight = _mm256_add_ps(mK0, mKF);
						pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
						pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mK5, mKA), cciCoeff_9, pxDownRight);

						//weight = 1 / (1+G^5)
						//weight1はgradUpRightを使う
						__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradDownRightを使う
						__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1+w2p2) / (w1+w2)
						mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
						mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
						__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);


						//0で最初の引数をとる
						__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
						mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

						_mm256_store_ps(buf[0], mDst);

						//  C G K O
						//  D H L P
						//  E I M Q
						//  F J N R	
						sp += 3;
#if 0 //interleave
						//UpRight G1
						const __m256 mKG = _mm256_loadu_ps(sp - ssize.width);
						gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

						const __m256 mKH = _mm256_loadu_ps(sp);
						const __m256 mKK = _mm256_loadu_ps(sp - ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

						const __m256 mKO = _mm256_loadu_ps(sp - ssize.width + 2);
						const __m256 mKL = _mm256_loadu_ps(sp + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

						const __m256 mKI = _mm256_loadu_ps(sp + ssize.width);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));

						const __m256 mKP = _mm256_loadu_ps(sp + 2);
						const __m256 mKM = _mm256_loadu_ps(sp + ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));

						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

						const __m256 mKJ = _mm256_loadu_ps(sp + (ssize.width << 1));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));

						const __m256 mKN = _mm256_loadu_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKQ = _mm256_loadu_ps(sp + ssize.width + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

						//DownRight G2
						gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));

						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));

						const __m256 mKR = _mm256_loadu_ps(sp + (ssize.width << 1) + 2);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
#else //load from sp for gradUpRight and gradDownRight
						//UpRight G1
						const __m256 mKG = _mm256_loadx_ps(sp - ssize.width);
						const __m256 mKK = _mm256_loadx_ps(sp - ssize.width + 1);
						const __m256 mKO = _mm256_loadx_ps(sp - ssize.width + 2);
						const __m256 mKH = _mm256_loadx_ps(sp);
						const __m256 mKL = _mm256_loadx_ps(sp + 1);
						const __m256 mKP = _mm256_loadx_ps(sp + 2);
						const __m256 mKJ = _mm256_loadx_ps(sp + (ssize.width << 1));
						const __m256 mKN = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
						const __m256 mKR = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);
						const __m256 mKI = _mm256_loadx_ps(sp + ssize.width);
						const __m256 mKM = _mm256_loadx_ps(sp + ssize.width + 1);
						const __m256 mKQ = _mm256_loadx_ps(sp + ssize.width + 2);

						gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

						//DownRight G2
						gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
#endif
						/* --- 2(b) --- */
						mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
						mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
						//if (1+G1) / (1+G2) > T then 135
						mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						//mTmp2 = _mm256_div_ps(mTmp0, mTmp1);
						maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						//mTmp2 = _mm256_div_ps(mTmp1, mTmp0);
						maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						/* --- 2(c) --- */
						//UpRight方向にエッジがある場合，UpRight方向に補間 p1
						pxUpRight = _mm256_add_ps(mKF, mKO);
						pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
						pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mKI, mKL), cciCoeff_9, pxUpRight);

						//DownRight方向にエッジがある場合，DownRight方向に補間 p2
						pxDownRight = _mm256_add_ps(mKC, mKR);
						pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
						pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mKH, mKM), cciCoeff_9, pxDownRight);

						//weight = 1 / (1+G^5)
						//weight1はgradUpRightを使う
						weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradDownRightを使う
						weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1+w2p2) / (w1+w2)
						mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
						mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
						pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

						//0で最初の引数をとる
						mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
						mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

						_mm256_storeu_ps(buf[0] + 3, mDst);

						sp -= (ssize.width + 1);
						sp += 8;
						buf[0] += 8; buf[1] += 8; buf[2] += 8; buf[3] += 8;
					}//main processing 1st line
					const int st = (start_x == 0) ? start_x : start_x + 8;

					for (int x = st; x < end_x; x += 8)//2列目以降．●は右下だけでいい
					{
#pragma region load
						//（奇数，奇数）の補間
						//（奇数，偶数），（偶数，奇数）の補間に必要な（奇数，奇数）の４画素を求める．
						sp += (ssize.width + 1);//(1,1)
						//  0   4   8   C
						//  1   5   9   D
						//  2   6   A   E
						//  3   7   B   F
						//  C G K O
						//  D H L P
						//  E I M Q
						//  F J N R
#if 1 //interleave
				//UpRight G1
						const __m256 mKG = _mm256_loadx_ps(sp - ssize.width);
						const __m256 mKD = _mm256_loadx_ps(sp - 1);

						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));

						const __m256 mKH = _mm256_loadx_ps(sp);
						const __m256 mKK = _mm256_loadx_ps(sp - ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));

						const __m256 mKO = _mm256_loadx_ps(sp - ssize.width + 2);
						const __m256 mKL = _mm256_loadx_ps(sp + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));

						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL));//

						const __m256 mKE = _mm256_loadx_ps(sp + ssize.width - 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));

						const __m256 mKI = _mm256_loadx_ps(sp + ssize.width);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));//

						const __m256 mKP = _mm256_loadx_ps(sp + 2);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));//
						const __m256 mKM = _mm256_loadx_ps(sp + ssize.width + 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));//

						const __m256 mKF = _mm256_loadx_ps(sp + (ssize.width << 1) - 1);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

						const __m256 mKJ = _mm256_loadx_ps(sp + (ssize.width << 1));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));//

						const __m256 mKN = _mm256_loadx_ps(sp + (ssize.width << 1) + 1);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));//
						const __m256 mKQ = _mm256_loadx_ps(sp + ssize.width + 2);
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));//

						//DownRight G2
						const __m256 mKC = _mm256_loadx_ps(sp - ssize.width - 1);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));
						const __m256 mKR = _mm256_loadx_ps(sp + (ssize.width << 1) + 2);
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));

#else
				//G1, G2
						const __m256 mKC = _mm256_loadx_ps(sp - ssize.width - 1);//(0,0)
						const __m256 mKG = _mm256_loadx_ps(sp - ssize.width + 0);//(0,1)
						const __m256 mKK = _mm256_loadx_ps(sp - ssize.width + 1);//(0,2)
						const __m256 mKO = _mm256_loadx_ps(sp - ssize.width + 2);//(0,3)
						const __m256 mKD = _mm256_loadx_ps(sp - 1);//(1,0)
						const __m256 mKH = _mm256_loadx_ps(sp + 0);//(1,1)
						const __m256 mKL = _mm256_loadx_ps(sp + 1);//(1,2)
						const __m256 mKP = _mm256_loadx_ps(sp + 2);//(1,3)
						const __m256 mKE = _mm256_loadx_ps(sp + ssize.width - 1);//(2,0)
						const __m256 mKI = _mm256_loadx_ps(sp + ssize.width + 0);//(2,1)
						const __m256 mKM = _mm256_loadx_ps(sp + ssize.width + 1);//(2,2)
						const __m256 mKQ = _mm256_loadx_ps(sp + ssize.width + 2);//(2,3)
						const __m256 mKF = _mm256_loadx_ps(sp + vpixelstep2 - 1);//(3,0)
						const __m256 mKJ = _mm256_loadx_ps(sp + vpixelstep2 + 0);//(3,1)
						const __m256 mKN = _mm256_loadx_ps(sp + vpixelstep2 + 1);//(3,2)
						const __m256 mKR = _mm256_loadx_ps(sp + vpixelstep2 + 2);//(3,3)

						//UpRight G1
						__m256 gradUpRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKD));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKH)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKE)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKO, mKL)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKI)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKP, mKM)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKJ)));
						gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKQ, mKN)));

						//DownRight G2
						__m256 gradDownRight = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKL));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKL, mKQ)));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKI)));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKN)));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKM, mKR)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKM)));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKJ)));//
						gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKK, mKP)));//
#endif
						sp -= (ssize.width + 1);
#pragma endregion
#pragma region grad
						/* --- 2(b) --- */
						__m256 mTmp0 = _mm256_add_ps(mOnef, gradUpRight);//G1=gradUpRight
						__m256 mTmp1 = _mm256_add_ps(mOnef, gradDownRight);
						//if (1+G1) / (1+G2) > T then 135
						__m256 mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						/* --- 2(c) --- */
						//UpRight方向にエッジがある場合，UpRight方向に補間 p1
						__m256 pxUpRight = _mm256_add_ps(mKF, mKO);
						pxUpRight = _mm256_mul_ps(cciCoeff_1, pxUpRight);
						pxUpRight = _mm256_fmadd_ps(_mm256_add_ps(mKI, mKL), cciCoeff_9, pxUpRight);

						//DownRight方向にエッジがある場合，DownRight方向に補間 p2
						__m256 pxDownRight = _mm256_add_ps(mKC, mKR);
						pxDownRight = _mm256_mul_ps(cciCoeff_1, pxDownRight);
						pxDownRight = _mm256_fmadd_ps(_mm256_add_ps(mKH, mKM), cciCoeff_9, pxDownRight);
#pragma endregion
#pragma region weight
						//weight = 1 / (1+G^5)
						//weight1はgradUpRightを使う
						__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradDownRightを使う
						__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
						weight2 = _mm256_rcp_ps(weight2);
#pragma endregion
#pragma region dest
						//p = (w1p1+w2p2) / (w1+w2)
						mTmp0 = _mm256_mul_ps(weight1, pxUpRight);
						mTmp1 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp0);
						__m256 pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

						//0で最初の引数をとる
						__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
						mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);
#pragma endregion
						_mm256_storeu_ps(buf[0] + 3, mDst);

						//(o,e)を補間-------------------------------------
						//		#:OddEven		@:EvenOdd
						//			S		|		X		
						//		X X X X X	|	X S X T X	
						//		8 X C @ G	|	X X X X X	
						//	  x X x # x X x	| X X C @ G X X
						//		9 X D X H	|	X # x X X	
						//		X t X t X	|	X D X H X	
						//			E		|		t	
#pragma region grad_from_buf012_src
					//horizontal
						const __m256 mKd1 = _mm256_loadx_ps(buf[2] + 1);
						const __m256 mKd2 = _mm256_loadx_ps(buf[2] + 2);
						__m256 gradHorizontal = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2));

						const __m256 mK8 = _mm256_loadx_ps(sp - 1);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKC)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));

						const __m256 mKx1 = _mm256_loadx_ps(buf[1] + 1);
						const __m256 mKx2 = _mm256_loadx_ps(buf[1] + 2);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
						const __m256 mK9 = _mm256_loadx_ps(sp + ssize.width - 1);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKD)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

						const __m256 mKt1 = _mm256_loadx_ps(buf[0] + 1);
						const __m256 mKt2 = _mm256_loadx_ps(buf[0] + 2);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKt1, mKt2)));

						//Vertical
						__m256 gradVertical = _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK9));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKt1)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKt2)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));

						//Horizontal方向にエッジがある場合，Horizontal方向に補間
						const __m256 mKx0 = _mm256_load_ps(buf[1]);
						const __m256 mKx3 = _mm256_loadx_ps(buf[1] + 3);
						__m256 pxHorizontal = _mm256_add_ps(mKx0, mKx3);
						pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
						pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKx1, mKx2), cciCoeff_9, pxHorizontal);

						//Vertical方向にエッジがある場合，Vertical方向に補間
						const __m256 mKS = _mm256_loadx_ps(sp - ssize.width);
						__m256 pxVertical = _mm256_add_ps(mKS, mKE);
						pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
						pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKD), cciCoeff_9, pxVertical);
#pragma endregion
#pragma region weight
						//weight = 1 / (1+G^5)
						//weight1はgradHorizontalを使う
						weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradVerticalを使う
						weight2 = _mm256_mul_ps(gradVertical, gradVertical);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1+w2p2) / (w1+w2)
						mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
						mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);

						pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

						mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
						mTmp1 = _mm256_add_ps(mOnef, gradVertical);

						//if (1+G1) / (1+G2) > T then 135
						//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
						mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						__m256 maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						//cmpの結果を論理演算に使うとバグる
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						__m256 maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//0で最初の引数をとる
						mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
						mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);
#pragma endregion
#pragma region dest
						//mk7:1234 5678
						//dst:abcd efgh-->a1b2c3d4...=
						__m256 mTmpHi = _mm256_unpackhi_ps(mDst, mKx2);//a 1 b 2 / e 5 f 6
						__m256 mTmpLo = _mm256_unpacklo_ps(mDst, mKx2);//c 3 d 4 / g 7 h 8
						__m256 mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);//1 0 2 0 / 5 0 6 0
						__m256 mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);//3 0 4 0 / 7 0 8 0

						_mm256_storea_ps(dp + 0 + dsize.width, mSrcLo);
						_mm256_storea_ps(dp + 8 + dsize.width, mSrcHi);
#pragma endregion
#pragma region grad_frombuf012_src
						//horizontal
						const __m256 mKT = _mm256_loadx_ps(sp - ssize.width + 1);
						gradHorizontal = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKT));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKd2)));

						const __m256 mKd3 = _mm256_loadx_ps(buf[2] + 3);
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKd3)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKG)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx1, mKx2)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKx2, mKx3)));
						gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKH)));

						//Vertical
						gradVertical = _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd1, mKx1));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKS, mKC)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKD)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd2, mKx2)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKT, mKG)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKG, mKH)));
						gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKd3, mKx3)));

						//Horizontal方向にエッジがある場合，Horizontal方向に補間
						pxHorizontal = _mm256_add_ps(mK8, mKK);
						pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
						pxHorizontal = _mm256_fmadd_ps(_mm256_add_ps(mKC, mKG), cciCoeff_9, pxHorizontal);


						//Vertical方向にエッジがある場合，Vertical方向に補間
						const __m256 mKd0 = _mm256_loadx_ps(buf[3] + 2);//1 0 2 0 / 3 0 4 0
						pxVertical = _mm256_add_ps(mKd0, mKt2);
						pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
						pxVertical = _mm256_fmadd_ps(_mm256_add_ps(mKd2, mKx2), cciCoeff_9, pxVertical);
#pragma endregion
#pragma region weight
						//weight = 1 / (1+G^5)
						//weight1はgradHorizontalを使う
						weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
						weight1 = _mm256_mul_ps(weight1, weight1);
						weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
						weight1 = _mm256_rcp_ps(weight1);

						//weight2はgradVerticalを使う
						weight2 = _mm256_mul_ps(gradVertical, gradVertical);
						weight2 = _mm256_mul_ps(weight2, weight2);
						weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
						weight2 = _mm256_rcp_ps(weight2);

						//p = (w1p1 + w2p2) / (w1 + w2)
						mTmp0 = _mm256_mul_ps(weight1, pxHorizontal);
						mTmp1 = _mm256_fmadd_ps(weight2, pxVertical, mTmp0);
						pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
						pxSmooth = _mm256_mul_ps(pxSmooth, mTmp1);

						mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
						mTmp1 = _mm256_add_ps(mOnef, gradVertical);

						//if (1+G1) / (1+G2) > T then 135
						//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
						mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
						maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//if (1+G2) / (1+G1) > T then 45
						//cmpの結果を論理演算に使うとバグる
						mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
						maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

						//0で最初の引数をとる
						mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
						mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);
#pragma endregion
#pragma region dest
						mTmpHi = _mm256_unpackhi_ps(mKC, mDst);
						mTmpLo = _mm256_unpacklo_ps(mKC, mDst);
						mSrcHi = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x13);
						mSrcLo = _mm256_permute2f128_ps(mTmpHi, mTmpLo, 0x02);

						_mm256_storea_ps(dp + 0, mSrcLo);
						_mm256_storea_ps(dp + 8, mSrcHi);//koko ha store yori hayai kamo
#pragma endregion
						sp += 8; dp += 16;
						buf[0] += 8; buf[1] += 8; buf[2] += 8; buf[3] += 8;
					}//x-main processing 
					buf[0] -= bstep; buf[1] -= bstep; buf[2] -= bstep; buf[3] -= bstep;
					ringbuffering4(buf[0], buf[3], buf[2], buf[1]);
				}//y
#pragma endregion
			}

#ifndef __USE_GLOBAL_BUFFER__
			_mm_free(buffer);
#endif
		}
	};

	void DCCI32FC1_SIMD_LoopFusionCVParallelFor(const Mat& src_, Mat& dest, const float threshold, int _, int divy, int divx)
	{
		CV_Assert(!src_.empty());
		CV_Assert(src_.type() == CV_32FC1);

		const Mat src = (src_.data == dest.data ? src_.clone() : src_);
		if (dest.size() != src.size() * 2) dest.create(src.size() * 2, CV_32FC1);

		//const int divy = 16;
		//const int divx = 4;
		cv::parallel_for_(Range{ 0, divx * divy }, DCCI32FC1_SIMD_LoopFusionInvorker{ src_, dest, threshold, Size(divx, divy) }, -1);
	}


	void DCCI(const Mat& src, Mat& dest, const float threshold)
	{
		CV_Assert(src.channels() == 1);
		if (src.depth() != CV_32F)
		{
			Mat tmp, out;
			src.convertTo(tmp, CV_32F, 1.f/255.f);
			DCCI32FC1_SIMD_LoopFusion(tmp, out, threshold, 1);
			out.convertTo(dest, src.depth(), 255.f);
		}
		else
		{
			DCCI32FC1_SIMD_LoopFusion(src, dest, threshold, 1);
		}
	}
}