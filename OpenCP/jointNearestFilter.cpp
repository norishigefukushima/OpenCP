#include "jointNearestFilter.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	//|s-b|<thresh d = b(center)
	//else b(srgmin)
	template <class T>
	void jointNearestFilterBF_(const Mat& src, const Mat& before, const Size ksize, Mat& dest, int thresh)
	{
		if (dest.empty())dest.create(src.size(), src.depth());
		Mat sim;
		const int radiusw = ksize.width / 2;
		const int radiush = ksize.height / 2;;
		copyMakeBorder(before, sim, radiush, radiush, radiusw, radiusw, cv::BORDER_DEFAULT);

		vector<int> _space_ofs_before(ksize.area());
		int* space_ofs_before = &_space_ofs_before[0];

		int maxk = 0;
		int th = thresh*thresh;
		for (int i = -radiush; i <= radiush; i++)
		{
			for (int j = -radiusw; j <= radiusw; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radiusw)
					continue;

				space_ofs_before[maxk++] = (int)(i*sim.cols + j);
			}
		}
		const int steps = sim.cols;
		const int step = dest.cols;
		T* jptr = sim.ptr<T>(radiush); jptr += radiusw;
		T* dst = dest.ptr<T>(0);
		T* sr = (T*)src.ptr<T>(0);
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				T val0 = sr[j];
				T valc = jptr[j];

				int minv = INT_MAX;
				T mind = 0;
				for (int k = 0; k < maxk; k++)
				{
					T val = jptr[j + space_ofs_before[k]];
					int ab = (int)((val - val0)*(val - val0));
					if (ab < minv)
					{
						minv = ab;
						mind = val;
					}
				}
				if (abs(valc - mind) < th)
					dst[j] = valc;
				else
					dst[j] = mind;
			}
			jptr += steps;
			dst += step;
			sr += step;
		}
	}



	template <class T>
	void jointNearestFilterBase_(const Mat& src, const Mat& before, const Size ksize, Mat& dest)
	{
		if (dest.empty())dest.create(src.size(), src.type());
		Mat sim;
		const int radiusw = ksize.width / 2;
		const int radiush = ksize.height / 2;;
		copyMakeBorder(before, sim, radiush, radiush, radiusw, radiusw, cv::BORDER_DEFAULT);

		vector<int> _space_ofs_before(ksize.area());
		int* space_ofs_before = &_space_ofs_before[0];

		int channels = src.channels();

		int maxk = 0;
		for (int i = -radiush; i <= radiush; i++)
		{
			for (int j = -radiusw; j <= radiusw; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radiusw)
					continue;

				space_ofs_before[maxk++] = (int)(i*sim.cols*channels + channels*j);
			}
		}

		const int steps = sim.cols*channels;
		const int step = dest.cols*channels;

		T* jptr = sim.ptr<T>(radiush); jptr += radiusw*channels;
		T* dst = dest.ptr<T>(0);
		T* sr = (T*)src.ptr<T>(0);
		if (channels == 1)
		{
			for (int i = 0; i < src.rows; i++)
			{
				for (int j = 0; j < src.cols; j++)
				{
					T val0 = sr[j];
					int minv = INT_MAX;
					T mind = 0;
					for (int k = 0; k < maxk; k++)
					{
						T val = jptr[j + space_ofs_before[k]];
						int ab = (int)((val - val0)*(val - val0));
						if (ab < minv)
						{
							minv = ab;
							mind = val;
						}
					}
					dst[j] = mind;
				}
				jptr += steps;
				dst += step;
				sr += step;
			}
		}
		else if (channels == 3)
		{
			for (int i = 0; i < src.rows; i++)
			{
				for (int j = 0; j < src.cols; j++)
				{
					T val0_b = sr[3 * j + 0];
					T val0_g = sr[3 * j + 1];
					T val0_r = sr[3 * j + 2];
					int minv = INT_MAX;
					T mind_b = 0;
					T mind_g = 0;
					T mind_r = 0;

					for (int k = 0; k < maxk; k++)
					{
						T val_b = jptr[3 * j + space_ofs_before[k] + 0];
						T val_g = jptr[3 * j + space_ofs_before[k] + 1];
						T val_r = jptr[3 * j + space_ofs_before[k] + 2];
						//L2
						int ab = (int)((val_b - val0_b)*(val_b - val0_b) + (val_g - val0_g)*(val_g - val0_g) + ((val_r - val0_r)*(val_r - val0_r)));

						if (ab < minv)
						{
							minv = ab;
							mind_b = val_b;
							mind_g = val_g;
							mind_r = val_r;
						}
					}
					dst[3 * j + 0] = mind_b;
					dst[3 * j + 1] = mind_g;
					dst[3 * j + 2] = mind_r;
				}
				jptr += steps;
				dst += step;
				sr += step;
			}
		}
	}

	void jointNearestFilter_32f(const Mat& src, const Mat& before, const Size ksize, Mat& dest)
	{
		const int rem = (4 - before.cols % 4) % 4;
		const int radiusw = ksize.width / 2;
		const int radiush = ksize.height / 2;;
		Mat sim; copyMakeBorder(before, sim, radiush, radiush, radiusw, radiusw + rem, cv::BORDER_DEFAULT);
		Mat dstim(Size(src.cols + rem, src.rows), CV_32F);
		vector<int> _space_ofs_before(ksize.area());
		int* space_ofs_before = &_space_ofs_before[0];

		int maxk = 0;
		for (int i = -radiush; i <= radiush; i++)
		{
			for (int j = -radiusw; j <= radiusw; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radiusw)
					continue;

				space_ofs_before[maxk++] = (int)(i*sim.cols + j);
			}
		}
		const int steps = sim.cols;
		const int dstep = dstim.cols;
		const int step = src.cols;
		float* jptr = sim.ptr<float>(radiush); jptr += radiusw;
		float* dst = dstim.ptr<float>(0);
		float* sr = (float*)src.ptr<float>(0);

		const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		for (int i = 0; i < src.rows; i++)
		{
			int j = 0;
			for (; j < src.cols; j += 4)
			{
				const __m128 ref = _mm_loadu_ps((sr + j));
				__m128 minv = _mm_set1_ps(FLT_MAX);
				__m128 mind = _mm_setzero_ps();
				int* ofs = &space_ofs_before[0];
				const float* jptrj = jptr + j;
				for (int k = 0; k < maxk; k++)
				{
					const __m128 val = _mm_loadu_ps((jptrj + *ofs++));
					const __m128 prev = minv;
					minv = _mm_min_ps(_mm_and_ps(_mm_sub_ps(val, ref), *(const __m128*)v32f_absmask), minv);

					__m128 msk = _mm_cmpeq_ps(minv, prev);
					mind = _mm_blendv_ps(val, mind, msk);
				}
				_mm_stream_ps((dst + j), mind);
			}
			jptr += steps;
			dst += dstep;
			sr += step;
		}
		Mat(dstim(Rect(0, 0, src.cols, src.rows))).copyTo(dest);
	}

	void jointNearestFilter_16s(const Mat& src, const Mat& before, const Size ksize, Mat& dest)
	{
		const int rem = (8 - before.cols % 8) % 8;
		const int radiusw = ksize.width / 2;
		const int radiush = ksize.height / 2;;
		Mat sim; copyMakeBorder(before, sim, radiush, radiush, radiusw, radiusw + rem, cv::BORDER_DEFAULT);
		Mat dstim(Size(src.cols + rem, src.rows), CV_16S);
		vector<int> _space_ofs_before(ksize.area());
		int* space_ofs_before = &_space_ofs_before[0];

		int maxk = 0;
		for (int i = -radiush; i <= radiush; i++)
		{
			for (int j = -radiusw; j <= radiusw; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radiusw)
					continue;

				space_ofs_before[maxk++] = (int)(i*sim.cols + j);
			}
		}
		const int steps = sim.cols;
		const int dstep = dstim.cols;
		const int step = src.cols;
		short * jptr = sim.ptr<short>(radiush); jptr += radiusw;
		short * dst = dstim.ptr<short>(0);
		short * sr = (short*)src.ptr<short>(0);
		for (int i = 0; i < src.rows; i++)
		{
			int j = 0;
			for (; j < src.cols; j += 8)
			{
				const __m128i ref = _mm_loadu_si128((__m128i*)(sr + j));
				__m128i minv = _mm_set1_epi16(SHRT_MAX);
				__m128i mind = _mm_setzero_si128();
				int* ofs = &space_ofs_before[0];
				const short* jptrj = jptr + j;
				for (int k = 0; k < maxk; k++)
				{
					const __m128i val = _mm_loadu_si128((__m128i*)(jptrj + *ofs++));
					const __m128i prev = minv;
					__m128i M = _mm_max_epi16(val, ref), m = _mm_min_epi16(val, ref);

					minv = _mm_min_epi16(_mm_subs_epi16(M, m), minv);

					__m128i msk = _mm_cmpeq_epi16(minv, prev);
					mind = _mm_blendv_epi8(val, mind, msk);
				}
				_mm_stream_si128((__m128i*)(dst + j), mind);
			}
			jptr += steps;
			dst += dstep;
			sr += step;
		}
		Mat(dstim(Rect(0, 0, src.cols, src.rows))).copyTo(dest);
	}



	void jointNearestFilter_16u(const Mat& src, const Mat& before, const Size ksize, Mat& dest)
	{
		const int rem = (8 - before.cols % 8) % 8;
		const int radiusw = ksize.width / 2;
		const int radiush = ksize.height / 2;;
		Mat sim; copyMakeBorder(before, sim, radiush, radiush, radiusw, radiusw + rem, cv::BORDER_DEFAULT);
		Mat dstim(Size(src.cols + rem, src.rows), CV_16U);
		vector<int> _space_ofs_before(ksize.area());
		int* space_ofs_before = &_space_ofs_before[0];

		int maxk = 0;
		for (int i = -radiush; i <= radiush; i++)
		{
			for (int j = -radiusw; j <= radiusw; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radiusw)
					continue;

				space_ofs_before[maxk++] = (int)(i*sim.cols + j);
			}
		}
		const int steps = sim.cols;
		const int dstep = dstim.cols;
		const int step = src.cols;
		unsigned short* jptr = sim.ptr<unsigned short>(radiush); jptr += radiusw;
		unsigned short* dst = dstim.ptr<unsigned short>(0);
		unsigned short* sr = (unsigned short*)src.ptr<unsigned short>(0);
		for (int i = 0; i < src.rows; i++)
		{
			int j = 0;
			for (; j < src.cols; j += 8)
			{
				const __m128i ref = _mm_loadu_si128((__m128i*)(sr + j));
				__m128i minv = _mm_set1_epi16(USHRT_MAX);
				__m128i mind = _mm_setzero_si128();
				int* ofs = &space_ofs_before[0];
				const unsigned short* jptrj = jptr + j;
				for (int k = 0; k < maxk; k++)
				{
					const __m128i val = _mm_loadu_si128((__m128i*)(jptrj + *ofs++));
					const __m128i prev = minv;
					minv = _mm_min_epu16(_mm_add_epi16(_mm_subs_epu16(val, ref), _mm_subs_epu16(ref, val)), minv);
					__m128i msk = _mm_cmpeq_epi16(minv, prev);
					mind = _mm_blendv_epi8(val, mind, msk);
				}
				_mm_stream_si128((__m128i*)(dst + j), mind);
			}
			jptr += steps;
			dst += dstep;
			sr += step;
		}
		Mat(dstim(Rect(0, 0, src.cols, src.rows))).copyTo(dest);
	}

	static void jointNearestFilter_8u(const Mat& src, const Mat& before, const Size ksize, Mat& dest)
	{
		const int rem = (16 - before.cols % 16) % 16;
		const int radiusw = ksize.width / 2;
		const int radiush = ksize.height / 2;;
		Mat sim; copyMakeBorder(before, sim, radiush, radiush, radiusw, radiusw + rem, cv::BORDER_DEFAULT);
		Mat dstim(Size(src.cols + rem, src.rows), CV_8U);
		vector<int> _space_ofs_before(ksize.area());
		int* space_ofs_before = &_space_ofs_before[0];

		int maxk = 0;
		for (int i = -radiush; i <= radiush; i++)
		{
			for (int j = -radiusw; j <= radiusw; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radiusw)
					continue;

				space_ofs_before[maxk++] = (int)(i*sim.cols + j);
			}
		}
		const int steps = sim.cols;
		const int dstep = dstim.cols;
		const int step = src.cols;
		uchar* jptr = sim.ptr<uchar>(radiush); jptr += radiusw;
		uchar* dst = dstim.ptr<uchar>(0);
		uchar* sr = (uchar*)src.ptr<uchar>(0);
		for (int i = 0; i < src.rows; i++)
		{
			int j = 0;
			for (; j < src.cols; j += 16)
			{
				const __m128i ref = _mm_loadu_si128((__m128i*)(sr + j));
				__m128i minv = _mm_set1_epi8(255);
				__m128i mind = _mm_setzero_si128();
				int* ofs = &space_ofs_before[0];
				const uchar* jptrj = jptr + j;
				for (int k = 0; k < maxk; k++)
				{
					const __m128i val = _mm_loadu_si128((__m128i*)(jptrj + *ofs++));
					const __m128i prev = minv;
					minv = _mm_min_epu8(_mm_add_epi8(_mm_subs_epu8(val, ref), _mm_subs_epu8(ref, val)), minv);
					const __m128i msk = _mm_cmpeq_epi8(minv, prev);
					mind = _mm_blendv_epi8(val, mind, msk);
				}
				_mm_stream_si128((__m128i*)(dst + j), mind);
			}
			jptr += steps;
			dst += dstep;
			sr += step;
		}
		Mat(dstim(Rect(0, 0, src.cols, src.rows))).copyTo(dest);
	}

	void jointNearestFilter(InputArray src_, InputArray before_, Size ksize, OutputArray dest_)
	{
		if (dest_.empty()) dest_.create(src_.size(), src_.type());

		Mat src = src_.getMat();
		Mat before = before_.getMat();
		Mat dest = dest_.getMat();
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
		if (haveSSE4)
		{
			if (src.depth() == CV_8U)
			{
				jointNearestFilter_8u(src, before, ksize, dest);
			}
			else if (src.depth() == CV_16S)
			{
				jointNearestFilter_16s(src, before, ksize, dest);
			}
			else if (src.depth() == CV_16U)
			{
				jointNearestFilter_16u(src, before, ksize, dest);
			}
			else if (src.depth() == CV_32F)
			{
				jointNearestFilter_32f(src, before, ksize, dest);
			}
			else if (src.depth() == CV_64F)
			{
				jointNearestFilterBase_<double>(src, before, ksize, dest);
			}
			else
			{
				cout << "un-supported type" << endl;
			}
		}
		/*else
		{
		if(src.depth()==CV_8U)
		{
		jointNearestFilterBF_<uchar>(src,before,ksize,dest);
		}
		else if(src.depth()==CV_16S)
		{
		jointNearestFilterBF_<short>(src,before,ksize,dest);
		}
		else if(src.depth()==CV_16U)
		{
		jointNearestFilterBF_<ushort>(src,before,ksize,dest);
		}
		else if(src.depth()==CV_32F)
		{
		jointNearestFilterBF_<float>(src,before,ksize,dest);
		}
		else if(src.depth()==CV_64F)
		{
		jointNearestFilterBF_<double>(src,before,ksize,dest);
		}
		}*/
	}


	void jointNearestFilterBase(InputArray src_, InputArray before_, Size ksize, OutputArray dest_)
	{
		if (dest_.empty()) dest_.create(src_.size(), src_.type());

		Mat src = src_.getMat();
		Mat before = before_.getMat();
		Mat dest = dest_.getMat();

		if (src.depth() == CV_8U)
		{
			jointNearestFilterBase_<uchar>(src, before, ksize, dest);
		}
		else if (src.depth() == CV_16S)
		{
			jointNearestFilterBase_<short>(src, before, ksize, dest);
		}
		else if (src.depth() == CV_16U)
		{
			jointNearestFilterBase_<ushort>(src, before, ksize, dest);
		}
		else if (src.depth() == CV_32F)
		{
			jointNearestFilterBase_<float>(src, before, ksize, dest);
		}
		else if (src.depth() == CV_64F)
		{
			jointNearestFilterBase_<double>(src, before, ksize, dest);
		}
	}

	void jointNearestFilterBF(const Mat& src, const Mat& before, const Size ksize, Mat& dest, int thresh)
	{
		if (src.depth() == CV_8U)
		{
			jointNearestFilterBF_<uchar>(src, before, ksize, dest, thresh);
		}
		else if (src.depth() == CV_16S)
		{
			jointNearestFilterBF_<short>(src, before, ksize, dest, thresh);
		}
		else if (src.depth() == CV_16U)
		{
			jointNearestFilterBF_<ushort>(src, before, ksize, dest, thresh);
		}
		else if (src.depth() == CV_32F)
		{
			jointNearestFilterBF_<float>(src, before, ksize, dest, thresh);
		}
		else if (src.depth() == CV_64F)
		{
			jointNearestFilterBF_<double>(src, before, ksize, dest, thresh);
		}
	}
}