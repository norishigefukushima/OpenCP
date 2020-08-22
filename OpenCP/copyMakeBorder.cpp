#include "copyMakeBorder.hpp"
#include "inlineSIMDfunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{

	void copyMakeBorderLineReflect(float* s, float* d, const int srcwidth, int left, int right, int type)
	{
		for (int i = 0; i < left; i += 8)
		{
			__m256 a = _mm256_load_ps(s + i);
			a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
			a = _mm256_permute2f128_ps(a, a, 1);
			_mm256_store_ps(d - i - 8 + left, a);
		}
		memcpy(d + left, s, sizeof(float) * srcwidth);
		for (int i = 0; i < right; i += 8)
		{
			__m256 a = _mm256_load_ps(s + srcwidth - 8 - i);
			a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
			a = _mm256_permute2f128_ps(a, a, 1);
			_mm256_store_ps(d + srcwidth + i + left, a);
		}
	}

	void copyMakeBorderReplicate8UC1(Mat& src, Mat& border, const int top, const int bottom, const int left, const int right)
	{
		const int LEFT = get_simd_ceil(left, 32);
		const int end = border.cols - right;

#pragma omp parallel for
		for (int j = 0; j < border.rows; j++)
		{
			uchar* s = src.ptr<uchar>(min(max(j - top, 0), src.rows - 1));
			uchar* d = border.ptr<uchar>(j);

			__m256i ms = _mm256_set1_epi8(s[0]);
			for (int i = 0; i < LEFT; i += 32)
				_mm256_storeu_si256((__m256i*)(d + i), ms);

			memcpy(d + left, s, sizeof(uchar) * src.cols);

			for (int i = end; i < border.cols; i++)
				d[i] = (s[src.cols - 1]);
		}
	}

	void copyMakeBorderReplicate32FC1(Mat& src, Mat& border, const int top, const int bottom, const int left, const int right)
	{
		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_ceil(right, 8);
		const int END = border.cols - RIGHT;
		const int end = border.cols - right;
#if 0
		//src top line
		{
			float* s = src.ptr<float>();
			float* d = border.ptr<float>(top);

			for (int i = 0; i < LEFT; i += 8)
				_mm256_storeu_ps(d + i, _mm256_set1_ps(s[0]));
			for (int i = END; i < END + RIGHT; i += 8)
				_mm256_storeu_ps(d + i, _mm256_set1_ps(s[src.cols - 1]));
			memcpy(d + left, s, sizeof(float) * src.cols);
		}
		//border upper
		for (int j = 0; j < top; j++)
		{
			float* s = border.ptr<float>(top);
			float* d = border.ptr<float>(j);
			memcpy(d, s, sizeof(float) * border.cols);
		}

		for (int j = top + 1; j < border.rows - bottom; j++)
		{
			float* s = src.ptr<float>(j - top);
			float* d = border.ptr<float>(j);

			for (int i = 0; i < LEFT; i += 8)
				_mm256_storeu_ps(d + i, _mm256_set1_ps(s[0]));
			for (int i = END; i < END + RIGHT; i += 8)
				_mm256_storeu_ps(d + i, _mm256_set1_ps(s[src.cols - 1]));
			memcpy(d + left, s, sizeof(float) * src.cols);
		}

		//border lower
		for (int j = border.rows - bottom; j < border.rows; j++)
		{
			float* s = border.ptr<float>(border.rows - bottom - 1);
			float* d = border.ptr<float>(j);
			memcpy(d, s, sizeof(float) * border.cols);
		}
#elif 0
		for (int j = 0; j < border.rows; j++)
		{
			float* s = src.ptr<float>(min(max(j - top, 0), src.rows - 1));
			float* d = border.ptr<float>(j);

			for (int i = 0; i < LEFT; i += 8)
				_mm256_storeu_ps(d + i, _mm256_set1_ps(s[0]));
			for (int i = END; i < END + RIGHT; i += 8)
				_mm256_storeu_ps(d + i, _mm256_set1_ps(s[src.cols - 1]));
			memcpy(d + left, s, sizeof(float) * src.cols);
		}
#else 
#pragma omp parallel for
		for (int j = 0; j < border.rows; j++)
		{
			float* s = src.ptr<float>(min(max(j - top, 0), src.rows - 1));
			float* d = border.ptr<float>(j);

			__m256 ms = _mm256_set1_ps(s[0]);
			for (int i = 0; i < LEFT; i += 8)
				_mm256_storeu_ps(d + i, ms);

			memcpy(d + left, s, sizeof(float) * src.cols);

			for (int i = end; i < border.cols; i++)
				d[i] = (s[src.cols - 1]);
		}
#endif
	}

	void copyMakeBorderReplicate32FC3(Mat& src, Mat& border, const int top, const int bottom, const int left, const int right)
	{
		const int LEFT = get_simd_floor(left, 8);
		const int RIGHT = get_simd_ceil(right, 8);
		const int END = border.cols - RIGHT;
		const int end = border.cols - right;
		const int end_simd = border.cols - (right - get_simd_floor(right, 8));
		const int e = (src.cols - 1) * 3;
#if 0
		//src top line
		{
			float* s = src.ptr<float>();
			float* d = border.ptr<float>(top);

			for (int i = 0; i < LEFT; i += 8)
				_mm256_storeu_ps_color(d + 3 * i, _mm256_set1_ps(s[0]), _mm256_set1_ps(s[1]), _mm256_set1_ps(s[2]));

			for (int i = END; i < END + RIGHT; i += 8)
				_mm256_storeu_ps_color(d + 3 * i, _mm256_set1_ps(s[e]), _mm256_set1_ps(s[e + 1]), _mm256_set1_ps(s[e + 2]));

			memcpy(d + 3 * left, s, sizeof(float) * src.cols * 3);
		}
		//border upper
		for (int j = 0; j < top; j++)
		{
			float* s = border.ptr<float>(top);
			float* d = border.ptr<float>(j);
			memcpy(d, s, sizeof(float) * border.cols * 3);
		}

		for (int j = top + 1; j < border.rows - bottom; j++)
		{
			float* s = src.ptr<float>(j - top);
			float* d = border.ptr<float>(j);

			for (int i = 0; i < LEFT; i += 8)
				_mm256_storeu_ps_color(d + 3 * i, _mm256_set1_ps(s[0]), _mm256_set1_ps(s[1]), _mm256_set1_ps(s[2]));

			for (int i = END; i < END + RIGHT; i += 8)
				_mm256_storeu_ps_color(d + 3 * i, _mm256_set1_ps(s[e]), _mm256_set1_ps(s[e + 1]), _mm256_set1_ps(s[e + 2]));

			memcpy(d + 3 * left, s, sizeof(float) * src.cols * 3);
		}

		//border lower
		for (int j = border.rows - bottom; j < border.rows; j++)
		{
			float* s = border.ptr<float>(border.rows - bottom - 1);
			float* d = border.ptr<float>(j);
			memcpy(d, s, sizeof(float) * border.cols * 3);
	}
#else

#pragma omp parallel for
		for (int j = 0; j < border.rows; j++)
		{
			float* s = src.ptr<float>(min(max(j - top, 0), src.rows - 1));
			float* d = border.ptr<float>(j);

			for (int i = 0; i < LEFT; i += 8)
				_mm256_storeu_ps_color(d + 3 * i, _mm256_set1_ps(s[0]), _mm256_set1_ps(s[1]), _mm256_set1_ps(s[2]));
			for (int i = LEFT; i < left; i++)
			{
				d[3 * i + 0] = s[0];
				d[3 * i + 1] = s[1];
				d[3 * i + 2] = s[2];
			}

			memcpy(d + 3 * left, s, sizeof(float) * src.cols * 3);

			for (int i = end; i < end_simd; i += 8)
				_mm256_storeu_ps_color(d + 3 * i, _mm256_set1_ps(s[e]), _mm256_set1_ps(s[e + 1]), _mm256_set1_ps(s[e + 2]));
			for (int i = end_simd; i < border.cols; i++)
			{
				d[3 * i + 0] = s[e + 0];
				d[3 * i + 1] = s[e + 1];
				d[3 * i + 2] = s[e + 2];
			}			
		}
#endif
}

	void copyMakeBorderReplicate(InputArray src_, cv::OutputArray border_, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src_.empty());
		CV_Assert(src_.depth() == CV_8U || src_.depth() == CV_32F);
		CV_Assert(src_.channels() == 1 || src_.channels() == 3);

		Mat src = src_.getMat();
		Size bsize = Size(src.cols + left + right, src.rows + top + bottom);
		if (border_.empty() || border_.size() != bsize || border_.type() != src_.type())
			border_.create(bsize, src.type());

		Mat border = border_.getMat();
		if (src.type() == CV_8UC1)copyMakeBorderReplicate8UC1(src, border, top, bottom, left, right);
		if (src.type() == CV_32FC1)copyMakeBorderReplicate32FC1(src, border, top, bottom, left, right);
		if (src.type() == CV_32FC3)copyMakeBorderReplicate32FC3(src, border, top, bottom, left, right);
	}


	void splitCopyMakeBorderReplicate32F(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);
		//CV_Assert(1 <= unroll && unroll <= 4);

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_ceil(right, 8);
		const int END = border[0].cols - RIGHT;
		const int SIMDW = get_simd_floor(src.cols, 8);
		//src top line

		{
			float* s = src.ptr<float>();
			float* b = border[0].ptr<float>(top);
			float* g = border[1].ptr<float>(top);
			float* r = border[2].ptr<float>(top);

			for (int i = 0; i < LEFT; i += 8)
			{
				_mm256_storeu_ps(b + i, _mm256_set1_ps(s[0]));
				_mm256_storeu_ps(g + i, _mm256_set1_ps(s[1]));
				_mm256_storeu_ps(r + i, _mm256_set1_ps(s[2]));
			}
			for (int i = END; i < END + RIGHT; i += 8)
			{
				_mm256_storeu_ps(b + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_ps(g + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_ps(r + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 2]));
			}
			for (int i = 0; i < SIMDW; i += 8)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				_mm256_storeu_ps(b + i + left, mb);
				_mm256_storeu_ps(g + i + left, mg);
				_mm256_storeu_ps(r + i + left, mr);
			}
			for (int i = SIMDW; i < src.cols; i++)
			{
				b[i + left] = s[3 * i + 0];
				g[i + left] = s[3 * i + 1];
				r[i + left] = s[3 * i + 2];
			}
		}

		//border upper
		for (int j = 0; j < top; j++)
		{
			for (int c = 0; c < 3; c++)
			{
				float* s = border[c].ptr<float>(top);
				float* d = border[c].ptr<float>(j);
				memcpy(d, s, sizeof(float) * border[0].cols);
			}
		}

#pragma omp parallel for schedule(dynamic)
		for (int j = top + 1; j < border[0].rows - bottom; j++)
		{
			float* s = src.ptr<float>(j - top);
			float* b = border[0].ptr<float>(j);
			float* g = border[1].ptr<float>(j);
			float* r = border[2].ptr<float>(j);

			for (int i = 0; i < LEFT; i += 8)
			{
				_mm256_storeu_ps(b + i, _mm256_set1_ps(s[0]));
				_mm256_storeu_ps(g + i, _mm256_set1_ps(s[1]));
				_mm256_storeu_ps(r + i, _mm256_set1_ps(s[2]));
			}
			for (int i = END; i < END + RIGHT; i += 8)
			{
				_mm256_storeu_ps(b + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_ps(g + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_ps(r + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 2]));
			}

			for (int i = 0; i < SIMDW; i += 8)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				_mm256_storeu_ps(b + i + left, mb);
				_mm256_storeu_ps(g + i + left, mg);
				_mm256_storeu_ps(r + i + left, mr);
			}
			for (int i = SIMDW; i < src.cols; i++)
			{
				b[i + left] = s[3 * i + 0];
				g[i + left] = s[3 * i + 1];
				r[i + left] = s[3 * i + 2];
			}
		}

		//border lower
		for (int j = border[0].rows - bottom; j < border[0].rows; j++)
		{
			for (int c = 0; c < 3; c++)
			{
				float* s = border[c].ptr<float>(border[0].rows - bottom - 1);
				float* d = border[c].ptr<float>(j);
				memcpy(d, s, sizeof(float) * border[0].cols);
			}
		}
	}

	void splitCopyMakeBorder(cv::InputArray src, cv::OutputArrayOfArrays dest, const int top, const int bottom, const int left, const int right, const int borderType, const cv::Scalar& color)
	{
		Mat s = src.getMat();

		vector<Mat> dst;
		if (dest.empty())
		{
			const Size borderSize = Size(s.cols + left + right, s.rows + top + bottom);

			dest.create(3, 1, src.type());
			dest.getMatVector(dst);
			dst[0].create(borderSize, src.depth());
			dst[1].create(borderSize, src.depth());
			dst[2].create(borderSize, src.depth());

			dest.getMatRef(0) = dst[0];
			dest.getMatRef(1) = dst[1];
			dest.getMatRef(2) = dst[2];
		}
		else
		{
			dest.getMatVector(dst);
		}

		if (borderType == cv::BORDER_REPLICATE)
		{
			if (src.depth() == CV_32F)splitCopyMakeBorderReplicate32F(s, dst, top, bottom, left, right);
			else
			{
				cout << "not implemented in splitCopyMakeBorder" << endl;
			}
		}
		else
		{
			cout << "not implemented in splitCopyMakeBorder" << endl;
		}
	}

}