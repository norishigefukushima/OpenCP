#include "copyMakeBorder.hpp"
#include "inlineSIMDfunctions.hpp"
#include "debugcp.hpp"
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

#pragma region copyMakeBorderReplicate
	static void copyMakeBorderReplicate8UC1(const Mat& src, Mat& border, const int top, const int bottom, const int left, const int right)
	{
		const int LEFT = get_simd_ceil(left, 32);
		const int end = border.cols - right;

#pragma omp parallel for schedule (dynamic)
		for (int j = 0; j < border.rows; j++)
		{
			const uchar* s = src.ptr<uchar>(min(max(j - top, 0), src.rows - 1));
			uchar* d = border.ptr<uchar>(j);

			__m256i ms = _mm256_set1_epi8(s[0]);
			for (int i = 0; i < LEFT; i += 32)
				_mm256_storeu_si256((__m256i*)(d + i), ms);

			memcpy(d + left, s, sizeof(uchar) * src.cols);

			for (int i = end; i < border.cols; i++)
				d[i] = (s[src.cols - 1]);
		}
	}

	static void copyMakeBorderReplicate16SC1(const Mat& src, Mat& border, const int top, const int bottom, const int left, const int right)
	{
		const int LEFT = get_simd_ceil(left, 16);
		const int end = border.cols - right;

#pragma omp parallel for schedule (dynamic)
		for (int j = 0; j < border.rows; j++)
		{
			const short* s = src.ptr<short>(min(max(j - top, 0), src.rows - 1));
			short* d = border.ptr<short>(j);

			__m256i ms = _mm256_set1_epi16(s[0]);
			for (int i = 0; i < LEFT; i += 16)
				_mm256_storeu_si256((__m256i*)(d + i), ms);

			memcpy(d + left, s, sizeof(short) * src.cols);

			for (int i = end; i < border.cols; i++)
				d[i] = (s[src.cols - 1]);
		}
	}

	static void copyMakeBorderReplicate32FC1(const Mat& src, Mat& border, const int top, const int bottom, const int left, const int right)
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
#pragma omp parallel for schedule (dynamic)
		for (int j = 0; j < border.rows; j++)
		{
			const float* s = src.ptr<float>(min(max(j - top, 0), src.rows - 1));
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

	static void copyMakeBorderReplicate32FC3(const Mat& src, Mat& border, const int top, const int bottom, const int left, const int right)
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

#pragma omp parallel for  schedule (dynamic)
		for (int j = 0; j < border.rows; j++)
		{
			const float* s = src.ptr<float>(min(max(j - top, 0), src.rows - 1));
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

	static void copyMakeBorderReplicate8UC3(const Mat& src, Mat& border, const int top, const int bottom, const int left, const int right)
	{
		const int LEFT = get_simd_floor(left, 32);
		const int end = border.cols - right;
		const int end_simd = border.cols - (right - get_simd_floor(right, 8));
		const int e = (src.cols - 1) * 3;

#pragma omp parallel for schedule (dynamic)
		for (int j = 0; j < border.rows; j++)
		{
			const uchar* s = src.ptr<uchar>(min(max(j - top, 0), src.rows - 1));
			uchar* d = border.ptr<uchar>(j);

			for (int i = 0; i < LEFT; i += 32)
				_mm256_store_epi8_color(d + 3 * i, _mm256_set1_epi8(s[0]), _mm256_set1_epi8(s[1]), _mm256_set1_epi8(s[2]));
			for (int i = LEFT; i < left; i++)
			{
				d[3 * i + 0] = s[0];
				d[3 * i + 1] = s[1];
				d[3 * i + 2] = s[2];
			}

			memcpy(d + 3 * left, s, sizeof(uchar) * src.cols * 3);

			for (int i = end; i < end_simd; i += 32)
				_mm256_store_epi8_color(d + 3 * i, _mm256_set1_epi8(s[e]), _mm256_set1_epi8(s[e + 1]), _mm256_set1_epi8(s[e + 2]));
			for (int i = end_simd; i < border.cols; i++)
			{
				d[3 * i + 0] = s[e + 0];
				d[3 * i + 1] = s[e + 1];
				d[3 * i + 2] = s[e + 2];
			}
		}
	}

	void copyMakeBorderReplicate(InputArray src_, cv::OutputArray border_, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src_.empty());
		CV_Assert(src_.depth() == CV_8U || src_.depth() == CV_32F || src_.depth() == CV_16S || src_.depth() == CV_16U || src_.depth() == CV_16F);
		CV_Assert(src_.channels() == 1 || src_.channels() == 3);

		Mat src = src_.getMat();
		Size bsize = Size(src.cols + left + right, src.rows + top + bottom);
		if (border_.empty() || border_.size() != bsize || border_.type() != src_.type())
		{
			border_.create(bsize, src.type());
		}

		Mat border = border_.getMat();

		if (src.type() == CV_8UC1)			copyMakeBorderReplicate8UC1(src, border, top, bottom, left, right);
		else if (src.type() == CV_8UC3)		copyMakeBorderReplicate8UC3(src, border, top, bottom, left, right);
		else if (src.type() == CV_32FC1)	copyMakeBorderReplicate32FC1(src, border, top, bottom, left, right);
		else if (src.type() == CV_32FC3)	copyMakeBorderReplicate32FC3(src, border, top, bottom, left, right);
		else if (src.type() == CV_16FC1 || src.type() == CV_16SC1 || src.type() == CV_16UC1) copyMakeBorderReplicate16SC1(src, border, top, bottom, left, right);
	}
#pragma endregion

#pragma region splitCopyMakeBordereplicate
	void splitCopyMakeBorderReplicate32F(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_ceil(right, 8);
		const int END = border[0].cols - RIGHT;
		const int end = border[0].cols - right;
		const int end_simd = border[0].cols - (right - get_simd_floor(right, 8));
		const int SIMDW = get_simd_floor(src.cols, 8);

#if 0
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
#else
#pragma omp parallel for //schedule (dynamic)
		for (int j = 0; j < border[0].rows; j++)
		{
			float* s = src.ptr<float>(border_replicate(j - top, src.rows - 1));

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

			/*for (int i = end; i < end_simd; i += 8)
			{
				_mm256_storeu_ps(b + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_ps(g + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_ps(r + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 2]));
			}
			for (int i = end_simd; i < border[0].cols; i++)
			{
				b[i] = s[3 * (src.cols - 1) + 0];
				g[i] = s[3 * (src.cols - 1) + 1];
				r[i] = s[3 * (src.cols - 1) + 2];
			}*/
		}
#endif
	}

	void splitCopyMakeBorderReplicate16S(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);

		const int LEFT = get_simd_ceil(left, 16);
		const int RIGHT = get_simd_ceil(right, 16);
		const int END = border[0].cols - RIGHT;
		const int end = border[0].cols - right;
		const int end_simd = border[0].cols - (right - get_simd_floor(right, 16));
		const int SIMDW = get_simd_floor(src.cols, 16);

#pragma omp parallel for //schedule (dynamic)
		for (int j = 0; j < border[0].rows; j++)
		{
			short* s = src.ptr<short>(border_replicate(j - top, src.rows - 1));

			short* b = border[0].ptr<short>(j);
			short* g = border[1].ptr<short>(j);
			short* r = border[2].ptr<short>(j);

			for (int i = 0; i < LEFT; i += 16)
			{
				_mm256_storeu_si256((__m256i*)(b + i), _mm256_set1_epi16(s[0]));
				_mm256_storeu_si256((__m256i*)(g + i), _mm256_set1_epi16(s[1]));
				_mm256_storeu_si256((__m256i*)(r + i), _mm256_set1_epi16(s[2]));
			}
			for (int i = END; i < END + RIGHT; i += 16)
			{
				_mm256_storeu_si256((__m256i*)(b + i), _mm256_set1_epi16(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_si256((__m256i*)(g + i), _mm256_set1_epi16(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_si256((__m256i*)(r + i), _mm256_set1_epi16(s[3 * (src.cols - 1) + 2]));
			}

			for (int i = 0; i < SIMDW; i += 16)
			{
				__m256i mb, mg, mr;
				_mm256_load_cvtepi16bgr2planar_epi16(s + 3 * i, mb, mg, mr);
				_mm256_storeu_si256((__m256i*)(b + i + left), mb);
				_mm256_storeu_si256((__m256i*)(g + i + left), mg);
				_mm256_storeu_si256((__m256i*)(r + i + left), mr);
			}
			for (int i = SIMDW; i < src.cols; i++)
			{
				b[i + left] = s[3 * i + 0];
				g[i + left] = s[3 * i + 1];
				r[i + left] = s[3 * i + 2];
			}

			/*for (int i = end; i < end_simd; i += 8)
			{
				_mm256_storeu_ps(b + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_ps(g + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_ps(r + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 2]));
			}
			for (int i = end_simd; i < border[0].cols; i++)
			{
				b[i] = s[3 * (src.cols - 1) + 0];
				g[i] = s[3 * (src.cols - 1) + 1];
				r[i] = s[3 * (src.cols - 1) + 2];
			}*/
		}
	}

	void splitCopyMakeBorderReplicate8U(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);

		const int LEFT = get_simd_ceil(left, 32);
		const int RIGHT = get_simd_ceil(right, 32);
		const int END = border[0].cols - RIGHT;
		const int end = border[0].cols - right;
		const int end_simd = border[0].cols - (right - get_simd_floor(right, 32));
		const int SIMDW = get_simd_floor(src.cols, 32);

#pragma omp parallel for
		for (int j = 0; j < border[0].rows; j++)
		{
			uchar* s = src.ptr<uchar>(border_replicate(j - top, src.rows - 1));

			uchar* b = border[0].ptr<uchar>(j);
			uchar* g = border[1].ptr<uchar>(j);
			uchar* r = border[2].ptr<uchar>(j);

			for (int i = 0; i < LEFT; i += 32)
			{
				_mm256_storeu_si256((__m256i*)(b + i), _mm256_set1_epi8(s[0]));
				_mm256_storeu_si256((__m256i*)(g + i), _mm256_set1_epi8(s[1]));
				_mm256_storeu_si256((__m256i*)(r + i), _mm256_set1_epi8(s[2]));
			}
			for (int i = END; i < END + RIGHT; i += 32)
			{
				_mm256_storeu_si256((__m256i*)(b + i), _mm256_set1_epi8(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_si256((__m256i*)(g + i), _mm256_set1_epi8(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_si256((__m256i*)(r + i), _mm256_set1_epi8(s[3 * (src.cols - 1) + 2]));
			}

			for (int i = 0; i < SIMDW; i += 32)
			{
				__m256i mb, mg, mr;
				_mm256_load_cvtepu8bgr2planar_si256(s + 3 * i, mb, mg, mr);
				_mm256_storeu_si256((__m256i*)(b + i + left), mb);
				_mm256_storeu_si256((__m256i*)(g + i + left), mg);
				_mm256_storeu_si256((__m256i*)(r + i + left), mr);
			}
			for (int i = SIMDW; i < src.cols; i++)
			{
				b[i + left] = s[3 * i + 0];
				g[i + left] = s[3 * i + 1];
				r[i + left] = s[3 * i + 2];
			}

			/*for (int i = end; i < end_simd; i += 8)
			{
				_mm256_storeu_ps(b + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_ps(g + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_ps(r + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 2]));
			}
			for (int i = end_simd; i < border[0].cols; i++)
			{
				b[i] = s[3 * (src.cols - 1) + 0];
				g[i] = s[3 * (src.cols - 1) + 1];
				r[i] = s[3 * (src.cols - 1) + 2];
			}*/
		}
	}

#ifdef __AVX512F__
	void splitCopyMakeBorderReplicate32F_AVX512(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);

		const int l_floor = get_simd_floor(left, 16);
		const int r_floor = get_simd_floor(right, 16);

		const bool isAligned = (left % 16 == 0) && (right % 16 == 0) && (src.cols % 16 == 0);
		if (isAligned)
		{
#pragma omp parallel for //schedule (dynamic)
			for (int y = 0; y < border[0].rows; y++)
			{
				const float* sptr = src.ptr<float>(min(src.rows - 1, max(0, y - top)));
				float* sptr_b = border[0].ptr<float>(y);
				float* sptr_g = border[1].ptr<float>(y);
				float* sptr_r = border[2].ptr<float>(y);

				__m512 rep_b = _mm512_set1_ps(sptr[0]);
				__m512 rep_g = _mm512_set1_ps(sptr[1]);
				__m512 rep_r = _mm512_set1_ps(sptr[2]);
				for (int x = 0; x < l_floor; x += 16)
				{
					_mm512_store_ps(sptr_b + x, rep_b);
					_mm512_store_ps(sptr_g + x, rep_g);
					_mm512_store_ps(sptr_r + x, rep_r);
				}

				sptr_b += left;
				sptr_g += left;
				sptr_r += left;
				for (int x = 0; x < src.cols; x += 16)
				{
					__m512 mb, mg, mr;
					_mm512_load_cvtps_bgr2planar_ps(sptr + 3 * x, mb, mg, mr);
					_mm512_store_ps(sptr_b, mb);
					_mm512_store_ps(sptr_g, mg);
					_mm512_store_ps(sptr_r, mr);
					sptr_b += 16; sptr_g += 16; sptr_r += 16;
					/*_mm512_load_cvtps_bgr2planar_ps(sptr + 3 * (x + 16), mb, mg, mr);
					_mm512_storeu_ps(sptr_b + 16, mb);
					_mm512_storeu_ps(sptr_g + 16, mg);
					_mm512_storeu_ps(sptr_r + 16, mr);
					sptr_b += 32; sptr_g += 32; sptr_r += 32;*/
					/*
					_mm512_load_cvtps_bgr2planar_ps(sptr + 3 * (x + 32), mb, mg, mr);
					_mm512_storeu_ps(sptr_b + 32, mb);
					_mm512_storeu_ps(sptr_g + 32, mg);
					_mm512_storeu_ps(sptr_r + 32, mr);
					_mm512_load_cvtps_bgr2planar_ps(sptr + 3 * (x + 48), mb, mg, mr);
					_mm512_storeu_ps(sptr_b + 48, mb);
					_mm512_storeu_ps(sptr_g + 48, mg);
					_mm512_storeu_ps(sptr_r + 48, mr);
					//sptr_b += 64; sptr_g += 64; sptr_r += 64;*/
				}

				rep_b = _mm512_set1_ps(sptr[3 * (src.cols - 1) + 0]);
				rep_g = _mm512_set1_ps(sptr[3 * (src.cols - 1) + 1]);
				rep_r = _mm512_set1_ps(sptr[3 * (src.cols - 1) + 2]);
				for (int x = 0; x < r_floor; x += 16)
				{
					_mm512_store_ps(sptr_b, rep_b);
					_mm512_store_ps(sptr_g, rep_g);
					_mm512_store_ps(sptr_r, rep_r);
				}
			}
		}
		else
		{
			const unsigned short l_mask = get_simd512_residualmask_epi32(left);
			const unsigned short r_mask = get_simd512_residualmask_epi32(right);
#pragma omp parallel for //schedule (dynamic)
			for (int y = 0; y < border[0].rows; y++)
			{
				const float* sptr = src.ptr<float>(min(src.rows - 1, max(0, y - top)));
				float* sptr_b = border[0].ptr<float>(y);
				float* sptr_g = border[1].ptr<float>(y);
				float* sptr_r = border[2].ptr<float>(y);

				__m512 rep_b = _mm512_set1_ps(sptr[0]);
				__m512 rep_g = _mm512_set1_ps(sptr[1]);
				__m512 rep_r = _mm512_set1_ps(sptr[2]);
				for (int x = 0; x < l_floor; x += 16)
				{
					_mm512_store_ps(sptr_b + x, rep_b);
					_mm512_store_ps(sptr_g + x, rep_g);
					_mm512_store_ps(sptr_r + x, rep_r);
				}
				{
					_mm512_mask_store_ps(sptr_b + l_floor, l_mask, rep_b);
					_mm512_mask_store_ps(sptr_g + l_floor, l_mask, rep_g);
					_mm512_mask_store_ps(sptr_r + l_floor, l_mask, rep_r);
				}

				sptr_b += left;
				sptr_g += left;
				sptr_r += left;
				for (int x = 0; x < src.cols; x += 16)
				{
					__m512 mb, mg, mr;
					_mm512_load_cvtps_bgr2planar_ps(sptr + 3 * x, mb, mg, mr);
					_mm512_storeu_ps(sptr_b, mb);
					_mm512_storeu_ps(sptr_g, mg);
					_mm512_storeu_ps(sptr_r, mr);
					sptr_b += 16; sptr_g += 16; sptr_r += 16;
					/*_mm512_load_cvtps_bgr2planar_ps(sptr + 3 * (x + 16), mb, mg, mr);
					_mm512_storeu_ps(sptr_b + 16, mb);
					_mm512_storeu_ps(sptr_g + 16, mg);
					_mm512_storeu_ps(sptr_r + 16, mr);
					sptr_b += 32; sptr_g += 32; sptr_r += 32;*/
					/*
					_mm512_load_cvtps_bgr2planar_ps(sptr + 3 * (x + 32), mb, mg, mr);
					_mm512_storeu_ps(sptr_b + 32, mb);
					_mm512_storeu_ps(sptr_g + 32, mg);
					_mm512_storeu_ps(sptr_r + 32, mr);
					_mm512_load_cvtps_bgr2planar_ps(sptr + 3 * (x + 48), mb, mg, mr);
					_mm512_storeu_ps(sptr_b + 48, mb);
					_mm512_storeu_ps(sptr_g + 48, mg);
					_mm512_storeu_ps(sptr_r + 48, mr);
					//sptr_b += 64; sptr_g += 64; sptr_r += 64;*/
				}

				rep_b = _mm512_set1_ps(sptr[3 * (src.cols - 1) + 0]);
				rep_g = _mm512_set1_ps(sptr[3 * (src.cols - 1) + 1]);
				rep_r = _mm512_set1_ps(sptr[3 * (src.cols - 1) + 2]);
				for (int x = 0; x < r_floor; x += 16)
				{
					_mm512_storeu_ps(sptr_b, rep_b);
					_mm512_storeu_ps(sptr_g, rep_g);
					_mm512_storeu_ps(sptr_r, rep_r);
				}
				{
					_mm512_mask_store_ps(sptr_b, r_mask, rep_b);
					_mm512_mask_store_ps(sptr_g, r_mask, rep_g);
					_mm512_mask_store_ps(sptr_r, r_mask, rep_r);
				}
			}
		}
	}

	void splitCopyMakeBorderReplicate16S_AVX512(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);

		const int LEFT = get_simd_ceil(left, 16);
		const int RIGHT = get_simd_ceil(right, 16);
		const int END = border[0].cols - RIGHT;
		const int end = border[0].cols - right;
		const int end_simd = border[0].cols - (right - get_simd_floor(right, 16));
		const int SIMDW = get_simd_floor(src.cols, 16);

#pragma omp parallel for //schedule (dynamic)
		for (int j = 0; j < border[0].rows; j++)
		{
			short* s = src.ptr<short>(border_replicate(j - top, src.rows - 1));

			short* b = border[0].ptr<short>(j);
			short* g = border[1].ptr<short>(j);
			short* r = border[2].ptr<short>(j);

			for (int i = 0; i < LEFT; i += 16)
			{
				_mm256_storeu_si256((__m256i*)(b + i), _mm256_set1_epi16(s[0]));
				_mm256_storeu_si256((__m256i*)(g + i), _mm256_set1_epi16(s[1]));
				_mm256_storeu_si256((__m256i*)(r + i), _mm256_set1_epi16(s[2]));
			}
			for (int i = END; i < END + RIGHT; i += 16)
			{
				_mm256_storeu_si256((__m256i*)(b + i), _mm256_set1_epi16(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_si256((__m256i*)(g + i), _mm256_set1_epi16(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_si256((__m256i*)(r + i), _mm256_set1_epi16(s[3 * (src.cols - 1) + 2]));
			}

			for (int i = 0; i < SIMDW; i += 16)
			{
				__m256i mb, mg, mr;
				_mm256_load_cvtepi16bgr2planar_epi16(s + 3 * i, mb, mg, mr);
				_mm256_storeu_si256((__m256i*)(b + i + left), mb);
				_mm256_storeu_si256((__m256i*)(g + i + left), mg);
				_mm256_storeu_si256((__m256i*)(r + i + left), mr);
			}
			for (int i = SIMDW; i < src.cols; i++)
			{
				b[i + left] = s[3 * i + 0];
				g[i + left] = s[3 * i + 1];
				r[i + left] = s[3 * i + 2];
			}

			/*for (int i = end; i < end_simd; i += 8)
			{
				_mm256_storeu_ps(b + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_ps(g + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_ps(r + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 2]));
			}
			for (int i = end_simd; i < border[0].cols; i++)
			{
				b[i] = s[3 * (src.cols - 1) + 0];
				g[i] = s[3 * (src.cols - 1) + 1];
				r[i] = s[3 * (src.cols - 1) + 2];
			}*/
		}
	}

	void splitCopyMakeBorderReplicate8U_AVX512(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);

		const int LEFT = get_simd_ceil(left, 32);
		const int RIGHT = get_simd_ceil(right, 32);
		const int END = border[0].cols - RIGHT;
		const int end = border[0].cols - right;
		const int end_simd = border[0].cols - (right - get_simd_floor(right, 32));
		const int SIMDW = get_simd_floor(src.cols, 32);

#pragma omp parallel for
		for (int j = 0; j < border[0].rows; j++)
		{
			uchar* s = src.ptr<uchar>(border_replicate(j - top, src.rows - 1));

			uchar* b = border[0].ptr<uchar>(j);
			uchar* g = border[1].ptr<uchar>(j);
			uchar* r = border[2].ptr<uchar>(j);

			for (int i = 0; i < LEFT; i += 32)
			{
				_mm256_storeu_si256((__m256i*)(b + i), _mm256_set1_epi8(s[0]));
				_mm256_storeu_si256((__m256i*)(g + i), _mm256_set1_epi8(s[1]));
				_mm256_storeu_si256((__m256i*)(r + i), _mm256_set1_epi8(s[2]));
			}
			for (int i = END; i < END + RIGHT; i += 32)
			{
				_mm256_storeu_si256((__m256i*)(b + i), _mm256_set1_epi8(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_si256((__m256i*)(g + i), _mm256_set1_epi8(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_si256((__m256i*)(r + i), _mm256_set1_epi8(s[3 * (src.cols - 1) + 2]));
			}

			for (int i = 0; i < SIMDW; i += 32)
			{
				__m256i mb, mg, mr;
				_mm256_load_cvtepu8bgr2planar_si256(s + 3 * i, mb, mg, mr);
				_mm256_storeu_si256((__m256i*)(b + i + left), mb);
				_mm256_storeu_si256((__m256i*)(g + i + left), mg);
				_mm256_storeu_si256((__m256i*)(r + i + left), mr);
			}
			for (int i = SIMDW; i < src.cols; i++)
			{
				b[i + left] = s[3 * i + 0];
				g[i + left] = s[3 * i + 1];
				r[i + left] = s[3 * i + 2];
			}

			/*for (int i = end; i < end_simd; i += 8)
			{
				_mm256_storeu_ps(b + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 0]));
				_mm256_storeu_ps(g + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 1]));
				_mm256_storeu_ps(r + i, _mm256_set1_ps(s[3 * (src.cols - 1) + 2]));
			}
			for (int i = end_simd; i < border[0].cols; i++)
			{
				b[i] = s[3 * (src.cols - 1) + 0];
				g[i] = s[3 * (src.cols - 1) + 1];
				r[i] = s[3 * (src.cols - 1) + 2];
			}*/
		}
	}
#endif


	void splitCopyMakeBorderReflect10132F(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_ceil(right, 8);
		const int END = border[0].cols - RIGHT;
		const int end = border[0].cols - right;
		const int end_simd = border[0].cols - (right - get_simd_floor(right, 8));
		const int SIMDW = get_simd_floor(src.cols, 8);
		const int e = src.cols - 1;
#pragma omp parallel for
		for (int j = 0; j < border[0].rows; j++)
		{
			float* s = src.ptr<float>(border_reflect101(j - top, src.rows - 1));

			float* b = border[0].ptr<float>(j);
			float* g = border[1].ptr<float>(j);
			float* r = border[2].ptr<float>(j);

			for (int i = 0; i < left; i++)
			{
				b[i] = s[3 * border_min_reflect101(i - left) + 0];
				g[i] = s[3 * border_min_reflect101(i - left) + 1];
				r[i] = s[3 * border_min_reflect101(i - left) + 2];
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

			for (int i = src.cols; i < border[0].cols; i++)
			{
				b[i] = s[3 * border_max_reflect101(i - left, e) + 0];
				g[i] = s[3 * border_max_reflect101(i - left, e) + 1];
				r[i] = s[3 * border_max_reflect101(i - left, e) + 2];
			}
		}
	}

	void splitCopyMakeBorderReflect1018U(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);

		const int LEFT = get_simd_ceil(left, 32);
		const int RIGHT = get_simd_floor(right, 32);
		const int END = border[0].cols - RIGHT;
		const int end = border[0].cols - right;
		const int end_simd = border[0].cols - (right - get_simd_floor(right, 32));
		const int SIMDW = get_simd_floor(src.cols, 8);

		const int e = src.cols - 1;
#pragma omp parallel for
		for (int j = 0; j < border[0].rows; j++)
		{
			uchar* s = src.ptr<uchar>(border_reflect101(j - top, src.rows - 1));

			uchar* b = border[0].ptr<uchar>(j);
			uchar* g = border[1].ptr<uchar>(j);
			uchar* r = border[2].ptr<uchar>(j);

			//for (int i = 0; i < LEFT; i += 32)
			for (int i = 0; i < left; i++)
			{
				b[i] = s[3 * border_min_reflect101(i - left) + 0];
				g[i] = s[3 * border_min_reflect101(i - left) + 1];
				r[i] = s[3 * border_min_reflect101(i - left) + 2];
			}

			for (int i = 0; i < SIMDW; i += 32)
			{
				__m256i mb, mg, mr;
				_mm256_load_cvtepu8bgr2planar_si256(s + 3 * i, mb, mg, mr);
				_mm256_storeu_si256((__m256i*)(b + i + left), mb);
				_mm256_storeu_si256((__m256i*)(g + i + left), mg);
				_mm256_storeu_si256((__m256i*)(r + i + left), mr);
			}
			for (int i = SIMDW; i < src.cols; i++)
			{
				b[i + left] = s[3 * i + 0];
				g[i + left] = s[3 * i + 1];
				r[i + left] = s[3 * i + 2];
			}

			for (int i = src.cols; i < border[0].cols; i++)
			{
				b[i] = s[3 * border_max_reflect101(i - left, e) + 0];
				g[i] = s[3 * border_max_reflect101(i - left, e) + 1];
				r[i] = s[3 * border_max_reflect101(i - left, e) + 2];
			}
		}
	}

	void splitCopyMakeBorderReflect32F(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_ceil(right, 8);
		const int END = border[0].cols - RIGHT;
		const int end = border[0].cols - right;
		const int end_simd = border[0].cols - (right - get_simd_floor(right, 8));
		const int SIMDW = get_simd_floor(src.cols, 8);
		const int e = src.cols - 1;
#pragma omp parallel for
		for (int j = 0; j < border[0].rows; j++)
		{
			float* s = src.ptr<float>(border_reflect(j - top, src.rows - 1));

			float* b = border[0].ptr<float>(j);
			float* g = border[1].ptr<float>(j);
			float* r = border[2].ptr<float>(j);

			for (int i = 0; i < left; i++)
			{
				b[i] = s[3 * border_min_reflect(i - left) + 0];
				g[i] = s[3 * border_min_reflect(i - left) + 1];
				r[i] = s[3 * border_min_reflect(i - left) + 2];
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

			for (int i = src.cols; i < border[0].cols; i++)
			{
				b[i] = s[3 * border_max_reflect(i - left, e) + 0];
				g[i] = s[3 * border_max_reflect(i - left, e) + 1];
				r[i] = s[3 * border_max_reflect(i - left, e) + 2];
			}
		}
	}

	void splitCopyMakeBorderReflect8U(Mat& src, vector<Mat>& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == 3);

		const int LEFT = get_simd_ceil(left, 32);
		const int RIGHT = get_simd_floor(right, 32);
		const int END = border[0].cols - RIGHT;
		const int end = border[0].cols - right;
		const int end_simd = border[0].cols - (right - get_simd_floor(right, 32));
		const int SIMDW = get_simd_floor(src.cols, 8);

		const int e = src.cols - 1;
#pragma omp parallel for
		for (int j = 0; j < border[0].rows; j++)
		{
			uchar* s = src.ptr<uchar>(border_reflect(j - top, src.rows - 1));

			uchar* b = border[0].ptr<uchar>(j);
			uchar* g = border[1].ptr<uchar>(j);
			uchar* r = border[2].ptr<uchar>(j);

			//for (int i = 0; i < LEFT; i += 32)
			for (int i = 0; i < left; i++)
			{
				b[i] = s[3 * border_min_reflect(i - left) + 0];
				g[i] = s[3 * border_min_reflect(i - left) + 1];
				r[i] = s[3 * border_min_reflect(i - left) + 2];
			}

			for (int i = 0; i < SIMDW; i += 32)
			{
				__m256i mb, mg, mr;
				_mm256_load_cvtepu8bgr2planar_si256(s + 3 * i, mb, mg, mr);
				_mm256_storeu_si256((__m256i*)(b + i + left), mb);
				_mm256_storeu_si256((__m256i*)(g + i + left), mg);
				_mm256_storeu_si256((__m256i*)(r + i + left), mr);
			}
			for (int i = SIMDW; i < src.cols; i++)
			{
				b[i + left] = s[3 * i + 0];
				g[i + left] = s[3 * i + 1];
				r[i + left] = s[3 * i + 2];
			}

			for (int i = src.cols; i < border[0].cols; i++)
			{
				b[i] = s[3 * border_max_reflect(i - left, e) + 0];
				g[i] = s[3 * border_max_reflect(i - left, e) + 1];
				r[i] = s[3 * border_max_reflect(i - left, e) + 2];
			}
		}
	}


	void splitCopyMakeBorder(cv::InputArray src, cv::OutputArrayOfArrays dest, const int top, const int bottom, const int left, const int right, const int borderType, const cv::Scalar& color)
	{
		CV_Assert(!src.empty());
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_16F);
		CV_Assert(borderType == cv::BORDER_REPLICATE || borderType == cv::BORDER_REFLECT101 || borderType == cv::BORDER_REFLECT);

		Mat s = src.getMat();
		const Size borderSize = Size(s.cols + left + right, s.rows + top + bottom);

		vector<Mat> dst;
		if (dest.empty())
		{
			dest.create(src.channels(), 1, src.depth());

			dest.getMatVector(dst);
			for (int i = 0; i < dst.size(); i++)
			{
				if (dst[i].empty())
				{
					dst[i].create(borderSize, src.depth());
					dest.getMatRef(i) = dst[i];
				}
			}
		}
		else
		{
			dest.getMatVector(dst);
			for (int i = 0; i < dst.size(); i++)
			{
				if (dst[i].empty() ||
					dst[i].depth() != src.depth() ||
					dst[i].cols != src.size().width + left + right ||
					dst[i].cols != src.size().height + top + bottom)
				{
					dst[i].create(borderSize, src.depth());
					dest.getMatRef(i) = dst[i];
				}
			}
		}

		if (borderType == cv::BORDER_REPLICATE)
		{
#ifdef __AVX512F__
			if (src.depth() == CV_8U)		splitCopyMakeBorderReplicate8U_AVX512(s, dst, top, bottom, left, right);
			else if (src.depth() == CV_32F) splitCopyMakeBorderReplicate32F_AVX512(s, dst, top, bottom, left, right);
			else if (src.depth() == CV_16F || src.depth() == CV_16S || src.depth() == CV_16U) splitCopyMakeBorderReplicate16S_AVX512(s, dst, top, bottom, left, right);
#else
			if (src.depth() == CV_8U)		splitCopyMakeBorderReplicate8U(s, dst, top, bottom, left, right);
			else if (src.depth() == CV_32F) splitCopyMakeBorderReplicate32F(s, dst, top, bottom, left, right);
			else if (src.depth() == CV_16F || src.depth() == CV_16S || src.depth() == CV_16U) splitCopyMakeBorderReplicate16S(s, dst, top, bottom, left, right);
#endif
		}
		else if (borderType == cv::BORDER_REFLECT101)
		{
			if (src.depth() == CV_8U)		splitCopyMakeBorderReflect1018U(s, dst, top, bottom, left, right);
			else if (src.depth() == CV_32F) splitCopyMakeBorderReflect10132F(s, dst, top, bottom, left, right);
		}
		else if (borderType == cv::BORDER_REFLECT)
		{
			if (src.depth() == CV_8U)		splitCopyMakeBorderReflect8U(s, dst, top, bottom, left, right);
			else if (src.depth() == CV_32F) splitCopyMakeBorderReflect32F(s, dst, top, bottom, left, right);
		}
		else
		{
			cout << "not implemented in splitCopyMakeBorder" << endl;
		}
	}
#pragma endregion

}