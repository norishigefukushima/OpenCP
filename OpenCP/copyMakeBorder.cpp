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

	void copyMakeBorderReplicate32FC1(Mat& src, Mat& border, const int top, const int bottom, const int left, const int right)
	{
		CV_Assert(!src.empty());
		border.create(Size(src.cols + left + right, src.rows + top + bottom), CV_32F);

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_ceil(right, 8);
		const int END = border.cols - RIGHT;

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