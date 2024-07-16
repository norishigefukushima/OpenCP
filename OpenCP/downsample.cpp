#include "downsample.hpp"
#include "inlineMathFunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	string getDowsamplingMethod(const Downsample method)
	{
		string ret = "";
		switch (method)
		{
		case Downsample::INTER_NEAREST:
			ret = "INTER_NEAREST"; break;
		case Downsample::INTER_LINEAR:
			ret = "INTER_LINEAR"; break;
		case Downsample::INTER_CUBIC:
			ret = "INTER_CUBIC"; break;
		case Downsample::INTER_AREA:
			ret = "INTER_AREA"; break;
		case Downsample::INTER_LANCZOS4:
			ret = "INTER_LANCZOS4"; break;
		case Downsample::CP_NEAREST:
			ret = "CP_NEAREST"; break;
		case Downsample::CP_LINEAR:
			ret = "CP_LINEAR"; break;
		case Downsample::CP_CUBIC:
			ret = "CP_CUBIC"; break;
		case Downsample::CP_AREA:
			ret = "CP_AREA"; break;
		case Downsample::CP_LANCZOS:
			ret = "CP_LANCZOS"; break;
		case Downsample::CP_GAUSS:
			ret = "CP_GAUSS"; break;
		case Downsample::CP_GAUSS_FAST:
			ret = "CP_GAUSS_Fast"; break;

		default:
			ret = "NO METHOD"; break;
		}

		return ret;
	}

	template <typename T>
	static void downsampleNN_(const Mat& src, Mat& dest, const int scale)
	{
		if (src.channels() == 1)
		{
			for (int j = 0, n = 0; j < src.rows; j += scale, n++)
			{
				const T* s = src.ptr<T>(j);
				T* d = dest.ptr<T>(n);
				for (int i = 0, m = 0; i < src.cols; i += scale, m++)
				{
					d[m] = s[i];
				}
			}
		}
		else
		{
			for (int j = 0, n = 0; j < src.rows; j += scale, n++)
			{
				const T* s = src.ptr<T>(j);
				T* d = dest.ptr<T>(n);
				for (int i = 0, m = 0; i < 3 * src.cols; i += 3 * scale, m += 3)
				{
					d[m + 0] = s[i + 0];
					d[m + 1] = s[i + 1];
					d[m + 2] = s[i + 2];
				}
			}
		}
	}

	template <>
	static void downsampleNN_<float>(const Mat& src, Mat& dest, const int scale)
	{
		if (src.channels() == 1)
		{
			const int w = src.cols / scale;
			for (int j = 0, n = 0; j < src.rows; j += scale, n++)
			{
				const float* s = src.ptr<float>(j);
				float* d = dest.ptr<float>(n);
				for (int i = 0; i < w; i++)
				{
					*d = *s;
					s += scale;
					d++;
				}
			}
		}
		else
		{
			for (int j = 0, n = 0; j < src.rows; j += scale, n++)
			{
				const float* s = src.ptr<float>(j);
				float* d = dest.ptr<float>(n);
				for (int i = 0, m = 0; i < 3 * src.cols; i += 3 * scale, m += 3)
				{
					d[m + 0] = s[i + 0];
					d[m + 1] = s[i + 1];
					d[m + 2] = s[i + 2];
				}
			}
		}
	}

	static void downsampleNN2_32F(const Mat& src, Mat& dest)
	{
		if (src.channels() == 1)
		{
			if (src.cols >= 32 && src.rows >= 32)
			{
				const int w = src.cols >> 1;
				const int simdw = w / 8;
				for (int j = 0; j < src.rows; j += 2)
				{
#if 0
					const float* s = src.ptr<float>(j);
					float* d = dest.ptr<float>(j >> 1);
					for (int i = 0; i < w; i++)
					{
						*d = *s;
						s += 2;
						d++;
					}
#else
					__m256* d = (__m256*)dest.ptr<float>(j >> 1);
					__m256* ms = (__m256*)src.ptr<float>(j);

					for (int i = 0; i < simdw; i++)
					{
						__m256 a = _mm256_shuffle_ps(*ms, *(ms + 1), _MM_SHUFFLE(2, 0, 2, 0));
						//*d = _mm256_castps256_ps128(_mm256_permute4x64_epi64(a,a,0x10));
						*d = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(a), _MM_SHUFFLE(3, 1, 2, 0)));
						d++;
						ms += 2;
					}
#endif
				}
			}
			else
			{
				const int w = src.cols >> 1;
				for (int j = 0; j < src.rows; j += 2)
				{
					const float* s = src.ptr<float>(j);
					float* d = dest.ptr<float>(j >> 1);
					for (int i = 0; i < w; i++)
					{
						*d = *s;
						s += 2;
						d++;
					}
				}
			}
		}
		else
		{
			for (int j = 0, n = 0; j < src.rows; j += 2, n++)
			{
				const float* s = src.ptr<float>(j);
				float* d = dest.ptr<float>(n);
				for (int i = 0, m = 0; i < 3 * src.cols; i += 3 * 2, m += 3)
				{
					d[m + 0] = s[i + 0];
					d[m + 1] = s[i + 1];
					d[m + 2] = s[i + 2];
				}
			}
		}
	}

	template <typename T>
	static void downsampleNNp1_(const Mat& src, Mat& dest, const int scale)
	{
		if (src.channels() == 1)
		{
			for (int j = 0, n = 0; j < src.rows; j += scale, n++)
			{
				const T* s = src.ptr<T>(j);
				T* d = dest.ptr<T>(n);
				for (int i = 0, m = 0; i < src.cols; i += scale, m++)
				{
					d[m] = s[i];
				}
				d[dest.cols - 1] = s[src.cols - 1];
			}
			const T* s = src.ptr<T>(src.rows - 1);
			T* d = dest.ptr<T>(dest.rows - 1);
			for (int i = 0, m = 0; i < src.cols; i += scale, m++)
			{
				d[m] = s[i];
			}
			d[dest.cols - 1] = s[src.cols - 1];
		}
		else
		{
			for (int j = 0, n = 0; j < src.rows; j += scale, n++)
			{
				const T* s = src.ptr<T>(j);
				T* d = dest.ptr<T>(n);
				for (int i = 0, m = 0; i < 3 * src.cols; i += 3 * scale, m += 3)
				{
					d[m + 0] = s[i + 0];
					d[m + 1] = s[i + 1];
					d[m + 2] = s[i + 2];
				}
				d[3 * (dest.cols - 1) + 0] = s[3 * (src.cols - 1) + 0];
				d[3 * (dest.cols - 1) + 1] = s[3 * (src.cols - 1) + 1];
				d[3 * (dest.cols - 1) + 2] = s[3 * (src.cols - 1) + 2];
			}

			{
				const T* s = src.ptr<T>(src.rows - 1);
				T* d = dest.ptr<T>(dest.rows - 1);
				for (int i = 0, m = 0; i < 3 * src.cols; i += 3 * scale, m += 3)
				{
					d[m + 0] = s[i + 0];
					d[m + 1] = s[i + 1];
					d[m + 2] = s[i + 2];
				}
				d[3 * (dest.cols - 1) + 0] = s[3 * (src.cols - 1) + 0];
				d[3 * (dest.cols - 1) + 1] = s[3 * (src.cols - 1) + 1];
				d[3 * (dest.cols - 1) + 2] = s[3 * (src.cols - 1) + 2];
			}
		}
	}


	static void hatConvolution(const Mat& src, Mat& dst, const int r)
	{
		const float invr = 1.f / r;
		const int d = 2 * r + 1;
		Mat kernel(d, 1, CV_32F);

		float wsum = 0.f;
		for (int i = -r; i <= r; i++)
		{
			const float dist = abs(i * invr);
			const float w = max(1.f - dist, 0.f);

			kernel.at<float>(i + r) = w;
			wsum += w;
		}

		for (int i = 0; i <= 2 * r; i++)
		{
			kernel.at<float>(i) /= wsum;
		}

		sepFilter2D(src, dst, src.depth(), kernel, kernel);
	}

	template <typename T>
	static void downsampleLinear_(const Mat& src, Mat& dest, const int scale, const int r, bool isPlus1 = false)
	{
		Mat conv;
		hatConvolution(src, conv, r);
		if (isPlus1) downsampleNNp1_<T>(conv, dest, scale);
		else downsampleNN_<T>(conv, dest, scale);
	}

	static void cubicConvolution(const Mat& src, Mat& dst, const int r, const double alpha)
	{
		const float invr = 1.0f / r;
		const float a = (float)alpha + 0.01f;
		const int d = 2 * r + 1;
		Mat kernel(d, 1, CV_32F);

		float wsum = 0.f;
		for (int i = -r; i <= r; i++)
		{
			const float dist = abs((float)i * invr);
			const float w = cp::cubic(dist, -a);
			//cout << w << endl;
			wsum += w;
			kernel.at<float>(i + r) = w;
		}
		//cout << "a: " << a << endl;
		//cout << kernel << endl;

		for (int i = 0; i <= 2 * r; i++)
		{
			kernel.at<float>(i) /= wsum;
		}

		sepFilter2D(src, dst, src.depth(), kernel, kernel);
	}

	template <typename T>
	static void downsampleCubic_(const Mat& src, Mat& dest, const int scale, const int r, const double alpha = 1.5, bool isPlus1 = false)
	{
		Mat conv;
		cubicConvolution(src, conv, r, alpha);
		if (isPlus1) downsampleNNp1_<T>(conv, dest, scale);
		else downsampleNN_<T>(conv, dest, scale);
	}

	template <typename T>
	static void downsampleArea_(const Mat& src_, Mat& dest, const int scale, const int r, bool isPlus1 = false)
	{
		Mat conv;
		const int d = 2 * r + 1;
		blur(src_, conv, Size(d, d));
		if (isPlus1) downsampleNNp1_<T>(conv, dest, scale);
		else downsampleNN_<T>(conv, dest, scale);
	}

	static void LanczosConvolution(const Mat& src, Mat& dst, const int r, const int order)
	{
		const float invr = float(order) / r;
		const int d = 2 * r + 1;
		Mat kernel(d, 1, CV_32F);

		float wsum = 0.f;
		for (int i = -r; i <= r; i++)
		{
			const float dist = abs(i * invr);
			float w = cp::lanczos(dist, (float)order);
			kernel.at<float>(i + r) = w;
			wsum += w;
		}

		for (int i = 0; i <= 2 * r; i++)
		{
			kernel.at<float>(i) /= wsum;
		}

		sepFilter2D(src, dst, src.depth(), kernel, kernel);
	}

	template <typename T>
	static void downsampleLanczos_(const Mat& src, Mat& dest, const int scale, const int r, const int order = 4, bool isPlus1 = false)
	{
		Mat conv;
		LanczosConvolution(src, conv, r, order);
		if (isPlus1) downsampleNNp1_<T>(conv, dest, scale);
		else downsampleNN_<T>(conv, dest, scale);
	}


	template <typename T>
	static void downsampleGauss_(const Mat& src, Mat& dest, const int scale, const int r, const double sigma_clip = 3.0, bool isPlus1 = false)
	{
		Mat conv, src32f;
		const int d = 2 * r + 1;

		src.convertTo(src32f, CV_32F);
		GaussianBlur(src32f, src32f, Size(d, d), r / sigma_clip);
		src32f.convertTo(conv, CV_8U);

		if (isPlus1) downsampleNNp1_<T>(conv, dest, scale);
		else downsampleNN_<T>(conv, dest, scale);
	}

	template <typename T>
	static void downsampleGaussFast_(const Mat& src, Mat& dest, const int scale, const int r, const double sigma_clip = 3.0, bool isPlus1 = false)
	{
		Mat conv;
		const int d = 2 * r + 1;
		GaussianBlur(src, conv, Size(d, d), r / sigma_clip);

		if (isPlus1) downsampleNNp1_<T>(conv, dest, scale);
		else downsampleNN_<T>(conv, dest, scale);
	}

	void downsample(cv::InputArray src_, cv::OutputArray dest_, const int scale, const Downsample downsample_method, const double parameter, double radius_ratio)
	{
		int r = scale >> 1;
		r = int(radius_ratio * r);

		if ((int)downsample_method <= INTER_LANCZOS4)
		{
			resize(src_, dest_, Size(), 1.0 / scale, 1.0 / scale, (int)downsample_method);
		}
		else
		{
			dest_.create(src_.size() / scale, src_.type());
			Mat src = src_.getMat();
			Mat dest = dest_.getMat();
			if (src.depth() == CV_8U)
			{
				switch (downsample_method)
				{
				case Downsample::CP_NEAREST:
					downsampleNN_<uchar>(src, dest, scale); break;
				case Downsample::CP_LINEAR:
					downsampleLinear_<uchar>(src, dest, scale, r); break;
				case Downsample::CP_CUBIC:
					if (parameter == 0)downsampleCubic_<uchar>(src, dest, scale, r);
					else downsampleCubic_<uchar>(src, dest, scale, r, parameter);
					break;
				case Downsample::CP_AREA:
					downsampleArea_<uchar>(src, dest, scale, r); break;
				case Downsample::CP_LANCZOS:
					if (parameter == 0)downsampleLanczos_<uchar>(src, dest, scale, (4 + 1) * r);
					else downsampleLanczos_<uchar>(src, dest, scale, int(parameter + 1) * r, (int)parameter);
					break;
				case Downsample::CP_GAUSS:
					if (parameter == 0)downsampleGauss_<uchar>(src, dest, scale, r);
					else downsampleGauss_<uchar>(src, dest, scale, r, parameter);
					break;
				case Downsample::CP_GAUSS_FAST:
					if (parameter == 0)downsampleGaussFast_<uchar>(src, dest, scale, r);
					else downsampleGaussFast_<uchar>(src, dest, scale, r, parameter);
					break;
				default:
					cout << "no method in downsample" << endl;
					break;
				}
			}
			else if (src.depth() == CV_32F)
			{
				switch (downsample_method)
				{
				case Downsample::CP_NEAREST:
					if (scale == 2 && src.cols % 2 == 0 && src.rows % 2 == 0) downsampleNN2_32F(src, dest);
					else downsampleNN_<float>(src, dest, scale); break;
				case Downsample::CP_LINEAR:
					downsampleLinear_<float>(src, dest, scale, r); break;
				case Downsample::CP_CUBIC:
					if (parameter == 0)downsampleCubic_<float>(src, dest, scale, r);
					else downsampleCubic_<float>(src, dest, scale, r, parameter);
					break;
				case Downsample::CP_AREA:
					downsampleArea_<float>(src, dest, scale, r); break;
				case Downsample::CP_LANCZOS:
					if (parameter == 0)downsampleLanczos_<float>(src, dest, scale, (4 + 1) * r);
					else downsampleLanczos_<float>(src, dest, scale, int(parameter + 1) * r, (int)parameter);
					break;
				case Downsample::CP_GAUSS:
					if (parameter == 0)downsampleGauss_<float>(src, dest, scale, r);
					else downsampleGauss_<float>(src, dest, scale, r, parameter);
					break;
				case Downsample::CP_GAUSS_FAST:
					if (parameter == 0)downsampleGaussFast_<float>(src, dest, scale, r);
					else downsampleGaussFast_<float>(src, dest, scale, r, parameter);
					break;
				default:
					cout << "no method in downsample" << endl;
					break;
				}
			}
			else
			{
				cout << "do not support this type in cp::downsample" << endl;
			}
		}
	}

	void downsamplePlus1(cv::InputArray src_, cv::OutputArray dest_, const int scale, const Downsample downsample_method, const double parameter, double radius_ratio)
	{
		int r = scale >> 1;
		r = int(radius_ratio * r);

		const int w = src_.size().width / scale + 1;
		const int h = src_.size().height / scale + 1;

		if ((int)downsample_method <= INTER_LANCZOS4)
		{
			resize(src_, dest_, Size(w, h), 0, 0, (int)downsample_method);
		}
		else
		{
			dest_.create(Size(w, h), src_.type());
			Mat src = src_.getMat();
			Mat dest = dest_.getMat();
			if (src.depth() == CV_8U)
			{
				switch (downsample_method)
				{
				case Downsample::CP_NEAREST:
					downsampleNNp1_<uchar>(src, dest, scale); break;
				case Downsample::CP_LINEAR:
					downsampleLinear_<uchar>(src, dest, scale, r, true); break;
				case Downsample::CP_CUBIC:
					if (parameter == 0)downsampleCubic_<uchar>(src, dest, scale, r);
					else downsampleCubic_<uchar>(src, dest, scale, r, parameter);
					break;
				case Downsample::CP_AREA:
					downsampleArea_<uchar>(src, dest, scale, r, true); break;
				case Downsample::CP_LANCZOS:
					if (parameter == 0)downsampleLanczos_<uchar>(src, dest, scale, (4 + 1) * r, 4, true);
					else downsampleLanczos_<uchar>(src, dest, scale, int(parameter + 1) * r, (int)parameter, true);
					break;
				case Downsample::CP_GAUSS:
					if (parameter == 0)downsampleGauss_<uchar>(src, dest, scale, r, 3.0, true);
					else downsampleGauss_<uchar>(src, dest, scale, r, parameter, true);
					break;
				case Downsample::CP_GAUSS_FAST:
					if (parameter == 0)downsampleGaussFast_<uchar>(src, dest, scale, r, 3.0, true);
					else downsampleGaussFast_<uchar>(src, dest, scale, r, parameter, true);
					break;
				default:
					cout << "no method in downsample" << endl;
					break;
				}
			}
			else if (src.depth() == CV_32F)
			{
				switch (downsample_method)
				{
				case Downsample::CP_NEAREST:
					//if (scale == 2 && src.cols % 2 == 0 && src.rows && 2)downsampleNN2_32F(src, dest);
					//else 
					downsampleNNp1_<float>(src, dest, scale); break;
				case Downsample::CP_LINEAR:
					downsampleLinear_<float>(src, dest, scale, r, true); break;
				case Downsample::CP_CUBIC:
					if (parameter == 0)downsampleCubic_<float>(src, dest, scale, r, 1.5, true);
					else downsampleCubic_<float>(src, dest, scale, r, parameter, true);
					break;
				case Downsample::CP_AREA:
					downsampleArea_<float>(src, dest, scale, r, true); break;
				case Downsample::CP_LANCZOS:
					if (parameter == 0)downsampleLanczos_<float>(src, dest, scale, (4 + 1) * r, true);
					else downsampleLanczos_<float>(src, dest, scale, int(parameter + 1) * r, (int)parameter, true);
					break;
				case Downsample::CP_GAUSS:
					if (parameter == 0)downsampleGauss_<float>(src, dest, scale, r, 3.0, true);
					else downsampleGauss_<float>(src, dest, scale, r, parameter, true);
					break;
				case Downsample::CP_GAUSS_FAST:
					if (parameter == 0)downsampleGaussFast_<float>(src, dest, scale, r, 3.0, true);
					else downsampleGaussFast_<float>(src, dest, scale, r, parameter, true);
					break;
				default:
					cout << "no method in downsample" << endl;
					break;
				}
			}
			else
			{
				cout << "do not support this type in cp::downsample" << endl;
			}
		}
	}
}