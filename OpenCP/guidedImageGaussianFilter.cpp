#include "guidedFilter.hpp"
#include "inlineSIMDFunctions.hpp"
const int BORDER_TYPE = cv::BORDER_REPLICATE;

using namespace cv;
namespace cp
{
	void guidedImageGaussianFilterGray(const Mat& src, Mat& dest, const int radius, const float sigma, const float eps)
	{
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const float e = eps;
		Mat sf; src.convertTo(sf, CV_32F);

		Mat mSrc(imsize, CV_32F);//mean_p
		GaussianBlur(sf, mSrc, ksize, sigma, sigma, BORDER_TYPE);//meanImSrc*K

		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		multiply(sf, sf, x1);//sf*sf
		GaussianBlur(x1, x3, ksize, sigma, sigma, BORDER_TYPE);//corrI:m*sf*sf

		multiply(mSrc, mSrc, x1);//;msf*msf
		x3 -= x1;//x3: m*sf*sf-msf*msf;
		x1 = x3 + e;
		divide(x3, x1, x3);
		multiply(x3, mSrc, x1);
		x1 -= mSrc;
		GaussianBlur(x3, x2, ksize, sigma, sigma, BORDER_TYPE);//x2*k
		GaussianBlur(x1, x3, ksize, sigma, sigma, BORDER_TYPE);//x3*k
		multiply(x2, sf, x1);//x1*K
		x2 = x1 - x3;//
		x2.convertTo(dest, src.type());
	}

	void guidedImageGaussianFilterGrayEnhance(const Mat& src, Mat& dest, const int radius, const float sigma, const float eps, const float m)
	{
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const float e = eps;
		Mat sf; src.convertTo(sf, CV_32F);

		Mat mSrc(imsize, CV_32F);//mean_p
		GaussianBlur(sf, mSrc, ksize, sigma, sigma, BORDER_TYPE);//meanImSrc*K

		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		multiply(sf, sf, x1);//sf*sf
		GaussianBlur(x1, x3, ksize, sigma, sigma, BORDER_TYPE);//corrI:m*sf*sf

		multiply(mSrc, mSrc, x1);//;msf*msf
		x3 -= x1;//x3: m*sf*sf-msf*msf;
		x1 = x3 + e;
		divide(x3, x1, x3);
		multiply(x3, mSrc, x1);
		x1 -= mSrc;

		x3 = (m * x3 + 1.f - m);
		GaussianBlur(x3, x2, ksize, sigma, sigma, BORDER_TYPE);//x2*k
		x1 = x1 * m;
		GaussianBlur(x1, x3, ksize, sigma, sigma, BORDER_TYPE);//x3*k
		multiply(x2, sf, x1);//x1*K
		x2 = x1 - x3;//
		x2.convertTo(dest, src.type());
	}

	static void gradientSquareRootPrewitt32F(InputArray ref, OutputArray dst)
	{
		dst.create(ref.size(), ref.type());

		Mat border;
		copyMakeBorder(ref, border, 1, 1, 1, 1, BORDER_DEFAULT);
		Mat s = ref.getMat();
		Mat d = dst.getMat();
		const int step = s.size().width;
		const int simdwidth = get_simd_floor(step, 8);
		const float normal = 1.f / 9.f;
		__m256 mnormal = _mm256_set1_ps(normal);
		for (int j = 0; j < ref.size().height; j++)
		{
			float* s = border.ptr<float>(j);
			float* dst = d.ptr<float>(j);
			for (int i = 0; i < simdwidth; i += 8)
			{
				__m256 mx = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(s + i + step)), _mm256_loadu_ps(s + i + 2 * step));
				mx = _mm256_sub_ps(mx, _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i + 2), _mm256_loadu_ps(s + i + 2 + step)), _mm256_loadu_ps(s + i + 2 + 2 * step)));
				__m256 my = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(s + i + 1)), _mm256_loadu_ps(s + i + 2));
				my = _mm256_sub_ps(my, _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i + 2 * step), _mm256_loadu_ps(s + i + 1 + 2 * step)), _mm256_loadu_ps(s + i + 2 + 2 * step)));
				_mm256_storeu_ps(dst + i, _mm256_sqrt_ps(_mm256_mul_ps(mnormal, _mm256_fmadd_ps(mx, mx, _mm256_mul_ps(my, my)))));
			}
			for (int i = simdwidth; i < step; i++)
			{
				float x = (s[i] + s[i + step] + s[i + 2 * step] - s[i + 2] - s[i + 2 + step] - s[i + 2 + 2 * step]);
				float y = (s[i] + s[i + 1] + s[i + 2] - s[i + 2 * step] - s[i + 1 + 2 * step] - s[i + 2 + 2 * step]);
				dst[i] = sqrt((x * x + y * y) * normal);
			}
		}
	}

	//Local Edge - Preserving Multiscale Decomposition for High Dynamic Range Image Tone Mapping TIP2013
	void localEdgePreservingFilter(const Mat& src, Mat& dest, const int radius, const float eps)
	{
		Size ksize = Size(2 * radius + 1, 2 * radius + 1);
		Mat sb; blur(src, sb, ksize);
		Mat sb2; blur(src.mul(src), sb2, ksize);
		Mat var = sb2 - sb.mul(sb);

		Mat grad; gradientSquareRootPrewitt32F(src, grad);
		blur(grad, grad, ksize);

		Mat a = var / (var + grad + eps);
		Mat b = sb - a.mul(sb);

		blur(a, a, ksize);
		blur(b, b, ksize);
		dest = a.mul(src) + b;
	}

	void localEdgePreservingGaussianFilter(const Mat& src, Mat& dest, const int radius, const float sigma, const float eps, float alpha, float beta)
	{
		Mat srcf; src.convertTo(srcf, CV_32F);
		Size ksize = Size(2 * radius + 1, 2 * radius + 1);

		Mat sb; GaussianBlur(srcf, sb, ksize, sigma);
		Mat sb2; GaussianBlur(srcf.mul(srcf), sb2, ksize, sigma);
		Mat var = sb2 - sb.mul(sb);

		Mat grad; gradientSquareRootPrewitt32F(srcf, grad);
		if (beta != 1.f) pow(grad, 2.0 - beta, grad);
		GaussianBlur(grad, grad, ksize, sigma);

		Mat a = var / (var + alpha * grad);
		Mat b = sb - a.mul(sb);

		GaussianBlur(a, a, ksize, sigma);
		GaussianBlur(b, b, ksize, sigma);
		dest = a.mul(srcf) + b;
		dest.convertTo(dest, src.depth());
	}
}