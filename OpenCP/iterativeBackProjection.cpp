#include "iterativeBackProjection.hpp"
#include "jointBilateralFilter.hpp"
#include "guidedFilter.hpp"
#include "timer.hpp"

#include "inlineSIMDFunctions.hpp"
#include "consoleImage.hpp"
#include "metrics.hpp"
#include "blend.hpp"

using namespace cv;
using namespace std;

namespace cp
{
	void normL1(Mat& src, Mat& dest, float lambda)
	{
		const int v = src.cols;

#pragma omp parallel for
		for (int j = 1; j < src.rows - 1; j++)
		{
			float* s = src.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 1; i < src.cols - 1; i++)
			{
				//float diff = abs(s[i - 1] - s[i]) + abs(s[i - v] - s[i]);
				float diff = sqrt((s[i - 1] - s[i]) * (s[i - 1] - s[i]) + (s[i - v] - s[i]) * (s[i - v] - s[i]));

				/*float diff = abs(s[i - 1] - s[i]);
					+ abs(s[i + 1] - s[i])
					+ abs(s[i + v] - s[i])
					+ abs(s[i - v] - s[i]);
*/
				d[i] *= 1.f / (1.f + lambda * diff);
			}
		}
	}

	void LucyRichardsonGaussTikhonov(const Mat& src, Mat& dest, const Size ksize, const float sigma, const float beta, const int iteration)
	{
		Mat srcf;
		Mat destf;
		Mat ratio;
		src.convertTo(srcf, CV_32FC3);
		src.convertTo(destf, CV_32FC3);
		Mat bdest;

		int i;
		Mat denominator = Mat::ones(srcf.size(), CV_32F);
		for (i = 0; i < iteration; i++)
		{
			GaussianBlur(destf, bdest, ksize, sigma);
			divide(srcf, bdest, ratio);
			GaussianBlur(ratio, ratio, ksize, sigma);

			//normL1(destf, denominator, beta);
			//multiply(ratio, denominator, ratio);
			multiply(ratio, ratio, ratio);
			//Laplacian(destf, denominator, destf.depth(), 1, 2.f*beta, -1.f);
			//divide(ratio, -denominator, ratio);

			multiply(destf, ratio, destf);
		}
		destf.convertTo(dest, src.type());
	}

	void LucyRichardsonGauss(const Mat& src, Mat& dest, const Size ksize, const float sigma, const int iteration)
	{
		Mat srcf;
		Mat destf;
		Mat ratio;
		src.convertTo(srcf, CV_32FC3);

		Mat bdest;
		srcf.copyTo(destf, CV_32FC3);
		for (int i = 0; i < iteration; i++)
		{
			GaussianBlur(destf, bdest, ksize, sigma);
			divide(srcf, bdest, ratio);
			GaussianBlur(ratio, ratio, ksize, sigma);

			multiply(destf, ratio, destf);
		}
		destf.convertTo(dest, CV_8UC3);
	}


	void fma(const Mat& src, const Mat& lambda, Mat& dest)
	{
		const int size = get_simd_floor(src.size().area(), 32);

		const float* s = src.ptr<float>();
		float* d = dest.ptr<float>();
		const float* l = lambda.ptr<float>();
		for (int i = 0; i < size; i += 32)
		{
			_mm256_store_ps(d + i + 0, _mm256_fmadd_ps(_mm256_load_ps(l + i + 0), _mm256_load_ps(s + i + 0), _mm256_load_ps(d + i + 0)));
			_mm256_store_ps(d + i + 8, _mm256_fmadd_ps(_mm256_load_ps(l + i + 8), _mm256_load_ps(s + i + 8), _mm256_load_ps(d + i + 8)));
			_mm256_store_ps(d + i + 16, _mm256_fmadd_ps(_mm256_load_ps(l + i + 16), _mm256_load_ps(s + i + 16), _mm256_load_ps(d + i + 16)));
			_mm256_store_ps(d + i + 24, _mm256_fmadd_ps(_mm256_load_ps(l + i + 24), _mm256_load_ps(s + i + 24), _mm256_load_ps(d + i + 24)));
		}
		for (int i = size; i < src.size().area(); i++)
		{
			d[i] += l[i] * s[i];
		}
	}

	void fma(const Mat& src, const float a, Mat& dest)
	{
		const int size = get_simd_floor(src.size().area(), 32);

		const float* s = src.ptr<float>();
		float* d = dest.ptr<float>();
		const __m256 ma = _mm256_set1_ps(a);
		for (int i = 0; i < size; i += 32)
		{
			_mm256_store_ps(d + i + 0, _mm256_fmadd_ps(ma, _mm256_load_ps(s + i + 0), _mm256_load_ps(d + i + 0)));
			_mm256_store_ps(d + i + 8, _mm256_fmadd_ps(ma, _mm256_load_ps(s + i + 8), _mm256_load_ps(d + i + 8)));
			_mm256_store_ps(d + i + 16, _mm256_fmadd_ps(ma, _mm256_load_ps(s + i + 16), _mm256_load_ps(d + i + 16)));
			_mm256_store_ps(d + i + 24, _mm256_fmadd_ps(ma, _mm256_load_ps(s + i + 24), _mm256_load_ps(d + i + 24)));
		}
		for (int i = size; i < src.size().area(); i++)
		{
			d[i] += a * s[i];
		}
	}

	void eth(Mat& error, Mat& ref, float th)
	{
		const int v = error.cols;
		float sigma = th;
		const float sigmap = -0.5f / (sigma * sigma);
#pragma omp parallel for
		for (int j = 1; j < error.rows - 1; j++)
		{
			float* r = ref.ptr<float>(j);
			float* e = error.ptr<float>(j);
			for (int i = 1; i < error.cols - 1; i++)
			{
				float diff =
					abs(r[i - 1] - r[i])
					+ abs(r[i + 1] - r[i])
					+ abs(r[i + v] - r[i])
					+ abs(r[i - v] - r[i]);


				e[i] *= (1.f - exp(diff * sigmap));
				//if (diff < th) error.at<float>(j, i) = 0.f;
			}
		}
	}

	void iterativeBackProjectionDeblurGaussianTV(const Mat& src, Mat& dest, const Size ksize, const float sigma, const float backprojection_sigma, const float lambda, const float th, const int iteration)
	{
		Mat srcf;
		Mat destf;
		Mat subf;
		src.convertTo(srcf, CV_32FC3);
		src.convertTo(destf, CV_32FC3);
		Mat bdest;

		//float lambdaamp = 0.99;
		//float l = lambda;
		for (int i = 0; i < iteration; i++)
		{
			GaussianBlur(destf, bdest, ksize, sigma);
			subtract(srcf, bdest, subf);
			//double e = norm(subf);

			GaussianBlur(subf, subf, ksize, backprojection_sigma);

			eth(subf, destf, th);
			//destf += lambda*subf;
			fma(subf, lambda, destf);

			//l *= lambdaamp;
		}
		destf.convertTo(dest, CV_8UC3);
	}

	void gradient(const Mat& src, Mat& dest)
	{
		int v = src.cols;
#pragma omp parallel for
		for (int j = 1; j < src.rows; j++)
		{
			float* s = (float*)src.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 1; i < src.cols; i += 8)
			{
				__m256 ms = _mm256_load_ps(s + i);
				__m256 px = _mm256_loadu_ps(s - 1 + i);
				__m256 py = _mm256_loadu_ps(s - v + i);

				__m256 a = _mm256_sub_ps(ms, px);
				px = _mm256_mul_ps(a, a);
				a = _mm256_sub_ps(ms, py);
				py = _mm256_mul_ps(a, a);
				a = _mm256_sqrt_ps(_mm256_add_ps(px, py));
				_mm256_storeu_ps(d + i, a);

				//float dx = s[i] - s[i - 1];
				//float dy = s[i] - s[i - v];
				//float grad = sqrt(dx*dx + dy*dy);
				//d[i] = grad;
			}
		}
	}

	void getstep(Mat& pgrad, Mat& cgrad, float lambda, Mat& dest)
	{
		int v = pgrad.cols;
#pragma omp parallel for
		for (int j = 0; j < pgrad.rows; j++)
		{
			float* s0 = pgrad.ptr<float>(j);
			float* s1 = cgrad.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 0; i < pgrad.cols; i++)
			{
				//d[i] = lambda +s0[i] / s1[i];
				//d[i] = lambda + min(2.f, 1.f*abs(s0[i]-s1[i]));
				d[i] = min(3.f, lambda + 1.f * abs(s0[i] - s1[i]));
			}
		}
	}

	void iterativeBackProjectionDeblurGaussianFast(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const float backprojection_sigma, const float lambda, const int iteration)
	{
		Mat srcf;
		Mat destf;
		Mat subf;
		src.convertTo(srcf, CV_32FC3);
		src.convertTo(destf, CV_32FC3);
		Mat bdest;

		//float lambdaamp = 0.99;
		//float l = lambda;
		Mat pgrad = Mat::ones(srcf.size(), CV_32F);
		Mat cgrad = Mat::ones(srcf.size(), CV_32F);
		Mat step = Mat::ones(srcf.size(), CV_32F);
		//gradient(srcf, pgrad);

		for (int i = 0; i < iteration; i++)
		{
			GaussianBlur(destf, bdest, ksize, sigma);
			subtract(srcf, bdest, subf);
			//double e = norm(subf);

			if (backprojection_sigma > 0.f)
			{
				GaussianBlur(subf, subf, ksize, backprojection_sigma);
			}

			getstep(pgrad, cgrad, lambda, step);

			//destf += lambda*subf;
			//destf += step.mul(subf);
			fma(subf, step, destf);

			cgrad.copyTo(pgrad);
			gradient(destf, cgrad);
			//Laplacian(destf, cgrad, destf.depth(), 1, 0.5);
		}
		destf.convertTo(dest, CV_8UC3);

		imshow("step", step * 0.333);
	}

	void iterativeBackProjectionDeblurGaussian(const Mat& src, Mat& dest, const Size ksize, const float sigma, const float backprojection_sigma, const float lambda, const int iteration, Mat& init)
	{
		Mat srcf;
		Mat destf;
		Mat subf;
		src.convertTo(srcf, CV_32FC3);

		if (init.empty()) src.convertTo(destf, CV_32FC3);
		else init.convertTo(destf, CV_32F);
		Mat bdest;

		//float lambdaamp = 0.99;
		//float l = lambda;

		for (int i = 0; i < iteration; i++)
		{
			GaussianBlur(destf, bdest, ksize, sigma);
			subtract(srcf, bdest, subf);
			//double e = norm(subf);

			if (backprojection_sigma > 0.f)
				GaussianBlur(subf, subf, ksize, backprojection_sigma);


			//destf += lambda*subf;
			/*if(i==0)fma(subf, 3, destf);
			else if (i==1)fma(subf, 2, destf);
			else	fma(subf, lambda, destf);*/
			fma(subf, lambda, destf);

			//l *= lambdaamp;
		}
		destf.convertTo(dest, src.depth());
	}

	void iterativeBackProjectionDeblurGaussian(const Mat& src, Mat& dest, const Size ksize, const float sigma, const float backprojection_sigma, const float lambda, const int iteration)
	{
		Mat temp;
		iterativeBackProjectionDeblurGaussian(src, dest, ksize, sigma, backprojection_sigma, lambda, iteration, temp);
	}


	void iterativeBackProjectionDeblurBoxGaussian(const Mat& src, Mat& dest, const int d, const Size ksize, const float backprojection_sigma, const float lambda, const int iteration, Mat& init)
	{
		Mat srcf;
		Mat destf;
		Mat subf;
		src.convertTo(srcf, CV_32F);

		if (init.empty()) src.convertTo(destf, CV_32F);
		else init.convertTo(destf, CV_32F);
		Mat bdest;

		//float lambdaamp = 0.99;
		//float l = lambda;

		Mat kernel = Mat::ones(d, d, CV_32F);
		kernel /= (d * d);
		if (d % 2 == 0)copyMakeBorder(kernel, kernel, 1, 0, 1, 0, BORDER_CONSTANT, Scalar::all(0));
		//if (d % 2 == 0)copyMakeBorder(kernel, kernel, 0, 1, 0, 1, BORDER_CONSTANT, Scalar::all(0));
		for (int i = 0; i < iteration; i++)
		{
			filter2D(destf, bdest, -1, kernel);
			//blur(destf, bdest, ksize);
			subtract(srcf, bdest, subf);
			//double e = norm(subf);

			if (backprojection_sigma > 0.f)
				GaussianBlur(subf, subf, ksize, backprojection_sigma);


			//destf += lambda*subf;
			/*if(i==0)fma(subf, 3, destf);
			else if (i==1)fma(subf, 2, destf);
			else	fma(subf, lambda, destf);*/
			fma(subf, lambda, destf);

			//l *= lambdaamp;
		}
		destf.convertTo(dest, src.depth());
	}

	void iterativeBackProjectionDeblurBoxGaussian(const Mat& src, Mat& dest, const int d, const Size ksize, const float backprojection_sigma, const float lambda, const int iteration)
	{
		Mat temp;
		iterativeBackProjectionDeblurBoxGaussian(src, dest, d, ksize, backprojection_sigma, lambda, iteration, temp);
	}


	void iterativeBackProjectionDeblurPascal3x3Gaussian(const Mat& src, Mat& dest, const Size ksize, const float backprojection_sigma, const float lambda, const int iteration, Mat& init)
	{
		Mat srcf;
		Mat destf;
		Mat subf;
		src.convertTo(srcf, CV_32FC3);

		if (init.empty()) src.convertTo(destf, CV_32FC3);
		else init.convertTo(destf, CV_32F);
		Mat bdest;

		//float lambdaamp = 0.99;
		//float l = lambda;
		Mat kernel = Mat::ones(3, 3, CV_32F);
		kernel.at<float>(0, 0) = 1.f;
		kernel.at<float>(0, 1) = 2.f;
		kernel.at<float>(0, 2) = 1.f;
		kernel.at<float>(1, 0) = 2.f;
		kernel.at<float>(1, 1) = 4.f;
		kernel.at<float>(1, 2) = 2.f;
		kernel.at<float>(2, 0) = 1.f;
		kernel.at<float>(2, 1) = 2.f;
		kernel.at<float>(2, 2) = 1.f;
		kernel /= 16.f;
		
		//if (d % 2 == 0)copyMakeBorder(kernel, kernel, 0, 1, 0, 1, BORDER_CONSTANT, Scalar::all(0));
		for (int i = 0; i < iteration; i++)
		{
			filter2D(destf, bdest, -1, kernel);
			subtract(srcf, bdest, subf);
			//double e = norm(subf);

			if (backprojection_sigma > 0.f)
				GaussianBlur(subf, subf, ksize, backprojection_sigma);


			//destf += lambda*subf;
			/*if(i==0)fma(subf, 3, destf);
			else if (i==1)fma(subf, 2, destf);
			else	fma(subf, lambda, destf);*/
			fma(subf, lambda, destf);

			//l *= lambdaamp;
		}
		destf.convertTo(dest, src.depth());
	}

	void iterativeBackProjectionDeblurPascal3x3Gaussian(const Mat& src, Mat& dest, const Size ksize, const float backprojection_sigma, const float lambda, const int iteration)
	{
		Mat temp;
		iterativeBackProjectionDeblurPascal3x3Gaussian(src, dest, ksize, backprojection_sigma, lambda, iteration, temp);
	}


	void iterativeBackProjectionDeblurBilateral(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const float backprojection_sigma_space, const float backprojection_sigma_color, const float lambda, const int iteration, cv::Mat& init)
	{
		Mat srcf;
		Mat bdest;
		Mat destf;
		Mat subf;

		src.convertTo(srcf, CV_32FC3);
		if (init.empty()) src.convertTo(destf, CV_32FC3);
		else init.convertTo(destf, CV_32FC3);

		for (int i = 0; i < iteration; i++)
		{
			GaussianBlur(destf, bdest, ksize, sigma);
			subtract(srcf, bdest, subf);

			jointBilateralFilter(subf, destf, subf, ksize, backprojection_sigma_color, backprojection_sigma_space);

			//fma(subf, lambda, destf);
			destf += ((float)lambda) * subf;

		}
		destf.convertTo(dest, src.type());
	}

	void iterativeBackProjectionDeblurBilateral(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const float backprojection_sigma_space, const float backprojection_sigma_color, const float lambda, const int iteration)
	{
		Mat temp;
		iterativeBackProjectionDeblurBilateral(src, dest, ksize, sigma, backprojection_sigma_space, backprojection_sigma_color, lambda, iteration, temp);
	}

	/*void iterativeBackProjectionDeblurBilateral(const Mat& src, Mat& dest, const Size ksize, const double sigma_space, const double backprojection_sigma_space, const double backprojection_sigma_color, const double lambda, const int iteration, Mat& init)
	{
		Mat srcf;
		Mat destf;
		Mat subf;

		src.convertTo(srcf, CV_32FC3);

		if (init.empty())
			init.convertTo(destf, CV_32FC3);
		else
			src.convertTo(destf, CV_32FC3);
		Mat bdest;

		double maxe = DBL_MAX;

		int i;
		for (i = 0; i < iteration; i++)
		{
			GaussianBlur(destf, bdest, ksize, sigma_space);

			subtract(srcf, bdest, subf);

			//normarize from 0 to 255 for joint birateral filter (range 0 to 255)
			double minv, maxv;
			minMaxLoc(subf, &minv, &maxv);
			subtract(subf, Scalar(minv, minv, minv), subf);
			multiply(subf, Scalar(2, 2, 2), subf);

			jointBilateralFilter(subf, destf, subf, ksize, backprojection_sigma_color, backprojection_sigma_space);

			multiply(subf, Scalar(0.5, 0.5, 0.5), subf);
			add(subf, Scalar(minv, minv, minv), subf);

			//double e = norm(subf);

			//imshow("a",subf);
			multiply(subf, Scalar(lambda, lambda, lambda), subf);

			add(destf, subf, destf);
			//destf += ((float)lambda)*subf;

			//printf("%f\n",e);

		}
		destf.convertTo(dest, CV_8UC3);
	}
	*/

	void iterativeBackProjectionDeblurGuidedImageFilter(const Mat& src, Mat& dest, const Size ksize, const double eps, const double sigma_space, const double lambda, const int iteration)
	{
		Mat srcf;
		Mat destf;
		Mat subf;
		src.convertTo(srcf, CV_32FC3);
		src.convertTo(destf, CV_32FC3);
		Mat bdest;

		double maxe = DBL_MAX;

		int i;
		for (i = 0; i < iteration; i++)
		{
			GaussianBlur(destf, bdest, ksize, sigma_space);

			subtract(srcf, bdest, subf);

			//normarize from 0 to 255 for joint birateral filter (range 0 to 255)
			double minv, maxv;
			minMaxLoc(subf, &minv, &maxv);
			subtract(subf, Scalar(minv, minv, minv), subf);
			multiply(subf, Scalar(2, 2, 2), subf);

			cp::guidedImageFilter(subf, destf, subf, ksize.width / 4, (float)eps);

			multiply(subf, Scalar(0.5, 0.5, 0.5), subf);
			add(subf, Scalar(minv, minv, minv), subf);

			//double e = norm(subf);

			//imshow("a",subf);
			multiply(subf, Scalar(lambda, lambda, lambda), subf);

			add(destf, subf, destf);
			//destf += ((float)lambda)*subf;

			//printf("%f\n",e);
			/*	if(i!=0)
			{
			if(maxe>e)
			{
			maxe=e;
			}
			else break;
			}*/
			//if(isWrite)
			//{
			//	destf.convertTo(dest,CV_8UC3);
			//	imwrite(format("B%03d.png",i),dest);
			//}
		}
		destf.convertTo(dest, CV_8UC3);
	}


	void guiIBP(Mat& src, Mat& ref, string wname)
	{
		namedWindow(wname);
		cp::ConsoleImage ci;
		int d = 2; createTrackbar("d", wname, &d, 10);
		int r = 2; createTrackbar("r", wname, &r, 10);
		int sigma = 10; createTrackbar("sigma10", wname, &sigma, 100);
		int back_sigma = 10; createTrackbar("back_sigma10", wname, &back_sigma, 100);
		int lambda = 100; createTrackbar("lambda100", wname, &lambda, 100);
		int iteration = 10; createTrackbar("iteration", wname, &iteration, 100);
		int key = 0;
		Mat dest;
		while (key != 'q')
		{
			Size ksize(2 * r + 1, 2 * r + 1);
			//iterativeBackProjectionDeblurBoxGaussian(src, dest, d, ksize, back_sigma * 0.1f, lambda * 0.01f, iteration);
			//iterativeBackProjectionDeblurGaussian(src, dest, ksize, sigma * 0.1f, back_sigma * 0.1f, lambda * 0.01f, iteration);
			iterativeBackProjectionDeblurPascal3x3Gaussian(src, dest, ksize, back_sigma * 0.1f, lambda * 0.01f, iteration);
			ci("src %f dB", getPSNR(src, ref));
			ci("IBP %f dB", getPSNR(dest, ref));
			ci.show();
			imshow(wname, dest);
			key = waitKey(1);
			if (key == 'd')guiAlphaBlend(dest, ref);
			if (key == 'c')guiAlphaBlend(src, ref);
		}
		destroyWindow(wname);
		destroyWindow("console");
	}
}