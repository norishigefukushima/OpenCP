#include "iterativeBackProjection.hpp"
#include "jointBilateralFilter.hpp"
#include "guidedFilter.hpp"
#include "timer.hpp"
#include "fftw3.h"
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
				float diff = sqrt((s[i - 1] - s[i])*(s[i - 1] - s[i]) + (s[i - v] - s[i])*(s[i - v] - s[i]));

				/*float diff = abs(s[i - 1] - s[i]);
					+ abs(s[i + 1] - s[i])
					+ abs(s[i + v] - s[i])
					+ abs(s[i - v] - s[i]);
*/
				d[i] *= 1.f/(1.f +lambda*diff);
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
		Mat denominator = Mat::ones(srcf.size(),CV_32F);
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
		src.convertTo(destf, CV_32FC3);
		Mat bdest;

		for (int i = 0; i < iteration; i++)
		{
			GaussianBlur(destf, bdest, ksize, sigma);
			divide(srcf, bdest, ratio);
			GaussianBlur(ratio, ratio, ksize, sigma);

			multiply(destf, ratio, destf);
		}
		destf.convertTo(dest, CV_8UC3);
	}


	void fma(const Mat& src, const Mat&  lambda, Mat& dest)
	{
		const float* s = src.ptr<float>(0);
		const float* l = lambda.ptr<float>(0);
		float* d = dest.ptr<float>(0);
		
		for (int i = 0; i < src.size().area(); i += 8)
		{
			__m256 ma = _mm256_load_ps(l + i);
			__m256 ms = _mm256_load_ps(s + i);
			__m256 md = _mm256_load_ps(d + i);
			md = _mm256_fmadd_ps(ma, ms, md);
			_mm256_store_ps(d + i, md);
		}
	}

	void fma(const Mat& src, const float a, Mat& dest)
	{
		const float* s = src.ptr<float>(0);
		float* d = dest.ptr<float>(0);
		__m256 ma = _mm256_set1_ps(a);
		for (int i = 0; i < src.size().area(); i += 8)
		{
			__m256 ms = _mm256_load_ps(s + i);
			__m256 md = _mm256_load_ps(d + i);
			md = _mm256_fmadd_ps(ma, ms, md);
			_mm256_store_ps(d + i, md);
		}
	}

	void eth(Mat& error, Mat& ref, float th)
	{
		const int v = error.cols;
		float sigma = th;
		const float sigmap = -0.5 / (sigma*sigma);
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


				e[i] *= (1.0 - exp(diff*sigmap));
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
			float* s = (float*) src.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 1; i < src.cols; i+=8)
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
				d[i] = min(3.f, lambda + 1.f*abs(s0[i] - s1[i]));
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

		imshow("step", step*0.333);
	}

	void iterativeBackProjectionDeblurGaussian(const Mat& src, Mat& dest, const Size ksize, const float sigma, const float backprojection_sigma, const float lambda, const int iteration, Mat& init)
	{
		Mat srcf;
		Mat destf;
		Mat subf;
		src.convertTo(srcf, CV_32FC3);

		if(init.empty()) src.convertTo(destf, CV_32FC3);
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
			destf += ((float)lambda)*subf;
			
		}
		destf.convertTo(dest, src.type());
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

			cp::guidedFilter(subf, destf, subf, ksize.width / 4, eps);

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
}