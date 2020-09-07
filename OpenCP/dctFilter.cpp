#include "dctFilter.hpp"

using namespace cv;

namespace cp
{
	template<class srcType>
	void createGaussianDCTIWiener_(Mat& src, const srcType sigma, const srcType eps)
	{
		srcType temp = sigma*(srcType)CV_PI / (src.cols);
		srcType a = -temp*temp / (srcType)2.0;

		for (int j = 0; j < src.rows; j++)
		{
			for (int i = 0; i < src.cols; i++)
			{
				srcType d = (srcType)(i*i + j*j);
				srcType v = exp(a*d);
				src.at<srcType>(j, i) = v*v / (v*v + eps);
			}
		}
	}

	template<class srcType>
	void createGaussianDCTITihkonov_(Mat& src, const srcType sigma, const srcType eps)
	{
		srcType temp = sigma*(srcType)CV_PI / (src.cols);
		srcType a = -temp*temp / (srcType)2.0;

		for (int j = 0; j < src.rows; j++)
		{
			for (int i = 0; i < src.cols; i++)
			{
				srcType d = (srcType)(i*i + j*j);
				srcType v = exp(a*d);
				//src.at<T>(j, i) = v*v / (v*v + eps);

				//v = (1.0 - eps)*v;

				//src.at<T>(j, i) = sqrt(v)*v / (sqrt(v)*v + eps);
				

				//src.at<T>(j, i) *= (1.f + eps);
				src.at<srcType>(j, i) = v*v*v*v / (v*v*v*v + eps);
				
				/*
				if(i==j)
					src.at<T>(j, i) = v*v / (v*v + eps);
				else
					src.at<T>(j, i) = v*v / (v*v + eps*0.5);
				*/
			}
		}
	}

	template<class srcType>
	void createGaussianDCTIT_(Mat& src, const srcType sigma, const srcType eps)
	{
		srcType temp = sigma*CV_PI / (src.cols);
		srcType a = -temp*temp / 2.0;

		for (int j = 0; j < src.rows; j++)
		{
			for (int i = 0; i < src.cols; i++)
			{
				srcType d = i*i + j*j;
				srcType v = exp(a*d);
				src.at<srcType>(j, i) = v + eps;
			}
		}
	}

	template<class srcType>
	void createGaussianDCTI_(Mat& src, const srcType sigma, const srcType max_clip_eps)
	{
		srcType temp = sigma*(srcType)CV_PI / (src.cols);
		srcType a = -temp*temp / (srcType)2.0;

		for (int j = 0; j < src.rows; j++)
		{
			for (int i = 0; i < src.cols; i++)
			{
				srcType d = (srcType)(i*i + j*j);
				srcType v = max(max_clip_eps, exp(a*d));
				src.at<srcType>(j, i) = v;
			}
		}
	}

	void GaussianFilterDCT64f(Mat& src, Mat& dest, const float sigma)
	{
		Mat srcf, destf;

		src.convertTo(srcf, CV_64F);
		Mat frec = Mat::zeros(src.size(), CV_64F);

		cv::dct(srcf, frec);

		Mat dctkernel = Mat::zeros(src.size(), CV_64F);

		createGaussianDCTI_<double>(dctkernel, sigma, FLT_MIN);

		idct(frec, destf);

		destf.convertTo(dest, src.depth(), 1.0);
	}

	//boundary condition is refrect
	class GaussianFilterDCT
	{
	private:
		Mat frec;
		float lutsigma;
		int lutsizex;
		int lutsizey;
		float* LUTx;
		float* LUTy;
		float denormalclip;

		void free()
		{
			_mm_free(LUTx);
			_mm_free(LUTy);
		}

		void setLUT(int width, int height, float sigma)
		{
			if (lutsizex != width || lutsizey != height)
			{
				free();

				LUTx = (float*)_mm_malloc(width * sizeof(float), 32);
				LUTy = (float*)_mm_malloc(height * sizeof(float), 32);
			}

			if (lutsigma != sigma)
			{
				lutsigma = sigma;

				if (width == height)
				{
					const float temp = sigma*(float)CV_PI / (width);
					const float a = -temp*temp / 2.f;
					for (int i = 0; i < width; i++)
					{
						//float arg = max(a*i*i, -87.3365f);
						float arg = max(a*i*i, denormalclip);//clip denormal number
						float v = exp(arg);
						LUTx[i] = v;
						LUTy[i] = v;
					}
				}
				else
				{
					const float tx = sigma*(float)CV_PI / (width);
					const float ax = -tx*tx / 2.f;
					const float ty = sigma*(float)CV_PI / (height);
					const float ay = -ty*ty / 2.f;
					for (int i = 0; i < width; i++)
					{

						float arg = max(ax*i*i, denormalclip);//clip denormal number
						float v = exp(arg);
						LUTx[i] = v;
					}
					for (int i = 0; i < height; i++)
					{
						float arg = max(ay*i*i, denormalclip);//clip denormal number
						float v = exp(arg);
						LUTy[i] = v;
					}
				}

			}
		}

		void multiplyDCTGaussianKernel(Mat& src)
		{
			CV_Assert(src.cols % 32 == 0);
			for (int j = 0; j < src.rows; j++)
			{
				float* f = src.ptr<float>(j);
				const __m256 b = _mm256_set1_ps(LUTy[j]);
				for (int i = 0; i < src.cols; i += 32)
				{
					__m256 a = _mm256_load_ps(LUTx + i);
					__m256 c = _mm256_load_ps(f + i);
					_mm256_store_ps(f + i, _mm256_mul_ps(b, _mm256_mul_ps(c, a)));
					a = _mm256_load_ps(LUTx + i + 8);
					c = _mm256_load_ps(f + i + 8);
					_mm256_store_ps(f + i + 8, _mm256_mul_ps(b, _mm256_mul_ps(c, a)));
					a = _mm256_load_ps(LUTx + i + 16);
					c = _mm256_load_ps(f + i + 16);
					_mm256_store_ps(f + i + 16, _mm256_mul_ps(b, _mm256_mul_ps(c, a)));
					a = _mm256_load_ps(LUTx + i + 24);
					c = _mm256_load_ps(f + i + 24);
					_mm256_store_ps(f + i + 24, _mm256_mul_ps(b, _mm256_mul_ps(c, a)));
				}
				//for (int i = 0; i <src.cols; i ++)
				//{
				//float v = LUTx[i] * LUTy[j];
				//f[i] *= v;
				//}
			}
		}
	public:
		~GaussianFilterDCT()
		{
			free();
		}


		GaussianFilterDCT()
		{
			denormalclip = -sqrt(87.3365f);
			lutsigma = 0.0;
			lutsizex = 0;
			lutsizey = 0;
			LUTx = 0;
			LUTy = 0;
		}

		void filter(Mat& src, Mat& dest, const float sigma)
		{
			setLUT(src.cols, src.rows, sigma);
			cv::dct(src, frec);

			multiplyDCTGaussianKernel(frec);
			idct(frec, dest);
		}
	};


	void deblurDCT64f(Mat& src, Mat& dest, const double sigma, const double eps, const int ks)
	{
		Mat srcf, destf;

		src.convertTo(srcf, CV_64F);
		Mat frec = Mat::zeros(src.size(), CV_64F);

		cv::dct(srcf, frec);

		Mat dctkernel = Mat::zeros(src.size(), CV_64F);
		/*
		createGaussianDCTI_<double>(dctkernel, sigma, eps, ks);
		divide(frec, dctkernel, frec);
		*/


		double temp = sigma*CV_PI / (src.cols);
		double a = -temp*temp / 2.0;
		int r = min((int)(ks * sigma), src.cols - 1);

		for (int j = 0; j <= r; j++)
		{
			for (int i = 0; i <= r; i++)
			{
				double d = i*i + j*j;
				double v = exp(a*d);

				frec.at<double>(j, i) /= (v + eps);
			}
		}

		idct(frec, destf);

		destf.convertTo(dest, src.depth());
	}

	void deblurDCT32f(Mat& src, Mat& dest, const float sigma, const float eps, const int ks)
	{
		Mat srcf, destf;
		src.convertTo(srcf, CV_32F);

		Mat dctkernel = Mat::zeros(src.size(), CV_32F);
		Mat frec = Mat::zeros(src.size(), CV_32F);

		cv::dct(srcf, frec);
		createGaussianDCTI_<float>(dctkernel, sigma, eps);
		divide(frec, dctkernel, frec);
		//multiply(frec, fkernel, frec);//for test

		idct(frec, destf);

		destf.convertTo(dest, src.depth(), 1.f + eps);

		//printMat(fkernel, 6, 6);
	}

	void deblurDCTTihkonov(Mat& src, Mat& dest, const float sigma, const float eps)
	{
		Mat srcf, destf;
		src.convertTo(srcf, CV_32F);

		Mat dctkernel = Mat::zeros(src.size(), CV_32F);
		Mat dctkernel2 = Mat::zeros(src.size(), CV_32F);
		Mat frec = Mat::zeros(src.size(), CV_32F);
		
		cv::dct(srcf, frec);

		createGaussianDCTI_<float>(dctkernel, sigma, FLT_EPSILON);
		createGaussianDCTITihkonov_<float>(dctkernel2, sigma, eps);
		divide(dctkernel2, dctkernel, dctkernel2);

		multiply(frec, dctkernel2, frec);
		idct(frec, destf);

		destf.convertTo(dest, src.depth(), (1.f + eps));
	}

	void deblurDCTWiener(Mat& src, Mat& dest, const float sigma, const float eps)
	{
		Mat srcf, destf;
		src.convertTo(srcf, CV_32F);

		Mat dctkernel = Mat::zeros(src.size(), CV_32F);
		Mat dctkernel2 = Mat::zeros(src.size(), CV_32F);
		Mat frec = Mat::zeros(src.size(), CV_32F);

		cv::dct(srcf, frec);

		createGaussianDCTI_<float>(dctkernel, sigma, FLT_EPSILON);
		createGaussianDCTIWiener_<float>(dctkernel2, sigma, eps);
		divide(dctkernel2, dctkernel, dctkernel2);

		multiply(frec, dctkernel2, frec);
		idct(frec, destf);

		destf.convertTo(dest, src.depth(), (1.f + eps));
	}

	void deblurdenoiseDCTWiener32f(Mat& src, Mat& dest, const float sigma, const float eps, const float th)
	{
		Mat srcf, destf;
		src.convertTo(srcf, CV_32F);

		Mat dctkernel = Mat::zeros(src.size(), CV_32F);
		Mat dctkernel2 = Mat::zeros(src.size(), CV_32F);
		Mat frec = Mat::zeros(src.size(), CV_32F);

		cv::dct(srcf, frec);

		createGaussianDCTI_<float>(dctkernel, sigma, FLT_EPSILON);
		createGaussianDCTIWiener_<float>(dctkernel2, sigma, eps);
		divide(dctkernel2, dctkernel, dctkernel2);

		float* s = frec.ptr<float>(0);
		for (int i = 1; i < frec.size().area(); i++)
		{
			s[i] = (abs(s[i]) < th) ? 0.f : s[i];
		}

		multiply(frec, dctkernel2, frec);
		idct(frec, destf);

		destf.convertTo(dest, src.depth(), (1.f + eps));
	}

	void deblurdenoiseDCT32f(Mat& src, Mat& dest, const float sigma, const float eps, const float th)
	{
		Mat srcf, destf;
		src.convertTo(srcf, CV_32F);

		Mat dctkernel = Mat::zeros(src.size(), CV_32F);
		Mat frec = Mat::zeros(src.size(), CV_32F);

		cv::dct(srcf, frec);
		createGaussianDCTI_<float>(dctkernel, sigma, eps);

		float* s = frec.ptr<float>(0);
		for (int i = 1; i < frec.size().area(); i++)
		{
			s[i] = (abs(s[i]) < th) ? 0.f : s[i];
		}

		divide(frec, dctkernel, frec);
		idct(frec, destf);

		destf.convertTo(dest, src.depth(), (1.f + eps));
	}
}
