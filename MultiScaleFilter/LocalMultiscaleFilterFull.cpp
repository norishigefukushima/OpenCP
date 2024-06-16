#include "multiscalefilter/MultiScaleFilter.hpp"
using namespace cv;
using namespace std;
namespace cp
{
	void LocalMultiScaleFilterFull::pyramid(const Mat& src_, Mat& dest_)
	{
		//rangeDescope(src_);

		Mat src, dest;
		const int r = (int)pow(2, level) * 4;
		copyMakeBorder(src_, src, r, r, r, r, borderType);

		//const bool isDebug = true;
		const bool isDebug = false;

		initRangeTable(sigma_range, boost);

		Mat srcf;
		if (src.depth() == CV_8U) src.convertTo(srcf, CV_32F);
		else srcf = src;

		//(1) build Gaussian pyramid
		ImageStack.resize(level + 1);
		srcf.copyTo(ImageStack[0]);
		buildGaussianPyramid(ImageStack[0], ImageStack, level, sigma_space);

		if (isDebug) showPyramid("GaussPy", ImageStack);

		vector<Mat> LaplacianPyramid(ImageStack.size());
		LaplacianPyramid[0].create(src.size(), ImageStack[0].depth());
		for (int l = 1; l < LaplacianPyramid.size(); l++)
		{
			LaplacianPyramid[l].create(LaplacianPyramid[l - 1].size() / 2, ImageStack[0].depth());
		}

		//(2) build Laplacian pyramid (0 to level)
		for (int l = 0; l <= level; l++)
		{
			const int height = src.rows / (int)pow(2, l);
			const int width = src.cols / (int)pow(2, l);

			if (l == level)
			{
				ImageStack[l](Rect(0, 0, width, height)).copyTo(LaplacianPyramid[l]);
			}
			else
			{
				if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
				{
#pragma omp parallel for schedule (dynamic)
					for (int j = 0; j < height; j++)
					{
						vector<Mat> llp;
						Mat rm(srcf.size(), CV_32F);
						for (int i = 0; i < width; i++)
						{
							const float g = ImageStack[l].at<float>(j, i);
							const float sigma = adaptiveSigmaMap[l].at<float>(j, i);
							const float boost = adaptiveBoostMap[l].at<float>(j, i);
							remap(srcf, rm, g, sigma, boost);
							buildLaplacianPyramid(rm, llp, l + 1, sigma_space);
							LaplacianPyramid[l].at<float>(j, i) = llp[l].at<float>(j, i);
						}
					}
				}
				else
				{
#pragma omp parallel for schedule (dynamic)
					for (int j = 0; j < height; j++)
					{
						vector<Mat> llp;
						Mat rm(src.size(), CV_32F);
						//Mat rm = srcf.clone();
						for (int i = 0; i < width; i++)
						{
							const float g = ImageStack[l].at<float>(j, i);
							remap(srcf, rm, g, sigma_range, boost);
							buildLaplacianPyramid(rm, llp, l + 1, sigma_space);
							LaplacianPyramid[l].at<float>(j, i) = llp[l].at<float>(j, i);
						}
					}
				}
			}

			if (isDebug) showPyramid("Laplacian Pyramid Paris2011", LaplacianPyramid);
			collapseLaplacianPyramid(LaplacianPyramid, dest, src.depth());
			dest(Rect(r, r, src_.cols, src_.rows)).copyTo(dest_);
		}
	}

	void LocalMultiScaleFilterFull::dog(const Mat& src, Mat& dest)
	{
		initRangeTable(sigma_range, boost);

		Mat srcf;
		if (src.depth() == CV_32F)
		{
			srcf = src;
		}
		else
		{
			src.convertTo(srcf, CV_32F);
		}

		const float sigma_lmax = (float)getPyramidSigma(sigma_space, level);
		const int rmax = (int)ceil(sigma_lmax * 3.f);
		const Size ksizemax(2 * rmax + 1, 2 * rmax + 1);

		const int r_pad = (int)pow(2, level + 1);//2^(level+1)
		vector<Mat>  LaplacianStack(level + 1);

		//(1) build Gaussian stack
		buildGaussianStack(srcf, ImageStack, sigma_space, level);

		for (int i = 0; i < level; i++)
		{
			LaplacianStack[i].create(ImageStack[0].size(), CV_32F);
		}
		Mat im;
		copyMakeBorder(srcf, im, rmax, rmax, rmax, rmax, BORDER_DEFAULT);

		//(2) build DoG stack (0 to level-1)
		for (int l = 0; l < level; l++)
		{
			const float sigma_l = (float)getPyramidSigma(sigma_space, l);
			const float sigma_lp = (float)getPyramidSigma(sigma_space, l + 1);
			const int r = (int)ceil(sigma_lp * 3.f);
			const Size ksize(2 * r + 1, 2 * r + 1);
			AutoBuffer<float> weight(ksize.area());
			AutoBuffer<int> index(ksize.area());
			setDoGKernel(weight, index, im.cols, ksize, sigma_l, sigma_lp);

#pragma omp parallel for schedule (dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				for (int i = 0; i < src.cols; i++)
				{
					const float g = ImageStack[l].at<float>(j, i);
					LaplacianStack[l].at<float>(j, i) = getDoGCoeffLn(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
					//if(l==0)LaplacianStack[l].at<float>(j, i) = getDoGCoeffLnNoremap(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
					//else LaplacianStack[l].at<float>(j, i) = getDoGCoeffLn(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
				}
			}
		}
		//(2) the last level is a copy of the last level DoG
		ImageStack[level].copyTo(LaplacianStack[level]);

		//(3) collapseDoG
		//showPyramid("stack", LaplacianStack, 10);

		collapseDoGStack(LaplacianStack, dest, src.depth());

		for (int l = 0; l < level; l++) LaplacianStack[l] = 2.f * LaplacianStack[l] + 127.5; Mat show; hconcat(LaplacianStack, show); imshowScale("stack", show);
	}

	void LocalMultiScaleFilterFull::dog(const Mat& src, const Mat& guide, Mat& dest)
	{
		initRangeTable(sigma_range, boost);

		Mat srcf, guidef;
		if (src.depth() == CV_32F)
		{
			srcf = src;
			guidef = guide;
		}
		else
		{
			src.convertTo(srcf, CV_32F);
			guide.convertTo(guidef, CV_32F);
		}

		const float sigma_lmax = (float)getPyramidSigma(sigma_space, level);
		const int rmax = (int)ceil(sigma_lmax * 3.f);
		const Size ksizemax(2 * rmax + 1, 2 * rmax + 1);

		const int r_pad = (int)pow(2, level + 1);//2^(level+1)
		vector<Mat> DifferenceStack(level + 1);
		vector<Mat> JointStack(level + 1);

		//(1) build Gaussian stack
		buildGaussianStack(srcf, ImageStack, sigma_space, level);
		buildGaussianStack(guidef, JointStack, sigma_space, level);

		for (int i = 0; i < level; i++)
		{
			DifferenceStack[i].create(ImageStack[0].size(), CV_32F);
		}
		Mat im, gm;
		copyMakeBorder(srcf, im, rmax, rmax, rmax, rmax, BORDER_DEFAULT);
		copyMakeBorder(guidef, gm, rmax, rmax, rmax, rmax, BORDER_DEFAULT);

		//(2) build DoG stack (0 to level-1)
		for (int l = 0; l < level; l++)
		{
			const float sigma_l = (float)getPyramidSigma(sigma_space, l);
			const float sigma_lp = (float)getPyramidSigma(sigma_space, l + 1);
			const int r = (int)ceil(sigma_lp * 3.f);
			const Size ksize(2 * r + 1, 2 * r + 1);
			AutoBuffer<float> weight(ksize.area());
			AutoBuffer<int> index(ksize.area());
			setDoGKernel(weight, index, im.cols, ksize, sigma_l, sigma_lp);

#pragma omp parallel for schedule (dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				for (int i = 0; i < src.cols; i++)
				{
					const float g = ImageStack[l].at<float>(j, i);
					const float h = JointStack[l].at<float>(j, i);
					//DifferenceStack[l].at<float>(j, i) = getDoGCoeffLn(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
					//DifferenceStack[l].at<float>(j, i) = getDoGCoeffLn(im, gm, g, h, j + rmax, i + rmax, ksize.area(), index, weight);
					 DifferenceStack[l].at<float>(j, i) = getDoGCoeffLn(gm, gm, h, h, j + rmax, i + rmax, ksize.area(), index, weight);
					//if(l==0)LaplacianStack[l].at<float>(j, i) = getDoGCoeffLnNoremap(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
					//else LaplacianStack[l].at<float>(j, i) = getDoGCoeffLn(im, g, j + rmax, i + rmax, ksize.area(), index, weight);
				}
			}
		}
		//(2) the last level is a copy of the last level DoG
		ImageStack[level].copyTo(DifferenceStack[level]);

		//(3) collapseDoG
		collapseDoGStack(DifferenceStack, dest, src.depth());

		for (int l = 0; l < level; l++) DifferenceStack[l] = 2.f * DifferenceStack[l] + 127.5; Mat show; hconcat(DifferenceStack, show); imshowScale("stack", show);
	}

	void LocalMultiScaleFilterFull::filter(const Mat& src, Mat& dest, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);
		this->pyramidComputeMethod = Fast;

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->level = level;
		this->boost = boost;
		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);

		freeSpaceWeight();
	}

	void LocalMultiScaleFilterFull::jointfilter(const Mat& src, const Mat& guide, Mat& dest, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);
		this->pyramidComputeMethod = Fast;

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->level = level;
		this->boost = boost;
		this->scalespaceMethod = scaleSpaceMethod;

		body(src, guide, dest);

		freeSpaceWeight();
	}

	void LocalMultiScaleFilterFull::setDoGKernel(float* weight, int* index, const int index_step, Size ksize, const float sigma1, const float sigma2)
	{
		CV_Assert(sigma2 > sigma1);

		const int r = ksize.width / 2;
		int count = 0;
		if (sigma1 == 0.f)
		{
			float sum2 = 0.f;
			const float coeff2 = float(1.0 / (-2.0 * sigma2 * sigma2));
			for (int j = -r; j <= r; j++)
			{
				for (int i = -r; i <= r; i++)
				{
					const float dist = float(j * j + i * i);
					const float v2 = exp(dist * coeff2);
					weight[count] = v2;
					index[count] = j * index_step + i;
					count++;
					sum2 += v2;
				}
			}
			sum2 = 1.f / sum2;
			for (int i = 0; i < ksize.area(); i++)
			{
				weight[i] = 0.f - weight[i] * sum2;
			}
			weight[ksize.area() / 2] = 1.f + weight[ksize.area() / 2];
		}
		else
		{
			AutoBuffer<float> buff(ksize.area());
			float sum1 = 0.f;
			float sum2 = 0.f;
			const float coeff1 = float(1.0 / (-2.0 * sigma1 * sigma1));
			const float coeff2 = float(1.0 / (-2.0 * sigma2 * sigma2));
			for (int j = -r; j <= r; j++)
			{
				for (int i = -r; i <= r; i++)
				{
					float dist = float(j * j + i * i);
					float v1 = exp(dist * coeff1);
					float v2 = exp(dist * coeff2);
					weight[count] = v1;
					buff[count] = v2;
					index[count] = j * index_step + i;
					count++;
					sum1 += v1;
					sum2 += v2;
				}
			}
			sum1 = 1.f / sum1;
			sum2 = 1.f / sum2;
			for (int i = 0; i < ksize.area(); i++)
			{
				weight[i] = weight[i] * sum1 - buff[i] * sum2;
			}
		}
	}

	float LocalMultiScaleFilterFull::getDoGCoeffLnNoremap(Mat& src, const float g, const int y, const int x, const int size, int* index, float* weight)
	{
		float* sptr = src.ptr<float>(y, x);
		const int simd_size = get_simd_floor(size, 8);

		const __m256 mg = _mm256_set1_ps(g);
		__m256 msum = _mm256_setzero_ps();
		for (int i = 0; i < simd_size; i += 8)
		{
			__m256i idx = _mm256_load_si256((const __m256i*)index);
			const __m256 ms = _mm256_i32gather_ps(sptr, idx, sizeof(float));
			msum = _mm256_fmadd_ps(_mm256_load_ps(weight), ms, msum);
			weight += 8;
			index += 8;
		}
		float sum = _mm256_reduceadd_ps(msum);

		for (int i = simd_size; i < size; i++)
		{
			const float s = sptr[*index];
			sum += *weight * s;
			weight++;
			index++;
		}

		return sum;
	}

	float LocalMultiScaleFilterFull::getDoGCoeffLn(const Mat& src, const float g, const int y, const int x, const int size, int* index, float* weight)
	{
		const float* sptr = src.ptr<float>(y, x);
		float* rptr = &rangeTable[0];
		const int simd_size = get_simd_floor(size, 8);
		//cout << "size : " << size << endl;
		//cout << "simd_size : " << simd_size << endl;

		const __m256 mg = _mm256_set1_ps(g);
		__m256 msum = _mm256_setzero_ps();
		for (int i = 0; i < simd_size; i += 8)
		{
			__m256i idx = _mm256_load_si256((const __m256i*)index);
			const __m256 ms = _mm256_i32gather_ps(sptr, idx, sizeof(float));
			const __m256 subsg = _mm256_sub_ps(ms, mg);
			const __m256 md = _mm256_fnmadd_ps(_mm256_i32gather_ps(rptr, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), subsg, ms);
			msum = _mm256_fmadd_ps(_mm256_load_ps(weight), md, msum);
			weight += 8;
			index += 8;
		}
		float sum = _mm256_reduceadd_ps(msum);

		for (int i = simd_size; i < size; i++)
		{
			const float s = sptr[*index];
			const float d = s - (s - g) * rangeTable[saturate_cast<uchar>(abs(s - g))];
			sum += *weight * d;
			weight++;
			index++;
		}

		return sum;
	}

	float LocalMultiScaleFilterFull::getDoGCoeffLn(const Mat& src, const Mat& guide, const float g, const float h, const int y, const int x, const int size, int* index, float* weight)
	{
		const float* sptr = src.ptr<float>(y, x);
		const float* jptr = guide.ptr<float>(y, x);
		float* rptr = &rangeTable[0];
		const int simd_size = get_simd_floor(size, 8);
		//cout << "size : " << size << endl;
		//cout << "simd_size : " << simd_size << endl;

		const __m256 mg = _mm256_set1_ps(g);
		const __m256 mh = _mm256_set1_ps(h);
		__m256 msum = _mm256_setzero_ps();
		for (int i = 0; i < simd_size; i += 8)
		{
			__m256i idx = _mm256_load_si256((const __m256i*)index);
			const __m256 ms = _mm256_i32gather_ps(sptr, idx, sizeof(float));
			const __m256 mj = _mm256_i32gather_ps(jptr, idx, sizeof(float));
			const __m256 subsg = _mm256_sub_ps(ms, mg);
			const __m256 subjh = _mm256_sub_ps(mj, mh);
			const __m256 md = _mm256_fnmadd_ps(_mm256_i32gather_ps(rptr, _mm256_cvtps_epi32(_mm256_abs_ps(subjh)), sizeof(float)), subsg, ms);
			msum = _mm256_fmadd_ps(_mm256_load_ps(weight), md, msum);
			weight += 8;
			index += 8;
		}
		float sum = _mm256_reduceadd_ps(msum);
		
		//float sum = 0.f; for (int i = 0; i < size; i++)
		for (int i = simd_size; i < size; i++)
		{
			const float s = sptr[*index];
			const float j = jptr[*index];
			const float d = s - (s - g) * rangeTable[saturate_cast<uchar>(abs(j - h))];
			sum += *weight * d;
			weight++;
			index++;
		}
		return sum;
	}
}