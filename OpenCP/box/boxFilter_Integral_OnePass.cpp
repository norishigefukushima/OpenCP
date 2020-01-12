#include "boxFilter_Integral_OnePass.h"
#include <iostream>

using namespace std;
using namespace cv;

//TODO: Color implement
boxFilter_Integral_OnePass::boxFilter_Integral_OnePass(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_base(_src, _dest, _r, _parallelType)
{
	copyMakeBorder(src, copy, r, r, r, r, BOX_FILTER_BORDER_TYPE);
	sum.create(copy.rows + 1, copy.cols + 1, CV_32F);
	ksize = 2 * r + 1;

	// colsum
	ColSum.create(sum.size(), CV_32F);
}

void boxFilter_Integral_OnePass::filter_naive_impl()
{
	for (int y = 1; y < ksize; y++)
	{
		float* copy_ptr = copy.ptr<float>(y - 1);
		float* colsum_prev_ptr = ColSum.ptr<float>(y - 1) + 1;
		float* colsum_ptr = ColSum.ptr<float>(y) + 1;
		float* sum_prev_ptr = sum.ptr<float>(y);
		float* sum_ptr = sum.ptr<float>(y) + 1;
		for (int x = 1; x < sum.cols; x++)
		{
			*colsum_ptr = *copy_ptr + *colsum_prev_ptr;
			*sum_ptr = *sum_prev_ptr + *colsum_ptr;
			colsum_ptr++;
			copy_ptr++;
			colsum_prev_ptr++;
			sum_ptr++;
			sum_prev_ptr++;
		}
	}
	for (int y = ksize; y < sum.rows; y++)
	{
		float* copy_ptr = copy.ptr<float>(y - 1);
		float* colsum_prev_ptr = ColSum.ptr<float>(y - 1) + 1;
		float* colsum_ptr = ColSum.ptr<float>(y) + 1;
		float* sum_prev_ptr = sum.ptr<float>(y);
		float* sum_ptr = sum.ptr<float>(y) + 1;
		float* A_ptr = sum.ptr<float>(y - ksize);
		float* B_ptr = sum.ptr<float>(y - ksize) + ksize;
		float* C_ptr = sum.ptr<float>(y);
		float* D_ptr = sum.ptr<float>(y) + ksize;
		float* dest_ptr = dest.ptr<float>(y - ksize);

		for (int x = 1; x < ksize; x++)
		{
			*colsum_ptr = *copy_ptr + *colsum_prev_ptr;
			*sum_ptr = *sum_prev_ptr + *colsum_ptr;
			colsum_ptr++;
			copy_ptr++;
			colsum_prev_ptr++;
			sum_ptr++;
			sum_prev_ptr++;
		}
		for (int x = ksize; x < sum.cols; x++)
		{
			*colsum_ptr = *copy_ptr + *colsum_prev_ptr;
			*sum_ptr = *sum_prev_ptr + *colsum_ptr;
			colsum_ptr++;
			copy_ptr++;
			colsum_prev_ptr++;
			sum_ptr++;
			sum_prev_ptr++;

			*dest_ptr = (*D_ptr - *C_ptr - *B_ptr + *A_ptr) * div;
			dest_ptr++;
			A_ptr++;
			B_ptr++;
			C_ptr++;
			D_ptr++;
		}
	}

	//for (int y = 1; y < ksize; y++)
	//{
	//	for (int x = 1; x < sum.cols; x++)
	//	{
	//		ColSum.at<float>(y, x) = copy.at<float>(y - 1, x - 1) + ColSum.at<float>(y - 1, x);
	//		sum.at<float>(y, x) = sum.at<float>(y, x - 1) + ColSum.at<float>(y, x);
	//	}
	//}
	//for (int y = ksize; y < sum.rows; y++)
	//{
	//	for (int x = 1; x < ksize; x++)
	//	{
	//		ColSum.at<float>(y, x) = copy.at<float>(y - 1, x - 1) + ColSum.at<float>(y - 1, x);
	//		sum.at<float>(y, x) = sum.at<float>(y, x - 1) + ColSum.at<float>(y, x);
	//	}
	//	for (int x = ksize; x < sum.cols; x++)
	//	{
	//		ColSum.at<float>(y, x) = copy.at<float>(y - 1, x - 1) + ColSum.at<float>(y - 1, x);
	//		sum.at<float>(y, x) = sum.at<float>(y, x - 1) + ColSum.at<float>(y, x);

	//		float A = sum.at<float>(y - ksize, x - ksize);
	//		float B = sum.at<float>(y - ksize, x);
	//		float C = sum.at<float>(y, x - ksize);
	//		float D = sum.at<float>(y, x);
	//		dest.at<float>(y - ksize, x - ksize) = (D - C - B + A) * div;
	//	}
	//}
}

void boxFilter_Integral_OnePass::filter_omp_impl()
{
	cout << "not supported" << endl;
}

void boxFilter_Integral_OnePass::operator()(const cv::Range& range) const
{
	cout << "not supported" << endl;
}



//TODO: Color implement
boxFilter_Integral_OnePass_Area::boxFilter_Integral_OnePass_Area(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_base(_src, _dest, _r, _parallelType)
{
	copyMakeBorder(src, copy, r, r, r, r, BOX_FILTER_BORDER_TYPE);
	ksize = 2 * r + 1;
}

void boxFilter_Integral_OnePass_Area::filter_naive_impl()
{
	for (int k = 0; k < row; k++)
	{
		Mat sum = Mat::zeros(ksize + 1, copy.cols + 1, CV_32F);
		Mat ColSum = Mat::zeros(ksize + 1, copy.cols + 1, CV_32F);
		for (int y = 1; y < ksize; y++)
		{
			for (int x = 1; x < sum.cols; x++)
			{
				ColSum.at<float>(y, x) = copy.at<float>(y - 1 + k, x - 1) + ColSum.at<float>(y - 1, x);
				sum.at<float>(y, x) = sum.at<float>(y, x - 1) + ColSum.at<float>(y, x);
			}
		}
		for (int x = 1; x < ksize; x++)
		{
			ColSum.at<float>(ksize, x) = copy.at<float>(ksize - 1 + k, x - 1) + ColSum.at<float>(ksize - 1, x);
			sum.at<float>(ksize, x) = sum.at<float>(ksize, x - 1) + ColSum.at<float>(ksize, x);
		}
		for (int x = ksize; x < sum.cols; x++)
		{
			ColSum.at<float>(ksize, x) = copy.at<float>(ksize - 1 + k, x - 1) + ColSum.at<float>(ksize - 1, x);
			sum.at<float>(ksize, x) = sum.at<float>(ksize, x - 1) + ColSum.at<float>(ksize, x);

			float C = sum.at<float>(ksize, x - ksize);
			float D = sum.at<float>(ksize, x);
			dest.at<float>(k, x - ksize) = (D - C)*div;
		}
	}
}

void boxFilter_Integral_OnePass_Area::filter_omp_impl()
{

}

void boxFilter_Integral_OnePass_Area::operator()(const cv::Range& range) const
{

}



/* --- box filter integral onepath 8u --- */
boxFilter_Integral_OnePass_8u::boxFilter_Integral_OnePass_8u(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_base(_src, _dest, _r, _parallelType)
{
	copyMakeBorder(src, copy, r, r, r, r, BOX_FILTER_BORDER_TYPE);
	sum.create(copy.rows + 1, copy.cols + 1, CV_32S);
	ksize = 2 * r + 1;
}

void boxFilter_Integral_OnePass_8u::filter_naive_impl()
{
	for (int y = 1; y < ksize; y++)
	{
		uint32_t* a_p = sum.ptr<uint32_t>(y - 1);
		uint32_t* b_p = sum.ptr<uint32_t>(y - 1) + 1;
		uint32_t* c_p = sum.ptr<uint32_t>(y);
		uint32_t* d_p = sum.ptr<uint32_t>(y) + 1;
		uint8_t* copy_p = copy.ptr<uint8_t>(y - 1);
		for (int x = 1; x < sum.cols; x++)
		{
			*d_p = *copy_p + *c_p + *b_p - *a_p;
			a_p++;
			b_p++;
			c_p++;
			d_p++;
			copy_p++;
		}
	}
	for (int y = ksize; y < sum.rows; y++)
	{
		uint32_t* a_p = sum.ptr<uint32_t>(y - 1);
		uint32_t* b_p = sum.ptr<uint32_t>(y - 1) + 1;
		uint32_t* c_p = sum.ptr<uint32_t>(y);
		uint32_t* d_p = sum.ptr<uint32_t>(y) + 1;
		uint8_t* copy_p = copy.ptr<uint8_t>(y - 1);
		for (int x = 1; x < ksize; x++)
		{
			*d_p = *copy_p + *c_p + *b_p - *a_p;
			a_p++;
			b_p++;
			c_p++;
			d_p++;
			copy_p++;
		}
		uint32_t* s_p1 = sum.ptr<uint32_t>(y - ksize);
		uint32_t* s_p2 = sum.ptr<uint32_t>(y - ksize) + ksize;
		uint32_t* s_p3 = sum.ptr<uint32_t>(y);
		uint32_t* s_p4 = sum.ptr<uint32_t>(y) + ksize;
		float* dp = dest.ptr<float>(y - ksize);
		for (int x = ksize; x < sum.cols; x++)
		{
			*d_p = *copy_p + *c_p + *b_p - *a_p;
			a_p++;
			b_p++;
			c_p++;
			d_p++;
			copy_p++;

			*dp = static_cast<float>(*s_p4 - *s_p3 - *s_p2 + *s_p1) * div;
			s_p1++;
			s_p2++;
			s_p3++;
			s_p4++;
			dp++;
		}
	}
}

void boxFilter_Integral_OnePass_8u::filter_omp_impl()
{
	cout << "not supported" << endl;
}

void boxFilter_Integral_OnePass_8u::operator()(const cv::Range& range) const
{
	cout << "not supported" << endl;
}
