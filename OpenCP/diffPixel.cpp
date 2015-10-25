#include <opencp.hpp>

using namespace std;
using namespace cv;

namespace cp
{

	template <class T>
	void HMIN(Mat& src, Mat& ref, Mat& dest, T thresh)
	{
		dest = Mat::zeros(src.size(), CV_8U);
		for (int j = 0; j < src.rows; j++)
		{
			T* s = src.ptr<T>(j);
			T* r = ref.ptr<T>(j);
			T* d = dest.ptr<T>(j);
			for (int i = 0; i<src.cols; i++)
			{
				if (abs(s[i] - r[i])>thresh)
				{
					if (s[i] < r[i])
					{
						d[i] = 255;
					}
					else
					{
						d[i - 1] = 255;
					}
				}
			}
		}
	}

	template <class T>
	void HMAX(Mat& src, Mat& ref, Mat& dest, T thresh)
	{
		dest = Mat::zeros(src.size(), CV_8U);
		for (int j = 0; j < src.rows; j++)
		{
			T* s = src.ptr<T>(j);
			T* r = ref.ptr<T>(j);
			T* d = dest.ptr<T>(j);
			for (int i = 0; i<src.cols; i++)
			{
				if (abs(s[i] - r[i])>thresh)
				{
					if (s[i] > r[i])
					{
						d[i] = 255;
					}
					else
					{
						d[i - 1] = 255;
					}
				}
			}
		}
	}

	template <class T>
	void VMIN(Mat& src, Mat& ref, Mat& dest, T thresh)
	{
		dest = Mat::zeros(src.size(), CV_8U);
		for (int j = 0; j < src.rows; j++)
		{
			T* s = src.ptr<T>(j);
			T* r = ref.ptr<T>(j);
			T* d = dest.ptr<T>(j);
			for (int i = 0; i<src.cols; i++)
			{
				if (abs(s[i] - r[i])>thresh)
				{
					if (s[i] < r[i])
					{
						d[i] = 255;
					}
					else
					{
						d[i - src.cols] = 255;
					}
				}
			}
		}
	}

	template <class T>
	void VMAX(Mat& src, Mat& ref, Mat& dest, T thresh)
	{
		dest = Mat::zeros(src.size(), CV_8U);
		for (int j = 0; j < src.rows; j++)
		{
			T* s = src.ptr<T>(j);
			T* r = ref.ptr<T>(j);
			T* d = dest.ptr<T>(j);
			for (int i = 0; i<src.cols; i++)
			{
				if (abs(s[i] - r[i])>thresh)
				{
					if (s[i] > r[i])
					{
						d[i] = 255;
					}
					else
					{
						d[i - src.cols] = 255;
					}
				}
			}
		}
	}
	void pixelDiffThresh(Mat& src, Mat& dest, double thresh, int direction)
	{
		Mat im;
		if (direction == PIXEL_DIFF_DIRECTION_H)
		{
			warpShift(src, im, 1, 0, BORDER_REPLICATE);
			compare(abs(im - src), thresh, dest, CMP_GT);
			Mat element = Mat::zeros(3, 1, CV_8U);
			element.at<uchar>(0, 1) = 255;
			element.at<uchar>(0, 2) = 255;
			dilate(dest, dest, element);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_HMIN)
		{
			warpShift(src, im, 1, 0, BORDER_REPLICATE);
			if (src.depth() == CV_8U)
				HMIN<uchar>(src, im, dest, (uchar)thresh);
			else if (src.depth() == CV_16U)
				HMIN<ushort>(src, im, dest, (ushort)thresh);
			if (src.depth() == CV_16S)
				HMIN<short>(src, im, dest, (short)thresh);
			if (src.depth() == CV_32F)
				HMIN<float>(src, im, dest, (float)thresh);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_HMAX)
		{
			warpShift(src, im, 1, 0, BORDER_REPLICATE);
			if (src.depth() == CV_8U)
				HMAX<uchar>(src, im, dest, (uchar)thresh);
			else if (src.depth() == CV_16U)
				HMAX<ushort>(src, im, dest, (ushort)thresh);
			if (src.depth() == CV_16S)
				HMAX<short>(src, im, dest, (short)thresh);
			if (src.depth() == CV_32F)
				HMAX<float>(src, im, dest, (float)thresh);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_VMIN)
		{
			warpShift(src, im, 0, 1, BORDER_REPLICATE);
			if (src.depth() == CV_8U)
				VMIN<uchar>(src, im, dest, (uchar)thresh);
			else if (src.depth() == CV_16U)
				VMIN<ushort>(src, im, dest, (ushort)thresh);
			if (src.depth() == CV_16S)
				VMIN<short>(src, im, dest, (short)thresh);
			if (src.depth() == CV_32F)
				VMIN<float>(src, im, dest, (float)thresh);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_VMAX)
		{
			warpShift(src, im, 0, 1, BORDER_REPLICATE);
			if (src.depth() == CV_8U)
				VMAX<uchar>(src, im, dest, (uchar)thresh);
			else if (src.depth() == CV_16U)
				VMAX<ushort>(src, im, dest, (ushort)thresh);
			if (src.depth() == CV_16S)
				VMAX<short>(src, im, dest, (short)thresh);
			if (src.depth() == CV_32F)
				VMAX<float>(src, im, dest, (float)thresh);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_HVMAX)
		{
			warpShift(src, im, 0, 1, BORDER_REPLICATE);
			if (src.depth() == CV_8U)
				VMAX<uchar>(src, im, dest, (uchar)thresh);
			else if (src.depth() == CV_16U)
				VMAX<ushort>(src, im, dest, (ushort)thresh);
			if (src.depth() == CV_16S)
				VMAX<short>(src, im, dest, (short)thresh);
			if (src.depth() == CV_32F)
				VMAX<float>(src, im, dest, (float)thresh);

			Mat temp;
			warpShift(src, im, 1, 0, BORDER_REPLICATE);
			if (src.depth() == CV_8U)
				HMAX<uchar>(src, im, temp, (uchar)thresh);
			else if (src.depth() == CV_16U)
				HMAX<ushort>(src, im, temp, (ushort)thresh);
			if (src.depth() == CV_16S)
				HMAX<short>(src, im, temp, (short)thresh);
			if (src.depth() == CV_32F)
				HMAX<float>(src, im, temp, (float)thresh);

			bitwise_or(dest, temp, dest);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_HVMIN)
		{
			warpShift(src, im, 0, 1, BORDER_REPLICATE);
			if (src.depth() == CV_8U)
				VMIN<uchar>(src, im, dest, (uchar)thresh);
			else if (src.depth() == CV_16U)
				VMIN<ushort>(src, im, dest, (ushort)thresh);
			if (src.depth() == CV_16S)
				VMIN<short>(src, im, dest, (short)thresh);
			if (src.depth() == CV_32F)
				VMIN<float>(src, im, dest, (float)thresh);

			Mat temp;
			warpShift(src, im, 1, 0, BORDER_REPLICATE);
			if (src.depth() == CV_8U)
				HMIN<uchar>(src, im, temp, (uchar)thresh);
			else if (src.depth() == CV_16U)
				HMIN<ushort>(src, im, temp, (ushort)thresh);
			if (src.depth() == CV_16S)
				HMIN<short>(src, im, temp, (short)thresh);
			if (src.depth() == CV_32F)
				HMIN<float>(src, im, temp, (float)thresh);

			bitwise_or(dest, temp, dest);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_V)
		{
			warpShift(src, im, 0, 1, BORDER_REPLICATE);
			compare(abs(im - src), thresh, dest, CMP_GT);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_HH)
		{
			warpShift(src, im, 1, 0, BORDER_REPLICATE);
			compare(abs(im - src), thresh, dest, CMP_GT);
			Mat element = Mat::zeros(Size(3, 1), CV_8U);
			element.at<uchar>(0, 1) = 255;
			element.at<uchar>(0, 2) = 255;
			dilate(dest, dest, element);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_VV)
		{
			warpShift(src, im, 0, 1, BORDER_REPLICATE);
			compare(abs(im - src), thresh, dest, CMP_GT);
			Mat element = Mat::zeros(Size(1, 3), CV_8U);
			element.at<uchar>(1, 0) = 255;
			element.at<uchar>(2, 0) = 255;
			dilate(dest, dest, element);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_HV)
		{
			warpShift(src, im, 1, 0, BORDER_REPLICATE);
			compare(abs(im - src), thresh, dest, CMP_GT);

			Mat temp;
			warpShift(src, im, 0, 1, BORDER_REPLICATE);
			compare(abs(im - src), thresh, temp, CMP_GT);

			bitwise_or(dest, temp, dest);
		}
		else if (direction == PIXEL_DIFF_DIRECTION_HHVV)
		{
			warpShift(src, im, 1, 0, BORDER_REPLICATE);
			compare(abs(im - src), thresh, dest, CMP_GT);
			Mat element = Mat::zeros(Size(3, 1), CV_8U);
			element.at<uchar>(0, 1) = 255;
			element.at<uchar>(0, 2) = 255;
			dilate(dest, dest, element);

			Mat temp;
			warpShift(src, im, 0, 1, BORDER_REPLICATE);
			compare(abs(im - src), thresh, temp, CMP_GT);
			Mat element2 = Mat::zeros(Size(1, 3), CV_8U);
			element2.at<uchar>(1, 0) = 255;
			element2.at<uchar>(2, 0) = 255;
			dilate(temp, temp, element2);

			bitwise_or(dest, temp, dest);
		}
	}

	void pixelDiffABS(Mat& src, Mat& dest, int direction)
	{
		Mat im;
		if (direction == PIXEL_DIFF_DIRECTION_H)
			warpShift(src, im, 1, 0, BORDER_REPLICATE);

		if (direction == PIXEL_DIFF_DIRECTION_V)
			warpShift(src, im, 0, 1, BORDER_REPLICATE);

		absdiff(im, src, dest);
	}


	//void pixelDiffThresh_(Mat& src, Mat& dest, T thresh, int direction=0)
	//{
	//	CV_Assert(src.channels()==1);
	//
	//	if(dest.empty())dest = Mat::zeros(src.size(),CV_8U);
	//	else dest.setTo(0);
	//
	//	if(direction>=0)
	//	{
	//		Mat im;copyMakeBorder(src,im,0,0,0,1,cv::BORDER_REPLICATE);
	//		for(int j=0;j<src.rows;j++)
	//		{
	//			uchar* s = im.ptr<uchar>(j);
	//			uchar* d = dest.ptr<uchar>(j);
	//			for(int i=0;i<src.cols;i++)
	//			{
	//				d[i] = (abs(s[i]-s[i+1])>thresh) ? 255:d[i] ;
	//			}
	//		}
	//	}
	//	if(direction<=0)
	//	{
	//		Mat im;copyMakeBorder(src,im,0,1,0,0,cv::BORDER_REPLICATE);
	//		for(int j=0;j<src.rows;j++)
	//		{
	//			uchar* s = im.ptr<uchar>(j);
	//			uchar* sv = im.ptr<uchar>(j+1);
	//			uchar* d = dest.ptr<uchar>(j);
	//			for(int i=0;i<src.cols;i++)
	//			{
	//				d[i] = (abs(s[i]-sv[i])>thresh) ? 255 :d[i];
	//			}
	//		}
	//	}
	//}
	//void pixelDiffABS(Mat& src, Mat& dest, int direction=0)
	//{
	//	CV_Assert(src.channels()==1);
	//	
	//	if(dest.empty())dest = Mat::zeros(src.size(),CV_8U);
	//	else dest.setTo(0);
	//
	//	if(direction>=0)
	//	{
	//		Mat im;copyMakeBorder(src,im,0,0,0,1,cv::BORDER_REPLICATE);
	//		for(int j=0;j<src.rows;j++)
	//		{
	//			uchar* s = im.ptr<uchar>(j);
	//			uchar* d = dest.ptr<uchar>(j);
	//			for(int i=0;i<src.cols;i++)
	//			{
	//				d[i] = (abs(s[i]-s[i+1]);
	//			}
	//		}
	//	}
	//	if(direction<=0)
	//	{
	//		Mat im;copyMakeBorder(src,im,0,1,0,0,cv::BORDER_REPLICATE);
	//		for(int j=0;j<src.rows;j++)
	//		{
	//			uchar* s = im.ptr<uchar>(j);
	//			uchar* sv = im.ptr<uchar>(j+1);
	//			uchar* d = dest.ptr<uchar>(j);
	//			for(int i=0;i<src.cols;i++)
	//			{
	//				d[i] = (abs(s[i]-sv[i]);
	//			}
	//		}
	//	}
	//}

}