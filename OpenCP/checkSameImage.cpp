#include "checkSameImage.hpp"

namespace cp
{
	bool CheckSameImage::checkSamplePoints(cv::Mat& src)
	{
		bool ret = true;
		cv::RNG rng(cv::getTickCount());
		for (int n = 0; n < (int)positions.size(); n++)
		{
			cv::Point pt = positions[n];
			switch (src.type())
			{
			case CV_8UC1:
				if (samples[n] != cv::Scalar(src.at<uchar>(pt)))
				{
					samples[n] = cv::Scalar(src.at<uchar>(pt));
					ret = false;
				}
				break;
			case CV_8UC3:
				if (samples[n] != cv::Scalar(src.at<cv::Vec3b>(pt)))
				{
					samples[n] = cv::Scalar(src.at<cv::Vec3b>(pt));
					ret = false;
				}
				break;
			case CV_16SC1:
				if (samples[n] != cv::Scalar(src.at<short>(pt)))
				{
					samples[n] = cv::Scalar(src.at<short>(pt));
					ret = false;
				}
				break;
			case CV_16SC3:
				if (samples[n] != cv::Scalar(src.at<cv::Vec3s>(pt)))
				{
					samples[n] = cv::Scalar(src.at<cv::Vec3s>(pt));
					ret = false;
				}
				break;
			case CV_16UC1:
				if (samples[n] != cv::Scalar(src.at<ushort>(pt)))
				{
					samples[n] = cv::Scalar(src.at<ushort>(pt));
					ret = false;
				}
				break;
			case CV_16UC3:
				if (samples[n] != cv::Scalar(src.at<cv::Vec3w>(pt)))
				{
					samples[n] = cv::Scalar(src.at<cv::Vec3w>(pt));
					ret = false;
				}
				break;
			case CV_32SC1:
				if (samples[n] != cv::Scalar(src.at<int>(pt)))
				{
					samples[n] = cv::Scalar(src.at<int>(pt));
					ret = false;
				}
				break;
			case CV_32SC3:
				if (samples[n] != cv::Scalar(src.at<cv::Vec3i>(pt)))
				{
					samples[n] = cv::Scalar(src.at<cv::Vec3i>(pt));
					ret = false;
				}
				break;
			case CV_32FC1:
				if (samples[n] != cv::Scalar(src.at<float>(pt)))
				{
					samples[n] = cv::Scalar(src.at<float>(pt));
					ret = false;
				}
				break;
			case CV_32FC3:
				if (samples[n] != cv::Scalar(src.at<cv::Vec3f>(pt)))
				{
					samples[n] = cv::Scalar(src.at<cv::Vec3f>(pt));
					ret = false;
				}
				break;
			case CV_64FC1:
				if (samples[n] != cv::Scalar(src.at<double>(pt)))
				{
					samples[n] = cv::Scalar(src.at<double>(pt));
					ret = false;
				}
				break;
			case CV_64FC3:
				if (samples[n] != cv::Scalar(src.at<cv::Vec3d>(pt)))
				{
					samples[n] = cv::Scalar(src.at<cv::Vec3d>(pt));
					ret = false;
				}
				break;

			default:
				std::cout << "not supported generateRandomSamplePoints" << std::endl;
				break;
			}
		}
		return ret;
	}

	void CheckSameImage::generateRandomSamplePoints(cv::Mat& src, const int num_check_points)
	{
		samples.resize(num_check_points);
		positions.resize(num_check_points);

		cv::RNG rng(cv::getTickCount());
		for (int n = 0; n < num_check_points; n++)
		{
			cv::Point pt(rng.uniform(0, src.cols - 1), rng.uniform(0, src.rows - 1));
			positions[n] = pt;
			switch (src.type())
			{
			case CV_8UC1: samples[n] = cv::Scalar(src.at<uchar>(pt)); break;
			case CV_8UC3: samples[n] = cv::Scalar(src.at<cv::Vec3b>(pt)); break;
			case CV_16SC1: samples[n] = cv::Scalar(src.at<short>(pt)); break;
			case CV_16SC3: samples[n] = cv::Scalar(src.at<cv::Vec3s>(pt)); break;
			case CV_16UC1: samples[n] = cv::Scalar(src.at<ushort>(pt)); break;
			case CV_16UC3: samples[n] = cv::Scalar(src.at<cv::Vec3w>(pt)); break;
			case CV_32SC1: samples[n] = cv::Scalar(src.at<int>(pt)); break;
			case CV_32SC3: samples[n] = cv::Scalar(src.at<cv::Vec3i>(pt)); break;
			case CV_32FC1: samples[n] = cv::Scalar(src.at<float>(pt)); break;
			case CV_32FC3: samples[n] = cv::Scalar(src.at<cv::Vec3f>(pt)); break;
			case CV_64FC1: samples[n] = cv::Scalar(src.at<double>(pt)); break;
			case CV_64FC3: samples[n] = cv::Scalar(src.at<cv::Vec3d>(pt)); break;

			default:
				std::cout << "not supported generateRandomSamplePoints" << std::endl;
				break;
			}
		}
	}

	bool CheckSameImage::isSameImage(cv::Mat& src, const int num_check_points)
	{
		CV_Assert(src.channels() == 1 || src.channels() == 3);
		bool ret = false;
		if (samples.size() != num_check_points)
		{
			generateRandomSamplePoints(src, num_check_points);
		}
		else
		{
			ret = checkSamplePoints(src);
		}

		return ret;
	}

	bool CheckSameImage::isSameImage(cv::Mat& src, cv::Mat& ref, const int num_check_points)
	{
		CV_Assert(src.channels() == 1 || src.channels() == 3);

		generateRandomSamplePoints(src, num_check_points);
		bool ret = checkSamplePoints(ref);

		return ret;
	}


	bool checkSameImage(cv::Mat& src, cv::Mat& ref, const int num_check_points)
	{
		CheckSameImage csi;
		return csi.isSameImage(src, ref, num_check_points);
	}
}