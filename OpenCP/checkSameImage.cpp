#include "checkSameImage.hpp"
#include "inlineSIMDFunctions.hpp"
#include "inlineCVFunctions.hpp"

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

	bool CheckSameImage::isSameFull(cv::InputArray src, cv::InputArray ref)
	{
		cv::Mat s = src.getMat();
		cv::Mat r = ref.getMat();
		int se = 32 / get_avx_element_size(s.depth());
		uchar* sptr = s.ptr<uchar>();
		uchar* rptr = r.ptr<uchar>();
		__m256i* ms = (__m256i*)sptr;
		__m256i* mr = (__m256i*)rptr;
		const int size = (int)s.total() * se;
		const int simdsize = get_simd_floor(size, 128);
		const int loopsize = simdsize / 128;
		int ret = 0;
		for (int i = 0; i < loopsize; i++)
		{
			ret |= _mm256_testc_si256(*ms, *mr); ms++; mr++;
			ret |= _mm256_testc_si256(*ms, *mr); ms++; mr++;
			ret |= _mm256_testc_si256(*ms, *mr); ms++; mr++;
			ret |= _mm256_testc_si256(*ms, *mr); ms++; mr++;
		}
		for (int i = simdsize; i < size; i++)
		{
			if (sptr[i] != rptr[i])ret = 0;
		}
		return ret;
	}

	void CheckSameImage::setUsePrev(const bool flag)
	{
		isUsePrev = flag;
	}

	bool CheckSameImage::isSame(cv::InputArray src, const int num_check_points)
	{
		CV_Assert(src.channels() == 1 || src.channels() == 3);
		cv::Mat s = src.getMat();
		bool ret = false;
		if (samples.size() != num_check_points)
		{

			generateRandomSamplePoints(s, num_check_points);
		}
		else
		{
			ret = checkSamplePoints(s);
		}

		return ret;
	}

	bool CheckSameImage::isSame(cv::InputArray src, cv::InputArray ref, const int num_check_points, const bool isShowMessage, const std::string ok_mes, const std::string ng_mes)
	{
		CV_Assert(src.channels() == 1 || src.channels() == 3);

		if (src.size() != ref.size())
		{
			if (isShowMessage)std::cout << "not same size. src: " << src.size() << ", answer: " << ref.size() << std::endl;
			return false;
		}

		if (src.depth() != ref.depth())
		{
			if (isShowMessage)std::cout << "not same depth. src: " << getDepthName(src.depth()) << ", answer: " << getDepthName(ref.depth()) << std::endl;
			return false;
		}

		bool ret = false;
		if (num_check_points <= 0)
		{
			if (isUsePrev)ref.copyTo(prev);
			ret = isSameFull(src, ref);
		}
		else
		{
			cv::Mat r = ref.getMat();
			cv::Mat s = src.getMat();
			generateRandomSamplePoints(r, num_check_points);
			ret = checkSamplePoints(s);
		}

		std::string m;
		if (ret)
		{
			m = ok_mes;
		}
		else
		{
			m = ng_mes;
		}

		if (isShowMessage) std::cout << m << std::endl;

		return ret;
	}

	bool isSame(cv::InputArray src, cv::InputArray ref, const int num_check_points, const bool isShowMessage, const std::string ok_mes, const std::string ng_mes)
	{
		CheckSameImage csi;
		csi.setUsePrev(false);
		return csi.isSame(src, ref, num_check_points, isShowMessage, ok_mes, ng_mes);
	}
}