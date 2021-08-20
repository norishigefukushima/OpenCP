#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT CheckSameImage
	{
	private:
		bool isUsePrev = true;
		cv::Mat prev;
		std::vector<cv::Point> positions;
		std::vector<cv::Scalar> samples;

		bool checkSamplePoints(cv::Mat& src);
		void generateRandomSamplePoints(cv::Mat& src, const int num_check_points);
		bool isSameFull(cv::InputArray src, cv::InputArray ref);

	public:
		/// <summary>
		/// set flag for using previous buffer in isSame(cv::InputArray, const int)
		/// </summary>
		/// <param name="flag">flags</param>
		void setUsePrev(const bool flag);

		/// <summary>
		/// check same image with the previous called image
		/// </summary>
		/// <param name="src">src image</param>
		/// <param name="num_check_points">number of random samples. if <=0, check full samples</param>
		/// <returns>true: same, false: not same</returns>
		bool isSame(cv::InputArray src, const int num_check_points = 10);

		/// <summary>
		/// check same image with the previous called image
		/// </summary>
		/// <param name="src">src image</param>
		/// <param name="ref">reference image, the image is pushed.</param>
		/// <param name="num_check_points">number of random samples. if <=0, check full samples</param>
		/// <param name="isShowMessage">flags for show console message or not.</param>
		/// <param name="ok_mes">message if(true)</param>
		/// <param name="ng_mes">message if(false)</param>
		/// <returns>true: same, false: not same</returns>
		bool isSame(cv::InputArray src, cv::InputArray ref, const int num_check_points = 0, const bool isShowMessage = true, const std::string ok_mes = "OK", const std::string ng_mes = "NG");
	};

	/// <summary>
	/// wrapper function of CheckSameImage.isSame. check same image with the previous called image
	/// </summary>
	/// <param name="src">src image</param>
	/// <param name="ref">reference image, the image is pushed.</param>
	/// <param name="num_check_points">number of random samples. if <=0, check full samples</param>
	/// <param name="isShowMessage">flags for show console message or not.</param>
	/// <param name="ok_mes">message if(true)</param>
	/// <param name="ng_mes">message if(false)</param>
	/// <returns>true: same, false: not same</returns>
	CP_EXPORT bool isSame(cv::InputArray src, cv::InputArray ref, const int num_check_points = 0, const bool isShowMessage = true, const std::string ok_mes = "OK", const std::string ng_mes = "NG");
}