#include <opencv2/opencv.hpp>

namespace cp
{
	void set1DSpaceKernel45(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle);
	void set1DSpaceKernel135(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle);
	void setSpaceKernel(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle);

	void set1DSpaceKernel135(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle);
	void set1DSpaceKernel45(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle);
	void setSpaceKernel(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle);

	void jointBilateralFilter_direction_8u(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, cv::Size kernelSize, double sigma_color, double sigma_space, int borderType, int direction, bool isRectangle);
}