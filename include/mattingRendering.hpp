#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT dispRefinement
	{
	private:

	public:

		int r;
		int th;
		int iter_ex;
		int th_r;
		int r_flip;
		int iter;
		int iter_g;
		int r_g;
		int eps_g;
		int th_FB;

		dispRefinement();
		void boundaryDetect(cv::Mat& src, cv::Mat& guid, cv::Mat& dest, cv::Mat& mask);
		void dispRefine(cv::Mat& src, cv::Mat& guid, cv::Mat& guid_mask, cv::Mat& alpha);
		void operator()(cv::Mat& src, cv::Mat& guid, cv::Mat& dest);
	};

	class CP_EXPORT mattingMethod
	{
	private:
		cv::Mat trimap;
		cv::Mat trimask;
		cv::Mat f;
		cv::Mat b;
		cv::Mat a;

	public:

		int r;
		int iter;
		int iter_g;
		int r_g;
		int eps_g;
		int th_FB;
		int r_Wgauss;
		int sigma_Wgauss;
		int th;

		mattingMethod();
		void boundaryDetect(cv::Mat& disp);
		void getAmap(cv::Mat& img);
		void getFBimg(cv::Mat& img);
		void operator()(cv::Mat& img, cv::Mat& disp, cv::Mat& alpha, cv::Mat& Fimg, cv::Mat& Bimg);
	};
}