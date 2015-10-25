#include "opencp.hpp"

using namespace cv;

namespace cp
{

	dispRefinement::dispRefinement()
	{
		r = 4;
		th = 10;
		iter_ex = 3;
		th_r = 50;
		r_flip = 1;
		iter = 3;
		iter_g = 2;
		r_g = 3;
		eps_g = 2;
		th_FB = 85;
	}

	void dispRefinement::boundaryDetect(Mat& src, Mat& guid, Mat& dest, Mat& mask)
	{
		int d = 2 * r + 1;
		Size kernel = Size(d, d);

		Mat trimap = Mat::zeros(src.size(), CV_8U);
		Mat trimask = Mat::zeros(src.size(), CV_8U);

		Mat dmax, dmin;
		maxFilter(src, dmax, kernel);
		minFilter(src, dmin, kernel);

		for (int i = 0; i < src.size().area(); i++)
		{
			uchar maxdiff = dmax.data[i] - src.data[i];
			uchar mindiff = src.data[i] - dmin.data[i];
			if (dmax.data[i] - dmin.data[i] < th)
			{
				// non boudnary region
				trimap.data[i] = 128;
				trimask.data[i] = 255;
			}
			else // boundary region
			{
				if (abs(src.data[i] - dmax.data[i]) < abs(src.data[i] - dmin.data[i]))
				{
					trimap.data[i] = 255; // foreground region
				}
				else
				{
					trimap.data[i] = 0; // background region
				}
			}
		}

		Mat trimax, trimin;
		for (int i = 0; i < iter_ex; i++)
		{
			maxFilter(trimap, trimax, Size(3, 3));
			compare(trimap, 0, mask, CMP_NE);
			trimax.copyTo(trimap, mask);

			minFilter(trimap, trimin, Size(3, 3));
			compare(trimap, 255, mask, CMP_NE);
			trimin.copyTo(trimap, mask);
		}

		trimap.copyTo(dest);
		trimask.copyTo(mask);
	}

	void dispRefinement::dispRefine(Mat& src, Mat& guid, Mat& guid_mask, Mat& alpha)
	{
		Mat guid_; guid.copyTo(guid_);
		Mat tmp;
		compare(alpha, 255.0*((double)th_r / 100.0), tmp, CMP_GT);
		guid_.setTo(255, tmp);
		compare(alpha, 255.0*((double)th_r / 100.0), tmp, CMP_LT);
		guid_.setTo(0, tmp);
		guid_.setTo(128, guid_mask);

		int d = 2 * r_flip + 1;
		Size kernel = Size(d, d);

		Mat dmax, dmin;
		maxFilter(src, dmax, kernel);
		minFilter(src, dmin, kernel);

		for (int i = 0; i < guid.size().area(); i++)
		{
			if (guid.data[i] != 128)
			{
				if (guid.data[i] == 0 && guid_.data[i] == 255)
				{
					src.data[i] = dmax.data[i];
				}
				else if (guid.data[i] == 255 && guid_.data[i] == 0)
				{
					src.data[i] = dmin.data[i];
				}
			}
		}
	}

	void dispRefinement::operator()(Mat& src, Mat& guid, Mat& dest)
	{
		Mat s; src.copyTo(s);
		Mat imgG; cv::cvtColor(guid, imgG, CV_BGR2GRAY);
		Mat ref; cv::cvtColor(guid, ref, CV_BGR2GRAY);

		for (int j = 0; j < iter; j++)
		{
			Mat bmap = Mat::zeros(src.size(), CV_8U);
			Mat bmask = Mat::zeros(src.size(), CV_8U);

			boundaryDetect(s, guid, bmap, bmask);

			Mat alpha; bmap.copyTo(alpha); bmap.setTo(128, bmask);

			Mat temp;
			for (int i = 0; i < iter_g; i++)
			{
				guidedFilter(alpha, ref, temp, r_g, eps_g / 100.0f);
				temp.copyTo(alpha);
				alpha.setTo(128, bmask);
			}
			Mat maskf, maskb;
			compare(alpha, 255.0*((double)th_FB / 100.0), maskf, CMP_GT);
			alpha.setTo(255, maskf);
			compare(alpha, 255.0*(1.0 - th_FB / 100.0), maskb, CMP_LT);
			alpha.setTo(0, maskb);
			dispRefine(s, bmap, bmask, alpha);
		}
		s.copyTo(dest);
	}

	mattingMethod::mattingMethod()
	{
		r = 3;
		th = 10;
		iter = 6;
		r_g = 3;
		eps_g = 2;
		iter_g = 1;
		th_FB = 85;
		r_Wgauss = 3;
		sigma_Wgauss = 9;
	}

	void mattingMethod::boundaryDetect(Mat& disp)
	{
		int d = 2 * r + 1;
		Size kernel = Size(d, d);

		Mat dmax;
		Mat dmin;
		maxFilter(disp, dmax, kernel);
		minFilter(disp, dmin, kernel);

		for (int i = 0; i < disp.size().area(); i++)
		{
			uchar maxdiff = dmax.data[i] - disp.data[i];
			uchar mindiff = disp.data[i] - dmin.data[i];

			if (dmax.data[i] - dmin.data[i] < th)
			{
				trimap.data[i] = 128;
				trimask.data[i] = 255;
			}
			else
			{
				if (abs(disp.data[i] - dmax.data[i]) < abs(disp.data[i] - dmin.data[i]))
				{
					trimap.data[i] = 255;
				}
				else
				{
					trimap.data[i] = 0;
				}
			}
		}

	}

	void mattingMethod::getAmap(Mat& img)
	{
		Mat mask;
		Mat trimax;
		Mat trimin;
		for (int i = 0; i < iter; i++)
		{
			maxFilter(trimap, trimax, Size(3, 3));
			compare(trimap, 0, mask, CMP_NE);
			trimax.copyTo(trimap, mask);

			minFilter(trimap, trimin, Size(3, 3));
			compare(trimap, 255, mask, CMP_NE);
			trimin.copyTo(trimap, mask);
		}

		Mat tmp;
		Mat imgG; cvtColor(img, imgG, CV_BGR2GRAY);
		for (int i = 0; i < iter_g; i++)
		{
			guidedFilter(trimap, imgG, tmp, r_g, eps_g / 100.0f);
			tmp.copyTo(trimap);
		}

	}

	void mattingMethod::getFBimg(Mat& img)
	{
		Mat tmp;
		Mat fmask;
		Mat bmask;
		Mat fimg;
		Mat bimg;

		GaussianBlur(trimap, tmp, Size(1, 1), 1.5);

		compare(tmp, 255 * ((double)th_FB / 100.0), fmask, CMP_GT);
		trimap.setTo(255, fmask);
		compare(tmp, 255.0*(1.0 - th_FB / 100.0), bmask, CMP_LT);
		trimap.setTo(0, bmask);

		int d = 2 * r_Wgauss + 1;
		weightedGaussianFilter(img, fmask, fimg, Size(d, d), (float)sigma_Wgauss);
		weightedGaussianFilter(img, bmask, bimg, Size(d, d), (float)sigma_Wgauss);

		Mat gb, gf;
		cvtColor(bimg, gb, CV_BGR2GRAY);
		cvtColor(fimg, gf, CV_BGR2GRAY);
		Mat fmask2, bmask2;
		compare(gb, 0, bmask2, CMP_NE);
		compare(gf, 0, fmask2, CMP_NE);
		bitwise_and(bmask2, fmask2, tmp);

		img.copyTo(bimg, ~tmp);
		img.copyTo(fimg, ~tmp);
		bitwise_or(fmask, bmask, tmp);
		img.copyTo(fimg, tmp);
		img.copyTo(bimg, tmp);

		fimg.copyTo(f, ~trimask);
		bimg.copyTo(b);
		trimap.setTo(128, trimask);
		trimap.copyTo(a, ~trimask);

	}

	void mattingMethod::operator()(Mat& img, Mat& disp, Mat& alpha, Mat& Fimg, Mat& Bimg)
	{
		trimap = Mat::zeros(img.size(), CV_8U);
		trimask = Mat::zeros(img.size(), CV_8U);

		f = Mat::zeros(img.size(), CV_8UC3);
		b = Mat::zeros(img.size(), CV_8UC3);
		a = Mat::zeros(img.size(), CV_8U);

		boundaryDetect(disp);

		getAmap(img);
		getFBimg(img);

		f.copyTo(Fimg);
		b.copyTo(Bimg);
		a.copyTo(alpha);
	}
}