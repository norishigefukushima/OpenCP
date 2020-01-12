#include "guidedFilter_Merge_OnePath_Fast.h"

using namespace std;
using namespace cv;

#define CALC_COVARIANCE()	\
	meanI_b = sumMeanI_b * div;	\
	meanI_g = sumMeanI_g * div;	\
	meanI_r = sumMeanI_r * div;	\
	meanP = sumMeanP * div;	\
	corrI_bb = sumCorrI_bb * div;	\
	corrI_bg = sumCorrI_bg * div;	\
	corrI_br = sumCorrI_br * div;	\
	corrI_gg = sumCorrI_gg * div;	\
	corrI_gr = sumCorrI_gr * div;	\
	corrI_rr = sumCorrI_rr * div;	\
	covIP_b = sumCovIP_b * div;	\
	covIP_g = sumCovIP_g * div;	\
	covIP_r = sumCovIP_r * div;	\
	\
	bb = corrI_bb - meanI_b * meanI_b;	\
	bg = corrI_bg - meanI_b * meanI_g;	\
	br = corrI_br - meanI_b * meanI_r;	\
	gg = corrI_gg - meanI_g * meanI_g;	\
	gr = corrI_gr - meanI_g * meanI_r;	\
	rr = corrI_rr - meanI_r * meanI_r;	\
	covb = covIP_b - meanI_b * meanP;	\
	covg = covIP_g - meanI_g * meanP;	\
	covr = covIP_r - meanI_r * meanP;	\
	\
	bb += eps;	\
	gg += eps;	\
	rr += eps;	\
	\
	det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);	\
	id = 1.f / det;	\
	\
	c0 = gg*rr - gr*gr;	\
	c1 = br*gr - bg*rr;	\
	c2 = bg*gr - br*gg;	\
	c4 = bb*rr - br*br;	\
	c5 = br*bg - bb*gr;	\
	c8 = bb*gg - bg*bg;	\
	\
	*ab_ptr = id * (covb*c0 + covg*c1 + covr*c2);	\
	*ag_ptr = id * (covb*c1 + covg*c4 + covr*c5);	\
	*ar_ptr = id * (covb*c2 + covg*c5 + covr*c8);	\
	*b_ptr = meanP - (*ab_ptr * meanI_b + *ag_ptr * meanI_g + *ar_ptr * meanI_r);


guidedFilter_Merge_OnePath_Fast::guidedFilter_Merge_OnePath_Fast(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
	: guidedFilter_Merge_OnePass(_src, _guide, _dest, _r, _eps, _parallelType)
{
	resize(src, src_resize, Size(), 0.5, 0.5);
	resize(guide, guide_resize, Size(), 0.5, 0.5);

	r = _r / 2;
	div = 1.f / ((2 * r + 1) * (2 * r + 1));

	row_resize = src_resize.rows;
	col_resize = src_resize.cols;
}

void guidedFilter_Merge_OnePath_Fast::filter_Guide1(int cn)
{
	Mat a(src_resize.size(), CV_32FC1);
	Mat b(src_resize.size(), CV_32FC1);
	Mat mean_a(src_resize.size(), CV_32FC1);
	Mat mean_b(src_resize.size(), CV_32FC1);

	// I'p' 2 a'b'
	{
		float sumMeanI = 0.f;
		float sumMeanP = 0.f;
		float sumCorrI = 0.f;
		float sumCorrIP = 0.f;

		float meanI = 0.f;
		float meanP = 0.f;
		float corrI = 0.f;
		float corrIP = 0.f;

		Mat columnSum = Mat::zeros(1, col_resize, CV_32FC4);
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = cp_prev;

		float* I_ptr_next = guide_resize.ptr<float>(0);
		float* p_ptr_next = src_resize.ptr<float>(0) + cn;
		for (int j = 0; j < col_resize; j++)
		{
			*(cp_next++) = *I_ptr_next * (r + 1);
			*(cp_next++) = (*I_ptr_next * *I_ptr_next) * (r + 1);
			*(cp_next++) = (*I_ptr_next * *p_ptr_next) * (r + 1);
			I_ptr_next++;

			*(cp_next++) = *p_ptr_next * (r + 1);
			p_ptr_next += p_cn;
		}
		cp_next = cp_prev;
		for (int i = 1; i < r; i++)
		{
			for (int j = 0; j < col_resize; j++)
			{
				*(cp_next++) += *I_ptr_next;
				*(cp_next++) += (*I_ptr_next * *I_ptr_next);
				*(cp_next++) += (*I_ptr_next * *p_ptr_next);
				I_ptr_next++;

				*(cp_next++) += *p_ptr_next;
				p_ptr_next += p_cn;
			}
			cp_next = cp_prev;
		}

		*cp_next += *I_ptr_next;
		sumMeanI += *(cp_next++) * (r + 1);

		*cp_next += (*I_ptr_next * *I_ptr_next);
		sumCorrI += *(cp_next++) * (r + 1);

		*cp_next += (*I_ptr_next * *p_ptr_next);
		sumCorrIP += *(cp_next++) * (r + 1);
		I_ptr_next++;

		*cp_next += *p_ptr_next;
		sumMeanP += *(cp_next++) * (r + 1);
		p_ptr_next += p_cn;
		for (int j = 1; j <= r; j++)
		{
			*cp_next += *I_ptr_next;
			sumMeanI += *(cp_next++);

			*cp_next += (*I_ptr_next * *I_ptr_next);
			sumCorrI += *(cp_next++);

			*cp_next += (*I_ptr_next * *p_ptr_next);
			sumCorrIP += *(cp_next++);
			I_ptr_next++;

			*cp_next += *p_ptr_next;
			sumMeanP += *(cp_next++);
			p_ptr_next += p_cn;
		}
		meanI = sumMeanI * div;
		corrI = sumCorrI * div;
		corrIP = sumCorrIP * div;
		meanP = sumMeanP * div;

		float varI = corrI - meanI * meanI;
		float covIP = corrIP - meanI * meanP;

		float* a_ptr = a.ptr<float>(0);
		float* b_ptr = b.ptr<float>(0);
		*a_ptr = covIP / (varI + eps);
		*b_ptr = meanP - *a_ptr * meanI;
		a_ptr++;
		b_ptr++;

		for (int j = 1; j <= r; j++)
		{
			*cp_next += *I_ptr_next;
			sumMeanI = sumMeanI - *cp_prev + *(cp_next++);

			*cp_next += (*I_ptr_next * *I_ptr_next);
			sumCorrI = sumCorrI - *(cp_prev + 1) + *(cp_next++);

			*cp_next += (*I_ptr_next * *p_ptr_next);
			sumCorrIP = sumCorrIP - *(cp_prev + 2) + *(cp_next++);
			I_ptr_next++;

			*cp_next += *p_ptr_next;
			sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
			p_ptr_next += p_cn;

			meanI = sumMeanI * div;
			corrI = sumCorrI * div;
			corrIP = sumCorrIP * div;
			meanP = sumMeanP * div;

			varI = corrI - meanI * meanI;
			covIP = corrIP - meanI * meanP;
			*a_ptr = covIP / (varI + eps);
			*b_ptr = meanP - *a_ptr * meanI;
			a_ptr++;
			b_ptr++;
		}
		for (int j = r + 1; j < col_resize - r; j++)
		{
			*cp_next += *I_ptr_next;
			sumMeanI = sumMeanI - *(cp_prev++) + *(cp_next++);

			*cp_next += (*I_ptr_next * *I_ptr_next);
			sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next++);

			*cp_next += (*I_ptr_next * *p_ptr_next);
			sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next++);
			I_ptr_next++;

			*cp_next += *p_ptr_next;
			sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
			p_ptr_next += p_cn;

			meanI = sumMeanI * div;
			corrI = sumCorrI * div;
			corrIP = sumCorrIP * div;
			meanP = sumMeanP * div;

			varI = corrI - meanI * meanI;
			covIP = corrIP - meanI * meanP;
			*a_ptr = covIP / (varI + eps);
			*b_ptr = meanP - *a_ptr * meanI;
			a_ptr++;
			b_ptr++;
		}
		cp_next -= columnSum.channels();
		for (int j = col_resize - r; j < col_resize; j++)
		{
			sumMeanI = sumMeanI - *(cp_prev++) + *cp_next;
			sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next + 1);
			sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next + 2);
			sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);

			meanI = sumMeanI * div;
			corrI = sumCorrI * div;
			corrIP = sumCorrIP * div;
			meanP = sumMeanP * div;

			varI = corrI - meanI * meanI;
			covIP = corrIP - meanI * meanP;
			*a_ptr = covIP / (varI + eps);
			*b_ptr = meanP - *a_ptr * meanI;
			a_ptr++;
			b_ptr++;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);

		float* I_ptr_prev = guide_resize.ptr<float>(0);
		float* p_ptr_prev = src_resize.ptr<float>(0) + cn;
		/*   0 < i <= r   */
		for (int i = 1; i <= r; i++)
		{
			sumMeanI = 0.f;
			sumCorrI = 0.f;
			sumCorrIP = 0.f;
			sumMeanP = 0.f;

			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
			sumMeanI += *(cp_next++) * (r + 1);

			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
			sumCorrI += *(cp_next++) * (r + 1);

			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
			sumCorrIP += *(cp_next++) * (r + 1);
			I_ptr_prev++;
			I_ptr_next++;

			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
			sumMeanP += *(cp_next++) * (r + 1);
			p_ptr_prev += p_cn;
			p_ptr_next += p_cn;
			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
				sumMeanI += *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
				sumCorrI += *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
				sumCorrIP += *(cp_next++);
				I_ptr_prev++;
				I_ptr_next++;

				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP += *(cp_next++);
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
			}
			meanI = sumMeanI * div;
			corrI = sumCorrI * div;
			corrIP = sumCorrIP * div;
			meanP = sumMeanP * div;

			varI = corrI - meanI * meanI;
			covIP = corrIP - meanI * meanP;
			*a_ptr = covIP / (varI + eps);
			*b_ptr = meanP - *a_ptr * meanI;
			a_ptr++;
			b_ptr++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
				sumMeanI = sumMeanI - *cp_prev + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
				sumCorrI = sumCorrI - *(cp_prev + 1) + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
				sumCorrIP = sumCorrIP - *(cp_prev + 2) + *(cp_next++);
				I_ptr_prev++;
				I_ptr_next++;

				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				meanI = sumMeanI * div;
				corrI = sumCorrI * div;
				corrIP = sumCorrIP * div;
				meanP = sumMeanP * div;

				varI = corrI - meanI * meanI;
				covIP = corrIP - meanI * meanP;
				*a_ptr = covIP / (varI + eps);
				*b_ptr = meanP - *a_ptr * meanI;
				a_ptr++;
				b_ptr++;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
				sumMeanI = sumMeanI - *(cp_prev++) + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
				sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
				sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next++);
				I_ptr_prev++;
				I_ptr_next++;

				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				meanI = sumMeanI * div;
				corrI = sumCorrI * div;
				corrIP = sumCorrIP * div;
				meanP = sumMeanP * div;

				varI = corrI - meanI * meanI;
				covIP = corrIP - meanI * meanP;
				*a_ptr = covIP / (varI + eps);
				*b_ptr = meanP - *a_ptr * meanI;
				a_ptr++;
				b_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanI = sumMeanI - *(cp_prev++) + *cp_next;
				sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next + 1);
				sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next + 2);
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);

				meanI = sumMeanI * div;
				corrI = sumCorrI * div;
				corrIP = sumCorrIP * div;
				meanP = sumMeanP * div;

				varI = corrI - meanI * meanI;
				covIP = corrIP - meanI * meanP;
				*a_ptr = covIP / (varI + eps);
				*b_ptr = meanP - *a_ptr * meanI;
				a_ptr++;
				b_ptr++;
			}
			I_ptr_prev = guide_resize.ptr<float>(0);
			p_ptr_prev = src_resize.ptr<float>(0) + cn;
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row_resize - r - 1   */
		for (int i = r + 1; i < row_resize - r - 1; i++)
		{
			sumMeanI = 0.f;
			sumCorrI = 0.f;
			sumCorrIP = 0.f;
			sumMeanP = 0.f;

			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
			sumMeanI += *(cp_next++) * (r + 1);

			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
			sumCorrI += *(cp_next++) * (r + 1);

			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
			sumCorrIP += *(cp_next++) * (r + 1);
			I_ptr_prev++;
			I_ptr_next++;

			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
			sumMeanP += *(cp_next++) * (r + 1);
			p_ptr_prev += p_cn;
			p_ptr_next += p_cn;
			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
				sumMeanI += *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
				sumCorrI += *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
				sumCorrIP += *(cp_next++);
				I_ptr_prev++;
				I_ptr_next++;

				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP += *(cp_next++);
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
			}
			meanI = sumMeanI * div;
			corrI = sumCorrI * div;
			corrIP = sumCorrIP * div;
			meanP = sumMeanP * div;

			varI = corrI - meanI * meanI;
			covIP = corrIP - meanI * meanP;
			*a_ptr = covIP / (varI + eps);
			*b_ptr = meanP - *a_ptr * meanI;
			a_ptr++;
			b_ptr++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
				sumMeanI = sumMeanI - *cp_prev + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
				sumCorrI = sumCorrI - *(cp_prev + 1) + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
				sumCorrIP = sumCorrIP - *(cp_prev + 2) + *(cp_next++);
				I_ptr_prev++;
				I_ptr_next++;

				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				meanI = sumMeanI * div;
				corrI = sumCorrI * div;
				corrIP = sumCorrIP * div;
				meanP = sumMeanP * div;

				varI = corrI - meanI * meanI;
				covIP = corrIP - meanI * meanP;
				*a_ptr = covIP / (varI + eps);
				*b_ptr = meanP - *a_ptr * meanI;
				a_ptr++;
				b_ptr++;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
				sumMeanI = sumMeanI - *(cp_prev++) + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
				sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
				sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next++);
				I_ptr_prev++;
				I_ptr_next++;

				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				meanI = sumMeanI * div;
				corrI = sumCorrI * div;
				corrIP = sumCorrIP * div;
				meanP = sumMeanP * div;

				varI = corrI - meanI * meanI;
				covIP = corrIP - meanI * meanP;
				*a_ptr = covIP / (varI + eps);
				*b_ptr = meanP - *a_ptr * meanI;
				a_ptr++;
				b_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanI = sumMeanI - *(cp_prev++) + *cp_next;
				sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next + 1);
				sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next + 2);
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);

				meanI = sumMeanI * div;
				corrI = sumCorrI * div;
				corrIP = sumCorrIP * div;
				meanP = sumMeanP * div;

				varI = corrI - meanI * meanI;
				covIP = corrIP - meanI * meanP;
				*a_ptr = covIP / (varI + eps);
				*b_ptr = meanP - *a_ptr * meanI;
				a_ptr++;
				b_ptr++;
			}
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   row_resize - r - 1 <= i < row_resize   */
		for (int i = row_resize - r - 1; i < row_resize; i++)
		{
			sumMeanI = 0.f;
			sumCorrI = 0.f;
			sumCorrIP = 0.f;
			sumMeanP = 0.f;

			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
			sumMeanI += *(cp_next++) * (r + 1);

			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
			sumCorrI += *(cp_next++) * (r + 1);

			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
			sumCorrIP += *(cp_next++) * (r + 1);
			I_ptr_prev++;
			I_ptr_next++;

			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
			sumMeanP += *(cp_next++) * (r + 1);
			p_ptr_prev += p_cn;
			p_ptr_next += p_cn;
			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
				sumMeanI += *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
				sumCorrI += *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
				sumCorrIP += *(cp_next++);
				I_ptr_prev++;
				I_ptr_next++;

				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP += *(cp_next++);
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
			}
			meanI = sumMeanI * div;
			corrI = sumCorrI * div;
			corrIP = sumCorrIP * div;
			meanP = sumMeanP * div;

			varI = corrI - meanI * meanI;
			covIP = corrIP - meanI * meanP;
			*a_ptr = covIP / (varI + eps);
			*b_ptr = meanP - *a_ptr * meanI;
			a_ptr++;
			b_ptr++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
				sumMeanI = sumMeanI - *cp_prev + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
				sumCorrI = sumCorrI - *(cp_prev + 1) + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
				sumCorrIP = sumCorrIP - *(cp_prev + 2) + *(cp_next++);
				I_ptr_prev++;
				I_ptr_next++;

				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				meanI = sumMeanI * div;
				corrI = sumCorrI * div;
				corrIP = sumCorrIP * div;
				meanP = sumMeanP * div;

				varI = corrI - meanI * meanI;
				covIP = corrIP - meanI * meanP;
				*a_ptr = covIP / (varI + eps);
				*b_ptr = meanP - *a_ptr * meanI;
				a_ptr++;
				b_ptr++;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
				sumMeanI = sumMeanI - *(cp_prev++) + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
				sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next++);

				*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
				sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next++);
				I_ptr_prev++;
				I_ptr_next++;

				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				meanI = sumMeanI * div;
				corrI = sumCorrI * div;
				corrIP = sumCorrIP * div;
				meanP = sumMeanP * div;

				varI = corrI - meanI * meanI;
				covIP = corrIP - meanI * meanP;
				*a_ptr = covIP / (varI + eps);
				*b_ptr = meanP - *a_ptr * meanI;
				a_ptr++;
				b_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanI = sumMeanI - *(cp_prev++) + *cp_next;
				sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next + 1);
				sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next + 2);
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);

				meanI = sumMeanI * div;
				corrI = sumCorrI * div;
				corrIP = sumCorrIP * div;
				meanP = sumMeanP * div;

				varI = corrI - meanI * meanI;
				covIP = corrIP - meanI * meanP;
				*a_ptr = covIP / (varI + eps);
				*b_ptr = meanP - *a_ptr * meanI;
				a_ptr++;
				b_ptr++;
			}
			I_ptr_next = guide_resize.ptr<float>(row_resize - 1);
			p_ptr_next = src_resize.ptr<float>(row_resize - 1) + cn;
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}

	// a'b' 2 mean_a' mean_b'
	{
		float sumMeanA = 0.f;
		float sumMeanB = 0.f;

		Mat columnSum = Mat::zeros(Size(col_resize, 1), CV_32FC2);
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = cp_prev;

		float* a_ptr_next = a.ptr<float>(0);
		float* b_ptr_next = b.ptr<float>(0);
		for (int j = 0; j < col_resize; j++)
		{
			*(cp_next++) = *a_ptr_next * (r + 1);
			a_ptr_next++;

			*(cp_next++) = *b_ptr_next * (r + 1);
			b_ptr_next++;
		}
		cp_next = cp_prev;
		for (int i = 1; i < r; i++)
		{
			for (int j = 0; j < col_resize; j++)
			{
				*(cp_next++) += *a_ptr_next;
				a_ptr_next++;

				*(cp_next++) += *b_ptr_next;
				b_ptr_next++;
			}
			cp_next = cp_prev;
		}

		*cp_next += *a_ptr_next;
		sumMeanA += *(cp_next++) * (r + 1);
		a_ptr_next++;

		*cp_next += *b_ptr_next;
		sumMeanB += *(cp_next++) * (r + 1);
		b_ptr_next++;

		for (int j = 1; j <= r; j++)
		{
			*cp_next += *a_ptr_next;
			sumMeanA += *(cp_next++);
			a_ptr_next++;

			*cp_next += *b_ptr_next;
			sumMeanB += *(cp_next++);
			b_ptr_next++;
		}
		float* meanA_ptr = mean_a.ptr<float>(0);
		float* meanB_ptr = mean_b.ptr<float>(0);
		*meanA_ptr = sumMeanA * div;
		*meanB_ptr = sumMeanB * div;
		meanA_ptr++;
		meanB_ptr++;

		for (int j = 1; j <= r; j++)
		{
			*cp_next += *a_ptr_next;
			sumMeanA = sumMeanA - *cp_prev + *(cp_next++);
			a_ptr_next++;

			*cp_next += *b_ptr_next;
			sumMeanB = sumMeanB - *(cp_prev + 1) + *(cp_next++);
			b_ptr_next++;

			*meanA_ptr = sumMeanA * div;
			*meanB_ptr = sumMeanB * div;
			meanA_ptr++;
			meanB_ptr++;
		}
		for (int j = r + 1; j < col_resize - r; j++)
		{
			*cp_next += *a_ptr_next;
			sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
			a_ptr_next++;

			*cp_next += *b_ptr_next;
			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
			b_ptr_next++;

			*meanA_ptr = sumMeanA * div;
			*meanB_ptr = sumMeanB * div;
			meanA_ptr++;
			meanB_ptr++;
		}
		cp_next -= columnSum.channels();
		for (int j = col_resize - r; j < col_resize; j++)
		{
			sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

			*meanA_ptr = sumMeanA * div;
			*meanB_ptr = sumMeanB * div;
			meanA_ptr++;
			meanB_ptr++;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);

		float* a_ptr_prev = a.ptr<float>(0);
		float* b_ptr_prev = b.ptr<float>(0);
		/*   0 < i <= r   */
		for (int i = 1; i <= r; i++)
		{
			sumMeanA = 0.f;
			sumMeanB = 0.f;

			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
			sumMeanA += *(cp_next++) * (r + 1);
			a_ptr_prev++;
			a_ptr_next++;

			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
			sumMeanB += *(cp_next++) * (r + 1);
			b_ptr_prev++;
			b_ptr_next++;
			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA += *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB += *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;
			}
			*meanA_ptr = sumMeanA * div;
			*meanB_ptr = sumMeanB * div;
			meanA_ptr++;
			meanB_ptr++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *cp_prev + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev + 1) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				*meanA_ptr = sumMeanA * div;
				*meanB_ptr = sumMeanB * div;
				meanA_ptr++;
				meanB_ptr++;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				*meanA_ptr = sumMeanA * div;
				*meanB_ptr = sumMeanB * div;
				meanA_ptr++;
				meanB_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

				*meanA_ptr = sumMeanA * div;
				*meanB_ptr = sumMeanB * div;
				meanA_ptr++;
				meanB_ptr++;
			}
			a_ptr_prev = a.ptr<float>(0);
			b_ptr_prev = b.ptr<float>(0);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row_resize - r - 1   */
		for (int i = r + 1; i < row_resize - r - 1; i++)
		{
			sumMeanA = 0.f;
			sumMeanB = 0.f;

			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
			sumMeanA += *(cp_next++) * (r + 1);
			a_ptr_prev++;
			a_ptr_next++;

			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
			sumMeanB += *(cp_next++) * (r + 1);
			b_ptr_prev++;
			b_ptr_next++;
			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA += *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB += *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;
			}
			*meanA_ptr = sumMeanA * div;
			*meanB_ptr = sumMeanB * div;
			meanA_ptr++;
			meanB_ptr++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *cp_prev + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev + 1) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				*meanA_ptr = sumMeanA * div;
				*meanB_ptr = sumMeanB * div;
				meanA_ptr++;
				meanB_ptr++;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				*meanA_ptr = sumMeanA * div;
				*meanB_ptr = sumMeanB * div;
				meanA_ptr++;
				meanB_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

				*meanA_ptr = sumMeanA * div;
				*meanB_ptr = sumMeanB * div;
				meanA_ptr++;
				meanB_ptr++;
			}
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   row_resize - r - 1 < i < row_resize   */
		for (int i = row_resize - r - 1; i < row_resize; i++)
		{
			sumMeanA = 0.f;
			sumMeanB = 0.f;

			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
			sumMeanA += *(cp_next++) * (r + 1);
			a_ptr_prev++;
			a_ptr_next++;

			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
			sumMeanB += *(cp_next++) * (r + 1);
			b_ptr_prev++;
			b_ptr_next++;
			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA += *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB += *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;
			}
			*meanA_ptr = sumMeanA * div;
			*meanB_ptr = sumMeanB * div;
			meanA_ptr++;
			meanB_ptr++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *cp_prev + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev + 1) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				*meanA_ptr = sumMeanA * div;
				*meanB_ptr = sumMeanB * div;
				meanA_ptr++;
				meanB_ptr++;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				*meanA_ptr = sumMeanA * div;
				*meanB_ptr = sumMeanB * div;
				meanA_ptr++;
				meanB_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

				*meanA_ptr = sumMeanA * div;
				*meanB_ptr = sumMeanB * div;
				meanA_ptr++;
				meanB_ptr++;
			}
			a_ptr_next = a.ptr<float>(row_resize - 1);
			b_ptr_next = b.ptr<float>(row_resize - 1);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}

	// resize
	resize(mean_a, mean_a, Size(), 2.0, 2.0);
	resize(mean_b, mean_b, Size(), 2.0, 2.0);

	// mean_a mean_b 2 q
	{
		float* a_ptr = mean_a.ptr<float>(0);
		float* b_ptr = mean_b.ptr<float>(0);
		float* I_ptr = guide.ptr<float>(0);
		float* q_ptr = dest.ptr<float>(0) + cn;

		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				*q_ptr = *a_ptr * *I_ptr + *b_ptr;
				a_ptr++;
				b_ptr++;
				I_ptr++;
				q_ptr += p_cn;
			}
		}
	}
}

void guidedFilter_Merge_OnePath_Fast::filter_Guide3(int cn)
{
	Mat a(src_resize.size(), CV_32FC3);
	Mat b(src_resize.size(), CV_32FC1);
	Mat mean_a(src_resize.size(), CV_32FC3);
	Mat mean_b(src_resize.size(), CV_32FC1);

	// Ip 2 ab
	{
		float sumMeanI_b = 0.f;
		float sumMeanI_g = 0.f;
		float sumMeanI_r = 0.f;
		float sumMeanP = 0.f;
		float sumCorrI_bb = 0.f;
		float sumCorrI_bg = 0.f;
		float sumCorrI_br = 0.f;
		float sumCorrI_gg = 0.f;
		float sumCorrI_gr = 0.f;
		float sumCorrI_rr = 0.f;
		float sumCovIP_b = 0.f;
		float sumCovIP_g = 0.f;
		float sumCovIP_r = 0.f;

		float meanI_b = 0.f;
		float meanI_g = 0.f;
		float meanI_r = 0.f;
		float meanP = 0.f;
		float corrI_bb = 0.f;
		float corrI_bg = 0.f;
		float corrI_br = 0.f;
		float corrI_gg = 0.f;
		float corrI_gr = 0.f;
		float corrI_rr = 0.f;
		float covIP_b = 0.f;
		float covIP_g = 0.f;
		float covIP_r = 0.f;

		Mat columnSum = Mat::zeros(Size(col_resize, 1), CV_32FC(13));

		float* Ib_ptr_next = guide_resize.ptr<float>(0);
		float* Ig_ptr_next = Ib_ptr_next + 1;
		float* Ir_ptr_next = Ib_ptr_next + 2;
		float* p_ptr_next = src_resize.ptr<float>(0) + cn;
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = cp_prev;

		/*   i == 0   */
		for (int j = 0; j < col_resize; j++)
		{
			*(cp_next++) = *Ib_ptr_next * (r + 1);
			*(cp_next++) = *Ig_ptr_next * (r + 1);
			*(cp_next++) = *Ir_ptr_next * (r + 1);
			*(cp_next++) = *p_ptr_next * (r + 1);
			*(cp_next++) = (*Ib_ptr_next * *Ib_ptr_next) * (r + 1);
			*(cp_next++) = (*Ib_ptr_next * *Ig_ptr_next) * (r + 1);
			*(cp_next++) = (*Ib_ptr_next * *Ir_ptr_next) * (r + 1);
			*(cp_next++) = (*Ig_ptr_next * *Ig_ptr_next) * (r + 1);
			*(cp_next++) = (*Ig_ptr_next * *Ir_ptr_next) * (r + 1);
			*(cp_next++) = (*Ir_ptr_next * *Ir_ptr_next) * (r + 1);
			*(cp_next++) = (*Ib_ptr_next * *p_ptr_next) * (r + 1);
			*(cp_next++) = (*Ig_ptr_next * *p_ptr_next) * (r + 1);
			*(cp_next++) = (*Ir_ptr_next * *p_ptr_next) * (r + 1);

			Ib_ptr_next += I_cn;
			Ig_ptr_next += I_cn;
			Ir_ptr_next += I_cn;
			p_ptr_next += p_cn;
		}
		cp_next = cp_prev;
		for (int i = 1; i < r; i++)
		{
			for (int j = 0; j < col_resize; j++)
			{
				*(cp_next++) += *Ib_ptr_next;
				*(cp_next++) += *Ig_ptr_next;
				*(cp_next++) += *Ir_ptr_next;
				*(cp_next++) += *p_ptr_next;
				*(cp_next++) += (*Ib_ptr_next * *Ib_ptr_next);
				*(cp_next++) += (*Ib_ptr_next * *Ig_ptr_next);
				*(cp_next++) += (*Ib_ptr_next * *Ir_ptr_next);
				*(cp_next++) += (*Ig_ptr_next * *Ig_ptr_next);
				*(cp_next++) += (*Ig_ptr_next * *Ir_ptr_next);
				*(cp_next++) += (*Ir_ptr_next * *Ir_ptr_next);
				*(cp_next++) += (*Ib_ptr_next * *p_ptr_next);
				*(cp_next++) += (*Ig_ptr_next * *p_ptr_next);
				*(cp_next++) += (*Ir_ptr_next * *p_ptr_next);

				Ib_ptr_next += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_next += p_cn;
			}
			cp_next = cp_prev;
		}

		*cp_next += *Ib_ptr_next;
		sumMeanI_b += *(cp_next++) * (r + 1);

		*cp_next += *Ig_ptr_next;
		sumMeanI_g += *(cp_next++) * (r + 1);
		*cp_next += *Ir_ptr_next;
		sumMeanI_r += *(cp_next++) * (r + 1);
		*cp_next += *p_ptr_next;
		sumMeanP += *(cp_next++) * (r + 1);
		*cp_next += (*Ib_ptr_next * *Ib_ptr_next);
		sumCorrI_bb += *(cp_next++) * (r + 1);
		*cp_next += (*Ib_ptr_next * *Ig_ptr_next);
		sumCorrI_bg += *(cp_next++) * (r + 1);
		*cp_next += (*Ib_ptr_next * *Ir_ptr_next);
		sumCorrI_br += *(cp_next++) * (r + 1);
		*cp_next += (*Ig_ptr_next * *Ig_ptr_next);
		sumCorrI_gg += *(cp_next++) * (r + 1);
		*cp_next += (*Ig_ptr_next * *Ir_ptr_next);
		sumCorrI_gr += *(cp_next++) * (r + 1);
		*cp_next += (*Ir_ptr_next * *Ir_ptr_next);
		sumCorrI_rr += *(cp_next++) * (r + 1);
		*cp_next += (*Ib_ptr_next * *p_ptr_next);
		sumCovIP_b += *(cp_next++) * (r + 1);
		*cp_next += (*Ig_ptr_next * *p_ptr_next);
		sumCovIP_g += *(cp_next++) * (r + 1);
		*cp_next += (*Ir_ptr_next * *p_ptr_next);
		sumCovIP_r += *(cp_next++) * (r + 1);

		Ib_ptr_next += I_cn;
		Ig_ptr_next += I_cn;
		Ir_ptr_next += I_cn;
		p_ptr_next += p_cn;

		for (int j = 1; j <= r; j++)
		{
			*cp_next += *Ib_ptr_next;
			sumMeanI_b += *(cp_next++);

			*cp_next += *Ig_ptr_next;
			sumMeanI_g += *(cp_next++);
			*cp_next += *Ir_ptr_next;
			sumMeanI_r += *(cp_next++);
			*cp_next += *p_ptr_next;
			sumMeanP += *(cp_next++);
			*cp_next += (*Ib_ptr_next * *Ib_ptr_next);
			sumCorrI_bb += *(cp_next++);
			*cp_next += (*Ib_ptr_next * *Ig_ptr_next);
			sumCorrI_bg += *(cp_next++);
			*cp_next += (*Ib_ptr_next * *Ir_ptr_next);
			sumCorrI_br += *(cp_next++);
			*cp_next += (*Ig_ptr_next * *Ig_ptr_next);
			sumCorrI_gg += *(cp_next++);
			*cp_next += (*Ig_ptr_next * *Ir_ptr_next);
			sumCorrI_gr += *(cp_next++);
			*cp_next += (*Ir_ptr_next * *Ir_ptr_next);
			sumCorrI_rr += *(cp_next++);
			*cp_next += (*Ib_ptr_next * *p_ptr_next);
			sumCovIP_b += *(cp_next++);
			*cp_next += (*Ig_ptr_next * *p_ptr_next);
			sumCovIP_g += *(cp_next++);
			*cp_next += (*Ir_ptr_next * *p_ptr_next);
			sumCovIP_r += *(cp_next++);

			Ib_ptr_next += I_cn;
			Ig_ptr_next += I_cn;
			Ir_ptr_next += I_cn;
			p_ptr_next += p_cn;
		}

		float bb, bg, br, gg, gr, rr, covb, covg, covr;
		float det, id, c0, c1, c2, c4, c5, c8;
		float* ab_ptr = a.ptr<float>(0);
		float* ag_ptr = ab_ptr + 1;
		float* ar_ptr = ab_ptr + 2;
		float* b_ptr = b.ptr<float>(0);

		CALC_COVARIANCE();
		ab_ptr += I_cn;
		ag_ptr += I_cn;
		ar_ptr += I_cn;
		b_ptr++;


		for (int j = 1; j <= r; j++)
		{
			*cp_next += *Ib_ptr_next;
			sumMeanI_b = sumMeanI_b - *cp_prev + *(cp_next++);

			*cp_next += *Ig_ptr_next;
			sumMeanI_g = sumMeanI_g - *(cp_prev + 1) + *(cp_next++);
			*cp_next += *Ir_ptr_next;
			sumMeanI_r = sumMeanI_r - *(cp_prev + 2) + *(cp_next++);
			*cp_next += *p_ptr_next;
			sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
			*cp_next += (*Ib_ptr_next * *Ib_ptr_next);
			sumCorrI_bb = sumCorrI_bb - *(cp_prev + 4) + *(cp_next++);
			*cp_next += (*Ib_ptr_next * *Ig_ptr_next);
			sumCorrI_bg = sumCorrI_bg - *(cp_prev + 5) + *(cp_next++);
			*cp_next += (*Ib_ptr_next * *Ir_ptr_next);
			sumCorrI_br = sumCorrI_br - *(cp_prev + 6) + *(cp_next++);
			*cp_next += (*Ig_ptr_next * *Ig_ptr_next);
			sumCorrI_gg = sumCorrI_gg - *(cp_prev + 7) + *(cp_next++);
			*cp_next += (*Ig_ptr_next * *Ir_ptr_next);
			sumCorrI_gr = sumCorrI_gr - *(cp_prev + 8) + *(cp_next++);
			*cp_next += (*Ir_ptr_next * *Ir_ptr_next);
			sumCorrI_rr = sumCorrI_rr - *(cp_prev + 9) + *(cp_next++);
			*cp_next += (*Ib_ptr_next * *p_ptr_next);
			sumCovIP_b = sumCovIP_b - *(cp_prev + 10) + *(cp_next++);
			*cp_next += (*Ig_ptr_next * *p_ptr_next);
			sumCovIP_g = sumCovIP_g - *(cp_prev + 11) + *(cp_next++);
			*cp_next += (*Ir_ptr_next * *p_ptr_next);
			sumCovIP_r = sumCovIP_r - *(cp_prev + 12) + *(cp_next++);

			Ib_ptr_next += I_cn;
			Ig_ptr_next += I_cn;
			Ir_ptr_next += I_cn;
			p_ptr_next += p_cn;

			CALC_COVARIANCE();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;
		}
		for (int j = r + 1; j < col_resize - r; j++)
		{
			*cp_next += *Ib_ptr_next;
			sumMeanI_b = sumMeanI_b - *(cp_prev++) + *(cp_next++);

			*cp_next += *Ig_ptr_next;
			sumMeanI_g = sumMeanI_g - *(cp_prev++) + *(cp_next++);
			*cp_next += *Ir_ptr_next;
			sumMeanI_r = sumMeanI_r - *(cp_prev++) + *(cp_next++);
			*cp_next += *p_ptr_next;
			sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
			*cp_next += (*Ib_ptr_next * *Ib_ptr_next);
			sumCorrI_bb = sumCorrI_bb - *(cp_prev++) + *(cp_next++);
			*cp_next += (*Ib_ptr_next * *Ig_ptr_next);
			sumCorrI_bg = sumCorrI_bg - *(cp_prev++) + *(cp_next++);
			*cp_next += (*Ib_ptr_next * *Ir_ptr_next);
			sumCorrI_br = sumCorrI_br - *(cp_prev++) + *(cp_next++);
			*cp_next += (*Ig_ptr_next * *Ig_ptr_next);
			sumCorrI_gg = sumCorrI_gg - *(cp_prev++) + *(cp_next++);
			*cp_next += (*Ig_ptr_next * *Ir_ptr_next);
			sumCorrI_gr = sumCorrI_gr - *(cp_prev++) + *(cp_next++);
			*cp_next += (*Ir_ptr_next * *Ir_ptr_next);
			sumCorrI_rr = sumCorrI_rr - *(cp_prev++) + *(cp_next++);
			*cp_next += (*Ib_ptr_next * *p_ptr_next);
			sumCovIP_b = sumCovIP_b - *(cp_prev++) + *(cp_next++);
			*cp_next += (*Ig_ptr_next * *p_ptr_next);
			sumCovIP_g = sumCovIP_g - *(cp_prev++) + *(cp_next++);
			*cp_next += (*Ir_ptr_next * *p_ptr_next);
			sumCovIP_r = sumCovIP_r - *(cp_prev++) + *(cp_next++);

			Ib_ptr_next += I_cn;
			Ig_ptr_next += I_cn;
			Ir_ptr_next += I_cn;
			p_ptr_next += p_cn;

			CALC_COVARIANCE();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;
		}
		cp_next -= columnSum.channels();
		for (int j = col_resize - r; j < col_resize; j++)
		{
			sumMeanI_b = sumMeanI_b - *(cp_prev++) + *cp_next;

			sumMeanI_g = sumMeanI_g - *(cp_prev++) + *(cp_next + 1);
			sumMeanI_r = sumMeanI_r - *(cp_prev++) + *(cp_next + 2);
			sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);
			sumCorrI_bb = sumCorrI_bb - *(cp_prev++) + *(cp_next + 4);
			sumCorrI_bg = sumCorrI_bg - *(cp_prev++) + *(cp_next + 5);
			sumCorrI_br = sumCorrI_br - *(cp_prev++) + *(cp_next + 6);
			sumCorrI_gg = sumCorrI_gg - *(cp_prev++) + *(cp_next + 7);
			sumCorrI_gr = sumCorrI_gr - *(cp_prev++) + *(cp_next + 8);
			sumCorrI_rr = sumCorrI_rr - *(cp_prev++) + *(cp_next + 9);
			sumCovIP_b = sumCovIP_b - *(cp_prev++) + *(cp_next + 10);
			sumCovIP_g = sumCovIP_g - *(cp_prev++) + *(cp_next + 11);
			sumCovIP_r = sumCovIP_r - *(cp_prev++) + *(cp_next + 12);

			CALC_COVARIANCE();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);

		float* Ib_ptr_prev = guide_resize.ptr<float>(0);
		float* Ig_ptr_prev = Ib_ptr_prev + 1;
		float* Ir_ptr_prev = Ib_ptr_prev + 2;
		float* p_ptr_prev = src_resize.ptr<float>(0) + cn;

		/*   0 < i <= r   */
		for (int i = 1; i <= r; i++)
		{
			sumMeanI_b = 0.f;
			sumMeanI_g = 0.f;
			sumMeanI_r = 0.f;
			sumMeanP = 0.f;
			sumCorrI_bb = 0.f;
			sumCorrI_bg = 0.f;
			sumCorrI_br = 0.f;
			sumCorrI_gg = 0.f;
			sumCorrI_gr = 0.f;
			sumCorrI_rr = 0.f;
			sumCovIP_b = 0.f;
			sumCovIP_g = 0.f;
			sumCovIP_r = 0.f;

			*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
			sumMeanI_b += *(cp_next++) * (r + 1);

			*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
			sumMeanI_g += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
			sumMeanI_r += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
			sumMeanP += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
			sumCorrI_bb += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
			sumCorrI_bg += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
			sumCorrI_br += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
			sumCorrI_gg += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
			sumCorrI_gr += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
			sumCorrI_rr += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
			sumCovIP_b += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
			sumCovIP_g += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
			sumCovIP_r += *(cp_next++) * (r + 1);

			Ib_ptr_prev += I_cn;
			Ib_ptr_next += I_cn;
			Ig_ptr_prev += I_cn;
			Ig_ptr_next += I_cn;
			Ir_ptr_prev += I_cn;
			Ir_ptr_next += I_cn;
			p_ptr_prev += p_cn;
			p_ptr_next += p_cn;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
				sumMeanI_b += *(cp_next++);

				*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
				sumMeanI_g += *(cp_next++);
				*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
				sumMeanI_r += *(cp_next++);
				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
				sumCorrI_bb += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
				sumCorrI_bg += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
				sumCorrI_br += *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
				sumCorrI_gg += *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
				sumCorrI_gr += *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
				sumCorrI_rr += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
				sumCovIP_b += *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
				sumCovIP_g += *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
				sumCovIP_r += *(cp_next++);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
			}
			CALC_COVARIANCE();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
				sumMeanI_b = sumMeanI_b - *cp_prev + *(cp_next++);

				*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
				sumMeanI_g = sumMeanI_g - *(cp_prev + 1) + *(cp_next++);
				*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
				sumMeanI_r = sumMeanI_r - *(cp_prev + 2) + *(cp_next++);
				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
				sumCorrI_bb = sumCorrI_bb - *(cp_prev + 4) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
				sumCorrI_bg = sumCorrI_bg - *(cp_prev + 5) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
				sumCorrI_br = sumCorrI_br - *(cp_prev + 6) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
				sumCorrI_gg = sumCorrI_gg - *(cp_prev + 7) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
				sumCorrI_gr = sumCorrI_gr - *(cp_prev + 8) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
				sumCorrI_rr = sumCorrI_rr - *(cp_prev + 9) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
				sumCovIP_b = sumCovIP_b - *(cp_prev + 10) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
				sumCovIP_g = sumCovIP_g - *(cp_prev + 11) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
				sumCovIP_r = sumCovIP_r - *(cp_prev + 12) + *(cp_next++);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_COVARIANCE();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
				sumMeanI_b = sumMeanI_b - *(cp_prev++) + *(cp_next++);

				*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
				sumMeanI_g = sumMeanI_g - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
				sumMeanI_r = sumMeanI_r - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
				sumCorrI_bb = sumCorrI_bb - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
				sumCorrI_bg = sumCorrI_bg - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
				sumCorrI_br = sumCorrI_br - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
				sumCorrI_gg = sumCorrI_gg - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
				sumCorrI_gr = sumCorrI_gr - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
				sumCorrI_rr = sumCorrI_rr - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
				sumCovIP_b = sumCovIP_b - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
				sumCovIP_g = sumCovIP_g - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
				sumCovIP_r = sumCovIP_r - *(cp_prev++) + *(cp_next++);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_COVARIANCE();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanI_b = sumMeanI_b - *(cp_prev++) + *cp_next;
				sumMeanI_g = sumMeanI_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanI_r = sumMeanI_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);
				sumCorrI_bb = sumCorrI_bb - *(cp_prev++) + *(cp_next + 4);
				sumCorrI_bg = sumCorrI_bg - *(cp_prev++) + *(cp_next + 5);
				sumCorrI_br = sumCorrI_br - *(cp_prev++) + *(cp_next + 6);
				sumCorrI_gg = sumCorrI_gg - *(cp_prev++) + *(cp_next + 7);
				sumCorrI_gr = sumCorrI_gr - *(cp_prev++) + *(cp_next + 8);
				sumCorrI_rr = sumCorrI_rr - *(cp_prev++) + *(cp_next + 9);
				sumCovIP_b = sumCovIP_b - *(cp_prev++) + *(cp_next + 10);
				sumCovIP_g = sumCovIP_g - *(cp_prev++) + *(cp_next + 11);
				sumCovIP_r = sumCovIP_r - *(cp_prev++) + *(cp_next + 12);

				CALC_COVARIANCE();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			Ib_ptr_prev = guide_resize.ptr<float>(0);
			Ig_ptr_prev = Ib_ptr_prev + 1;
			Ir_ptr_prev = Ib_ptr_prev + 2;
			p_ptr_prev = src_resize.ptr<float>(0) + cn;
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row - r - 1   */
		for (int i = r + 1; i < row_resize - r - 1; i++)
		{
			sumMeanI_b = 0.f;
			sumMeanI_g = 0.f;
			sumMeanI_r = 0.f;
			sumMeanP = 0.f;
			sumCorrI_bb = 0.f;
			sumCorrI_bg = 0.f;
			sumCorrI_br = 0.f;
			sumCorrI_gg = 0.f;
			sumCorrI_gr = 0.f;
			sumCorrI_rr = 0.f;
			sumCovIP_b = 0.f;
			sumCovIP_g = 0.f;
			sumCovIP_r = 0.f;

			*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
			sumMeanI_b += *(cp_next++) * (r + 1);

			*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
			sumMeanI_g += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
			sumMeanI_r += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
			sumMeanP += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
			sumCorrI_bb += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
			sumCorrI_bg += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
			sumCorrI_br += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
			sumCorrI_gg += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
			sumCorrI_gr += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
			sumCorrI_rr += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
			sumCovIP_b += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
			sumCovIP_g += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
			sumCovIP_r += *(cp_next++) * (r + 1);

			Ib_ptr_prev += I_cn;
			Ib_ptr_next += I_cn;
			Ig_ptr_prev += I_cn;
			Ig_ptr_next += I_cn;
			Ir_ptr_prev += I_cn;
			Ir_ptr_next += I_cn;
			p_ptr_prev += p_cn;
			p_ptr_next += p_cn;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
				sumMeanI_b += *(cp_next++);

				*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
				sumMeanI_g += *(cp_next++);
				*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
				sumMeanI_r += *(cp_next++);
				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
				sumCorrI_bb += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
				sumCorrI_bg += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
				sumCorrI_br += *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
				sumCorrI_gg += *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
				sumCorrI_gr += *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
				sumCorrI_rr += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
				sumCovIP_b += *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
				sumCovIP_g += *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
				sumCovIP_r += *(cp_next++);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
			}
			CALC_COVARIANCE();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
				sumMeanI_b = sumMeanI_b - *cp_prev + *(cp_next++);

				*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
				sumMeanI_g = sumMeanI_g - *(cp_prev + 1) + *(cp_next++);
				*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
				sumMeanI_r = sumMeanI_r - *(cp_prev + 2) + *(cp_next++);
				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
				sumCorrI_bb = sumCorrI_bb - *(cp_prev + 4) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
				sumCorrI_bg = sumCorrI_bg - *(cp_prev + 5) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
				sumCorrI_br = sumCorrI_br - *(cp_prev + 6) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
				sumCorrI_gg = sumCorrI_gg - *(cp_prev + 7) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
				sumCorrI_gr = sumCorrI_gr - *(cp_prev + 8) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
				sumCorrI_rr = sumCorrI_rr - *(cp_prev + 9) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
				sumCovIP_b = sumCovIP_b - *(cp_prev + 10) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
				sumCovIP_g = sumCovIP_g - *(cp_prev + 11) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
				sumCovIP_r = sumCovIP_r - *(cp_prev + 12) + *(cp_next++);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_COVARIANCE();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
				sumMeanI_b = sumMeanI_b - *(cp_prev++) + *(cp_next++);

				*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
				sumMeanI_g = sumMeanI_g - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
				sumMeanI_r = sumMeanI_r - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
				sumCorrI_bb = sumCorrI_bb - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
				sumCorrI_bg = sumCorrI_bg - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
				sumCorrI_br = sumCorrI_br - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
				sumCorrI_gg = sumCorrI_gg - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
				sumCorrI_gr = sumCorrI_gr - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
				sumCorrI_rr = sumCorrI_rr - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
				sumCovIP_b = sumCovIP_b - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
				sumCovIP_g = sumCovIP_g - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
				sumCovIP_r = sumCovIP_r - *(cp_prev++) + *(cp_next++);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_COVARIANCE();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanI_b = sumMeanI_b - *(cp_prev++) + *cp_next;
				sumMeanI_g = sumMeanI_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanI_r = sumMeanI_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);
				sumCorrI_bb = sumCorrI_bb - *(cp_prev++) + *(cp_next + 4);
				sumCorrI_bg = sumCorrI_bg - *(cp_prev++) + *(cp_next + 5);
				sumCorrI_br = sumCorrI_br - *(cp_prev++) + *(cp_next + 6);
				sumCorrI_gg = sumCorrI_gg - *(cp_prev++) + *(cp_next + 7);
				sumCorrI_gr = sumCorrI_gr - *(cp_prev++) + *(cp_next + 8);
				sumCorrI_rr = sumCorrI_rr - *(cp_prev++) + *(cp_next + 9);
				sumCovIP_b = sumCovIP_b - *(cp_prev++) + *(cp_next + 10);
				sumCovIP_g = sumCovIP_g - *(cp_prev++) + *(cp_next + 11);
				sumCovIP_r = sumCovIP_r - *(cp_prev++) + *(cp_next + 12);

				CALC_COVARIANCE();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   row - r - 1 <= i < row   */
		for (int i = row_resize - r - 1; i < row_resize; i++)
		{
			sumMeanI_b = 0.f;
			sumMeanI_g = 0.f;
			sumMeanI_r = 0.f;
			sumMeanP = 0.f;
			sumCorrI_bb = 0.f;
			sumCorrI_bg = 0.f;
			sumCorrI_br = 0.f;
			sumCorrI_gg = 0.f;
			sumCorrI_gr = 0.f;
			sumCorrI_rr = 0.f;
			sumCovIP_b = 0.f;
			sumCovIP_g = 0.f;
			sumCovIP_r = 0.f;

			*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
			sumMeanI_b += *(cp_next++) * (r + 1);

			*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
			sumMeanI_g += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
			sumMeanI_r += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
			sumMeanP += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
			sumCorrI_bb += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
			sumCorrI_bg += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
			sumCorrI_br += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
			sumCorrI_gg += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
			sumCorrI_gr += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
			sumCorrI_rr += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
			sumCovIP_b += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
			sumCovIP_g += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
			sumCovIP_r += *(cp_next++) * (r + 1);

			Ib_ptr_prev += I_cn;
			Ib_ptr_next += I_cn;
			Ig_ptr_prev += I_cn;
			Ig_ptr_next += I_cn;
			Ir_ptr_prev += I_cn;
			Ir_ptr_next += I_cn;
			p_ptr_prev += p_cn;
			p_ptr_next += p_cn;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
				sumMeanI_b += *(cp_next++);

				*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
				sumMeanI_g += *(cp_next++);
				*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
				sumMeanI_r += *(cp_next++);
				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
				sumCorrI_bb += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
				sumCorrI_bg += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
				sumCorrI_br += *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
				sumCorrI_gg += *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
				sumCorrI_gr += *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
				sumCorrI_rr += *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
				sumCovIP_b += *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
				sumCovIP_g += *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
				sumCovIP_r += *(cp_next++);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
			}
			CALC_COVARIANCE();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
				sumMeanI_b = sumMeanI_b - *cp_prev + *(cp_next++);

				*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
				sumMeanI_g = sumMeanI_g - *(cp_prev + 1) + *(cp_next++);
				*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
				sumMeanI_r = sumMeanI_r - *(cp_prev + 2) + *(cp_next++);
				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
				sumCorrI_bb = sumCorrI_bb - *(cp_prev + 4) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
				sumCorrI_bg = sumCorrI_bg - *(cp_prev + 5) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
				sumCorrI_br = sumCorrI_br - *(cp_prev + 6) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
				sumCorrI_gg = sumCorrI_gg - *(cp_prev + 7) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
				sumCorrI_gr = sumCorrI_gr - *(cp_prev + 8) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
				sumCorrI_rr = sumCorrI_rr - *(cp_prev + 9) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
				sumCovIP_b = sumCovIP_b - *(cp_prev + 10) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
				sumCovIP_g = sumCovIP_g - *(cp_prev + 11) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
				sumCovIP_r = sumCovIP_r - *(cp_prev + 12) + *(cp_next++);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_COVARIANCE();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *Ib_ptr_prev + *Ib_ptr_next;
				sumMeanI_b = sumMeanI_b - *(cp_prev++) + *(cp_next++);

				*cp_next = *cp_next - *Ig_ptr_prev + *Ig_ptr_next;
				sumMeanI_g = sumMeanI_g - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *Ir_ptr_prev + *Ir_ptr_next;
				sumMeanI_r = sumMeanI_r - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ib_ptr_prev) + (*Ib_ptr_next * *Ib_ptr_next);
				sumCorrI_bb = sumCorrI_bb - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ig_ptr_prev) + (*Ib_ptr_next * *Ig_ptr_next);
				sumCorrI_bg = sumCorrI_bg - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *Ir_ptr_prev) + (*Ib_ptr_next * *Ir_ptr_next);
				sumCorrI_br = sumCorrI_br - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ig_ptr_prev) + (*Ig_ptr_next * *Ig_ptr_next);
				sumCorrI_gg = sumCorrI_gg - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *Ir_ptr_prev) + (*Ig_ptr_next * *Ir_ptr_next);
				sumCorrI_gr = sumCorrI_gr - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *Ir_ptr_prev) + (*Ir_ptr_next * *Ir_ptr_next);
				sumCorrI_rr = sumCorrI_rr - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ib_ptr_prev * *p_ptr_prev) + (*Ib_ptr_next * *p_ptr_next);
				sumCovIP_b = sumCovIP_b - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ig_ptr_prev * *p_ptr_prev) + (*Ig_ptr_next * *p_ptr_next);
				sumCovIP_g = sumCovIP_g - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - (*Ir_ptr_prev * *p_ptr_prev) + (*Ir_ptr_next * *p_ptr_next);
				sumCovIP_r = sumCovIP_r - *(cp_prev++) + *(cp_next++);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_COVARIANCE();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanI_b = sumMeanI_b - *(cp_prev++) + *cp_next;
				sumMeanI_g = sumMeanI_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanI_r = sumMeanI_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);
				sumCorrI_bb = sumCorrI_bb - *(cp_prev++) + *(cp_next + 4);
				sumCorrI_bg = sumCorrI_bg - *(cp_prev++) + *(cp_next + 5);
				sumCorrI_br = sumCorrI_br - *(cp_prev++) + *(cp_next + 6);
				sumCorrI_gg = sumCorrI_gg - *(cp_prev++) + *(cp_next + 7);
				sumCorrI_gr = sumCorrI_gr - *(cp_prev++) + *(cp_next + 8);
				sumCorrI_rr = sumCorrI_rr - *(cp_prev++) + *(cp_next + 9);
				sumCovIP_b = sumCovIP_b - *(cp_prev++) + *(cp_next + 10);
				sumCovIP_g = sumCovIP_g - *(cp_prev++) + *(cp_next + 11);
				sumCovIP_r = sumCovIP_r - *(cp_prev++) + *(cp_next + 12);

				CALC_COVARIANCE();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			Ib_ptr_next = guide_resize.ptr<float>(row_resize - 1);
			Ig_ptr_next = Ib_ptr_next + 1;
			Ir_ptr_next = Ib_ptr_next + 2;
			p_ptr_next = src_resize.ptr<float>(row_resize - 1) + cn;
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}

	// ab 2 q
	{
		float sumMeanA_b = 0.f;
		float sumMeanA_g = 0.f;
		float sumMeanA_r = 0.f;
		float sumMeanB = 0.f;

		Mat columnSum = Mat::zeros(Size(col_resize, 1), CV_32FC4);
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = cp_prev;

		float* ab_ptr_next = a.ptr<float>(0);
		float* ag_ptr_next = ab_ptr_next + 1;
		float* ar_ptr_next = ab_ptr_next + 2;
		float* b_ptr_next = b.ptr<float>(0);
		for (int j = 0; j < col_resize; j++)
		{
			*(cp_next++) = *ab_ptr_next * (r + 1);
			*(cp_next++) = *ag_ptr_next * (r + 1);
			*(cp_next++) = *ar_ptr_next * (r + 1);
			*(cp_next++) = *b_ptr_next * (r + 1);

			ab_ptr_next += I_cn;
			ag_ptr_next += I_cn;
			ar_ptr_next += I_cn;
			b_ptr_next++;
		}
		cp_next = cp_prev;
		for (int i = 1; i < r; i++)
		{
			for (int j = 0; j < col_resize; j++)
			{
				*(cp_next++) += *ab_ptr_next;
				*(cp_next++) += *ag_ptr_next;
				*(cp_next++) += *ar_ptr_next;
				*(cp_next++) += *b_ptr_next;

				ab_ptr_next += I_cn;
				ag_ptr_next += I_cn;
				ar_ptr_next += I_cn;
				b_ptr_next++;
			}
			cp_next = cp_prev;
		}

		*cp_next += *ab_ptr_next;
		sumMeanA_b += *(cp_next++) * (r + 1);

		*cp_next += *ag_ptr_next;
		sumMeanA_g += *(cp_next++) * (r + 1);
		*cp_next += *ar_ptr_next;
		sumMeanA_r += *(cp_next++) * (r + 1);
		*cp_next += *b_ptr_next;
		sumMeanB += *(cp_next++) * (r + 1);

		ab_ptr_next += I_cn;
		ag_ptr_next += I_cn;
		ar_ptr_next += I_cn;
		b_ptr_next++;

		for (int j = 1; j <= r; j++)
		{
			*cp_next += *ab_ptr_next;
			sumMeanA_b += *(cp_next++);
			*cp_next += *ag_ptr_next;
			sumMeanA_g += *(cp_next++);
			*cp_next += *ar_ptr_next;
			sumMeanA_r += *(cp_next++);
			*cp_next += *b_ptr_next;
			sumMeanB += *(cp_next++);

			ab_ptr_next += I_cn;
			ag_ptr_next += I_cn;
			ar_ptr_next += I_cn;
			b_ptr_next++;
		}
		float* a_ptr = mean_a.ptr<float>(0);
		float* b_ptr = mean_b.ptr<float>(0);

		*(a_ptr++) = sumMeanA_b * div;
		*(a_ptr++) = sumMeanA_g * div;
		*(a_ptr++) = sumMeanA_r * div;
		*(b_ptr++) = sumMeanB * div;

		for (int j = 1; j <= r; j++)
		{
			*cp_next += *ab_ptr_next;
			sumMeanA_b = sumMeanA_b - *cp_prev + *(cp_next++);

			*cp_next += *ag_ptr_next;
			sumMeanA_g = sumMeanA_g - *(cp_prev + 1) + *(cp_next++);
			*cp_next += *ar_ptr_next;
			sumMeanA_r = sumMeanA_r - *(cp_prev + 2) + *(cp_next++);
			*cp_next += *b_ptr_next;
			sumMeanB = sumMeanB - *(cp_prev + 3) + *(cp_next++);

			ab_ptr_next += I_cn;
			ag_ptr_next += I_cn;
			ar_ptr_next += I_cn;
			b_ptr_next++;

			*(a_ptr++) = sumMeanA_b * div;
			*(a_ptr++) = sumMeanA_g * div;
			*(a_ptr++) = sumMeanA_r * div;
			*(b_ptr++) = sumMeanB * div;
		}
		for (int j = r + 1; j < col_resize - r; j++)
		{
			*cp_next += *ab_ptr_next;
			sumMeanA_b = sumMeanA_b - *(cp_prev++) + *(cp_next++);

			*cp_next += *ag_ptr_next;
			sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next++);
			*cp_next += *ar_ptr_next;
			sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next++);
			*cp_next += *b_ptr_next;
			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);

			ab_ptr_next += I_cn;
			ag_ptr_next += I_cn;
			ar_ptr_next += I_cn;
			b_ptr_next++;

			*(a_ptr++) = sumMeanA_b * div;
			*(a_ptr++) = sumMeanA_g * div;
			*(a_ptr++) = sumMeanA_r * div;
			*(b_ptr++) = sumMeanB * div;
		}
		cp_next -= columnSum.channels();
		for (int j = col_resize - r; j < col_resize; j++)
		{
			sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;

			sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
			sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

			*(a_ptr++) = sumMeanA_b * div;
			*(a_ptr++) = sumMeanA_g * div;
			*(a_ptr++) = sumMeanA_r * div;
			*(b_ptr++) = sumMeanB * div;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);

		float* ab_ptr_prev = a.ptr<float>(0);
		float* ag_ptr_prev = ab_ptr_prev + 1;
		float* ar_ptr_prev = ab_ptr_prev + 2;
		float* b_ptr_prev = b.ptr<float>(0);
		/*   0 < i <= r   */
		for (int i = 1; i <= r; i++)
		{
			sumMeanA_b = 0.f;
			sumMeanA_g = 0.f;
			sumMeanA_r = 0.f;
			sumMeanB = 0.f;

			*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
			sumMeanA_b += *(cp_next++) * (r + 1);

			*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
			sumMeanA_g += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
			sumMeanA_r += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
			sumMeanB += *(cp_next++) * (r + 1);

			ab_ptr_prev += I_cn;
			ab_ptr_next += I_cn;
			ag_ptr_prev += I_cn;
			ag_ptr_next += I_cn;
			ar_ptr_prev += I_cn;
			ar_ptr_next += I_cn;
			b_ptr_prev++;
			b_ptr_next++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
				sumMeanA_b += *(cp_next++);

				*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
				sumMeanA_g += *(cp_next++);
				*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
				sumMeanA_r += *(cp_next++);
				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB += *(cp_next++);

				ab_ptr_prev += I_cn;
				ab_ptr_next += I_cn;
				ag_ptr_prev += I_cn;
				ag_ptr_next += I_cn;
				ar_ptr_prev += I_cn;
				ar_ptr_next += I_cn;
				b_ptr_prev++;
				b_ptr_next++;
			}
			*(a_ptr++) = sumMeanA_b * div;
			*(a_ptr++) = sumMeanA_g * div;
			*(a_ptr++) = sumMeanA_r * div;
			*(b_ptr++) = sumMeanB * div;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
				sumMeanA_b = sumMeanA_b - *cp_prev + *(cp_next++);

				*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev + 1) + *(cp_next++);
				*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
				sumMeanA_r = sumMeanA_r - *(cp_prev + 2) + *(cp_next++);
				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev + 3) + *(cp_next++);

				ab_ptr_prev += I_cn;
				ab_ptr_next += I_cn;
				ag_ptr_prev += I_cn;
				ag_ptr_next += I_cn;
				ar_ptr_prev += I_cn;
				ar_ptr_next += I_cn;
				b_ptr_prev++;
				b_ptr_next++;

				*(a_ptr++) = sumMeanA_b * div;
				*(a_ptr++) = sumMeanA_g * div;
				*(a_ptr++) = sumMeanA_r * div;
				*(b_ptr++) = sumMeanB * div;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *(cp_next++);

				*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);

				ab_ptr_prev += I_cn;
				ab_ptr_next += I_cn;
				ag_ptr_prev += I_cn;
				ag_ptr_next += I_cn;
				ar_ptr_prev += I_cn;
				ar_ptr_next += I_cn;
				b_ptr_prev++;
				b_ptr_next++;

				*(a_ptr++) = sumMeanA_b * div;
				*(a_ptr++) = sumMeanA_g * div;
				*(a_ptr++) = sumMeanA_r * div;
				*(b_ptr++) = sumMeanB * div;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;

				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

				*(a_ptr++) = sumMeanA_b * div;
				*(a_ptr++) = sumMeanA_g * div;
				*(a_ptr++) = sumMeanA_r * div;
				*(b_ptr++) = sumMeanB * div;
			}
			ab_ptr_prev = a.ptr<float>(0);
			ag_ptr_prev = ab_ptr_prev + 1;
			ar_ptr_prev = ab_ptr_prev + 2;
			b_ptr_prev = b.ptr<float>(0);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row - r - 1   */
		for (int i = r + 1; i < row_resize - r - 1; i++)
		{
			sumMeanA_b = 0.f;
			sumMeanA_g = 0.f;
			sumMeanA_r = 0.f;
			sumMeanB = 0.f;

			*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
			sumMeanA_b += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
			sumMeanA_g += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
			sumMeanA_r += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
			sumMeanB += *(cp_next++) * (r + 1);

			ab_ptr_prev += I_cn;
			ab_ptr_next += I_cn;
			ag_ptr_prev += I_cn;
			ag_ptr_next += I_cn;
			ar_ptr_prev += I_cn;
			ar_ptr_next += I_cn;
			b_ptr_prev++;
			b_ptr_next++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
				sumMeanA_b += *(cp_next++);
				*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
				sumMeanA_g += *(cp_next++);
				*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
				sumMeanA_r += *(cp_next++);
				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB += *(cp_next++);

				ab_ptr_prev += I_cn;
				ab_ptr_next += I_cn;
				ag_ptr_prev += I_cn;
				ag_ptr_next += I_cn;
				ar_ptr_prev += I_cn;
				ar_ptr_next += I_cn;
				b_ptr_prev++;
				b_ptr_next++;
			}
			*(a_ptr++) = sumMeanA_b * div;
			*(a_ptr++) = sumMeanA_g * div;
			*(a_ptr++) = sumMeanA_r * div;
			*(b_ptr++) = sumMeanB * div;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
				sumMeanA_b = sumMeanA_b - *cp_prev + *(cp_next++);
				*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev + 1) + *(cp_next++);
				*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
				sumMeanA_r = sumMeanA_r - *(cp_prev + 2) + *(cp_next++);
				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev + 3) + *(cp_next++);

				ab_ptr_prev += I_cn;
				ab_ptr_next += I_cn;
				ag_ptr_prev += I_cn;
				ag_ptr_next += I_cn;
				ar_ptr_prev += I_cn;
				ar_ptr_next += I_cn;
				b_ptr_prev++;
				b_ptr_next++;

				*(a_ptr++) = sumMeanA_b * div;
				*(a_ptr++) = sumMeanA_g * div;
				*(a_ptr++) = sumMeanA_r * div;
				*(b_ptr++) = sumMeanB * div;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);

				ab_ptr_prev += I_cn;
				ab_ptr_next += I_cn;
				ag_ptr_prev += I_cn;
				ag_ptr_next += I_cn;
				ar_ptr_prev += I_cn;
				ar_ptr_next += I_cn;
				b_ptr_prev++;
				b_ptr_next++;

				*(a_ptr++) = sumMeanA_b * div;
				*(a_ptr++) = sumMeanA_g * div;
				*(a_ptr++) = sumMeanA_r * div;
				*(b_ptr++) = sumMeanB * div;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

				*(a_ptr++) = sumMeanA_b * div;
				*(a_ptr++) = sumMeanA_g * div;
				*(a_ptr++) = sumMeanA_r * div;
				*(b_ptr++) = sumMeanB * div;
			}
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   row - r - 1 < i < row   */
		for (int i = row_resize - r - 1; i < row_resize; i++)
		{
			sumMeanA_b = 0.f;
			sumMeanA_g = 0.f;
			sumMeanA_r = 0.f;
			sumMeanB = 0.f;

			*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
			sumMeanA_b += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
			sumMeanA_g += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
			sumMeanA_r += *(cp_next++) * (r + 1);
			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
			sumMeanB += *(cp_next++) * (r + 1);

			ab_ptr_prev += I_cn;
			ab_ptr_next += I_cn;
			ag_ptr_prev += I_cn;
			ag_ptr_next += I_cn;
			ar_ptr_prev += I_cn;
			ar_ptr_next += I_cn;
			b_ptr_prev++;
			b_ptr_next++;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
				sumMeanA_b += *(cp_next++);
				*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
				sumMeanA_g += *(cp_next++);
				*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
				sumMeanA_r += *(cp_next++);
				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB += *(cp_next++);

				ab_ptr_prev += I_cn;
				ab_ptr_next += I_cn;
				ag_ptr_prev += I_cn;
				ag_ptr_next += I_cn;
				ar_ptr_prev += I_cn;
				ar_ptr_next += I_cn;
				b_ptr_prev++;
				b_ptr_next++;
			}
			*(a_ptr++) = sumMeanA_b * div;
			*(a_ptr++) = sumMeanA_g * div;
			*(a_ptr++) = sumMeanA_r * div;
			*(b_ptr++) = sumMeanB * div;

			for (int j = 1; j <= r; j++)
			{
				*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
				sumMeanA_b = sumMeanA_b - *cp_prev + *(cp_next++);
				*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev + 1) + *(cp_next++);
				*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
				sumMeanA_r = sumMeanA_r - *(cp_prev + 2) + *(cp_next++);
				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev + 3) + *(cp_next++);

				ab_ptr_prev += I_cn;
				ab_ptr_next += I_cn;
				ag_ptr_prev += I_cn;
				ag_ptr_next += I_cn;
				ar_ptr_prev += I_cn;
				ar_ptr_next += I_cn;
				b_ptr_prev++;
				b_ptr_next++;

				*(a_ptr++) = sumMeanA_b * div;
				*(a_ptr++) = sumMeanA_g * div;
				*(a_ptr++) = sumMeanA_r * div;
				*(b_ptr++) = sumMeanB * div;
			}
			for (int j = r + 1; j < col_resize - r; j++)
			{
				*cp_next = *cp_next - *ab_ptr_prev + *ab_ptr_next;
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *ag_ptr_prev + *ag_ptr_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *ar_ptr_prev + *ar_ptr_next;
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next++);
				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);

				ab_ptr_prev += I_cn;
				ab_ptr_next += I_cn;
				ag_ptr_prev += I_cn;
				ag_ptr_next += I_cn;
				ar_ptr_prev += I_cn;
				ar_ptr_next += I_cn;
				b_ptr_prev++;
				b_ptr_next++;

				*(a_ptr++) = sumMeanA_b * div;
				*(a_ptr++) = sumMeanA_g * div;
				*(a_ptr++) = sumMeanA_r * div;
				*(b_ptr++) = sumMeanB * div;
			}
			cp_next -= columnSum.channels();
			for (int j = col_resize - r; j < col_resize; j++)
			{
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

				*(a_ptr++) = sumMeanA_b * div;
				*(a_ptr++) = sumMeanA_g * div;
				*(a_ptr++) = sumMeanA_r * div;
				*(b_ptr++) = sumMeanB * div;
			}
			ab_ptr_next = a.ptr<float>(row_resize - 1);
			ag_ptr_next = ab_ptr_next + 1;
			ar_ptr_next = ab_ptr_next + 2;
			b_ptr_next = b.ptr<float>(row_resize - 1);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}

	// resize
	resize(mean_a, mean_a, Size(), 2.0, 2.0);
	resize(mean_b, mean_b, Size(), 2.0, 2.0);

	// mean_a mean_b 2 
	{
		float* a_ptr = mean_a.ptr<float>(0);
		float* b_ptr = mean_b.ptr<float>(0);
		float* I_ptr = guide.ptr<float>(0);
		float* q_ptr = dest.ptr<float>(0) + cn;

		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				*q_ptr = (*a_ptr * *I_ptr) + (*(a_ptr+1) * *(I_ptr+1)) + (*(a_ptr+2) * *(I_ptr+2)) + *b_ptr;
				a_ptr += I_cn;
				b_ptr++;
				I_ptr += I_cn;
				q_ptr += p_cn;
			}
		}
	}
}
