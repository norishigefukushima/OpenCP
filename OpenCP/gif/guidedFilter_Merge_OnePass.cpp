#include "guidedFilter_Merge_OnePass.h"
#include <intrin.h>

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


guidedFilter_Merge_OnePass::guidedFilter_Merge_OnePass(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
	: GuidedFilterBase(_src, _guide, _dest, _r, _eps), parallelType(_parallelType)
{
	div = 1.f / ((2 * r + 1)*(2 * r + 1));
	row = src.rows;
	col = src.cols;
	p_cn = src.channels();
	I_cn = guide.channels();
}

void guidedFilter_Merge_OnePass::filter()
{
	if (parallelType == NAIVE)
	{
		if (I_cn == 1)
		{
			for (int i = 0; i < p_cn; i++)
				filter_Guide1(i);
		}
		else if (I_cn == 3)
		{
			for (int i = 0; i < p_cn; i++)
				filter_Guide3(i);
		}
	}
	else
	{
		if (I_cn == 1)
		{
#pragma omp parallel for
			for (int i = 0; i < p_cn; i++)
				filter_Guide1(i);
		}
		else if (I_cn == 3)
		{
#pragma omp parallel for
			for (int i = 0; i < p_cn; i++)
				filter_Guide3(i);
		}
	}
}

void guidedFilter_Merge_OnePass::filterVector()
{
	cv::merge(vsrc, src);
	cv::merge(vguide, guide);

	if (parallelType == NAIVE)
	{
		if (I_cn == 1)
		{
			for (int i = 0; i < p_cn; i++)
				filter_Guide1(i);
		}
		else if (I_cn == 3)
		{
			for (int i = 0; i < p_cn; i++)
				filter_Guide3(i);
		}
	}
	else
	{
		if (I_cn == 1)
		{
#pragma omp parallel for
			for (int i = 0; i < p_cn; i++)
				filter_Guide1(i);
		}
		else if (I_cn == 3)
		{
#pragma omp parallel for
			for (int i = 0; i < p_cn; i++)
				filter_Guide3(i);
		}
	}

	split(dest, vdest);
}

void guidedFilter_Merge_OnePass::filter_Guide1(int cn)
{
	Mat a, b;
	a.create(src.size(), CV_32FC1);
	b.create(src.size(), CV_32FC1);

	// Ip 2 ab
	{
		float sumMeanI = 0.f;
		float sumMeanP = 0.f;
		float sumCorrI = 0.f;
		float sumCorrIP = 0.f;

		float meanI = 0.f;
		float meanP = 0.f;
		float corrI = 0.f;
		float corrIP = 0.f;

		Mat columnSum = Mat::zeros(1, col, CV_32FC4);
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = cp_prev;

		float* I_ptr_next = guide.ptr<float>(0);
		float* p_ptr_next = src.ptr<float>(0) + cn;
		for (int j = 0; j < col; j++)
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
			for (int j = 0; j < col; j++)
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
		for (int j = r + 1; j < col - r; j++)
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
		for (int j = col - r; j < col; j++)
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

		float* I_ptr_prev = guide.ptr<float>(0);
		float* p_ptr_prev = src.ptr<float>(0) + cn;
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
			for (int j = r + 1; j < col - r; j++)
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
			for (int j = col - r; j < col; j++)
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
			I_ptr_prev = guide.ptr<float>(0);
			p_ptr_prev = src.ptr<float>(0) + cn;
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row - r - 1   */
		for (int i = r + 1; i < row - r - 1; i++)
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
			for (int j = r + 1; j < col - r; j++)
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
			for (int j = col - r; j < col; j++)
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

		/*   row - r - 1 <= i < row   */
		for (int i = row - r - 1; i < row; i++)
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
			for (int j = r + 1; j < col - r; j++)
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
			for (int j = col - r; j < col; j++)
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
			I_ptr_next = guide.ptr<float>(row - 1);
			p_ptr_next = src.ptr<float>(row - 1) + cn;
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}

	// ab 2 q
	{
		float sumMeanA = 0.f;
		float sumMeanB = 0.f;

		float meanA = 0.f;
		float meanB = 0.f;

		Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC2);
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = cp_prev;

		float* a_ptr_next = a.ptr<float>(0);
		float* b_ptr_next = b.ptr<float>(0);
		for (int j = 0; j < col; j++)
		{
			*(cp_next++) = *a_ptr_next * (r + 1);
			a_ptr_next++;

			*(cp_next++) = *b_ptr_next * (r + 1);
			b_ptr_next++;
		}
		cp_next = cp_prev;
		for (int i = 1; i < r; i++)
		{
			for (int j = 0; j < col; j++)
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
		meanA = sumMeanA * div;
		meanB = sumMeanB * div;

		float* q_ptr = dest.ptr<float>(0) + cn;
		float* I_ptr = guide.ptr<float>(0);
		*q_ptr = meanA * *I_ptr + meanB;
		q_ptr += p_cn;
		I_ptr++;

		for (int j = 1; j <= r; j++)
		{
			*cp_next += *a_ptr_next;
			sumMeanA = sumMeanA - *cp_prev + *(cp_next++);
			a_ptr_next++;

			*cp_next += *b_ptr_next;
			sumMeanB = sumMeanB - *(cp_prev + 1) + *(cp_next++);
			b_ptr_next++;

			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			*cp_next += *a_ptr_next;
			sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
			a_ptr_next++;

			*cp_next += *b_ptr_next;
			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
			b_ptr_next++;

			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;
		}
		cp_next -= columnSum.channels();
		for (int j = col - r; j < col; j++)
		{
			sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;
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
			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;

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

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			for (int j = r + 1; j < col - r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			a_ptr_prev = a.ptr<float>(0);
			b_ptr_prev = b.ptr<float>(0);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row - r - 1   */
		for (int i = r + 1; i < row - r - 1; i++)
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
			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;

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

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			for (int j = r + 1; j < col - r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   row - r - 1 < i < row   */
		for (int i = row - r - 1; i < row; i++)
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
			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;

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

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			for (int j = r + 1; j < col - r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			a_ptr_next = a.ptr<float>(row - 1);
			b_ptr_next = b.ptr<float>(row - 1);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}
}



void guidedFilter_Merge_OnePass::filter_Guide3(int cn)
{
	Mat a, b;
	a.create(src.size(), CV_32FC3);
	b.create(src.size(), CV_32FC1);

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

		Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC(13));

		float* Ib_ptr_next = guide.ptr<float>(0);
		float* Ig_ptr_next = guide.ptr<float>(0) + 1;
		float* Ir_ptr_next = guide.ptr<float>(0) + 2;
		float* p_ptr_next = src.ptr<float>(0) + cn;
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = cp_prev;

		/*   i == 0   */
		for (int j = 0; j < col; j++)
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
			for (int j = 0; j < col; j++)
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

		/*{
			meanI_b = sumMeanI_b * div;
			meanI_g = sumMeanI_g * div;
			meanI_r = sumMeanI_r * div;
			meanP = sumMeanP * div;
			corrI_bb = sumCorrI_bb * div;
			corrI_bg = sumCorrI_bg * div;
			corrI_br = sumCorrI_br * div;
			corrI_gg = sumCorrI_gg * div;
			corrI_gr = sumCorrI_gr * div;
			corrI_rr = sumCorrI_rr * div;
			covIP_b = sumCovIP_b * div;
			covIP_g = sumCovIP_g * div;
			covIP_r = sumCovIP_r * div;

			bb = corrI_bb - meanI_b * meanI_b;
			bg = corrI_bg - meanI_b * meanI_g;
			br = corrI_br - meanI_b * meanI_r;
			gg = corrI_gg - meanI_g * meanI_g;
			gr = corrI_gr - meanI_g * meanI_r;
			rr = corrI_rr - meanI_r * meanI_r;
			covb = covIP_b - meanI_b * meanP;
			covg = covIP_g - meanI_g * meanP;
			covr = covIP_r - meanI_r * meanP;

			bb += eps;
			gg += eps;
			rr += eps;

			det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
			id = 1.f / det;

			c0 = gg*rr - gr*gr;
			c1 = br*gr - bg*rr;
			c2 = bg*gr - br*gg;
			c4 = bb*rr - br*br;
			c5 = br*bg - bb*gr;
			c8 = bb*gg - bg*bg;

			*ab_ptr = id * (covb*c0 + covg*c1 + covr*c2);
			*ag_ptr = id * (covb*c1 + covg*c4 + covr*c5);
			*ar_ptr = id * (covb*c2 + covg*c5 + covr*c8);
			*b_ptr = meanP - (*ab_ptr * meanI_b + *ag_ptr * meanI_g + *ar_ptr * meanI_r);

		}*/
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
		for (int j = r + 1; j < col - r; j++)
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
		for (int j = col - r; j < col; j++)
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

		float* Ib_ptr_prev = guide.ptr<float>(0);
		float* Ig_ptr_prev = guide.ptr<float>(0) + 1;
		float* Ir_ptr_prev = guide.ptr<float>(0) + 2;
		float* p_ptr_prev = src.ptr<float>(0) + cn;

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
			for (int j = r + 1; j < col - r; j++)
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
			for (int j = col - r; j < col; j++)
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
			Ib_ptr_prev = guide.ptr<float>(0);
			Ig_ptr_prev = guide.ptr<float>(0) + 1;
			Ir_ptr_prev = guide.ptr<float>(0) + 2;
			p_ptr_prev = src.ptr<float>(0) + cn;
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row - r - 1   */
		for (int i = r + 1; i < row - r - 1; i++)
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
			for (int j = r + 1; j < col - r; j++)
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
			for (int j = col - r; j < col; j++)
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
		for (int i = row - r - 1; i < row; i++)
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
			for (int j = r + 1; j < col - r; j++)
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
			for (int j = col - r; j < col; j++)
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
			Ib_ptr_next = guide.ptr<float>(row - 1);
			Ig_ptr_next = Ib_ptr_next + 1;
			Ir_ptr_next = Ib_ptr_next + 2;
			p_ptr_next = src.ptr<float>(row - 1) + cn;
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}

	// ab 2 q
	{
		float sumMeanA_b = 0.f;
		float sumMeanA_g = 0.f;
		float sumMeanA_r = 0.f;
		float sumMeanB = 0.f;

		float meanA_b = 0.f;
		float meanA_g = 0.f;
		float meanA_r = 0.f;
		float meanB = 0.f;

		Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC4);
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = cp_prev;

		float* ab_ptr_next = a.ptr<float>(0);
		float* ag_ptr_next = ab_ptr_next + 1;
		float* ar_ptr_next = ab_ptr_next + 2;
		float* b_ptr_next = b.ptr<float>(0);
		for (int j = 0; j < col; j++)
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
			for (int j = 0; j < col; j++)
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
		meanA_b = sumMeanA_b * div;
		meanA_g = sumMeanA_g * div;
		meanA_r = sumMeanA_r * div;
		meanB = sumMeanB * div;

		float* q_ptr = dest.ptr<float>(0) + cn;
		float* I_ptr = guide.ptr<float>(0);
		*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
		q_ptr += p_cn;
		I_ptr += I_cn;

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

			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;
		}
		for (int j = r + 1; j < col - r; j++)
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

			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;
		}
		cp_next -= columnSum.channels();
		for (int j = col - r; j < col; j++)
		{
			sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;

			sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
			sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;
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
			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;

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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			for (int j = r + 1; j < col - r; j++)
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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;

				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			ab_ptr_prev = a.ptr<float>(0);
			ag_ptr_prev = ab_ptr_prev + 1;
			ar_ptr_prev = ab_ptr_prev + 2;
			b_ptr_prev = b.ptr<float>(0);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row - r - 1   */
		for (int i = r + 1; i < row - r - 1; i++)
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
			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;

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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			for (int j = r + 1; j < col - r; j++)
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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   row - r - 1 < i < row   */
		for (int i = row - r - 1; i < row; i++)
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
			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;

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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			for (int j = r + 1; j < col - r; j++)
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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			ab_ptr_next = a.ptr<float>(row - 1);
			ag_ptr_next = ab_ptr_next + 1;
			ar_ptr_next = ab_ptr_next + 2;
			b_ptr_next = b.ptr<float>(row - 1);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}
}



//#define _MERGE_

void guidedFilter_Merge_OnePass_SIMD::filter_Guide1(int cn)
{
	Mat a, b;
	a.create(src.size(), CV_32FC1);
	b.create(src.size(), CV_32FC1);

	__m128 mDiv = _mm_set1_ps(div);
	__m128 mBorder = _mm_set1_ps(static_cast<float>(r + 1));

#ifdef _MERGE_
	int index = 0;
	__m256 mEps = _mm256_set1_ps(eps);
	__m256 mVar = _mm256_setzero_ps();
	__m256 mCov = _mm256_setzero_ps();
	__m256 mMeanI = _mm256_setzero_ps();
	__m256 mMeanP = _mm256_setzero_ps();
	__m256 mA = _mm256_setzero_ps();
	__m256 mB = _mm256_setzero_ps();
#endif

	// Ip 2 ab
	{
		__m128 mSum = _mm_setzero_ps();
		__m128 mMean = _mm_setzero_ps();

		Mat columnSum = Mat::zeros(1, col, CV_32FC4);
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = columnSum.ptr<float>(0);
		__m128 mCol_prev = _mm_setzero_ps();
		__m128 mCol_next = _mm_setzero_ps();
		__m128 mTmp = _mm_setzero_ps();

		float* I_ptr_next = guide.ptr<float>(0);
		float* p_ptr_next = src.ptr<float>(0) + cn;
		for (int j = 0; j < col; j++)
		{
			mCol_next = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
			mCol_next = _mm_mul_ps(mCol_next, mBorder);
			_mm_storeu_ps(cp_next, mCol_next);

			I_ptr_next++;
			p_ptr_next += p_cn;
			cp_next += 4;
		}
		cp_next = cp_prev;
		for (int i = 1; i < r; i++)
		{
			for (int j = 0; j < col; j++)
			{
				mCol_next = _mm_loadu_ps(cp_next);
				mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
				mCol_next = _mm_add_ps(mCol_next, mTmp);
				_mm_storeu_ps(cp_next, mCol_next);

				I_ptr_next++;
				p_ptr_next += p_cn;
				cp_next += 4;
			}
			cp_next = cp_prev;
		}

		mCol_next = _mm_loadu_ps(cp_next);
		mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
		mCol_next = _mm_add_ps(mCol_next, mTmp);
		_mm_storeu_ps(cp_next, mCol_next);

		I_ptr_next++;
		p_ptr_next += p_cn;
		cp_next += 4;

		mSum = _mm_mul_ps(mCol_next, mBorder);

		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_loadu_ps(cp_next);
			mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
			mCol_next = _mm_add_ps(mCol_next, mTmp);
			_mm_storeu_ps(cp_next, mCol_next);

			I_ptr_next++;
			p_ptr_next += p_cn;
			cp_next += 4;

			mSum = _mm_add_ps(mSum, mCol_next);
		}
		mTmp = _mm_mul_ps(mSum, mDiv);

		float* a_ptr = a.ptr<float>(0);
		float* b_ptr = b.ptr<float>(0);

#ifndef _MERGE_
		
		float varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
		float covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

		*a_ptr = covIP / (varI + eps);
		*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
		a_ptr++;
		b_ptr++;
#else
		mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
		mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
		mMeanI.m256_f32[index] = mTmp.m128_f32[0];
		mMeanP.m256_f32[index] = mTmp.m128_f32[3];
		index++;
#endif
		mCol_prev = _mm_loadu_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_loadu_ps(cp_next);
			mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
			mCol_next = _mm_add_ps(mCol_next, mTmp);
			_mm_storeu_ps(cp_next, mCol_next);

			I_ptr_next++;
			p_ptr_next += p_cn;
			cp_next += 4;

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);

			mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
			float varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
			float covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

			*a_ptr = covIP / (varI + eps);
			*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
			a_ptr++;
			b_ptr++;
#else
			mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
			mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
			mMeanI.m256_f32[index] = mTmp.m128_f32[0];
			mMeanP.m256_f32[index] = mTmp.m128_f32[3];
			index++;
			if (index == 8)
			{
				mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
				mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
				_mm256_storeu_ps(a_ptr, mA);
				_mm256_storeu_ps(b_ptr, mB);
				a_ptr += 8;
				b_ptr += 8;
				index = 0;
			}
#endif
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_next = _mm_loadu_ps(cp_next);
			mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
			mCol_next = _mm_add_ps(mCol_next, mTmp);
			_mm_storeu_ps(cp_next, mCol_next);

			mCol_prev = _mm_loadu_ps(cp_prev);

			I_ptr_next++;
			p_ptr_next += p_cn;
			cp_next += 4;
			cp_prev += 4;

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);

			mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
			varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
			covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

			*a_ptr = covIP / (varI + eps);
			*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
			a_ptr++;
			b_ptr++;

#else
			mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
			mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
			mMeanI.m256_f32[index] = mTmp.m128_f32[0];
			mMeanP.m256_f32[index] = mTmp.m128_f32[3];
			index++;
			if (index == 8)
			{
				mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
				mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
				_mm256_storeu_ps(a_ptr, mA);
				_mm256_storeu_ps(b_ptr, mB);
				a_ptr += 8;
				b_ptr += 8;
				index = 0;
			}
#endif
		}
		cp_next -= 4;
		mCol_next = _mm_loadu_ps(cp_next);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev = _mm_loadu_ps(cp_prev);
			cp_prev += 4;

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);

			mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
			varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
			covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

			*a_ptr = covIP / (varI + eps);
			*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
			a_ptr++;
			b_ptr++;
#else
			mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
			mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
			mMeanI.m256_f32[index] = mTmp.m128_f32[0];
			mMeanP.m256_f32[index] = mTmp.m128_f32[3];
			index++;
			if (index == 8)
			{
				mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
				mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
				_mm256_storeu_ps(a_ptr, mA);
				_mm256_storeu_ps(b_ptr, mB);
				a_ptr += 8;
				b_ptr += 8;
				index = 0;
			}
#endif
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);

		float* I_ptr_prev = guide.ptr<float>(0);
		float* p_ptr_prev = src.ptr<float>(0) + cn;
		/*   0 < i <= r   */
		for (int i = 1; i <= r; i++)
		{
			mSum = _mm_setzero_ps();

			mCol_next = _mm_loadu_ps(cp_next);
			mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
			mCol_next = _mm_sub_ps(mCol_next, mTmp);
			mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
			mCol_next = _mm_add_ps(mCol_next, mTmp);
			_mm_storeu_ps(cp_next, mCol_next);

			I_ptr_prev++;
			I_ptr_next++;
			p_ptr_prev += p_cn;
			p_ptr_next += p_cn;
			cp_next += 4;

			mSum = _mm_mul_ps(mCol_next, mBorder);

			for (int j = 1; j <= r; j++)
			{
				mCol_next = _mm_loadu_ps(cp_next);
				mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
				mCol_next = _mm_sub_ps(mCol_next, mTmp);
				mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
				mCol_next = _mm_add_ps(mCol_next, mTmp);
				_mm_storeu_ps(cp_next, mCol_next);

				I_ptr_prev++;
				I_ptr_next++;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
				cp_next += 4;

				mSum = _mm_add_ps(mSum, mCol_next);
			}
			mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
			varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
			covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

			*a_ptr = covIP / (varI + eps);
			*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
			a_ptr++;
			b_ptr++;
#else
			mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
			mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
			mMeanI.m256_f32[index] = mTmp.m128_f32[0];
			mMeanP.m256_f32[index] = mTmp.m128_f32[3];
			index++;
			if (index == 8)
			{
				mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
				mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
				_mm256_storeu_ps(a_ptr, mA);
				_mm256_storeu_ps(b_ptr, mB);
				a_ptr += 8;
				b_ptr += 8;
				index = 0;
			}
#endif

			mCol_prev = _mm_loadu_ps(cp_prev);
			for (int j = 1; j <= r; j++)
			{
				mCol_next = _mm_loadu_ps(cp_next);
				mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
				mCol_next = _mm_sub_ps(mCol_next, mTmp);
				mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
				mCol_next = _mm_add_ps(mCol_next, mTmp);
				_mm_storeu_ps(cp_next, mCol_next);

				I_ptr_prev++;
				I_ptr_next++;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
				cp_next += 4;

				mSum = _mm_sub_ps(mSum, mCol_prev);
				mSum = _mm_add_ps(mSum, mCol_next);

				mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
				varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
				covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

				*a_ptr = covIP / (varI + eps);
				*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
				a_ptr++;
				b_ptr++;
#else
				mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
				mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
				mMeanI.m256_f32[index] = mTmp.m128_f32[0];
				mMeanP.m256_f32[index] = mTmp.m128_f32[3];
				index++;
				if (index == 8)
				{
					mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
					mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
					_mm256_storeu_ps(a_ptr, mA);
					_mm256_storeu_ps(b_ptr, mB);
					a_ptr += 8;
					b_ptr += 8;
					index = 0;
				}
#endif
			}
			for (int j = r + 1; j < col - r; j++)
			{
				mCol_next = _mm_loadu_ps(cp_next);
				mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
				mCol_next = _mm_sub_ps(mCol_next, mTmp);
				mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
				mCol_next = _mm_add_ps(mCol_next, mTmp);
				_mm_storeu_ps(cp_next, mCol_next);

				mCol_prev = _mm_loadu_ps(cp_prev);

				I_ptr_prev++;
				I_ptr_next++;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
				cp_prev += 4;
				cp_next += 4;

				mSum = _mm_sub_ps(mSum, mCol_prev);
				mSum = _mm_add_ps(mSum, mCol_next);

				mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
				varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
				covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

				*a_ptr = covIP / (varI + eps);
				*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
				a_ptr++;
				b_ptr++;
#else
				mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
				mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
				mMeanI.m256_f32[index] = mTmp.m128_f32[0];
				mMeanP.m256_f32[index] = mTmp.m128_f32[3];
				index++;
				if (index == 8)
				{
					mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
					mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
					_mm256_storeu_ps(a_ptr, mA);
					_mm256_storeu_ps(b_ptr, mB);
					a_ptr += 8;
					b_ptr += 8;
					index = 0;
				}
#endif
			}
			cp_next -= 4;
			mCol_next = _mm_loadu_ps(cp_next);
			for (int j = col - r; j < col; j++)
			{
				mCol_prev = _mm_loadu_ps(cp_prev);
				cp_prev += 4;

				mSum = _mm_sub_ps(mSum, mCol_prev);
				mSum = _mm_add_ps(mSum, mCol_next);

				mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
				varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
				covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

				*a_ptr = covIP / (varI + eps);
				*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
				a_ptr++;
				b_ptr++;
#else
				mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
				mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
				mMeanI.m256_f32[index] = mTmp.m128_f32[0];
				mMeanP.m256_f32[index] = mTmp.m128_f32[3];
				index++;
				if (index == 8)
				{
					mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
					mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
					_mm256_storeu_ps(a_ptr, mA);
					_mm256_storeu_ps(b_ptr, mB);
					a_ptr += 8;
					b_ptr += 8;
					index = 0;
				}
#endif
			}
			I_ptr_prev = guide.ptr<float>(0);
			p_ptr_prev = src.ptr<float>(0) + cn;
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row - r - 1   */
		for (int i = r + 1; i < row - r - 1; i++)
		{
			mSum = _mm_setzero_ps();

			mCol_next = _mm_loadu_ps(cp_next);
			mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
			mCol_next = _mm_sub_ps(mCol_next, mTmp);
			mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
			mCol_next = _mm_add_ps(mCol_next, mTmp);
			_mm_storeu_ps(cp_next, mCol_next);

			I_ptr_prev++;
			I_ptr_next++;
			p_ptr_prev += p_cn;
			p_ptr_next += p_cn;
			cp_next += 4;

			mSum = _mm_mul_ps(mCol_next, mBorder);

			for (int j = 1; j <= r; j++)
			{
				mCol_next = _mm_loadu_ps(cp_next);
				mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
				mCol_next = _mm_sub_ps(mCol_next, mTmp);
				mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
				mCol_next = _mm_add_ps(mCol_next, mTmp);
				_mm_storeu_ps(cp_next, mCol_next);

				I_ptr_prev++;
				I_ptr_next++;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
				cp_next += 4;

				mSum = _mm_add_ps(mSum, mCol_next);
			}
			mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
			varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
			covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

			*a_ptr = covIP / (varI + eps);
			*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
			a_ptr++;
			b_ptr++;
#else
			mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
			mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
			mMeanI.m256_f32[index] = mTmp.m128_f32[0];
			mMeanP.m256_f32[index] = mTmp.m128_f32[3];
			index++;
			if (index == 8)
			{
				mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
				mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
				_mm256_storeu_ps(a_ptr, mA);
				_mm256_storeu_ps(b_ptr, mB);
				a_ptr += 8;
				b_ptr += 8;
				index = 0;
			}
#endif

			mCol_prev = _mm_loadu_ps(cp_prev);
			for (int j = 1; j <= r; j++)
			{
				mCol_next = _mm_loadu_ps(cp_next);
				mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
				mCol_next = _mm_sub_ps(mCol_next, mTmp);
				mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
				mCol_next = _mm_add_ps(mCol_next, mTmp);
				_mm_storeu_ps(cp_next, mCol_next);

				I_ptr_prev++;
				I_ptr_next++;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
				cp_next += 4;

				mSum = _mm_sub_ps(mSum, mCol_prev);
				mSum = _mm_add_ps(mSum, mCol_next);

				mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
				varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
				covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

				*a_ptr = covIP / (varI + eps);
				*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
				a_ptr++;
				b_ptr++;
#else
				mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
				mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
				mMeanI.m256_f32[index] = mTmp.m128_f32[0];
				mMeanP.m256_f32[index] = mTmp.m128_f32[3];
				index++;
				if (index == 8)
				{
					mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
					mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
					_mm256_storeu_ps(a_ptr, mA);
					_mm256_storeu_ps(b_ptr, mB);
					a_ptr += 8;
					b_ptr += 8;
					index = 0;
				}
#endif
			}
			for (int j = r + 1; j < col - r; j++)
			{
				mCol_next = _mm_loadu_ps(cp_next);
				mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
				mCol_next = _mm_sub_ps(mCol_next, mTmp);
				mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
				mCol_next = _mm_add_ps(mCol_next, mTmp);
				_mm_storeu_ps(cp_next, mCol_next);

				mCol_prev = _mm_loadu_ps(cp_prev);

				I_ptr_prev++;
				I_ptr_next++;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
				cp_prev += 4;
				cp_next += 4;

				mSum = _mm_sub_ps(mSum, mCol_prev);
				mSum = _mm_add_ps(mSum, mCol_next);

				mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
				varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
				covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

				*a_ptr = covIP / (varI + eps);
				*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
				a_ptr++;
				b_ptr++;
#else
				mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
				mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
				mMeanI.m256_f32[index] = mTmp.m128_f32[0];
				mMeanP.m256_f32[index] = mTmp.m128_f32[3];
				index++;
				if (index == 8)
				{
					mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
					mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
					_mm256_storeu_ps(a_ptr, mA);
					_mm256_storeu_ps(b_ptr, mB);
					a_ptr += 8;
					b_ptr += 8;
					index = 0;
				}
#endif
			}
			cp_next -= 4;
			mCol_next = _mm_loadu_ps(cp_next);
			for (int j = col - r; j < col; j++)
			{
				mCol_prev = _mm_loadu_ps(cp_prev);
				cp_prev += 4;

				mSum = _mm_sub_ps(mSum, mCol_prev);
				mSum = _mm_add_ps(mSum, mCol_next);

				mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
				varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
				covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

				*a_ptr = covIP / (varI + eps);
				*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
				a_ptr++;
				b_ptr++;
#else
				mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
				mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
				mMeanI.m256_f32[index] = mTmp.m128_f32[0];
				mMeanP.m256_f32[index] = mTmp.m128_f32[3];
				index++;
				if (index == 8)
				{
					mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
					mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
					_mm256_storeu_ps(a_ptr, mA);
					_mm256_storeu_ps(b_ptr, mB);
					a_ptr += 8;
					b_ptr += 8;
					index = 0;
				}
#endif
			}
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   row - r - 1 < i < row   */
		for (int i = row - r - 1; i < row; i++)
		{
			mSum = _mm_setzero_ps();

			mCol_next = _mm_loadu_ps(cp_next);
			mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
			mCol_next = _mm_sub_ps(mCol_next, mTmp);
			mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
			mCol_next = _mm_add_ps(mCol_next, mTmp);
			_mm_storeu_ps(cp_next, mCol_next);

			I_ptr_prev++;
			I_ptr_next++;
			p_ptr_prev += p_cn;
			p_ptr_next += p_cn;
			cp_next += 4;

			mSum = _mm_mul_ps(mCol_next, mBorder);

			for (int j = 1; j <= r; j++)
			{
				mCol_next = _mm_loadu_ps(cp_next);
				mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
				mCol_next = _mm_sub_ps(mCol_next, mTmp);
				mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
				mCol_next = _mm_add_ps(mCol_next, mTmp);
				_mm_storeu_ps(cp_next, mCol_next);

				I_ptr_prev++;
				I_ptr_next++;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
				cp_next += 4;

				mSum = _mm_add_ps(mSum, mCol_next);
			}
			mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
			varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
			covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

			*a_ptr = covIP / (varI + eps);
			*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
			a_ptr++;
			b_ptr++;
#else
			mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
			mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
			mMeanI.m256_f32[index] = mTmp.m128_f32[0];
			mMeanP.m256_f32[index] = mTmp.m128_f32[3];
			index++;
			if (index == 8)
			{
				mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
				mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
				_mm256_storeu_ps(a_ptr, mA);
				_mm256_storeu_ps(b_ptr, mB);
				a_ptr += 8;
				b_ptr += 8;
				index = 0;
			}
#endif

			mCol_prev = _mm_loadu_ps(cp_prev);
			for (int j = 1; j <= r; j++)
			{
				mCol_next = _mm_loadu_ps(cp_next);
				mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
				mCol_next = _mm_sub_ps(mCol_next, mTmp);
				mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
				mCol_next = _mm_add_ps(mCol_next, mTmp);
				_mm_storeu_ps(cp_next, mCol_next);

				I_ptr_prev++;
				I_ptr_next++;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
				cp_next += 4;

				mSum = _mm_sub_ps(mSum, mCol_prev);
				mSum = _mm_add_ps(mSum, mCol_next);

				mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
				varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
				covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

				*a_ptr = covIP / (varI + eps);
				*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
				a_ptr++;
				b_ptr++;
#else
				mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
				mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
				mMeanI.m256_f32[index] = mTmp.m128_f32[0];
				mMeanP.m256_f32[index] = mTmp.m128_f32[3];
				index++;
				if (index == 8)
				{
					mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
					mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
					_mm256_storeu_ps(a_ptr, mA);
					_mm256_storeu_ps(b_ptr, mB);
					a_ptr += 8;
					b_ptr += 8;
					index = 0;
				}
#endif
			}
			for (int j = r + 1; j < col - r; j++)
			{
				mCol_next = _mm_loadu_ps(cp_next);
				mTmp = _mm_setr_ps(*I_ptr_prev, *I_ptr_prev * *I_ptr_prev, *I_ptr_prev * *p_ptr_prev, *p_ptr_prev);
				mCol_next = _mm_sub_ps(mCol_next, mTmp);
				mTmp = _mm_setr_ps(*I_ptr_next, *I_ptr_next * *I_ptr_next, *I_ptr_next * *p_ptr_next, *p_ptr_next);
				mCol_next = _mm_add_ps(mCol_next, mTmp);
				_mm_storeu_ps(cp_next, mCol_next);

				mCol_prev = _mm_loadu_ps(cp_prev);

				I_ptr_prev++;
				I_ptr_next++;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
				cp_prev += 4;
				cp_next += 4;

				mSum = _mm_sub_ps(mSum, mCol_prev);
				mSum = _mm_add_ps(mSum, mCol_next);

				mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
				varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
				covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

				*a_ptr = covIP / (varI + eps);
				*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
				a_ptr++;
				b_ptr++;
#else
				mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
				mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
				mMeanI.m256_f32[index] = mTmp.m128_f32[0];
				mMeanP.m256_f32[index] = mTmp.m128_f32[3];
				index++;
				if (index == 8)
				{
					mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
					mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
					_mm256_storeu_ps(a_ptr, mA);
					_mm256_storeu_ps(b_ptr, mB);
					a_ptr += 8;
					b_ptr += 8;
					index = 0;
				}
#endif
			}
			cp_next -= 4;
			mCol_next = _mm_loadu_ps(cp_next);
			for (int j = col - r; j < col; j++)
			{
				mCol_prev = _mm_loadu_ps(cp_prev);
				cp_prev += 4;

				mSum = _mm_sub_ps(mSum, mCol_prev);
				mSum = _mm_add_ps(mSum, mCol_next);

				mTmp = _mm_mul_ps(mSum, mDiv);

#ifndef _MERGE_
				varI = ((float*)&mTmp)[1] - ((float*)&mTmp)[0] * ((float*)&mTmp)[0];
				covIP = ((float*)&mTmp)[2] - ((float*)&mTmp)[0] * ((float*)&mTmp)[3];

				*a_ptr = covIP / (varI + eps);
				*b_ptr = ((float*)&mTmp)[3] - *a_ptr * ((float*)&mTmp)[0];
				a_ptr++;
				b_ptr++;
#else
				mVar.m256_f32[index] = mTmp.m128_f32[1] - mTmp.m128_f32[0] * mTmp.m128_f32[0];
				mCov.m256_f32[index] = mTmp.m128_f32[2] - mTmp.m128_f32[0] * mTmp.m128_f32[3];
				mMeanI.m256_f32[index] = mTmp.m128_f32[0];
				mMeanP.m256_f32[index] = mTmp.m128_f32[3];
				index++;
				if (index == 8)
				{
					mA = _mm256_div_ps(mCov, _mm256_add_ps(mVar, mEps));
					mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));
					_mm256_storeu_ps(a_ptr, mA);
					_mm256_storeu_ps(b_ptr, mB);
					a_ptr += 8;
					b_ptr += 8;
					index = 0;
				}
#endif
			}
			I_ptr_next = guide.ptr<float>(row - 1);
			p_ptr_next = src.ptr<float>(row - 1) + cn;
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}

	// ab 2 q
	{
		float sumMeanA = 0.f;
		float sumMeanB = 0.f;

		float meanA = 0.f;
		float meanB = 0.f;

		Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC2);
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = cp_prev;

		float* a_ptr_next = a.ptr<float>(0);
		float* b_ptr_next = b.ptr<float>(0);
		for (int j = 0; j < col; j++)
		{
			*(cp_next++) = *a_ptr_next * (r + 1);
			a_ptr_next++;

			*(cp_next++) = *b_ptr_next * (r + 1);
			b_ptr_next++;
		}
		cp_next = cp_prev;
		for (int i = 1; i < r; i++)
		{
			for (int j = 0; j < col; j++)
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
		meanA = sumMeanA * div;
		meanB = sumMeanB * div;

		float* q_ptr = dest.ptr<float>(0) + cn;
		float* I_ptr = guide.ptr<float>(0);
		*q_ptr = meanA * *I_ptr + meanB;
		q_ptr += p_cn;
		I_ptr++;

		for (int j = 1; j <= r; j++)
		{
			*cp_next += *a_ptr_next;
			sumMeanA = sumMeanA - *cp_prev + *(cp_next++);
			a_ptr_next++;

			*cp_next += *b_ptr_next;
			sumMeanB = sumMeanB - *(cp_prev + 1) + *(cp_next++);
			b_ptr_next++;

			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			*cp_next += *a_ptr_next;
			sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
			a_ptr_next++;

			*cp_next += *b_ptr_next;
			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
			b_ptr_next++;

			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;
		}
		cp_next -= columnSum.channels();
		for (int j = col - r; j < col; j++)
		{
			sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;
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
			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;

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

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			for (int j = r + 1; j < col - r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			a_ptr_prev = a.ptr<float>(0);
			b_ptr_prev = b.ptr<float>(0);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row - r - 1   */
		for (int i = r + 1; i < row - r - 1; i++)
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
			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;

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

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			for (int j = r + 1; j < col - r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   row - r - 1 < i < row   */
		for (int i = row - r - 1; i < row; i++)
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
			meanA = sumMeanA * div;
			meanB = sumMeanB * div;

			*q_ptr = meanA * *I_ptr + meanB;
			q_ptr += p_cn;
			I_ptr++;

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

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			for (int j = r + 1; j < col - r; j++)
			{
				*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
				sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
				a_ptr_prev++;
				a_ptr_next++;

				*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
				b_ptr_prev++;
				b_ptr_next++;

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

				meanA = sumMeanA * div;
				meanB = sumMeanB * div;

				*q_ptr = meanA * *I_ptr + meanB;
				q_ptr += p_cn;
				I_ptr++;
			}
			a_ptr_next = a.ptr<float>(row - 1);
			b_ptr_next = b.ptr<float>(row - 1);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}
}


#define CALC_CONVARIANCE_SIMD()	\
	mTmp1 = _mm256_mul_ps(mSum1, mDiv);	\
	mTmp2 = _mm256_mul_ps(mSum2, mDiv);	\
	\
	bb =   ((float*)&mTmp2)[0] - ((float*)&mTmp1)[0] * ((float*)&mTmp1)[0];	\
	bg =   ((float*)&mTmp2)[1] - ((float*)&mTmp1)[0] * ((float*)&mTmp1)[1];	\
	br =   ((float*)&mTmp2)[2] - ((float*)&mTmp1)[0] * ((float*)&mTmp1)[2];	\
	gg =   ((float*)&mTmp2)[3] - ((float*)&mTmp1)[1] * ((float*)&mTmp1)[1];	\
	gr =   ((float*)&mTmp2)[4] - ((float*)&mTmp1)[1] * ((float*)&mTmp1)[2];	\
	rr =   ((float*)&mTmp2)[5] - ((float*)&mTmp1)[2] * ((float*)&mTmp1)[2];	\
	covb = ((float*)&mTmp1)[4] - ((float*)&mTmp1)[0] * ((float*)&mTmp1)[3];	\
	covg = ((float*)&mTmp1)[5] - ((float*)&mTmp1)[1] * ((float*)&mTmp1)[3];	\
	covr = ((float*)&mTmp1)[6] - ((float*)&mTmp1)[2] * ((float*)&mTmp1)[3];	\
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
	*b_ptr = ((float*)&mTmp1)[3] - (*ab_ptr * ((float*)&mTmp1)[0] + *ag_ptr * ((float*)&mTmp1)[1] + *ar_ptr * ((float*)&mTmp1)[2]);


void guidedFilter_Merge_OnePass_SIMD::filter_Guide3(int cn)
{
	Mat a, b;
	a.create(src.size(), CV_32FC3);
	b.create(src.size(), CV_32FC1);

	__m256 mDiv = _mm256_set1_ps(div);
	__m256 mBorder = _mm256_set1_ps(static_cast<float>(r + 1));

	// Ip 2 ab
	{
		__m256 mSum1 = _mm256_setzero_ps(); //  Ib,  Ig,  Ir,   p, Ipb, Ipg, Ipr,   0
		__m256 mSum2 = _mm256_setzero_ps(); //  bb,  bg,  br,  gg,  gr,  rr,   0,   0
		__m256 mMean1 = _mm256_setzero_ps();
		__m256 mMean2 = _mm256_setzero_ps();

		Mat columnSum = Mat::zeros(Size(col + 2, 1), CV_32FC(13));
		float* cp_prev1 = columnSum.ptr<float>(0);
		float* cp_next1 = cp_prev1;
		float* cp_prev2 = columnSum.ptr<float>(0) + 7;
		float* cp_next2 = cp_prev2;

		__m256 mCol_prev1 = _mm256_setzero_ps();
		__m256 mCol_next1 = _mm256_setzero_ps();
		__m256 mCol_prev2 = _mm256_setzero_ps();
		__m256 mCol_next2 = _mm256_setzero_ps();
		__m256 mTmp1 = _mm256_setzero_ps();
		__m256 mTmp2 = _mm256_setzero_ps();

		float* Ib_ptr_next = guide.ptr<float>(0);
		float* Ig_ptr_next = guide.ptr<float>(0) + 1;
		float* Ir_ptr_next = guide.ptr<float>(0) + 2;
		float* p_ptr_next = src.ptr<float>(0) + cn;

		/*   i == 0   */
		for (int j = 0; j < col; j++)
		{
			mCol_next1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
				*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
			mCol_next1 = _mm256_mul_ps(mCol_next1, mBorder);
			_mm256_storeu_ps(cp_next1, mCol_next1);
			cp_next1 += 13;

			mCol_next2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
				*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
			mCol_next2 = _mm256_mul_ps(mCol_next2, mBorder);
			_mm256_storeu_ps(cp_next2, mCol_next2);
			cp_next2 += 13;

			Ib_ptr_next += I_cn;
			Ig_ptr_next += I_cn;
			Ir_ptr_next += I_cn;
			p_ptr_next += p_cn;
		}
		cp_next1 = cp_prev1;
		cp_next2 = cp_prev2;
		for (int i = 1; i < r; i++)
		{
			for (int j = 0; j < col; j++)
			{
				mCol_next1 = _mm256_loadu_ps(cp_next1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
					*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
				mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
				_mm256_storeu_ps(cp_next1, mCol_next1);
				cp_next1 += 13;

				mCol_next2 = _mm256_loadu_ps(cp_next2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
					*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
				mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
				_mm256_storeu_ps(cp_next2, mCol_next2);
				cp_next2 += 13;

				Ib_ptr_next += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_next += p_cn;
			}
			cp_next1 = cp_prev1;
			cp_next2 = cp_prev2;
		}

		mCol_next1 = _mm256_loadu_ps(cp_next1);
		mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
			*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
		mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
		_mm256_storeu_ps(cp_next1, mCol_next1);
		cp_next1 += 13;
		mSum1 = _mm256_mul_ps(mCol_next1, mBorder);

		mCol_next2 = _mm256_loadu_ps(cp_next2);
		mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
			*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
		mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
		_mm256_storeu_ps(cp_next2, mCol_next2);
		cp_next2 += 13;
		mSum2 = _mm256_mul_ps(mCol_next2, mBorder);

		Ib_ptr_next += I_cn;
		Ig_ptr_next += I_cn;
		Ir_ptr_next += I_cn;
		p_ptr_next += p_cn;

		for (int j = 1; j <= r; j++)
		{
			mCol_next1 = _mm256_loadu_ps(cp_next1);
			mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
				*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
			mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
			_mm256_storeu_ps(cp_next1, mCol_next1);
			cp_next1 += 13;
			mSum1 = _mm256_add_ps(mSum1, mCol_next1);

			mCol_next2 = _mm256_loadu_ps(cp_next2);
			mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
				*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
			mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
			_mm256_storeu_ps(cp_next2, mCol_next2);
			cp_next2 += 13;
			mSum2 = _mm256_add_ps(mSum2, mCol_next2);

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

		CALC_CONVARIANCE_SIMD();
		ab_ptr += I_cn;
		ag_ptr += I_cn;
		ar_ptr += I_cn;
		b_ptr++;

		mCol_prev1 = _mm256_loadu_ps(cp_prev1);
		mCol_prev2 = _mm256_loadu_ps(cp_prev2);
		for (int j = 1; j <= r; j++)
		{
			mCol_next1 = _mm256_loadu_ps(cp_next1);
			mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
				*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
			mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
			_mm256_storeu_ps(cp_next1, mCol_next1);
			cp_next1 += 13;
			mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
			mSum1 = _mm256_add_ps(mSum1, mCol_next1);

			mCol_next2 = _mm256_loadu_ps(cp_next2);
			mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
				*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
			mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
			_mm256_storeu_ps(cp_next2, mCol_next2);
			cp_next2 += 13;
			mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
			mSum2 = _mm256_add_ps(mSum2, mCol_next2);

			Ib_ptr_next += I_cn;
			Ig_ptr_next += I_cn;
			Ir_ptr_next += I_cn;
			p_ptr_next += p_cn;

			CALC_CONVARIANCE_SIMD();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_prev1 = _mm256_loadu_ps(cp_prev1);
			mCol_next1 = _mm256_loadu_ps(cp_next1);
			mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
				*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
			mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
			_mm256_storeu_ps(cp_next1, mCol_next1);
			cp_prev1 += 13;
			cp_next1 += 13;
			mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
			mSum1 = _mm256_add_ps(mSum1, mCol_next1);

			mCol_prev2 = _mm256_loadu_ps(cp_prev2);
			mCol_next2 = _mm256_loadu_ps(cp_next2);
			mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
				*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
			mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
			_mm256_storeu_ps(cp_next2, mCol_next2);
			cp_prev2 += 13;
			cp_next2 += 13;
			mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
			mSum2 = _mm256_add_ps(mSum2, mCol_next2);

			Ib_ptr_next += I_cn;
			Ig_ptr_next += I_cn;
			Ir_ptr_next += I_cn;
			p_ptr_next += p_cn;

			CALC_CONVARIANCE_SIMD();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;
		}
		cp_next1 -= 13;
		cp_next2 -= 13;
		mCol_next1 = _mm256_loadu_ps(cp_next1);
		mCol_next2 = _mm256_loadu_ps(cp_next2);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev1 = _mm256_loadu_ps(cp_prev1);
			cp_prev1 += 13;
			mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
			mSum1 = _mm256_add_ps(mSum1, mCol_next1);

			mCol_prev2 = _mm256_loadu_ps(cp_prev2);
			cp_prev2 += 13;
			mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
			mSum2 = _mm256_add_ps(mSum2, mCol_next2);

			CALC_CONVARIANCE_SIMD();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;
		}
		cp_prev1 = cp_next1 = columnSum.ptr<float>(0);
		cp_prev2 = cp_next2 = columnSum.ptr<float>(0) + 7;

		float* Ib_ptr_prev = guide.ptr<float>(0);
		float* Ig_ptr_prev = guide.ptr<float>(0) + 1;
		float* Ir_ptr_prev = guide.ptr<float>(0) + 2;
		float* p_ptr_prev = src.ptr<float>(0) + cn;
		/*   0 < r <= r   */
		for (int i = 1; i <= r; i++)
		{
			mSum1 = _mm256_setzero_ps();
			mSum2 = _mm256_setzero_ps();

			mCol_next1 = _mm256_loadu_ps(cp_next1);
			mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
				*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
			mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
			mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
				*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
			mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
			_mm256_storeu_ps(cp_next1, mCol_next1);
			cp_next1 += 13;
			mSum1 = _mm256_mul_ps(mCol_next1, mBorder);

			mCol_next2 = _mm256_loadu_ps(cp_next2);
			mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
				*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
			mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
			mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
				*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
			mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
			_mm256_storeu_ps(cp_next2, mCol_next2);
			cp_next2 += 13;
			mSum2 = _mm256_mul_ps(mCol_next2, mBorder);

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
				mCol_next1 = _mm256_loadu_ps(cp_next1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
					*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
				mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
					*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
				mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
				_mm256_storeu_ps(cp_next1, mCol_next1);
				cp_next1 += 13;
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_next2 = _mm256_loadu_ps(cp_next2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
					*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
				mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
					*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
				mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
				_mm256_storeu_ps(cp_next2, mCol_next2);
				cp_next2 += 13;
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
			}
			CALC_CONVARIANCE_SIMD();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;

			mCol_prev1 = _mm256_loadu_ps(cp_prev1);
			mCol_prev2 = _mm256_loadu_ps(cp_prev2);
			for (int j = 1; j <= r; j++)
			{
				mCol_next1 = _mm256_loadu_ps(cp_next1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
					*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
				mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
					*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
				mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
				_mm256_storeu_ps(cp_next1, mCol_next1);
				cp_next1 += 13;
				mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_next2 = _mm256_loadu_ps(cp_next2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
					*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
				mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
					*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
				mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
				_mm256_storeu_ps(cp_next2, mCol_next2);
				cp_next2 += 13;
				mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_CONVARIANCE_SIMD();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			for (int j = r + 1; j < col - r; j++)
			{
				mCol_prev1 = _mm256_loadu_ps(cp_prev1);
				mCol_next1 = _mm256_loadu_ps(cp_next1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
					*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
				mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
					*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
				mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
				_mm256_storeu_ps(cp_next1, mCol_next1);
				cp_prev1 += 13;
				cp_next1 += 13;
				mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_prev2 = _mm256_loadu_ps(cp_prev2);
				mCol_next2 = _mm256_loadu_ps(cp_next2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
					*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
				mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
					*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
				mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
				_mm256_storeu_ps(cp_next2, mCol_next2);
				cp_prev2 += 13;
				cp_next2 += 13;
				mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_CONVARIANCE_SIMD();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			cp_next1 -= 13;
			cp_next2 -= 13;
			mCol_next1 = _mm256_loadu_ps(cp_next1);
			mCol_next2 = _mm256_loadu_ps(cp_next2);
			for (int j = col - r; j < col; j++)
			{
				mCol_prev1 = _mm256_loadu_ps(cp_prev1);
				cp_prev1 += 13;
				mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_prev2 = _mm256_loadu_ps(cp_prev2);
				cp_prev2 += 13;
				mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				CALC_CONVARIANCE_SIMD();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			Ib_ptr_prev = guide.ptr<float>(0);
			Ig_ptr_prev = guide.ptr<float>(0) + 1;
			Ir_ptr_prev = guide.ptr<float>(0) + 2;
			p_ptr_prev = src.ptr<float>(0) + cn;
			cp_prev1 = cp_next1 = columnSum.ptr<float>(0);
			cp_prev2 = cp_next2 = columnSum.ptr<float>(0) + 7;
		}

		/*   r < i < row - r - 1   */
		for (int i = r + 1; i < row - r - 1; i++)
		{
			mSum1 = _mm256_setzero_ps();
			mSum2 = _mm256_setzero_ps();

			mCol_next1 = _mm256_loadu_ps(cp_next1);
			mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
				*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
			mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
			mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
				*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
			mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
			_mm256_storeu_ps(cp_next1, mCol_next1);
			cp_next1 += 13;
			mSum1 = _mm256_mul_ps(mCol_next1, mBorder);

			mCol_next2 = _mm256_loadu_ps(cp_next2);
			mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
				*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
			mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
			mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
				*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
			mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
			_mm256_storeu_ps(cp_next2, mCol_next2);
			cp_next2 += 13;
			mSum2 = _mm256_mul_ps(mCol_next2, mBorder);

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
				mCol_next1 = _mm256_loadu_ps(cp_next1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
					*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
				mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
					*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
				mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
				_mm256_storeu_ps(cp_next1, mCol_next1);
				cp_next1 += 13;
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_next2 = _mm256_loadu_ps(cp_next2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
					*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
				mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
					*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
				mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
				_mm256_storeu_ps(cp_next2, mCol_next2);
				cp_next2 += 13;
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
			}
			CALC_CONVARIANCE_SIMD();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;

			mCol_prev1 = _mm256_loadu_ps(cp_prev1);
			mCol_prev2 = _mm256_loadu_ps(cp_prev2);
			for (int j = 1; j <= r; j++)
			{
				mCol_next1 = _mm256_loadu_ps(cp_next1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
					*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
				mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
					*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
				mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
				_mm256_storeu_ps(cp_next1, mCol_next1);
				cp_next1 += 13;
				mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_next2 = _mm256_loadu_ps(cp_next2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
					*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
				mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
					*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
				mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
				_mm256_storeu_ps(cp_next2, mCol_next2);
				cp_next2 += 13;
				mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_CONVARIANCE_SIMD();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			for (int j = r + 1; j < col - r; j++)
			{
				mCol_prev1 = _mm256_loadu_ps(cp_prev1);
				mCol_next1 = _mm256_loadu_ps(cp_next1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
					*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
				mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
					*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
				mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
				_mm256_storeu_ps(cp_next1, mCol_next1);
				cp_prev1 += 13;
				cp_next1 += 13;
				mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_prev2 = _mm256_loadu_ps(cp_prev2);
				mCol_next2 = _mm256_loadu_ps(cp_next2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
					*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
				mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
					*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
				mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
				_mm256_storeu_ps(cp_next2, mCol_next2);
				cp_prev2 += 13;
				cp_next2 += 13;
				mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_CONVARIANCE_SIMD();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			cp_next1 -= 13;
			cp_next2 -= 13;
			mCol_next1 = _mm256_loadu_ps(cp_next1);
			mCol_next2 = _mm256_loadu_ps(cp_next2);
			for (int j = col - r; j < col; j++)
			{
				mCol_prev1 = _mm256_loadu_ps(cp_prev1);
				cp_prev1 += 13;
				mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_prev2 = _mm256_loadu_ps(cp_prev2);
				cp_prev2 += 13;
				mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				CALC_CONVARIANCE_SIMD();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			cp_prev1 = cp_next1 = columnSum.ptr<float>(0);
			cp_prev2 = cp_next2 = columnSum.ptr<float>(0) + 7;
		}

		/*   row - r - 1 <= i < row   */
		for (int i = row - r - 1; i < row; i++)
		{
			mSum1 = _mm256_setzero_ps();
			mSum2 = _mm256_setzero_ps();

			mCol_next1 = _mm256_loadu_ps(cp_next1);
			mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
				*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
			mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
			mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
				*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
			mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
			_mm256_storeu_ps(cp_next1, mCol_next1);
			cp_next1 += 13;
			mSum1 = _mm256_mul_ps(mCol_next1, mBorder);

			mCol_next2 = _mm256_loadu_ps(cp_next2);
			mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
				*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
			mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
			mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
				*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
			mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
			_mm256_storeu_ps(cp_next2, mCol_next2);
			cp_next2 += 13;
			mSum2 = _mm256_mul_ps(mCol_next2, mBorder);

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
				mCol_next1 = _mm256_loadu_ps(cp_next1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
					*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
				mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
					*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
				mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
				_mm256_storeu_ps(cp_next1, mCol_next1);
				cp_next1 += 13;
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_next2 = _mm256_loadu_ps(cp_next2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
					*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
				mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
					*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
				mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
				_mm256_storeu_ps(cp_next2, mCol_next2);
				cp_next2 += 13;
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;
			}
			CALC_CONVARIANCE_SIMD();
			ab_ptr += I_cn;
			ag_ptr += I_cn;
			ar_ptr += I_cn;
			b_ptr++;

			mCol_prev1 = _mm256_loadu_ps(cp_prev1);
			mCol_prev2 = _mm256_loadu_ps(cp_prev2);
			for (int j = 1; j <= r; j++)
			{
				mCol_next1 = _mm256_loadu_ps(cp_next1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
					*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
				mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
					*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
				mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
				_mm256_storeu_ps(cp_next1, mCol_next1);
				cp_next1 += 13;
				mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_next2 = _mm256_loadu_ps(cp_next2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
					*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
				mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
					*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
				mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
				_mm256_storeu_ps(cp_next2, mCol_next2);
				cp_next2 += 13;
				mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_CONVARIANCE_SIMD();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			for (int j = r + 1; j < col - r; j++)
			{
				mCol_prev1 = _mm256_loadu_ps(cp_prev1);
				mCol_next1 = _mm256_loadu_ps(cp_next1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_prev, *Ig_ptr_prev, *Ir_ptr_prev, *p_ptr_prev,
					*Ib_ptr_prev * *p_ptr_prev, *Ig_ptr_prev * *p_ptr_prev, *Ir_ptr_prev * *p_ptr_prev, 0);
				mCol_next1 = _mm256_sub_ps(mCol_next1, mTmp1);
				mTmp1 = _mm256_setr_ps(*Ib_ptr_next, *Ig_ptr_next, *Ir_ptr_next, *p_ptr_next,
					*Ib_ptr_next * *p_ptr_next, *Ig_ptr_next * *p_ptr_next, *Ir_ptr_next * *p_ptr_next, 0);
				mCol_next1 = _mm256_add_ps(mCol_next1, mTmp1);
				_mm256_storeu_ps(cp_next1, mCol_next1);
				cp_prev1 += 13;
				cp_next1 += 13;
				mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_prev2 = _mm256_loadu_ps(cp_prev2);
				mCol_next2 = _mm256_loadu_ps(cp_next2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_prev * *Ib_ptr_prev, *Ib_ptr_prev * *Ig_ptr_prev, *Ib_ptr_prev * *Ir_ptr_prev,
					*Ig_ptr_prev * *Ig_ptr_prev, *Ig_ptr_prev * *Ir_ptr_prev, *Ir_ptr_prev * *Ir_ptr_prev, 0, 0);
				mCol_next2 = _mm256_sub_ps(mCol_next2, mTmp2);
				mTmp2 = _mm256_setr_ps(*Ib_ptr_next * *Ib_ptr_next, *Ib_ptr_next * *Ig_ptr_next, *Ib_ptr_next * *Ir_ptr_next,
					*Ig_ptr_next * *Ig_ptr_next, *Ig_ptr_next * *Ir_ptr_next, *Ir_ptr_next * *Ir_ptr_next, 0, 0);
				mCol_next2 = _mm256_add_ps(mCol_next2, mTmp2);
				_mm256_storeu_ps(cp_next2, mCol_next2);
				cp_prev2 += 13;
				cp_next2 += 13;
				mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				Ib_ptr_prev += I_cn;
				Ib_ptr_next += I_cn;
				Ig_ptr_prev += I_cn;
				Ig_ptr_next += I_cn;
				Ir_ptr_prev += I_cn;
				Ir_ptr_next += I_cn;
				p_ptr_prev += p_cn;
				p_ptr_next += p_cn;

				CALC_CONVARIANCE_SIMD();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			cp_next1 -= 13;
			cp_next2 -= 13;
			mCol_next1 = _mm256_loadu_ps(cp_next1);
			mCol_next2 = _mm256_loadu_ps(cp_next2);
			for (int j = col - r; j < col; j++)
			{
				mCol_prev1 = _mm256_loadu_ps(cp_prev1);
				cp_prev1 += 13;
				mSum1 = _mm256_sub_ps(mSum1, mCol_prev1);
				mSum1 = _mm256_add_ps(mSum1, mCol_next1);

				mCol_prev2 = _mm256_loadu_ps(cp_prev2);
				cp_prev2 += 13;
				mSum2 = _mm256_sub_ps(mSum2, mCol_prev2);
				mSum2 = _mm256_add_ps(mSum2, mCol_next2);

				CALC_CONVARIANCE_SIMD();
				ab_ptr += I_cn;
				ag_ptr += I_cn;
				ar_ptr += I_cn;
				b_ptr++;
			}
			Ib_ptr_next = guide.ptr<float>(row - 1);
			Ig_ptr_next = guide.ptr<float>(row - 1) + 1;
			Ir_ptr_next = guide.ptr<float>(row - 1) + 2;
			p_ptr_next = src.ptr<float>(row - 1) + cn;
			cp_prev1 = cp_next1 = columnSum.ptr<float>(0);
			cp_prev2 = cp_next2 = columnSum.ptr<float>(0) + 7;
		}
	}

	// ab 2 Ip
	{
		float sumMeanA_b = 0.f;
		float sumMeanA_g = 0.f;
		float sumMeanA_r = 0.f;
		float sumMeanB = 0.f;

		float meanA_b = 0.f;
		float meanA_g = 0.f;
		float meanA_r = 0.f;
		float meanB = 0.f;

		Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC4);
		float* cp_prev = columnSum.ptr<float>(0);
		float* cp_next = cp_prev;

		float* ab_ptr_next = a.ptr<float>(0);
		float* ag_ptr_next = ab_ptr_next + 1;
		float* ar_ptr_next = ab_ptr_next + 2;
		float* b_ptr_next = b.ptr<float>(0);
		for (int j = 0; j < col; j++)
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
			for (int j = 0; j < col; j++)
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
		meanA_b = sumMeanA_b * div;
		meanA_g = sumMeanA_g * div;
		meanA_r = sumMeanA_r * div;
		meanB = sumMeanB * div;

		float* q_ptr = dest.ptr<float>(0) + cn;
		float* I_ptr = guide.ptr<float>(0);
		*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
		q_ptr += p_cn;
		I_ptr += I_cn;

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

			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;
		}
		for (int j = r + 1; j < col - r; j++)
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

			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;
		}
		cp_next -= columnSum.channels();
		for (int j = col - r; j < col; j++)
		{
			sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;

			sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
			sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;
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
			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;

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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			for (int j = r + 1; j < col - r; j++)
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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;

				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			ab_ptr_prev = a.ptr<float>(0);
			ag_ptr_prev = ab_ptr_prev + 1;
			ar_ptr_prev = ab_ptr_prev + 2;
			b_ptr_prev = b.ptr<float>(0);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   r < i < row - r - 1   */
		for (int i = r + 1; i < row - r - 1; i++)
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
			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;

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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			for (int j = r + 1; j < col - r; j++)
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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}

		/*   row - r - 1 < i < row   */
		for (int i = row - r - 1; i < row; i++)
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
			meanA_b = sumMeanA_b * div;
			meanA_g = sumMeanA_g * div;
			meanA_r = sumMeanA_r * div;
			meanB = sumMeanB * div;

			*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
			q_ptr += p_cn;
			I_ptr += I_cn;

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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			for (int j = r + 1; j < col - r; j++)
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

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			cp_next -= columnSum.channels();
			for (int j = col - r; j < col; j++)
			{
				sumMeanA_b = sumMeanA_b - *(cp_prev++) + *cp_next;
				sumMeanA_g = sumMeanA_g - *(cp_prev++) + *(cp_next + 1);
				sumMeanA_r = sumMeanA_r - *(cp_prev++) + *(cp_next + 2);
				sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 3);

				meanA_b = sumMeanA_b * div;
				meanA_g = sumMeanA_g * div;
				meanA_r = sumMeanA_r * div;
				meanB = sumMeanB * div;

				*q_ptr = (meanA_b * *I_ptr + meanA_g * *(I_ptr + 1) + meanA_r * *(I_ptr + 2)) + meanB;
				q_ptr += p_cn;
				I_ptr += I_cn;
			}
			ab_ptr_next = a.ptr<float>(row - 1);
			ag_ptr_next = ab_ptr_next + 1;
			ar_ptr_next = ab_ptr_next + 2;
			b_ptr_next = b.ptr<float>(row - 1);
			cp_prev = cp_next = columnSum.ptr<float>(0);
		}
	}
}




void guidedFilter_Merge_OnePass_LoopFusion::filter_Guide1(int cn)
{
	Mat a(src.size(), CV_32FC1);
	Mat b(src.size(), CV_32FC1);

	float sumMeanI = 0.f;
	float sumCorrI = 0.f;
	float sumCorrIP = 0.f;
	float sumMeanP = 0.f;
	float sumMeanA = 0.f;
	float sumMeanB = 0.f;

	float meanI = 0.f;
	float corrI = 0.f;
	float corrIP = 0.f;
	float meanP = 0.f;

	Mat ipColumnSum = Mat::zeros(1, col, CV_32FC4);
	float* ipCol_ptr_prev = ipColumnSum.ptr<float>(0);
	float* ipCol_ptr_next = ipCol_ptr_prev;

	float* I_ptr_next = guide.ptr<float>(0);
	float* p_ptr_next = src.ptr<float>(0) + cn;
	for (int j = 0; j < col; j++)
	{
		*(ipCol_ptr_next++) = *I_ptr_next * (r + 1);
		*(ipCol_ptr_next++) = (*I_ptr_next * *I_ptr_next) * (r + 1);
		*(ipCol_ptr_next++) = (*I_ptr_next * *p_ptr_next) * (r + 1);
		I_ptr_next++;

		*(ipCol_ptr_next++) = *p_ptr_next * (r + 1);
		p_ptr_next += p_cn;
	}
	ipCol_ptr_next = ipCol_ptr_prev;
	for (int i = 1; i < r; i++)
	{
		for (int j = 0; j < col; j++)
		{
			*(ipCol_ptr_next++) += *I_ptr_next;
			*(ipCol_ptr_next++) += (*I_ptr_next * *I_ptr_next);
			*(ipCol_ptr_next++) += (*I_ptr_next * *p_ptr_next);
			I_ptr_next++;

			*(ipCol_ptr_next++) += *p_ptr_next;
			p_ptr_next += p_cn;
		}
		ipCol_ptr_next = ipCol_ptr_prev;
	}

	*ipCol_ptr_next += *I_ptr_next;
	sumMeanI += *(ipCol_ptr_next++) * (r + 1);

	*ipCol_ptr_next += (*I_ptr_next * *I_ptr_next);
	sumCorrI += *(ipCol_ptr_next++) * (r + 1);

	*ipCol_ptr_next += (*I_ptr_next * *p_ptr_next);
	sumCorrIP += *(ipCol_ptr_next++) * (r + 1);
	I_ptr_next++;

	*ipCol_ptr_next += *p_ptr_next;
	sumMeanP += *(ipCol_ptr_next++) * (r + 1);
	p_ptr_next += p_cn;

	for (int j = 1; j <= r; j++)
	{
		*ipCol_ptr_next += *I_ptr_next;
		sumMeanI += *(ipCol_ptr_next++);

		*ipCol_ptr_next += (*I_ptr_next * *I_ptr_next);
		sumCorrI += *(ipCol_ptr_next++);

		*ipCol_ptr_next += (*I_ptr_next * *p_ptr_next);
		sumCorrIP += *(ipCol_ptr_next++);
		I_ptr_next++;

		*ipCol_ptr_next += *p_ptr_next;
		sumMeanP += *(ipCol_ptr_next++);
		p_ptr_next += p_cn;
	}
	meanI = sumMeanI * div;
	corrI = sumCorrI * div;
	corrIP = sumCorrIP * div;
	meanP = sumMeanP * div;

	float varI = corrI - meanI * meanI;
	float covIP = corrIP - meanI * meanP;

	Mat abColumnSum = Mat::zeros(1, col, CV_32FC2);
	float* abCol_ptr_prev = abColumnSum.ptr<float>(0);
	float* abCol_ptr_next = abCol_ptr_prev;

	*abCol_ptr_next = (covIP / (varI + eps)) * (r + 1);
	*(abCol_ptr_next + 1) = (meanP - *abCol_ptr_next * meanI) * (r + 1);
	abCol_ptr_next += abColumnSum.channels();

	for (int j = 1; j <= r; j++)
	{
		*ipCol_ptr_next += *I_ptr_next;
		sumMeanI = sumMeanI - *ipCol_ptr_prev + *(ipCol_ptr_next++);

		*ipCol_ptr_next += (*I_ptr_next * *I_ptr_next);
		sumCorrI = sumCorrI - *(ipCol_ptr_prev + 1) + *(ipCol_ptr_next++);

		*ipCol_ptr_next += (*I_ptr_next * *p_ptr_next);
		sumCorrIP = sumCorrIP - *(ipCol_ptr_prev + 2) + *(ipCol_ptr_next++);
		I_ptr_next++;

		*ipCol_ptr_next += *p_ptr_next;
		sumMeanP = sumMeanP - *(ipCol_ptr_prev + 3) + *(ipCol_ptr_next++);
		p_ptr_next += p_cn;

		meanI = sumMeanI * div;
		corrI = sumCorrI * div;
		corrIP = sumCorrIP * div;
		meanP = sumMeanP * div;

		varI = corrI - meanI * meanI;
		covIP = corrIP - meanI * meanP;

		*abCol_ptr_next = (covIP / (varI + eps)) * (r + 1);
		*(abCol_ptr_next + 1) = meanP - *abCol_ptr_next * meanI;
		abCol_ptr_next += abColumnSum.channels();
	}
	for (int j = r + 1; j < col - r; j++)
	{
		*ipCol_ptr_next += *I_ptr_next;
		sumMeanI = sumMeanI - *(ipCol_ptr_prev++) + *(ipCol_ptr_next++);

		*ipCol_ptr_next += (*I_ptr_next * *I_ptr_next);
		sumCorrI = sumCorrI - *(ipCol_ptr_prev++) + *(ipCol_ptr_next++);

		*ipCol_ptr_next += (*I_ptr_next * *p_ptr_next);
		sumCorrIP = sumCorrIP - *(ipCol_ptr_prev++) + *(ipCol_ptr_next++);
		I_ptr_next++;

		*ipCol_ptr_next += *p_ptr_next;
		sumMeanP = sumMeanP - *(ipCol_ptr_prev++) + *(ipCol_ptr_next++);
		p_ptr_next += p_cn;

		meanI = sumMeanI * div;
		corrI = sumCorrI * div;
		corrIP = sumCorrIP * div;
		meanP = sumMeanP * div;

		varI = corrI - meanI * meanI;
		covIP = corrIP - meanI * meanP;

		*abCol_ptr_next = (covIP / (varI + eps)) * (r + 1);
		*(abCol_ptr_next + 1) = meanP - *abCol_ptr_next * meanI;
		abCol_ptr_next += abColumnSum.channels();
	}
	ipCol_ptr_next -= ipColumnSum.channels();
	for (int j = col - r; j < col; j++)
	{
		sumMeanI = sumMeanI - *(ipCol_ptr_prev++) + *ipCol_ptr_next;
		sumCorrI = sumCorrI - *(ipCol_ptr_prev++) + *(ipCol_ptr_next + 1);
		sumCorrIP = sumCorrIP - *(ipCol_ptr_prev++) + *(ipCol_ptr_next + 2);
		sumMeanP = sumMeanP - *(ipCol_ptr_prev++) + *(ipCol_ptr_next + 3);

		meanI = sumMeanI * div;
		corrI = sumCorrI * div;
		corrIP = sumCorrIP * div;
		meanP = sumMeanP * div;

		varI = corrI - meanI * meanI;
		covIP = corrIP - meanI * meanP;

		*abCol_ptr_next = (covIP / (varI + eps)) * (r + 1);
		*(abCol_ptr_next + 1) = meanP - *abCol_ptr_next * meanI;
		abCol_ptr_next += abColumnSum.channels();
	}
	ipCol_ptr_prev = ipCol_ptr_next = ipColumnSum.ptr<float>(0);

	float* I_ptr_prev = guide.ptr<float>(0);
	float* p_ptr_prev = src.ptr<float>(0) + cn;
	/*   0 < i < r   */
	for (int i = 1; i < r; i++)
	{
	}

	//Mat a(src.size(), CV_32FC1);
	//Mat b(src.size(), CV_32FC1);

	//// Ip 2 ab
	//{
	//	float sumMeanI = 0.f;
	//	float sumMeanP = 0.f;
	//	float sumCorrI = 0.f;
	//	float sumCorrIP = 0.f;

	//	float meanI = 0.f;
	//	float meanP = 0.f;
	//	float corrI = 0.f;
	//	float corrIP = 0.f;

	//	Mat columnSum = Mat::zeros(1, col, CV_32FC4);
	//	float* cp_prev = columnSum.ptr<float>(0);
	//	float* cp_next = cp_prev;

	//	float* I_ptr_next = guide.ptr<float>(0);
	//	float* p_ptr_next = src.ptr<float>(0) + cn;
	//	for (int j = 0; j < col; j++)
	//	{
	//		*(cp_next++) = *I_ptr_next * (r + 1);
	//		*(cp_next++) = (*I_ptr_next * *I_ptr_next) * (r + 1);
	//		*(cp_next++) = (*I_ptr_next * *p_ptr_next) * (r + 1);
	//		I_ptr_next++;

	//		*(cp_next++) = *p_ptr_next * (r + 1);
	//		p_ptr_next += p_cn;
	//	}
	//	cp_next = cp_prev;
	//	for (int i = 1; i < r; i++)
	//	{
	//		for (int j = 0; j < col; j++)
	//		{
	//			*(cp_next++) += *I_ptr_next;
	//			*(cp_next++) += (*I_ptr_next * *I_ptr_next);
	//			*(cp_next++) += (*I_ptr_next * *p_ptr_next);
	//			I_ptr_next++;

	//			*(cp_next++) += *p_ptr_next;
	//			p_ptr_next += p_cn;
	//		}
	//		cp_next = cp_prev;
	//	}

	//	*cp_next += *I_ptr_next;
	//	sumMeanI += *(cp_next++) * (r + 1);

	//	*cp_next += (*I_ptr_next * *I_ptr_next);
	//	sumCorrI += *(cp_next++) * (r + 1);

	//	*cp_next += (*I_ptr_next * *p_ptr_next);
	//	sumCorrIP += *(cp_next++) * (r + 1);
	//	I_ptr_next++;

	//	*cp_next += *p_ptr_next;
	//	sumMeanP += *(cp_next++) * (r + 1);
	//	p_ptr_next += p_cn;
	//	for (int j = 1; j <= r; j++)
	//	{
	//		*cp_next += *I_ptr_next;
	//		sumMeanI += *(cp_next++);

	//		*cp_next += (*I_ptr_next * *I_ptr_next);
	//		sumCorrI += *(cp_next++);

	//		*cp_next += (*I_ptr_next * *p_ptr_next);
	//		sumCorrIP += *(cp_next++);
	//		I_ptr_next++;

	//		*cp_next += *p_ptr_next;
	//		sumMeanP += *(cp_next++);
	//		p_ptr_next += p_cn;
	//	}
	//	meanI = sumMeanI * div;
	//	corrI = sumCorrI * div;
	//	corrIP = sumCorrIP * div;
	//	meanP = sumMeanP * div;

	//	float varI = corrI - meanI * meanI;
	//	float covIP = corrIP - meanI * meanP;

	//	float* a_ptr = a.ptr<float>(0);
	//	float* b_ptr = b.ptr<float>(0);
	//	*a_ptr = covIP / (varI + eps);
	//	*b_ptr = meanP - *a_ptr * meanI;
	//	a_ptr++;
	//	b_ptr++;

	//	for (int j = 1; j <= r; j++)
	//	{
	//		*cp_next += *I_ptr_next;
	//		sumMeanI = sumMeanI - *cp_prev + *(cp_next++);

	//		*cp_next += (*I_ptr_next * *I_ptr_next);
	//		sumCorrI = sumCorrI - *(cp_prev + 1) + *(cp_next++);

	//		*cp_next += (*I_ptr_next * *p_ptr_next);
	//		sumCorrIP = sumCorrIP - *(cp_prev + 2) + *(cp_next++);
	//		I_ptr_next++;

	//		*cp_next += *p_ptr_next;
	//		sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
	//		p_ptr_next += p_cn;

	//		meanI = sumMeanI * div;
	//		corrI = sumCorrI * div;
	//		corrIP = sumCorrIP * div;
	//		meanP = sumMeanP * div;

	//		varI = corrI - meanI * meanI;
	//		covIP = corrIP - meanI * meanP;
	//		*a_ptr = covIP / (varI + eps);
	//		*b_ptr = meanP - *a_ptr * meanI;
	//		a_ptr++;
	//		b_ptr++;
	//	}
	//	for (int j = r + 1; j < col - r; j++)
	//	{
	//		*cp_next += *I_ptr_next;
	//		sumMeanI = sumMeanI - *(cp_prev++) + *(cp_next++);

	//		*cp_next += (*I_ptr_next * *I_ptr_next);
	//		sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next++);

	//		*cp_next += (*I_ptr_next * *p_ptr_next);
	//		sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next++);
	//		I_ptr_next++;

	//		*cp_next += *p_ptr_next;
	//		sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
	//		p_ptr_next += p_cn;

	//		meanI = sumMeanI * div;
	//		corrI = sumCorrI * div;
	//		corrIP = sumCorrIP * div;
	//		meanP = sumMeanP * div;

	//		varI = corrI - meanI * meanI;
	//		covIP = corrIP - meanI * meanP;
	//		*a_ptr = covIP / (varI + eps);
	//		*b_ptr = meanP - *a_ptr * meanI;
	//		a_ptr++;
	//		b_ptr++;
	//	}
	//	cp_next -= columnSum.channels();
	//	for (int j = col - r; j < col; j++)
	//	{
	//		sumMeanI = sumMeanI - *(cp_prev++) + *cp_next;
	//		sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next + 1);
	//		sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next + 2);
	//		sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);

	//		meanI = sumMeanI * div;
	//		corrI = sumCorrI * div;
	//		corrIP = sumCorrIP * div;
	//		meanP = sumMeanP * div;

	//		varI = corrI - meanI * meanI;
	//		covIP = corrIP - meanI * meanP;
	//		*a_ptr = covIP / (varI + eps);
	//		*b_ptr = meanP - *a_ptr * meanI;
	//		a_ptr++;
	//		b_ptr++;
	//	}
	//	cp_prev = cp_next = columnSum.ptr<float>(0);

	//	float* I_ptr_prev = guide.ptr<float>(0);
	//	float* p_ptr_prev = src.ptr<float>(0) + cn;
	//	/*   0 < i <= r   */
	//	for (int i = 1; i <= r; i++)
	//	{
	//		sumMeanI = 0.f;
	//		sumCorrI = 0.f;
	//		sumCorrIP = 0.f;
	//		sumMeanP = 0.f;

	//		*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//		sumMeanI += *(cp_next++) * (r + 1);

	//		*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//		sumCorrI += *(cp_next++) * (r + 1);

	//		*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//		sumCorrIP += *(cp_next++) * (r + 1);
	//		I_ptr_prev++;
	//		I_ptr_next++;

	//		*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//		sumMeanP += *(cp_next++) * (r + 1);
	//		p_ptr_prev += p_cn;
	//		p_ptr_next += p_cn;
	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//			sumMeanI += *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//			sumCorrI += *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//			sumCorrIP += *(cp_next++);
	//			I_ptr_prev++;
	//			I_ptr_next++;

	//			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//			sumMeanP += *(cp_next++);
	//			p_ptr_prev += p_cn;
	//			p_ptr_next += p_cn;
	//		}
	//		meanI = sumMeanI * div;
	//		corrI = sumCorrI * div;
	//		corrIP = sumCorrIP * div;
	//		meanP = sumMeanP * div;

	//		varI = corrI - meanI * meanI;
	//		covIP = corrIP - meanI * meanP;
	//		*a_ptr = covIP / (varI + eps);
	//		*b_ptr = meanP - *a_ptr * meanI;
	//		a_ptr++;
	//		b_ptr++;

	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//			sumMeanI = sumMeanI - *cp_prev + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//			sumCorrI = sumCorrI - *(cp_prev + 1) + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//			sumCorrIP = sumCorrIP - *(cp_prev + 2) + *(cp_next++);
	//			I_ptr_prev++;
	//			I_ptr_next++;

	//			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//			sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
	//			p_ptr_prev += p_cn;
	//			p_ptr_next += p_cn;

	//			meanI = sumMeanI * div;
	//			corrI = sumCorrI * div;
	//			corrIP = sumCorrIP * div;
	//			meanP = sumMeanP * div;

	//			varI = corrI - meanI * meanI;
	//			covIP = corrIP - meanI * meanP;
	//			*a_ptr = covIP / (varI + eps);
	//			*b_ptr = meanP - *a_ptr * meanI;
	//			a_ptr++;
	//			b_ptr++;
	//		}
	//		for (int j = r + 1; j < col - r; j++)
	//		{
	//			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//			sumMeanI = sumMeanI - *(cp_prev++) + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//			sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//			sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next++);
	//			I_ptr_prev++;
	//			I_ptr_next++;

	//			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//			sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
	//			p_ptr_prev += p_cn;
	//			p_ptr_next += p_cn;

	//			meanI = sumMeanI * div;
	//			corrI = sumCorrI * div;
	//			corrIP = sumCorrIP * div;
	//			meanP = sumMeanP * div;

	//			varI = corrI - meanI * meanI;
	//			covIP = corrIP - meanI * meanP;
	//			*a_ptr = covIP / (varI + eps);
	//			*b_ptr = meanP - *a_ptr * meanI;
	//			a_ptr++;
	//			b_ptr++;
	//		}
	//		cp_next -= columnSum.channels();
	//		for (int j = col - r; j < col; j++)
	//		{
	//			sumMeanI = sumMeanI - *(cp_prev++) + *cp_next;
	//			sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next + 1);
	//			sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next + 2);
	//			sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);

	//			meanI = sumMeanI * div;
	//			corrI = sumCorrI * div;
	//			corrIP = sumCorrIP * div;
	//			meanP = sumMeanP * div;

	//			varI = corrI - meanI * meanI;
	//			covIP = corrIP - meanI * meanP;
	//			*a_ptr = covIP / (varI + eps);
	//			*b_ptr = meanP - *a_ptr * meanI;
	//			a_ptr++;
	//			b_ptr++;
	//		}
	//		I_ptr_prev = guide.ptr<float>(0);
	//		p_ptr_prev = src.ptr<float>(0) + cn;
	//		cp_prev = cp_next = columnSum.ptr<float>(0);
	//	}

	//	/*   r < i < row - r - 1   */
	//	for (int i = r + 1; i < row - r - 1; i++)
	//	{
	//		sumMeanI = 0.f;
	//		sumCorrI = 0.f;
	//		sumCorrIP = 0.f;
	//		sumMeanP = 0.f;

	//		*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//		sumMeanI += *(cp_next++) * (r + 1);

	//		*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//		sumCorrI += *(cp_next++) * (r + 1);

	//		*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//		sumCorrIP += *(cp_next++) * (r + 1);
	//		I_ptr_prev++;
	//		I_ptr_next++;

	//		*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//		sumMeanP += *(cp_next++) * (r + 1);
	//		p_ptr_prev += p_cn;
	//		p_ptr_next += p_cn;
	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//			sumMeanI += *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//			sumCorrI += *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//			sumCorrIP += *(cp_next++);
	//			I_ptr_prev++;
	//			I_ptr_next++;

	//			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//			sumMeanP += *(cp_next++);
	//			p_ptr_prev += p_cn;
	//			p_ptr_next += p_cn;
	//		}
	//		meanI = sumMeanI * div;
	//		corrI = sumCorrI * div;
	//		corrIP = sumCorrIP * div;
	//		meanP = sumMeanP * div;

	//		varI = corrI - meanI * meanI;
	//		covIP = corrIP - meanI * meanP;
	//		*a_ptr = covIP / (varI + eps);
	//		*b_ptr = meanP - *a_ptr * meanI;
	//		a_ptr++;
	//		b_ptr++;

	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//			sumMeanI = sumMeanI - *cp_prev + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//			sumCorrI = sumCorrI - *(cp_prev + 1) + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//			sumCorrIP = sumCorrIP - *(cp_prev + 2) + *(cp_next++);
	//			I_ptr_prev++;
	//			I_ptr_next++;

	//			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//			sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
	//			p_ptr_prev += p_cn;
	//			p_ptr_next += p_cn;

	//			meanI = sumMeanI * div;
	//			corrI = sumCorrI * div;
	//			corrIP = sumCorrIP * div;
	//			meanP = sumMeanP * div;

	//			varI = corrI - meanI * meanI;
	//			covIP = corrIP - meanI * meanP;
	//			*a_ptr = covIP / (varI + eps);
	//			*b_ptr = meanP - *a_ptr * meanI;
	//			a_ptr++;
	//			b_ptr++;
	//		}
	//		for (int j = r + 1; j < col - r; j++)
	//		{
	//			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//			sumMeanI = sumMeanI - *(cp_prev++) + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//			sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//			sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next++);
	//			I_ptr_prev++;
	//			I_ptr_next++;

	//			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//			sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
	//			p_ptr_prev += p_cn;
	//			p_ptr_next += p_cn;

	//			meanI = sumMeanI * div;
	//			corrI = sumCorrI * div;
	//			corrIP = sumCorrIP * div;
	//			meanP = sumMeanP * div;

	//			varI = corrI - meanI * meanI;
	//			covIP = corrIP - meanI * meanP;
	//			*a_ptr = covIP / (varI + eps);
	//			*b_ptr = meanP - *a_ptr * meanI;
	//			a_ptr++;
	//			b_ptr++;
	//		}
	//		cp_next -= columnSum.channels();
	//		for (int j = col - r; j < col; j++)
	//		{
	//			sumMeanI = sumMeanI - *(cp_prev++) + *cp_next;
	//			sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next + 1);
	//			sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next + 2);
	//			sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);

	//			meanI = sumMeanI * div;
	//			corrI = sumCorrI * div;
	//			corrIP = sumCorrIP * div;
	//			meanP = sumMeanP * div;

	//			varI = corrI - meanI * meanI;
	//			covIP = corrIP - meanI * meanP;
	//			*a_ptr = covIP / (varI + eps);
	//			*b_ptr = meanP - *a_ptr * meanI;
	//			a_ptr++;
	//			b_ptr++;
	//		}
	//		cp_prev = cp_next = columnSum.ptr<float>(0);
	//	}

	//	/*   row - r - 1 <= i < row   */
	//	for (int i = row - r - 1; i < row; i++)
	//	{
	//		sumMeanI = 0.f;
	//		sumCorrI = 0.f;
	//		sumCorrIP = 0.f;
	//		sumMeanP = 0.f;

	//		*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//		sumMeanI += *(cp_next++) * (r + 1);

	//		*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//		sumCorrI += *(cp_next++) * (r + 1);

	//		*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//		sumCorrIP += *(cp_next++) * (r + 1);
	//		I_ptr_prev++;
	//		I_ptr_next++;

	//		*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//		sumMeanP += *(cp_next++) * (r + 1);
	//		p_ptr_prev += p_cn;
	//		p_ptr_next += p_cn;
	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//			sumMeanI += *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//			sumCorrI += *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//			sumCorrIP += *(cp_next++);
	//			I_ptr_prev++;
	//			I_ptr_next++;

	//			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//			sumMeanP += *(cp_next++);
	//			p_ptr_prev += p_cn;
	//			p_ptr_next += p_cn;
	//		}
	//		meanI = sumMeanI * div;
	//		corrI = sumCorrI * div;
	//		corrIP = sumCorrIP * div;
	//		meanP = sumMeanP * div;

	//		varI = corrI - meanI * meanI;
	//		covIP = corrIP - meanI * meanP;
	//		*a_ptr = covIP / (varI + eps);
	//		*b_ptr = meanP - *a_ptr * meanI;
	//		a_ptr++;
	//		b_ptr++;

	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//			sumMeanI = sumMeanI - *cp_prev + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//			sumCorrI = sumCorrI - *(cp_prev + 1) + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//			sumCorrIP = sumCorrIP - *(cp_prev + 2) + *(cp_next++);
	//			I_ptr_prev++;
	//			I_ptr_next++;

	//			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//			sumMeanP = sumMeanP - *(cp_prev + 3) + *(cp_next++);
	//			p_ptr_prev += p_cn;
	//			p_ptr_next += p_cn;

	//			meanI = sumMeanI * div;
	//			corrI = sumCorrI * div;
	//			corrIP = sumCorrIP * div;
	//			meanP = sumMeanP * div;

	//			varI = corrI - meanI * meanI;
	//			covIP = corrIP - meanI * meanP;
	//			*a_ptr = covIP / (varI + eps);
	//			*b_ptr = meanP - *a_ptr * meanI;
	//			a_ptr++;
	//			b_ptr++;
	//		}
	//		for (int j = r + 1; j < col - r; j++)
	//		{
	//			*cp_next = *cp_next - *I_ptr_prev + *I_ptr_next;
	//			sumMeanI = sumMeanI - *(cp_prev++) + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *I_ptr_prev) + (*I_ptr_next * *I_ptr_next);
	//			sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next++);

	//			*cp_next = *cp_next - (*I_ptr_prev * *p_ptr_prev) + (*I_ptr_next * *p_ptr_next);
	//			sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next++);
	//			I_ptr_prev++;
	//			I_ptr_next++;

	//			*cp_next = *cp_next - *p_ptr_prev + *p_ptr_next;
	//			sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next++);
	//			p_ptr_prev += p_cn;
	//			p_ptr_next += p_cn;

	//			meanI = sumMeanI * div;
	//			corrI = sumCorrI * div;
	//			corrIP = sumCorrIP * div;
	//			meanP = sumMeanP * div;

	//			varI = corrI - meanI * meanI;
	//			covIP = corrIP - meanI * meanP;
	//			*a_ptr = covIP / (varI + eps);
	//			*b_ptr = meanP - *a_ptr * meanI;
	//			a_ptr++;
	//			b_ptr++;
	//		}
	//		cp_next -= columnSum.channels();
	//		for (int j = col - r; j < col; j++)
	//		{
	//			sumMeanI = sumMeanI - *(cp_prev++) + *cp_next;
	//			sumCorrI = sumCorrI - *(cp_prev++) + *(cp_next + 1);
	//			sumCorrIP = sumCorrIP - *(cp_prev++) + *(cp_next + 2);
	//			sumMeanP = sumMeanP - *(cp_prev++) + *(cp_next + 3);

	//			meanI = sumMeanI * div;
	//			corrI = sumCorrI * div;
	//			corrIP = sumCorrIP * div;
	//			meanP = sumMeanP * div;

	//			varI = corrI - meanI * meanI;
	//			covIP = corrIP - meanI * meanP;
	//			*a_ptr = covIP / (varI + eps);
	//			*b_ptr = meanP - *a_ptr * meanI;
	//			a_ptr++;
	//			b_ptr++;
	//		}
	//		I_ptr_next = guide.ptr<float>(row - 1);
	//		p_ptr_next = src.ptr<float>(row - 1) + cn;
	//		cp_prev = cp_next = columnSum.ptr<float>(0);
	//	}
	//}

	//// ab 2 q
	//{
	//	float sumMeanA = 0.f;
	//	float sumMeanB = 0.f;

	//	float meanA = 0.f;
	//	float meanB = 0.f;

	//	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC2);
	//	float* cp_prev = columnSum.ptr<float>(0);
	//	float* cp_next = cp_prev;

	//	float* a_ptr_next = a.ptr<float>(0);
	//	float* b_ptr_next = b.ptr<float>(0);
	//	for (int j = 0; j < col; j++)
	//	{
	//		*(cp_next++) = *a_ptr_next * (r + 1);
	//		a_ptr_next++;

	//		*(cp_next++) = *b_ptr_next * (r + 1);
	//		b_ptr_next++;
	//	}
	//	cp_next = cp_prev;
	//	for (int i = 1; i < r; i++)
	//	{
	//		for (int j = 0; j < col; j++)
	//		{
	//			*(cp_next++) += *a_ptr_next;
	//			a_ptr_next++;

	//			*(cp_next++) += *b_ptr_next;
	//			b_ptr_next++;
	//		}
	//		cp_next = cp_prev;
	//	}

	//	*cp_next += *a_ptr_next;
	//	sumMeanA += *(cp_next++) * (r + 1);
	//	a_ptr_next++;

	//	*cp_next += *b_ptr_next;
	//	sumMeanB += *(cp_next++) * (r + 1);
	//	b_ptr_next++;

	//	for (int j = 1; j <= r; j++)
	//	{
	//		*cp_next += *a_ptr_next;
	//		sumMeanA += *(cp_next++);
	//		a_ptr_next++;

	//		*cp_next += *b_ptr_next;
	//		sumMeanB += *(cp_next++);
	//		b_ptr_next++;
	//	}
	//	meanA = sumMeanA * div;
	//	meanB = sumMeanB * div;

	//	float* q_ptr = dest.ptr<float>(0) + cn;
	//	float* I_ptr = guide.ptr<float>(0);
	//	*q_ptr = meanA * *I_ptr + meanB;
	//	q_ptr += p_cn;
	//	I_ptr++;

	//	for (int j = 1; j <= r; j++)
	//	{
	//		*cp_next += *a_ptr_next;
	//		sumMeanA = sumMeanA - *cp_prev + *(cp_next++);
	//		a_ptr_next++;

	//		*cp_next += *b_ptr_next;
	//		sumMeanB = sumMeanB - *(cp_prev + 1) + *(cp_next++);
	//		b_ptr_next++;

	//		meanA = sumMeanA * div;
	//		meanB = sumMeanB * div;

	//		*q_ptr = meanA * *I_ptr + meanB;
	//		q_ptr += p_cn;
	//		I_ptr++;
	//	}
	//	for (int j = r + 1; j < col - r; j++)
	//	{
	//		*cp_next += *a_ptr_next;
	//		sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
	//		a_ptr_next++;

	//		*cp_next += *b_ptr_next;
	//		sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
	//		b_ptr_next++;

	//		meanA = sumMeanA * div;
	//		meanB = sumMeanB * div;

	//		*q_ptr = meanA * *I_ptr + meanB;
	//		q_ptr += p_cn;
	//		I_ptr++;
	//	}
	//	cp_next -= columnSum.channels();
	//	for (int j = col - r; j < col; j++)
	//	{
	//		sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
	//		sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

	//		meanA = sumMeanA * div;
	//		meanB = sumMeanB * div;

	//		*q_ptr = meanA * *I_ptr + meanB;
	//		q_ptr += p_cn;
	//		I_ptr++;
	//	}
	//	cp_prev = cp_next = columnSum.ptr<float>(0);

	//	float* a_ptr_prev = a.ptr<float>(0);
	//	float* b_ptr_prev = b.ptr<float>(0);
	//	/*   0 < i <= r   */
	//	for (int i = 1; i <= r; i++)
	//	{
	//		sumMeanA = 0.f;
	//		sumMeanB = 0.f;

	//		*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//		sumMeanA += *(cp_next++) * (r + 1);
	//		a_ptr_prev++;
	//		a_ptr_next++;

	//		*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//		sumMeanB += *(cp_next++) * (r + 1);
	//		b_ptr_prev++;
	//		b_ptr_next++;
	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//			sumMeanA += *(cp_next++);
	//			a_ptr_prev++;
	//			a_ptr_next++;

	//			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//			sumMeanB += *(cp_next++);
	//			b_ptr_prev++;
	//			b_ptr_next++;
	//		}
	//		meanA = sumMeanA * div;
	//		meanB = sumMeanB * div;

	//		*q_ptr = meanA * *I_ptr + meanB;
	//		q_ptr += p_cn;
	//		I_ptr++;

	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//			sumMeanA = sumMeanA - *cp_prev + *(cp_next++);
	//			a_ptr_prev++;
	//			a_ptr_next++;

	//			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//			sumMeanB = sumMeanB - *(cp_prev + 1) + *(cp_next++);
	//			b_ptr_prev++;
	//			b_ptr_next++;

	//			meanA = sumMeanA * div;
	//			meanB = sumMeanB * div;

	//			*q_ptr = meanA * *I_ptr + meanB;
	//			q_ptr += p_cn;
	//			I_ptr++;
	//		}
	//		for (int j = r + 1; j < col - r; j++)
	//		{
	//			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//			sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
	//			a_ptr_prev++;
	//			a_ptr_next++;

	//			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
	//			b_ptr_prev++;
	//			b_ptr_next++;

	//			meanA = sumMeanA * div;
	//			meanB = sumMeanB * div;

	//			*q_ptr = meanA * *I_ptr + meanB;
	//			q_ptr += p_cn;
	//			I_ptr++;
	//		}
	//		cp_next -= columnSum.channels();
	//		for (int j = col - r; j < col; j++)
	//		{
	//			sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
	//			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

	//			meanA = sumMeanA * div;
	//			meanB = sumMeanB * div;

	//			*q_ptr = meanA * *I_ptr + meanB;
	//			q_ptr += p_cn;
	//			I_ptr++;
	//		}
	//		a_ptr_prev = a.ptr<float>(0);
	//		b_ptr_prev = b.ptr<float>(0);
	//		cp_prev = cp_next = columnSum.ptr<float>(0);
	//	}

	//	/*   r < i < row - r - 1   */
	//	for (int i = r + 1; i < row - r - 1; i++)
	//	{
	//		sumMeanA = 0.f;
	//		sumMeanB = 0.f;

	//		*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//		sumMeanA += *(cp_next++) * (r + 1);
	//		a_ptr_prev++;
	//		a_ptr_next++;

	//		*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//		sumMeanB += *(cp_next++) * (r + 1);
	//		b_ptr_prev++;
	//		b_ptr_next++;
	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//			sumMeanA += *(cp_next++);
	//			a_ptr_prev++;
	//			a_ptr_next++;

	//			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//			sumMeanB += *(cp_next++);
	//			b_ptr_prev++;
	//			b_ptr_next++;
	//		}
	//		meanA = sumMeanA * div;
	//		meanB = sumMeanB * div;

	//		*q_ptr = meanA * *I_ptr + meanB;
	//		q_ptr += p_cn;
	//		I_ptr++;

	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//			sumMeanA = sumMeanA - *cp_prev + *(cp_next++);
	//			a_ptr_prev++;
	//			a_ptr_next++;

	//			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//			sumMeanB = sumMeanB - *(cp_prev + 1) + *(cp_next++);
	//			b_ptr_prev++;
	//			b_ptr_next++;

	//			meanA = sumMeanA * div;
	//			meanB = sumMeanB * div;

	//			*q_ptr = meanA * *I_ptr + meanB;
	//			q_ptr += p_cn;
	//			I_ptr++;
	//		}
	//		for (int j = r + 1; j < col - r; j++)
	//		{
	//			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//			sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
	//			a_ptr_prev++;
	//			a_ptr_next++;

	//			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
	//			b_ptr_prev++;
	//			b_ptr_next++;

	//			meanA = sumMeanA * div;
	//			meanB = sumMeanB * div;

	//			*q_ptr = meanA * *I_ptr + meanB;
	//			q_ptr += p_cn;
	//			I_ptr++;
	//		}
	//		cp_next -= columnSum.channels();
	//		for (int j = col - r; j < col; j++)
	//		{
	//			sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
	//			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

	//			meanA = sumMeanA * div;
	//			meanB = sumMeanB * div;

	//			*q_ptr = meanA * *I_ptr + meanB;
	//			q_ptr += p_cn;
	//			I_ptr++;
	//		}
	//		cp_prev = cp_next = columnSum.ptr<float>(0);
	//	}

	//	/*   row - r - 1 < i < row   */
	//	for (int i = row - r - 1; i < row; i++)
	//	{
	//		sumMeanA = 0.f;
	//		sumMeanB = 0.f;

	//		*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//		sumMeanA += *(cp_next++) * (r + 1);
	//		a_ptr_prev++;
	//		a_ptr_next++;

	//		*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//		sumMeanB += *(cp_next++) * (r + 1);
	//		b_ptr_prev++;
	//		b_ptr_next++;
	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//			sumMeanA += *(cp_next++);
	//			a_ptr_prev++;
	//			a_ptr_next++;

	//			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//			sumMeanB += *(cp_next++);
	//			b_ptr_prev++;
	//			b_ptr_next++;
	//		}
	//		meanA = sumMeanA * div;
	//		meanB = sumMeanB * div;

	//		*q_ptr = meanA * *I_ptr + meanB;
	//		q_ptr += p_cn;
	//		I_ptr++;

	//		for (int j = 1; j <= r; j++)
	//		{
	//			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//			sumMeanA = sumMeanA - *cp_prev + *(cp_next++);
	//			a_ptr_prev++;
	//			a_ptr_next++;

	//			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//			sumMeanB = sumMeanB - *(cp_prev + 1) + *(cp_next++);
	//			b_ptr_prev++;
	//			b_ptr_next++;

	//			meanA = sumMeanA * div;
	//			meanB = sumMeanB * div;

	//			*q_ptr = meanA * *I_ptr + meanB;
	//			q_ptr += p_cn;
	//			I_ptr++;
	//		}
	//		for (int j = r + 1; j < col - r; j++)
	//		{
	//			*cp_next = *cp_next - *a_ptr_prev + *a_ptr_next;
	//			sumMeanA = sumMeanA - *(cp_prev++) + *(cp_next++);
	//			a_ptr_prev++;
	//			a_ptr_next++;

	//			*cp_next = *cp_next - *b_ptr_prev + *b_ptr_next;
	//			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next++);
	//			b_ptr_prev++;
	//			b_ptr_next++;

	//			meanA = sumMeanA * div;
	//			meanB = sumMeanB * div;

	//			*q_ptr = meanA * *I_ptr + meanB;
	//			q_ptr += p_cn;
	//			I_ptr++;
	//		}
	//		cp_next -= columnSum.channels();
	//		for (int j = col - r; j < col; j++)
	//		{
	//			sumMeanA = sumMeanA - *(cp_prev++) + *cp_next;
	//			sumMeanB = sumMeanB - *(cp_prev++) + *(cp_next + 1);

	//			meanA = sumMeanA * div;
	//			meanB = sumMeanB * div;

	//			*q_ptr = meanA * *I_ptr + meanB;
	//			q_ptr += p_cn;
	//			I_ptr++;
	//		}
	//		a_ptr_next = a.ptr<float>(row - 1);
	//		b_ptr_next = b.ptr<float>(row - 1);
	//		cp_prev = cp_next = columnSum.ptr<float>(0);
	//	}
	//}
}



void guidedFilter_Merge_OnePass_LoopFusion::filter_Guide3(int cn)
{

}
