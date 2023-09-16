#if 0
void FastGuided(cv::Mat& src, cv::Mat& guidance, cv::Mat& dest, int radius, float eps);
void FastGuided64F(cv::Mat& src, cv::Mat& guidance, cv::Mat& dest, int radius, double eps);

//local imp
class GuidedImageUpsample
{
	std::vector<cv::Mat> v;//src
	std::vector<cv::Mat> srcf;

	std::vector<cv::Mat> I;//guide
	cv::Mat resize_If_b;
	cv::Mat resize_If_g;
	cv::Mat resize_If_r;
	cv::Mat If_b;
	cv::Mat If_g;
	cv::Mat If_r;

	cv::Mat mean_I_b;
	cv::Mat mean_I_g;
	cv::Mat mean_I_r;

	cv::Mat corr_I_bb;
	cv::Mat corr_I_bg;
	cv::Mat corr_I_br;
	cv::Mat corr_I_gg;
	cv::Mat corr_I_gr;
	cv::Mat corr_I_rr;

	cv::Mat var_I_bb;
	cv::Mat var_I_bg;
	cv::Mat var_I_br;
	cv::Mat var_I_gg;
	cv::Mat var_I_gr;
	cv::Mat var_I_rr;
	cv::Mat temp;

	std::vector<cv::Mat> mean_p;
	std::vector<cv::Mat> corr_Ip_b;
	std::vector<cv::Mat> corr_Ip_g;
	std::vector<cv::Mat> corr_Ip_r;
	std::vector<cv::Mat> cov_Ip_b;
	std::vector<cv::Mat> cov_Ip_g;
	std::vector<cv::Mat> cov_Ip_r;
	std::vector<cv::Mat> a_b;
	std::vector<cv::Mat> a_g;
	std::vector<cv::Mat> a_r;
	std::vector<cv::Mat> b;

	std::vector<cv::Mat> resize_mean_a_b;
	std::vector<cv::Mat> resize_mean_a_g;
	std::vector<cv::Mat> resize_mean_a_r;
	std::vector<cv::Mat> resize_mean_b;
	std::vector<cv::Mat> resize_q;

	std::vector<cv::Mat> d;
public:
	void upsample64f(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, const double eps);
	void upsample32f(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, const float eps);
	void filter32f(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, const float eps);
};

void GuidedImageUpsample::upsample64f(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, const double eps)
{
	//const int downsampleMethod = INTER_LANCZOS4;
	const int downsampleMethod = INTER_NEAREST;
	const int upsampleMethod = INTER_CUBIC;

	const int scale = guide.cols / src.cols;
	const Size upsize = guide.size();
	const Size downsize = src.size();

	const int imtype = CV_64F;
	const int ddepth = CV_64F;

	const Size ksize(2 * r + 1, 2 * r + 1);

	const int size = downsize.area();

	const Point anchor(-1, -1);

	v.resize(3);
	srcf.resize(3);

	I.resize(3);
	resize_If_b.create(upsize, imtype);
	resize_If_g.create(upsize, imtype);
	resize_If_r.create(upsize, imtype);
	If_b.create(downsize, imtype);
	If_g.create(downsize, imtype);
	If_r.create(downsize, imtype);

	mean_I_b.create(downsize, imtype);
	mean_I_g.create(downsize, imtype);
	mean_I_r.create(downsize, imtype);

	corr_I_bb.create(downsize, imtype);
	corr_I_bg.create(downsize, imtype);
	corr_I_br.create(downsize, imtype);
	corr_I_gg.create(downsize, imtype);
	corr_I_gr.create(downsize, imtype);
	corr_I_rr.create(downsize, imtype);

	var_I_bb.create(downsize, imtype);
	var_I_bg.create(downsize, imtype);
	var_I_br.create(downsize, imtype);
	var_I_gg.create(downsize, imtype);
	var_I_gr.create(downsize, imtype);
	var_I_rr.create(downsize, imtype);
	temp.create(downsize, imtype);

	mean_p.resize(3);
	corr_Ip_b.resize(3);
	corr_Ip_g.resize(3);
	corr_Ip_r.resize(3);
	cov_Ip_b.resize(3);
	cov_Ip_g.resize(3);
	cov_Ip_r.resize(3);
	a_b.resize(3);
	a_g.resize(3);
	a_r.resize(3);
	b.resize(3);

	resize_mean_a_b.resize(3);
	resize_mean_a_g.resize(3);
	resize_mean_a_r.resize(3);
	resize_mean_b.resize(3);
	resize_q.resize(3);

	d.resize(3);

	cv::split(src, v);
	cv::split(guide, I);
	{
		//cvt8u32f(src, srcf);
		v[0].convertTo(srcf[0], CV_64F);
		v[1].convertTo(srcf[1], CV_64F);
		v[2].convertTo(srcf[2], CV_64F);
		I[0].convertTo(resize_If_b, CV_64F);
		I[1].convertTo(resize_If_g, CV_64F);
		I[2].convertTo(resize_If_r, CV_64F);
	}


	//downsample input image
	cv::resize(resize_If_b, If_b, downsize, 0.0, 0.0, downsampleMethod);
	cv::resize(resize_If_g, If_g, downsize, 0.0, 0.0, downsampleMethod);
	cv::resize(resize_If_r, If_r, downsize, 0.0, 0.0, downsampleMethod);

	cv::boxFilter(If_b, mean_I_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	cv::boxFilter(If_g, mean_I_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	cv::boxFilter(If_r, mean_I_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, If_b, corr_I_bb);
	boxFilter(corr_I_bb, corr_I_bb, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_b, If_g, corr_I_bg);
	boxFilter(corr_I_bg, corr_I_bg, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_b, If_r, corr_I_br);
	boxFilter(corr_I_br, corr_I_br, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, If_g, corr_I_gg);
	boxFilter(corr_I_gg, corr_I_gg, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, If_r, corr_I_gr);
	boxFilter(corr_I_gr, corr_I_gr, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, If_r, corr_I_rr);
	boxFilter(corr_I_rr, corr_I_rr, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(mean_I_b, mean_I_b, temp);
	var_I_bb = corr_I_bb - temp;
	multiply(mean_I_b, mean_I_g, temp);
	var_I_bg = corr_I_bg - temp;
	multiply(mean_I_b, mean_I_r, temp);
	var_I_br = corr_I_br - temp;
	multiply(mean_I_g, mean_I_g, temp);
	var_I_gg = corr_I_gg - temp;
	multiply(mean_I_g, mean_I_r, temp);
	var_I_gr = corr_I_gr - temp;
	multiply(mean_I_r, mean_I_r, temp);
	var_I_rr = corr_I_rr - temp;

	var_I_bb += eps;
	var_I_gg += eps;
	var_I_rr += eps;

#pragma omp parallel for
	for (int c = 0; c < 3; c++)
	{
		//mean_p[c].create(downsize, imtype);

		//corr_Ip_b[c].create(downsize, imtype);
		//corr_Ip_g[c].create(downsize, imtype);
		//corr_Ip_r[c].create(downsize, imtype);		
		//cov_Ip_b[c].create(downsize, imtype);
		//cov_Ip_g[c].create(downsize, imtype);
		//cov_Ip_r[c].create(downsize, imtype);

		a_b[c].create(downsize, imtype);
		a_g[c].create(downsize, imtype);
		a_r[c].create(downsize, imtype);

		//b[c].create(downsize, imtype);

		//mean_a_b[c].create(downsize, imtype);
		//mean_a_g[c].create(downsize, imtype);
		//mean_a_r[c].create(downsize, imtype);
		//mean_b[c].create(downsize, imtype);
		/*
		q[c].create(upsize, imtype);
		resize_mean_a_b[c].create(upsize, imtype);
		resize_mean_a_g[c].create(upsize, imtype);
		resize_mean_a_r[c].create(upsize, imtype);
		resize_mean_b[c].create(upsize, imtype);
		resize_q[c].create(upsize, imtype);
		*/

		boxFilter(srcf[c], mean_p[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);

		multiply(If_b, srcf[c], corr_Ip_b[c]);
		boxFilter(corr_Ip_b[c], corr_Ip_b[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		multiply(If_g, srcf[c], corr_Ip_g[c]);
		boxFilter(corr_Ip_g[c], corr_Ip_g[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		multiply(If_r, srcf[c], corr_Ip_r[c]);
		boxFilter(corr_Ip_r[c], corr_Ip_r[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);

		//implace srcf
		multiply(mean_I_b, mean_p[c], srcf[c]);
		cov_Ip_b[c] = corr_Ip_b[c] - srcf[c];
		multiply(mean_I_g, mean_p[c], srcf[c]);
		cov_Ip_g[c] = corr_Ip_g[c] - srcf[c];
		multiply(mean_I_r, mean_p[c], srcf[c]);
		cov_Ip_r[c] = corr_Ip_r[c] - srcf[c];

		{
			double* bb = var_I_bb.ptr<double>(0);
			double* bg = var_I_bg.ptr<double>(0);
			double* br = var_I_br.ptr<double>(0);
			double* gg = var_I_gg.ptr<double>(0);
			double* gr = var_I_gr.ptr<double>(0);
			double* rr = var_I_rr.ptr<double>(0);
			double* covb = cov_Ip_b[c].ptr<double>(0);
			double* covg = cov_Ip_g[c].ptr<double>(0);
			double* covr = cov_Ip_r[c].ptr<double>(0);
			double* ab = a_b[c].ptr<double>(0);
			double* ag = a_g[c].ptr<double>(0);
			double* ar = a_r[c].ptr<double>(0);

			for (int i = size; i--; bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++)
			{
				const double det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
					- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
				const double id = 1.0 / det;

				double c0 = *gg * *rr - *gr * *gr;
				double c1 = *gr * *br - *bg * *rr;
				double c2 = *bg * *gr - *br * *gg;
				double c4 = *bb * *rr - *br * *br;
				double c5 = *bg * *br - *bb * *gr;
				double c8 = *bb * *gg - *bg * *bg;

				*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
				*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
				*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);
			}
		}

		//implace corr_Ip_b,g,r[c]
		multiply(a_b[c], mean_I_b, corr_Ip_b[c]);
		multiply(a_g[c], mean_I_g, corr_Ip_g[c]);
		multiply(a_r[c], mean_I_r, corr_Ip_r[c]);
		b[c] = mean_p[c] - (corr_Ip_b[c] + corr_Ip_g[c] + corr_Ip_r[c]);

		boxFilter(a_b[c], a_b[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		boxFilter(a_g[c], a_g[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		boxFilter(a_r[c], a_r[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		boxFilter(b[c], b[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);

		//-----------------------------------------------------------------------resize
		//resize(a_b[c], resize_mean_a_b[c], upsize, 0.0, 0.0, upsampleMethod);
		//resize(a_g[c], resize_mean_a_g[c], upsize, 0.0, 0.0, upsampleMethod);
		//resize(a_r[c], resize_mean_a_r[c], upsize, 0.0, 0.0, upsampleMethod);
		//resize(b[c],   resize_mean_b[c],   upsize, 0.0, 0.0, upsampleMethod);
		float alpha = -1.5f;
		cp::upsampleCubic(a_b[c], resize_mean_a_b[c], scale, alpha);
		cp::upsampleCubic(a_g[c], resize_mean_a_g[c], scale, alpha);
		cp::upsampleCubic(a_r[c], resize_mean_a_r[c], scale, alpha);
		cp::upsampleCubic(b[c], resize_mean_b[c], scale, alpha);
		//-----------------------------------------------------------------------resize

		multiply(resize_mean_a_b[c], resize_If_b, resize_mean_a_b[c]);
		multiply(resize_mean_a_g[c], resize_If_g, resize_mean_a_g[c]);
		multiply(resize_mean_a_r[c], resize_If_r, resize_mean_a_r[c]);

		//implace srcf
		srcf[c] = resize_mean_a_b[c] + resize_mean_a_g[c] + resize_mean_a_r[c] + resize_mean_b[c];

		srcf[c].convertTo(d[c], v[c].depth());
	}

	merge(d, dest);
}

void GuidedImageUpsample::upsample32f(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, const float eps)
{
	//const int downsampleMethod = INTER_LANCZOS4;
	const int downsampleMethod = INTER_NEAREST;
	const int upsampleMethod = INTER_CUBIC;

	const int scale = guide.cols / src.cols;
	const Size upsize = guide.size();
	const Size downsize = src.size();

	const int imtype = CV_32F;
	const int ddepth = CV_32F;

	const Size ksize(2 * r + 1, 2 * r + 1);

	const int size = downsize.area();

	const Point anchor(-1, -1);

	v.resize(3);
	srcf.resize(3);

	I.resize(3);
	resize_If_b.create(upsize, imtype);
	resize_If_g.create(upsize, imtype);
	resize_If_r.create(upsize, imtype);
	If_b.create(downsize, imtype);
	If_g.create(downsize, imtype);
	If_r.create(downsize, imtype);

	mean_I_b.create(downsize, imtype);
	mean_I_g.create(downsize, imtype);
	mean_I_r.create(downsize, imtype);

	corr_I_bb.create(downsize, imtype);
	corr_I_bg.create(downsize, imtype);
	corr_I_br.create(downsize, imtype);
	corr_I_gg.create(downsize, imtype);
	corr_I_gr.create(downsize, imtype);
	corr_I_rr.create(downsize, imtype);

	var_I_bb.create(downsize, imtype);
	var_I_bg.create(downsize, imtype);
	var_I_br.create(downsize, imtype);
	var_I_gg.create(downsize, imtype);
	var_I_gr.create(downsize, imtype);
	var_I_rr.create(downsize, imtype);
	temp.create(downsize, imtype);

	mean_p.resize(3);
	corr_Ip_b.resize(3);
	corr_Ip_g.resize(3);
	corr_Ip_r.resize(3);
	cov_Ip_b.resize(3);
	cov_Ip_g.resize(3);
	cov_Ip_r.resize(3);
	a_b.resize(3);
	a_g.resize(3);
	a_r.resize(3);
	b.resize(3);

	resize_mean_a_b.resize(3);
	resize_mean_a_g.resize(3);
	resize_mean_a_r.resize(3);
	resize_mean_b.resize(3);
	resize_q.resize(3);

	d.resize(3);

	cv::split(src, v);
	cv::split(guide, I);
	{
		//cvt8u32f(src, srcf);
		v[0].convertTo(srcf[0], ddepth);
		v[1].convertTo(srcf[1], ddepth);
		v[2].convertTo(srcf[2], ddepth);
		I[0].convertTo(resize_If_b, ddepth);
		I[1].convertTo(resize_If_g, ddepth);
		I[2].convertTo(resize_If_r, ddepth);
	}


	//downsample input image
	cv::resize(resize_If_b, If_b, downsize, 0.0, 0.0, downsampleMethod);
	cv::resize(resize_If_g, If_g, downsize, 0.0, 0.0, downsampleMethod);
	cv::resize(resize_If_r, If_r, downsize, 0.0, 0.0, downsampleMethod);

	cv::boxFilter(If_b, mean_I_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	cv::boxFilter(If_g, mean_I_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	cv::boxFilter(If_r, mean_I_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, If_b, corr_I_bb);
	boxFilter(corr_I_bb, corr_I_bb, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_b, If_g, corr_I_bg);
	boxFilter(corr_I_bg, corr_I_bg, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_b, If_r, corr_I_br);
	boxFilter(corr_I_br, corr_I_br, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, If_g, corr_I_gg);
	boxFilter(corr_I_gg, corr_I_gg, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, If_r, corr_I_gr);
	boxFilter(corr_I_gr, corr_I_gr, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, If_r, corr_I_rr);
	boxFilter(corr_I_rr, corr_I_rr, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(mean_I_b, mean_I_b, temp);
	var_I_bb = corr_I_bb - temp;
	multiply(mean_I_b, mean_I_g, temp);
	var_I_bg = corr_I_bg - temp;
	multiply(mean_I_b, mean_I_r, temp);
	var_I_br = corr_I_br - temp;
	multiply(mean_I_g, mean_I_g, temp);
	var_I_gg = corr_I_gg - temp;
	multiply(mean_I_g, mean_I_r, temp);
	var_I_gr = corr_I_gr - temp;
	multiply(mean_I_r, mean_I_r, temp);
	var_I_rr = corr_I_rr - temp;

	var_I_bb += eps;
	var_I_gg += eps;
	var_I_rr += eps;

#pragma omp parallel for
	for (int c = 0; c < 3; c++)
	{
		//mean_p[c].create(downsize, imtype);

		//corr_Ip_b[c].create(downsize, imtype);
		//corr_Ip_g[c].create(downsize, imtype);
		//corr_Ip_r[c].create(downsize, imtype);		
		//cov_Ip_b[c].create(downsize, imtype);
		//cov_Ip_g[c].create(downsize, imtype);
		//cov_Ip_r[c].create(downsize, imtype);

		a_b[c].create(downsize, imtype);
		a_g[c].create(downsize, imtype);
		a_r[c].create(downsize, imtype);

		//b[c].create(downsize, imtype);

		//mean_a_b[c].create(downsize, imtype);
		//mean_a_g[c].create(downsize, imtype);
		//mean_a_r[c].create(downsize, imtype);
		//mean_b[c].create(downsize, imtype);
		/*
		q[c].create(upsize, imtype);
		resize_mean_a_b[c].create(upsize, imtype);
		resize_mean_a_g[c].create(upsize, imtype);
		resize_mean_a_r[c].create(upsize, imtype);
		resize_mean_b[c].create(upsize, imtype);
		resize_q[c].create(upsize, imtype);
		*/

		boxFilter(srcf[c], mean_p[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);

		multiply(If_b, srcf[c], corr_Ip_b[c]);
		boxFilter(corr_Ip_b[c], corr_Ip_b[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		multiply(If_g, srcf[c], corr_Ip_g[c]);
		boxFilter(corr_Ip_g[c], corr_Ip_g[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		multiply(If_r, srcf[c], corr_Ip_r[c]);
		boxFilter(corr_Ip_r[c], corr_Ip_r[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);

		//implace srcf
		multiply(mean_I_b, mean_p[c], srcf[c]);
		cov_Ip_b[c] = corr_Ip_b[c] - srcf[c];
		multiply(mean_I_g, mean_p[c], srcf[c]);
		cov_Ip_g[c] = corr_Ip_g[c] - srcf[c];
		multiply(mean_I_r, mean_p[c], srcf[c]);
		cov_Ip_r[c] = corr_Ip_r[c] - srcf[c];

		{
			float* bb = var_I_bb.ptr<float>(0);
			float* bg = var_I_bg.ptr<float>(0);
			float* br = var_I_br.ptr<float>(0);
			float* gg = var_I_gg.ptr<float>(0);
			float* gr = var_I_gr.ptr<float>(0);
			float* rr = var_I_rr.ptr<float>(0);
			float* covb = cov_Ip_b[c].ptr<float>(0);
			float* covg = cov_Ip_g[c].ptr<float>(0);
			float* covr = cov_Ip_r[c].ptr<float>(0);
			float* ab = a_b[c].ptr<float>(0);
			float* ag = a_g[c].ptr<float>(0);
			float* ar = a_r[c].ptr<float>(0);

			for (int i = size; i--; bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++)
			{
				const double det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
					- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
				const double id = 1.f / det;

				double c0 = *gg * *rr - *gr * *gr;
				double c1 = *gr * *br - *bg * *rr;
				double c2 = *bg * *gr - *br * *gg;
				double c4 = *bb * *rr - *br * *br;
				double c5 = *bg * *br - *bb * *gr;
				double c8 = *bb * *gg - *bg * *bg;

				*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
				*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
				*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);
			}
		}

		//implace corr_Ip_b,g,r[c]
		multiply(a_b[c], mean_I_b, corr_Ip_b[c]);
		multiply(a_g[c], mean_I_g, corr_Ip_g[c]);
		multiply(a_r[c], mean_I_r, corr_Ip_r[c]);
		b[c] = mean_p[c] - (corr_Ip_b[c] + corr_Ip_g[c] + corr_Ip_r[c]);

		boxFilter(a_b[c], a_b[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		boxFilter(a_g[c], a_g[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		boxFilter(a_r[c], a_r[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		boxFilter(b[c], b[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);

		//-----------------------------------------------------------------------resize
		//resize(a_b[c], resize_mean_a_b[c], upsize, 0.0, 0.0, upsampleMethod);
		//resize(a_g[c], resize_mean_a_g[c], upsize, 0.0, 0.0, upsampleMethod);
		//resize(a_r[c], resize_mean_a_r[c], upsize, 0.0, 0.0, upsampleMethod);
		//resize(b[c],   resize_mean_b[c],   upsize, 0.0, 0.0, upsampleMethod);
		float alpha = -1.5f;
		cp::upsampleCubic(a_b[c], resize_mean_a_b[c], scale, alpha);
		cp::upsampleCubic(a_g[c], resize_mean_a_g[c], scale, alpha);
		cp::upsampleCubic(a_r[c], resize_mean_a_r[c], scale, alpha);
		cp::upsampleCubic(b[c], resize_mean_b[c], scale, alpha);
		//-----------------------------------------------------------------------resize

		multiply(resize_mean_a_b[c], resize_If_b, resize_mean_a_b[c]);
		multiply(resize_mean_a_g[c], resize_If_g, resize_mean_a_g[c]);
		multiply(resize_mean_a_r[c], resize_If_r, resize_mean_a_r[c]);

		//srcfを破壊可能．output qの代わりに使用．
		srcf[c] = resize_mean_a_b[c] + resize_mean_a_g[c] + resize_mean_a_r[c] + resize_mean_b[c];

		srcf[c].convertTo(d[c], v[c].depth());
	}

	merge(d, dest);
}

void GuidedImageUpsample::filter32f(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, const float eps)
{
	const Size imsize = guide.size();

	const int imtype = CV_32F;
	const int ddepth = CV_32F;

	const Size ksize(2 * r + 1, 2 * r + 1);

	const int size = imsize.area();

	const Point anchor(-1, -1);

	v.resize(3);
	srcf.resize(3);

	I.resize(3);
	If_b.create(imsize, imtype);
	If_g.create(imsize, imtype);
	If_r.create(imsize, imtype);

	mean_I_b.create(imsize, imtype);
	mean_I_g.create(imsize, imtype);
	mean_I_r.create(imsize, imtype);

	corr_I_bb.create(imsize, imtype);
	corr_I_bg.create(imsize, imtype);
	corr_I_br.create(imsize, imtype);
	corr_I_gg.create(imsize, imtype);
	corr_I_gr.create(imsize, imtype);
	corr_I_rr.create(imsize, imtype);

	var_I_bb.create(imsize, imtype);
	var_I_bg.create(imsize, imtype);
	var_I_br.create(imsize, imtype);
	var_I_gg.create(imsize, imtype);
	var_I_gr.create(imsize, imtype);
	var_I_rr.create(imsize, imtype);
	temp.create(imsize, imtype);

	mean_p.resize(3);
	corr_Ip_b.resize(3);
	corr_Ip_g.resize(3);
	corr_Ip_r.resize(3);
	cov_Ip_b.resize(3);
	cov_Ip_g.resize(3);
	cov_Ip_r.resize(3);
	a_b.resize(3);
	a_g.resize(3);
	a_r.resize(3);
	b.resize(3);

	d.resize(3);

	cv::split(src, v);
	cv::split(guide, I);
	{
		//cvt8u32f(src, srcf);
		v[0].convertTo(srcf[0], ddepth);
		v[1].convertTo(srcf[1], ddepth);
		v[2].convertTo(srcf[2], ddepth);
		I[0].convertTo(If_b, ddepth);
		I[1].convertTo(If_g, ddepth);
		I[2].convertTo(If_r, ddepth);
	}

	cv::boxFilter(If_b, mean_I_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	cv::boxFilter(If_g, mean_I_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	cv::boxFilter(If_r, mean_I_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, If_b, corr_I_bb);
	boxFilter(corr_I_bb, corr_I_bb, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_b, If_g, corr_I_bg);
	boxFilter(corr_I_bg, corr_I_bg, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_b, If_r, corr_I_br);
	boxFilter(corr_I_br, corr_I_br, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, If_g, corr_I_gg);
	boxFilter(corr_I_gg, corr_I_gg, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, If_r, corr_I_gr);
	boxFilter(corr_I_gr, corr_I_gr, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, If_r, corr_I_rr);
	boxFilter(corr_I_rr, corr_I_rr, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(mean_I_b, mean_I_b, temp);
	var_I_bb = corr_I_bb - temp;
	multiply(mean_I_b, mean_I_g, temp);
	var_I_bg = corr_I_bg - temp;
	multiply(mean_I_b, mean_I_r, temp);
	var_I_br = corr_I_br - temp;
	multiply(mean_I_g, mean_I_g, temp);
	var_I_gg = corr_I_gg - temp;
	multiply(mean_I_g, mean_I_r, temp);
	var_I_gr = corr_I_gr - temp;
	multiply(mean_I_r, mean_I_r, temp);
	var_I_rr = corr_I_rr - temp;

	var_I_bb += eps;
	var_I_gg += eps;
	var_I_rr += eps;

	//#pragma omp parallel for
	for (int c = 0; c < 3; c++)
	{
		a_b[c].create(imsize, imtype);
		a_g[c].create(imsize, imtype);
		a_r[c].create(imsize, imtype);


		boxFilter(srcf[c], mean_p[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);

		multiply(If_b, srcf[c], corr_Ip_b[c]);
		boxFilter(corr_Ip_b[c], corr_Ip_b[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		multiply(If_g, srcf[c], corr_Ip_g[c]);
		boxFilter(corr_Ip_g[c], corr_Ip_g[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		multiply(If_r, srcf[c], corr_Ip_r[c]);
		boxFilter(corr_Ip_r[c], corr_Ip_r[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);

		//implace srcf
		multiply(mean_I_b, mean_p[c], srcf[c]);
		cov_Ip_b[c] = corr_Ip_b[c] - srcf[c];
		multiply(mean_I_g, mean_p[c], srcf[c]);
		cov_Ip_g[c] = corr_Ip_g[c] - srcf[c];
		multiply(mean_I_r, mean_p[c], srcf[c]);
		cov_Ip_r[c] = corr_Ip_r[c] - srcf[c];

		{
			float* bb = var_I_bb.ptr<float>(0);
			float* bg = var_I_bg.ptr<float>(0);
			float* br = var_I_br.ptr<float>(0);
			float* gg = var_I_gg.ptr<float>(0);
			float* gr = var_I_gr.ptr<float>(0);
			float* rr = var_I_rr.ptr<float>(0);
			float* covb = cov_Ip_b[c].ptr<float>(0);
			float* covg = cov_Ip_g[c].ptr<float>(0);
			float* covr = cov_Ip_r[c].ptr<float>(0);
			float* ab = a_b[c].ptr<float>(0);
			float* ag = a_g[c].ptr<float>(0);
			float* ar = a_r[c].ptr<float>(0);

			for (int i = size; i--; bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++)
			{
				const double det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
					- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
				const double id = 1.f / det;

				double c0 = *gg * *rr - *gr * *gr;
				double c1 = *gr * *br - *bg * *rr;
				double c2 = *bg * *gr - *br * *gg;
				double c4 = *bb * *rr - *br * *br;
				double c5 = *bg * *br - *bb * *gr;
				double c8 = *bb * *gg - *bg * *bg;

				*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
				*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
				*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);
			}
		}

		//implace corr_Ip_b,g,r[c]
		multiply(a_b[c], mean_I_b, corr_Ip_b[c]);
		multiply(a_g[c], mean_I_g, corr_Ip_g[c]);
		multiply(a_r[c], mean_I_r, corr_Ip_r[c]);
		b[c] = mean_p[c] - (corr_Ip_b[c] + corr_Ip_g[c] + corr_Ip_r[c]);

		boxFilter(a_b[c], a_b[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		boxFilter(a_g[c], a_g[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		boxFilter(a_r[c], a_r[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);
		boxFilter(b[c], b[c], ddepth, ksize, anchor, true, BORDER_REPLICATE);

		multiply(a_b[c], If_b, a_b[c]);
		multiply(a_g[c], If_g, a_g[c]);
		multiply(a_r[c], If_r, a_r[c]);

		//srcfを破壊可能．output qの代わりに使用．
		srcf[c] = a_b[c] + a_g[c] + a_r[c] + b[c];

		srcf[c].convertTo(d[c], v[c].depth());
	}

	merge(d, dest);
}

void FastGuided64F(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int r, double eps)
{
	const Size upsize = guide.size();
	const Size downsize = src.size();

	const int imtype = CV_64F;
	const int ddepth = CV_64F;

	const Size ksize(2 * r + 1, 2 * r + 1);

	const int size = downsize.area();

	const Point anchor(-1, -1);

	vector<Mat> I(3);
	vector<Mat> v(3);
	vector<Mat> d(3);
	Mat srcf0(downsize, imtype);
	Mat srcf1(downsize, imtype);
	Mat srcf2(downsize, imtype);
	Mat If_b(downsize, imtype);
	Mat If_g(downsize, imtype);
	Mat If_r(downsize, imtype);
	Mat mean_p(downsize, imtype);
	Mat mean_I_b(downsize, imtype);
	Mat mean_I_g(downsize, imtype);
	Mat mean_I_r(downsize, imtype);
	Mat corr_I_bb(downsize, imtype);
	Mat corr_I_bg(downsize, imtype);
	Mat corr_I_br(downsize, imtype);
	Mat corr_I_gg(downsize, imtype);
	Mat corr_I_gr(downsize, imtype);
	Mat corr_I_rr(downsize, imtype);
	Mat corr_Ip_b(downsize, imtype);
	Mat corr_Ip_g(downsize, imtype);
	Mat corr_Ip_r(downsize, imtype);
	Mat var_I_bb(downsize, imtype);
	Mat var_I_bg(downsize, imtype);
	Mat var_I_br(downsize, imtype);
	Mat var_I_gg(downsize, imtype);
	Mat var_I_gr(downsize, imtype);
	Mat var_I_rr(downsize, imtype);
	Mat cov_Ip_b(downsize, imtype);
	Mat cov_Ip_g(downsize, imtype);
	Mat cov_Ip_r(downsize, imtype);
	Mat a_b(downsize, imtype);
	Mat a_g(downsize, imtype);
	Mat a_r(downsize, imtype);
	Mat b(downsize, imtype);
	Mat mean_a_b(downsize, imtype);
	Mat mean_a_g(downsize, imtype);
	Mat mean_a_r(downsize, imtype);
	Mat mean_b(downsize, imtype);
	Mat q(downsize, imtype);
	Mat temp(downsize, imtype);
	Mat temp_b(downsize, imtype);
	Mat temp_g(downsize, imtype);
	Mat temp_r(downsize, imtype);

	Mat resize_If_b(upsize, imtype);
	Mat resize_If_g(upsize, imtype);
	Mat resize_If_r(upsize, imtype);
	Mat resize_mean_a_b(upsize, imtype);
	Mat resize_mean_a_g(upsize, imtype);
	Mat resize_mean_a_r(upsize, imtype);
	Mat resize_mean_b(upsize, imtype);
	Mat resize_q(upsize, imtype);

	split(src, v);
	split(guide, I);
	{
		//cvt8u32f(src, srcf);
		v[0].convertTo(srcf0, CV_64F);
		v[1].convertTo(srcf1, CV_64F);
		v[2].convertTo(srcf2, CV_64F);
		I[0].convertTo(resize_If_b, CV_64F);
		I[1].convertTo(resize_If_g, CV_64F);
		I[2].convertTo(resize_If_r, CV_64F);
	}

	//const int downsampleMethod = INTER_LANCZOS4;
	const int downsampleMethod = INTER_AREA;

	//downsample input image
	resize(resize_If_b, If_b, downsize, 0.0, 0.0, downsampleMethod);
	resize(resize_If_g, If_g, downsize, 0.0, 0.0, downsampleMethod);
	resize(resize_If_r, If_r, downsize, 0.0, 0.0, downsampleMethod);
	//

	//src_blue

	boxFilter(If_b, mean_I_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(If_g, mean_I_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(If_r, mean_I_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(srcf0, mean_p, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, If_b, corr_I_bb);
	boxFilter(corr_I_bb, corr_I_bb, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_b, If_g, corr_I_bg);
	boxFilter(corr_I_bg, corr_I_bg, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_b, If_r, corr_I_br);
	boxFilter(corr_I_br, corr_I_br, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, If_g, corr_I_gg);
	boxFilter(corr_I_gg, corr_I_gg, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, If_r, corr_I_gr);
	boxFilter(corr_I_gr, corr_I_gr, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, If_r, corr_I_rr);
	boxFilter(corr_I_rr, corr_I_rr, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, srcf0, corr_Ip_b);
	boxFilter(corr_Ip_b, corr_Ip_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, srcf0, corr_Ip_g);
	boxFilter(corr_Ip_g, corr_Ip_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, srcf0, corr_Ip_r);
	boxFilter(corr_Ip_r, corr_Ip_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(mean_I_b, mean_I_b, temp);
	var_I_bb = corr_I_bb - temp;
	multiply(mean_I_b, mean_I_g, temp);
	var_I_bg = corr_I_bg - temp;
	multiply(mean_I_b, mean_I_r, temp);
	var_I_br = corr_I_br - temp;
	multiply(mean_I_g, mean_I_g, temp);
	var_I_gg = corr_I_gg - temp;
	multiply(mean_I_g, mean_I_r, temp);
	var_I_gr = corr_I_gr - temp;
	multiply(mean_I_r, mean_I_r, temp);
	var_I_rr = corr_I_rr - temp;

	var_I_bb += eps;
	var_I_gg += eps;
	var_I_rr += eps;

	multiply(mean_I_b, mean_p, temp);
	cov_Ip_b = corr_Ip_b - temp;
	multiply(mean_I_g, mean_p, temp);
	cov_Ip_g = corr_Ip_g - temp;
	multiply(mean_I_r, mean_p, temp);
	cov_Ip_r = corr_Ip_r - temp;
	{
		double* bb = var_I_bb.ptr<double>(0);
		double* bg = var_I_bg.ptr<double>(0);
		double* br = var_I_br.ptr<double>(0);
		double* gg = var_I_gg.ptr<double>(0);
		double* gr = var_I_gr.ptr<double>(0);
		double* rr = var_I_rr.ptr<double>(0);
		double* covb = cov_Ip_b.ptr<double>(0);
		double* covg = cov_Ip_g.ptr<double>(0);
		double* covr = cov_Ip_r.ptr<double>(0);
		double* ab = a_b.ptr<double>(0);
		double* ag = a_g.ptr<double>(0);
		double* ar = a_r.ptr<double>(0);

		for (int i = size; i--; bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++)
		{
			const double det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
				- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
			const double id = 1.f / det;

			double c0 = *gg * *rr - *gr * *gr;
			double c1 = *gr * *br - *bg * *rr;
			double c2 = *bg * *gr - *br * *gg;
			double c4 = *bb * *rr - *br * *br;
			double c5 = *bg * *br - *bb * *gr;
			double c8 = *bb * *gg - *bg * *bg;

			*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
			*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
			*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);
		}
	}
	multiply(a_b, mean_I_b, temp_b);
	multiply(a_g, mean_I_g, temp_g);
	multiply(a_r, mean_I_r, temp_r);
	b = mean_p - (temp_b + temp_g + temp_r);

	boxFilter(a_b, mean_a_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_g, mean_a_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_r, mean_a_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(b, mean_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	//-----------------------------------------------------------------------resize
	resize(mean_a_b, resize_mean_a_b, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_g, resize_mean_a_g, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_r, resize_mean_a_r, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_b, resize_mean_b, upsize, 0.0, 0.0, INTER_CUBIC);
	//-----------------------------------------------------------------------resize

	multiply(resize_mean_a_b, resize_If_b, resize_mean_a_b);
	multiply(resize_mean_a_g, resize_If_g, resize_mean_a_g);
	multiply(resize_mean_a_r, resize_If_r, resize_mean_a_r);

	q = resize_mean_a_b + resize_mean_a_g + resize_mean_a_r + resize_mean_b;

	q.convertTo(d[0], v[0].type());

	//src_green
	boxFilter(srcf1, mean_p, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, srcf1, corr_Ip_b);
	boxFilter(corr_Ip_b, corr_Ip_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, srcf1, corr_Ip_g);
	boxFilter(corr_Ip_g, corr_Ip_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, srcf1, corr_Ip_r);
	boxFilter(corr_Ip_r, corr_Ip_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);


	multiply(mean_I_b, mean_p, temp);
	cov_Ip_b = corr_Ip_b - temp;
	multiply(mean_I_g, mean_p, temp);
	cov_Ip_g = corr_Ip_g - temp;
	multiply(mean_I_r, mean_p, temp);
	cov_Ip_r = corr_Ip_r - temp;

	{
		double* bb = var_I_bb.ptr<double>(0);
		double* bg = var_I_bg.ptr<double>(0);
		double* br = var_I_br.ptr<double>(0);
		double* gg = var_I_gg.ptr<double>(0);
		double* gr = var_I_gr.ptr<double>(0);
		double* rr = var_I_rr.ptr<double>(0);
		double* covb = cov_Ip_b.ptr<double>(0);
		double* covg = cov_Ip_g.ptr<double>(0);
		double* covr = cov_Ip_r.ptr<double>(0);
		double* ab = a_b.ptr<double>(0);
		double* ag = a_g.ptr<double>(0);
		double* ar = a_r.ptr<double>(0);

		for (int i = size; i--; bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++)
		{
			const double det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
				- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
			const double id = 1.f / det;

			double c0 = *gg * *rr - *gr * *gr;
			double c1 = *gr * *br - *bg * *rr;
			double c2 = *bg * *gr - *br * *gg;
			double c4 = *bb * *rr - *br * *br;
			double c5 = *bg * *br - *bb * *gr;
			double c8 = *bb * *gg - *bg * *bg;

			*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
			*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
			*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);
		}
	}

	multiply(a_b, mean_I_b, temp_b);
	multiply(a_g, mean_I_g, temp_g);
	multiply(a_r, mean_I_r, temp_r);
	b = mean_p - (temp_b + temp_g + temp_r);

	boxFilter(a_b, mean_a_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_g, mean_a_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_r, mean_a_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(b, mean_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	//-----------------------------------------------------------------------resize
	resize(mean_a_b, resize_mean_a_b, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_g, resize_mean_a_g, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_r, resize_mean_a_r, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_b, resize_mean_b, upsize, 0.0, 0.0, INTER_CUBIC);
	//-----------------------------------------------------------------------resize

	multiply(resize_mean_a_b, resize_If_b, resize_mean_a_b);
	multiply(resize_mean_a_g, resize_If_g, resize_mean_a_g);
	multiply(resize_mean_a_r, resize_If_r, resize_mean_a_r);

	q = resize_mean_a_b + resize_mean_a_g + resize_mean_a_r + resize_mean_b;

	q.convertTo(d[1], v[1].type());

	//src_red

	boxFilter(srcf2, mean_p, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, srcf2, corr_Ip_b);
	boxFilter(corr_Ip_b, corr_Ip_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, srcf2, corr_Ip_g);
	boxFilter(corr_Ip_g, corr_Ip_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, srcf2, corr_Ip_r);
	boxFilter(corr_Ip_r, corr_Ip_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(mean_I_b, mean_p, temp);
	cov_Ip_b = corr_Ip_b - temp;
	multiply(mean_I_g, mean_p, temp);
	cov_Ip_g = corr_Ip_g - temp;
	multiply(mean_I_r, mean_p, temp);
	cov_Ip_r = corr_Ip_r - temp;

	{
		double* bb = var_I_bb.ptr<double>(0);
		double* bg = var_I_bg.ptr<double>(0);
		double* br = var_I_br.ptr<double>(0);
		double* gg = var_I_gg.ptr<double>(0);
		double* gr = var_I_gr.ptr<double>(0);
		double* rr = var_I_rr.ptr<double>(0);
		double* covb = cov_Ip_b.ptr<double>(0);
		double* covg = cov_Ip_g.ptr<double>(0);
		double* covr = cov_Ip_r.ptr<double>(0);
		double* ab = a_b.ptr<double>(0);
		double* ag = a_g.ptr<double>(0);
		double* ar = a_r.ptr<double>(0);

		for (int i = size; i--; bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++)
		{
			const double det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
				- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
			const double id = 1.f / det;

			double c0 = *gg * *rr - *gr * *gr;
			double c1 = *gr * *br - *bg * *rr;
			double c2 = *bg * *gr - *br * *gg;
			double c4 = *bb * *rr - *br * *br;
			double c5 = *bg * *br - *bb * *gr;
			double c8 = *bb * *gg - *bg * *bg;

			*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
			*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
			*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);
		}
	}

	multiply(a_b, mean_I_b, temp_b);
	multiply(a_g, mean_I_g, temp_g);
	multiply(a_r, mean_I_r, temp_r);
	b = mean_p - (temp_b + temp_g + temp_r);

	boxFilter(a_b, mean_a_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_g, mean_a_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_r, mean_a_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(b, mean_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	//-----------------------------------------------------------------------resize
	resize(mean_a_b, resize_mean_a_b, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_g, resize_mean_a_g, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_r, resize_mean_a_r, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_b, resize_mean_b, upsize, 0.0, 0.0, INTER_CUBIC);
	//-----------------------------------------------------------------------resize

	multiply(resize_mean_a_b, resize_If_b, resize_mean_a_b);
	multiply(resize_mean_a_g, resize_If_g, resize_mean_a_g);
	multiply(resize_mean_a_r, resize_If_r, resize_mean_a_r);

	q = resize_mean_a_b + resize_mean_a_g + resize_mean_a_r + resize_mean_b;

	q.convertTo(d[2], v[2].type());

	merge(d, dest);
}

void FastGuided(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int r, float eps)
{
	const Size upsize = guide.size();
	const Size downsize = src.size();

	const int imtype = CV_32F;
	const int ddepth = CV_32F;

	const Size ksize(2 * r + 1, 2 * r + 1);

	const int size = downsize.area();

	const Point anchor(-1, -1);

	vector<Mat> I(3);
	vector<Mat> v(3);
	vector<Mat> d(3);
	Mat srcf0(downsize, imtype);
	Mat srcf1(downsize, imtype);
	Mat srcf2(downsize, imtype);
	Mat If_b(downsize, imtype);
	Mat If_g(downsize, imtype);
	Mat If_r(downsize, imtype);
	Mat mean_p(downsize, imtype);
	Mat mean_I_b(downsize, imtype);
	Mat mean_I_g(downsize, imtype);
	Mat mean_I_r(downsize, imtype);
	Mat corr_I_bb(downsize, imtype);
	Mat corr_I_bg(downsize, imtype);
	Mat corr_I_br(downsize, imtype);
	Mat corr_I_gg(downsize, imtype);
	Mat corr_I_gr(downsize, imtype);
	Mat corr_I_rr(downsize, imtype);
	Mat corr_Ip_b(downsize, imtype);
	Mat corr_Ip_g(downsize, imtype);
	Mat corr_Ip_r(downsize, imtype);
	Mat var_I_bb(downsize, imtype);
	Mat var_I_bg(downsize, imtype);
	Mat var_I_br(downsize, imtype);
	Mat var_I_gg(downsize, imtype);
	Mat var_I_gr(downsize, imtype);
	Mat var_I_rr(downsize, imtype);
	Mat cov_Ip_b(downsize, imtype);
	Mat cov_Ip_g(downsize, imtype);
	Mat cov_Ip_r(downsize, imtype);
	Mat a_b(downsize, imtype);
	Mat a_g(downsize, imtype);
	Mat a_r(downsize, imtype);
	Mat b(downsize, imtype);
	Mat mean_a_b(downsize, imtype);
	Mat mean_a_g(downsize, imtype);
	Mat mean_a_r(downsize, imtype);
	Mat mean_b(downsize, imtype);
	Mat q(downsize, imtype);
	Mat temp(downsize, imtype);
	Mat temp_b(downsize, imtype);
	Mat temp_g(downsize, imtype);
	Mat temp_r(downsize, imtype);

	Mat resize_If_b(upsize, imtype);
	Mat resize_If_g(upsize, imtype);
	Mat resize_If_r(upsize, imtype);
	Mat resize_mean_a_b(upsize, imtype);
	Mat resize_mean_a_g(upsize, imtype);
	Mat resize_mean_a_r(upsize, imtype);
	Mat resize_mean_b(upsize, imtype);
	Mat resize_q(upsize, imtype);

	split(src, v);
	split(guide, I);
	{
		//cvt8u32f(src, srcf);
		v[0].convertTo(srcf0, CV_32F);
		v[1].convertTo(srcf1, CV_32F);
		v[2].convertTo(srcf2, CV_32F);
		I[0].convertTo(resize_If_b, CV_32F);
		I[1].convertTo(resize_If_g, CV_32F);
		I[2].convertTo(resize_If_r, CV_32F);
	}

	//downsample input image
	resize(resize_If_b, If_b, downsize, 0.0, 0.0, INTER_AREA);
	resize(resize_If_g, If_g, downsize, 0.0, 0.0, INTER_AREA);
	resize(resize_If_r, If_r, downsize, 0.0, 0.0, INTER_AREA);
	//

	//src_blue

	boxFilter(If_b, mean_I_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(If_g, mean_I_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(If_r, mean_I_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(srcf0, mean_p, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, If_b, corr_I_bb);
	boxFilter(corr_I_bb, corr_I_bb, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_b, If_g, corr_I_bg);
	boxFilter(corr_I_bg, corr_I_bg, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_b, If_r, corr_I_br);
	boxFilter(corr_I_br, corr_I_br, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, If_g, corr_I_gg);
	boxFilter(corr_I_gg, corr_I_gg, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, If_r, corr_I_gr);
	boxFilter(corr_I_gr, corr_I_gr, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, If_r, corr_I_rr);
	boxFilter(corr_I_rr, corr_I_rr, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, srcf0, corr_Ip_b);
	boxFilter(corr_Ip_b, corr_Ip_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, srcf0, corr_Ip_g);
	boxFilter(corr_Ip_g, corr_Ip_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, srcf0, corr_Ip_r);
	boxFilter(corr_Ip_r, corr_Ip_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(mean_I_b, mean_I_b, temp);
	var_I_bb = corr_I_bb - temp;
	multiply(mean_I_b, mean_I_g, temp);
	var_I_bg = corr_I_bg - temp;
	multiply(mean_I_b, mean_I_r, temp);
	var_I_br = corr_I_br - temp;
	multiply(mean_I_g, mean_I_g, temp);
	var_I_gg = corr_I_gg - temp;
	multiply(mean_I_g, mean_I_r, temp);
	var_I_gr = corr_I_gr - temp;
	multiply(mean_I_r, mean_I_r, temp);
	var_I_rr = corr_I_rr - temp;

	var_I_bb += eps;
	var_I_gg += eps;
	var_I_rr += eps;

	multiply(mean_I_b, mean_p, temp);
	cov_Ip_b = corr_Ip_b - temp;
	multiply(mean_I_g, mean_p, temp);
	cov_Ip_g = corr_Ip_g - temp;
	multiply(mean_I_r, mean_p, temp);
	cov_Ip_r = corr_Ip_r - temp;
	{
		float* bb = var_I_bb.ptr<float>(0);
		float* bg = var_I_bg.ptr<float>(0);
		float* br = var_I_br.ptr<float>(0);
		float* gg = var_I_gg.ptr<float>(0);
		float* gr = var_I_gr.ptr<float>(0);
		float* rr = var_I_rr.ptr<float>(0);
		float* covb = cov_Ip_b.ptr<float>(0);
		float* covg = cov_Ip_g.ptr<float>(0);
		float* covr = cov_Ip_r.ptr<float>(0);
		float* ab = a_b.ptr<float>(0);
		float* ag = a_g.ptr<float>(0);
		float* ar = a_r.ptr<float>(0);

		for (int i = size; i--; bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++)
		{
			const float det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
				- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
			const float id = 1.f / det;

			float c0 = *gg * *rr - *gr * *gr;
			float c1 = *gr * *br - *bg * *rr;
			float c2 = *bg * *gr - *br * *gg;
			float c4 = *bb * *rr - *br * *br;
			float c5 = *bg * *br - *bb * *gr;
			float c8 = *bb * *gg - *bg * *bg;

			*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
			*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
			*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);
		}
	}
	multiply(a_b, mean_I_b, temp_b);
	multiply(a_g, mean_I_g, temp_g);
	multiply(a_r, mean_I_r, temp_r);
	b = mean_p - (temp_b + temp_g + temp_r);

	boxFilter(a_b, mean_a_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_g, mean_a_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_r, mean_a_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(b, mean_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	//-----------------------------------------------------------------------resize
	resize(mean_a_b, resize_mean_a_b, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_g, resize_mean_a_g, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_r, resize_mean_a_r, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_b, resize_mean_b, upsize, 0.0, 0.0, INTER_CUBIC);
	//-----------------------------------------------------------------------resize

	multiply(resize_mean_a_b, resize_If_b, resize_mean_a_b);
	multiply(resize_mean_a_g, resize_If_g, resize_mean_a_g);
	multiply(resize_mean_a_r, resize_If_r, resize_mean_a_r);

	q = resize_mean_a_b + resize_mean_a_g + resize_mean_a_r + resize_mean_b;

	q.convertTo(d[0], v[0].type());

	//src_green



	boxFilter(srcf1, mean_p, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, srcf1, corr_Ip_b);
	boxFilter(corr_Ip_b, corr_Ip_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, srcf1, corr_Ip_g);
	boxFilter(corr_Ip_g, corr_Ip_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, srcf1, corr_Ip_r);
	boxFilter(corr_Ip_r, corr_Ip_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);


	multiply(mean_I_b, mean_p, temp);
	cov_Ip_b = corr_Ip_b - temp;
	multiply(mean_I_g, mean_p, temp);
	cov_Ip_g = corr_Ip_g - temp;
	multiply(mean_I_r, mean_p, temp);
	cov_Ip_r = corr_Ip_r - temp;

	{
		float* bb = var_I_bb.ptr<float>(0);
		float* bg = var_I_bg.ptr<float>(0);
		float* br = var_I_br.ptr<float>(0);
		float* gg = var_I_gg.ptr<float>(0);
		float* gr = var_I_gr.ptr<float>(0);
		float* rr = var_I_rr.ptr<float>(0);
		float* covb = cov_Ip_b.ptr<float>(0);
		float* covg = cov_Ip_g.ptr<float>(0);
		float* covr = cov_Ip_r.ptr<float>(0);
		float* ab = a_b.ptr<float>(0);
		float* ag = a_g.ptr<float>(0);
		float* ar = a_r.ptr<float>(0);

		for (int i = size; i--; bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++)
		{
			const float det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
				- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
			const float id = 1.f / det;

			float c0 = *gg * *rr - *gr * *gr;
			float c1 = *gr * *br - *bg * *rr;
			float c2 = *bg * *gr - *br * *gg;
			float c4 = *bb * *rr - *br * *br;
			float c5 = *bg * *br - *bb * *gr;
			float c8 = *bb * *gg - *bg * *bg;

			*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
			*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
			*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);
		}
	}

	multiply(a_b, mean_I_b, temp_b);
	multiply(a_g, mean_I_g, temp_g);
	multiply(a_r, mean_I_r, temp_r);
	b = mean_p - (temp_b + temp_g + temp_r);

	boxFilter(a_b, mean_a_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_g, mean_a_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_r, mean_a_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(b, mean_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	//-----------------------------------------------------------------------resize
	resize(mean_a_b, resize_mean_a_b, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_g, resize_mean_a_g, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_r, resize_mean_a_r, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_b, resize_mean_b, upsize, 0.0, 0.0, INTER_CUBIC);
	//-----------------------------------------------------------------------resize

	multiply(resize_mean_a_b, resize_If_b, resize_mean_a_b);
	multiply(resize_mean_a_g, resize_If_g, resize_mean_a_g);
	multiply(resize_mean_a_r, resize_If_r, resize_mean_a_r);

	q = resize_mean_a_b + resize_mean_a_g + resize_mean_a_r + resize_mean_b;

	q.convertTo(d[1], v[1].type());

	//src_red

	boxFilter(srcf2, mean_p, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(If_b, srcf2, corr_Ip_b);
	boxFilter(corr_Ip_b, corr_Ip_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_g, srcf2, corr_Ip_g);
	boxFilter(corr_Ip_g, corr_Ip_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	multiply(If_r, srcf2, corr_Ip_r);
	boxFilter(corr_Ip_r, corr_Ip_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	multiply(mean_I_b, mean_p, temp);
	cov_Ip_b = corr_Ip_b - temp;
	multiply(mean_I_g, mean_p, temp);
	cov_Ip_g = corr_Ip_g - temp;
	multiply(mean_I_r, mean_p, temp);
	cov_Ip_r = corr_Ip_r - temp;

	{
		float* bb = var_I_bb.ptr<float>(0);
		float* bg = var_I_bg.ptr<float>(0);
		float* br = var_I_br.ptr<float>(0);
		float* gg = var_I_gg.ptr<float>(0);
		float* gr = var_I_gr.ptr<float>(0);
		float* rr = var_I_rr.ptr<float>(0);
		float* covb = cov_Ip_b.ptr<float>(0);
		float* covg = cov_Ip_g.ptr<float>(0);
		float* covr = cov_Ip_r.ptr<float>(0);
		float* ab = a_b.ptr<float>(0);
		float* ag = a_g.ptr<float>(0);
		float* ar = a_r.ptr<float>(0);

		for (int i = size; i--; bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++)
		{
			const float det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
				- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
			const float id = 1.f / det;

			float c0 = *gg * *rr - *gr * *gr;
			float c1 = *gr * *br - *bg * *rr;
			float c2 = *bg * *gr - *br * *gg;
			float c4 = *bb * *rr - *br * *br;
			float c5 = *bg * *br - *bb * *gr;
			float c8 = *bb * *gg - *bg * *bg;

			*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
			*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
			*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);
		}
	}

	multiply(a_b, mean_I_b, temp_b);
	multiply(a_g, mean_I_g, temp_g);
	multiply(a_r, mean_I_r, temp_r);
	b = mean_p - (temp_b + temp_g + temp_r);

	boxFilter(a_b, mean_a_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_g, mean_a_g, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(a_r, mean_a_r, ddepth, ksize, anchor, true, BORDER_REPLICATE);
	boxFilter(b, mean_b, ddepth, ksize, anchor, true, BORDER_REPLICATE);

	resize(mean_a_b, resize_mean_a_b, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_g, resize_mean_a_g, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_a_r, resize_mean_a_r, upsize, 0.0, 0.0, INTER_CUBIC);
	resize(mean_b, resize_mean_b, upsize, 0.0, 0.0, INTER_CUBIC);

	//-----------------------------------------------------------------------resize
	multiply(resize_mean_a_b, resize_If_b, resize_mean_a_b);
	multiply(resize_mean_a_g, resize_If_g, resize_mean_a_g);
	multiply(resize_mean_a_r, resize_If_r, resize_mean_a_r);
	//-----------------------------------------------------------------------resize

	q = resize_mean_a_b + resize_mean_a_g + resize_mean_a_r + resize_mean_b;

	q.convertTo(d[2], v[2].type());

	merge(d, dest);

}
#endif