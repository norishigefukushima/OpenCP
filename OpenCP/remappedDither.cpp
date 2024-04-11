#include "remappedDither.hpp"
#include "inlineSIMDFunctions.hpp"
#include "inlineMathFunctions.hpp"
#include "color.hpp"


#include "detailEnhancement.hpp"
#include "statisticalFilter.hpp"
#include "debugcp.hpp"
//#include <spatialfilter/SpatialFilter.hpp>
using namespace std;
using namespace cv;

namespace cp
{
	std::string getImageSamplingMethodName(const int isample_method)
	{
		std::string isampling_name;
		if (isample_method == IMAGE_TEXTURNESS_FLOYD_STEINBERG) isampling_name = "TEX_FLOYD";
		else if (isample_method == IMAGE_TEXTURENESS_OSTRO) isampling_name = "TEX_OSTRO";
		else if (isample_method == IMAGE_TEXTURENESS_SIERRA2) isampling_name = "TEX_SIEERA2";
		else if (isample_method == IMAGE_TEXTURENESS_SIERRA3) isampling_name = "TEX_SIEERA3";
		else if (isample_method == IMAGE_TEXTURENESS_JARVIS) isampling_name = "TEX_JARVIS";
		else if (isample_method == IMAGE_TEXTURENESS_STUCKI) isampling_name = "TEX_STUCKI";
		else if (isample_method == IMAGE_TEXTURENESS_BURKES) isampling_name = "TEX_BURKES";
		else if (isample_method == IMAGE_TEXTURENESS_STEAVENSON) isampling_name = "TEX_STEAVENSON";
		else if (isample_method == IMAGE_DEPTH) isampling_name = "DEPTH";
		else if (isample_method == IMAGE_FLAT_FLOYD_STEINBERG) isampling_name = "FLAT_FLOYD";
		else if (isample_method == IMAGE_FLAT_OSTRO) isampling_name = "FLAT_OSTRO";
		else if (isample_method == IMAGE_FLAT_SIERRA2) isampling_name = "FLAT_SIERRA2";
		else if (isample_method == IMAGE_FLAT_SIERRA3) isampling_name = "FLAT_SIERRA3";
		else if (isample_method == IMAGE_FLAT_JARVIS) isampling_name = "FLAT_JARVIS";
		else if (isample_method == IMAGE_FLAT_STUCKI) isampling_name = "FLAT_STUCKI";
		else if (isample_method == IMAGE_FLAT_BURKES) isampling_name = "FLAT_BURKES";
		else if (isample_method == IMAGE_FLAT_STEAVENSON) isampling_name = "FLAT_STEAVENSON";
		else if (isample_method == IMAGE_AREA_MONTECARLO) isampling_name = "AREA_MONTECARLO";
		else if (isample_method == IMAGE_BLUENOISE) isampling_name = "BLUENOISE";
		else isampling_name = "Default";

		return isampling_name;
	}

	std::string getDitheringPostProcessName(const int method)
	{
		std::string ret = "";
		switch (method)
		{
		case NO_POSTPROCESS: ret = "NO_POSTPROCESS";
			break;
		case FlipBottomCopy:ret = "FlipBottomCopy";
			break;
		case FlipTopCopy:ret = "FlipTopCopy";
			break;
		case RANDOM_ROTATION:ret = "RANDOM_ROTATION";
			break;
		default: ret = "";
			break;
		}
		return ret;
	}

	static void generateHistogramFromScaledInput(cv::Mat& src, cv::Mat& histogram, const int num_bins)
	{
		histogram = Mat::zeros(num_bins, 1, CV_32F);
		float* s = src.ptr<float>();
		for (int i = 0; i < src.size().area(); i++)
		{
			int index = min(int(s[i] * (num_bins)), num_bins - 1);
			histogram.at<float>(index) += 1.f;
		}
	}

	float IntensityRemappedDither::compute_s(const Mat& src)
	{
		float s = 0.f;

		const int nf = max(cvRound(n * sampling_ratio), 2);
		int m = min(1000, nf);
		//int m = ; 
		//int m = (int)ceil(bin_ratio * n * sampling_ratio);//number of bins in the processing histogram

		int histSize[] = { m };

		int channels[] = { 0 };
		int dims = 1;
		float value_ranges[] = { 0.f,1.f };
		const float* ranges[] = { value_ranges };

		MatND histogram;

		cv::calcHist(&src, 1, channels, Mat(), histogram, dims, histSize, ranges, true, false);
		//generateHistogramFromScaledInput(src, histogram, m);

		const float inv_num_of_bins = 1.f / (float)(m);
		float H_k = 0.f;//cumulative sum fpr h_i
		float X_k = 0.f;//cumulative sum fpr h_i*xi
		float* h = histogram.ptr<float>();

		/*for (int i = 0; i < histogram.size().area(); i++)
		{
			cout << i << "|" << h[i] << endl;
		}
		getchar();*/
		for (int i = 0; i < m; i++)
		{
			//const float x_i = (i + 0.5f) * inv_num_of_bins;
			const float x_i = (i)*inv_num_of_bins;
			X_k += x_i * h[i];
			H_k += h[i];
			s = (H_k - n * (1.f - sampling_ratio)) / X_k;
			if (s * x_i > 1.f) break;
		}

		return s;
	}

	void IntensityRemappedDither::remap(const Mat& src, Mat& dest)
	{
		dest.create(src.size(), src.type());
		if (scale == 0.f) scale = compute_s(src);

		//scaling image by s
		const float* src_ptr = src.ptr<float>();
		float* dest_ptr = dest.ptr<float>();
		const int simd_n = get_simd_floor(n, 8);
		__m256 ms = _mm256_set1_ps(scale);
		__m256 ones = _mm256_set1_ps(1.f);
		for (int i = 0; i < simd_n; i += 8)
		{
			_mm256_store_ps(dest_ptr + i, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(src_ptr + i), ms), ones));
		}
		for (int i = simd_n; i < n; i++)
		{
			dest_ptr[i] = min(scale * src_ptr[i], 1.f);
		}
	}

	IntensityRemappedDither::IntensityRemappedDither(Size image_size, float sampling_ratio, const int dither_method, const int dither_scanorder, const int dither_postprocess) :
		n(image_size.area()), sampling_ratio(sampling_ratio), dither_method(dither_method), dither_scanorder(dither_scanorder), dither_postprocess(dither_postprocess)
	{
		;
	}

	int IntensityRemappedDither::body(const Mat& src, Mat& dest)
	{
		CV_Assert(!dest.empty());

		remap(src, imagebuff);

		int sample_num = ditherDestruction(imagebuff, dest, dither_method, dither_scanorder);

		if (dither_postprocess == FlipBottomCopy)
		{
			int r = src.rows / 2;
			Mat flp;
			flip(dest, flp, 1);
			for (int j = 0; j < r; j++)
			{
				uchar* d = dest.ptr(j);
				uchar* s = flp.ptr(2 * r - j);
				memcpy(d, s, sizeof(uchar) * src.cols);
			}
			sample_num = countNonZero(dest);
		}
		else if (dither_postprocess == FlipTopCopy)
		{
			Mat flp;
			flip(dest, flp, 0);
			for (int j = 0; j < src.rows / 2; j++)
			{
				uchar* s = flp.ptr(src.rows / 2 + j);
				uchar* d = dest.ptr(src.rows / 2 + j);
				memcpy(d, s, sizeof(uchar) * src.cols);
			}
			sample_num = countNonZero(dest);
		}
		else if (dither_postprocess == RANDOM_ROTATION)
		{
			srand((unsigned int)cv::getTickCount());
			int n = rand() % 4;
			if (n != 3) cv::rotate(dest, dest, n);
		}

		return sample_num;
	}

	int IntensityRemappedDither::generate(const Mat& src, Mat& dest, const float ratio)
	{
		sampling_ratio = ratio;
		scale = compute_s(src);
		return body(src, dest);
	}

	int generateSamplingMaskRemappedDitherWeight(const cv::Mat& weight, cv::Mat& dest, const float sampling_ratio, const int dithering_method, const int dithering_order, const float bin_ratio, const int maskOption)
	{
		IntensityRemappedDither dither(weight.size(), sampling_ratio, dithering_method, dithering_order, maskOption);
		return dither.generate(weight, dest, sampling_ratio);
	}

	void Swap(cv::Mat& importanceMap, int px, int py, int qx, int qy)
	{
		float tmp = importanceMap.at<float>(py, px);
		importanceMap.at<float>(py, px) = importanceMap.at<float>(qy, qx);
		importanceMap.at<float>(qy, qx) = tmp;

	}
	float Comp(cv::Mat& importanceMap, Size mapSize, int ix, int iy)
	{
		const int W = importanceMap.cols;
		const int H = importanceMap.rows;

		const float sigma_i = 2.1f;
		const float sigma_s = 1.f;

		vector<float> sum(H);
		for (int i = 0; i < H; i++)sum[i] = 0.f;

		//small area approximation
		for (int oy = -14; oy <= 14; ++oy)
		{
			for (int ox = -14; ox <= 14; ++ox)
			{
				int sx = ix + ox;
				if (sx < 0)
					sx += W;
				if (sx >= W)
					sx -= W;

				int sy = iy + oy;
				if (sy < 0)
					sy += H;
				if (sy >= H)
					sy -= H;

				float dx = (float)abs(ix - sx);
				if (dx > W / 2)
					dx = W - dx;

				float dy = (float)abs(iy - sy);
				if (dy > H / 2)
					dy = H - dy;

				const float a = (dx * dx + dy * dy) / (sigma_i * sigma_i);

				const float b = sqrt(abs(importanceMap.at<float>(iy, ix) - importanceMap.at<float>(sy, sx))) / (sigma_s * sigma_s);

				sum[sy] += exp(-a - b);
			}
		}

		float total = 0.f;
		for (int sy = 0; sy < H; ++sy)
			total += sum[sy];

		return total;
	}
	void createImportanceMapFlatBlueNoise(cv::Mat& dest, int& sample_num, float sampling_ratio)
	{
		RNG rng;
		sample_num = 0;

		if (sampling_ratio == 1.f)
		{
			dest.setTo(255);
			sample_num = dest.size().area();
			return;
		}

		Mat temp = Mat::zeros(dest.size(), CV_32F);
		int ns = (int)(dest.size().area() * sampling_ratio);

		for (int n = 0; n < ns;)
		{
			int x = rng.uniform(0, temp.cols);
			int y = rng.uniform(0, temp.rows);

			if (temp.at<float>(y, x) == 0.f)
			{
				temp.at<float>(y, x) = 1.f;
				sample_num++;
				n++;
			}
		}

		//initial energy
		Mat energy(Size(dest.cols, dest.rows), CV_32F);
		for (int iy = 0; iy < dest.rows; ++iy)
		{
			for (int ix = 0; ix < dest.cols; ++ix)
			{
				energy.at<float>(iy, ix) = Comp(temp, temp.size(), ix, iy);
			}
		}

		const int kMaxIteration = 5000;

		for (int i = 0; i < kMaxIteration; i++)
		{
			float current_energy = 0.f;

			for (int iy = 0; iy < dest.rows; ++iy)
			{
				for (int ix = 0; ix < dest.cols; ++ix)
				{
					current_energy += energy.at<float>(iy, ix);
				}
			}
			//cout << i<<": " << current_energy<< endl;
			const int px = rng.uniform(0, dest.cols);
			const int py = rng.uniform(0, dest.rows);
			const int qx = rng.uniform(0, dest.cols);
			const int qy = rng.uniform(0, dest.rows);

			float next_energy = current_energy;
			next_energy -= energy.at<float>(py, px);
			next_energy -= energy.at<float>(qy, qx);

			Swap(temp, px, py, qx, qy);

			const float e0 = Comp(temp, temp.size(), px, py);
			const float e1 = Comp(temp, temp.size(), qx, qy);

			next_energy += (e0 + e1);

			if (next_energy < current_energy)
			{
				energy.at<float>(py, px) = e0;
				energy.at<float>(qy, qx) = e1;
				continue;
			}

			Swap(temp, px, py, qx, qy);
		}

		temp.convertTo(dest, CV_8U, 255.f);
	}

	void generateSamplingMaskRemappedDitherFlat(cv::RNG& rng, cv::Mat& dest, int& sample_num, const float sampling_ratio, int dithering_method, const bool isCircle)
	{
		if (sampling_ratio == 1.f)
		{
			dest.setTo(255);
			sample_num = dest.size().area();
			return;
		}

		Mat filtered(dest.size(), CV_32F);
		rng.fill(filtered, RNG::UNIFORM, 0.f, 1.f, true);
		if (isCircle)
		{
			Mat mask;
			cp::setCircleMask(mask, dest.size());
			filtered.setTo(0, mask);
		}

		sample_num = generateSamplingMaskRemappedDitherWeight(filtered, dest, sampling_ratio, dithering_method, DITHER_SCANORDER::MEANDERING, 0.1f, DITHER_POSTPROCESS::RANDOM_ROTATION);
	}

	void generateSamplingMaskRemappedDitherGaussian(cv::RNG& rng, cv::Mat& dest, int& sample_num, const float sampling_ratio, int dithering_method, int dithering_order, const float sigma)
	{
		if (sampling_ratio == 1.f)
		{
			dest.setTo(255);
			sample_num = dest.size().area();
			return;
		}

		Mat mask(dest.size(), CV_32F);

		const int r = dest.rows / 2;
		float coeff = -1.f / (2.f * sigma * sigma);

		for (int j = 0; j < dest.rows; j++)
		{
			for (int i = 0; i < dest.cols; i++)
			{
				float r2 = float((i - r) * (i - r) + (j - r) * (j - r));
				//if (sqrt(r2) > r)continue;
				mask.at<float>(j, i) = rng.uniform(0.f, (float)exp(r2 * coeff));
				//filtered.at<float>(j, i) = max((float)exp(r2 * coeff), rng.uniform(0.f,0.05f));
			}
		}
		//filtered.at<float>(r, r) = 0.f;
#if 0
		double sample_lambda = 0.5;
		double sample_sigma1 = sigma;
		double sample_sigma2 = sigma * 0.9;
		for (int j = 0; j < dest.rows; j++)
		{
			const double y = (j - r);
			for (int i = 0; i < dest.cols; i++)
			{
				const double x = (i - r);
				const double d = sqrt(x * x + y * y);

				//filtered.at<float>(j, i) = (float)(max(wmin,exp(d*d / (-2.0*sample_sigma1 *sample_sigma1)) - sample_lambda * exp(d*d / (-2.0*sample_sigma2*sample_sigma2))));

				const float v = (float)(exp(d * d / (-2.0 * sample_sigma1 * sample_sigma1)) - sample_lambda * exp(d * d / (-2.0 * sample_sigma2 * sample_sigma2)));
				//filtered.at<float>(j, i) = v;
				filtered.at<float>(j, i) = rng.uniform(0.f, v);
			}
		}
#endif
		sample_num = generateSamplingMaskRemappedDitherWeight(mask, dest, sampling_ratio, dithering_method, dithering_order, 0.1f, 3);
	}

	void generateSamplingMaskRemappedDitherDepthSigma(const cv::Mat& dispmap, cv::Mat& dest, const int inforcus_disp, const float sigma_base, const float inc_sigma, int& sample_num, float sampling_ratio)
	{
		if (sampling_ratio == 1.f)
		{
			sample_num = dispmap.size().area();
			dest.setTo(255);
			return;
		}
		if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != dispmap.size())dest.create(dispmap.size(), CV_8UC1);
		//	CV_Assert(src.type() == dest.type());

		Mat filtered(dispmap.size(), CV_32F);
		double minv, maxv;
		minMaxLoc(dispmap, &minv, &maxv);
		float smax = (float)((maxv - inforcus_disp) * inc_sigma);
		{
			//Timer t("tt1");
			const int size = dispmap.size().area();

			uchar* d = (uchar*)dispmap.ptr<uchar>(0);
			float* dst = filtered.ptr<float>(0);
			for (int i = 0; i < size; i++)
			{
				//float sigma = min(5.f,abs((float)d[i] - inforcus_disp))*inc_sigma;
				float sigma = abs((float)d[i] - inforcus_disp) * inc_sigma;
				//dst[i] = (smax - sigma)/smax;
				//dst[i] = (sigma) / smax;
				//dst[i] = (smax-sigma) / smax + 1;
				float v = ((smax - sigma) / smax) * 0.4f;
				dst[i] = max(v * v * v, 0.05f);
			}
			//GaussianBlur(filtered, filtered, Size(3, 3), 20, 20);
		}

		//imshowNormalize("imp", filtered); waitKey();


		//remapのためのヒストグラム計算(Appendix)
		{
			//Timer t("tt2");
			int binNum = 1000;
			int histSize[] = { binNum };

			float value_ranges[] = { 0,1 };
			const float* ranges[] = { value_ranges };
			MatND hist;

			int channels[] = { 0 };
			int dims = 1;

			calcHist(&filtered, 1, channels, Mat(), hist, dims, histSize, ranges, true, false); //ヒストグラム計算

			int h_k = 0;
			float x_k = 0.f;

			float s = 0.f;
			float x = 0.f;
			const float ibinNum = 1.f / binNum;
			int n = dispmap.size().area();
			for (int i = 0; s * x <= 1.f; i++)
			{
				h_k += saturate_cast<int>(hist.at<float>(i));

				x = 1.f * i * ibinNum + 0.0005f;
				x_k += x * saturate_cast<int>(hist.at<float>(i));
				s = (h_k - n * (1 - sampling_ratio)) / x_k;
			}

			const __m256 ms = _mm256_set1_ps(s);
			const __m256 ones = _mm256_set1_ps(1.f);
#pragma omp parallel for schedule (dynamic)
			for (int y = 0; y < dispmap.rows; y++)
			{
				float* hpi_ptr = filtered.ptr<float>(y);
				for (int x = 0; x < dispmap.cols; x += 8)
				{
					//result[x] = min(v[x] * s, 1.f);
					_mm256_store_ps(hpi_ptr + x, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(hpi_ptr + x), ms), ones));
				}
			}
		}
		{
			//Timer t("tt3");
			sample_num = ditherDestruction(filtered, dest, FLOYD_STEINBERG);
		}
	}

	/*
	Mat calcCenterSurround(Mat& center, Mat& surround)
	{
		Mat csmap(center.size(), center.type());
		resize(surround, csmap, csmap.size());
		csmap = abs(csmap - center);
		return csmap;
	}

	vector<Mat> CreateCenterSurroundPyramid(vector<Mat>& pyramid)
	{
		vector<Mat> cspyr(6);
		cspyr[0] = calcCenterSurround(pyramid[2], pyramid[5]);
		cspyr[1] = calcCenterSurround(pyramid[2], pyramid[6]);
		cspyr[2] = calcCenterSurround(pyramid[3], pyramid[6]);
		cspyr[3] = calcCenterSurround(pyramid[3], pyramid[7]);
		cspyr[4] = calcCenterSurround(pyramid[4], pyramid[7]);
		cspyr[5] = calcCenterSurround(pyramid[4], pyramid[8]);
		return cspyr;
	}

	void normalizeRange(Mat& src)
	{
		double minval, maxval;
		minMaxLoc(src, &minval, &maxval);
		src -= minval;
		if (minval < maxval)
			src /= maxval - minval;
	}

	void trimPeaks(Mat& src, int step)
	{
		const int w = src.cols;
		const int h = src.rows;

		const double M = 1.0;
		normalizeRange(src);
		double m = 0.0;
		for (int y = 0; y < h - step; y += step)
		{
			for (int x = 0; x < w - step; x += step)
			{
				Mat roi(src, Rect(x, y, step, step));
				double minval = 0.0, maxval = 0.0;
				minMaxLoc(roi, &minval, &maxval);
				m += maxval;
			}
		}
		m /= (w / step - (w % step ? 0 : 1)) * (h / step - (h % step ? 0 : 1));
		src *= (M - m) * (M - m);
	}

	Mat calcSaliencyMap(Mat& src)
	{
		const int STEP = 8;
		const float WEIGHT_I = 0.333f;
		const float WEIGHT_C = 0.333f;
		const float WEIGHT_O = 0.333f;

		Mat srcf;
		src.convertTo(srcf, CV_32F, 1.f / 255);

		const int r = 8;
		const Size ksize = Size(2 * r + 1, 2 * r + 1);
		const float sigma = r / CV_PI;
		const float lambda = r + 1;
		const float deg45 = CV_PI / 4.0;
		Mat gabor000 = getGaborKernel(ksize, sigma, deg45 * 0, lambda, 1.0, 0.0, CV_32F);
		Mat gabor045 = getGaborKernel(ksize, sigma, deg45 * 1, lambda, 1.0, 0.0, CV_32F);
		Mat gabor090 = getGaborKernel(ksize, sigma, deg45 * 2, lambda, 1.0, 0.0, CV_32F);
		Mat gabor135 = getGaborKernel(ksize, sigma, deg45 * 3, lambda, 1.0, 0.0, CV_32F);

		const int NUM_SCALES = 9;
		vector<Mat> pyramidI(NUM_SCALES);
		vector<Mat> pyramidRG(NUM_SCALES);
		vector<Mat> pyramidBY(NUM_SCALES);
		vector<Mat> pyramid000(NUM_SCALES);
		vector<Mat> pyramid045(NUM_SCALES);
		vector<Mat> pyramid090(NUM_SCALES);
		vector<Mat> pyramid135(NUM_SCALES);

		Mat scaled = srcf;
		for (int s = 0; s < NUM_SCALES; ++s)
		{
			const int w = scaled.cols;
			const int h = scaled.rows;

			vector<Mat> colors;
			split(scaled, colors);
			Mat imageI = (colors[0] + colors[1] + colors[2]) / 3.0f;
			pyramidI[s] = imageI;

			double minval, maxval;
			minMaxLoc(imageI, &minval, &maxval);
			Mat r(h, w, CV_32F);
			Mat g(h, w, CV_32F);
			Mat b(h, w, CV_32F);
			for (int j = 0; j < h; ++j)
			{
				for (int i = 0; i < w; ++i)
				{
					if (imageI.at<float>(j, i) < 0.1f * maxval) //最大ピークの1/10以下の画素は除外
						continue;
					r.at<float>(j, i) = colors[2].at<float>(j, i) / imageI.at<float>(j, i);
					g.at<float>(j, i) = colors[1].at<float>(j, i) / imageI.at<float>(j, i);
					b.at<float>(j, i) = colors[0].at<float>(j, i) / imageI.at<float>(j, i);
				}
			}

			Mat R = max(0.0f, r - (g + b) / 2);
			Mat G = max(0.0f, g - (b + r) / 2);
			Mat B = max(0.0f, b - (r + g) / 2);
			Mat Y = max(0.0f, (r + g) / 2 - abs(r - g) / 2 - b);
			pyramidRG[s] = R - G;
			pyramidBY[s] = B - Y;

			filter2D(imageI, pyramid000[s], -1, gabor000);
			filter2D(imageI, pyramid045[s], -1, gabor045);
			filter2D(imageI, pyramid090[s], -1, gabor090);
			filter2D(imageI, pyramid135[s], -1, gabor135);

			pyrDown(scaled, scaled);
		}


		vector<Mat> cspyrI = CreateCenterSurroundPyramid(pyramidI);
		vector<Mat> cspyrRG = CreateCenterSurroundPyramid(pyramidRG);
		vector<Mat> cspyrBY = CreateCenterSurroundPyramid(pyramidBY);
		vector<Mat> cspyr000 = CreateCenterSurroundPyramid(pyramid000);
		vector<Mat> cspyr045 = CreateCenterSurroundPyramid(pyramid045);
		vector<Mat> cspyr090 = CreateCenterSurroundPyramid(pyramid090);
		vector<Mat> cspyr135 = CreateCenterSurroundPyramid(pyramid135);

		Mat temp(srcf.size(), CV_32F);
		Mat conspI(srcf.size(), CV_32F);
		Mat conspC(srcf.size(), CV_32F);
		Mat consp000(srcf.size(), CV_32F);
		Mat consp045(srcf.size(), CV_32F);
		Mat consp090(srcf.size(), CV_32F);
		Mat consp135(srcf.size(), CV_32F);
		for (int t = 0; t<int(cspyrI.size()); ++t)
		{
			trimPeaks(cspyrI[t], STEP); resize(cspyrI[t], temp, srcf.size()); conspI += temp;

			trimPeaks(cspyrRG[t], STEP); resize(cspyrRG[t], temp, srcf.size()); conspC += temp;
			trimPeaks(cspyrBY[t], STEP); resize(cspyrBY[t], temp, srcf.size()); conspC += temp;

			trimPeaks(cspyr000[t], STEP); resize(cspyr000[t], temp, srcf.size()); consp000 += temp;
			trimPeaks(cspyr045[t], STEP); resize(cspyr045[t], temp, srcf.size()); consp045 += temp;
			trimPeaks(cspyr090[t], STEP); resize(cspyr090[t], temp, srcf.size()); consp090 += temp;
			trimPeaks(cspyr135[t], STEP); resize(cspyr135[t], temp, srcf.size()); consp135 += temp;
		}
		trimPeaks(consp000, STEP);
		trimPeaks(consp045, STEP);
		trimPeaks(consp090, STEP);
		trimPeaks(consp135, STEP);
		Mat conspO = consp000 + consp045 + consp090 + consp135;

		trimPeaks(conspI, STEP);
		trimPeaks(conspC, STEP);
		trimPeaks(conspO, STEP);
		Mat saliency = WEIGHT_I * conspI + WEIGHT_C * conspC + WEIGHT_O * conspO;
		normalizeRange(saliency);
		return saliency;
	}
	*/


	struct xorshift_32_4 {
		static unsigned int x[4];
		static int cnt;
		unsigned int operator()() {
			if (cnt != 4) return x[cnt++];
			__m128i a;
			a = _mm_loadu_si128((__m128i*)x);
			a = _mm_xor_si128(a, _mm_slli_epi32(a, 13));
			a = _mm_xor_si128(a, _mm_srli_epi32(a, 17));
			a = _mm_xor_si128(a, _mm_slli_epi32(a, 5));
			_mm_storeu_si128((__m128i*)x, a);
			cnt = 1;
			return x[0];
		}
	};
	int xorshift_32_4::cnt = 0;
	unsigned int xorshift_32_4::x[] = { 0xf247756d, 0x1654caaa, 0xb2f5e564, 0x7d986dd7 };

	//https://github.com/Tlapesium/cpp-lib/blob/master/utility/xorshift.cpp
		//need AVX2
	struct _MM_XORSHIFT32_AVX2
	{
		__m256 normal;
		__m256i x;
		int cnt = 0;
		//public:
		_MM_XORSHIFT32_AVX2()
		{
			x = _mm256_set_epi32(0xd5eae750, 0xc784b986, 0x16bcf701, 0x65032360, 0xb628094f, 0xd8281e7b, 0xecfa5dc8, 0x3b828203);
			normal = _mm256_set1_ps(1.f / (INT_MAX));
		}
		__m256 next32f()
		{
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
			x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
			return _mm256_mul_ps(normal, _mm256_abs_ps(_mm256_cvtepi32_ps(x)));
		}
		__m256i next()
		{
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
			x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
			return x;
		}

		unsigned int operator()()
		{
			if (cnt != 8) return x.m256i_i32[cnt++];
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
			x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
			cnt = 1;
			return x.m256i_i32[0];
		}
	};

	struct xorshift_64_4 {
		static unsigned long long int x[4];
		static int cnt;
		unsigned long long int operator()() {
			if (cnt != 4) return x[cnt++];
			__m256i a;
			a = _mm256_loadu_si256((__m256i*)x);
			a = _mm256_xor_si256(a, _mm256_slli_epi64(a, 7));
			a = _mm256_xor_si256(a, _mm256_srli_epi64(a, 9));
			_mm256_storeu_si256((__m256i*)x, a);
			cnt = 1;
			return x[0];
		}
	};

	int xorshift_64_4::cnt = 0;
	unsigned long long int xorshift_64_4::x[] = { 0xf77bcfb23d5143cfULL, 0xbda154512ac6f703ULL, 0xb2ef653838c2edf3ULL, 0xa7dbfba7cef3c195ULL };

	//type 0 Gaussian (sr is not used)
	//type 1 gradient (sr is not used)
	//type 2 bilateral (sr is valid)
	class TexturenessSampling
	{
		RNG rng;
		Mat histogram;

		Mat importance;
		Mat src_32f;
		Mat gray;
		//Ptr<cp::SpatialFilterBase> gf = cp::createSpatialFilter(cp::SpatialFilterAlgorithm::SlidingDCT5_AVX, CV_32F, cp::SpatialKernel::GAUSSIAN);

		float edgeDiffusionSigma = 1.f;

		void gradientMax(const Mat& src, Mat& dest, const bool isAccumurate)
		{
			if (dest.empty()) dest.create(src.size(), src.type());

			if (isAccumurate)
			{
				for (int j = 0; j < src.rows; j++)
				{
					const float* sc = src.ptr<float>(j);
					const float* sm = src.ptr<float>(max(j - 1, 0));
					const float* sp = src.ptr<float>(min(j + 1, src.rows - 1));
					//const float* smm = src.ptr<float>(max(j - 2, 0));
					//const float* spp = src.ptr<float>(min(j + 2, src.rows - 1));
					float* d = dest.ptr<float>(j);
					if constexpr (true)
					{
						for (int i = 0; i < src.cols; i += 8)
						{
							const __m256 mscp = _mm256_loadu_ps(sc + i - 1);
							const __m256 mscc = _mm256_loadu_ps(sc + i - 0);
							const __m256 mscm = _mm256_loadu_ps(sc + i + 1);
							const __m256 msp = _mm256_loadu_ps(sp + i + 0);
							const __m256 msm = _mm256_loadu_ps(sm + i + 0);

							_mm256_storeu_ps(d + i, _mm256_add_ps(_mm256_loadu_ps(d + i), _mm256_max_ps(
								_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
								_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm)))
							));
						}
					}
					else
					{
						for (int i = 1; i < src.cols - 1; i++)
						{
							//const float dx = 0.25f * (sc[i - 1] + sc[i + 1] + sc[i - 2] + sc[i + 2]);
							//const float dy = 0.25f * (sm[i] + sp[i] + smm[i] + spp[i]);
							//const float dx = 0.5f * (sc[i - 1] + sc[i + 1]);
							//const float dy = 0.5f * (sm[i] + sp[i]);

							//d[i] += sqrt((sc[i] - dx) * (sc[i] - dx) + (sc[i] - dy) * (sc[i] - dy));
							//d[i] += sqrt(max((sc[i] - dx) * (sc[i] - dx), (sc[i] - dy) * (sc[i] - dy)));
							d[i] += max(max(abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1])), max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
							//d[i] += 0.5f * ((abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1])) + max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
						}
					}
					d[0] = 0.f;
					d[src.cols - 1] = 0.f;
				}
			}
			else
			{
				for (int j = 0; j < src.rows; j++)
				{
					const float* sc = src.ptr<float>(j);
					const float* sm = src.ptr<float>(max(j - 1, 0));
					const float* sp = src.ptr<float>(min(j + 1, src.rows - 1));
					//const float* smm = src.ptr<float>(max(j - 2, 0));
					//const float* spp = src.ptr<float>(min(j + 2, src.rows - 1));
					float* d = dest.ptr<float>(j);
					if constexpr (true)
					{
						for (int i = 0; i < src.cols; i += 8)
						{
							const __m256 mscp = _mm256_loadu_ps(sc + i - 1);
							const __m256 mscc = _mm256_loadu_ps(sc + i - 0);
							const __m256 mscm = _mm256_loadu_ps(sc + i + 1);
							const __m256 msp = _mm256_loadu_ps(sp + i + 0);
							const __m256 msm = _mm256_loadu_ps(sm + i + 0);

							_mm256_storeu_ps(d + i, _mm256_max_ps(
								_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
								_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm))
							));
						}
					}
					else
					{
						for (int i = 1; i < src.cols - 1; i++)
						{
							//const float dx = 0.25f * (sc[i - 1] + sc[i + 1] + sc[i - 2] + sc[i + 2]);
							//const float dy = 0.25f * (sm[i] + sp[i] + smm[i] + spp[i]);
							//const float dx = 0.5f * (sc[i - 1] + sc[i + 1]);
							//const float dy = 0.5f * (sm[i] + sp[i]);

							//d[i] = sqrt((sc[i] - dx) * (sc[i] - dx) + (sc[i] - dy) * (sc[i] - dy));
							//d[i] = sqrt(max((sc[i] - dx) * (sc[i] - dx), (sc[i] - dy) * (sc[i] - dy)));
							d[i] = max(max(abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1])), max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
							//d[i] = 0.5f*((abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1]))+ max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
						}
					}
					d[0] = 0.f;
					d[src.cols - 1] = 0.f;
				}
			}
		}

		void gradientMaxDiffuse3x3(const Mat& src, Mat& dest, const bool isAccumurate, const float sigma)
		{
			if (dest.empty()) dest.create(src.size(), CV_32F);
			vector<Mat> linebuffer(3);
			for (int l = 0; l < 3; l++)
			{
				linebuffer[l].create(Size(src.cols + 2, 1), CV_32F);
				linebuffer[l].at<float>(0, 0) = 0.f;
				linebuffer[l].at<float>(0, 1) = 0.f;
				linebuffer[l].at<float>(0, linebuffer[l].cols - 2) = 0.f;
				linebuffer[l].at<float>(0, linebuffer[l].cols - 1) = 0.f;
			}
			float w0 = exp(-1.f / (2.f * sigma * sigma));
			float normal = 2.f * w0 + 1.f;
			const __m256 g0 = _mm256_set1_ps(w0 / normal);
			const __m256 g1 = _mm256_set1_ps(1.f / normal);
			if (isAccumurate)
			{
				for (int j = 0; j < src.rows; j++)
				{
					const float* sc = src.ptr<float>(j);
					const float* sm = src.ptr<float>(max(j - 1, 0));
					const float* sp = src.ptr<float>(min(j + 1, src.rows - 1));
					float* d = dest.ptr<float>(j);
					float* b = linebuffer[j % 3].ptr<float>(0, 1);
					for (int i = 0; i < src.cols; i += 8)
					{
						const __m256 mscp = _mm256_loadu_ps(sc + i - 1);
						const __m256 mscc = _mm256_loadu_ps(sc + i - 0);
						const __m256 mscm = _mm256_loadu_ps(sc + i + 1);
						const __m256 msp = _mm256_loadu_ps(sp + i + 0);
						const __m256 msm = _mm256_loadu_ps(sm + i + 0);

						_mm256_storeu_ps(b + i, _mm256_max_ps(
							_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
							_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm))
						));
					}
					b[0] = 0.f;
					b[src.cols - 1] = 0.f;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 blur = _mm256_mul_ps(g0, _mm256_loadu_ps(b + i - 1));
						blur = _mm256_fmadd_ps(g1, _mm256_loadu_ps(b + i + 0), blur);
						blur = _mm256_fmadd_ps(g0, _mm256_loadu_ps(b + i + 1), blur);
						_mm256_storeu_ps(b + i, blur);
					}
					for (int i = 0; i < src.cols; i += 8)
					{
						_mm256_store_ps(d + i, _mm256_add_ps(_mm256_loadu_ps(d + i), _mm256_loadu_ps(b + i)));
					}
				}
			}
			else
			{
				for (int j = 0; j < src.rows; j++)
				{
					const float* sc = src.ptr<float>(j);
					const float* sm = src.ptr<float>(max(j - 1, 0));
					const float* sp = src.ptr<float>(min(j + 1, src.rows - 1));
					float* d = dest.ptr<float>(j);
					float* b = linebuffer[j % 3].ptr<float>(0, 1);
					for (int i = 0; i < src.cols; i += 8)
					{
						const __m256 mscp = _mm256_loadu_ps(sc + i - 1);
						const __m256 mscc = _mm256_loadu_ps(sc + i - 0);
						const __m256 mscm = _mm256_loadu_ps(sc + i + 1);
						const __m256 msp = _mm256_loadu_ps(sp + i + 0);
						const __m256 msm = _mm256_loadu_ps(sm + i + 0);

						_mm256_storeu_ps(b + i, _mm256_max_ps(
							_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
							_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm))
						));
					}
					b[0] = 0.f;
					b[src.cols - 1] = 0.f;
					for (int i = 0; i < src.cols; i += 8)
					{
						__m256 blur = _mm256_mul_ps(g0, _mm256_loadu_ps(b + i - 1));
						blur = _mm256_fmadd_ps(g1, _mm256_loadu_ps(b + i + 0), blur);
						blur = _mm256_fmadd_ps(g0, _mm256_loadu_ps(b + i + 1), blur);
						_mm256_storeu_ps(b + i, blur);
					}
					for (int i = 0; i < src.cols; i += 8)
					{
						_mm256_store_ps(d + i, _mm256_loadu_ps(b + i));
					}
				}
			}
		}

		//return maxval;
		template<int ch>
		float gradientMaxDiffuse3x3(const vector<Mat>& src, Mat& dest, const bool isAccumurate, const float sigma)
		{
			if (dest.empty()) dest.create(src[0].size(), CV_32F);

			vector<Mat> linebuffer(3);
			for (int l = 0; l < 3; l++)
			{
				linebuffer[l].create(Size(src[0].cols + 2, 1), CV_32F);
				linebuffer[l].at<float>(0, 0) = 0.f;
				linebuffer[l].at<float>(0, 1) = 0.f;
				linebuffer[l].at<float>(0, linebuffer[l].cols - 2) = 0.f;
				linebuffer[l].at<float>(0, linebuffer[l].cols - 1) = 0.f;
			}

			const float w0 = exp(-1.f / (2.f * sigma * sigma));
			const float normal = 2.f * w0 + 1.f;
			const __m256 g0 = _mm256_set1_ps(w0 / normal);
			const __m256 g1 = _mm256_set1_ps(1.f / normal);
			const int w = src[0].cols;
			const int h = src[0].rows;
			__m256 mmax = _mm256_setzero_ps();
			for (int j = 0; j < h; j++)
			{
				AutoBuffer<const float*> sc(ch);
				AutoBuffer<const float*> sm(ch);
				AutoBuffer<const float*> sp(ch);
				for (int c = 0; c < ch; c++)
				{
					sc[c] = src[c].ptr<float>(j);
					sm[c] = src[c].ptr<float>(max(j - 1, 0));
					sp[c] = src[c].ptr<float>(min(j + 1, h - 1));
				}
	
				float* b = linebuffer[j % 3].ptr<float>(0, 1);
				for (int i = 0; i < h; i += 8)
				{
					__m256 macc = _mm256_setzero_ps();
					for (int c = 0; c < ch; c++)
					{
						const __m256 mscp = _mm256_loadu_ps(sc[c] + i - 1);
						const __m256 mscc = _mm256_loadu_ps(sc[c] + i - 0);
						const __m256 mscm = _mm256_loadu_ps(sc[c] + i + 1);
						const __m256 msp = _mm256_loadu_ps(sp[c] + i + 0);
						const __m256 msm = _mm256_loadu_ps(sm[c] + i + 0);

						macc = _mm256_add_ps(macc, _mm256_max_ps(
							_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
							_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm))
						));
					}
					_mm256_store_ps(b + i, macc);
				}
				b[0] = 0.f;
				b[w - 1] = 0.f;
				float* d = dest.ptr<float>(j);
				for (int i = 0; i < w; i += 8)
				{
					__m256 blur = _mm256_mul_ps(g0, _mm256_loadu_ps(b + i - 1));
					blur = _mm256_fmadd_ps(g1, _mm256_loadu_ps(b + i + 0), blur);
					blur = _mm256_fmadd_ps(g0, _mm256_loadu_ps(b + i + 1), blur);
					_mm256_storeu_ps(d + i, blur);
					mmax = _mm256_max_ps(mmax, blur);
				}
			}

			return _mm256_reducemax_ps(mmax);
		}


		float getmax(const Mat& src)
		{
			const int size = src.size().area();
			const int simdSize = get_simd_floor(size, 8);
			__m256 mmax = _mm256_setzero_ps();
			const float* s = src.ptr<float>();
			for (int i = 0; i < simdSize; i += 8)
			{
				mmax = _mm256_max_ps(mmax, _mm256_load_ps(s + i));
			}
			return _mm256_reducemax_ps(mmax);
		}

		void normalize_max1(Mat& srcdest)
		{
			const int size = srcdest.size().area();
			const int simdSize = get_simd_floor(size, 8);
			__m256 mmax = _mm256_setzero_ps();
			float* s = srcdest.ptr<float>();
			for (int i = 0; i < simdSize; i += 8)
			{
				mmax = _mm256_max_ps(mmax, _mm256_load_ps(s + i));
			}
			float maxval = 0.f;
			for (int i = 0; i < 8; i++)
			{
				maxval = max(maxval, mmax.m256_f32[i]);
			}

			__m256 minv = _mm256_set1_ps(1.f / maxval);
			for (int i = 0; i < simdSize; i += 8)
			{
				_mm256_store_ps(s + i, _mm256_mul_ps(minv, _mm256_load_ps(s + i)));
			}
		}

		float computeTexturenessDoG(const vector<Mat>& src, Mat& dest, const float sigma)
		{
			for (int c = 0; c < src.size(); c++)
			{
				const int r = (int)ceil(sigma * 1.5);
				const int d = 2 * r + 1;
				GaussianBlur(src[c], src_32f, Size(d, d), sigma);
				//Laplacian(src[c], src_32f, CV_32F, d);
				//gf->filter(src[c], src_32f, ss1, 1);
				if (c == 0)
				{
					absdiff(src_32f, src[c], dest);
				}
				else
				{
					absdiff(src_32f, src[c], src_32f);
					add(src_32f, dest, dest);
				}
			}

			if (edgeDiffusionSigma != 0.f)
			{
				Size ksize = Size(3, 3);
				GaussianBlur(dest, dest, ksize, edgeDiffusionSigma);
			}
			return getmax(dest);
			//normalize_max1(dest);
			//normalize(dest, dest, 0.f, 1.f, NORM_MINMAX);
		}
		float computeTexturenessDoG(const Mat& src, Mat& dest, const float sigma)
		{
			vector<Mat> a(1);
			a[0] = src;
			return computeTexturenessDoG(a, dest, sigma);
		}

		float computeTexturenessGradientMax(const vector<Mat>& src, Mat& dest)
		{
			float maxval = 0.f;
			if (src.size() == 3) maxval = gradientMaxDiffuse3x3<3>(src, dest, false, edgeDiffusionSigma);
			//gradientMaxDiffuse3x3(src[0], dest, false, edgeDiffusionSigma);
			/*for (int c = 0; c < src.size(); c++)
			{
				gradientMaxDiffuse3x3(src[c], dest, c == 0 ? false : true, edgeDiffusionSigma);
			}*/
			/*
			for (int c = 0; c < src.size(); c++)
			{
				gradientMax(src[c], dest, c == 0 ? false : true);
			}

			if (edgeDiffusionSigma != 0)
			{
				Size ksize = Size(3, 3);
				GaussianBlur(dest, dest, ksize, edgeDiffusionSigma);
			}
			*/
			//mul
			return maxval;
			//multiply(dest, 1.f / maxval, dest);
			//normalize_max1(dest);
			//normalize(dest, dest, 0.f, 1.f, NORM_MINMAX);
		}
		float computeTexturenessGradientMax(const Mat& src, Mat& dest)
		{
			vector<Mat> a(1);
			a[0] = src;
			return computeTexturenessGradientMax(a, dest);
		}

		void computeTexturenessBilateral(const vector<Mat>& src, Mat& dest, const float sigmaSpace, const float sr = 70.f)
		{
			for (int c = 0; c < src.size(); c++)
			{
				bilateralFilter(src[c], src_32f, (int)ceil(sigmaSpace * 2) * 2 + 1, sr, sigmaSpace);
				//bilateralFilterLocalStatisticsPrior(src[c], src_32f, float(sr), (float)ss1, sr * 0.8f);
				if (c == 0)
				{
					absdiff(src_32f, src[c], dest);
				}
				else
				{
					absdiff(src_32f, src[c], src_32f);
					add(src_32f, dest, dest);
				}
			}

			if (edgeDiffusionSigma != 0)
			{
				Size ksize = Size(3, 3);
				GaussianBlur(dest, dest, ksize, edgeDiffusionSigma);
			}

			normalize(dest, dest, 0.f, 1.f, NORM_MINMAX);
		}

		void remap(Mat& srcdest, const float scale, const float rangemax)
		{
			const int n = srcdest.size().area();
			const __m256 ms = _mm256_set1_ps(scale / rangemax);
			const __m256 ones = _mm256_set1_ps(1.f);
			//#pragma omp parallel for schedule (dynamic)
			const int sizeSimd = n / 8;
			float* srcPtr = srcdest.ptr<float>();
			//result[i] = min(v[i] * s, 1.f);
			for (int i = 0; i < sizeSimd; i++)
			{
				_mm256_store_ps(srcPtr, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(srcPtr), ms), ones));
				srcPtr += 8;
			}
		}
		float computeScaleHistogram(Mat& src, const float sampling_ratio, const int numBins = 100, const float rangemin = 0.f, const float rangemax = 1.f)
		{
			//cp::Timer t("tt2");
			//int binNum = (int)(bin_ratio*dest.size().area()*sampling_ratio);
			if constexpr (false) //opencv
			{
				int histSize[] = { numBins };
				float value_ranges[] = { rangemin, rangemax };
				const float* ranges[] = { value_ranges };
				const int channels[] = { 0 };
				const int dims = 1;
				calcHist(&src, 1, channels, Mat(), histogram, dims, histSize, ranges, true, false);
			}
			else
			{
				histogram.create(Size(numBins, 1), CV_32S);
				histogram.setTo(0);
				int* h = histogram.ptr<int>();
				const float* s = src.ptr<float>();
				const float amp = numBins / rangemax;
				for (int i = 0; i < src.size().area(); i++)
				{
					const int idx = int(s[i] * amp);
					h[idx]++;
				}
			}
			
			int H_k = 0;//cumulative sum of histogram
			float X_k = 0.f;//sum of hi*xi

			float scale = 0.f;//scaling factor
			float x = 0.f;//bin center
			const float inv_m = 1.f / numBins;//1/m
			const float offset = inv_m * 0.5f;
			const int n = src.size().area();
			const int nt = int(n * (1.f - sampling_ratio));
			const float sx_max = 1.f + FLT_EPSILON;
			const float sx_min = 1.f - FLT_EPSILON;
			//cout << n<<","<<nt<<"," <<sampling_ratio<< endl;
			for (int i = 0; i < numBins; i++)
			{
				//const int h_i = saturate_cast<int>(hist.at<float>(i));
				const int h_i = histogram.at<int>(i);
				H_k += h_i;

				x = i * inv_m + offset;
				X_k += x * h_i;

				scale = (H_k - nt) / X_k;//eq (5)
				const float sx = scale * x;
				if (sx_min < sx /*&& sx < sx_max*/)
				{
					break;
				}
			}
			return scale;
		}

		//source [0:1]
		//_MM_XORSHIFT32_AVX2 mrand;
		int randDither(const Mat& src, Mat& dest)
		{
			const __m256 normal = _mm256_set1_ps(1.f / (INT_MAX * 2.f));
			__m256i x = _mm256_set_epi32(0xd5eae750, 0xc784b986, 0x16bcf701, 0x65032360, 0xb628094f, 0xd8281e7b, 0xecfa5dc8, 0x3b828203);

			const float* s = src.ptr<float>();
			uchar* d = dest.ptr<uchar>();

			const __m256 maxval = _mm256_set1_ps(255.f);
			const __m256 mone = _mm256_set1_ps(1.0f);
			__m256i mcount = _mm256_setzero_si256();
			const int size = src.size().area();
			const __m256i fmask = _mm256_set1_epi32(0x3f800000);
			//unsigned res = (v >> 9) | 0x3f800000;
			//return (*(float*)&res) - 1.0f;

			if constexpr (true)
			{
				for (int i = 0; i < size; i += 8)
				{
					x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
					x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
					x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
					//__m256 mrng = _mm256_mul_ps(normal, _mm256_abs_ps(_mm256_cvtepi32_ps(x)));
					//__m256 mrng = _mm256_fmadd_ps(normal, _mm256_cvtepi32_ps(x), mone);
					__m256 mrng = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_srli_epi32(x, 8), fmask)), mone);
					//__m256 mrng = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_srli_epi32(x, 9), fmask)), mone);
					__m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(s + i), mrng, _CMP_LE_OQ);
					__m256 mdst = _mm256_andnot_ps(mask, maxval);
					mcount = _mm256_sub_epi32(mcount, _mm256_castps_si256(mask));
					_mm_storel_epi64((__m128i*)(d + i), _mm256_cvtps_epu8(mdst));
				}
			}
			else
			{
				for (int i = 0; i < size; i += 16)
				{
					x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
					x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
					x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
					__m256 mrng = _mm256_mul_ps(normal, _mm256_abs_ps(_mm256_cvtepi32_ps(x)));
					__m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(s + i), mrng, _CMP_LE_OQ);
					__m256 mdst = _mm256_andnot_ps(mask, maxval);
					mcount = _mm256_sub_epi32(mcount, _mm256_castps_si256(mask));
					_mm_storel_epi64((__m128i*)(d + i), _mm256_cvtps_epu8(mdst));

					x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
					x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
					x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
					mrng = _mm256_mul_ps(normal, _mm256_abs_ps(_mm256_cvtepi32_ps(x)));
					mask = _mm256_cmp_ps(_mm256_loadu_ps(s + i + 8), mrng, _CMP_LE_OQ);
					mdst = _mm256_andnot_ps(mask, maxval);
					mcount = _mm256_sub_epi32(mcount, _mm256_castps_si256(mask));
					_mm_storel_epi64((__m128i*)(d + i + 8), _mm256_cvtps_epu8(mdst));
				}
			}

			const int count = size - _mm256_reduceadd_epi32(mcount);
			//cout << count<<"/"<< src.size().area() << endl;
			/*
			//RNG rng;
			_MM_XORSHIFT32_AVX2 mrand;

			const float* s = src.ptr<float>();
			uchar* d = dest.ptr<uchar>();
			int count = 0;
			for (int i = 0; i < src.size().area(); i++)
			{
				__m256 vv = mrand.next32f();
				//const float v = rng.uniform(0.f, 1.f);
				const float v = vv.m256_f32[0];
				if ((v > s[i]))
				{
					d[i] = 0;
				}
				else
				{
					d[i] = 255;
					count++;
				}
			}
			*/
			/*randu(dest, 0, 256);
			const float* s = src.ptr<float>();
			uchar* d = dest.ptr<uchar>();
			int count = 0;
			for (int i = 0; i < src.size().area(); i++)
			{
				if ((d[i] > s[i] * 255.f))
				{
					d[i] = 0;
				}
				else
				{
					d[i] = 255;
					count++;
				}
			}*/

			return count;
		}

		template<int channels>
		int remapRandomDitherSamples(const int sample_num, const float scale, const float rangemax, const Mat& prob, const vector<cv::Mat>& guide, cv::Mat& dest)
		{
			const float smax = scale / rangemax;
			//const int channels = (int)guide.size();
			if (dest.size() != Size(sample_num, channels)) dest.create(Size(sample_num, channels), CV_32F);

			AutoBuffer<const float*> s(channels);
			AutoBuffer<float*> d(channels);
			for (int c = 0; c < channels; c++)
			{
				s[c] = guide[c].ptr<float>();
			}
			for (int c = 0; c < channels; c++)
			{
				d[c] = dest.ptr<float>(c);
			}

			const int size = prob.size().area();
			int count = 0;
			const float* pptr = prob.ptr<float>();
			for (int i = 0; i < size; i++)
			{
				const float v = rng.uniform(0.f, 1.f);
				if ((v <= min(1.f, smax * pptr[i])))
				{
					for (int c = 0; c < channels; c++)
					{
						d[c][count] = s[c][i];
					}
					count++;
					if (count == sample_num) return sample_num;
				}
			}
			for (int i = count; i < sample_num; i++)
			{
				const int idx = rng.uniform(0, count);
				for (int c = 0; c < channels; c++)
				{
					d[c][i] = d[c][idx];
				}
			}

			return count;
		}

		int dither(Mat& inportance, Mat& dest, const int ditheringMethod)
		{
			int sample_num = 0;
			if (ditheringMethod >= 0)
			{
				sample_num = ditherDestruction(importance, dest, ditheringMethod, MEANDERING);
			}
			else
			{
				sample_num = randDither(importance, dest);
			}
			return sample_num;
		}

	public:
		TexturenessSampling(const float edgeDiffusionSigma) :
			edgeDiffusionSigma(edgeDiffusionSigma)
		{
			;
		}

		void generateDoG(const vector<Mat>& src, Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const float sigma)
		{
			//generate(src[0], dest, sample_num, sampling_ratio, ditheringMethod); 
			//cp::imshowScale("src[0]", src[0]); imshow("importance", importance); imshow("mask", dest); waitKey();
			//return;

			if (isUseAverage)
			{
				cp::cvtColorAverageGray(src, gray, true);
				generateDoG(gray, dest, sample_num, sampling_ratio, ditheringMethod, sigma);
			}
			else
			{
				if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != src[0].size()) dest.create(src[0].size(), CV_8UC1);

				importance.create(src[0].rows, src[0].cols, CV_32F);//importance map (n pixels)
				float maxval = computeTexturenessDoG(src, importance, sigma);
				computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
				sample_num = dither(importance, dest, ditheringMethod);
				//cout << sample_num << endl;
				//cp::imshowScale("src[0]", src[0]); cp::imshowScale("importance", importance, 255); imshow("mask", dest); waitKey();
			}
		}

		void generateDoG(const Mat& src, Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod, const float sigma)
		{
			//CV_Assert(src.depth() == CV_32F);	
			if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != src.size())dest.create(src.size(), CV_8UC1);

			importance.create(src.rows, src.cols, CV_32F);//importance map (n pixels)
			float maxval = computeTexturenessDoG(src, importance, sigma);
			float scale = computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
			remap(importance, scale, maxval);
			sample_num = dither(importance, dest, ditheringMethod);
		}

		void generateGradientMax(const vector<Mat>& src, Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage)
		{
			//generate(src[0], dest, sample_num, sampling_ratio, ditheringMethod); 
			//cp::imshowScale("src[0]", src[0]); imshow("importance", importance); imshow("mask", dest); waitKey();
			//return;

			if (isUseAverage)
			{
				cout << "generateGradientMax use average" << endl;
				cp::cvtColorAverageGray(src, gray, true);
				generateGradientMax(gray, dest, sample_num, sampling_ratio, ditheringMethod);
			}
			else
			{
				if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != src[0].size()) dest.create(src[0].size(), CV_8UC1);

				importance.create(src[0].rows, src[0].cols, CV_32F);//importance map (n pixels)
				const float maxval = computeTexturenessGradientMax(src, importance);
				computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
				sample_num = dither(importance, dest, ditheringMethod);
			}
		}
		void generateGradientMax(const Mat& src, Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod)
		{
			//CV_Assert(src.depth() == CV_32F);	
			if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != src.size())dest.create(src.size(), CV_8UC1);

			importance.create(src.rows, src.cols, CV_32F);//importance map (n pixels)
			const float maxval = computeTexturenessGradientMax(src, importance);
			computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
			sample_num = dither(importance, dest, ditheringMethod);
		}

		int generateGradientMaxSamples(const vector<Mat>& image, Mat& samples, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage)
		{
			int sample_num = 0;
			//generate(src[0], dest, sample_num, sampling_ratio, ditheringMethod); 
			//cp::imshowScale("src[0]", src[0]); imshow("importance", importance); imshow("mask", dest); waitKey();
			//return;

			if (isUseAverage)
			{
				cout << "generateGradientMax use average" << endl;
				cp::cvtColorAverageGray(image, gray, true);
				generateGradientMax(gray, samples, sample_num, sampling_ratio, ditheringMethod);
			}
			else
			{
				importance.create(image[0].rows, image[0].cols, CV_32F);//importance map (n pixels)
				const float maxval = computeTexturenessGradientMax(image, importance);
				const float scale = computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
				sample_num = remapRandomDitherSamples<3>(int(image[0].size().area() * (sampling_ratio)), scale, maxval, importance, image, samples);
			}
			return sample_num;
		}
	};

	void generateSamplingMaskRemappedTextureness(const Mat& src, Mat& dest, int& sample_num, const float samplingRatio, const int ditheringMethod, const float bin_ratio)
	{
		//CV_Assert(src.depth() == CV_32F);
#if 0
		static int ss1 = 2; createTrackbar("ss1", "", &ss1, 10);
		static int ss2 = 0; createTrackbar("ss2", "", &ss2, 100);
		static int sr = 70; createTrackbar("sr", "", &sr, 255);
		static int type = 0; createTrackbar("type", "", &type, 2);
#else
		float ss1 = 3.f;
		float ss2 = 1.f;
		float sr = 70.f;
		const int type = 0;
#endif
		if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != src.size())dest.create(src.size(), CV_8UC1);
		TexturenessSampling tims(ss2);
		tims.generateDoG(src, dest, sample_num, samplingRatio, ditheringMethod, false, ss1);
	}


	inline int reagionDiff(Mat& dispMap, int posX, int posY)
	{
		int diff = 0;
		int r = 2;
		for (int y = -r; y <= r; y++)
		{
			for (int x = -r; x <= r; x++)
			{
				if (x == 0 && y == 0)
					continue;
				diff += abs(dispMap.at<uchar>(posY, posX) - dispMap.at<uchar>(posY + y, posX + x));
			}
		}
		return diff;
	}

	inline void reagionRefine(Mat& importanceMap8u, int grid_space, int posX, int posY, int& sample_num, int sample_num_end)
	{
		if (sample_num >= sample_num_end) return;
		int k = grid_space / 2;

		for (int y = -k; y <= k; y++)
		{
			for (int x = -k; x <= k; x++)
			{
				if (importanceMap8u.at<uchar>(posY + y, posX + x) == 255)
					continue;
				importanceMap8u.at<uchar>(posY + y, posX + x) = 255;
				sample_num++;
				if (sample_num >= sample_num_end)
					break;
			}
		}

	}

	struct Sort
	{
		bool operator() (cv::Point3i pt1, cv::Point3i pt2) { return (pt1.z < pt2.z); }
	}sortobj;

	void createSamplingOffset(Mat& importanceMap, int*& importance_ofs, int*& importance_ofs_store, int& sample_num, const int r, const int src_step, bool isAVXPadding)
	{
		const int sstep = src_step;
		const int dstep = importanceMap.cols;

		CV_Assert(importanceMap.depth() == CV_8U);

		int num = sample_num;// countNonZero(importanceMap);
		int pad;
		if (isAVXPadding)
		{
			pad = (8 - num % 8) % 8;
			//pad = (isAVXPadding) ? pad : 0;
			sample_num = num + pad;
		}
		else
		{
			pad = (4 - num % 4) % 4;
			sample_num = num + pad;
		}

		if (isAVXPadding)
		{
			importance_ofs = (int*)_mm_malloc(sizeof(int) * (sample_num), 32);
			importance_ofs_store = (int*)_mm_malloc(sizeof(int) * (sample_num), 32);
		}
		else
		{
			importance_ofs = (int*)_mm_malloc(sizeof(int) * (sample_num), 16);
			importance_ofs_store = (int*)_mm_malloc(sizeof(int) * (sample_num), 16);
		}

		int index = 0;

		for (int y = 0; y < importanceMap.rows; y++)
		{
			uchar* spix = importanceMap.ptr<uchar>(y);
			for (int x = 0; x < importanceMap.cols; x++)
			{
				if (spix[x] != 0)
				{
					importance_ofs[index] = (y + r) * sstep + (x + r);
					importance_ofs_store[index++] = y * dstep + x;
				}
			}
		}
		for (int i = 0; i < pad; i++)
		{
			importance_ofs[index] = (importanceMap.rows - 1 + r) * sstep + (importanceMap.cols - 1 + r);
			importance_ofs_store[index++] = (importanceMap.rows - 1) * dstep + importanceMap.cols - 1;
		}
	}


	void generateSamplingMaskRemappedDitherTexturenessPackedAoS(cv::Mat& src, cv::Mat& dest, const float sampling_ratio, int ditheringMethod)
	{
		CV_Assert(src.depth() == CV_32F);
		if (src.channels() == 3)
		{
			//cout << "3 channels in createImportanceSampledReshapedImage" << endl;
			//int64 start, end;
			// src...8UC3
			cv::Mat src_gray;
			//cv::cvtColor(src, src_gray, COLOR_BGR2GRAY);
			cp::cvtColorAverageGray(src, src_gray);

			int sample_num;
			cv::Mat mask;
			//start = cv::getTickCount();
			generateSamplingMaskRemappedTextureness(src_gray, mask, sample_num, sampling_ratio, ditheringMethod);
			dest.create(Size(1, sample_num), CV_32FC3);
			//end = cv::getTickCount();
			//std::cout << "Createmask time:" << (end - start) * 1000 / (cv::getTickFrequency()) << std::endl;

			//imshow("mask", mask);
			//waitKey();


			//start = cv::getTickCount();
			Vec3f* reshaped_ptr = dest.ptr<Vec3f>();
			for (int y = 0; y < mask.rows; y++)
			{
				uchar* mask_ptr = mask.ptr<uchar>(y);
				Vec3f* src_ptr = src.ptr<Vec3f>(y);
				for (int x = 0; x < mask.cols; x++)
				{
					if (mask_ptr[x] == 255)
					{
						reshaped_ptr[0] = src_ptr[x];
						reshaped_ptr++;
					}
				}
			}
			//end = cv::getTickCount();
			//std::cout << "packing time:" << (end - start) * 1000 / (cv::getTickFrequency()) << std::endl;

			//cout << dest.size() << endl;
			//cout << dest.channels() << endl;
		}
		else if (src.channels() == 1)
		{
			//cout << "1 channels in createImportanceSampledReshapedImage" << endl;
			int sample_num;
			cv::Mat mask;
			//start = cv::getTickCount();
			generateSamplingMaskRemappedTextureness(src, mask, sample_num, sampling_ratio, ditheringMethod);
			dest.create(Size(1, sample_num), CV_32F);

			float* reshaped_ptr = dest.ptr<float>();
			for (int y = 0; y < mask.rows; y++)
			{
				uchar* mask_ptr = mask.ptr<uchar>(y);
				float* src_ptr = src.ptr<float>(y);
				for (int x = 0; x < mask.cols; x++)
				{
					if (mask_ptr[x] == 255)
					{
						reshaped_ptr[0] = src_ptr[x];
						reshaped_ptr++;
					}
				}
			}
		}
		else
		{
			//cout <<src.channels()<< " channels in createImportanceSampledReshapedImage" << endl;
			//int64 start, end;
			// src...8UC3
			cv::Mat src_gray(src.size(), CV_32F);
			float* in = src.ptr<float>();
			float* sss = src_gray.ptr<float>();
			const int channels = src.channels();
			for (int i = 0; i < src.size().area(); i++)
			{
				sss[i] = in[channels * i + 0];
			}

			int sample_num;
			cv::Mat mask;
			//start = cv::getTickCount();
			generateSamplingMaskRemappedTextureness(src_gray, mask, sample_num, sampling_ratio, ditheringMethod);
			dest.create(Size(1, sample_num), CV_MAKETYPE(CV_32F, channels));

			float* reshaped_ptr = dest.ptr<float>();
			for (int y = 0; y < mask.rows; y++)
			{
				uchar* mask_ptr = mask.ptr<uchar>(y);
				float* src_ptr = src.ptr<float>(y);
				for (int x = 0; x < mask.cols; x++)
				{
					if (mask_ptr[x] == 255)
					{
						for (int c = 0; c < channels; c++)
						{
							reshaped_ptr[c] = src_ptr[channels * x + c];
						}
						reshaped_ptr += channels;
					}
				}
			}
		}
	}


	static void mask2sample(const int sample_num, const Mat& sampleMask, const vector<cv::Mat>& guide, cv::Mat& dest)
	{
		const int channels = (int)guide.size();
		if (dest.size() != Size(sample_num, channels)) dest.create(Size(sample_num, channels), CV_32F);

		AutoBuffer<const float*> s(channels);
		AutoBuffer<float*> d(channels);
		for (int c = 0; c < channels; c++)
		{
			d[c] = dest.ptr<float>(c);
		}

		for (int y = 0, count = 0; y < sampleMask.rows; y++)
		{
			for (int c = 0; c < channels; c++)
			{
				s[c] = guide[c].ptr<float>(y);
			}

			const uchar* mask_ptr = sampleMask.ptr<uchar>(y);
			for (int x = 0; x < sampleMask.cols; x++)
			{
				if (mask_ptr[x] == 255)
				{
					for (int c = 0; c < channels; c++)
					{
						d[c][count] = s[c][x];
					}
					count++;
					if (count == sample_num) return;
				}
			}
		}
	}

	void generateSamplingMaskRemappedDitherTexturenessDoGPackedSoA(const vector<cv::Mat>& guide, cv::Mat& dest, const float sampling_ratio, const bool isUseAverage, int ditheringMethod, cv::Mat sampleMask, const float sigma)
	{
		CV_Assert(guide[0].depth() == CV_32F);

		//TexturenessImportanceMapSampling ims(2.9, 0.7, 70, 0);
		//TexturenessImportanceMapSampling ims(3.5, 1.5, 70, 0);
		TexturenessSampling ims(0.7f);
		int sample_num = 0;
		ims.generateDoG(guide, sampleMask, sample_num, sampling_ratio, ditheringMethod, isUseAverage, sigma);
		sample_num = get_simd_floor(sample_num, 8);
		mask2sample(sample_num, sampleMask, guide, dest);
	}

	void generateSamplingMaskRemappedDitherTexturenessGradientMaxPackedSoA(const vector<cv::Mat>& guide, cv::Mat& dest, const float sampling_ratio, const bool isUseAverage, int ditheringMethod, cv::Mat sampleMask)
	{
		CV_Assert(guide[0].depth() == CV_32F);

		//TexturenessImportanceMapSampling ims(2.9, 0.7, 70, 0);
		TexturenessSampling ims(0.7f);
		//TexturenessImportanceMapSampling ims(3.5, 1.5, 70, 0);
		/*
		ims.generateGradientMax(guide, sampleMask, sample_num, sampling_ratio, ditheringMethod, isUseAverage);
		sample_num = get_simd_floor(sample_num, 8);
		mask2sample(sample_num, sampleMask, guide, dest);
		*/
		int sample_num = ims.generateGradientMaxSamples(guide, dest, sampling_ratio, ditheringMethod, isUseAverage);
	}

	void generateSamplingMaskRemappedDitherTexturenessPackedSoA(const vector<cv::Mat>& src, vector<cv::Mat>& guide, cv::Mat& dest, const float sampling_ratio, int ditheringMethod)
	{
		CV_Assert(guide[0].depth() == CV_32F);
		const int channels = (int)guide.size();

		if (channels == 3)
		{
			//cout << "3 channels in createImportanceSampledReshapedImage" << endl;
			//int64 start, end;
			cv::Mat src_gray;
			//cp::cvtColorAverageGray(guide, src_gray);
			guide[0].copyTo(src_gray);

			int sample_num = 0;
			cv::Mat mask;
			//start = cv::getTickCount();
			generateSamplingMaskRemappedTextureness(src_gray, mask, sample_num, sampling_ratio, ditheringMethod);
			sample_num = get_simd_floor(sample_num, 8);
			//print_debug(sample_num);

			dest.create(Size(sample_num, 3), CV_32F);
			//end = cv::getTickCount();
			//std::cout << "Createmask time:" << (end - start) * 1000 / (cv::getTickFrequency()) << std::endl;
			//imshow("mask", mask);
			//waitKey();
			//start = cv::getTickCount();
			float* d0 = dest.ptr<float>(0);
			float* d1 = dest.ptr<float>(1);
			float* d2 = dest.ptr<float>(2);
			for (int y = 0, count = 0; y < mask.rows; y++)
			{
				uchar* mask_ptr = mask.ptr<uchar>(y);
				float* s0 = guide[0].ptr<float>(y);
				float* s1 = guide[1].ptr<float>(y);
				float* s2 = guide[2].ptr<float>(y);
				for (int x = 0; x < mask.cols; x++)
				{
					if (mask_ptr[x] == 255)
					{
						d0[count] = s0[x];
						d1[count] = s1[x];
						d2[count] = s2[x];
						count++;
						if (count == sample_num)return;
					}
				}
			}
			//end = cv::getTickCount();
			//std::cout << "packing time:" << (end - start) * 1000 / (cv::getTickFrequency()) << std::endl;
		}
		else if (channels == 1)
		{
			//cout << "1 channels in createImportanceSampledReshapedImage" << endl;
			int sample_num;
			cv::Mat mask;
			//start = cv::getTickCount();
			generateSamplingMaskRemappedTextureness(guide[0], mask, sample_num, sampling_ratio, ditheringMethod);
			sample_num = get_simd_floor(sample_num, 8);
			dest.create(Size(sample_num, 1), CV_32F);

			float* d = dest.ptr<float>();
			for (int y = 0, count = 0; y < mask.rows; y++)
			{
				uchar* mask_ptr = mask.ptr<uchar>(y);
				float* s = guide[0].ptr<float>(y);
				for (int x = 0; x < mask.cols; x++)
				{
					if (mask_ptr[x] == 255)
					{
						d[count] = s[x];
						count++;
						if (count == sample_num)return;
					}
				}
			}
		}
		else
		{
			//cout <<src.channels()<< " channels in createImportanceSampledReshapedImage" << endl;
			//int64 start, end;
			cv::Mat src_gray(guide[0].size(), CV_32F);
			float* in = guide[0].ptr<float>();
			float* sss = src_gray.ptr<float>();

			cp::cvtColorAverageGray(src, src_gray);
			//cp::detailEnhancementGauss(src_gray, src_gray, 1, 2, 1);

			//print_matinfo_detail(src_gray);
			//guide[1].copyTo(src_gray);

			int sample_num = 0;
			cv::Mat mask;
			//start = cv::getTickCount();
			generateSamplingMaskRemappedTextureness(src_gray, mask, sample_num, sampling_ratio, ditheringMethod);
			sample_num = get_simd_floor(sample_num, 8);
			dest.create(Size(sample_num, channels), CV_32F);

			AutoBuffer<float*> s(channels);
			AutoBuffer<float*> d(channels);
			for (int c = 0; c < channels; c++)
			{
				d[c] = dest.ptr<float>(c);
			}

			for (int y = 0, count = 0; y < mask.rows; y++)
			{
				uchar* mask_ptr = mask.ptr<uchar>(y);
				for (int c = 0; c < channels; c++)
				{
					s[c] = guide[c].ptr<float>(y);
				}

				for (int x = 0; x < mask.cols; x++)
				{
					if (mask_ptr[x] == 255)
					{
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][x];
						}
						count++;
						if (count == sample_num)return;
					}
				}
			}
		}
	}
}