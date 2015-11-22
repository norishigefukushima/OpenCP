#include "opencp.hpp"
#define USE_OPENMP 1

using namespace std;
using namespace cv;

namespace cp
{
	RealtimeO1BilateralFilter::RealtimeO1BilateralFilter()
	{
		isSaveMemory = false;
		setBinDepth(CV_32F);
		setColorNorm(L1SQR);
		downsampleSizeSplatting = 1;
		downsampleSizeBlurring = 1;
		downsampleMethod = INTER_AREA;
		upsampleMethod = INTER_CUBIC;

		bin2num.resize(256);
		idx.resize(256);
		a.resize(256);
	}

	RealtimeO1BilateralFilter::~RealtimeO1BilateralFilter()
	{
		;
	}

	void RealtimeO1BilateralFilter::setBinDepth(int depth)
	{
		bin_depth = depth;
	}

	void RealtimeO1BilateralFilter::createBin(Size imsize, int number_of_bin, int channles)
	{
		normalize_sub_range.clear();
		sub_range.clear();

		for (int i = 0; i < num_bin; i++)
		{
			sub_range.push_back(Mat::zeros(imsize, CV_MAKETYPE(bin_depth, channles)));
			normalize_sub_range.push_back(Mat::zeros(imsize, CV_MAKETYPE(bin_depth, channles)));
		}
	}

	void RealtimeO1BilateralFilter::setColorLUT(double sigma_color, int channlels)
	{
		double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		if (bin_depth == CV_32F)
		{
			for (int i = 0; i < 256 * channlels; i++)
			{
				//color_weight_32F[i] = max((float)std::exp(i*i*gauss_color_coeff), 0.00000f);//avoid 0 value
				color_weight_32F[i] = (float)std::exp(i*i*gauss_color_coeff);
			}
		}
		else if (bin_depth == CV_64F)
		{
			for (int i = 0; i < 256 * channlels; i++)
			{
				//color_weight[i] = max((float)std::exp(i*i*gauss_color_coeff), 0.00001f);//avoid 0 value
				color_weight_64F[i] = std::exp(i*i*gauss_color_coeff);
			}
		}
		else
		{
			fprintf(stderr, "only support 32F and 64F for LUT\n");
		}
	}

	void RealtimeO1BilateralFilter::showBinIndex()
	{
		int prev = idx[0];
		for (int i = 0; i < 256; i++)
		{
			printf("%3d %2d %f", i, idx[i], a[i]);
			if (idx[i] != prev)
			{
				//printf("%d %d\n", i, idx[i]);
				prev = idx[i];
				printf(" *");
			}
			printf("\n");
		}
	}

	void RealtimeO1BilateralFilter::disposeBin(int number_of_bin)
	{
		float astep = (255.f / (float)(number_of_bin - 1));
		int bin = 0;
		int pindex = 0;

		bin2num[bin] = 0;
		for (int i = 0; i < 256; i++)
		{
			idx[i] = (uchar)((float)i / astep);
			if (pindex != idx[i])
			{
				bin++;
				pindex = idx[i];
				bin2num[bin] = i;
			}
		}
		bin2num[number_of_bin - 1] = 255;

		for (int i = 0; i < 255; i++)
		{
			a[i] = 1.f - (i - bin2num[idx[i]]) / (float)(bin2num[idx[i] + 1] - bin2num[idx[i]]);
		}

		//fourcely set (last -1) bin for avoiding error in interpoation
		idx[255] = (uchar)(number_of_bin - 2);
		a[255] = 0.f;
		//bin2num[number_of_bin - 1] = bin2num[number_of_bin - 2];
	}

	void RealtimeO1BilateralFilter::setColorNorm(int normType_)
	{
		normType = normType_;
	}

	void RealtimeO1BilateralFilter::blurring(const Mat& src_, Mat& dest_)
	{
		Mat src, dest;

		downsampleSizeBlurring = max(downsampleSizeBlurring, 1);
		downsampleSizeSplatting = max(downsampleSizeSplatting, 1);

		int dsize = downsampleSizeSplatting*downsampleSizeBlurring;
		if (downsampleSizeBlurring == 1)
		{
			src = src_;
			dest = dest_;
		}
		else
		{
			resize(src_, src, Size(src_.cols / downsampleSizeBlurring, src_.rows / downsampleSizeBlurring), 0.0, 0.0, downsampleMethod);
		}

		if (filter_type == FIR_SEPARABLE)
		{
			Size kernel = Size(2 * (radius / dsize) + 1, 2 * (radius / dsize) + 1);
			GaussianBlur(src, dest, kernel, sigma_space / dsize, 0.0, BORDER_REPLICATE);
		}
		else if (filter_type == IIR_AM)
		{
			if (bin_depth == CV_64F) GaussianFilter(src, dest, (sigma_space / dsize), GAUSSIAN_FILTER_AM2, filterK);
			else GaussianFilter(src, dest, (sigma_space / dsize), GAUSSIAN_FILTER_AM, filterK);
		}
		else if (filter_type == IIR_SR)
		{
			GaussianFilter(src, dest, (sigma_space / dsize), GAUSSIAN_FILTER_SR, filterK);
		}
		else if (filter_type == IIR_Deriche)
		{
			GaussianFilter(src, dest, (sigma_space / dsize), GAUSSIAN_FILTER_DERICHE, filterK);
		}
		else if (filter_type == IIR_YVY)
		{
			GaussianFilter(src, dest, (sigma_space / dsize), GAUSSIAN_FILTER_VYV, filterK);
		}
		if (downsampleSizeBlurring != 1)
		{
			resize(dest, dest_, src_.size(), 0.0, 0.0, upsampleMethod);
		}
	}

	template <typename T, typename S>
	void RealtimeO1BilateralFilter::splattingColor(const T* s, S* su, S* sd, const uchar* j, const uchar* v, const int imageSize, const int channels, const int type)
	{
		if (type == L1SQR)
		{
			if (channels == 3)
			{
				if (typeid(S) == typeid(float))
				{
					for (int i = 0; i < imageSize; i++)
					{
						const float coeff = color_weight_32F[abs(j[3 * i + 0] - v[0]) + abs(j[3 * i + 1] - v[1]) + abs(j[3 * i + 2] - v[2])];
						su[3 * i + 0] = (S)coeff*(S)s[3 * i + 0];
						su[3 * i + 1] = (S)coeff*(S)s[3 * i + 1];
						su[3 * i + 2] = (S)coeff*(S)s[3 * i + 2];
						sd[3 * i + 0] = (S)coeff;
						sd[3 * i + 1] = (S)coeff;
						sd[3 * i + 2] = (S)coeff;
					}
				}
				else if (typeid(S) == typeid(double))
				{
					for (int i = 0; i < imageSize; i++)
					{
						const double coeff = color_weight_64F[abs(j[3 * i + 0] - v[0]) + abs(j[3 * i + 1] - v[1]) + abs(j[3 * i + 2] - v[2])];
						su[3 * i + 0] = (S)coeff*(S)s[3 * i + 0];
						su[3 * i + 1] = (S)coeff*(S)s[3 * i + 1];
						su[3 * i + 2] = (S)coeff*(S)s[3 * i + 2];
						sd[3 * i + 0] = (S)coeff;
						sd[3 * i + 1] = (S)coeff;
						sd[3 * i + 2] = (S)coeff;
					}
				}
			}
			else if (channels == 1)
			{
				if (typeid(S) == typeid(float))
				{
					for (int i = 0; i < imageSize; i++)
					{
						const float coeff = color_weight_32F[abs(j[3 * i + 0] - v[0]) + abs(j[3 * i + 1] - v[1]) + abs(j[3 * i + 2] - v[2])];
						su[i] = coeff*(float)s[i];
						sd[i] = coeff;
					}
				}
				else if (typeid(S) == typeid(double))
				{
					for (int i = 0; i < imageSize; i++)
					{
						const double coeff = color_weight_64F[abs(j[3 * i + 0] - v[0]) + abs(j[3 * i + 1] - v[1]) + abs(j[3 * i + 2] - v[2])];
						su[i] = (S)coeff*(S)s[i];
						sd[i] = (S)coeff;
					}
				}
			}
		}
		else if (type == L2)
		{
			if (channels == 3)
			{
				if (typeid(S) == typeid(float))
				{
					for (int i = 0; i < imageSize; i++)
					{
						const float coeff =
							color_weight_32F[abs(j[3 * i + 0] - v[0])] *
							color_weight_32F[abs(j[3 * i + 1] - v[1])] *
							color_weight_32F[abs(j[3 * i + 2] - v[2])];
						su[3 * i + 0] = (S)coeff*(S)s[3 * i + 0];
						su[3 * i + 1] = (S)coeff*(S)s[3 * i + 1];
						su[3 * i + 2] = (S)coeff*(S)s[3 * i + 2];
						sd[3 * i + 0] = (S)coeff;
						sd[3 * i + 1] = (S)coeff;
						sd[3 * i + 2] = (S)coeff;
					}
				}
				else if (typeid(S) == typeid(double))
				{
					for (int i = 0; i < imageSize; i++)
					{
						const double coeff =
							color_weight_64F[abs(j[3 * i + 0] - v[0])] *
							color_weight_64F[abs(j[3 * i + 1] - v[1])] *
							color_weight_64F[abs(j[3 * i + 2] - v[2])];
						su[3 * i + 0] = (S)coeff*(S)s[3 * i + 0];
						su[3 * i + 1] = (S)coeff*(S)s[3 * i + 1];
						su[3 * i + 2] = (S)coeff*(S)s[3 * i + 2];
						sd[3 * i + 0] = (S)coeff;
						sd[3 * i + 1] = (S)coeff;
						sd[3 * i + 2] = (S)coeff;
					}
				}
			}
			else if (channels == 1)
			{
				if (typeid(S) == typeid(float))
				{
					for (int i = 0; i < imageSize; i++)
					{
						const float coeff =
							color_weight_32F[abs(j[3 * i + 0] - v[0])] *
							color_weight_32F[abs(j[3 * i + 1] - v[1])] *
							color_weight_32F[abs(j[3 * i + 2] - v[2])];
						su[i] = coeff*(float)s[i];
						sd[i] = coeff;
					}
				}
				else if (typeid(S) == typeid(double))
				{
					for (int i = 0; i < imageSize; i++)
					{
						const double coeff =
							color_weight_64F[abs(j[3 * i + 0] - v[0])] *
							color_weight_64F[abs(j[3 * i + 1] - v[1])] *
							color_weight_64F[abs(j[3 * i + 2] - v[2])];
						su[i] = (S)coeff*(S)s[i];
						sd[i] = (S)coeff;
					}
				}
			}
		}
		else
		{
			cout << "not support norm in spplatting color" << endl;
		}
	}

	template <typename T, typename S>
	void RealtimeO1BilateralFilter::splatting(const T* s, S* su, S* sd, const uchar* j, const uchar v, const int imageSize, const int channels)
	{
		if (channels == 3)
		{
			if (typeid(S) == typeid(float))
			{
				for (int i = 0; i < imageSize; i++)
				{
					const float coeff = color_weight_32F[cvRound(abs(j[i] - v))];
					su[3 * i + 0] = (S)coeff*(S)s[3 * i + 0];
					su[3 * i + 1] = (S)coeff*(S)s[3 * i + 1];
					su[3 * i + 2] = (S)coeff*(S)s[3 * i + 2];
					sd[3 * i + 0] = (S)coeff;
					sd[3 * i + 1] = (S)coeff;
					sd[3 * i + 2] = (S)coeff;
				}
			}
			else if (typeid(S) == typeid(double))
			{
				for (int i = 0; i < imageSize; i++)
				{
					const double coeff = color_weight_64F[cvRound(abs(j[i] - v))];
					su[3 * i + 0] = (S)coeff*(S)s[3 * i + 0];
					su[3 * i + 1] = (S)coeff*(S)s[3 * i + 1];
					su[3 * i + 2] = (S)coeff*(S)s[3 * i + 2];
					sd[3 * i + 0] = (S)coeff;
					sd[3 * i + 1] = (S)coeff;
					sd[3 * i + 2] = (S)coeff;
				}
			}
		}
		else if (channels == 1)
		{
			if (typeid(S) == typeid(float))
			{
				for (int i = 0; i < imageSize; i++)
				{
					const float coeff = color_weight_32F[cvRound(abs(j[i] - v))];
					su[i] = coeff*(float)s[i];
					sd[i] = coeff;
				}
			}
			else if (typeid(S) == typeid(double))
			{
				for (int i = 0; i < imageSize; i++)
				{
					const double coeff = color_weight_64F[cvRound(abs(j[i] - v))];
					su[i] = (S)coeff*(S)s[i];
					sd[i] = (S)coeff;
				}
			}
		}
	}

	void RealtimeO1BilateralFilter::body(InputArray src_, InputArray joint_, OutputArray dest_, bool save_memorySize)
	{
		if (dest_.empty()) dest_.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat joint = joint_.getMat();
		Mat dest = dest_.getMat();

		if (bin_depth == CV_32F)
		{
			if (src.depth() == CV_8U)  (save_memorySize) ? bodySaveMemorySize_<uchar, float>(src, joint, dest) : body_<uchar, float>(src, joint, dest);
			if (src.depth() == CV_16U) (save_memorySize) ? bodySaveMemorySize_<ushort, float>(src, joint, dest) : body_<ushort, float>(src, joint, dest);
			if (src.depth() == CV_16S) (save_memorySize) ? bodySaveMemorySize_<short, float>(src, joint, dest) : body_<short, float>(src, joint, dest);
			if (src.depth() == CV_32S) (save_memorySize) ? bodySaveMemorySize_<int, float>(src, joint, dest) : body_<int, float>(src, joint, dest);
			if (src.depth() == CV_32F) (save_memorySize) ? bodySaveMemorySize_<float, float>(src, joint, dest) : body_<float, float>(src, joint, dest);
			if (src.depth() == CV_64F) (save_memorySize) ? bodySaveMemorySize_<double, float>(src, joint, dest) : body_<double, float>(src, joint, dest);
		}
		else if (bin_depth == CV_64F)
		{
			if (src.depth() == CV_8U)  (save_memorySize) ? bodySaveMemorySize_<uchar, double>(src, joint, dest) : body_<uchar, double>(src, joint, dest);
			if (src.depth() == CV_16U) (save_memorySize) ? bodySaveMemorySize_<ushort, double>(src, joint, dest) : body_<ushort, double>(src, joint, dest);
			if (src.depth() == CV_16S) (save_memorySize) ? bodySaveMemorySize_<short, double>(src, joint, dest) : body_<short, double>(src, joint, dest);
			if (src.depth() == CV_32S) (save_memorySize) ? bodySaveMemorySize_<int, double>(src, joint, dest) : body_<int, double>(src, joint, dest);
			if (src.depth() == CV_32F) (save_memorySize) ? bodySaveMemorySize_<float, double>(src, joint, dest) : body_<float, double>(src, joint, dest);
			if (src.depth() == CV_64F) (save_memorySize) ? bodySaveMemorySize_<double, double>(src, joint, dest) : body_<double, double>(src, joint, dest);
		}
	}

	template <typename T, typename S>
	void RealtimeO1BilateralFilter::bodySaveMemorySize_(const Mat& src_, const Mat& guide_, Mat& dest)
	{
		if (bgrid.size() != 1 || bgrid[0].size() != src_.size())
		{
			bgrid.clear();
			bgrid.resize(1);
		}

		Mat src;
		Mat guide;
		Mat dst;
		if (typeid(S) == typeid(float)) dst = Mat::zeros(src_.size(), CV_MAKETYPE(CV_32F, src_.channels()));
		else dst = Mat::zeros(src_.size(), CV_MAKETYPE(CV_64F, src_.channels()));

		if (downsampleSizeSplatting != 1)
		{
			resize(src_, src, Size(src_.size().width / downsampleSizeSplatting, src_.size().height / downsampleSizeSplatting), 0, 0, downsampleMethod);
			resize(guide_, guide, Size(src_.size().width / downsampleSizeSplatting, src_.size().height / downsampleSizeSplatting), 0, 0, downsampleMethod);
		}
		else
		{
			src = src_;
			guide = guide_;
		}

		num_bin = max(num_bin, 2);// for 0 and 255
		if (dest.empty() || dest.size() != src_.size() || dest.type() != src_.type()) dest.create(src_.size(), src_.type());

		const int imageSize = src.size().area();
		const int imageSizeFull = src_.size().area();

		Mat joint = guide;

		setColorLUT(sigma_color, guide.channels());
		if (sub_range.size() != 1 || sub_range[0].size().area() != imageSize || sub_range[0].depth() != bin_depth)
		{
			createBin(src.size(), 1, src.channels());
		}
		disposeBin(num_bin);

		const T* s = src.ptr<T>(0);
		const uchar* j = joint.ptr<uchar>(0);
		const uchar* jfull = guide_.ptr<uchar>(0);
		S* d = dst.ptr<S>(0);
		if (src.channels() == 1 && guide.channels() == 1)
		{
			for (int b = 0; b < num_bin; b++)
			{
				S* su = sub_range[0].ptr<S>(0);//upper
				S* sd = normalize_sub_range[0].ptr<S>(0);//down

				uchar v = bin2num[b];
				splatting<T, S>(s, su, sd, j, v, imageSize, src.channels());

				blurring(sub_range[0], sub_range[0]);
				blurring(normalize_sub_range[0], normalize_sub_range[0]);

				if (downsampleSizeSplatting == 1)
				{
					divide(sub_range[0], normalize_sub_range[0], bgrid[0]);
				}
				else
				{
					divide(sub_range[0], normalize_sub_range[0], sub_range[0]);
					resize(sub_range[0], bgrid[0], src_.size(), 0, 0, upsampleMethod);
				}

				if (typeid(S) == typeid(float))
				{
					for (int i = 0; i < imageSizeFull; i++)
					{
						int id = idx[jfull[i]];
						float ca = a[jfull[i]];
						if (id + 1 == b)
							d[i] += (1.f - ca)*bgrid[0].at<float>(i);
						if (id == b)
							d[i] += (ca)*bgrid[0].at<float>(i);
					}
				}
				else if (typeid(S) == typeid(double))
				{
					for (int i = 0; i < imageSizeFull; i++)
					{
						int id = idx[jfull[i]];
						double ca = (double)a[jfull[i]];
						if (id + 1 == b)
							d[i] += (S)((1.0 - ca)*bgrid[0].at<double>(i));
						if (id == b)
							d[i] += (S)(ca*bgrid[0].at<double>(i));
					}
				}
			}
			dst.convertTo(dest, src.type());
		}
		else if (src.channels() == 3 && guide.channels() == 1)
		{
			for (int b = 0; b < num_bin; b++)
			{
				S* su = sub_range[0].ptr<S>(0);//upper
				S* sd = normalize_sub_range[0].ptr<S>(0);//down

				uchar v = bin2num[b];
				splatting<T, S>(s, su, sd, j, v, imageSize, src.channels());

				blurring(sub_range[0], sub_range[0]);
				blurring(normalize_sub_range[0], normalize_sub_range[0]);

				if (downsampleSizeSplatting == 1)
				{
					divide(sub_range[0], normalize_sub_range[0], bgrid[0]);
				}
				else
				{
					divide(sub_range[0], normalize_sub_range[0], sub_range[0]);
					resize(sub_range[0], bgrid[0], src_.size(), 0, 0, upsampleMethod);
				}

				if (typeid(S) == typeid(float))
				{
					for (int i = 0; i < imageSizeFull; i++)
					{
						int id = idx[jfull[i]];
						float ca = a[jfull[i]];
						if (id + 1 == b)
						{
							d[3 * i + 0] += (1.0f - ca)*bgrid[0].at<float>(3 * i + 0);
							d[3 * i + 1] += (1.0f - ca)*bgrid[0].at<float>(3 * i + 1);
							d[3 * i + 2] += (1.0f - ca)*bgrid[0].at<float>(3 * i + 2);
						}
						if (id == b)
						{
							d[3 * i + 0] += ca*bgrid[0].at<float>(3 * i + 0);
							d[3 * i + 1] += ca*bgrid[0].at<float>(3 * i + 1);
							d[3 * i + 2] += ca*bgrid[0].at<float>(3 * i + 2);
						}
					}
				}
				else if (typeid(S) == typeid(double))
				{
					for (int i = 0; i < imageSizeFull; i++)
					{
						int id = idx[jfull[i]];
						double ca = (double)a[jfull[i]];
						if (id + 1 == b)
						{
							d[3 * i + 0] += (S)((1.0 - ca)*bgrid[0].at<double>(3 * i + 0));
							d[3 * i + 1] += (S)((1.0 - ca)*bgrid[0].at<double>(3 * i + 1));
							d[3 * i + 2] += (S)((1.0 - ca)*bgrid[0].at<double>(3 * i + 2));
						}
						if (id == b)
						{
							d[3 * i + 0] += (S)(ca*bgrid[0].at<double>(3 * i + 0));
							d[3 * i + 1] += (S)(ca*bgrid[0].at<double>(3 * i + 1));
							d[3 * i + 2] += (S)(ca*bgrid[0].at<double>(3 * i + 2));
						}
					}
				}
			}
			dst.convertTo(dest, src.type());
		}
		else if (src.channels() == 1 && guide.channels() == 3)
		{
			for (int b = 0; b < num_bin; b++)
			{
				CV_DECL_ALIGNED(16) uchar bgr[3];
				for (int g = 0; g < num_bin; g++)
				{
					for (int r = 0; r < num_bin; r++)
					{
						S* su = sub_range[0].ptr<S>(0);//upper
						S* sd = normalize_sub_range[0].ptr<S>(0);//down

						bgr[0] = bin2num[b];
						bgr[1] = bin2num[g];
						bgr[2] = bin2num[r];
						splattingColor<T, S>(s, su, sd, j, bgr, imageSize, src.channels(), normType);

						blurring(sub_range[0], sub_range[0]);
						blurring(normalize_sub_range[0], normalize_sub_range[0]);

						if (downsampleSizeSplatting == 1)
						{
							divide(sub_range[0], normalize_sub_range[0], bgrid[0]);
						}
						else
						{
							divide(sub_range[0], normalize_sub_range[0], sub_range[0]);
							resize(sub_range[0], bgrid[0], src_.size(), 0, 0, upsampleMethod);
						}

						if (typeid(S) == typeid(float))
						{
							for (int i = 0; i < imageSizeFull; i++)
							{
								float inter = 1.f;
								int id = idx[jfull[3 * i + 0]];
								float ca = a[jfull[3 * i + 0]];

								if (id + 1 == b) inter *= (1.f - ca);
								else if (id == b) inter *= ca;
								else goto jump1;

								id = idx[jfull[3 * i + 1]];
								ca = a[jfull[3 * i + 1]];
								if (id + 1 == g) inter *= (1.f - ca);
								else if (id == g)inter *= ca;
								else goto jump1;

								id = idx[jfull[3 * i + 2]];
								ca = a[jfull[3 * i + 2]];
								if (id + 1 == r) inter *= (1.f - ca);
								else if (id == r) inter *= ca;
								else goto jump1;

								d[i] += inter*bgrid[0].at<S>(i);
							jump1:;
							}
						}
						else if (typeid(S) == typeid(double))
						{
							for (int i = 0; i < imageSizeFull; i++)
							{
								double inter = 1.0;
								int id = idx[jfull[3 * i + 0]];
								double ca = a[jfull[3 * i + 0]];

								if (id + 1 == b) inter *= (1.0 - ca);
								else if (id == b) inter *= ca;
								else goto jump2;

								id = idx[jfull[3 * i + 1]];
								ca = a[jfull[3 * i + 1]];
								if (id + 1 == g) inter *= (1.0 - ca);
								else if (id == g)inter *= ca;
								else goto jump2;

								id = idx[jfull[3 * i + 2]];
								ca = a[jfull[3 * i + 2]];
								if (id + 1 == r) inter *= (1.0 - ca);
								else if (id == r) inter *= ca;
								else goto jump2;

								d[i] += (S)inter*bgrid[0].at<S>(i);
							jump2:;
							}
						}
					}
				}
			}
			dst.convertTo(dest, src.type());
		}
		else if (src.channels() == 3 && guide.channels() == 3)
		{
			for (int b = 0; b < num_bin; b++)
			{
				CV_DECL_ALIGNED(16) uchar bgr[3];
				for (int g = 0; g < num_bin; g++)
				{
					for (int r = 0; r < num_bin; r++)
					{
						S* su = sub_range[0].ptr<S>(0);//upper
						S* sd = normalize_sub_range[0].ptr<S>(0);//down

						bgr[0] = bin2num[b];
						bgr[1] = bin2num[g];
						bgr[2] = bin2num[r];
						splattingColor<T, S>(s, su, sd, j, bgr, imageSize, src.channels(), normType);

						blurring(sub_range[0], sub_range[0]);
						blurring(normalize_sub_range[0], normalize_sub_range[0]);

						if (downsampleSizeSplatting == 1)
						{
							divide(sub_range[0], normalize_sub_range[0], bgrid[0]);
						}
						else
						{
							divide(sub_range[0], normalize_sub_range[0], sub_range[0]);
							resize(sub_range[0], bgrid[0], src_.size(), 0, 0, upsampleMethod);
						}

						if (typeid(S) == typeid(float))
						{
							for (int i = 0; i < imageSizeFull; i++)
							{
								float inter = 1.f;
								int id = idx[jfull[3 * i + 0]];
								float ca = a[jfull[3 * i + 0]];

								if (id + 1 == b) inter *= (1.f - ca);
								else if (id == b) inter *= ca;
								else goto jump3;

								id = idx[jfull[3 * i + 1]];
								ca = a[jfull[3 * i + 1]];
								if (id + 1 == g) inter *= (1.f - ca);
								else if (id == g)inter *= ca;
								else goto jump3;

								id = idx[jfull[3 * i + 2]];
								ca = a[jfull[3 * i + 2]];
								if (id + 1 == r) inter *= (1.f - ca);
								else if (id == r) inter *= ca;
								else goto jump3;

								d[3 * i + 0] += inter*bgrid[0].at<S>(3 * i + 0);
								d[3 * i + 1] += inter*bgrid[0].at<S>(3 * i + 1);
								d[3 * i + 2] += inter*bgrid[0].at<S>(3 * i + 2);
							jump3:
								;
							}
						}
						else if (typeid(S) == typeid(double))
						{
							for (int i = 0; i < imageSizeFull; i++)
							{
								double inter = 1.0;
								int id = idx[jfull[3 * i + 0]];
								double ca = (double)a[jfull[3 * i + 0]];

								if (id + 1 == b) inter *= (1.0 - ca);
								else if (id == b) inter *= ca;
								else goto jump4;

								id = idx[jfull[3 * i + 1]];
								ca = a[jfull[3 * i + 1]];
								if (id + 1 == g) inter *= (1.0 - ca);
								else if (id == g)inter *= ca;
								else goto jump4;

								id = idx[jfull[3 * i + 2]];
								ca = a[jfull[3 * i + 2]];
								if (id + 1 == r) inter *= (1.0 - ca);
								else if (id == r) inter *= ca;
								else goto jump4;

								d[3 * i + 0] += (S)inter*bgrid[0].at<S>(3 * i + 0);
								d[3 * i + 1] += (S)inter*bgrid[0].at<S>(3 * i + 1);
								d[3 * i + 2] += (S)inter*bgrid[0].at<S>(3 * i + 2);
							jump4:;
							}
						}
					}
				}
			}
			dst.convertTo(dest, src.type());
		}
	}

	template <typename T, typename S>
	void RealtimeO1BilateralFilter::body_(const Mat& src_, const Mat& guide_, Mat& dest)
	{
		if (bgrid.size() != num_bin || bgrid[0].size() != src_.size())
		{
			bgrid.clear();
			bgrid.resize(num_bin);
		}

		Mat src;
		Mat guide;
		if (downsampleSizeSplatting != 1)
		{
			resize(src_, src, Size(src_.size().width / downsampleSizeSplatting, src_.size().height / downsampleSizeSplatting), 0, 0, downsampleMethod);
			resize(guide_, guide, Size(src_.size().width / downsampleSizeSplatting, src_.size().height / downsampleSizeSplatting), 0, 0, downsampleMethod);
		}
		else
		{
			src = src_;
			guide = guide_;
		}

		num_bin = max(num_bin, 2);// for 0 and 255
		if (dest.empty() || dest.size() != src_.size() || dest.type() != src.type()) dest.create(src_.size(), src.type());

		const int imageSize = src.size().area();
		const int imageSizeFull = src_.size().area();

		setColorLUT(sigma_color, guide.channels());
		if (sub_range.size() != num_bin || sub_range[0].size().area() != imageSize || sub_range[0].depth() != bin_depth)
		{
			createBin(src.size(), num_bin, src.channels());
			disposeBin(num_bin);
		}

		const T* s = src.ptr<T>(0);
		const uchar* j = guide.ptr<uchar>(0);
		const uchar* jfull = guide_.ptr<uchar>(0);
		T* d = dest.ptr<T>(0);
		if (src.channels() == 1 && guide.channels() == 1)
		{
#ifdef USE_OPENMP
#pragma omp parallel for schedule (dynamic)
			for (int b = 0; b < num_bin; b++)
#else
			for (int b = 0; b < num_bin; b++)
#endif
			{
				S* su = sub_range[b].ptr<S>(0);//upper
				S* sd = normalize_sub_range[b].ptr<S>(0);//down

				uchar v = bin2num[b];

				splatting<T, S>(s, su, sd, j, v, imageSize, src.channels());

				blurring(sub_range[b], sub_range[b]);
				blurring(normalize_sub_range[b], normalize_sub_range[b]);
				if (downsampleSizeSplatting == 1)
				{
					divide(sub_range[b], normalize_sub_range[b], bgrid[b]);
				}
				else
				{
					divide(sub_range[b], normalize_sub_range[b], sub_range[b]);
					resize(sub_range[b], bgrid[b], src_.size(), 0, 0, upsampleMethod);
				}
			}
			for (int i = 0; i < imageSizeFull; i++)
			{
				int id = idx[jfull[i]];
				S ca = (S)a[jfull[i]];
				S ica = (S)(1.f - ca);
				d[i] = saturate_cast<T>(ca*bgrid[id].at<S>(i)+ica*bgrid[id + 1].at<S>(i));
			}
		}
		else if (src.channels() == 3 && guide.channels() == 1)
		{
#ifdef USE_OPENMP
#pragma omp parallel for schedule (dynamic)
			for (int b = 0; b < num_bin; b++)
#else
			for (int b = 0; b < num_bin; b++)
#endif
			{
				S* su = sub_range[b].ptr<S>(0);//upper
				S* sd = normalize_sub_range[b].ptr<S>(0);//down

				uchar v = bin2num[b];

				splatting<T, S>(s, su, sd, j, v, imageSize, src.channels());

				blurring(sub_range[b], sub_range[b]);
				blurring(normalize_sub_range[b], normalize_sub_range[b]);

				if (downsampleSizeSplatting == 1)
				{
					divide(sub_range[b], normalize_sub_range[b], bgrid[b]);
				}
				else
				{
					divide(sub_range[b], normalize_sub_range[b], sub_range[b]);
					resize(sub_range[b], bgrid[b], src_.size(), 0, 0, upsampleMethod);
				}
			}

			for (int i = 0; i < imageSizeFull; i++)
			{
				int id = idx[jfull[i]];
				S ca = (S)a[jfull[i]];
				S ica = (S)(1.f - ca);

				d[3 * i + 0] = saturate_cast<T>(ca*bgrid[id].at<S>(3 * i + 0) + ica*bgrid[id + 1].at<S>(3 * i + 0));
				d[3 * i + 1] = saturate_cast<T>(ca*bgrid[id].at<S>(3 * i + 1) + ica*bgrid[id + 1].at<S>(3 * i + 1));
				d[3 * i + 2] = saturate_cast<T>(ca*bgrid[id].at<S>(3 * i + 2) + ica*bgrid[id + 1].at<S>(3 * i + 2));
			}
		}
		else if (src.channels() == 1 && guide.channels() == 3)
		{
			vector<Mat> dst(num_bin);
			if (typeid(S) == typeid(double)) for (int b = 0; b < num_bin; b++) dst[b] = Mat::zeros(src_.size(), CV_64FC1);
			else for (int b = 0; b < num_bin; b++) dst[b] = Mat::zeros(src_.size(), CV_32FC1);
#ifdef USE_OPENMP
#pragma omp parallel for schedule (dynamic)
			for (int b = 0; b < num_bin; b++)
#else
			for (int b = 0; b < num_bin; b++)
#endif
			{
				S* dtemp = dst[b].ptr<S>(0);
				CV_DECL_ALIGNED(16) uchar bgr[3];
				for (int g = 0; g < num_bin; g++)
				{
					for (int r = 0; r < num_bin; r++)
					{
						S* su = sub_range[b].ptr<S>(0);//upper
						S* sd = normalize_sub_range[b].ptr<S>(0);//down

						bgr[0] = bin2num[b];
						bgr[1] = bin2num[g];
						bgr[2] = bin2num[r];
						splattingColor<T, S>(s, su, sd, j, bgr, imageSize, src.channels(), normType);

						blurring(sub_range[b], sub_range[b]);
						blurring(normalize_sub_range[b], normalize_sub_range[b]);
						if (downsampleSizeSplatting == 1)
						{
							divide(sub_range[b], normalize_sub_range[b], bgrid[b]);
						}
						else
						{
							divide(sub_range[b], normalize_sub_range[b], sub_range[b]);
							resize(sub_range[b], bgrid[b], src_.size(), 0, 0, upsampleMethod);
						}

						for (int i = 0; i < imageSizeFull; i++)
						{
							S inter = (S)1.0;
							int id = idx[jfull[3 * i + 0]];
							S ca = a[jfull[3 * i + 0]];

							if (id + 1 == b) inter *= ((S)1.0 - ca);
							else if (id == b) inter *= ca;
							else goto jump1;

							id = idx[jfull[3 * i + 1]];
							ca = a[jfull[3 * i + 1]];
							if (id + 1 == g) inter *= ((S)1.0 - ca);
							else if (id == g)inter *= ca;
							else goto jump1;

							id = idx[jfull[3 * i + 2]];
							ca = a[jfull[3 * i + 2]];
							if (id + 1 == r) inter *= ((S)1.0 - ca);
							else if (id == r) inter *= ca;
							else goto jump1;

							dtemp[i] += inter*bgrid[b].at<S>(i);
						jump1:;
						}
					}
				}
			}
			for (int b = 1; b < num_bin; b++)
			{
				dst[0] += dst[b];
			}
			dst[0].convertTo(dest, src.type());
		}
		else if (src.channels() == 3 && guide.channels() == 3)
		{
			vector<Mat> dst(num_bin);
			if (typeid(S) == typeid(double)) for (int b = 0; b < num_bin; b++) dst[b] = Mat::zeros(src_.size(), CV_64FC3);
			else for (int b = 0; b < num_bin; b++) dst[b] = Mat::zeros(src_.size(), CV_32FC3);
#ifdef USE_OPENMP
#pragma omp parallel for schedule (dynamic)
			for (int b = 0; b < num_bin; b++)
#else
			for (int b = 0; b < num_bin; b++)
#endif
			{
				for (int g = 0; g < num_bin; g++)
				{
					for (int r = 0; r < num_bin; r++)
					{
						S* su = sub_range[b].ptr<S>(0);//upper
						S* sd = normalize_sub_range[b].ptr<S>(0);//down
						CV_DECL_ALIGNED(16) uchar bgr[3];
						bgr[0] = bin2num[b];
						bgr[1] = bin2num[g];
						bgr[2] = bin2num[r];
						splattingColor<T, S>(s, su, sd, j, bgr, imageSize, src.channels(), normType);

						blurring(sub_range[b], sub_range[b]);
						blurring(normalize_sub_range[b], normalize_sub_range[b]);
						if (downsampleSizeSplatting == 1)
						{
							divide(sub_range[b], normalize_sub_range[b], bgrid[b]);
						}
						else
						{
							divide(sub_range[b], normalize_sub_range[b], sub_range[b]);
							resize(sub_range[b], bgrid[b], src_.size(), 0, 0, upsampleMethod);
						}

						S* bgridPtr = bgrid[b].ptr<S>(0);
						S* dtemp = dst[b].ptr<S>(0);
						uchar* ref = (uchar*)jfull;
						for (int i = 0; i < imageSizeFull; i++)
						{
							S inter = (S)1.0;
							int id = idx[ref[0]];
							S ca = a[ref[0]];

							if (id + 1 == b) inter *= ((S)1.0 - ca);
							else if (id == b) inter *= ca;
							else goto jump2;

							id = idx[ref[1]];
							ca = a[ref[1]];
							if (id + 1 == g) inter *= ((S)1.0 - ca);
							else if (id == g)inter *= ca;
							else goto jump2;

							id = idx[ref[2]];
							ca = a[ref[2]];
							if (id + 1 == r) inter *= ((S)1.0 - ca);
							else if (id == r) inter *= ca;
							else goto jump2;

							dtemp[0] += inter*bgridPtr[0];
							dtemp[1] += inter*bgridPtr[1];
							dtemp[2] += inter*bgridPtr[2];
						jump2:
							dtemp += 3;
							bgridPtr += 3;
							ref += 3;
						}
					}
				}
			}

			for (int b = 1; b < num_bin; b++)
			{
				dst[0] += dst[b];
			}

			if (src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_16S || src.depth() == CV_32S)
				dst[0].convertTo(dest, src.type(), 1.0, 0.5);
			else
				dst[0].convertTo(dest, src.type());
		}
	}

	void RealtimeO1BilateralFilter::gaussFIR(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, int r, float sigma_color_, float sigma_space_, int num_bin_)
	{
		radius = r;
		sigma_color = sigma_color_;
		sigma_space = sigma_space_;
		num_bin = num_bin_;
		filter_type = FIR_SEPARABLE;

		body(src, joint, dest, isSaveMemory);
	}

	void RealtimeO1BilateralFilter::gaussFIR(cv::InputArray src, cv::OutputArray dest, int r, float sigma_color, float sigma_space, int num_bin)
	{
		Mat joint;
		if (src.depth() != CV_8U)src.getMat().convertTo(joint, CV_8U);
		else joint = src.getMat();
		gaussFIR(src, joint, dest, r, sigma_color, sigma_space, num_bin);
	}

	void RealtimeO1BilateralFilter::gaussIIR(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, float sigma_color_, float sigma_space_, int num_bin_, int method, int K)
	{
		filterK = K;
		sigma_color = sigma_color_;
		sigma_space = sigma_space_;
		num_bin = num_bin_;
		filter_type = method;

		body(src, joint, dest, isSaveMemory);
	}

	void RealtimeO1BilateralFilter::gaussIIR(cv::InputArray src, cv::OutputArray dest, float sigma_color, float sigma_space, int num_bin, int method, int K)
	{
		Mat joint;
		if (src.depth() != CV_8U)src.getMat().convertTo(joint, CV_8U);
		else joint = src.getMat();
		gaussIIR(src, joint, dest, sigma_color, sigma_space, num_bin, method, K);
	}

}