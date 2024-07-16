#include <opencp.hpp>
#include "patchPCA.hpp"

using namespace cv;
using namespace std;

/*namespace util
{
	void eigen(cv::Mat& src, cv::Mat& eval, cv::Mat& evec);
	void eigenconvtest(cv::Mat& src, cv::Mat& dest);
}*/

//#define SCHEDULE schedule(dynamic)
#define SCHEDULE schedule(static)
namespace cp
{
#pragma region util
class BlockCovMatrixOperator
{
	int D = 0;
	int r;
	int channels = 0;
	Mat cov;
public:
	BlockCovMatrixOperator(Mat& src, const int channels)
	{
		cov = src.clone();
		this->channels = channels;
		this->D = cov.cols / channels;
		this->r = D / 2;
	}

	void getCovMatrix(Mat& dest)
	{
		cov.copyTo(dest);
	}

	vector<double> getBlockElementPatchCoordinate(Point i, Point j)
	{
		vector<double> ret;
		const int k = D * i.y + i.x;
		const int l = D * j.y + j.x;
		for (int cy = 0; cy < channels; cy++)
		{
			for (int cx = 0; cx < channels; cx++)
			{
				ret.push_back(cov.at<double>(channels * l + cy, channels * k + cx));
			}
		}
	}

	vector<double> setBlockElementPatchCoordinate(Point i, Point j, vector<double>& val, bool isBlockTranspose = false)
	{
		const int k = D * i.y + i.x;
		const int l = D * j.y + j.x;
		if (isBlockTranspose)
		{
			for (int cy = 0, idx = 0; cy < channels; cy++)
			{
				for (int cx = 0; cx < channels; cx++)
				{
					cov.at<double>(channels * l + cx, channels * k + cy) = val[idx++];
				}
			}
		}
		else
		{
			for (int cy = 0, idx = 0; cy < channels; cy++)
			{
				for (int cx = 0; cx < channels; cx++)
				{
					cov.at<double>(channels * l + cy, channels * k + cx) = val[idx++];
				}
			}
		}
	}
};

class MatrixDraw
{
	const int step = 50;
	const int rows;
	const int cols;
	Mat grid;
public:
	MatrixDraw(int rows, int cols) : rows(rows), cols(cols)
	{
		grid = Mat::zeros(Size(step * cols, step * rows), CV_8UC3);
		grid.setTo(255);

		for (int j = 0; j < rows; j++)
		{
			for (int i = 0; i < cols; i++)
			{
				rectangle(grid, Rect(i * step, j * step, step, step), COLOR_BLACK, 2);
			}
		}
	}

	void setGrid(Mat& dest)
	{
		grid.copyTo(dest);
	}

	void set(Mat& dest, const int row, const int col)
	{
		if (dest.empty())setGrid(dest);
		rectangle(dest, Rect(col * step, row * step, step, step), COLOR_GRAY150, cv::FILLED);
	}
};
//start+(start+1)+,...,+end
inline int sum_from(const int end, int start = 0)
{
	int ret = 0;
	for (int i = start; i <= end; i++) ret += i;
	return ret;
}

void testCovMat(string wname, Mat& src, Mat& ref, vector<vector<Point>>& covset, const bool isWait, int channels)
{
	const int step = 50;
	const int rows = src.rows;
	const int cols = src.cols;
	Mat show1 = Mat::zeros(Size(step * cols, step * rows), CV_8UC3);
	Mat show2 = Mat::zeros(Size(step * cols, step * rows), CV_8UC3);
	namedWindow(wname);
	static int dir = 0; createTrackbar("dir", wname, &dir, max((int)covset.size() - 1, 1)); setTrackbarMin("dir", wname, -1);
	static int sw = 0; createTrackbar("sw", wname, &sw, 2);
	static int boost = 0; createTrackbar("boost", wname, &boost, 10);

	int key = 0;
	while (key != 'q')
	{
		const double amp = pow(10.0, boost);
		show1.setTo(255);
		show2.setTo(255);
		if (channels == 1)
		{
			for (int j = 0; j < rows; j++)
			{
				for (int i = 0; i < cols; i++)
				{
					rectangle(show1, Rect(i * step, j * step, step, step), COLOR_GRAY200, 5);
					rectangle(show2, Rect(i * step, j * step, step, step), COLOR_GRAY200, 5);

					if (dir < 0)
					{
						if (i == j)rectangle(show2, Rect(i * step, j * step, step, step), COLOR_GRAY150, cv::FILLED);
					}
					else
					{
						for (int n = 0; n < covset[dir].size(); n++)
						{
							if (covset[dir][n].x == i && covset[dir][n].y == j)
								rectangle(show2, Rect(i * step, j * step, step, step), COLOR_GRAY150, cv::FILLED);
						}
					}

					const int x = i;
					const int y = j;
					rectangle(show1, Rect(x * step, y * step, step, step), COLOR_GRAY200, cv::FILLED);
					rectangle(show2, Rect(x * step, y * step, step, step), COLOR_BLACK);
					Mat v = src;
					if (sw == 1)v = ref;
					else if (sw == 2)v = Mat(ref - src).clone();
					cv::addText(show1, cv::format("%7.2f", v.at<double>(y, x)), Point(int(step * x + 0.1 * step), int(step * y + step * 0.65)), "Times", 9);
					cv::addText(show2, cv::format("%7.2f", v.at<double>(y, x)), Point(int(step * x + 0.1 * step), int(step * y + step * 0.65)), "Times", 9);
				}
			}
		}
		else if (channels == 3)
		{
			for (int j = 0; j < rows; j += 3)
			{
				for (int i = 0; i < cols; i += 3)
				{
					rectangle(show1, Rect(i * step, j * step, step * 3, step * 3), COLOR_GRAY200, 5);
					rectangle(show2, Rect(i * step, j * step, step * 3, step * 3), COLOR_GRAY200, 5);
					if (dir < 0)
					{
						if (i == j)rectangle(show2, Rect(i * step, j * step, step * 3, step * 3), COLOR_GRAY150, cv::FILLED);
					}
					else
					{
						for (int n = 0; n < covset[dir].size(); n++)
						{
							if (covset[dir][n].x * 3 == i && covset[dir][n].y * 3 == j)
								rectangle(show2, Rect(i * step, j * step, step * 3, step * 3), COLOR_GRAY150, cv::FILLED);
						}
					}

					for (int l = 0; l < 3; l++)
					{
						for (int k = 0; k < 3; k++)
						{
							const int x = i + k;
							const int y = j + l;

							int b = (k == 0 || l == 0) ? 255 : 80;
							int g = (k == 1 || l == 1) ? 255 : 80;
							int r = (k == 2 || l == 2) ? 255 : 80;
							Scalar color = Scalar(b, g, r, 0);

							rectangle(show1, Rect(x * step, y * step, step, step), color, cv::FILLED);
							rectangle(show2, Rect(x * step, y * step, step, step), COLOR_BLACK);
							Mat v = src;
							if (sw == 1)v = ref;
							else if (sw == 2)v = Mat(abs(Mat(ref - src))).clone();
							if (sw == 2)
							{
								cv::addText(show1, format("%7.2f", amp * v.at<double>(y, x)), Point(int(step * x + 0.1 * step), int(step * y + step * 0.65)), "Times", 9);
								cv::addText(show2, format("%7.2f", amp * v.at<double>(y, x)), Point(int(step * x + 0.1 * step), int(step * y + step * 0.65)), "Times", 9);
							}
							else
							{
								cv::addText(show1, format("%7.2f", v.at<double>(y, x)), Point(int(step * x + 0.1 * step), int(step * y + step * 0.65)), "Times", 9);
								cv::addText(show2, format("%7.2f", v.at<double>(y, x)), Point(int(step * x + 0.1 * step), int(step * y + step * 0.65)), "Times", 9);
							}
						}
					}
				}
			}
		}
		double a = 0.5;
		addWeighted(show1, a, show2, 1.0 - a, 0.0, show1);
		imshow(wname, show1);
		key = waitKey(1);
		if (!isWait)break;
	}
}

double GrassmannDistance(const Mat& src, const Mat& ref)
{
	double dst = 0.0, acoss = 0.0;
	Mat AtB, S, U, Vt;
	AtB = src * ref.t();
	SVD::compute(AtB, S, U, Vt, 2);
	switch (src.type())
	{
	case CV_32F:
		for (int i = 0; i < src.rows; i++)
		{
			acoss = acos(max(min((double)S.at<float>(i, 0), 1.0), -1.0));
			dst += acoss * acoss;
		}
		break;

	case CV_64F:
		for (int i = 0; i < src.rows; i++)
		{
			acoss = acos(max(min(S.at<double>(i, 0), 1.0), -1.0));
			dst += acoss * acoss;
		}
		break;
	}
	return dst;
}

double getPSNRVector(vector<Mat>& src, vector<Mat>& ref)
{
	CV_Assert(src.size() == ref.size());
	CV_Assert(src[0].size() == ref[0].size());

	const int imsize = (int)src[0].size().area();
	const int total_size = (int)(imsize * src.size());
	double ret = 0.0;
	for (int c = 0; c < src.size(); c++)
	{
		float* s = src[c].ptr<float>();
		float* r = ref[c].ptr<float>();
		__m256 error = _mm256_setzero_ps();
		for (int i = 0; i < imsize; i += 8)
		{
			const __m256 sub = _mm256_sub_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(r + i));
			/*if (i == 0 || i == 8)
			{
				cout << c << ":"; print_m256(_mm256_loadu_ps(s + i));
				cout << c << ":"; print_m256(_mm256_loadu_ps(r + i));
			}*/
			error = _mm256_fmadd_ps(sub, sub, error);
		}
		ret += _mm256_reduceadd_pspd(error);
	}
	const double mse = ret / total_size;
	return 10.0 * log10(255.0 * 255.0 / mse);
}
#pragma endregion

#pragma region flags

inline bool is32F(const DRIM2COLType method)
{
	bool ret = false;
	switch (method)
	{
	case DRIM2COLType::FULL_SUB_FULL_32F:
	case DRIM2COLType::FULL_SUB_HALF_32F:
	case DRIM2COLType::FULL_SUB_REP_32F:

	case DRIM2COLType::MEAN_SUB_HALF_32F:
	case DRIM2COLType::MEAN_SUB_REP_32F:
	case DRIM2COLType::MEAN_SUB_CONV_32F:
	case DRIM2COLType::MEAN_SUB_CONVF_32F:
	case DRIM2COLType::MEAN_SUB_FFT_32F:
	case DRIM2COLType::MEAN_SUB_FFTF_32F:

	case DRIM2COLType::NO_SUB_HALF_32F:
	case DRIM2COLType::NO_SUB_REP_32F:
	case DRIM2COLType::NO_SUB_CONV_32F:
	case DRIM2COLType::NO_SUB_CONVF_32F:
	case DRIM2COLType::NO_SUB_FFT_32F:
	case DRIM2COLType::NO_SUB_FFTF_32F:

	case DRIM2COLType::CONST_SUB_HALF_32F:
	case DRIM2COLType::CONST_SUB_REP_32F:
	case DRIM2COLType::CONST_SUB_CONV_32F:
	case DRIM2COLType::CONST_SUB_CONVF_32F:
	case DRIM2COLType::CONST_SUB_FFT_32F:
	case DRIM2COLType::CONST_SUB_FFTF_32F:
		ret = true; break;
	default:
		break;
	}
	return ret;
}

inline bool isSepCov(DRIM2COLType type, bool isSVD)
{
	bool ret = false;
	if (isSVD)
	{
		if (
			type == DRIM2COLType::MEAN_SUB_SEPSVD ||
			type == DRIM2COLType::NO_SUB_SEPSVD ||
			type == DRIM2COLType::CONST_SUB_SEPSVD)
		{
			ret = true;
		}
	}
	else
	{
		if (
			type == DRIM2COLType::MEAN_SUB_SEPCOVX ||
			type == DRIM2COLType::MEAN_SUB_SEPCOVY ||
			type == DRIM2COLType::NO_SUB_SEPCOVX ||
			type == DRIM2COLType::NO_SUB_SEPCOVY ||
			type == DRIM2COLType::CONST_SUB_SEPCOVX ||
			type == DRIM2COLType::CONST_SUB_SEPCOVY ||
			type == DRIM2COLType::MEAN_SUB_SEPCOVXXt ||
			type == DRIM2COLType::NO_SUB_SEPCOVXXt ||
			type == DRIM2COLType::CONST_SUB_SEPCOVXXt
			)
		{
			ret = true;
		}
	}
	return ret;
}

string getDRIM2COLName(const DRIM2COLType method)
{
	string ret = "";
	switch (method)
	{
	case DRIM2COLType::FULL_SUB_FULL_32F:			ret = "FULL_SUB_FULL_32F"; break;
	case DRIM2COLType::FULL_SUB_FULL_64F:			ret = "FULL_SUB_FULL_64F"; break;
	case DRIM2COLType::FULL_SUB_HALF_32F:			ret = "FULL_SUB_HALF_32F"; break;
	case DRIM2COLType::FULL_SUB_HALF_64F:			ret = "FULL_SUB_HALF_64F"; break;
	case DRIM2COLType::FULL_SUB_REP_32F:			ret = "FULL_SUB_REP_32F"; break;
	case DRIM2COLType::FULL_SUB_REP_64F:			ret = "FULL_SUB_REP_64F"; break;

	case DRIM2COLType::MEAN_SUB_HALF_32F:			ret = "MEAN_SUB_HALF_32F"; break;
	case DRIM2COLType::NO_SUB_HALF_32F:				ret = "NO_SUB_HALF_32F"; break;
	case DRIM2COLType::CONST_SUB_HALF_32F:			ret = "CONST_SUB_HALF_32F"; break;
	case DRIM2COLType::MEAN_SUB_HALF_64F:			ret = "MEAN_SUB_HALF_64F"; break;
	case DRIM2COLType::NO_SUB_HALF_64F:				ret = "NO_SUB_HALF_64F"; break;
	case DRIM2COLType::CONST_SUB_HALF_64F:			ret = "CONST_SUB_HALF_64F"; break;

	case DRIM2COLType::MEAN_SUB_REP_32F:			ret = "MEAN_SUB_REP_32F"; break;
	case DRIM2COLType::NO_SUB_REP_32F:				ret = "NO_SUB_REP_32F"; break;
	case DRIM2COLType::CONST_SUB_REP_32F:			ret = "CONST_SUB_REP_32F"; break;
	case DRIM2COLType::MEAN_SUB_REP_64F:			ret = "MEAN_SUB_REP_64F"; break;
	case DRIM2COLType::NO_SUB_REP_64F:				ret = "NO_SUB_REP_64F"; break;
	case DRIM2COLType::CONST_SUB_REP_64F:			ret = "CONST_SUB_REP_64F"; break;

	case DRIM2COLType::MEAN_SUB_CONV_32F:			ret = "MEAN_SUB_CONV_32F"; break;
	case DRIM2COLType::NO_SUB_CONV_32F:				ret = "NO_SUB_CONV_32F"; break;
	case DRIM2COLType::CONST_SUB_CONV_32F:			ret = "CONST_SUB_CONV_32F"; break;
	case DRIM2COLType::MEAN_SUB_CONV_64F:			ret = "MEAN_SUB_CONV_64F"; break;
	case DRIM2COLType::NO_SUB_CONV_64F:				ret = "NO_SUB_CONV_64F"; break;
	case DRIM2COLType::CONST_SUB_CONV_64F:			ret = "CONST_SUB_CONV_64F"; break;

	case DRIM2COLType::MEAN_SUB_CONVF_32F:			ret = "MEAN_SUB_CONVF_32F"; break;
	case DRIM2COLType::NO_SUB_CONVF_32F:			ret = "NO_SUB_CONVF_32F"; break;
	case DRIM2COLType::CONST_SUB_CONVF_32F:			ret = "CONST_SUB_CONVF_32F"; break;
	case DRIM2COLType::MEAN_SUB_CONVF_64F:			ret = "MEAN_SUB_CONVF_64F"; break;
	case DRIM2COLType::NO_SUB_CONVF_64F:			ret = "NO_SUB_CONVF_64F"; break;
	case DRIM2COLType::CONST_SUB_CONVF_64F:			ret = "CONST_SUB_CONVF_64F"; break;

	case DRIM2COLType::MEAN_SUB_FFT_32F:			ret = "MEAN_SUB_FFT_32F"; break;
	case DRIM2COLType::NO_SUB_FFT_32F:				ret = "NO_SUB_FFT_32F"; break;
	case DRIM2COLType::CONST_SUB_FFT_32F:			ret = "CONST_SUB_FFT_32F"; break;
	case DRIM2COLType::MEAN_SUB_FFT_64F:			ret = "MEAN_SUB_FFT_64F"; break;
	case DRIM2COLType::NO_SUB_FFT_64F:				ret = "NO_SUB_FFT_64F"; break;
	case DRIM2COLType::CONST_SUB_FFT_64F:			ret = "CONST_SUB_FFT_64F"; break;

	case DRIM2COLType::MEAN_SUB_FFTF_32F:			ret = "MEAN_SUB_FFTF_32F"; break;
	case DRIM2COLType::NO_SUB_FFTF_32F:				ret = "NO_SUB_FFTF_32F"; break;
	case DRIM2COLType::CONST_SUB_FFTF_32F:			ret = "CONST_SUB_FFTF_32F"; break;
	case DRIM2COLType::MEAN_SUB_FFTF_64F:			ret = "MEAN_SUB_FFTF_64F"; break;
	case DRIM2COLType::NO_SUB_FFTF_64F:				ret = "NO_SUB_FFTF_64F"; break;
	case DRIM2COLType::CONST_SUB_FFTF_64F:			ret = "CONST_SUB_FFTF_64F"; break;

	case DRIM2COLType::TEST:						ret = "TEST"; break;

	case DRIM2COLType::OPENCV_PCA:					ret = "OPENCV_PCA"; break;
	case DRIM2COLType::OPENCV_COV:					ret = "OPENCV_COV"; break;

	case DRIM2COLType::MEAN_SUB_SEPSVD:            ret = "MEAN_SUB_SEPSVD";		break;
	case DRIM2COLType::NO_SUB_SEPSVD:			   ret = "NO_SUB_SEPSVD";		break;
	case DRIM2COLType::CONST_SUB_SEPSVD:		   ret = "CONST_SUB_SEPSVD";	break;
	case DRIM2COLType::MEAN_SUB_SEPCOVX:		   ret = "MEAN_SUB_SEPCOVX";	break;
	case DRIM2COLType::NO_SUB_SEPCOVX:			   ret = "NO_SUB_SEPCOVX";		break;
	case DRIM2COLType::CONST_SUB_SEPCOVX:		   ret = "CONST_SUB_SEPCOVX";	break;
	case DRIM2COLType::MEAN_SUB_SEPCOVY:		   ret = "MEAN_SUB_SEPCOVY";	break;
	case DRIM2COLType::NO_SUB_SEPCOVY:			   ret = "NO_SUB_SEPCOVY";		break;
	case DRIM2COLType::CONST_SUB_SEPCOVY:		   ret = "CONST_SUB_SEPCOVY";	break;
	case DRIM2COLType::MEAN_SUB_SEPCOVXXt:		   ret = "MEAN_SUB_SEPCOVXXt";	break;
	case DRIM2COLType::NO_SUB_SEPCOVXXt:		   ret = "NO_SUB_SEPCOVXXt";	break;
	case DRIM2COLType::CONST_SUB_SEPCOVXXt:		   ret = "CONST_SUB_SEPCOVXXt"; break;

	default:
		break;
	}
	return ret;
}

inline DRIM2COLElementSkipElement getElementSkipType(DRIM2COLType method)
{
	DRIM2COLElementSkipElement ret = DRIM2COLElementSkipElement::FULL;
	switch (method)
	{
	case DRIM2COLType::FULL_SUB_FULL_32F:
	case DRIM2COLType::FULL_SUB_FULL_64F:
		ret = DRIM2COLElementSkipElement::FULL; break;

	case DRIM2COLType::FULL_SUB_HALF_32F:
	case DRIM2COLType::FULL_SUB_HALF_64F:
	case DRIM2COLType::MEAN_SUB_HALF_32F:
	case DRIM2COLType::NO_SUB_HALF_32F:
	case DRIM2COLType::CONST_SUB_HALF_32F:
	case DRIM2COLType::MEAN_SUB_HALF_64F:
	case DRIM2COLType::NO_SUB_HALF_64F:
		ret = DRIM2COLElementSkipElement::HALF; break;

	case DRIM2COLType::FULL_SUB_REP_32F:
	case DRIM2COLType::FULL_SUB_REP_64F:
	case DRIM2COLType::MEAN_SUB_REP_32F:
	case DRIM2COLType::NO_SUB_REP_32F:
	case DRIM2COLType::CONST_SUB_REP_32F:
	case DRIM2COLType::MEAN_SUB_REP_64F:
	case DRIM2COLType::NO_SUB_REP_64F:
	case DRIM2COLType::CONST_SUB_REP_64F:
		ret = DRIM2COLElementSkipElement::REP; break;

	case DRIM2COLType::MEAN_SUB_CONV_32F:
	case DRIM2COLType::NO_SUB_CONV_32F:
	case DRIM2COLType::CONST_SUB_CONV_32F:
	case DRIM2COLType::MEAN_SUB_CONV_64F:
	case DRIM2COLType::NO_SUB_CONV_64F:
	case DRIM2COLType::CONST_SUB_CONV_64F:
		ret = DRIM2COLElementSkipElement::CONV; break;

	case DRIM2COLType::MEAN_SUB_CONVF_32F:
	case DRIM2COLType::NO_SUB_CONVF_32F:
	case DRIM2COLType::CONST_SUB_CONVF_32F:
	case DRIM2COLType::MEAN_SUB_CONVF_64F:
	case DRIM2COLType::NO_SUB_CONVF_64F:
	case DRIM2COLType::CONST_SUB_CONVF_64F:
		ret = DRIM2COLElementSkipElement::CONVF; break;

	case DRIM2COLType::MEAN_SUB_FFT_32F:
	case DRIM2COLType::NO_SUB_FFT_32F:
	case DRIM2COLType::CONST_SUB_FFT_32F:
	case DRIM2COLType::MEAN_SUB_FFT_64F:
	case DRIM2COLType::NO_SUB_FFT_64F:
	case DRIM2COLType::CONST_SUB_FFT_64F:
		ret = DRIM2COLElementSkipElement::FFT; break;

	case DRIM2COLType::MEAN_SUB_FFTF_32F:
	case DRIM2COLType::NO_SUB_FFTF_32F:
	case DRIM2COLType::CONST_SUB_FFTF_32F:
	case DRIM2COLType::MEAN_SUB_FFTF_64F:
	case DRIM2COLType::NO_SUB_FFTF_64F:
	case DRIM2COLType::CONST_SUB_FFTF_64F:
		ret = DRIM2COLElementSkipElement::FFTF; break;
	}
	return ret;
}

string getDRIM2COLElementSkipTypeName(DRIM2COLElementSkipElement method)
{
	string ret = "";
	switch (method)
	{
	case DRIM2COLElementSkipElement::FULL:	ret = "FULL"; break;
	case DRIM2COLElementSkipElement::HALF:	ret = "HALF"; break;
	case DRIM2COLElementSkipElement::REP:	ret = "REP"; break;
	case DRIM2COLElementSkipElement::CONV:	ret = "CONV"; break;
	case DRIM2COLElementSkipElement::CONVF:	ret = "CONVF"; break;
	case DRIM2COLElementSkipElement::FFT:	ret = "FFT"; break;
	default: break;
	}
	return ret;
}

CalcPatchCovarMatrix::CenterMethod CalcPatchCovarMatrix::getCenterMethod(const DRIM2COLType method)
{
	CalcPatchCovarMatrix::CenterMethod cm = CalcPatchCovarMatrix::CenterMethod::FULL;
	switch (method)
	{
	case DRIM2COLType::MEAN_SUB_HALF_32F:
	case DRIM2COLType::MEAN_SUB_HALF_64F:
	case DRIM2COLType::MEAN_SUB_REP_32F:
	case DRIM2COLType::MEAN_SUB_REP_64F:
	case DRIM2COLType::MEAN_SUB_CONV_32F:
	case DRIM2COLType::MEAN_SUB_CONV_64F:
	case DRIM2COLType::MEAN_SUB_CONVF_32F:
	case DRIM2COLType::MEAN_SUB_CONVF_64F:
	case DRIM2COLType::MEAN_SUB_FFT_32F:
	case DRIM2COLType::MEAN_SUB_FFT_64F:
	case DRIM2COLType::MEAN_SUB_FFTF_32F:
	case DRIM2COLType::MEAN_SUB_FFTF_64F:
		cm = CalcPatchCovarMatrix::CenterMethod::MEAN; break;

	case DRIM2COLType::CONST_SUB_HALF_32F:
	case DRIM2COLType::CONST_SUB_HALF_64F:
	case DRIM2COLType::CONST_SUB_REP_32F:
	case DRIM2COLType::CONST_SUB_REP_64F:
	case DRIM2COLType::CONST_SUB_CONV_32F:
	case DRIM2COLType::CONST_SUB_CONV_64F:
	case DRIM2COLType::CONST_SUB_CONVF_32F:
	case DRIM2COLType::CONST_SUB_CONVF_64F:
	case DRIM2COLType::CONST_SUB_FFT_32F:
	case DRIM2COLType::CONST_SUB_FFT_64F:
	case DRIM2COLType::CONST_SUB_FFTF_32F:
	case DRIM2COLType::CONST_SUB_FFTF_64F:
		cm = CalcPatchCovarMatrix::CenterMethod::CONST_; break;

	case DRIM2COLType::NO_SUB_HALF_32F:
	case DRIM2COLType::NO_SUB_HALF_64F:
	case DRIM2COLType::NO_SUB_REP_32F:
	case DRIM2COLType::NO_SUB_REP_64F:
	case DRIM2COLType::NO_SUB_CONV_32F:
	case DRIM2COLType::NO_SUB_CONV_64F:
	case DRIM2COLType::NO_SUB_CONVF_32F:
	case DRIM2COLType::NO_SUB_CONVF_64F:
	case DRIM2COLType::NO_SUB_FFT_32F:
	case DRIM2COLType::NO_SUB_FFT_64F:
	case DRIM2COLType::NO_SUB_FFTF_32F:
	case DRIM2COLType::NO_SUB_FFTF_64F:
		cm = CalcPatchCovarMatrix::CenterMethod::NO; break;

	case DRIM2COLType::FULL_SUB_FULL_32F:
	case DRIM2COLType::FULL_SUB_FULL_64F:
	case DRIM2COLType::FULL_SUB_HALF_32F:
	case DRIM2COLType::FULL_SUB_HALF_64F:
	case DRIM2COLType::FULL_SUB_REP_32F:
	case DRIM2COLType::FULL_SUB_REP_64F:
		cm = CalcPatchCovarMatrix::CenterMethod::FULL; break;

	default:
		break;
	}
	return cm;
}
#pragma endregion

#pragma region public

int CalcPatchCovarMatrix::getNumElements(int patch_rad, int color_channel, const DRIM2COLType method)
{
	DRIM2COLElementSkipElement type = getElementSkipType(method);
	int ret = 0;
	const int D = 2 * patch_rad + 1;
	const int DD = D * D;
	const int dim = DD * color_channel;
	const int CC = color_channel * color_channel;
	switch (type)
	{
	case DRIM2COLElementSkipElement::FULL:ret = dim * dim;
		break;
	case DRIM2COLElementSkipElement::HALF:ret = sum_from(DD) * CC;
		break;
	case DRIM2COLElementSkipElement::REP:ret = (((2 * D - 1) * (2 * D - 1)) / 2) * CC + sum_from(color_channel);
		break;
	case DRIM2COLElementSkipElement::CONV:ret = DD * CC;
		break;
	case DRIM2COLElementSkipElement::CONVF:ret = ((2 * D - 1) * (2 * D - 1)) * CC;
		break;
	default:
		break;
	}
	return ret;
}

void CalcPatchCovarMatrix::setBorder(const int border)
{
	this->border = border;
}

void CalcPatchCovarMatrix::setConstSub(const int const_sub)
{
	this->const_sub = const_sub;
}

void CalcPatchCovarMatrix::computeCov(const vector<Mat>& src, const int patch_rad, Mat& cov, const DRIM2COLType method, const int skip, const bool isParallel)
{
	if (isSepCov(method, true) || isSepCov(method, false))
	{
		cout << "call computeCovSep for " << getDRIM2COLName(method) << endl;
		return;
	}
	this->patch_rad = patch_rad;
	D = 2 * patch_rad + 1;
	color_channels = (int)src.size();
	dim = color_channels * D * D;

	cov.create(dim, dim, CV_64F);
	cov.setTo(0);
	//cout << method << endl;
	//cp::Timer t;

	if (method == DRIM2COLType::TEST)
	{
		simdOMPCovFullCenterHalfElementTEST32F(src, cov, border);
		return;
	}

	if (method == DRIM2COLType::OPENCV_PCA || method == DRIM2COLType::OPENCV_COV)
	{
		Mat highDimGuide(src[0].size(), CV_MAKE_TYPE(CV_32F, dim));
		IM2COL(src, highDimGuide, patch_rad, border);
		Mat x = highDimGuide.reshape(1, src[0].size().area());
		Mat mean = Mat::zeros(dim, 1, CV_64F);
		cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
	}
	else
	{
		if (skip == 1)
		{
			CenterMethod centerMethod = getCenterMethod(method);

			if (!isParallel) omp_set_num_threads(1);

			if (is32F(method))
			{
				const DRIM2COLElementSkipElement type = getElementSkipType(method);
				//cout << getDRIM2COLElementSkipTypeName(type)<<endl;
				if (type == DRIM2COLElementSkipElement::FULL)
				{
					//cout << "FULL-FULL" << endl;
					simdOMPCovFullCenterFullElement32F(src, cov, border);
				}
				else if (type == DRIM2COLElementSkipElement::HALF)
				{
					if (centerMethod == CenterMethod::FULL)
					{
						//cout << "simdOMPCovFullSubHalfElement32F" << endl;
						simdOMPCovFullCenterHalfElement32F(src, cov, border);
					}
					else
					{
						if (color_channels == 1 && patch_rad == 1)      simdOMPCov_RepCenterHalfElement32F<1, 1>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 2 && patch_rad == 1) simdOMPCov_RepCenterHalfElement32F<2, 1>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 3 && patch_rad == 1) simdOMPCov_RepCenterHalfElement32F<3, 1>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 4 && patch_rad == 1) simdOMPCov_RepCenterHalfElement32F<4, 1>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 6 && patch_rad == 1) simdOMPCov_RepCenterHalfElement32F<6, 1>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 1 && patch_rad == 2) simdOMPCov_RepCenterHalfElement32F<1, 2>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 2 && patch_rad == 2) simdOMPCov_RepCenterHalfElement32F<2, 2>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 3 && patch_rad == 2) simdOMPCov_RepCenterHalfElement32F<3, 2>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 4 && patch_rad == 2) simdOMPCov_RepCenterHalfElement32F<4, 2>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 6 && patch_rad == 2) simdOMPCov_RepCenterHalfElement32F<6, 2>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 1 && patch_rad == 3) simdOMPCov_RepCenterHalfElement32F<1, 3>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 2 && patch_rad == 3) simdOMPCov_RepCenterHalfElement32F<2, 3>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 3 && patch_rad == 3) simdOMPCov_RepCenterHalfElement32F<3, 3>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 4 && patch_rad == 3) simdOMPCov_RepCenterHalfElement32F<4, 3>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 6 && patch_rad == 3) simdOMPCov_RepCenterHalfElement32F<6, 3>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 1 && patch_rad == 4) simdOMPCov_RepCenterHalfElement32F<1, 4>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 2 && patch_rad == 4) simdOMPCov_RepCenterHalfElement32F<2, 4>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 3 && patch_rad == 4) simdOMPCov_RepCenterHalfElement32F<3, 4>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 4 && patch_rad == 4) simdOMPCov_RepCenterHalfElement32F<4, 4>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 6 && patch_rad == 4) simdOMPCov_RepCenterHalfElement32F<6, 4>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 1 && patch_rad == 5) simdOMPCov_RepCenterHalfElement32F<1, 5>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 2 && patch_rad == 5) simdOMPCov_RepCenterHalfElement32F<2, 5>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 3 && patch_rad == 5) simdOMPCov_RepCenterHalfElement32F<3, 5>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 4 && patch_rad == 5) simdOMPCov_RepCenterHalfElement32F<4, 5>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 6 && patch_rad == 5) simdOMPCov_RepCenterHalfElement32F<6, 5>(src, cov, centerMethod, (float)const_sub);
						else simdOMPCov_RepCenterHalfElement32FCn(src, cov, centerMethod, (float)const_sub);
					}
				}
				else if (type == DRIM2COLElementSkipElement::REP)
				{
					if (centerMethod == CenterMethod::FULL)
					{
						simdOMPCovFullCenterRepElement32F(src, cov, border);
					}
					else
					{
						if (color_channels == 1 && patch_rad == 1)      simdOMPCov_RepCenterRepElement32F<1, 1>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 2 && patch_rad == 1) simdOMPCov_RepCenterRepElement32F<2, 1>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 3 && patch_rad == 1) simdOMPCov_RepCenterRepElement32F<3, 1>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 4 && patch_rad == 1) simdOMPCov_RepCenterRepElement32F<4, 1>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 6 && patch_rad == 1) simdOMPCov_RepCenterRepElement32F<6, 1>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 1 && patch_rad == 2) simdOMPCov_RepCenterRepElement32F<1, 2>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 2 && patch_rad == 2) simdOMPCov_RepCenterRepElement32F<2, 2>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 3 && patch_rad == 2) simdOMPCov_RepCenterRepElement32F<3, 2>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 4 && patch_rad == 2) simdOMPCov_RepCenterRepElement32F<4, 2>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 6 && patch_rad == 2) simdOMPCov_RepCenterRepElement32F<6, 2>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 1 && patch_rad == 3) simdOMPCov_RepCenterRepElement32F<1, 3>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 2 && patch_rad == 3) simdOMPCov_RepCenterRepElement32F<2, 3>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 3 && patch_rad == 3) simdOMPCov_RepCenterRepElement32F<3, 3>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 4 && patch_rad == 3) simdOMPCov_RepCenterRepElement32F<4, 3>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 6 && patch_rad == 3) simdOMPCov_RepCenterRepElement32F<6, 3>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 1 && patch_rad == 4) simdOMPCov_RepCenterRepElement32F<1, 4>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 2 && patch_rad == 4) simdOMPCov_RepCenterRepElement32F<2, 4>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 3 && patch_rad == 4) simdOMPCov_RepCenterRepElement32F<3, 4>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 4 && patch_rad == 4) simdOMPCov_RepCenterRepElement32F<4, 4>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 6 && patch_rad == 4) simdOMPCov_RepCenterRepElement32F<6, 4>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 1 && patch_rad == 5) simdOMPCov_RepCenterRepElement32F<1, 5>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 2 && patch_rad == 5) simdOMPCov_RepCenterRepElement32F<2, 5>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 3 && patch_rad == 5) simdOMPCov_RepCenterRepElement32F<3, 5>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 4 && patch_rad == 5) simdOMPCov_RepCenterRepElement32F<4, 5>(src, cov, centerMethod, (float)const_sub);
						else if (color_channels == 6 && patch_rad == 5) simdOMPCov_RepCenterRepElement32F<6, 5>(src, cov, centerMethod, (float)const_sub);
						else simdOMPCov_RepCenterRepElement32FCn(src, cov, centerMethod, (float)const_sub);
					}
				}
				else if (type == DRIM2COLElementSkipElement::CONV)
				{
					if (centerMethod == CenterMethod::FULL)
					{
						cout << "nosupport (Full-Conv), call dummy" << endl;
						simdOMPCovFullCenterRepElement32F(src, cov, border);
					}
					else
					{
						if (color_channels == 1 && patch_rad == 1)      simdOMPCov_RepCenterConvElement32F<1, 1>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 2 && patch_rad == 1) simdOMPCov_RepCenterConvElement32F<2, 1>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 3 && patch_rad == 1) simdOMPCov_RepCenterConvElement32F<3, 1>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 4 && patch_rad == 1) simdOMPCov_RepCenterConvElement32F<4, 1>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 6 && patch_rad == 1) simdOMPCov_RepCenterConvElement32F<6, 1>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 1 && patch_rad == 2) simdOMPCov_RepCenterConvElement32F<1, 2>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 2 && patch_rad == 2) simdOMPCov_RepCenterConvElement32F<2, 2>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 3 && patch_rad == 2) simdOMPCov_RepCenterConvElement32F<3, 2>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 4 && patch_rad == 2) simdOMPCov_RepCenterConvElement32F<4, 2>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 6 && patch_rad == 2) simdOMPCov_RepCenterConvElement32F<6, 2>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 1 && patch_rad == 3) simdOMPCov_RepCenterConvElement32F<1, 3>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 2 && patch_rad == 3) simdOMPCov_RepCenterConvElement32F<2, 3>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 3 && patch_rad == 3) simdOMPCov_RepCenterConvElement32F<3, 3>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 4 && patch_rad == 3) simdOMPCov_RepCenterConvElement32F<4, 3>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 6 && patch_rad == 3) simdOMPCov_RepCenterConvElement32F<6, 3>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 1 && patch_rad == 4) simdOMPCov_RepCenterConvElement32F<1, 4>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 2 && patch_rad == 4) simdOMPCov_RepCenterConvElement32F<2, 4>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 3 && patch_rad == 4) simdOMPCov_RepCenterConvElement32F<3, 4>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 4 && patch_rad == 4) simdOMPCov_RepCenterConvElement32F<4, 4>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 6 && patch_rad == 4) simdOMPCov_RepCenterConvElement32F<6, 4>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 1 && patch_rad == 5) simdOMPCov_RepCenterConvElement32F<1, 5>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 2 && patch_rad == 5) simdOMPCov_RepCenterConvElement32F<2, 5>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 3 && patch_rad == 5) simdOMPCov_RepCenterConvElement32F<3, 5>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 4 && patch_rad == 5) simdOMPCov_RepCenterConvElement32F<4, 5>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 6 && patch_rad == 5) simdOMPCov_RepCenterConvElement32F<6, 5>(src, cov, centerMethod, (float)const_sub, border);
						else simdOMPCov_RepCenterConvElement32FCn(src, cov, centerMethod, (float)const_sub, border);
					}
				}
				else if (type == DRIM2COLElementSkipElement::CONVF)
				{
					if (centerMethod == CenterMethod::FULL)
					{
						cout << "nosupport (Full-ConvF), call dummy" << endl;
						simdOMPCovFullCenterRepElement32F(src, cov, border);
					}
					else
					{
						if (color_channels == 1 && patch_rad == 1)      simdOMPCov_RepCenterConvFElement32F<1, 1>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 2 && patch_rad == 1) simdOMPCov_RepCenterConvFElement32F<2, 1>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 3 && patch_rad == 1) simdOMPCov_RepCenterConvFElement32F<3, 1>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 4 && patch_rad == 1) simdOMPCov_RepCenterConvFElement32F<4, 1>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 6 && patch_rad == 1) simdOMPCov_RepCenterConvFElement32F<6, 1>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 1 && patch_rad == 2) simdOMPCov_RepCenterConvFElement32F<1, 2>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 2 && patch_rad == 2) simdOMPCov_RepCenterConvFElement32F<2, 2>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 3 && patch_rad == 2) simdOMPCov_RepCenterConvFElement32F<3, 2>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 4 && patch_rad == 2) simdOMPCov_RepCenterConvFElement32F<4, 2>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 6 && patch_rad == 2) simdOMPCov_RepCenterConvFElement32F<6, 2>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 1 && patch_rad == 3) simdOMPCov_RepCenterConvFElement32F<1, 3>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 2 && patch_rad == 3) simdOMPCov_RepCenterConvFElement32F<2, 3>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 3 && patch_rad == 3) simdOMPCov_RepCenterConvFElement32F<3, 3>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 4 && patch_rad == 3) simdOMPCov_RepCenterConvFElement32F<4, 3>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 6 && patch_rad == 3) simdOMPCov_RepCenterConvFElement32F<6, 3>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 1 && patch_rad == 4) simdOMPCov_RepCenterConvFElement32F<1, 4>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 2 && patch_rad == 4) simdOMPCov_RepCenterConvFElement32F<2, 4>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 3 && patch_rad == 4) simdOMPCov_RepCenterConvFElement32F<3, 4>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 4 && patch_rad == 4) simdOMPCov_RepCenterConvFElement32F<4, 4>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 6 && patch_rad == 4) simdOMPCov_RepCenterConvFElement32F<6, 4>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 1 && patch_rad == 5) simdOMPCov_RepCenterConvFElement32F<1, 5>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 2 && patch_rad == 5) simdOMPCov_RepCenterConvFElement32F<2, 5>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 3 && patch_rad == 5) simdOMPCov_RepCenterConvFElement32F<3, 5>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 4 && patch_rad == 5) simdOMPCov_RepCenterConvFElement32F<4, 5>(src, cov, centerMethod, (float)const_sub, border);
						else if (color_channels == 6 && patch_rad == 5) simdOMPCov_RepCenterConvFElement32F<6, 5>(src, cov, centerMethod, (float)const_sub, border);
						else simdOMPCov_RepCenterConvFElement32FCn(src, cov, centerMethod, (float)const_sub, border);
					}
				}
				else if (type == DRIM2COLElementSkipElement::FFT)
				{
					if (centerMethod == CenterMethod::FULL)
					{
						cout << "nosupport (Full-ConvF), call dummy" << endl;
						simdOMPCovFullCenterRepElement32F(src, cov, border);
					}
					else
					{
						covFFT(src, cov, centerMethod, (float)const_sub);
					}
				}
				else if (type == DRIM2COLElementSkipElement::FFTF)
				{
					if (centerMethod == CenterMethod::FULL)
					{
						cout << "nosupport (Full-ConvF), call dummy" << endl;
						simdOMPCovFullCenterRepElement32F(src, cov, border);
					}
					else
					{
						//cout << "FFTF" << endl;
						covFFTFull(src, cov, centerMethod, (float)const_sub);
					}
				}
				else
				{
					cout << "do not have this type (computeCov): " << endl;
				}
			}
			else //64F
			{
				const DRIM2COLElementSkipElement type = getElementSkipType(method);
				if (type == DRIM2COLElementSkipElement::FULL)
				{
					simdOMPCovFullCenterFullElement32F(src, cov, border);
				}
				else if (type == DRIM2COLElementSkipElement::HALF)
				{
					if (centerMethod == CenterMethod::FULL)
					{
						//cout << "simdOMPCovFullSubHalfElement32F" << endl;
						simdOMPCovFullCenterHalfElement32F(src, cov, border);
					}
					else
					{
						if (color_channels == 1 && dim == 1)       simdOMPCov_RepCenterHalfElement64F<1, 1>(src, cov, centerMethod);
						else if (color_channels == 3 && dim == 3)  simdOMPCov_RepCenterHalfElement64F<3, 3>(src, cov, centerMethod);
						else if (color_channels == 1 && dim == 9)  simdOMPCov_RepCenterHalfElement64F<1, 9>(src, cov, centerMethod);
						else if (color_channels == 3 && dim == 27) simdOMPCov_RepCenterHalfElement64F<3, 27>(src, cov, centerMethod);
						else if (color_channels == 1 && dim == 25) simdOMPCov_RepCenterHalfElement64F<1, 25>(src, cov, centerMethod);
						else if (color_channels == 3 && dim == 75) simdOMPCov_RepCenterHalfElement64F<3, 75>(src, cov, centerMethod);
						else if (color_channels == 1 && dim == 49) simdOMPCov_RepCenterHalfElement64F<1, 49>(src, cov, centerMethod);
						else if (color_channels == 3 && dim == 147)simdOMPCov_RepCenterHalfElement64F<3, 147>(src, cov, centerMethod);
						else if (color_channels == 1 && dim == 81) simdOMPCov_RepCenterHalfElement64F<1, 81>(src, cov, centerMethod);
						else if (color_channels == 3 && dim == 243)simdOMPCov_RepCenterHalfElement64F<3, 243>(src, cov, centerMethod);
						else simdOMPCov_RepCenterHalfElement64FCn(src, cov, centerMethod);
					}
				}
				else if (type == DRIM2COLElementSkipElement::REP)
				{
					if (centerMethod == CenterMethod::FULL)
					{
						simdOMPCovFullCenterRepElement32F(src, cov, border);
					}
					else
					{
						if (color_channels == 1 && dim == 1)       simdOMPCov_RepCenterRepElement32F<1, 1>(src, cov, centerMethod);
						else if (color_channels == 3 && dim == 3)  simdOMPCov_RepCenterRepElement32F<3, 3>(src, cov, centerMethod);
						else if (color_channels == 1 && dim == 9)  simdOMPCov_RepCenterRepElement32F<1, 9>(src, cov, centerMethod);
						else if (color_channels == 3 && dim == 27) simdOMPCov_RepCenterRepElement32F<3, 27>(src, cov, centerMethod);
						else if (color_channels == 1 && dim == 25) simdOMPCov_RepCenterRepElement32F<1, 25>(src, cov, centerMethod);
						else if (color_channels == 3 && dim == 75) simdOMPCov_RepCenterRepElement32F<3, 75>(src, cov, centerMethod);
						else if (color_channels == 1 && dim == 49) simdOMPCov_RepCenterRepElement32F<1, 49>(src, cov, centerMethod);
						else if (color_channels == 3 && dim == 147)simdOMPCov_RepCenterRepElement32F<3, 147>(src, cov, centerMethod);
						else if (color_channels == 1 && dim == 81) simdOMPCov_RepCenterRepElement32F<1, 81>(src, cov, centerMethod);
						else if (color_channels == 3 && dim == 243)simdOMPCov_RepCenterRepElement32F<3, 243>(src, cov, centerMethod);
						else simdOMPCov_RepCenterRepElement32FCn(src, cov, centerMethod);
					}
				}

				/*if (color_channels == 1 && dim == 1)simd_64F<1, 1>(src, cov, covsubMethod);
				else if (color_channels == 3 && dim == 3)simd_64F<3, 3>(src, cov, covsubMethod);
				else if (color_channels == 1 && dim == 9)simd_64F<1, 9>(src, cov, covsubMethod);
				else if (color_channels == 3 && dim == 27)simd_64F<3, 27>(src, cov, covsubMethod);
				else if (color_channels == 1 && dim == 25)simd_64F<1, 25>(src, cov, covsubMethod);
				else if (color_channels == 3 && dim == 75)simd_64F<3, 75>(src, cov, covsubMethod);
				else if (color_channels == 1 && dim == 49) simd_64F<1, 49>(src, cov, covsubMethod);
				else if (color_channels == 3 && dim == 147)simd_64F<3, 147>(src, cov, covsubMethod);
				else if (color_channels == 1 && dim == 81) simd_64F<1, 81>(src, cov, covsubMethod);
				else if (color_channels == 3 && dim == 243)simd_64F<3, 243>(src, cov, covsubMethod);
				else simd_64FCn(src, cov, covsubMethod);*/
			}

			if (!isParallel) omp_set_num_threads(omp_get_num_procs());
		}
		else
		{
			simd_32FSkipCn(src, cov, skip);
		}
	}
}

//see vector<Mat> version for detailed infomation, just call the function.
void CalcPatchCovarMatrix::computeCov(const Mat& src, const int patch_rad, Mat& cov, const DRIM2COLType method, const int skip, const bool isParallel)
{
	vector<Mat> vsrc(src.channels());
	if (src.channels() == 1) vsrc[0] = src;
	else split(src, vsrc);

	computeCov(vsrc, patch_rad, cov, method, skip, isParallel);
}

void CalcPatchCovarMatrix::computeSeparateCovXXt(const vector<Mat>& src, const int patch_rad, const int borderType, vector<Mat>& covmat) {
	const int D = patch_rad * 2 + 1;
	const int DD = D * D;
	const int channels = (int)src.size();
	const int cols = src[0].cols, rows = src[0].rows;
	const float Normfactor = (float)D / (cols * rows);
	Mat srcborder, reduceVec, reduceScolor;
	vector<double> mcovx, mcovy;

	covmat.resize(2);

	if (channels == 1)
	{
		mcovx.resize((DD - D) / 2 + D);

		copyMakeBorder(src[0], srcborder, patch_rad, patch_rad, patch_rad, patch_rad, borderType);

		covmat[0].create(D, D, CV_64F);
		reduceVec.create(1, srcborder.cols, CV_64F);

		covmat[0].setTo(0.);

		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				reduceVec.setTo(0.);
				for (int x = 0; x < cols + 2 * patch_rad; x++)
				{
					for (int y = 0; y < rows; y++)
					{
						reduceVec.at<double>(0, x) += srcborder.at<float>(y + i, x) * srcborder.at<float>(y + j, x);
					}
				}

				for (int k = 0; k < D - 1; k++)
				{
					reduceVec.at<double>(0, k) *= double(k + 1) / D;
					reduceVec.at<double>(0, reduceVec.cols - 1 - k) *= double(k + 1) / D;
				}

				for (int x = 0; x < cols + 2 * patch_rad; x++)
				{
					mcovx[idx] += reduceVec.at<double>(0, x);
				}
				idx++;
			}
		}

		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				covmat[0].at<double>(i, j) = covmat[0].at<double>(j, i) = mcovx[idx++] * Normfactor;
			}
		}


		mcovy.resize((DD - D) / 2 + D);
		srcborder = srcborder.t();

		covmat[1].create(D, D, CV_64F);
		covmat[1].setTo(0.);
		reduceVec.create(1, srcborder.cols, CV_64F);

		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				reduceVec.setTo(0.);
				for (int x = 0; x < rows + 2 * patch_rad; x++)
				{
					for (int y = 0; y < cols; y++)
					{
						reduceVec.at<double>(0, x) += srcborder.at<float>(y + i, x) * srcborder.at<float>(y + j, x);
					}
				}

				for (int k = 0; k < D - 1; k++)
				{
					reduceVec.at<double>(0, k) *= double(k + 1) / D;
					reduceVec.at<double>(0, reduceVec.cols - 1 - k) *= double(k + 1) / D;
				}

				for (int x = 0; x < rows + 2 * patch_rad; x++) {
					mcovy[idx] += reduceVec.at<double>(0, x);
				}
				idx++;
			}
		}

		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				covmat[1].at<double>(i, j) = covmat[1].at<double>(j, i) = mcovy[idx++] * Normfactor;
			}
		}
	}
	else
	{
		Mat srcborder, srcborderx, srcboprdery;
		vector<Mat> vsrc, vsrcborder(channels);

		covmat[0].create(channels * D, channels * D, CV_64F);
		mcovx.resize(channels * channels * ((DD - D) / 2 + D));
		vsrcborder.resize(channels);
		covmat.resize(2);
		for (int c = 0; c < channels; c++)
		{
			copyMakeBorder(src[c], vsrcborder[c], patch_rad, patch_rad, patch_rad, patch_rad, borderType);
		}

		reduceVec.create(1, cols + 2 * patch_rad, CV_64F);

		covmat[0].setTo(0.);

		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				for (int c1 = 0; c1 < channels; c1++)
				{
					for (int c2 = 0; c2 < channels; c2++)
					{
						reduceVec.setTo(0.);
						for (int x = 0; x < cols + 2 * patch_rad; x++)
						{
							for (int y = 0; y < rows; y++)
							{
								reduceVec.at<double>(0, x) += vsrcborder[c1].at<float>(y + i, x) * vsrcborder[c2].at<float>(y + j, x);
							}
						}

						for (int k = 0; k < D - 1; k++)
						{
							reduceVec.at<double>(0, k) *= double(k + 1) / D;
							reduceVec.at<double>(0, reduceVec.cols - 1 - k) *= double(k + 1) / D;
						}

						for (int x = 0; x < cols + 2 * patch_rad; x++)
						{
							mcovx[idx] += reduceVec.at<double>(0, x);
						}
						idx++;
					}
				}
			}
		}
		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				for (int c1 = 0; c1 < channels; c1++)
				{
					for (int c2 = 0; c2 < channels; c2++)
					{
						covmat[0].at<double>(i * channels + c1, j * channels + c2) = covmat[0].at<double>(j * channels + c2, i * channels + c1) = mcovx[idx++] * Normfactor;
					}
				}
			}
		}

		for (int c = 0; c < channels; c++)
		{
			copyMakeBorder(src[c].t(), vsrcborder[c], patch_rad, patch_rad, patch_rad, patch_rad, borderType);
		}

		mcovy.resize((DD - D) / 2 + D);
		reduceVec.create(1, rows + 2 * patch_rad, CV_64F);

		covmat[1].create(D, D, CV_64F);
		covmat[1].setTo(0.);

		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				covmat[1].at<double>(i, j) = 0;
				for (int c = 0; c < channels; c++)
				{
					reduceVec.setTo(0.);
					for (int x = 0; x < rows + 2 * patch_rad; x++)
					{
						for (int y = 0; y < cols; y++)
						{
							reduceVec.at<double>(0, x) += vsrcborder[c].at<float>(y + i, x) * vsrcborder[c].at<float>(y + j, x);
						}
					}
					for (int k = 0; k < D - 1; k++)
					{
						reduceVec.at<double>(0, k) *= double(k + 1) / D;
						reduceVec.at<double>(0, reduceVec.cols - 1 - k) *= double(k + 1) / D;
					}
					for (int x = 0; x < cols + 2 * patch_rad; x++)
					{
						mcovy[idx] += reduceVec.at<double>(0, x);
					}
				}
				idx++;
			}
		}

		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				covmat[1].at<double>(i, j) = covmat[1].at<double>(j, i) = mcovy[idx++] * Normfactor;
			}
		}
	}
}

void CalcPatchCovarMatrix::computeSeparateCov(const vector<Mat>& src, const int patch_rad, const int borderType, vector<Mat>& covmat)
{
	const int D = patch_rad * 2 + 1;
	const int DD = D * D;
	const int channels = (int)src.size();
	const int cols = src[0].cols, rows = src[0].rows;
	const float Normfactor = (float)D / (cols * rows);

	const int threadMax = omp_get_num_procs();
	//print_debug(threadMax);
	vector<float> mcovh;
	vector<float> mcovv;

	covmat.resize(2);
	dataBorder.resize(channels);

	const bool isSIMD = true;
	if (channels == 1)
	{
		copyMakeBorder(src[0], dataBorder[0], patch_rad, patch_rad, patch_rad, patch_rad, borderType);

		vector<vector<float>> mtcovh(threadMax);
		vector<vector<float>> mtcovv(threadMax);

		mcovh.resize((DD - D) / 2 + D);
		mcovv.resize((DD - D) / 2 + D);
		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				mcovh[idx] = mcovv[idx] = 0.f;
				idx++;
			}
		}
		for (int t = 0; t < threadMax; t++)
		{
			mtcovh[t].resize((DD - D) / 2 + D);
			mtcovv[t].resize((DD - D) / 2 + D);
			for (int i = 0, idx = 0; i < D; i++)
			{
				for (int j = i; j < D; j++)
				{
					mtcovh[t][idx] = mtcovv[t][idx] = 0.f;
					idx++;
				}
			}
		}

		covmat[0].create(D, D, CV_64F);
		covmat[0].setTo(0.0);
		covmat[1].create(D, D, CV_64F);
		covmat[1].setTo(0.0);

		if (isSIMD)
		{
#pragma omp parallel for schedule (static)
			for (int y = 0; y < rows; y++)
			{
				const int t = omp_get_thread_num();
				for (int i = 0, idx = 0; i < D; i++)
				{
					for (int j = i; j < D; j++)
					{
						const float* sh = dataBorder[0].ptr<float>(y + i, patch_rad);
						const float* th = dataBorder[0].ptr<float>(y + j, patch_rad);
						const float* sv = dataBorder[0].ptr<float>(y + patch_rad, i);
						const float* tv = dataBorder[0].ptr<float>(y + patch_rad, j);
						__m256 psumh = _mm256_setzero_ps();
						__m256 psumv = _mm256_setzero_ps();
						for (int x = 0; x < cols; x += 8)
						{
							psumh = _mm256_fmadd_ps(_mm256_loadu_ps(sh + x), _mm256_loadu_ps(th + x), psumh);
							psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv + x), _mm256_loadu_ps(tv + x), psumv);
						}
						mtcovh[t][idx] += _mm256_reduceadd_ps(psumh);
						mtcovv[t][idx] += _mm256_reduceadd_ps(psumv);
						idx++;
					}
				}
			}

			//reduction
			for (int i = 0, idx = 0; i < D; i++)
			{
				for (int j = i; j < D; j++)
				{
					for (int t = 0; t < threadMax; t++)
					{
						mcovh[idx] += mtcovh[t][idx];
						mcovv[idx] += mtcovv[t][idx];
					}
					idx++;
				}
			}
		}
		else
		{
			for (int y = 0; y < rows; y++)
			{
				for (int i = 0, idx = 0; i < D; i++)
				{
					for (int j = i; j < D; j++)
					{
						const float* sh = dataBorder[0].ptr<float>(y + i, patch_rad);
						const float* th = dataBorder[0].ptr<float>(y + j, patch_rad);
						const float* sv = dataBorder[0].ptr<float>(y + patch_rad, i);
						const float* tv = dataBorder[0].ptr<float>(y + patch_rad, j);
						for (int x = 0; x < cols; x++)
						{
							mcovh[idx] += sh[x] * th[x];
							mcovv[idx] += sv[x] * tv[x];
						}
						idx++;
					}
				}
			}
		}

		//v
		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				covmat[0].at<double>(i, j) = covmat[0].at<double>(j, i) = mcovh[idx++] * Normfactor;
			}
		}
		//h
		for (int j = 0, idx = 0; j < D; j++)
		{
			for (int i = j; i < D; i++)
			{
				covmat[1].at<double>(i, j) = covmat[1].at<double>(j, i) = mcovv[idx++] * Normfactor;
			}
		}
	}
	else
	{
		for (int c = 0; c < channels; c++)
		{
			copyMakeBorder(src[c], dataBorder[c], patch_rad, patch_rad, patch_rad, patch_rad, borderType);
		}

		vector<vector<float>> mtcovh(threadMax);
		vector<vector<float>> mtcovv(threadMax);
		mcovh.resize(channels * channels * ((DD - D) / 2 + D));
		mcovv.resize((DD - D) / 2 + D);
		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				for (int c1 = 0; c1 < channels; c1++)
				{
					for (int c2 = 0; c2 < channels; c2++)
					{
						mcovh[idx++] = 0.f;
					}
				}
			}
		}
		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				mcovv[idx++] = 0.f;
			}
		}
		for (int t = 0; t < threadMax; t++)
		{
			mtcovh[t].resize(channels * channels * ((DD - D) / 2 + D));
			mtcovv[t].resize((DD - D) / 2 + D);
			//h
			for (int i = 0, idx = 0; i < D; i++)
			{
				for (int j = i; j < D; j++)
				{
					for (int c1 = 0; c1 < channels; c1++)
					{
						for (int c2 = 0; c2 < channels; c2++)
						{
							mtcovh[t][idx++] = 0.f;
						}
					}
				}
			}
			//v
			for (int i = 0, idx = 0; i < D; i++)
			{
				for (int j = i; j < D; j++)
				{
					mtcovv[t][idx++] = 0.f;
				}
			}
		}

		covmat[0].create(channels * D, channels * D, CV_64F);
		covmat[0].setTo(0.0);
		covmat[1].create(D, D, CV_64F);
		covmat[1].setTo(0.0);
		if (channels == 2)
		{
			if (isSIMD)
			{
#pragma omp parallel for schedule (static)
				for (int y = 0; y < rows; y++)
				{
					const int t = omp_get_thread_num();
					for (int i = 0, idxh = 0, idxv = 0; i < D; i++)
					{
						for (int j = i; j < D; j++)
						{
							//h
							const float* sh0 = dataBorder[0].ptr<float>(y + i, patch_rad);
							const float* sh1 = dataBorder[1].ptr<float>(y + i, patch_rad);
							const float* th0 = dataBorder[0].ptr<float>(y + j, patch_rad);
							const float* th1 = dataBorder[1].ptr<float>(y + j, patch_rad);
							__m256 psumh00 = _mm256_setzero_ps();
							__m256 psumh01 = _mm256_setzero_ps();
							__m256 psumh10 = _mm256_setzero_ps();
							__m256 psumh11 = _mm256_setzero_ps();
							for (int x = 0; x < cols; x += 8)
							{
								const __m256 msh0 = _mm256_loadu_ps(sh0 + x);
								const __m256 msh1 = _mm256_loadu_ps(sh1 + x);
								const __m256 mth0 = _mm256_loadu_ps(th0 + x);
								const __m256 mth1 = _mm256_loadu_ps(th1 + x);
								psumh00 = _mm256_fmadd_ps(msh0, mth0, psumh00);
								psumh01 = _mm256_fmadd_ps(msh0, mth1, psumh01);
								psumh10 = _mm256_fmadd_ps(msh1, mth0, psumh10);
								psumh11 = _mm256_fmadd_ps(msh1, mth1, psumh11);
							}
							mtcovh[t][idxh + 0] += _mm256_reduceadd_ps(psumh00);
							mtcovh[t][idxh + 1] += _mm256_reduceadd_ps(psumh01);
							mtcovh[t][idxh + 2] += _mm256_reduceadd_ps(psumh10);
							mtcovh[t][idxh + 3] += _mm256_reduceadd_ps(psumh11);
							idxh += 4;

							//v
							const float* sv0 = dataBorder[0].ptr<float>(y + patch_rad, i);
							const float* tv0 = dataBorder[0].ptr<float>(y + patch_rad, j);
							const float* sv1 = dataBorder[1].ptr<float>(y + patch_rad, i);
							const float* tv1 = dataBorder[1].ptr<float>(y + patch_rad, j);
							__m256 psumv = _mm256_setzero_ps();
							for (int x = 0; x < cols; x += 8)
							{
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv0 + x), _mm256_loadu_ps(tv0 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv1 + x), _mm256_loadu_ps(tv1 + x), psumv);
							}
							mtcovv[t][idxv] += _mm256_reduceadd_ps(psumv);
							idxv++;
						}
					}
				}
			}
			else
			{
				for (int y = 0; y < rows; y++)
				{
					for (int i = 0, idxh = 0, idxv = 0; i < D; i++)
					{
						for (int j = i; j < D; j++)
						{
							for (int c1 = 0; c1 < channels; c1++)
							{
								const float* sh = dataBorder[c1].ptr<float>(y + i, patch_rad);
								for (int c2 = 0; c2 < channels; c2++)
								{
									const float* th = dataBorder[c2].ptr<float>(y + j, patch_rad);
									for (int x = 0; x < cols; x++)
									{
										mcovh[idxh] += sh[x] * th[x];
									}
									idxh++;
								}
							}

							for (int c = 0; c < channels; c++)
							{
								const float* sv = dataBorder[c].ptr<float>(y + patch_rad, i);
								const float* tv = dataBorder[c].ptr<float>(y + patch_rad, j);
								for (int x = 0; x < cols; x++)
								{
									mcovv[idxv] += sv[x] * tv[x];
								}
							}
							idxv++;
						}
					}
				}
			}
		}
		else if (channels == 3)
		{
			if (isSIMD)
			{
#pragma omp parallel for schedule (static)
				for (int y = 0; y < rows; y++)
				{
					const int t = omp_get_thread_num();
					for (int i = 0, idxh = 0, idxv = 0; i < D; i++)
					{
						for (int j = i; j < D; j++)
						{
							//h
							const float* sh0 = dataBorder[0].ptr<float>(y + i, patch_rad);
							const float* sh1 = dataBorder[1].ptr<float>(y + i, patch_rad);
							const float* sh2 = dataBorder[2].ptr<float>(y + i, patch_rad);
							const float* th0 = dataBorder[0].ptr<float>(y + j, patch_rad);
							const float* th1 = dataBorder[1].ptr<float>(y + j, patch_rad);
							const float* th2 = dataBorder[2].ptr<float>(y + j, patch_rad);
							__m256 psumh00 = _mm256_setzero_ps();
							__m256 psumh01 = _mm256_setzero_ps();
							__m256 psumh02 = _mm256_setzero_ps();
							__m256 psumh10 = _mm256_setzero_ps();
							__m256 psumh11 = _mm256_setzero_ps();
							__m256 psumh12 = _mm256_setzero_ps();
							__m256 psumh20 = _mm256_setzero_ps();
							__m256 psumh21 = _mm256_setzero_ps();
							__m256 psumh22 = _mm256_setzero_ps();
							for (int x = 0; x < cols; x += 8)
							{
								const __m256 msh0 = _mm256_loadu_ps(sh0 + x);
								const __m256 msh1 = _mm256_loadu_ps(sh1 + x);
								const __m256 msh2 = _mm256_loadu_ps(sh2 + x);
								const __m256 mth0 = _mm256_loadu_ps(th0 + x);
								const __m256 mth1 = _mm256_loadu_ps(th1 + x);
								const __m256 mth2 = _mm256_loadu_ps(th2 + x);
								psumh00 = _mm256_fmadd_ps(msh0, mth0, psumh00);
								psumh01 = _mm256_fmadd_ps(msh0, mth1, psumh01);
								psumh02 = _mm256_fmadd_ps(msh0, mth2, psumh02);
								psumh10 = _mm256_fmadd_ps(msh1, mth0, psumh10);
								psumh11 = _mm256_fmadd_ps(msh1, mth1, psumh11);
								psumh12 = _mm256_fmadd_ps(msh1, mth2, psumh12);
								psumh20 = _mm256_fmadd_ps(msh2, mth0, psumh20);
								psumh21 = _mm256_fmadd_ps(msh2, mth1, psumh21);
								psumh22 = _mm256_fmadd_ps(msh2, mth2, psumh22);
							}
							mtcovh[t][idxh + 0] += _mm256_reduceadd_ps(psumh00);
							mtcovh[t][idxh + 1] += _mm256_reduceadd_ps(psumh01);
							mtcovh[t][idxh + 2] += _mm256_reduceadd_ps(psumh02);
							mtcovh[t][idxh + 3] += _mm256_reduceadd_ps(psumh10);
							mtcovh[t][idxh + 4] += _mm256_reduceadd_ps(psumh11);
							mtcovh[t][idxh + 5] += _mm256_reduceadd_ps(psumh12);
							mtcovh[t][idxh + 6] += _mm256_reduceadd_ps(psumh20);
							mtcovh[t][idxh + 7] += _mm256_reduceadd_ps(psumh21);
							mtcovh[t][idxh + 8] += _mm256_reduceadd_ps(psumh22);
							idxh += 9;

							//v
							const float* sv0 = dataBorder[0].ptr<float>(y + patch_rad, i);
							const float* tv0 = dataBorder[0].ptr<float>(y + patch_rad, j);
							const float* sv1 = dataBorder[1].ptr<float>(y + patch_rad, i);
							const float* tv1 = dataBorder[1].ptr<float>(y + patch_rad, j);
							const float* sv2 = dataBorder[2].ptr<float>(y + patch_rad, i);
							const float* tv2 = dataBorder[2].ptr<float>(y + patch_rad, j);
							__m256 psumv = _mm256_setzero_ps();
							for (int x = 0; x < cols; x += 8)
							{
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv0 + x), _mm256_loadu_ps(tv0 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv1 + x), _mm256_loadu_ps(tv1 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv2 + x), _mm256_loadu_ps(tv2 + x), psumv);
							}
							mtcovv[t][idxv] += _mm256_reduceadd_ps(psumv);
							idxv++;
						}
					}
				}
			}
			else
			{
				for (int y = 0; y < rows; y++)
				{
					for (int i = 0, idxh = 0, idxv = 0; i < D; i++)
					{
						for (int j = i; j < D; j++)
						{
							for (int c1 = 0; c1 < channels; c1++)
							{
								const float* sh = dataBorder[c1].ptr<float>(y + i, patch_rad);
								for (int c2 = 0; c2 < channels; c2++)
								{
									const float* th = dataBorder[c2].ptr<float>(y + j, patch_rad);
									for (int x = 0; x < cols; x++)
									{
										mcovh[idxh] += sh[x] * th[x];
									}
									idxh++;
								}
							}

							for (int c = 0; c < channels; c++)
							{
								const float* sv = dataBorder[c].ptr<float>(y + patch_rad, i);
								const float* tv = dataBorder[c].ptr<float>(y + patch_rad, j);
								for (int x = 0; x < cols; x++)
								{
									mcovv[idxv] += sv[x] * tv[x];
								}
							}
							idxv++;
						}
					}
				}
			}
		}
		else if (channels == 4)
		{
			if (isSIMD)
			{
#pragma omp parallel for schedule (static)
				for (int y = 0; y < rows; y++)
				{
					const int t = omp_get_thread_num();
					for (int i = 0, idxh = 0, idxv = 0; i < D; i++)
					{
						for (int j = i; j < D; j++)
						{
							//h
							const float* sh0 = dataBorder[0].ptr<float>(y + i, patch_rad);
							const float* sh1 = dataBorder[1].ptr<float>(y + i, patch_rad);
							const float* sh2 = dataBorder[2].ptr<float>(y + i, patch_rad);
							const float* sh3 = dataBorder[3].ptr<float>(y + i, patch_rad);
							const float* th0 = dataBorder[0].ptr<float>(y + j, patch_rad);
							const float* th1 = dataBorder[1].ptr<float>(y + j, patch_rad);
							const float* th2 = dataBorder[2].ptr<float>(y + j, patch_rad);
							const float* th3 = dataBorder[3].ptr<float>(y + j, patch_rad);
							__m256 psumh00 = _mm256_setzero_ps();
							__m256 psumh01 = _mm256_setzero_ps();
							__m256 psumh02 = _mm256_setzero_ps();
							__m256 psumh03 = _mm256_setzero_ps();
							__m256 psumh10 = _mm256_setzero_ps();
							__m256 psumh11 = _mm256_setzero_ps();
							__m256 psumh12 = _mm256_setzero_ps();
							__m256 psumh13 = _mm256_setzero_ps();
							__m256 psumh20 = _mm256_setzero_ps();
							__m256 psumh21 = _mm256_setzero_ps();
							__m256 psumh22 = _mm256_setzero_ps();
							__m256 psumh23 = _mm256_setzero_ps();
							__m256 psumh30 = _mm256_setzero_ps();
							__m256 psumh31 = _mm256_setzero_ps();
							__m256 psumh32 = _mm256_setzero_ps();
							__m256 psumh33 = _mm256_setzero_ps();
							for (int x = 0; x < cols; x += 8)
							{
								const __m256 msh0 = _mm256_loadu_ps(sh0 + x);
								const __m256 msh1 = _mm256_loadu_ps(sh1 + x);
								const __m256 msh2 = _mm256_loadu_ps(sh2 + x);
								const __m256 msh3 = _mm256_loadu_ps(sh3 + x);
								const __m256 mth0 = _mm256_loadu_ps(th0 + x);
								const __m256 mth1 = _mm256_loadu_ps(th1 + x);
								const __m256 mth2 = _mm256_loadu_ps(th2 + x);
								const __m256 mth3 = _mm256_loadu_ps(th3 + x);
								psumh00 = _mm256_fmadd_ps(msh0, mth0, psumh00);
								psumh01 = _mm256_fmadd_ps(msh0, mth1, psumh01);
								psumh02 = _mm256_fmadd_ps(msh0, mth2, psumh02);
								psumh03 = _mm256_fmadd_ps(msh0, mth3, psumh03);
								psumh10 = _mm256_fmadd_ps(msh1, mth0, psumh10);
								psumh11 = _mm256_fmadd_ps(msh1, mth1, psumh11);
								psumh12 = _mm256_fmadd_ps(msh1, mth2, psumh12);
								psumh13 = _mm256_fmadd_ps(msh1, mth3, psumh13);
								psumh20 = _mm256_fmadd_ps(msh2, mth0, psumh20);
								psumh21 = _mm256_fmadd_ps(msh2, mth1, psumh21);
								psumh22 = _mm256_fmadd_ps(msh2, mth2, psumh22);
								psumh23 = _mm256_fmadd_ps(msh2, mth3, psumh23);
								psumh30 = _mm256_fmadd_ps(msh3, mth0, psumh30);
								psumh31 = _mm256_fmadd_ps(msh3, mth1, psumh31);
								psumh32 = _mm256_fmadd_ps(msh3, mth2, psumh32);
								psumh33 = _mm256_fmadd_ps(msh3, mth3, psumh33);
							}
							mtcovh[t][idxh + 0] += _mm256_reduceadd_ps(psumh00);
							mtcovh[t][idxh + 1] += _mm256_reduceadd_ps(psumh01);
							mtcovh[t][idxh + 2] += _mm256_reduceadd_ps(psumh02);
							mtcovh[t][idxh + 3] += _mm256_reduceadd_ps(psumh03);
							mtcovh[t][idxh + 4] += _mm256_reduceadd_ps(psumh10);
							mtcovh[t][idxh + 5] += _mm256_reduceadd_ps(psumh11);
							mtcovh[t][idxh + 6] += _mm256_reduceadd_ps(psumh12);
							mtcovh[t][idxh + 7] += _mm256_reduceadd_ps(psumh13);
							mtcovh[t][idxh + 8] += _mm256_reduceadd_ps(psumh20);
							mtcovh[t][idxh + 9] += _mm256_reduceadd_ps(psumh21);
							mtcovh[t][idxh + 10] += _mm256_reduceadd_ps(psumh22);
							mtcovh[t][idxh + 11] += _mm256_reduceadd_ps(psumh23);
							mtcovh[t][idxh + 12] += _mm256_reduceadd_ps(psumh30);
							mtcovh[t][idxh + 13] += _mm256_reduceadd_ps(psumh31);
							mtcovh[t][idxh + 14] += _mm256_reduceadd_ps(psumh32);
							mtcovh[t][idxh + 15] += _mm256_reduceadd_ps(psumh33);
							idxh += 16;

							//v
							const float* sv0 = dataBorder[0].ptr<float>(y + patch_rad, i);
							const float* tv0 = dataBorder[0].ptr<float>(y + patch_rad, j);
							const float* sv1 = dataBorder[1].ptr<float>(y + patch_rad, i);
							const float* tv1 = dataBorder[1].ptr<float>(y + patch_rad, j);
							const float* sv2 = dataBorder[2].ptr<float>(y + patch_rad, i);
							const float* tv2 = dataBorder[2].ptr<float>(y + patch_rad, j);
							const float* sv3 = dataBorder[3].ptr<float>(y + patch_rad, i);
							const float* tv3 = dataBorder[3].ptr<float>(y + patch_rad, j);
							__m256 psumv = _mm256_setzero_ps();
							for (int x = 0; x < cols; x += 8)
							{
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv0 + x), _mm256_loadu_ps(tv0 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv1 + x), _mm256_loadu_ps(tv1 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv2 + x), _mm256_loadu_ps(tv2 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv3 + x), _mm256_loadu_ps(tv3 + x), psumv);
							}
							mtcovv[t][idxv] += _mm256_reduceadd_ps(psumv);
							idxv++;
						}
					}
				}
			}
			else
			{
				for (int y = 0; y < rows; y++)
				{
					for (int i = 0, idxh = 0, idxv = 0; i < D; i++)
					{
						for (int j = i; j < D; j++)
						{
							for (int c1 = 0; c1 < channels; c1++)
							{
								const float* sh = dataBorder[c1].ptr<float>(y + i, patch_rad);
								for (int c2 = 0; c2 < channels; c2++)
								{
									const float* th = dataBorder[c2].ptr<float>(y + j, patch_rad);
									for (int x = 0; x < cols; x++)
									{
										mcovh[idxh] += sh[x] * th[x];
									}
									idxh++;
								}
							}

							for (int c = 0; c < channels; c++)
							{
								const float* sv = dataBorder[c].ptr<float>(y + patch_rad, i);
								const float* tv = dataBorder[c].ptr<float>(y + patch_rad, j);
								for (int x = 0; x < cols; x++)
								{
									mcovv[idxv] += sv[x] * tv[x];
								}
							}
							idxv++;
						}
					}
				}
			}
		}
		else if (channels == 6)
		{
			if (isSIMD)
			{
#pragma omp parallel for schedule (static)
				for (int y = 0; y < rows; y++)
				{
					const int t = omp_get_thread_num();
					for (int i = 0, idxh = 0, idxv = 0; i < D; i++)
					{
						for (int j = i; j < D; j++)
						{
							//h
							const float* sh0 = dataBorder[0].ptr<float>(y + i, patch_rad);
							const float* sh1 = dataBorder[1].ptr<float>(y + i, patch_rad);
							const float* sh2 = dataBorder[2].ptr<float>(y + i, patch_rad);
							const float* sh3 = dataBorder[3].ptr<float>(y + i, patch_rad);
							const float* sh4 = dataBorder[4].ptr<float>(y + i, patch_rad);
							const float* sh5 = dataBorder[5].ptr<float>(y + i, patch_rad);
							const float* th0 = dataBorder[0].ptr<float>(y + j, patch_rad);
							const float* th1 = dataBorder[1].ptr<float>(y + j, patch_rad);
							const float* th2 = dataBorder[2].ptr<float>(y + j, patch_rad);
							const float* th3 = dataBorder[3].ptr<float>(y + j, patch_rad);
							const float* th4 = dataBorder[4].ptr<float>(y + j, patch_rad);
							const float* th5 = dataBorder[5].ptr<float>(y + j, patch_rad);
							__m256 psumh00 = _mm256_setzero_ps();
							__m256 psumh01 = _mm256_setzero_ps();
							__m256 psumh02 = _mm256_setzero_ps();
							__m256 psumh03 = _mm256_setzero_ps();
							__m256 psumh04 = _mm256_setzero_ps();
							__m256 psumh05 = _mm256_setzero_ps();
							__m256 psumh10 = _mm256_setzero_ps();
							__m256 psumh11 = _mm256_setzero_ps();
							__m256 psumh12 = _mm256_setzero_ps();
							__m256 psumh13 = _mm256_setzero_ps();
							__m256 psumh14 = _mm256_setzero_ps();
							__m256 psumh15 = _mm256_setzero_ps();
							__m256 psumh20 = _mm256_setzero_ps();
							__m256 psumh21 = _mm256_setzero_ps();
							__m256 psumh22 = _mm256_setzero_ps();
							__m256 psumh23 = _mm256_setzero_ps();
							__m256 psumh24 = _mm256_setzero_ps();
							__m256 psumh25 = _mm256_setzero_ps();
							__m256 psumh30 = _mm256_setzero_ps();
							__m256 psumh31 = _mm256_setzero_ps();
							__m256 psumh32 = _mm256_setzero_ps();
							__m256 psumh33 = _mm256_setzero_ps();
							__m256 psumh34 = _mm256_setzero_ps();
							__m256 psumh35 = _mm256_setzero_ps();
							__m256 psumh40 = _mm256_setzero_ps();
							__m256 psumh41 = _mm256_setzero_ps();
							__m256 psumh42 = _mm256_setzero_ps();
							__m256 psumh43 = _mm256_setzero_ps();
							__m256 psumh44 = _mm256_setzero_ps();
							__m256 psumh45 = _mm256_setzero_ps();
							__m256 psumh50 = _mm256_setzero_ps();
							__m256 psumh51 = _mm256_setzero_ps();
							__m256 psumh52 = _mm256_setzero_ps();
							__m256 psumh53 = _mm256_setzero_ps();
							__m256 psumh54 = _mm256_setzero_ps();
							__m256 psumh55 = _mm256_setzero_ps();
							for (int x = 0; x < cols; x += 8)
							{
								const __m256 msh0 = _mm256_loadu_ps(sh0 + x);
								const __m256 msh1 = _mm256_loadu_ps(sh1 + x);
								const __m256 msh2 = _mm256_loadu_ps(sh2 + x);
								const __m256 msh3 = _mm256_loadu_ps(sh3 + x);
								const __m256 msh4 = _mm256_loadu_ps(sh4 + x);
								const __m256 msh5 = _mm256_loadu_ps(sh5 + x);
								const __m256 mth0 = _mm256_loadu_ps(th0 + x);
								const __m256 mth1 = _mm256_loadu_ps(th1 + x);
								const __m256 mth2 = _mm256_loadu_ps(th2 + x);
								const __m256 mth3 = _mm256_loadu_ps(th3 + x);
								const __m256 mth4 = _mm256_loadu_ps(th4 + x);
								const __m256 mth5 = _mm256_loadu_ps(th5 + x);
								psumh00 = _mm256_fmadd_ps(msh0, mth0, psumh00);
								psumh01 = _mm256_fmadd_ps(msh0, mth1, psumh01);
								psumh02 = _mm256_fmadd_ps(msh0, mth2, psumh02);
								psumh03 = _mm256_fmadd_ps(msh0, mth3, psumh03);
								psumh04 = _mm256_fmadd_ps(msh0, mth4, psumh04);
								psumh05 = _mm256_fmadd_ps(msh0, mth5, psumh05);
								psumh10 = _mm256_fmadd_ps(msh1, mth0, psumh10);
								psumh11 = _mm256_fmadd_ps(msh1, mth1, psumh11);
								psumh12 = _mm256_fmadd_ps(msh1, mth2, psumh12);
								psumh13 = _mm256_fmadd_ps(msh1, mth3, psumh13);
								psumh14 = _mm256_fmadd_ps(msh1, mth4, psumh14);
								psumh15 = _mm256_fmadd_ps(msh1, mth5, psumh15);
								psumh20 = _mm256_fmadd_ps(msh2, mth0, psumh20);
								psumh21 = _mm256_fmadd_ps(msh2, mth1, psumh21);
								psumh22 = _mm256_fmadd_ps(msh2, mth2, psumh22);
								psumh23 = _mm256_fmadd_ps(msh2, mth3, psumh23);
								psumh24 = _mm256_fmadd_ps(msh2, mth4, psumh24);
								psumh25 = _mm256_fmadd_ps(msh2, mth5, psumh25);
								psumh30 = _mm256_fmadd_ps(msh3, mth0, psumh30);
								psumh31 = _mm256_fmadd_ps(msh3, mth1, psumh31);
								psumh32 = _mm256_fmadd_ps(msh3, mth2, psumh32);
								psumh33 = _mm256_fmadd_ps(msh3, mth3, psumh33);
								psumh34 = _mm256_fmadd_ps(msh3, mth4, psumh34);
								psumh35 = _mm256_fmadd_ps(msh3, mth5, psumh35);
								psumh40 = _mm256_fmadd_ps(msh4, mth0, psumh40);
								psumh41 = _mm256_fmadd_ps(msh4, mth1, psumh41);
								psumh42 = _mm256_fmadd_ps(msh4, mth2, psumh42);
								psumh43 = _mm256_fmadd_ps(msh4, mth3, psumh43);
								psumh44 = _mm256_fmadd_ps(msh4, mth4, psumh44);
								psumh45 = _mm256_fmadd_ps(msh4, mth5, psumh45);
								psumh50 = _mm256_fmadd_ps(msh5, mth0, psumh50);
								psumh51 = _mm256_fmadd_ps(msh5, mth1, psumh51);
								psumh52 = _mm256_fmadd_ps(msh5, mth2, psumh52);
								psumh53 = _mm256_fmadd_ps(msh5, mth3, psumh53);
								psumh54 = _mm256_fmadd_ps(msh5, mth4, psumh54);
								psumh55 = _mm256_fmadd_ps(msh5, mth5, psumh55);
							}
							mtcovh[t][idxh + 0] += _mm256_reduceadd_ps(psumh00);
							mtcovh[t][idxh + 1] += _mm256_reduceadd_ps(psumh01);
							mtcovh[t][idxh + 2] += _mm256_reduceadd_ps(psumh02);
							mtcovh[t][idxh + 3] += _mm256_reduceadd_ps(psumh03);
							mtcovh[t][idxh + 4] += _mm256_reduceadd_ps(psumh04);
							mtcovh[t][idxh + 5] += _mm256_reduceadd_ps(psumh05);
							mtcovh[t][idxh + 6] += _mm256_reduceadd_ps(psumh10);
							mtcovh[t][idxh + 7] += _mm256_reduceadd_ps(psumh11);
							mtcovh[t][idxh + 8] += _mm256_reduceadd_ps(psumh12);
							mtcovh[t][idxh + 9] += _mm256_reduceadd_ps(psumh13);
							mtcovh[t][idxh + 10] += _mm256_reduceadd_ps(psumh14);
							mtcovh[t][idxh + 11] += _mm256_reduceadd_ps(psumh15);
							mtcovh[t][idxh + 12] += _mm256_reduceadd_ps(psumh20);
							mtcovh[t][idxh + 13] += _mm256_reduceadd_ps(psumh21);
							mtcovh[t][idxh + 14] += _mm256_reduceadd_ps(psumh22);
							mtcovh[t][idxh + 15] += _mm256_reduceadd_ps(psumh23);
							mtcovh[t][idxh + 16] += _mm256_reduceadd_ps(psumh24);
							mtcovh[t][idxh + 17] += _mm256_reduceadd_ps(psumh25);
							mtcovh[t][idxh + 18] += _mm256_reduceadd_ps(psumh30);
							mtcovh[t][idxh + 19] += _mm256_reduceadd_ps(psumh31);
							mtcovh[t][idxh + 20] += _mm256_reduceadd_ps(psumh32);
							mtcovh[t][idxh + 21] += _mm256_reduceadd_ps(psumh33);
							mtcovh[t][idxh + 22] += _mm256_reduceadd_ps(psumh34);
							mtcovh[t][idxh + 23] += _mm256_reduceadd_ps(psumh35);
							mtcovh[t][idxh + 24] += _mm256_reduceadd_ps(psumh40);
							mtcovh[t][idxh + 25] += _mm256_reduceadd_ps(psumh41);
							mtcovh[t][idxh + 26] += _mm256_reduceadd_ps(psumh42);
							mtcovh[t][idxh + 27] += _mm256_reduceadd_ps(psumh43);
							mtcovh[t][idxh + 28] += _mm256_reduceadd_ps(psumh44);
							mtcovh[t][idxh + 29] += _mm256_reduceadd_ps(psumh45);
							mtcovh[t][idxh + 30] += _mm256_reduceadd_ps(psumh50);
							mtcovh[t][idxh + 31] += _mm256_reduceadd_ps(psumh51);
							mtcovh[t][idxh + 32] += _mm256_reduceadd_ps(psumh52);
							mtcovh[t][idxh + 33] += _mm256_reduceadd_ps(psumh53);
							mtcovh[t][idxh + 34] += _mm256_reduceadd_ps(psumh54);
							mtcovh[t][idxh + 35] += _mm256_reduceadd_ps(psumh55);
							idxh += 36;

							//v
							const float* sv0 = dataBorder[0].ptr<float>(y + patch_rad, i);
							const float* tv0 = dataBorder[0].ptr<float>(y + patch_rad, j);
							const float* sv1 = dataBorder[1].ptr<float>(y + patch_rad, i);
							const float* tv1 = dataBorder[1].ptr<float>(y + patch_rad, j);
							const float* sv2 = dataBorder[2].ptr<float>(y + patch_rad, i);
							const float* tv2 = dataBorder[2].ptr<float>(y + patch_rad, j);
							const float* sv3 = dataBorder[3].ptr<float>(y + patch_rad, i);
							const float* tv3 = dataBorder[3].ptr<float>(y + patch_rad, j);
							const float* sv4 = dataBorder[4].ptr<float>(y + patch_rad, i);
							const float* tv4 = dataBorder[4].ptr<float>(y + patch_rad, j);
							const float* sv5 = dataBorder[5].ptr<float>(y + patch_rad, i);
							const float* tv5 = dataBorder[5].ptr<float>(y + patch_rad, j);
							__m256 psumv = _mm256_setzero_ps();
							for (int x = 0; x < cols; x += 8)
							{
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv0 + x), _mm256_loadu_ps(tv0 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv1 + x), _mm256_loadu_ps(tv1 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv2 + x), _mm256_loadu_ps(tv2 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv3 + x), _mm256_loadu_ps(tv3 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv4 + x), _mm256_loadu_ps(tv4 + x), psumv);
								psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv5 + x), _mm256_loadu_ps(tv5 + x), psumv);
							}
							mtcovv[t][idxv] += _mm256_reduceadd_ps(psumv);
							idxv++;
						}
					}
				}
			}
			else
			{
				for (int y = 0; y < rows; y++)
				{
					for (int i = 0, idxh = 0, idxv = 0; i < D; i++)
					{
						for (int j = i; j < D; j++)
						{
							for (int c1 = 0; c1 < channels; c1++)
							{
								const float* sh = dataBorder[c1].ptr<float>(y + i, patch_rad);
								for (int c2 = 0; c2 < channels; c2++)
								{
									const float* th = dataBorder[c2].ptr<float>(y + j, patch_rad);
									for (int x = 0; x < cols; x++)
									{
										mcovh[idxh] += sh[x] * th[x];
									}
									idxh++;
								}
							}

							for (int c = 0; c < channels; c++)
							{
								const float* sv = dataBorder[c].ptr<float>(y + patch_rad, i);
								const float* tv = dataBorder[c].ptr<float>(y + patch_rad, j);
								for (int x = 0; x < cols; x++)
								{
									mcovv[idxv] += sv[x] * tv[x];
								}
							}
							idxv++;
						}
					}
				}
			}
		}
		else
		{
			if (isSIMD)
			{
#pragma omp parallel for schedule (static)
				for (int y = 0; y < rows; y++)
				{
					const int t = omp_get_thread_num();
					for (int i = 0, idxh = 0, idxv = 0; i < D; i++)
					{
						for (int j = i; j < D; j++)
						{
							for (int c1 = 0; c1 < channels; c1++)
							{
								const float* sh = dataBorder[c1].ptr<float>(y + i, patch_rad);
								for (int c2 = 0; c2 < channels; c2++)
								{
									const float* th = dataBorder[c2].ptr<float>(y + j, patch_rad);
									__m256 psumh = _mm256_setzero_ps();
									for (int x = 0; x < cols; x += 8)
									{
										psumh = _mm256_fmadd_ps(_mm256_loadu_ps(sh + x), _mm256_loadu_ps(th + x), psumh);

									}
									mtcovh[t][idxh] += _mm256_reduceadd_ps(psumh);
									idxh++;
								}
							}

							for (int c = 0; c < channels; c++)
							{
								const float* sv = dataBorder[c].ptr<float>(y + patch_rad, i);
								const float* tv = dataBorder[c].ptr<float>(y + patch_rad, j);
								__m256 psumv = _mm256_setzero_ps();
								for (int x = 0; x < cols; x += 8)
								{
									psumv = _mm256_fmadd_ps(_mm256_loadu_ps(sv + x), _mm256_loadu_ps(tv + x), psumv);
								}
								mtcovv[t][idxv] += _mm256_reduceadd_ps(psumv);
							}
							idxv++;
						}
					}
				}
			}
			else
			{
				for (int y = 0; y < rows; y++)
				{
					for (int i = 0, idxh = 0, idxv = 0; i < D; i++)
					{
						for (int j = i; j < D; j++)
						{
							for (int c1 = 0; c1 < channels; c1++)
							{
								const float* sh = dataBorder[c1].ptr<float>(y + i, patch_rad);
								for (int c2 = 0; c2 < channels; c2++)
								{
									const float* th = dataBorder[c2].ptr<float>(y + j, patch_rad);
									for (int x = 0; x < cols; x++)
									{
										mcovh[idxh] += sh[x] * th[x];
									}
									idxh++;
								}
							}

							for (int c = 0; c < channels; c++)
							{
								const float* sv = dataBorder[c].ptr<float>(y + patch_rad, i);
								const float* tv = dataBorder[c].ptr<float>(y + patch_rad, j);
								for (int x = 0; x < cols; x++)
								{
									mcovv[idxv] += sv[x] * tv[x];
								}
							}
							idxv++;
						}
					}
				}
			}
		}

		//reduction	
		//h
		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				for (int c1 = 0; c1 < channels; c1++)
				{
					for (int c2 = 0; c2 < channels; c2++)
					{
						for (int t = 0; t < threadMax; t++)
						{
							mcovh[idx] += mtcovh[t][idx];
						}
						idx++;
					}
				}
			}
		}
		//v
		for (int i = 0, idx = 0; i < D; i++)
		{
			for (int j = i; j < D; j++)
			{
				for (int t = 0; t < threadMax; t++)
				{
					mcovv[idx] += mtcovv[t][idx];
				}
				idx++;
			}
		}
	}

	//set covmat
	//h
	for (int i = 0, idx = 0; i < D; i++)
	{
		for (int j = i; j < D; j++)
		{
			for (int c1 = 0; c1 < channels; c1++)
			{
				for (int c2 = 0; c2 < channels; c2++)
				{
					covmat[0].at<double>(i * channels + c1, j * channels + c2) = covmat[0].at<double>(j * channels + c2, i * channels + c1) = mcovh[idx++] * Normfactor;
				}
			}
		}
	}
	//v
	for (int i = 0, idx = 0; i < D; i++)
	{
		for (int j = i; j < D; j++)
		{
			covmat[1].at<double>(i, j) = covmat[1].at<double>(j, i) = mcovv[idx++] * Normfactor;
		}
	}
}
#pragma endregion

#pragma region setter
void CalcPatchCovarMatrix::getScanorder(int* scan, const int step, const int channels, const bool isReverse)
{
	for (int j = 0, idx = 0; j < D; j++)
	{
		for (int i = 0; i < D; i++)
		{
			for (int c = 0; c < channels; c++)
			{
				if (!isReverse)  scan[idx] = channels * step * (j - patch_rad) + channels * (i - patch_rad) + c;
				else scan[idx] = channels * step * (patch_rad - j) + channels * (patch_rad - i) + c;
				idx++;
			}
		}
	}
}

void CalcPatchCovarMatrix::getScanorderBorder(int* scan, const int step, const int channels)
{
	for (int j = 0, idx = 0; j < D; j++)
	{
		for (int i = 0; i < D; i++)
		{
			for (int c = 0; c < channels; c++)
			{
				scan[idx] = channels * step * (j)+channels * i + c;
				idx++;
			}
		}
	}
}

void CalcPatchCovarMatrix::setCovHalf(Mat& dest, const vector<double>& covElem, const double normalSize)
{
	for (int y = 0, idx = 0; y < dim; y++)
	{
		for (int x = y; x < dim; x++)
		{
			dest.at<double>(y, x) = dest.at<double>(x, y) = covElem[idx++] * normalSize;
		}
	}
}

void CalcPatchCovarMatrix::setCovRep(const vector<double>& meanv, const vector<double>& varElem, Mat& dest, vector<double>& covElem, vector<vector<Point>>& covset, const double normalSize)
{
	if (color_channels == 1)
	{
		for (int i = 0; i < D * D; i++)
		{
			//covar.at<double>(i, i) = mulSum[i] * normalSize - meanv[0] * meanv[0];
			dest.at<double>(i, i) = varElem[0] * normalSize;
		}

		const int size = (int)covset.size();
		for (int k = 0; k < size; k++)
		{
			const double cov_val = covElem[k] * normalSize;
			for (int l = 0; l < covset[k].size(); l++)
			{
				Point pt = covset[k][l];
				dest.at<double>(pt) = cov_val;
			}
		}
	}
	else
	{
		//Mat db = Mat::zeros(Size(D * D * color_channels, D * D * color_channels), CV_8U);

		//diag
		for (int i = 0; i < D * D; i++)
		{
			for (int cy = 0, vidx = 0; cy < color_channels; cy++)
			{
				for (int cx = cy; cx < color_channels; cx++)
				{
					const double v = varElem[vidx++] * normalSize;
					const int x = i * color_channels + cx;
					const int y = i * color_channels + cy;
					const int x2 = i * color_channels + cy;
					const int y2 = i * color_channels + cx;
					dest.at<double>(y2, x2) = dest.at<double>(y, x) = v;
					//db.at<uchar>(y, x) = 255;
				}
			}
		}

		//dir
		const int dir_size = (int)covset.size();
		for (int k = 0, vidx = 0; k < dir_size; k++)
		{
			for (int cy = 0; cy < color_channels; cy++)
			{
				for (int cx = 0; cx < color_channels; cx++)
				{
					const double cov_val = covElem[vidx++] * normalSize;
					bool srcTrans = false;
					{
						//l=0;
						const int x = covset[k][0].x * color_channels + cy;
						const int y = covset[k][0].y * color_channels + cx;
						dest.at<double>(y, x) = cov_val;
						//dest.at<double>(y, x) = 0;
						if (covset[k][0].x < covset[k][0].y)
						{
							srcTrans = true;
						}

					}
					for (int l = 1; l < covset[k].size(); l++)
					{
						if (!srcTrans)
						{
							if (covset[k][l].x > covset[k][l].y)
							{
								const int x = covset[k][l].x * color_channels + cy;
								const int y = covset[k][l].y * color_channels + cx;
								//dest.at<double>(y, x) = 0;
								dest.at<double>(y, x) = cov_val;
								//db.at<uchar>(y, x) = 255;
							}
							else
							{
								const int x = covset[k][l].x * color_channels + cx;
								const int y = covset[k][l].y * color_channels + cy;
								dest.at<double>(y, x) = cov_val;
								//db.at<uchar>(y, x) = 255;
							}
						}
						else
						{
							if (covset[k][l].x < covset[k][l].y)
							{
								const int x = covset[k][l].x * color_channels + cy;
								const int y = covset[k][l].y * color_channels + cx;
								dest.at<double>(y, x) = cov_val;
								//db.at<uchar>(y, x) = 255;
							}
							else
							{
								const int x = covset[k][l].x * color_channels + cx;
								const int y = covset[k][l].y * color_channels + cy;
								dest.at<double>(y, x) = cov_val;
								//db.at<uchar>(y, x) = 255;
							}
						}
					}
				}
			}
			//cp::imshowResize("test", db, Size(), 3, 3); waitKey();
		}
	}
}
#pragma endregion

#pragma region naive
void CalcPatchCovarMatrix::naive(const vector<Mat>& src, Mat& cov, const int skip)
{
	vector<Mat> sub(color_channels);
	vector<double> meanv(color_channels);
	vector<double> var(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		double aa, vv;
		cp::average_variance(src[c], aa, vv);
		subtract(src[c], aa, sub[c]);
		meanv[c] = 0.0;
		var[c] = vv;
	}

	const int DD = D * D;
	AutoBuffer<int> scan(DD);
	getScanorder(scan, src[0].cols, 1);

	vector<double> sum(dim * dim);
	for (int i = 0; i < dim * dim; i++) sum[i] = 0.0;

	AutoBuffer<float> patch(dim);

	int count = 0;
	for (int j = patch_rad; j < src[0].rows - patch_rad; j += skip)
	{
		for (int i = patch_rad; i < src[0].rows - patch_rad; i += skip)
		{
			AutoBuffer<float*> sptr(color_channels);
			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] = sub[c].ptr<float>(j, i);
			}

			for (int k = 0, idx = 0; k < DD; k++)
			{
				for (int c = 0; c < color_channels; c++)
				{
					patch[idx++] = sptr[c][scan[k]];
				}
			}
			for (int k = 0, idx = 0; k < dim; k++)
			{
				for (int l = k + 1; l < dim; l++)
				{
					sum[idx++] += patch[k] * patch[l];
				}
			}
			count++;
		}
	}

	//setCovRepMeanOld(meanv, var, cov, sum, 1.0 / count);
}

void CalcPatchCovarMatrix::naive(const vector<Mat>& src, Mat& cov, Mat& mask)
{
	vector<Mat> sub(color_channels);
	vector<double> meanv(color_channels);
	vector<double> var(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		double aa, vv;
		cp::average_variance(src[c], aa, vv);
		subtract(src[c], aa, sub[c]);
		meanv[c] = 0.0;
		var[c] = vv;
	}

	const int DD = D * D;
	AutoBuffer<int> scan(DD);
	getScanorder(scan, src[0].cols, 1);

	vector<double> sum(dim * dim);
	for (int i = 0; i < dim * dim; i++) sum[i] = 0.0;

	AutoBuffer<float> patch(dim);

	int count = 0;
	for (int j = patch_rad; j < src[0].rows - patch_rad; j++)
	{
		for (int i = patch_rad; i < src[0].rows - patch_rad; i++)
		{
			if (mask.at<uchar>(j, i) == 0)continue;

			AutoBuffer<float*> sptr(color_channels);
			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] = sub[c].ptr<float>(j, i);
			}

			for (int k = 0, idx = 0; k < DD; k++)
			{
				for (int c = 0; c < color_channels; c++)
				{
					patch[idx++] = sptr[c][scan[k]];
				}
			}
			for (int k = 0, idx = 0; k < dim; k++)
			{
				for (int l = k + 1; l < dim; l++)
				{
					sum[idx++] += patch[k] * patch[l];
				}
			}
			count++;
		}
	}

	//setCovRepMeanOld(meanv, var, cov, sum, 1.0 / count);
}
#pragma endregion

#pragma region util
inline int getIndex(int j, int i, int step, int patch_rad)
{
	return step * (j - patch_rad) + (i - patch_rad);
}

inline int getComputeHalfCovNoDiagElementSize(const int dim)
{
	int ret = 0;
	for (int k = 0; k < dim; k++)
	{
		for (int l = k + 1; l < dim; l++)
		{
			ret++;
		}
	}
	return ret;
}

inline int getComputeHalfCovElementSize(const int dim)
{
	int ret = 0;
	for (int k = 0; k < dim; k++)
	{
		for (int l = k; l < dim; l++)
		{
			ret++;
		}
	}
	return ret;
}

void sub_const(InputArray src_, const double val, Mat& dest)
{
	Mat src = src_.getMat();
	if (src.depth() == CV_32F)
	{
		dest.create(src.size(), CV_32F);
		const int simdsize = get_simd_floor(src.size().area(), 32);
		const int S = simdsize / 32;
		float* sptr = src.ptr<float>();
		float* dptr = dest.ptr<float>();
		const __m256 msub = _mm256_set1_ps((float)val);
		for (int i = 0; i < S; i++)
		{
			_mm256_storeu_ps(dptr, _mm256_sub_ps(_mm256_loadu_ps(sptr), msub));
			_mm256_storeu_ps(dptr + 8, _mm256_sub_ps(_mm256_loadu_ps(sptr + 8), msub));
			_mm256_storeu_ps(dptr + 16, _mm256_sub_ps(_mm256_loadu_ps(sptr + 16), msub));
			_mm256_storeu_ps(dptr + 24, _mm256_sub_ps(_mm256_loadu_ps(sptr + 24), msub));
			sptr += 32;
			dptr += 32;
		}
		sptr = src.ptr<float>();
		dptr = dest.ptr<float>();
		for (int i = simdsize; i < src.size().area(); i++)
		{
			dptr[i] = sptr[i] - (float)val;
		}
	}
	else if (src.depth() == CV_64F)
	{
		dest.create(src.size(), CV_64F);
		const int simdsize = get_simd_floor(src.size().area(), 16);
		const int S = simdsize / 16;
		double* sptr = src.ptr<double>();
		double* dptr = dest.ptr<double>();
		const __m256d msub = _mm256_set1_pd(val);
		for (int i = 0; i < S; i++)
		{
			_mm256_storeu_pd(dptr, _mm256_sub_pd(_mm256_loadu_pd(sptr), msub));
			_mm256_storeu_pd(dptr + 4, _mm256_sub_pd(_mm256_loadu_pd(sptr + 4), msub));
			_mm256_storeu_pd(dptr + 8, _mm256_sub_pd(_mm256_loadu_pd(sptr + 8), msub));
			_mm256_storeu_pd(dptr + 12, _mm256_sub_pd(_mm256_loadu_pd(sptr + 12), msub));
			sptr += 16;
			dptr += 16;
		}
		sptr = src.ptr<double>();
		dptr = dest.ptr<double>();
		for (int i = simdsize; i < src.size().area(); i++)
		{
			dptr[i] = sptr[i] - val;
		}
	}
	else
	{
		cout << "no support type: sub_const " << cp::getDepthName(src.depth()) << endl;
	}
}

inline void _mm256_storeu_cvtps_pd(double* dest, __m256 src)
{
	_mm256_storeu_pd(dest + 0, _mm256_cvtps_pd(_mm256_castps256_ps128(src)));
	_mm256_storeu_pd(dest + 4, _mm256_cvtps_pd(_mm256_castps256hi_ps128(src)));
}

void sub_const32to64(InputArray src_, const float val, Mat& dest)
{
	Mat src = src_.getMat();
	CV_Assert(src.depth() == CV_32F);
	dest.create(src.size(), CV_64F);
	const int simdsize = get_simd_floor(src.size().area(), 32);
	const int S = simdsize / 32;
	float* sptr = src.ptr<float>();
	double* dptr = dest.ptr<double>();
	const __m256 msub = _mm256_set1_ps(val);
	for (int i = 0; i < S; i++)
	{
		_mm256_storeu_cvtps_pd(dptr, _mm256_sub_ps(_mm256_loadu_ps(sptr), msub));
		_mm256_storeu_cvtps_pd(dptr + 8, _mm256_sub_ps(_mm256_loadu_ps(sptr + 8), msub));
		_mm256_storeu_cvtps_pd(dptr + 16, _mm256_sub_ps(_mm256_loadu_ps(sptr + 16), msub));
		_mm256_storeu_cvtps_pd(dptr + 24, _mm256_sub_ps(_mm256_loadu_ps(sptr + 24), msub));
		sptr += 32;
		dptr += 32;
	}
	sptr = src.ptr<float>();
	dptr = dest.ptr<double>();
	for (int i = simdsize; i < src.size().area(); i++)
	{
		dptr[i] = double(sptr[i] - val);
	}
}

void averagesub_rect(InputArray src, const int patch_r, const int x, const int y, Mat& dest, const int borderType)
{
	Mat im; copyMakeBorder(src, im, patch_r, patch_r, patch_r, patch_r, borderType);
	Mat shift = im(Rect(x, y, src.size().width, src.size().height)).clone();
	double val = cp::average(shift);
	sub_const(shift, val, dest);
}

void sub_rect(InputArray src, const int patch_r, const int x, const int y, const float val, Mat& dest, const int borderType)
{
	Mat im; copyMakeBorder(src, im, patch_r, patch_r, patch_r, patch_r, borderType);
	Mat shift = im(Rect(x, y, src.size().width, src.size().height)).clone();
	sub_const(shift, val, dest);
}

void CalcPatchCovarMatrix::computeSepCov(const vector<Mat>& src, const int patch_rad, vector<Mat>& vcov, const DRIM2COLType method, const int skip, const bool isParallel) {

	if (!isSepCov(method, true) && !isSepCov(method, false))
	{
		cout << "call computeCov for " << getDRIM2COLName(method) << endl;
		return;
	}
	this->patch_rad = patch_rad;
	D = 2 * patch_rad + 1;
	color_channels = (int)src.size();
	dim = color_channels * D * D;
	//vector<Mat> data(color_channels);
	data.resize(color_channels);

	vcov.resize(2);

	if (method == DRIM2COLType::MEAN_SUB_SEPCOVX)
	{
		for (int c = 0; c < color_channels; c++)
		{
			sub_const(src[c], float(cp::average(src[c])), data[c]);
		}
		computeSeparateCov(data, patch_rad, border, vcov);
	}

	if (method == DRIM2COLType::MEAN_SUB_SEPCOVY)
	{
		//Mat temp;
		for (int c = 0; c < color_channels; c++)
		{
			//temp = src[c].t();
			//temp.copyTo(srct[c]);
			//transpose(src[c], srct[c]);
			sub_const(src[c].t(), float(cp::average(src[c])), data[c]);
		}
		computeSeparateCov(data, patch_rad, border, vcov);
	}

	if (method == DRIM2COLType::NO_SUB_SEPCOVX)
	{
		computeSeparateCov(src, patch_rad, border, vcov);
	}

	if (method == DRIM2COLType::NO_SUB_SEPCOVY)
	{
		vector<Mat> srct(src.size());
		Mat temp;
		for (int c = 0; c < src.size(); c++)
		{
			temp = src[c].t();
			temp.copyTo(srct[c]);
		}
		computeSeparateCov(srct, patch_rad, border, vcov);
	}

	if (method == DRIM2COLType::CONST_SUB_SEPCOVX)
	{
		for (int c = 0; c < color_channels; c++)
		{
			sub_const(src[c], const_sub, data[c]);
		}
		computeSeparateCov(data, patch_rad, border, vcov);
	}

	if (method == DRIM2COLType::CONST_SUB_SEPCOVY)
	{
		vector<Mat> srct(src.size());
		Mat temp;
		for (int c = 0; c < src.size(); c++)
		{
			//temp = src[c].t();
			//temp.copyTo(srct[c]);
			sub_const(src[c].t(), const_sub, data[c]);
		}
		computeSeparateCov(data, patch_rad, border, vcov);
	}

	if (method == DRIM2COLType::MEAN_SUB_SEPCOVXXt)
	{
		for (int c = 0; c < color_channels; c++)
		{
			sub_const(src[c], float(cp::average(src[c])), data[c]);
		}
		computeSeparateCovXXt(data, patch_rad, border, vcov);
	}

	if (method == DRIM2COLType::NO_SUB_SEPCOVXXt)
	{
		computeSeparateCovXXt(src, patch_rad, border, vcov);
	}

	if (method == DRIM2COLType::CONST_SUB_SEPCOVXXt)
	{
		for (int c = 0; c < color_channels; c++)
		{
			sub_const(src[c], const_sub, data[c]);
		}
		computeSeparateCovXXt(data, patch_rad, border, vcov);
	}
}

void CalcPatchCovarMatrix::computeSepCov(const Mat& src, const int patch_rad, vector<Mat>& cov, const DRIM2COLType method, const int skip, const bool isParallel)
{
	vector<Mat> vsrc(src.channels());
	if (src.channels() == 1) vsrc[0] = src;
	else split(src, vsrc);

	computeSepCov(vsrc, patch_rad, cov, method, skip, isParallel);
}


class RepresentiveCovarianceComputer
{
	struct RPCOV
	{
		Point direct;
		Point set;
		vector<Point> sharedElement;
		vector<Point> sharedElementIndex;
		RPCOV(Point direct, Point set, vector<Point>& sharedElement, vector<Point>& sharedElementIndex) : direct(direct), set(set), sharedElement(sharedElement), sharedElementIndex(sharedElementIndex)
		{
			;
		}
	};
	vector<RPCOV> list;
	int r;
	int channels;
	inline int get_num_direction(int r)
	{
		const int D = 2 * r + 1;
		//print_debug3((2 * D - 1) * (2 * D - 1) / 2, 2 * D * (D - 1) + 1);
		return 2 * D * (D - 1) + 1;
	}

	void computeAllDirection(const int r, vector<Point>& direction_list, bool isFixDirectionCenter)
	{
		if (isFixDirectionCenter)
		{
			int idx = 0;
			const int D = 2 * r + 1;

			direction_list.resize(D * D - 1);
			for (int i = 0; i < D; i++)
			{
				for (int j = 0; j < D; j++)
				{
					if (i == 0 && j == 0)continue;
					direction_list[idx].x = i;
					direction_list[idx].y = j;
					idx++;
				}
			}
		}
		else
		{
			int idx = 0;
			const int D = 2 * r + 1;
			direction_list.resize(get_num_direction(r));
			for (int j = 0; j < D; j++)
			{
				direction_list[idx].x = 0;
				direction_list[idx].y = j;
				idx++;
			}
			for (int i = 1; i < D; i++)
			{
				for (int j = -D + 1; j < D; j++)
				{
					direction_list[idx].x = i;
					direction_list[idx].y = j;
					idx++;
				}
			}
		}
	}

public:

	RepresentiveCovarianceComputer(const int channels, const int patch_rad, const bool isFixDirectionCenter = false) :r(patch_rad), channels(channels)
	{
		const int D = patch_rad * 2 + 1;
		const int DD = D * D;

		vector<Point> direction_list;
		computeAllDirection(patch_rad, direction_list, isFixDirectionCenter);
		if (isFixDirectionCenter)
		{
			for (const Point& dir : direction_list)
			{
				int counter = 0;
				Point set = dir;
				vector<Point> sharedElement;
				vector<Point> sharedElementIndex;

				for (int j = 0; j < D; j++)
				{
					for (int i = 0; i < D; i++)
					{
						if (i == 0 && j == 0)continue;
						Point a = Point(i, j) + dir;
						if (a.x < D && a.x >= 0 && a.y < D && a.y >= 0)
						{
							sharedElement.push_back(Point(j, i));
							sharedElementIndex.push_back(Point((j + dir.y) * D + (i + dir.x), j * D + i));
							sharedElementIndex.push_back(Point((j - dir.y) * D + i - dir.x, j * D + i));
						}
					}
				}
				list.push_back(RPCOV(dir, set, sharedElement, sharedElementIndex));
			}
		}
		else
		{
			//cp::Timer t;
			int idx = 0;
			for (const Point& dir : direction_list)
			{
				if (dir.x == 0 && dir.y == 0) continue;
				//cout << idx++ << endl;
				int counter = 0;
				Point set;
				vector<Point> sharedElement;
				vector<Point> sharedElementIndex;

				// plus
				int i_min = max(-dir.x, 0);
				int i_max = min(D - 1 - dir.x, D - 1);
				int j_min = max(-dir.y, 0);
				int j_max = min(D - 1 - dir.y, D - 1);

				for (int i = i_min; i <= i_max; i++)
				{
					for (int j = j_min; j <= j_max; j++)
					{
						if (counter == 0)
						{
							set = Point(i, j);
						}
						sharedElement.push_back(Point(j, i));
						sharedElementIndex.push_back(Point((j + dir.y) * D + (i + dir.x), j * D + i));
						counter++;
					}
				}

				//minus
				i_min = max(dir.x, 0);
				i_max = min(D - 1 + dir.x, D - 1);
				j_min = max(dir.y, 0);
				j_max = min(D - 1 + dir.y, D - 1);

				for (int i = i_min; i <= i_max; i++)
				{
					for (int j = j_min; j <= j_max; j++)
					{
						//sharedElement.push_back(Point(i, j));
						sharedElementIndex.push_back(Point((j - dir.y) * D + i - dir.x, j * D + i));
					}
				}

				list.push_back(RPCOV(dir, set, sharedElement, sharedElementIndex));
				//cout << "OK" << endl;
			}
		}
	}

	vector<vector<Point>> getSharedSet()
	{
		vector<vector<Point>> ret;
		for (int i = 0; i < list.size(); i++)
		{
			ret.push_back(list[i].sharedElementIndex);
		}

		return ret;
	}

	void getIndexFull(int* first, int* second)//for FullSub
	{
		const int D = r * 2 + 1;
		const int DD = D * D;
		const int step = D * channels;

		int idx = 0;
		{
			for (int cb = 0; cb < channels; cb++)
			{
				for (int ca = cb; ca < channels; ca++)
				{
					first[idx] = ca;
					second[idx] = cb;
					idx++;
				}
			}
		}
		for (int i = 0; i < list.size(); i++)
		{
			Point a = list[i].set;
			Point b = a + list[i].direct;
			for (int cb = 0; cb < channels; cb++)
			{
				for (int ca = 0; ca < channels; ca++)
				{
					first[idx] = a.y * step + channels * a.x + ca;
					second[idx] = b.y * step + channels * b.x + cb;
					idx++;
				}
			}
		}
	}

	void getIndex(int* first, int* second, const int width)
	{
		Point c = Point(r, r);
		for (int i = 0; i < list.size(); i++)
		{
			Point a = list[i].set - c;
			Point b = a + list[i].direct;

			first[i] = a.y * width + a.x;
			second[i] = b.y * width + b.x;
		}
	}

	void getIndexBorder(int* first, int* second, const int width)
	{
		for (int i = 0; i < list.size(); i++)
		{
			Point a = list[i].set;
			Point b = a + list[i].direct;

			first[i] = a.y * width + a.x;
			second[i] = b.y * width + b.x;
		}
	}

	void print()
	{
		int idx = 1;
		int size = (int)list.size();
		for (const RPCOV& rpc : list)
		{
			cout << idx++ << "/" << size << ": direct " << rpc.direct << endl;
			cout << ", 1st set: " << "," << rpc.set << endl;
			cout << "shared element : ";
			for (const Point& e : rpc.sharedElement)
			{
				cout << e << "-" << e + rpc.direct << ",";
			}
			cout << endl;
			cout << "shared index: ";
			for (const Point& e : rpc.sharedElementIndex)
			{
				cout << e << ",";
			}
			cout << endl << endl;
		}
	}

	int getNumberOfDirections()
	{
		return (int)list.size();
	}

	double getRatio()
	{
		const int D = (2 * r + 1) * (2 * r + 1);
		return(double)list.size() / (D * D);
	}
};
#pragma endregion

#pragma region full mean
//template<int color_channels, int dim>
void CalcPatchCovarMatrix::simdOMPCovFullCenterFullElement32F(const vector<Mat>& src_, Mat& cov, const int border)
{
	//cout << "simdOMPCovFull32F" << endl;
	const int simd_step = 8;
	const int DD = dim / color_channels;

	vector<Mat> data(dim);
	/*for (int y = 0, idx = 0; y < D; y++)
	{
		for (int x = 0; x < D; x++)
		{
			for (int c = 0; c < color_channels; c++)
			{
				sub_rect(src_[c], patch_rad, x, y, float(mean[c][y * D + x]), data[idx++], border);//full
			}
		}
	}*/
#pragma omp parallel for schedule(dynamic)
	for (int d = 0; d < dim; d++)
	{
		const int y = d / (D * color_channels);
		const int xc = d % (D * color_channels);
		const int x = xc / color_channels;
		const int c = xc % color_channels;
		averagesub_rect(src_[c], patch_rad, x, y, data[d], border);//full
	}

	const int thread_max = omp_get_max_threads();

	const int xst = 0;
	const int yst = 0;
	const int xend = data[0].cols - 0;
	const int yend = data[0].rows - 0;
	const int simd_end = get_simd_floor(xend - xst, simd_step);
	const __m256i mask = get_simd_residualmask_epi32(xend - xst);
	const bool isRem = ((xend - xst) == simd_end) ? false : true;

	const double normalSize = 1.0 / ((data[0].rows) * (xend - xst));
	//const double normalSize = 1.0;
	const int covElementSize = dim * dim;

	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * covElementSize * thread_max, AVX_ALIGN);
	for (int i = 0; i < covElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

#pragma omp parallel for SCHEDULE
	for (int y = yst; y < yend; y++)
	{
		const int tindex = omp_get_thread_num();
		__m256* mcov_local = &mbuff[covElementSize * tindex];
		AutoBuffer<__m256> msrc_local(dim);
		AutoBuffer<const float*> sptr(dim);
		for (int d = 0; d < dim; d++)
		{
			sptr[d] = data[d].ptr<float>(y);
		}

		for (int i = xst; i < simd_end; i += simd_step)
		{
			for (int d = 0; d < dim; d++)
			{
				msrc_local[d] = _mm256_loadu_ps(sptr[d]);
			}

			for (int y = 0, idx = 0; y < DD; y++)
			{
				for (int cy = 0; cy < color_channels; cy++)
				{
					const int Y = y * color_channels + cy;
					for (int x = 0; x < DD; x++)
					{
						for (int cx = 0; cx < color_channels; cx++)
						{
							const int X = x * color_channels + cx;
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[X], msrc_local[Y], mcov_local[idx]);
							idx++;
						}
					}
				}
			}

			for (int d = 0; d < dim; d++)
			{
				sptr[d] += simd_step;
			}
		}
		if (isRem)
		{
			for (int d = 0; d < dim; d++)
			{
				msrc_local[d] = _mm256_maskload_ps(sptr[d], mask);
			}

			for (int y = 0, idx = 0; y < DD; y++)
			{
				for (int cy = 0; cy < color_channels; cy++)
				{
					const int Y = y * color_channels + cy;
					for (int x = 0; x < DD; x++)
					{
						for (int cx = 0; cx < color_channels; cx++)
						{
							const int X = x * color_channels + cx;
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[X], msrc_local[Y], mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
	}

	//reduction
	AutoBuffer<double> covElem(covElementSize);
	for (int d = 0; d < covElementSize; d++) covElem[d] = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		__m256* mcov = &mbuff[covElementSize * t];
		for (int d = 0; d < covElementSize; d++)
		{
			covElem[d] += _mm256_reduceadd_pspd(mcov[d]);
		}
	}

	for (int y = 0, idx = 0; y < DD; y++)
	{
		for (int cy = 0; cy < color_channels; cy++)
		{
			for (int x = 0; x < DD; x++)
			{
				for (int cx = 0; cx < color_channels; cx++)
				{
					const double val = covElem[idx] * normalSize;
					cov.at<double>(y * color_channels + cy, x * color_channels + cx) = val;
					idx++;
				}
			}
		}
	}

	//util::vizCovMat("Full", cov);
	_mm_free(mbuff);
}

void CalcPatchCovarMatrix::simdOMPCovFullCenterHalfElement32F(const vector<Mat>& src_, Mat& cov, const int border)
{
	//cout << "simdOMPCovFull32F" << endl;
	const int simd_step = 8;
	const int DD = dim / color_channels;

	vector<Mat> data(dim);
	/*for (int y = 0, idx = 0; y < D; y++)
	{
		for (int x = 0; x < D; x++)
		{
			for (int c = 0; c < color_channels; c++)
			{
				sub_rect(src_[c], patch_rad, x, y, float(mean[c][y * D + x]), data[idx++], border);//full
			}
		}
	}*/
#pragma omp parallel for schedule(dynamic)
	for (int d = 0; d < dim; d++)
	{
		const int y = d / (D * color_channels);
		const int xc = d % (D * color_channels);
		const int x = xc / color_channels;
		const int c = xc % color_channels;
		averagesub_rect(src_[c], patch_rad, x, y, data[d], border);//full
	}

	const int thread_max = omp_get_max_threads();

	const int xst = 0;
	const int yst = 0;
	const int xend = data[0].cols - 0;
	const int yend = data[0].rows - 0;
	const int simd_end = get_simd_floor(xend - xst, simd_step);
	const __m256i mask = get_simd_residualmask_epi32(xend - xst);
	const bool isRem = ((xend - xst) == simd_end) ? false : true;

	const double normalSize = 1.0 / ((data[0].rows) * (xend - xst));
	//const double normalSize = 1.0;
	const int covElementSize = dim * dim;

	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * covElementSize * thread_max, AVX_ALIGN);
	for (int i = 0; i < covElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

#pragma omp parallel for SCHEDULE
	for (int y = yst; y < yend; y++)
	{
		const int tindex = omp_get_thread_num();
		__m256* mcov_local = &mbuff[covElementSize * tindex];
		AutoBuffer<__m256> msrc_local(dim);
		AutoBuffer<const float*> sptr(dim);
		for (int d = 0; d < dim; d++)
		{
			sptr[d] = data[d].ptr<float>(y);
		}

		for (int i = xst; i < simd_end; i += simd_step)
		{
			for (int d = 0; d < dim; d++)
			{
				msrc_local[d] = _mm256_loadu_ps(sptr[d]);
			}

			for (int y = 0; y < dim; y++)
			{
				for (int x = y; x < dim; x++)
				{
					mcov_local[dim * y + x] = _mm256_fmadd_ps(msrc_local[x], msrc_local[y], mcov_local[dim * y + x]);
				}
			}

			for (int d = 0; d < dim; d++)
			{
				sptr[d] += simd_step;
			}
		}
		if (isRem)
		{
			for (int d = 0; d < dim; d++)
			{
				msrc_local[d] = _mm256_maskload_ps(sptr[d], mask);
			}

			for (int y = 0; y < dim; y++)
			{
				for (int x = y; x < dim; x++)
				{
					mcov_local[dim * y + x] = _mm256_fmadd_ps(msrc_local[x], msrc_local[y], mcov_local[dim * y + x]);
				}
			}
		}
	}

	//reduction
	AutoBuffer<double> covElem(covElementSize);
	for (int i = 0; i < covElementSize; i++) covElem[i] = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		__m256* mvar = &mbuff[covElementSize * t];
		for (int d = 0; d < covElementSize; d++)
		{
			covElem[d] += _mm256_reduceadd_pspd(mvar[d]);
		}
	}

	for (int y = 0, idx = 0; y < DD; y++)
	{
		for (int cy = 0; cy < color_channels; cy++)
		{
			for (int x = 0; x < DD; x++)
			{
				for (int cx = 0; cx < color_channels; cx++)
				{
					const double val = covElem[idx] * normalSize;
					cov.at<double>(y * color_channels + cy, x * color_channels + cx) = val;
					idx++;
				}
			}
		}
	}
	//copy for lower triangular matrix
	for (int y = 0; y < dim; y++)
	{
		for (int x = y; x < dim; x++)
		{
			cov.at<double>(x, y) = cov.at<double>(y, x);
		}
	}

	_mm_free(mbuff);
}

void CalcPatchCovarMatrix::simdOMPCovFullCenterRepElement32F(const vector<Mat>& src_, Mat& cov, const int border)
{
	//cout << "simdOMPCovFull32F" << endl;
	RepresentiveCovarianceComputer rcc(color_channels, patch_rad);
	vector<vector<Point>> ss = rcc.getSharedSet();
	const int directionElementSize = (int)ss.size();
	const int computeElementSize = (directionElementSize)*color_channels * color_channels + sum_from(color_channels);
	AutoBuffer<int> first(computeElementSize);
	AutoBuffer<int> second(computeElementSize);
	rcc.getIndexFull(first, second);

	const int simd_step = 8;
	const int DD = dim / color_channels;

	vector<Mat> data(dim);
	/*for (int y = 0, idx = 0; y < D; y++)
	{
		for (int x = 0; x < D; x++)
		{
			for (int c = 0; c < color_channels; c++)
			{
				sub_rect(src_[c], patch_rad, x, y, float(mean[c][y * D + x]), data[idx++], border);//full
			}
		}
	}*/
#pragma omp parallel for schedule(dynamic)
	for (int d = 0; d < dim; d++)
	{
		const int y = d / (D * color_channels);
		const int xc = d % (D * color_channels);
		const int x = xc / color_channels;
		const int c = xc % color_channels;
		averagesub_rect(src_[c], patch_rad, x, y, data[d], border);//full
	}


	const int thread_max = omp_get_max_threads();

	const int xst = 0;
	const int yst = 0;
	const int xend = data[0].cols - 0;
	const int yend = data[0].rows - 0;
	const int simd_end = get_simd_floor(xend - xst, simd_step);
	const __m256i mask = get_simd_residualmask_epi32(xend - xst);
	const bool isRem = ((xend - xst) == simd_end) ? false : true;

	const double normalSize = 1.0 / ((data[0].rows) * (xend - xst));

	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * computeElementSize * thread_max, AVX_ALIGN);
	for (int i = 0; i < computeElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

#pragma omp parallel for SCHEDULE
	for (int y = yst; y < yend; y++)
	{
		const int tindex = omp_get_thread_num();
		__m256* mcov_local = &mbuff[computeElementSize * tindex];
		AutoBuffer<__m256> msrc_local(dim);
		AutoBuffer<const float*> sptr(dim);
		for (int d = 0; d < dim; d++)
		{
			sptr[d] = data[d].ptr<float>(y);
		}

		for (int i = xst; i < simd_end; i += simd_step)
		{
			for (int d = 0; d < dim; d++)
			{
				msrc_local[d] = _mm256_loadu_ps(sptr[d]);
			}

			for (int d = 0; d < computeElementSize; d++)
			{
				mcov_local[d] = _mm256_fmadd_ps(msrc_local[first[d]], msrc_local[second[d]], mcov_local[d]);
			}

			for (int d = 0; d < dim; d++)
			{
				sptr[d] += simd_step;
			}
		}
		if (isRem)
		{
			for (int d = 0; d < dim; d++)
			{
				msrc_local[d] = _mm256_maskload_ps(sptr[d], mask);
			}

			for (int d = 0; d < computeElementSize; d++)
			{
				mcov_local[d] = _mm256_fmadd_ps(msrc_local[first[d]], msrc_local[second[d]], mcov_local[d]);
			}
		}
	}

	//reduction
	const int varElementSize = sum_from(color_channels);
	const int covElementSize = directionElementSize * color_channels * color_channels;
	vector<double> varElem(varElementSize);
	vector<double> covElem(covElementSize);
	for (int i = 0; i < varElementSize; i++) varElem[i] = 0.0;
	for (int i = 0; i < covElementSize; i++) covElem[i] = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		__m256* mvar = &mbuff[computeElementSize * t];
		__m256* mcov = &mbuff[computeElementSize * t + varElementSize];
		for (int d = 0; d < varElementSize; d++)
		{
			varElem[d] += _mm256_reduceadd_pspd(mvar[d]);
		}
		for (int d = 0; d < covElementSize; d++)
		{
			covElem[d] += _mm256_reduceadd_pspd(mcov[d]);
		}
	}

	vector <double> mm(color_channels);
	for (int c = 0; c < color_channels; c++)mm[c] = 0.0;
	setCovRep(mm, varElem, cov, covElem, ss, normalSize);

	_mm_free(mbuff);
}

void CalcPatchCovarMatrix::simdOMPCovFullCenterHalfElementTEST32F(const vector<Mat>& src_, Mat& cov, const int border)
{
	//cout << "simdOMPCovFull32F" << endl;
	const int simd_step = 8;
	const int DD = dim / color_channels;

	vector<Mat> data(dim);
	/*for (int y = 0, idx = 0; y < D; y++)
	{
		for (int x = 0; x < D; x++)
		{
			for (int c = 0; c < color_channels; c++)
			{
				sub_rect(src_[c], patch_rad, x, y, float(mean[c][y * D + x]), data[idx++], border);//full
			}
		}
	}*/
	vector<double> ave(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		ave[c] = cp::average(src_[c]);
	}
#pragma omp parallel for schedule(dynamic)
	for (int d = 0; d < dim; d++)
	{
		const int y = d / (D * color_channels);
		const int xc = d % (D * color_channels);
		const int x = xc / color_channels;
		const int c = xc % color_channels;
		//sub_rect(src_[c], patch_rad, x, y, float(ave[c]), data[d], border);//mean
		sub_rect(src_[c], patch_rad, x, y, (float)const_sub, data[d], border);//const
		//sub_rect(src_[c], patch_rad, x, y, 0.f, data[d], border);//nosub
		//averagesub_rect(src_[c], patch_rad, x, y, data[d], border);//full
	}

	const int thread_max = omp_get_max_threads();

	const int xst = 0;
	const int yst = 0;
	const int xend = data[0].cols - 0;
	const int yend = data[0].rows - 0;
	const int simd_end = get_simd_floor(xend - xst, simd_step);
	const __m256i mask = get_simd_residualmask_epi32(xend - xst);
	const bool isRem = ((xend - xst) == simd_end) ? false : true;

	const double normalSize = 1.0 / ((data[0].rows) * (xend - xst));
	//const double normalSize = 1.0;
	const int covElementSize = dim * dim;

	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * covElementSize * thread_max, AVX_ALIGN);
	for (int i = 0; i < covElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

#pragma omp parallel for SCHEDULE
	for (int y = yst; y < yend; y++)
	{
		const int tindex = omp_get_thread_num();
		__m256* mcov_local = &mbuff[covElementSize * tindex];
		AutoBuffer<__m256> msrc_local(dim);
		AutoBuffer<const float*> sptr(dim);
		for (int d = 0; d < dim; d++)
		{
			sptr[d] = data[d].ptr<float>(y);
		}

		for (int i = xst; i < simd_end; i += simd_step)
		{
			for (int d = 0; d < dim; d++)
			{
				msrc_local[d] = _mm256_loadu_ps(sptr[d]);
			}

			for (int y = 0; y < dim; y++)
			{
				for (int x = y; x < dim; x++)
				{
					mcov_local[dim * y + x] = _mm256_fmadd_ps(msrc_local[x], msrc_local[y], mcov_local[dim * y + x]);
				}
			}

			for (int d = 0; d < dim; d++)
			{
				sptr[d] += simd_step;
			}
		}
		if (isRem)
		{
			for (int d = 0; d < dim; d++)
			{
				msrc_local[d] = _mm256_maskload_ps(sptr[d], mask);
			}

			for (int y = 0; y < dim; y++)
			{
				for (int x = y; x < dim; x++)
				{
					mcov_local[dim * y + x] = _mm256_fmadd_ps(msrc_local[x], msrc_local[y], mcov_local[dim * y + x]);
				}
			}
		}
	}

	//reduction
	AutoBuffer<double> covElem(covElementSize);
	for (int i = 0; i < covElementSize; i++) covElem[i] = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		__m256* mvar = &mbuff[covElementSize * t];
		for (int d = 0; d < covElementSize; d++)
		{
			covElem[d] += _mm256_reduceadd_pspd(mvar[d]);
		}
	}

	for (int y = 0, idx = 0; y < DD; y++)
	{
		for (int cy = 0; cy < color_channels; cy++)
		{
			for (int x = 0; x < DD; x++)
			{
				for (int cx = 0; cx < color_channels; cx++)
				{
					const double val = covElem[idx] * normalSize;
					cov.at<double>(y * color_channels + cy, x * color_channels + cx) = val;
					idx++;
				}
			}
		}
	}
	//copy for lower triangular matrix
	for (int y = 0; y < dim; y++)
	{
		for (int x = y; x < dim; x++)
		{
			cov.at<double>(x, y) = cov.at<double>(y, x);
		}
	}

	//debug process for covariance matrix is here! 
	const bool isDebug = true;
	if (isDebug)
	{
		const bool isDir = false;//else direction Copy
		const bool isVar = true;//else Diag Variance Copy
		Mat ref = cov.clone();
		if (isVar)
		{
			vector<vector<Point>> covset(1);
			for (int d = 0; d < DD; d++)covset[0].push_back(Point(d, d));

			//util::vizCovMat("Rep", cov);
			const int rep = 0;
			//const int rep = DD / 2;//representive cov is half position
			for (int d = 0; d < DD; d++)
			{
				for (int cy = 0; cy < color_channels; cy++)
				{
					for (int cx = 0; cx < color_channels; cx++)
					{
						//cout << cy << "," << cx << ":" << cov.at<double>(d * color_channels + cy, d * color_channels + cx) << endl;
						//cout <<cy<<","<<cx<<":"<< cov.at<double>(d * color_channels + cy, d * color_channels + cx) - cov.at<double>(rep * color_channels + cy, rep * color_channels + cx) << endl;
						cov.at<double>(d * color_channels + cy, d * color_channels + cx) = cov.at<double>(rep * color_channels + cy, rep * color_channels + cx);
					}
				}
			}
			//util::vizCovMat("Full-   (var)", ref, color_channels);
			//util::vizCovMat("Full-Rep(var)", cov, color_channels);
			//testCovMat("test(var)", ref, cov, covset, false, );
		}

		if (isDir)
		{
			RepresentiveCovarianceComputer rcc(color_channels, patch_rad);
			vector<vector<Point>> covset = rcc.getSharedSet();
			//under debug
			const int dir_size = (int)covset.size();
			for (int k = 0; k < dir_size; k++)
			{
				Mat cov_val(color_channels, color_channels, CV_64F);
				for (int cy = 0; cy < color_channels; cy++)
				{
					for (int cx = 0; cx < color_channels; cx++)
					{
						const int x = covset[k][0].x * color_channels + cx;
						const int y = covset[k][0].y * color_channels + cy;
						if (covset[k][0].x > covset[k][0].y)
						{
							cov_val.at<double>(cy, cx) = cov.at<double>(y, x);
						}
						else
						{
							cov_val.at<double>(cx, cy) = cov.at<double>(y, x);
						}
					}
				}

				for (int cy = 0; cy < color_channels; cy++)
				{
					for (int cx = 0; cx < color_channels; cx++)
					{
						for (int l = 1; l < covset[k].size(); l++)
						{
							if (covset[k][l].x > covset[k][l].y)
							{
								const int x = covset[k][l].x * color_channels + cx;
								const int y = covset[k][l].y * color_channels + cy;
								cov.at<double>(y, x) = cov_val.at<double>(cy, cx);
							}
							else
							{
								const int x = covset[k][l].x * color_channels + cx;
								const int y = covset[k][l].y * color_channels + cy;
								cov.at<double>(y, x) = cov_val.at<double>(cx, cy);
							}
						}
					}
				}
				//cp::imshowResize("test", db, Size(), 3, 3); waitKey();
			}
			//Mat a = ref - cov;
			//util::vizCovMat("Full-   (var)", ref, color_channels);
			//util::vizCovMat("Full-Rep(var)", cov, color_channels);
			//testCovMat("Full-Rep(all)", ref, cov, covset, false, color_channels);
		}
	}

	_mm_free(mbuff);
}
#pragma endregion

#pragma region dir_conv_FFT

vector<Point> dir2cov(const int dx, const int dy, const int D)
{
	vector<Point> ret;
	const int ymin = max(0 - dy, 0);
	const int xmin = max(0 - dx, 0);
	const int ymax = min(D - dy, D);
	const int xmax = min(D - dx, D);
	for (int y = ymin; y < ymax; y++)
	{
		for (int x = xmin; x < xmax; x++)
		{
			int j = y + dy;
			int i = x + dx;
			ret.push_back(Point(D * y + x, D * j + i));
		}
	}
	return ret;
}

vector<Point> dir2covQuad(const int dx, const int dy, const int D)
{
	vector<Point> ret;
	/*
	const int ymin = dy;
	const int xmin = dx;
	const int ymax = D;
	const int xmax = D;
	for (int y = ymin; y < ymax; y++)
	{
		for (int x = xmin; x < xmax; x++)
		{
			int j = y + dy;
			int i = x + dx;
			ret.push_back(Point(D * y + x, D * j + i));
			j = y + dy;
			i = x - dx;
			ret.push_back(Point(D * y + x, D * j + i));

			j = y - dy;
			i = x + dx;
			ret.push_back(Point(D * y + x, D * j + i));

			j = y - dy;
			i = x - dx;
			ret.push_back(Point(D * y + x, D * j + i));
		}
	}*/


	for (int y = 0; y < D; y++)
	{
		for (int x = 0; x < D; x++)
		{
			int j = y + dy;
			if (j >= 0 && j < D)
			{
				int i = x + dx;
				if (i >= 0 && i < D)
				{
					ret.push_back(Point(D * y + x, D * j + i));
				}
				i = x - dx;
				if (i >= 0 && i < D)
				{
					ret.push_back(Point(D * y + x, D * j + i));
				}
			}
			j = y - dy;
			if (j >= 0 && j < D)
			{
				int i = x + dx;
				if (i >= 0 && i < D)
				{
					ret.push_back(Point(D * y + x, D * j + i));
				}
				i = x - dx;
				if (i >= 0 && i < D)
				{
					ret.push_back(Point(D * y + x, D * j + i));
				}
			}
		}
	}
	return ret;
}

template<int color_channels, int patch_rad>
void CalcPatchCovarMatrix::simdOMPCov_RepCenterConvElement32F(const vector<Mat>& src_, Mat& cov, const CenterMethod method, const float constant_sub, const int border)
{
	//cout << "simdOMPCov_RepCenterConvElement32F" << endl;
	const int simd_step = 8;
	const int DD = dim / color_channels;

	constexpr int D = 2 * patch_rad + 1;

	vector<Mat> data(color_channels);
	dataBorder.resize(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			//cout << "Conv mean" << endl; 
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
		}
		else if (method == CenterMethod::CONST_)
		{
			//cout << "Conv const: " <<const_sub<< endl;
			//subtract(src_[c], constant_sub, data[c]);
			sub_const(src_[c], constant_sub, data[c]);
		}
		else if (method == CenterMethod::NO)
		{
			//cout << "Conv nosub" << endl;
			//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			//else meanForCov[0] = 0.0;
			data[c] = src_[c];
		}
		else
		{
			cout << int(method) << endl;
			cout << "No support method (simdOMPCovRepCov32F)" << endl;
		}

		{
			copyMakeBorder(data[c], dataBorder[c], patch_rad, patch_rad, patch_rad, patch_rad, border);
			//copyMakeBorder(data[c], dataBorder[c], 0, 2 * patch_rad, 0, 2*patch_rad, border);
		}
	}

	const int thread_max = omp_get_max_threads();

	const int xst = 0;
	const int yst = 0;
	const int xend = data[0].cols - 0;
	const int yend = data[0].rows - 0;
	const int simd_end = get_simd_floor(xend - xst, simd_step);
	const __m256i mask = get_simd_residualmask_epi32(xend - xst);
	const bool isRem = ((xend - xst) == simd_end) ? false : true;

	const double normalSize = 1.0 / ((data[0].rows) * (xend - xst));
	constexpr int covElementSize = D * D * color_channels * color_channels;
	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * covElementSize * thread_max, AVX_ALIGN);

	//const int center = patch_rad * dataBorder[0].cols + patch_rad;
	const int center = 0;

	for (int i = 0; i < covElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

	if (color_channels == 1)
	{
#pragma omp parallel for SCHEDULE
		for (int y = yst; y < yend; y++)
		{
			const int tindex = omp_get_thread_num();
			__m256* mcov_local = &mbuff[covElementSize * tindex];
			const float* sptr = dataBorder[0].ptr<float>(y);

			for (int i = xst; i < simd_end; i += simd_step)
			{
				const __m256 mc = _mm256_loadu_ps(sptr + center);

				for (int y = 0, idx = 0; y < D; y++)
				{
					const float* ref = sptr + y * dataBorder[0].cols;
					for (int x = 0; x < D; x++)
					{
						mcov_local[idx] = _mm256_fmadd_ps(mc, _mm256_loadu_ps(ref + x), mcov_local[idx]);
						idx++;
					}
				}
				sptr += simd_step;
			}
			if (isRem)
			{
				const __m256 mc = _mm256_maskload_ps(sptr + center, mask);
				for (int y = 0, idx = 0; y < D; y++)
				{
					const float* ref = sptr + y * dataBorder[0].cols;
					for (int x = 0; x < D; x++)
					{
						mcov_local[idx] = _mm256_fmadd_ps(mc, _mm256_maskload_ps(ref + x, mask), mcov_local[idx]);
						idx++;
					}
				}
			}
		}
	}
	else if (color_channels == 3)
	{
#pragma omp parallel for SCHEDULE
		for (int y = yst; y < yend; y++)
		{
			const int tindex = omp_get_thread_num();
			__m256* mcov_local = &mbuff[covElementSize * tindex];
			AutoBuffer<const float*> sptr(3);
			sptr[0] = dataBorder[0].ptr<float>(y);
			sptr[1] = dataBorder[1].ptr<float>(y);
			sptr[2] = dataBorder[2].ptr<float>(y);

			for (int i = xst; i < simd_end; i += simd_step)
			{
				int idx = 0;
				const __m256 mxv0 = _mm256_loadu_ps(sptr[0] + center);
				const __m256 mxv1 = _mm256_loadu_ps(sptr[1] + center);
				const __m256 mxv2 = _mm256_loadu_ps(sptr[2] + center);
				for (int cy = 0; cy < 3; cy++)
				{
					for (int y = 0; y < D; y++)
					{
						const float* ref = sptr[cy] + y * dataBorder[0].cols;
						for (int x = 0; x < D; x++)
						{
							const __m256 myv = _mm256_loadu_ps(ref + x);
							mcov_local[idx + 0] = _mm256_fmadd_ps(mxv0, myv, mcov_local[idx + 0]);
							mcov_local[idx + 1] = _mm256_fmadd_ps(mxv1, myv, mcov_local[idx + 1]);
							mcov_local[idx + 2] = _mm256_fmadd_ps(mxv2, myv, mcov_local[idx + 2]);
							idx += 3;
						}
					}
				}

				sptr[0] += simd_step;
				sptr[1] += simd_step;
				sptr[2] += simd_step;
			}
			if (isRem)
			{
				int idx = 0;
				const __m256 mc0 = _mm256_maskload_ps(sptr[0] + center, mask);
				const __m256 mc1 = _mm256_maskload_ps(sptr[1] + center, mask);
				const __m256 mc2 = _mm256_maskload_ps(sptr[2] + center, mask);
				for (int cy = 0; cy < 3; cy++)
				{
					for (int y = 0; y < D; y++)
					{
						const float* ref = sptr[cy] + y * dataBorder[0].cols;
						for (int x = 0; x < D; x++)
						{
							const __m256 yv = _mm256_maskload_ps(ref + x, mask);
							mcov_local[idx + 0] = _mm256_fmadd_ps(mc0, yv, mcov_local[idx + 0]);
							mcov_local[idx + 1] = _mm256_fmadd_ps(mc1, yv, mcov_local[idx + 1]);
							mcov_local[idx + 2] = _mm256_fmadd_ps(mc2, yv, mcov_local[idx + 2]);
							idx += 3;
						}
					}
				}
			}
		}
	}
	else if (color_channels == 6)
	{
#pragma omp parallel for SCHEDULE
		for (int y = yst; y < yend; y++)
		{
			const int tindex = omp_get_thread_num();
			__m256* mcov_local = &mbuff[covElementSize * tindex];
			AutoBuffer<const float*> sptr(6);
			sptr[0] = dataBorder[0].ptr<float>(y);
			sptr[1] = dataBorder[1].ptr<float>(y);
			sptr[2] = dataBorder[2].ptr<float>(y);
			sptr[3] = dataBorder[3].ptr<float>(y);
			sptr[4] = dataBorder[4].ptr<float>(y);
			sptr[5] = dataBorder[5].ptr<float>(y);

			for (int i = xst; i < simd_end; i += simd_step)
			{
				int idx = 0;
				const __m256 mxv0 = _mm256_loadu_ps(sptr[0] + center);
				const __m256 mxv1 = _mm256_loadu_ps(sptr[1] + center);
				const __m256 mxv2 = _mm256_loadu_ps(sptr[2] + center);
				const __m256 mxv3 = _mm256_loadu_ps(sptr[3] + center);
				const __m256 mxv4 = _mm256_loadu_ps(sptr[4] + center);
				const __m256 mxv5 = _mm256_loadu_ps(sptr[5] + center);
				for (int cy = 0; cy < 6; cy++)
				{
					for (int y = 0; y < D; y++)
					{
						const float* ref = sptr[cy] + y * dataBorder[0].cols;
						for (int x = 0; x < D; x++)
						{
							const __m256 myv = _mm256_loadu_ps(ref + x);
							mcov_local[idx + 0] = _mm256_fmadd_ps(mxv0, myv, mcov_local[idx + 0]);
							mcov_local[idx + 1] = _mm256_fmadd_ps(mxv1, myv, mcov_local[idx + 1]);
							mcov_local[idx + 2] = _mm256_fmadd_ps(mxv2, myv, mcov_local[idx + 2]);
							mcov_local[idx + 3] = _mm256_fmadd_ps(mxv3, myv, mcov_local[idx + 3]);
							mcov_local[idx + 4] = _mm256_fmadd_ps(mxv4, myv, mcov_local[idx + 4]);
							mcov_local[idx + 5] = _mm256_fmadd_ps(mxv5, myv, mcov_local[idx + 5]);
							idx += 6;
						}
					}
				}

				sptr[0] += simd_step;
				sptr[1] += simd_step;
				sptr[2] += simd_step;
				sptr[3] += simd_step;
				sptr[4] += simd_step;
				sptr[5] += simd_step;
			}
			if (isRem)
			{
				int idx = 0;
				const __m256 mc0 = _mm256_maskload_ps(sptr[0] + center, mask);
				const __m256 mc1 = _mm256_maskload_ps(sptr[1] + center, mask);
				const __m256 mc2 = _mm256_maskload_ps(sptr[2] + center, mask);
				const __m256 mc3 = _mm256_maskload_ps(sptr[3] + center, mask);
				const __m256 mc4 = _mm256_maskload_ps(sptr[4] + center, mask);
				const __m256 mc5 = _mm256_maskload_ps(sptr[5] + center, mask);
				for (int cy = 0; cy < 6; cy++)
				{
					for (int y = 0; y < D; y++)
					{
						const float* ref = sptr[cy] + y * dataBorder[0].cols;
						for (int x = 0; x < D; x++)
						{
							const __m256 yv = _mm256_maskload_ps(ref + x, mask);
							mcov_local[idx + 0] = _mm256_fmadd_ps(mc0, yv, mcov_local[idx + 0]);
							mcov_local[idx + 1] = _mm256_fmadd_ps(mc1, yv, mcov_local[idx + 1]);
							mcov_local[idx + 2] = _mm256_fmadd_ps(mc2, yv, mcov_local[idx + 2]);
							mcov_local[idx + 3] = _mm256_fmadd_ps(mc3, yv, mcov_local[idx + 3]);
							mcov_local[idx + 4] = _mm256_fmadd_ps(mc4, yv, mcov_local[idx + 4]);
							mcov_local[idx + 5] = _mm256_fmadd_ps(mc5, yv, mcov_local[idx + 5]);
							idx += 6;
						}
					}
				}
			}
		}
	}
	else
	{
#pragma omp parallel for SCHEDULE
		for (int y = yst; y < yend; y++)
		{
			const int tindex = omp_get_thread_num();
			__m256* mcov_local = &mbuff[covElementSize * tindex];
			AutoBuffer<const float*> sptr(color_channels);
			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] = dataBorder[c].ptr<float>(y);
			}
			AutoBuffer<__m256> mxv(color_channels);
			for (int i = xst; i < simd_end; i += simd_step)
			{
				for (int cx = 0; cx < color_channels; cx++)
				{
					mxv[cx] = _mm256_loadu_ps(sptr[cx] + center);
				}
				int idx = 0;
				for (int cy = 0; cy < color_channels; cy++)
				{
					for (int y = 0; y < D; y++)
					{
						const float* ref = sptr[cy] + y * dataBorder[0].cols;
						for (int x = 0; x < D; x++)
						{
							const __m256 myv = _mm256_loadu_ps(ref + x);
							for (int cx = 0; cx < color_channels; cx++)
							{
								mcov_local[idx] = _mm256_fmadd_ps(mxv[cx], myv, mcov_local[idx]);
								idx++;
							}
						}
					}
				}

				for (int c = 0; c < color_channels; c++)
				{
					sptr[c] += simd_step;
				}
			}
			if (isRem)
			{
				for (int cx = 0; cx < color_channels; cx++)
				{
					mxv[cx] = _mm256_maskload_ps(sptr[cx] + center, mask);
				}
				int idx = 0;
				for (int cy = 0, idx = 0; cy < color_channels; cy++)
				{
					for (int y = 0; y < D; y++)
					{
						const float* ref = sptr[cy] + y * dataBorder[0].cols;
						for (int x = 0; x < D; x++)
						{
							const __m256 myv = _mm256_maskload_ps(ref + x, mask);
							for (int cx = 0; cx < color_channels; cx++)
							{
								mcov_local[idx] = _mm256_fmadd_ps(mxv[cx], myv, mcov_local[idx]);
								idx++;
							}
						}
					}
				}
			}
		}
	}

	//reduction
	AutoBuffer<double> covElem(covElementSize);
	for (int i = 0; i < covElementSize; i++) covElem[i] = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		__m256* mvar = &mbuff[covElementSize * t];
		for (int d = 0; d < covElementSize; d++)
		{
			covElem[d] += _mm256_reduceadd_pspd(mvar[d]);
		}
	}

	for (int cy = 0, idx = 0; cy < color_channels; cy++)
	{
		for (int y = 0; y < D; y++)
		{
			for (int x = 0; x < D; x++)
			{
				vector<Point> pt = dir2covQuad(x, y, D);
				for (int cx = 0; cx < color_channels; cx++)
				{
					const double val = covElem[idx] * normalSize;
					for (int d = 0; d < pt.size(); d++)
					{
						cov.at<double>(pt[d].y * color_channels + cy, pt[d].x * color_channels + cx) = val;
					}
					idx++;
				}
			}
		}
	}

	_mm_free(mbuff);
}

void CalcPatchCovarMatrix::simdOMPCov_RepCenterConvElement32FCn(const vector<Mat>& src_, Mat& cov, const CenterMethod method, const float constant_sub, const int border)
{
	//cout << "simdOMPCovFull32F" << endl;
	const int simd_step = 8;
	const int DD = dim / color_channels;

	const int D = 2 * patch_rad + 1;
	const int R = 2 * D - 1;
	const int Rh = R / 2;

	vector<Mat> data(color_channels);
	dataBorder.resize(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			//cout << "Rep mean" << endl; 
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
		}
		else if (method == CenterMethod::CONST_)
		{
			//cout << "Rep const" << endl;
			//subtract(src_[c], constant_sub, data[c]);
			sub_const(src_[c], constant_sub, data[c]);
		}
		else if (method == CenterMethod::NO)
		{
			//cout << "Rep nosub" << endl;
			//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			//else meanForCov[0] = 0.0;
			data[c] = src_[c];
		}
		else
		{
			cout << int(method) << endl;
			cout << "No support method (simdOMPCovRepCov32F)" << endl;
		}

		{
			copyMakeBorder(data[c], dataBorder[c], Rh, Rh, Rh, Rh, border);
		}
	}

	const int thread_max = omp_get_max_threads();

	const int xst = 0;
	const int yst = 0;
	const int xend = data[0].cols - 0;
	const int yend = data[0].rows - 0;
	const int simd_end = get_simd_floor(xend - xst, simd_step);
	const __m256i mask = get_simd_residualmask_epi32(xend - xst);
	const bool isRem = ((xend - xst) == simd_end) ? false : true;

	const double normalSize = 1.0 / ((data[0].rows) * (xend - xst));
	//const double normalSize = 1.0;
	const int covElementSize = R * R * color_channels * color_channels;
	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * covElementSize * thread_max, AVX_ALIGN);

	const int center = Rh * dataBorder[0].cols + Rh;

	for (int i = 0; i < covElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

#pragma omp parallel for SCHEDULE
	for (int y = yst; y < yend; y++)
	{
		const int tindex = omp_get_thread_num();
		__m256* mcov_local = &mbuff[covElementSize * tindex];
		AutoBuffer<__m256> msrc_local(dim);
		AutoBuffer<const float*> sptr(color_channels);
		for (int c = 0; c < color_channels; c++)
		{
			sptr[c] = dataBorder[c].ptr<float>(y);
		}

		for (int i = xst; i < simd_end; i += simd_step)
		{
			for (int cx = 0, idx = 0; cx < color_channels; cx++)
			{
				const __m256 mc = _mm256_loadu_ps(sptr[cx] + center);
				for (int cy = 0; cy < color_channels; cy++)
				{
					for (int y = 0; y < R; y++)
					{
						for (int x = 0; x < R; x++)
						{
							const int pos = y * dataBorder[0].cols + x;
							mcov_local[idx] = _mm256_fmadd_ps(mc, _mm256_loadu_ps(sptr[cy] + pos), mcov_local[idx]);
							idx++;
						}
					}
				}
			}

			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] += simd_step;
			}
		}
		if (isRem)
		{
			for (int cx = 0, idx = 0; cx < color_channels; cx++)
			{
				const __m256 mc = _mm256_maskload_ps(sptr[cx] + center, mask);
				for (int cy = 0; cy < color_channels; cy++)
				{
					for (int y = 0; y < R; y++)
					{
						for (int x = 0; x < R; x++)
						{
							const int pos = y * dataBorder[0].cols + x;
							mcov_local[idx] = _mm256_fmadd_ps(mc, _mm256_maskload_ps(sptr[cy] + pos, mask), mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
	}

	//reduction
	AutoBuffer<double> covElem(covElementSize);
	for (int i = 0; i < covElementSize; i++) covElem[i] = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		__m256* mvar = &mbuff[covElementSize * t];
		for (int d = 0; d < covElementSize; d++)
		{
			covElem[d] += _mm256_reduceadd_pspd(mvar[d]);
		}
	}

	for (int cx = 0, idx = 0; cx < color_channels; cx++)
	{
		for (int cy = 0; cy < color_channels; cy++)
		{
			for (int y = 0; y < R; y++)
			{
				for (int x = 0; x < R; x++)
				{
					vector<Point> pt = dir2cov(x - Rh, y - Rh, D);
					const double val = covElem[idx] * normalSize;
					for (int d = 0; d < pt.size(); d++)
					{
						cov.at<double>(pt[d].y * color_channels + cy, pt[d].x * color_channels + cx) = val;
					}
					idx++;
				}
			}
		}
	}

	_mm_free(mbuff);
}

template<int color_channels, int patch_rad>
void CalcPatchCovarMatrix::simdOMPCov_RepCenterConvFElement32F(const vector<Mat>& src_, Mat& cov, const CenterMethod method, const float constant_sub, const int border)
{
	//cout << "simdOMPCov_RepCenterConvFElement32F" << endl;
	const int simd_step = 8;
	const int DD = dim / color_channels;

	constexpr int D = 2 * patch_rad + 1;
	constexpr int R = 2 * D - 1;
	constexpr int RR = R * R;
	constexpr int Rh = R / 2;

	vector<Mat> data(color_channels);
	dataBorder.resize(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			//cout << "ConvF mean" << endl; 
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
		}
		else if (method == CenterMethod::CONST_)
		{
			//cout << "ConvF const: " <<const_sub<< endl;
			sub_const(src_[c], constant_sub, data[c]);
		}
		else if (method == CenterMethod::NO)
		{
			//cout << "ConvF nosub" << endl;
			data[c] = src_[c];
		}
		else
		{
			cout << int(method) << endl;
			cout << "No support method (simdOMPCovRepCov32F)" << endl;
		}

		{
			copyMakeBorder(data[c], dataBorder[c], Rh, Rh, Rh, Rh, border);
		}
	}

	const int thread_max = omp_get_max_threads();

	const int xst = 0;
	const int yst = 0;
	const int xend = data[0].cols - 0;
	const int yend = data[0].rows - 0;
	const int simd_end = get_simd_floor(xend - xst, simd_step);
	const __m256i mask = get_simd_residualmask_epi32(xend - xst);
	const bool isRem = ((xend - xst) == simd_end) ? false : true;

	const double normalSize = 1.0 / ((data[0].rows) * (xend - xst));
	//const double normalSize = 1.0;
	constexpr int covElementSize = R * R * color_channels * color_channels;
	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * covElementSize * thread_max, AVX_ALIGN);

	const int center = Rh * dataBorder[0].cols + Rh;

	for (int i = 0; i < covElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

#pragma omp parallel for SCHEDULE
	for (int y = yst; y < yend; y++)
	{
		const int tindex = omp_get_thread_num();
		__m256* mcov_local = &mbuff[covElementSize * tindex];
		AutoBuffer<__m256> msrc_local(covElementSize);
		AutoBuffer<const float*> sptr(color_channels);
		for (int c = 0; c < color_channels; c++)
		{
			sptr[c] = dataBorder[c].ptr<float>(y);
		}

		AutoBuffer<__m256> mxv(color_channels);
		for (int i = xst; i < simd_end; i += simd_step)
		{
			for (int cx = 0; cx < color_channels; cx++)
			{
				mxv[cx] = _mm256_loadu_ps(sptr[cx] + center);
			}
			int idx = 0;
			for (int cy = 0; cy < color_channels; cy++)
			{
				for (int y = 0; y < R; y++)
				{
					const float* ref = sptr[cy] + y * dataBorder[0].cols;
					for (int x = 0; x < R; x++)
					{
						const __m256 myv = _mm256_loadu_ps(ref + x);
						for (int cx = 0; cx < color_channels; cx++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(mxv[cx], myv, mcov_local[idx]);
							idx++;
						}
					}
				}
			}

			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] += simd_step;
			}
		}
		if (isRem)
		{
			for (int cx = 0; cx < color_channels; cx++)
			{
				mxv[cx] = _mm256_maskload_ps(sptr[cx] + center, mask);
			}
			int idx = 0;
			for (int cy = 0; cy < color_channels; cy++)
			{
				for (int y = 0; y < R; y++)
				{
					const float* ref = sptr[cy] + y * dataBorder[0].cols;
					for (int x = 0; x < R; x++)
					{
						const __m256 myv = _mm256_maskload_ps(ref + x, mask);
						for (int cx = 0; cx < color_channels; cx++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(mxv[cx], myv, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
	}

	//reduction
	AutoBuffer<double> covElem(covElementSize);
	for (int i = 0; i < covElementSize; i++) covElem[i] = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		__m256* mvar = &mbuff[covElementSize * t];
		for (int d = 0; d < covElementSize; d++)
		{
			covElem[d] += _mm256_reduceadd_pspd(mvar[d]);
		}
	}

#pragma omp parallel for
	for (int y = 0; y < R; y++)
	{
		for (int x = 0; x < R; x++)
		{
			vector<Point> pt = dir2cov(x - Rh, y - Rh, D);
			for (int cy = 0; cy < color_channels; cy++)
			{
				for (int cx = 0; cx < color_channels; cx++)
				{
					const int idx = RR * color_channels * cy + R * color_channels * y + color_channels * x + cx;
					const double val = covElem[idx] * normalSize;
					for (int d = 0; d < pt.size(); d++)
					{
						cov.at<double>(pt[d].y * color_channels + cy, pt[d].x * color_channels + cx) = val;
					}
				}
			}
		}
	}

	_mm_free(mbuff);
}

void CalcPatchCovarMatrix::simdOMPCov_RepCenterConvFElement32FCn(const vector<Mat>& src_, Mat& cov, const CenterMethod method, const float constant_sub, const int border)
{
	//cout << "simdOMPCov_RepCenterConvQElement32F" << endl;
	const int simd_step = 8;
	const int DD = dim / color_channels;

	const int D = 2 * patch_rad + 1;
	const int R = 2 * D - 1;
	const int Rh = R / 2;

	vector<Mat> data(color_channels);
	dataBorder.resize(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			//cout << "Conv mean" << endl; 
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
		}
		else if (method == CenterMethod::CONST_)
		{
			//cout << "Conv const: " <<const_sub<< endl;
			//subtract(src_[c], constant_sub, data[c]);
			sub_const(src_[c], constant_sub, data[c]);
		}
		else if (method == CenterMethod::NO)
		{
			//cout << "Conv nosub" << endl;
			//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			//else meanForCov[0] = 0.0;
			data[c] = src_[c];
		}
		else
		{
			cout << int(method) << endl;
			cout << "No support method (simdOMPCovRepCov32F)" << endl;
		}

		{
			copyMakeBorder(data[c], dataBorder[c], Rh, Rh, Rh, Rh, border);
		}
	}

	const int thread_max = omp_get_max_threads();

	const int xst = 0;
	const int yst = 0;
	const int xend = data[0].cols - 0;
	const int yend = data[0].rows - 0;
	const int simd_end = get_simd_floor(xend - xst, simd_step);
	const __m256i mask = get_simd_residualmask_epi32(xend - xst);
	const bool isRem = ((xend - xst) == simd_end) ? false : true;

	const double normalSize = 1.0 / ((data[0].rows) * (xend - xst));
	//const double normalSize = 1.0;
	const int covElementSize = R * R * color_channels * color_channels;
	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * covElementSize * thread_max, AVX_ALIGN);

	const int center = Rh * dataBorder[0].cols + Rh;

	for (int i = 0; i < covElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

#pragma omp parallel for SCHEDULE
	for (int y = yst; y < yend; y++)
	{
		const int tindex = omp_get_thread_num();
		__m256* mcov_local = &mbuff[covElementSize * tindex];
		AutoBuffer<__m256> msrc_local(covElementSize);
		AutoBuffer<const float*> sptr(color_channels);
		for (int c = 0; c < color_channels; c++)
		{
			sptr[c] = dataBorder[c].ptr<float>(y);
		}

		for (int i = xst; i < simd_end; i += simd_step)
		{
			for (int cx = 0, idx = 0; cx < color_channels; cx++)
			{
				const __m256 mc = _mm256_loadu_ps(sptr[cx] + center);
				for (int cy = 0; cy < color_channels; cy++)
				{
					for (int y = 0; y < R; y++)
					{
						for (int x = 0; x < R; x++)
						{
							const int pos = y * dataBorder[0].cols + x;
							mcov_local[idx] = _mm256_fmadd_ps(mc, _mm256_loadu_ps(sptr[cy] + pos), mcov_local[idx]);
							idx++;
						}
					}
				}
			}

			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] += simd_step;
			}
		}
		if (isRem)
		{
			for (int cx = 0, idx = 0; cx < color_channels; cx++)
			{
				const __m256 mc = _mm256_maskload_ps(sptr[cx] + center, mask);
				for (int cy = 0; cy < color_channels; cy++)
				{
					for (int y = 0; y < R; y++)
					{
						for (int x = 0; x < R; x++)
						{
							const int pos = y * dataBorder[0].cols + x;
							mcov_local[idx] = _mm256_fmadd_ps(mc, _mm256_maskload_ps(sptr[cy] + pos, mask), mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
	}

	//reduction
	AutoBuffer<double> covElem(covElementSize);
	for (int i = 0; i < covElementSize; i++) covElem[i] = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		__m256* mvar = &mbuff[covElementSize * t];
		for (int d = 0; d < covElementSize; d++)
		{
			covElem[d] += _mm256_reduceadd_pspd(mvar[d]);
		}
	}

	for (int cx = 0, idx = 0; cx < color_channels; cx++)
	{
		for (int cy = 0; cy < color_channels; cy++)
		{
			for (int y = 0; y < R; y++)
			{
				for (int x = 0; x < R; x++)
				{
					vector<Point> pt = dir2cov(x - Rh, y - Rh, D);
					const double val = covElem[idx] * normalSize;
					for (int d = 0; d < pt.size(); d++)
					{
						cov.at<double>(pt[d].y * color_channels + cy, pt[d].x * color_channels + cx) = val;
					}
					idx++;
				}
			}
		}
	}

	_mm_free(mbuff);
}


cv::Mat crossCorrelation(const cv::Mat& src, const cv::Mat& ref, int maxShift)
{
	CV_Assert(src.depth() == CV_32F);
	CV_Assert(ref.depth() == CV_32F);
	//int width = cv::getOptimalDFTSize(std::max(src.cols, ref.cols));
	//int height = cv::getOptimalDFTSize(std::max(src.rows, ref.rows));

	cv::Mat fft1;
	cv::Mat fft2;
	cv::dft(src, fft1, 0, src.rows);
	cv::dft(ref, fft2, 0, ref.rows);
	cv::mulSpectrums(fft1, fft2, fft1, 0, true);
	cv::idft(fft1, fft1, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT, maxShift);
	return fft1;
}

void CalcPatchCovarMatrix::covFFT(const vector<Mat>& src_, Mat& cov, const CenterMethod method, const float constant_sub)
{
	//cout << "simdOMPCov_RepCenterConvElement32F" << endl;
	const int simd_step = 8;
	const int DD = dim / color_channels;
	const int D = 2 * patch_rad + 1;

	vector<Mat> data(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			//cout << "Conv mean" << endl; 
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
		}
		else if (method == CenterMethod::CONST_)
		{
			//cout << "Conv const: " <<const_sub<< endl;
			//subtract(src_[c], constant_sub, data[c]);
			sub_const(src_[c], constant_sub, data[c]);
		}
		else if (method == CenterMethod::NO)
		{
			//cout << "Conv nosub" << endl;
			//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			//else meanForCov[0] = 0.0;
			data[c] = src_[c];
		}
		else
		{
			cout << int(method) << endl;
			cout << "No support method (simdOMPCovRepCov32F)" << endl;
		}
	}

	const double normalSize = 1.0 / ((data[0].rows) * (data[0].cols));
	//const int center = patch_rad * dataBorder[0].cols + patch_rad;
	const int center = 0;

	const int covElementSize = D * D * color_channels * color_channels;
	AutoBuffer<double> covElem(covElementSize);
	const int cc = color_channels * color_channels;
#pragma omp parallel for
	for (int c = 0; c < cc; c++)
	{
		const int cy = c / color_channels;
		const int cx = c % color_channels;

		Mat v = crossCorrelation(data[cx], data[cy], DD);
		for (int y = 0; y < D; y++)
		{
			for (int x = 0; x < D; x++)
			{
				int idx = color_channels * DD * cy + D * color_channels * y + color_channels * x + cx;
				covElem[idx] = v.at<float>(y, x);
			}
		}
	}

#pragma omp parallel for
	for (int y = 0; y < D; y++)
	{
		for (int x = 0; x < D; x++)
		{
			vector<Point> pt = dir2covQuad(x, y, D);
			for (int c = 0; c < cc; c++)
			{
				const int cy = c / color_channels;
				const int cx = c % color_channels;

				int idx = color_channels * DD * cy + D * color_channels * y + color_channels * x + cx;
				const double val = covElem[idx] * normalSize;
				for (int d = 0; d < pt.size(); d++)
				{
					cov.at<double>(pt[d].y * color_channels + cy, pt[d].x * color_channels + cx) = val;
				}
			}
		}
	}
}

void CalcPatchCovarMatrix::covFFTFull(const vector<Mat>& src_, Mat& cov, const CenterMethod method, const float constant_sub)
{
	//cout << "simdOMPCov_RepCenterConvElement32F" << endl;
	const int simd_step = 8;
	const int DD = dim / color_channels;
	const int D = 2 * patch_rad + 1;

	vector<Mat> data(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			//cout << "Conv mean" << endl; 
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
		}
		else if (method == CenterMethod::CONST_)
		{
			//cout << "Conv const: " <<const_sub<< endl;
			//subtract(src_[c], constant_sub, data[c]);
			sub_const(src_[c], constant_sub, data[c]);
		}
		else if (method == CenterMethod::NO)
		{
			//cout << "Conv nosub" << endl;
			//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			//else meanForCov[0] = 0.0;
			data[c] = src_[c];
		}
		else
		{
			cout << int(method) << endl;
			cout << "No support method (simdOMPCovRepCov32F)" << endl;
		}
	}

	const double normalSize = 1.0 / ((data[0].rows) * (data[0].cols));
	//const int center = patch_rad * dataBorder[0].cols + patch_rad;
	const int center = 0;

	const int R = 2 * D - 1;
	const int RR = R * R;
	const int Rh = R / 2;
	const int CC = color_channels * color_channels;
	const int covElementSize = RR * CC;
	AutoBuffer<double> covElem(covElementSize);
#pragma omp parallel for
	for (int c = 0; c < CC; c++)
	{
		const int cy = c / color_channels;
		const int cx = c % color_channels;

		if (false)
		{
			Mat s0;
			copyMakeBorder(data[cx], s0, Rh, 0, Rh, 0, cv::BORDER_WRAP);
			s0 = s0(Rect(0, 0, data[0].cols, data[0].rows)).clone();
			Mat s1;
			copyMakeBorder(data[cy], s1, Rh, 0, Rh, 0, cv::BORDER_WRAP);
			s1 = s1(Rect(0, 0, data[0].cols, data[0].rows)).clone();

			Mat v = crossCorrelation(s0, s1, R);
			for (int y = 0; y < R; y++)
			{
				for (int x = 0; x < R; x++)
				{
					const int idx = RR * color_channels * cx + RR * cy + R * y + x;
					covElem[idx] = v.at<float>(y, x);
				}
			}
		}
		else
		{
			Mat v;
			{
				//cp::Timer t;
				v = crossCorrelation(data[cx], data[cy], data[0].rows);
			}
			for (int y = 0; y < R; y++)
			{
				for (int x = 0; x < R; x++)
				{
					const int X = (x - Rh < 0) ? data[0].cols - 0 + (x - Rh) : x - Rh;
					const int Y = (y - Rh < 0) ? data[0].rows - 0 + (y - Rh) : y - Rh;
					const int idx = RR * color_channels * cx + RR * cy + R * y + x;
					//print_debug3(X, Y, idx);
					covElem[idx] = v.at<float>(Y, X);
				}
			}
		}
	}

#pragma omp parallel for
	for (int y = 0; y < R; y++)
	{
		for (int x = 0; x < R; x++)
		{
			vector<Point> pt = dir2cov(x - Rh, y - Rh, D);
			for (int cx = 0, idx = 0; cx < color_channels; cx++)
			{
				for (int cy = 0; cy < color_channels; cy++)
				{
					const int idx = RR * color_channels * cx + RR * cy + R * y + x;
					const double val = covElem[idx] * normalSize;
					for (int d = 0; d < pt.size(); d++)
					{
						cov.at<double>(pt[d].y * color_channels + cy, pt[d].x * color_channels + cx) = val;
					}
				}
			}
		}
	}
}
#pragma endregion

#pragma region representative_mean

template<int color_channels, int patch_rad>
void CalcPatchCovarMatrix::simdOMPCov_RepCenterHalfElement32F(const vector<Mat>& src_, Mat& destCovarianceMatrix, const CenterMethod method, const float constant_sub)
{
	bool isBorder = true;

	//cout << "simdOMPCovRepMean32F" << endl;
	const int simd_step = 8;
	data.resize(color_channels);
	dataBorder.resize(color_channels);

	const int thread_max = omp_get_max_threads();
	constexpr int dim = (2 * patch_rad + 1) * (2 * patch_rad + 1) * color_channels;
	constexpr int DD = (2 * patch_rad + 1) * (2 * patch_rad + 1);
	int* scan = (int*)_mm_malloc(sizeof(int) * DD, AVX_ALIGN);
	const int computeElementSize = getComputeHalfCovElementSize(dim);
	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * computeElementSize * thread_max, AVX_ALIGN);

	double normalSize = 0.0;
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			sub_const(src_[c], float(cp::average(src_[c])), data[c]);
		}
		else if (method == CenterMethod::CONST_)
		{
			sub_const(src_[c], constant_sub, data[c]);
		}
		else if (method == CenterMethod::NO)
		{
			//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			//else meanForCov[0] = 0.0;
			data[c] = src_[c];
		}
		else
		{
			cout << int(method) << endl;
			cout << "No support method (simdOMPCovRepMean32F)" << endl;
		}
		if (isBorder)
		{
			copyMakeBorder(data[c], dataBorder[c], patch_rad, patch_rad, patch_rad, patch_rad, border);
		}
	}

	if (isBorder)
	{
		getScanorderBorder(scan, dataBorder[0].cols, 1);
		const int simd_end = get_simd_floor(src_[0].cols, simd_step);
		const __m256i mask = get_simd_residualmask_epi32(src_[0].cols);
		const bool isRem = (src_[0].cols == simd_end) ? false : true;
		normalSize = 1.0 / ((src_[0].rows) * (src_[0].cols));

		for (int i = 0; i < computeElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

		if (color_channels == 1)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				const float* sptr = dataBorder[0].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr + *scptr);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					sptr += simd_step;
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_maskload_ps(sptr + *scptr, mask);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
		else if (color_channels == 2)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				AutoBuffer<const float*> sptr(2);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr[0] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[1] + *scptr);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_maskload_ps(sptr[0] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[1] + *scptr, mask);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
		else if (color_channels == 3)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				AutoBuffer<const float*> sptr(3);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr[0] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[1] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[2] + *scptr);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_maskload_ps(sptr[0] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[1] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[2] + *scptr, mask);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
		else if (color_channels == 4)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				AutoBuffer<const float*> sptr(4);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);
				sptr[3] = dataBorder[3].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr[0] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[1] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[2] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[3] + *scptr);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
					sptr[3] += simd_step;
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_maskload_ps(sptr[0] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[1] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[2] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[3] + *scptr, mask);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
		else if (color_channels == 6)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				AutoBuffer<const float*> sptr(6);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);
				sptr[3] = dataBorder[3].ptr<float>(j);
				sptr[4] = dataBorder[4].ptr<float>(j);
				sptr[5] = dataBorder[5].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr[0] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[1] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[2] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[3] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[4] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[5] + *scptr);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
					sptr[3] += simd_step;
					sptr[4] += simd_step;
					sptr[5] += simd_step;
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_maskload_ps(sptr[0] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[1] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[2] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[3] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[4] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[5] + *scptr, mask);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
		else
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				AutoBuffer<const float*> sptr(color_channels);
				for (int c = 0; c < color_channels; c++)
				{
					sptr[c] = dataBorder[c].ptr<float>(j);
				}

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						for (int c = 0; c < color_channels; c++)
						{
							*msrcptr++ = _mm256_loadu_ps(sptr[c] + *scptr);
						}
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					for (int c = 0; c < color_channels; c++)
					{
						sptr[c] += simd_step;
					}
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						for (int c = 0; c < color_channels; c++)
						{
							*msrcptr++ = _mm256_maskload_ps(sptr[c] + *scptr, mask);
						}
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
	}
	else
	{
		for (int c = 0; c < color_channels; c++)
		{
			if (method == CenterMethod::MEAN)
			{
				sub_const(src_[c], float(cp::average(src_[c])), data[c]);
			}
			else if (method == CenterMethod::CONST_)
			{
				sub_const(src_[c], constant_sub, data[c]);
			}
			else if (method == CenterMethod::NO)
			{
				//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
				//else meanForCov[0] = 0.0;
				data[c] = src_[c];
			}
			else
			{
				cout << "No support method (simdOMPCovRepMean32F)" << endl;
			}
		}

		getScanorder(scan, data[0].cols, 1, false);
		const int simd_end_x = get_simd_floor(data[0].cols - 2 * patch_rad, simd_step) + patch_rad;
		normalSize = 1.0 / ((data[0].rows - 2 * patch_rad) * (simd_end_x - patch_rad));

		for (int i = 0; i < computeElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

		const int yend = data[0].rows - 2 * patch_rad;
#pragma omp parallel for SCHEDULE
		for (int y = 0; y < yend; y++)
		{
			const int j = y + patch_rad;
			const int tindex = omp_get_thread_num();

			__m256* mcov_local = &mbuff[computeElementSize * tindex];

			AutoBuffer<__m256> msrc_local(dim);
			AutoBuffer<const float*> sptr(color_channels);
			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] = data[c].ptr<float>(j, patch_rad);
			}

			for (int i = patch_rad; i < simd_end_x; i += simd_step)
			{
				//load data to register or L1 cache
				__m256* msrcptr = &msrc_local[0];
				int* scptr = &scan[0];
				for (int k = 0; k < DD; k++)
				{
					for (int c = 0; c < color_channels; c++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr[c] + *scptr);
					}
					scptr++;
				}

				//compute covariance
				for (int y = 0, idx = 0; y < dim; y++)
				{
					const __m256 my = msrc_local[y];
					for (int x = y; x < dim; x++)
					{
						mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
						idx++;
					}
				}

				for (int c = 0; c < color_channels; c++)
				{
					sptr[c] += simd_step;
				}
			}
		}
	}

	//reduction
	vector<double> covElem(computeElementSize);
	for (double c : covElem) c = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		const int sstep = computeElementSize * t;
		__m256* mcov = &mbuff[sstep];
		for (int i = 0; i < computeElementSize; i++)
		{
			covElem[i] += _mm256_reduceadd_pspd(mcov[i]);
		}
	}

	setCovHalf(destCovarianceMatrix, covElem, normalSize);

	_mm_free(scan);
	_mm_free(mbuff);
}

void CalcPatchCovarMatrix::simdOMPCov_RepCenterHalfElement32FCn(const vector<Mat>& src_, Mat& destCovarianceMatrix, const CenterMethod method, const float constant_sub)
{
	bool isBorder = true;

	//cout << "simdOMPCovRepMean32F" << endl;
	const int simd_step = 8;
	data.resize(color_channels);
	dataBorder.resize(color_channels);

	const int thread_max = omp_get_max_threads();
	const int dim = (2 * patch_rad + 1) * (2 * patch_rad + 1) * color_channels;
	const int DD = (2 * patch_rad + 1) * (2 * patch_rad + 1);
	int* scan = (int*)_mm_malloc(sizeof(int) * DD, AVX_ALIGN);
	const int computeElementSize = getComputeHalfCovElementSize(dim);
	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * computeElementSize * thread_max, AVX_ALIGN);

	double normalSize = 0.0;
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			sub_const(src_[c], float(cp::average(src_[c])), data[c]);
		}
		else if (method == CenterMethod::CONST_)
		{
			sub_const(src_[c], constant_sub, data[c]);
		}
		else if (method == CenterMethod::NO)
		{
			//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			//else meanForCov[0] = 0.0;
			data[c] = src_[c];
		}
		else
		{
			cout << int(method) << endl;
			cout << "No support method (simdOMPCovRepMean32F)" << endl;
		}
		if (isBorder)
		{
			copyMakeBorder(data[c], dataBorder[c], patch_rad, patch_rad, patch_rad, patch_rad, border);
		}
	}

	if (isBorder)
	{
		getScanorderBorder(scan, dataBorder[0].cols, 1);
		const int simd_end = get_simd_floor(src_[0].cols, simd_step);
		const __m256i mask = get_simd_residualmask_epi32(src_[0].cols);
		const bool isRem = (src_[0].cols == simd_end) ? false : true;
		normalSize = 1.0 / ((src_[0].rows) * (src_[0].cols));

		for (int i = 0; i < computeElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

		if (color_channels == 1)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				const float* sptr = dataBorder[0].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr + *scptr);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					sptr += simd_step;
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_maskload_ps(sptr + *scptr, mask);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
		else if (color_channels == 2)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				AutoBuffer<const float*> sptr(2);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr[0] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[1] + *scptr);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_maskload_ps(sptr[0] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[1] + *scptr, mask);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
		else if (color_channels == 3)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				AutoBuffer<const float*> sptr(3);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr[0] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[1] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[2] + *scptr);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_maskload_ps(sptr[0] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[1] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[2] + *scptr, mask);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
		else if (color_channels == 4)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				AutoBuffer<const float*> sptr(4);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);
				sptr[3] = dataBorder[3].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr[0] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[1] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[2] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[3] + *scptr);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
					sptr[3] += simd_step;
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_maskload_ps(sptr[0] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[1] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[2] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[3] + *scptr, mask);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
		else if (color_channels == 6)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				AutoBuffer<const float*> sptr(6);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);
				sptr[3] = dataBorder[3].ptr<float>(j);
				sptr[4] = dataBorder[4].ptr<float>(j);
				sptr[5] = dataBorder[5].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr[0] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[1] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[2] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[3] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[4] + *scptr);
						*msrcptr++ = _mm256_loadu_ps(sptr[5] + *scptr);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
					sptr[3] += simd_step;
					sptr[4] += simd_step;
					sptr[5] += simd_step;
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						*msrcptr++ = _mm256_maskload_ps(sptr[0] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[1] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[2] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[3] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[4] + *scptr, mask);
						*msrcptr++ = _mm256_maskload_ps(sptr[5] + *scptr, mask);
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
		else
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < src_[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];

				AutoBuffer<__m256> msrc_local(dim);
				AutoBuffer<const float*> sptr(color_channels);
				for (int c = 0; c < color_channels; c++)
				{
					sptr[c] = dataBorder[c].ptr<float>(j);
				}

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						for (int c = 0; c < color_channels; c++)
						{
							*msrcptr++ = _mm256_loadu_ps(sptr[c] + *scptr);
						}
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}

					for (int c = 0; c < color_channels; c++)
					{
						sptr[c] += simd_step;
					}
				}
				if (isRem)
				{
					//load data to register or L1 cache
					__m256* msrcptr = &msrc_local[0];
					int* scptr = &scan[0];
					for (int k = 0; k < DD; k++)
					{
						for (int c = 0; c < color_channels; c++)
						{
							*msrcptr++ = _mm256_maskload_ps(sptr[c] + *scptr, mask);
						}
						scptr++;
					}

					//compute covariance
					for (int y = 0, idx = 0; y < dim; y++)
					{
						const __m256 my = msrc_local[y];
						for (int x = y; x < dim; x++)
						{
							mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
							idx++;
						}
					}
				}
			}
		}
	}
	else
	{
		for (int c = 0; c < color_channels; c++)
		{
			if (method == CenterMethod::MEAN)
			{
				sub_const(src_[c], float(cp::average(src_[c])), data[c]);
			}
			else if (method == CenterMethod::CONST_)
			{
				sub_const(src_[c], constant_sub, data[c]);
			}
			else if (method == CenterMethod::NO)
			{
				//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
				//else meanForCov[0] = 0.0;
				data[c] = src_[c];
			}
			else
			{
				cout << "No support method (simdOMPCovRepMean32F)" << endl;
			}
		}

		getScanorder(scan, data[0].cols, 1, false);
		const int simd_end_x = get_simd_floor(data[0].cols - 2 * patch_rad, simd_step) + patch_rad;
		normalSize = 1.0 / ((data[0].rows - 2 * patch_rad) * (simd_end_x - patch_rad));

		for (int i = 0; i < computeElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

		const int yend = data[0].rows - 2 * patch_rad;
#pragma omp parallel for SCHEDULE
		for (int y = 0; y < yend; y++)
		{
			const int j = y + patch_rad;
			const int tindex = omp_get_thread_num();

			__m256* mcov_local = &mbuff[computeElementSize * tindex];

			AutoBuffer<__m256> msrc_local(dim);
			AutoBuffer<const float*> sptr(color_channels);
			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] = data[c].ptr<float>(j, patch_rad);
			}

			for (int i = patch_rad; i < simd_end_x; i += simd_step)
			{
				//load data to register or L1 cache
				__m256* msrcptr = &msrc_local[0];
				int* scptr = &scan[0];
				for (int k = 0; k < DD; k++)
				{
					for (int c = 0; c < color_channels; c++)
					{
						*msrcptr++ = _mm256_loadu_ps(sptr[c] + *scptr);
					}
					scptr++;
				}

				//compute covariance
				for (int y = 0, idx = 0; y < dim; y++)
				{
					const __m256 my = msrc_local[y];
					for (int x = y; x < dim; x++)
					{
						mcov_local[idx] = _mm256_fmadd_ps(msrc_local[x], my, mcov_local[idx]);
						idx++;
					}
				}

				for (int c = 0; c < color_channels; c++)
				{
					sptr[c] += simd_step;
				}
			}
		}
	}

	//reduction
	vector<double> covElem(computeElementSize);
	for (double c : covElem) c = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		const int sstep = computeElementSize * t;
		__m256* mcov = &mbuff[sstep];
		for (int i = 0; i < computeElementSize; i++)
		{
			covElem[i] += _mm256_reduceadd_pspd(mcov[i]);
		}
	}

	setCovHalf(destCovarianceMatrix, covElem, normalSize);

	_mm_free(scan);
	_mm_free(mbuff);
}

double average64(const Mat& src, const int left = 0, const int right = 0, const int top = 0, const int bottom = 0, const bool isNormalize = true)
{
	CV_Assert(src.type() == CV_64FC1);
	__m256d msum1 = _mm256_setzero_pd();
	__m256d msum2 = _mm256_setzero_pd();
	__m256d msum3 = _mm256_setzero_pd();
	__m256d msum4 = _mm256_setzero_pd();

	const bool isFull = (left == 0 && right == 0 && top == 0 && bottom == 0);
	const int size = (isFull) ? src.size().area() : (src.cols - (left + right)) * (src.rows - (top + bottom));

	double sum = 0.0;
	if (isFull)
	{
		const double* sptr = src.ptr<double>();
		const int simdSize = get_simd_floor(size, 16);
		for (int i = 0; i < simdSize; i += 16)
		{
			msum1 = _mm256_add_pd(_mm256_load_pd(sptr + i + 0), msum1);
			msum2 = _mm256_add_pd(_mm256_load_pd(sptr + i + 4), msum2);
			msum3 = _mm256_add_pd(_mm256_load_pd(sptr + i + 8), msum3);
			msum4 = _mm256_add_pd(_mm256_load_pd(sptr + i + 12), msum4);
		}
		sum = _mm256_reduceadd_pd(msum1) + _mm256_reduceadd_pd(msum2) + _mm256_reduceadd_pd(msum3) + _mm256_reduceadd_pd(msum4);
		double rem = 0.0;
		for (int i = simdSize; i < size; i++)
		{
			rem += sptr[i];
		}
		sum += rem;
	}
	else
	{
		const int simdend = get_simd_floor(src.cols - (left + right), 32) + left;
		for (int j = top; j < src.rows - bottom; j++)
		{
			const double* sptr = src.ptr<double>(j);
			msum1 = _mm256_setzero_pd();
			msum2 = _mm256_setzero_pd();
			msum3 = _mm256_setzero_pd();
			msum4 = _mm256_setzero_pd();
			for (int i = left; i < simdend; i += 16)
			{
				msum1 = _mm256_add_pd(_mm256_loadu_pd(sptr + i + 0), msum1);
				msum2 = _mm256_add_pd(_mm256_loadu_pd(sptr + i + 4), msum2);
				msum3 = _mm256_add_pd(_mm256_loadu_pd(sptr + i + 8), msum3);
				msum4 = _mm256_add_pd(_mm256_loadu_pd(sptr + i + 12), msum4);
			}
			sum += _mm256_reduceadd_pd(msum1) + _mm256_reduceadd_pd(msum2) + _mm256_reduceadd_pd(msum3) + _mm256_reduceadd_pd(msum4);
			double rem = 0.0;
			for (int i = simdend; i < src.cols - right; i++)
			{
				rem += sptr[i];
			}
			sum += rem;
		}
	}

	if (isNormalize)return sum / size;
	else return sum;
}

template<int color_channels, int dim>
void CalcPatchCovarMatrix::simdOMPCov_RepCenterHalfElement64F(const vector<Mat>& src_, Mat& cov, const CenterMethod method, const double constant_sub)
{
	//cout << "simdOMPCovRepMean32F" << endl;
	const int simd_step = 4;
	data.resize(color_channels);

	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			if (src_[0].depth() == CV_64F) sub_const(src_[c], average64(src_[c]), data[c]);
			if (src_[0].depth() == CV_32F) sub_const32to64(src_[c], (float)cp::average(src_[c]), data[c]);
		}
		else if (method == CenterMethod::CONST_)
		{
			if (src_[0].depth() == CV_64F) sub_const(src_[c], constant_sub, data[c]);
			if (src_[0].depth() == CV_32F) sub_const32to64(src_[c], (float)constant_sub, data[c]);
		}
		else if (method == CenterMethod::NO)
		{
			//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			//else meanForCov[0] = 0.0;
			if (src_[c].depth() == CV_32F)src_[c].convertTo(data[c], CV_64F);
			else data[c] = src_[c];
		}
		else
		{
			cout << "No support method (simdOMPCovRepMean32F)" << endl;
		}
	}

	const int DD = dim / color_channels;

	int* scan = (int*)_mm_malloc(sizeof(int) * DD, AVX_ALIGN);
	getScanorder(scan, data[0].cols, 1, false);

	const int thread_max = omp_get_max_threads();
	const int simd_end_x = get_simd_floor(data[0].cols - 2 * patch_rad, simd_step) + patch_rad;
	const double normalSize = 1.0 / ((data[0].rows - 2 * patch_rad) * (simd_end_x - patch_rad));

	const int computeElementSize = getComputeHalfCovElementSize(dim);
	__m256d* mbuff = (__m256d*)_mm_malloc(sizeof(__m256d) * computeElementSize * thread_max, AVX_ALIGN);
	for (int i = 0; i < computeElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_pd();

	const int yend = data[0].rows - 2 * patch_rad;
#pragma omp parallel for SCHEDULE
	for (int y = 0; y < yend; y++)
	{
		const int j = y + patch_rad;
		const int tindex = omp_get_thread_num();

		__m256d* mcov_local = &mbuff[computeElementSize * tindex];

		AutoBuffer<__m256d> msrc_local(dim);
		AutoBuffer<const double*> sptr(color_channels);
		for (int c = 0; c < color_channels; c++)
		{
			sptr[c] = data[c].ptr<double>(j, patch_rad);
		}

		for (int i = patch_rad; i < simd_end_x; i += simd_step)
		{
			//load data to register or L1 cache
			__m256d* msrcptr = &msrc_local[0];
			int* scptr = &scan[0];
			for (int k = 0; k < DD; k++)
			{
				for (int c = 0; c < color_channels; c++)
				{
					*msrcptr++ = _mm256_loadu_pd(sptr[c] + *scptr);
				}
				scptr++;
			}

			//compute covariance
			for (int y = 0, idx = 0; y < dim; y++)
			{
				const __m256d my = msrc_local[y];
				for (int x = y; x < dim; x++)
				{
					mcov_local[idx] = _mm256_fmadd_pd(msrc_local[x], my, mcov_local[idx]);
					idx++;
				}
			}

			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] += simd_step;
			}
		}
	}

	//reduction
	vector<double> covElem(computeElementSize);
	for (double c : covElem) c = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		const int sstep = computeElementSize * t;
		__m256d* mcov = &mbuff[sstep];
		for (int i = 0; i < computeElementSize; i++)
		{
			covElem[i] += _mm256_reduceadd_pd(mcov[i]);
		}
	}

	setCovHalf(cov, covElem, normalSize);

	_mm_free(scan);
	_mm_free(mbuff);
}

void CalcPatchCovarMatrix::simdOMPCov_RepCenterHalfElement64FCn(const vector<Mat>& src_, Mat& cov, const CenterMethod method, const double constant_sub)
{}
#pragma endregion

#pragma region representative_covariance

template<int color_channels, int patch_rad>
void CalcPatchCovarMatrix::simdOMPCov_RepCenterRepElement32F(const vector<Mat>& src_, Mat& destCovarianceMatrix, const CenterMethod method, const float constant_sub)
{
	bool isBorder = true;
	dim = (2 * patch_rad + 1) * (2 * patch_rad + 1) * color_channels;
	//cout << "simdOMPCovRepCov32F: "<<color_channels<<","<<dim << endl;

	RepresentiveCovarianceComputer rcc(color_channels, patch_rad);
	const int simd_step = 8;
	data.resize(color_channels);
	dataBorder.resize(color_channels);

	double normalSize = 0.0;

	vector<double> meanForCov(color_channels);

	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			//cout << "Rep mean" << endl; 
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
			meanForCov[c] = 0.0;
		}
		else if (method == CenterMethod::CONST_)
		{
			//cout << "Rep const" << endl;
			//subtract(src_[c], constant_sub, data[c]);
			sub_const(src_[c], constant_sub, data[c]);
			meanForCov[c] = 0.0;
		}
		else if (method == CenterMethod::NO)
		{
			//cout << "Rep nosub" << endl;
			//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			//else meanForCov[0] = 0.0;
			data[c] = src_[c];
			meanForCov[0] = 0.0;
		}
		else
		{
			cout << int(method) << endl;
			cout << "No support method (simdOMPCovRepCov32F)" << endl;
		}
		if (isBorder)
		{
			copyMakeBorder(data[c], dataBorder[c], patch_rad, patch_rad, patch_rad, patch_rad, border);
		}
	}

	const int thread_max = omp_get_max_threads();

	//constexpr int directionElementSize = 2 * (2 * patch_rad + 1) * (2 * patch_rad);
	constexpr int directionElementSize = (4 * patch_rad + 1) * (4 * patch_rad + 1) / 2;
	const int computeCovElementSize = directionElementSize * color_channels * color_channels;
	const int computeVarElementSize = sum_from(color_channels);
	const int computeElementSize = computeCovElementSize + computeVarElementSize;
	//print_debug3(computeCovElementSize, computeVarElementSize, computeElementSize);

	AutoBuffer<int> first(directionElementSize);
	AutoBuffer<int> second(directionElementSize);

	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * computeElementSize * thread_max, AVX_ALIGN);
	for (int i = 0; i < computeElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

	if (isBorder)
	{
		rcc.getIndexBorder(first, second, dataBorder[0].cols);
		const int simd_end = get_simd_floor(src_[0].cols, simd_step);
		const __m256i mask = get_simd_residualmask_epi32(src_[0].cols);
		const bool isRem = (src_[0].cols == simd_end) ? false : true;
		normalSize = 1.0 / ((src_[0].rows) * (src_[0].cols));

		if (color_channels == 1)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];

				const float* sptr = dataBorder[0].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					mvar_local[0] = _mm256_fmadd_ps(_mm256_loadu_ps(sptr), _mm256_loadu_ps(sptr), mvar_local[0]);

					for (int k = 0; k < directionElementSize; k++)
					{
						mcov_local[k] = _mm256_fmadd_ps(_mm256_loadu_ps(sptr + first[k]), _mm256_loadu_ps(sptr + second[k]), mcov_local[k]);
					}
					sptr += simd_step;
				}
				if (isRem)
				{
					mvar_local[0] = _mm256_fmadd_ps(_mm256_maskload_ps(sptr, mask), _mm256_maskload_ps(sptr, mask), mvar_local[0]);

					for (int k = 0; k < directionElementSize; k++)
					{
						mcov_local[k] = _mm256_fmadd_ps(_mm256_maskload_ps(sptr + first[k], mask), _mm256_maskload_ps(sptr + second[k], mask), mcov_local[k]);
					}
				}
			}
		}
		else if (color_channels == 2)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];
				AutoBuffer<const float*> sptr(2);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//var (diag)
					const __m256 mg = _mm256_loadu_ps(sptr[0]);
					const __m256 ma = _mm256_loadu_ps(sptr[1]);
					mvar_local[0] = _mm256_fmadd_ps(mg, mg, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(ma, mg, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(ma, ma, mvar_local[2]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 4)
					{
						const __m256 mg1 = _mm256_loadu_ps(sptr[0] + first[k]);
						const __m256 ma1 = _mm256_loadu_ps(sptr[1] + first[k]);
						const __m256 mg2 = _mm256_loadu_ps(sptr[0] + second[k]);
						const __m256 ma2 = _mm256_loadu_ps(sptr[1] + second[k]);
						mcov_local[vidx + 0] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(ma1, mg2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mg1, ma2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(ma1, ma2, mcov_local[vidx + 3]);
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
				}
				if (isRem)
				{
					//var (diag)
					const __m256 mg = _mm256_maskload_ps(sptr[0], mask);
					const __m256 ma = _mm256_maskload_ps(sptr[1], mask);
					mvar_local[0] = _mm256_fmadd_ps(mg, mg, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(ma, mg, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(ma, ma, mvar_local[2]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 4)
					{
						const __m256 mg1 = _mm256_maskload_ps(sptr[0] + first[k], mask);
						const __m256 ma1 = _mm256_maskload_ps(sptr[1] + first[k], mask);
						const __m256 mg2 = _mm256_maskload_ps(sptr[0] + second[k], mask);
						const __m256 ma2 = _mm256_maskload_ps(sptr[1] + second[k], mask);
						mcov_local[vidx + 0] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(ma1, mg2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mg1, ma2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(ma1, ma2, mcov_local[vidx + 3]);
					}
				}
			}
		}
		else if (color_channels == 3)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];
				AutoBuffer<const float*> sptr(3);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//var (diag)
					const __m256 mb = _mm256_loadu_ps(sptr[0]);
					const __m256 mg = _mm256_loadu_ps(sptr[1]);
					const __m256 mr = _mm256_loadu_ps(sptr[2]);
					mvar_local[0] = _mm256_fmadd_ps(mb, mb, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(mg, mb, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(mr, mb, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(mg, mg, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(mr, mg, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(mr, mr, mvar_local[5]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 9)
					{
						const __m256 mb1 = _mm256_loadu_ps(sptr[0] + first[k]);
						const __m256 mg1 = _mm256_loadu_ps(sptr[1] + first[k]);
						const __m256 mr1 = _mm256_loadu_ps(sptr[2] + first[k]);
						const __m256 mb2 = _mm256_loadu_ps(sptr[0] + second[k]);
						const __m256 mg2 = _mm256_loadu_ps(sptr[1] + second[k]);
						const __m256 mr2 = _mm256_loadu_ps(sptr[2] + second[k]);
						mcov_local[vidx + 0] = _mm256_fmadd_ps(mb1, mb2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(mg1, mb2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mr1, mb2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(mb1, mg2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(mr1, mg2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(mb1, mr2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(mg1, mr2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(mr1, mr2, mcov_local[vidx + 8]);
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
				}
				if (isRem)
				{
					//var (diag)
					const __m256 mb = _mm256_maskload_ps(sptr[0], mask);
					const __m256 mg = _mm256_maskload_ps(sptr[1], mask);
					const __m256 mr = _mm256_maskload_ps(sptr[2], mask);
					mvar_local[0] = _mm256_fmadd_ps(mb, mb, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(mg, mb, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(mr, mb, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(mg, mg, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(mr, mg, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(mr, mr, mvar_local[5]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 9)
					{
						const __m256 mb1 = _mm256_maskload_ps(sptr[0] + first[k], mask);
						const __m256 mg1 = _mm256_maskload_ps(sptr[1] + first[k], mask);
						const __m256 mr1 = _mm256_maskload_ps(sptr[2] + first[k], mask);
						const __m256 mb2 = _mm256_maskload_ps(sptr[0] + second[k], mask);
						const __m256 mg2 = _mm256_maskload_ps(sptr[1] + second[k], mask);
						const __m256 mr2 = _mm256_maskload_ps(sptr[2] + second[k], mask);
						mcov_local[vidx + 0] = _mm256_fmadd_ps(mb1, mb2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(mg1, mb2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mr1, mb2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(mb1, mg2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(mr1, mg2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(mb1, mr2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(mg1, mr2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(mr1, mr2, mcov_local[vidx + 8]);
					}
				}
			}
		}
		else if (color_channels == 4)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];
				AutoBuffer<const float*> sptr(4);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);
				sptr[3] = dataBorder[3].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//var (diag)
					const __m256 mb = _mm256_loadu_ps(sptr[0]);
					const __m256 mg = _mm256_loadu_ps(sptr[1]);
					const __m256 mr = _mm256_loadu_ps(sptr[2]);
					const __m256 ma = _mm256_loadu_ps(sptr[3]);
					mvar_local[0] = _mm256_fmadd_ps(mb, mb, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(mg, mb, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(mr, mb, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(ma, mb, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(mg, mg, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(mr, mg, mvar_local[5]);
					mvar_local[6] = _mm256_fmadd_ps(ma, mg, mvar_local[6]);
					mvar_local[7] = _mm256_fmadd_ps(mr, mr, mvar_local[7]);
					mvar_local[8] = _mm256_fmadd_ps(ma, mr, mvar_local[8]);
					mvar_local[9] = _mm256_fmadd_ps(ma, ma, mvar_local[9]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 16)
					{
						const __m256 mb1 = _mm256_loadu_ps(sptr[0] + first[k]);
						const __m256 mg1 = _mm256_loadu_ps(sptr[1] + first[k]);
						const __m256 mr1 = _mm256_loadu_ps(sptr[2] + first[k]);
						const __m256 ma1 = _mm256_loadu_ps(sptr[3] + first[k]);
						const __m256 mb2 = _mm256_loadu_ps(sptr[0] + second[k]);
						const __m256 mg2 = _mm256_loadu_ps(sptr[1] + second[k]);
						const __m256 mr2 = _mm256_loadu_ps(sptr[2] + second[k]);
						const __m256 ma2 = _mm256_loadu_ps(sptr[3] + second[k]);

						mcov_local[vidx + 0] = _mm256_fmadd_ps(mb1, mb2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(mg1, mb2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mr1, mb2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(ma1, mb2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(mb1, mg2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(mr1, mg2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(ma1, mg2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(mb1, mr2, mcov_local[vidx + 8]);
						mcov_local[vidx + 9] = _mm256_fmadd_ps(mg1, mr2, mcov_local[vidx + 9]);
						mcov_local[vidx + 10] = _mm256_fmadd_ps(mr1, mr2, mcov_local[vidx + 10]);
						mcov_local[vidx + 11] = _mm256_fmadd_ps(ma1, mr2, mcov_local[vidx + 11]);
						mcov_local[vidx + 12] = _mm256_fmadd_ps(mb1, ma2, mcov_local[vidx + 12]);
						mcov_local[vidx + 13] = _mm256_fmadd_ps(mg1, ma2, mcov_local[vidx + 13]);
						mcov_local[vidx + 14] = _mm256_fmadd_ps(mr1, ma2, mcov_local[vidx + 14]);
						mcov_local[vidx + 15] = _mm256_fmadd_ps(ma1, ma2, mcov_local[vidx + 15]);
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
					sptr[3] += simd_step;
				}
				if (isRem)
				{
					//var (diag)
					const __m256 mb = _mm256_maskload_ps(sptr[0], mask);
					const __m256 mg = _mm256_maskload_ps(sptr[1], mask);
					const __m256 mr = _mm256_maskload_ps(sptr[2], mask);
					const __m256 ma = _mm256_maskload_ps(sptr[3], mask);
					mvar_local[0] = _mm256_fmadd_ps(mb, mb, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(mg, mb, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(mr, mb, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(ma, mb, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(mg, mg, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(mr, mg, mvar_local[5]);
					mvar_local[6] = _mm256_fmadd_ps(ma, mg, mvar_local[6]);
					mvar_local[7] = _mm256_fmadd_ps(mr, mr, mvar_local[7]);
					mvar_local[8] = _mm256_fmadd_ps(ma, mr, mvar_local[8]);
					mvar_local[9] = _mm256_fmadd_ps(ma, ma, mvar_local[9]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 16)
					{
						const __m256 mb1 = _mm256_maskload_ps(sptr[0] + first[k], mask);
						const __m256 mg1 = _mm256_maskload_ps(sptr[1] + first[k], mask);
						const __m256 mr1 = _mm256_maskload_ps(sptr[2] + first[k], mask);
						const __m256 ma1 = _mm256_maskload_ps(sptr[3] + first[k], mask);
						const __m256 mb2 = _mm256_maskload_ps(sptr[0] + second[k], mask);
						const __m256 mg2 = _mm256_maskload_ps(sptr[1] + second[k], mask);
						const __m256 mr2 = _mm256_maskload_ps(sptr[2] + second[k], mask);
						const __m256 ma2 = _mm256_maskload_ps(sptr[3] + second[k], mask);

						mcov_local[vidx + 0] = _mm256_fmadd_ps(mb1, mb2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(mg1, mb2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mr1, mb2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(ma1, mb2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(mb1, mg2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(mr1, mg2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(ma1, mg2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(mb1, mr2, mcov_local[vidx + 8]);
						mcov_local[vidx + 9] = _mm256_fmadd_ps(mg1, mr2, mcov_local[vidx + 9]);
						mcov_local[vidx + 10] = _mm256_fmadd_ps(mr1, mr2, mcov_local[vidx + 10]);
						mcov_local[vidx + 11] = _mm256_fmadd_ps(ma1, mr2, mcov_local[vidx + 11]);
						mcov_local[vidx + 12] = _mm256_fmadd_ps(mb1, ma2, mcov_local[vidx + 12]);
						mcov_local[vidx + 13] = _mm256_fmadd_ps(mg1, ma2, mcov_local[vidx + 13]);
						mcov_local[vidx + 14] = _mm256_fmadd_ps(mr1, ma2, mcov_local[vidx + 14]);
						mcov_local[vidx + 15] = _mm256_fmadd_ps(ma1, ma2, mcov_local[vidx + 15]);
					}
				}
			}
		}
		else if (color_channels == 6)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];
				AutoBuffer<const float*> sptr(6);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);
				sptr[3] = dataBorder[3].ptr<float>(j);
				sptr[4] = dataBorder[4].ptr<float>(j);
				sptr[5] = dataBorder[5].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//var (diag)
					const __m256 m0 = _mm256_loadu_ps(sptr[0]);
					const __m256 m1 = _mm256_loadu_ps(sptr[1]);
					const __m256 m2 = _mm256_loadu_ps(sptr[2]);
					const __m256 m3 = _mm256_loadu_ps(sptr[3]);
					const __m256 m4 = _mm256_loadu_ps(sptr[4]);
					const __m256 m5 = _mm256_loadu_ps(sptr[5]);
					mvar_local[0] = _mm256_fmadd_ps(m0, m0, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(m1, m0, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(m2, m0, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(m3, m0, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(m4, m0, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(m5, m0, mvar_local[5]);
					mvar_local[6] = _mm256_fmadd_ps(m1, m1, mvar_local[6]);
					mvar_local[7] = _mm256_fmadd_ps(m2, m1, mvar_local[7]);
					mvar_local[8] = _mm256_fmadd_ps(m3, m1, mvar_local[8]);
					mvar_local[9] = _mm256_fmadd_ps(m4, m1, mvar_local[9]);
					mvar_local[10] = _mm256_fmadd_ps(m5, m1, mvar_local[10]);
					mvar_local[11] = _mm256_fmadd_ps(m2, m2, mvar_local[11]);
					mvar_local[12] = _mm256_fmadd_ps(m3, m2, mvar_local[12]);
					mvar_local[13] = _mm256_fmadd_ps(m4, m2, mvar_local[13]);
					mvar_local[14] = _mm256_fmadd_ps(m5, m2, mvar_local[14]);
					mvar_local[15] = _mm256_fmadd_ps(m3, m3, mvar_local[15]);
					mvar_local[16] = _mm256_fmadd_ps(m4, m3, mvar_local[16]);
					mvar_local[17] = _mm256_fmadd_ps(m5, m3, mvar_local[17]);
					mvar_local[18] = _mm256_fmadd_ps(m4, m4, mvar_local[18]);
					mvar_local[19] = _mm256_fmadd_ps(m5, m4, mvar_local[19]);
					mvar_local[20] = _mm256_fmadd_ps(m5, m5, mvar_local[20]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 36)
					{
						const __m256 m0_1 = _mm256_loadu_ps(sptr[0] + first[k]);
						const __m256 m1_1 = _mm256_loadu_ps(sptr[1] + first[k]);
						const __m256 m2_1 = _mm256_loadu_ps(sptr[2] + first[k]);
						const __m256 m3_1 = _mm256_loadu_ps(sptr[3] + first[k]);
						const __m256 m4_1 = _mm256_loadu_ps(sptr[4] + first[k]);
						const __m256 m5_1 = _mm256_loadu_ps(sptr[5] + first[k]);
						const __m256 m0_2 = _mm256_loadu_ps(sptr[0] + second[k]);
						const __m256 m1_2 = _mm256_loadu_ps(sptr[1] + second[k]);
						const __m256 m2_2 = _mm256_loadu_ps(sptr[2] + second[k]);
						const __m256 m3_2 = _mm256_loadu_ps(sptr[3] + second[k]);
						const __m256 m4_2 = _mm256_loadu_ps(sptr[4] + second[k]);
						const __m256 m5_2 = _mm256_loadu_ps(sptr[5] + second[k]);

						mcov_local[vidx + 0] = _mm256_fmadd_ps(m0_1, m0_2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(m1_1, m0_2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(m2_1, m0_2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(m3_1, m0_2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(m4_1, m0_2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(m5_1, m0_2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(m0_1, m1_2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(m1_1, m1_2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(m2_1, m1_2, mcov_local[vidx + 8]);
						mcov_local[vidx + 9] = _mm256_fmadd_ps(m3_1, m1_2, mcov_local[vidx + 9]);
						mcov_local[vidx + 10] = _mm256_fmadd_ps(m4_1, m1_2, mcov_local[vidx + 10]);
						mcov_local[vidx + 11] = _mm256_fmadd_ps(m5_1, m1_2, mcov_local[vidx + 11]);
						mcov_local[vidx + 12] = _mm256_fmadd_ps(m0_1, m2_2, mcov_local[vidx + 12]);
						mcov_local[vidx + 13] = _mm256_fmadd_ps(m1_1, m2_2, mcov_local[vidx + 13]);
						mcov_local[vidx + 14] = _mm256_fmadd_ps(m2_1, m2_2, mcov_local[vidx + 14]);
						mcov_local[vidx + 15] = _mm256_fmadd_ps(m3_1, m2_2, mcov_local[vidx + 15]);
						mcov_local[vidx + 16] = _mm256_fmadd_ps(m4_1, m2_2, mcov_local[vidx + 16]);
						mcov_local[vidx + 17] = _mm256_fmadd_ps(m5_1, m2_2, mcov_local[vidx + 17]);
						mcov_local[vidx + 18] = _mm256_fmadd_ps(m0_1, m3_2, mcov_local[vidx + 18]);
						mcov_local[vidx + 19] = _mm256_fmadd_ps(m1_1, m3_2, mcov_local[vidx + 19]);
						mcov_local[vidx + 20] = _mm256_fmadd_ps(m2_1, m3_2, mcov_local[vidx + 20]);
						mcov_local[vidx + 21] = _mm256_fmadd_ps(m3_1, m3_2, mcov_local[vidx + 21]);
						mcov_local[vidx + 22] = _mm256_fmadd_ps(m4_1, m3_2, mcov_local[vidx + 22]);
						mcov_local[vidx + 23] = _mm256_fmadd_ps(m5_1, m3_2, mcov_local[vidx + 23]);
						mcov_local[vidx + 24] = _mm256_fmadd_ps(m0_1, m4_2, mcov_local[vidx + 24]);
						mcov_local[vidx + 25] = _mm256_fmadd_ps(m1_1, m4_2, mcov_local[vidx + 25]);
						mcov_local[vidx + 26] = _mm256_fmadd_ps(m2_1, m4_2, mcov_local[vidx + 26]);
						mcov_local[vidx + 27] = _mm256_fmadd_ps(m3_1, m4_2, mcov_local[vidx + 27]);
						mcov_local[vidx + 28] = _mm256_fmadd_ps(m4_1, m4_2, mcov_local[vidx + 28]);
						mcov_local[vidx + 29] = _mm256_fmadd_ps(m5_1, m4_2, mcov_local[vidx + 29]);
						mcov_local[vidx + 30] = _mm256_fmadd_ps(m0_1, m5_2, mcov_local[vidx + 30]);
						mcov_local[vidx + 31] = _mm256_fmadd_ps(m1_1, m5_2, mcov_local[vidx + 31]);
						mcov_local[vidx + 32] = _mm256_fmadd_ps(m2_1, m5_2, mcov_local[vidx + 32]);
						mcov_local[vidx + 33] = _mm256_fmadd_ps(m3_1, m5_2, mcov_local[vidx + 33]);
						mcov_local[vidx + 34] = _mm256_fmadd_ps(m4_1, m5_2, mcov_local[vidx + 34]);
						mcov_local[vidx + 35] = _mm256_fmadd_ps(m5_1, m5_2, mcov_local[vidx + 35]);
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
					sptr[3] += simd_step;
					sptr[4] += simd_step;
					sptr[5] += simd_step;
				}
				if (isRem)
				{
					//var (diag)
					const __m256 m0 = _mm256_maskload_ps(sptr[0], mask);
					const __m256 m1 = _mm256_maskload_ps(sptr[1], mask);
					const __m256 m2 = _mm256_maskload_ps(sptr[2], mask);
					const __m256 m3 = _mm256_maskload_ps(sptr[3], mask);
					const __m256 m4 = _mm256_maskload_ps(sptr[4], mask);
					const __m256 m5 = _mm256_maskload_ps(sptr[5], mask);

					mvar_local[0] = _mm256_fmadd_ps(m0, m0, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(m1, m0, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(m2, m0, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(m3, m0, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(m4, m0, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(m5, m0, mvar_local[5]);
					mvar_local[6] = _mm256_fmadd_ps(m1, m1, mvar_local[6]);
					mvar_local[7] = _mm256_fmadd_ps(m2, m1, mvar_local[7]);
					mvar_local[8] = _mm256_fmadd_ps(m3, m1, mvar_local[8]);
					mvar_local[9] = _mm256_fmadd_ps(m4, m1, mvar_local[9]);
					mvar_local[10] = _mm256_fmadd_ps(m5, m1, mvar_local[10]);
					mvar_local[11] = _mm256_fmadd_ps(m2, m2, mvar_local[11]);
					mvar_local[12] = _mm256_fmadd_ps(m3, m2, mvar_local[12]);
					mvar_local[13] = _mm256_fmadd_ps(m4, m2, mvar_local[13]);
					mvar_local[14] = _mm256_fmadd_ps(m5, m2, mvar_local[14]);
					mvar_local[15] = _mm256_fmadd_ps(m3, m3, mvar_local[15]);
					mvar_local[16] = _mm256_fmadd_ps(m4, m3, mvar_local[16]);
					mvar_local[17] = _mm256_fmadd_ps(m5, m3, mvar_local[17]);
					mvar_local[18] = _mm256_fmadd_ps(m4, m4, mvar_local[18]);
					mvar_local[19] = _mm256_fmadd_ps(m5, m4, mvar_local[19]);
					mvar_local[20] = _mm256_fmadd_ps(m5, m5, mvar_local[20]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 36)
					{
						const __m256 m0_1 = _mm256_maskload_ps(sptr[0] + first[k], mask);
						const __m256 m1_1 = _mm256_maskload_ps(sptr[1] + first[k], mask);
						const __m256 m2_1 = _mm256_maskload_ps(sptr[2] + first[k], mask);
						const __m256 m3_1 = _mm256_maskload_ps(sptr[3] + first[k], mask);
						const __m256 m4_1 = _mm256_maskload_ps(sptr[4] + first[k], mask);
						const __m256 m5_1 = _mm256_maskload_ps(sptr[5] + first[k], mask);
						const __m256 m0_2 = _mm256_maskload_ps(sptr[0] + second[k], mask);
						const __m256 m1_2 = _mm256_maskload_ps(sptr[1] + second[k], mask);
						const __m256 m2_2 = _mm256_maskload_ps(sptr[2] + second[k], mask);
						const __m256 m3_2 = _mm256_maskload_ps(sptr[3] + second[k], mask);
						const __m256 m4_2 = _mm256_maskload_ps(sptr[4] + second[k], mask);
						const __m256 m5_2 = _mm256_maskload_ps(sptr[5] + second[k], mask);

						mcov_local[vidx + 0] = _mm256_fmadd_ps(m0_1, m0_2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(m1_1, m0_2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(m2_1, m0_2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(m3_1, m0_2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(m4_1, m0_2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(m5_1, m0_2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(m0_1, m1_2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(m1_1, m1_2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(m2_1, m1_2, mcov_local[vidx + 8]);
						mcov_local[vidx + 9] = _mm256_fmadd_ps(m3_1, m1_2, mcov_local[vidx + 9]);
						mcov_local[vidx + 10] = _mm256_fmadd_ps(m4_1, m1_2, mcov_local[vidx + 10]);
						mcov_local[vidx + 11] = _mm256_fmadd_ps(m5_1, m1_2, mcov_local[vidx + 11]);
						mcov_local[vidx + 12] = _mm256_fmadd_ps(m0_1, m2_2, mcov_local[vidx + 12]);
						mcov_local[vidx + 13] = _mm256_fmadd_ps(m1_1, m2_2, mcov_local[vidx + 13]);
						mcov_local[vidx + 14] = _mm256_fmadd_ps(m2_1, m2_2, mcov_local[vidx + 14]);
						mcov_local[vidx + 15] = _mm256_fmadd_ps(m3_1, m2_2, mcov_local[vidx + 15]);
						mcov_local[vidx + 16] = _mm256_fmadd_ps(m4_1, m2_2, mcov_local[vidx + 16]);
						mcov_local[vidx + 17] = _mm256_fmadd_ps(m5_1, m2_2, mcov_local[vidx + 17]);
						mcov_local[vidx + 18] = _mm256_fmadd_ps(m0_1, m3_2, mcov_local[vidx + 18]);
						mcov_local[vidx + 19] = _mm256_fmadd_ps(m1_1, m3_2, mcov_local[vidx + 19]);
						mcov_local[vidx + 20] = _mm256_fmadd_ps(m2_1, m3_2, mcov_local[vidx + 20]);
						mcov_local[vidx + 21] = _mm256_fmadd_ps(m3_1, m3_2, mcov_local[vidx + 21]);
						mcov_local[vidx + 22] = _mm256_fmadd_ps(m4_1, m3_2, mcov_local[vidx + 22]);
						mcov_local[vidx + 23] = _mm256_fmadd_ps(m5_1, m3_2, mcov_local[vidx + 23]);
						mcov_local[vidx + 24] = _mm256_fmadd_ps(m0_1, m4_2, mcov_local[vidx + 24]);
						mcov_local[vidx + 25] = _mm256_fmadd_ps(m1_1, m4_2, mcov_local[vidx + 25]);
						mcov_local[vidx + 26] = _mm256_fmadd_ps(m2_1, m4_2, mcov_local[vidx + 26]);
						mcov_local[vidx + 27] = _mm256_fmadd_ps(m3_1, m4_2, mcov_local[vidx + 27]);
						mcov_local[vidx + 28] = _mm256_fmadd_ps(m4_1, m4_2, mcov_local[vidx + 28]);
						mcov_local[vidx + 29] = _mm256_fmadd_ps(m5_1, m4_2, mcov_local[vidx + 29]);
						mcov_local[vidx + 30] = _mm256_fmadd_ps(m0_1, m5_2, mcov_local[vidx + 30]);
						mcov_local[vidx + 31] = _mm256_fmadd_ps(m1_1, m5_2, mcov_local[vidx + 31]);
						mcov_local[vidx + 32] = _mm256_fmadd_ps(m2_1, m5_2, mcov_local[vidx + 32]);
						mcov_local[vidx + 33] = _mm256_fmadd_ps(m3_1, m5_2, mcov_local[vidx + 33]);
						mcov_local[vidx + 34] = _mm256_fmadd_ps(m4_1, m5_2, mcov_local[vidx + 34]);
						mcov_local[vidx + 35] = _mm256_fmadd_ps(m5_1, m5_2, mcov_local[vidx + 35]);
					}
				}
			}
		}
		else
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];

				AutoBuffer<const float*> sptr(color_channels);
				AutoBuffer<__m256> mv1(color_channels);
				AutoBuffer<__m256> mv2(color_channels);

				for (int c = 0; c < color_channels; c++)
				{
					sptr[c] = dataBorder[c].ptr<float>(j);
				}

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//var (diag)
					for (int c = 0; c < color_channels; c++)mv1[c] = _mm256_loadu_ps(sptr[c]);
					for (int cy = 0, vidx = 0; cy < color_channels; cy++)
					{
						for (int cx = cy; cx < color_channels; cx++)
						{
							mvar_local[vidx] = _mm256_fmadd_ps(mv1[cx], mv1[cy], mvar_local[vidx]);
							vidx++;
						}
					}

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++)
					{
						for (int c = 0; c < color_channels; c++)
						{
							mv1[c] = _mm256_loadu_ps(sptr[c] + first[k]);
							mv2[c] = _mm256_loadu_ps(sptr[c] + second[k]);
						}
						for (int cy = 0; cy < color_channels; cy++)
						{
							for (int cx = 0; cx < color_channels; cx++)
							{
								mcov_local[vidx] = _mm256_fmadd_ps(mv1[cx], mv2[cy], mcov_local[vidx]);
								vidx++;
							}
						}
					}

					for (int c = 0; c < color_channels; c++)
					{
						sptr[c] += simd_step;
					}
				}
				if (isRem)
				{
					//var (diag)
					for (int c = 0; c < color_channels; c++)mv1[c] = _mm256_maskload_ps(sptr[c], mask);
					for (int cy = 0, vidx = 0; cy < color_channels; cy++)
					{
						for (int cx = cy; cx < color_channels; cx++)
						{
							mvar_local[vidx] = _mm256_fmadd_ps(mv1[cx], mv1[cy], mvar_local[vidx]);
							vidx++;
						}
					}

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++)
					{
						for (int c = 0; c < color_channels; c++)
						{
							mv1[c] = _mm256_maskload_ps(sptr[c] + first[k], mask);
							mv2[c] = _mm256_maskload_ps(sptr[c] + second[k], mask);
						}
						for (int cy = 0; cy < color_channels; cy++)
						{
							for (int cx = 0; cx < color_channels; cx++)
							{
								mcov_local[vidx] = _mm256_fmadd_ps(mv1[cx], mv2[cy], mcov_local[vidx]);
								vidx++;
							}
						}
					}
				}
			}
		}
	}
	else
	{
		rcc.getIndex(first, second, data[0].cols);
		const int simd_end_x = get_simd_floor(data[0].cols - 2 * patch_rad, simd_step) + patch_rad;
		normalSize = 1.0 / ((data[0].rows - 2 * patch_rad) * (simd_end_x - patch_rad));
		const int yend = data[0].rows - 2 * patch_rad;
		if (color_channels == 1)
		{
#pragma omp parallel for SCHEDULE
			for (int y = 0; y < yend; y++)
			{
				const int j = y + patch_rad;
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[(directionElementSize + 1) * tindex];
				__m256* mvar_local = &mbuff[(directionElementSize + 1) * tindex + directionElementSize];
				const float* sptr = data[0].ptr<float>(j, patch_rad);

				for (int i = patch_rad; i < simd_end_x; i += simd_step)
				{
					const __m256 v = _mm256_loadu_ps(sptr);
					mvar_local[0] = _mm256_fmadd_ps(v, v, mvar_local[0]);

					for (int k = 0; k < directionElementSize; k++)
					{
						mcov_local[k] = _mm256_fmadd_ps(_mm256_loadu_ps(sptr + first[k]), _mm256_loadu_ps(sptr + second[k]), mcov_local[k]);
					}

					sptr += simd_step;
				}
			}
		}
		else
		{
#pragma omp parallel for SCHEDULE
			for (int y = 0; y < yend; y++)
			{
				const int j = y + patch_rad;
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];
				AutoBuffer<const float*> sptr(color_channels);
				for (int c = 0; c < color_channels; c++)
				{
					sptr[c] = data[c].ptr<float>(j, patch_rad);
				}

				for (int i = patch_rad; i < simd_end_x; i += simd_step)
				{
					//diag (var)
					for (int cy = 0, vidx = 0; cy < color_channels; cy++)
					{
						for (int cx = 0; cx < color_channels; cx++)
						{
							mvar_local[vidx] = _mm256_fmadd_ps(_mm256_loadu_ps(sptr[cx]), _mm256_loadu_ps(sptr[cy]), mvar_local[vidx]);
							vidx++;
						}
					}

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++)
					{
						for (int cy = 0; cy < color_channels; cy++)
						{
							for (int cx = 0; cx < color_channels; cx++)
							{
								mcov_local[vidx] = _mm256_fmadd_ps(_mm256_loadu_ps(sptr[cx] + first[k]), _mm256_loadu_ps(sptr[cy] + second[k]), mcov_local[vidx]);
								vidx++;
							}
						}
					}

					for (int c = 0; c < color_channels; c++)
					{
						sptr[c] += simd_step;
					}
				}
			}
		}
	}

	//reduction
	vector<double> varElem(computeVarElementSize);
	vector<double> covElem(computeCovElementSize);
	for (double v : varElem) v = 0.0;
	for (double c : covElem) c = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		const int sstep = computeElementSize * t;
		__m256* mcov = &mbuff[sstep];
		for (int i = 0; i < computeCovElementSize; i++)
		{
			covElem[i] += _mm256_reduceadd_pspd(mcov[i]);
		}

		__m256* mvar = &mbuff[sstep + computeCovElementSize];
		for (int i = 0; i < computeVarElementSize; i++)
		{
			varElem[i] += _mm256_reduceadd_pspd(mvar[i]);
		}
	}

	setCovRep(meanForCov, varElem, destCovarianceMatrix, covElem, rcc.getSharedSet(), normalSize);

	_mm_free(mbuff);
}

void CalcPatchCovarMatrix::simdOMPCov_RepCenterRepElement32FCn(const vector<Mat>& src_, Mat& destCovarianceMatrix, const CenterMethod method, const float constant_sub)
{
	bool isBorder = true;
	dim = (2 * patch_rad + 1) * (2 * patch_rad + 1) * color_channels;
	//cout << "simdOMPCovRepCov32F: "<<color_channels<<","<<dim << endl;

	RepresentiveCovarianceComputer rcc(color_channels, patch_rad);

	const int simd_step = 8;
	data.resize(color_channels);
	dataBorder.resize(color_channels);

	double normalSize = 0.0;

	vector<double> meanForCov(color_channels);

	for (int c = 0; c < color_channels; c++)
	{
		if (method == CenterMethod::MEAN)
		{
			//cout << "Rep mean" << endl; 
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
			meanForCov[c] = 0.0;
		}
		else if (method == CenterMethod::CONST_)
		{
			//cout << "Rep const" << endl;
			//subtract(src_[c], constant_sub, data[c]);
			sub_const(src_[c], constant_sub, data[c]);
			meanForCov[c] = 0.0;
		}
		else if (method == CenterMethod::NO)
		{
			//cout << "Rep nosub" << endl;
			//if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			//else meanForCov[0] = 0.0;
			data[c] = src_[c];
			meanForCov[0] = 0.0;
		}
		else
		{
			cout << int(method) << endl;
			cout << "No support method (simdOMPCovRepCov32F)" << endl;
		}
		if (isBorder)
		{
			copyMakeBorder(data[c], dataBorder[c], patch_rad, patch_rad, patch_rad, patch_rad, border);
		}
	}

	const int thread_max = omp_get_max_threads();

	const int directionElementSize = 2 * (2 * patch_rad + 1) * (2 * patch_rad);
	const int computeCovElementSize = directionElementSize * color_channels * color_channels;
	const int computeVarElementSize = sum_from(color_channels);
	const int computeElementSize = computeCovElementSize + computeVarElementSize;

	AutoBuffer<int> first(directionElementSize);
	AutoBuffer<int> second(directionElementSize);

	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * computeElementSize * thread_max, AVX_ALIGN);
	for (int i = 0; i < computeElementSize * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

	if (isBorder)
	{
		rcc.getIndexBorder(first, second, dataBorder[0].cols);
		const int simd_end = get_simd_floor(src_[0].cols, simd_step);
		const __m256i mask = get_simd_residualmask_epi32(src_[0].cols);
		const bool isRem = (src_[0].cols == simd_end) ? false : true;
		normalSize = 1.0 / ((src_[0].rows) * (src_[0].cols));

		if (color_channels == 1)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];

				const float* sptr = dataBorder[0].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					mvar_local[0] = _mm256_fmadd_ps(_mm256_loadu_ps(sptr), _mm256_loadu_ps(sptr), mvar_local[0]);

					for (int k = 0; k < directionElementSize; k++)
					{
						mcov_local[k] = _mm256_fmadd_ps(_mm256_loadu_ps(sptr + first[k]), _mm256_loadu_ps(sptr + second[k]), mcov_local[k]);
					}
					sptr += simd_step;
				}
				if (isRem)
				{
					mvar_local[0] = _mm256_fmadd_ps(_mm256_maskload_ps(sptr, mask), _mm256_maskload_ps(sptr, mask), mvar_local[0]);

					for (int k = 0; k < directionElementSize; k++)
					{
						mcov_local[k] = _mm256_fmadd_ps(_mm256_maskload_ps(sptr + first[k], mask), _mm256_maskload_ps(sptr + second[k], mask), mcov_local[k]);
					}
				}
			}
		}
		else if (color_channels == 2)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];
				AutoBuffer<const float*> sptr(2);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//var (diag)
					const __m256 mg = _mm256_loadu_ps(sptr[0]);
					const __m256 ma = _mm256_loadu_ps(sptr[1]);
					mvar_local[0] = _mm256_fmadd_ps(mg, mg, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(ma, mg, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(ma, ma, mvar_local[2]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 4)
					{
						const __m256 mg1 = _mm256_loadu_ps(sptr[0] + first[k]);
						const __m256 ma1 = _mm256_loadu_ps(sptr[1] + first[k]);
						const __m256 mg2 = _mm256_loadu_ps(sptr[0] + second[k]);
						const __m256 ma2 = _mm256_loadu_ps(sptr[1] + second[k]);
						mcov_local[vidx + 0] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(ma1, mg2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mg1, ma2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(ma1, ma2, mcov_local[vidx + 3]);
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
				}
				if (isRem)
				{
					//var (diag)
					const __m256 mg = _mm256_maskload_ps(sptr[0], mask);
					const __m256 ma = _mm256_maskload_ps(sptr[1], mask);
					mvar_local[0] = _mm256_fmadd_ps(mg, mg, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(ma, mg, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(ma, ma, mvar_local[2]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 4)
					{
						const __m256 mg1 = _mm256_maskload_ps(sptr[0] + first[k], mask);
						const __m256 ma1 = _mm256_maskload_ps(sptr[1] + first[k], mask);
						const __m256 mg2 = _mm256_maskload_ps(sptr[0] + second[k], mask);
						const __m256 ma2 = _mm256_maskload_ps(sptr[1] + second[k], mask);
						mcov_local[vidx + 0] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(ma1, mg2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mg1, ma2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(ma1, ma2, mcov_local[vidx + 3]);
					}
				}
			}
		}
		else if (color_channels == 3)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];
				AutoBuffer<const float*> sptr(3);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//var (diag)
					const __m256 mb = _mm256_loadu_ps(sptr[0]);
					const __m256 mg = _mm256_loadu_ps(sptr[1]);
					const __m256 mr = _mm256_loadu_ps(sptr[2]);
					mvar_local[0] = _mm256_fmadd_ps(mb, mb, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(mg, mb, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(mr, mb, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(mg, mg, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(mr, mg, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(mr, mr, mvar_local[5]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 9)
					{
						const __m256 mb1 = _mm256_loadu_ps(sptr[0] + first[k]);
						const __m256 mg1 = _mm256_loadu_ps(sptr[1] + first[k]);
						const __m256 mr1 = _mm256_loadu_ps(sptr[2] + first[k]);
						const __m256 mb2 = _mm256_loadu_ps(sptr[0] + second[k]);
						const __m256 mg2 = _mm256_loadu_ps(sptr[1] + second[k]);
						const __m256 mr2 = _mm256_loadu_ps(sptr[2] + second[k]);
						mcov_local[vidx + 0] = _mm256_fmadd_ps(mb1, mb2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(mg1, mb2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mr1, mb2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(mb1, mg2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(mr1, mg2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(mb1, mr2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(mg1, mr2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(mr1, mr2, mcov_local[vidx + 8]);
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
				}
				if (isRem)
				{
					//var (diag)
					const __m256 mb = _mm256_maskload_ps(sptr[0], mask);
					const __m256 mg = _mm256_maskload_ps(sptr[1], mask);
					const __m256 mr = _mm256_maskload_ps(sptr[2], mask);
					mvar_local[0] = _mm256_fmadd_ps(mb, mb, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(mg, mb, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(mr, mb, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(mg, mg, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(mr, mg, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(mr, mr, mvar_local[5]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 9)
					{
						const __m256 mb1 = _mm256_maskload_ps(sptr[0] + first[k], mask);
						const __m256 mg1 = _mm256_maskload_ps(sptr[1] + first[k], mask);
						const __m256 mr1 = _mm256_maskload_ps(sptr[2] + first[k], mask);
						const __m256 mb2 = _mm256_maskload_ps(sptr[0] + second[k], mask);
						const __m256 mg2 = _mm256_maskload_ps(sptr[1] + second[k], mask);
						const __m256 mr2 = _mm256_maskload_ps(sptr[2] + second[k], mask);
						mcov_local[vidx + 0] = _mm256_fmadd_ps(mb1, mb2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(mg1, mb2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mr1, mb2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(mb1, mg2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(mr1, mg2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(mb1, mr2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(mg1, mr2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(mr1, mr2, mcov_local[vidx + 8]);
					}
				}
			}
		}
		else if (color_channels == 4)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];
				AutoBuffer<const float*> sptr(4);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);
				sptr[3] = dataBorder[3].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//var (diag)
					const __m256 mb = _mm256_loadu_ps(sptr[0]);
					const __m256 mg = _mm256_loadu_ps(sptr[1]);
					const __m256 mr = _mm256_loadu_ps(sptr[2]);
					const __m256 ma = _mm256_loadu_ps(sptr[3]);
					mvar_local[0] = _mm256_fmadd_ps(mb, mb, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(mg, mb, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(mr, mb, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(ma, mb, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(mg, mg, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(mr, mg, mvar_local[5]);
					mvar_local[6] = _mm256_fmadd_ps(ma, mg, mvar_local[6]);
					mvar_local[7] = _mm256_fmadd_ps(mr, mr, mvar_local[7]);
					mvar_local[8] = _mm256_fmadd_ps(ma, mr, mvar_local[8]);
					mvar_local[9] = _mm256_fmadd_ps(ma, ma, mvar_local[9]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 16)
					{
						const __m256 mb1 = _mm256_loadu_ps(sptr[0] + first[k]);
						const __m256 mg1 = _mm256_loadu_ps(sptr[1] + first[k]);
						const __m256 mr1 = _mm256_loadu_ps(sptr[2] + first[k]);
						const __m256 ma1 = _mm256_loadu_ps(sptr[3] + first[k]);
						const __m256 mb2 = _mm256_loadu_ps(sptr[0] + second[k]);
						const __m256 mg2 = _mm256_loadu_ps(sptr[1] + second[k]);
						const __m256 mr2 = _mm256_loadu_ps(sptr[2] + second[k]);
						const __m256 ma2 = _mm256_loadu_ps(sptr[3] + second[k]);

						mcov_local[vidx + 0] = _mm256_fmadd_ps(mb1, mb2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(mg1, mb2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mr1, mb2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(ma1, mb2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(mb1, mg2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(mr1, mg2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(ma1, mg2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(mb1, mr2, mcov_local[vidx + 8]);
						mcov_local[vidx + 9] = _mm256_fmadd_ps(mg1, mr2, mcov_local[vidx + 9]);
						mcov_local[vidx + 10] = _mm256_fmadd_ps(mr1, mr2, mcov_local[vidx + 10]);
						mcov_local[vidx + 11] = _mm256_fmadd_ps(ma1, mr2, mcov_local[vidx + 11]);
						mcov_local[vidx + 12] = _mm256_fmadd_ps(mb1, ma2, mcov_local[vidx + 12]);
						mcov_local[vidx + 13] = _mm256_fmadd_ps(mg1, ma2, mcov_local[vidx + 13]);
						mcov_local[vidx + 14] = _mm256_fmadd_ps(mr1, ma2, mcov_local[vidx + 14]);
						mcov_local[vidx + 15] = _mm256_fmadd_ps(ma1, ma2, mcov_local[vidx + 15]);
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
					sptr[3] += simd_step;
				}
				if (isRem)
				{
					//var (diag)
					const __m256 mb = _mm256_maskload_ps(sptr[0], mask);
					const __m256 mg = _mm256_maskload_ps(sptr[1], mask);
					const __m256 mr = _mm256_maskload_ps(sptr[2], mask);
					const __m256 ma = _mm256_maskload_ps(sptr[3], mask);
					mvar_local[0] = _mm256_fmadd_ps(mb, mb, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(mg, mb, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(mr, mb, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(ma, mb, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(mg, mg, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(mr, mg, mvar_local[5]);
					mvar_local[6] = _mm256_fmadd_ps(ma, mg, mvar_local[6]);
					mvar_local[7] = _mm256_fmadd_ps(mr, mr, mvar_local[7]);
					mvar_local[8] = _mm256_fmadd_ps(ma, mr, mvar_local[8]);
					mvar_local[9] = _mm256_fmadd_ps(ma, ma, mvar_local[9]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 16)
					{
						const __m256 mb1 = _mm256_maskload_ps(sptr[0] + first[k], mask);
						const __m256 mg1 = _mm256_maskload_ps(sptr[1] + first[k], mask);
						const __m256 mr1 = _mm256_maskload_ps(sptr[2] + first[k], mask);
						const __m256 ma1 = _mm256_maskload_ps(sptr[3] + first[k], mask);
						const __m256 mb2 = _mm256_maskload_ps(sptr[0] + second[k], mask);
						const __m256 mg2 = _mm256_maskload_ps(sptr[1] + second[k], mask);
						const __m256 mr2 = _mm256_maskload_ps(sptr[2] + second[k], mask);
						const __m256 ma2 = _mm256_maskload_ps(sptr[3] + second[k], mask);

						mcov_local[vidx + 0] = _mm256_fmadd_ps(mb1, mb2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(mg1, mb2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(mr1, mb2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(ma1, mb2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(mb1, mg2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(mg1, mg2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(mr1, mg2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(ma1, mg2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(mb1, mr2, mcov_local[vidx + 8]);
						mcov_local[vidx + 9] = _mm256_fmadd_ps(mg1, mr2, mcov_local[vidx + 9]);
						mcov_local[vidx + 10] = _mm256_fmadd_ps(mr1, mr2, mcov_local[vidx + 10]);
						mcov_local[vidx + 11] = _mm256_fmadd_ps(ma1, mr2, mcov_local[vidx + 11]);
						mcov_local[vidx + 12] = _mm256_fmadd_ps(mb1, ma2, mcov_local[vidx + 12]);
						mcov_local[vidx + 13] = _mm256_fmadd_ps(mg1, ma2, mcov_local[vidx + 13]);
						mcov_local[vidx + 14] = _mm256_fmadd_ps(mr1, ma2, mcov_local[vidx + 14]);
						mcov_local[vidx + 15] = _mm256_fmadd_ps(ma1, ma2, mcov_local[vidx + 15]);
					}
				}
			}
		}
		else if (color_channels == 6)
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];
				AutoBuffer<const float*> sptr(6);
				sptr[0] = dataBorder[0].ptr<float>(j);
				sptr[1] = dataBorder[1].ptr<float>(j);
				sptr[2] = dataBorder[2].ptr<float>(j);
				sptr[3] = dataBorder[3].ptr<float>(j);
				sptr[4] = dataBorder[4].ptr<float>(j);
				sptr[5] = dataBorder[5].ptr<float>(j);

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//var (diag)
					const __m256 m0 = _mm256_loadu_ps(sptr[0]);
					const __m256 m1 = _mm256_loadu_ps(sptr[1]);
					const __m256 m2 = _mm256_loadu_ps(sptr[2]);
					const __m256 m3 = _mm256_loadu_ps(sptr[3]);
					const __m256 m4 = _mm256_loadu_ps(sptr[4]);
					const __m256 m5 = _mm256_loadu_ps(sptr[5]);
					mvar_local[0] = _mm256_fmadd_ps(m0, m0, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(m1, m0, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(m2, m0, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(m3, m0, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(m4, m0, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(m5, m0, mvar_local[5]);
					mvar_local[6] = _mm256_fmadd_ps(m1, m1, mvar_local[6]);
					mvar_local[7] = _mm256_fmadd_ps(m2, m1, mvar_local[7]);
					mvar_local[8] = _mm256_fmadd_ps(m3, m1, mvar_local[8]);
					mvar_local[9] = _mm256_fmadd_ps(m4, m1, mvar_local[9]);
					mvar_local[10] = _mm256_fmadd_ps(m5, m1, mvar_local[10]);
					mvar_local[11] = _mm256_fmadd_ps(m2, m2, mvar_local[11]);
					mvar_local[12] = _mm256_fmadd_ps(m3, m2, mvar_local[12]);
					mvar_local[13] = _mm256_fmadd_ps(m4, m2, mvar_local[13]);
					mvar_local[14] = _mm256_fmadd_ps(m5, m2, mvar_local[14]);
					mvar_local[15] = _mm256_fmadd_ps(m3, m3, mvar_local[15]);
					mvar_local[16] = _mm256_fmadd_ps(m4, m3, mvar_local[16]);
					mvar_local[17] = _mm256_fmadd_ps(m5, m3, mvar_local[17]);
					mvar_local[18] = _mm256_fmadd_ps(m4, m4, mvar_local[18]);
					mvar_local[19] = _mm256_fmadd_ps(m5, m4, mvar_local[19]);
					mvar_local[20] = _mm256_fmadd_ps(m5, m5, mvar_local[20]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 36)
					{
						const __m256 m0_1 = _mm256_loadu_ps(sptr[0] + first[k]);
						const __m256 m1_1 = _mm256_loadu_ps(sptr[1] + first[k]);
						const __m256 m2_1 = _mm256_loadu_ps(sptr[2] + first[k]);
						const __m256 m3_1 = _mm256_loadu_ps(sptr[3] + first[k]);
						const __m256 m4_1 = _mm256_loadu_ps(sptr[4] + first[k]);
						const __m256 m5_1 = _mm256_loadu_ps(sptr[5] + first[k]);
						const __m256 m0_2 = _mm256_loadu_ps(sptr[0] + second[k]);
						const __m256 m1_2 = _mm256_loadu_ps(sptr[1] + second[k]);
						const __m256 m2_2 = _mm256_loadu_ps(sptr[2] + second[k]);
						const __m256 m3_2 = _mm256_loadu_ps(sptr[3] + second[k]);
						const __m256 m4_2 = _mm256_loadu_ps(sptr[4] + second[k]);
						const __m256 m5_2 = _mm256_loadu_ps(sptr[5] + second[k]);

						mcov_local[vidx + 0] = _mm256_fmadd_ps(m0_1, m0_2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(m1_1, m0_2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(m2_1, m0_2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(m3_1, m0_2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(m4_1, m0_2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(m5_1, m0_2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(m0_1, m1_2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(m1_1, m1_2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(m2_1, m1_2, mcov_local[vidx + 8]);
						mcov_local[vidx + 9] = _mm256_fmadd_ps(m3_1, m1_2, mcov_local[vidx + 9]);
						mcov_local[vidx + 10] = _mm256_fmadd_ps(m4_1, m1_2, mcov_local[vidx + 10]);
						mcov_local[vidx + 11] = _mm256_fmadd_ps(m5_1, m1_2, mcov_local[vidx + 11]);
						mcov_local[vidx + 12] = _mm256_fmadd_ps(m0_1, m2_2, mcov_local[vidx + 12]);
						mcov_local[vidx + 13] = _mm256_fmadd_ps(m1_1, m2_2, mcov_local[vidx + 13]);
						mcov_local[vidx + 14] = _mm256_fmadd_ps(m2_1, m2_2, mcov_local[vidx + 14]);
						mcov_local[vidx + 15] = _mm256_fmadd_ps(m3_1, m2_2, mcov_local[vidx + 15]);
						mcov_local[vidx + 16] = _mm256_fmadd_ps(m4_1, m2_2, mcov_local[vidx + 16]);
						mcov_local[vidx + 17] = _mm256_fmadd_ps(m5_1, m2_2, mcov_local[vidx + 17]);
						mcov_local[vidx + 18] = _mm256_fmadd_ps(m0_1, m3_2, mcov_local[vidx + 18]);
						mcov_local[vidx + 19] = _mm256_fmadd_ps(m1_1, m3_2, mcov_local[vidx + 19]);
						mcov_local[vidx + 20] = _mm256_fmadd_ps(m2_1, m3_2, mcov_local[vidx + 20]);
						mcov_local[vidx + 21] = _mm256_fmadd_ps(m3_1, m3_2, mcov_local[vidx + 21]);
						mcov_local[vidx + 22] = _mm256_fmadd_ps(m4_1, m3_2, mcov_local[vidx + 22]);
						mcov_local[vidx + 23] = _mm256_fmadd_ps(m5_1, m3_2, mcov_local[vidx + 23]);
						mcov_local[vidx + 24] = _mm256_fmadd_ps(m0_1, m4_2, mcov_local[vidx + 24]);
						mcov_local[vidx + 25] = _mm256_fmadd_ps(m1_1, m4_2, mcov_local[vidx + 25]);
						mcov_local[vidx + 26] = _mm256_fmadd_ps(m2_1, m4_2, mcov_local[vidx + 26]);
						mcov_local[vidx + 27] = _mm256_fmadd_ps(m3_1, m4_2, mcov_local[vidx + 27]);
						mcov_local[vidx + 28] = _mm256_fmadd_ps(m4_1, m4_2, mcov_local[vidx + 28]);
						mcov_local[vidx + 29] = _mm256_fmadd_ps(m5_1, m4_2, mcov_local[vidx + 29]);
						mcov_local[vidx + 30] = _mm256_fmadd_ps(m0_1, m5_2, mcov_local[vidx + 30]);
						mcov_local[vidx + 31] = _mm256_fmadd_ps(m1_1, m5_2, mcov_local[vidx + 31]);
						mcov_local[vidx + 32] = _mm256_fmadd_ps(m2_1, m5_2, mcov_local[vidx + 32]);
						mcov_local[vidx + 33] = _mm256_fmadd_ps(m3_1, m5_2, mcov_local[vidx + 33]);
						mcov_local[vidx + 34] = _mm256_fmadd_ps(m4_1, m5_2, mcov_local[vidx + 34]);
						mcov_local[vidx + 35] = _mm256_fmadd_ps(m5_1, m5_2, mcov_local[vidx + 35]);
					}

					sptr[0] += simd_step;
					sptr[1] += simd_step;
					sptr[2] += simd_step;
					sptr[3] += simd_step;
					sptr[4] += simd_step;
					sptr[5] += simd_step;
				}
				if (isRem)
				{
					//var (diag)
					const __m256 m0 = _mm256_maskload_ps(sptr[0], mask);
					const __m256 m1 = _mm256_maskload_ps(sptr[1], mask);
					const __m256 m2 = _mm256_maskload_ps(sptr[2], mask);
					const __m256 m3 = _mm256_maskload_ps(sptr[3], mask);
					const __m256 m4 = _mm256_maskload_ps(sptr[4], mask);
					const __m256 m5 = _mm256_maskload_ps(sptr[5], mask);

					mvar_local[0] = _mm256_fmadd_ps(m0, m0, mvar_local[0]);
					mvar_local[1] = _mm256_fmadd_ps(m1, m0, mvar_local[1]);
					mvar_local[2] = _mm256_fmadd_ps(m2, m0, mvar_local[2]);
					mvar_local[3] = _mm256_fmadd_ps(m3, m0, mvar_local[3]);
					mvar_local[4] = _mm256_fmadd_ps(m4, m0, mvar_local[4]);
					mvar_local[5] = _mm256_fmadd_ps(m5, m0, mvar_local[5]);
					mvar_local[6] = _mm256_fmadd_ps(m1, m1, mvar_local[6]);
					mvar_local[7] = _mm256_fmadd_ps(m2, m1, mvar_local[7]);
					mvar_local[8] = _mm256_fmadd_ps(m3, m1, mvar_local[8]);
					mvar_local[9] = _mm256_fmadd_ps(m4, m1, mvar_local[9]);
					mvar_local[10] = _mm256_fmadd_ps(m5, m1, mvar_local[10]);
					mvar_local[11] = _mm256_fmadd_ps(m2, m2, mvar_local[11]);
					mvar_local[12] = _mm256_fmadd_ps(m3, m2, mvar_local[12]);
					mvar_local[13] = _mm256_fmadd_ps(m4, m2, mvar_local[13]);
					mvar_local[14] = _mm256_fmadd_ps(m5, m2, mvar_local[14]);
					mvar_local[15] = _mm256_fmadd_ps(m3, m3, mvar_local[15]);
					mvar_local[16] = _mm256_fmadd_ps(m4, m3, mvar_local[16]);
					mvar_local[17] = _mm256_fmadd_ps(m5, m3, mvar_local[17]);
					mvar_local[18] = _mm256_fmadd_ps(m4, m4, mvar_local[18]);
					mvar_local[19] = _mm256_fmadd_ps(m5, m4, mvar_local[19]);
					mvar_local[20] = _mm256_fmadd_ps(m5, m5, mvar_local[20]);

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++, vidx += 36)
					{
						const __m256 m0_1 = _mm256_maskload_ps(sptr[0] + first[k], mask);
						const __m256 m1_1 = _mm256_maskload_ps(sptr[1] + first[k], mask);
						const __m256 m2_1 = _mm256_maskload_ps(sptr[2] + first[k], mask);
						const __m256 m3_1 = _mm256_maskload_ps(sptr[3] + first[k], mask);
						const __m256 m4_1 = _mm256_maskload_ps(sptr[4] + first[k], mask);
						const __m256 m5_1 = _mm256_maskload_ps(sptr[5] + first[k], mask);
						const __m256 m0_2 = _mm256_maskload_ps(sptr[0] + second[k], mask);
						const __m256 m1_2 = _mm256_maskload_ps(sptr[1] + second[k], mask);
						const __m256 m2_2 = _mm256_maskload_ps(sptr[2] + second[k], mask);
						const __m256 m3_2 = _mm256_maskload_ps(sptr[3] + second[k], mask);
						const __m256 m4_2 = _mm256_maskload_ps(sptr[4] + second[k], mask);
						const __m256 m5_2 = _mm256_maskload_ps(sptr[5] + second[k], mask);

						mcov_local[vidx + 0] = _mm256_fmadd_ps(m0_1, m0_2, mcov_local[vidx + 0]);
						mcov_local[vidx + 1] = _mm256_fmadd_ps(m1_1, m0_2, mcov_local[vidx + 1]);
						mcov_local[vidx + 2] = _mm256_fmadd_ps(m2_1, m0_2, mcov_local[vidx + 2]);
						mcov_local[vidx + 3] = _mm256_fmadd_ps(m3_1, m0_2, mcov_local[vidx + 3]);
						mcov_local[vidx + 4] = _mm256_fmadd_ps(m4_1, m0_2, mcov_local[vidx + 4]);
						mcov_local[vidx + 5] = _mm256_fmadd_ps(m5_1, m0_2, mcov_local[vidx + 5]);
						mcov_local[vidx + 6] = _mm256_fmadd_ps(m0_1, m1_2, mcov_local[vidx + 6]);
						mcov_local[vidx + 7] = _mm256_fmadd_ps(m1_1, m1_2, mcov_local[vidx + 7]);
						mcov_local[vidx + 8] = _mm256_fmadd_ps(m2_1, m1_2, mcov_local[vidx + 8]);
						mcov_local[vidx + 9] = _mm256_fmadd_ps(m3_1, m1_2, mcov_local[vidx + 9]);
						mcov_local[vidx + 10] = _mm256_fmadd_ps(m4_1, m1_2, mcov_local[vidx + 10]);
						mcov_local[vidx + 11] = _mm256_fmadd_ps(m5_1, m1_2, mcov_local[vidx + 11]);
						mcov_local[vidx + 12] = _mm256_fmadd_ps(m0_1, m2_2, mcov_local[vidx + 12]);
						mcov_local[vidx + 13] = _mm256_fmadd_ps(m1_1, m2_2, mcov_local[vidx + 13]);
						mcov_local[vidx + 14] = _mm256_fmadd_ps(m2_1, m2_2, mcov_local[vidx + 14]);
						mcov_local[vidx + 15] = _mm256_fmadd_ps(m3_1, m2_2, mcov_local[vidx + 15]);
						mcov_local[vidx + 16] = _mm256_fmadd_ps(m4_1, m2_2, mcov_local[vidx + 16]);
						mcov_local[vidx + 17] = _mm256_fmadd_ps(m5_1, m2_2, mcov_local[vidx + 17]);
						mcov_local[vidx + 18] = _mm256_fmadd_ps(m0_1, m3_2, mcov_local[vidx + 18]);
						mcov_local[vidx + 19] = _mm256_fmadd_ps(m1_1, m3_2, mcov_local[vidx + 19]);
						mcov_local[vidx + 20] = _mm256_fmadd_ps(m2_1, m3_2, mcov_local[vidx + 20]);
						mcov_local[vidx + 21] = _mm256_fmadd_ps(m3_1, m3_2, mcov_local[vidx + 21]);
						mcov_local[vidx + 22] = _mm256_fmadd_ps(m4_1, m3_2, mcov_local[vidx + 22]);
						mcov_local[vidx + 23] = _mm256_fmadd_ps(m5_1, m3_2, mcov_local[vidx + 23]);
						mcov_local[vidx + 24] = _mm256_fmadd_ps(m0_1, m4_2, mcov_local[vidx + 24]);
						mcov_local[vidx + 25] = _mm256_fmadd_ps(m1_1, m4_2, mcov_local[vidx + 25]);
						mcov_local[vidx + 26] = _mm256_fmadd_ps(m2_1, m4_2, mcov_local[vidx + 26]);
						mcov_local[vidx + 27] = _mm256_fmadd_ps(m3_1, m4_2, mcov_local[vidx + 27]);
						mcov_local[vidx + 28] = _mm256_fmadd_ps(m4_1, m4_2, mcov_local[vidx + 28]);
						mcov_local[vidx + 29] = _mm256_fmadd_ps(m5_1, m4_2, mcov_local[vidx + 29]);
						mcov_local[vidx + 30] = _mm256_fmadd_ps(m0_1, m5_2, mcov_local[vidx + 30]);
						mcov_local[vidx + 31] = _mm256_fmadd_ps(m1_1, m5_2, mcov_local[vidx + 31]);
						mcov_local[vidx + 32] = _mm256_fmadd_ps(m2_1, m5_2, mcov_local[vidx + 32]);
						mcov_local[vidx + 33] = _mm256_fmadd_ps(m3_1, m5_2, mcov_local[vidx + 33]);
						mcov_local[vidx + 34] = _mm256_fmadd_ps(m4_1, m5_2, mcov_local[vidx + 34]);
						mcov_local[vidx + 35] = _mm256_fmadd_ps(m5_1, m5_2, mcov_local[vidx + 35]);
					}
				}
			}
		}
		else
		{
#pragma omp parallel for SCHEDULE
			for (int j = 0; j < data[0].rows; j++)
			{
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];

				AutoBuffer<const float*> sptr(color_channels);
				AutoBuffer<__m256> mv1(color_channels);
				AutoBuffer<__m256> mv2(color_channels);

				for (int c = 0; c < color_channels; c++)
				{
					sptr[c] = dataBorder[c].ptr<float>(j);
				}

				for (int i = 0; i < simd_end; i += simd_step)
				{
					//var (diag)
					for (int c = 0; c < color_channels; c++)mv1[c] = _mm256_loadu_ps(sptr[c]);
					for (int cy = 0, vidx = 0; cy < color_channels; cy++)
					{
						for (int cx = cy; cx < color_channels; cx++)
						{
							mvar_local[vidx] = _mm256_fmadd_ps(mv1[cx], mv1[cy], mvar_local[vidx]);
							vidx++;
						}
					}

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++)
					{
						for (int c = 0; c < color_channels; c++)
						{
							mv1[c] = _mm256_loadu_ps(sptr[c] + first[k]);
							mv2[c] = _mm256_loadu_ps(sptr[c] + second[k]);
						}
						for (int cy = 0; cy < color_channels; cy++)
						{
							for (int cx = 0; cx < color_channels; cx++)
							{
								mcov_local[vidx] = _mm256_fmadd_ps(mv1[cx], mv2[cy], mcov_local[vidx]);
								vidx++;
							}
						}
					}

					for (int c = 0; c < color_channels; c++)
					{
						sptr[c] += simd_step;
					}
				}
				if (isRem)
				{
					//var (diag)
					for (int c = 0; c < color_channels; c++)mv1[c] = _mm256_maskload_ps(sptr[c], mask);
					for (int cy = 0, vidx = 0; cy < color_channels; cy++)
					{
						for (int cx = cy; cx < color_channels; cx++)
						{
							mvar_local[vidx] = _mm256_fmadd_ps(mv1[cx], mv1[cy], mvar_local[vidx]);
							vidx++;
						}
					}

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++)
					{
						for (int c = 0; c < color_channels; c++)
						{
							mv1[c] = _mm256_maskload_ps(sptr[c] + first[k], mask);
							mv2[c] = _mm256_maskload_ps(sptr[c] + second[k], mask);
						}
						for (int cy = 0; cy < color_channels; cy++)
						{
							for (int cx = 0; cx < color_channels; cx++)
							{
								mcov_local[vidx] = _mm256_fmadd_ps(mv1[cx], mv2[cy], mcov_local[vidx]);
								vidx++;
							}
						}
					}
				}
			}
		}
	}
	else
	{
		rcc.getIndex(first, second, data[0].cols);
		const int simd_end_x = get_simd_floor(data[0].cols - 2 * patch_rad, simd_step) + patch_rad;
		normalSize = 1.0 / ((data[0].rows - 2 * patch_rad) * (simd_end_x - patch_rad));
		const int yend = data[0].rows - 2 * patch_rad;
		if (color_channels == 1)
		{
#pragma omp parallel for SCHEDULE
			for (int y = 0; y < yend; y++)
			{
				const int j = y + patch_rad;
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[(directionElementSize + 1) * tindex];
				__m256* mvar_local = &mbuff[(directionElementSize + 1) * tindex + directionElementSize];
				const float* sptr = data[0].ptr<float>(j, patch_rad);

				for (int i = patch_rad; i < simd_end_x; i += simd_step)
				{
					const __m256 v = _mm256_loadu_ps(sptr);
					mvar_local[0] = _mm256_fmadd_ps(v, v, mvar_local[0]);

					for (int k = 0; k < directionElementSize; k++)
					{
						mcov_local[k] = _mm256_fmadd_ps(_mm256_loadu_ps(sptr + first[k]), _mm256_loadu_ps(sptr + second[k]), mcov_local[k]);
					}

					sptr += simd_step;
				}
			}
		}
		else
		{
#pragma omp parallel for SCHEDULE
			for (int y = 0; y < yend; y++)
			{
				const int j = y + patch_rad;
				const int tindex = omp_get_thread_num();

				__m256* mcov_local = &mbuff[computeElementSize * tindex];
				__m256* mvar_local = &mbuff[computeElementSize * tindex + computeCovElementSize];
				AutoBuffer<const float*> sptr(color_channels);
				for (int c = 0; c < color_channels; c++)
				{
					sptr[c] = data[c].ptr<float>(j, patch_rad);
				}

				for (int i = patch_rad; i < simd_end_x; i += simd_step)
				{
					//diag (var)
					for (int cy = 0, vidx = 0; cy < color_channels; cy++)
					{
						for (int cx = 0; cx < color_channels; cx++)
						{
							mvar_local[vidx] = _mm256_fmadd_ps(_mm256_loadu_ps(sptr[cx]), _mm256_loadu_ps(sptr[cy]), mvar_local[vidx]);
							vidx++;
						}
					}

					//cov
					for (int k = 0, vidx = 0; k < directionElementSize; k++)
					{
						for (int cy = 0; cy < color_channels; cy++)
						{
							for (int cx = 0; cx < color_channels; cx++)
							{
								mcov_local[vidx] = _mm256_fmadd_ps(_mm256_loadu_ps(sptr[cx] + first[k]), _mm256_loadu_ps(sptr[cy] + second[k]), mcov_local[vidx]);
								vidx++;
							}
						}
					}

					for (int c = 0; c < color_channels; c++)
					{
						sptr[c] += simd_step;
					}
				}
			}
		}
	}

	//reduction
	vector<double> varElem(computeVarElementSize);
	vector<double> covElem(computeCovElementSize);
	for (double v : varElem) v = 0.0;
	for (double c : covElem) c = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		const int sstep = computeElementSize * t;
		__m256* mcov = &mbuff[sstep];
		for (int i = 0; i < computeCovElementSize; i++)
		{
			covElem[i] += _mm256_reduceadd_pspd(mcov[i]);
		}

		__m256* mvar = &mbuff[sstep + computeCovElementSize];
		for (int i = 0; i < computeVarElementSize; i++)
		{
			varElem[i] += _mm256_reduceadd_pspd(mvar[i]);
		}
	}

	setCovRep(meanForCov, varElem, destCovarianceMatrix, covElem, rcc.getSharedSet(), normalSize);

	_mm_free(mbuff);
}
#pragma endregion

#pragma region sep_cov
//20 x 5 matrix -> 4x5 x 5 matrix -> transpose 5x5 with 4 channel -> 1d vector
void channalUnitTranspose1DVector64F(const Mat& src, Mat& dest, const int channels)
{
	CV_Assert(src.channels() == 1);
	Mat s = src.clone();
	dest.create(1, s.cols * s.rows, CV_64F);
	const int w = s.cols / channels;
	const int h = s.rows;
	const double* sptr = s.ptr<double>();
	double* dptr = dest.ptr<double>();
	int idx = 0;
	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < w; i++)
		{
			for (int c = 0; c < channels; c++)
			{
				dptr[idx++] = sptr[w * channels * i + channels * j + c];
			}
		}
	}

}

void separateEvecSVD(const Mat& evec, const int src_channels, Mat& dstEvec, Mat& dstEvecH, Mat& dstEvecV, int order)
{
	separateEvecSVD(evec, src_channels, evec.rows, dstEvec, dstEvecH, dstEvecV, order);
}

void separateEvecSVD(const Mat& evec, const int src_channels, const int dchannels, Mat& dstEvec, Mat& dstEvecH, Mat& dstEvecV, int order)
{
	Mat umat, umatsep, w, u, vt, ui;
	vector<Mat> Ui, Uxi, Uyi;

	if (src_channels == 1)
	{
		const int D = (int)sqrt(evec.rows);
		for (int i = 0; i < dchannels; i++)
		{
			umat = evec(Rect(0, i, evec.cols, 1)).reshape(0, D);
			SVDecomp(umat, w, u, vt);
			umatsep = u(Rect(0, 0, order, D)) * vt(Rect(0, 0, D, order));
			umatsep = umatsep.reshape(0, 1);
			Uxi.push_back(u(Rect(0, 0, order, D)).t());
			Uyi.push_back(vt(Rect(0, 0, D, order)).clone());
			Ui.push_back(umatsep);
		}
		vconcat(Uxi, dstEvecH);
		vconcat(Uyi, dstEvecV);
		vconcat(Ui, dstEvec);
	}
	else
	{
		const int D = (int)sqrt(evec.rows / src_channels);
		const int DD = D * D;
		for (int i = 0; i < dchannels; i++)
		{
			vector<Mat> vumatsep;
			umat = evec(Rect(0, i, evec.cols, 1)).reshape(0, D);
			channalUnitTranspose1DVector64F(umat, umat, src_channels);
			umat = umat.clone().reshape(0, D);
			SVDecomp(umat, w, u, vt);
			umatsep = u(Rect(0, 0, order, D)) * vt(Rect(0, 0, src_channels * D, order));
			channalUnitTranspose1DVector64F(umatsep, umatsep, src_channels);
			//umatsep = umatsep.reshape(0, 1);
			Uxi.push_back(vt(Rect(0, 0, src_channels * D, order)).clone());
			Uyi.push_back(u(Rect(0, 0, order, D)).t());
			Ui.push_back(umatsep);
		}
		vconcat(Uxi, dstEvecH);
		vconcat(Uyi, dstEvecV);
		vconcat(Ui, dstEvec);
	}
}

void calcEvecfromSepCov(const Mat& XXt, const Mat& XtX, const int dchannel, Mat& dstEvec, Mat& dstEvecH, Mat& dstEvecV)
{
	CV_Assert(!XXt.empty());
	CV_Assert(!XtX.empty());
	const int src_channels = XXt.cols / XtX.cols;
	Mat Ux, Uy, evalx, evaly;
	Mat Sortidx, evalxy;
	vector<Point> Uidx;
	Mat Uxyi;
	vector<Mat>vUxy;
	vector<Mat>vUx;
	vector<Mat>vUy;

	if (XXt.cols == XtX.cols)
	{
		int D = XXt.rows;
		cv::eigen(XXt, evalx, Ux);
		cv::eigen(XtX, evaly, Uy);
		for (int i = 0; i < D; i++)
		{
			for (int j = 0; j < D; j++)
			{
				Uidx.push_back(Point(i, j));
			}
		}
		evalxy = evalx * evaly.t();

		sortIdx(evalxy.reshape(0, 1), Sortidx, SORT_EVERY_ROW + SORT_DESCENDING);

		for (int i = 0; i < dchannel; i++)
		{
			Uxyi = Ux(Rect(0, Uidx[Sortidx.at<int>(0, i)].x, D, 1)).t() * Uy(Rect(0, Uidx[Sortidx.at<int>(0, i)].y, D, 1));
			vUx.push_back(Ux(Rect(0, Uidx[Sortidx.at<int>(0, i)].x, D, 1)).clone());
			vUy.push_back(Uy(Rect(0, Uidx[Sortidx.at<int>(0, i)].y, D, 1)).clone());
			//Uxyi = Uy(Rect(0, Uidx[Sortidx.at<int>(0, i)].y, D, 1)).t() * Ux(Rect(0, Uidx[Sortidx.at<int>(0, i)].x, D, 1));
			Uxyi = Uxyi.reshape(0, 1);
			vUxy.push_back(Uxyi);
		}
		vconcat(vUxy, dstEvec);
		vconcat(vUx, dstEvecH);
		vconcat(vUy, dstEvecV);
	}
	else //(XXt.cols != XtX.cols)
	{
		const int Dx = XXt.cols;
		const int Dy = XtX.cols;
		cv::eigen(XXt, evalx, Ux);
		cv::eigen(XtX, evaly, Uy);
		//cout << evalx << endl;
		//cout << evaly << endl;
		for (int i = 0; i < Dx; i++)
		{
			for (int j = 0; j < Dy; j++)
			{
				Uidx.push_back(Point(i, j));
			}
		}
		evalxy = evalx * evaly.t();
		sortIdx(evalxy.reshape(0, 1), Sortidx, SORT_EVERY_ROW + SORT_DESCENDING);

		for (int i = 0; i < dchannel; i++)
		{
			const int sidx = Sortidx.at<int>(i);
			Uxyi = Uy(Rect(0, Uidx[sidx].y, Dy, 1)).t() * Ux(Rect(0, Uidx[sidx].x, Dx, 1));
			vUx.push_back(Ux(Rect(0, Uidx[sidx].x, Dx, 1)).clone());
			vUy.push_back(Uy(Rect(0, Uidx[sidx].y, Dy, 1)).clone());

			channalUnitTranspose1DVector64F(Uxyi, Uxyi, src_channels);//with transpose
			//Uxyi = Uxyi.reshape(1, 1);//without transpose

			vUxy.push_back(Uxyi);
		}
		vconcat(vUxy, dstEvec);
		vconcat(vUx, dstEvecH);
		vconcat(vUy, dstEvecV);
	}
}

//void computeSeparateCov(const vector<Mat>& src, const int patch_rad, const int borderType, Mat& covmat) {
//	const float D = float(patch_rad * 2 + 1);
//	const float DD = float(D * D);
//	const int channels = src.size();
//	const int cols = src[0].cols, rows = src[0].rows;
//	Mat srcborder, reduceVec, reduceScolor;
//
//	if (channels == 1)	{
//		vector<Mat> vcovmat(2);
//		copyMakeBorder(src[0], srcborder, patch_rad, patch_rad, patch_rad, patch_rad, borderType);
//
//		for (int d = 0; d < 2; d++) {
//			if (d == 1) srcborder = srcborder.t();
//			vcovmat[d].create(D, D, CV_64F);
//			for (int i = 0; i < D; i++)
//			{
//				for (int j = 0; j < D; j++)
//				{
//					reduce(srcborder(Rect(0, i, srcborder.cols, rows)).mul(srcborder(Rect(0, j, srcborder.cols, rows))), reduceVec, 0, REDUCE_SUM, CV_32F);
//					for (int k = 0; k < D - 1; k++)
//					{
//						reduceVec.at<float>(0, k) *= float(k + 1) / D;
//						reduceVec.at<float>(0, reduceVec.cols - 1 - k) *= float(k + 1) / D;
//					}
//					reduce(reduceVec, reduceScolor, 1, REDUCE_SUM, CV_32F);
//					vcovmat[d].at<double>(i, j) = reduceScolor.at<float>(0, 0) * D;
//				}
//			}
//			merge(vcovmat, covmat);
//		}
//	}
//}
#pragma endregion

#pragma region im2col

void imshowDRIM2COLEigenVec(string wname, Mat& evec, const int channels)
{
	const int w = (int)sqrt(evec.cols / channels);
	vector<Mat> v;
	for (int i = 0; i < evec.rows; i++)
	{
		Mat a = evec.row(i).reshape(1, w).clone();
		//if (i == 0)print_mat(a);
		Mat b;
		copyMakeBorder(a, b, 0, 1, 0, 1, BORDER_CONSTANT, Scalar::all(1));
		v.push_back(b);
	}
	Mat dest;
	cp::concat(v, dest, w, w);
	copyMakeBorder(dest, dest, 1, 0, 1, 0, BORDER_CONSTANT, Scalar::all(1));
	resize(dest, dest, Size(1024, 1024), 1, 1, INTER_NEAREST);

	cp::imshowScale(wname, dest, 255, 128);
}

void DRIM2COLEigenVec(const Mat& src, const Mat& evec, Mat& dest, const int r, const int channels, const int border, const bool isParallel)
{
	//cout << evec.cols << endl;
	//print_matinfo(evec);
	dest.create(src.size(), CV_MAKE_TYPE(CV_32F, evec.rows));

	const int D = 2 * r + 1;
	const int dim = D * D * src.channels();//D * D * src.channels()
	bool isSIMD = true;
	AutoBuffer<const float*> eptr(evec.rows);
	for (int m = 0; m < evec.rows; m++)
	{
		eptr[m] = evec.ptr<float>(m);
	}

	if (isParallel)
	{
		if (src.channels() == 1)
		{
			Mat srcborder;
			copyMakeBorder(src, srcborder, r, r, r, r, border);
			if (isSIMD)
			{
#pragma omp parallel for
				for (int j = 0; j < src.rows; j++)
				{
					AutoBuffer<__m256> patch(dim);
					for (int i = 0; i < src.cols; i += 8)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								patch[idx++] = _mm256_loadu_ps(srcborder.ptr<float>(j + l, i + m));
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							__m256 mval = _mm256_setzero_ps();
							for (int l = 0; l < dim; l++)
							{
								mval = _mm256_fmadd_ps(patch[l], _mm256_set1_ps(eptr[m][l]), mval);
							}
							for (int mm = 0; mm < 8; mm++)
							{
								//d[evec.rows * mm + m] = mval.m256_f32[mm];
								d[evec.rows * mm + m] = ((float*)&mval)[mm];
							}
						}
					}
				}
			}
			else
			{
#pragma omp parallel for
				for (int j = 0; j < src.rows; j++)
				{
					AutoBuffer<float> patch(dim);
					for (int i = 0; i < src.cols; i++)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								patch[idx++] = srcborder.at<float>(j + l, i + m);
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							float val = 0.f;
							for (int l = 0; l < dim; l++)
							{
								val += patch[l] * eptr[m][l];
							}
							d[m] = val;
						}
					}
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> srcborder;
			cp::splitCopyMakeBorder(src, srcborder, r, r, r, r, border);
			bool isSIMD = true;

			if (isSIMD)
			{
#pragma omp parallel for
				for (int j = 0; j < src.rows; j++)
				{
					AutoBuffer<__m256> patch(dim);
					for (int i = 0; i < src.cols; i += 8)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								for (int c = 0; c < 3; c++)
								{
									patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + l, i + m));
								}
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							__m256 mval = _mm256_setzero_ps();
							for (int l = 0; l < dim; l++)
							{
								mval = _mm256_fmadd_ps(patch[l], _mm256_set1_ps(eptr[m][l]), mval);
							}
							for (int mm = 0; mm < 8; mm++)
							{
								//d[evec.rows * mm + m] = mval.m256_f32[mm];
								d[evec.rows * mm + m] = ((float*)&mval)[mm];
							}
						}
					}
				}
			}
			else
			{
#pragma omp parallel for
				for (int j = 0; j < src.rows; j++)
				{
					AutoBuffer<float> patch(dim);
					for (int i = 0; i < src.cols; i++)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								for (int c = 0; c < 3; c++)
								{
									patch[idx++] = srcborder[c].at<float>(j + l, i + m);
								}
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							float val = 0.f;
							for (int l = 0; l < dim; l++)
							{
								val += patch[l] * eptr[m][l];
							}
							d[m] = val;
						}
					}
				}
			}
		}
	}
	else
	{
		if (src.channels() == 1)
		{
			Mat srcborder;
			copyMakeBorder(src, srcborder, r, r, r, r, border);
			if (isSIMD)
			{
				AutoBuffer<__m256> patch(dim);
				for (int j = 0; j < src.rows; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								patch[idx++] = _mm256_loadu_ps(srcborder.ptr<float>(j + l, i + m));
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							__m256 mval = _mm256_setzero_ps();
							for (int l = 0; l < dim; l++)
							{
								mval = _mm256_fmadd_ps(patch[l], _mm256_set1_ps(eptr[m][l]), mval);
							}
							for (int mm = 0; mm < 8; mm++)
							{
								//d[evec.rows * mm + m] = mval.m256_f32[mm];
								d[evec.rows * mm + m] = ((float*)&mval)[mm];
							}
						}
					}
				}
			}
			else
			{
				AutoBuffer<float> patch(dim);
				for (int j = 0; j < src.rows; j++)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								patch[idx++] = srcborder.at<float>(j + l, i + m);
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							float val = 0.f;
							for (int l = 0; l < dim; l++)
							{
								val += patch[l] * eptr[m][l];
							}
							d[m] = val;
						}
					}
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> srcborder;
			cp::splitCopyMakeBorder(src, srcborder, r, r, r, r, border);
			bool isSIMD = true;
			if (isSIMD)
			{
				AutoBuffer<__m256> patch(dim);
				for (int j = 0; j < src.rows; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								for (int c = 0; c < 3; c++)
								{
									patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + l, i + m));
								}
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							__m256 mval = _mm256_setzero_ps();
							for (int l = 0; l < dim; l++)
							{
								mval = _mm256_fmadd_ps(patch[l], _mm256_set1_ps(eptr[m][l]), mval);
							}
							for (int mm = 0; mm < 8; mm++)
							{
								//d[evec.rows * mm + m] = mval.m256_f32[mm];
								d[evec.rows * mm + m] = ((float*)&mval)[mm];
							}
						}
					}
				}
			}
			else
			{
				AutoBuffer<float> patch(dim);
				for (int j = 0; j < src.rows; j++)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								for (int c = 0; c < 3; c++)
								{
									patch[idx++] = srcborder[c].at<float>(j + l, i + m);
								}
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							float val = 0.f;
							for (int l = 0; l < dim; l++)
							{
								val += patch[l] * eptr[m][l];
							}
							d[m] = val;
						}
					}
				}
			}
		}
	}
}

template<int dest_channels>
void DRIM2COLEigenVec(const vector<Mat>& src, const Mat& evec, vector<Mat>& dest, const int r, const int border)
{
	//cout << evec.cols << endl;
	//print_matinfo(evec);
	const int src_channels = (int)src.size();

	if (dest.size() != dest_channels) dest.resize(dest_channels);
	for (int c = 0; c < dest_channels; c++)
	{
		dest[c].create(src[0].size(), CV_32F);
	}


	const int D = 2 * r + 1;
	const int DD = D * D;
	const int dim = DD * src_channels;//D * D * src_channels()

	AutoBuffer<const float*> eptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eptr[m] = evec.ptr<float>(m);
	}

	vector<Mat> srcborder(src_channels);
	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(src[c], srcborder[c], r, r, r, r, border);
	}

	bool isSIMD = true;
	if (isSIMD)
	{
		const int unroll = 32;
		const int width = get_simd_floor(src[0].cols, unroll);
		const int height = src[0].rows;

		AutoBuffer<float*> sptr(src_channels);

		const int step = srcborder[0].cols;
		for (int c = 0; c < src_channels; c++)
		{
			sptr[c] = srcborder[c].ptr<float>();
		}

		if (unroll == 8)
		{
			const int simd_end = get_simd_floor(width, 8);
			const __m256i mask = get_simd_residualmask_epi32(width);
			const bool isRem = (width == simd_end) ? false : true;
			AutoBuffer<__m256> patch(dim);
			AutoBuffer<float*> dptr(dest_channels);
			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < simd_end; i += 8)
				{
					for (int c = 0; c < dest_channels; c++)
					{
						dptr[c] = dest[c].ptr<float>(j, i);
					}

					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + i;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								patch[idx++] = _mm256_loadu_ps(sptr[c] + index);
							}
						}
					}

					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval = _mm256_setzero_ps();
						for (int d = 0; d < dim; d++)
						{
							mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
						}
						_mm256_storeu_ps(dptr[c], mval);
					}
				}
				if (isRem)
				{
					for (int c = 0; c < dest_channels; c++)
					{
						dptr[c] = dest[c].ptr<float>(j, simd_end);
					}

					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + simd_end;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								patch[idx++] = _mm256_maskload_ps(sptr[c] + index, mask);
							}
						}
					}

					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval = _mm256_setzero_ps();
						for (int d = 0; d < dim; d++)
						{
							mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
						}
						_mm256_maskstore_ps(dptr[c], mask, mval);
					}
				}
			}
		}
		else if (unroll == 16)
		{
		}
		else if (unroll == 32)
		{
			const int simd_end = get_simd_floor(width, 32);
			const int rem32 = width - simd_end;
			const int simdx8 = rem32 / 8;
			const int simd_end8 = simd_end + simdx8 * 8;
			const int rem = rem32 - simdx8 * 8;
			const __m256i mask = get_simd_residualmask_epi32(rem);
			const bool isRem = (rem == 0) ? false : true;

			AutoBuffer<__m256> me(dim * dest_channels);
			for (int c = 0, idx = 0; c < dest_channels; c++)
			{
				for (int d = 0; d < dim; d++)
				{
					me[idx++] = _mm256_set1_ps(eptr[c][d]);
				}
			}
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < height; j++)
			{
				AutoBuffer<float*> dptr(dest_channels);
				AutoBuffer<__m256> patch1(dim);
				AutoBuffer<__m256> patch2(dim);
				AutoBuffer<__m256> patch3(dim);
				AutoBuffer<__m256> patch4(dim);
				for (int c = 0; c < dest_channels; c++)
				{
					dptr[c] = dest[c].ptr<float>(j);
				}

				for (int i = 0; i < simd_end; i += 32)
				{
					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + i;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								float* s = sptr[c] + index;
								patch1[idx] = _mm256_loadu_ps(s);
								patch2[idx] = _mm256_loadu_ps(s + 8);
								patch3[idx] = _mm256_loadu_ps(s + 16);
								patch4[idx] = _mm256_loadu_ps(s + 24);
								idx++;
							}
						}
					}

					int idx = 0;
					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval1 = _mm256_mul_ps(patch1[0], me[idx]);
						__m256 mval2 = _mm256_mul_ps(patch2[0], me[idx]);
						__m256 mval3 = _mm256_mul_ps(patch3[0], me[idx]);
						__m256 mval4 = _mm256_mul_ps(patch4[0], me[idx]);
						__m256 m1 = patch1[1];
						__m256 m2 = patch2[1];
						__m256 m3 = patch3[1];
						__m256 m4 = patch4[1];
						__m256 e = me[1];
						for (int d = 1; d < dim - 1; d++)
						{
							mval1 = _mm256_fmadd_ps(m1, e, mval1);
							mval2 = _mm256_fmadd_ps(m2, e, mval2);
							mval3 = _mm256_fmadd_ps(m3, e, mval3);
							mval4 = _mm256_fmadd_ps(m4, e, mval4);
							idx++;
							m1 = patch1[d + 1];
							m2 = patch2[d + 1];
							m3 = patch3[d + 1];
							m4 = patch4[d + 1];
							e = me[idx];
						}
						{
							mval1 = _mm256_fmadd_ps(m1, me[idx], mval1);
							mval2 = _mm256_fmadd_ps(m2, me[idx], mval2);
							mval3 = _mm256_fmadd_ps(m3, me[idx], mval3);
							mval4 = _mm256_fmadd_ps(m4, me[idx], mval4);
						}
						_mm256_storeu_ps(dptr[c] + i, mval1);
						_mm256_storeu_ps(dptr[c] + i + 8, mval2);
						_mm256_storeu_ps(dptr[c] + i + 16, mval3);
						_mm256_storeu_ps(dptr[c] + i + 24, mval4);
					}
				}

				for (int i = simd_end; i < simd_end8; i += 8)
				{
					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + i;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								float* s = sptr[c] + index;
								patch1[idx] = _mm256_loadu_ps(s);
								idx++;
							}
						}
					}

					int idx = 0;
					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval1 = _mm256_mul_ps(patch1[0], me[idx]);
						__m256 m1 = patch1[1];
						__m256 e = me[1];
						for (int d = 1; d < dim - 1; d++)
						{
							mval1 = _mm256_fmadd_ps(m1, e, mval1);
							idx++;
							m1 = patch1[d + 1];
							e = me[idx];
						}
						{
							mval1 = _mm256_fmadd_ps(m1, me[idx], mval1);
						}
						_mm256_storeu_ps(dptr[c] + i, mval1);
					}
				}

				if (isRem)
				{
					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + simd_end8;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								float* s = sptr[c] + index;
								patch1[idx] = _mm256_maskload_ps(s, mask);
								idx++;
							}
						}
					}

					int idx = 0;
					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval1 = _mm256_mul_ps(patch1[0], me[idx]);
						__m256 m1 = patch1[1];
						__m256 e = me[1];
						for (int d = 1; d < dim - 1; d++)
						{
							mval1 = _mm256_fmadd_ps(m1, e, mval1);
							idx++;
							m1 = patch1[d + 1];
							e = me[idx];
						}
						{
							mval1 = _mm256_fmadd_ps(m1, me[idx], mval1);
						}
						_mm256_maskstore_ps(dptr[c] + simd_end8, mask, mval1);
					}
				}
			}
		}
	}
	else
	{
		const int width = src[0].cols;
		const int height = src[0].rows;
		AutoBuffer<float> patch(dim);
		AutoBuffer<float*> dptr(dest_channels);

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				for (int c = 0; c < dest_channels; c++)
				{
					dptr[c] = dest[c].ptr<float>(j, i);
				}

				for (int l = 0, idx = 0; l < D; l++)
				{
					for (int m = 0; m < D; m++)
					{
						for (int c = 0; c < src_channels; c++)
						{
							patch[idx++] = srcborder[c].at<float>(j + l, i + m);
						}
					}
				}

				for (int c = 0; c < dest_channels; c++)
				{
					float val = 0.f;
					for (int d = 0; d < dim; d++)
					{
						val += patch[d] * eptr[c][d];
					}
					*dptr[c] = val;
				}
			}
		}
	}
}

void DRIM2COLEigenVecCn(const vector<Mat>& src, const Mat& evec, vector<Mat>& dest, const int r, const int border)
{
	//cout << evec.cols << endl;
	//print_matinfo(evec);
	const int src_channels = (int)src.size();
	const int dest_channels = evec.rows;

	if (dest.size() != dest_channels) dest.resize(dest_channels);
	for (int c = 0; c < dest_channels; c++)
	{
		dest[c].create(src[0].size(), CV_32F);
	}

	const int D = 2 * r + 1;
	const int DD = D * D;
	const int dim = DD * src_channels;//D * D * src_channels()

	AutoBuffer<const float*> eptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eptr[m] = evec.ptr<float>(m);
	}

	vector<Mat> srcborder(src_channels);
	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(src[c], srcborder[c], r, r, r, r, border);
	}

	bool isSIMD = true;
	if (isSIMD)
	{
		const int unroll = 32;
		const int width = get_simd_floor(src[0].cols, unroll);
		const int height = src[0].rows;

		AutoBuffer<float*> sptr(src_channels);

		const int step = srcborder[0].cols;
		for (int c = 0; c < src_channels; c++)
		{
			sptr[c] = srcborder[c].ptr<float>();
		}

		if (unroll == 8)
		{
			const int simd_end = get_simd_floor(width, 8);
			const __m256i mask = get_simd_residualmask_epi32(width);
			const bool isRem = (width == simd_end) ? false : true;
			//if (isRem)cout << "isRem: true" << endl;

			AutoBuffer<__m256> me(dim * dest_channels);
			for (int c = 0, idx = 0; c < dest_channels; c++)
			{
				for (int d = 0; d < dim; d++)
				{
					me[idx++] = _mm256_set1_ps(eptr[c][d]);
				}
			}
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < height; j++)
			{
				AutoBuffer<float*> dptr(dest_channels);
				AutoBuffer<__m256> patch(dim);
				for (int i = 0; i < simd_end; i += 8)
				{
					for (int c = 0; c < dest_channels; c++)
					{
						dptr[c] = dest[c].ptr<float>(j, i);
					}

					//load 
					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + i;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								patch[idx++] = _mm256_loadu_ps(sptr[c] + index);
							}
						}
					}

					//projection
					for (int c = 0, idx = 0; c < dest_channels; c++)
					{
						__m256 mval = _mm256_mul_ps(patch[0], me[idx]);
						idx++;
						for (int d = 1; d < dim; d++)
						{
							mval = _mm256_fmadd_ps(patch[d], me[idx], mval);
							idx++;
						}
						_mm256_storeu_ps(dptr[c], mval);
					}
				}
				if (isRem)
				{
					for (int c = 0; c < dest_channels; c++)
					{
						dptr[c] = dest[c].ptr<float>(j, simd_end);
					}

					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + simd_end;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								patch[idx++] = _mm256_maskload_ps(sptr[c] + index, mask);
							}
						}
					}

					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval = _mm256_setzero_ps();
						for (int d = 0; d < dim; d++)
						{
							mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
						}
						_mm256_maskstore_ps(dptr[c], mask, mval);
					}
				}
			}
		}
		else if (unroll == 16)
		{
			;
		}
		else if (unroll == 32)
		{
			const int simd_end = get_simd_floor(width, 32);
			const int rem32 = width - simd_end;
			const int simdx8 = rem32 / 8;
			const int simd_end8 = simd_end + simdx8 * 8;
			const int rem = rem32 - simdx8 * 8;
			const __m256i mask = get_simd_residualmask_epi32(rem);
			const bool isRem = (rem == 0) ? false : true;
			//if (isRem)cout << "isRem: true" << endl;

			AutoBuffer<__m256> me(dim * dest_channels);
			for (int c = 0, idx = 0; c < dest_channels; c++)
			{
				for (int d = 0; d < dim; d++)
				{
					me[idx++] = _mm256_set1_ps(eptr[c][d]);
				}
			}
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < height; j++)
			{
				AutoBuffer<float*> dptr(dest_channels);
				AutoBuffer<__m256> patch1(dim);
				AutoBuffer<__m256> patch2(dim);
				AutoBuffer<__m256> patch3(dim);
				AutoBuffer<__m256> patch4(dim);
				for (int c = 0; c < dest_channels; c++)
				{
					dptr[c] = dest[c].ptr<float>(j);
				}

				for (int i = 0; i < simd_end; i += 32)
				{
					//load
					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + i;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								const float* s = sptr[c] + index;
								patch1[idx] = _mm256_loadu_ps(s);
								patch2[idx] = _mm256_loadu_ps(s + 8);
								patch3[idx] = _mm256_loadu_ps(s + 16);
								patch4[idx] = _mm256_loadu_ps(s + 24);
								idx++;
							}
						}
					}

					//projection
					for (int c = 0, idx = 0; c < dest_channels; c++)
					{
						__m256 mval1 = _mm256_mul_ps(patch1[0], me[idx]);
						__m256 mval2 = _mm256_mul_ps(patch2[0], me[idx]);
						__m256 mval3 = _mm256_mul_ps(patch3[0], me[idx]);
						__m256 mval4 = _mm256_mul_ps(patch4[0], me[idx]);
						idx++;
						__m256 m1 = patch1[1];
						__m256 m2 = patch2[1];
						__m256 m3 = patch3[1];
						__m256 m4 = patch4[1];
						__m256 e = me[idx];
						for (int d = 1; d < dim - 1; d++)
						{
							mval1 = _mm256_fmadd_ps(m1, e, mval1);
							mval2 = _mm256_fmadd_ps(m2, e, mval2);
							mval3 = _mm256_fmadd_ps(m3, e, mval3);
							mval4 = _mm256_fmadd_ps(m4, e, mval4);
							idx++;
							m1 = patch1[d + 1];
							m2 = patch2[d + 1];
							m3 = patch3[d + 1];
							m4 = patch4[d + 1];
							e = me[idx];
						}
						{
							mval1 = _mm256_fmadd_ps(m1, e, mval1);
							mval2 = _mm256_fmadd_ps(m2, e, mval2);
							mval3 = _mm256_fmadd_ps(m3, e, mval3);
							mval4 = _mm256_fmadd_ps(m4, e, mval4);
							idx++;
						}
						_mm256_storeu_ps(dptr[c] + i, mval1);
						_mm256_storeu_ps(dptr[c] + i + 8, mval2);
						_mm256_storeu_ps(dptr[c] + i + 16, mval3);
						_mm256_storeu_ps(dptr[c] + i + 24, mval4);
					}
				}

				for (int i = simd_end; i < simd_end8; i += 8)
				{
					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + i;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								float* s = sptr[c] + index;
								patch1[idx] = _mm256_loadu_ps(s);
								idx++;
							}
						}
					}

					int idx = 0;
					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval1 = _mm256_mul_ps(patch1[0], me[idx]);
						__m256 m1 = patch1[1];
						idx++;
						__m256 e = me[idx];
						for (int d = 1; d < dim - 1; d++)
						{
							mval1 = _mm256_fmadd_ps(m1, e, mval1);
							idx++;
							m1 = patch1[d + 1];
							e = me[idx];
						}
						{
							mval1 = _mm256_fmadd_ps(m1, me[idx], mval1);
							idx++;
						}
						_mm256_storeu_ps(dptr[c] + i, mval1);
					}
				}
				if (isRem)
				{
					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + simd_end8;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								float* s = sptr[c] + index;
								patch1[idx] = _mm256_maskload_ps(s, mask);
								idx++;
							}
						}
					}

					int idx = 0;
					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval1 = _mm256_mul_ps(patch1[0], me[idx]);
						idx++;
						__m256 m1 = patch1[1];
						__m256 e = me[idx];
						for (int d = 1; d < dim - 1; d++)
						{
							mval1 = _mm256_fmadd_ps(m1, e, mval1);
							idx++;
							m1 = patch1[d + 1];
							e = me[idx];
						}
						{
							mval1 = _mm256_fmadd_ps(m1, me[idx], mval1);
							idx++;
						}
						_mm256_maskstore_ps(dptr[c] + simd_end8, mask, mval1);
					}
				}
			}
		}
	}
	else
	{
		const int width = src[0].cols;
		const int height = src[0].rows;
		AutoBuffer<float> patch(dim);
		AutoBuffer<float*> dptr(dest_channels);

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				for (int d = 0; d < dest_channels; d++)
				{
					float val = 0.f;
					for (int l = 0, idx = 0; l < D; l++)
					{
						for (int m = 0; m < D; m++)
						{
							for (int c = 0; c < src_channels; c++)
							{
								val += srcborder[c].at<float>(j + l, i + m) * evec.at<float>(d, idx);
								idx++;
							}
						}
					}
					dest[d].at<float>(j, i) = val;
				}

				/*
				for (int c = 0; c < dest_channels; c++)
				{
					dptr[c] = dest[c].ptr<float>(j, i);
				}

				for (int l = 0, idx = 0; l < D; l++)
				{
					for (int m = 0; m < D; m++)
					{
						for (int c = 0; c < src_channels; c++)
						{
							patch[idx++] = srcborder[c].at<float>(j + l, i + m);
						}
					}
				}

				for (int d = 0; d < dest_channels; d++)
				{
					float val = 0.f;
					for (int e = 0; e < dim; e++)
					{
						val += patch[e] * eptr[d][e];
					}
					*dptr[d] = val;
				}*/
			}
		}
	}
}

void DRIM2COLSepEigenVecCn(const vector<Mat>& src, const Mat& evecH, const Mat& evecV, vector<Mat>& dest, const int r, const int border)
{
	const int src_channels = (int)src.size();
	const int dest_channels = evecH.rows;

	if (dest.size() != dest_channels) dest.resize(dest_channels);
	for (int c = 0; c < dest_channels; c++)
	{
		dest[c].create(src[0].size(), CV_32F);
	}

	const int width = src[0].cols;
	const int height = src[0].rows;

	const int D = 2 * r + 1;
	//const int dimV = D * src_channels;//D * src_channels
	//const int dimH = D;//D
	const int dimV = D;//D
	const int dimH = D * src_channels;//D * src_channels

	AutoBuffer<const float*> eHptr(dest_channels);
	AutoBuffer<const float*> eVptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eHptr[m] = evecH.ptr<float>(m);
		eVptr[m] = evecV.ptr<float>(m);
	}
	const int wb8 = get_simd_ceil(width + 2 * r, 8);
	const int off = wb8 - width + 2 * r;
	vector<Mat> srcborder(src_channels);
	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(src[c], srcborder[c], r, r, r, r + off, border);
	}

	int bb = 0;
	const int stx = bb;
	const int edbx = srcborder[0].cols - bb;
	const int edx = width - bb;
	const int sty = bb;
	const int edy = height - bb;

	const bool isSIMD = true;
	if (isSIMD)
	{
		const int unroll = 8;
			if (unroll == 8)
			{
				const int simd_bend = get_simd_floor(edbx - stx, 8);
				const int simd_end = get_simd_floor(edx - stx, 8);
				const __m256i mask = get_simd_residualmask_epi32(edx - stx);
				const int rem = edx - stx - simd_end;
				const bool isRem = (rem == 0) ? false : true;
#pragma omp parallel for schedule(dynamic)
				for (int j = sty; j < edy; j++)
				{
					AutoBuffer<__m256> patch(src_channels * D);
					Mat projVHI(dest_channels, srcborder[0].cols, CV_32F);
					// V loop
					for (int i = stx; i < simd_bend; i += 8)
					{
						//load data
						for (int v = 0, idx = 0; v < D; v++)
						{
							for (int c = 0; c < src_channels; c++)
							{
								patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + v, i));
							}
						}

						//projection
						for (int d = 0; d < dest_channels; d++)
						{
							__m256 mval = _mm256_mul_ps(patch[0], _mm256_set1_ps(eHptr[d][0]));
							for (int h = 1; h < dimH; h++)
							{
								mval = _mm256_fmadd_ps(patch[h], _mm256_set1_ps(eHptr[d][h]), mval);
							}
							_mm256_storeu_ps(projVHI.ptr<float>(d, i), mval);
						}
					}
					// H loop
					for (int i = stx; i < simd_end; i += 8)
					{
						for (int d = 0; d < dest_channels; d++)
						{
							const float* p = projVHI.ptr<float>(d, i);
							__m256 mval = _mm256_mul_ps(_mm256_loadu_ps(p), _mm256_set1_ps(eVptr[d][0]));
							for (int h = 1; h < D; h++)
							{
								mval = _mm256_fmadd_ps(_mm256_loadu_ps(p + h), _mm256_set1_ps(eVptr[d][h]), mval);
							}
							_mm256_storeu_ps(dest[d].ptr<float>(j, i), mval);
						}
					}
				}
			}
			else if (unroll == 32)
			{
				const int simd_bend = get_simd_floor(edbx - stx, 32);
				const int simd_end = get_simd_floor(edx - stx, 32);
				const __m256i mask = get_simd_residualmask_epi32(edx - stx);
				const int rem = edx - stx - simd_end;
				const bool isRem = (rem == 0) ? false : true;
#pragma omp parallel for schedule(dynamic)
				for (int j = sty; j < edy; j++)
				{
					AutoBuffer<__m256> patch1(src_channels * D);
					AutoBuffer<__m256> patch2(src_channels * D);
					AutoBuffer<__m256> patch3(src_channels * D);
					AutoBuffer<__m256> patch4(src_channels * D);
					Mat projVHI(dest_channels, srcborder[0].cols, CV_32F);
					// V loop
					for (int i = stx; i < simd_bend; i += 32)
					{
						//load data
						for (int v = 0, idx = 0; v < D; v++)
						{
							for (int c = 0; c < src_channels; c++)
							{
								patch1[idx] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + v, i + 0));
								patch2[idx] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + v, i + 8));
								patch3[idx] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + v, i + 16));
								patch4[idx] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + v, i + 24));
								idx++;
							}
						}

						//projection
						for (int d = 0; d < dest_channels; d++)
						{
							const __m256 me = _mm256_set1_ps(eHptr[d][0]);
							__m256 mval1 = _mm256_mul_ps(patch1[0], me);
							__m256 mval2 = _mm256_mul_ps(patch2[0], me);
							__m256 mval3 = _mm256_mul_ps(patch3[0], me);
							__m256 mval4 = _mm256_mul_ps(patch4[0], me);
							for (int h = 1; h < dimH; h++)
							{
								const __m256 me = _mm256_set1_ps(eHptr[d][h]);
								mval1 = _mm256_fmadd_ps(patch1[h], me, mval1);
								mval2 = _mm256_fmadd_ps(patch2[h], me, mval2);
								mval3 = _mm256_fmadd_ps(patch3[h], me, mval3);
								mval4 = _mm256_fmadd_ps(patch4[h], me, mval4);
							}
							_mm256_storeu_ps(projVHI.ptr<float>(d, i + 0), mval1);
							_mm256_storeu_ps(projVHI.ptr<float>(d, i + 8), mval2);
							_mm256_storeu_ps(projVHI.ptr<float>(d, i + 16), mval3);
							_mm256_storeu_ps(projVHI.ptr<float>(d, i + 24), mval4);
						}
					}
					// H loop
					for (int i = stx; i < simd_end; i += 32)
					{
						for (int d = 0; d < dest_channels; d++)
						{
							const float* p = projVHI.ptr<float>(d, i);
							const __m256 me = _mm256_set1_ps(eVptr[d][0]);
							__m256 mval1 = _mm256_mul_ps(_mm256_loadu_ps(p), me);
							__m256 mval2 = _mm256_mul_ps(_mm256_loadu_ps(p + 8), me);
							__m256 mval3 = _mm256_mul_ps(_mm256_loadu_ps(p + 16), me);
							__m256 mval4 = _mm256_mul_ps(_mm256_loadu_ps(p + 24), me);
							for (int h = 1; h < D; h++)
							{
								const __m256 me = _mm256_set1_ps(eVptr[d][h]);
								mval1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + h), me, mval1);
								mval2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + h + 8), me, mval2);
								mval3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + h + 16), me, mval3);
								mval4 = _mm256_fmadd_ps(_mm256_loadu_ps(p + h + 24), me, mval4);
							}
							_mm256_storeu_ps(dest[d].ptr<float>(j, i), mval1);
							_mm256_storeu_ps(dest[d].ptr<float>(j, i + 8), mval2);
							_mm256_storeu_ps(dest[d].ptr<float>(j, i + 16), mval3);
							_mm256_storeu_ps(dest[d].ptr<float>(j, i + 24), mval4);
						}
					}
				}
			}
	}
	else
	{
#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < height; j++)
		{
			Mat projVHI = Mat::zeros(dest_channels, srcborder[0].cols, CV_32F);
			//V
			for (int i = 0; i < srcborder[0].cols; i++)
			{
				for (int d = 0; d < dest_channels; d++)
				{
					float mval = 0.f;
					for (int v = 0, idx = 0; v < D; v++)
					{
						for (int c = 0; c < src_channels; c++)
						{
							mval = fma(srcborder[c].at<float>(j + v, i), eHptr[d][idx], mval);
							idx++;
						}
					}
					projVHI.at<float>(d, i) = mval;
				}
			}
			//H
			for (int i = 0; i < width; i++)
			{
				for (int d = 0; d < dest_channels; d++)
				{
					float mval = 0.f;
					for (int h = 0; h < D; h++)
					{
						mval = fma(projVHI.at<float>(d, i + h), eVptr[d][h], mval);
					}
					dest[d].at<float>(j, i) = mval;
				}
			}
		}
	}
}

double DRIM2COLTestSepCn(const vector<Mat>& src, const Mat& evec, const Mat& evecH, const Mat& evecV, vector<Mat>& dest, const int r, const int border)
{
	Mat evecInv = evec.t();
	Mat evecHInv = evecH.t();
	Mat evecVInv = evecV.t();
	const int src_channels = (int)src.size();
	const int dest_channels = evecH.rows;

	if (dest.size() != dest_channels) dest.resize(dest_channels);
	for (int c = 0; c < dest_channels; c++)
	{
		dest[c].create(src[0].size(), CV_32F);
	}

	const int width = src[0].cols;
	const int height = src[0].rows;

	const int D = 2 * r + 1;
	//const int dimV = D * src_channels;//D * src_channels
	//const int dimH = D;//D
	const int dimV = D;//D
	const int dimH = D * src_channels;//D * src_channels
	const int dim = dimH * dimV;

	AutoBuffer<const float*> eHptr(dest_channels);
	AutoBuffer<const float*> eVptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eHptr[m] = evecH.ptr<float>(m);
		eVptr[m] = evecV.ptr<float>(m);
	}
	const int wb8 = get_simd_ceil(width + 2 * r, 8);
	const int off = wb8 - width + 2 * r;
	vector<Mat> srcborder(src_channels);
	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(src[c], srcborder[c], r, r, r, r + off, border);
	}

	vector<Mat> projection(dest_channels);
	for (int c = 0; c < dest_channels; c++)
	{
		projection[c].create(srcborder[0].size(), CV_32F);
	}

	const bool isRep = true;
	const bool is2DTest = false;
	double ret = 0.0;
	if (isRep)
	{
		if (is2DTest)
		{
			//projection
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i++)
				{
					//projection
					for (int d = 0; d < dest_channels; d++)
					{
						float val = 0.f;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								for (int c = 0; c < src_channels; c++)
								{
									//val = fma(srcborder[c].at<float>(j + l, i + m), evecH.at<float>(d, src_channels * m + c) * evecV.at<float>(d, l), val);//without reshape
									val = fma(srcborder[c].at<float>(j + m, i + l), evecH.at<float>(d, src_channels * m + c) * evecV.at<float>(d, l), val);
								}
							}
						}
						dest[d].at<float>(j, i) = val;
					}
				}
			}

			//reprojection
			const int thread_max = omp_get_num_procs();
			AutoBuffer<double> error(thread_max);
			for (int t = 0; t < thread_max; t++)error[t] = 0.0;
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < height; j++)
			{
				const int t = omp_get_thread_num();
				for (int i = 0; i < width; i++)
				{
					for (int d = 0; d < dim; d++)
					{
						/*
						without reshape
						float val = 0.f;
						for (int c = 0; c < dest_channels; c++)
						{
							//val = fma(dest[c].at<float>(j, i), evecInv.at<float>(d, c), val);
							val = fma(dest[c].at<float>(j, i), evecHInv.at<float>(src_channels * H + C, c) * evecVInv.at<float>(V, c), val);
						}
						const int H = (d % (dimH)) / src_channels;
						const int V = d / (dimH);
						const int I = i + H;
						const int J = j + V;
						const int C = d % src_channels;
						*/
						float val = 0.f;

						const int H = (d % (dimH)) / src_channels;
						const int V = d / (dimH);
						const int C = d % src_channels;
						for (int c = 0; c < dest_channels; c++)
						{
							//val = fma(dest[c].at<float>(j, i), evecInv.at<float>(d, c), val);
							val = fma(dest[c].at<float>(j, i), evecHInv.at<float>(src_channels * H + C, c) * evecVInv.at<float>(V, c), val);
						}

						const int I = i + V;
						const int J = j + H;

						error[t] += (val - srcborder[C].at<float>(J, I)) * (val - srcborder[C].at<float>(J, I));
					}
				}
			}
			double total_error = 0.0;
			for (int t = 0; t < thread_max; t++)total_error += error[t];
			const double mse = total_error / (width * height * dim);
			ret = 10.0 * log10(255.0 * 255.0 / mse);
			return ret;
		}
		else
		{
			const bool isHV = false;
			if (isHV)
			{
				//projection
				// H loop
#pragma omp parallel for schedule(dynamic)
				for (int j = 0; j < srcborder[0].rows; j++)
				{
					for (int i = 0; i < width; i++)
					{
						//projection
						for (int d = 0; d < dest_channels; d++)
						{
							float mval = 0.f;
							for (int l = 0, idx = 0; l < D; l++)
							{
								for (int c = 0; c < src_channels; c++)
								{
									mval += srcborder[c].at<float>(j, i + l) * eHptr[d][idx];
									idx++;
								}
							}
							projection[d].at<float>(j, i) = mval;
						}
					}
				}

				//V loop
#pragma omp parallel for schedule(dynamic)
				for (int j = 0; j < height; j++)
				{
					for (int i = 0; i < width; i++)
					{
						//projection
						for (int d = 0; d < dest_channels; d++)
						{
							float mval = 0.f;
							for (int v = 0; v < dimV; v++)
							{
								mval += projection[d].at<float>(j + v, i) * eVptr[d][v];
							}
							dest[d].at<float>(j, i) = mval;
						}
					}
				}
			}
			else
			{
				//projection
				// V loop
#pragma omp parallel for schedule(dynamic)
				for (int j = 0; j < height; j++)
				{
					Mat projVHI = Mat::zeros(dest_channels, srcborder[0].cols, CV_32F);
					for (int i = 0; i < srcborder[0].cols; i++)
					{
						//projection
						for (int d = 0; d < dest_channels; d++)
						{
							float mval = 0.f;
							for (int v = 0, idx = 0; v < D; v++)
							{
								for (int c = 0; c < src_channels; c++)
								{
									mval = fma(srcborder[c].at<float>(j + v, i), eHptr[d][idx++], mval);
								}
							}
							projVHI.at<float>(d, i) = mval;
						}
					}
					for (int i = 0; i < width; i++)
					{
						//projection
						for (int d = 0; d < dest_channels; d++)
						{
							float mval = 0.f;
							for (int h = 0; h < D; h++)
							{
								mval = fma(projVHI.at<float>(d, i + h), eVptr[d][h], mval);
							}
							dest[d].at<float>(j, i) = mval;
						}
					}
				}
			}
			//reprojection
			const int thread_max = omp_get_num_procs();
			AutoBuffer<double> error(thread_max);
			for (int t = 0; t < thread_max; t++)error[t] = 0.0;
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < height; j++)
			{
				const int t = omp_get_thread_num();
				for (int i = 0; i < width; i++)
				{
					for (int d = 0; d < dim; d++)
					{
						const int H = (d % (dimH)) / src_channels;
						const int V = d / (dimH);
						const int C = d % src_channels;
						float val = 0.f;
						//without reshape
						/*for (int c = 0; c < dest_channels; c++)
						{
							//val = fma(dest[c].at<float>(j, i), evecInv.at<float>(d, c), val);
							val = fma(dest[c].at<float>(j, i), evecHInv.at<float>(src_channels * H + C, c) * evecVInv.at<float>(V, c), val);
						}
						const int I = i + H;
						const int J = j + V;*/

						//with reshape
						for (int c = 0; c < dest_channels; c++)
						{
							//val = fma(dest[c].at<float>(j, i), evecInv.at<float>(d, c), val);
							val = fma(dest[c].at<float>(j, i), evecHInv.at<float>(src_channels * H + C, c) * evecVInv.at<float>(V, c), val);
						}
						const int I = i + V;
						const int J = j + H;

						error[t] += (val - srcborder[C].at<float>(J, I)) * (val - srcborder[C].at<float>(J, I));
					}
				}
			}
			double total_error = 0.0;
			for (int t = 0; t < thread_max; t++)total_error += error[t];
			const double mse = total_error / (width * height * dim);
			ret = 10.0 * log10(255.0 * 255.0 / mse);
			return ret;
		}
	}
	else
	{
		if (is2DTest)
		{
			vector<Mat> ev(dest_channels);
			for (int d = 0; d < dest_channels; d++)
			{
				ev[d].create(evecV.cols, evecH.cols, CV_32F);
				for (int j = 0; j < evecV.cols; j++)
				{
					for (int i = 0; i < evecH.cols; i++)
					{
						ev[d].at<float>(j, i) = evecH.at<float>(d, i) * evecV.at<float>(d, j);
						//ev[d].at<float>(j, i) = evec.at<float>(d, evecH.cols * j + i);
					}
				}
			}
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i++)
				{
					//projection
					for (int d = 0; d < dest_channels; d++)
					{
						float val = 0.f;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								for (int c = 0; c < src_channels; c++)
								{
									val = fma(srcborder[c].at<float>(j + l, i + m), ev[d].at<float>(l, src_channels * m + c), val);
									//val = fma(srcborder[c].at<float>(i + m, j + l), ev[d].at<float>(l, src_channels * m + c), val);
									//val += srcborder[c].at<float>(j + l, i + m) * ev[d].at<float>(l, src_channels * m + c);
								}
							}
						}
						dest[d].at<float>(j, i) = val;
					}
				}
			}
		}
		else
		{
			// H loop
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < srcborder[0].rows; j++)
			{
				for (int i = 0; i < width; i++)
				{
					//projection
					for (int d = 0; d < dest_channels; d++)
					{
						float mval = 0.f;
						for (int l = 0, idx = 0; l < D; l++)
						{
							for (int c = 0; c < src_channels; c++)
							{
								mval += srcborder[c].at<float>(j, i + l) * eHptr[d][idx];
								idx++;
							}
						}
						projection[d].at<float>(j, i) = mval;
					}
				}
			}

			//V loop
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i++)
				{
					//projection
					for (int d = 0; d < dest_channels; d++)
					{
						float mval = 0.f;
						for (int h = 0; h < dimV; h++)
						{
							mval += projection[d].at<float>(j + h, i) * eVptr[d][h];
						}
						dest[d].at<float>(j, i) = mval;
					}
				}
			}
		}
	}
}

void DRIM2COL(const Mat& src, Mat& dst, const int neighborhood_r, const int dest_channels, const int border, const int method, const bool isParallel, const double const_sub)
{
	CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);

	const int width = src.cols;
	const int height = src.rows;
	const int ch = src.channels();
	const int d = 2 * neighborhood_r + 1;
	const int patch_area = d * d;
	const int dim = ch * patch_area;
	const Size imsize = src.size();
	const int imarea = imsize.area();
	const int num_points = cvRound((float)imarea * 0.1);

	if (double(src.size().area() * 255 * 255) > FLT_MAX)
	{
		cout << "overflow in float" << endl;
	}

	if (method < (int)DRIM2COLType::OPENCV_PCA)
	{
		Mat cov, eval, evec;
		CalcPatchCovarMatrix pcov; pcov.computeCov(src, neighborhood_r, cov, (DRIM2COLType)method, 1, isParallel);

		eigen(cov, eval, evec);
		//if (isParallel) imshowNeighboorhoodEigenVectors("evec", evec, 1);

		Mat transmat;
		evec(Rect(0, 0, evec.cols, dest_channels)).convertTo(transmat, CV_32F);
		DRIM2COLEigenVec(src, transmat, dst, neighborhood_r, 1, border, isParallel);
	}
	else
	{
		Mat highDimGuide(src.size(), CV_MAKE_TYPE(CV_32F, dim));
		{
			//cp::Timer t("cvt HDI");
			IM2COL(src, highDimGuide, neighborhood_r, border);
		}

		if (method == (int)DRIM2COLType::OPENCV_PCA)
		{
			PCA pca(highDimGuide.reshape(1, imsize.area()), cv::Mat(), cv::PCA::DATA_AS_ROW, dest_channels);
			dst = pca.project(highDimGuide.reshape(1, imsize.area())).reshape(dest_channels, src.rows);
		}
		else if (method == (int)DRIM2COLType::OPENCV_COV)
		{
			Mat x = highDimGuide.reshape(1, imsize.area());
			Mat cov, mean;
			cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
			//print_mat(cov);
			Mat eval, evec;
			eigen(cov, eval, evec);

			Mat transmat;
			evec(Rect(0, 0, evec.cols, dest_channels)).convertTo(transmat, CV_32F);
			//transmat = Mat::eye(transmat.size(), CV_32F);
			cv::transform(highDimGuide, dst, transmat);
		}
		else if (method == (int)DRIM2COLType::NO_SUB_SEPSVD)
		{
			Mat cov, eval, evec, U, transmat, Uv, Uh;
			CalcPatchCovarMatrix pcov; pcov.computeCov(src, neighborhood_r, cov, DRIM2COLType::FULL_SUB_HALF_32F, 1, isParallel);
			cv::eigen(cov, eval, evec);

			separateEvecSVD(evec, ch, U, Uv, Uh);

			U(Rect(0, 0, evec.cols, dest_channels)).convertTo(transmat, CV_32F);;
			DRIM2COLEigenVec(src, transmat, dst, neighborhood_r, 1, border, isParallel);
		}
		else if (method == (int)DRIM2COLType::NO_SUB_SEPCOVX)
		{
			Mat cov, eval, evec, U, transmat;
			vector<Mat> vcov(2);
			CalcPatchCovarMatrix pcov; pcov.computeSepCov(src, neighborhood_r, vcov, (DRIM2COLType)method, 1, isParallel);

			Mat Uh, Uv;
			calcEvecfromSepCov(vcov[0], vcov[1], dest_channels, U, Uh, Uv);

			U(Rect(0, 0, U.cols, dest_channels)).convertTo(transmat, CV_32F);
			DRIM2COLEigenVec(src, transmat, dst, neighborhood_r, 1, border, isParallel);
		}
	}
}

void DRIM2COL(const vector<Mat>& vsrc, vector<Mat>& vdst, const int neighborhood_r, const int dest_channels, const int border, const int method, const bool isParallel, const double const_sub)
{
	CV_Assert(vsrc[0].depth() == CV_8U || vsrc[0].depth() == CV_32F);

	const int width = vsrc[0].cols;
	const int height = vsrc[0].rows;
	const int ch = (int)vsrc.size();
	const int d = 2 * neighborhood_r + 1;
	const int patch_area = d * d;
	const int dim = ch * patch_area;
	const int channels = min(dest_channels, ch * patch_area);
	const Size imsize = vsrc[0].size();
	const int imarea = imsize.area();
	const int num_points = cvRound((float)imarea * 0.1);

	if (double(vsrc[0].size().area() * 255 * 255) > FLT_MAX)
	{
		cout << "overflow in float" << endl;
	}

	if (method < (int)DRIM2COLType::OPENCV_PCA)
	{
		Mat cov;
		CalcPatchCovarMatrix pcov;
		pcov.setBorder(border);
		pcov.computeCov(vsrc, neighborhood_r, cov, (DRIM2COLType)method, 1, isParallel);

		Mat eval, evec;
		eigen(cov, eval, evec);
		Mat U32F;
		evec(Rect(0, 0, evec.cols, channels)).convertTo(U32F, CV_32F);

		switch (channels)
		{
		case 1: DRIM2COLEigenVec<1>(vsrc, U32F, vdst, neighborhood_r, border); break;
		case 2: DRIM2COLEigenVec<2>(vsrc, U32F, vdst, neighborhood_r, border); break;
		case 3: DRIM2COLEigenVec<3>(vsrc, U32F, vdst, neighborhood_r, border); break;
		case 4: DRIM2COLEigenVec<4>(vsrc, U32F, vdst, neighborhood_r, border); break;
		case 5: DRIM2COLEigenVec<5>(vsrc, U32F, vdst, neighborhood_r, border); break;
		case 6: DRIM2COLEigenVec<6>(vsrc, U32F, vdst, neighborhood_r, border); break;
		case 7: DRIM2COLEigenVec<7>(vsrc, U32F, vdst, neighborhood_r, border); break;
		case 8: DRIM2COLEigenVec<8>(vsrc, U32F, vdst, neighborhood_r, border); break;
		case 9: DRIM2COLEigenVec<9>(vsrc, U32F, vdst, neighborhood_r, border); break;
		default:
			DRIM2COLEigenVecCn(vsrc, U32F, vdst, neighborhood_r, border);
			break;
		}
	}
	else if (isSepCov((DRIM2COLType)method, true))
	{
		Mat cov;
		CalcPatchCovarMatrix pcov;
		pcov.setBorder(border);
		pcov.setConstSub((int)const_sub);
		if (method == (int)DRIM2COLType::MEAN_SUB_SEPSVD) pcov.computeCov(vsrc, neighborhood_r, cov, DRIM2COLType::MEAN_SUB_HALF_32F, 1, isParallel);
		if (method == (int)DRIM2COLType::NO_SUB_SEPSVD) pcov.computeCov(vsrc, neighborhood_r, cov, DRIM2COLType::NO_SUB_HALF_32F, 1, isParallel);
		if (method == (int)DRIM2COLType::CONST_SUB_SEPSVD) pcov.computeCov(vsrc, neighborhood_r, cov, DRIM2COLType::CONST_SUB_HALF_32F, 1, isParallel);
		Mat eval, evec, U, U32F;
		cv::eigen(cov, eval, evec);
		Mat Uh, Uv;
		separateEvecSVD(evec, ch, U, Uh, Uv);
		U(Rect(0, 0, evec.cols, channels)).convertTo(U32F, CV_32F);
		Mat Uh32F, Uv32F;
		Uv(Rect(0, 0, Uv.cols, channels)).convertTo(Uv32F, CV_32F);
		Uh(Rect(0, 0, Uh.cols, channels)).convertTo(Uh32F, CV_32F);

		DRIM2COLSepEigenVecCn(vsrc, Uh32F, Uv32F, vdst, neighborhood_r, border);
		//DRIM2COLEigenVecCn(vsrc, U32F, vdst, neighborhood_r, border);
	}
	else if (isSepCov((DRIM2COLType)method, false))
	{
		vector<Mat> vcov;
		CalcPatchCovarMatrix pcov;
		pcov.setBorder(border);
		pcov.setConstSub((int)const_sub);
		pcov.computeSepCov(vsrc, neighborhood_r, vcov, (DRIM2COLType)method, 1, isParallel);

		Mat U, Uh, Uv;
		calcEvecfromSepCov(vcov[0], vcov[1], channels, U, Uh, Uv);
		//Mat transmat= cp::convert(U, CV_32F);
		Mat Uh32F = cp::convert(Uh, CV_32F);
		Mat Uv32F = cp::convert(Uv, CV_32F);
		Mat U32F = cp::convert(U, CV_32F);

		if (DRIM2COLType::MEAN_SUB_SEPCOVX == (DRIM2COLType)method ||
			DRIM2COLType::NO_SUB_SEPCOVX == (DRIM2COLType)method ||
			DRIM2COLType::CONST_SUB_SEPCOVX == (DRIM2COLType)method)
		{
			DRIM2COLSepEigenVecCn(vsrc, Uh32F, Uv32F, vdst, neighborhood_r, border);
			//DRIM2COLEigenVecCn(vsrc, U32F, vdst, neighborhood_r, border);
		}
		else
		{
			vector<Mat> vsrct(vsrc.size());
			for (int c = 0; c < vsrct.size(); c++)
			{
				transpose(vsrc[c], vsrct[c]);
			}
			DRIM2COLSepEigenVecCn(vsrct, Uh32F, Uv32F, vdst, neighborhood_r, border);
		}
	}
	else
	{
		Mat highDimGuide(vsrc[0].size(), CV_MAKE_TYPE(CV_32F, dim));
		{
			//cp::Timer t("cvt HDI");
			IM2COL(vsrc, highDimGuide, neighborhood_r, border);
		}
		if (method == (int)DRIM2COLType::OPENCV_PCA)
		{
			PCA pca(highDimGuide.reshape(1, imsize.area()), cv::Mat(), cv::PCA::DATA_AS_ROW, channels);
			Mat temp = pca.project(highDimGuide.reshape(1, imsize.area())).reshape(channels, vsrc[0].rows);
			split(temp, vdst);
		}
		else if (method == (int)DRIM2COLType::OPENCV_COV)
		{
			Mat x = highDimGuide.reshape(1, imsize.area());
			Mat cov, mean;
			cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
			//cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL |  cv::COVAR_ROWS);
			//print_mat(cov);
			Mat eval, evec;
			eigen(cov, eval, evec);

			Mat transmat;
			//print_matinfo(evec);
			evec(Rect(0, 0, evec.cols, channels)).convertTo(transmat, CV_32F);
			//transmat = Mat::eye(transmat.size(), CV_32F);

			switch (channels)
			{
			case 1: DRIM2COLEigenVec<1>(vsrc, transmat, vdst, neighborhood_r, border); break;
			case 2: DRIM2COLEigenVec<2>(vsrc, transmat, vdst, neighborhood_r, border); break;
			case 3: DRIM2COLEigenVec<3>(vsrc, transmat, vdst, neighborhood_r, border); break;
			case 4: DRIM2COLEigenVec<4>(vsrc, transmat, vdst, neighborhood_r, border); break;
			case 5: DRIM2COLEigenVec<5>(vsrc, transmat, vdst, neighborhood_r, border); break;
			case 6: DRIM2COLEigenVec<6>(vsrc, transmat, vdst, neighborhood_r, border); break;
			case 7: DRIM2COLEigenVec<7>(vsrc, transmat, vdst, neighborhood_r, border); break;
			case 8: DRIM2COLEigenVec<8>(vsrc, transmat, vdst, neighborhood_r, border); break;
			case 9: DRIM2COLEigenVec<9>(vsrc, transmat, vdst, neighborhood_r, border); break;
			default:
				DRIM2COLEigenVecCn(vsrc, transmat, vdst, neighborhood_r, border);
				break;
			}
			/*
			Mat temp;
			cv::transform(highDimGuide, temp, transmat);
			split(temp, dst);
			*/
		}
	}
}

void DRIM2COLTile(const Mat& src, Mat& dest, const int neighborhood_r, const int dest_channels, const int border, const int method, const Size div)
{
	dest.create(src.size(), CV_MAKE_TYPE(CV_32F, dest_channels));
	const int channels = src.channels();

	const int vecsize = sizeof(__m256) / sizeof(float);//8

	if (div.area() == 1)
	{
		DRIM2COL(src, dest, neighborhood_r, dest_channels, border, method);
	}
	else
	{
		int r = neighborhood_r;
		const int R = get_simd_ceil(r, 8);
		Size tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
		Size divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

		vector<Mat> split_dst(channels);

		for (int c = 0; c < channels; c++)
		{
			split_dst[c].create(tileSize, CV_32FC1);
		}

		const int thread_max = omp_get_max_threads();
		vector<vector<Mat>>	subImageInput(thread_max);
		vector<vector<Mat>>	subImageOutput(thread_max);
		vector<Mat>	subImageOutput2(thread_max);
		for (int n = 0; n < thread_max; n++)
		{
			subImageInput[n].resize(channels);
			subImageOutput[n].resize(channels);
		}

		std::vector<cv::Mat> srcSplit;
		if (src.channels() != 3)split(src, srcSplit);

#pragma omp parallel for schedule(static)
		for (int n = 0; n < div.area(); n++)
		{
			const int thread_num = omp_get_thread_num();
			const cv::Point idx = cv::Point(n % div.width, n / div.width);

			if (src.channels() == 3)
			{
				cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, border, vecsize, vecsize, vecsize, vecsize);
			}
			else
			{
				for (int c = 0; c < srcSplit.size(); c++)
				{
					cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, border, vecsize, vecsize, vecsize, vecsize);
				}
			}
			DRIM2COL(subImageInput[thread_num], subImageOutput[thread_num], neighborhood_r, dest_channels, border, method);
			merge(subImageOutput[thread_num], subImageOutput2[thread_num]);
			cp::pasteTileAlign(subImageOutput2[thread_num], dest, div, idx, r, 8, 8);
		}
	}
}

#pragma endregion

#pragma region reprojectDRIM2COL

void reprojectNeighborhoodEigenVec(const vector<Mat>& vsrc, vector<Mat>& dest, const Mat& evec, const int r, const int border)
{
	const Mat evecInv = evec.t();
	const int src_channels = (int)vsrc.size();
	const int dest_channels = evec.rows;

	const int width = vsrc[0].cols;
	const int simd_end = get_simd_floor(width, 8);
	const __m256i mask = get_simd_residualmask_epi32(width);
	const bool isRem = (width == simd_end) ? false : true;
	const int height = vsrc[0].rows;

	const int D = 2 * r + 1;
	const int DD = D * D;
	const int dim = DD * src_channels;//D * D * src_channels()
	dest.resize(dim);
	for (int i = 0; i < dim; i++) dest[i].create(vsrc[0].size(), vsrc[0].type());

	AutoBuffer<const float*> eptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eptr[m] = evec.ptr<float>(m);
	}

	AutoBuffer<const float*> einvptr(dim);
	for (int d = 0; d < dim; d++)
	{
		einvptr[d] = evecInv.ptr<float>(d);
	}

	vector<Mat> srcborder(src_channels);
	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(vsrc[c], srcborder[c], r, r, r, r, border);
	}

	AutoBuffer<__m256> patch(dim);
	AutoBuffer<__m256> proj(dest_channels);
	AutoBuffer<__m256> repatch(dim);
	//project
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < simd_end; i += 8)
		{
			for (int l = 0, idx = 0; l < D; l++)
			{
				for (int k = 0; k < D; k++)
				{
					for (int c = 0; c < src_channels; c++)
					{
						patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + l, i + k));
					}
				}
			}

			for (int c = 0; c < dest_channels; c++)
			{
				__m256 mval = _mm256_setzero_ps();
				for (int d = 0; d < dim; d++)
				{
					mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
				}
				proj[c] = mval;
			}
			for (int d = 0; d < dim; d++)
			{
				__m256 mrep = _mm256_setzero_ps();
				for (int c = 0; c < dest_channels; c++)
				{
					mrep = _mm256_fmadd_ps(proj[c], _mm256_set1_ps(einvptr[d][c]), mrep);
				}
				_mm256_storeu_ps(dest[d].ptr<float>(j, i), mrep);
			}
		}

		if (isRem)
		{
			for (int l = 0, idx = 0; l < D; l++)
			{
				for (int k = 0; k < D; k++)
				{
					for (int c = 0; c < src_channels; c++)
					{
						patch[idx++] = _mm256_maskload_ps(srcborder[c].ptr<float>(j + l, simd_end + k), mask);
					}
				}
			}

			for (int c = 0; c < dest_channels; c++)
			{
				__m256 mval = _mm256_setzero_ps();
				for (int d = 0; d < dim; d++)
				{
					mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
				}
				proj[c] = mval;
			}
			for (int d = 0; d < dim; d++)
			{
				__m256 mrep = _mm256_setzero_ps();
				for (int c = 0; c < dest_channels; c++)
				{
					mrep = _mm256_fmadd_ps(proj[c], _mm256_set1_ps(einvptr[d][c]), mrep);
				}
				_mm256_maskstore_ps(dest[d].ptr<float>(j, simd_end), mask, mrep);
			}
		}
	}
}

void reprojectNeighborhoodSepEigenVec(const vector<Mat>& vsrc, vector<Mat>& dest, const Mat& evecH, const Mat& evecV, const int patch_radius, const int borderType)
{
	Mat evecHInv = evecH.t();
	Mat evecVInv = evecV.t();
	const int src_channels = (int)vsrc.size();
	const int dest_channels = evecH.rows;

	const int width = vsrc[0].cols;
	const int height = vsrc[0].rows;

	const int D = 2 * patch_radius + 1;
	//const int dimV = D * src_channels;//D * src_channels()
	//const int dimH = D;//D

	AutoBuffer<const float*> eHptr(dest_channels);
	AutoBuffer<const float*> eVptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eHptr[m] = evecH.ptr<float>(m);
		eVptr[m] = evecV.ptr<float>(m);
	}

	const int wb8 = get_simd_ceil(width + 2 * patch_radius, 8);
	const int off = wb8 - width + 2 * patch_radius;
	vector<Mat> srcborder(src_channels);
	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(vsrc[c], srcborder[c], patch_radius, patch_radius, patch_radius, patch_radius + off, borderType);
	}

	dest.resize(D * D * src_channels);
	for (int c = 0; c < D * D * src_channels; c++)
	{
		dest[c].create(vsrc[0].size(), CV_32F);
	}
	const int bb = 0;
	const int stx = bb;
	const int edx = width - bb;
	const int edbx = srcborder[0].cols - bb;
	const int simd_end = get_simd_floor(edx - stx, 8);
	const int rem = edx - stx - simd_end;
	const __m256i mask = get_simd_residualmask_epi32(edx - stx);
	const bool isRem = (rem == 0) ? false : true;
	const int sty = bb;
	const int edy = height - bb;

	const int unroll = 8;
	if (unroll == 8)
	{
		const int simd_bend = get_simd_floor(edbx - stx, 8);
		const int simd_end = get_simd_floor(edx - stx, 8);
		const __m256i mask = get_simd_residualmask_epi32(edx - stx);
		const int rem = edx - stx - simd_end;
		const bool isRem = (rem == 0) ? false : true;
#pragma omp parallel for schedule(dynamic)
		for (int j = sty; j < edy; j++)
		{
			const int t = omp_get_thread_num();
			AutoBuffer<__m256> patch(src_channels * D);
			Mat projVHI(dest_channels, srcborder[0].cols, CV_32F);
			//projection
			for (int i = stx; i < simd_bend; i += 8)
			{
				//load data
				for (int v = 0, idx = 0; v < D; v++)
				{
					for (int c = 0; c < src_channels; c++)
					{
						patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + v, i));
					}
				}

				//projection
				for (int d = 0; d < dest_channels; d++)
				{
					__m256 mval = _mm256_mul_ps(patch[0], _mm256_set1_ps(eHptr[d][0]));
					for (int h = 1; h < D * src_channels; h++)
					{
						mval = _mm256_fmadd_ps(patch[h], _mm256_set1_ps(eHptr[d][h]), mval);
					}
					_mm256_storeu_ps(projVHI.ptr<float>(d, i), mval);
				}
			}
			// H loop
			for (int i = stx; i < simd_end; i += 8)
			{
				AutoBuffer<__m256> rep(dest_channels);
				for (int d = 0; d < dest_channels; d++)
				{
					const float* p = projVHI.ptr<float>(d, i);
					__m256 mval = _mm256_setzero_ps();
					for (int h = 0; h < D; h++)
					{
						mval = _mm256_fmadd_ps(_mm256_loadu_ps(p + h), _mm256_set1_ps(eVptr[d][h]), mval);
					}
					rep[d] = mval;
				}
				//reprojection
				for (int h = 0, idx = 0; h < D; h++)
				{
					for (int v = 0; v < D; v++)
					{
						for (int c = 0; c < src_channels; c++)
						{
							__m256 mval = _mm256_setzero_ps();
							for (int d = 0; d < dest_channels; d++)
							{
								mval = _mm256_fmadd_ps(rep[d], _mm256_set1_ps(evecHInv.at<float>(src_channels * h + c, d) * evecVInv.at<float>(v, d)), mval);
							}
							_mm256_storeu_ps(dest[idx].ptr<float>(j, i), mval);
							idx++;
						}
					}
				}
			}
		}
	}
}

void reprojectIM2COL(const vector<Mat>& vsrc, vector<Mat>& dest, const int neighborhood_r, const int border)
{
	CV_Assert(vsrc[0].depth() == CV_8U || vsrc[0].depth() == CV_32F);
	const int channels = (int)vsrc.size();
	vector<Mat> vborder(channels);
	for (int c = 0; c < channels; c++)
	{
		copyMakeBorder(vsrc[c], vborder[c], neighborhood_r, neighborhood_r, neighborhood_r, neighborhood_r, border);
	}

	const int D = 2 * neighborhood_r + 1;
	dest.resize(channels * D * D);
	/*for (int j = 0, idx = 0; j < D; j++)
	{
		for (int i = 0; i < D; i++)
		{
			for (int c = 0; c < channels; c++)
			{
				vborder[c](Rect(i, j, vsrc[0].cols, vsrc[0].rows)).copyTo(dest[idx++]);
			}
		}
	}*/
	for (int j = 0, idx = 0; j < D; j++)
	{
		for (int i = 0; i < D; i++)
		{
			for (int c = 0; c < channels; c++)
			{
				vborder[c](Rect(j, i, vsrc[0].cols, vsrc[0].rows)).copyTo(dest[idx++]);
			}
		}
	}
}

//see vector varsion for detailed implementation
void reprojectDRIM2COL(const Mat& src, Mat& dest, const int neighborhood_r, const int dest_channels, const int border, const DRIM2COLType type, const bool isParallel, const double const_sub)
{
	vector<Mat> temp;
	vector<Mat> temp2;
	split(src, temp);
	reprojectDRIM2COL(temp, temp2, neighborhood_r, dest_channels, border, type, isParallel, const_sub);
}

void reprojectDRIM2COL(const vector<Mat>& vsrc, vector<Mat>& dest, const int neighborhood_r, const int dest_channels, const int border, const DRIM2COLType type, const bool isParallel, const double const_sub)
{
	CV_Assert(vsrc[0].depth() == CV_8U || vsrc[0].depth() == CV_32F);

	const int width = vsrc[0].cols;
	const int height = vsrc[0].rows;
	const int ch = (int)vsrc.size();
	const int d = 2 * neighborhood_r + 1;
	const int patch_area = d * d;
	const int dim = ch * patch_area;
	const int channels = min(dest_channels, dim);
	const Size imsize = vsrc[0].size();
	const int imarea = imsize.area();
	const int num_points = cvRound((float)imarea * 0.1);

	if (double(vsrc[0].size().area() * 255 * 255) > FLT_MAX)
	{
		cout << "overflow in float" << endl;
	}

	if ((int)type < (int)DRIM2COLType::OPENCV_PCA)
	{
		Mat cov;
		CalcPatchCovarMatrix pcov;
		pcov.setBorder(border);
		pcov.setConstSub((int)const_sub);
		pcov.computeCov(vsrc, neighborhood_r, cov, (DRIM2COLType)type, 1, isParallel);

		Mat eval, evec;
		cv::eigen(cov, eval, evec);

		Mat U32F;
		evec(Rect(0, 0, evec.cols, channels)).convertTo(U32F, CV_32F);
		reprojectNeighborhoodEigenVec(vsrc, dest, U32F, neighborhood_r, border);
	}
	else if (isSepCov(type, true))
	{
		Mat cov;
		CalcPatchCovarMatrix pcov;
		pcov.setBorder(border);
		pcov.setConstSub((int)const_sub);
		if (type == DRIM2COLType::MEAN_SUB_SEPSVD) pcov.computeCov(vsrc, neighborhood_r, cov, DRIM2COLType::MEAN_SUB_HALF_32F, 1, isParallel);
		if (type == DRIM2COLType::NO_SUB_SEPSVD) pcov.computeCov(vsrc, neighborhood_r, cov, DRIM2COLType::NO_SUB_HALF_32F, 1, isParallel);
		if (type == DRIM2COLType::CONST_SUB_SEPSVD) pcov.computeCov(vsrc, neighborhood_r, cov, DRIM2COLType::CONST_SUB_HALF_32F, 1, isParallel);
		Mat eval, evec, U, U32F, Uh, Uv;
		cv::eigen(cov, eval, evec);
		separateEvecSVD(evec, ch, U, Uh, Uv);

		U(Rect(0, 0, evec.cols, channels)).convertTo(U32F, CV_32F);
		reprojectNeighborhoodEigenVec(vsrc, dest, U32F, neighborhood_r, border);
	}
	else if (isSepCov(type, false))
	{
		vector<Mat> vcov;
		CalcPatchCovarMatrix pcov;
		pcov.setBorder(border);
		pcov.setConstSub((int)const_sub);
		pcov.computeSepCov(vsrc, neighborhood_r, vcov, type, 1, isParallel);

		Mat U, Uh, Uv;
		calcEvecfromSepCov(vcov[0], vcov[1], channels, U, Uh, Uv);
		//Mat transmat= cp::convert(U, CV_32F);
		Mat U32F = cp::convert(U, CV_32F);
		Mat Uh32F = cp::convert(Uh, CV_32F);
		Mat Uv32F = cp::convert(Uv, CV_32F);
		if (DRIM2COLType::MEAN_SUB_SEPCOVX == type ||
			DRIM2COLType::NO_SUB_SEPCOVX == type ||
			DRIM2COLType::CONST_SUB_SEPCOVX == type)
		{
			reprojectNeighborhoodEigenVec(vsrc, dest, U32F, neighborhood_r, border);
			//reprojectNeighborhoodSepEigenVec(vsrc, dest, Uh32F, Uv32F, neighborhood_r, border);
			//vector<Mat> vtemp;
			//ret = DRIM2COLTestCn(vsrc, transmat, Uh32F, Uv32F, vtemp, neighborhood_r, border);
		}
		else
		{
			vector<Mat> vsrct(vsrc.size());
			for (int c = 0; c < vsrct.size(); c++)
			{
				transpose(vsrc[c], vsrct[c]);
			}
			reprojectNeighborhoodSepEigenVec(vsrct, dest, Uh32F, Uv32F, neighborhood_r, border);
		}
		/*
		{
			//using full
			Mat transmatInv = transmat.t();
			ret = getPSNRReprojectDRIM2COLEigenVec(vsrc, transmat, transmatInv, neighborhood_r, border, bb, isNormalizeDimension);
		}*/
	}
	else
	{
		Mat highDimGuide(vsrc[0].size(), CV_MAKE_TYPE(CV_32F, dim));
		{
			//cp::Timer t("cvt HDI");
			IM2COL(vsrc, highDimGuide, neighborhood_r, border);
		}
		Mat x = highDimGuide.reshape(1, imsize.area());
		Mat cov, mean;
		cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
		//cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL |  cv::COVAR_ROWS);
		//print_mat(cov);
		Mat eval, evec;
		eigen(cov, eval, evec);

		Mat U32F;
		evec(Rect(0, 0, evec.cols, channels)).convertTo(U32F, CV_32F);
		reprojectNeighborhoodEigenVec(vsrc, dest, U32F, neighborhood_r, border);
	}
}
#pragma endregion

#pragma region getPSNR_DRIM2COL

double getPSNRReprojectDRIM2COLEigenVec(const vector<Mat>& vsrc, const Mat& evec, const int patch_radius, const int borderType, const int bb, const bool isNormalizeDimension)
{
	//const Mat evecInv = evec.t();
	Mat evecInv;
	invert(evec, evecInv, DECOMP_SVD);

	const int src_channels = (int)vsrc.size();
	const int dest_channels = evec.rows;

	const int width = vsrc[0].cols;
	const int height = vsrc[0].rows;

	const int D = 2 * patch_radius + 1;
	const int DD = D * D;
	const int dim = DD * src_channels;//D * D * src_channels()

	AutoBuffer<const float*> eptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eptr[m] = evec.ptr<float>(m);
	}

	AutoBuffer<const float*> einvptr(dim);
	for (int d = 0; d < dim; d++)
	{
		einvptr[d] = evecInv.ptr<float>(d);
	}

	vector<Mat> srcborder(src_channels);

	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(vsrc[c], srcborder[c], patch_radius, patch_radius, patch_radius, patch_radius, borderType);
	}

	const int stx = bb;
	const int edx = width - bb;
	const int simd_end = get_simd_floor(edx - stx, 8);
	const int rem = edx - stx - simd_end;
	const __m256i mask = get_simd_residualmask_epi32(edx - stx);
	const bool isRem = (rem == 0) ? false : true;
	const int sty = bb;
	const int edy = height - bb;
	const int area = (isNormalizeDimension) ? (edx - stx) * (edy - sty) * dim : (edx - stx) * (edy - sty);

	const int thread_max = omp_get_max_threads();
	AutoBuffer<__m256d> ssd0(thread_max);
	AutoBuffer<__m256d> ssd1(thread_max);
	for (int t = 0; t < thread_max; t++)
	{
		ssd0[t] = _mm256_setzero_pd();
		ssd1[t] = _mm256_setzero_pd();
	}
#pragma omp parallel for
	for (int j = sty; j < edy; j++)
	{
		const int t = omp_get_thread_num();
		AutoBuffer<__m256> patch(dim);
		AutoBuffer<__m256> proj(dest_channels);
		AutoBuffer<__m256> repatch(dim);
		for (int i = stx; i < simd_end; i += 8)
		{
			//load data
			for (int l = 0, idx = 0; l < D; l++)
			{
				for (int k = 0; k < D; k++)
				{
					for (int c = 0; c < src_channels; c++)
					{
						patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + l, i + k));
					}
				}
			}
			//projection
			for (int c = 0; c < dest_channels; c++)
			{
				__m256 mval = _mm256_setzero_ps();
				for (int d = 0; d < dim; d++)
				{
					mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
				}
				proj[c] = mval;
			}
			//reprojection
			for (int d = 0; d < dim; d++)
			{
				__m256 mrep = _mm256_setzero_ps();
				for (int c = 0; c < dest_channels; c++)
				{
					mrep = _mm256_fmadd_ps(proj[c], _mm256_set1_ps(einvptr[d][c]), mrep);
				}
				__m256 sub = _mm256_sub_ps(patch[d], mrep);
				__m256d sd0 = _mm256_cvtps_pd(_mm256_castps256_ps128(sub));
				__m256d sd1 = _mm256_cvtps_pd(_mm256_castps256hi_ps128(sub));
				ssd0[t] = _mm256_fmadd_pd(sd0, sd0, ssd0[t]);
				ssd1[t] = _mm256_fmadd_pd(sd1, sd1, ssd1[t]);
			}
		}

		//last rem 
		if (isRem)
		{
			for (int l = 0, idx = 0; l < D; l++)
			{
				for (int k = 0; k < D; k++)
				{
					for (int c = 0; c < src_channels; c++)
					{
						patch[idx++] = _mm256_maskload_ps(srcborder[c].ptr<float>(j + l, simd_end + k), mask);
					}
				}
			}

			for (int c = 0; c < dest_channels; c++)
			{
				__m256 mval = _mm256_setzero_ps();
				for (int d = 0; d < dim; d++)
				{
					mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
				}
				proj[c] = mval;
			}

			for (int d = 0; d < dim; d++)
			{
				__m256 mrep = _mm256_setzero_ps();
				for (int c = 0; c < dest_channels; c++)
				{
					mrep = _mm256_fmadd_ps(proj[c], _mm256_set1_ps(einvptr[d][c]), mrep);
				}
				__m256 sub = _mm256_sub_ps(patch[d], mrep);
				__m256d sd0 = _mm256_cvtps_pd(_mm256_castps256_ps128(sub));
				__m256d sd1 = _mm256_cvtps_pd(_mm256_castps256hi_ps128(sub));
				ssd0[t] = _mm256_fmadd_pd(sd0, sd0, ssd0[t]);
				ssd1[t] = _mm256_fmadd_pd(sd1, sd1, ssd1[t]);
			}
		}
	}

	double mse = 0.0;
	for (int t = 0; t < thread_max; t++)
	{
		mse += _mm256_reduceadd_pd(ssd0[t]) + _mm256_reduceadd_pd(ssd1[t]);
	}
	mse = mse / area;
	return 10.0 * log10(255.0 * 255.0 / mse);
}

double getPSNRReprojectDRIM2COLSepEigenVec(const vector<Mat>& vsrc, const Mat& evecH, const Mat& evecV, const int patch_radius, const int borderType, const int bb, const bool isNormalizeDimension)
{
	Mat evecHInv = evecH.t();
	Mat evecVInv = evecV.t();

	const int src_channels = (int)vsrc.size();
	const int dest_channels = evecH.rows;

	const int width = vsrc[0].cols;
	const int height = vsrc[0].rows;

	const int D = 2 * patch_radius + 1;
	//const int dimV = D * src_channels;//D * src_channels()
	//const int dimH = D;//D

	AutoBuffer<const float*> eHptr(dest_channels);
	AutoBuffer<const float*> eVptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eHptr[m] = evecH.ptr<float>(m);
		eVptr[m] = evecV.ptr<float>(m);
	}

	const int wb8 = get_simd_ceil(width + 2 * patch_radius, 8);
	const int off = wb8 - width + 2 * patch_radius;
	vector<Mat> srcborder(src_channels);
	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(vsrc[c], srcborder[c], patch_radius, patch_radius, patch_radius, patch_radius + off, borderType);
	}
	const int stx = bb;
	const int edx = width - bb;
	const int edbx = srcborder[0].cols - bb;
	const int simd_end = get_simd_floor(edx - stx, 8);
	const int rem = edx - stx - simd_end;
	const __m256i mask = get_simd_residualmask_epi32(edx - stx);
	const bool isRem = (rem == 0) ? false : true;
	const int sty = bb;
	const int edy = height - bb;
	const int area = (isNormalizeDimension) ? (edx - stx) * (edy - sty) * D * D * src_channels : (edx - stx) * (edy - sty);

	const int thread_max = omp_get_max_threads();
	AutoBuffer<__m256d> ssd0(thread_max);
	AutoBuffer<__m256d> ssd1(thread_max);
	for (int t = 0; t < thread_max; t++)
	{
		ssd0[t] = _mm256_setzero_pd();
		ssd1[t] = _mm256_setzero_pd();
	}

	const int unroll = 8;
	if (unroll == 8)
	{
		const int simd_bend = get_simd_floor(edbx - stx, 8);
		const int simd_end = get_simd_floor(edx - stx, 8);
		const __m256i mask = get_simd_residualmask_epi32(edx - stx);
		const int rem = edx - stx - simd_end;
		const bool isRem = (rem == 0) ? false : true;
#pragma omp parallel for schedule(dynamic)
		for (int j = sty; j < edy; j++)
		{
			const int t = omp_get_thread_num();
			AutoBuffer<__m256> patch(src_channels * D);
			Mat projVHI(dest_channels, srcborder[0].cols, CV_32F);
			//projection
			for (int i = stx; i < simd_bend; i += 8)
			{
				//load data
				for (int v = 0, idx = 0; v < D; v++)
				{
					for (int c = 0; c < src_channels; c++)
					{
						patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + v, i));
					}
				}

				//projection
				for (int d = 0; d < dest_channels; d++)
				{
					__m256 mval = _mm256_mul_ps(patch[0], _mm256_set1_ps(eHptr[d][0]));
					for (int h = 1; h < D * src_channels; h++)
					{
						mval = _mm256_fmadd_ps(patch[h], _mm256_set1_ps(eHptr[d][h]), mval);
					}
					_mm256_storeu_ps(projVHI.ptr<float>(d, i), mval);
				}
			}
			// H loop
			for (int i = stx; i < simd_end; i += 8)
			{
				AutoBuffer<__m256> rep(dest_channels);
				for (int d = 0; d < dest_channels; d++)
				{
					const float* p = projVHI.ptr<float>(d, i);
					__m256 mval = _mm256_setzero_ps();
					for (int h = 0; h < D; h++)
					{
						mval = _mm256_fmadd_ps(_mm256_loadu_ps(p + h), _mm256_set1_ps(eVptr[d][h]), mval);
					}
					rep[d] = mval;
				}
				//reprojection
				for (int v = 0; v < D; v++)
				{
					for (int h = 0; h < D; h++)
					{
						for (int c = 0; c < src_channels; c++)
						{
							__m256 mval = _mm256_setzero_ps();
							for (int d = 0; d < dest_channels; d++)
							{
								mval = _mm256_fmadd_ps(rep[d], _mm256_set1_ps(evecHInv.at<float>(src_channels * h + c, d) * evecVInv.at<float>(v, d)), mval);
							}

							const __m256 sub = _mm256_sub_ps(_mm256_loadu_ps(srcborder[c].ptr<float>(j + h, i + v)), mval);
							const __m256d sd0 = _mm256_cvtps_pd(_mm256_castps256_ps128(sub));
							const __m256d sd1 = _mm256_cvtps_pd(_mm256_castps256hi_ps128(sub));
							ssd0[t] = _mm256_fmadd_pd(sd0, sd0, ssd0[t]);
							ssd1[t] = _mm256_fmadd_pd(sd1, sd1, ssd1[t]);
						}
					}
				}
			}
		}
	}

	double mse = 0.0;
	for (int t = 0; t < thread_max; t++)
	{
		mse += _mm256_reduceadd_pd(ssd0[t]) + _mm256_reduceadd_pd(ssd1[t]);
	}
	mse = mse / area;
	return 10.0 * log10(255.0 * 255.0 / mse);
}

//see vector varsion for detailed implementation
double getPSNRReprojectDRIM2COL(const Mat& src, const int neighborhood_r, const int dest_channels, const int border, const DRIM2COLType type, const int bb, const bool isNormalizeDimension, const bool isParallel, const double const_sub)
{
	vector<Mat> temp;
	split(src, temp);
	return getPSNRReprojectDRIM2COL(temp, neighborhood_r, dest_channels, border, type, bb, isNormalizeDimension, isParallel, const_sub);
}

double getPSNRReprojectDRIM2COL(const vector<Mat>& vsrc, const int neighborhood_r, const int dest_channels, const int border, const DRIM2COLType type, const int bb, const bool isNormalizeDimension, const bool isParallel, const double const_sub)
{
	CV_Assert(vsrc[0].depth() == CV_8U || vsrc[0].depth() == CV_32F);

	const int width = vsrc[0].cols;
	const int height = vsrc[0].rows;
	const int ch = (int)vsrc.size();
	const int d = 2 * neighborhood_r + 1;
	const int patch_area = d * d;
	const int dim = ch * patch_area;
	const int channels = min(dest_channels, dim);
	const Size imsize = vsrc[0].size();
	const int imarea = imsize.area();
	const int num_points = cvRound((float)imarea * 0.1);

	double ret = 0.0;
	if (double((int)vsrc[0].size().area() * 255 * 255) > FLT_MAX)
	{
		cout << "overflow in float" << endl;
	}

	if ((int)type < (int)DRIM2COLType::OPENCV_PCA)
	{
		//cout << "Reproject for PSNR" << endl;
		Mat cov;
		CalcPatchCovarMatrix pcov;
		pcov.setBorder(border);
		pcov.setConstSub((int)const_sub);
		pcov.computeCov(vsrc, neighborhood_r, cov, (DRIM2COLType)type, 1, isParallel);

		Mat eval, evec;
		cv::eigen(cov, eval, evec);
		//util::eigen(cov, eval, evec);

		Mat U32F;
		evec(Rect(0, 0, evec.cols, channels)).convertTo(U32F, CV_32F);
		//print_mat(transmatInv * transmat);
		//print_matinfo_detail(transmatInv * transmat);
		ret = getPSNRReprojectDRIM2COLEigenVec(vsrc, U32F, neighborhood_r, border, bb, isNormalizeDimension);
	}
	else if (isSepCov(type, true))
	{
		Mat cov;
		CalcPatchCovarMatrix pcov;
		pcov.setBorder(border);
		pcov.setConstSub((int)const_sub);
		if (type == DRIM2COLType::MEAN_SUB_SEPSVD) pcov.computeCov(vsrc, neighborhood_r, cov, DRIM2COLType::MEAN_SUB_HALF_32F, 1, isParallel);
		if (type == DRIM2COLType::NO_SUB_SEPSVD) pcov.computeCov(vsrc, neighborhood_r, cov, DRIM2COLType::NO_SUB_HALF_32F, 1, isParallel);
		if (type == DRIM2COLType::CONST_SUB_SEPSVD) pcov.computeCov(vsrc, neighborhood_r, cov, DRIM2COLType::CONST_SUB_HALF_32F, 1, isParallel);
		Mat eval, evec, U, U32F, Uh, Uh32F, Uv, Uv32F;
		cv::eigen(cov, eval, evec);
		separateEvecSVD(evec, ch, U, Uh, Uv);

		U(Rect(0, 0, evec.cols, channels)).convertTo(U32F, CV_32F);
		Uh(Rect(0, 0, Uh.cols, channels)).convertTo(Uh32F, CV_32F);
		Uv(Rect(0, 0, Uv.cols, channels)).convertTo(Uv32F, CV_32F);

		ret = getPSNRReprojectDRIM2COLEigenVec(vsrc, U32F, neighborhood_r, border, bb, isNormalizeDimension);
		//ret = getPSNRReprojectDRIM2COLSepEigenVec(vsrc, Uh32F, Uv32F, neighborhood_r, border, bb, isNormalizeDimension);
	}
	else if (isSepCov(type, false))
	{
		vector<Mat> vcov;
		CalcPatchCovarMatrix pcov;
		pcov.setBorder(border);
		pcov.setConstSub((int)const_sub);
		pcov.computeSepCov(vsrc, neighborhood_r, vcov, type, 1, isParallel);

		Mat U, Uh, Uv;
		calcEvecfromSepCov(vcov[0], vcov[1], channels, U, Uh, Uv);
		//Mat transmat= cp::convert(U, CV_32F);
		Mat U32F = cp::convert(U, CV_32F);
		Mat Uh32F = cp::convert(Uh, CV_32F);
		Mat Uv32F = cp::convert(Uv, CV_32F);
		if (DRIM2COLType::MEAN_SUB_SEPCOVX == type ||
			DRIM2COLType::NO_SUB_SEPCOVX == type ||
			DRIM2COLType::CONST_SUB_SEPCOVX == type)
		{

			ret = getPSNRReprojectDRIM2COLEigenVec(vsrc, U32F, neighborhood_r, border, bb, isNormalizeDimension);
			//ret = getPSNRReprojectDRIM2COLSepEigenVec(vsrc, Uh32F, Uv32F, neighborhood_r, border, bb, isNormalizeDimension);
			//vector<Mat> vtemp;
			//ret = DRIM2COLTestCn(vsrc, transmat, Uh32F, Uv32F, vtemp, neighborhood_r, border);
		}
		else
		{
			vector<Mat> vsrct(vsrc.size());
			for (int c = 0; c < vsrct.size(); c++)
			{
				transpose(vsrc[c], vsrct[c]);
			}
			ret = getPSNRReprojectDRIM2COLSepEigenVec(vsrct, Uh32F, Uv32F, neighborhood_r, border, bb, isNormalizeDimension);
		}
		/*
		{
			//using full
			Mat transmatInv = transmat.t();
			ret = getPSNRReprojectDRIM2COLEigenVec(vsrc, transmat, transmatInv, neighborhood_r, border, bb, isNormalizeDimension);
		}*/
	}
	else
	{
		Mat highDimGuide(vsrc[0].size(), CV_MAKE_TYPE(CV_32F, dim));
		{
			//cp::Timer t("cvt HDI");
			IM2COL(vsrc, highDimGuide, neighborhood_r, border);
		}
		Mat x = highDimGuide.reshape(1, imsize.area());
		Mat cov, mean;
		cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
		//cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL |  cv::COVAR_ROWS);

		//print_mat(cov);
		Mat eval, evec;
		eigen(cov, eval, evec);

		Mat U32F;
		//print_matinfo(evec);
		evec(Rect(0, 0, evec.cols, channels)).convertTo(U32F, CV_32F);
		//transmat = Mat::eye(transmat.size(), CV_32F);
		ret = getPSNRReprojectDRIM2COLEigenVec(vsrc, U32F, neighborhood_r, border, bb, isNormalizeDimension);
	}
	return ret;
}
#pragma endregion

#pragma region patchPCADenoise
void patchPCAEigenVec(const vector<Mat>& vsrc, vector<Mat>& dest, const Mat& evec, const Mat& evecInv, const int r, const float th, const int border)
{
	const int src_channels = (int)vsrc.size();
	const int dest_channels = evec.rows;

	const int width = vsrc[0].cols;
	const int simd_end = get_simd_floor(width, 8);
	const __m256i mask = get_simd_residualmask_epi32(width);
	const bool isRem = (width == simd_end) ? false : true;
	const int height = vsrc[0].rows;

	const int D = 2 * r + 1;
	const int DD = D * D;
	const int dim = DD * src_channels;//D * D * src_channels()

	const __m256 mdiv = _mm256_set1_ps(1.f / DD);
	const __m256 mth = _mm256_set1_ps(th);
	AutoBuffer<const float*> eptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eptr[m] = evec.ptr<float>(m);
	}

	AutoBuffer<const float*> einvptr(dim);
	for (int d = 0; d < dim; d++)
	{
		einvptr[d] = evecInv.ptr<float>(d);
	}

	vector<Mat> srcborder(src_channels);
	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(vsrc[c], srcborder[c], r, r, r, r, border);
	}
	dest.resize(vsrc.size());
	vector<Mat> destborder(vsrc.size());
	for (int i = 0; i < vsrc.size(); i++)
	{
		destborder[i] = Mat::zeros(srcborder[0].size(), vsrc[0].depth());
	}

	AutoBuffer<__m256> patch(dim);
	AutoBuffer<__m256> proj(dest_channels);
	AutoBuffer<__m256> repatch(dim);

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < simd_end; i += 8)
		{
			for (int l = 0, idx = 0; l < D; l++)
			{
				for (int k = 0; k < D; k++)
				{
					for (int c = 0; c < src_channels; c++)
					{
						patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + l, i + k));
					}
				}
			}

			for (int c = 0; c < dest_channels; c++)
			{
				__m256 mval = _mm256_setzero_ps();
				for (int d = 0; d < dim; d++)
				{
					mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
				}

				__m256 mm = _mm256_cmp_ps(_mm256_abs_ps(mval), mth, _CMP_LE_OQ);
				proj[c] = _mm256_andnot_ps(mm, mval);
			}

			for (int l = 0, idx = 0; l < D; l++)
			{
				for (int k = 0; k < D; k++)
				{
					for (int c = 0; c < src_channels; c++)
					{
						const int d = D * src_channels * l + src_channels * k + c;
						__m256 mrep = _mm256_setzero_ps();
						for (int c = 0; c < dest_channels; c++)
						{
							mrep = _mm256_fmadd_ps(proj[c], _mm256_set1_ps(einvptr[d][c]), mrep);
						}
						_mm256_storeu_ps(destborder[c].ptr<float>(j + l, i + k), _mm256_fmadd_ps(mdiv, mrep, _mm256_loadu_ps(destborder[c].ptr<float>(j + l, i + k))));
					}
				}
			}
		}

		if (isRem)
		{
			for (int l = 0, idx = 0; l < D; l++)
			{
				for (int k = 0; k < D; k++)
				{
					for (int c = 0; c < src_channels; c++)
					{
						patch[idx++] = _mm256_maskload_ps(srcborder[c].ptr<float>(j + l, simd_end + k), mask);
					}
				}
			}

			for (int c = 0; c < dest_channels; c++)
			{
				__m256 mval = _mm256_setzero_ps();
				for (int d = 0; d < dim; d++)
				{
					mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
				}
				proj[c] = mval;
			}

			for (int l = 0, idx = 0; l < D; l++)
			{
				for (int k = 0; k < D; k++)
				{
					for (int c = 0; c < src_channels; c++)
					{
						const int d = D * src_channels * l + src_channels * k + c;
						__m256 mrep = _mm256_setzero_ps();
						for (int c = 0; c < dest_channels; c++)
						{
							mrep = _mm256_fmadd_ps(proj[c], _mm256_set1_ps(einvptr[d][c]), mrep);
						}
						_mm256_maskstore_ps(destborder[c].ptr<float>(j + l, simd_end + k), mask, _mm256_fmadd_ps(mdiv, mrep, _mm256_maskload_ps(destborder[c].ptr<float>(j + l, simd_end + k), mask)));
					}
				}
			}
		}
	}

	for (int c = 0; c < src_channels; c++)
	{
		destborder[c](Rect(r, r, vsrc[0].cols, vsrc[0].rows)).copyTo(dest[c]);
	}
}

void patchPCADenoise(const vector<Mat>& vsrc, vector<Mat>& dest, const int neighborhood_r, const int dest_channels, const float th, const int border, const int method, const bool isParallel)
{
	CV_Assert(vsrc[0].depth() == CV_8U || vsrc[0].depth() == CV_32F);

	const int width = vsrc[0].cols;
	const int height = vsrc[0].rows;
	const int ch = (int)vsrc.size();
	const int d = 2 * neighborhood_r + 1;
	const int patch_area = d * d;
	const int dim = ch * patch_area;
	const int channels = min(dest_channels, dim);
	const Size imsize = vsrc[0].size();
	const int imarea = imsize.area();
	const int num_points = cvRound((float)imarea * 0.1);

	if (double(vsrc[0].size().area() * 255 * 255) > FLT_MAX)
	{
		cout << "overflow in float" << endl;
	}

	if (method < (int)DRIM2COLType::OPENCV_PCA)
	{
		Mat cov, eval, evec;
		CalcPatchCovarMatrix pcov;
		pcov.setBorder(border);
		pcov.computeCov(vsrc, neighborhood_r, cov, (DRIM2COLType)method, 1, isParallel);

		cv::eigen(cov, eval, evec);

		Mat transmat;
		evec(Rect(0, 0, evec.cols, channels)).convertTo(transmat, CV_32F);

		Mat transmatInv = transmat.t();
		patchPCAEigenVec(vsrc, dest, transmat, transmatInv, neighborhood_r, th, border);
	}
	else
	{
		Mat highDimGuide(vsrc[0].size(), CV_MAKE_TYPE(CV_32F, dim));
		{
			//cp::Timer t("cvt HDI");
			IM2COL(vsrc, highDimGuide, neighborhood_r, border);
		}
		Mat x = highDimGuide.reshape(1, imsize.area());
		Mat cov, mean;
		cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
		//cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL |  cv::COVAR_ROWS);
		//print_mat(cov);
		Mat eval, evec;
		eigen(cov, eval, evec);

		Mat transmat;
		evec(Rect(0, 0, evec.cols, channels)).convertTo(transmat, CV_32F);
		Mat transmatInv = transmat.t();
		patchPCAEigenVec(vsrc, dest, transmat, transmatInv, neighborhood_r, th, border);
	}
}

void patchPCADenoise(const Mat& src, Mat& dest, const int neighborhood_r, const int dest_channels, const float th, const int border, const int method, const bool isParallel)
{
	vector<Mat> temp;
	vector<Mat> temp2;
	split(src, temp);
	patchPCADenoise(temp, temp2, neighborhood_r, dest_channels, th, border, method, isParallel);
	merge(temp2, dest);
}
#pragma endregion

#pragma region debug
void CalcPatchCovarMatrix::simd_32FSkipCn(const vector<Mat>& src, Mat& cov, const int skip)
{
	const int simd_step = 8;
	const int x_step = 8 * skip;

	vector<double> meanv(color_channels);
	vector<double> var(color_channels);
	for (int i = 0; i < color_channels; i++)
	{
		if (color_channels != 1) meanv[i] = cp::average(src[i]);
		else meanv[0] = 0.0;
	}

	const int DD = dim / color_channels;
	const int center = (DD / 2) * color_channels;

	AutoBuffer<int> scan(DD);
	getScanorder(scan, src[0].cols, 1);

	vector<double> sum(dim * dim);
	for (int i = 0; i < dim * dim; i++) sum[i] = 0.0;
	AutoBuffer<__m256> msum(dim * dim);
	AutoBuffer<__m256> msrc(dim);
	AutoBuffer<__m256> mvar(color_channels);
	for (int i = 0; i < dim * dim; i++) msum[i] = _mm256_setzero_ps();
	for (int i = 0; i < color_channels; i++) mvar[i] = _mm256_setzero_ps();

	const int simd_end_x = get_simd_floor(src[0].cols - 2 * patch_rad, simd_step) + patch_rad;
	const double normalSize = 1.0 / ((src[0].rows - 2 * patch_rad) * (simd_end_x - patch_rad));

	__m256i gatherstep = _mm256_setr_epi32(0, x_step, 2 * x_step, 3 * x_step, 4 * x_step, 5 * x_step, 6 * x_step, 7 * x_step);

	for (int j = patch_rad; j < src[0].rows - patch_rad; j += skip)
	{
		for (int i = patch_rad; i < simd_end_x; i += x_step)
		{
			AutoBuffer<const float*> sptr(color_channels);
			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] = src[c].ptr<float>(j, i);
			}

			for (int k = 0, idx = 0; k < DD; k++)
			{
				for (int c = 0; c < color_channels; c++)
				{
					msrc[idx++] = _mm256_i32gather_ps(sptr[c] + scan[k], gatherstep, 4);
				}
			}

			for (int c = 0; c < color_channels; c++)
			{
				mvar[c] = _mm256_fmadd_ps(msrc[c + center], msrc[c + center], mvar[c]);
			}

			for (int k = 0, idx = 0; k < dim; k++)
			{
				for (int l = k + 1; l < dim; l++)
				{
					msum[idx] = _mm256_fmadd_ps(msrc[k], msrc[l], msum[idx]);
					idx++;
				}
			}
		}
	}

	for (int k = 0, idx = 0; k < dim; k++)
	{
		for (int l = k + 1; l < dim; l++)
		{
			//sum[idx] = _mm256_reduceadd_ps(msum[idx]);
			sum[idx] = _mm256_reduceadd_pspd(msum[idx]);
			idx++;
		}
	}

	for (int c = 0; c < color_channels; c++)
	{
		//var[c] = _mm256_reduceadd_ps(mvar[c]);
		var[c] = _mm256_reduceadd_pspd(mvar[c]);
	}

	//setCoVar(meanv, var, cov, sum, normalSize);
	//setCovRepMeanOld(meanv, var, cov, sum, 1);
}

void computeAllRelativePosition(const int r, vector<Point>& RP_list)
{
	int range = 2 * r;
	int idx = 0;
	int RPnum = (range * 2 + 2) * range + 1;
	RP_list.resize(RPnum);
	for (int j = 0; j <= range; j++)
	{
		RP_list[idx].x = 0;
		RP_list[idx].y = j;
		idx++;
	}
	for (int i = 1; i <= range; i++)
	{
		for (int j = -range; j <= range; j++)
		{
			RP_list[idx].x = i;
			RP_list[idx].y = j;
			idx++;
		}
	}
	/*
	int D = 2 * r + 1;
	int idx = 0;
	int RPnum = 2 * D * D;
	RP_list.resize(RPnum);

	for (int j = 0; j < D; j++)
	{
		RP_list[idx].x = 0;
		RP_list[idx].y = j;
		idx++;
	}

	for (int i = 1; i < D; i++)
	{
		for (int j = -(D - 1); j < D; j++)
		{
			RP_list[idx].x = i;
			RP_list[idx].y = j;
			idx++;
		}
	}
	*/
}

void computeCovRepresentive(const Mat& src, const int patch_rad, Mat& covmat)
{
	const int D = patch_rad * 2 + 1;
	const int DD = D * D;
	const int channels = src.channels();

	vector<Point> RP_list;
	int i_min, i_max, j_min, j_max;
	int counter = 0;

	computeAllRelativePosition(patch_rad, RP_list);
	//cout << RP_list.size() << endl;
	if (channels == 1)
	{
		float repcovtemp;
		covmat.create(DD, DD, CV_32F);
		Mat srcborder;
		Mat subdata;
		copyMakeBorder(src, srcborder, patch_rad, patch_rad, patch_rad, patch_rad, 4);
		const double ave = cp::average(src);
		sub_const(srcborder, float(ave), subdata);
		//vector<vector<vector<vector<float>>>> repcov;
		//repcov.resize(D);
		//for (int i = 0; i < D; i++)
		//{
		//	repcov[i].resize(D);
		//	for (int j = 0; j < D; j++)
		//	{
		//		repcov[i][j].resize(D);
		//		for (int k = 0; k < D; k++)
		//		{
		//			repcov[i][j][k].resize(D);
		//		}
		//	}
		//}

		for (const Point& RP : RP_list)
		{
			// plus
			i_min = max(-RP.x, 0);
			i_max = min(D - 1 - RP.x, D - 1);
			j_min = max(-RP.y, 0);
			j_max = min(D - 1 - RP.y, D - 1);
			//print_debug4(i_min, i_max, j_min, j_max);
			// i
			for (int i = i_min; i <= i_max; i++)
			{
				// j
				for (int j = j_min; j <= j_max; j++)
				{
					if (counter == 0)
					{
						cout << RP.x << "," << RP.y << endl;
						//compute covariance
						repcovtemp = float(subdata(Rect(i, j, src.cols, src.rows)).dot(subdata(Rect(i + RP.x, j + RP.y, src.cols, src.rows))));
						repcovtemp /= float(src.cols * src.rows);
					}
					//RP���ɓ��������U��Ή����鋤���U�s��̗v�f�֑}��
					//repcov[i][j][i + RP[0]][j + RP[1]] = repcovtemp;
					covmat.at<float>(j * D + i, (j + RP.y) * D + i + RP.x) = repcovtemp;
					counter++;
				}
			}

			if (counter == DD)
			{
				counter = 0;
				continue;
			}

			// minus
			i_min = max(RP.x, 0);
			i_max = min(D - 1 + RP.x, D - 1);
			j_min = max(RP.y, 0);
			j_max = min(D - 1 + RP.y, D - 1);

			// i
			for (int i = i_min; i <= i_max; i++)
			{
				// j
				for (int j = j_min; j <= j_max; j++)
				{
					if (counter == 0)
					{
						repcovtemp = float(subdata(Rect(i, j, src.cols, src.rows)).dot(subdata(Rect(i - RP.x, j - RP.y, src.cols, src.rows))));
					}
					//repcov[i][j][i - RP[0]][j - RP[1]] = repcovtemp;
					covmat.at<float>(j * D + i, (j - RP.y) * D + i - RP.x) = repcovtemp;
					counter++;
				}
			}
			counter = 0;
		}

		//matricization

		//for (int i = 0; i < D; i++)
		//{
		//	for (int j = 0; j < D; j++)
		//	{
		//		for (int k = 0; k < D; k++)
		//		{
		//			for (int l = 0; l < D; l++)
		//			{
		//				covmat.at<float>(j * D + i, l * D + k) = repcov[i][j][k][l];
		//			}
		//		}
		//	}
		//}
	}
	else if (channels == 3)
	{
		cout << "color" << endl;

		Mat repcovtemp;
		repcovtemp.create(channels, channels, CV_32F);
		covmat.create(DD * channels, DD * channels, CV_32F);
		vector<Mat> srcborder(channels), vsrc(channels), data(channels);

		split(src, vsrc);
		cp::splitCopyMakeBorder(src, srcborder, patch_rad, patch_rad, patch_rad, patch_rad, 4);
		for (int c = 0; c < channels; c++)
		{
			double ave = cp::average(vsrc[c]);
			data[c].create(vsrc[c].cols, vsrc[c].rows, CV_32F);
			sub_const(srcborder[c], float(ave), data[c]);
		}

		for (const Point& RP : RP_list)
		{
			// plus
			//i+RP[0], j+RP[1]
			i_min = max(-RP.x, 0);
			i_max = min(D - 1 - RP.x, D - 1);
			j_min = max(-RP.y, 0);
			j_max = min(D - 1 - RP.y, D - 1);

			// i
			for (int i = i_min; i <= i_max; i++)
			{
				// j
				for (int j = j_min; j <= j_max; j++)
				{
					if (counter == 0)
					{
						cout << Point(i, j) << "," << RP << endl;
						for (int c1 = 0; c1 < channels; c1++)
						{
							for (int c2 = 0; c2 < channels; c2++)
							{
								//compute covariance
								repcovtemp.at<float>(c1, c2) = float(data[c1](Rect(i, j, src.cols, src.rows)).dot(data[c2](Rect(i + RP.x, j + RP.y, src.cols, src.rows))));
								repcovtemp.at<float>(c1, c2) /= float(src.cols * src.rows);
							}
						}
					}

					for (int c1 = 0; c1 < channels; c1++)
					{
						for (int c2 = 0; c2 < channels; c2++)
						{
							covmat.at<float>((j * D + i) * channels + c1, ((j + RP.y) * D + i + RP.x) * channels + c2) = repcovtemp.at<float>(c1, c2);
							//covmat.at<float>((channels* j + c1)* D + (channels * i + c2), ((channels* j + c1) + RP[1]) * D + (channels * i + c2) + RP[0]) = repcovtemp.at<float>(c1, c2);
						}
					}
					counter++;
				}
			}

			if (counter == DD)
			{
				counter = 0;
				continue;
			}

			// minus
			i_min = max(RP.x, 0);
			i_max = min(D - 1 + RP.x, D - 1);
			j_min = max(RP.y, 0);
			j_max = min(D - 1 + RP.y, D - 1);

			// i
			for (int i = i_min; i <= i_max; i++)
			{
				// j
				for (int j = j_min; j <= j_max; j++)
				{
					if (counter == 0)
					{
						cout << "op: " << Point(i, j) << "," << RP << endl;
						for (int c1 = 0; c1 < channels; c1++)
						{
							for (int c2 = 0; c2 < channels; c2++)
							{
								//compute covariance
								repcovtemp.at<float>(c1, c2) = float(data[c1](Rect(i, j, src.cols, src.rows)).dot(data[c2](Rect(i + RP.x, j + RP.y, src.cols, src.rows))));
								repcovtemp.at<float>(c1, c2) /= (float)(src.cols * src.rows);
							}
						}
					}
					for (int c1 = 0; c1 < channels; c1++)
					{
						for (int c2 = 0; c2 < channels; c2++)
						{
							covmat.at<float>((j * D + i) * channels + c1, ((j - RP.y) * D + i - RP.x) * channels + c2) = repcovtemp.at<float>(c1, c2);
							//covmat.at<float>((channels * j + c1) * D + (channels * i + c2), ((channels * j + c1) + RP[1]) * D + (channels * i + c2) + RP[0]) = repcovtemp.at<float>(c1, c2);
						}
					}
					counter++;
				}
			}
			counter = 0;
		}
	}

}

void colorPCA(const Mat& src, Mat& dst, Mat& transmat) {
	//Mat X, cov, evec, U, transmat, temp;
	//cov = X.t() * X;
	//eigen(cov, evec, U);
	//U.convertTo(transmat, CV_32F);
	//dst.create(Size(X.cols, X.rows), CV_64F);
	//for (int i = 0; i < X.cols; i++) {
	//	for (int j = 0; j < X.rows; j++) {
	//		temp = U(Rect(0, j, U.cols, 1)) * X(Rect(i, 0, 1, X.rows));
	//		dst.at<double>(i, j) = temp.at<float>(0, 0);
	//	}
	//}
	CalcPatchCovarMatrix pcov;
	Mat cov, evec, U;
	pcov.computeCov(src, 0, cov, DRIM2COLType::MEAN_SUB_HALF_64F, 1, 1);
	eigen(cov, evec, U);
	U.convertTo(transmat, CV_32F);
	DRIM2COLEigenVec(src, transmat, dst, 0, 1, 4, 1);
}

void computeCovRepresentiveTest(const int channels, const int patch_rad)
{
	RepresentiveCovarianceComputer rcc(channels, patch_rad);
	rcc.print();
	cout << rcc.getNumberOfDirections() << "/" << int(rcc.getNumberOfDirections() / rcc.getRatio()) << ": " << rcc.getRatio() << endl;
}
#pragma endregion
}