#include "IM2COL.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	struct Neighborhood32F
	{
	public:
		float distance;
		int ofs;
		int x;
		int y;
		Neighborhood32F(const float distance, const int ofs)
		{
			this->distance = distance;
			this->ofs = ofs;
		}

		Neighborhood32F(const float distance, const int ofs, int x, int y)
		{
			this->distance = distance;
			this->ofs = ofs;

			this->x = x;
			this->y = y;
		}
	};

	bool cmpNeighborhood32F(const Neighborhood32F& a, const Neighborhood32F& b)
	{
		return a.distance < b.distance;
	}

	void IM2COL(const Mat& src, Mat& dst, const int neighborhood_r, const int border)
	{
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
		CV_Assert(!src.empty());

		const int width = src.cols;
		const int height = src.rows;
		const int channels = src.channels();
		const int d = 2 * neighborhood_r + 1;
		const int patch_area = d * d;
		const int dest_dim = channels * patch_area;

		Mat srcborder; copyMakeBorder(src, srcborder, neighborhood_r, neighborhood_r, neighborhood_r, neighborhood_r, border);
		dst.create(src.size(), CV_MAKE_TYPE(CV_32F, dest_dim));

		AutoBuffer<int> scan(dest_dim);
		vector<Neighborhood32F> data;
		{
			for (int j = 0; j < d; j++)
			{
				for (int i = 0; i < d; i++)
				{
					for (int k = 0; k < channels; k++)
					{
						float distance = float((j - neighborhood_r) * (j - neighborhood_r) + (i - neighborhood_r) * (i - neighborhood_r));
						if (channels == 3)
						{
							if (k == 0) distance += 0.0002f;
							if (k == 2) distance += 0.0001f;
						}
						int offset = channels * srcborder.cols * j + channels * i + k;
						//data.push_back(Neighborhood32F(distance, offset));
						data.push_back(Neighborhood32F(distance, offset, i - 1, j - 1));
					}
				}
			}
		}
		//sort(data.begin(), data.end(), cmpNeighborhood32F);
		for (int i = 0; i < dest_dim; i++)
		{
			scan[i] = data[i].ofs;
			//cout <<i<<": "<< scan[i] <<"| "<< data[i].x<<","<< data[i].y << endl;
		}
		//getchar();

		if (src.depth() == CV_8U)
		{
			//CalcTime t("pca bf");
			int count = 0;
#pragma omp parallel for
			for (int y = 0; y < height; y++)
			{
				uchar* dp = dst.ptr<uchar>(y);

				for (int x = 0; x < width; x++)
				{
					float* ip = dst.ptr<float>(y, x);

					int count = 0;
					for (int j = 0; j < d; j++)
					{
						uchar* sp = srcborder.ptr<uchar>(y + j);

						for (int i = 0; i < d; i++)
						{
							for (int k = 0; k < channels; k++)
							{
								ip[count++] = (float)sp[channels * (x + i) + k];
							}
						}
					}
				}
			}
		}
		else if (src.depth() == CV_32F)
		{
			//CalcTime t("pca bf");

#pragma omp parallel for
			for (int y = 0; y < height; y++)
			{
				float* dp = dst.ptr<float>(y);

				for (int x = 0; x < width; x++)
				{
					float* sptr = srcborder.ptr<float>(y, x);
					float* ip = dst.ptr<float>(y, x);
					for (int i = 0; i < dest_dim; i++)
					{
						ip[i] = sptr[scan[i]];
					}
				}
			}
		}
	}

	void IM2COL(const vector<Mat>& src, Mat& dst, const int neighborhood_r, const int border)
	{
		CV_Assert(src[0].depth() == CV_8U || src[0].depth() == CV_32F);

		const int width = src[0].cols;
		const int height = src[0].rows;
		const int channels = (int)src.size();
		const int D = 2 * neighborhood_r + 1;
		const int DD = D * D;
		const int dest_dim = channels * DD;

		vector<Mat> srcborder(src.size());
		for (int c = 0; c < src.size(); c++)
		{
			copyMakeBorder(src[c], srcborder[c], neighborhood_r, neighborhood_r, neighborhood_r, neighborhood_r, border);
		}
		dst.create(src[0].size(), CV_MAKE_TYPE(CV_32F, dest_dim));

		AutoBuffer<int> scan(dest_dim);
		vector<Neighborhood32F> data;
		{
			for (int j = 0; j < D; j++)
			{
				for (int i = 0; i < D; i++)
				{
					float distance = float((j - neighborhood_r) * (j - neighborhood_r) + (i - neighborhood_r) * (i - neighborhood_r));
					int offset = srcborder[0].cols * j + i;
					//data.push_back(Neighborhood32F(distance, offset));
					data.push_back(Neighborhood32F(distance, offset, i - 1, j - 1));
				}
			}
		}

		//sort(data.begin(), data.end(), cmpNeighborhood32F);
		for (int i = 0; i < dest_dim; i++)
		{
			scan[i] = data[i].ofs;
			//cout <<i<<": "<< scan[i] <<"| "<< data[i].x<<","<< data[i].y << endl;
		}
		//getchar();

		if (src[0].depth() == CV_8U)
		{
#pragma omp parallel for schedule (dynamic)
			for (int y = 0; y < height; y++)
			{
				AutoBuffer<uchar*> sptr(channels);
				uchar* dp = dst.ptr<uchar>(y);

				for (int x = 0; x < width; x++)
				{
					uchar* ip = dst.ptr<uchar>(y, x);
					for (int c = 0; c < channels; c++)
					{
						sptr[c] = srcborder[c].ptr<uchar>(y, x);
					}

					for (int i = 0, idx = 0; i < DD; i++)
					{
						for (int c = 0; c < channels; c++)
						{
							ip[idx++] = sptr[c][scan[i]];
						}
					}
				}
			}
		}
		else if (src[0].depth() == CV_32F)
		{
#pragma omp parallel for schedule (dynamic)
			for (int y = 0; y < height; y++)
			{
				AutoBuffer<float*> sptr(channels);
				float* dp = dst.ptr<float>(y);

				for (int x = 0; x < width; x++)
				{
					float* ip = dst.ptr<float>(y, x);
					for (int c = 0; c < channels; c++)
					{
						sptr[c] = srcborder[c].ptr<float>(y, x);
					}

					for (int i = 0, idx = 0; i < DD; i++)
					{
						for (int c = 0; c < channels; c++)
						{
							ip[idx++] = sptr[c][scan[i]];
						}
					}
				}
			}
		}
	}
}