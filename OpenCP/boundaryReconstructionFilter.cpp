#include "boundaryReconstructionFilter.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	template <class srcType>
	struct BRFData
	{
		float distance;
		int count;
		srcType val;
		float sub;
	};

	template <class srcType>
	void boundaryReconstructionFilter_(InputArray src_, OutputArray dest_, Size ksize, const float frec, const float color, const float space)
	{
		dest_.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat dest = dest_.getMat();
		if (dest.empty())dest.create(src.size(), src.type());
		Mat sim;
		const int radiusw = ksize.width / 2;
		const int radiush = ksize.height / 2;;
		copyMakeBorder(src, sim, radiush, radiush, radiusw, radiusw, cv::BORDER_DEFAULT);

		vector<int> _space_ofs_before(ksize.area());
		vector<float> _space_dist(ksize.area());
		int* space_ofs_before = &_space_ofs_before[0];
		float* space_dist = &_space_dist[0];

		int maxk = 0;
		for (int i = -radiush; i <= radiush; i++)
		{
			for (int j = -radiusw; j <= radiusw; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radiusw)
					continue;
				//if( r==0) continue;
				space_dist[maxk] = (float)r;
				space_ofs_before[maxk++] = (int)(i*sim.cols + j);
			}
		}
		const int steps = sim.cols;
		const int step = src.size().width;


#pragma omp parallel for
		for (int i = 0; i < src.size().height; i++)
		{
			srcType* jptr = sim.ptr<srcType>(i + radiush); jptr += radiusw;
			srcType* dst = dest.ptr<srcType>(i);
			srcType* sr = (srcType*)src.ptr<srcType>(i);
			for (int j = 0; j < src.cols; j++)
			{
				srcType val0 = sr[j];

				vector<BRFData<srcType>> rdata(0);
				for (int k = 0; k < maxk; k++)
				{
					const srcType val = jptr[j + space_ofs_before[k]];

					bool flag = true;
					for (int n = 0; n < (int)rdata.size(); n++)
					{
						if (val == rdata[n].val)
						{
							flag = false;
							rdata[n].count++;
							rdata[n].distance += space_dist[k];
							break;
						}
					}
					if (flag)
					{
						BRFData<srcType> rd;
						rd.count = 1;
						rd.distance = space_dist[k];
						rd.val = val;

						rdata.push_back(rd);
					}
				}

				if (rdata.size() == 1)
				{
					dst[j] = rdata[0].val;
					continue;
				}

				float maxDis = 0.f;
				float minDis = FLT_MAX;
				int maxOcc = 0;
				int minOcc = maxk;

				srcType maxDiff = 0;
				srcType minDiff = 255;
				for (int n = 0; n < (int)rdata.size(); n++)
				{
					rdata[n].distance = (float)(rdata[n].distance / (double)rdata[n].count);
					rdata[n].sub = (float)abs(rdata[n].val - val0);
					maxDis = std::max<float>(rdata[n].distance, maxDis);
					minDis = std::min<float>(rdata[n].distance, minDis);
					maxOcc = std::max<int>(rdata[n].count, maxOcc);
					minOcc = std::min<int>(rdata[n].count, minOcc);
					maxDiff = std::max<srcType>((srcType)abs(rdata[n].sub), maxDiff);
					minDiff = std::min<srcType>((srcType)abs(rdata[n].sub), minDiff);
				}

				float divOcc = (maxOcc == minOcc) ? 0.00000001f : 1.0f / (float)(maxOcc - minOcc);
				float divDiff = (maxDiff == minDiff) ? 0.00000001f : 1.0f / (float)(maxDiff - minDiff);
				float divDis = (maxDis == minDis) ? 0.00000001f : 1.0f / (float)(maxDis - minDis);


				float maxE = 0.f;
				srcType mind = val0;

				for (int n = 0; n < (int)rdata.size(); n++)
				{
					float J = frec*(rdata[n].count - minOcc)*divOcc;
					J += color*((float)maxDiff - rdata[n].sub)*divDiff;
					J += space*(maxDis - rdata[n].distance)*divDis;

					if (J > maxE)
					{
						maxE = J;
						mind = rdata[n].val;
					}
				}
				dst[j] = mind;
			}
			jptr += steps;
			dst += step;
			sr += step;
		}
	}

	void boundaryReconstructionFilter(InputArray src, OutputArray dest, Size ksize, const float frec, const float color, const float space)
	{
		if (src.type() == CV_8U)
		{
			boundaryReconstructionFilter_<uchar>(src, dest, ksize, frec, color, space);
		}
		else if (src.type() == CV_16S)
		{
			boundaryReconstructionFilter_<short>(src, dest, ksize, frec, color, space);
		}
		else if (src.type() == CV_16U)
		{
			boundaryReconstructionFilter_<unsigned short>(src, dest, ksize, frec, color, space);
		}
		else if (src.type() == CV_32F)
		{
			boundaryReconstructionFilter_<float>(src, dest, ksize, frec, color, space);
		}
		else if (src.type() == CV_64F)
		{
			boundaryReconstructionFilter_<double>(src, dest, ksize, frec, color, space);
		}
	}
}