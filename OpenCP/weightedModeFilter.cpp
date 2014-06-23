#include "opencp.hpp"
#include <opencv2/core/internal.hpp>

#define sqr(a) ((a)*(a))

class Histogram
{
public:

	enum
	{
		L0_NORM = 0,
		L1_NORM,
		L2_NORM,
		EXP
	};

	enum
	{
		NO_WEIGHT = 0,
		GAUSSIAN,
		BILATERAL,
		NO_WEIGHT_MEDIAN,
		GAUSSIAN_MEDIAN,
		BILATERAL_MEDIAN
	};

	int truncate;
	__declspec(align(16)) float hist[256];

	Histogram(int truncate_val, int mode_);

	void clear();
	void add(float addval, int bin, int metric=0);
	enum
	{
		MAX=0,
		MEDIAN
	};
	int mode;
	int returnVal();
	int returnMax();
	int returnMedian();
};

int Histogram::returnVal()
{
	if(mode==0)
	return returnMax();
	else
	return returnMedian();

}
Histogram::Histogram(int truncate_val, int mode_ = Histogram::MAX)
{
	truncate = truncate_val;
	mode=mode;
	clear();
}
void Histogram::clear()
{
	int ssecount = 256/4;
	float* h = hist;
	for(int i=0;i<ssecount;i++) 
	{
		_mm_store_ps(h,_mm_setzero_ps());
		h+=4;
	}
}
void Histogram::add(float addval, int bin, int metric)
{
	hist[bin] += addval;

	if(metric != L0_NORM)
	{
		float val;
		if(metric == L1_NORM)
		{for(int i=1; i<truncate; i++)
		{
			val = addval*((float)(truncate-i)/(float)truncate);
			if(bin+i<=255)	hist[bin+i] += val;
			if(bin-i>=0)	hist[bin-i] += val;
		}
		}
		else if(metric == L2_NORM)
		{
			float div = 1.f/(float)(sqr(truncate));
			for(int i=1; i<truncate; i++)
			{
				val = addval*((float)sqr((truncate-i))*div);
				if(bin+i<=255)	hist[bin+i] += val;
				if(bin-i>=0)	hist[bin-i] += val;
			}
		}
		else if(metric == EXP)
		{
			for(int i=1; i<truncate; i++)
			{
				val = addval*(1-exp(-((float)sqr(i)/(float)(2*sqr(truncate)))));
				if(bin+i<=255)	hist[bin+i] += val;
				if(bin-i>=0)	hist[bin-i] += val;
			}
		}
	}
}
int Histogram::returnMax()
{
	float maxv = 0.f;
	int maxbin;

	for(int i=0; i<256; i++)
	{
		if(hist[i] > maxv)
		{
			maxv = hist[i];
			maxbin = i;
		}
	}
	return maxbin;
}
int Histogram::returnMedian()
{
	float maxval=0.f;
	for(int i=0; i<256; i++)
	{
		maxval+=hist[i];
	}
	const float half_max = maxval*0.5f;

	int maxbin;
	maxval=0.f;
	for(int i=0; i<256; i++)
	{
		maxval+=hist[i];
		if(maxval > half_max)
		{
			maxbin = i;
			break;
		}
	}
	return maxbin;
}

void bgrSplit(Mat& src, Mat& b, Mat& g, Mat& r)
{
	vector<Mat> div;

	split(src, div);
	vector<Mat>::const_iterator it = div.begin();
	Mat btemp, gtemp, rtemp;
	btemp = *it++;
	gtemp = *it++;
	rtemp = *it;
	b = btemp.clone();
	g = gtemp.clone();
	r = rtemp.clone();
}
void bgrMerge(Mat& dst, Mat& b, Mat& g, Mat& r, int rtype)
{
	vector<Mat> merge;

	merge.push_back(b);
	merge.push_back(g);
	merge.push_back(r);
	Mat mixed(b.size(), rtype);
	int fromTo[] = {0,0, 1,1, 2,2};

	vector<Mat> mix;
	mix.push_back(mixed);

	mixChannels(merge, mix, fromTo, 3);

	dst = mixed.clone();
}

void weightedHistogramFilter(Mat& src, Mat& guide, Mat& dst, int r, int truncate, double sig_s, double sig_c, int metric, int method)
{
	if(dst.empty()) dst.create(src.size(), src.type());

	int width = src.cols;
	int height = src.rows;

	Mat src2; copyMakeBorder(src, src2, r, r, r, r, 1);

	int mode = Histogram::MAX;
	if(method>=Histogram::NO_WEIGHT_MEDIAN) 
	{
		mode = Histogram::MEDIAN;
		method -=Histogram::NO_WEIGHT_MEDIAN;
	}

	if(guide.channels()==3)
	{
		Mat B, G, R;
		bgrSplit(guide, B, G, R);

		Mat guideB; copyMakeBorder(B, guideB, r, r, r, r, 1);
		Mat guideG; copyMakeBorder(G, guideG, r, r, r, r, 1);
		Mat guideR; copyMakeBorder(R, guideR, r, r, r, r, 1);

		float* lutc = new float[768];
		float* luts = new float[(2*r+1)*(2*r+1)];


		for(int j=0,idx=0; j<2*r+1; j++)
		{
			for(int i=0; i<2*r+1; i++,idx++)
			{
				luts[idx] = (float)(exp(-(sqr(i-r)+sqr(j-r))/(2*sqr(sig_s))));
			}
		}
		for(int i=0;i<256*3;i++)
		{
			lutc[i]=(float)(exp(-sqr(i)/(2*sqr(sig_c))));
		}

		if(method == Histogram::BILATERAL)
		{
#pragma omp parallel for
			for(int y=0; y<height; y++)
			{
				Histogram h(truncate,mode);
				for(int x=0; x<width; x++)
				{
					h.clear();
					const uchar bb = B.at<uchar>(y,x);
					const uchar gg = G.at<uchar>(y,x);
					const uchar rr = R.at<uchar>(y,x);

					for(int j=0,idx=0; j<2*r+1; j++)
					{
						uchar* sp = src2.ptr<uchar>(y+j); sp+=x;
						uchar* bp = guideB.ptr<uchar>(y+j); bp+=x;
						uchar* gp = guideG.ptr<uchar>(y+j); gp+=x;
						uchar* rp = guideR.ptr<uchar>(y+j); rp+=x;

						for(int i=0; i<2*r+1; i++,idx++)
						{
							float addval = luts[idx];

							int diff = abs(bb-*bp)+abs(gg-*gp)+abs(rr-*rp);
							addval*= lutc[diff];	
							h.add(addval, *sp, metric);

							sp++;
							bp++;
							gp++;
							rp++;
						}
					}
					dst.at<uchar>(y,x) = h.returnVal();
				}
			}
		}
		if(method == Histogram::GAUSSIAN)
		{
#pragma omp parallel for
			for(int y=0; y<height; y++)
			{
				Histogram h(truncate,mode);
				for(int x=0; x<width; x++)
				{
					h.clear();
					for(int j=0,idx=0; j<2*r+1; j++)
					{
						uchar* sp = src2.ptr<uchar>(y+j); sp+=x;
						for(int i=0; i<2*r+1; i++,idx++)
						{
							float addval = luts[idx];
							h.add(addval, *sp, metric);
							sp++;
						}
					}
					dst.at<uchar>(y,x) = h.returnVal();
				}
			}
		}
		else if(method == Histogram::NO_WEIGHT)
		{
#pragma omp parallel for
			for(int y=0; y<height; y++)
			{
				for(int x=0; x<width; x++)
				{
					Histogram h(truncate,mode);

					for(int j=0,idx=0; j<2*r+1; j++)
					{
						for(int i=0; i<2*r+1; i++,idx++)
						{
							float addval = 1.f;
							h.add(addval, src2.at<uchar>(y+j,x+i), metric);
						}
					}
					dst.at<uchar>(y,x) = h.returnVal();
				}
			}
		}
		delete[] lutc;
		delete[] luts;
	}
	else if(guide.channels()==1)
	{
		Mat G; copyMakeBorder(guide, G, r, r, r, r, 1);

		float* lutc = new float[256];
		float* luts = new float[(2*r+1)*(2*r+1)];


		for(int j=0,idx=0; j<2*r+1; j++)
		{
			for(int i=0; i<2*r+1; i++,idx++)
			{
				luts[idx] = (float)(exp(-(sqr(i-r)+sqr(j-r))/(2*sqr(sig_s))));
			}
		}
		for(int i=0;i<256;i++)
		{
			lutc[i]=(float)(exp(-sqr(i)/(2.0*sqr(sig_c))));
		}

		if(method == Histogram::BILATERAL)
		{
#pragma omp parallel for
			for(int y=0; y<height; y++)
			{
				Histogram h(truncate,mode);
				for(int x=0; x<width; x++)
				{
					h.clear();
					const uchar gg = guide.at<uchar>(y,x);
					for(int j=0,idx=0; j<2*r+1; j++)
					{
						uchar* sp = src2.ptr<uchar>(y+j); sp+=x;
						uchar* gp = G.ptr<uchar>(y+j); gp+=x;
						for(int i=0; i<2*r+1; i++,idx++)
						{
							float addval = luts[idx];
							int diff = abs(gg-*gp);
							addval*= lutc[diff];	
							h.add(addval, *sp, metric);
							sp++;
							gp++;
						}
					}
					dst.at<uchar>(y,x) = h.returnVal();
				}
			}
		}
		if(method == Histogram::GAUSSIAN)
		{
#pragma omp parallel for
			for(int y=0; y<height; y++)
			{
				Histogram h(truncate,mode);
				for(int x=0; x<width; x++)
				{
					h.clear();
					for(int j=0,idx=0; j<2*r+1; j++)
					{
						uchar* sp = src2.ptr<uchar>(y+j); sp+=x;
						for(int i=0; i<2*r+1; i++,idx++)
						{
							float addval = luts[idx];
							h.add(addval, *sp, metric);
							sp++;
						}
					}
					dst.at<uchar>(y,x) = h.returnVal();
				}
			}
		}
		else if(method == Histogram::NO_WEIGHT)
		{
#pragma omp parallel for
			for(int y=0; y<height; y++)
			{
				for(int x=0; x<width; x++)
				{
					Histogram h(truncate,mode);

					for(int j=0,idx=0; j<2*r+1; j++)
					{
						for(int i=0; i<2*r+1; i++,idx++)
						{
							float addval = 1.f;
							h.add(addval, src2.at<uchar>(y+j,x+i), metric);
						}
					}
					dst.at<uchar>(y,x) = h.returnVal();
				}
			}
		}
		delete[] lutc;
		delete[] luts;
	}
}

void weightedModeFilter(Mat& src, Mat& guide, Mat& dst, int r, int truncate, double sig_s, double sig_c, int metric, int method)
{
	weightedHistogramFilter(src,guide,dst,r,truncate, sig_s, sig_c, metric, method);
}

void weightedMedianFilter(Mat& src, Mat& guide, Mat& dst, int r, int truncate, double sig_s, double sig_c, int metric, int method)
{
	weightedHistogramFilter(src,guide,dst,r,truncate, sig_s, sig_c, metric, method+Histogram::NO_WEIGHT_MEDIAN);
}
