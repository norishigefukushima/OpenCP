#include "opencp.hpp"
#include <opencv2/core/internal.hpp>
void weightedGaussianFilter(Mat& src, Mat& weight, Mat& dest,Size ksize, float sigma, int border_type)
{
	Mat temp;
	if(src.channels()==3)cvtColor(weight,temp,CV_GRAY2BGR);
	else temp=weight;
	Mat weightf;temp.convertTo(weightf,CV_32F);

	Mat sw;
	Mat wsrc;src.convertTo(wsrc,CV_MAKETYPE(CV_32F,src.channels()));
	Mat destf;
	//boxFilter(weightf,sw,CV_32F,ksize,pt,true,border_type);
	GaussianBlur(weightf,sw,ksize,sigma,0.0,border_type);
	cv::multiply(wsrc,weightf,wsrc);//sf*sf
	GaussianBlur(wsrc,destf,ksize,sigma,0.0,border_type);
	cv::divide(destf,sw,destf);
	destf.convertTo(dest,CV_MAKETYPE(CV_8U,src.channels()));
}

void GaussianFilter_8u_ignore_boudary(const Mat& src,  Mat& dest, int r, float sigma)
{
	Mat sc;
	if(dest.empty())dest.create(src.size(),src.type());
	if(src.data==dest.data)src.copyTo(sc);
	else sc = src;
	
	float* lut = (float*)_mm_malloc(sizeof(float)*(2*r+1)*(2*r+1),16);
	
	double gauss_space_coeff = -0.5/(sigma*sigma);

	int maxk=0;

	for(int l=-r;l<=r;l++)
	{
		for(int k=-r;k<=r;k++)
		{
			double cr = std::sqrt((double)k*k + (double)l*l);
			if( cr > r )
				continue;
			lut[maxk++] = (float)std::exp(r*r*gauss_space_coeff);

		}
	}
	if(src.channels()==3)
	{
		for(int j=r;j<src.rows-r;j++)
		{
			uchar* d = dest.ptr<uchar>(j);
			for(int i=r;i<src.cols-r;i++)
			{
				//for(int k=0;k<maxk;k++)
				float sum_b = 0.f;
				float sum_g = 0.f;
				float sum_r = 0.f;
				float weight = 0.f;
				float nor = 0.f;
				int m=0;
				for(int l=-r;l<=r;l++)
				{
					uchar* s = sc.ptr<uchar>(j+l)+3*i;
					for(int k=-r;k<=r;k++)
					{
						float w = lut[m++];
						weight+=w;
						sum_b += w*s[3*k+0];
						sum_g += w*s[3*k+1];
						sum_r += w*s[3*k+2];
					}
				}
				float div = 1.f/weight;
				d[3*i  ]=saturate_cast<uchar>(sum_b*div);
				d[3*i+1]=saturate_cast<uchar>(sum_g*div);
				d[3*i+2]=saturate_cast<uchar>(sum_r*div);
			}
		}
	}
	else if(src.channels()==1)
	{
		for(int j=r;j<src.rows-r;j++)
		{
			uchar* d = dest.ptr<uchar>(j);
			for(int i=r;i<src.cols-r;i++)
			{
				//for(int k=0;k<maxk;k++)
				float sum = 0.f;
				float weight = 0.f;
				float nor = 0.f;
				int m=0;
				for(int l=-r;l<=r;l++)
				{
					uchar* s = sc.ptr<uchar>(j+l)+i;
					for(int k=-r;k<=r;k++)
					{
						float w = lut[m++];
						weight+=w;
						sum += w*s[k];
					}
				}
				float div = 1.f/weight;
				d[i  ]=saturate_cast<uchar>(sum*div);
			}
		}
	}

	_mm_free(lut);
}

void GaussianFilter_8u_ignore_boudary(const Mat& src, Mat& dest, int r, float sigma, Mat& mask)
{
	if(dest.empty())src.copyTo(dest);
	if(src.data!=dest.data)src.copyTo(dest);
	Mat sc=src;
	//if(src.data==dest.data) sc= src.clone();
	//else sc = src;
	
	float* lut = (float*)_mm_malloc(sizeof(float)*(2*r+1)*(2*r+1),16);
	
	double gauss_space_coeff = -0.5/(sigma*sigma);

	int maxk=0;
	for(int l=-r;l<=r;l++)
	{
		for(int k=-r;k<=r;k++)
		{
			double cr = std::sqrt((double)k*k + (double)l*l);
			//if( cr > r )continue;
			lut[maxk++] = (float)std::exp(r*r*gauss_space_coeff);

		}
	}
	if(src.channels()==3)
	{
		for(int j=r;j<src.rows-r;j++)
		{
			uchar* d = dest.ptr<uchar>(j);
			uchar* msk = mask.ptr<uchar>(j);
			for(int i=r;i<src.cols-r;i++)
			{
				if(msk[i]!=0)
				{
					float sum_b = 0.f;
					float sum_g = 0.f;
					float sum_r = 0.f;
					float weight = 0.f;
					float nor = 0.f;
					int m=0;
					for(int l=-r;l<=r;l++)
					{
						uchar* s = sc.ptr<uchar>(j+l)+3*i;
						for(int k=-r;k<=r;k++)
						{
							float w = lut[m++];
							weight+=w;
							sum_b += w*s[3*k+0];
							sum_g += w*s[3*k+1];
							sum_r += w*s[3*k+2];
						}
					}
					float div = 1.f/weight;
					d[3*i  ]=saturate_cast<uchar>(sum_b*div);
					d[3*i+1]=saturate_cast<uchar>(sum_g*div);
					d[3*i+2]=saturate_cast<uchar>(sum_r*div);
				}
			}
		}
	}
	else if(src.channels()==1)
	{
		for(int j=r;j<src.rows-r;j++)
		{
			uchar* d = dest.ptr<uchar>(j);
			uchar* msk = mask.ptr<uchar>(j);
			for(int i=r;i<src.cols-r;i++)
			{
				if(msk[i]!=0)
				{
				//for(int k=0;k<maxk;k++)
				float sum = 0.f;
				float weight = 0.f;
				float nor = 0.f;
				int m=0;
				for(int l=-r;l<=r;l++)
				{
					uchar* s = sc.ptr<uchar>(j+l)+i;
					for(int k=-r;k<=r;k++)
					{
						float w = lut[m++];
						weight+=w;
						sum += w*s[k];
					}
				}
				float div = 1.f/weight;
				d[i  ]=saturate_cast<uchar>(sum*div);
				}
			}
		}
	}

	_mm_free(lut);
}


void GaussianFilter(const Mat src, Mat& dest, int r, float sigma, int method, Mat& mask)
{
	if(mask.empty())
	{
		GaussianFilter_8u_ignore_boudary(src,dest,r,sigma);
	}
	else
	{
		GaussianFilter_8u_ignore_boudary(src,dest,r,sigma,mask);
	}
}