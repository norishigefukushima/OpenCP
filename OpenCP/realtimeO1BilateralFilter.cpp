#include "opencp.hpp"

void RealtimeO1BilateralFilter::createBin(Size imsize, int num_bin, int channles)
{
	normalize_sub_range.resize(num_bin);
	sub_range.resize(num_bin);

	for(int i=0;i<num_bin;i++)
	{
		sub_range[i].create(imsize,CV_MAKETYPE(CV_32F,channles));
		normalize_sub_range[i].create(imsize,CV_MAKETYPE(CV_32F,channles));
	}
}

void RealtimeO1BilateralFilter::setColorLUT(float sigma_color)
{
	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	for(int i=0;i<256;i++)
	{
		//color_weight[i] = max((float)std::exp(i*i*gauss_color_coeff), 0.00001f);//avoid 0 value
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
	}
}

int inline RealtimeO1BilateralFilter::bin2num(const int bin_index)
{
	int v = (int)(255.f/(float)(num_bin-1)*(bin_index)) ;

	if(bin_index==num_bin-1) v=254;
	return v;
}

void RealtimeO1BilateralFilter::allocateBin(int num_bin)
{	
	float astep = (255.f/(float)(num_bin-1));
	//std::cout<<"=================="<<std::endl;
	for(int i=0;i<256;i++)
	{
		if(i==255)
		{
			idx[i]=(uchar)(num_bin-2);
			a[i] = 0.f;	
		}
		else
		{
			idx[i]=(uchar)((float)i/astep);
			int step = (int)(astep*(idx[i]+1)) - (int)(astep*idx[i]);
			a[i] = 1.f-(i- bin2num(idx[i]))/(float)(bin2num(idx[i]+1)- bin2num(idx[i]));
			//std::cout<<i<<","<<(int)(astep*(idx[i]+1))<<","<<(int)(astep*idx[i])<<","<<num_bin-1<<","<<a[i]<<std::endl;
		}
	}
}

RealtimeO1BilateralFilter::RealtimeO1BilateralFilter()
{
	downsample_size = 1;
	idx.resize(256);
	a.resize(256);
}

void RealtimeO1BilateralFilter::filter(const Mat& src_, Mat& dest_)
{
	Mat src, dest;

	if(downsample_size == 1)
	{
		src = src_;
		dest=dest_;
	}
	else
	{
		resize(src_,src, Size(src_.cols/downsample_size, src_.rows/downsample_size),0.0,0.0,INTER_AREA);
		//resize(src_,src, Size(src_.cols/downsample_size, src_.rows/downsample_size),0.0,0.0,INTER_AREA);
	}

	if(filter_type==FIR_SEPARABLE)
	{
		Size kernel = Size(2*(radius/downsample_size)+1,2*(radius/downsample_size)+1);
		GaussianBlur(src,dest,kernel,sigma_space/downsample_size,0.0,BORDER_REPLICATE);
	}
	else if(filter_type==IIR_AM)
	{
		GaussianBlurIIR(src,dest,sigma_space/downsample_size,filter_iteration);
	}
	else if(filter_type==IIR_SR)
	{
		GaussianBlurSR(src,dest,sigma_space/downsample_size);
	}	
	else if(filter_type==FIR_BOX)
	{
		Size kernel = Size(2*(radius/downsample_size)+1,2*(radius/downsample_size)+1);
		for(int i=0;i<filter_iteration;i++)
		{
			boxFilter(src,dest,CV_32F,kernel,Point(-1,-1),false);
		}
	}

	if(downsample_size != 1)
	{
		//resize(dest, dest_,src_.size(),0.0,0.0,INTER_LINEAR);
		resize(dest, dest_,src_.size(),0.0,0.0,INTER_CUBIC);
	}
}

void RealtimeO1BilateralFilter::body(const Mat& src, const Mat& joint, Mat& dest)
{
	CV_Assert(joint.channels()==1);
	num_bin = max(num_bin,2);// for 0 and 255
	dest.create(src.size(),src.type());

	setColorLUT(sigma_color);
	if(sub_range.size()!=num_bin || sub_range[0].size().area()!=src.size().area())
	{
		createBin(src.size(),num_bin, src.channels());
		allocateBin(num_bin);
	}

	if(src.channels()==3)
	{
		const uchar* s = src.ptr<uchar>(0);
		const uchar* j = joint.ptr<uchar>(0);
		uchar* d = dest.ptr<uchar>(0);
#pragma omp parallel for schedule (dynamic)
		for(int b=0;b<num_bin;b++)
		{
			Size kernel = Size(2*radius+1,2*radius+1);
			float* su = sub_range[b].ptr<float>(0);//upper
			float* sd = normalize_sub_range[b].ptr<float>(0);//down

			uchar v = (uchar)bin2num(b);

			for(int i=0;i<src.size().area();i++)
			{
				const float coeff = color_weight[ abs(j[i] - v)];

				su[3*i+0] = coeff*s[3*i+0];
				su[3*i+1] = coeff*s[3*i+1];
				su[3*i+2] = coeff*s[3*i+2];
				sd[3*i+0] = coeff;
				sd[3*i+1] = coeff;
				sd[3*i+2] = coeff;
			}

			filter(sub_range[b],sub_range[b]);
			filter(normalize_sub_range[b],normalize_sub_range[b]);

			divide(sub_range[b],normalize_sub_range[b],sub_range[b]);
		}

		for(int i=0;i<src.size().area();i++)
		{
			int id = idx[j[i]];
			float ca = a[j[i]];

			d[3*i+0] = saturate_cast<uchar>(ca*sub_range[id].at<float>(3*i+0)+(1.0f-ca)*sub_range[id+1].at<float>(3*i+0));
			d[3*i+1] = saturate_cast<uchar>(ca*sub_range[id].at<float>(3*i+1)+(1.0f-ca)*sub_range[id+1].at<float>(3*i+1));
			d[3*i+2] = saturate_cast<uchar>(ca*sub_range[id].at<float>(3*i+2)+(1.0f-ca)*sub_range[id+1].at<float>(3*i+2));
		}
	}
	else if(src.channels()==1)
	{
		const uchar* s = src.ptr<uchar>(0);
		const uchar* j = joint.ptr<uchar>(0);
		uchar* d = dest.ptr<uchar>(0);

#pragma omp parallel for schedule (dynamic)
		for(int b=0;b<num_bin;b++)
		{
			Size kernel = Size(2*radius+1,2*radius+1);
			float* su = sub_range[b].ptr<float>(0);//upper
			float* sd = normalize_sub_range[b].ptr<float>(0);//down

			uchar v = (uchar)bin2num(b);

			//uchar v = (uchar)(255.f/(float)((num_bin-1)*(b))+0.5);
			//uchar v = (uchar)(255.f/(float)(num_bin-1)*(b));
			//uchar v = (uchar)(255.f/(float)((num_bin-1)*(b)));
			//uchar v = 0;
			
			for(int i=0;i<src.size().area();i++)
			{
				const float coeff = color_weight[ abs(j[i] - v)];

				su[i] = coeff*s[i];
				sd[i] = coeff;
			}
			
			filter(sub_range[b],sub_range[b]);
			filter(normalize_sub_range[b],normalize_sub_range[b]);
			
			divide(sub_range[b],normalize_sub_range[b],sub_range[b]);
		}

		for(int i=0;i<src.size().area();i++)
		{
			int id = idx[j[i]];
			float ca = a[j[i]];
			d[i] = saturate_cast<uchar>(ca*sub_range[id].at<float>(i)+(1.0f-ca)*sub_range[id+1].at<float>(i));
		}
	}
}

void RealtimeO1BilateralFilter::gauss(Mat& src, Mat& joint, Mat& dest, int r_, float sigma_color_, float sigma_space_, int num_bin_)
{
	radius=r_;
	sigma_color=sigma_color_;
	sigma_space=sigma_space_;
	num_bin=num_bin_;
	filter_type = FIR_SEPARABLE;

	body(src, joint, dest);
}

void RealtimeO1BilateralFilter::gauss(Mat& src, Mat& dest, int r, float sigma_color, float sigma_space, int num_bin)
{
	Mat joint;
	if(src.channels()==1) joint = src;
	else cvtColor(src,joint,COLOR_BGR2GRAY);
	gauss(src,joint,dest,r,sigma_color,sigma_space,num_bin);
}

void RealtimeO1BilateralFilter::gauss_iir(Mat& src, Mat& joint, Mat& dest, float sigma_color_, float sigma_space_, int num_bin_, int iteration)
{
	filter_iteration=iteration;
	sigma_color=sigma_color_;
	sigma_space=sigma_space_;
	num_bin=num_bin_;
	filter_type = IIR_AM;

	body(src, joint, dest);
}

void RealtimeO1BilateralFilter::gauss_iir(Mat& src, Mat& dest, float sigma_color, float sigma_space, int num_bin, int iter)
{
	Mat joint;
	if(src.channels()==1) joint = src;
	else cvtColor(src,joint,COLOR_BGR2GRAY);
	gauss_iir(src,joint,dest,sigma_color,sigma_space,num_bin,iter);
}

void RealtimeO1BilateralFilter::gauss_sr(Mat& src, Mat& joint, Mat& dest, float sigma_color_, float sigma_space_, int num_bin_)
{
	sigma_color=sigma_color_;
	sigma_space=sigma_space_;
	num_bin=num_bin_;
	filter_type = IIR_SR;

	body(src, joint, dest);
}

void RealtimeO1BilateralFilter::gauss_sr(Mat& src, Mat& dest, float sigma_color, float sigma_space, int num_bin)
{
	Mat joint;
	if(src.channels()==1) joint = src;
	else cvtColor(src,joint,COLOR_BGR2GRAY);
	gauss_sr(src,joint,dest,sigma_color,sigma_space,num_bin);
}

void RealtimeO1BilateralFilter::box(Mat& src, Mat& joint, Mat& dest, int r, float sigma_color_, int num_bin_, int iteration)
{
	filter_iteration=iteration;
	sigma_color=sigma_color_;
	num_bin=num_bin_;
	filter_type = FIR_BOX;

	body(src, joint, dest);
}

void RealtimeO1BilateralFilter::box(Mat& src, Mat& dest, int r, float sigma_color, int num_bin, int box_iter)
{
	Mat joint;
	if(src.channels()==1) joint = src;
	else cvtColor(src,joint,COLOR_BGR2GRAY);

	box(src,joint,dest, r,sigma_color, num_bin,box_iter);
}