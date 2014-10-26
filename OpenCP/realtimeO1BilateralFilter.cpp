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
		color_weight[i] = max((float)std::exp(i*i*gauss_color_coeff), 0.00001f);//avoid 0 value
	}
}

void RealtimeO1BilateralFilter::allocateBin(int num_bin)
{
	float astep = (255.f/(float)(num_bin-1));

	for(int i=0;i<256;i++)
	{
		if(i==0)
		{
			idx[i]=0;
			a[i] = 1.f;	
		}
		else if(i==255)
		{
			idx[i]=(uchar)(num_bin-2);
			a[i] = 0.f;	
		}
		else
		{
			idx[i]=(uchar)((float)i/astep);
			int step = (int)(astep*(idx[i]+1)) - (int)astep*idx[i];
			a[i] = 1.f- (float)(i - (int)astep*idx[i])/(float)step;
			//cout<<i<<","<<(int)(255.0/(float)(num_bin-1)*(idx[i]))<<","<<idx[i]<<","<<num_bin-1<<","<<a[i]<<endl;
		}
	}
}


RealtimeO1BilateralFilter::RealtimeO1BilateralFilter()
{
	idx.resize(256);
	a.resize(256);
}

//#include "spectral_recursive_filter.hpp"

void RealtimeO1BilateralFilter::gauss(Mat& src, Mat& joint, Mat& dest, int r, float sigma_color, float sigma_space, int num_bin)
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
		uchar* s = src.ptr<uchar>(0);
		uchar* j = joint.ptr<uchar>(0);
		uchar* d = dest.ptr<uchar>(0);
#pragma omp parallel for schedule (dynamic)
		for(int b=0;b<num_bin;b++)
		{
			Size kernel = Size(2*r+1,2*r+1);
			float* su = sub_range[b].ptr<float>(0);//upper
			float* sd = normalize_sub_range[b].ptr<float>(0);//down

			uchar v = (uchar)(255.0/(float)(num_bin-1)*(b));

			if(b==0)v=0;
			if(b==num_bin-1)v=255;

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

			/*spectral_recursive_filter::gauss srf_gauss(sigma_space,sigma_space);
			srf_gauss.filter(sub_range[b],sub_range[b]);
			srf_gauss.filter(normalize_sub_range[b],normalize_sub_range[b]);*/
		/*				if(r<=0)
			{
				GaussianBlurIIR(sub_range[b],sub_range[b],sigma_space,4);
				GaussianBlurIIR(normalize_sub_range[b],normalize_sub_range[b],sigma_space,4);
			}
			else*/
			{
				GaussianBlur(sub_range[b],sub_range[b],kernel,sigma_space);
				GaussianBlur(normalize_sub_range[b],normalize_sub_range[b],kernel,sigma_space);
			}
			

			max(normalize_sub_range[b],0.00001f,normalize_sub_range[b]);//a
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
		uchar* s = src.ptr<uchar>(0);
		uchar* j = joint.ptr<uchar>(0);
		uchar* d = dest.ptr<uchar>(0);
#pragma omp parallel for schedule (dynamic)
		for(int b=0;b<num_bin;b++)
		{
			Size kernel = Size(2*r+1,2*r+1);
			float* su = sub_range[b].ptr<float>(0);//upper
			float* sd = normalize_sub_range[b].ptr<float>(0);//down

			uchar v = (uchar)(255.0/(float)(num_bin-1)*(b));

			if(b==0)v=0;
			if(b==num_bin-1)v=255;

			for(int i=0;i<src.size().area();i++)
			{
				const float coeff = color_weight[ abs(j[i] - v)];

				su[i] = coeff*s[i];
				sd[i] = coeff;
			}

			//spectral_recursive_filter::gauss srf_gauss(sigma_space,sigma_space);
			//srf_gauss.filter(sub_range[b],sub_range[b]);
			//srf_gauss.filter(normalize_sub_range[b],normalize_sub_range[b]);
		/*	if(r<=0)
			{
				GaussianBlurIIR(sub_range[b],sub_range[b],sigma_space,4);
				GaussianBlurIIR(normalize_sub_range[b],normalize_sub_range[b],sigma_space,4);
			}
			else*/
			{
				GaussianBlur(sub_range[b],sub_range[b],kernel,sigma_space);
				GaussianBlur(normalize_sub_range[b],normalize_sub_range[b],kernel,sigma_space);
			}
			max(normalize_sub_range[b],0.00001f,normalize_sub_range[b]);//a
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

void RealtimeO1BilateralFilter::gauss(Mat& src, Mat& dest, int r, float sigma_color, float sigma_space, int num_bin)
{
	Mat joint;
	if(src.channels()==1) joint = src;
	else cvtColor(src,joint,COLOR_BGR2GRAY);
	gauss(src,joint,dest,r,sigma_color,sigma_space,num_bin);
}

void RealtimeO1BilateralFilter::box(Mat& src, Mat& dest, int r, float sigma_color, int num_bin, int box_iter)
{
	Mat joint;
	if(src.channels()==1) joint = src;
	else cvtColor(src,joint,COLOR_BGR2GRAY);

	box(src,joint,dest, r,sigma_color, num_bin,box_iter);
}

void RealtimeO1BilateralFilter::box(Mat& src, Mat& joint, Mat& dest, int r, float sigma_color, int num_bin, int box_iter)
{
	CV_Assert(joint.channels()==1);

	box_iter = max(box_iter,1);
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
		uchar* s = src.ptr<uchar>(0);
		uchar* j = joint.ptr<uchar>(0);
		uchar* d = dest.ptr<uchar>(0);
#pragma omp parallel for schedule (dynamic)
		for(int b=0;b<num_bin;b++)
		{
			Size kernel = Size(2*r+1,2*r+1);
			float* su = sub_range[b].ptr<float>(0);//upper
			float* sd = normalize_sub_range[b].ptr<float>(0);//down

			uchar v = (uchar)(255.0/(float)(num_bin-1)*(b));

			if(b==0)v=0;
			if(b==num_bin-1)v=255;

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

			for(int i=0;i<box_iter;i++)
			{
				boxFilter(sub_range[b],sub_range[b],CV_32F,kernel,Point(-1,-1),false);
				boxFilter(normalize_sub_range[b],normalize_sub_range[b],CV_32F,kernel,Point(-1,-1),false);
			}

			max(normalize_sub_range[b],0.00001f,normalize_sub_range[b]);//a
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
		uchar* s = src.ptr<uchar>(0);
		uchar* j = joint.ptr<uchar>(0);
		uchar* d = dest.ptr<uchar>(0);
#pragma omp parallel for schedule (dynamic)
		for(int b=0;b<num_bin;b++)
		{
			Size kernel = Size(2*r+1,2*r+1);
			float* su = sub_range[b].ptr<float>(0);//upper
			float* sd = normalize_sub_range[b].ptr<float>(0);//down

			uchar v = (uchar)(255.0/(float)(num_bin-1)*(b));

			if(b==0)v=0;
			if(b==num_bin-1)v=255;

			for(int i=0;i<src.size().area();i++)
			{
				const float coeff = color_weight[ abs(j[i] - v)];

				su[i] = coeff*s[i];
				sd[i] = coeff;
			}

			for(int i=0;i<box_iter;i++)
			{
				boxFilter(sub_range[b],sub_range[b],CV_32F,kernel,Point(-1,-1),false);
				boxFilter(normalize_sub_range[b],normalize_sub_range[b],CV_32F,kernel,Point(-1,-1),false);
			}

			max(normalize_sub_range[b],0.00001f,normalize_sub_range[b]);//a
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