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

// Alvarez-Mazorra
//L. Alvarez, L. Mazorra, "Signal and image restoration using shock filters and anisotropic diffusion," SIAM Journal on Numerical Analysis, vol. 31, no. 2, pp. 590?605, 1994.
void gaussian_am(float *image, const int width, const int height, const float sigma, const int iteration)
{
	const int num_pixels = width*height;

	float nu, boundary_scale, post_scale;
	float *ptr;
	int i;
	int step;

	if(sigma <= 0 || iteration < 0)
		return;

	double lambda = (sigma*sigma)/(2.0*iteration);
	double dnu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
	nu = (float)dnu;
	boundary_scale = (float)(1.0/(1.0 - dnu));
	post_scale = (float)(pow(dnu/lambda,2*iteration));

	int x = 0;
	const __m128 mnu = _mm_set1_ps(nu);
	const __m128 mboundaryscale = _mm_set1_ps(boundary_scale);

	bool isSSEt = false;

	if(!isSSEt)
	{
		//Filter horizontally along each row 
		for(int y = 0; y < height; y++)
		{
			for(step = 0; step < iteration; step++)
			{
				ptr = image + width*y;
				ptr[0] *= boundary_scale;

				//rightwards
				for(x = 1; x < width; x++)
				{
					ptr[x] += nu*ptr[x - 1];
				}

				ptr[x = width - 1] *= boundary_scale;

				//leftwards
				for(; x > 0; x--)
				{
					ptr[x - 1] += nu*ptr[x];
				}
			}
		}
	}
	else
	{
		Mat buff(Size(height,width),CV_32F);
		float* buffptr = buff.ptr<float>(0);
		Mat s(Size(width,height),CV_32F,image);
		cv::transpose(s,buff);
		for(; x < height; x+=4)
		{	
			for(step = 0; step < iteration; step++)
			{
				ptr = buffptr +x;

				//ptr[0] *= boundaryscale;
				{
					__m128 im = _mm_load_ps(ptr);
					_mm_store_ps(ptr,_mm_mul_ps(mboundaryscale,im));
				}

				//downwards 
				for(i = height; i < num_pixels; i += height)
				{
					__m128 im = _mm_load_ps(ptr+i-height);
					__m128 dm = _mm_load_ps(ptr+i);
					_mm_store_ps(ptr+i,_mm_add_ps(dm,_mm_mul_ps(mnu,im)));
				}
				//ptr[i = numpixels - width] *= boundaryscale;
				{
					__m128 im = _mm_loadu_ps(ptr+num_pixels - height);
					_mm_store_ps(ptr+num_pixels - height,_mm_mul_ps(mboundaryscale,im));
				}

				i = num_pixels - height;
				//upwards
				for(; i > 0; i -= height)
				{
					__m128 im = _mm_load_ps(ptr+i);
					__m128 dm = _mm_load_ps(ptr+i-height);
					_mm_store_ps(ptr+i-height,_mm_add_ps(dm,_mm_mul_ps(mnu,im)));
				}
			}
		}
		cv::transpose(buff,s);
	}

	x = 0;
	if(width%4==0)
	{
		for(; x < width; x+=4)
		{	
			for(step = 0; step < iteration; step++)
			{
				ptr = image + x;

				//ptr[0] *= boundaryscale;
				{
					__m128 im = _mm_load_ps(ptr);
					_mm_store_ps(ptr,_mm_mul_ps(mboundaryscale,im));
				}

				//downwards 
				for(i = width; i < num_pixels; i += width)
				{
					__m128 im = _mm_load_ps(ptr+i-width);
					__m128 dm = _mm_load_ps(ptr+i);
					_mm_store_ps(ptr+i,_mm_add_ps(dm,_mm_mul_ps(mnu,im)));
				}
				//ptr[i = numpixels - width] *= boundaryscale;
				{
					__m128 im = _mm_loadu_ps(ptr+num_pixels - width);
					_mm_store_ps(ptr+num_pixels - width,_mm_mul_ps(mboundaryscale,im));
				}

				i = num_pixels - width;
				//upwards
				for(; i > 0; i -= width)
				{
					__m128 im = _mm_load_ps(ptr+i);
					__m128 dm = _mm_load_ps(ptr+i-width);
					_mm_store_ps(ptr+i-width,_mm_add_ps(dm,_mm_mul_ps(mnu,im)));
				}
			}
		}
	}
	else
	{
		for(; x <= width-4; x+=4)
		{	
			for(step = 0; step < iteration; step++)
			{
				ptr = image + x;

				//ptr[0] *= boundaryscale;
				{
					__m128 im = _mm_loadu_ps(ptr);
					_mm_storeu_ps(ptr,_mm_mul_ps(mboundaryscale,im));
				}

				//downwards 
				for(i = width; i < num_pixels; i += width)
				{
					__m128 im = _mm_loadu_ps(ptr+i-width);
					__m128 dm = _mm_loadu_ps(ptr+i);
					_mm_storeu_ps(ptr+i,_mm_add_ps(dm,_mm_mul_ps(mnu,im)));
				}


				//ptr[i = numpixels - width] *= boundaryscale;
				{
					__m128 im = _mm_loadu_ps(ptr+num_pixels - width);
					_mm_storeu_ps(ptr+num_pixels - width,_mm_mul_ps(mboundaryscale,im));
				}

				i = num_pixels - width;
				//upwards
				for(; i > 0; i -= width)
				{
					__m128 im = _mm_loadu_ps(ptr+i);
					__m128 dm = _mm_loadu_ps(ptr+i-width);
					_mm_storeu_ps(ptr+i-width,_mm_add_ps(dm,_mm_mul_ps(mnu,im)));
				}
			}
		}
		for(; x < width; x++)
		{
			for(step = 0; step < iteration; step++)
			{
				ptr = image + x;
				ptr[0] *= boundary_scale;

				//downwards
				for(i = width; i < num_pixels; i += width)
					ptr[i] += nu*ptr[i - width];

				i = num_pixels - width;
				ptr[i] *= boundary_scale;

				//upwards
				for(; i > 0; i -= width)
					ptr[i - width] += nu*ptr[i];
			}
		}
	}

	i=0;
	const __m128 mpostscale = _mm_set1_ps(post_scale);
	for(; i <= num_pixels-4; i+=4)
	{
		__m128 im = _mm_load_ps(image+i);
		_mm_store_ps(image+i,_mm_mul_ps(mpostscale,im));

	}
	for(; i < num_pixels; i++)
	{
		image[i] *= post_scale;
	}

	return;
}

void GaussianBlurAM(InputArray src_, OutputArray dest, float sigma, int iteration)
{
	Mat src = src_.getMat();
	Mat srcf;
	if(src.depth()!=CV_32F) src.convertTo(srcf,CV_32F);
	else srcf = src;

	if(src.channels()==1)
	{	 
		gaussian_am(srcf.ptr<float>(0),src.cols, src.rows, sigma, iteration);
	}
	else if (src.channels()==3)
	{
		vector<Mat> plane;
		split(srcf,plane);
		// cvtColorBGR2PLANE(src,plane);
		gaussian_am(plane[0].ptr<float>(0),src.cols,src.rows, sigma, iteration);
		gaussian_am(plane[1].ptr<float>(0),src.cols,src.rows, sigma, iteration);
		gaussian_am(plane[2].ptr<float>(0),src.cols,src.rows, sigma, iteration);

		merge(plane,dest);
	}

	if(src.depth()!=CV_32F) srcf.convertTo(dest,src.type(),1.0,0.5);
	else srcf.copyTo(dest);
}
