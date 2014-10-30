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

// Alvarez–Mazorra
//L. Alvarez, L. Mazorra, "Signal and image restoration using shock filters and anisotropic diffusion," SIAM Journal on Numerical Analysis, vol. 31, no. 2, pp. 590–605, 1994.

void gaussian_am(float *image, const int width, const int height, const float sigma, const int iteration)
{
    const int num_pixels = width*height;

    float nu, boundary_scale, post_scale;
    float *ptr;
    long i, x, y;
    int step;
    
    if(sigma <= 0 || iteration < 0)
        return;
    
    double lambda = (sigma*sigma)/(2.0*iteration);
    double dnu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
    nu = (float)dnu;
    boundary_scale = (float)(1.0/(1.0 - dnu));
    post_scale = (float)(pow(dnu/lambda,2*iteration));
    
    //Filter horizontally along each row 
    for(y = 0; y < height; y++)
    {
        for(step = 0; step < iteration; step++)
        {
            ptr = image + width*y;
            ptr[0] *= boundary_scale;
            
            //rightwards
            for(x = 1; x < width; x++)
                ptr[x] += nu*ptr[x - 1];
            
            ptr[x = width - 1] *= boundary_scale;
            
            //leftwards
            for(; x > 0; x--)
                ptr[x - 1] += nu*ptr[x];
        }
    }
    
    // Filter vertically along each column
	x = 0;

	for(; x <= width-4; x+=4)
	{	
		const __m128 mnu = _mm_set1_ps(nu);
		const __m128 mboundaryscale = _mm_set1_ps(boundary_scale);

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
    
    i=0;
	const __m128 mpostscale = _mm_set1_ps(post_scale);
    for(; i <= num_pixels-4; i+=4)
	{
		__m128 im = _mm_loadu_ps(image+i);
		_mm_storeu_ps(image+i,_mm_mul_ps(mpostscale,im));

	}
	for(; i < num_pixels; i++)
	{
		image[i] *= post_scale;
	}
    
    return;
}

void GaussianBlurIIR(InputArray src_, OutputArray dest, float sigma, int iteration)
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


//spectral recursive Gaussian Filter
//K. Sugimoto and S. Kamata: "Fast Gaussian filter with second-order shift property of DCT-5", Proc. IEEE Int. Conf. on Image Process. (ICIP2013), pp.514-518 (Sep. 2013).
namespace spectral_recursive_filter
{
//*************************************************************************************************
#define USE_SSE			//providing SSE computation
#define USE_OPENCV2		//providing OpenCV2 interface


	//extrapolation functions
//(atE() and atS() require variables w and h in their scope respectively)
/*
/// AA|ABCDE|EE (cv::BORDER_REPLICATE)
#define atW(x) (std::max(x,0))
#define atN(y) (std::max(y,0))
#define atE(x) (std::min(x,w-1))
#define atS(y) (std::min(y,h-1))
/*/
/// CB|ABCDE|DC (cv::BORDER_REFLECT_101)
#define atW(x) (std::abs(x))
#define atN(y) (std::abs(y))
#define atE(x) (w-1-std::abs(w-1-(x)))
#define atS(y) (h-1-std::abs(h-1-(y)))

class gauss
{
private:
	static const int K=2; //order of approximation (do not change!)

private:
	double sx,sy; //scale of Gaussian
	int rx,ry; //kernel radius
	std::vector<double> spectX,spectY;
	std::vector<double> tableX,tableY; //look-up tables

public:
	gauss(double sx,double sy):sx(sx),sy(sy)
	{
		if(sx<0.0 || sy<0.0)
			throw std::invalid_argument("\'sx\' and \'sy\' should be nonnegative!");
	
		rx=estimate_radius(sx);
		ry=estimate_radius(sy);
		spectX=gen_spectrum(sx,rx);
		spectY=gen_spectrum(sy,ry);
		tableX=build_lookup_table(rx,spectX);
		tableY=build_lookup_table(ry,spectY);
	}
	~gauss(){}

private:
	static inline double phase(int r)
	{
		return 2.0*CV_PI/(r+1+r); //DCT/DST-5
	}
	static inline int estimate_radius(double s)
	{
		//return (s<4.0) ? int(3.3333*s-0.3333+0.5) : int(3.4113*s-0.6452+0.5); //K==3
		return (s<4.0) ? int(3.0000*s-0.2000+0.5) : int(3.0000*s+0.5); //K==2
	}
	static inline std::vector<double> gen_spectrum(double s,int r)
	{
		const double phi=phase(r);
		std::vector<double> spect(K);
		for(int k=1;k<=K;k++)
			spect[k-1]=2.0*exp(-0.5*s*s*phi*phi*k*k);
		return spect;
	}
	static inline std::vector<double> build_lookup_table(int r,std::vector<double>& spect)
	{
		assert(spect.size()==K);
		const double phi=phase(r);
		std::vector<double> table(K*(1+r));
		for(int u=0;u<=r;++u)
			for(int k=1;k<=K;++k)
				table[K*u+k-1]=cos(k*phi*u)*spect[k-1];
		return table;
	}
	
	template <typename T>
	inline void filter_h(int w,int h,T* src,T* dst)
	{
		throw std::invalid_argument("Unsupported element type!");
	}
	template <typename T>
	inline void filter_v(int w,int h,T* src,T* dst)
	{
		throw std::invalid_argument("Unsupported element type!");
	}
	template <typename T>
	inline void filter_sse_h(int w,int h,T* src,T* dst)
	{
		throw std::invalid_argument("Unsupported element type!");
	}
	template <typename T>
	inline void filter_sse_v(int w,int h,T* src,T* dst)
	{
		throw std::invalid_argument("Unsupported element type!");
	}

public:

#ifdef USE_SSE
	template <typename T>
	void filter(int w,int h,T* src,T* dst)
	{
		if(w<=4.0*sx || h<=4.0*sy)
			throw std::invalid_argument("\'sx\' and \'sy\' should be less than about w/4 or h/4!");
		
		//filtering is skipped if s==0.0
		if(sx==0.0 && sy==0.0)
			return;
		else if(sx==0.0)
			filter_sse_v<T>(w,h,src,dst);
		else if(sy==0.0)
			filter_sse_h<T>(w,h,src,dst);
		else
		{
			filter_sse_v<T>(w,h,src,dst);
			filter_sse_h<T>(w,h,dst,dst); //only filter_h() allows src==dst.
		}
	}
#else
	template <typename T>
	void filter(int w,int h,T* src,T* dst)
	{
		if(w<=4.0*sx || h<=4.0*sy)
			throw std::invalid_argument("\'sx\' and \'sy\' should be less than about w/4 or h/4!");
		
		//filtering is skipped if s==0.0
		if(sx==0.0 && sy==0.0)
			return;
		else if(sx==0.0)
			filter_v<T>(w,h,src,dst);
		else if(sy==0.0)
			filter_h<T>(w,h,src,dst);
		else
		{
			filter_v<T>(w,h,src,dst);
			filter_h<T>(w,h,dst,dst); //only filter_h() allows src==dst.
		}
	}
#endif
	
#ifdef USE_OPENCV2 //OpenCV2 interface for easy function call.
	void filter(const cv::Mat& src,cv::Mat& dst)
	{
		//checking the format of input/output images
		if(src.size()!=dst.size())
			throw std::invalid_argument("\'src\' and \'dst\' should have the same size!");
		if(src.type()!=dst.type())
			throw std::invalid_argument("\'src\' and \'dst\' should have the same element type!");
		if(src.channels()!=1 || dst.channels()!=1)
			throw std::invalid_argument("Multi-channel images are unsupported!");
		if(src.isSubmatrix() || dst.isSubmatrix())
			throw std::invalid_argument("Subimages are unsupported!");

		switch(src.type())
		{
		case CV_32FC1:
			filter<float>(src.cols,src.rows,reinterpret_cast<float*>(src.data),reinterpret_cast<float*>(dst.data));
			break;
		default:
			throw std::invalid_argument("Unsupported element type!");
			break;
		}
	}
#endif
};

//*************************************************************************************************

template<>
inline void gauss::filter_h<float>(int w,int h,float* src,float* dst)
{
	const int r=rx;
	const float norm=float(1.0/(r+1+r));
	std::vector<float> table(tableX.size());
	for(int t=0;t<int(table.size());++t)
		table[t]=float(tableX[t]);
	
	const float cf11=float(table[K*1+0]*2.0/spectX[0]), cfR1=table[K*r+0];
	const float cf12=float(table[K*1+1]*2.0/spectX[1]), cfR2=table[K*r+1];

	float sum,a1,a2,b1,b2;
	float dA,dB,delta;
	std::vector<float> buf(w); //to allow for src==dst
	for(int y=0;y<h;++y)
	{
		std::copy(&src[w*y],&src[w*y+w],buf.begin());

		sum=buf[0];
		a1=buf[0]*table[0]; b1=buf[1]*table[0];
		a2=buf[0]*table[1]; b2=buf[1]*table[1];
		for(int u=1;u<=r;++u)
		{
			const float sumA=buf[atW(0-u)]+buf[0+u];
			const float sumB=buf[atW(1-u)]+buf[1+u];
			sum+=sumA;
			a1+=sumA*table[K*u+0]; b1+=sumB*table[K*u+0];
			a2+=sumA*table[K*u+1]; b2+=sumB*table[K*u+1];
		}

		//the first pixel (x=0)
		float* q=&dst[w*y];
		q[0]=norm*(sum+a1+a2);
		dA=buf[atE(0+r+1)]-buf[atW(0-r)];
		sum+=dA;

		//the other pixels (0<x<w)
		int x=1;
		while(true) //four-length ring buffers
		{
			q[x]=norm*(sum+b1+b2);
			dB=buf[atE(x+r+1)]-buf[atW(x-r)]; delta=dA-dB;
			sum+=dB;
			a1+=-cf11*b1+cfR1*delta;
			a2+=-cf12*b2+cfR2*delta;
			x++; if(w<=x) break;
			
			q[x]=norm*(sum-a1-a2);
			dA=buf[atE(x+r+1)]-buf[atW(x-r)]; delta=dB-dA;
			sum+=dA;
			b1+=+cf11*a1+cfR1*delta;
			b2+=+cf12*a2+cfR2*delta;
			x++; if(w<=x) break;
			
			q[x]=norm*(sum-b1-b2);
			dB=buf[atE(x+r+1)]-buf[atW(x-r)]; delta=dA-dB;
			sum+=dB;
			a1+=-cf11*b1-cfR1*delta;
			a2+=-cf12*b2-cfR2*delta;
			x++; if(w<=x) break;
			
			q[x]=norm*(sum+a1+a2);
			dA=buf[atE(x+r+1)]-buf[atW(x-r)]; delta=dB-dA;
			sum+=dA;
			b1+=+cf11*a1-cfR1*delta;
			b2+=+cf12*a2-cfR2*delta;
			x++; if(w<=x) break;
		}
	}
}
template<>
inline void gauss::filter_v<float>(int w,int h,float* src,float* dst)
{
	const int r=ry;
	const float norm=float(1.0/(r+1+r));
	std::vector<float> table(tableY.size());
	for(int t=0;t<int(table.size());++t)
		table[t]=float(tableY[t]);

	//work space to keep raster scanning
	std::vector<float> workspace((2*K+1)*w);

	//calculating the first and second terms
	for(int x=0;x<w;++x)
	{
		float* ws=&workspace[(2*K+1)*x];
		ws[0]=src[x];
		ws[1]=src[x+w]*table[0]; ws[2]=src[x]*table[0];
		ws[3]=src[x+w]*table[1]; ws[4]=src[x]*table[1];
	}
	for(int v=1;v<=r;++v)
	{
		for(int x=0;x<w;++x)
		{
			const float sum0=src[x+w*atN(0-v)]+src[x+w*(0+v)];
			const float sum1=src[x+w*atN(1-v)]+src[x+w*(1+v)];
			float* ws=&workspace[(2*K+1)*x];
			ws[0]+=sum0;
			ws[1]+=sum1*table[K*v+0]; ws[2]+=sum0*table[K*v+0];
			ws[3]+=sum1*table[K*v+1]; ws[4]+=sum0*table[K*v+1];
		}
	}

	const float cf11=float(table[K*1+0]*2.0/spectY[0]), cfR1=table[K*r+0];
	const float cf12=float(table[K*1+1]*2.0/spectY[1]), cfR2=table[K*r+1];
	
	float *q,*p0N,*p0S,*p1N,*p1S;
	for(int y=0;y<1;++y) //the first line (y=0)
	{
		q=&dst[w*y];
		p1N=&src[w*atN(0-r  )]; p1S=&src[w*atS(0+r+1)];
		for(int x=0;x<w;++x)
		{
			float* ws=&workspace[(2*K+1)*x];
			q[x]=norm*(ws[0]+ws[2]+ws[4]);
			ws[0]+=p1S[x]-p1N[x];
		}
	}
	for(int y=1;y<h;++y) //remaining lines (with two-length ring buffers)
	{
		q=&dst[w*y];
		p0N=&src[w*atN(y-r-1)]; p0S=&src[w*atS(y+r  )];
		p1N=&src[w*atN(y-r  )]; p1S=&src[w*atS(y+r+1)];
		for(int x=0;x<w;++x)
		{
			float* ws=&workspace[(2*K+1)*x];
			q[x]=norm*(ws[0]+ws[1]+ws[3]);

			const float d0=p0S[x]-p0N[x];
			const float d1=p1S[x]-p1N[x];
			const float delta=d1-d0;

			ws[0]+=d1;
			ws[2]=cfR1*delta+cf11*ws[1]-ws[2];
			ws[4]=cfR2*delta+cf12*ws[3]-ws[4];
		}
		y++; if(h<=y) break; //to the next line
		
		q=&dst[w*y];
		p0N=&src[w*atN(y-r-1)]; p0S=&src[w*atS(y+r  )];
		p1N=&src[w*atN(y-r  )]; p1S=&src[w*atS(y+r+1)];
		for(int x=0;x<w;++x)
		{
			float* ws=&workspace[(2*K+1)*x];
			q[x]=norm*(ws[0]+ws[2]+ws[4]);

			const float d0=p0S[x]-p0N[x];
			const float d1=p1S[x]-p1N[x];
			const float delta=d1-d0;

			ws[0]+=d1;
			ws[1]=cfR1*delta+cf11*ws[2]-ws[1];
			ws[3]=cfR2*delta+cf12*ws[4]-ws[3];
		}
	}
}

template<>
inline void gauss::filter_sse_h<float>(int w,int h,float* src,float* dst)
{
	const int B=sizeof(__m128)/sizeof(float);
	const int r=rx;
	const float norm=float(1.0/(r+1+r));
	std::vector<float> table(tableX.size());
	for(int t=0;t<int(table.size());++t)
		table[t]=float(tableX[t]);
	
	const float cf11=float(table[K*1+0]*2.0/spectX[0]), cfR1=table[K*r+0];
	const float cf12=float(table[K*1+1]*2.0/spectX[1]), cfR2=table[K*r+1];
	
	//to allow for src==dst
	std::vector<float> buf(B*w);

	assert(h%B==0);
	for(int y=0;y<h/B*B;y+=B)
	{
		std::copy(&src[w*y],&src[w*(y+B)],buf.begin());
		
		__m128 pv0=_mm_set_ps(buf[w*3+0],buf[w*2+0],buf[w*1+0],buf[w*0+0]);
		__m128 pv1=_mm_set_ps(buf[w*3+1],buf[w*2+1],buf[w*1+1],buf[w*0+1]);
		
		__m128 sum=pv0;
		__m128 a1=_mm_mul_ps(_mm_set1_ps(table[0]),pv0);
		__m128 b1=_mm_mul_ps(_mm_set1_ps(table[0]),pv1);
		__m128 a2=_mm_mul_ps(_mm_set1_ps(table[1]),pv0);
		__m128 b2=_mm_mul_ps(_mm_set1_ps(table[1]),pv1);

		for(int u=1;u<=r;++u)
		{
			const float* p0M=&buf[atW(0-u)];
			const float* p1M=&buf[atW(1-u)];
			const float* p0P=&buf[   (0+u)];
			const float* p1P=&buf[   (1+u)];
			__m128 pv0M=_mm_set_ps(p0M[w*3],p0M[w*2],p0M[w*1],p0M[w*0]);
			__m128 pv1M=_mm_set_ps(p1M[w*3],p1M[w*2],p1M[w*1],p1M[w*0]);
			__m128 pv0P=_mm_set_ps(p0P[w*3],p0P[w*2],p0P[w*1],p0P[w*0]);
			__m128 pv1P=_mm_set_ps(p1P[w*3],p1P[w*2],p1P[w*1],p1P[w*0]);
			__m128 sumA=_mm_add_ps(pv0M,pv0P);
			__m128 sumB=_mm_add_ps(pv1M,pv1P);
			
			sum=_mm_add_ps(sum,sumA);
			a1=_mm_add_ps(a1,_mm_mul_ps(_mm_set1_ps(table[K*u+0]),sumA));
			b1=_mm_add_ps(b1,_mm_mul_ps(_mm_set1_ps(table[K*u+0]),sumB));
			a2=_mm_add_ps(a2,_mm_mul_ps(_mm_set1_ps(table[K*u+1]),sumA));
			b2=_mm_add_ps(b2,_mm_mul_ps(_mm_set1_ps(table[K*u+1]),sumB));
		}
		
		//sliding convolution
		float *pA,*pB;
		__m128 pvA,pvB;
		__m128 dvA,dvB,delta;
		__m128 qv[B];

		//the first four pixels (0<=x<B)
		for(int x=0;x<B;x+=B)
		{
			//the first pixel (x=0)
			qv[0]=_mm_mul_ps(_mm_set1_ps(norm),_mm_add_ps(sum,_mm_add_ps(a1,a2)));
			
			pA=&buf[atW(0-r  )]; pvA=_mm_set_ps(pA[w*3],pA[w*2],pA[w*1],pA[w*0]);
			pB=&buf[   (0+r+1)]; pvB=_mm_set_ps(pB[w*3],pB[w*2],pB[w*1],pB[w*0]);
			dvA=_mm_sub_ps(pvB,pvA);

			sum=_mm_add_ps(sum,dvA);
			//b1=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1),delta),_mm_mul_ps(_mm_set1_ps(cf11),a1)),b1);
			//b2=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2),delta),_mm_mul_ps(_mm_set1_ps(cf12),a2)),b2);

			//the second pixel (x=1)
			qv[1]=_mm_mul_ps(_mm_set1_ps(norm),_mm_add_ps(sum,_mm_add_ps(b1,b2)));
			
			pA=&buf[atW(1-r  )]; pvA=_mm_set_ps(pA[w*3],pA[w*2],pA[w*1],pA[w*0]);
			pB=&buf[   (1+r+1)]; pvB=_mm_set_ps(pB[w*3],pB[w*2],pB[w*1],pB[w*0]);
			dvB=_mm_sub_ps(pvB,pvA);
			delta=_mm_sub_ps(dvB,dvA);

			sum=_mm_add_ps(sum,dvB);
			a1=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1),delta),_mm_mul_ps(_mm_set1_ps(cf11),b1)),a1);
			a2=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2),delta),_mm_mul_ps(_mm_set1_ps(cf12),b2)),a2);
		
			//the third pixel (x=2)
			qv[2]=_mm_mul_ps(_mm_set1_ps(norm),_mm_add_ps(sum,_mm_add_ps(a1,a2)));
		
			pA=&buf[atW(2-r  )]; pvA=_mm_set_ps(pA[w*3],pA[w*2],pA[w*1],pA[w*0]);
			pB=&buf[   (2+r+1)]; pvB=_mm_set_ps(pB[w*3],pB[w*2],pB[w*1],pB[w*0]);
			dvA=_mm_sub_ps(pvB,pvA);
			delta=_mm_sub_ps(dvA,dvB);

			sum=_mm_add_ps(sum,dvA);
			b1=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1),delta),_mm_mul_ps(_mm_set1_ps(cf11),a1)),b1);
			b2=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2),delta),_mm_mul_ps(_mm_set1_ps(cf12),a2)),b2);

			//the forth pixel (x=3)
			qv[3]=_mm_mul_ps(_mm_set1_ps(norm),_mm_add_ps(sum,_mm_add_ps(b1,b2)));
		
			pA=&buf[atW(3-r  )]; pvA=_mm_set_ps(pA[w*3],pA[w*2],pA[w*1],pA[w*0]);
			pB=&buf[   (3+r+1)]; pvB=_mm_set_ps(pB[w*3],pB[w*2],pB[w*1],pB[w*0]);
			dvB=_mm_sub_ps(pvB,pvA);
			delta=_mm_sub_ps(dvB,dvA);

			sum=_mm_add_ps(sum,dvB);
			a1=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1),delta),_mm_mul_ps(_mm_set1_ps(cf11),b1)),a1);
			a2=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2),delta),_mm_mul_ps(_mm_set1_ps(cf12),b2)),a2);
			
			//output with transposition
			_MM_TRANSPOSE4_PS(qv[0],qv[1],qv[2],qv[3]);
			_mm_storeu_ps(&dst[w*(y+0)],qv[0]);
			_mm_storeu_ps(&dst[w*(y+1)],qv[1]);
			_mm_storeu_ps(&dst[w*(y+2)],qv[2]);
			_mm_storeu_ps(&dst[w*(y+3)],qv[3]);
		}
		
		//the other pixels (B<=x<w)
		for(int x=B;x<w/B*B;x+=B) //four-length ring buffers
		{
			//the first pixel (x=0)
			qv[0]=_mm_mul_ps(_mm_set1_ps(norm),_mm_add_ps(sum,_mm_add_ps(a1,a2)));

			pA=&buf[atW(x+0-r  )]; pvA=_mm_set_ps(pA[w*3],pA[w*2],pA[w*1],pA[w*0]);
			pB=&buf[atE(x+0+r+1)]; pvB=_mm_set_ps(pB[w*3],pB[w*2],pB[w*1],pB[w*0]);
			dvA=_mm_sub_ps(pvB,pvA);
			delta=_mm_sub_ps(dvA,dvB);

			sum=_mm_add_ps(sum,dvA);
			b1=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1),delta),_mm_mul_ps(_mm_set1_ps(cf11),a1)),b1);
			b2=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2),delta),_mm_mul_ps(_mm_set1_ps(cf12),a2)),b2);

			//the second pixel (x=1)
			qv[1]=_mm_mul_ps(_mm_set1_ps(norm),_mm_add_ps(sum,_mm_add_ps(b1,b2)));
		
			pA=&buf[atW(x+1-r  )]; pvA=_mm_set_ps(pA[w*3],pA[w*2],pA[w*1],pA[w*0]);
			pB=&buf[atE(x+1+r+1)]; pvB=_mm_set_ps(pB[w*3],pB[w*2],pB[w*1],pB[w*0]);
			dvB=_mm_sub_ps(pvB,pvA);
			delta=_mm_sub_ps(dvB,dvA);

			sum=_mm_add_ps(sum,dvB);
			a1=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1),delta),_mm_mul_ps(_mm_set1_ps(cf11),b1)),a1);
			a2=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2),delta),_mm_mul_ps(_mm_set1_ps(cf12),b2)),a2);
		
			//the third pixel (x=2)
			qv[2]=_mm_mul_ps(_mm_set1_ps(norm),_mm_add_ps(sum,_mm_add_ps(a1,a2)));
		
			pA=&buf[atW(x+2-r  )]; pvA=_mm_set_ps(pA[w*3],pA[w*2],pA[w*1],pA[w*0]);
			pB=&buf[atE(x+2+r+1)]; pvB=_mm_set_ps(pB[w*3],pB[w*2],pB[w*1],pB[w*0]);
			dvA=_mm_sub_ps(pvB,pvA);
			delta=_mm_sub_ps(dvA,dvB);

			sum=_mm_add_ps(sum,dvA);
			b1=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1),delta),_mm_mul_ps(_mm_set1_ps(cf11),a1)),b1);
			b2=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2),delta),_mm_mul_ps(_mm_set1_ps(cf12),a2)),b2);
		
			//the forth pixel (x=3)
			qv[3]=_mm_mul_ps(_mm_set1_ps(norm),_mm_add_ps(sum,_mm_add_ps(b1,b2)));
		
			pA=&buf[atW(x+3-r  )]; pvA=_mm_set_ps(pA[w*3],pA[w*2],pA[w*1],pA[w*0]);
			pB=&buf[atE(x+3+r+1)]; pvB=_mm_set_ps(pB[w*3],pB[w*2],pB[w*1],pB[w*0]);
			dvB=_mm_sub_ps(pvB,pvA);
			delta=_mm_sub_ps(dvB,dvA);

			sum=_mm_add_ps(sum,dvB);
			a1=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1),delta),_mm_mul_ps(_mm_set1_ps(cf11),b1)),a1);
			a2=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2),delta),_mm_mul_ps(_mm_set1_ps(cf12),b2)),a2);
			
			//output with transposition
			_MM_TRANSPOSE4_PS(qv[0],qv[1],qv[2],qv[3]);
			_mm_storeu_ps(&dst[w*(y+0)+x],qv[0]);
			_mm_storeu_ps(&dst[w*(y+1)+x],qv[1]);
			_mm_storeu_ps(&dst[w*(y+2)+x],qv[2]);
			_mm_storeu_ps(&dst[w*(y+3)+x],qv[3]);
		}
	}
}
template<>
inline void gauss::filter_sse_v<float>(int w,int h,float* src,float* dst)
{
	assert(w%sizeof(__m128)==0);

	const int B=sizeof(__m128)/sizeof(float);
	const int r=ry;
	const float norm=float(1.0/(r+1+r));
	std::vector<float> table(tableY.size());
	for(int t=0;t<int(table.size());++t)
		table[t]=float(tableY[t]);

	//work space to keep raster scanning
	float* workspace=reinterpret_cast<float*>(_mm_malloc(sizeof(float)*(2*K+1)*w,sizeof(__m128)));
	
	//calculating the first and second terms
	for(int x=0;x<w/B*B;x+=B)
	{
		float* ws=&workspace[(2*K+1)*x];
		__m128 p0=_mm_load_ps(&src[x]);
		__m128 p1=_mm_load_ps(&src[x+w]);
		_mm_store_ps(&ws[B*0],p0);
		_mm_store_ps(&ws[B*1],_mm_mul_ps(p1,_mm_set1_ps(table[0])));
		_mm_store_ps(&ws[B*2],_mm_mul_ps(p0,_mm_set1_ps(table[0])));
		_mm_store_ps(&ws[B*3],_mm_mul_ps(p1,_mm_set1_ps(table[1])));
		_mm_store_ps(&ws[B*4],_mm_mul_ps(p0,_mm_set1_ps(table[1])));
	}
	for(int v=1;v<=r;++v)
	{
		for(int x=0;x<w/B*B;x+=B)
		{
			float* ws=&workspace[(2*K+1)*x];
			__m128 sum0=_mm_add_ps(_mm_load_ps(&src[x+w*atN(0-v)]),_mm_load_ps(&src[x+w*(0+v)]));
			__m128 sum1=_mm_add_ps(_mm_load_ps(&src[x+w*atN(1-v)]),_mm_load_ps(&src[x+w*(1+v)]));
			_mm_store_ps(&ws[B*0],_mm_add_ps(_mm_load_ps(&ws[B*0]),sum0));
			_mm_store_ps(&ws[B*1],_mm_add_ps(_mm_load_ps(&ws[B*1]),_mm_mul_ps(sum1,_mm_set1_ps(table[K*v+0]))));
			_mm_store_ps(&ws[B*2],_mm_add_ps(_mm_load_ps(&ws[B*2]),_mm_mul_ps(sum0,_mm_set1_ps(table[K*v+0]))));
			_mm_store_ps(&ws[B*3],_mm_add_ps(_mm_load_ps(&ws[B*3]),_mm_mul_ps(sum1,_mm_set1_ps(table[K*v+1]))));
			_mm_store_ps(&ws[B*4],_mm_add_ps(_mm_load_ps(&ws[B*4]),_mm_mul_ps(sum0,_mm_set1_ps(table[K*v+1]))));
		}
	}
	
	const float cf11=float(table[K*1+0]*2.0/spectY[0]), cfR1=table[K*r+0];
	const float cf12=float(table[K*1+1]*2.0/spectY[1]), cfR2=table[K*r+1];

	//sliding convolution
	for(int y=0;y<1;++y) //the first line (y=0)
	{
		float* q=&dst[w*y];
		const float* p1N=&src[w*atN(y-r  )];
		const float* p1S=&src[w*atS(y+r+1)];
		for(int x=0;x<w/B*B;x+=B)
		{
			float* ws=&workspace[(2*K+1)*x];
			const __m128 a0=_mm_load_ps(&ws[B*0]);
			const __m128 a2=_mm_load_ps(&ws[B*2]);
			const __m128 a4=_mm_load_ps(&ws[B*4]);
			_mm_store_ps(&q[x],_mm_mul_ps(_mm_set1_ps(norm),_mm_add_ps(a0,_mm_add_ps(a2,a4))));

			const __m128 d=_mm_sub_ps(_mm_load_ps(&p1S[x]),_mm_load_ps(&p1N[x]));
			_mm_store_ps(&ws[B*0],_mm_add_ps(a0,d));
		}
	}
	for(int y=1;y<h;++y) //the other lines
	{
		float* q=&dst[w*y];
		const float* p0N=&src[w*atN(y-r-1)];
		const float* p1N=&src[w*atN(y-r  )];
		const float* p0S=&src[w*atS(y+r  )];
		const float* p1S=&src[w*atS(y+r+1)];
		for(int x=0;x<w/B*B;x+=B)
		{
			float* ws=&workspace[(2*K+1)*x];
			const __m128 a0=_mm_load_ps(&ws[B*0]);
			const __m128 a1=_mm_load_ps(&ws[B*1]);
			const __m128 a3=_mm_load_ps(&ws[B*3]);
			_mm_store_ps(&q[x],_mm_mul_ps(_mm_set1_ps(norm),_mm_add_ps(a0,_mm_add_ps(a1,a3))));

			const __m128 d0=_mm_sub_ps(_mm_load_ps(&p0S[x]),_mm_load_ps(&p0N[x]));
			const __m128 d1=_mm_sub_ps(_mm_load_ps(&p1S[x]),_mm_load_ps(&p1N[x]));
			const __m128 delta=_mm_sub_ps(d1,d0);

			_mm_store_ps(&ws[B*0],_mm_add_ps(a0,d1));
			_mm_store_ps(&ws[B*1],_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1),delta),_mm_mul_ps(_mm_set1_ps(cf11),a1)),_mm_load_ps(&ws[B*2])));
			_mm_store_ps(&ws[B*2],a1);
			_mm_store_ps(&ws[B*3],_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2),delta),_mm_mul_ps(_mm_set1_ps(cf12),a3)),_mm_load_ps(&ws[B*4])));
			_mm_store_ps(&ws[B*4],a3);
		}
	}
	_mm_free(workspace);
}
}

void GaussianBlurSR_cliped(InputArray src_, OutputArray dest, float sigma)
{
	Mat src = src_.getMat();
	Mat srcf;
	if(src.depth()!=CV_32F) src.convertTo(srcf,CV_32F);
	else srcf = src;

	Mat srcf2(src.size(),CV_32FC(src.channels()));

	if(src.channels()==1)
	{	 
		spectral_recursive_filter::gauss srf_gauss(sigma,sigma);
		srf_gauss.filter(srcf, srcf2);
	}
	else if (src.channels()==3)
	{
		vector<Mat> plane;
		split(srcf,plane);

		Mat temp(src.size(),CV_32F);
		spectral_recursive_filter::gauss srf_gauss(sigma,sigma);
		srf_gauss.filter(plane[0],temp);temp.copyTo(plane[0]);
		srf_gauss.filter(plane[1],temp);temp.copyTo(plane[1]);
		srf_gauss.filter(plane[2],temp);temp.copyTo(plane[2]);
		
		merge(plane,srcf2);
	}

	if(src.depth()!=CV_32F && src.depth()!=CV_64F) srcf2.convertTo(dest,src.type(),1.0,0.5);
	else if(src.depth()==CV_64F)srcf2.convertTo(dest,src.type());
	else srcf2.copyTo(dest);
}

void GaussianBlurSR(InputArray src, OutputArray dest, float sigma)
{
	const int SIMDSTEP = 4;
	int xpad = src.size().width%4;
	int ypad = src.size().height%4;

	xpad = (SIMDSTEP-xpad)%SIMDSTEP;
	ypad = (SIMDSTEP-ypad)%SIMDSTEP;

	if(xpad==0 && ypad==0)
	{
		GaussianBlurSR_cliped(src,dest,sigma);
	}
	else
	{
		Mat s, d;
		copyMakeBorder(src, s, 0, ypad, 0, xpad, BORDER_REPLICATE);
		GaussianBlurSR_cliped(s,d,sigma);
		Mat(d(Rect(Point(0,0),src.size()))).copyTo(dest);
	}
}